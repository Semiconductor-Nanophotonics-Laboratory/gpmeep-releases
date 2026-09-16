/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#include "meep_internals.hpp"
#include "bicgstab.hpp"
#include "gpu_backend_internal.hpp"
#include "gpu_grid_index.hpp"

#if MEEP_HAVE_CUDA
#include "meep_cuda/runtime.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

namespace meep {

namespace {

class distributed_cw_failure_guard {
public:
  explicit distributed_cw_failure_guard(bool armed)
      : armed_(armed)
#if __cplusplus >= 201703L
        , uncaught_exceptions_on_entry_(std::uncaught_exceptions())
#endif
  {}

  ~distributed_cw_failure_guard() {
    bool unwinding = false;
#if __cplusplus >= 201703L
    unwinding =
        std::uncaught_exceptions() > uncaught_exceptions_on_entry_;
#else
    unwinding = std::uncaught_exception();
#endif
    if (armed_ && unwinding)
      meep::abort(
          "rank-local failure during distributed solve_cw; aborting the "
          "communicator to prevent a Krylov or boundary deadlock");
  }

private:
  bool armed_;
#if __cplusplus >= 201703L
  int uncaught_exceptions_on_entry_;
#endif
};

class solve_cw_state_guard {
public:
  solve_cw_state_guard(fields &owner, int saved_t,
                       complex<double> omega)
      : owner_(owner), saved_t_(saved_t), active_(true) {
    owner_.set_solve_cw_omega(omega);
  }

  ~solve_cw_state_guard() {
    if (active_) restore();
  }

  void restore() {
    if (!active_) return;
    owner_.t = saved_t_;
    owner_.unset_solve_cw_omega();
    active_ = false;
  }

  solve_cw_state_guard(const solve_cw_state_guard &) = delete;
  solve_cw_state_guard &operator=(const solve_cw_state_guard &) = delete;

private:
  fields &owner_;
  int saved_t_;
  bool active_;
};

#if MEEP_HAVE_CUDA
class cuda_cw_workspace {
public:
  explicit cuda_cw_workspace(size_t scalar_count)
      : values_(nullptr), scalar_count_(scalar_count) {
    if (scalar_count == 0 ||
        scalar_count > std::numeric_limits<size_t>::max() / sizeof(float))
      throw std::overflow_error("CUDA CW workspace size overflow");
    values_ = static_cast<float *>(
        meep_cuda::allocate_device_bytes(scalar_count * sizeof(float)));
  }

  ~cuda_cw_workspace() { meep_cuda::free_device(values_); }

  cuda_cw_workspace(const cuda_cw_workspace &) = delete;
  cuda_cw_workspace &operator=(const cuda_cw_workspace &) = delete;

  float *values() const noexcept { return values_; }
  size_t scalar_count() const noexcept { return scalar_count_; }

private:
  float *values_;
  size_t scalar_count_;
};

class resident_cw_plan_guard {
public:
  explicit resident_cw_plan_guard(
      gpu::detail::resident_cw_vector_plan *plan)
      : plan_(plan) {}
  ~resident_cw_plan_guard() {
    gpu::detail::destroy_resident_cw_vector_plan(plan_);
  }

  resident_cw_plan_guard(const resident_cw_plan_guard &) = delete;
  resident_cw_plan_guard &operator=(const resident_cw_plan_guard &) = delete;

  gpu::detail::resident_cw_vector_plan *get() const noexcept {
    return plan_;
  }
  void reset(gpu::detail::resident_cw_vector_plan *plan) {
    if (plan == plan_) return;
    gpu::detail::destroy_resident_cw_vector_plan(plan_);
    plan_ = plan;
  }

private:
  gpu::detail::resident_cw_vector_plan *plan_;
};

class cuda_bicgstab_vector_ops : public bicgstab_vector_ops {
public:
  cuda_bicgstab_vector_ops()
      : fp64_workspace_(static_cast<double *>(
            meep_cuda::allocate_device_bytes(
                (meep_cuda::vector_reduction_partial_capacity + 1) *
                sizeof(double)))),
        fp64_result_(fp64_workspace_ +
                     meep_cuda::vector_reduction_partial_capacity),
        fp32_result_(nullptr), finite_result_(nullptr),
        breakdown_injected_(false) {
    try {
      fp32_result_ = static_cast<float *>(
          meep_cuda::allocate_device_bytes(sizeof(float)));
      finite_result_ = static_cast<int *>(
          meep_cuda::allocate_device_bytes(sizeof(int)));
    }
    catch (...) {
      meep_cuda::free_device(finite_result_);
      meep_cuda::free_device(fp32_result_);
      meep_cuda::free_device(fp64_workspace_);
      throw;
    }
  }

  ~cuda_bicgstab_vector_ops() override {
    meep_cuda::free_device(finite_result_);
    meep_cuda::free_device(fp32_result_);
    meep_cuda::free_device(fp64_workspace_);
  }

  double dot(size_t n, const realnum *x,
             const realnum *y) override {
    meep_cuda::initialize_fp64_result(fp64_result_);
    meep_cuda::vector_dot_fp32(
        reinterpret_cast<const float *>(x),
        reinterpret_cast<const float *>(y), n, fp64_workspace_,
        meep_cuda::vector_reduction_partial_capacity, fp64_result_);
    double local = 0.0;
    meep_cuda::copy_to_host(&local, fp64_result_, sizeof(local));
    if (!breakdown_injected_ && am_master() &&
        std::getenv("MEEP_GPU_TEST_CW_KRYLOV_RANK_BREAKDOWN")) {
      local = std::numeric_limits<double>::quiet_NaN();
      breakdown_injected_ = true;
    }
    if (!and_to_all(std::isfinite(local)))
      throw bicgstab_numerical_breakdown(
          "nonfinite CUDA CW Krylov dot product");
    return sum_to_all(local);
  }

  double norm2(size_t n, const realnum *x) override {
    meep_cuda::initialize_fp32_result(fp32_result_);
    meep_cuda::initialize_int_result(finite_result_, 1);
    meep_cuda::vector_max_abs_fp32(
        reinterpret_cast<const float *>(x), n, fp32_result_,
        finite_result_);
    float local_maximum = 0.0f;
    int local_finite = 0;
    meep_cuda::copy_to_host(
        &local_maximum, fp32_result_, sizeof(local_maximum));
    meep_cuda::copy_to_host(
        &local_finite, finite_result_, sizeof(local_finite));
    if (!and_to_all(local_finite == 1 &&
                    std::isfinite(local_maximum)))
      throw bicgstab_numerical_breakdown(
          "nonfinite CUDA CW Krylov vector");
    const double maximum = max_to_all(
        static_cast<double>(local_maximum));
    if (maximum == 0.0) return 0.0;
    const double inverse = 1.0 / maximum;
    if (!std::isfinite(inverse) || inverse == 0.0)
      throw bicgstab_numerical_breakdown(
          "CUDA CW Krylov norm scale is outside FP32 range");
    meep_cuda::initialize_fp64_result(fp64_result_);
    meep_cuda::vector_scaled_sum_squares_fp32(
        reinterpret_cast<const float *>(x), n, inverse, fp64_workspace_,
        meep_cuda::vector_reduction_partial_capacity, fp64_result_);
    double local_sum = 0.0;
    meep_cuda::copy_to_host(&local_sum, fp64_result_, sizeof(local_sum));
    if (!and_to_all(std::isfinite(local_sum) && local_sum >= 0.0))
      throw bicgstab_numerical_breakdown(
          "nonfinite CUDA CW Krylov squared norm");
    const double sum = sum_to_all(local_sum);
    if (!std::isfinite(sum) || sum < 0.0)
      throw bicgstab_numerical_breakdown(
          "nonfinite distributed CUDA CW Krylov norm");
    return maximum * std::sqrt(sum);
  }

  void fill(size_t n, realnum *x, realnum value) override {
    meep_cuda::vector_fill_fp32(
        reinterpret_cast<float *>(x), n, static_cast<float>(value));
  }

  void copy(size_t n, realnum *destination,
            const realnum *source) override {
    meep_cuda::vector_copy_fp32(
        reinterpret_cast<float *>(destination),
        reinterpret_cast<const float *>(source), n);
  }

  void scale(size_t n, realnum *x, double value) override {
    const double scalar = checked_scalar(value, "scale");
    meep_cuda::vector_scale_fp32(
        reinterpret_cast<float *>(x), n, scalar);
  }

  void xpay(size_t n, realnum *x, double value,
            const realnum *y) override {
    const double scalar = checked_scalar(value, "xpay");
    meep_cuda::vector_xpay_fp32(
        reinterpret_cast<float *>(x),
        reinterpret_cast<const float *>(y), n, scalar);
  }

  void left_minus_scale(size_t n, realnum *output,
                        const realnum *left,
                        double value) override {
    const double scalar = checked_scalar(value, "left-minus-scale");
    meep_cuda::vector_left_minus_scale_fp32(
        reinterpret_cast<float *>(output),
        reinterpret_cast<const float *>(left), n, scalar);
  }

private:
  static double checked_scalar(double value, const char *operation) {
    if (!std::isfinite(value))
      throw bicgstab_numerical_breakdown(
          std::string("nonfinite CUDA CW Krylov ") + operation +
          " coefficient");
    return value;
  }

  double *fp64_workspace_;
  double *fp64_result_;
  float *fp32_result_;
  int *finite_result_;
  bool breakdown_injected_;
};
#endif

} // namespace

static void fields_to_array(const fields &f, complex<realnum> *x) {
  for (int i = 0; i < f.num_chunks; ++i)
    if (f.chunks[i]->is_mine())
      gpu::detail::sync_resident_cache_for_owner(f.chunks[i]);

  size_t ix = 0;
  for (int i = 0; i < f.num_chunks; i++)
    if (f.chunks[i]->is_mine()) FOR_COMPONENTS(c) {
        if (is_D(c) || is_B(c)) {
          realnum *fr, *fi;
#define COPY_FROM_FIELD(fld)                                                                       \
  if ((fr = f.chunks[i]->fld[0]) && (fi = f.chunks[i]->fld[1]))                                    \
    LOOP_OVER_VOL_OWNED(f.chunks[i]->gv, c, idx)                                                   \
  x[ix++] = complex<double>(fr[idx], fi[idx]);
          COPY_FROM_FIELD(f[c]);
          COPY_FROM_FIELD(f_u[c]);
          COPY_FROM_FIELD(f_cond[c]);
          COPY_FROM_FIELD(f_bfast[c]);
          component c2 = field_type_component(is_D(c) ? E_stuff : H_stuff, c);
          COPY_FROM_FIELD(f_w[c2]);
          if (f.chunks[i]->f_w[c2][0]) COPY_FROM_FIELD(f[c2]);
#undef COPY_FROM_FIELD
        }
      }
}

static void array_to_fields(const complex<realnum> *x, fields &f) {
  // The Krylov solver writes the canonical host arrays directly. Preserve
  // device-authoritative values first.  The field pointers and grid topology
  // do not change, so keep their resident allocations and address-bearing
  // curl/boundary plans; only arrays actually overwritten below need upload
  // invalidation.
  for (int i = 0; i < f.num_chunks; ++i)
    if (f.chunks[i]->is_mine())
      gpu::detail::sync_resident_cache_for_owner(f.chunks[i]);

  // Validate every resident array before writing any host field.  If one
  // chunk is unexpectedly active or device authoritative, this loop throws
  // before the first host value changes.  Invalidation itself is harmless if
  // a later chunk rejects the transaction: it only causes a conservative
  // host-to-device refresh on the next use.
  for (int i = 0; i < f.num_chunks; ++i)
    if (f.chunks[i]->is_mine()) {
      std::vector<const void *> host_writes;
      FOR_COMPONENTS(c) {
        if (is_D(c) || is_B(c)) {
          realnum *fr;
          realnum *fi;
#define COLLECT_FIELD_WRITE(fld)                                                                  \
  do {                                                                                            \
    if ((fr = f.chunks[i]->fld[0]) && (fi = f.chunks[i]->fld[1])) {                              \
      host_writes.push_back(fr);                                                                  \
      host_writes.push_back(fi);                                                                  \
    }                                                                                             \
  } while (false)
          COLLECT_FIELD_WRITE(f[c]);
          COLLECT_FIELD_WRITE(f_u[c]);
          COLLECT_FIELD_WRITE(f_cond[c]);
          COLLECT_FIELD_WRITE(f_bfast[c]);
          component c2 =
              field_type_component(is_D(c) ? E_stuff : H_stuff, c);
          COLLECT_FIELD_WRITE(f_w[c2]);
          if (f.chunks[i]->f_w[c2][0]) COLLECT_FIELD_WRITE(f[c2]);
#undef COLLECT_FIELD_WRITE
        }
      }
      gpu::detail::prepare_resident_host_writes_for_owner(
          f.chunks[i], host_writes.data(), host_writes.size());
    }

  size_t ix = 0;
  for (int i = 0; i < f.num_chunks; i++)
    if (f.chunks[i]->is_mine()) FOR_COMPONENTS(c) {
        if (is_D(c) || is_B(c)) {
          realnum *fr, *fi;
#define COPY_TO_FIELD(fld)                                                                         \
  do {                                                                                             \
    if ((fr = f.chunks[i]->fld[0]) && (fi = f.chunks[i]->fld[1])) {                               \
      LOOP_OVER_VOL_OWNED(f.chunks[i]->gv, c, idx) {                                               \
        fr[idx] = real(x[ix]);                                                                     \
        fi[idx] = imag(x[ix++]);                                                                   \
      }                                                                                            \
    }                                                                                              \
  } while (false)
          COPY_TO_FIELD(f[c]);
          COPY_TO_FIELD(f_u[c]);
          COPY_TO_FIELD(f_cond[c]);
          COPY_TO_FIELD(f_bfast[c]);
          component c2 = field_type_component(is_D(c) ? E_stuff : H_stuff, c);
          COPY_TO_FIELD(f_w[c2]);
          if (f.chunks[i]->f_w[c2][0]) COPY_TO_FIELD(f[c2]);
#undef COPY_TO_FIELD
        }
      }

  f.step_boundaries(D_stuff);
  f.update_eh(E_stuff, true);
  f.step_boundaries(E_stuff);

  /* done in f.step before updating D:
  f.step_boundaries(B_stuff);
  f.update_eh(H_stuff);
  f.step_boundaries(H_stuff); */
}

typedef struct {
  size_t n;
  fields *f;
  complex<double> iomega;
} fieldop_data;

#if MEEP_HAVE_CUDA
static std::vector<gpu::detail::cw_field_vector_segment_fp32>
build_resident_cw_vector_segments(fields &f, size_t complex_count) {
  std::vector<gpu::detail::cw_field_vector_segment_fp32> segments;
  size_t complex_offset = 0;
  for (int chunk_index = 0; chunk_index < f.num_chunks; ++chunk_index)
    if (f.chunks[chunk_index]->is_mine()) FOR_COMPONENTS(c) {
        fields_chunk *chunk = f.chunks[chunk_index];
        if (!is_D(c) && !is_B(c)) continue;
        realnum *field_real = nullptr;
        realnum *field_imaginary = nullptr;
#define APPEND_RESIDENT_CW_FIELD(fld)                                      \
  do {                                                                      \
    field_real = chunk->fld[0];                                             \
    field_imaginary = chunk->fld[1];                                        \
    if (field_real && field_imaginary) {                                    \
      const size_t point_count = chunk->gv.nowned(c);                       \
      segments.push_back(                                                   \
          {chunk, reinterpret_cast<float *>(field_real),                    \
           reinterpret_cast<float *>(field_imaginary), chunk->gv.ntot(),    \
           gpu::detail::make_index_space_fp32(                              \
               chunk->gv, chunk->gv.little_owned_corner(c),                 \
               chunk->gv.big_corner()),                                    \
           point_count, complex_offset});                                   \
      if (point_count > complex_count - complex_offset)                     \
        throw std::logic_error(                                             \
            "CUDA CW field layout exceeds the packed vector");             \
      complex_offset += point_count;                                        \
    }                                                                       \
  } while (false)
        APPEND_RESIDENT_CW_FIELD(f[c]);
        APPEND_RESIDENT_CW_FIELD(f_u[c]);
        APPEND_RESIDENT_CW_FIELD(f_cond[c]);
        APPEND_RESIDENT_CW_FIELD(f_bfast[c]);
        const component c2 =
            field_type_component(is_D(c) ? E_stuff : H_stuff, c);
        APPEND_RESIDENT_CW_FIELD(f_w[c2]);
        if (chunk->f_w[c2][0]) APPEND_RESIDENT_CW_FIELD(f[c2]);
#undef APPEND_RESIDENT_CW_FIELD
      }
  if (complex_offset != complex_count || segments.empty())
    throw std::logic_error(
        "CUDA CW field layout does not match the packed vector");
  return segments;
}

typedef struct {
  fields *f;
  gpu::detail::resident_cw_vector_plan *plan;
  complex<realnum> iomega;
} cuda_fieldop_data;

static void cuda_fieldop(const realnum *x, realnum *y, void *data_) {
  cuda_fieldop_data *data = static_cast<cuda_fieldop_data *>(data_);
  gpu::detail::scatter_resident_cw_vector_fp32(
      data->plan, reinterpret_cast<const float *>(x));
  data->f->step_boundaries(D_stuff);
  data->f->update_eh(E_stuff, true);
  data->f->step_boundaries(E_stuff);
  data->f->step();
  gpu::detail::gather_resident_cw_field_operator_fp32(
      data->plan, reinterpret_cast<const float *>(x),
      reinterpret_cast<float *>(y),
      static_cast<float>(1.0 / data->f->dt),
      {real(data->iomega), imag(data->iomega)});
}
#endif

static void fieldop(const realnum *xr, realnum *yr, void *data_) {
  const complex<realnum> *x = reinterpret_cast<const complex<realnum> *>(xr);
  complex<realnum> *y = reinterpret_cast<complex<realnum> *>(yr);
  fieldop_data *data = (fieldop_data *)data_;
  array_to_fields(x, *data->f);
  data->f->step();
  fields_to_array(*data->f, y);
  size_t n = data->n;
  realnum dt_inv = 1.0 / data->f->dt;
  complex<realnum> iomega = complex<realnum>(real(data->iomega), imag(data->iomega));
  for (size_t i = 0; i < n; ++i)
    y[i] = (y[i] - x[i]) * dt_inv + iomega * x[i];
}

#if MEEP_HAVE_CUDA
static bool use_resident_cw_solver(const fields &f,
                                   const complex<double> *) {
  const char *value = std::getenv("MEEP_GPU_CW_SOLVER");
  const std::string requested = value && *value ? value : "auto";
  if (requested != "auto" && requested != "host" &&
      requested != "resident")
    throw std::invalid_argument(
        "invalid MEEP_GPU_CW_SOLVER='" + requested +
        "' (expected auto, host, or resident)");
  if (requested == "host") return false;
  const bool fp32_build = sizeof(realnum) == sizeof(float);
  const bool cuda_selected = f.gpu_cuda_execution_selected();
  const bool supported = !f.fluxes;
  if (requested == "auto")
    return fp32_build && cuda_selected && supported;
  if (!fp32_build)
    throw std::runtime_error(
        "resident CUDA CW solver requires an FP32 build");
  if (!cuda_selected)
    throw std::runtime_error(
        "resident CUDA CW solver requires a CUDA-selected fields owner");
  if (f.fluxes)
    throw std::runtime_error(
        "resident CUDA CW solver does not yet support legacy flux monitors");
  return true;
}

static int solve_cw_resident_cuda(
    fields &f, int L, size_t N, realnum *host_x,
    const realnum *host_b, double tol, int *iters, size_t nwork,
    bool quiet, complex<double> iomega, int restart_interval,
    bool publish_finite_breakdown_candidate,
    bool inject_finite_breakdown_for_testing,
    bool *candidate_published) {
  if (N == 0 || N % 2 != 0 || !host_x || !host_b || !iters)
    throw std::invalid_argument(
        "resident CUDA CW solver received an invalid workspace");
  if (N > std::numeric_limits<size_t>::max() / 2 ||
      nwork > std::numeric_limits<size_t>::max() - 2 * N)
    throw std::overflow_error("resident CUDA CW workspace size overflow");
  std::unique_ptr<cuda_cw_workspace> workspace;
  std::vector<std::unique_ptr<gpu::detail::resident_curl_session> >
      sessions(static_cast<size_t>(f.num_chunks));
  resident_cw_plan_guard plan(nullptr);
  std::unique_ptr<cuda_bicgstab_vector_ops> vector_ops;
  std::exception_ptr local_setup_failure;
  try {
    workspace.reset(new cuda_cw_workspace(nwork + 2 * N));
    if (am_master() &&
        std::getenv("MEEP_GPU_TEST_CW_SETUP_RANK_FAILURE"))
      throw std::runtime_error(
          "injected rank-local resident CW setup failure");
    float *device_work = workspace->values();
    meep_cuda::copy_to_device(
        device_work + nwork, host_x, N * sizeof(realnum));
    meep_cuda::copy_to_device(
        device_work + nwork + N, host_b, N * sizeof(realnum));
    for (int chunk_index = 0; chunk_index < f.num_chunks; ++chunk_index)
      if (f.chunks[chunk_index]->is_mine())
        sessions[static_cast<size_t>(chunk_index)].reset(
            new gpu::detail::resident_curl_session(
                f.chunks[chunk_index], true));
    const std::vector<gpu::detail::cw_field_vector_segment_fp32> segments =
        build_resident_cw_vector_segments(f, N / 2);
    plan.reset(gpu::detail::create_resident_cw_vector_plan(
        segments.data(), segments.size(), N / 2));
    vector_ops.reset(new cuda_bicgstab_vector_ops());
  }
  catch (...) { local_setup_failure = std::current_exception(); }

  // No rank may enter Krylov collectives until every rank has completed all
  // rank-local CUDA allocations and resident-plan construction.
  if (!and_to_all(!local_setup_failure)) {
    if (local_setup_failure) std::rethrow_exception(local_setup_failure);
    throw std::runtime_error(
        "resident CUDA CW setup failed on another MPI rank");
  }

  float *device_work = workspace->values();
  float *device_x = device_work + nwork;
  float *device_b = device_x + N;
  cuda_fieldop_data operator_data = {
      &f, plan.get(),
      complex<realnum>(real(iomega), imag(iomega))};
  int result = 0;
  try {
    result = static_cast<int>(
        restart_interval > 0
            ? bicgstabL_restarted_with_vector_ops(
                  L, N, reinterpret_cast<realnum *>(device_x), cuda_fieldop,
                  &operator_data,
                  reinterpret_cast<const realnum *>(device_b), tol, iters,
                  reinterpret_cast<realnum *>(device_work), quiet,
                  restart_interval, *vector_ops)
            : bicgstabL_with_vector_ops(
                  L, N, reinterpret_cast<realnum *>(device_x), cuda_fieldop,
                  &operator_data,
                  reinterpret_cast<const realnum *>(device_b), tol, iters,
                  reinterpret_cast<realnum *>(device_work), quiet,
                  *vector_ops));
  }
  catch (const std::exception &error) {
    std::fprintf(stderr,
                 "meep: resident CUDA CW solver failed on rank %d: %s\n",
                 my_rank(), error.what());
    std::fflush(stderr);
    throw;
  }
  catch (...) {
    std::fprintf(stderr,
                 "meep: resident CUDA CW solver failed on rank %d with an "
                 "unknown exception\n",
                 my_rank());
    std::fflush(stderr);
    throw;
  }

  if (candidate_published) *candidate_published = false;
  if (inject_finite_breakdown_for_testing) result = -1;

  // A max-iteration exit can occur immediately after a vector update. Verify
  // that the candidate remains finite before publishing it. Shift-and-invert
  // may additionally consume a finite, nonzero initial breakdown candidate,
  // matching the host solver's established FP32 recovery semantics. Ordinary
  // CW solves and later eigen iterations retain the safer pre-solve field.
  if (result >= 0 || publish_finite_breakdown_candidate) {
    double candidate_norm = 0.0;
    try {
      candidate_norm = vector_ops->norm2(
          N, reinterpret_cast<const realnum *>(device_x));
    }
    catch (const bicgstab_numerical_breakdown &) {
      result = -1;
    }
    const bool publish = result >= 0 ||
                         (publish_finite_breakdown_candidate &&
                          candidate_norm > 0.0);
    if (publish) {
      meep_cuda::copy_to_host(host_x, device_x, N * sizeof(realnum));
      if (candidate_published) *candidate_published = true;
    }
  }
  for (auto &session : sessions)
    if (session) session->finish(false);
  return result;
}
#else
static bool use_resident_cw_solver(const fields &,
                                   const complex<double> *) {
  const char *value = std::getenv("MEEP_GPU_CW_SOLVER");
  if (value && std::string(value) == "resident")
    throw std::runtime_error(
        "resident CUDA CW solver called in a CPU-only build");
  if (value && *value && std::string(value) != "auto" &&
      std::string(value) != "host")
    throw std::invalid_argument(
        std::string("invalid MEEP_GPU_CW_SOLVER='") + value +
        "' (expected auto, host, or resident)");
  return false;
}

static int solve_cw_resident_cuda(
    fields &, int, size_t, realnum *, const realnum *, double, int *,
    size_t, bool, complex<double>, int, bool, bool, bool *) {
  throw std::runtime_error(
      "resident CUDA CW solver called in a CPU-only build");
}
#endif

static int cw_reliable_restart_interval() {
  const char *value = std::getenv("MEEP_GPU_CW_RELIABLE_RESTART");
  if (!value || !*value || std::string(value) == "auto")
    // A short interval is robust for small, well-conditioned fixtures but
    // repeatedly discards the high-order recurrence on resonant FP32
    // systems.  One hundred iterations still bounds recursive-residual
    // drift while allowing BiCGSTAB-L to traverse those spectra.
    return sizeof(realnum) == sizeof(float) ? 100 : 0;
  if (std::string(value) == "off" || std::string(value) == "0")
    return 0;
  errno = 0;
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (errno || !end || *end || parsed < 1 ||
      parsed > std::numeric_limits<int>::max())
    throw std::invalid_argument(
        std::string("invalid MEEP_GPU_CW_RELIABLE_RESTART='") + value +
        "' (expected auto, off, 0, or a positive iteration interval)");
  return static_cast<int>(parsed);
}

static bool cw_true_residual_diagnostic_enabled() {
  const char *value =
      std::getenv("MEEP_GPU_CW_DIAGNOSTIC_TRUE_RESIDUAL");
  if (!value || !*value || std::string(value) == "0" ||
      std::string(value) == "off" || std::string(value) == "false")
    return false;
  if (std::string(value) == "1" || std::string(value) == "on" ||
      std::string(value) == "true")
    return true;
  throw std::invalid_argument(
      std::string("invalid MEEP_GPU_CW_DIAGNOSTIC_TRUE_RESIDUAL='") +
      value + "' (expected 0/1, off/on, or false/true)");
}

static bool cw_test_boolean(const char *setting) {
  const char *value = std::getenv(setting);
  if (!value || !*value || std::string(value) == "0") return false;
  if (std::string(value) == "1") return true;
  throw std::invalid_argument(
      std::string("invalid ") + setting + "='" + value +
      "' (expected 0 or 1)");
}

static bool collectively_agree_on_cw_boolean(bool local_choice,
                                             const char *setting) {
  const bool any_rank = or_to_all(local_choice);
  const bool every_rank = and_to_all(local_choice);
  if (any_rank != every_rank)
    throw std::runtime_error(
        std::string(setting) + " differs across MPI ranks");
  return every_rank;
}

static int collectively_agree_on_cw_restart_interval(int local_interval) {
  const int minimum = min_to_all(local_interval);
  const int maximum = max_to_all(local_interval);
  if (minimum != maximum)
    throw std::runtime_error(
        "MEEP_GPU_CW_RELIABLE_RESTART differs across MPI ranks");
  return minimum;
}

static int collectively_agree_on_cw_order(int local_order) {
  const int minimum = min_to_all(local_order);
  const int maximum = max_to_all(local_order);
  if (minimum != maximum)
    throw std::runtime_error(
        "solve_cw BiCGSTAB-L order differs across MPI ranks");
  return minimum;
}

// Rayleigh-quotient estimate <x,Ax>/<x,x> for eigenfrequency given approximate eigenvector x
// (length n), overwriting x with Ax and b with x/|x|.
static complex<double> estimate_eigfreq(complex<realnum> *b, complex<realnum> *x, size_t n,
                                        fieldop_data *data) {
  memcpy(b, x, n * sizeof(complex<realnum>));
  fieldop(reinterpret_cast<realnum *>(b), reinterpret_cast<realnum *>(x), (void *)data);
  complex<double> bdotx(0, 0);
  double bnorm2 = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const complex<realnum> bi = b[i];
    bnorm2 += real(bi) * real(bi) + imag(bi) * imag(bi);
    const complex<realnum> bx = conj(bi) * x[i];
    bdotx += complex<double>(real(bx), imag(bx));
  }
  bnorm2 = sum_to_all(bnorm2);
  bdotx = sum_to_all(bdotx);
  complex<double> quotient;
  if (std::isfinite(bnorm2) && bnorm2 > 0.0 &&
      std::isfinite(real(bdotx)) && std::isfinite(imag(bdotx))) {
    // Retain Meep's established arithmetic in the ordinary range so the
    // shift-and-invert convergence trajectory and oracle remain unchanged.
    const double bnorminv = 1.0 / sqrt(bnorm2);
    for (size_t i = 0; i < n; ++i) b[i] *= bnorminv;
    quotient = bdotx / bnorm2;
  }
  else {
    // Extreme finite FP32 vectors can under/overflow their unscaled products.
    // Recompute the Rayleigh quotient after global-max scaling in that rare
    // case instead of dividing by zero or publishing a non-finite estimate.
    double local_maximum = 0.0;
    for (size_t i = 0; i < n; ++i)
      local_maximum = std::max(
          local_maximum,
          std::max(std::abs(static_cast<double>(real(b[i]))),
                   std::abs(static_cast<double>(imag(b[i])))));
    const double global_maximum = max_to_all(local_maximum);
    complex<double> scaled_bdotx(0, 0);
    double scaled_bnorm2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
      const complex<double> scaled_b(
          static_cast<double>(real(b[i])) / global_maximum,
          static_cast<double>(imag(b[i])) / global_maximum);
      const complex<double> scaled_x(
          static_cast<double>(real(x[i])) / global_maximum,
          static_cast<double>(imag(x[i])) / global_maximum);
      scaled_bnorm2 += norm(scaled_b);
      scaled_bdotx += conj(scaled_b) * scaled_x;
    }
    scaled_bnorm2 = sum_to_all(scaled_bnorm2);
    scaled_bdotx = sum_to_all(scaled_bdotx);
    const double scaled_norm = sqrt(scaled_bnorm2);
    for (size_t i = 0; i < n; ++i)
      b[i] = complex<realnum>(
          static_cast<double>(real(b[i])) / global_maximum / scaled_norm,
          static_cast<double>(imag(b[i])) / global_maximum / scaled_norm);
    quotient = scaled_bdotx / scaled_bnorm2;
  }
  complex<double> iomega =
      data->iomega - quotient; // unshifted eigenvalue
  // now, invert: iomega = (1 - exp(-i * (2 * pi * frequency) * dt)) / dt)
  // to get frequency = log(1 - iomega * dt) / (-2 pi i * dt)
  double dt = data->f->dt;
  return log(1.0 - iomega * dt) / complex<double>(0, -2 * pi * dt);
}

static bool finite_nonzero_cw_vector(const complex<realnum> *values,
                                     size_t n) {
  bool local_finite = true;
  double local_maximum = 0.0;
  for (size_t index = 0; index < n; ++index) {
    const double real_value = static_cast<double>(real(values[index]));
    const double imag_value = static_cast<double>(imag(values[index]));
    local_finite = local_finite && std::isfinite(real_value) &&
                   std::isfinite(imag_value);
    local_maximum = std::max(
        local_maximum, std::max(std::abs(real_value), std::abs(imag_value)));
  }
  const bool globally_finite = and_to_all(local_finite);
  const double global_maximum = max_to_all(local_maximum);
  return globally_finite && std::isfinite(global_maximum) &&
         global_maximum > 0.0;
}

/* Solve for the CW (constant frequency) field response at the given
   frequency to the sources (with amplitude given by the current sources
   at the current time).  The solver halts at a fractional convergence
   of tol, or when maxiters is reached, or when convergence fails;
   returns true if convergence succeeds and false if it fails.

   The parameter L determines the order of the iterative algorithm
   that is used.  L should always be positive and should normally be
   >= 2.  Larger values of L will often lead to faster convergence, at
   the expense of more memory and more work per iteration.

   If the optional argument eigfreq is non-NULL, then the solver is used for a
   shift-and-invert power iteration to find the closest eigenfrequency and
   eigenvector to frequency: the solver is iterated up to eigiters times,
   or until the estimated eigenfreq stops changing by <= eigtol (relative). */
bool fields::solve_cw(double tol, int maxiters, complex<double> frequency, int L,
                      complex<double> *eigfreq, double eigtol, int eigiters) {
  if (is_real) meep::abort("solve_cw is incompatible with use_real_fields()");
  if (L < 1) meep::abort("solve_cw called with L = %d < 1", L);
  distributed_cw_failure_guard distributed_failure_guard(
      count_processors() > 1);
  if (am_master() && std::getenv("MEEP_GPU_TEST_CW_RANK_FAILURE"))
    throw std::runtime_error(
        "injected rank-local solve_cw failure for MPI deadlock regression");
  const int solver_L = collectively_agree_on_cw_order(L);
  int tsave = t; // save time (gets incremented by iterations)
  int iters;

  solve_cw_state_guard solve_state(
      *this, tsave, 2 * pi * frequency);

  step(); // step once to make sure everything is allocated

  size_t N = 0; // size of linear system (on this processor, at least)
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) {
      FOR_COMPONENTS(c) {
        if (chunks[i]->f[c][0] && (is_D(c) || is_B(c))) {
          component c2 = field_type_component(is_D(c) ? E_stuff : H_stuff, c);
          /* unknowns are just D and B in non-PML regions, but in PML
             regions the E, U, W, and C fields are also unknowns (in
             principle, we might be able to compute these extra fields
             in frequency domain via scalinb by the appropriate s
             factors, rather than storing them, but I had some
             problems getting that working) */
          N += 2 * chunks[i]->gv.nowned(c) *
               (1 + (chunks[i]->f_u[c][0] != NULL) + (chunks[i]->f_w[c2][0] != NULL) * 2 +
                (chunks[i]->f_cond[c][0] != NULL) + (chunks[i]->f_bfast[c][0] != NULL));
        }
      }
    }

  const bool resident_cuda_solver = collectively_agree_on_cw_boolean(
      use_resident_cw_solver(*this, eigfreq), "MEEP_GPU_CW_SOLVER");
  iters = maxiters;
  size_t nwork =
      (size_t)bicgstabL(solver_L, N, 0, 0, 0, 0, tol, &iters, 0, true);
  const size_t eigfreq_state_size = eigfreq ? N : 0;
  if (N > std::numeric_limits<size_t>::max() / 2 ||
      nwork > std::numeric_limits<size_t>::max() - 2 * N ||
      nwork + 2 * N >
          std::numeric_limits<size_t>::max() - eigfreq_state_size)
    throw std::overflow_error("solve_cw host workspace size overflow");
  std::unique_ptr<realnum[]> work_owner(
      new realnum[nwork + 2 * N + eigfreq_state_size]);
  realnum *work = work_owner.get();
  complex<realnum> *x = reinterpret_cast<complex<realnum> *>(work + nwork);
  complex<realnum> *b = reinterpret_cast<complex<realnum> *>(work + nwork + N);
  complex<realnum> *last_good = eigfreq
                                    ? reinterpret_cast<complex<realnum> *>(
                                          work + nwork + 2 * N)
                                    : nullptr;

  fields_to_array(*this, x); // initial guess = initial fields
  if (last_good) memcpy(last_good, x, N * sizeof(realnum));

  // get J amplitudes from current time step
  zero_fields(); // note that we've saved the fields in x above
  calc_sources(time());
  step_source(B_stuff, true);
  step_boundaries(B_stuff);
  update_eh(H_stuff);
  calc_sources(time() + 0.5 * dt);
  step_source(D_stuff, true);
  step_boundaries(D_stuff);
  update_eh(E_stuff);
  fields_to_array(*this, b);
  double mdt_inv = -1.0 / dt;
  for (size_t i = 0; i < N / 2; ++i)
    b[i] *= mdt_inv;
  {
    double bmax = 0;
    for (size_t i = 0; i < N / 2; ++i) {
      double babs = abs(b[i]);
      if (babs > bmax) bmax = babs;
    }
    am_now_working_on(MpiAllTime);
    if (max_to_all(bmax) == 0.0) meep::abort("zero current amplitudes in solve_cw");
    finished_working();
  }

  fieldop_data data;
  data.f = this;
  data.n = N / 2;
  data.iomega = ((1.0 - exp(complex<double>(0., -1.) * (2 * pi * frequency) * dt)) * (1.0 / dt));
  iters = maxiters;

  const int reliable_restart_interval =
      collectively_agree_on_cw_restart_interval(
          eigfreq ? 0 : cw_reliable_restart_interval());
  const bool diagnose_true_residual = collectively_agree_on_cw_boolean(
      !eigfreq && cw_true_residual_diagnostic_enabled(),
      "MEEP_GPU_CW_DIAGNOSTIC_TRUE_RESIDUAL");
  const bool inject_eigen_inner_failure =
      collectively_agree_on_cw_boolean(
          eigfreq && cw_test_boolean(
                         "MEEP_GPU_TEST_CW_EIGEN_INNER_FAILURE"),
          "MEEP_GPU_TEST_CW_EIGEN_INNER_FAILURE");
  const bool inject_initial_resident_breakdown =
      collectively_agree_on_cw_boolean(
          eigfreq && cw_test_boolean(
                         "MEEP_GPU_TEST_CW_EIGEN_INITIAL_BREAKDOWN"),
          "MEEP_GPU_TEST_CW_EIGEN_INITIAL_BREAKDOWN");
  const auto solve_linear_system =
      [&](complex<realnum> *solution, const complex<realnum> *rhs,
          int *iteration_count, bool publish_finite_breakdown_candidate,
          bool inject_finite_breakdown,
          bool *candidate_published) -> int {
    if (resident_cuda_solver) {
      // Shift-and-invert is sensitive to periodically discarding a valid
      // high-order recurrence. Use the entire remaining budget as one batch:
      // apparent recursive convergence is still independently checked with
      // b-Ax, and a false convergence restarts only from the current device
      // solution within the original iteration limit. Ordinary CW retains
      // its configured periodic reliable-restart policy.
      const int resident_restart_interval =
          eigfreq ? std::max(1, *iteration_count)
                  : reliable_restart_interval;
      return solve_cw_resident_cuda(
          *this, solver_L, N, reinterpret_cast<realnum *>(solution),
          reinterpret_cast<const realnum *>(rhs), tol, iteration_count,
          nwork, verbosity == 0, data.iomega,
          resident_restart_interval, publish_finite_breakdown_candidate,
          inject_finite_breakdown, candidate_published);
    }
    if (candidate_published) *candidate_published = true;
    return static_cast<int>(
        reliable_restart_interval > 0
            ? bicgstabL_restarted(
                  solver_L, N, reinterpret_cast<realnum *>(solution),
                  fieldop, &data, reinterpret_cast<const realnum *>(rhs),
                  tol, iteration_count, work, verbosity == 0,
                  reliable_restart_interval)
            : bicgstabL(
                  solver_L, N, reinterpret_cast<realnum *>(solution),
                  fieldop, &data, reinterpret_cast<const realnum *>(rhs),
                  tol, iteration_count, work, verbosity == 0));
  };
  bool initial_candidate_published = false;
  int ierr = solve_linear_system(
      x, b, &iters, eigfreq != nullptr,
      resident_cuda_solver && inject_initial_resident_breakdown,
      &initial_candidate_published);

  if (diagnose_true_residual) {
    // Re-evaluate b-Ax with the ordinary field operator.  BiCGSTAB-L's
    // recursively updated residual can drift in FP32; this diagnostic keeps
    // that distinct from a packed-field or resident-operator error.
    fieldop(reinterpret_cast<realnum *>(x), work, &data);
    double local_residual_squared = 0.0;
    double local_rhs_squared = 0.0;
    for (size_t index = 0; index < N; ++index) {
      const double residual = static_cast<double>(
          reinterpret_cast<realnum *>(b)[index]) -
                              static_cast<double>(work[index]);
      const double rhs = static_cast<double>(
          reinterpret_cast<realnum *>(b)[index]);
      local_residual_squared += residual * residual;
      local_rhs_squared += rhs * rhs;
    }
    const double residual_squared = sum_to_all(local_residual_squared);
    const double rhs_squared = sum_to_all(local_rhs_squared);
    const double relative_true_residual = std::sqrt(
        residual_squared / std::max(rhs_squared, 1e-300));
    master_printf("CW diagnostic true relative residual = %.17g\n",
                  relative_true_residual);
  }

  if (ierr < 0 && resident_cuda_solver)
    master_printf(
        "WARNING: resident FP32 solve_cw encountered a numerical breakdown "
        "at requested BiCGSTAB-L order %d. A finite nonzero initial "
        "shift-and-invert candidate may be refined; other calls retain the "
        "pre-solve field. Retry with a different L/reliable-restart interval "
        "or MEEP_GPU_CW_SOLVER=host.\n",
        solver_L);

  if (verbosity > 0) {
    long long approximate_timesteps =
        static_cast<long long>(iters) * 2 * solver_L;
    if (reliable_restart_interval > 0 && iters > 0) {
      const long long batches =
          (static_cast<long long>(iters) + reliable_restart_interval - 1) /
          reliable_restart_interval;
      approximate_timesteps += 2 * batches;
    }
    master_printf(
        "Finished solve_cw after %d CG iters (~ %lld timesteps including "
        "true-residual checks).\n",
        iters, approximate_timesteps);
    if (ierr) master_printf(" -- CONVERGENCE FAILURE (%d) in solve_cw!\n", ierr);
  }

  // do additional shift-and-invert iterations to find eigenfrequency
  if (eigfreq) {
    // FP32 BiCGSTAB-L can flag recursive-residual breakdown after producing
    // a finite useful first approximation.  Preserve the established
    // shift-and-invert recovery semantics: a successful later inner solve
    // validates that seed.  Unlike the old shadowed status, any *later*
    // inner-solve failure invalidates the published eigenfrequency.
    bool eigfreq_valid = initial_candidate_published &&
                         finite_nonzero_cw_vector(x, data.n);
    bool completed_inner_solve = false;
    if (eigfreq_valid) {
      *eigfreq = estimate_eigfreq(b, x, data.n, &data);
      eigfreq_valid = std::isfinite(real(*eigfreq)) &&
                      std::isfinite(imag(*eigfreq)) &&
                      finite_nonzero_cw_vector(b, data.n);
    }
    if (eigfreq_valid) {
      memcpy(last_good, b, N * sizeof(realnum));
    }
    else {
      ierr = ierr == 0 ? 1 : ierr;
      memcpy(x, last_good, N * sizeof(realnum));
    }
    if (eigfreq_valid && verbosity > 0) {
      master_printf("Initial eigen-frequency estimate = %g%+gi\n",
                    real(*eigfreq), imag(*eigfreq));
    }
    for (int eigiter = 0; eigfreq_valid && eigiter < eigiters; ++eigiter) {
      iters = maxiters;
      const int eig_ierr =
          inject_eigen_inner_failure && eigiter == 0
              ? 1
              : solve_linear_system(
                    x, b, &iters, false, false, nullptr);
      if (eig_ierr != 0) {
        // Do not estimate or publish an eigenfrequency from a failed inner
        // solve.  The old block-local `ierr` shadowed the return status and
        // could therefore report success here.
        ierr = eig_ierr;
        eigfreq_valid = false;
        memcpy(x, last_good, N * sizeof(realnum));
        if (verbosity > 0)
          master_printf(
              "Eigensolver step %d: %d CG iters. -- CONVERGENCE "
              "FAILURE (%d) in solve_cw!\n",
              eigiter + 1, iters, eig_ierr);
        break;
      }
      if (!finite_nonzero_cw_vector(x, data.n)) {
        ierr = 1;
        eigfreq_valid = false;
        memcpy(x, last_good, N * sizeof(realnum));
        if (verbosity > 0)
          master_printf(
              "Eigensolver step %d produced a non-finite or zero linear "
              "solution. -- CONVERGENCE FAILURE in solve_cw!\n",
              eigiter + 1);
        break;
      }
      completed_inner_solve = true;
      const complex<double> newfreq =
          estimate_eigfreq(b, x, data.n, &data);
      if (!std::isfinite(real(newfreq)) ||
          !std::isfinite(imag(newfreq)) ||
          !finite_nonzero_cw_vector(b, data.n)) {
        ierr = 1;
        eigfreq_valid = false;
        memcpy(x, last_good, N * sizeof(realnum));
        if (verbosity > 0)
          master_printf(
              "Eigensolver step %d produced a non-finite or zero "
              "eigenpair candidate. -- CONVERGENCE FAILURE in solve_cw!\n",
              eigiter + 1);
        break;
      }
      const complex<double> dfreq = newfreq - *eigfreq;
      if (verbosity > 0)
        master_printf(
            "Eigensolver step %d: %d CG iters, freq = %g%+gi "
            "(change = %g%+gi).\n",
            eigiter + 1, iters, real(newfreq), imag(newfreq),
            real(dfreq), imag(dfreq));
      *eigfreq = newfreq;
      memcpy(last_good, b, N * sizeof(realnum));
      if (abs(dfreq) <= eigtol * abs(newfreq)) break; // converged
    }
    if (eigfreq_valid && (ierr == 0 || completed_inner_solve)) {
      ierr = 0;
      memcpy(x, b, N * sizeof(realnum));
    }
    else {
      ierr = ierr == 0 ? 1 : ierr;
      memcpy(x, last_good, N * sizeof(realnum));
      *eigfreq = complex<double>(
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN());
    }
  }

  array_to_fields(x, *this);
  step(); // ensure H/B are updated and synced with E/D

  solve_state.restore();
  update_dfts();

  return !ierr;
}

/* as solve_cw, but infers frequency from sources */
bool fields::solve_cw(double tol, int maxiters, int L, complex<double> *eigfreq, double eigtol,
                      int eigiters) {
  complex<double> freq = 0.0;
  for (src_time *s = sources; s; s = s->next) {
    complex<double> sf = s->frequency();
    if (sf != freq && freq != 0.0 && sf != 0.0)
      meep::abort("must pass frequency to solve_cw if sources do not agree");
    if (sf != 0.0) freq = sf;
  }
  if (freq == 0.0) meep::abort("must pass frequency to solve_cw if sources do not specify one");
  return solve_cw(tol, maxiters, freq, L, eigfreq, eigtol, eigiters);
}

} // namespace meep
