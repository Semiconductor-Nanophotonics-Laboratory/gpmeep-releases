/* Correctness test for the first integrated FP32 CUDA step_db curl path. */

#include <meep.hpp>
#include <meep/gpu.hpp>
#include "gpu_backend_internal.hpp"
#include "gpu_grid_index.hpp"
#include "bicgstab.hpp"
#include "meep_internals.hpp"
#include "meep_cuda/runtime.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <climits>
#include <complex>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <dirent.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <linux/fs.h>
#include <time.h>
#include <unistd.h>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

using namespace meep;

namespace meep {
// Friend-only test access keeps the production update implementation private
// while allowing the regression to exercise it without a full Maxwell step.
class fields_chunk_test_access {
public:
  static bool update_pols(fields_chunk &owner, field_type ft) {
    return owner.update_pols(ft);
  }
  static void step_db(fields &owner, field_type ft) {
    owner.step_db(ft);
  }
};

namespace gpu {
namespace detail {

// This class is the sole non-fields friend of the replay token and exists
// only in this integrated regression translation unit.
class boundary_descriptor_replay_token_test_access {
public:
  static boundary_descriptor_replay_token make(
      const void *identity, std::uint64_t generation) noexcept {
    return boundary_descriptor_replay_token(identity, generation);
  }
};

} // namespace detail
} // namespace gpu
} // namespace meep

namespace {

double vacuum(const vec &) { return 1.0; }
double lossy(const vec &) { return 0.17; }
double weak_chi2(const vec &) { return 0.012; }
double weak_chi3(const vec &) { return 0.006; }
double gyrotropic_sigma(const vec &) { return 0.08; }
double multilevel_sigma(const vec &) { return 0.045; }
std::complex<double> initialized_profile(const vec &point) {
  return std::complex<double>(0.03 * point.x(), -0.02 * point.y());
}

void require(bool condition, const std::string &message);

void require_saturating_boundary_topology_generation() {
  const std::uint64_t maximum =
      std::numeric_limits<std::uint64_t>::max();
  std::atomic<std::uint64_t> ordinary(0);
  require(
      gpu::detail::next_saturating_boundary_topology_generation(
          &ordinary) == 1 && ordinary.load() == 1,
      "boundary topology generation did not begin monotonically at one");

  std::atomic<std::uint64_t> saturated(maximum - 1);
  require(
      gpu::detail::next_saturating_boundary_topology_generation(
          &saturated) == maximum && saturated.load() == maximum,
      "boundary topology generation did not issue UINT64_MAX exactly once");
  require(
      gpu::detail::next_saturating_boundary_topology_generation(
          &saturated) == 0 &&
          gpu::detail::next_saturating_boundary_topology_generation(
              &saturated) == 0 &&
          saturated.load() == maximum,
      "saturated boundary topology generation wrapped or resumed replay");
  require(
      gpu::detail::next_saturating_boundary_topology_generation(nullptr) ==
          0,
      "null boundary topology generation state was replay eligible");
  if (am_master())
    std::cout
        << "PASS: boundary topology generation saturates permanently at UINT64_MAX\n";
}

void write_stderr_line_atomically(const std::string &line) {
  // POSIX guarantees that a pipe write no larger than PIPE_BUF (at least
  // 512 bytes) is not interleaved with another writer.  Open MPI forwards
  // these rank-local stderr records; split iostream insertions can otherwise
  // corrupt an exact per-rank qualification marker.
  if (line.find('\n') != std::string::npos || line.size() >= 512)
    throw std::runtime_error(
        "atomic stderr evidence must be one bounded line");
  const std::string payload = line + '\n';
  ssize_t count;
  do {
    count = write(STDERR_FILENO, payload.data(), payload.size());
  } while (count < 0 && errno == EINTR);
  if (count != static_cast<ssize_t>(payload.size()))
    throw std::runtime_error("could not atomically emit stderr evidence");
}

struct diagonal_bicgstab_data {
  const std::vector<realnum> *diagonal;
};

void diagonal_bicgstab_operator(const realnum *x, realnum *y,
                                void *opaque) {
  const diagonal_bicgstab_data *data =
      static_cast<const diagonal_bicgstab_data *>(opaque);
  for (std::size_t index = 0; index < data->diagonal->size(); ++index)
    y[index] = (*data->diagonal)[index] * x[index];
}

void require_restarted_bicgstab_boundaries() {
  const std::size_t count = 37;
  std::vector<realnum> diagonal(count);
  std::vector<realnum> rhs(count);
  std::vector<realnum> exact(count);
  for (std::size_t index = 0; index < count; ++index) {
    diagonal[index] = static_cast<realnum>(
        0.7 + 0.025 * static_cast<double>(index));
    rhs[index] = static_cast<realnum>(
        0.3 * std::sin(0.17 * static_cast<double>(index + 1)) + 0.2);
    exact[index] = rhs[index] / diagonal[index];
  }
  diagonal_bicgstab_data data = {&diagonal};
  int query_iterations = 0;
  const std::ptrdiff_t workspace_count = bicgstabL_restarted(
      2, count, exact.data(), diagonal_bicgstab_operator, &data,
      rhs.data(), 1e-6, &query_iterations, nullptr, true, 5);
  require(workspace_count > 0,
          "restarted BiCGSTAB-L workspace query failed");
  std::vector<realnum> workspace(
      static_cast<std::size_t>(workspace_count));

  std::vector<realnum> solution = exact;
  int iterations = 10;
  require(bicgstabL_restarted(
              2, count, solution.data(), diagonal_bicgstab_operator, &data,
              rhs.data(), 1e-6, &iterations, workspace.data(), true, 5) == 0 &&
              iterations == 0 && solution == exact,
          "restarted BiCGSTAB-L rejected or changed an exact initial solution");

  solution.assign(count, realnum(0));
  iterations = 200;
  require(bicgstabL_restarted(
              2, count, solution.data(), diagonal_bicgstab_operator, &data,
              rhs.data(), 1e-6, &iterations, workspace.data(), true, 5) == 0,
          "restarted BiCGSTAB-L diagonal solve did not converge");
  double residual_squared = 0.0;
  double rhs_squared = 0.0;
  for (std::size_t index = 0; index < count; ++index) {
    const double residual = static_cast<double>(rhs[index]) -
        static_cast<double>(diagonal[index]) *
        static_cast<double>(solution[index]);
    residual_squared += residual * residual;
    rhs_squared += static_cast<double>(rhs[index]) * rhs[index];
  }
  require(std::sqrt(residual_squared / rhs_squared) <= 1.05e-6,
          "restarted BiCGSTAB-L accepted a false true-residual convergence");

  auto rejects = [&](int *iteration_pointer, double tolerance,
                     int interval) {
    try {
      bicgstabL_restarted(
          2, count, solution.data(), diagonal_bicgstab_operator, &data,
          rhs.data(), tolerance, iteration_pointer, workspace.data(), true,
          interval);
      return false;
    }
    catch (const std::invalid_argument &) { return true; }
  };
  iterations = -1;
  require(rejects(nullptr, 1e-6, 5) &&
              rejects(&iterations, 1e-6, 5),
          "restarted BiCGSTAB-L accepted an invalid iteration limit");
  iterations = 10;
  require(rejects(&iterations, 1e-6, 0) &&
              rejects(&iterations,
                      std::numeric_limits<double>::quiet_NaN(), 5) &&
              rejects(&iterations, -1e-6, 5),
          "restarted BiCGSTAB-L accepted an invalid interval or tolerance");
}

void require_cuda_krylov_vector_primitives() {
#if MEEP_HAVE_CUDA
  const std::size_t count = 4099;
  std::vector<float> left(count);
  std::vector<float> output(count, 0.25f);
  std::vector<float> expected(count);
  for (std::size_t index = 0; index < count; ++index)
    left[index] = static_cast<float>(
        0.75 * std::sin(0.013 * static_cast<double>(index)) -
        0.1 * std::cos(0.031 * static_cast<double>(index)));

  float *device_left = nullptr;
  float *device_output = nullptr;
  float *device_copy = nullptr;
  double *device_fp64_workspace = nullptr;
  double *device_fp64 = nullptr;
  float *device_fp32 = nullptr;
  try {
    device_left = static_cast<float *>(
        meep_cuda::allocate_device_bytes(count * sizeof(float)));
    device_output = static_cast<float *>(
        meep_cuda::allocate_device_bytes(count * sizeof(float)));
    device_copy = static_cast<float *>(
        meep_cuda::allocate_device_bytes(count * sizeof(float)));
    device_fp64_workspace = static_cast<double *>(
        meep_cuda::allocate_device_bytes(
            (meep_cuda::vector_reduction_partial_capacity + 1) *
            sizeof(double)));
    device_fp64 = device_fp64_workspace +
        meep_cuda::vector_reduction_partial_capacity;
    device_fp32 = static_cast<float *>(
        meep_cuda::allocate_device_bytes(sizeof(float)));
    meep_cuda::copy_to_device(
        device_left, left.data(), count * sizeof(float));
    meep_cuda::vector_fill_fp32(device_output, count, 0.25f, 191);
    const double left_minus_scale = 0.375000000037;
    const double vector_scale = -0.750000000041;
    const double xpay_scale = 0.125000000029;
    meep_cuda::vector_left_minus_scale_fp32(
        device_output, device_left, count, left_minus_scale, 191);
    meep_cuda::vector_scale_fp32(
        device_output, count, vector_scale, 191);
    meep_cuda::vector_xpay_fp32(
        device_output, device_left, count, xpay_scale, 191);
    meep_cuda::vector_copy_fp32(device_copy, device_output, count);
    meep_cuda::copy_to_host(
        output.data(), device_copy, count * sizeof(float));

    for (std::size_t index = 0; index < count; ++index) {
      float value = static_cast<float>(
          static_cast<double>(left[index]) -
          left_minus_scale * static_cast<double>(0.25f));
      value = static_cast<float>(
          static_cast<double>(value) * vector_scale);
      expected[index] = static_cast<float>(
          static_cast<double>(value) +
          xpay_scale * static_cast<double>(left[index]));
      require(output[index] == expected[index],
              "CUDA Krylov elementwise primitive changed legacy host "
              "double-coefficient/FP32-store semantics");
    }

    meep_cuda::initialize_fp64_result(device_fp64);
    meep_cuda::vector_dot_fp32(
        device_output, device_left, count, device_fp64_workspace,
        meep_cuda::vector_reduction_partial_capacity, device_fp64, 191);
    double dot = 0.0;
    meep_cuda::copy_to_host(&dot, device_fp64, sizeof(dot));
    double expected_dot = 0.0;
    for (std::size_t index = 0; index < count; ++index)
      expected_dot += static_cast<double>(output[index] * left[index]);
    require(std::abs(dot - expected_dot) <=
                5e-13 * std::max(1.0, std::abs(expected_dot)),
            "CUDA Krylov FP64 dot reduction disagrees with FP32 products");
    for (int repetition = 0; repetition < 32; ++repetition) {
      meep_cuda::vector_dot_fp32(
          device_output, device_left, count, device_fp64_workspace,
          meep_cuda::vector_reduction_partial_capacity, device_fp64, 191);
      double repeated_dot = 0.0;
      meep_cuda::copy_to_host(
          &repeated_dot, device_fp64, sizeof(repeated_dot));
      require(repeated_dot == dot,
              "CUDA Krylov dot reduction is not bitwise repeatable");
    }

    meep_cuda::initialize_fp32_result(device_fp32);
    meep_cuda::vector_max_abs_fp32(
        device_output, count, device_fp32, nullptr, 191);
    float maximum = 0.0f;
    meep_cuda::copy_to_host(&maximum, device_fp32, sizeof(maximum));
    float expected_maximum = 0.0f;
    for (float value : output)
      expected_maximum = std::max(expected_maximum, std::abs(value));
    require(maximum == expected_maximum,
            "CUDA Krylov maximum reduction disagrees with FP32 reference");

    const double scale = 0.625000000033;
    meep_cuda::initialize_fp64_result(device_fp64);
    meep_cuda::vector_scaled_sum_squares_fp32(
        device_output, count, scale, device_fp64_workspace,
        meep_cuda::vector_reduction_partial_capacity, device_fp64, 191);
    double sum_squares = 0.0;
    meep_cuda::copy_to_host(
        &sum_squares, device_fp64, sizeof(sum_squares));
    double expected_sum_squares = 0.0;
    for (float value : output) {
      const double scaled = static_cast<double>(value) * scale;
      expected_sum_squares += scaled * scaled;
    }
    require(std::abs(sum_squares - expected_sum_squares) <=
                5e-13 * std::max(1.0, std::abs(expected_sum_squares)),
            "CUDA Krylov scaled norm reduction disagrees with legacy "
            "host FP64 scaling");
    for (int repetition = 0; repetition < 32; ++repetition) {
      meep_cuda::vector_scaled_sum_squares_fp32(
          device_output, count, scale, device_fp64_workspace,
          meep_cuda::vector_reduction_partial_capacity, device_fp64, 191);
      double repeated_sum_squares = 0.0;
      meep_cuda::copy_to_host(
          &repeated_sum_squares, device_fp64,
          sizeof(repeated_sum_squares));
      require(repeated_sum_squares == sum_squares,
              "CUDA Krylov scaled norm reduction is not bitwise repeatable");
    }
  }
  catch (...) {
    meep_cuda::free_device(device_fp32);
    meep_cuda::free_device(device_fp64_workspace);
    meep_cuda::free_device(device_copy);
    meep_cuda::free_device(device_output);
    meep_cuda::free_device(device_left);
    throw;
  }
  meep_cuda::free_device(device_fp32);
  meep_cuda::free_device(device_fp64_workspace);
  meep_cuda::free_device(device_copy);
  meep_cuda::free_device(device_output);
  meep_cuda::free_device(device_left);
#else
  throw std::logic_error(
      "CUDA Krylov vector primitive test requires a CUDA build");
#endif
}

void append_cw_test_segments(
    fields &f,
    std::vector<gpu::detail::cw_field_vector_segment_fp32> *segments,
    std::vector<float> *packed) {
#if MEEP_SINGLE
  if (!segments || !packed)
    throw std::invalid_argument(
        "CW packed-vector test outputs must be non-null");
  segments->clear();
  packed->clear();
  std::size_t complex_offset = 0;
  for (int chunk_index = 0; chunk_index < f.num_chunks; ++chunk_index)
    if (f.chunks[chunk_index]->is_mine()) FOR_COMPONENTS(c) {
        fields_chunk *chunk = f.chunks[chunk_index];
        if (!is_D(c) && !is_B(c)) continue;
        realnum *field_real = nullptr;
        realnum *field_imaginary = nullptr;
#define APPEND_CW_TEST_FIELD(fld)                                           \
  do {                                                                      \
    field_real = chunk->fld[0];                                             \
    field_imaginary = chunk->fld[1];                                        \
    if (field_real && field_imaginary) {                                    \
      const std::size_t point_count = chunk->gv.nowned(c);                  \
      segments->push_back(                                                  \
          {chunk, field_real, field_imaginary, chunk->gv.ntot(),            \
           gpu::detail::make_index_space_fp32(                              \
               chunk->gv, chunk->gv.little_owned_corner(c),                 \
               chunk->gv.big_corner()),                                    \
           point_count, complex_offset});                                   \
      LOOP_OVER_VOL_OWNED(chunk->gv, c, field_index) {                      \
        packed->push_back(field_real[field_index]);                         \
        packed->push_back(field_imaginary[field_index]);                    \
      }                                                                     \
      complex_offset += point_count;                                        \
    }                                                                       \
  } while (false)
        APPEND_CW_TEST_FIELD(f[c]);
        APPEND_CW_TEST_FIELD(f_u[c]);
        APPEND_CW_TEST_FIELD(f_cond[c]);
        APPEND_CW_TEST_FIELD(f_bfast[c]);
        const component c2 =
            field_type_component(is_D(c) ? E_stuff : H_stuff, c);
        APPEND_CW_TEST_FIELD(f_w[c2]);
        if (chunk->f_w[c2][0]) APPEND_CW_TEST_FIELD(f[c2]);
#undef APPEND_CW_TEST_FIELD
      }
  require(packed->size() == 2 * complex_offset,
          "CW packed-vector test constructed an inconsistent layout");
#else
  (void)f;
  (void)segments;
  (void)packed;
  throw std::logic_error(
      "CW packed-vector test requires an FP32 build");
#endif
}

void require_cuda_cw_field_vector_roundtrip() {
#if MEEP_HAVE_CUDA
  const grid_volume gv = vol2d(1.8, 1.6, 8.0);
  structure s(gv, vacuum, pml(0.3));
  fields f(&s, 0.0, 0.0, true, 64, 64);
  continuous_src_time source(0.27);
  f.add_point_source(Ez, source, vec(0.65, 0.8), 0.8);
  for (int step = 0; step < 4; ++step) f.step();
  for (int chunk_index = 0; chunk_index < f.num_chunks; ++chunk_index)
    if (f.chunks[chunk_index]->is_mine())
      gpu::detail::sync_resident_cache_for_owner(f.chunks[chunk_index]);

  std::vector<gpu::detail::cw_field_vector_segment_fp32> segments;
  std::vector<float> expected;
  append_cw_test_segments(f, &segments, &expected);
  require(!segments.empty() && !expected.empty(),
          "CW packed-vector roundtrip found no field segments");

  std::vector<std::unique_ptr<gpu::detail::resident_curl_session> >
      sessions(static_cast<std::size_t>(f.num_chunks));
  for (int chunk_index = 0; chunk_index < f.num_chunks; ++chunk_index)
    if (f.chunks[chunk_index]->is_mine())
      sessions[static_cast<std::size_t>(chunk_index)].reset(
          new gpu::detail::resident_curl_session(
              f.chunks[chunk_index], true));

  gpu::detail::resident_cw_vector_plan *plan = nullptr;
  float *device_first = nullptr;
  float *device_second = nullptr;
  try {
    plan = gpu::detail::create_resident_cw_vector_plan(
        segments.data(), segments.size(), expected.size() / 2);
    const std::size_t bytes = expected.size() * sizeof(float);
    device_first = static_cast<float *>(
        meep_cuda::allocate_device_bytes(bytes));
    device_second = static_cast<float *>(
        meep_cuda::allocate_device_bytes(bytes));
    gpu::detail::gather_resident_cw_vector_fp32(plan, device_first);
    std::vector<float> gathered(expected.size());
    meep_cuda::copy_to_host(gathered.data(), device_first, bytes);
    require(gathered == expected,
            "CUDA CW field gather changed packed ordering or values");

    std::vector<float> transformed(expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index)
      transformed[index] = expected[index] +
                           static_cast<float>(
                               static_cast<int>(index % 7) - 3) *
                               0.0009765625f;
    meep_cuda::copy_to_device(device_first, transformed.data(), bytes);
    gpu::detail::scatter_resident_cw_vector_fp32(plan, device_first);
    gpu::detail::gather_resident_cw_vector_fp32(plan, device_second);
    meep_cuda::copy_to_host(gathered.data(), device_second, bytes);
    require(gathered == transformed,
            "CUDA CW field scatter/gather roundtrip changed FP32 values");

    gpu::detail::destroy_resident_cw_vector_plan(plan);
    plan = nullptr;
    meep_cuda::free_device(device_second);
    device_second = nullptr;
    meep_cuda::free_device(device_first);
    device_first = nullptr;
    for (auto &session : sessions)
      if (session) session->finish(false);

    for (int chunk_index = 0; chunk_index < f.num_chunks; ++chunk_index)
      if (f.chunks[chunk_index]->is_mine())
        gpu::detail::sync_resident_cache_for_owner(f.chunks[chunk_index]);
    std::vector<gpu::detail::cw_field_vector_segment_fp32> final_segments;
    std::vector<float> final_packed;
    append_cw_test_segments(f, &final_segments, &final_packed);
    require(final_packed == transformed,
            "CUDA CW scatter did not publish device-authoritative fields");
  }
  catch (...) {
    gpu::detail::destroy_resident_cw_vector_plan(plan);
    meep_cuda::free_device(device_second);
    meep_cuda::free_device(device_first);
    throw;
  }
#else
  throw std::logic_error(
      "CUDA CW field-vector roundtrip requires a CUDA build");
#endif
}

class distributed_outer_unwind_probe {
public:
  distributed_outer_unwind_probe(fields &owner, bool &step_completed,
                                 bool &initialization_completed)
      : owner_(owner), step_completed_(step_completed),
        initialization_completed_(initialization_completed) {}
  ~distributed_outer_unwind_probe() {
    // Both calls complete normally while the deliberately thrown outer
    // exception is still unwinding. Distributed failure guards must detect
    // their own incomplete exit, not the process-wide uncaught flag.
    owner_.step();
    step_completed_ = true;
    owner_.initialize_field(Ez, initialized_profile);
    initialization_completed_ = true;
  }

private:
  fields &owner_;
  bool &step_completed_;
  bool &initialization_completed_;
};

class anisotropic_test_material : public material_function {
public:
  void eff_chi1inv_row(component c, double row[3], const volume &,
                       double = DEFAULT_SUBPIXEL_TOL,
                       int = DEFAULT_SUBPIXEL_MAXEVAL) override {
    static const double inverse[3][3] = {
        {0.55, 0.03, 0.02},
        {0.03, 0.48, 0.025},
        {0.02, 0.025, 0.60}};
    const int direction = component_index(c);
    for (int column = 0; column < 3; ++column)
      row[column] = inverse[direction][column];
  }

  void sigma_row(component c, double row[3], const vec &) override {
    static const double sigma[3][3] = {
        {0.18, 0.012, 0.008},
        {0.012, 0.15, 0.010},
        {0.008, 0.010, 0.20}};
    const int direction = component_index(c);
    for (int column = 0; column < 3; ++column)
      row[column] = sigma[direction][column];
  }
};

struct run_result {
  std::vector<std::complex<double> > samples;
  gpu::dispatch_statistics statistics{};
  gpu::resident_statistics resident{};
  gpu::field_update_statistics field_updates{};
  gpu::polarization_statistics polarizations{};
  gpu::source_statistics sources{};
  gpu::boundary_statistics boundaries{};
  gpu::dft_statistics dfts{};
  gpu::dft_reduction_statistics dft_reductions{};
  gpu::multi_gpu_statistics multi_gpu{};
  gpu::tile_coalescing_statistics tile_coalescing{};
  bool every_rank_cuda_curl = false;
  bool every_rank_remote_exchange = false;
  std::uint64_t minimum_rank_cuda_curl_calls = 0;
  std::uint64_t maximum_rank_cuda_curl_calls = 0;
  double workload_seconds = 0.0;
};

struct ldos_workload_result {
  run_result physics;
  gpu::ldos_statistics ldos{};
};

struct ldos_benchmark_result {
  struct timed_update_record {
    std::uint64_t start_monotonic_ns = 0;
    std::uint64_t stop_monotonic_ns = 0;
    int cpu_before = -1;
    int cpu_after = -1;
    std::vector<int> affinity_before;
    std::vector<int> affinity_after;
    std::uint64_t elapsed_ns = 0;
    std::uint64_t voluntary_context_switches = 0;
    std::uint64_t involuntary_context_switches = 0;
    std::uint64_t minor_faults = 0;
    std::uint64_t major_faults = 0;
  };
  run_result physics;
  gpu::ldos_statistics ldos{};
  double reduction_seconds = 0.0;
  std::uint64_t voluntary_context_switches = 0;
  std::uint64_t involuntary_context_switches = 0;
  std::uint64_t major_faults = 0;
  std::uint64_t minor_faults = 0;
  std::vector<timed_update_record> timed_updates;
};

struct dense_field_array {
  int chunk_index = -1;
  int component_index = -1;
  int complex_part = -1;
  std::vector<double> values;
};

struct dense_long_run_result {
  std::vector<dense_field_array> arrays;
  gpu::dispatch_statistics statistics{};
  gpu::field_update_statistics field_updates{};
  int executed_timesteps = 0;
  double grid_sx = 0.0;
  double grid_sy = 0.0;
  double resolution = 0.0;
  double source_frequency = 0.0;
  int source_components[2] = {-1, -1};
  double source_parameters[2][4] = {};
  double energy = 0.0;
};

enum test_dimension { one_dimensional_case, two_dimensional_case, three_dimensional_case };

class scoped_environment_override {
public:
  scoped_environment_override(const char *name, const char *replacement)
      : name_(name), had_value_(false) {
    const char *current = std::getenv(name);
    if (current) {
      had_value_ = true;
      value_ = current;
    }
    if (replacement)
      setenv(name, replacement, 1);
    else
      unsetenv(name);
  }

  ~scoped_environment_override() {
    if (had_value_)
      setenv(name_.c_str(), value_.c_str(), 1);
    else
      unsetenv(name_.c_str());
  }

  scoped_environment_override(const scoped_environment_override &) = delete;
  scoped_environment_override &operator=(
      const scoped_environment_override &) = delete;

private:
  std::string name_;
  bool had_value_;
  std::string value_;
};

void require(bool condition, const std::string &message);

std::uint64_t global_sum(std::uint64_t value) {
  require(value <= static_cast<std::uint64_t>(
                       std::numeric_limits<std::size_t>::max()),
          "GPU statistic does not fit the MPI size_t reduction");
  return static_cast<std::uint64_t>(
      sum_to_all(static_cast<std::size_t>(value)));
}

std::uint64_t global_minimum(std::uint64_t value) {
  constexpr std::uint64_t maximum_exact_binary64_integer =
      UINT64_C(9007199254740992);
  require(value <= maximum_exact_binary64_integer,
          "GPU statistic does not fit the exact MPI minimum reduction");
  return static_cast<std::uint64_t>(
      -max_to_all(-static_cast<double>(value)));
}

std::uint64_t global_maximum(std::uint64_t value) {
  // mympi exposes integer minimum/sum reductions but only a double maximum.
  // Binary64 represents every integer through 2^53 exactly, vastly beyond
  // any dispatch count accepted by this bounded test executable.
  constexpr std::uint64_t maximum_exact_binary64_integer =
      UINT64_C(9007199254740992);
  require(value <= maximum_exact_binary64_integer,
          "GPU statistic does not fit the exact MPI maximum reduction");
  return static_cast<std::uint64_t>(
      max_to_all(static_cast<double>(value)));
}

void aggregate_statistics(run_result &result) {
  result.multi_gpu = gpu::get_multi_gpu_statistics();
  result.tile_coalescing = gpu::get_tile_coalescing_statistics();
  result.every_rank_cuda_curl =
      and_to_all(result.statistics.cuda_curl_calls > 0 &&
                 result.statistics.cpu_curl_calls == 0);
  result.every_rank_remote_exchange =
      and_to_all(count_processors() == 1 ||
                 (result.multi_gpu.mpi_messages > 0 &&
                  result.multi_gpu.mpi_scalars > 0 &&
                  result.multi_gpu.cuda_aware_bytes +
                          result.multi_gpu.pinned_staging_bytes >
                      0));
  result.minimum_rank_cuda_curl_calls =
      global_minimum(result.statistics.cuda_curl_calls);
  result.maximum_rank_cuda_curl_calls =
      global_maximum(result.statistics.cuda_curl_calls);
  result.statistics = {
      global_sum(result.statistics.cpu_curl_calls),
      global_sum(result.statistics.cpu_curl_points),
      global_sum(result.statistics.cuda_curl_calls),
      global_sum(result.statistics.cuda_curl_points),
      global_sum(result.statistics.host_to_device_bytes),
      global_sum(result.statistics.device_to_host_bytes)};
  result.resident = {
      global_sum(result.resident.host_to_device_bytes_avoided),
      global_sum(result.resident.device_to_host_bytes_avoided),
      global_sum(result.resident.device_buffer_allocations),
      global_sum(result.resident.device_buffer_reuses)};
  result.field_updates = {
      global_sum(result.field_updates.cpu_update_eh_calls),
      global_sum(result.field_updates.cpu_update_eh_points),
      global_sum(result.field_updates.cuda_update_eh_calls),
      global_sum(result.field_updates.cuda_update_eh_points)};
  result.polarizations = {
      global_sum(result.polarizations.cpu_update_calls),
      global_sum(result.polarizations.cpu_update_points),
      global_sum(result.polarizations.cuda_update_calls),
      global_sum(result.polarizations.cuda_update_points)};
  result.sources = {
      global_sum(result.sources.cpu_update_calls),
      global_sum(result.sources.cpu_update_points),
      global_sum(result.sources.cuda_update_calls),
      global_sum(result.sources.cuda_update_points)};
  result.boundaries = {
      global_sum(result.boundaries.cpu_update_calls),
      global_sum(result.boundaries.cpu_update_points),
      global_sum(result.boundaries.cuda_update_calls),
      global_sum(result.boundaries.cuda_update_points)};
  result.dfts = {
      global_sum(result.dfts.cpu_update_calls),
      global_sum(result.dfts.cpu_update_points),
      global_sum(result.dfts.cuda_update_calls),
      global_sum(result.dfts.cuda_update_points)};
  result.dft_reductions = {
      global_sum(result.dft_reductions.cpu_reduction_calls),
      global_sum(result.dft_reductions.cpu_submitted_pairs),
      global_sum(result.dft_reductions.cpu_point_frequency_terms),
      global_sum(result.dft_reductions.cuda_reduction_calls),
      global_sum(result.dft_reductions.cuda_submitted_pairs),
      global_sum(result.dft_reductions.cuda_point_frequency_terms),
      global_sum(result.dft_reductions.cuda_descriptor_uploads),
      global_sum(result.dft_reductions.cuda_plan_reuses),
      global_sum(result.dft_reductions.cuda_kernel_launches),
      global_sum(
          result.dft_reductions.cuda_result_device_to_host_bytes),
      global_sum(
          result.dft_reductions
              .full_dft_device_to_host_bytes_avoided),
      global_sum(result.dft_reductions.mpi_allreduce_calls),
      global_sum(result.dft_reductions.mpi_allreduce_bytes)};
  result.multi_gpu = {
      global_sum(result.multi_gpu.mpi_messages),
      global_sum(result.multi_gpu.mpi_scalars),
      global_sum(result.multi_gpu.cuda_aware_bytes),
      global_sum(result.multi_gpu.pinned_staging_bytes),
      global_sum(result.multi_gpu.pinned_device_to_host_bytes),
      global_sum(result.multi_gpu.pinned_host_to_device_bytes)};
  result.tile_coalescing = {
      global_sum(result.tile_coalescing.curl_chunk_phases),
      global_sum(result.tile_coalescing.curl_input_tiles),
      global_sum(result.tile_coalescing.update_eh_chunk_phases),
      global_sum(result.tile_coalescing.update_eh_input_tiles)};
}

const char *case_name(test_dimension which) {
  switch (which) {
    case one_dimensional_case: return "1D";
    case two_dimensional_case: return "2D-tiled";
    case three_dimensional_case: return "3D";
  }
  return "unknown";
}

grid_volume make_volume(test_dimension which) {
  switch (which) {
    case one_dimensional_case: return vol1d(2.5, 12.0);
    case two_dimensional_case: return vol2d(2.5, 2.0, 12.0);
    case three_dimensional_case: return vol3d(1.2, 1.0, 0.8, 8.0);
  }
  throw std::invalid_argument("unknown GPU curl test dimension");
}

run_result run_case(gpu::backend_mode mode, test_dimension which,
                    bool configure_backend = true) {
  if (configure_backend) gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();

  const grid_volume gv = make_volume(which);
  structure s(gv, vacuum, no_pml(), identity(), 2);
  fields f(&s, 0.0, 0.0, true, which == two_dimensional_case ? 64 : 0);
  f.use_real_fields();

  gaussian_src_time electric_source(0.31, 0.12);
  gaussian_src_time magnetic_source(0.23, 0.10);
  std::vector<vec> points;
  if (which == one_dimensional_case) {
    f.add_point_source(Ex, electric_source, vec(0.71), 0.7);
    f.add_point_source(Hy, magnetic_source, vec(1.63), -0.4);
    points = {vec(0.37), vec(1.19), vec(2.01)};
  }
  else if (which == two_dimensional_case) {
    f.add_volume_source(Ez, electric_source,
                        volume(vec(0.71, 0.25), vec(0.71, 1.75)), 0.7);
    f.add_point_source(Hz, magnetic_source, vec(1.63, 1.27), -0.4);
    points = {vec(0.37, 0.42), vec(1.19, 0.91), vec(2.01, 1.53)};
  }
  else {
    f.add_point_source(Ex, electric_source, vec(0.31, 0.43, 0.27), 0.7);
    f.add_point_source(Hz, magnetic_source, vec(0.91, 0.67, 0.53), -0.4);
    points = {vec(0.17, 0.23, 0.19), vec(0.61, 0.49, 0.37),
              vec(1.03, 0.79, 0.61)};
  }

  for (int step = 0; step < 72; ++step)
    f.step();

  run_result result;
  for (const vec &point : points)
    for (component c = Ex; c <= Hz; c = component(c + 1))
      if (gv.has_field(c)) result.samples.push_back(f.get_field(c, point));
  result.samples.push_back(std::complex<double>(f.field_energy(), 0.0));
  result.statistics = gpu::get_dispatch_statistics();
  result.resident = gpu::get_resident_statistics();
  result.field_updates = gpu::get_field_update_statistics();
  result.polarizations = gpu::get_polarization_statistics();
  result.sources = gpu::get_source_statistics();
  result.boundaries = gpu::get_boundary_statistics();
  result.dfts = gpu::get_dft_statistics();
  aggregate_statistics(result);
  return result;
}

run_result run_material_case(gpu::backend_mode mode, bool complex_bloch,
                             int loop_tile_base = 64,
                             bool volume_source_batch_fixture = false,
                             int statistics_reset_step = -1) {
  gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();

  const grid_volume gv = vol2d(2.0, 2.0, 10.0);
  anisotropic_test_material anisotropic;
  structure s(gv, anisotropic, pml(0.4));
  s.add_susceptibility(
      anisotropic, E_stuff, lorentzian_susceptibility(0.42, 0.08));
  s.add_susceptibility(
      anisotropic, H_stuff, lorentzian_susceptibility(0.35, 0.06, true));
  s.set_chi2(weak_chi2);
  s.set_chi3(weak_chi3);
  const component conductive_components[] = {Dx, Dy, Dz, Bx, By, Bz};
  for (component c : conductive_components)
    if (gv.has_field(c)) s.set_conductivity(c, lossy);
  fields f(&s, 0.0, 0.0, true, loop_tile_base, loop_tile_base);
  if (complex_bloch)
    f.use_bloch(vec(0.13, 0.07));
  else
    f.use_real_fields();
  gaussian_src_time source(0.3, 0.1);
  const std::complex<double> primary_amplitude =
      complex_bloch ? std::complex<double>(0.8, -0.3)
                    : std::complex<double>(1.0, 0.0);
  const std::complex<double> secondary_amplitude =
      complex_bloch ? std::complex<double>(0.25, 0.1)
                    : std::complex<double>(0.3, 0.0);
  if (volume_source_batch_fixture) {
    f.add_volume_source(Ez, source, f.v, primary_amplitude);
    f.add_volume_source(Ex, source, f.v, secondary_amplitude);
  }
  else {
    f.add_point_source(Ez, source, gv.center(), primary_amplitude);
    f.add_point_source(Ex, source, vec(0.83, 1.17), secondary_amplitude);
  }
  component dft_components[] = {Ez, Hz};
  dft_fields dft_monitor =
      f.add_dft_fields(dft_components, 2, f.v, 0.22, 0.38, 3);
  (void)dft_monitor;
  for (int step = 0; step < 48; ++step) {
    f.step();
    if (step + 1 == statistics_reset_step)
      gpu::reset_dispatch_statistics();
  }

  run_result result;
  const vec points[] = {vec(0.37, 0.41), vec(1.0, 1.0), vec(1.61, 1.53)};
  for (const vec &point : points)
    for (component c = Ex; c <= Hz; c = component(c + 1))
      if (gv.has_field(c)) result.samples.push_back(f.get_field(c, point));
  result.samples.push_back(std::complex<double>(f.field_energy(), 0.0));
  result.samples.push_back(std::complex<double>(f.dft_norm(), 0.0));
  result.statistics = gpu::get_dispatch_statistics();
  result.resident = gpu::get_resident_statistics();
  result.field_updates = gpu::get_field_update_statistics();
  result.polarizations = gpu::get_polarization_statistics();
  result.sources = gpu::get_source_statistics();
  result.boundaries = gpu::get_boundary_statistics();
  result.dfts = gpu::get_dft_statistics();
  aggregate_statistics(result);
  return result;
}

run_result run_gyrotropic_case(gpu::backend_mode mode,
                               gyrotropy_model model,
                               test_dimension which) {
  gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();

  const grid_volume gv =
      which == two_dimensional_case
          ? vol2d(2.4, 2.0, 12.0)
          : vol3d(1.2, 1.0, 0.8, 8.0);
  structure s(gv, vacuum, pml(0.3));
  const double alpha =
      model == GYROTROPIC_SATURATED ? 2e-4 : 0.0;
  const vec bias =
      which == two_dimensional_case
          ? (model == GYROTROPIC_SATURATED
                 ? vec(0.0, 0.0, 1.0)
                 : vec(0.0, 0.0, 0.17))
          : (model == GYROTROPIC_SATURATED
                 ? vec(0.21, -0.14, 0.95)
                 : vec(0.11, -0.07, 0.13));
  s.add_susceptibility(
      gyrotropic_sigma, E_stuff,
      gyrotropic_susceptibility(
          bias, 0.74, 0.025, alpha, model));
  fields f(&s, 0.0, 0.0, true, 64, 64);
  f.use_real_fields();
  gaussian_src_time source(0.29, 0.11);
  std::vector<vec> points;
  if (which == two_dimensional_case) {
    f.add_point_source(Ex, source, vec(0.83, 0.91), 0.8);
    f.add_point_source(Ey, source, vec(1.57, 1.19), -0.23);
    points = {
        vec(0.41, 0.47), vec(1.21, 1.03), vec(2.01, 1.61)};
  }
  else {
    f.add_point_source(
        Ex, source, vec(0.31, 0.43, 0.27), 0.8);
    f.add_point_source(
        Ey, source, vec(0.73, 0.61, 0.49), -0.23);
    f.add_point_source(
        Ez, source, vec(0.91, 0.37, 0.59), 0.17);
    points = {
        vec(0.17, 0.23, 0.19), vec(0.61, 0.49, 0.37),
        vec(1.03, 0.79, 0.61)};
  }
  for (int step = 0; step < 40; ++step)
    f.step();

  run_result result;
  for (const vec &point : points)
    for (component c = Ex; c <= Hz; c = component(c + 1))
      if (gv.has_field(c))
        result.samples.push_back(f.get_field(c, point));
  result.samples.push_back(
      std::complex<double>(f.field_energy(), 0.0));
  result.statistics = gpu::get_dispatch_statistics();
  result.resident = gpu::get_resident_statistics();
  result.field_updates = gpu::get_field_update_statistics();
  result.polarizations = gpu::get_polarization_statistics();
  result.sources = gpu::get_source_statistics();
  result.boundaries = gpu::get_boundary_statistics();
  aggregate_statistics(result);
  return result;
}

run_result run_multilevel_case(gpu::backend_mode mode,
                               test_dimension which) {
  gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();

  const grid_volume gv =
      which == one_dimensional_case
          ? vol1d(2.4, 16.0)
          : which == two_dimensional_case
                ? vol2d(2.4, 2.0, 12.0)
                : vol3d(1.2, 1.0, 0.8, 8.0);
  structure s(gv, vacuum, pml(which == one_dimensional_case ? 0.25 : 0.3),
              identity(), 2);
  const realnum gamma_matrix[] = {
      0.030f, 0.000f, 0.000f,
      -0.030f, 0.020f, 0.000f,
      0.000f, -0.020f, 0.000f};
  const realnum initial_populations[] = {1.0f, 0.2f, 0.05f};
  const realnum alpha[] = {
      -0.37894034f, 0.0f,
      0.37894034f, -0.28420526f,
      0.0f, 0.28420526f};
  const realnum frequencies[] = {0.42f, 0.56f};
  const realnum linewidths[] = {0.04f, 0.05f};
  const realnum transition_sigma[] = {
      0.18f, 0.16f, 0.14f, 0.12f, 0.10f,
      0.15f, 0.13f, 0.11f, 0.09f, 0.07f};
  s.add_susceptibility(
      multilevel_sigma, E_stuff,
      multilevel_susceptibility(
          3, 2, gamma_matrix, initial_populations, alpha, frequencies,
          linewidths, transition_sigma));

  fields f(&s, 0.0, 0.0, true, 64, 64);
  if (which == two_dimensional_case)
    f.use_bloch(vec(0.07, 0.05));
  else
    f.use_real_fields();
  gaussian_src_time source(0.31, 0.13);
  std::vector<vec> points;
  if (which == one_dimensional_case) {
    f.add_point_source(Ex, source, vec(0.83), 0.8);
    points = {vec(0.31), vec(1.17), vec(2.03)};
  }
  else if (which == two_dimensional_case) {
    f.add_point_source(
        Ex, source, vec(0.83, 0.71),
        std::complex<double>(0.8, -0.17));
    f.add_point_source(
        Ey, source, vec(1.57, 1.29),
        std::complex<double>(-0.23, 0.11));
    points = {
        vec(0.41, 0.47), vec(1.21, 1.03), vec(2.01, 1.61)};
  }
  else {
    f.add_point_source(
        Ex, source, vec(0.31, 0.43, 0.27), 0.8);
    f.add_point_source(
        Ey, source, vec(0.73, 0.61, 0.49), -0.23);
    f.add_point_source(
        Ez, source, vec(0.91, 0.37, 0.59), 0.17);
    points = {
        vec(0.17, 0.23, 0.19), vec(0.61, 0.49, 0.37),
        vec(1.03, 0.79, 0.61)};
  }
  for (int step = 0; step < 48; ++step)
    f.step();

  run_result result;
  for (const vec &point : points)
    for (component c = Ex; c <= Hz; c = component(c + 1))
      if (gv.has_field(c))
        result.samples.push_back(f.get_field(c, point));
  result.samples.push_back(
      std::complex<double>(f.field_energy(), 0.0));
  result.statistics = gpu::get_dispatch_statistics();
  result.resident = gpu::get_resident_statistics();
  result.field_updates = gpu::get_field_update_statistics();
  result.polarizations = gpu::get_polarization_statistics();
  result.sources = gpu::get_source_statistics();
  result.boundaries = gpu::get_boundary_statistics();
  aggregate_statistics(result);
  return result;
}

run_result run_first_step_material_case(gpu::backend_mode mode) {
  gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();
  const grid_volume gv = vol2d(2.0, 2.0, 10.0);
  anisotropic_test_material anisotropic;
  structure s(gv, anisotropic, pml(0.4));
  s.set_chi2(weak_chi2);
  s.set_chi3(weak_chi3);
  fields f(&s, 0.0, 0.0, true, 64, 64);
  f.use_real_fields();
  gaussian_src_time source(0.3, 0.1);
  f.add_point_source(Ez, source, gv.center(), 1.0);
  f.step();

  run_result result;
  const vec points[] = {gv.center(), vec(0.23, 1.0), vec(1.77, 1.0)};
  for (const vec &point : points) {
    result.samples.push_back(f.get_field(Dz, point));
    result.samples.push_back(f.get_field(Ez, point));
  }
  result.samples.push_back(std::complex<double>(f.field_energy(), 0.0));
  result.statistics = gpu::get_dispatch_statistics();
  result.field_updates = gpu::get_field_update_statistics();
  aggregate_statistics(result);
  return result;
}

run_result run_beta_case(gpu::backend_mode mode, double beta,
                         bool complex_fields) {
  gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();
  const grid_volume gv = vol2d(1.8, 1.6, 10.0);
  structure s(gv, vacuum, pml(0.3));
  const component conductive_components[] = {Dx, Dy, Dz, Bx, By, Bz};
  for (component c : conductive_components)
    if (gv.has_field(c)) s.set_conductivity(c, lossy);
  fields f(&s, 0.0, beta, true, 64, 64);
  if (complex_fields)
    f.use_bloch(vec(0.0, 0.0));
  else
    f.use_real_fields();
  gaussian_src_time source(0.29, 0.11);
  f.add_point_source(
      Ez, source, gv.center(),
      complex_fields ? std::complex<double>(0.8, -0.23)
                     : std::complex<double>(0.8, 0.0));
  f.add_point_source(
      Hz, source, vec(0.67, 0.83),
      complex_fields ? std::complex<double>(-0.19, 0.11)
                     : std::complex<double>(-0.19, 0.0));
  for (int step = 0; step < 12; ++step)
    f.step();

  run_result result;
  const vec points[] = {gv.center(), vec(0.43, 0.57), vec(1.31, 1.07)};
  for (const vec &point : points)
    for (component c = Ex; c <= Hz; c = component(c + 1))
      if (gv.has_field(c)) result.samples.push_back(f.get_field(c, point));
  result.samples.push_back(std::complex<double>(f.field_energy(), 0.0));
  result.statistics = gpu::get_dispatch_statistics();
  result.resident = gpu::get_resident_statistics();
  result.field_updates = gpu::get_field_update_statistics();
  result.sources = gpu::get_source_statistics();
  aggregate_statistics(result);
  return result;
}

enum class curl_replay_direct_transition { beta, bfast };

struct curl_replay_transition_result {
  run_result physics;
  gpu::curl_phase_replay_statistics before_transition{};
  gpu::curl_phase_replay_statistics after_transition{};
};

curl_replay_transition_result run_curl_replay_direct_transition_case(
    gpu::backend_mode mode, curl_replay_direct_transition transition) {
  gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();
  const grid_volume gv = vol2d(1.8, 1.6, 10.0);
  structure s(gv, vacuum, pml(0.3));
  fields f(&s, 0.0, 0.0, true, 64, 64);
  f.use_bloch(vec(0.0, 0.0));
  gaussian_src_time source(0.29, 0.11);
  f.add_point_source(
      Ez, source, gv.center(), std::complex<double>(0.8, -0.23));
  f.add_point_source(
      Hz, source, vec(0.67, 0.83),
      std::complex<double>(-0.19, 0.11));
  for (int step = 0; step < 16; ++step) f.step();

  curl_replay_transition_result result;
  result.before_transition = gpu::get_curl_phase_replay_statistics();
  if (transition == curl_replay_direct_transition::beta) {
    f.beta = 0.19;
    for (int chunk_index = 0; chunk_index < f.num_chunks; ++chunk_index)
      f.chunks[chunk_index]->beta = f.beta;
  }
  else {
    f.bfast_scaled_k = {0.13, -0.07, 0.05};
    for (int chunk_index = 0; chunk_index < f.num_chunks; ++chunk_index)
      f.chunks[chunk_index]->bfast_scaled_k = f.bfast_scaled_k;
  }

  // The first changed step is the critical complete-plan -> direct-operation
  // transition. A stale replay here would silently omit beta/BFAST work.
  f.step();
  result.after_transition = gpu::get_curl_phase_replay_statistics();
  for (int step = 0; step < 3; ++step) f.step();

  const vec points[] = {
      gv.center(), vec(0.43, 0.57), vec(1.31, 1.07)};
  for (const vec &point : points)
    for (component c = Ex; c <= Hz; c = component(c + 1))
      if (gv.has_field(c))
        result.physics.samples.push_back(f.get_field(c, point));
  result.physics.samples.push_back(
      std::complex<double>(f.field_energy(), 0.0));
  result.physics.statistics = gpu::get_dispatch_statistics();
  result.physics.resident = gpu::get_resident_statistics();
  result.physics.field_updates = gpu::get_field_update_statistics();
  result.physics.sources = gpu::get_source_statistics();
  aggregate_statistics(result.physics);
  return result;
}

#if MEEP_SINGLE
void run_single_curl_phase(fields &owner, field_type ft,
                           bool resident) {
  std::vector<std::unique_ptr<gpu::detail::resident_curl_session> >
      sessions;
  if (resident) {
    sessions.reserve(static_cast<std::size_t>(owner.num_chunks));
    for (int chunk_index = 0; chunk_index < owner.num_chunks;
         ++chunk_index)
      if (owner.chunks[chunk_index]->is_mine())
        sessions.emplace_back(
            new gpu::detail::resident_curl_session(
                owner.chunks[chunk_index], true));
  }
  try {
    meep::fields_chunk_test_access::step_db(owner, ft);
  }
  catch (...) {
    for (auto &session : sessions)
      if (session && session->active()) session->finish(false);
    throw;
  }
  for (auto &session : sessions)
    if (session && session->active()) session->finish(false);
}

struct curl_replay_host_write_result {
  run_result physics;
  gpu::curl_phase_replay_statistics before{};
  gpu::curl_phase_replay_statistics after{};
  gpu::curl_phase_replay_statistics after_replay{};
  gpu::dispatch_statistics dispatch_before{};
  gpu::dispatch_statistics dispatch_after{};
  gpu::dispatch_statistics dispatch_after_replay{};
  gpu::resident_statistics resident_before{};
  gpu::resident_statistics resident_after{};
  std::size_t expected_upload_bytes = 0;
};

curl_replay_host_write_result run_curl_replay_host_write_case(
    gpu::backend_mode mode) {
  gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();
  const grid_volume gv = vol2d(1.8, 1.6, 10.0);
  structure s(gv, vacuum, pml(0.3));
  fields f(&s, 0.0, 0.0, true, 64, 64);
  f.use_bloch(vec(0.0, 0.0));
  gaussian_src_time source(0.29, 0.11);
  f.add_point_source(
      Ez, source, gv.center(), std::complex<double>(0.8, -0.23));
  f.add_point_source(
      Hz, source, vec(0.67, 0.83),
      std::complex<double>(-0.19, 0.11));
  for (int step = 0; step < 16; ++step) f.step();

  fields_chunk *target = nullptr;
  for (int chunk_index = 0; chunk_index < f.num_chunks; ++chunk_index)
    if (f.chunks[chunk_index]->is_mine() &&
        f.chunks[chunk_index]->f[Ez][0]) {
      target = f.chunks[chunk_index];
      break;
    }
  require(target != nullptr,
          "curl replay host-write fixture found no resident Ez field");
  if (mode == gpu::backend_mode::cuda) {
    for (int chunk_index = 0; chunk_index < f.num_chunks; ++chunk_index)
      if (f.chunks[chunk_index]->is_mine())
        gpu::detail::sync_resident_cache_for_owner(
            f.chunks[chunk_index]);
    const void *host_writes[] = {target->f[Ez][0]};
    gpu::detail::prepare_resident_host_writes_for_owner(
        target, host_writes, 1);
  }
  for (std::size_t index = 0; index < target->gv.ntot(); ++index)
    target->f[Ez][0][index] +=
        static_cast<realnum>(
            0.00390625 * (1 + static_cast<int>(index % 5)));

  curl_replay_host_write_result result;
  gpu::reset_dispatch_statistics();
  result.before = gpu::get_curl_phase_replay_statistics();
  result.dispatch_before = gpu::get_dispatch_statistics();
  result.resident_before = gpu::get_resident_statistics();
  result.expected_upload_bytes =
      target->gv.ntot() * sizeof(realnum);
  run_single_curl_phase(
      f, B_stuff, mode == gpu::backend_mode::cuda);
  result.after = gpu::get_curl_phase_replay_statistics();
  result.dispatch_after = gpu::get_dispatch_statistics();
  result.resident_after = gpu::get_resident_statistics();
  run_single_curl_phase(
      f, B_stuff, mode == gpu::backend_mode::cuda);
  result.after_replay = gpu::get_curl_phase_replay_statistics();
  result.dispatch_after_replay = gpu::get_dispatch_statistics();
  if (mode == gpu::backend_mode::cuda)
    for (int chunk_index = 0; chunk_index < f.num_chunks; ++chunk_index)
      if (f.chunks[chunk_index]->is_mine())
        gpu::detail::sync_resident_cache_for_owner(
            f.chunks[chunk_index]);

  const vec points[] = {
      gv.center(), vec(0.43, 0.57), vec(1.31, 1.07)};
  for (const vec &point : points)
    for (component c = Ex; c <= Hz; c = component(c + 1))
      if (gv.has_field(c))
        result.physics.samples.push_back(f.get_field(c, point));
  result.physics.samples.push_back(
      std::complex<double>(f.field_energy(), 0.0));
  result.physics.statistics = gpu::get_dispatch_statistics();
  result.physics.resident = gpu::get_resident_statistics();
  result.physics.field_updates = gpu::get_field_update_statistics();
  result.physics.sources = gpu::get_source_statistics();
  aggregate_statistics(result.physics);
  return result;
}
#endif

run_result run_bfast_case(
    gpu::backend_mode mode, test_dimension which, bool use_pml,
    bool use_conductivity, bool complex_fields) {
  gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();
  const grid_volume gv = make_volume(which);
  structure s(gv, vacuum, use_pml ? pml(0.3) : no_pml());
  if (use_conductivity) {
    const component conductive_components[] = {
        Dx, Dy, Dz, Bx, By, Bz};
    for (component c : conductive_components)
      if (gv.has_field(c)) s.set_conductivity(c, lossy);
  }
  fields f(
      &s, 0.0, 0.0, true, which == one_dimensional_case ? 0 : 32,
      which == one_dimensional_case ? 0 : 32,
      {0.13, -0.07, 0.05});
  if (complex_fields) {
    if (which == one_dimensional_case)
      f.use_bloch(vec(0.0));
    else if (which == two_dimensional_case)
      f.use_bloch(vec(0.0, 0.0));
    else
      f.use_bloch(vec(0.0, 0.0, 0.0));
  }
  else
    f.use_real_fields();

  gaussian_src_time source(0.31, 0.12);
  std::vector<vec> points;
  const std::complex<double> primary_amplitude =
      complex_fields ? std::complex<double>(0.7, -0.19)
                     : std::complex<double>(0.7, 0.0);
  const std::complex<double> secondary_amplitude =
      complex_fields ? std::complex<double>(-0.23, 0.11)
                     : std::complex<double>(-0.23, 0.0);
  if (which == one_dimensional_case) {
    f.add_point_source(Ex, source, vec(0.71), primary_amplitude);
    f.add_point_source(Hy, source, vec(1.63), secondary_amplitude);
    points = {vec(0.37), vec(1.19), vec(2.01)};
  }
  else if (which == two_dimensional_case) {
    f.add_point_source(Ez, source, vec(0.71, 0.53), primary_amplitude);
    f.add_point_source(Hx, source, vec(1.63, 1.27), secondary_amplitude);
    points = {
        vec(0.37, 0.42), vec(1.19, 0.91), vec(2.01, 1.53)};
  }
  else {
    f.add_point_source(
        Ex, source, vec(0.31, 0.43, 0.27), primary_amplitude);
    f.add_point_source(
        Hz, source, vec(0.91, 0.67, 0.53), secondary_amplitude);
    points = {
        vec(0.17, 0.23, 0.19), vec(0.61, 0.49, 0.37),
        vec(1.03, 0.79, 0.61)};
  }

  // More than two steps is essential: BFAST's persistent F recurrence,
  // including the historical bare single-operand branch, is otherwise
  // indistinguishable from several superficially plausible formulas.
  for (int step = 0; step < 18; ++step)
    f.step();

  run_result result;
  for (const vec &point : points)
    for (component c = Ex; c <= Hz; c = component(c + 1))
      if (gv.has_field(c))
        result.samples.push_back(f.get_field(c, point));
  result.samples.push_back(
      std::complex<double>(f.field_energy(), 0.0));
  result.statistics = gpu::get_dispatch_statistics();
  result.resident = gpu::get_resident_statistics();
  result.field_updates = gpu::get_field_update_statistics();
  result.sources = gpu::get_source_statistics();
  result.boundaries = gpu::get_boundary_statistics();
  aggregate_statistics(result);
  return result;
}

run_result run_cylindrical_case(gpu::backend_mode mode, int azimuthal_m,
                                bool real_fields,
                                bool zero_near_origin,
                                bool dispersive,
                                bool use_pml) {
  gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();
  const grid_volume gv = volcyl(1.6, 1.4, 10.0);
  structure s(
      gv, vacuum, use_pml ? pml(0.25) : no_pml(), identity(), 2, 0.4);
  if (dispersive) {
    s.add_susceptibility(
        vacuum, E_stuff, lorentzian_susceptibility(0.43, 0.07));
    s.add_susceptibility(
        vacuum, H_stuff, lorentzian_susceptibility(0.36, 0.05, true));
  }
  s.set_chi2(weak_chi2);
  s.set_chi3(weak_chi3);
  const component conductive_components[] = {Dr, Dp, Dz, Br, Bp, Bz};
  for (component c : conductive_components)
    if (gv.has_field(c)) s.set_conductivity(c, lossy);
  fields f(
      &s, static_cast<double>(azimuthal_m), 0.0, zero_near_origin, 32,
      32);
  if (real_fields)
    f.use_real_fields();
  else
    f.use_bloch(0.0);
  gaussian_src_time source(0.31, 0.12);
  f.add_point_source(
      Ep, source, veccyl(0.53, 0.61),
      real_fields ? std::complex<double>(0.7, 0.0)
                  : std::complex<double>(0.7, -0.21));
  f.add_point_source(
      Ez, source, veccyl(0.37, 0.83),
      real_fields ? std::complex<double>(-0.23, 0.0)
                  : std::complex<double>(-0.23, 0.14));
  component dft_components[] = {Ep, Ez};
  dft_fields dft_monitor =
      f.add_dft_fields(dft_components, 2, f.v, 0.21, 0.37, 3);
  (void)dft_monitor;
  for (int step = 0; step < 24; ++step)
    f.step();

  run_result result;
  const vec points[] = {
      veccyl(0.0, 0.47), veccyl(0.11, 0.53),
      veccyl(0.61, 0.79), veccyl(1.29, 1.08)};
  for (const vec &point : points)
    for (component c = Er; c <= Hz; c = component(c + 1))
      if (gv.has_field(c)) result.samples.push_back(f.get_field(c, point));
  result.samples.push_back(
      std::complex<double>(f.field_energy(), 0.0));
  result.samples.push_back(std::complex<double>(f.dft_norm(), 0.0));
  result.statistics = gpu::get_dispatch_statistics();
  result.resident = gpu::get_resident_statistics();
  result.field_updates = gpu::get_field_update_statistics();
  result.polarizations = gpu::get_polarization_statistics();
  result.sources = gpu::get_source_statistics();
  result.boundaries = gpu::get_boundary_statistics();
  result.dfts = gpu::get_dft_statistics();
  aggregate_statistics(result);
  return result;
}

run_result run_backend_transition_case(bool transition_to_cpu) {
  gpu::set_backend(transition_to_cpu ? gpu::backend_mode::cuda
                                     : gpu::backend_mode::cpu);
  gpu::reset_dispatch_statistics();
  const grid_volume gv = vol2d(1.8, 1.6, 10.0);
  structure s(gv, vacuum, no_pml());
  fields f(&s, 0.0, 0.0, true, 64, 64);
  f.use_real_fields();
  gaussian_src_time source(0.29, 0.11);
  f.add_point_source(Ez, source, gv.center(), 0.8);
  for (int step = 0; step < 6; ++step)
    f.step();
  if (transition_to_cpu) gpu::set_backend(gpu::backend_mode::cpu);
  for (int step = 0; step < 6; ++step)
    f.step();

  run_result result;
  const vec points[] = {gv.center(), vec(0.43, 0.57), vec(1.31, 1.07)};
  for (const vec &point : points)
    result.samples.push_back(f.get_field(Ez, point));
  result.statistics = gpu::get_dispatch_statistics();
  aggregate_statistics(result);
  return result;
}

run_result run_initialize_field_case(gpu::backend_mode mode) {
  gpu::set_backend(mode);
  const grid_volume gv = vol2d(1.8, 1.6, 10.0);
  structure s(gv, vacuum, no_pml());
  fields f(&s, 0.0, 0.0, true, 64, 64);
  f.use_real_fields();
  gaussian_src_time source(0.29, 0.11);
  f.add_point_source(Ez, source, gv.center(), 0.8);
  for (int step = 0; step < 5; ++step)
    f.step();
  f.initialize_field(Ez, initialized_profile);
  for (int step = 0; step < 40; ++step)
    f.step();

  run_result result;
  const vec points[] = {gv.center(), vec(0.43, 0.57), vec(1.31, 1.07)};
  for (const vec &point : points)
    result.samples.push_back(f.get_field(Ez, point));
  aggregate_statistics(result);
  return result;
}

run_result run_initialize_field_before_step_case(
    gpu::backend_mode mode) {
  gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();
  const grid_volume gv = vol2d(1.8, 1.6, 10.0);
  structure s(gv, vacuum, no_pml());
  fields f(&s, 0.0, 0.0, true, 64, 64);
  f.use_real_fields();
  f.initialize_field(Ez, initialized_profile);

  run_result result;
  result.statistics = gpu::get_dispatch_statistics();
  result.resident = gpu::get_resident_statistics();
  result.field_updates = gpu::get_field_update_statistics();
  result.polarizations = gpu::get_polarization_statistics();
  result.sources = gpu::get_source_statistics();
  result.boundaries = gpu::get_boundary_statistics();
  result.dfts = gpu::get_dft_statistics();
  aggregate_statistics(result);
  return result;
}

void require_fields_copy_lifecycle() {
  gpu::set_backend(gpu::backend_mode::cuda);
  const grid_volume gv = vol2d(1.8, 1.6, 10.0);
  structure s(gv, vacuum, no_pml());
  s.add_susceptibility(
      vacuum, E_stuff, lorentzian_susceptibility(0.31, 0.05));
  s.add_susceptibility(
      vacuum, E_stuff, lorentzian_susceptibility(0.43, 0.07, true));
  fields original(&s, 0.0, 0.0, true, 64, 64);
  original.use_real_fields();
  gaussian_src_time source(0.29, 0.11);
  original.add_point_source(Ez, source, gv.center(), 0.8);
  for (int step = 0; step < 8; ++step)
    original.step();

  fields clone(original);
  original.remove_sources();
  for (int step = 0; step < 4; ++step) {
    original.step();
    clone.step();
  }
  const vec points[] = {gv.center(), vec(0.43, 0.57), vec(1.31, 1.07)};
  for (const vec &point : points) {
    const std::complex<double> reference = original.get_field(Ez, point);
    const std::complex<double> candidate = clone.get_field(Ez, point);
    const double error = std::abs(reference - candidate);
    const double tolerance =
        3e-6 + 3e-4 * std::max(std::abs(reference), std::abs(candidate));
    require(error <= tolerance,
            "fields copy lost device-resident or polarization state");
  }
}

void require_fields_tile_base_contract() {
  gpu::set_backend(gpu::backend_mode::cpu);
  const grid_volume gv = vol2d(1.0, 1.0, 4.0);
  structure s(gv, vacuum, no_pml());
  // meep::abort is catchable for a singleton process, but intentionally calls
  // MPI_Abort for a multi-rank job.  Exercise the invalid-constructor contract
  // in the dedicated singleton qualification instead of aborting make check's
  // two-rank harness; the positive copy-lifecycle contract remains collective.
  if (count_processors() == 1) {
    const int invalid_bases[][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    for (const auto &bases : invalid_bases) {
      bool rejected = false;
      try {
        fields invalid(&s, 0.0, 0.0, true, bases[0], bases[1]);
      }
      catch (const std::runtime_error &) {
        rejected = true;
      }
      require(rejected, "fields must reject negative and base-one DB/EH tile settings");
    }
  }

  fields original(&s, 0.0, 0.0, true, 7, 11);
  fields clone(original);
  require(clone.loop_tile_base_db == 7 && clone.loop_tile_base_eh == 11,
          "fields copy construction must preserve both nondefault tile bases");
}

void require(bool condition, const std::string &message) {
  if (!and_to_all(condition)) throw std::runtime_error(message);
}

void require_coalesced_transfer_batch_planner() {
  const std::size_t sizes[] = {4, 6, 3, 7, 1};
  const std::vector<gpu::detail::coalesced_transfer_batch> batches =
      gpu::detail::plan_coalesced_transfer_batches(sizes, 5, 10);
  require(batches.size() == 3,
          "coalesced transfer planner returned the wrong batch count");
  require(batches[0].begin == 0 && batches[0].end == 2 &&
              batches[0].scalar_count == 10 &&
              batches[1].begin == 2 && batches[1].end == 4 &&
              batches[1].scalar_count == 10 &&
              batches[2].begin == 4 && batches[2].end == 5 &&
              batches[2].scalar_count == 1,
          "coalesced transfer planner did not preserve contiguous order");

  const std::size_t exact[] = {10};
  const auto exact_batches =
      gpu::detail::plan_coalesced_transfer_batches(exact, 1, 10);
  require(exact_batches.size() == 1 &&
              exact_batches[0].scalar_count == 10,
          "coalesced transfer planner rejected an exact-limit batch");

  const std::size_t split[] = {6, 5};
  const auto split_batches =
      gpu::detail::plan_coalesced_transfer_batches(split, 2, 10);
  require(split_batches.size() == 2 && split_batches[0].end == 1 &&
              split_batches[1].begin == 1,
          "coalesced transfer planner failed to split before overflow");

  bool oversized_rejected = false;
  const std::size_t oversized[] = {11};
  try {
    (void)gpu::detail::plan_coalesced_transfer_batches(
        oversized, 1, 10);
  }
  catch (const std::length_error &) {
    oversized_rejected = true;
  }
  require(oversized_rejected,
          "coalesced transfer planner accepted an oversized logical block");

  bool zero_rejected = false;
  const std::size_t zero[] = {0};
  try {
    (void)gpu::detail::plan_coalesced_transfer_batches(zero, 1, 10);
  }
  catch (const std::invalid_argument &) {
    zero_rejected = true;
  }
  require(zero_rejected,
          "coalesced transfer planner accepted an empty logical block");
}

void require_phase_block_operation_index_planner() {
  const std::size_t starts[] = {0, 2, 2, 5};
  const std::vector<std::uint32_t> indices =
      gpu::detail::plan_phase_block_operation_indices(starts, 4, 6);
  const std::vector<std::uint32_t> expected = {0, 0, 2, 2, 2, 3};
  require(indices == expected,
          "phase block-map planner mishandled a zero-block operation");

  require(gpu::detail::plan_phase_block_operation_indices(
              nullptr, 0, 0).empty(),
          "phase block-map planner rejected an empty phase");

  bool null_rejected = false;
  try {
    (void)gpu::detail::plan_phase_block_operation_indices(
        nullptr, 1, 1);
  }
  catch (const std::invalid_argument &) {
    null_rejected = true;
  }
  require(null_rejected,
          "phase block-map planner accepted null operation starts");

  bool ownerless_rejected = false;
  try {
    (void)gpu::detail::plan_phase_block_operation_indices(
        nullptr, 0, 1);
  }
  catch (const std::logic_error &) {
    ownerless_rejected = true;
  }
  require(ownerless_rejected,
          "phase block-map planner accepted ownerless blocks");

  bool nonzero_first_rejected = false;
  const std::size_t nonzero_first[] = {1};
  try {
    (void)gpu::detail::plan_phase_block_operation_indices(
        nonzero_first, 1, 2);
  }
  catch (const std::logic_error &) {
    nonzero_first_rejected = true;
  }
  require(nonzero_first_rejected,
          "phase block-map planner accepted a nonzero first prefix");

  bool decreasing_rejected = false;
  const std::size_t decreasing[] = {0, 3, 2};
  try {
    (void)gpu::detail::plan_phase_block_operation_indices(
        decreasing, 3, 4);
  }
  catch (const std::logic_error &) {
    decreasing_rejected = true;
  }
  require(decreasing_rejected,
          "phase block-map planner accepted a decreasing prefix");

  bool range_rejected = false;
  const std::size_t outside[] = {0, 7};
  try {
    (void)gpu::detail::plan_phase_block_operation_indices(
        outside, 2, 6);
  }
  catch (const std::logic_error &) {
    range_rejected = true;
  }
  require(range_rejected,
          "phase block-map planner accepted an out-of-range prefix");

  if (std::numeric_limits<std::size_t>::max() >
      static_cast<std::size_t>(
          std::numeric_limits<std::uint32_t>::max())) {
    bool operation_overflow_rejected = false;
    const std::size_t zero = 0;
    try {
      (void)gpu::detail::plan_phase_block_operation_indices(
          &zero,
          static_cast<std::size_t>(
              std::numeric_limits<std::uint32_t>::max()) + 1,
          0);
    }
    catch (const std::overflow_error &) {
      operation_overflow_rejected = true;
    }
    require(operation_overflow_rejected,
            "phase block-map planner accepted an unrepresentable "
            "operation count");
  }
}

void require_source_amplitude_l1_cache() {
  continuous_src_time source(0.25);
  std::vector<std::ptrdiff_t> indices = {1, 3, 5};
  std::vector<std::complex<double> > amplitudes = {
      {3.0, 4.0}, {-5.0, 12.0}, {8.0, 15.0}};
  src_vol profile(Ez, &source, std::move(indices),
                  std::move(amplitudes));
  require(profile.amplitude_l1() == 35.0,
          "source amplitude L1 cache is wrong after construction");
  profile.set_amplitude(1, std::complex<double>(0.0, 0.0));
  require(profile.amplitude_l1() == 22.0,
          "source amplitude L1 cache is stale after complex mutation");
  profile.set_amplitude(2, 6.0);
  require(profile.amplitude_l1() == 11.0,
          "source amplitude L1 cache is stale after real mutation");

  std::vector<std::ptrdiff_t> other_indices = {1, 3, 5};
  std::vector<std::complex<double> > other_amplitudes = {
      {0.0, -4.0}, {0.0, 0.0}, {-6.0, 0.0}};
  src_vol other(Ez, &source, std::move(other_indices),
                std::move(other_amplitudes));
  require(src_vol::combinable(profile, other),
          "source amplitude L1 fixture profiles are not combinable");
  profile.add_amplitudes_from(other);
  require(profile.amplitude_l1() == 3.0,
          "source amplitude L1 cache is stale after profile combination");
}

void require_halo_curl_partition_combinatorics() {
  std::uint64_t cases = 0;
  for (int parity = 0; parity < 8; ++parity)
    for (int nx = 1; nx <= 5; ++nx)
      for (int ny = 1; ny <= 5; ++ny)
        for (int nz = 1; nz <= 5; ++nz)
          for (int mask = 0; mask < 8; ++mask) {
            const int lower[3] = {
                parity & 1, (parity >> 1) & 1, (parity >> 2) & 1};
            const int point_counts[3] = {nx, ny, nz};
            const int upper[3] = {
                lower[0] + 2 * (nx - 1),
                lower[1] + 2 * (ny - 1),
                lower[2] + 2 * (nz - 1)};
            const bool active[3] = {
                (mask & 1) != 0, (mask & 2) != 0,
                (mask & 4) != 0};
            bool expected_interior = true;
            std::uint64_t expected_interior_points = 1;
            for (int axis = 0; axis < 3; ++axis) {
              if (active[axis] && point_counts[axis] < 3)
                expected_interior = false;
              const int count = active[axis]
                                    ? point_counts[axis] - 2
                                    : point_counts[axis];
              if (count > 0)
                expected_interior_points *=
                    static_cast<std::uint64_t>(count);
            }
            if (!expected_interior) expected_interior_points = 0;
            const gpu::detail::curl_partition_validation validation =
                gpu::detail::validate_curl_partition_for_testing(
                    lower, upper, active);
            require(validation.full_points ==
                        static_cast<std::uint64_t>(nx * ny * nz) &&
                        validation.interior_exists == expected_interior &&
                        validation.interior_points ==
                            expected_interior_points &&
                        validation.interior_points +
                                validation.shell_points ==
                            validation.full_points &&
                        validation.interior_shell_disjoint &&
                        validation.exact_cover,
                    "halo/curl inclusive Yee-box partition failed a parity, "
                    "extent, or active-axis combination");
            ++cases;
          }
  require(cases == 8000,
          "halo/curl partition combinatoric test count changed");
}

void compare_results(const run_result &reference, const run_result &candidate,
                     test_dimension which) {
  require(reference.samples.size() == candidate.samples.size(), "sample count mismatch");
  for (std::size_t i = 0; i < reference.samples.size(); ++i) {
    require(std::isfinite(reference.samples[i].real()) &&
                std::isfinite(reference.samples[i].imag()) &&
                std::isfinite(candidate.samples[i].real()) &&
                std::isfinite(candidate.samples[i].imag()),
            "CPU/CUDA comparison encountered a non-finite sample");
    const double error = std::abs(reference.samples[i] - candidate.samples[i]);
    const double scale = std::max(std::abs(reference.samples[i]), std::abs(candidate.samples[i]));
    const double tolerance = 3e-6 + 3e-4 * scale;
    if (error > tolerance) {
      std::cerr << "sample " << i << " differs: CPU=" << reference.samples[i]
                << " candidate=" << candidate.samples[i] << " error=" << error
                << " tolerance=" << tolerance << " case=" << case_name(which) << '\n';
      throw std::runtime_error("CPU/CUDA step_db result mismatch");
    }
  }
}

struct owned_lorentzian_state {
  fields_chunk *chunk;
  const susceptibility *susceptibility_owner;
  gpu::detail::lorentzian_internal_state_view state;
};

std::vector<owned_lorentzian_state> collect_lorentzian_states(
    fields &owner, component c) {
  std::vector<owned_lorentzian_state> result;
  for (int chunk_index = 0; chunk_index < owner.num_chunks; ++chunk_index) {
    fields_chunk *chunk = owner.chunks[chunk_index];
    if (!chunk->is_mine()) continue;
    for (polarization_state *polarization = chunk->pol[type(c)];
         polarization; polarization = polarization->next) {
      if (!polarization->data || !polarization->s) continue;
      const gpu::detail::lorentzian_internal_state_view state =
          gpu::detail::lorentzian_internal_state_for_testing(
              polarization->data, static_cast<int>(c), 0);
      if (state.polarization && state.auxiliary)
        result.push_back({chunk, polarization->s, state});
    }
  }
  require(!result.empty(),
          "integrated ADE state regression found no local polarization");
  return result;
}

void update_owned_polarizations(fields &owner, field_type ft) {
  for (int chunk_index = 0; chunk_index < owner.num_chunks; ++chunk_index)
    if (owner.chunks[chunk_index]->is_mine())
      (void)meep::fields_chunk_test_access::update_pols(
          *owner.chunks[chunk_index], ft);
}

void require_standard_drude_neutral_state(gpu::backend_mode mode,
                                          int iterations) {
  require(iterations > 0,
          "Drude neutral-state regression requires positive iterations");
  gpu::set_backend(gpu::backend_mode::cpu);
  const grid_volume gv = vol1d(1.0, 8.0);
  structure s(gv, vacuum, no_pml());
  s.add_susceptibility(
      vacuum, E_stuff, lorentzian_susceptibility(0.31, 0.07, true));
  fields f(&s);
  f.use_real_fields();
  f.require_component(Ex);
  // Allocate through fields_chunk::update_pols, not the standalone formula
  // helpers. The zero field leaves both freshly initialized states at zero.
  update_owned_polarizations(f, E_stuff);
  std::vector<owned_lorentzian_state> states =
      collect_lorentzian_states(f, Ex);
  for (owned_lorentzian_state &entry : states) {
    require(entry.state.kind ==
                gpu::detail::lorentzian_state_kind::increment,
            "standard Drude did not allocate increment auxiliary state");
    realnum *polarization =
        static_cast<realnum *>(entry.state.polarization);
    realnum *increment = static_cast<realnum *>(entry.state.auxiliary);
    PLOOP_OVER_VOL_OWNED(entry.chunk->gv, Ex, index) {
      polarization[index] = realnum(0.75);
      increment[index] = realnum(0.0);
    }
  }

  gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();
  for (int iteration = 0; iteration < iterations; ++iteration)
    update_owned_polarizations(f, E_stuff);
  if (mode == gpu::backend_mode::cuda)
    for (owned_lorentzian_state &entry : states)
      gpu::detail::sync_resident_cache_for_owner(entry.chunk);

  bool local_bitwise_invariant = true;
  for (const owned_lorentzian_state &entry : states) {
    const realnum *polarization =
        static_cast<const realnum *>(entry.state.polarization);
    const realnum *increment =
        static_cast<const realnum *>(entry.state.auxiliary);
    const realnum expected_polarization = realnum(0.75);
    const realnum expected_increment = realnum(0.0);
    LOOP_OVER_VOL_OWNED(entry.chunk->gv, Ex, index) {
      local_bitwise_invariant =
          local_bitwise_invariant &&
          std::memcmp(&polarization[index], &expected_polarization,
                      sizeof(realnum)) == 0 &&
          std::memcmp(&increment[index], &expected_increment,
                      sizeof(realnum)) == 0;
    }
  }
  require(local_bitwise_invariant,
          "production Drude neutral mode changed P or zero increment");
  const gpu::polarization_statistics statistics =
      gpu::get_polarization_statistics();
  require(mode == gpu::backend_mode::cuda
              ? statistics.cuda_update_calls > 0 &&
                    statistics.cpu_update_calls == 0
              : statistics.cpu_update_calls > 0 &&
                    statistics.cuda_update_calls == 0,
          "production Drude state regression used the wrong backend");
}

void require_standard_lorentz_previous_state(gpu::backend_mode mode) {
  gpu::set_backend(gpu::backend_mode::cpu);
  // This matches the small-phase, very-weak-loss regime exercised by the
  // upstream anisotropic-dispersion test.  Its decay rate is more accurately
  // represented by the previous-P state even though omega*dt is small.
  const grid_volume gv = vol1d(1.0, 200.0);
  structure s(gv, vacuum, no_pml());
  constexpr realnum omega = realnum(1.1);
  constexpr realnum gamma = realnum(1.0e-5);
  s.add_susceptibility(
      vacuum, E_stuff, lorentzian_susceptibility(omega, gamma, false));
  fields f(&s);
  f.use_real_fields();
  f.require_component(Ex);
  update_owned_polarizations(f, E_stuff);
  std::vector<owned_lorentzian_state> states =
      collect_lorentzian_states(f, Ex);
  for (owned_lorentzian_state &entry : states) {
    require(entry.state.kind ==
                gpu::detail::lorentzian_state_kind::previous,
            "standard Lorentz pole did not allocate previous-P state");
    realnum *polarization =
        static_cast<realnum *>(entry.state.polarization);
    realnum *previous = static_cast<realnum *>(entry.state.auxiliary);
    PLOOP_OVER_VOL_OWNED(entry.chunk->gv, Ex, index) {
      polarization[index] = realnum(0.75);
      previous[index] = realnum(-0.125);
    }
  }

  gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();
  update_owned_polarizations(f, E_stuff);
  if (mode == gpu::backend_mode::cuda)
    for (owned_lorentzian_state &entry : states)
      gpu::detail::sync_resident_cache_for_owner(entry.chunk);

  const realnum omega2pi = 2 * pi * omega;
  const realnum gamma2pi = 2 * pi * gamma;
  const realnum omega_dt_squared =
      omega2pi * omega2pi * f.dt * f.dt;
  const realnum gamma_inverse =
      1 / (1 + gamma2pi * f.dt / 2);
  const realnum gamma_previous = 1 - gamma2pi * f.dt / 2;
  const realnum expected =
      gamma_inverse *
      (realnum(0.75) * (2 - omega_dt_squared) -
       gamma_previous * realnum(-0.125));
  bool local_previous_exact = true;
  bool local_value_close = true;
  for (const owned_lorentzian_state &entry : states) {
    const realnum *polarization =
        static_cast<const realnum *>(entry.state.polarization);
    const realnum *previous =
        static_cast<const realnum *>(entry.state.auxiliary);
    LOOP_OVER_VOL_OWNED(entry.chunk->gv, Ex, index) {
      local_previous_exact =
          local_previous_exact && previous[index] == realnum(0.75);
      const double error =
          std::abs(static_cast<double>(polarization[index] - expected));
      local_value_close =
          local_value_close &&
          error <= 8 * std::numeric_limits<realnum>::epsilon() *
                       std::max(1.0, std::abs(static_cast<double>(expected)));
    }
  }
  require(local_previous_exact && local_value_close,
          "production weak-damping Lorentz pole did not preserve legacy "
          "previous-P recurrence");
  const gpu::polarization_statistics statistics =
      gpu::get_polarization_statistics();
  require(mode == gpu::backend_mode::cuda
              ? statistics.cuda_update_calls > 0 &&
                    statistics.cpu_update_calls == 0
              : statistics.cpu_update_calls > 0 &&
                    statistics.cuda_update_calls == 0,
          "production Lorentz state regression used the wrong backend");
}

void require_fine_grid_lorentz_increment_state(gpu::backend_mode mode) {
  gpu::set_backend(gpu::backend_mode::cpu);
  // First exercise the coefficient-cancellation override with a deliberately
  // weak-loss pole. At resolution 4000 the lowest Au frequency has a
  // restoring coefficient smaller than one FP32 ulp near 2, so it must use
  // the increment state independently of its damping ratio.
  {
    const grid_volume cancellation_gv = vol1d(0.01, 4000.0);
    structure cancellation_structure(cancellation_gv, vacuum, no_pml());
    cancellation_structure.add_susceptibility(
        vacuum, E_stuff,
        lorentzian_susceptibility(realnum(0.33472008806800069),
                                  realnum(1.0e-5), false));
    fields cancellation_fields(&cancellation_structure);
    cancellation_fields.use_real_fields();
    cancellation_fields.require_component(Ex);
    update_owned_polarizations(cancellation_fields, E_stuff);
    const std::vector<owned_lorentzian_state> cancellation_states =
        collect_lorentzian_states(cancellation_fields, Ex);
    for (const owned_lorentzian_state &entry : cancellation_states)
      require(entry.state.kind ==
                  gpu::detail::lorentzian_state_kind::increment,
              "cancellation-prone weak-loss Lorentz pole did not allocate "
              "increment state");
  }

  // At resolution 500 the highest Au pole's coefficient is well represented,
  // but its lossy, sub-0.1-radian phase advance still benefits from storing
  // the polarization increment. Exercise the full recurrence on this second,
  // independently selected branch.
  const grid_volume gv = vol1d(0.01, 500.0);
  structure s(gv, vacuum, no_pml());
  constexpr realnum omega = realnum(10.743304995339203);
  constexpr realnum gamma = realnum(1.7857115059820567);
  s.add_susceptibility(
      vacuum, E_stuff, lorentzian_susceptibility(omega, gamma, false));
  fields f(&s);
  f.use_real_fields();
  f.require_component(Ex);
  update_owned_polarizations(f, E_stuff);
  std::vector<owned_lorentzian_state> states =
      collect_lorentzian_states(f, Ex);
  for (owned_lorentzian_state &entry : states) {
    require(entry.state.kind ==
                gpu::detail::lorentzian_state_kind::increment,
            "fine-grid Lorentz pole did not allocate increment state");
    realnum *polarization =
        static_cast<realnum *>(entry.state.polarization);
    realnum *increment =
        static_cast<realnum *>(entry.state.auxiliary);
    PLOOP_OVER_VOL_OWNED(entry.chunk->gv, Ex, index) {
      polarization[index] = realnum(0.75);
      increment[index] = realnum(-0.125);
    }
  }

  gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();
  update_owned_polarizations(f, E_stuff);
  if (mode == gpu::backend_mode::cuda)
    for (owned_lorentzian_state &entry : states)
      gpu::detail::sync_resident_cache_for_owner(entry.chunk);

  const realnum omega2pi = 2 * pi * omega;
  const realnum gamma2pi = 2 * pi * gamma;
  const realnum omega_dt_squared =
      omega2pi * omega2pi * f.dt * f.dt;
  const realnum gamma_inverse = 1 / (1 + gamma2pi * f.dt / 2);
  const realnum gamma_previous = 1 - gamma2pi * f.dt / 2;
  const realnum expected_increment =
      gamma_inverse *
      (gamma_previous * realnum(-0.125) -
       omega_dt_squared * realnum(0.75));
  const realnum expected_polarization =
      realnum(0.75) + expected_increment;
  bool local_increment_contract = true;
  for (const owned_lorentzian_state &entry : states) {
    const realnum *polarization =
        static_cast<const realnum *>(entry.state.polarization);
    const realnum *increment =
        static_cast<const realnum *>(entry.state.auxiliary);
    LOOP_OVER_VOL_OWNED(entry.chunk->gv, Ex, index) {
      const double p_error = std::abs(static_cast<double>(
          polarization[index] - expected_polarization));
      const double increment_error = std::abs(static_cast<double>(
          increment[index] - expected_increment));
      const double scale = std::max(
          1.0, std::max(std::abs(static_cast<double>(expected_polarization)),
                        std::abs(static_cast<double>(expected_increment))));
      local_increment_contract =
          local_increment_contract &&
          std::isfinite(static_cast<double>(polarization[index])) &&
          std::isfinite(static_cast<double>(increment[index])) &&
          p_error <= 8 * std::numeric_limits<realnum>::epsilon() * scale &&
          increment_error <=
              8 * std::numeric_limits<realnum>::epsilon() * scale;
    }
  }
  require(local_increment_contract,
          "fine-grid Lorentz pole violated the increment recurrence");
  const gpu::polarization_statistics statistics =
      gpu::get_polarization_statistics();
  require(mode == gpu::backend_mode::cuda
              ? statistics.cuda_update_calls > 0 &&
                    statistics.cpu_update_calls == 0
              : statistics.cpu_update_calls > 0 &&
                    statistics.cuda_update_calls == 0,
          "fine-grid Lorentz regression used the wrong backend");
}

void require_noisy_lorentzian_drude_state_contract(bool drude) {
  gpu::set_backend(gpu::backend_mode::cpu);
  const grid_volume gv = vol1d(1.0, 8.0);
  structure s(gv, vacuum, no_pml());
  constexpr unsigned long seed = 0x5a17UL;
  constexpr realnum noise_amplitude = realnum(0.37);
  constexpr realnum omega = realnum(0.31);
  constexpr realnum gamma = realnum(0.07);
  s.add_susceptibility(
      vacuum, E_stuff,
      noisy_lorentzian_susceptibility(
          noise_amplitude, omega, gamma, drude));
  fields f(&s);
  f.use_real_fields();
  f.require_component(Ex);

  set_random_seed(seed);
  update_owned_polarizations(f, E_stuff);
  std::vector<owned_lorentzian_state> states =
      collect_lorentzian_states(f, Ex);
  const realnum gamma2pi = 2 * pi * gamma;
  const realnum omega2pi = 2 * pi * omega;
  const realnum noise_scale =
      omega2pi * noise_amplitude * sqrt(gamma2pi) * f.dt * f.dt /
      (1 + gamma2pi * f.dt / 2);

  // Replay the public fixed-seed generator in the exact serial order used by
  // noisy_lorentzian_susceptibility::update_P.  Replaying one update leaves
  // the global generator at the same position as the production first
  // update; replaying two after the second update likewise preserves the
  // production stream while supplying an independent scalar oracle.
  const auto replay_noise =
      [&](int update_number) {
        require(update_number > 0,
                "noisy ADE oracle requires a positive update number");
        set_random_seed(seed);
        std::vector<std::vector<realnum> > result(states.size());
        for (std::size_t state_index = 0; state_index < states.size();
             ++state_index)
          result[state_index].assign(states[state_index].state.point_count,
                                     realnum(0.0));
        for (int update = 0; update < update_number; ++update)
          for (std::size_t state_index = 0; state_index < states.size();
               ++state_index) {
            const owned_lorentzian_state &entry = states[state_index];
            const realnum *sigma =
                entry.susceptibility_owner->sigma[Ex][X];
            require(sigma != nullptr,
                    "noisy ADE oracle found no diagonal susceptibility");
            LOOP_OVER_VOL_OWNED(entry.chunk->gv, Ex, index) {
              result[state_index][index] = static_cast<realnum>(
                  gaussian_random(0, noise_scale * sqrt(sigma[index])));
            }
          }
        return result;
      };
  const std::vector<std::vector<realnum> > first_expected_noise =
      replay_noise(1);
  std::vector<std::vector<realnum> > first_p(states.size());
  std::vector<std::vector<realnum> > first_auxiliary(states.size());
  bool local_first_contract = true;
  bool local_nonzero_noise = false;
  for (std::size_t state_index = 0; state_index < states.size();
       ++state_index) {
    const owned_lorentzian_state &entry = states[state_index];
    require(entry.state.kind ==
                (drude ? gpu::detail::lorentzian_state_kind::increment
                       : gpu::detail::lorentzian_state_kind::previous),
            "noisy susceptibility allocated the wrong auxiliary state");
    const realnum *polarization =
        static_cast<const realnum *>(entry.state.polarization);
    const realnum *auxiliary =
        static_cast<const realnum *>(entry.state.auxiliary);
    first_p[state_index].assign(
        polarization, polarization + entry.state.point_count);
    first_auxiliary[state_index].assign(
        auxiliary, auxiliary + entry.state.point_count);
    LOOP_OVER_VOL_OWNED(entry.chunk->gv, Ex, index) {
      local_nonzero_noise =
          local_nonzero_noise || polarization[index] != realnum(0.0);
      local_first_contract =
          local_first_contract &&
          std::memcmp(&polarization[index],
                      &first_expected_noise[state_index][index],
                      sizeof(realnum)) == 0 &&
          (drude ? polarization[index] == auxiliary[index]
                 : auxiliary[index] == realnum(0.0));
    }
  }
  require(or_to_all(local_nonzero_noise) && local_first_contract,
          "first noisy Lorentzian/Drude update disagreed with the fixed-seed "
          "scalar oracle or violated its state contract");

  update_owned_polarizations(f, E_stuff);
  const std::vector<std::vector<realnum> > second_expected_noise =
      replay_noise(2);
  const realnum omega_dt_squared =
      omega2pi * omega2pi * f.dt * f.dt;
  const realnum gamma_inverse =
      1 / (1 + gamma2pi * f.dt / 2);
  const realnum gamma_previous = 1 - gamma2pi * f.dt / 2;
  bool local_second_contract = true;
  for (std::size_t state_index = 0; state_index < states.size();
       ++state_index) {
    const owned_lorentzian_state &entry = states[state_index];
    const realnum *polarization =
        static_cast<const realnum *>(entry.state.polarization);
    const realnum *auxiliary =
        static_cast<const realnum *>(entry.state.auxiliary);
    const std::vector<realnum> &p1 = first_p[state_index];
    const std::vector<realnum> &aux1 = first_auxiliary[state_index];
    LOOP_OVER_VOL_OWNED(entry.chunk->gv, Ex, index) {
      const realnum noise = second_expected_noise[state_index][index];
      if (!drude) {
        const realnum base_polarization =
            gamma_inverse *
            (p1[index] * (2 - omega_dt_squared) -
             gamma_previous * aux1[index]);
        const realnum expected_polarization =
            base_polarization + noise;
        const double error = std::abs(static_cast<double>(
            polarization[index] - expected_polarization));
        const double scale =
            std::max(1.0, std::abs(static_cast<double>(
                              expected_polarization)));
        local_second_contract =
            local_second_contract && auxiliary[index] == p1[index] &&
            std::isfinite(static_cast<double>(polarization[index])) &&
            error <= 8 * std::numeric_limits<realnum>::epsilon() * scale;
      }
      else {
        const realnum base_increment =
            gamma_inverse * (gamma_previous * aux1[index]);
        const realnum expected_increment = base_increment + noise;
        const realnum expected_polarization = p1[index] + expected_increment;
        const double p_error = std::abs(static_cast<double>(
            polarization[index] - expected_polarization));
        const double auxiliary_error = std::abs(static_cast<double>(
            auxiliary[index] - expected_increment));
        const double scale = std::max(
            1.0, std::max(std::abs(static_cast<double>(expected_polarization)),
                          std::abs(static_cast<double>(expected_increment))));
        local_second_contract =
            local_second_contract &&
            std::isfinite(static_cast<double>(polarization[index])) &&
            std::isfinite(static_cast<double>(auxiliary[index])) &&
            p_error <= 8 * std::numeric_limits<realnum>::epsilon() * scale &&
            auxiliary_error <=
                8 * std::numeric_limits<realnum>::epsilon() * scale;
      }
    }
  }
  require(local_second_contract,
          "second noisy Lorentzian/Drude update disagreed with the "
          "fixed-seed scalar oracle or applied noise to the wrong auxiliary "
          "state");
}

void require_integrated_ade_state_contract(bool cuda_available) {
#if MEEP_SINGLE
  // This catches the original FP32 failure mode through the production
  // fields/susceptibility path. One million zero-drive updates must retain
  // the Drude neutral root bit-for-bit instead of repeatedly evaluating
  // 2*P-Pprev.
  require_standard_drude_neutral_state(gpu::backend_mode::cpu, 1000000);
  require_standard_lorentz_previous_state(gpu::backend_mode::cpu);
  require_fine_grid_lorentz_increment_state(gpu::backend_mode::cpu);
  require_noisy_lorentzian_drude_state_contract(false);
  require_noisy_lorentzian_drude_state_contract(true);
  if (cuda_available) {
    // The header-level CUDA primitive has its own million-step gate. These
    // integrated updates prove that production host dispatch selects that
    // increment kernel for Drude and the legacy kernel for Lorentz.
    require_standard_drude_neutral_state(gpu::backend_mode::cuda, 64);
    require_standard_lorentz_previous_state(gpu::backend_mode::cuda);
    require_fine_grid_lorentz_increment_state(gpu::backend_mode::cuda);
  }
#else
  (void)cuda_available;
#endif
}

void require_mixed_dispersive_multi_gpu_contract() {
  require(count_processors() == 2,
          "mixed dispersive multi-GPU regression requires two MPI ranks");
  const run_result cpu =
      run_material_case(gpu::backend_mode::cpu, false);
  const run_result cuda =
      run_material_case(gpu::backend_mode::cuda, false);
  compare_results(cpu, cuda, two_dimensional_case);
  require(cuda.every_rank_cuda_curl && cuda.every_rank_remote_exchange &&
              cuda.polarizations.cuda_update_calls > 0 &&
              cuda.polarizations.cpu_update_calls == 0 &&
              cuda.multi_gpu.mpi_messages > 0 &&
              cuda.multi_gpu.mpi_scalars > 0,
          "anisotropic Lorentz+Drude multi-GPU regression did not use "
          "exclusive CUDA polarization and remote-boundary paths");
  const std::string transport =
      std::getenv("MEEP_GPU_MPI_TRANSPORT")
          ? std::getenv("MEEP_GPU_MPI_TRANSPORT")
          : "auto";
  if (transport == "pinned" || transport == "host")
    require(cuda.multi_gpu.pinned_staging_bytes > 0 &&
                cuda.multi_gpu.cuda_aware_bytes == 0,
            "mixed dispersive multi-GPU pinned run used the wrong transport");
  if (transport == "cuda-aware" || transport == "device")
    require(cuda.multi_gpu.cuda_aware_bytes > 0 &&
                cuda.multi_gpu.pinned_staging_bytes == 0,
            "mixed dispersive multi-GPU CUDA-aware run used the wrong "
            "transport");
}

void compare_cw_results(const run_result &reference,
                        const run_result &candidate) {
  require(reference.samples.size() == candidate.samples.size(),
          "CW full-vector sample count mismatch");
  require(reference.samples.size() > 9,
          "CW comparison is missing its full packed unknown vector");
  double raw_error_squared = 0.0;
  double reference_squared = 0.0;
  std::complex<double> phase_cross(0.0, 0.0);
  bool local_finite = true;
  for (std::size_t index = 0; index < reference.samples.size(); ++index) {
    const std::complex<double> expected = reference.samples[index];
    const std::complex<double> actual = candidate.samples[index];
    local_finite = local_finite && std::isfinite(expected.real()) &&
                   std::isfinite(expected.imag()) &&
                   std::isfinite(actual.real()) &&
                   std::isfinite(actual.imag());
    raw_error_squared += std::norm(expected - actual);
    reference_squared += std::norm(expected);
    if (index >= 9) phase_cross += std::conj(actual) * expected;
  }
  require(local_finite,
          "CW full-vector comparison encountered a nonfinite value");
  raw_error_squared = sum_to_all(raw_error_squared);
  reference_squared = sum_to_all(reference_squared);
  phase_cross = sum_to_all(phase_cross);
  require(std::isfinite(phase_cross.real()) &&
              std::isfinite(phase_cross.imag()) &&
              std::abs(phase_cross) > 0.0,
          "CW full-vector phase anchor is degenerate");
  const std::complex<double> phase =
      phase_cross / std::abs(phase_cross);

  double aligned_error_squared = 0.0;
  double maximum_error = 0.0;
  double maximum_contract_ratio = 0.0;
  double maximum_physical_contract_ratio = 0.0;
  double maximum_raw_physical_contract_ratio = 0.0;
  for (std::size_t index = 0; index < reference.samples.size(); ++index) {
    const std::complex<double> expected = reference.samples[index];
    const std::complex<double> actual = phase * candidate.samples[index];
    const double error = std::abs(expected - actual);
    maximum_error = std::max(maximum_error, error);
    aligned_error_squared += error * error;
    const double scale = std::max(std::abs(expected), std::abs(actual));
    // The first nine values are physical E/H point probes and retain a
    // strict pointwise contract.  The remaining packed vector includes PML
    // auxiliaries whose exact value may be close to zero; for those, use an
    // FP32-appropriate absolute floor in addition to the global L2 gate.
    const double absolute_relative_limit =
        index < 9 ? 3e-5 + 2e-3 * scale
                  : 5e-4 + 5e-3 * scale;
    const double contract_ratio = error / absolute_relative_limit;
    maximum_contract_ratio =
        std::max(maximum_contract_ratio, contract_ratio);
    if (index < 9)
      maximum_physical_contract_ratio =
          std::max(maximum_physical_contract_ratio, contract_ratio);
    if (index < 9) {
      const double raw_error =
          std::abs(expected - candidate.samples[index]);
      const double raw_scale =
          std::max(std::abs(expected),
                   std::abs(candidate.samples[index]));
      maximum_raw_physical_contract_ratio = std::max(
          maximum_raw_physical_contract_ratio,
          raw_error / (3e-5 + 2e-3 * raw_scale));
    }
  }
  aligned_error_squared = sum_to_all(aligned_error_squared);
  maximum_error = max_to_all(maximum_error);
  maximum_contract_ratio = max_to_all(maximum_contract_ratio);
  maximum_physical_contract_ratio =
      max_to_all(maximum_physical_contract_ratio);
  maximum_raw_physical_contract_ratio =
      max_to_all(maximum_raw_physical_contract_ratio);
  const double raw_relative_l2 = std::sqrt(
      raw_error_squared / std::max(reference_squared, 1e-300));
  const double relative_l2 = std::sqrt(
      aligned_error_squared / std::max(reference_squared, 1e-300));
  const std::size_t global_sample_count =
      sum_to_all(reference.samples.size());
  if (am_master())
    std::cout << "cw-full-vector-agreement: samples="
              << global_sample_count
              << " raw-relative-l2=" << raw_relative_l2
              << " phase-angle=" << std::arg(phase)
              << " aligned-max-abs=" << maximum_error
              << " aligned-relative-l2=" << relative_l2
              << " max-contract-ratio=" << maximum_contract_ratio
              << " physical-contract-ratio="
              << maximum_physical_contract_ratio
              << " raw-physical-contract-ratio="
              << maximum_raw_physical_contract_ratio
              << '\n';
  require(raw_relative_l2 <= 2e-3,
          "CW CPU/CUDA raw full-vector relative-L2 error exceeds 0.2%");
  require(std::abs(std::arg(phase)) <= 2e-3,
          "CW CPU/CUDA global phase error exceeds 0.002 radians");
  require(relative_l2 <= 2e-3,
          "CW CPU/CUDA full-vector relative-L2 error exceeds 0.2%");
  require(maximum_contract_ratio <= 1.0,
          "CW CPU/CUDA packed-vector error exceeds the FP32 residual "
          "contract");
  require(maximum_physical_contract_ratio <= 1.0,
          "CW CPU/CUDA physical point-probe error exceeds the strict "
          "FP32 contract");
  require(maximum_raw_physical_contract_ratio <= 1.0,
          "CW CPU/CUDA raw physical point-probe error exceeds the strict "
          "FP32 contract");
}

dense_long_run_result run_dense_long_field_case(gpu::backend_mode mode) {
  constexpr int timesteps = 1024;
  constexpr double grid_sx = 2.5;
  constexpr double grid_sy = 2.0;
  constexpr double resolution = 16.0;
  constexpr double source_frequency = 0.27;
  constexpr double source_parameters[2][4] = {
      {0.83, 0.91, 0.8, -0.31},
      {1.57, 1.19, -0.23, 0.17}};
#if !MEEP_SINGLE
  require(mode == gpu::backend_mode::cpu,
          "FP64 dense reference build only supports the CPU backend");
#endif
  gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();

  // This deliberately excites both 2D polarizations in a complex field so
  // every Cartesian E/H/D/B component and both real/imaginary arrays are
  // covered.  The compact grid keeps the regression cheap while 1024 updates
  // expose phase error which short point-sample tests can hide.
  const grid_volume gv = vol2d(grid_sx, grid_sy, resolution);
  structure s(gv, vacuum, pml(0.35));
  fields f(&s, 0.0, 0.0, true, 64, 64);
  f.use_bloch(vec(0.11, 0.07));
  continuous_src_time source(source_frequency);
  f.add_point_source(
      Ez, source, vec(source_parameters[0][0], source_parameters[0][1]),
      std::complex<double>(source_parameters[0][2], source_parameters[0][3]));
  f.add_point_source(
      Ex, source, vec(source_parameters[1][0], source_parameters[1][1]),
      std::complex<double>(source_parameters[1][2], source_parameters[1][3]));

  dense_long_run_result result;
  result.grid_sx = grid_sx;
  result.grid_sy = grid_sy;
  result.resolution = resolution;
  result.source_frequency = source_frequency;
  result.source_components[0] = static_cast<int>(Ez);
  result.source_components[1] = static_cast<int>(Ex);
  std::memcpy(result.source_parameters, source_parameters,
              sizeof(source_parameters));
  for (int step = 0; step < timesteps; ++step) {
    f.step();
    ++result.executed_timesteps;
  }
  result.statistics = gpu::get_dispatch_statistics();
  result.field_updates = gpu::get_field_update_statistics();

  // CUDA owns the authoritative arrays after stepping.  One bulk owner sync
  // per chunk makes the complete primary field state observable without the
  // sparse scalar-query path changing what this regression covers.
#if MEEP_SINGLE
  if (mode == gpu::backend_mode::cuda)
    for (int chunk_index = 0; chunk_index < f.num_chunks; ++chunk_index)
      if (f.chunks[chunk_index]->is_mine())
        gpu::detail::sync_resident_cache_for_owner(f.chunks[chunk_index]);
#endif

  for (int chunk_index = 0; chunk_index < f.num_chunks; ++chunk_index) {
    fields_chunk *chunk = f.chunks[chunk_index];
    if (!chunk->is_mine()) continue;
    const std::size_t array_size = chunk->gv.ntot();
    for (int component_index = 0;
         component_index < NUM_FIELD_COMPONENTS; ++component_index)
      for (int part = 0; part < 2; ++part) {
        const realnum *array = chunk->f[component_index][part];
        if (!array) continue;
        dense_field_array snapshot;
        snapshot.chunk_index = chunk_index;
        snapshot.component_index = component_index;
        snapshot.complex_part = part;
        snapshot.values.reserve(array_size);
        for (std::size_t index = 0; index < array_size; ++index)
          snapshot.values.push_back(static_cast<double>(array[index]));
        result.arrays.push_back(std::move(snapshot));
      }
  }
  result.energy = f.field_energy();
  return result;
}

std::string canonical_runtime_path(const std::string &path,
                                   const char *description) {
  require(!path.empty() && path[0] == '/',
          std::string(description) + " path is not absolute");
  char resolved[PATH_MAX];
  require(realpath(path.c_str(), resolved) != nullptr,
          std::string("could not canonicalize ") + description + " path");
  return std::string(resolved);
}

std::string current_executable_path() {
  char path[PATH_MAX];
  const ssize_t length = readlink("/proc/self/exe", path, sizeof(path) - 1);
  require(length > 0 && static_cast<std::size_t>(length) < sizeof(path),
          "could not identify the dense evidence executable");
  path[length] = '\0';
  return canonical_runtime_path(path, "executable");
}

std::string loaded_libmeep_path() {
  std::ifstream maps("/proc/self/maps");
  require(maps.good(),
          "could not inspect loaded libraries for dense evidence");
  std::set<std::string> candidates;
  std::string line;
  while (std::getline(maps, line)) {
    const std::size_t path_start = line.find('/');
    if (path_start == std::string::npos) continue;
    std::string path = line.substr(path_start);
    const std::string deleted_suffix = " (deleted)";
    const bool deleted =
        path.size() >= deleted_suffix.size() &&
        path.compare(path.size() - deleted_suffix.size(),
                     deleted_suffix.size(), deleted_suffix) == 0;
    if (deleted) path.resize(path.size() - deleted_suffix.size());
    const std::size_t filename = path.rfind('/');
    if (filename != std::string::npos &&
        path.compare(filename + 1, 10, "libmeep.so") == 0) {
      if (deleted)
        throw std::runtime_error(
            "loaded libmeep was replaced or deleted during dense evidence");
      candidates.insert(canonical_runtime_path(path, "loaded libmeep"));
    }
  }
  require(candidates.size() == 1,
          "dense evidence did not identify exactly one loaded libmeep");
  return *candidates.begin();
}

std::string json_escape(const std::string &value) {
  std::ostringstream escaped;
  for (const unsigned char byte : value) {
    switch (byte) {
      case '\"': escaped << "\\\""; break;
      case '\\': escaped << "\\\\"; break;
      case '\b': escaped << "\\b"; break;
      case '\f': escaped << "\\f"; break;
      case '\n': escaped << "\\n"; break;
      case '\r': escaped << "\\r"; break;
      case '\t': escaped << "\\t"; break;
      default:
        if (byte < 0x20) {
          escaped << "\\u00" << std::hex << std::setw(2)
                  << std::setfill('0') << static_cast<unsigned>(byte)
                  << std::dec << std::setfill(' ');
        }
        else escaped << static_cast<char>(byte);
    }
  }
  return escaped.str();
}

std::string canonical_decimal(double value) {
  require(std::isfinite(value),
          "canonical evidence decimal is not finite");
  if (value == 0.0) return "0";
  std::ostringstream formatted;
  formatted.imbue(std::locale::classic());
  formatted << std::setprecision(std::numeric_limits<double>::max_digits10)
            << value;
  require(formatted.good(), "could not format canonical evidence decimal");
  std::string result = formatted.str();
  const std::size_t exponent = result.find_first_of("eE");
  if (exponent != std::string::npos) {
    result[exponent] = 'e';
    std::size_t digits = exponent + 1;
    bool negative = false;
    if (digits < result.size() &&
        (result[digits] == '+' || result[digits] == '-')) {
      negative = result[digits] == '-';
      result.erase(digits, 1);
    }
    while (digits + 1 < result.size() && result[digits] == '0')
      result.erase(digits, 1);
    if (negative) result.insert(digits, 1, '-');
  }
  return result;
}

std::uint64_t sample_digest(
    const std::vector<std::complex<double>> &samples) {
  std::uint64_t digest = UINT64_C(14695981039346656037);
  for (const std::complex<double> &sample : samples) {
    const double values[] = {sample.real(), sample.imag()};
    for (const double value : values) {
      std::uint64_t bits = 0;
      static_assert(sizeof(bits) == sizeof(value),
                    "double digest width changed");
      std::memcpy(&bits, &value, sizeof(bits));
      for (int byte = 0; byte < 8; ++byte) {
        digest ^= (bits >> (8 * byte)) & UINT64_C(0xff);
        digest *= UINT64_C(1099511628211);
      }
    }
  }
  return digest;
}

std::string hex_digest(std::uint64_t value) {
  std::ostringstream stream;
  stream << std::hex << std::setw(16) << std::setfill('0') << value;
  return stream.str();
}

struct numerical_error_summary {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
};

numerical_error_summary compare_sample_errors(
    const std::vector<std::complex<double>> &reference,
    const std::vector<std::complex<double>> &candidate) {
  require(reference.size() == candidate.size(),
          "LDOS evidence sample count mismatch");
  numerical_error_summary result;
  for (std::size_t index = 0; index < reference.size(); ++index) {
    const double absolute = std::abs(reference[index] - candidate[index]);
    const double scale =
        std::max(std::abs(reference[index]), std::abs(candidate[index]));
    result.maximum_absolute = std::max(result.maximum_absolute, absolute);
    result.maximum_relative = std::max(
        result.maximum_relative,
        absolute / std::max(scale, std::numeric_limits<double>::min()));
  }
  return result;
}

std::string json_samples(
    const std::vector<std::complex<double>> &samples) {
  std::ostringstream stream;
  stream << '[';
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index) stream << ',';
    stream << '[' << canonical_decimal(samples[index].real()) << ','
           << canonical_decimal(samples[index].imag()) << ']';
  }
  stream << ']';
  return stream.str();
}

struct mapped_runtime_file {
  std::string path;
  std::uint64_t device;
  std::uint64_t inode;
  std::uint64_t size;
  std::int64_t mtime_ns;
  std::int64_t ctime_ns;
};

struct mapped_runtime_special {
  std::string path;
  std::string kind;
  std::uint64_t device;
  std::uint64_t inode;
  std::uint64_t mode;
  std::uint64_t rdev;
  bool deleted;
};

struct runtime_mapping_inventory {
  std::vector<mapped_runtime_file> regular;
  std::vector<mapped_runtime_special> special;
};

std::string decode_procfs_path(const std::string &encoded) {
  std::string decoded;
  decoded.reserve(encoded.size());
  for (std::size_t index = 0; index < encoded.size();) {
    if (encoded[index] == '\\' && index + 3 < encoded.size() &&
        encoded[index + 1] >= '0' && encoded[index + 1] <= '7' &&
        encoded[index + 2] >= '0' && encoded[index + 2] <= '7' &&
        encoded[index + 3] >= '0' && encoded[index + 3] <= '7') {
      const unsigned value =
          static_cast<unsigned>(encoded[index + 1] - '0') * 64u +
          static_cast<unsigned>(encoded[index + 2] - '0') * 8u +
          static_cast<unsigned>(encoded[index + 3] - '0');
      decoded.push_back(static_cast<char>(value));
      index += 4;
      continue;
    }
    require(encoded[index] != '\\',
            "runtime mapping contains a malformed procfs escape");
    decoded.push_back(encoded[index++]);
  }
  return decoded;
}

bool runtime_family_required(const std::string &path) {
  std::string lower = path;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char value) { return std::tolower(value); });
  const std::size_t slash = lower.rfind('/');
  const std::string name =
      slash == std::string::npos ? lower : lower.substr(slash + 1);
  const auto soname = [&name](const std::string &prefix) {
    return name == prefix ||
           (name.size() > prefix.size() &&
            name.compare(0, prefix.size(), prefix) == 0 &&
            (name[prefix.size()] == '.' || name[prefix.size()] == '_'));
  };
  return soname("libmeep.so") || soname("libmpi.so") ||
         soname("libcuda.so") || soname("libcudart.so") ||
         lower.find("ld-linux") != std::string::npos ||
         lower.find("/ld.so") != std::string::npos;
}

bool nvidia_character_device_path(const std::string &path) {
  if (path == "/dev/nvidiactl" || path == "/dev/nvidia-uvm" ||
      path == "/dev/nvidia-uvm-tools")
    return true;
  const std::string prefix = "/dev/nvidia";
  if (path.compare(0, prefix.size(), prefix) != 0 ||
      path.size() == prefix.size())
    return false;
  return std::all_of(path.begin() + prefix.size(), path.end(),
                     [](unsigned char value) { return std::isdigit(value); });
}

bool ephemeral_data_mapping_path(const std::string &path) {
  return path.compare(0, 9, "/dev/shm/") == 0 ||
         path.compare(0, 7, "/memfd:") == 0 || path == "/dev/zero";
}

runtime_mapping_inventory loaded_runtime_files() {
  std::ifstream maps("/proc/self/maps");
  require(maps.good(), "could not inspect runtime file mappings");
  struct regular_candidate {
    mapped_runtime_file record;
    bool executable = false;
    bool required = false;
  };
  std::map<std::pair<std::uint64_t, std::uint64_t>, regular_candidate>
      candidates;
  std::set<std::pair<std::string, std::uint64_t>> special_seen;
  runtime_mapping_inventory result;
  std::string line;
  while (std::getline(maps, line)) {
    std::istringstream parser(line);
    std::string addresses, permissions, offset, device_text;
    std::uint64_t mapped_inode = 0;
    if (!(parser >> addresses >> permissions >> offset >> device_text >>
          mapped_inode))
      throw std::runtime_error("could not parse a runtime mapping record");
    std::string mapped_path;
    std::getline(parser, mapped_path);
    const std::size_t first = mapped_path.find_first_not_of(' ');
    if (first == std::string::npos || mapped_path[first] != '/') continue;
    mapped_path = decode_procfs_path(mapped_path.substr(first));
    const std::string deleted_suffix = " (deleted)";
    const bool deleted =
        mapped_path.size() >= deleted_suffix.size() &&
        mapped_path.compare(mapped_path.size() - deleted_suffix.size(),
                            deleted_suffix.size(), deleted_suffix) == 0;
    if (deleted) mapped_path.resize(mapped_path.size() - deleted_suffix.size());
    const bool executable = permissions.find('x') != std::string::npos;
    if (deleted && (executable || runtime_family_required(mapped_path)))
      throw std::runtime_error(
          "an executable/required runtime mapping was replaced or deleted");
    unsigned mapped_major = 0, mapped_minor = 0;
    char tail = '\0';
    require(std::sscanf(device_text.c_str(), "%x:%x%c", &mapped_major,
                        &mapped_minor, &tail) == 2,
            "runtime mapping has an invalid kernel device/inode identity");
    const std::uint64_t mapped_device = static_cast<std::uint64_t>(
        makedev(mapped_major, mapped_minor));
    if (deleted) {
      require(!executable && ephemeral_data_mapping_path(mapped_path),
              "unexpected deleted absolute runtime mapping");
      const auto key = std::make_pair(mapped_path, mapped_inode);
      if (special_seen.insert(key).second)
        result.special.push_back({mapped_path, "ephemeral-data", mapped_device,
                                  mapped_inode, 0, 0, true});
      continue;
    }
    char resolved[PATH_MAX];
    require(realpath(mapped_path.c_str(), resolved) != nullptr,
            "could not canonicalize a runtime mapping");
    const int descriptor = open(resolved, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    require(descriptor >= 0,
            "could not securely open a runtime mapping pathname");
    struct stat info;
    const int status = fstat(descriptor, &info);
    close(descriptor);
    require(status == 0, "could not inspect a runtime mapping");
    require(static_cast<std::uint64_t>(info.st_dev) ==
                    mapped_device &&
                static_cast<std::uint64_t>(info.st_ino) == mapped_inode,
            "runtime mapping pathname no longer names the mapped inode");
    if (!S_ISREG(info.st_mode)) {
      const bool nvidia = nvidia_character_device_path(resolved);
      const bool ephemeral = ephemeral_data_mapping_path(resolved);
      require(!executable &&
                  ((nvidia && S_ISCHR(info.st_mode)) || ephemeral),
              "unexpected special or executable runtime mapping");
      const auto key = std::make_pair(
          std::string(resolved), static_cast<std::uint64_t>(info.st_ino));
      if (special_seen.insert(key).second)
        result.special.push_back(
            {resolved, nvidia ? "nvidia-character-device" : "ephemeral-data",
             mapped_device, static_cast<std::uint64_t>(info.st_ino),
             static_cast<std::uint64_t>(info.st_mode),
             static_cast<std::uint64_t>(info.st_rdev), false});
      continue;
    }
    require(mapped_inode > 0,
            "regular runtime mapping has an invalid inode identity");
    const auto identity = std::make_pair(
        static_cast<std::uint64_t>(info.st_dev),
        static_cast<std::uint64_t>(info.st_ino));
    regular_candidate &candidate = candidates[identity];
    if (candidate.record.path.empty())
      candidate.record =
          {resolved, identity.first, identity.second,
           static_cast<std::uint64_t>(info.st_size),
           static_cast<std::int64_t>(info.st_mtim.tv_sec) * INT64_C(1000000000) +
               info.st_mtim.tv_nsec,
           static_cast<std::int64_t>(info.st_ctim.tv_sec) * INT64_C(1000000000) +
               info.st_ctim.tv_nsec};
    candidate.executable = candidate.executable || executable;
    candidate.required =
        candidate.required || runtime_family_required(candidate.record.path);
  }
  for (const auto &entry : candidates)
    if (entry.second.executable || entry.second.required)
      result.regular.push_back(entry.second.record);
  std::sort(result.regular.begin(), result.regular.end(),
            [](const mapped_runtime_file &left,
               const mapped_runtime_file &right) {
              if (left.path != right.path) return left.path < right.path;
              if (left.device != right.device) return left.device < right.device;
              return left.inode < right.inode;
            });
  std::sort(result.special.begin(), result.special.end(),
            [](const mapped_runtime_special &left,
               const mapped_runtime_special &right) {
              if (left.path != right.path) return left.path < right.path;
              return left.inode < right.inode;
            });
  require(!result.regular.empty(), "runtime mapping attestation is empty");
  return result;
}

void emit_rank_records_one_writer(const std::string &local_record) {
  require(local_record.size() <= static_cast<std::size_t>(INT_MAX),
          "structured rank record is too large for MPI gathering");
#ifdef HAVE_MPI
  const int rank = my_rank();
  const int ranks = count_processors();
  const int local_size = static_cast<int>(local_record.size());
  std::vector<int> sizes(rank == 0 ? ranks : 0);
  require(MPI_Gather(&local_size, 1, MPI_INT,
                     rank == 0 ? sizes.data() : nullptr, 1, MPI_INT, 0,
                     MPI_COMM_WORLD) == MPI_SUCCESS,
          "could not gather structured rank record sizes");
  std::vector<int> displacements;
  std::vector<char> payload;
  if (rank == 0) {
    displacements.resize(ranks);
    std::size_t total = 0;
    for (int index = 0; index < ranks; ++index) {
      require(sizes[index] >= 0 &&
                  total + static_cast<std::size_t>(sizes[index]) <=
                      static_cast<std::size_t>(INT_MAX),
              "gathered structured rank payload is too large");
      displacements[index] = static_cast<int>(total);
      total += static_cast<std::size_t>(sizes[index]);
    }
    payload.resize(total);
  }
  require(MPI_Gatherv(local_record.data(), local_size, MPI_CHAR,
                      rank == 0 ? payload.data() : nullptr,
                      rank == 0 ? sizes.data() : nullptr,
                      rank == 0 ? displacements.data() : nullptr, MPI_CHAR, 0,
                      MPI_COMM_WORLD) == MPI_SUCCESS,
          "could not gather structured rank records");
  if (rank == 0) {
    for (int index = 0; index < ranks; ++index)
      std::cout.write(payload.data() + displacements[index], sizes[index])
          << '\n';
    std::cout << std::flush;
  }
#else
  if (am_master()) std::cout << local_record << '\n' << std::flush;
#endif
}

void write_new_file_atomically(const std::string &path,
                               const std::string &payload) {
  require(!path.empty() && path[0] == '/',
          "dense evidence output path must be absolute");
  struct stat existing;
  errno = 0;
  require(lstat(path.c_str(), &existing) != 0 && errno == ENOENT,
          "refusing to replace stale dense long-horizon evidence");

  const std::string temporary =
      path + ".tmp." + std::to_string(static_cast<long long>(getpid()));
  const int descriptor = open(
      temporary.c_str(),
      O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
  require(descriptor >= 0,
          "could not securely create dense evidence temporary file");
  bool descriptor_open = true;
  bool temporary_exists = true;
  try {
    std::size_t written = 0;
    while (written < payload.size()) {
      const ssize_t count = write(
          descriptor, payload.data() + written, payload.size() - written);
      if (count < 0 && errno == EINTR) continue;
      require(count > 0, "failed while writing dense long-horizon evidence");
      written += static_cast<std::size_t>(count);
    }
    require(fsync(descriptor) == 0,
            "could not fsync dense long-horizon evidence");
    require(close(descriptor) == 0,
            "could not close dense long-horizon evidence");
    descriptor_open = false;
    require(syscall(SYS_renameat2, AT_FDCWD, temporary.c_str(), AT_FDCWD,
                    path.c_str(), RENAME_NOREPLACE) == 0,
            "refusing to replace raced or stale dense evidence output");
    temporary_exists = false;

    const std::size_t separator = path.rfind('/');
    const std::string parent = separator == 0 ? "/" : path.substr(0, separator);
    const int directory = open(
        parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    require(directory >= 0,
            "could not open dense evidence directory for fsync");
    const int sync_result = fsync(directory);
    const int close_result = close(directory);
    require(sync_result == 0 && close_result == 0,
            "could not fsync dense evidence directory");
  }
  catch (...) {
    if (descriptor_open) close(descriptor);
    if (temporary_exists) unlink(temporary.c_str());
    throw;
  }
}

std::vector<int> cpu_affinity_for_task(pid_t task) {
  cpu_set_t set;
  CPU_ZERO(&set);
  require(sched_getaffinity(task, sizeof(set), &set) == 0,
          "could not read LDOS worker CPU affinity");
  std::vector<int> result;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
    if (CPU_ISSET(cpu, &set)) result.push_back(cpu);
  require(!result.empty(), "LDOS worker CPU affinity is empty");
  return result;
}

std::vector<int> current_cpu_affinity() {
  return cpu_affinity_for_task(0);
}

void preinitialize_check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<int> parse_ldos_cpu_list(const char *text) {
  preinitialize_check(
      text && *text,
      "focused LDOS benchmark requires an explicit Linux CPU list");
  std::vector<int> result;
  const char *cursor = text;
  while (*cursor) {
    preinitialize_check(
        std::isdigit(static_cast<unsigned char>(*cursor)),
        "focused LDOS CPU list is not canonical");
    preinitialize_check(
        *cursor != '0' || !std::isdigit(
            static_cast<unsigned char>(cursor[1])),
        "focused LDOS CPU list has a leading zero");
    unsigned long value = 0;
    do {
      const unsigned int digit = static_cast<unsigned int>(*cursor - '0');
      preinitialize_check(
          value <= (static_cast<unsigned long>(CPU_SETSIZE - 1) - digit) / 10,
          "focused LDOS CPU list exceeds CPU_SETSIZE");
      value = value * 10 + digit;
      ++cursor;
    } while (std::isdigit(static_cast<unsigned char>(*cursor)));
    preinitialize_check(
        std::find(result.begin(), result.end(), static_cast<int>(value)) ==
            result.end(),
        "focused LDOS CPU list contains a duplicate");
    result.push_back(static_cast<int>(value));
    if (!*cursor) break;
    preinitialize_check(
        *cursor == ',' && cursor[1],
        "focused LDOS CPU list has invalid separators");
    ++cursor;
  }
  preinitialize_check(
      std::is_sorted(result.begin(), result.end()),
      "focused LDOS CPU list must be sorted");
  return result;
}

void preinitialize_ldos_cpu_affinity() {
  if (!std::getenv("MEEP_GPU_TEST_LDOS_BENCHMARK_ONLY")) return;
  const std::vector<int> cpus =
      parse_ldos_cpu_list(std::getenv("MEEP_GPU_LDOS_CPU_LIST"));
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int cpu : cpus) CPU_SET(cpu, &set);
  preinitialize_check(
      sched_setaffinity(0, sizeof(set), &set) == 0,
      "could not pre-initialize focused LDOS worker CPU affinity");
  cpu_set_t observed;
  CPU_ZERO(&observed);
  preinitialize_check(
      sched_getaffinity(0, sizeof(observed), &observed) == 0,
      "could not verify focused LDOS worker CPU affinity");
  std::vector<int> observed_cpus;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
    if (CPU_ISSET(cpu, &observed)) observed_cpus.push_back(cpu);
  preinitialize_check(
      observed_cpus == cpus,
      "focused LDOS worker CPU affinity was not applied exactly");
}

unsigned long long process_start_time_ticks() {
  std::ifstream stream("/proc/self/stat");
  std::string text;
  std::getline(stream, text);
  const std::size_t close = text.rfind(')');
  require(stream.good() || stream.eof(),
          "could not read focused LDOS worker process identity");
  require(close != std::string::npos && close + 2 < text.size(),
          "focused LDOS worker process identity is malformed");
  std::istringstream fields(text.substr(close + 2));
  std::string field;
  for (int index = 0; index <= 19; ++index)
    require(static_cast<bool>(fields >> field),
            "focused LDOS worker process identity is incomplete");
  char *end = nullptr;
  errno = 0;
  const unsigned long long value = std::strtoull(field.c_str(), &end, 10);
  require(errno == 0 && end && *end == '\0' && value > 0,
          "focused LDOS worker start time is invalid");
  return value;
}

std::vector<pid_t> process_task_ids() {
  DIR *directory = opendir("/proc/self/task");
  require(directory != nullptr,
          "could not enumerate focused LDOS worker tasks");
  std::vector<pid_t> result;
  errno = 0;
  while (dirent *entry = readdir(directory)) {
    char *end = nullptr;
    const long value = std::strtol(entry->d_name, &end, 10);
    if (end && *end == '\0' && value > 0 && value <= INT_MAX)
      result.push_back(static_cast<pid_t>(value));
  }
  const int read_error = errno;
  const int close_error = closedir(directory);
  require(read_error == 0 && close_error == 0,
          "could not finish enumerating focused LDOS worker tasks");
  std::sort(result.begin(), result.end());
  require(!result.empty(), "focused LDOS worker has no observable tasks");
  return result;
}

void append_json_integer_array(std::ostringstream &output,
                               const std::vector<int> &values) {
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) output << ',';
    output << values[index];
  }
  output << ']';
}

void append_timed_updates_json(
    std::ostringstream &output,
    const std::vector<ldos_benchmark_result::timed_update_record> &updates) {
  output << '[';
  for (std::size_t index = 0; index < updates.size(); ++index) {
    if (index) output << ',';
    const ldos_benchmark_result::timed_update_record &record = updates[index];
    output << "{\"start_monotonic_ns\":" << record.start_monotonic_ns
           << ",\"stop_monotonic_ns\":" << record.stop_monotonic_ns
           << ",\"cpu_before\":" << record.cpu_before
           << ",\"cpu_after\":" << record.cpu_after
           << ",\"affinity_before\":";
    append_json_integer_array(output, record.affinity_before);
    output << ",\"affinity_after\":";
    append_json_integer_array(output, record.affinity_after);
    output << ",\"elapsed_ns\":" << record.elapsed_ns
           << ",\"voluntary_context_switches\":"
           << record.voluntary_context_switches
           << ",\"involuntary_context_switches\":"
           << record.involuntary_context_switches
           << ",\"minor_faults\":" << record.minor_faults
           << ",\"major_faults\":" << record.major_faults;
    output << '}';
  }
  output << ']';
}

std::vector<int> wait_for_ldos_controller_gate() {
  const char *directory = std::getenv("MEEP_GPU_LDOS_CONTROL_DIR");
  const char *nonce = std::getenv("MEEP_GPU_LDOS_NONCE");
  require(directory && directory[0] == '/',
          "focused LDOS benchmark requires an absolute controller directory");
  require(nonce && std::strlen(nonce) == 64 &&
              std::all_of(nonce, nonce + 64, [](char value) {
                return std::isdigit(static_cast<unsigned char>(value)) ||
                       (value >= 'a' && value <= 'f');
              }),
          "focused LDOS benchmark controller nonce is invalid");
  const int rank = my_rank();
  const std::string ready_path =
      std::string(directory) + "/rank-" + std::to_string(rank) + ".ready";
  std::ostringstream ready;
  ready << "{\"schema_version\":1,\"rank\":" << rank
        << ",\"pid\":" << static_cast<long long>(getpid())
        << ",\"start_time_ticks\":" << process_start_time_ticks()
        << ",\"device_ordinal\":" << gpu::selected_device()
        << ",\"device_uuid\":\""
        << json_escape(gpu::selected_device_identifier())
        << "\",\"nonce\":\"" << nonce << "\",\"task_affinities\":{";
  const std::vector<pid_t> tasks = process_task_ids();
  for (std::size_t index = 0; index < tasks.size(); ++index) {
    if (index) ready << ',';
    ready << '\"' << static_cast<long long>(tasks[index]) << "\":";
    append_json_integer_array(ready, cpu_affinity_for_task(tasks[index]));
  }
  ready << "}}";
  write_new_file_atomically(ready_path, ready.str());

  const std::string go_path = std::string(directory) + "/GO";
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(60);
  int go_descriptor = -1;
  while ((go_descriptor = open(
              go_path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC)) < 0) {
    require(errno == ENOENT,
            "could not securely open focused LDOS controller GO gate");
    require(std::chrono::steady_clock::now() < deadline,
            "timed out waiting for focused LDOS controller GO gate");
    usleep(10000);
  }
  struct stat info;
  require(fstat(go_descriptor, &info) == 0 && S_ISREG(info.st_mode) &&
              info.st_nlink == 1 && info.st_uid == geteuid() &&
              (info.st_mode & 0777) == 0400,
          "focused LDOS controller GO gate metadata is invalid");
  std::ostringstream expected_stream;
  expected_stream << "{\"schema_version\":1,\"nonce\":\"" << nonce
                  << "\",\"ranks\":" << count_processors() << "}\n";
  const std::string expected = expected_stream.str();
  std::vector<char> bytes(expected.size() + 1);
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t count = read(
        go_descriptor, bytes.data() + offset, bytes.size() - offset);
    require(count >= 0 || errno == EINTR,
            "could not read focused LDOS controller GO gate");
    if (count < 0) continue;
    if (count == 0) break;
    offset += static_cast<std::size_t>(count);
  }
  const int close_result = close(go_descriptor);
  require(close_result == 0 && offset == expected.size() &&
              std::memcmp(bytes.data(), expected.data(), expected.size()) == 0,
          "focused LDOS controller GO gate bytes differ from binding");
  const std::vector<int> affinity = current_cpu_affinity();
  all_wait();
  return affinity;
}

void publish_ldos_results_ready_and_wait_release() {
  const char *directory = std::getenv("MEEP_GPU_LDOS_CONTROL_DIR");
  const char *nonce = std::getenv("MEEP_GPU_LDOS_NONCE");
  require(directory && directory[0] == '/' && nonce &&
              std::strlen(nonce) == 64 &&
              std::all_of(nonce, nonce + 64, [](char value) {
                return std::isdigit(static_cast<unsigned char>(value)) ||
                       (value >= 'a' && value <= 'f');
              }),
          "focused LDOS completion handshake environment is invalid");
  struct timespec monotonic;
  require(clock_gettime(CLOCK_MONOTONIC, &monotonic) == 0 &&
              monotonic.tv_sec >= 0 && monotonic.tv_nsec >= 0 &&
              monotonic.tv_nsec < 1000000000L,
          "could not timestamp focused LDOS RESULTS_READY");
  const std::uint64_t monotonic_ns =
      static_cast<std::uint64_t>(monotonic.tv_sec) * 1000000000ULL +
      static_cast<std::uint64_t>(monotonic.tv_nsec);
  const int rank = my_rank();
  const std::string ready_path =
      std::string(directory) + "/rank-" + std::to_string(rank) +
      ".results-ready";
  std::ostringstream ready;
  ready << "{\"schema_version\":1,\"state\":\"RESULTS_READY\",\"rank\":"
        << rank << ",\"pid\":" << static_cast<long long>(getpid())
        << ",\"start_time_ticks\":" << process_start_time_ticks()
        << ",\"device_ordinal\":" << gpu::selected_device()
        << ",\"device_uuid\":\""
        << json_escape(gpu::selected_device_identifier())
        << "\",\"nonce\":\"" << nonce << "\",\"monotonic_ns\":"
        << monotonic_ns << '}';
  write_new_file_atomically(ready_path, ready.str());

  const std::string release_path = std::string(directory) + "/RELEASE";
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(60);
  int release_descriptor = -1;
  while ((release_descriptor = open(
              release_path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC)) < 0) {
    require(errno == ENOENT,
            "could not securely open focused LDOS RELEASE gate");
    require(std::chrono::steady_clock::now() < deadline,
            "timed out waiting for focused LDOS RELEASE gate");
    usleep(10000);
  }
  struct stat info;
  require(fstat(release_descriptor, &info) == 0 && S_ISREG(info.st_mode) &&
              info.st_nlink == 1 && info.st_uid == geteuid() &&
              (info.st_mode & 0777) == 0400,
          "focused LDOS RELEASE gate metadata is invalid");
  std::ostringstream expected_stream;
  expected_stream << "{\"schema_version\":1,\"nonce\":\"" << nonce
                  << "\",\"ranks\":" << count_processors()
                  << ",\"state\":\"RELEASE\"}\n";
  const std::string expected = expected_stream.str();
  std::vector<char> bytes(expected.size() + 1);
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t count = read(
        release_descriptor, bytes.data() + offset, bytes.size() - offset);
    require(count >= 0 || errno == EINTR,
            "could not read focused LDOS RELEASE gate");
    if (count < 0) continue;
    if (count == 0) break;
    offset += static_cast<std::size_t>(count);
  }
  const int close_result = close(release_descriptor);
  require(close_result == 0 && offset == expected.size() &&
              std::memcmp(bytes.data(), expected.data(), expected.size()) == 0,
          "focused LDOS RELEASE gate bytes differ from binding");
}

void write_dense_long_evidence(const char *path,
                               const dense_long_run_result &result,
                               gpu::backend_mode mode) {
  require(path && *path, "dense long-horizon evidence path is empty");
  require(mode == gpu::backend_mode::cpu ||
              mode == gpu::backend_mode::cuda,
          "dense long-horizon evidence backend must be cpu or cuda");
#if MEEP_SINGLE
  const char *lane = mode == gpu::backend_mode::cuda
                         ? "cuda-fp32"
                         : "cpu-fp32";
  constexpr int precision_bits = 32;
#else
  require(mode == gpu::backend_mode::cpu,
          "the CPU-FP64 evidence lane cannot request CUDA");
  const char *lane = "cpu-fp64";
  constexpr int precision_bits = 64;
#endif

  std::size_t scalar_count = 0;
  for (const dense_field_array &array : result.arrays) {
    require(array.chunk_index >= 0 && array.component_index >= 0 &&
                array.component_index < NUM_FIELD_COMPONENTS &&
                (array.complex_part == 0 || array.complex_part == 1) &&
                !array.values.empty(),
            "dense long-horizon evidence array metadata is invalid");
    require(array.values.size() <=
                std::numeric_limits<std::size_t>::max() - scalar_count,
            "dense long-horizon evidence scalar count overflowed");
    scalar_count += array.values.size();
    for (double value : array.values)
      require(std::isfinite(value),
              "dense long-horizon evidence contains a non-finite field");
  }
  require(!result.arrays.empty() && scalar_count > 0 &&
              result.executed_timesteps > 0 &&
              std::isfinite(result.grid_sx) &&
              std::isfinite(result.grid_sy) &&
              std::isfinite(result.resolution) &&
              std::isfinite(result.source_frequency) &&
              std::isfinite(result.energy),
          "dense long-horizon evidence is empty or has non-finite energy");
  for (const auto &source : result.source_parameters)
    for (double parameter : source)
      require(std::isfinite(parameter),
              "dense long-horizon source identity is non-finite");
  const std::string executable_path = current_executable_path();
  const std::string libmeep_path = loaded_libmeep_path();
  require(executable_path.find('\n') == std::string::npos &&
              executable_path.find('=') == std::string::npos &&
              libmeep_path.find('\n') == std::string::npos &&
              libmeep_path.find('=') == std::string::npos,
          "runtime provenance path cannot be encoded in dense evidence");

  std::ostringstream output;
  output << "GPMEEP_DENSE_LONG_HORIZON_EVIDENCE_V1\n"
         << "lane=" << lane << '\n'
         << "precision_bits=" << precision_bits << '\n'
         << "backend="
         << (mode == gpu::backend_mode::cuda ? "cuda" : "cpu") << '\n'
         << "executable_path=" << executable_path << '\n'
         << "libmeep_path=" << libmeep_path << '\n'
         << "timesteps=" << result.executed_timesteps << '\n'
         << "grid_sx=" << std::hexfloat << result.grid_sx << '\n'
         << "grid_sy=" << std::hexfloat << result.grid_sy << '\n'
         << "resolution=" << std::hexfloat << result.resolution << '\n'
         << "source_count=2\n"
         << "source_frequency=" << std::hexfloat
         << result.source_frequency << '\n';
  for (int source_index = 0; source_index < 2; ++source_index)
    output << "source" << source_index << "_component="
           << std::defaultfloat << result.source_components[source_index]
           << '\n'
           << "source" << source_index << "_x=" << std::hexfloat
           << result.source_parameters[source_index][0] << '\n'
           << "source" << source_index << "_y=" << std::hexfloat
           << result.source_parameters[source_index][1] << '\n'
           << "source" << source_index << "_real=" << std::hexfloat
           << result.source_parameters[source_index][2] << '\n'
           << "source" << source_index << "_imag=" << std::hexfloat
           << result.source_parameters[source_index][3] << '\n';
  output
         << std::defaultfloat
         << "array_count=" << result.arrays.size() << '\n'
         << "scalar_count=" << scalar_count << '\n'
         << "cpu_curl_calls=" << result.statistics.cpu_curl_calls << '\n'
         << "cuda_curl_calls=" << result.statistics.cuda_curl_calls << '\n'
         << "cpu_update_eh_calls="
         << result.field_updates.cpu_update_eh_calls << '\n'
         << "cuda_update_eh_calls="
         << result.field_updates.cuda_update_eh_calls << '\n'
         << "energy=" << std::hexfloat << result.energy << '\n'
         << std::defaultfloat;
  for (const dense_field_array &array : result.arrays) {
    output << "array=" << array.chunk_index << ','
           << array.component_index << ',' << array.complex_part << ','
           << array.values.size() << '\n';
    for (double value : array.values)
      output << "value=" << std::hexfloat << value << '\n';
    output << std::defaultfloat;
  }
  output << "end=1\n";
  require(output.good(), "failed while formatting dense long-horizon evidence");
  write_new_file_atomically(path, output.str());
}

std::uint64_t dense_fixture_hash(const dense_long_run_result &result) {
  // This checksum detects accidental fixture truncation or mismatch.  Release
  // evidence binds the fixture itself with SHA-256 outside this test binary.
  std::uint64_t hash = 1469598103934665603ULL;
  auto add_bytes = [&](const void *data, std::size_t size) {
    const unsigned char *bytes =
        static_cast<const unsigned char *>(data);
    for (std::size_t index = 0; index < size; ++index) {
      hash ^= bytes[index];
      hash *= 1099511628211ULL;
    }
  };
  for (const dense_field_array &array : result.arrays) {
    add_bytes(&array.chunk_index, sizeof(array.chunk_index));
    add_bytes(&array.component_index, sizeof(array.component_index));
    add_bytes(&array.complex_part, sizeof(array.complex_part));
    const std::uint64_t count =
        static_cast<std::uint64_t>(array.values.size());
    add_bytes(&count, sizeof(count));
    if (!array.values.empty())
      add_bytes(array.values.data(), array.values.size() * sizeof(double));
  }
  add_bytes(&result.energy, sizeof(result.energy));
  return hash;
}

#if !MEEP_SINGLE
void write_dense_fp64_fixture(const char *path,
                              const dense_long_run_result &result) {
  require(path && *path, "FP64 dense fixture output path is empty");
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  require(output.good(), "could not create FP64 dense fixture");
  const std::uint64_t magic = 0x47504d44464c4431ULL; // GPMDFLD1
  const std::uint64_t array_count =
      static_cast<std::uint64_t>(result.arrays.size());
  output.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
  output.write(reinterpret_cast<const char *>(&array_count),
               sizeof(array_count));
  for (const dense_field_array &array : result.arrays) {
    const std::int32_t metadata[] = {
        static_cast<std::int32_t>(array.chunk_index),
        static_cast<std::int32_t>(array.component_index),
        static_cast<std::int32_t>(array.complex_part)};
    const std::uint64_t count =
        static_cast<std::uint64_t>(array.values.size());
    output.write(reinterpret_cast<const char *>(metadata), sizeof(metadata));
    output.write(reinterpret_cast<const char *>(&count), sizeof(count));
    if (count)
      output.write(
          reinterpret_cast<const char *>(array.values.data()),
          static_cast<std::streamsize>(count * sizeof(double)));
  }
  output.write(reinterpret_cast<const char *>(&result.energy),
               sizeof(result.energy));
  const std::uint64_t hash = dense_fixture_hash(result);
  output.write(reinterpret_cast<const char *>(&hash), sizeof(hash));
  output.flush();
  require(output.good(), "failed while writing FP64 dense fixture");
}
#endif

#if MEEP_SINGLE
dense_long_run_result read_dense_fp64_fixture(const char *path) {
  require(path && *path, "FP64 dense fixture input path is empty");
  std::ifstream input(path, std::ios::binary);
  require(input.good(), "could not open FP64 dense fixture");
  std::uint64_t magic = 0;
  std::uint64_t array_count = 0;
  input.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  input.read(reinterpret_cast<char *>(&array_count), sizeof(array_count));
  require(input.good() && magic == 0x47504d44464c4431ULL &&
              array_count > 0 && array_count <= 4096,
          "FP64 dense fixture header is invalid");
  dense_long_run_result result;
  result.arrays.reserve(static_cast<std::size_t>(array_count));
  std::uint64_t total_values = 0;
  for (std::uint64_t array_index = 0; array_index < array_count;
       ++array_index) {
    std::int32_t metadata[3] = {};
    std::uint64_t count = 0;
    input.read(reinterpret_cast<char *>(metadata), sizeof(metadata));
    input.read(reinterpret_cast<char *>(&count), sizeof(count));
    total_values += count;
    require(input.good() && metadata[0] >= 0 && metadata[1] >= 0 &&
                metadata[1] < NUM_FIELD_COMPONENTS &&
                (metadata[2] == 0 || metadata[2] == 1) && count > 0 &&
                count <= 1000000 && total_values <= 10000000,
            "FP64 dense fixture array metadata is invalid");
    dense_field_array array;
    array.chunk_index = metadata[0];
    array.component_index = metadata[1];
    array.complex_part = metadata[2];
    array.values.resize(static_cast<std::size_t>(count));
    input.read(reinterpret_cast<char *>(array.values.data()),
               static_cast<std::streamsize>(count * sizeof(double)));
    require(input.good(), "FP64 dense fixture array is truncated");
    result.arrays.push_back(std::move(array));
  }
  std::uint64_t stored_hash = 0;
  input.read(reinterpret_cast<char *>(&result.energy),
             sizeof(result.energy));
  input.read(reinterpret_cast<char *>(&stored_hash), sizeof(stored_hash));
  require(input.good() && input.peek() == std::ifstream::traits_type::eof() &&
              stored_hash == dense_fixture_hash(result),
          "FP64 dense fixture footer or checksum is invalid");
  return result;
}

struct dense_error_metrics {
  std::size_t scalar_count = 0;
  double linf_absolute = 0.0;
  double linf_relative = 0.0;
  double l2_relative = 0.0;
  double energy_relative = 0.0;
};

dense_error_metrics compare_dense_field_arrays(
    const dense_long_run_result &reference,
    const dense_long_run_result &candidate,
    const char *comparison_name) {
  require(reference.arrays.size() == candidate.arrays.size(),
          std::string(comparison_name) + " field-array count differs");
  dense_error_metrics metrics;
  long double squared_error = 0.0L;
  long double squared_reference = 0.0L;
  double reference_linf = 0.0;
  for (std::size_t array_index = 0;
       array_index < reference.arrays.size(); ++array_index) {
    const dense_field_array &expected = reference.arrays[array_index];
    const dense_field_array &actual = candidate.arrays[array_index];
    require(expected.chunk_index == actual.chunk_index &&
                expected.component_index == actual.component_index &&
                expected.complex_part == actual.complex_part &&
                expected.values.size() == actual.values.size(),
            std::string(comparison_name) +
                " field-array metadata differs");
    for (std::size_t index = 0; index < expected.values.size(); ++index) {
      const double expected_value = expected.values[index];
      const double actual_value = actual.values[index];
      require(std::isfinite(expected_value) && std::isfinite(actual_value),
              std::string(comparison_name) +
                  " encountered a non-finite field value");
      const double error = std::abs(expected_value - actual_value);
      metrics.linf_absolute = std::max(metrics.linf_absolute, error);
      reference_linf = std::max(reference_linf,
                                std::abs(expected_value));
      squared_error += static_cast<long double>(error) *
                       static_cast<long double>(error);
      squared_reference += static_cast<long double>(expected_value) *
                           static_cast<long double>(expected_value);
      ++metrics.scalar_count;
    }
  }
  require(reference_linf > 0.0 && squared_reference > 0.0L,
          std::string(comparison_name) + " reference field is zero");
  metrics.linf_relative = metrics.linf_absolute / reference_linf;
  metrics.l2_relative = std::sqrt(
      static_cast<double>(squared_error / squared_reference));
  metrics.energy_relative =
      std::abs(reference.energy - candidate.energy) /
      std::max(std::abs(reference.energy), 1e-30);
  require(std::isfinite(metrics.linf_relative) &&
              std::isfinite(metrics.l2_relative) &&
              std::isfinite(metrics.energy_relative),
          std::string(comparison_name) + " norm is non-finite");
  return metrics;
}
#endif

void require_dense_long_field_agreement() {
#if MEEP_SINGLE
  constexpr int timesteps = 1024;
  constexpr double maximum_linf_absolute = 5e-5;
  constexpr double maximum_linf_relative = 1e-6;
  constexpr double maximum_l2_relative = 8e-6;
  constexpr double maximum_energy_relative = 4e-6;

  const dense_long_run_result cpu =
      run_dense_long_field_case(gpu::backend_mode::cpu);
  const dense_long_run_result cuda =
      run_dense_long_field_case(gpu::backend_mode::cuda);
  require(cpu.arrays.size() == cuda.arrays.size() &&
              cpu.arrays.size() >= 24,
          "dense long-horizon CPU/CUDA field-array topology differs or did "
          "not cover all Cartesian E/H/D/B complex arrays");

  std::size_t scalar_count = 0;
  long double squared_error = 0.0L;
  long double squared_reference = 0.0L;
  double linf_absolute = 0.0;
  double reference_linf = 0.0;
  bool covered_component_part[NUM_FIELD_COMPONENTS][2] = {};
  for (std::size_t array_index = 0; array_index < cpu.arrays.size();
       ++array_index) {
    const dense_field_array &reference = cpu.arrays[array_index];
    const dense_field_array &candidate = cuda.arrays[array_index];
    require(reference.chunk_index == candidate.chunk_index &&
                reference.component_index == candidate.component_index &&
                reference.complex_part == candidate.complex_part &&
                reference.values.size() == candidate.values.size(),
            "dense long-horizon CPU/CUDA field-array metadata differs");
    covered_component_part[reference.component_index]
                          [reference.complex_part] = true;
    for (std::size_t index = 0; index < reference.values.size(); ++index) {
      const double cpu_value = reference.values[index];
      const double cuda_value = candidate.values[index];
      require(std::isfinite(cpu_value) && std::isfinite(cuda_value),
              "dense long-horizon field comparison found a non-finite "
              "value");
      const double error = std::abs(cpu_value - cuda_value);
      linf_absolute = std::max(linf_absolute, error);
      reference_linf = std::max(reference_linf, std::abs(cpu_value));
      squared_error +=
          static_cast<long double>(error) * static_cast<long double>(error);
      squared_reference += static_cast<long double>(cpu_value) *
                           static_cast<long double>(cpu_value);
      ++scalar_count;
    }
  }
  const component required_components[] = {
      Ex, Ey, Ez, Hx, Hy, Hz, Dx, Dy, Dz, Bx, By, Bz};
  for (component c : required_components)
    require(covered_component_part[c][0] &&
                covered_component_part[c][1],
            "dense long-horizon oracle omitted a Cartesian E/H/D/B real "
            "or imaginary array");
  require(scalar_count >= 20000 && reference_linf > 1e-5 &&
              squared_reference > 1e-12L,
          "dense long-horizon oracle did not cover a nontrivial whole-grid "
          "field");
  const double linf_relative = linf_absolute / reference_linf;
  const double l2_relative = std::sqrt(
      static_cast<double>(squared_error / squared_reference));
  const double energy_scale = std::max(std::abs(cpu.energy), 1e-30);
  const double energy_relative =
      std::abs(cpu.energy - cuda.energy) / energy_scale;
  require(std::isfinite(linf_relative) && std::isfinite(l2_relative) &&
              std::isfinite(cpu.energy) && std::isfinite(cuda.energy) &&
              std::isfinite(energy_relative),
          "dense long-horizon norm or energy comparison is non-finite");

  if (am_master())
    std::cout << "dense-long-field: grid=2.5x2.0 resolution=16 steps="
              << timesteps << " arrays=" << cpu.arrays.size()
              << " scalars=" << scalar_count
              << " linf_abs=" << linf_absolute << "/"
              << maximum_linf_absolute
              << " linf_rel=" << linf_relative << "/"
              << maximum_linf_relative
              << " l2_rel=" << l2_relative << "/"
              << maximum_l2_relative
              << " energy_rel=" << energy_relative << "/"
              << maximum_energy_relative << '\n';

  require(linf_absolute <= maximum_linf_absolute &&
              linf_relative <= maximum_linf_relative &&
              l2_relative <= maximum_l2_relative &&
              energy_relative <= maximum_energy_relative,
          "dense long-horizon CPU-FP32/CUDA-FP32 whole-field norm gate "
          "failed");

  const char *fp64_fixture_path =
      std::getenv("MEEP_GPU_TEST_DENSE_FP64_REFERENCE");
  if (fp64_fixture_path) {
    constexpr double maximum_fp64_linf_absolute = 5e-3;
    constexpr double maximum_fp64_linf_relative = 2e-4;
    constexpr double maximum_fp64_l2_relative = 2e-4;
    constexpr double maximum_fp64_energy_relative = 2e-4;
    const dense_long_run_result fp64 =
        read_dense_fp64_fixture(fp64_fixture_path);
    const dense_error_metrics fp32_error = compare_dense_field_arrays(
        fp64, cpu, "CPU-FP64/CPU-FP32 dense comparison");
    const dense_error_metrics cuda_error = compare_dense_field_arrays(
        fp64, cuda, "CPU-FP64/CUDA-FP32 dense comparison");
    require(fp32_error.scalar_count == scalar_count &&
                cuda_error.scalar_count == scalar_count,
            "FP64 dense fixture did not cover the same complete field");
    if (am_master())
      std::cout
          << "dense-fp64-reference: fixture-fnv64=0x" << std::hex
          << dense_fixture_hash(fp64) << std::dec
          << " cpu-fp32-linf_abs=" << fp32_error.linf_absolute
          << " cuda-fp32-linf_abs=" << cuda_error.linf_absolute
          << "/" << maximum_fp64_linf_absolute
          << " cpu-fp32-linf_rel=" << fp32_error.linf_relative
          << " cuda-fp32-linf_rel=" << cuda_error.linf_relative
          << "/" << maximum_fp64_linf_relative
          << " cpu-fp32-l2_rel=" << fp32_error.l2_relative
          << " cuda-fp32-l2_rel=" << cuda_error.l2_relative
          << "/" << maximum_fp64_l2_relative
          << " cpu-fp32-energy_rel=" << fp32_error.energy_relative
          << " cuda-fp32-energy_rel=" << cuda_error.energy_relative
          << "/" << maximum_fp64_energy_relative << '\n';
    require(
        fp32_error.linf_absolute <= maximum_fp64_linf_absolute &&
            cuda_error.linf_absolute <= maximum_fp64_linf_absolute &&
            fp32_error.linf_relative <= maximum_fp64_linf_relative &&
            cuda_error.linf_relative <= maximum_fp64_linf_relative &&
            fp32_error.l2_relative <= maximum_fp64_l2_relative &&
            cuda_error.l2_relative <= maximum_fp64_l2_relative &&
            fp32_error.energy_relative <= maximum_fp64_energy_relative &&
            cuda_error.energy_relative <= maximum_fp64_energy_relative,
        "CPU-FP32 or CUDA-FP32 dense whole-field result exceeded the "
        "independent CPU-FP64 reference gate");
  }
  require(cpu.statistics.cpu_curl_calls > 0 &&
              cpu.statistics.cuda_curl_calls == 0 &&
              cuda.statistics.cuda_curl_calls > 0 &&
              cuda.statistics.cpu_curl_calls == 0 &&
              cuda.field_updates.cuda_update_eh_calls > 0 &&
              cuda.field_updates.cpu_update_eh_calls == 0,
          "dense long-horizon oracle did not use exclusive CPU and CUDA "
          "field-update paths");
#endif
}

void require_phase_batched_launch_equivalence() {
#if MEEP_SINGLE
  const bool curl_replay_globally_disabled =
      std::getenv("MEEP_GPU_DISABLE_CURL_PHASE_REPLAY") != nullptr;
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  run_result fused;
  {
    scoped_environment_override curl_enable(
        "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL", "1");
    scoped_environment_override update_enable(
        "MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH", "1");
    scoped_environment_override source_enable(
        "MEEP_GPU_ENABLE_PHASE_BATCHED_SOURCE", "1");
    scoped_environment_override curl_batch(
        "MEEP_GPU_DISABLE_PHASE_BATCHED_CURL", nullptr);
    scoped_environment_override update_batch(
        "MEEP_GPU_DISABLE_PHASE_BATCHED_UPDATE_EH", nullptr);
    scoped_environment_override source_batch(
        "MEEP_GPU_DISABLE_PHASE_BATCHED_SOURCE", nullptr);
    require(
        gpu::detail::phase_batched_curl_mode() ==
                gpu::detail::phase_batch_mode::forced &&
            gpu::detail::phase_batched_update_eh_mode() ==
                gpu::detail::phase_batch_mode::forced,
        "forced phase-batch fixture did not install both policies");
    fused = run_material_case(gpu::backend_mode::cuda, false, 64, true);
  }
  const gpu::phase_batch_policy_statistics fused_policy =
      gpu::get_phase_batch_policy_statistics();
  require(gpu::get_live_resident_device_buffers() == live_before,
          "phase-batched CUDA run leaked a cached descriptor or mirror");

  run_result automatic;
  {
    scoped_environment_override curl_enable(
        "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL", nullptr);
    scoped_environment_override update_enable(
        "MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH", nullptr);
    scoped_environment_override source_enable(
        "MEEP_GPU_ENABLE_PHASE_BATCHED_SOURCE", nullptr);
    scoped_environment_override curl_disable(
        "MEEP_GPU_DISABLE_PHASE_BATCHED_CURL", nullptr);
    scoped_environment_override update_disable(
        "MEEP_GPU_DISABLE_PHASE_BATCHED_UPDATE_EH", nullptr);
    scoped_environment_override source_disable(
        "MEEP_GPU_DISABLE_PHASE_BATCHED_SOURCE", nullptr);
    automatic =
        run_material_case(gpu::backend_mode::cuda, false, 64, true);
  }
  const gpu::phase_batch_policy_statistics automatic_policy =
      gpu::get_phase_batch_policy_statistics();
  const gpu::curl_phase_replay_statistics automatic_replay =
      gpu::get_curl_phase_replay_statistics();
  require(gpu::get_live_resident_device_buffers() == live_before,
          "automatic phase-batched CUDA run leaked a resident buffer");
  compare_results(automatic, fused, two_dimensional_case);
  require(
      fused_policy.curl_forced_batches > 0 &&
          fused_policy.update_eh_forced_batches > 0 &&
          fused_policy.curl_batched_operations > 0 &&
          fused_policy.update_eh_batched_operations > 0,
      "forced phase-batch statistics did not prove batched operations: "
      "curl-forced=" +
          std::to_string(fused_policy.curl_forced_batches) +
          " update-eh-forced=" +
          std::to_string(fused_policy.update_eh_forced_batches) +
          " curl-operations=" +
          std::to_string(fused_policy.curl_batched_operations) +
          " update-eh-operations=" +
          std::to_string(fused_policy.update_eh_batched_operations) +
          " update-eh-unbatched=" +
          std::to_string(fused_policy.update_eh_unbatched_operations) +
          " update-eh-auto-checks=" +
          std::to_string(fused_policy.update_eh_automatic_checks));
  require(
      automatic_policy.curl_automatic_checks ==
              automatic_policy.curl_automatic_selected +
                  automatic_policy.curl_automatic_rejected &&
          automatic_policy.update_eh_automatic_checks ==
              automatic_policy.update_eh_automatic_selected +
                  automatic_policy.update_eh_automatic_rejected &&
          automatic_policy.curl_automatic_selected > 0 &&
          automatic_policy.update_eh_automatic_selected > 0 &&
          automatic_policy.curl_batched_operations > 0 &&
          automatic_policy.update_eh_batched_operations > 0 &&
          (curl_replay_globally_disabled
               ? automatic_replay.checks == 0 &&
                     automatic_replay.hits == 0 &&
                     automatic_replay.unready == 0 &&
                     automatic_replay.generation_misses == 0 &&
                     automatic_replay.mirror_misses == 0
               : automatic_replay.checks ==
                         automatic_replay.hits + automatic_replay.unready +
                             automatic_replay.generation_misses +
                             automatic_replay.mirror_misses &&
                     automatic_replay.hits > 0) &&
          automatic.statistics.cuda_curl_calls ==
              fused.statistics.cuda_curl_calls &&
          automatic.field_updates.cuda_update_eh_calls ==
              fused.field_updates.cuda_update_eh_calls,
      "automatic phase policy did not select the favorable small topology");

  run_result replay_disabled;
  gpu::curl_phase_replay_statistics replay_disabled_statistics{};
  {
    scoped_environment_override replay_disable(
        "MEEP_GPU_DISABLE_CURL_PHASE_REPLAY", "1");
    scoped_environment_override curl_enable(
        "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL", nullptr);
    scoped_environment_override update_enable(
        "MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH", nullptr);
    scoped_environment_override source_enable(
        "MEEP_GPU_ENABLE_PHASE_BATCHED_SOURCE", nullptr);
    scoped_environment_override curl_batch(
        "MEEP_GPU_DISABLE_PHASE_BATCHED_CURL", nullptr);
    scoped_environment_override update_batch(
        "MEEP_GPU_DISABLE_PHASE_BATCHED_UPDATE_EH", nullptr);
    scoped_environment_override source_batch(
        "MEEP_GPU_DISABLE_PHASE_BATCHED_SOURCE", nullptr);
    replay_disabled =
        run_material_case(gpu::backend_mode::cuda, false, 64, true);
    replay_disabled_statistics =
        gpu::get_curl_phase_replay_statistics();
  }
  require(gpu::get_live_resident_device_buffers() == live_before,
          "curl replay-disabled control leaked a resident buffer");
  compare_results(automatic, replay_disabled, two_dimensional_case);
  require(
      replay_disabled_statistics.checks == 0 &&
          replay_disabled_statistics.hits == 0 &&
          replay_disabled_statistics.unready == 0 &&
          replay_disabled_statistics.generation_misses == 0 &&
          replay_disabled_statistics.mirror_misses == 0 &&
          replay_disabled.statistics.cuda_curl_calls ==
              automatic.statistics.cuda_curl_calls &&
          replay_disabled.statistics.cuda_curl_points ==
              automatic.statistics.cuda_curl_points,
      "curl phase replay opt-out changed results or logical dispatch");

  run_result direct_mixed;
  gpu::curl_phase_replay_statistics direct_mixed_replay{};
  {
    scoped_environment_override curl_enable(
        "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL", nullptr);
    scoped_environment_override curl_disable(
        "MEEP_GPU_DISABLE_PHASE_BATCHED_CURL", nullptr);
    direct_mixed =
        run_beta_case(gpu::backend_mode::cuda, 0.19, true);
    direct_mixed_replay = gpu::get_curl_phase_replay_statistics();
  }
  require(gpu::get_live_resident_device_buffers() == live_before,
          "curl replay mixed-direct control leaked a resident buffer");
  require(
      direct_mixed.statistics.cuda_curl_calls > 0 &&
          (curl_replay_globally_disabled
               ? direct_mixed_replay.checks == 0 &&
                     direct_mixed_replay.hits == 0
               : direct_mixed_replay.checks > 0 &&
                     direct_mixed_replay.hits == 0 &&
                     direct_mixed_replay.checks ==
                         direct_mixed_replay.unready +
                             direct_mixed_replay.generation_misses +
                             direct_mixed_replay.mirror_misses),
      "curl replay accepted a phase containing a direct beta operation");

  run_result unfused;
  {
    scoped_environment_override curl_enable(
        "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL", nullptr);
    scoped_environment_override update_enable(
        "MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH", nullptr);
    scoped_environment_override source_enable(
        "MEEP_GPU_ENABLE_PHASE_BATCHED_SOURCE", nullptr);
    scoped_environment_override curl_batch(
        "MEEP_GPU_DISABLE_PHASE_BATCHED_CURL", "1");
    scoped_environment_override update_batch(
        "MEEP_GPU_DISABLE_PHASE_BATCHED_UPDATE_EH", "1");
    scoped_environment_override source_batch(
        "MEEP_GPU_DISABLE_PHASE_BATCHED_SOURCE", nullptr);
    unfused = run_material_case(gpu::backend_mode::cuda, false, 64, true);
  }
  require(gpu::get_live_resident_device_buffers() == live_before,
          "unfused CUDA control run leaked a resident buffer");
  compare_results(unfused, fused, two_dimensional_case);
  require(
      fused.statistics.cpu_curl_calls == 0 &&
          unfused.statistics.cpu_curl_calls == 0 &&
          fused.field_updates.cpu_update_eh_calls == 0 &&
          unfused.field_updates.cpu_update_eh_calls == 0 &&
          fused.sources.cpu_update_calls == 0 &&
          unfused.sources.cpu_update_calls == 0,
      "phase-batched equivalence case silently dispatched CPU field work");
  require(
      fused.statistics.cuda_curl_points ==
              unfused.statistics.cuda_curl_points &&
          fused.field_updates.cuda_update_eh_points ==
              unfused.field_updates.cuda_update_eh_points &&
          fused.sources.cuda_update_points ==
              unfused.sources.cuda_update_points,
      "phase batching changed the number of updated field points");
  require(
      fused.statistics.cuda_curl_calls <
              unfused.statistics.cuda_curl_calls &&
          fused.field_updates.cuda_update_eh_calls <
              unfused.field_updates.cuda_update_eh_calls &&
          fused.sources.cuda_update_calls <
              unfused.sources.cuda_update_calls,
      "phase batching did not reduce curl, E/H, and source launch counts: "
      "curl=" + std::to_string(unfused.statistics.cuda_curl_calls) + "->" +
          std::to_string(fused.statistics.cuda_curl_calls) +
          " update-eh=" +
          std::to_string(unfused.field_updates.cuda_update_eh_calls) + "->" +
          std::to_string(fused.field_updates.cuda_update_eh_calls) +
          " sources=" +
          std::to_string(unfused.sources.cuda_update_calls) + "->" +
          std::to_string(fused.sources.cuda_update_calls));
  require(fused.resident.device_buffer_reuses > 0,
          "phase-batched descriptor plans were not reused");

  run_result explicitly_disabled;
  {
    scoped_environment_override curl_enable(
        "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL", "1");
    scoped_environment_override update_enable(
        "MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH", "1");
    scoped_environment_override source_enable(
        "MEEP_GPU_ENABLE_PHASE_BATCHED_SOURCE", "1");
    scoped_environment_override curl_disable(
        "MEEP_GPU_DISABLE_PHASE_BATCHED_CURL", "1");
    scoped_environment_override update_disable(
        "MEEP_GPU_DISABLE_PHASE_BATCHED_UPDATE_EH", "1");
    scoped_environment_override source_disable(
        "MEEP_GPU_DISABLE_PHASE_BATCHED_SOURCE", "1");
    explicitly_disabled =
        run_material_case(gpu::backend_mode::cuda, false, 64, true);
  }
  compare_results(unfused, explicitly_disabled, two_dimensional_case);
  require(
      explicitly_disabled.statistics.cuda_curl_calls ==
              unfused.statistics.cuda_curl_calls &&
          explicitly_disabled.field_updates.cuda_update_eh_calls ==
              unfused.field_updates.cuda_update_eh_calls &&
          explicitly_disabled.sources.cuda_update_calls ==
              unfused.sources.cuda_update_calls,
      "phase-batch disable controls did not override explicit opt-in");

  run_result zero_opt_in;
  {
    scoped_environment_override curl_enable(
        "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL", "0");
    scoped_environment_override update_enable(
        "MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH", "0");
    scoped_environment_override source_enable(
        "MEEP_GPU_ENABLE_PHASE_BATCHED_SOURCE", "0");
    scoped_environment_override curl_disable(
        "MEEP_GPU_DISABLE_PHASE_BATCHED_CURL", nullptr);
    scoped_environment_override update_disable(
        "MEEP_GPU_DISABLE_PHASE_BATCHED_UPDATE_EH", nullptr);
    scoped_environment_override source_disable(
        "MEEP_GPU_DISABLE_PHASE_BATCHED_SOURCE", nullptr);
    zero_opt_in =
        run_material_case(gpu::backend_mode::cuda, false, 64, true);
  }
  compare_results(unfused, zero_opt_in, two_dimensional_case);
  require(
      zero_opt_in.statistics.cuda_curl_calls ==
              unfused.statistics.cuda_curl_calls &&
          zero_opt_in.field_updates.cuda_update_eh_calls ==
              unfused.field_updates.cuda_update_eh_calls &&
          zero_opt_in.sources.cuda_update_calls ==
              unfused.sources.cuda_update_calls,
      "phase-batch opt-in accepted a value other than exact 1");
  if (am_master())
    std::cout
        << "phase-batched-launches: curl="
        << unfused.statistics.cuda_curl_calls << "->"
        << fused.statistics.cuda_curl_calls << " update-eh="
        << unfused.field_updates.cuda_update_eh_calls << "->"
        << fused.field_updates.cuda_update_eh_calls << " sources="
        << unfused.sources.cuda_update_calls << "->"
        << fused.sources.cuda_update_calls << " points="
        << fused.statistics.cuda_curl_points << "/"
        << fused.field_updates.cuda_update_eh_points << "/"
        << fused.sources.cuda_update_points << '\n';
#endif
}

void require_curl_replay_direct_transition_contract() {
#if MEEP_SINGLE
  const bool replay_disabled =
      std::getenv("MEEP_GPU_DISABLE_CURL_PHASE_REPLAY") != nullptr;
  const curl_replay_direct_transition transitions[] = {
      curl_replay_direct_transition::beta,
      curl_replay_direct_transition::bfast};
  for (curl_replay_direct_transition transition : transitions) {
    const curl_replay_transition_result cpu =
        run_curl_replay_direct_transition_case(
            gpu::backend_mode::cpu, transition);
    const curl_replay_transition_result cuda =
        run_curl_replay_direct_transition_case(
            gpu::backend_mode::cuda, transition);
    compare_results(cpu.physics, cuda.physics, two_dimensional_case);
    require(cuda.physics.statistics.cuda_curl_calls > 0 &&
                cuda.physics.statistics.cpu_curl_calls == 0,
            "curl replay direct-transition fixture used CPU curl work");
    if (replay_disabled) {
      require(cuda.before_transition.checks == 0 &&
                  cuda.after_transition.checks == 0,
              "globally disabled curl replay emitted transition telemetry");
      continue;
    }
    const std::uint64_t transition_checks =
        cuda.after_transition.checks - cuda.before_transition.checks;
    const std::uint64_t transition_hits =
        cuda.after_transition.hits - cuda.before_transition.hits;
    const std::uint64_t transition_unready =
        cuda.after_transition.unready -
        cuda.before_transition.unready;
    require(
        cuda.before_transition.hits > 0 && transition_checks > 0 &&
            transition_hits == 0 && transition_unready == transition_checks &&
            cuda.after_transition.generation_misses ==
                cuda.before_transition.generation_misses &&
            cuda.after_transition.mirror_misses ==
                cuda.before_transition.mirror_misses,
        transition == curl_replay_direct_transition::beta
            ? "stable curl replay was not rejected by a beta transition"
            : "stable curl replay was not rejected by a BFAST transition");
  }
#endif
}

void require_curl_replay_invalidation_and_failure_contract() {
#if MEEP_SINGLE
  // The injected replay failure and exact host-array transaction are a
  // process-local contract. The authoritative qualification runs this full
  // binary separately with one MPI rank; distributed runs exercise the
  // ordinary replay path but must not duplicate this process-local fault
  // transaction on every rank.
  if (count_processors() != 1) {
    if (am_master())
      std::cout
          << "SKIP: curl replay invalidation/failure transaction requires "
             "one MPI rank\n";
    return;
  }
  scoped_environment_override curl_enable(
      "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL", "1");
  scoped_environment_override curl_disable(
      "MEEP_GPU_DISABLE_PHASE_BATCHED_CURL", nullptr);
  const bool replay_disabled =
      std::getenv("MEEP_GPU_DISABLE_CURL_PHASE_REPLAY") != nullptr;

  const curl_replay_host_write_result host_cpu =
      run_curl_replay_host_write_case(gpu::backend_mode::cpu);
  const curl_replay_host_write_result host_cuda =
      run_curl_replay_host_write_case(gpu::backend_mode::cuda);
  compare_results(
      host_cpu.physics, host_cuda.physics, two_dimensional_case);
  require(host_cuda.physics.statistics.cuda_curl_calls > 0 &&
              host_cuda.physics.statistics.cpu_curl_calls == 0,
          "curl replay host-write fixture used CPU curl work");
  if (replay_disabled) {
    require(host_cuda.after_replay.checks == 0,
            "globally disabled curl replay emitted invalidation telemetry");
    if (am_master())
      std::cout
          << "PASS: curl replay opt-out preserves host-write physics and "
             "emits zero replay telemetry\n";
    return;
  }

  require(
      host_cuda.after.checks - host_cuda.before.checks == 1 &&
          host_cuda.after.mirror_misses -
                  host_cuda.before.mirror_misses ==
              1 &&
          host_cuda.after.hits == host_cuda.before.hits &&
          host_cuda.after.unready == host_cuda.before.unready &&
          host_cuda.after.generation_misses ==
              host_cuda.before.generation_misses &&
          host_cuda.dispatch_after.host_to_device_bytes -
                  host_cuda.dispatch_before.host_to_device_bytes ==
              host_cuda.expected_upload_bytes &&
          host_cuda.resident_after.device_buffer_allocations ==
              host_cuda.resident_before.device_buffer_allocations,
      "curl replay did not reject and exactly re-upload a host-invalidated "
      "input mirror");
  require(
      host_cuda.after_replay.checks - host_cuda.after.checks == 1 &&
          host_cuda.after_replay.hits - host_cuda.after.hits == 1 &&
          host_cuda.after_replay.unready == host_cuda.after.unready &&
          host_cuda.after_replay.generation_misses ==
              host_cuda.after.generation_misses &&
          host_cuda.after_replay.mirror_misses ==
              host_cuda.after.mirror_misses &&
          host_cuda.dispatch_after_replay.host_to_device_bytes ==
              host_cuda.dispatch_after.host_to_device_bytes,
      "curl replay did not resume without a second upload after host-write "
      "publication");

  // A new unrelated mirror advances the cache allocation generation without
  // deleting the existing curl plan. This deterministically reaches the
  // generation guard and then proves the ordinary rebuild cadence.
  std::vector<float> generation_probe(29, 0.125f);
  gpu::set_backend(gpu::backend_mode::cuda);
  gpu::reset_dispatch_statistics();
  const grid_volume generation_gv = vol2d(1.8, 1.6, 10.0);
  structure generation_structure(
      generation_gv, vacuum, pml(0.3));
  fields generation_fields(
      &generation_structure, 0.0, 0.0, true, 64, 64);
  generation_fields.use_bloch(vec(0.0, 0.0));
  gaussian_src_time generation_source(0.29, 0.11);
  generation_fields.add_point_source(
      Ez, generation_source, generation_gv.center(),
      std::complex<double>(0.8, -0.23));
  for (int step = 0; step < 16; ++step) generation_fields.step();
  fields_chunk *generation_chunk = nullptr;
  for (int chunk_index = 0;
       chunk_index < generation_fields.num_chunks; ++chunk_index)
    if (generation_fields.chunks[chunk_index]->is_mine()) {
      generation_chunk = generation_fields.chunks[chunk_index];
      break;
    }
  require(generation_chunk != nullptr,
          "curl replay generation fixture found no local chunk");
  const int b_phase_key = static_cast<int>(B_stuff);
  const gpu::detail::curl_phase_replay_plan_snapshot generation_warm =
      gpu::detail::get_curl_phase_replay_plan_snapshot_for_testing(
          &generation_fields, b_phase_key);
  require(generation_warm.exists && generation_warm.complete_phase &&
              generation_warm.stable_reuses >= 2,
          "curl replay generation fixture did not warm a complete plan");
  {
    gpu::detail::resident_curl_session resident(
        generation_chunk, true);
    gpu::detail::resident_ensure_mirror_fp32(
        resident.cache(), generation_probe.data(),
        generation_probe.size());
    resident.finish(false);
  }
  gpu::reset_dispatch_statistics();
  run_single_curl_phase(generation_fields, B_stuff, true);
  const gpu::curl_phase_replay_statistics generation_first =
      gpu::get_curl_phase_replay_statistics();
  const gpu::detail::curl_phase_replay_plan_snapshot
      generation_rebuilt =
          gpu::detail::get_curl_phase_replay_plan_snapshot_for_testing(
              &generation_fields, b_phase_key);
  const gpu::dispatch_statistics generation_first_dispatch =
      gpu::get_dispatch_statistics();
  require(
      generation_first.checks == 1 &&
          generation_first.generation_misses == 1 &&
          generation_first.hits == 0 && generation_first.unready == 0 &&
          generation_first.mirror_misses == 0 &&
          generation_first_dispatch.cuda_curl_calls == 1 &&
          generation_first_dispatch.cpu_curl_calls == 0 &&
          generation_rebuilt.exists && generation_rebuilt.complete_phase &&
          generation_rebuilt.stable_reuses == 0,
      "curl replay generation miss did not fall back to one ordinary "
      "complete-plan rebuild");
  run_single_curl_phase(generation_fields, B_stuff, true);
  const gpu::detail::curl_phase_replay_plan_snapshot generation_reuse_one =
      gpu::detail::get_curl_phase_replay_plan_snapshot_for_testing(
          &generation_fields, b_phase_key);
  run_single_curl_phase(generation_fields, B_stuff, true);
  const gpu::detail::curl_phase_replay_plan_snapshot generation_reuse_two =
      gpu::detail::get_curl_phase_replay_plan_snapshot_for_testing(
          &generation_fields, b_phase_key);
  run_single_curl_phase(generation_fields, B_stuff, true);
  const gpu::curl_phase_replay_statistics generation_recovered =
      gpu::get_curl_phase_replay_statistics();
  require(
      generation_reuse_one.complete_phase &&
          generation_reuse_one.stable_reuses == 1 &&
          generation_reuse_two.complete_phase &&
          generation_reuse_two.stable_reuses == 2 &&
          generation_recovered.checks == 4 &&
          generation_recovered.hits == 1 &&
          generation_recovered.unready == 2 &&
          generation_recovered.generation_misses == 1 &&
          generation_recovered.mirror_misses == 0 &&
          generation_recovered.checks ==
              generation_recovered.hits +
                  generation_recovered.unready +
                  generation_recovered.generation_misses +
                  generation_recovered.mirror_misses,
      "curl replay generation recovery did not require two stable ordinary "
      "reuses before replay");

  gpu::detail::reset_resident_cache_for_owner_reusing_allocations(
      generation_chunk);
  const gpu::detail::curl_phase_replay_plan_snapshot reset_snapshot =
      gpu::detail::get_curl_phase_replay_plan_snapshot_for_testing(
          &generation_fields, b_phase_key);
  gpu::reset_dispatch_statistics();
  run_single_curl_phase(generation_fields, B_stuff, true);
  const gpu::curl_phase_replay_statistics reset_rebuild =
      gpu::get_curl_phase_replay_statistics();
  const gpu::detail::curl_phase_replay_plan_snapshot reset_rebuilt_plan =
      gpu::detail::get_curl_phase_replay_plan_snapshot_for_testing(
          &generation_fields, b_phase_key);
  require(!reset_snapshot.exists && reset_rebuild.checks == 1 &&
              reset_rebuild.unready == 1 && reset_rebuild.hits == 0 &&
              reset_rebuild.generation_misses == 0 &&
              reset_rebuild.mirror_misses == 0 &&
              reset_rebuilt_plan.exists &&
              reset_rebuilt_plan.complete_phase &&
              reset_rebuilt_plan.stable_reuses == 0,
          "curl replay cache reset did not discard and ordinarily rebuild "
          "its address-bearing plan");

  // Inject immediately before a warmed replay kernel submission. The failed
  // transaction must make the plan incomplete, reset its stability proof,
  // conserve outcome counters, and recover only after two ordinary reuses.
  gpu::reset_dispatch_statistics();
  const grid_volume failure_gv = vol2d(1.8, 1.6, 10.0);
  structure failure_structure(failure_gv, vacuum, pml(0.3));
  fields failure_fields(
      &failure_structure, 0.0, 0.0, true, 64, 64);
  failure_fields.use_bloch(vec(0.0, 0.0));
  gaussian_src_time failure_source(0.29, 0.11);
  failure_fields.add_point_source(
      Ez, failure_source, failure_gv.center(),
      std::complex<double>(0.8, -0.23));
  for (int step = 0; step < 16; ++step) failure_fields.step();
  const gpu::detail::curl_phase_replay_plan_snapshot failure_warm =
      gpu::detail::get_curl_phase_replay_plan_snapshot_for_testing(
          &failure_fields, b_phase_key);
  require(failure_warm.exists && failure_warm.complete_phase &&
              failure_warm.stable_reuses >= 2,
          "curl replay failure fixture did not warm a complete plan");
  for (int chunk_index = 0; chunk_index < failure_fields.num_chunks;
       ++chunk_index)
    if (failure_fields.chunks[chunk_index]->is_mine())
      gpu::detail::sync_resident_cache_for_owner(
          failure_fields.chunks[chunk_index]);

  struct field_bytes_snapshot {
    const realnum *pointer;
    std::vector<realnum> values;
  };
  std::vector<field_bytes_snapshot> magnetic_before;
  for (int chunk_index = 0; chunk_index < failure_fields.num_chunks;
       ++chunk_index) {
    fields_chunk *chunk = failure_fields.chunks[chunk_index];
    if (!chunk->is_mine()) continue;
    const int complex_parts = 2 - chunk->is_real;
    for (component c = Bx; c <= Bz; c = component(c + 1))
      for (int complex_part = 0; complex_part < complex_parts;
           ++complex_part)
        if (chunk->f[c][complex_part])
          magnetic_before.push_back(
              {chunk->f[c][complex_part],
               std::vector<realnum>(
                   chunk->f[c][complex_part],
                   chunk->f[c][complex_part] + chunk->gv.ntot())});
  }
  require(!magnetic_before.empty(),
          "curl replay failure fixture found no magnetic field arrays");
  gpu::reset_dispatch_statistics();
  gpu::detail::set_curl_phase_replay_launch_failures_for_testing(1);
  bool injected = false;
  try {
    run_single_curl_phase(failure_fields, B_stuff, true);
  }
  catch (const std::runtime_error &error) {
    injected = std::string(error.what()) ==
               "injected CUDA curl phase replay launch failure";
  }
  gpu::detail::set_curl_phase_replay_launch_failures_for_testing(0);
  for (int chunk_index = 0; chunk_index < failure_fields.num_chunks;
       ++chunk_index)
    if (failure_fields.chunks[chunk_index]->is_mine())
      gpu::detail::sync_resident_cache_for_owner(
          failure_fields.chunks[chunk_index]);
  bool magnetic_unchanged = true;
  for (const field_bytes_snapshot &snapshot : magnetic_before)
    magnetic_unchanged =
        magnetic_unchanged &&
        std::memcmp(
            snapshot.pointer, snapshot.values.data(),
            snapshot.values.size() * sizeof(realnum)) == 0;
  const gpu::curl_phase_replay_statistics failure_statistics =
      gpu::get_curl_phase_replay_statistics();
  const gpu::dispatch_statistics failure_dispatch =
      gpu::get_dispatch_statistics();
  const gpu::phase_batch_policy_statistics failure_policy =
      gpu::get_phase_batch_policy_statistics();
  const gpu::detail::curl_phase_replay_plan_snapshot failure_snapshot =
      gpu::detail::get_curl_phase_replay_plan_snapshot_for_testing(
          &failure_fields, b_phase_key);
  require(
      injected && magnetic_unchanged && failure_statistics.checks == 1 &&
          failure_statistics.unready == 1 &&
          failure_statistics.hits == 0 &&
          failure_statistics.generation_misses == 0 &&
          failure_statistics.mirror_misses == 0 &&
          failure_statistics.checks ==
              failure_statistics.hits + failure_statistics.unready +
                  failure_statistics.generation_misses +
                  failure_statistics.mirror_misses &&
          failure_dispatch.cuda_curl_calls == 0 &&
          failure_dispatch.cuda_curl_points == 0 &&
          failure_policy.curl_forced_batches == 0 &&
          !failure_snapshot.complete_phase &&
          failure_snapshot.stable_reuses == 0,
      "curl replay launch failure did not preserve fields and invalidate "
      "the replay transaction exactly once");
  run_single_curl_phase(failure_fields, B_stuff, true);
  const gpu::detail::curl_phase_replay_plan_snapshot failure_reuse_one =
      gpu::detail::get_curl_phase_replay_plan_snapshot_for_testing(
          &failure_fields, b_phase_key);
  run_single_curl_phase(failure_fields, B_stuff, true);
  const gpu::detail::curl_phase_replay_plan_snapshot failure_reuse_two =
      gpu::detail::get_curl_phase_replay_plan_snapshot_for_testing(
          &failure_fields, b_phase_key);
  run_single_curl_phase(failure_fields, B_stuff, true);
  const gpu::curl_phase_replay_statistics failure_recovered =
      gpu::get_curl_phase_replay_statistics();
  require(
      failure_reuse_one.exists && failure_reuse_one.complete_phase &&
          failure_reuse_one.stable_reuses == 1 &&
          failure_reuse_two.exists && failure_reuse_two.complete_phase &&
          failure_reuse_two.stable_reuses == 2 &&
          failure_recovered.checks == 4 &&
          failure_recovered.hits == 1 &&
          failure_recovered.unready == 3 &&
          failure_recovered.generation_misses == 0 &&
          failure_recovered.mirror_misses == 0 &&
          failure_recovered.checks ==
              failure_recovered.hits + failure_recovered.unready +
                  failure_recovered.generation_misses +
                  failure_recovered.mirror_misses,
      "curl replay launch failure did not recover through two ordinary "
      "stable reuses before replay");
  if (am_master())
    std::cout
        << "PASS: curl replay host-write, generation, cache-reset, and "
           "launch-failure transactions recover deterministically\n";
#endif
}

void require_phase_batched_source_split_and_replay_contract() {
#if MEEP_SINGLE
  gpu::set_backend(gpu::backend_mode::cuda);
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  int cache_owner = 0;
  int phase_owner = 0;
  constexpr std::size_t profile_points = 3;
  constexpr std::size_t destination_count =
      meep_cuda::indexed_source_phase_max_operations + 3;
  constexpr std::size_t split_operation_count = destination_count + 1;
  const std::ptrdiff_t indices[profile_points] = {1, 4, 6};
  const float amplitudes[2 * profile_points] = {
      0.25f, -0.10f, -0.35f, 0.20f, 0.15f, 0.05f};
  std::vector<std::vector<float>> destinations(
      destination_count, std::vector<float>(8, 0.75f));
  std::vector<std::vector<float>> expected = destinations;

  const auto scale_for = [](std::size_t operation,
                            std::size_t pass) {
    return gpu::detail::complex_value_fp32{
        0.01f * static_cast<float>(operation + pass + 1),
        -0.0075f * static_cast<float>((operation + 2 * pass) % 7)};
  };
  const auto apply_reference = [&](std::size_t operation,
                                   std::size_t pass) {
    const std::size_t destination =
        operation == destination_count - 1
            ? meep_cuda::indexed_source_phase_max_operations
            : operation == destination_count
                  ? destination_count - 1
                  : operation;
    const gpu::detail::complex_value_fp32 scale =
        scale_for(operation, pass);
    for (std::size_t point = 0; point < profile_points; ++point) {
      const float amplitude_real = amplitudes[2 * point];
      const float amplitude_imaginary = amplitudes[2 * point + 1];
      const float value =
          amplitude_real * scale.real -
          amplitude_imaginary * scale.imag;
      expected[destination][indices[point]] -= value;
    }
  };
  gpu::detail::resident_curl_session resident(&cache_owner, true);
  const auto run_pass = [&](std::size_t operation_count,
                            std::size_t pass) {
    gpu::detail::resident_source_phase_batch batch(
        &phase_owner, 911, true);
    for (std::size_t operation = 0; operation < operation_count;
         ++operation) {
      const std::size_t destination =
          operation == destination_count - 1
              ? meep_cuda::indexed_source_phase_max_operations
              : operation == destination_count
                    ? destination_count - 1
                    : operation;
      require(
          gpu::detail::resident_indexed_source_subtract_fp32(
              resident.cache(), destinations[destination].data(),
              destinations[destination].size(), indices, amplitudes,
              profile_points, nullptr, scale_for(operation, pass), false),
          "active source phase did not collect an indexed update");
      apply_reference(operation, pass);
    }
    batch.finish();
  };
  const auto require_values_match = [&] {
    for (std::size_t destination = 0;
         destination < destination_count; ++destination)
      for (std::size_t index = 0;
           index < destinations[destination].size(); ++index)
        require(
            std::abs(destinations[destination][index] -
                     expected[destination][index]) <= 2.0e-6f,
            "phase-batched source split changed update order or values");
  };

  gpu::reset_dispatch_statistics();
  run_pass(split_operation_count, 0);
  const gpu::source_statistics first =
      gpu::get_source_statistics();
  require(
      first.cuda_update_calls == 3 &&
          first.cuda_update_points ==
              split_operation_count * profile_points,
      "source phase did not split at the 32-operation limit and repeated "
      "destination boundary");

  gpu::reset_dispatch_statistics();
  run_pass(split_operation_count, 1);
  const gpu::source_statistics replay_sources =
      gpu::get_source_statistics();
  const gpu::dispatch_statistics replay_dispatch =
      gpu::get_dispatch_statistics();
  require(
      replay_sources.cuda_update_calls == 3 &&
          replay_sources.cuda_update_points ==
              split_operation_count * profile_points &&
          replay_dispatch.host_to_device_bytes == 0,
      "stable source topology replay uploaded descriptors or field data");

  const std::uint64_t live_before_prune =
      gpu::get_live_resident_device_buffers();
  gpu::reset_dispatch_statistics();
  run_pass(1, 2);
  require(
      gpu::get_live_resident_device_buffers() + 3 == live_before_prune,
      "shrinking source topology retained obsolete descriptor groups");

  gpu::reset_dispatch_statistics();
  run_pass(1, 3);
  const gpu::source_statistics compact_replay_sources =
      gpu::get_source_statistics();
  const gpu::dispatch_statistics compact_replay_dispatch =
      gpu::get_dispatch_statistics();
  require(
      compact_replay_sources.cuda_update_calls == 1 &&
          compact_replay_sources.cuda_update_points == profile_points &&
          compact_replay_dispatch.host_to_device_bytes == 0,
      "compacted source topology did not become a 0-byte-H2D replay");

  // Restore a real persistent two-operation plan and prove its stable replay
  // before replacing a captured source profile. A singleton deliberately
  // clears its plan, so it cannot exercise descriptor invalidation.
  gpu::reset_dispatch_statistics();
  run_pass(2, 4);
  gpu::reset_dispatch_statistics();
  run_pass(2, 5);
  require(
      gpu::get_source_statistics().cuda_update_calls == 1 &&
          gpu::get_source_statistics().cuda_update_points ==
              2 * profile_points &&
          gpu::get_dispatch_statistics().host_to_device_bytes == 0,
      "restored source descriptor plan did not become a 0-byte-H2D replay");

  resident.finish(false);
  const std::uint64_t live_before_profile_replacement =
      gpu::get_live_resident_device_buffers();
  gpu::detail::discard_resident_source_profile_for_owner(
      &cache_owner, indices, amplitudes);
  require(
      gpu::get_live_resident_device_buffers() ==
          live_before_profile_replacement,
      "source-profile recycling changed the live allocation count");
  {
    gpu::detail::resident_curl_session replacement_resident(
        &cache_owner, true);
    gpu::reset_dispatch_statistics();
    run_pass(2, 6);
    const gpu::source_statistics replacement_sources =
        gpu::get_source_statistics();
    const gpu::dispatch_statistics replacement_dispatch =
        gpu::get_dispatch_statistics();
    const std::size_t profile_bytes =
        profile_points *
        (sizeof(std::ptrdiff_t) + 2 * sizeof(float));
    require(
        replacement_sources.cuda_update_calls == 1 &&
            replacement_sources.cuda_update_points ==
                2 * profile_points &&
            replacement_dispatch.host_to_device_bytes > profile_bytes &&
            gpu::get_live_resident_device_buffers() ==
                live_before_profile_replacement,
        "source replacement reused a stale descriptor or discarded "
        "reusable allocations");
    replacement_resident.finish(true);
  }
  require_values_match();
  gpu::detail::destroy_resident_source_phase_batches_for_owner(
      &phase_owner);
  gpu::detail::destroy_resident_cache_for_owner(&cache_owner);

  // Public source profiles may repeat a destination index. Use more than one
  // CUDA block and an exactly representable contribution so this catches a
  // lost non-atomic read/modify/write without adding a tolerance ambiguity.
  int duplicate_cache_owner = 0;
  int duplicate_phase_owner = 0;
  constexpr std::size_t duplicate_points = 513;
  std::vector<std::ptrdiff_t> duplicate_indices(duplicate_points, 2);
  std::vector<float> duplicate_amplitudes(2 * duplicate_points, 0.0f);
  for (std::size_t point = 0; point < duplicate_points; ++point)
    duplicate_amplitudes[2 * point] = 1.0f / 1024.0f;
  std::vector<float> duplicate_destination(4, 1.0f);
  const std::ptrdiff_t auxiliary_indices[1] = {1};
  const float auxiliary_amplitudes[2] = {0.25f, 0.0f};
  std::vector<float> auxiliary_destination(4, 0.5f);
  gpu::reset_dispatch_statistics();
  {
    gpu::detail::resident_curl_session duplicate_resident(
        &duplicate_cache_owner, true);
    gpu::detail::resident_source_phase_batch duplicate_batch(
        &duplicate_phase_owner, 912, true);
    require(
        gpu::detail::resident_indexed_source_subtract_fp32(
            duplicate_resident.cache(), duplicate_destination.data(),
            duplicate_destination.size(), duplicate_indices.data(),
            duplicate_amplitudes.data(), duplicate_points, nullptr,
            {1.0f, 0.0f}, false),
        "duplicate-index source phase did not collect an indexed update");
    require(
        gpu::detail::resident_indexed_source_subtract_fp32(
            duplicate_resident.cache(), auxiliary_destination.data(),
            auxiliary_destination.size(), auxiliary_indices,
            auxiliary_amplitudes, 1, nullptr, {1.0f, 0.0f}, false),
        "duplicate-index source phase did not collect its independent "
        "second operation");
    duplicate_batch.finish();
    duplicate_resident.finish(true);
  }
  const gpu::source_statistics duplicate_sources =
      gpu::get_source_statistics();
  require(
      duplicate_destination[2] ==
              1.0f - static_cast<float>(duplicate_points) / 1024.0f &&
          auxiliary_destination[1] == 0.25f &&
          duplicate_sources.cpu_update_calls == 0 &&
          duplicate_sources.cuda_update_calls == 1 &&
          duplicate_sources.cuda_update_points == duplicate_points + 1,
      "multi-operation phase-batched source did not preserve duplicate-index "
      "atomics or exact call accounting");
  gpu::detail::destroy_resident_source_phase_batches_for_owner(
      &duplicate_phase_owner);
  gpu::detail::destroy_resident_cache_for_owner(&duplicate_cache_owner);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "source split/replay lifecycle leaked a resident allocation");
  if (am_master())
    std::cout
        << "PASS: phase-batched source 32-operation split, repeated-"
           "destination ordering, duplicate-index atomics, descriptor "
           "pruning, replacement refresh, and 0-byte replay\n";
#endif
}

void require_resident_ldos_reduction_contract() {
#if MEEP_SINGLE && MEEP_HAVE_CUDA
  gpu::set_backend(gpu::backend_mode::cuda);
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  int electric_owner = 0;
  int magnetic_owner = 0;
  constexpr std::size_t field_count = 911;
  std::vector<float> electric_real(field_count);
  std::vector<float> electric_imaginary(field_count);
  std::vector<float> real_only(field_count);
  std::vector<float> magnetic_real(field_count);
  std::vector<float> magnetic_imaginary(field_count);
  for (std::size_t index = 0; index < field_count; ++index) {
    const float coordinate = static_cast<float>(index);
    electric_real[index] = 0.25f * std::sin(0.017f * coordinate);
    electric_imaginary[index] =
        -0.20f * std::cos(0.013f * coordinate);
    real_only[index] = 0.15f * std::sin(0.031f * coordinate);
    magnetic_real[index] =
        -0.30f * std::cos(0.019f * coordinate);
    magnetic_imaginary[index] =
        0.22f * std::sin(0.023f * coordinate);
  }
  std::vector<float> resident_electric_real(field_count);
  std::vector<float> resident_electric_imaginary(field_count);
  std::vector<float> resident_real_only(field_count);
  std::vector<float> resident_magnetic_real(field_count);
  std::vector<float> resident_magnetic_imaginary(field_count);
  for (std::size_t index = 0; index < field_count; ++index) {
    resident_electric_real[index] =
        -0.73f * electric_real[index] + 0.011f;
    resident_electric_imaginary[index] =
        1.17f * electric_imaginary[index] - 0.007f;
    resident_real_only[index] = 0.61f * real_only[index] + 0.019f;
    resident_magnetic_real[index] =
        -0.89f * magnetic_real[index] - 0.013f;
    resident_magnetic_imaginary[index] =
        1.09f * magnetic_imaginary[index] + 0.005f;
  }

  const std::vector<std::size_t> profile_counts = {521, 37, 403};
  std::vector<std::vector<std::ptrdiff_t> > indices(3);
  std::vector<std::vector<float> > amplitudes(3);
  for (std::size_t profile = 0; profile < profile_counts.size(); ++profile) {
    indices[profile].resize(profile_counts[profile]);
    amplitudes[profile].resize(2 * profile_counts[profile]);
    for (std::size_t point = 0; point < profile_counts[profile]; ++point) {
      indices[profile][point] = static_cast<std::ptrdiff_t>(
          (17 * point + 29 * profile + 3) % field_count);
      amplitudes[profile][2 * point] =
          0.08f * std::sin(0.07f * static_cast<float>(point + 1));
      amplitudes[profile][2 * point + 1] =
          -0.06f * std::cos(
              0.05f * static_cast<float>(point + profile + 1));
    }
  }

  double expected[4] = {0.0, 0.0, 0.0, 0.0};
  const auto accumulate_reference = [&expected](
      const std::vector<float> &field_real,
      const std::vector<float> *field_imaginary,
      const std::vector<std::ptrdiff_t> &profile_indices,
      const std::vector<float> &profile_amplitudes, bool magnetic) {
    for (std::size_t point = 0; point < profile_indices.size(); ++point) {
      const std::size_t index =
          static_cast<std::size_t>(profile_indices[point]);
      const float fr = field_real[index];
      const float fi = field_imaginary
                           ? (*field_imaginary)[index]
                           : 0.0f;
      const float ar = profile_amplitudes[2 * point];
      const float ai = profile_amplitudes[2 * point + 1];
      const float product_real = fr * ar + fi * ai;
      const float product_imaginary = fi * ar - fr * ai;
      const std::size_t offset = magnetic ? 2 : 0;
      expected[offset] += static_cast<double>(product_real);
      expected[offset + 1] += static_cast<double>(product_imaginary);
    }
  };
  accumulate_reference(
      resident_electric_real, &resident_electric_imaginary, indices[0],
      amplitudes[0], false);
  accumulate_reference(
      resident_real_only, nullptr, indices[1], amplitudes[1], false);
  accumulate_reference(
      resident_magnetic_real, &resident_magnetic_imaginary, indices[2],
      amplitudes[2], true);

  gpu::detail::resident_curl_session electric_session(
      &electric_owner, true);
  gpu::detail::resident_curl_session magnetic_session(
      &magnetic_owner, true);
  gpu::detail::resident_copy_fp32(
      electric_session.cache(), electric_real.data(),
      resident_electric_real.data(), field_count);
  gpu::detail::resident_copy_fp32(
      electric_session.cache(), electric_imaginary.data(),
      resident_electric_imaginary.data(), field_count);
  gpu::detail::resident_copy_fp32(
      electric_session.cache(), real_only.data(),
      resident_real_only.data(), field_count);
  gpu::detail::resident_copy_fp32(
      magnetic_session.cache(), magnetic_real.data(),
      resident_magnetic_real.data(), field_count);
  gpu::detail::resident_copy_fp32(
      magnetic_session.cache(), magnetic_imaginary.data(),
      resident_magnetic_imaginary.data(), field_count);
  gpu::detail::ldos_reduction_request_fp32 requests[3] = {
      {electric_session.cache(), electric_real.data(),
       electric_imaginary.data(), field_count, indices[0].data(),
       amplitudes[0].data(), profile_counts[0], false},
      {electric_session.cache(), real_only.data(), nullptr, field_count,
       indices[1].data(), amplitudes[1].data(), profile_counts[1], false},
      {magnetic_session.cache(), magnetic_real.data(),
       magnetic_imaginary.data(), field_count, indices[2].data(),
       amplitudes[2].data(), profile_counts[2], true}};

  const auto require_result = [&expected](const double actual[4]) {
    for (std::size_t value = 0; value < 4; ++value)
      require(
          std::abs(actual[value] - expected[value]) <=
              2.0e-12 * std::max(1.0, std::abs(expected[value])),
          "resident CUDA LDOS reduction disagrees with FP32-product/FP64-"
          "sum reference");
  };

  gpu::reset_dispatch_statistics();
  double first[4] = {};
  gpu::detail::resident_reduce_ldos_fp32(requests, 3, first);
  require_result(first);
  const gpu::detail::ldos_reduction_statistics first_statistics =
      gpu::detail::get_ldos_reduction_statistics();
  const gpu::dispatch_statistics first_dispatch =
      gpu::get_dispatch_statistics();
  const std::uint64_t expected_avoided_bytes =
      5 * field_count * sizeof(float);
  require(
      first_statistics.batch_calls == 1 &&
          first_statistics.submitted_profiles == 3 &&
          first_statistics.source_points ==
              profile_counts[0] + profile_counts[1] + profile_counts[2] &&
          first_statistics.descriptor_uploads == 1 &&
          first_statistics.kernel_launches == 2 &&
          first_statistics.result_device_to_host_bytes == 4 * sizeof(double) &&
          first_statistics.full_field_device_to_host_bytes_avoided ==
              expected_avoided_bytes &&
          first_dispatch.device_to_host_bytes == 4 * sizeof(double),
      "resident CUDA LDOS first-use statistics are inconsistent");

  gpu::reset_dispatch_statistics();
  double replay[4] = {};
  gpu::detail::resident_reduce_ldos_fp32(requests, 3, replay);
  require_result(replay);
  const gpu::detail::ldos_reduction_statistics replay_statistics =
      gpu::detail::get_ldos_reduction_statistics();
  const gpu::dispatch_statistics replay_dispatch =
      gpu::get_dispatch_statistics();
  require(
      replay_statistics.batch_calls == 1 &&
          replay_statistics.descriptor_uploads == 0 &&
          replay_statistics.kernel_launches == 2 &&
          replay_statistics.full_field_device_to_host_bytes_avoided ==
              expected_avoided_bytes &&
          replay_dispatch.host_to_device_bytes == 0 &&
          replay_dispatch.device_to_host_bytes == 4 * sizeof(double),
      "resident CUDA LDOS stable topology did not become a 32-byte-D2H, "
      "zero-H2D replay");
  for (std::size_t repetition = 0; repetition < 32; ++repetition) {
    double repeated[4] = {};
    gpu::detail::resident_reduce_ldos_fp32(requests, 3, repeated);
    require(
        std::memcmp(repeated, replay, sizeof(replay)) == 0,
        "resident CUDA LDOS reduction changed bits across repetitions");
  }

  const std::uint64_t live_before_reentry =
      gpu::get_live_resident_device_buffers();
  gpu::reset_dispatch_statistics();
  double reentry_result[4] = {7.0, -8.0, 9.0, -10.0};
  const double reentry_result_before[4] = {7.0, -8.0, 9.0, -10.0};
  bool rejected_reentry = false;
  try {
    gpu::detail::resident_reduce_ldos_reentrant_for_testing(
        requests, 3, reentry_result);
  }
  catch (const std::logic_error &) { rejected_reentry = true; }
  const gpu::detail::ldos_reduction_statistics reentry_statistics =
      gpu::detail::get_ldos_reduction_statistics();
  const gpu::dispatch_statistics reentry_dispatch =
      gpu::get_dispatch_statistics();
  const gpu::resident_statistics reentry_resident =
      gpu::get_resident_statistics();
  require(
      rejected_reentry &&
          std::memcmp(reentry_result, reentry_result_before,
                      sizeof(reentry_result)) == 0 &&
          reentry_statistics.batch_calls == 0 &&
          reentry_statistics.descriptor_uploads == 0 &&
          reentry_statistics.kernel_launches == 0 &&
          reentry_dispatch.host_to_device_bytes == 0 &&
          reentry_dispatch.device_to_host_bytes == 0 &&
          reentry_resident.device_buffer_allocations == 0 &&
          gpu::get_live_resident_device_buffers() == live_before_reentry,
      "deterministic LDOS reentry rejection changed result, counters, or "
      "resident allocations");

  const gpu::detail::ldos_reduction_request_fp32 reordered_requests[3] = {
      requests[1], requests[0], requests[2]};
  const std::uint64_t live_before_commit_failure =
      gpu::get_live_resident_device_buffers();
  gpu::reset_dispatch_statistics();
  gpu::detail::fail_next_ldos_plan_commit_for_testing();
  double failed_commit_result[4] = {11.0, -12.0, 13.0, -14.0};
  const double failed_commit_result_before[4] = {
      11.0, -12.0, 13.0, -14.0};
  bool rejected_commit = false;
  try {
    gpu::detail::resident_reduce_ldos_fp32(
        reordered_requests, 3, failed_commit_result);
  }
  catch (const std::bad_alloc &) { rejected_commit = true; }
  const gpu::detail::ldos_reduction_statistics failed_commit_statistics =
      gpu::detail::get_ldos_reduction_statistics();
  const gpu::dispatch_statistics failed_commit_dispatch =
      gpu::get_dispatch_statistics();
  const gpu::resident_statistics failed_commit_resident =
      gpu::get_resident_statistics();
  require(
      rejected_commit &&
          std::memcmp(failed_commit_result, failed_commit_result_before,
                      sizeof(failed_commit_result)) == 0 &&
          failed_commit_statistics.batch_calls == 0 &&
          failed_commit_statistics.descriptor_uploads == 0 &&
          failed_commit_statistics.kernel_launches == 0 &&
          failed_commit_dispatch.host_to_device_bytes > 0 &&
          failed_commit_dispatch.device_to_host_bytes == 0 &&
          failed_commit_resident.device_buffer_allocations == 1 &&
          gpu::get_live_resident_device_buffers() ==
              live_before_commit_failure,
      "failed inactive LDOS plan commit changed caller/active/live state or "
      "hid its physical upload");
  gpu::reset_dispatch_statistics();
  double after_commit_failure[4] = {};
  gpu::detail::resident_reduce_ldos_fp32(
      requests, 3, after_commit_failure);
  require(
      std::memcmp(after_commit_failure, replay, sizeof(replay)) == 0 &&
          gpu::detail::get_ldos_reduction_statistics()
                  .descriptor_uploads == 0 &&
          gpu::get_dispatch_statistics().host_to_device_bytes == 0,
      "LDOS active plan was not preserved across failed inactive commit");
  require(
      electric_real != resident_electric_real &&
          electric_imaginary != resident_electric_imaginary &&
          real_only != resident_real_only &&
          magnetic_real != resident_magnetic_real &&
          magnetic_imaginary != resident_magnetic_imaginary,
      "resident CUDA LDOS unexpectedly published full fields to the host");

  std::ptrdiff_t bad_index =
      static_cast<std::ptrdiff_t>(field_count);
  const float bad_amplitude[2] = {1.0f, 0.0f};
  const gpu::detail::ldos_reduction_request_fp32 bad_request = {
      electric_session.cache(), electric_real.data(), nullptr, field_count,
      &bad_index, bad_amplitude, 1, false};
  const gpu::detail::ldos_reduction_request_fp32 malformed_batch[2] = {
      requests[0], bad_request};
  const std::uint64_t live_before_rejection =
      gpu::get_live_resident_device_buffers();
  gpu::reset_dispatch_statistics();
  bool rejected_bad_index = false;
  double rejected_result[4] = {1.0, -2.0, 3.0, -4.0};
  const double rejected_result_before[4] = {1.0, -2.0, 3.0, -4.0};
  try {
    gpu::detail::resident_reduce_ldos_fp32(
        malformed_batch, 2, rejected_result);
  }
  catch (const std::out_of_range &) { rejected_bad_index = true; }
  const gpu::detail::ldos_reduction_statistics rejected_statistics =
      gpu::detail::get_ldos_reduction_statistics();
  const gpu::dispatch_statistics rejected_dispatch =
      gpu::get_dispatch_statistics();
  require(
      rejected_bad_index &&
          std::memcmp(rejected_result, rejected_result_before,
                      sizeof(rejected_result)) == 0 &&
          rejected_statistics.batch_calls == 0 &&
          rejected_statistics.descriptor_uploads == 0 &&
          rejected_statistics.kernel_launches == 0 &&
          rejected_dispatch.host_to_device_bytes == 0 &&
          rejected_dispatch.device_to_host_bytes == 0 &&
          gpu::get_live_resident_device_buffers() == live_before_rejection,
      "malformed later LDOS request changed result, counters, or resident "
      "allocations before rejection");

  double recovered[4] = {};
  gpu::detail::resident_reduce_ldos_fp32(requests, 3, recovered);
  require(
      std::memcmp(recovered, replay, sizeof(replay)) == 0,
      "resident CUDA LDOS did not recover after atomic batch rejection");

  electric_session.finish(false);
  magnetic_session.finish(false);
  const std::uint64_t live_before_profile_replacement =
      gpu::get_live_resident_device_buffers();
  gpu::detail::discard_resident_source_profile_for_owner(
      &magnetic_owner, indices[2].data(), amplitudes[2].data());
  require(
      gpu::get_live_resident_device_buffers() ==
          live_before_profile_replacement,
      "LDOS source-profile recycling changed the live allocation count");
  {
    gpu::detail::resident_curl_session replacement_electric_session(
        &electric_owner, true);
    gpu::detail::resident_curl_session replacement_magnetic_session(
        &magnetic_owner, true);
    gpu::reset_dispatch_statistics();
    double after_replacement[4] = {};
    gpu::detail::resident_reduce_ldos_fp32(
        requests, 3, after_replacement);
    const gpu::detail::ldos_reduction_statistics replacement_statistics =
        gpu::detail::get_ldos_reduction_statistics();
    const gpu::dispatch_statistics replacement_dispatch =
        gpu::get_dispatch_statistics();
    require(
        std::memcmp(after_replacement, replay, sizeof(replay)) == 0 &&
            replacement_statistics.descriptor_uploads == 1 &&
            replacement_statistics.kernel_launches == 2 &&
            replacement_dispatch.host_to_device_bytes >
                2 * profile_counts[2] * sizeof(float) &&
            replacement_dispatch.device_to_host_bytes ==
                4 * sizeof(double) &&
            gpu::get_live_resident_device_buffers() ==
                live_before_profile_replacement + 1 &&
            gpu::get_resident_statistics().device_buffer_allocations == 1,
        "cross-cache LDOS source replacement reused a stale descriptor or "
        "failed to warm exactly one bounded inactive plan allocation");
    replacement_electric_session.finish(false);
    replacement_magnetic_session.finish(false);
  }
  const std::uint64_t warm_plan_live =
      gpu::get_live_resident_device_buffers();
  require(
      warm_plan_live == live_before_profile_replacement + 1,
      "LDOS two-slot warm-up did not retain exactly one additional plan");

  gpu::reset_dispatch_statistics();
  gpu::detail::fail_next_ldos_plan_commit_for_testing();
  double reused_slot_failure_result[4] = {
      15.0, -16.0, 17.0, -18.0};
  const double reused_slot_failure_result_before[4] = {
      15.0, -16.0, 17.0, -18.0};
  bool rejected_reused_slot_commit = false;
  {
    gpu::detail::resident_curl_session failure_electric_session(
        &electric_owner, true);
    gpu::detail::resident_curl_session failure_magnetic_session(
        &magnetic_owner, true);
    try {
      gpu::detail::resident_reduce_ldos_fp32(
          reordered_requests, 3, reused_slot_failure_result);
    }
    catch (const std::bad_alloc &) {
      rejected_reused_slot_commit = true;
    }
    failure_electric_session.finish(false);
    failure_magnetic_session.finish(false);
  }
  require(
      rejected_reused_slot_commit &&
          std::memcmp(reused_slot_failure_result,
                      reused_slot_failure_result_before,
                      sizeof(reused_slot_failure_result)) == 0 &&
          gpu::detail::get_ldos_reduction_statistics().batch_calls == 0 &&
          gpu::detail::get_ldos_reduction_statistics().kernel_launches == 0 &&
          gpu::get_dispatch_statistics().host_to_device_bytes > 0 &&
          gpu::get_dispatch_statistics().device_to_host_bytes == 0 &&
          gpu::get_resident_statistics().device_buffer_allocations == 0 &&
          gpu::detail::resident_ldos_plan_buffer_count_for_testing(
              requests[0].cache) == 2 &&
          gpu::get_live_resident_device_buffers() == warm_plan_live,
      "failed reused-inactive LDOS commit changed the active plan, caller, "
      "or bounded allocation state");
  {
    gpu::detail::resident_curl_session recovery_electric_session(
        &electric_owner, true);
    gpu::detail::resident_curl_session recovery_magnetic_session(
        &magnetic_owner, true);
    gpu::reset_dispatch_statistics();
    double reused_slot_recovery[4] = {};
    gpu::detail::resident_reduce_ldos_fp32(
        requests, 3, reused_slot_recovery);
    require(
        std::memcmp(reused_slot_recovery, replay, sizeof(replay)) == 0 &&
            gpu::detail::get_ldos_reduction_statistics()
                    .descriptor_uploads == 0 &&
            gpu::get_dispatch_statistics().host_to_device_bytes == 0 &&
            gpu::get_live_resident_device_buffers() == warm_plan_live,
        "LDOS active plan was not reusable after an overwritten inactive "
        "slot failed before commit");
    recovery_electric_session.finish(false);
    recovery_magnetic_session.finish(false);
  }

  for (std::size_t replacement = 0; replacement < 16; ++replacement) {
    gpu::detail::discard_resident_source_profile_for_owner(
        &magnetic_owner, indices[2].data(), amplitudes[2].data());
    gpu::detail::resident_curl_session replacement_electric_session(
        &electric_owner, true);
    gpu::detail::resident_curl_session replacement_magnetic_session(
        &magnetic_owner, true);
    gpu::reset_dispatch_statistics();
    double churn_result[4] = {};
    gpu::detail::resident_reduce_ldos_fp32(
        requests, 3, churn_result);
    const gpu::detail::ldos_reduction_statistics churn_statistics =
        gpu::detail::get_ldos_reduction_statistics();
    const gpu::resident_statistics churn_resident =
        gpu::get_resident_statistics();
    require(
        std::memcmp(churn_result, replay, sizeof(replay)) == 0 &&
            churn_statistics.descriptor_uploads == 1 &&
            churn_statistics.kernel_launches == 2 &&
            churn_resident.device_buffer_allocations == 0 &&
            gpu::get_live_resident_device_buffers() == warm_plan_live,
        "warmed LDOS A/B source replacement allocated a third plan buffer, "
        "changed bits, or leaked resident state");
    replacement_electric_session.finish(false);
    replacement_magnetic_session.finish(false);
  }

  gpu::detail::reset_resident_cache_for_owner_reusing_allocations(
      &magnetic_owner);
  require(
      gpu::get_live_resident_device_buffers() == warm_plan_live,
      "same-device non-result reset discarded retained LDOS allocations");
  {
    gpu::detail::resident_curl_session reset_electric_session(
        &electric_owner, true);
    gpu::detail::resident_curl_session reset_magnetic_session(
        &magnetic_owner, true);
    requests[2].cache = reset_magnetic_session.cache();
    gpu::reset_dispatch_statistics();
    double reset_result[4] = {};
    gpu::detail::resident_reduce_ldos_fp32(requests, 3, reset_result);
    require(
        std::memcmp(reset_result, replay, sizeof(replay)) == 0 &&
            gpu::detail::get_ldos_reduction_statistics()
                    .descriptor_uploads == 1 &&
            gpu::get_resident_statistics().device_buffer_allocations == 0 &&
            gpu::get_live_resident_device_buffers() == warm_plan_live,
        "same-device non-result reset did not reuse mirrors and the bounded "
        "inactive LDOS plan");
    reset_electric_session.finish(true);
    reset_magnetic_session.finish(true);
  }

  gpu::detail::reset_resident_cache_for_owner_reusing_allocations(
      &electric_owner);
  require(
      gpu::get_live_resident_device_buffers() == warm_plan_live &&
          gpu::detail::resident_ldos_plan_buffer_count_for_testing(
              requests[0].cache) == 2,
      "same-device result-cache reset discarded a bounded LDOS plan slot");
  {
    gpu::detail::resident_curl_session reset_result_session(
        &electric_owner, true);
    gpu::detail::resident_curl_session retained_non_result_session(
        &magnetic_owner, true);
    requests[0].cache = reset_result_session.cache();
    requests[1].cache = reset_result_session.cache();
    requests[2].cache = retained_non_result_session.cache();
    gpu::reset_dispatch_statistics();
    double reset_result[4] = {};
    gpu::detail::resident_reduce_ldos_fp32(requests, 3, reset_result);
    require(
        std::memcmp(reset_result, replay, sizeof(replay)) == 0 &&
            gpu::detail::get_ldos_reduction_statistics()
                    .descriptor_uploads == 1 &&
            gpu::get_resident_statistics().device_buffer_allocations == 0 &&
            gpu::detail::resident_ldos_plan_buffer_count_for_testing(
                requests[0].cache) == 2 &&
            gpu::get_live_resident_device_buffers() == warm_plan_live,
        "same-device result-cache reset did not retain/rewrite the bounded "
        "inactive plan without allocation");
    reset_result_session.finish(true);
    retained_non_result_session.finish(true);
  }

  require(
      electric_real == resident_electric_real &&
          electric_imaginary == resident_electric_imaginary &&
          real_only == resident_real_only &&
          magnetic_real == resident_magnetic_real &&
          magnetic_imaginary == resident_magnetic_imaginary,
      "resident CUDA LDOS test did not preserve device-authoritative fields");

  gpu::detail::destroy_resident_cache_for_owner(&magnetic_owner);
  const std::uint64_t live_after_non_result_destroy =
      gpu::get_live_resident_device_buffers();
  const std::uint64_t destroyed_non_result_buffers =
      warm_plan_live - live_after_non_result_destroy;
  require(
      live_after_non_result_destroy < warm_plan_live &&
          gpu::detail::resident_ldos_plan_buffer_count_for_testing(
              requests[0].cache) == 2,
      "non-result LDOS cache destruction failed to release its buffers or "
      "discarded a bounded result-cache plan slot");
  {
    gpu::detail::resident_curl_session recreated_electric_session(
        &electric_owner, true);
    gpu::detail::resident_curl_session recreated_magnetic_session(
        &magnetic_owner, true);
    requests[2].cache = recreated_magnetic_session.cache();
    gpu::reset_dispatch_statistics();
    double recreated_result[4] = {};
    gpu::detail::resident_reduce_ldos_fp32(
        requests, 3, recreated_result);
    const gpu::detail::ldos_reduction_statistics recreated_statistics =
        gpu::detail::get_ldos_reduction_statistics();
    const gpu::resident_statistics recreated_resident =
        gpu::get_resident_statistics();
    const std::size_t recreated_plan_buffers =
        gpu::detail::resident_ldos_plan_buffer_count_for_testing(
            requests[0].cache);
    const std::uint64_t recreated_live =
        gpu::get_live_resident_device_buffers();
    require(
        std::memcmp(recreated_result, replay, sizeof(replay)) == 0 &&
            recreated_statistics.descriptor_uploads == 1 &&
            recreated_resident.device_buffer_allocations == 4 &&
            recreated_plan_buffers == 2 &&
            recreated_live ==
                live_after_non_result_destroy +
                    recreated_resident.device_buffer_allocations,
        "recreated non-result LDOS cache used a stale cross-cache pointer or "
        "reallocated the warmed result plan (freed=" +
            std::to_string(destroyed_non_result_buffers) +
            ", allocated=" +
            std::to_string(recreated_resident.device_buffer_allocations) +
            ", plans=" + std::to_string(recreated_plan_buffers) +
            ", live=" + std::to_string(recreated_live) +
            ", destroyed-live=" +
            std::to_string(live_after_non_result_destroy) + ")");
    recreated_electric_session.finish(true);
    recreated_magnetic_session.finish(true);
  }
  gpu::detail::destroy_resident_cache_for_owner(&magnetic_owner);
  gpu::detail::destroy_resident_cache_for_owner(&electric_owner);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "resident CUDA LDOS lifecycle leaked a device allocation");
  if (am_master())
    std::cout
        << "PASS: resident CUDA LDOS deterministic batched reduction, "
           "32-byte replay, deterministic reentry/commit recovery, bounded "
           "A/B plan reuse, cross-cache reset/destroy/recreate, and lifecycle\n";
#endif
}

void require_resident_ldos_two_device_migration_contract() {
#if MEEP_SINGLE && MEEP_HAVE_CUDA
  require(count_processors() == 1,
          "LDOS two-device migration contract requires one MPI rank");
  std::vector<int> compatible_ordinals;
  for (const gpu::device_info &device : gpu::enumerate_devices())
    if (device.compatible) compatible_ordinals.push_back(device.ordinal);
  require(compatible_ordinals.size() >= 2,
          "LDOS two-device migration contract requires two compatible "
          "physical GPUs");

  const int original_ordinal = gpu::selected_device();
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  int result_owner = 0;
  int non_result_owner = 0;
  constexpr std::size_t field_count = 769;
  constexpr std::size_t source_count = 321;
  std::vector<float> electric(field_count);
  std::vector<float> magnetic(field_count);
  std::vector<std::ptrdiff_t> electric_indices(source_count);
  std::vector<std::ptrdiff_t> magnetic_indices(source_count);
  std::vector<float> electric_amplitudes(2 * source_count);
  std::vector<float> magnetic_amplitudes(2 * source_count);
  for (std::size_t index = 0; index < field_count; ++index) {
    electric[index] =
        0.21f * std::sin(0.013f * static_cast<float>(index + 1));
    magnetic[index] =
        -0.18f * std::cos(0.017f * static_cast<float>(index + 2));
  }
  for (std::size_t point = 0; point < source_count; ++point) {
    electric_indices[point] = static_cast<std::ptrdiff_t>(
        (19 * point + 7) % field_count);
    magnetic_indices[point] = static_cast<std::ptrdiff_t>(
        (23 * point + 11) % field_count);
    electric_amplitudes[2 * point] =
        0.04f * std::sin(0.031f * static_cast<float>(point + 1));
    electric_amplitudes[2 * point + 1] =
        -0.03f * std::cos(0.029f * static_cast<float>(point + 3));
    magnetic_amplitudes[2 * point] =
        -0.05f * std::cos(0.027f * static_cast<float>(point + 2));
    magnetic_amplitudes[2 * point + 1] =
        0.02f * std::sin(0.037f * static_cast<float>(point + 5));
  }

  const auto run_on_device = [&](int ordinal) {
    gpu::select_device(ordinal);
    gpu::reset_dispatch_statistics();
    double rebuilt[4] = {};
    {
      gpu::detail::resident_curl_session result_session(
          &result_owner, true);
      gpu::detail::resident_curl_session non_result_session(
          &non_result_owner, true);
      const gpu::detail::ldos_reduction_request_fp32 requests[2] = {
          {result_session.cache(), electric.data(), nullptr, field_count,
           electric_indices.data(), electric_amplitudes.data(), source_count,
           false},
          {non_result_session.cache(), magnetic.data(), nullptr, field_count,
           magnetic_indices.data(), magnetic_amplitudes.data(), source_count,
           true}};
      gpu::detail::resident_reduce_ldos_fp32(requests, 2, rebuilt);
      result_session.finish(false);
      non_result_session.finish(false);
    }
    gpu::detail::discard_resident_source_profile_for_owner(
        &non_result_owner, magnetic_indices.data(),
        magnetic_amplitudes.data());
    {
      gpu::detail::resident_curl_session result_session(
          &result_owner, true);
      gpu::detail::resident_curl_session non_result_session(
          &non_result_owner, true);
      const gpu::detail::ldos_reduction_request_fp32 requests[2] = {
          {result_session.cache(), electric.data(), nullptr, field_count,
           electric_indices.data(), electric_amplitudes.data(), source_count,
           false},
          {non_result_session.cache(), magnetic.data(), nullptr, field_count,
           magnetic_indices.data(), magnetic_amplitudes.data(), source_count,
           true}};
      gpu::detail::resident_reduce_ldos_fp32(requests, 2, rebuilt);
      gpu::reset_dispatch_statistics();
      double replay[4] = {};
      gpu::detail::resident_reduce_ldos_fp32(requests, 2, replay);
      require(
          std::memcmp(rebuilt, replay, sizeof(replay)) == 0 &&
              gpu::selected_device() == ordinal &&
              gpu::detail::get_ldos_reduction_statistics()
                      .descriptor_uploads == 0 &&
              gpu::get_dispatch_statistics().host_to_device_bytes == 0 &&
              gpu::get_dispatch_statistics().device_to_host_bytes ==
                  4 * sizeof(double) &&
              gpu::detail::resident_ldos_plan_buffer_count_for_testing(
                  requests[0].cache) == 2,
          "LDOS migration target did not provide fixed-device bitwise, "
          "zero-H2D replay");
      result_session.finish(true);
      non_result_session.finish(true);
    }
    return std::vector<double>(rebuilt, rebuilt + 4);
  };

  const std::vector<double> first =
      run_on_device(compatible_ordinals[0]);
  const std::uint64_t warm_live =
      gpu::get_live_resident_device_buffers();
  const std::vector<double> second =
      run_on_device(compatible_ordinals[1]);
  require(gpu::get_live_resident_device_buffers() == warm_live,
          "LDOS device migration did not free and recreate both bounded plan "
          "slots and mirrors at a stable live count");
  for (std::size_t value = 0; value < 4; ++value)
    require(
        std::abs(first[value] - second[value]) <=
            2.0e-6 * std::max(1.0, std::abs(first[value])),
        "LDOS two-device migration changed a numerical result");

  const std::vector<double> returned =
      run_on_device(compatible_ordinals[0]);
  require(
      std::memcmp(first.data(), returned.data(), 4 * sizeof(double)) == 0 &&
          gpu::get_live_resident_device_buffers() == warm_live,
      "LDOS migration back to the original test GPU changed fixed-device "
      "bits or resident lifecycle state");

  gpu::detail::destroy_resident_cache_for_owner(&non_result_owner);
  gpu::detail::destroy_resident_cache_for_owner(&result_owner);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "LDOS two-device migration leaked resident allocations");
  if (original_ordinal >= 0) gpu::select_device(original_ordinal);
  if (am_master())
    std::cout
        << "PASS: resident CUDA LDOS bounded-plan migration across two "
           "physical GPUs with fixed-device replay and lifecycle recovery\n";
#endif
}

void require_cuda_tile_coalescing_equivalence() {
#if MEEP_SINGLE
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  run_result coalesced;
  {
    scoped_environment_override control(
        "MEEP_GPU_DISABLE_TILE_COALESCING", nullptr);
    coalesced = run_material_case(gpu::backend_mode::cuda, false, 8);
  }
  require(gpu::get_live_resident_device_buffers() == live_before,
          "coalesced CUDA tile run leaked resident state");

  run_result replay_coalesced;
  gpu::curl_phase_replay_statistics replay_statistics{};
  {
    scoped_environment_override control(
        "MEEP_GPU_DISABLE_TILE_COALESCING", nullptr);
    // Reset after the replay plan is warm so the measured interval contains
    // only steady-state execution. Replay must preserve the same public
    // tile-coalescing telemetry as the ordinary descriptor walk.
    replay_coalesced =
        run_material_case(gpu::backend_mode::cuda, false, 8, false, 32);
    replay_statistics = gpu::get_curl_phase_replay_statistics();
  }
  require(gpu::get_live_resident_device_buffers() == live_before,
          "steady replay CUDA tile run leaked resident state");
  require(replay_coalesced.samples == coalesced.samples,
          "steady curl replay changed a tile-coalesced field, energy, or "
          "DFT observable");
  require(
      replay_coalesced.tile_coalescing.curl_chunk_phases > 0 &&
          replay_coalesced.tile_coalescing.curl_input_tiles >
              replay_coalesced.tile_coalescing.curl_chunk_phases,
      "steady curl replay omitted CUDA tile-coalescing telemetry");
  if (std::getenv("MEEP_GPU_DISABLE_CURL_PHASE_REPLAY") == nullptr)
    require(replay_statistics.hits > 0,
            "steady tile-coalescing fixture did not exercise curl replay");

  run_result legacy;
  {
    scoped_environment_override control(
        "MEEP_GPU_DISABLE_TILE_COALESCING", "1");
    legacy = run_material_case(gpu::backend_mode::cuda, false, 8);
  }
  require(gpu::get_live_resident_device_buffers() == live_before,
          "legacy CUDA tile run leaked resident state");
  require(coalesced.samples == legacy.samples,
          "CUDA tile coalescing changed a field, energy, or DFT observable");
  require(coalesced.statistics.cpu_curl_calls == 0 &&
              legacy.statistics.cpu_curl_calls == 0 &&
              coalesced.field_updates.cpu_update_eh_calls == 0 &&
              legacy.field_updates.cpu_update_eh_calls == 0,
          "CUDA tile coalescing equivalence silently used CPU field work");
  require(coalesced.statistics.cuda_curl_points ==
              legacy.statistics.cuda_curl_points &&
              coalesced.field_updates.cuda_update_eh_points ==
              legacy.field_updates.cuda_update_eh_points,
          "CUDA tile coalescing changed the field-update point count");
  if (am_master())
    std::cout
        << "tile-coalescing-exact: curl-phases="
        << coalesced.tile_coalescing.curl_chunk_phases
        << " input-tiles="
        << coalesced.tile_coalescing.curl_input_tiles
        << " update-eh-phases="
        << coalesced.tile_coalescing.update_eh_chunk_phases
        << " input-tiles="
        << coalesced.tile_coalescing.update_eh_input_tiles << '\n';
  require(
      coalesced.tile_coalescing.curl_chunk_phases > 0 &&
          coalesced.tile_coalescing.curl_input_tiles >
              coalesced.tile_coalescing.curl_chunk_phases &&
          coalesced.tile_coalescing.update_eh_chunk_phases > 0 &&
          coalesced.tile_coalescing.update_eh_input_tiles >
              coalesced.tile_coalescing.update_eh_chunk_phases,
      "CUDA tile coalescing statistics did not prove both curl and E/H paths");
  require(
      legacy.tile_coalescing.curl_chunk_phases == 0 &&
          legacy.tile_coalescing.curl_input_tiles == 0 &&
      legacy.tile_coalescing.update_eh_chunk_phases == 0 &&
          legacy.tile_coalescing.update_eh_input_tiles == 0,
      "disabled CUDA tile coalescing reported optimized execution");
#endif
}

struct dft_phase_sharing_result {
  std::vector<std::complex<realnum> > complete_dft_storage;
  gpu::detail::dft_batch_statistics statistics{};
  double accumulated_magnitude = 0.0;
};

void require_overlapping_dft_batch_rejected_before_prepare() {
  float output[64] = {};
  const auto request = [](float *storage, std::size_t points,
                          std::size_t frequencies) {
    return gpu::detail::dft_update_request_fp32{
        nullptr, storage, nullptr, nullptr, 0, nullptr, nullptr,
        points, nullptr, frequencies, 0.0, 1.0, 0.0, 0, 0};
  };
  const auto require_rejected = [&](const char *label,
                                    gpu::detail::dft_update_request_fp32 first,
                                    gpu::detail::dft_update_request_fp32 second) {
    gpu::reset_dispatch_statistics();
    const std::uint64_t live_before =
        gpu::get_live_resident_device_buffers();
    bool rejected = false;
    try {
      const gpu::detail::dft_update_request_fp32 requests[] = {
          first, second};
      gpu::detail::resident_update_dft_batch_fp32(requests, 2);
    }
    catch (const std::invalid_argument &error) {
      rejected = std::string(error.what()).find("overlap") !=
                 std::string::npos;
    }
    const gpu::detail::dft_batch_statistics statistics =
        gpu::detail::get_dft_batch_statistics();
    require(
        rejected && statistics.batch_calls == 0 &&
            statistics.submitted_updates == 0 &&
            statistics.phase_preparation_launches == 0 &&
            statistics.phase_reuses == 0 &&
            statistics.update_kernel_launches == 0 &&
            gpu::get_live_resident_device_buffers() == live_before,
        std::string("overlapping DFT output was not rejected before ") +
            label + " preparation");
  };

  require_rejected(
      "duplicate-output", request(output, 4, 2), request(output, 4, 2));
  require_rejected(
      "partial-overlap", request(output, 4, 2),
      request(output + 4, 4, 2));
  require_rejected(
      "same-base-resize", request(output, 2, 2),
      request(output, 6, 2));
  gpu::detail::dft_update_request_fp32 aliased_input =
      request(output, 4, 2);
  aliased_input.field_real = output;
  aliased_input.field_array_count = 64;
  require_rejected(
      "output-input-alias", aliased_input,
      request(output + 32, 4, 2));

  float first_output[16] = {};
  float second_output[16] = {};
  float shared_input[16] = {};
  gpu::detail::dft_update_request_fp32 inconsistent_extents[] = {
      request(first_output, 2, 2), request(second_output, 2, 2)};
  inconsistent_extents[0].field_real = shared_input;
  inconsistent_extents[0].field_array_count = 8;
  inconsistent_extents[1].field_real = shared_input;
  inconsistent_extents[1].field_array_count = 16;
  gpu::reset_dispatch_statistics();
  bool inconsistent_extent_rejected = false;
  try {
    gpu::detail::resident_update_dft_batch_fp32(
        inconsistent_extents, 2);
  }
  catch (const std::invalid_argument &error) {
    inconsistent_extent_rejected =
        std::string(error.what()).find("different byte extents") !=
        std::string::npos;
  }
  const gpu::detail::dft_batch_statistics extent_statistics =
      gpu::detail::get_dft_batch_statistics();
  require(
      inconsistent_extent_rejected && extent_statistics.batch_calls == 0 &&
          extent_statistics.submitted_updates == 0 &&
          extent_statistics.phase_preparation_launches == 0 &&
          extent_statistics.phase_reuses == 0 &&
          extent_statistics.update_kernel_launches == 0,
      "one DFT input base with inconsistent mirror extents was not rejected "
      "before preparation");

  gpu::detail::dft_update_request_fp32 overlapping_inputs[] = {
      request(first_output, 2, 2), request(second_output, 2, 2)};
  overlapping_inputs[0].field_real = shared_input;
  overlapping_inputs[0].field_array_count = 16;
  overlapping_inputs[1].field_real = shared_input + 4;
  overlapping_inputs[1].field_array_count = 8;
  gpu::reset_dispatch_statistics();
  bool overlapping_input_rejected = false;
  try {
    gpu::detail::resident_update_dft_batch_fp32(
        overlapping_inputs, 2);
  }
  catch (const std::invalid_argument &error) {
    overlapping_input_rejected =
        std::string(error.what()).find("input mirrors") !=
        std::string::npos;
  }
  const gpu::detail::dft_batch_statistics overlap_statistics =
      gpu::detail::get_dft_batch_statistics();
  require(
      overlapping_input_rejected && overlap_statistics.batch_calls == 0 &&
          overlap_statistics.submitted_updates == 0 &&
          overlap_statistics.phase_preparation_launches == 0 &&
          overlap_statistics.phase_reuses == 0 &&
          overlap_statistics.update_kernel_launches == 0,
      "partially overlapping DFT input mirrors were not rejected before "
      "preparation");
}

void require_malformed_dft_batch_is_atomic_and_recoverable() {
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  int owner = 0;
  std::vector<float> output(8, 0.125f);
  std::vector<float> rejected_output(8, -0.25f);
  const std::vector<float> output_before = output;
  const std::vector<float> rejected_before = rejected_output;
  std::vector<float> field = {
      0.0f, 0.75f, -0.5f, 1.25f, 0.375f, -0.625f, 0.875f, -1.0f};
  const std::ptrdiff_t valid_indices[] = {1, 3};
  const std::ptrdiff_t invalid_indices[] = {
      1, static_cast<std::ptrdiff_t>(field.size())};
  const float weights[] = {1.0f, 0.5f};
  const double frequencies[] = {0.31, 0.47};

  gpu::reset_dispatch_statistics();
  bool rejected = false;
  {
    gpu::detail::resident_curl_session resident(&owner, true);
    const gpu::detail::dft_update_request_fp32 requests[] = {
        {resident.cache(), output.data(), field.data(), nullptr,
         field.size(), valid_indices, weights, 2, frequencies, 2,
         0.37, 0.8, -0.2, 0, 0},
        {resident.cache(), rejected_output.data(), field.data(), nullptr,
         field.size(), invalid_indices, weights, 2, frequencies, 2,
         0.37, 0.8, -0.2, 0, 0}};
    try {
      gpu::detail::resident_update_dft_batch_fp32(requests, 2);
    }
    catch (const std::out_of_range &) {
      rejected = true;
    }
  }

  const gpu::detail::dft_batch_statistics rejected_statistics =
      gpu::detail::get_dft_batch_statistics();
  require(
      rejected && output == output_before &&
          rejected_output == rejected_before &&
          rejected_statistics.batch_calls == 0 &&
          rejected_statistics.submitted_updates == 0 &&
          rejected_statistics.phase_preparation_launches == 0 &&
          rejected_statistics.phase_reuses == 0 &&
          rejected_statistics.update_kernel_launches == 0 &&
          !gpu::detail::resident_phase_is_active_for_owner(&owner),
      "a malformed later DFT request partially updated output or retained "
      "an active resident session");

  gpu::reset_dispatch_statistics();
  {
    gpu::detail::resident_curl_session resident(&owner, true);
    const gpu::detail::dft_update_request_fp32 request = {
        resident.cache(), output.data(), field.data(), nullptr,
        field.size(), valid_indices, weights, 2, frequencies, 2,
        0.37, 0.8, -0.2, 0, 0};
    gpu::detail::resident_update_dft_batch_fp32(&request, 1);
    resident.finish();
  }
  const gpu::detail::dft_batch_statistics recovery_statistics =
      gpu::detail::get_dft_batch_statistics();
  bool recovery_values_match = true;
  for (std::size_t point = 0; point < 2; ++point)
    for (std::size_t frequency = 0; frequency < 2; ++frequency) {
      const std::complex<double> phase =
          std::polar(1.0, frequencies[frequency] * 0.37) *
          std::complex<double>(0.8, -0.2);
      const std::complex<double> expected =
          std::complex<double>(0.125, 0.125) +
          phase * static_cast<double>(
                      field[valid_indices[point]] * weights[point]);
      const std::size_t output_index = 2 * (point * 2 + frequency);
      const std::complex<double> observed(
          output[output_index], output[output_index + 1]);
      recovery_values_match =
          recovery_values_match &&
          std::abs(observed - expected) <=
              2e-6 * std::max(1.0, std::abs(expected));
    }
  require(
      output != output_before && recovery_values_match &&
          recovery_statistics.batch_calls == 1 &&
          recovery_statistics.submitted_updates == 1 &&
          recovery_statistics.phase_preparation_launches == 1 &&
          recovery_statistics.phase_reuses == 0 &&
          recovery_statistics.update_kernel_launches == 1 &&
          !gpu::detail::resident_phase_is_active_for_owner(&owner),
      "a resident DFT cache did not recover after rejecting a malformed "
      "later request");
  gpu::detail::destroy_resident_cache_for_owner(&owner);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "malformed DFT batch recovery leaked a resident allocation");
}

void require_direct_dft_update_sync_and_phase_key_mutation() {
  scoped_environment_override sharing_control(
      "MEEP_GPU_DISABLE_DFT_PHASE_SHARING", nullptr);
  scoped_environment_override batch_enable(
      "MEEP_GPU_ENABLE_MULTI_MONITOR_DFT_BATCH", "1");
  scoped_environment_override batch_disable(
      "MEEP_GPU_DISABLE_MULTI_MONITOR_DFT_BATCH", nullptr);
  gpu::set_backend(gpu::backend_mode::cuda);
  const grid_volume gv = vol2d(2.0, 1.8, 14.0);
  structure s(gv, vacuum, no_pml(), identity(), 4);
  fields f(&s, 0.0, 0.0, true, 64, 64);
  f.use_real_fields();
  gaussian_src_time source(0.29, 0.10);
  f.add_point_source(Ez, source, gv.center(), 0.8);
  component components[] = {Ez};
  const volume monitor_region(vec(0.20, 0.15), vec(1.80, 1.65));
  dft_fields first = f.add_dft_fields(
      components, 1, monitor_region, 0.18, 0.42, 5, true, 1);
  dft_fields second = f.add_dft_fields(
      components, 1, monitor_region, 0.18, 0.42, 5, true, 1);

  std::vector<const fields_chunk *> cache_owners;
  for (dft_chunk *chunk = first.chunks; chunk;
       chunk = chunk->next_in_dft)
    if (std::find(cache_owners.begin(), cache_owners.end(), chunk->fc) ==
        cache_owners.end())
      cache_owners.push_back(chunk->fc);
  require(and_to_all(cache_owners.size() >= 2),
          "direct DFT update regression did not span multiple resident "
          "caches on every rank");

  for (int step = 0; step < 8; ++step)
    f.step();
  std::vector<std::complex<realnum> > before;
  for (dft_chunk *chunk = first.chunks; chunk;
       chunk = chunk->next_in_dft) {
    gpu::detail::sync_resident_cache_for_owner(chunk->fc);
    before.insert(
        before.end(), chunk->dft,
        chunk->dft + chunk->N * chunk->omega.size());
  }

  gpu::reset_dispatch_statistics();
  f.update_dfts();
  const gpu::detail::dft_batch_statistics shared =
      gpu::detail::get_dft_batch_statistics();
  double local_change = 0.0;
  std::size_t before_offset = 0;
  for (dft_chunk *chunk = first.chunks; chunk;
       chunk = chunk->next_in_dft) {
    const std::size_t value_count = chunk->N * chunk->omega.size();
    for (std::size_t index = 0; index < value_count; ++index)
      local_change +=
          std::abs(chunk->dft[index] - before[before_offset + index]);
    before_offset += value_count;
  }
  bool twin_outputs_match = true;
  dft_chunk *first_chunk = first.chunks;
  dft_chunk *second_chunk = second.chunks;
  while (first_chunk && second_chunk) {
    const std::size_t value_count =
        first_chunk->N * first_chunk->omega.size();
    twin_outputs_match =
        twin_outputs_match && first_chunk->N == second_chunk->N &&
        first_chunk->omega.size() == second_chunk->omega.size() &&
        std::memcmp(first_chunk->dft, second_chunk->dft,
                    value_count * sizeof(std::complex<realnum>)) == 0;
    first_chunk = first_chunk->next_in_dft;
    second_chunk = second_chunk->next_in_dft;
  }
  twin_outputs_match =
      twin_outputs_match && !first_chunk && !second_chunk;
  bool sessions_finished = true;
  for (const fields_chunk *owner : cache_owners)
    sessions_finished =
        sessions_finished &&
        !gpu::detail::resident_phase_is_active_for_owner(owner);
  require(
      and_to_all(shared.batch_calls == 1 && shared.submitted_updates >= 4 &&
                 shared.phase_preparation_launches == 1 &&
                 shared.phase_reuses + 1 == shared.submitted_updates &&
                 shared.update_kernel_launches == 1 &&
                 shared.multi_monitor_automatic_checks == 0 &&
                 shared.multi_monitor_automatic_selected == 0 &&
                 shared.multi_monitor_automatic_rejected == 0 &&
                 shared.multi_monitor_forced_batches == 1 &&
                 shared.multi_monitor_batched_updates ==
                     shared.submitted_updates &&
                 shared.multi_monitor_unbatched_updates == 0 &&
                 twin_outputs_match && sessions_finished) &&
          sum_to_all(local_change) > 0.0,
      "direct fields::update_dfts did not share one phase, synchronize every "
      "cache to host, or finish its owned resident sessions");

  for (dft_chunk *chunk = second.chunks; chunk;
       chunk = chunk->next_in_dft)
    chunk->scale *= std::complex<double>(0.71, -0.23);
  gpu::reset_dispatch_statistics();
  f.update_dfts();
  const gpu::detail::dft_batch_statistics mutated =
      gpu::detail::get_dft_batch_statistics();
  bool public_phases_match = true;
  for (dft_chunk *chunk = second.chunks; chunk;
       chunk = chunk->next_in_dft)
    for (std::size_t frequency = 0; frequency < chunk->omega.size();
         ++frequency) {
      const std::complex<double> expected =
          std::polar(1.0, chunk->omega[frequency] * f.time()) *
          chunk->scale;
      public_phases_match =
          public_phases_match &&
          std::abs(std::complex<double>(
                       chunk->dft_phase[frequency].real(),
                       chunk->dft_phase[frequency].imag()) -
                   expected) <=
              2e-6 * std::max(1.0, std::abs(expected));
    }
  require(
      and_to_all(mutated.batch_calls == 1 &&
                 mutated.submitted_updates == shared.submitted_updates &&
                 mutated.phase_preparation_launches ==
                     shared.phase_preparation_launches + 1 &&
                 mutated.phase_reuses + 1 == shared.phase_reuses &&
                 mutated.update_kernel_launches == 1 &&
                 mutated.multi_monitor_automatic_checks == 0 &&
                 mutated.multi_monitor_automatic_selected == 0 &&
                 mutated.multi_monitor_automatic_rejected == 0 &&
                 mutated.multi_monitor_forced_batches == 1 &&
                 mutated.multi_monitor_batched_updates ==
                     mutated.submitted_updates &&
                 public_phases_match),
      "mutating one DFT monitor scale did not split exactly one phase key");

  second.remove();
  first.remove();
}

enum class dft_monitor_batch_control { automatic, forced, disabled };

dft_phase_sharing_result run_dft_phase_sharing_case(
    gpu::backend_mode mode, bool disable_sharing, bool complex_fields,
    dft_monitor_batch_control batch_control) {
  scoped_environment_override sharing_control(
      "MEEP_GPU_DISABLE_DFT_PHASE_SHARING",
      disable_sharing ? "1" : nullptr);
  scoped_environment_override batch_enable(
      "MEEP_GPU_ENABLE_MULTI_MONITOR_DFT_BATCH",
      batch_control == dft_monitor_batch_control::forced ? "1" : nullptr);
  scoped_environment_override batch_disable(
      "MEEP_GPU_DISABLE_MULTI_MONITOR_DFT_BATCH",
      batch_control == dft_monitor_batch_control::disabled ? "1" : nullptr);
  gpu::set_backend(mode);
  const grid_volume gv = vol2d(2.0, 1.8, 14.0);
  structure s(gv, vacuum, no_pml(), identity(), 4);
  fields f(&s, 0.0, 0.0, true, 64, 64);
  if (complex_fields)
    f.use_bloch(vec(0.07, -0.03));
  else
    f.use_real_fields();
  gaussian_src_time source(0.29, 0.10);
  f.add_point_source(
      Ez, source, gv.center(),
      complex_fields ? std::complex<double>(0.8, -0.21)
                     : std::complex<double>(0.8, 0.0));

  component components[] = {Ez};
  component magnetic_components[] = {Hy};
  const volume monitor_region(vec(0.25, 0.20), vec(1.75, 1.60));
  dft_fields first = f.add_dft_fields(
      components, 1, monitor_region, 0.18, 0.42, 5, true, 1);
  dft_fields second = f.add_dft_fields(
      components, 1, monitor_region, 0.18, 0.42, 5, true, 1);
  dft_fields third = f.add_dft_fields(
      components, 1, monitor_region, 0.18, 0.42, 5, true, 1);
  dft_fields fourth = f.add_dft_fields(
      components, 1, monitor_region, 0.18, 0.42, 5, true, 1);
  dft_fields fifth = f.add_dft_fields(
      components, 1, monitor_region, 0.18, 0.42, 5, true, 2, true);
  dft_fields magnetic = f.add_dft_fields(
      magnetic_components, 1, monitor_region, 0.18, 0.42, 5, true, 1);
  dft_fields alternate_frequencies = f.add_dft_fields(
      components, 1, monitor_region, 0.18, 0.42, 5, true, 1);
  bool persistent_monitor_is_expanded = false;
  for (dft_chunk *chunk = fifth.chunks; chunk;
       chunk = chunk->next_in_dft)
    persistent_monitor_is_expanded =
        persistent_monitor_is_expanded ||
        (chunk->persist &&
         (!(chunk->is == chunk->is_old) || !(chunk->ie == chunk->ie_old)));
  require(and_to_all(persistent_monitor_is_expanded),
          "persistent DFT sharing regression did not exercise an "
          "adjoint-expanded monitor on every rank");
  dft_fields *monitors[] = {
      &first, &second, &third, &fourth, &fifth,
      &magnetic, &alternate_frequencies};
  std::vector<const fields_chunk *> cache_owners;
  for (dft_fields *monitor : monitors)
    for (dft_chunk *chunk = monitor->chunks; chunk;
         chunk = chunk->next_in_dft)
      if (std::find(
              cache_owners.begin(), cache_owners.end(), chunk->fc) ==
          cache_owners.end())
        cache_owners.push_back(chunk->fc);
  require(
      and_to_all(cache_owners.size() >= 2),
      "DFT phase-sharing equivalence did not span multiple resident caches "
      "on every rank");

  // Two non-adjacent monitor groups deliberately use a second phase key.
  // This verifies exact grouping by omega/time/scale rather than accidental
  // reuse of the most recently prepared phase array.
  for (dft_chunk *chunk = second.chunks; chunk;
       chunk = chunk->next_in_dft)
    chunk->scale *= std::complex<double>(0.73, -0.19);
  for (dft_chunk *chunk = fourth.chunks; chunk;
       chunk = chunk->next_in_dft)
    chunk->scale *= std::complex<double>(0.73, -0.19);

  gpu::reset_dispatch_statistics();
  for (int step = 0; step < 12; ++step)
    f.step();
  for (dft_chunk *chunk = alternate_frequencies.chunks; chunk;
       chunk = chunk->next_in_dft)
    chunk->omega[2] += 0.017;
  for (int step = 12; step < 24; ++step)
    f.step();

  dft_phase_sharing_result result;
  for (dft_fields *monitor : monitors)
    for (dft_chunk *chunk = monitor->chunks; chunk;
         chunk = chunk->next_in_dft) {
      gpu::detail::sync_resident_cache_for_owner(chunk->fc);
      const std::size_t value_count = chunk->N * chunk->omega.size();
      result.complete_dft_storage.insert(
          result.complete_dft_storage.end(), chunk->dft,
          chunk->dft + value_count);
      for (std::size_t index = 0; index < value_count; ++index)
        result.accumulated_magnitude += std::abs(chunk->dft[index]);
    }
  result.statistics = gpu::detail::get_dft_batch_statistics();
  fifth.remove();
  return result;
}

void require_dft_phase_sharing_equivalence() {
  require_malformed_dft_batch_is_atomic_and_recoverable();
  require_direct_dft_update_sync_and_phase_key_mutation();
  for (bool complex_fields : {false, true}) {
    const std::uint64_t live_before =
        gpu::get_live_resident_device_buffers();
    const dft_phase_sharing_result sharing =
        run_dft_phase_sharing_case(
            gpu::backend_mode::cuda, false, complex_fields,
            dft_monitor_batch_control::forced);
    require(gpu::get_live_resident_device_buffers() == live_before,
            "DFT phase-sharing run leaked a resident allocation");
    const dft_phase_sharing_result control =
        run_dft_phase_sharing_case(
            gpu::backend_mode::cuda, true, complex_fields,
            dft_monitor_batch_control::forced);
    require(gpu::get_live_resident_device_buffers() == live_before,
            "DFT no-sharing control run leaked a resident allocation");

    const dft_phase_sharing_result unbatched =
        run_dft_phase_sharing_case(
            gpu::backend_mode::cuda, false, complex_fields,
            dft_monitor_batch_control::disabled);
    require(gpu::get_live_resident_device_buffers() == live_before,
            "disabled multi-monitor DFT run leaked a resident allocation");
    const dft_phase_sharing_result automatic =
        run_dft_phase_sharing_case(
            gpu::backend_mode::cuda, false, complex_fields,
            dft_monitor_batch_control::automatic);
    require(gpu::get_live_resident_device_buffers() == live_before,
            "automatic multi-monitor DFT run leaked a resident allocation");

    const auto arrays_match = [](const dft_phase_sharing_result &left,
                                 const dft_phase_sharing_result &right) {
      return left.complete_dft_storage.size() ==
                 right.complete_dft_storage.size() &&
             (left.complete_dft_storage.empty() ||
              std::memcmp(
                  left.complete_dft_storage.data(),
                  right.complete_dft_storage.data(),
                  left.complete_dft_storage.size() *
                      sizeof(std::complex<realnum>)) == 0);
    };
    const bool local_exact =
        sharing.complete_dft_storage.size() ==
            control.complete_dft_storage.size() &&
        arrays_match(sharing, control) &&
        arrays_match(sharing, unbatched) &&
        arrays_match(sharing, automatic);
    require(and_to_all(local_exact),
            "DFT phase sharing or multi-monitor batching changed a bit in a "
            "complete monitor array");
    require(sum_to_all(sharing.accumulated_magnitude) > 0.0,
            "DFT phase-sharing equivalence accumulated no signal");

    const dft_phase_sharing_result cpu =
        run_dft_phase_sharing_case(
            gpu::backend_mode::cpu, false, complex_fields,
            dft_monitor_batch_control::automatic);
    bool local_cpu_agreement =
        cpu.complete_dft_storage.size() ==
        sharing.complete_dft_storage.size();
    if (local_cpu_agreement)
      for (std::size_t index = 0;
           index < sharing.complete_dft_storage.size(); ++index) {
        const double error = std::abs(
            cpu.complete_dft_storage[index] -
            sharing.complete_dft_storage[index]);
        const double scale = std::max(
            std::abs(cpu.complete_dft_storage[index]),
            std::abs(sharing.complete_dft_storage[index]));
        if (!std::isfinite(error) || error > 3e-6 + 3e-4 * scale) {
          local_cpu_agreement = false;
          break;
        }
      }
    require(and_to_all(local_cpu_agreement),
            "DFT phase-sharing full monitor arrays disagree with CPU");

    const std::uint64_t sharing_updates =
        global_sum(sharing.statistics.submitted_updates);
    const std::uint64_t sharing_preparations =
        global_sum(sharing.statistics.phase_preparation_launches);
    const std::uint64_t sharing_reuses =
        global_sum(sharing.statistics.phase_reuses);
    const std::uint64_t control_updates =
        global_sum(control.statistics.submitted_updates);
    const std::uint64_t control_preparations =
        global_sum(control.statistics.phase_preparation_launches);
    const std::uint64_t control_reuses =
        global_sum(control.statistics.phase_reuses);
    const std::uint64_t sharing_launches =
        global_sum(sharing.statistics.update_kernel_launches);
    const std::uint64_t unbatched_launches =
        global_sum(unbatched.statistics.update_kernel_launches);
    const std::uint64_t automatic_launches =
        global_sum(automatic.statistics.update_kernel_launches);
    require(
        sharing_updates > 0 && sharing_updates == control_updates &&
            sharing_preparations < sharing_updates &&
            sharing_preparations + sharing_reuses == sharing_updates,
        "DFT phase sharing did not eliminate phase-preparation launches");
    require(
        control_preparations == control_updates && control_reuses == 0,
        "disabled DFT phase sharing unexpectedly reused a prepared phase");
    const bool batching_controls_valid =
        sharing.statistics.multi_monitor_automatic_checks == 0 &&
            sharing.statistics.multi_monitor_automatic_selected == 0 &&
            sharing.statistics.multi_monitor_automatic_rejected == 0 &&
            sharing.statistics.multi_monitor_forced_batches > 0 &&
            sharing.statistics.multi_monitor_batched_updates ==
                sharing.statistics.submitted_updates &&
            sharing.statistics.multi_monitor_unbatched_updates == 0 &&
            sharing.statistics.multi_monitor_plan_uploads > 0 &&
            sharing.statistics.multi_monitor_plan_reuses > 0 &&
            sharing.statistics
                    .multi_monitor_metadata_host_to_device_bytes > 0 &&
            sharing_launches < sharing_updates &&
            unbatched.statistics.multi_monitor_unbatched_updates ==
                unbatched.statistics.submitted_updates &&
            unbatched.statistics.multi_monitor_batched_updates == 0 &&
            unbatched_launches == sharing_updates &&
            automatic.statistics.multi_monitor_automatic_checks > 0 &&
            automatic.statistics.multi_monitor_automatic_selected == 0 &&
            automatic.statistics.multi_monitor_automatic_rejected ==
                automatic.statistics.multi_monitor_automatic_checks &&
            automatic.statistics.multi_monitor_batched_updates == 0 &&
            automatic.statistics.multi_monitor_unbatched_updates ==
                automatic.statistics.submitted_updates &&
            automatic.statistics.multi_monitor_plan_uploads == 0 &&
            automatic.statistics.multi_monitor_plan_reuses == 0 &&
            automatic_launches == sharing_updates;
    if (!batching_controls_valid && am_master())
      std::cerr
          << "DFT multi-monitor diagnostics: fields="
          << (complex_fields ? "complex" : "real")
          << " sharing={updates:"
          << sharing.statistics.submitted_updates
          << ",launches:" << sharing_launches
          << ",forced:"
          << sharing.statistics.multi_monitor_forced_batches
          << ",batched:"
          << sharing.statistics.multi_monitor_batched_updates
          << ",unbatched:"
          << sharing.statistics.multi_monitor_unbatched_updates
          << ",uploads:"
          << sharing.statistics.multi_monitor_plan_uploads
          << ",reuses:"
          << sharing.statistics.multi_monitor_plan_reuses
          << ",metadata_bytes:"
          << sharing.statistics
                 .multi_monitor_metadata_host_to_device_bytes
          << "} disabled={updates:"
          << unbatched.statistics.submitted_updates
          << ",launches:" << unbatched_launches
          << ",batched:"
          << unbatched.statistics.multi_monitor_batched_updates
          << ",unbatched:"
          << unbatched.statistics.multi_monitor_unbatched_updates
          << "} automatic={updates:"
          << automatic.statistics.submitted_updates
          << ",launches:" << automatic_launches
          << ",checks:"
          << automatic.statistics.multi_monitor_automatic_checks
          << ",selected:"
          << automatic.statistics.multi_monitor_automatic_selected
          << ",rejected:"
          << automatic.statistics.multi_monitor_automatic_rejected
          << ",batched:"
          << automatic.statistics.multi_monitor_batched_updates
          << ",unbatched:"
          << automatic.statistics.multi_monitor_unbatched_updates
          << ",uploads:"
          << automatic.statistics.multi_monitor_plan_uploads
          << ",reuses:"
          << automatic.statistics.multi_monitor_plan_reuses
          << "} global_updates=" << sharing_updates << '\n';
    require(
        batching_controls_valid,
        "multi-monitor DFT forced/disabled/fail-closed automatic controls did "
        "not prove one physical update launch per forced batch");
    const std::uint64_t compared_bytes = global_sum(
        static_cast<std::uint64_t>(
            sharing.complete_dft_storage.size() *
            sizeof(std::complex<realnum>)));
    if (am_master())
      std::cout << "dft-phase-sharing-exact: fields="
                << (complex_fields ? "complex" : "real")
                << " updates=" << sharing_updates
                << " preparations=" << control_preparations << "->"
                << sharing_preparations << " reuses=" << sharing_reuses
                << " update-launches=" << unbatched_launches << "->"
                << sharing_launches
                << " bytes=" << compared_bytes << '\n';
  }
}

struct dft_array_materialization_result {
  std::vector<std::complex<double> > values;
  gpu::dft_materialization_statistics statistics{};
};

std::size_t dft_chunk_scalar_count(dft_chunk *chunks) {
  std::size_t total = 0;
  for (dft_chunk *chunk = chunks; chunk; chunk = chunk->next_in_dft) {
    if (chunk->N &&
        chunk->omega.size() >
            std::numeric_limits<std::size_t>::max() / chunk->N / 2)
      throw std::overflow_error("test DFT checkpoint scalar count overflow");
    const std::size_t count = 2 * chunk->N * chunk->omega.size();
    if (count > std::numeric_limits<std::size_t>::max() - total)
      throw std::overflow_error("test DFT checkpoint total overflow");
    total += count;
  }
  return total;
}

std::size_t dft_chunk_count(dft_chunk *chunks, bool nonempty_only) {
  std::size_t count = 0;
  for (dft_chunk *chunk = chunks; chunk; chunk = chunk->next_in_dft)
    if (!nonempty_only || (chunk->N && !chunk->omega.empty())) ++count;
  return count;
}

void append_dft_checkpoint_ranges(
    dft_chunk *chunks,
    std::vector<gpu::detail::resident_checkpoint_range_fp32> &ranges) {
  for (dft_chunk *chunk = chunks; chunk; chunk = chunk->next_in_dft)
    ranges.push_back(
        {chunk->fc, reinterpret_cast<const float *>(chunk->dft),
         2 * chunk->N * chunk->omega.size()});
}

std::size_t attached_dft_chunk_count(const fields &f) {
  std::size_t count = 0;
  for (int chunk_index = 0; chunk_index < f.num_chunks; ++chunk_index)
    if (f.chunks[chunk_index]->is_mine())
      for (dft_chunk *chunk = f.chunks[chunk_index]->dft_chunks; chunk;
           chunk = chunk->next_in_chunk)
        ++count;
  return count;
}

std::vector<std::complex<double> > capture_dft_array(
    fields &f, dft_flux &monitor, component c, int frequency) {
  int rank = -1;
  std::size_t dims[3] = {0, 0, 0};
  std::unique_ptr<std::complex<realnum>[]> raw(
      f.get_dft_array(monitor, c, frequency, &rank, dims));
  require(raw != nullptr, "checkpoint get_dft_array returned null");
  std::size_t count = rank == 0 ? 1 : 1;
  for (int dimension = 0; dimension < rank; ++dimension) {
    require(dims[dimension] <=
                std::numeric_limits<std::size_t>::max() / count,
            "checkpoint get_dft_array dimension overflow");
    count *= dims[dimension];
  }
  std::vector<std::complex<double> > values(count);
  for (std::size_t index = 0; index < count; ++index)
    values[index] = std::complex<double>(raw[index]);
  return values;
}

void require_complex_vectors_close(
    const std::vector<std::complex<double> > &expected,
    const std::vector<std::complex<double> > &actual,
    const std::string &label, double relative, double absolute);

struct public_dft_array_case_result {
  std::vector<std::string> labels;
  std::vector<std::vector<std::complex<double> > > arrays;
  gpu::dft_materialization_statistics statistics{};
  bool saw_nonidentity_symmetry = false;
  bool saw_nonzero_measure = false;
  bool saw_nontrivial_stored_weight = false;
};

public_dft_array_case_result run_public_dft_array_case(
    gpu::backend_mode mode) {
  gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();
  const grid_volume gv = vol2d(1.8, 1.6, 10.0);
  const symmetry reflection = mirror(X, gv);
  structure s(gv, vacuum, no_pml(), reflection, 4);
  fields f(&s, 0.0, 0.0, true, 64, 64);
  f.use_real_fields();
  gaussian_src_time source(0.27, 0.10);
  f.add_point_source(Ez, source, gv.center(), 0.8);

  const std::vector<double> frequencies = {0.23, 0.31};
  dft_flux plane = f.add_dft_flux_plane(
      volume(vec(1.35, 0.25), vec(1.35, 1.35)), frequencies);
  dft_flux box = f.add_dft_flux_box(
      volume(vec(0.35, 0.25), vec(1.45, 1.35)), frequencies);
  volume_list force_region(
      volume(vec(1.30, 0.30), vec(1.30, 1.30)), Sx, 0.75);
  dft_force force = f.add_dft_force(&force_region, frequencies, 1);
  volume_list near_region(
      volume(vec(0.30, 0.25), vec(1.50, 0.25)), Sy, -1.25,
      new volume_list(
          volume(vec(1.50, 0.25), vec(1.50, 1.35)), Sx, 0.75,
          new volume_list(
              volume(vec(0.30, 1.35), vec(1.50, 1.35)), Sy, 1.5,
              new volume_list(
                  volume(vec(0.30, 0.25), vec(0.30, 1.35)), Sx,
                  -0.5))));
  dft_near2far near =
      f.add_dft_near2far(&near_region, frequencies, 1);

  for (int step = 0; step < 36; ++step) f.step();

  public_dft_array_case_result result;
  const auto inspect_chunks = [&](dft_chunk *chunks) {
    for (dft_chunk *chunk = chunks; chunk; chunk = chunk->next_in_dft) {
      result.saw_nonidentity_symmetry =
          result.saw_nonidentity_symmetry || chunk->sn != 0;
      result.saw_nonzero_measure =
          result.saw_nonzero_measure ||
          (chunk->include_dV_and_interp_weights &&
           (chunk->dV0 != 0.0 || chunk->dV1 != 0.0));
      const double stored_magnitude = std::abs(chunk->stored_weight);
      result.saw_nontrivial_stored_weight =
          result.saw_nontrivial_stored_weight ||
          (stored_magnitude > 0.0 &&
           std::abs(stored_magnitude - 1.0) > 1e-12);
    }
  };
  const auto append_monitor =
      [&](auto &monitor, const std::vector<dft_chunk *> &chunklists,
          const char *prefix) {
        std::vector<component> components;
        for (dft_chunk *chunks : chunklists) {
          inspect_chunks(chunks);
          for (dft_chunk *chunk = chunks; chunk;
               chunk = chunk->next_in_dft)
            if (std::find(components.begin(), components.end(), chunk->c) ==
                components.end())
              components.push_back(chunk->c);
        }
        require(!components.empty(),
                std::string(prefix) + " public monitor has no DFT chunks");
        for (component c : components)
          for (size_t frequency = 0; frequency < frequencies.size();
               ++frequency) {
            int rank = -1;
            size_t dims[3] = {0, 0, 0};
            std::unique_ptr<std::complex<realnum>[]> raw(
                f.get_dft_array(monitor, c,
                                static_cast<int>(frequency), &rank, dims));
            require(raw != nullptr,
                    std::string(prefix) + " get_dft_array returned null");
            size_t count = 1;
            for (int dim = 0; dim < rank; ++dim) {
              require(dims[dim] <=
                          std::numeric_limits<size_t>::max() / count,
                      std::string(prefix) + " array geometry overflow");
              count *= dims[dim];
            }
            std::vector<std::complex<double> > values(count);
            for (size_t point = 0; point < count; ++point) {
              values[point] = std::complex<double>(raw[point]);
              require(std::isfinite(values[point].real()) &&
                          std::isfinite(values[point].imag()),
                      std::string(prefix) +
                          " materialized a non-finite public value");
            }
            result.labels.push_back(
                std::string(prefix) + ":c=" +
                std::to_string(static_cast<int>(c)) + ":f=" +
                std::to_string(frequency) + ":rank=" +
                std::to_string(rank));
            result.arrays.push_back(std::move(values));
          }
      };

  append_monitor(plane, {plane.E, plane.H}, "flux-plane");
  append_monitor(box, {box.E, box.H}, "flux-closed-box");
  append_monitor(force, {force.offdiag1, force.offdiag2, force.diag},
                 "force");
  append_monitor(near, {near.F}, "near2far-closed-surface");
  result.statistics = gpu::get_dft_materialization_statistics();

  plane.remove();
  box.remove();
  force.remove();
  near.remove();
  return result;
}

void require_public_dft_array_materialization_equivalence() {
  require(count_processors() == 1,
          "public DFT array materialization regression is singleton-only");
  const public_dft_array_case_result cpu =
      run_public_dft_array_case(gpu::backend_mode::cpu);
  const public_dft_array_case_result cuda =
      run_public_dft_array_case(gpu::backend_mode::cuda);
  require(cpu.labels == cuda.labels &&
              cpu.arrays.size() == cuda.arrays.size(),
          "CPU/CUDA public DFT array cases differ in topology");
  for (size_t array = 0; array < cpu.arrays.size(); ++array)
    require_complex_vectors_close(
        cpu.arrays[array], cuda.arrays[array], cpu.labels[array],
        1.5e-3, 5e-6);
  require(cpu.saw_nonidentity_symmetry &&
              cuda.saw_nonidentity_symmetry,
          "public DFT regression did not exercise a transformed symmetry "
          "chunk");
  require(cpu.saw_nonzero_measure && cuda.saw_nonzero_measure,
          "public DFT regression did not exercise nonzero dV weights");
  require(cpu.saw_nontrivial_stored_weight &&
              cuda.saw_nontrivial_stored_weight,
          "public DFT regression did not exercise nontrivial stored weights");
  require(cuda.statistics.cuda_array_calls == cuda.arrays.size() &&
              cuda.statistics.cuda_array_points >
                  cuda.statistics.cuda_array_calls &&
              cuda.statistics.cpu_array_calls == 0 &&
              cuda.statistics.cuda_kernel_launches > 0 &&
              cuda.statistics.cuda_result_device_to_host_bytes > 0 &&
              cuda.statistics.full_dft_device_to_host_bytes_avoided > 0,
          "public flux/force/near2far arrays did not remain on strict CUDA "
          "materialization");
  if (am_master())
    std::cout << "PUBLIC_DFT_ARRAY_MATERIALIZATION arrays="
              << cuda.arrays.size()
              << " points=" << cuda.statistics.cuda_array_points
              << " cpu_fallbacks=" << cuda.statistics.cpu_array_calls
              << " kernels=" << cuda.statistics.cuda_kernel_launches
              << " symmetry=yes nonzero_dV=yes stored_weight=yes"
              << '\n';
}

void require_complex_vectors_close(
    const std::vector<std::complex<double> > &expected,
    const std::vector<std::complex<double> > &actual,
    const std::string &label, double relative = 1.2e-3,
    double absolute = 3.0e-6) {
  require(expected.size() == actual.size(), label + " size differs");
  double reference = 0.0;
  double error = 0.0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    reference = std::max(reference, std::abs(expected[index]));
    error = std::max(error, std::abs(expected[index] - actual[index]));
  }
  require(error <= absolute + relative * reference,
          label + " error=" + std::to_string(error) +
              " reference=" + std::to_string(reference));
}

struct dft_checkpoint_case_result {
  std::vector<double> baseline_flux;
  std::vector<double> scaled_flux;
  std::vector<std::complex<double> > baseline_array;
  std::vector<std::complex<double> > scaled_array;
  gpu::dft_checkpoint_statistics clean_save{};
  gpu::dft_checkpoint_statistics dirty_save{};
  gpu::dft_checkpoint_statistics load{};
  gpu::dft_scale_statistics scale{};
  gpu::dft_reduction_statistics reductions{};
  gpu::dft_materialization_statistics materializations{};
  gpu::dft_statistics step_dfts{};
  gpu::dft_statistics pre_save_dfts{};
  std::size_t expected_dirty_save_avoided_bytes = 0;
  std::size_t local_scalar_count = 0;
  std::size_t local_chunk_count = 0;
  std::size_t local_nonempty_chunk_count = 0;
  std::size_t attached_before_save = 0;
  std::size_t attached_after_save = 0;
};

dft_checkpoint_case_result run_dft_checkpoint_case(
    gpu::backend_mode mode, const std::string &tag) {
  gpu::set_backend(mode);
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  dft_checkpoint_case_result result;
  std::string clean_path;
  std::string checkpoint_path;
  {
    const grid_volume gv = vol2d(2.0, 1.6, 14.0);
    structure s(gv, vacuum, no_pml(), identity(), 4);
    fields f(&s, 0.0, 0.0, true, 64, 64);
    f.use_real_fields();
    gaussian_src_time source(0.31, 0.11);
    f.add_point_source(Ez, source, vec(0.63, 0.79), 0.9);
    const std::vector<double> frequencies = {0.0, 0.27, 0.39};
    dft_flux monitor = f.add_dft_flux(
        Y, volume(vec(0.38, 0.82), vec(1.62, 0.82)), frequencies,
        true, true, 1);
    result.local_scalar_count =
        dft_chunk_scalar_count(monitor.E) +
        dft_chunk_scalar_count(monitor.H);
    result.local_chunk_count =
        dft_chunk_count(monitor.E, false) +
        dft_chunk_count(monitor.H, false);
    result.local_nonempty_chunk_count =
        dft_chunk_count(monitor.E, true) +
        dft_chunk_count(monitor.H, true);
    require(result.local_scalar_count > 0 &&
                result.local_nonempty_chunk_count > 0,
            "checkpoint regression created no local monitor payload");
    result.attached_before_save = attached_dft_chunk_count(f);

    const std::string clean_name =
        "gpu-step-db-dft-checkpoint-clean-" + tag;
    gpu::reset_dispatch_statistics();
    monitor.save_hdf5(f, clean_name.c_str(), "clean");
    result.clean_save = gpu::get_dft_checkpoint_statistics();
    result.attached_after_save = attached_dft_chunk_count(f);
    clean_path = clean_name + ".h5";

    for (int step = 0; step < 32; ++step) f.step();
    result.pre_save_dfts = gpu::get_dft_statistics();

    if (mode == gpu::backend_mode::cuda) {
      std::vector<gpu::detail::resident_checkpoint_range_fp32> ranges;
      append_dft_checkpoint_ranges(monitor.E, ranges);
      append_dft_checkpoint_ranges(monitor.H, ranges);
      result.expected_dirty_save_avoided_bytes =
          gpu::detail::summarize_resident_checkpoint_ranges_fp32(
              ranges.data(), ranges.size())
              .full_cache_device_to_host_bytes_avoided;
    }

    const std::string checkpoint_name =
        "gpu-step-db-dft-checkpoint-dirty-" + tag;
    gpu::reset_dispatch_statistics();
    monitor.save_hdf5(f, checkpoint_name.c_str(), "state");
    result.dirty_save = gpu::get_dft_checkpoint_statistics();
    checkpoint_path = checkpoint_name + ".h5";

    monitor.scale_dfts(std::complex<double>(-0.21, 0.13));
    gpu::reset_dispatch_statistics();
    monitor.load_hdf5(f, checkpoint_name.c_str(), "state");
    result.load = gpu::get_dft_checkpoint_statistics();

    gpu::reset_dispatch_statistics();
    std::unique_ptr<double[]> restored_flux(monitor.flux());
    result.baseline_flux.assign(restored_flux.get(),
                                restored_flux.get() + 3);
    std::unique_ptr<double[]> repeated_flux(monitor.flux());
    for (std::size_t frequency = 0; frequency < 3; ++frequency)
      require(std::abs(repeated_flux[frequency] -
                       result.baseline_flux[frequency]) <=
                  1.0e-12 +
                      1.0e-9 * std::abs(result.baseline_flux[frequency]),
              "repeated checkpoint flux changed before scaling");
    result.baseline_array = capture_dft_array(f, monitor, monitor.cE, 1);

    const std::complex<double> scale(-0.375, 0.125);
    monitor.scale_dfts(scale);
    std::unique_ptr<double[]> scaled_flux(monitor.flux());
    result.scaled_flux.assign(scaled_flux.get(), scaled_flux.get() + 3);
    result.scaled_array = capture_dft_array(f, monitor, monitor.cE, 1);
    f.step();
    result.scale = gpu::get_dft_scale_statistics();
    result.reductions = gpu::get_dft_reduction_statistics();
    result.materializations =
        gpu::get_dft_materialization_statistics();
    result.step_dfts = gpu::get_dft_statistics();

    const double flux_scale = std::norm(scale);
    for (std::size_t frequency = 0; frequency < 3; ++frequency)
      require(std::abs(result.scaled_flux[frequency] -
                       flux_scale * result.baseline_flux[frequency]) <=
                  2.0e-6 + 1.5e-3 *
                                   std::abs(flux_scale *
                                            result.baseline_flux[frequency]),
              "checkpoint resident complex scale violated flux linearity");
    std::vector<std::complex<double> > expected_array =
        result.baseline_array;
    for (std::complex<double> &value : expected_array) value *= scale;
    require_complex_vectors_close(
        expected_array, result.scaled_array,
        "checkpoint resident complex array scale");
    monitor.remove();
  }
  if (!clean_path.empty()) unlink(clean_path.c_str());
  if (!checkpoint_path.empty()) unlink(checkpoint_path.c_str());
  gpu::set_backend(gpu::backend_mode::cpu);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "checkpoint regression leaked resident device buffers");
  return result;
}

void require_dft_checkpoint_residency_and_numerics() {
  require(count_processors() == 1,
          "focused checkpoint regression requires one rank");
  const dft_checkpoint_case_result cpu = run_dft_checkpoint_case(
      gpu::backend_mode::cpu, "cpu");
  const dft_checkpoint_case_result cuda = run_dft_checkpoint_case(
      gpu::backend_mode::cuda, "cuda");
  require(cpu.baseline_flux.size() == cuda.baseline_flux.size() &&
              cpu.scaled_flux.size() == cuda.scaled_flux.size(),
          "CPU/CUDA checkpoint flux result sizes differ");
  for (std::size_t frequency = 0;
       frequency < cpu.baseline_flux.size(); ++frequency) {
    const double reference = std::max(
        std::abs(cpu.baseline_flux[frequency]),
        std::abs(cuda.baseline_flux[frequency]));
    require(std::abs(cpu.baseline_flux[frequency] -
                     cuda.baseline_flux[frequency]) <=
                3.0e-6 + 1.5e-3 * reference,
            "CPU/CUDA loaded checkpoint flux differs");
    const double scaled_reference = std::max(
        std::abs(cpu.scaled_flux[frequency]),
        std::abs(cuda.scaled_flux[frequency]));
    require(std::abs(cpu.scaled_flux[frequency] -
                     cuda.scaled_flux[frequency]) <=
                3.0e-6 + 1.5e-3 * scaled_reference,
            "CPU/CUDA scaled checkpoint flux differs");
  }
  require_complex_vectors_close(
      cpu.baseline_array, cuda.baseline_array,
      "CPU/CUDA loaded checkpoint DFT array");
  require_complex_vectors_close(
      cpu.scaled_array, cuda.scaled_array,
      "CPU/CUDA resident-scaled checkpoint DFT array");

  require(cpu.clean_save.cpu_save_dataset_calls == 2 &&
              cpu.clean_save.cpu_save_values == cpu.local_scalar_count &&
              cpu.clean_save.cuda_save_dataset_calls == 0 &&
              cpu.load.cpu_load_dataset_calls == 2 &&
              cpu.load.cpu_load_values == cpu.local_scalar_count &&
              cpu.scale.cpu_scale_calls == cpu.local_chunk_count &&
              cpu.scale.cpu_scale_values == cpu.local_scalar_count / 2 &&
              cpu.scale.cuda_scale_calls == 0,
          "CPU checkpoint/scale telemetry is not exact");
  require(cuda.clean_save.cuda_save_dataset_calls == 2 &&
              cuda.clean_save.cuda_save_values ==
                  cuda.local_scalar_count &&
              cuda.clean_save.cuda_save_device_to_host_bytes == 0 &&
              cuda.clean_save.cpu_save_dataset_calls == 0,
          "clean pre-step CUDA checkpoint was not a zero-transfer logical "
          "save");
  require(cuda.dirty_save.cuda_save_dataset_calls == 2 &&
              cuda.dirty_save.cuda_save_values ==
                  cuda.local_scalar_count &&
              cuda.dirty_save.cuda_save_device_to_host_bytes ==
                  cuda.local_scalar_count * sizeof(realnum) &&
              cuda.dirty_save
                      .cuda_save_full_cache_device_to_host_bytes_avoided >
                  0 &&
              cuda.dirty_save
                      .cuda_save_full_cache_device_to_host_bytes_avoided ==
                  cuda.expected_dirty_save_avoided_bytes &&
              cuda.dirty_save.cpu_save_dataset_calls == 0,
          "dirty CUDA checkpoint did not stage exactly its DFT payload: "
          "calls=" +
              std::to_string(cuda.dirty_save.cuda_save_dataset_calls) +
              " values=" +
              std::to_string(cuda.dirty_save.cuda_save_values) +
              " expected-values=" +
              std::to_string(cuda.local_scalar_count) + " d2h=" +
              std::to_string(
                  cuda.dirty_save.cuda_save_device_to_host_bytes) +
              " expected-d2h=" +
              std::to_string(cuda.local_scalar_count * sizeof(realnum)) +
              " avoided=" +
              std::to_string(
                  cuda.dirty_save
                      .cuda_save_full_cache_device_to_host_bytes_avoided) +
              " cpu-calls=" +
              std::to_string(
                  cuda.dirty_save.cpu_save_dataset_calls) +
              " pre-save-cuda-dft-calls=" +
              std::to_string(cuda.pre_save_dfts.cuda_update_calls) +
              " pre-save-cpu-dft-calls=" +
              std::to_string(cuda.pre_save_dfts.cpu_update_calls) +
              " attached-before=" +
              std::to_string(cuda.attached_before_save) +
              " attached-after=" +
              std::to_string(cuda.attached_after_save));
  require(cuda.load.cuda_load_dataset_calls == 2 &&
              cuda.load.cuda_load_values == cuda.local_scalar_count &&
              cuda.load.cuda_load_host_to_device_bytes ==
                  cuda.local_scalar_count * sizeof(realnum) &&
              cuda.load.cpu_load_dataset_calls == 0,
          "CUDA checkpoint load H2D telemetry is not exact");
  require(cuda.scale.cuda_scale_calls == cuda.local_chunk_count &&
              cuda.scale.cuda_scale_values ==
                  cuda.local_scalar_count / 2 &&
              cuda.scale.cuda_scale_kernel_launches ==
                  cuda.local_nonempty_chunk_count &&
              cuda.scale.cuda_scale_host_to_device_bytes == 0 &&
              cuda.scale.cpu_scale_calls == 0,
          "load-minus resident scale discarded or re-uploaded its DFT");
  require(cuda.reductions.cuda_descriptor_uploads == 2 &&
              cuda.reductions.cuda_plan_reuses >= 1 &&
              cuda.reductions.cpu_reduction_calls == 0,
          "checkpoint load/scale did not invalidate and rebuild the CUDA "
          "DFT reduction plan");
  require(cuda.materializations.cuda_array_calls == 2 &&
              cuda.materializations.cuda_kernel_launches == 2 &&
              cuda.materializations.cpu_array_calls == 0,
          "resident scale reused a stale DFT materialization plan");
  require(cuda.step_dfts.cuda_update_calls > 0 &&
              cuda.step_dfts.cpu_update_calls == 0,
          "the first post-load-minus timestep fell back to CPU DFT update");
}

void require_zero_value_dft_checkpoint() {
  require(count_processors() == 1,
          "zero-value checkpoint regression requires one rank");
  gpu::set_backend(gpu::backend_mode::cuda);
  const grid_volume gv = vol2d(1.0, 1.0, 8.0);
  structure s(gv, vacuum, no_pml());
  fields f(&s);
  const char *name = "gpu-step-db-dft-checkpoint-zero";
  std::string path;
  gpu::reset_dispatch_statistics();
  {
    std::unique_ptr<h5file> file(
        f.open_h5file(name, h5file::WRITE));
    path = file->file_name();
    save_dft_hdf5(nullptr, "zero", file.get(), nullptr, true);
  }
  {
    std::unique_ptr<h5file> file(
        f.open_h5file(name, h5file::READONLY));
    load_dft_hdf5(nullptr, "zero", file.get(), nullptr, true);
  }
  const gpu::dft_checkpoint_statistics statistics =
      gpu::get_dft_checkpoint_statistics();
  require(statistics.cuda_save_dataset_calls == 1 &&
              statistics.cuda_save_values == 0 &&
              statistics.cuda_save_device_to_host_bytes == 0 &&
              statistics.cuda_load_dataset_calls == 1 &&
              statistics.cuda_load_values == 0 &&
              statistics.cuda_load_host_to_device_bytes == 0 &&
              statistics.cpu_save_dataset_calls == 0 &&
              statistics.cpu_load_dataset_calls == 0,
          "zero-value CUDA checkpoint was not a successful logical no-op");
  if (!path.empty()) unlink(path.c_str());
  gpu::set_backend(gpu::backend_mode::cpu);
}

void require_dft_checkpoint_failure_atomicity() {
  require(count_processors() == 1,
          "checkpoint failure-atomic regression requires one rank");
  gpu::set_backend(gpu::backend_mode::cuda);
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  std::string path;
  {
    const grid_volume gv = vol2d(2.2, 1.8, 14.0);
    structure s(gv, vacuum, no_pml(), identity(), 8);
    fields f(&s, 0.0, 0.0, true, 64, 64);
    f.use_real_fields();
    gaussian_src_time source(0.29, 0.10);
    f.add_point_source(Ez, source, vec(0.71, 0.87), 0.8);
    const std::vector<double> frequencies = {0.21, 0.29, 0.38};
    dft_flux monitor = f.add_dft_flux(
        Y, volume(vec(0.32, 0.91), vec(1.88, 0.91)), frequencies,
        true, true, 1);
    for (int step = 0; step < 24; ++step) f.step();
    require(dft_chunk_count(monitor.E, true) >= 2,
            "checkpoint atomicity fixture needs multiple E chunks");
    require(dft_chunk_count(monitor.H, true) >= 1,
            "checkpoint atomicity fixture needs an H chunk");
    const std::vector<std::complex<double> > checkpoint_e_array =
        capture_dft_array(f, monitor, monitor.cE, 1);
    const std::vector<std::complex<double> > checkpoint_h_array =
        capture_dft_array(f, monitor, monitor.cH, 1);

    const char *name = "gpu-step-db-dft-checkpoint-atomic";
    path = std::string(name) + ".h5";
    monitor.save_hdf5(f, name, "atomic");
    monitor.scale_dfts(std::complex<double>(-0.43, 0.27));

    // All E uploads succeed, then the first H upload fails after its mirror
    // has been allocated/registered but before the copy.  This proves both
    // public E/H transaction atomicity and rollback of a partially prepared
    // resident mirror.
    gpu::detail::set_dft_checkpoint_h2d_failure_after_for_testing(
        static_cast<std::int64_t>(dft_chunk_count(monitor.E, true)));
    bool h2d_failure_observed = false;
    try {
      monitor.load_hdf5(f, name, "atomic");
    }
    catch (const std::runtime_error &error) {
      h2d_failure_observed =
          std::string(error.what()).find(
              "injected CUDA DFT checkpoint H2D failure") !=
          std::string::npos;
    }
    gpu::detail::set_dft_checkpoint_h2d_failure_after_for_testing(-1);
    require(h2d_failure_observed,
            "later-dataset checkpoint H2D failure was not injected");
    const std::vector<std::complex<double> > after_failed_e_load =
        capture_dft_array(f, monitor, monitor.cE, 1);
    const std::vector<std::complex<double> > after_failed_h_load =
        capture_dft_array(f, monitor, monitor.cH, 1);
    require_complex_vectors_close(
        checkpoint_e_array, after_failed_e_load,
        "failure-atomic E checkpoint host/resident state", 1.0e-6,
        1.0e-7);
    require_complex_vectors_close(
        checkpoint_h_array, after_failed_h_load,
        "failure-atomic H checkpoint host/resident state", 1.0e-6,
        1.0e-7);

    // A dirty save must fail before modifying authority.  Retrying the same
    // operation after clearing the deterministic hook must write the exact
    // checkpoint and leave all resident buffers live.
    f.step();
    gpu::detail::set_dft_checkpoint_d2h_failure_after_for_testing(1);
    bool d2h_failure_observed = false;
    try {
      monitor.save_hdf5(f, name, "retry");
    }
    catch (const std::runtime_error &error) {
      d2h_failure_observed =
          std::string(error.what()).find(
              "injected CUDA DFT checkpoint D2H failure") !=
          std::string::npos;
    }
    gpu::detail::set_dft_checkpoint_d2h_failure_after_for_testing(-1);
    require(d2h_failure_observed,
            "dirty-save checkpoint D2H failure was not injected");
    monitor.save_hdf5(f, name, "retry");
    monitor.remove();
  }
  if (!path.empty()) unlink(path.c_str());
  gpu::set_backend(gpu::backend_mode::cpu);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "checkpoint failure regression leaked resident buffers");
}

void trigger_distributed_dft_checkpoint_rank_failure(bool load_failure) {
  require(count_processors() == 2,
          "distributed checkpoint failure probe requires two ranks");
  gpu::set_backend(gpu::backend_mode::cuda);
  const grid_volume gv = vol2d(2.0, 1.6, 12.0);
  structure s(gv, vacuum, no_pml(), identity(), 8);
  fields f(&s, 0.0, 0.0, true, 64, 64);
  f.use_real_fields();
  gaussian_src_time source(0.30, 0.10);
  f.add_point_source(Ez, source, vec(0.63, 0.79), 0.8);
  component components[] = {Ez};
  dft_fields monitor = f.add_dft_fields(
      components, 1, f.v, 0.22, 0.38, 3, true, 1);
  for (int step = 0; step < 8; ++step) f.step();
  const char *name = "gpu-step-db-dft-checkpoint-rank-failure";
  if (load_failure) {
    {
      std::unique_ptr<h5file> file(
          f.open_h5file(name, h5file::WRITE));
      save_dft_hdf5(monitor.chunks, "fields", file.get(), nullptr, true);
    }
    monitor.scale_dfts(std::complex<double>(-0.25, 0.125));
    if (my_rank() == 0)
      gpu::detail::set_dft_checkpoint_h2d_failure_after_for_testing(0);
    std::unique_ptr<h5file> file(
        f.open_h5file(name, h5file::READONLY));
    load_dft_hdf5(monitor.chunks, "fields", file.get(), nullptr, true);
  }
  else {
    if (my_rank() == 0)
      gpu::detail::set_dft_checkpoint_d2h_failure_after_for_testing(0);
    std::unique_ptr<h5file> file(
        f.open_h5file(name, h5file::WRITE));
    save_dft_hdf5(monitor.chunks, "fields", file.get(), nullptr, true);
  }
  meep::abort(
      "distributed DFT checkpoint rank failure did not abort the "
      "communicator");
}

void require_distributed_zero_work_dft_checkpoint() {
  require(count_processors() == 2,
          "distributed zero-work checkpoint requires two ranks");
  gpu::set_backend(gpu::backend_mode::cuda);
  const grid_volume gv = vol2d(2.4, 1.6, 12.0);
  structure s(gv, vacuum, no_pml(), identity(), 8);
  fields f(&s, 0.0, 0.0, true, 64, 64);
  f.use_real_fields();
  gaussian_src_time source(0.30, 0.10);
  f.add_point_source(Ez, source, vec(0.31, 0.79), 0.8);
  component components[] = {Ez};
  const volume point(vec(0.31, 0.79), vec(0.31, 0.79));
  dft_fields monitor = f.add_dft_fields(
      components, 1, point, 0.22, 0.38, 3, true, 1);
  for (int step = 0; step < 8; ++step) f.step();
  const std::size_t local_values =
      dft_chunk_scalar_count(monitor.chunks);
  require(or_to_all(local_values == 0) && or_to_all(local_values > 0),
          "point checkpoint did not create one zero-work and one owner "
          "rank");
  const char *name = "gpu-step-db-dft-checkpoint-zero-rank";
  std::string path;
  gpu::reset_dispatch_statistics();
  {
    std::unique_ptr<h5file> file(
        f.open_h5file(name, h5file::WRITE));
    path = file->file_name();
    save_dft_hdf5(monitor.chunks, "fields", file.get(), nullptr, true);
  }
  {
    std::unique_ptr<h5file> file(
        f.open_h5file(name, h5file::READONLY));
    load_dft_hdf5(monitor.chunks, "fields", file.get(), nullptr, true);
  }
  const gpu::dft_checkpoint_statistics statistics =
      gpu::get_dft_checkpoint_statistics();
  require(and_to_all(
              statistics.cuda_save_dataset_calls == 1 &&
              statistics.cuda_save_values == local_values &&
              statistics.cuda_save_device_to_host_bytes ==
                  local_values * sizeof(realnum) &&
              statistics.cuda_load_dataset_calls == 1 &&
              statistics.cuda_load_values == local_values &&
              statistics.cuda_load_host_to_device_bytes ==
                  local_values * sizeof(realnum) &&
              statistics.cpu_save_dataset_calls == 0 &&
              statistics.cpu_load_dataset_calls == 0),
          "distributed empty-rank checkpoint telemetry is not exact");
  gpu::reset_dispatch_statistics();
  monitor.scale_dfts(std::complex<double>(-0.25, 0.125));
  const gpu::dft_scale_statistics scale_statistics =
      gpu::get_dft_scale_statistics();
  const std::size_t local_chunks = dft_chunk_count(monitor.chunks, false);
  const std::size_t local_nonempty_chunks =
      dft_chunk_count(monitor.chunks, true);
  require(and_to_all(
              scale_statistics.cpu_scale_calls == 0 &&
              scale_statistics.cpu_scale_values == 0 &&
              scale_statistics.cuda_scale_calls == local_chunks &&
              scale_statistics.cuda_scale_values == local_values / 2 &&
              scale_statistics.cuda_scale_kernel_launches ==
                  local_nonempty_chunks &&
              scale_statistics.cuda_scale_host_to_device_bytes == 0),
          "distributed empty-rank resident DFT scale telemetry is not "
          "exact");
  monitor.remove();
  all_wait();
  if (am_master() && !path.empty()) unlink(path.c_str());
  all_wait();
  gpu::set_backend(gpu::backend_mode::cpu);
}

dft_array_materialization_result run_dft_array_materialization_case(
    gpu::backend_mode mode) {
  gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();
  const grid_volume gv = vol2d(1.6, 1.4, 12.0);
  structure s(gv, vacuum, no_pml(), identity(), 4);
  fields f(&s, 0.0, 0.0, true, 32, 32);
  f.use_real_fields();
  gaussian_src_time source(0.29, 0.10);
  f.add_point_source(Ez, source, vec(0.61, 0.73), 0.8);
  component components[] = {Ez};
  const volume point_monitor(vec(0.91, 0.67), vec(0.91, 0.67));
  constexpr int frequency_count = 7;
  dft_fields monitor = f.add_dft_fields(
      components, 1, point_monitor, 0.18, 0.42, frequency_count,
      true, 1);
  for (int step = 0; step < 28; ++step) f.step();

  dft_array_materialization_result result;
  const auto capture_spectrum = [&]() {
    for (int frequency = 0; frequency < frequency_count; ++frequency) {
      int rank = -1;
      size_t dims[3] = {0, 0, 0};
      std::unique_ptr<std::complex<realnum>[]> values(
          f.get_dft_array(monitor, Ez, frequency, &rank, dims));
      require(values != nullptr,
              "point get_dft_array returned an empty result");
      require(rank == 0,
              "point get_dft_array did not collapse to a scalar rank");
      result.values.push_back(std::complex<double>(values[0]));
    }
  };
  capture_spectrum();
  capture_spectrum();
  f.step();
  capture_spectrum();
  // A positive real scale changes the resident content generation while
  // preserving the exact +0 bit pattern of the cylindrical-axis oracle.
  monitor.scale_dfts(std::complex<double>(2.0, 0.0));
  capture_spectrum();
  result.statistics = gpu::get_dft_materialization_statistics();
  monitor.remove();
  return result;
}

void require_low_level_dft_materialization_cache_invalidation() {
  gpu::set_backend(gpu::backend_mode::cuda);
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();

  // Preparing the second source mirror must not make the first dependency's
  // cache-wide allocation generation stale within the same request batch.
  int shared_owner = 0;
  const float shared_first[] = {1.0f, -0.25f};
  const float shared_second[] = {-0.5f, 0.75f};
  const std::ptrdiff_t shared_first_destination[] = {0};
  const std::ptrdiff_t shared_second_destination[] = {1};
  const float shared_weight[] = {1.0f};
  const gpu::detail::dft_materialization_request_fp32 shared_requests[] = {
      {&shared_owner, shared_first, shared_first_destination, shared_weight,
       1, 1, 1, 0, 1, 1.0f, 0.0f, nullptr},
      {&shared_owner, shared_second, shared_second_destination, shared_weight,
       1, 1, 1, 0, 1, 1.0f, 0.0f, nullptr}};
  float shared_output[4] = {};
  gpu::reset_dispatch_statistics();
  gpu::detail::resident_materialize_dft_array_fp32(
      shared_requests, 2, shared_output, 2, 1);
  gpu::detail::resident_materialize_dft_array_fp32(
      shared_requests, 2, shared_output, 2, 1);
  require(gpu::get_dft_materialization_statistics()
                  .cuda_kernel_launches == 1,
          "same-owner multi-mirror DFT topology missed its second call");
  gpu::detail::destroy_resident_cache_for_owner(&shared_owner);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "same-owner multi-mirror DFT test leaked device buffers");

  // Materialization topology identity is bit-exact for FP32 metadata. Signed
  // zero is observable after multiplication, and NaN payloads must not force
  // perpetual cache misses merely because operator== rejects every NaN.
  int signed_zero_owner = 0;
  const float signed_zero_dft[] = {1.0f, 0.0f};
  const std::ptrdiff_t signed_zero_destination[] = {0};
  const float plus_zero_weight[] = {0.0f};
  const float minus_zero_weight[] = {-0.0f};
  gpu::detail::dft_materialization_request_fp32 signed_zero_request = {
      &signed_zero_owner, signed_zero_dft, signed_zero_destination,
      plus_zero_weight, 1, 1, 1, 0, 1, 1.0f, 0.0f, nullptr};
  float signed_zero_output[2] = {};
  gpu::reset_dispatch_statistics();
  gpu::detail::resident_materialize_dft_array_fp32(
      &signed_zero_request, 1, signed_zero_output, 1, 1);
  signed_zero_request.point_weights = minus_zero_weight;
  gpu::detail::resident_materialize_dft_array_fp32(
      &signed_zero_request, 1, signed_zero_output, 1, 1);
  require(gpu::get_dft_materialization_statistics()
                  .cuda_kernel_launches == 2 &&
              std::signbit(signed_zero_output[0]),
          "DFT materialization topology treated signed-zero point weights "
          "as identical");

  const float unit_weight[] = {1.0f};
  signed_zero_request.point_weights = unit_weight;
  signed_zero_request.inverse_stored_weight_real = 0.0f;
  gpu::reset_dispatch_statistics();
  gpu::detail::resident_materialize_dft_array_fp32(
      &signed_zero_request, 1, signed_zero_output, 1, 1);
  signed_zero_request.inverse_stored_weight_real = -0.0f;
  gpu::detail::resident_materialize_dft_array_fp32(
      &signed_zero_request, 1, signed_zero_output, 1, 1);
  require(gpu::get_dft_materialization_statistics()
                  .cuda_kernel_launches == 2,
          "DFT materialization topology treated signed-zero inverse "
          "weights as identical");
  gpu::detail::destroy_resident_cache_for_owner(&signed_zero_owner);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "signed-zero DFT materialization test leaked device buffers");

  // A malformed fresh request, including a malformed later member of a
  // batch, must be rejected before creating either owner's resident cache.
  int malformed_first_owner = 0;
  int malformed_second_owner = 0;
  const float malformed_first_dft[] = {1.0f, 0.0f};
  const float malformed_second_dft[] = {2.0f, 0.0f};
  const std::ptrdiff_t valid_destination[] = {0};
  const std::ptrdiff_t invalid_destination[] = {1};
  const float valid_weight[] = {1.0f};
  const gpu::detail::dft_materialization_request_fp32 malformed_batch[] = {
      {&malformed_first_owner, malformed_first_dft, valid_destination,
       valid_weight, 1, 1, 1, 0, 1, 1.0f, 0.0f, nullptr},
      {&malformed_second_owner, malformed_second_dft,
       invalid_destination, valid_weight,
       1, 1, 1, 0, 1, 1.0f, 0.0f, nullptr}};
  bool malformed_batch_rejected = false;
  float malformed_output[2] = {};
  try {
    gpu::detail::resident_materialize_dft_array_fp32(
        malformed_batch, 2, malformed_output, 1, 1);
  }
  catch (const std::invalid_argument &) {
    malformed_batch_rejected = true;
  }
  const auto malformed_first_snapshot =
      gpu::detail::get_resident_cache_snapshot_for_testing(
          &malformed_first_owner);
  const auto malformed_second_snapshot =
      gpu::detail::get_resident_cache_snapshot_for_testing(
          &malformed_second_owner);
  require(malformed_batch_rejected &&
              !malformed_first_snapshot.exists &&
              !malformed_second_snapshot.exists &&
              gpu::get_live_resident_device_buffers() == live_before,
          "malformed DFT materialization batch mutated a fresh owner cache");

  // A later owner which is already inside a resident phase must also be
  // rejected during the read-only batch preflight.  In particular, the
  // valid fresh owner which precedes it must not acquire a cache or mirror.
  int phase_fresh_owner = 0;
  int phase_active_owner = 0;
  const float phase_fresh_dft[] = {3.0f, -0.5f};
  const float phase_active_dft[] = {-2.0f, 0.25f};
  const gpu::detail::dft_materialization_request_fp32 phase_batch[] = {
      {&phase_fresh_owner, phase_fresh_dft, valid_destination,
       valid_weight, 1, 1, 1, 0, 1, 1.0f, 0.0f, nullptr},
      {&phase_active_owner, phase_active_dft, valid_destination,
       valid_weight, 1, 1, 1, 0, 1, 1.0f, 0.0f, nullptr}};
  {
    gpu::detail::resident_curl_session active_session(
        &phase_active_owner, true);
    const auto active_before =
        gpu::detail::get_resident_cache_snapshot_for_testing(
            &phase_active_owner);
    const std::uint64_t phase_live_before =
        gpu::get_live_resident_device_buffers();
    gpu::reset_dispatch_statistics();
    bool phase_batch_rejected = false;
    try {
      gpu::detail::resident_materialize_dft_array_fp32(
          phase_batch, 2, malformed_output, 1, 1);
    }
    catch (const std::logic_error &) { phase_batch_rejected = true; }
    const auto fresh_after =
        gpu::detail::get_resident_cache_snapshot_for_testing(
            &phase_fresh_owner);
    const auto active_after =
        gpu::detail::get_resident_cache_snapshot_for_testing(
            &phase_active_owner);
    const gpu::dft_materialization_statistics phase_statistics =
        gpu::get_dft_materialization_statistics();
    require(phase_batch_rejected && !fresh_after.exists &&
                active_before.exists && active_before.phase_active &&
                active_after.exists && active_after.phase_active &&
                active_after.device_ordinal == active_before.device_ordinal &&
                active_after.epoch == active_before.epoch &&
                active_after.mirror_count == active_before.mirror_count &&
                gpu::get_live_resident_device_buffers() ==
                    phase_live_before &&
                phase_statistics.cuda_kernel_launches == 0 &&
                phase_statistics.cuda_result_device_to_host_bytes == 0,
            "active-phase DFT materialization preflight mutated an earlier "
            "fresh owner or the active owner");
    active_session.finish(false);
  }
  gpu::detail::destroy_resident_cache_for_owner(&phase_active_owner);
  require(!gpu::detail::get_resident_cache_snapshot_for_testing(
               &phase_fresh_owner).exists &&
              gpu::get_live_resident_device_buffers() == live_before,
          "active-phase DFT materialization preflight leaked a cache");

  // Once a source is device-authoritative, a conflicting extent or an
  // interior/straddling alias must not free it, upload stale host bytes, or
  // advance its cache epoch. A valid retry proves the retained content.
  int range_owner = 0;
  std::vector<float> range_dft = {
      1.0f, -0.5f, 2.0f, 0.25f, -1.5f, 0.75f,
      0.5f, 1.25f, -0.75f, -0.25f, 1.5f, -1.0f};
  const gpu::detail::dft_materialization_request_fp32 range_request = {
      &range_owner, range_dft.data(), valid_destination, valid_weight,
      6, 1, 1, 0, 1, 1.0f, 0.0f, nullptr};
  float range_output[2] = {};
  gpu::detail::resident_materialize_dft_array_fp32(
      &range_request, 1, range_output, 1, 1);
  (void)gpu::detail::scale_resident_complex_dft_fp32_for_owner(
      &range_owner, range_dft.data(), 6, 2.0, 0.0);
  gpu::detail::resident_materialize_dft_array_fp32(
      &range_request, 1, range_output, 1, 1);
  require(range_output[0] == 2.0f && range_output[1] == -1.0f,
          "range preflight fixture did not retain device-authoritative DFT "
          "content");
  const std::uint64_t range_live =
      gpu::get_live_resident_device_buffers();
  const auto range_snapshot =
      gpu::detail::get_resident_cache_snapshot_for_testing(&range_owner);
  const auto require_range_rejected =
      [&](const float *source, std::size_t storage_points,
          const char *message) {
        gpu::detail::dft_materialization_request_fp32 invalid =
            range_request;
        invalid.dft_real_imag = source;
        invalid.storage_point_count = storage_points;
        bool rejected = false;
        try {
          gpu::detail::resident_materialize_dft_array_fp32(
              &invalid, 1, range_output, 1, 1);
        }
        catch (const std::invalid_argument &) { rejected = true; }
        const auto after =
            gpu::detail::get_resident_cache_snapshot_for_testing(
                &range_owner);
        require(rejected && range_snapshot.exists && after.exists &&
                    range_snapshot.epoch == after.epoch &&
                    range_snapshot.mirror_count == after.mirror_count &&
                    !after.phase_active &&
                    gpu::get_live_resident_device_buffers() == range_live,
                message);
        gpu::detail::resident_materialize_dft_array_fp32(
            &range_request, 1, range_output, 1, 1);
        require(range_output[0] == 2.0f && range_output[1] == -1.0f,
                "rejected DFT source range changed retained content");
      };
  require_range_rejected(
      range_dft.data(), 7,
      "DFT materialization accepted a conflicting complete-source extent");
  require_range_rejected(
      range_dft.data() + 2, 1,
      "DFT materialization accepted an interior resident source range");
  require_range_rejected(
      range_dft.data() + 11, 1,
      "DFT materialization accepted a straddling resident source range");
  gpu::detail::destroy_resident_cache_for_owner(&range_owner);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "DFT materialization range-preflight test leaked device buffers");

  // An explicit metadata flag represents a zero cylindrical measure even in
  // fast-math builds.  Exactly-zero DFT data must remain zero, not 0/0 NaN.
  int axis_owner = 0;
  const float axis_dft[] = {0.0f, 0.0f};
  const std::ptrdiff_t axis_destination[] = {0};
  const float axis_weight[] = {1.0f};
  const std::uint8_t axis_zero_divisor[] = {1u};
  const gpu::detail::dft_materialization_request_fp32 axis_request = {
      &axis_owner, axis_dft, axis_destination, axis_weight,
      1, 1, 1, 0, 1, 1.0f, 0.0f, axis_zero_divisor};
  float axis_output[2] = {1.0f, 1.0f};
  gpu::detail::resident_materialize_dft_array_fp32(
      &axis_request, 1, axis_output, 1, 1);
  require(axis_output[0] == 0.0f && axis_output[1] == 0.0f &&
              std::isfinite(axis_output[0]) &&
              std::isfinite(axis_output[1]),
          "zero cylindrical-axis DFT sample materialized as Inf/NaN");
  gpu::detail::destroy_resident_cache_for_owner(&axis_owner);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "cylindrical-axis DFT test leaked device buffers");

  int first_owner = 0;
  int second_owner = 0;
  std::vector<float> first = {
      1.0f, -0.5f, 2.0f, 0.25f, -1.5f, 0.75f,
      0.5f, 1.25f, -0.75f, -0.25f, 1.5f, -1.0f};
  std::vector<float> second = {
      -2.0f, 0.5f, 0.75f, -1.25f, 1.0f, 2.0f};
  const std::ptrdiff_t first_destinations[] = {0, 1};
  const std::ptrdiff_t second_destinations[] = {2};
  const float first_weights[] = {1.0f, 0.5f};
  const float second_weights[] = {-2.0f};

  const auto capture = [&](std::vector<std::complex<double> > *observed) {
    for (std::size_t frequency = 0; frequency < 3; ++frequency) {
      const gpu::detail::dft_materialization_request_fp32 requests[] = {
          {&first_owner, first.data(), first_destinations, first_weights,
           2, 2, 3, frequency, 1, 0.75f, -0.25f, nullptr},
          {&second_owner, second.data(), second_destinations,
           second_weights, 1, 1, 3, frequency, 1, -0.5f, 0.125f,
           nullptr}};
      float output[6] = {};
      gpu::detail::resident_materialize_dft_array_fp32(
          requests, 2, output, 3, 1);
      for (std::size_t point = 0; point < 3; ++point)
        observed->push_back(std::complex<double>(
            output[2 * point], output[2 * point + 1]));
    }
  };
  const auto expected = [&]() {
    std::vector<std::complex<double> > values;
    for (std::size_t frequency = 0; frequency < 3; ++frequency) {
      for (std::size_t point = 0; point < 2; ++point) {
        const std::size_t index = 2 * (point * 3 + frequency);
        values.push_back(
            std::complex<double>(first[index], first[index + 1]) *
            std::complex<double>(0.75, -0.25) *
            static_cast<double>(first_weights[point]));
      }
      const std::size_t index = 2 * frequency;
      values.push_back(
          std::complex<double>(second[index], second[index + 1]) *
          std::complex<double>(-0.5, 0.125) *
          static_cast<double>(second_weights[0]));
    }
    return values;
  };
  const auto require_matches = [&](const std::vector<std::complex<double> > &a,
                                   const std::vector<std::complex<double> > &b,
                                   const char *message) {
    require(a.size() == b.size(), message);
    for (std::size_t index = 0; index < a.size(); ++index)
      require(std::abs(a[index] - b[index]) <=
                  2e-6 * std::max(1.0, std::abs(b[index])),
              message);
  };

  gpu::reset_dispatch_statistics();
  std::vector<std::complex<double> > first_capture;
  capture(&first_capture);
  capture(&first_capture);
  require_matches(
      std::vector<std::complex<double> >(
          first_capture.begin(), first_capture.begin() + 9),
      expected(), "all-frequency DFT cache changed the first result");
  require_matches(
      std::vector<std::complex<double> >(
          first_capture.begin() + 9, first_capture.end()),
      expected(), "all-frequency DFT cache changed a reused result");
  require(
      gpu::get_dft_materialization_statistics()
              .cuda_kernel_launches == 1,
      "three-frequency DFT cache did not reduce six calls to one kernel");

  gpu::detail::discard_resident_mirror_for_owner(
      &second_owner, second.data());
  second[0] += 0.625f;
  std::vector<std::complex<double> > after_discard;
  capture(&after_discard);
  require_matches(after_discard, expected(),
                  "discarded DFT mirror reused stale cached content");
  require(gpu::get_dft_materialization_statistics()
                  .cuda_kernel_launches == 2,
          "DFT mirror discard did not invalidate the all-frequency cache");

  gpu::detail::destroy_resident_cache_for_owner(&second_owner);
  second[2] -= 0.375f;
  std::vector<std::complex<double> > after_owner_aba;
  capture(&after_owner_aba);
  require_matches(after_owner_aba, expected(),
                  "owner-cache ABA reused stale DFT materialization");
  require(gpu::get_dft_materialization_statistics()
                  .cuda_kernel_launches == 3,
          "owner destruction did not invalidate a dependent DFT cache");

  gpu::detail::destroy_resident_cache_for_owner(&second_owner);
  gpu::detail::destroy_resident_cache_for_owner(&first_owner);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "DFT materialization cache invalidation leaked device buffers");
}

void require_backend_dft_output_staging_contract() {
  require(count_processors() == 1 || count_processors() == 2,
          "focused DFT output staging contract requires one or two ranks");
  gpu::set_backend(gpu::backend_mode::cuda);
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  const std::uint64_t pinned_before =
      gpu::detail::get_live_dft_output_pinned_buffers_for_testing();

  int first_owner = 0;
  int second_owner = 0;
  std::vector<float> first = {
      1.0f, 0.5f, 2.0f, -0.5f, 3.0f, 1.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      // Spare host allocation tail makes the straddling-alias regression a
      // valid host range while the resident source remains exactly 12 floats.
      0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> second = {
      -1.0f, 0.25f, -2.0f, 0.5f, -3.0f, 0.75f};
  const std::ptrdiff_t first_destinations[] = {1, 0};
  const std::ptrdiff_t second_destinations[] = {0};
  const float first_weights[] = {0.5f, 1.0f};
  const float second_weights[] = {-2.0f};
  const std::uint8_t first_zero_divisors[] = {0u, 1u};
  const gpu::detail::dft_output_staging_request_fp32 requests[] = {
      {&first_owner, first.data(), first_destinations, first_weights,
       first_zero_divisors, 2, 2, 3, 0.75f, -0.25f, 0},
      {&second_owner, second.data(), second_destinations, second_weights,
       nullptr, 1, 1, 3, -0.5f, 0.125f, 2}};

  // Make both complete source allocations device-authoritative before the
  // first stage. This proves that output staging returns only its bounded
  // planar tile and accounts the full DFT D2H traffic it avoids.
  (void)gpu::detail::scale_resident_complex_dft_fp32_for_owner(
      &first_owner, first.data(), 6, 1.0, 0.0);
  (void)gpu::detail::scale_resident_complex_dft_fp32_for_owner(
      &second_owner, second.data(), 3, 1.0, 0.0);
  gpu::reset_dispatch_statistics();

  const auto expected = [&](std::size_t source_frequency,
                            double first_scale) {
    std::vector<std::complex<double> > packed(3);
    const std::size_t first_index = 2 * source_frequency;
    packed[1] =
        std::complex<double>(first[first_index],
                             first[first_index + 1]) *
        std::complex<double>(0.75, -0.25) * 0.5 * first_scale;
    packed[0] = std::complex<double>(0.0, 0.0);
    const std::size_t second_index = 2 * source_frequency;
    packed[2] =
        std::complex<double>(second[second_index],
                             second[second_index + 1]) *
        std::complex<double>(-0.5, 0.125) * -2.0;
    return packed;
  };
  const auto require_tile = [&](
      const gpu::detail::resident_dft_output_staging_view_fp32 &view,
      std::size_t start, std::size_t count, double first_scale,
      const char *message) {
    require(view.planar && view.output_point_count == 3 &&
                view.frequency_start == start &&
                view.frequency_count == count &&
                view.frequency_capacity == 2,
            message);
    for (std::size_t tile_frequency = 0;
         tile_frequency < count; ++tile_frequency) {
      const auto reference = expected(start + tile_frequency, first_scale);
      const std::size_t plane = tile_frequency * 2 * 3;
      for (std::size_t point = 0; point < 3; ++point) {
        const std::complex<double> observed(
            view.planar[plane + point],
            view.planar[plane + 3 + point]);
        require(std::abs(observed - reference[point]) <=
                    3e-6 * std::max(1.0, std::abs(reference[point])),
                message);
      }
    }
  };

  gpu::detail::resident_dft_output_staging_view_fp32 first_view =
      gpu::detail::resident_stage_dft_output_fp32(
          requests, 2, 3, 0, 2, 2);
  require_tile(first_view, 0, 2, 1.0,
               "first resident DFT output tile disagrees");
  const std::uint64_t live_after_first =
      gpu::get_live_resident_device_buffers();
  require(live_after_first == live_before + 4,
          "first DFT output stage did not retain exactly two source and "
          "two plan device buffers");

  // Reuse the exact retained topology but request a different frequency
  // window, so the private D2H scratch receives observably different bytes.
  // Failure after that transfer and before publication must leave every byte
  // of the already-returned view unchanged.
  gpu::detail::set_dft_output_staging_d2h_failure_after_for_testing(0);
  bool reuse_publication_failure_observed = false;
  try {
    (void)gpu::detail::resident_stage_dft_output_fp32(
        requests, 2, 3, 1, 2, 2);
  }
  catch (const std::runtime_error &error) {
    reuse_publication_failure_observed =
        std::string(error.what()).find(
            "injected CUDA DFT output staging D2H publication failure") !=
        std::string::npos;
  }
  gpu::detail::set_dft_output_staging_d2h_failure_after_for_testing(-1);
  require(reuse_publication_failure_observed &&
              gpu::get_live_resident_device_buffers() == live_after_first,
          "failed reused DFT output publication changed buffer inventory");
  require_tile(first_view, 0, 2, 1.0,
               "failed reused DFT output publication modified prior view");

  // Force a different-capacity replacement through allocation, metadata H2D,
  // and kernel launch, then fail its bounded D2H. The old plan/view must stay
  // committed and every replacement allocation must unwind.
  gpu::detail::set_dft_output_staging_d2h_failure_after_for_testing(0);
  bool replacement_failure_observed = false;
  try {
    (void)gpu::detail::resident_stage_dft_output_fp32(
        requests, 2, 3, 0, 2, 3);
  }
  catch (const std::runtime_error &error) {
    replacement_failure_observed =
        std::string(error.what()).find(
            "injected CUDA DFT output staging D2H publication failure") !=
        std::string::npos;
  }
  gpu::detail::set_dft_output_staging_d2h_failure_after_for_testing(-1);
  require(replacement_failure_observed &&
              gpu::get_live_resident_device_buffers() == live_after_first,
          "failed DFT output plan replacement leaked or replaced buffers");
  require_tile(first_view, 0, 2, 1.0,
               "failed DFT output replacement invalidated prior view");

  bool rejected_empty = false;
  try {
    (void)gpu::detail::resident_stage_dft_output_fp32(
        nullptr, 0, 3, 0, 1, 2);
  }
  catch (const std::invalid_argument &) { rejected_empty = true; }
  require(rejected_empty,
          "DFT output staging accepted an empty local request list");

  // A fresh malformed owner must not acquire even an empty resident cache.
  // This catches cache creation and epoch advancement that live-device-buffer
  // telemetry alone cannot observe.
  int malformed_fresh_owner = 0;
  const std::ptrdiff_t invalid_destination[] = {1};
  const float valid_weight[] = {1.0f};
  const gpu::detail::dft_output_staging_request_fp32 malformed_fresh = {
      &malformed_fresh_owner, first.data(), invalid_destination,
      valid_weight, nullptr, 1, 1, 3, 1.0f, 0.0f, 0};
  const auto fresh_before =
      gpu::detail::get_resident_cache_snapshot_for_testing(
          &malformed_fresh_owner);
  bool rejected_fresh_metadata = false;
  try {
    (void)gpu::detail::resident_stage_dft_output_fp32(
        &malformed_fresh, 1, 1, 0, 1, 1);
  }
  catch (const std::invalid_argument &) {
    rejected_fresh_metadata = true;
  }
  const auto fresh_after =
      gpu::detail::get_resident_cache_snapshot_for_testing(
          &malformed_fresh_owner);
  require(rejected_fresh_metadata && !fresh_before.exists &&
              !fresh_after.exists,
          "malformed DFT output metadata created a resident cache");

  // A conflicting extent for an existing device-authoritative mirror must be
  // rejected before ensure_resident_mirror can free it. The retained plan,
  // returned pinned view, and exact resident-buffer inventory stay unchanged.
  gpu::detail::dft_output_staging_request_fp32 extent_mismatch[] = {
      requests[0], requests[1]};
  extent_mismatch[0].storage_point_count = 3;
  bool rejected_extent = false;
  try {
    (void)gpu::detail::resident_stage_dft_output_fp32(
        extent_mismatch, 2, 3, 0, 2, 2);
  }
  catch (const std::invalid_argument &) { rejected_extent = true; }
  require(rejected_extent &&
              gpu::get_live_resident_device_buffers() == live_after_first,
          "conflicting DFT output source extent changed authoritative "
          "resident state");
  require_tile(first_view, 0, 2, 1.0,
               "rejected source extent invalidated the prior pinned view");

  // An exact base-address lookup is insufficient: an interior or straddling
  // subrange would otherwise create a second mirror whose H2D upload can
  // overwrite device-authoritative DFT state with stale host bytes. Both
  // shapes pass the ordinary descriptor checks and must be rejected against
  // the existing complete first-source mirror before a phase begins.
  const std::ptrdiff_t overlap_destination[] = {0};
  const float overlap_weight[] = {1.0f};
  const auto require_overlap_rejected = [&](const float *source,
                                             const char *message) {
    const gpu::detail::dft_output_staging_request_fp32 overlap_request = {
        &first_owner, source, overlap_destination, overlap_weight, nullptr,
        1, 1, 3, 1.0f, 0.0f, 0};
    const auto before =
        gpu::detail::get_resident_cache_snapshot_for_testing(&first_owner);
    bool rejected = false;
    try {
      (void)gpu::detail::resident_stage_dft_output_fp32(
          &overlap_request, 1, 1, 0, 1, 1);
    }
    catch (const std::invalid_argument &) { rejected = true; }
    const auto after =
        gpu::detail::get_resident_cache_snapshot_for_testing(&first_owner);
    require(rejected &&
                gpu::get_live_resident_device_buffers() == live_after_first &&
                before.exists && after.exists &&
                before.epoch == after.epoch &&
                before.mirror_count == after.mirror_count &&
                !after.phase_active,
            message);
    require_tile(first_view, 0, 2, 1.0,
                 "rejected overlapping source invalidated prior view");
  };
  require_overlap_rejected(
      first.data() + 2,
      "DFT output staging accepted an interior resident source range");
  require_overlap_rejected(
      first.data() + 10,
      "DFT output staging accepted a straddling resident source range");

  const std::ptrdiff_t duplicate_destinations[] = {0, 0};
  gpu::detail::dft_output_staging_request_fp32 malformed[] = {
      requests[0], requests[1]};
  malformed[0].destination_indices = duplicate_destinations;
  const auto malformed_existing_before =
      gpu::detail::get_resident_cache_snapshot_for_testing(&first_owner);
  bool rejected_malformed = false;
  try {
    (void)gpu::detail::resident_stage_dft_output_fp32(
        malformed, 2, 3, 2, 1, 2);
  }
  catch (const std::invalid_argument &) { rejected_malformed = true; }
  const auto malformed_existing_after =
      gpu::detail::get_resident_cache_snapshot_for_testing(&first_owner);
  require(rejected_malformed &&
              gpu::get_live_resident_device_buffers() == live_after_first &&
              malformed_existing_before.exists &&
              malformed_existing_after.exists &&
              malformed_existing_before.epoch ==
                  malformed_existing_after.epoch &&
              malformed_existing_before.mirror_count ==
                  malformed_existing_after.mirror_count &&
              !malformed_existing_after.phase_active,
          "rejected DFT output metadata changed the retained plan");

  gpu::detail::resident_dft_output_staging_view_fp32 short_view =
      gpu::detail::resident_stage_dft_output_fp32(
          requests, 2, 3, 2, 1, 2);
  require_tile(short_view, 2, 1, 1.0,
               "short reused resident DFT output tile disagrees");

  // Removing a non-owner dependency must invalidate the plan retained by
  // first_owner. Recreating the same host identity then exercises the cache
  // allocation generation's ABA defense.
  gpu::detail::destroy_resident_cache_for_owner(&second_owner);
  require(short_view.lifetime && short_view.planar,
          "dependency destruction released a live pinned staging view");
  require_tile(short_view, 2, 1, 1.0,
               "dependency destruction invalidated a live pinned view");
  require_tile(gpu::detail::resident_stage_dft_output_fp32(
                   requests, 2, 3, 0, 2, 2),
               0, 2, 1.0,
               "cross-cache ABA changed resident DFT output staging");

  // A device-side scientific mutation changes the dependency content
  // generation and explicitly invalidates every retained consumer plan.
  (void)gpu::detail::scale_resident_complex_dft_fp32_for_owner(
      &first_owner, first.data(), 6, 2.0, 0.0);
  require_tile(gpu::detail::resident_stage_dft_output_fp32(
                   requests, 2, 3, 0, 2, 2),
               0, 2, 2.0,
               "content-generation refresh reused stale DFT output");

  const gpu::dft_materialization_statistics statistics =
      gpu::get_dft_materialization_statistics();
  const auto align_up = [](std::size_t value, std::size_t alignment) {
    const std::size_t remainder = value % alignment;
    return remainder ? value + alignment - remainder : value;
  };
  const std::size_t expected_destination_bytes =
      3 * sizeof(std::ptrdiff_t);
  const std::size_t expected_weight_offset =
      align_up(expected_destination_bytes, alignof(float));
  const std::size_t expected_zero_offset =
      expected_weight_offset + 3 * sizeof(float);
  const std::size_t expected_descriptor_offset = align_up(
      expected_zero_offset + 3 * sizeof(std::uint8_t),
      alignof(meep_cuda::dft_output_staging_operation_fp32));
  const std::size_t expected_block_map_offset = align_up(
      expected_descriptor_offset +
          2 * sizeof(meep_cuda::dft_output_staging_operation_fp32),
      alignof(std::uint32_t));
  const std::size_t expected_metadata_bytes =
      expected_block_map_offset + 2 * sizeof(std::uint32_t);
  const std::uint64_t expected_bulk_workspace =
      expected_metadata_bytes + 3 * (3 * 2 * 2 * sizeof(float));
  require(statistics.cpu_output_calls == 0 &&
              statistics.cpu_output_points == 0 &&
              statistics.cuda_output_staging_calls == 4 &&
              statistics.cuda_output_staging_points == 12 &&
              statistics.cuda_output_staging_frequencies == 7 &&
              statistics.cuda_output_staging_descriptor_uploads == 3 &&
              statistics.cuda_output_staging_plan_reuses == 1 &&
              statistics.cuda_output_staging_kernel_launches == 4 &&
              statistics.cuda_output_staging_result_device_to_host_bytes ==
                  168 &&
              statistics
                      .cuda_output_staging_full_dft_device_to_host_bytes_avoided ==
                  240 &&
              statistics.cuda_output_staging_workspace_ceiling_bytes ==
                  expected_bulk_workspace,
          "resident DFT output staging telemetry is inconsistent");

  // Topology identity is bit-exact for FP32 metadata. In particular, changing
  // +0 to -0 in a point weight must upload a new descriptor plan instead of
  // silently reusing metadata with different signed-zero semantics.
  const float plus_zero_weights[] = {0.0f, 1.0f};
  const float minus_zero_weights[] = {-0.0f, 1.0f};
  gpu::detail::dft_output_staging_request_fp32 plus_zero_requests[] = {
      requests[0], requests[1]};
  gpu::detail::dft_output_staging_request_fp32 minus_zero_requests[] = {
      requests[0], requests[1]};
  plus_zero_requests[0].point_weights = plus_zero_weights;
  minus_zero_requests[0].point_weights = minus_zero_weights;
  gpu::reset_dispatch_statistics();
  gpu::detail::resident_dft_output_staging_view_fp32 plus_zero_view =
      gpu::detail::resident_stage_dft_output_fp32(
          plus_zero_requests, 2, 3, 0, 1, 2);
  gpu::detail::resident_dft_output_staging_view_fp32 minus_zero_view =
      gpu::detail::resident_stage_dft_output_fp32(
          minus_zero_requests, 2, 3, 0, 1, 2);
  const gpu::dft_materialization_statistics signed_zero_statistics =
      gpu::get_dft_materialization_statistics();
  require(plus_zero_view.lifetime && minus_zero_view.lifetime &&
              signed_zero_statistics.cuda_output_staging_calls == 2 &&
              signed_zero_statistics
                      .cuda_output_staging_descriptor_uploads == 2 &&
              signed_zero_statistics.cuda_output_staging_plan_reuses == 0,
          "DFT output topology treated signed-zero metadata as identical");

  // The inverse stored weight is descriptor metadata too. Its signed-zero
  // bits must participate in topology identity independently of point
  // weights.
  gpu::detail::dft_output_staging_request_fp32 plus_zero_inverse[] = {
      requests[0], requests[1]};
  gpu::detail::dft_output_staging_request_fp32 minus_zero_inverse[] = {
      requests[0], requests[1]};
  plus_zero_inverse[0].inverse_stored_weight_real = 0.0f;
  minus_zero_inverse[0].inverse_stored_weight_real = -0.0f;
  gpu::reset_dispatch_statistics();
  auto plus_zero_inverse_view =
      gpu::detail::resident_stage_dft_output_fp32(
          plus_zero_inverse, 2, 3, 0, 1, 2);
  auto minus_zero_inverse_view =
      gpu::detail::resident_stage_dft_output_fp32(
          minus_zero_inverse, 2, 3, 0, 1, 2);
  const gpu::dft_materialization_statistics inverse_zero_statistics =
      gpu::get_dft_materialization_statistics();
  require(plus_zero_inverse_view.lifetime &&
              minus_zero_inverse_view.lifetime &&
              inverse_zero_statistics.cuda_output_staging_calls == 2 &&
              inverse_zero_statistics
                      .cuda_output_staging_descriptor_uploads == 2 &&
              inverse_zero_statistics.cuda_output_staging_plan_reuses == 0,
          "DFT output topology treated signed-zero inverse weights as "
          "identical");

  // Exercise the ordinary DFT-update path, which advances only the source
  // content generation once all update inputs are resident. The second stage
  // must reuse the exact descriptor plan while launching against the new
  // generation; scale_resident_complex_dft cannot prove this because it
  // deliberately invalidates every dependent plan.
  float zero_update_field[] = {0.0f};
  const std::ptrdiff_t zero_update_indices[] = {0, 0};
  const float zero_update_weights[] = {0.0f, 0.0f};
  const double update_frequencies[] = {0.2, 0.3, 0.4};
  const auto perform_zero_dft_update = [&]() {
    gpu::detail::resident_curl_session session(&first_owner, true);
    require(session.active() && session.cache(),
            "normal DFT update could not open its resident phase");
    gpu::detail::resident_update_dft_fp32(
        session.cache(), first.data(), zero_update_field, nullptr, 1,
        zero_update_indices, zero_update_weights, 2,
        update_frequencies, 3, 0.0, 0.0, 0.0, 0, 0);
    session.finish(false);
  };
  perform_zero_dft_update();  // prepares stable update-input mirrors
  gpu::reset_dispatch_statistics();
  require_tile(gpu::detail::resident_stage_dft_output_fp32(
                   requests, 2, 3, 0, 2, 2),
               0, 2, 2.0,
               "prepared normal DFT update changed staged output");
  perform_zero_dft_update();  // content generation only
  require_tile(gpu::detail::resident_stage_dft_output_fp32(
                   requests, 2, 3, 0, 2, 2),
               0, 2, 2.0,
               "normal DFT content-generation refresh returned stale data");
  const gpu::dft_materialization_statistics content_statistics =
      gpu::get_dft_materialization_statistics();
  require(content_statistics.cuda_output_staging_calls == 2 &&
              content_statistics.cuda_output_staging_descriptor_uploads ==
                  1 &&
              content_statistics.cuda_output_staging_plan_reuses == 1 &&
              content_statistics.cuda_output_staging_kernel_launches == 2,
          "normal DFT content generation did not reuse the staging plan");

  require(gpu::detail::get_live_dft_output_pinned_buffers_for_testing() >
              pinned_before,
          "DFT output views did not retain their pinned publication storage");
  first_view.lifetime.reset();
  short_view.lifetime.reset();
  plus_zero_view.lifetime.reset();
  minus_zero_view.lifetime.reset();
  plus_zero_inverse_view.lifetime.reset();
  minus_zero_inverse_view.lifetime.reset();
  gpu::detail::destroy_resident_cache_for_owner(&second_owner);
  gpu::detail::destroy_resident_cache_for_owner(&first_owner);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "resident DFT output staging leaked device buffers");
  require(gpu::detail::get_live_dft_output_pinned_buffers_for_testing() ==
              pinned_before,
          "resident DFT output staging leaked pinned host buffers");
  if (am_master())
    std::cout
        << "PASS: resident DFT output planar staging, pinned bounded D2H, "
           "plan reuse, validation atomicity, cross-cache ABA, content "
           "generation, telemetry, and lifecycle\n";
}

void require_public_dft_output_failure_atomicity() {
  require(count_processors() == 1,
          "public DFT output failure-atomic regression is singleton-only");
  gpu::set_backend(gpu::backend_mode::cuda);
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  const std::uint64_t pinned_before =
      gpu::detail::get_live_dft_output_pinned_buffers_for_testing();

  {
  const grid_volume gv = vol2d(1.6, 1.4, 12.0);
  const volume monitor_volume(vec(0.10, 0.10), vec(0.38, 0.34));
  const std::vector<double> frequencies = {0.23, 0.29, 0.35};
  component components[] = {Ez};
  structure s(gv, vacuum, no_pml(), identity(), 4);
  fields f(&s, 0.0, 0.0, true, 32, 32);
  f.use_real_fields();
  gaussian_src_time source(0.29, 0.10);
  f.add_point_source(Ez, source, vec(0.24, 0.22), 0.8);
  dft_fields monitor = f.add_dft_fields(
      components, 1, monitor_volume, frequencies, true, 1);
  for (int step = 0; step < 32; ++step) f.step();

  int reference_rank = -1;
  size_t reference_dims[3] = {0, 0, 0};
  size_t reference_point_count = 0;
  std::vector<std::vector<std::complex<double> > > reference;
  for (size_t frequency = 0; frequency < frequencies.size(); ++frequency) {
    int rank = -1;
    size_t dims[3] = {0, 0, 0};
    std::unique_ptr<std::complex<realnum>[]> raw(
        f.get_dft_array(monitor, Ez, static_cast<int>(frequency),
                        &rank, dims));
    require(raw != nullptr && rank > 0,
            "failure-atomic DFT output oracle was empty or scalar");
    size_t count = 1;
    for (int dim = 0; dim < rank; ++dim) {
      require(dims[dim] <=
                  std::numeric_limits<size_t>::max() / count,
              "failure-atomic DFT output oracle dimensions overflow");
      count *= dims[dim];
    }
    if (frequency == 0) {
      reference_rank = rank;
      std::copy(dims, dims + 3, reference_dims);
      reference_point_count = count;
    }
    else {
      require(rank == reference_rank && count == reference_point_count,
              "failure-atomic DFT output oracle geometry changed");
      for (int dim = 0; dim < rank; ++dim)
        require(dims[dim] == reference_dims[dim],
                "failure-atomic DFT output oracle dimension changed");
    }
    std::vector<std::complex<double> > values(count);
    for (size_t point = 0; point < count; ++point)
      values[point] = std::complex<double>(raw[point]);
    reference.push_back(std::move(values));
  }
  require(reference_point_count > 1,
          "failure-atomic DFT output oracle is degenerate");

  const char *output_name =
      "gpu-step-db-public-dft-output-failure-atomic";
  const std::string output_path = f.h5file_name(output_name);
  const std::string temporary_name =
      std::string(output_name) + ".gpmeep-tmp";
  const std::string temporary_path =
      f.h5file_name(temporary_name.c_str());
  unlink(output_path.c_str());
  unlink(temporary_path.c_str());

  const std::string sentinel =
      "gpmeep-existing-final-must-survive-partial-output\n";
  {
    std::ofstream output(
        output_path, std::ios::binary | std::ios::trunc);
    require(output.good(),
            "could not create failure-atomic DFT output sentinel");
    output.write(sentinel.data(),
                 static_cast<std::streamsize>(sentinel.size()));
    output.close();
    require(output.good(),
            "could not close failure-atomic DFT output sentinel");
  }
  struct stat sentinel_status;
  require(stat(output_path.c_str(), &sentinel_status) == 0 &&
              S_ISREG(sentinel_status.st_mode),
          "could not stat failure-atomic DFT output sentinel");
  const auto read_bytes = [](const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.good())
      throw std::runtime_error(
          "could not read failure-atomic DFT output artifact");
    std::ostringstream bytes;
    bytes << input.rdbuf();
    if (!input.good() && !input.eof())
      throw std::runtime_error(
          "failed while reading failure-atomic DFT output artifact");
    return bytes.str();
  };
  require(read_bytes(output_path) == sentinel,
          "failure-atomic DFT output sentinel was not written exactly");

  gpu::reset_dispatch_statistics();
  gpu::detail::fail_next_cuda_dft_output_after_dataset_for_testing();
  bool injected_failure_observed = false;
  try {
    f.output_dft(monitor, output_name);
  }
  catch (const std::runtime_error &error) {
    injected_failure_observed =
        std::string(error.what()).find(
            "injected CUDA DFT output failure after an HDF5 dataset") !=
        std::string::npos;
  }
  // Clear a stale hook if an unrelated exception prevented its consumption;
  // the exact error assertion below will still reject that path.
  (void)gpu::detail::
      consume_cuda_dft_output_dataset_failure_for_testing();
  struct stat partial_status;
  const bool partial_exists =
      stat(temporary_path.c_str(), &partial_status) == 0 &&
      partial_status.st_size > 0;
  struct stat preserved_status;
  const bool final_identity_preserved =
      stat(output_path.c_str(), &preserved_status) == 0 &&
      preserved_status.st_dev == sentinel_status.st_dev &&
      preserved_status.st_ino == sentinel_status.st_ino &&
      preserved_status.st_size == sentinel_status.st_size;
  const gpu::dft_materialization_statistics failed_statistics =
      gpu::get_dft_materialization_statistics();
  require(injected_failure_observed && partial_exists &&
              final_identity_preserved &&
              read_bytes(output_path) == sentinel &&
              failed_statistics.cuda_output_calls == 0 &&
              failed_statistics.cpu_output_calls == 0 &&
              failed_statistics.cuda_output_points == 0 &&
              failed_statistics.cpu_output_points == 0 &&
              failed_statistics.cpu_array_calls == 0 &&
              failed_statistics.cuda_array_calls == 0 &&
              failed_statistics.cuda_output_staging_calls == 1 &&
              failed_statistics.cuda_output_staging_points ==
                  reference_point_count &&
              failed_statistics.cuda_output_staging_frequencies ==
                  frequencies.size() &&
              failed_statistics
                      .cuda_output_staging_descriptor_uploads == 1 &&
              failed_statistics.cuda_output_staging_plan_reuses == 0 &&
              failed_statistics.cuda_output_staging_kernel_launches == 1 &&
              failed_statistics
                      .cuda_output_staging_result_device_to_host_bytes ==
                  2 * reference_point_count * frequencies.size() *
                      sizeof(float) &&
              failed_statistics
                      .cuda_output_staging_full_dft_device_to_host_bytes_avoided ==
                  2 * reference_point_count * frequencies.size() *
                      sizeof(float) &&
              failed_statistics.cuda_output_staging_workspace_ceiling_bytes >
                  3 * failed_statistics
                          .cuda_output_staging_result_device_to_host_bytes &&
              failed_statistics.cuda_output_staging_workspace_ceiling_bytes <=
                  3 * failed_statistics
                          .cuda_output_staging_result_device_to_host_bytes +
                      4096 &&
              failed_statistics.array_mpi_allreduce_calls == 0 &&
              failed_statistics.array_mpi_allreduce_bytes == 0,
          "failed CUDA output_dft replaced its final file, lost its partial "
          "temporary, or committed success telemetry");

  // The injected point is exactly after ez_0.r closes. Prove the temporary
  // is a deterministic one-dataset HDF5 prefix with the correct payload,
  // rather than merely a nonempty scratch file.
  {
    h5file partial(temporary_path.c_str(), h5file::READONLY,
                   false /* parallel */, true /* local */);
    require(partial.dataset_exists("ez_0.r") &&
                !partial.dataset_exists("ez_0.i") &&
                !partial.dataset_exists("ez_1.r") &&
                !partial.dataset_exists("ez_1.i") &&
                !partial.dataset_exists("ez_2.r") &&
                !partial.dataset_exists("ez_2.i"),
            "failed CUDA output_dft temporary has the wrong dataset prefix");
    int partial_rank = -1;
    size_t partial_dims[3] = {0, 0, 0};
    std::unique_ptr<float[]> partial_payload(static_cast<float *>(
        partial.read("ez_0.r", &partial_rank, partial_dims, 3, true)));
    require(partial_payload && partial_rank == reference_rank,
            "failed CUDA output_dft temporary rank is invalid");
    for (int dim = 0; dim < reference_rank; ++dim)
      require(partial_dims[dim] == reference_dims[dim],
              "failed CUDA output_dft temporary dimensions are invalid");
    for (size_t point = 0; point < reference_point_count; ++point)
      require(partial_payload[point] ==
                  static_cast<float>(reference[0][point].real()),
              "failed CUDA output_dft temporary payload is invalid");
  }

  // Poison the completed prefix with a dataset which is not part of the
  // public inventory.  The retry must open its first dataset with WRITE,
  // truncate this stale file, and therefore remove the marker.
  {
    h5file stale(temporary_path.c_str(), h5file::READWRITE,
                 false /* parallel */, true /* local */);
    const size_t marker_dims[1] = {1};
    float marker = 42.25f;
    stale.write("gpmeep_stale_marker", 1, marker_dims, &marker, true);
  }

  // The next ordinary call must truncate the stale partial file, publish a
  // complete replacement, and reuse the resident staging topology.
  gpu::reset_dispatch_statistics();
  f.output_dft(monitor, output_name);
  const gpu::dft_materialization_statistics recovered_statistics =
      gpu::get_dft_materialization_statistics();
  struct stat recovered_status;
  const bool final_was_replaced =
      stat(output_path.c_str(), &recovered_status) == 0 &&
      S_ISREG(recovered_status.st_mode) &&
      (recovered_status.st_dev != sentinel_status.st_dev ||
       recovered_status.st_ino != sentinel_status.st_ino);
  require(access(temporary_path.c_str(), F_OK) != 0 &&
              final_was_replaced && read_bytes(output_path) != sentinel &&
              recovered_statistics.cuda_output_calls == 1 &&
              recovered_statistics.cuda_output_points ==
                  reference_point_count * frequencies.size() &&
              recovered_statistics.cpu_output_calls == 0 &&
              recovered_statistics.cpu_output_points == 0 &&
              recovered_statistics.cuda_output_staging_calls == 1 &&
              recovered_statistics.cuda_output_staging_points ==
                  reference_point_count &&
              recovered_statistics.cuda_output_staging_frequencies ==
                  frequencies.size() &&
              recovered_statistics.cuda_output_staging_kernel_launches == 1 &&
              recovered_statistics.cuda_output_staging_descriptor_uploads ==
                  0 &&
              recovered_statistics.cuda_output_staging_plan_reuses == 1 &&
              recovered_statistics
                      .cuda_output_staging_result_device_to_host_bytes ==
                  2 * reference_point_count * frequencies.size() *
                      sizeof(float) &&
              recovered_statistics
                      .cuda_output_staging_full_dft_device_to_host_bytes_avoided ==
                  2 * reference_point_count * frequencies.size() *
                      sizeof(float) &&
              recovered_statistics
                      .cuda_output_staging_workspace_ceiling_bytes ==
                  failed_statistics
                      .cuda_output_staging_workspace_ceiling_bytes &&
              recovered_statistics.cpu_array_calls == 0 &&
              recovered_statistics.cuda_array_calls == 0 &&
              recovered_statistics.array_mpi_allreduce_calls == 0 &&
              recovered_statistics.array_mpi_allreduce_bytes == 0,
          "CUDA output_dft did not recover by atomically replacing the stale "
          "partial artifact");

  std::unique_ptr<h5file> output(new h5file(
      output_path.c_str(), h5file::READONLY,
      false /* parallel */, true /* local */));
  require(!output->dataset_exists("gpmeep_stale_marker"),
          "recovered failure-atomic DFT output retained a stale dataset");
  double maximum_error = 0.0;
  for (size_t frequency = 0; frequency < frequencies.size(); ++frequency) {
    int real_rank = -1;
    int imaginary_rank = -1;
    size_t real_dims[3] = {0, 0, 0};
    size_t imaginary_dims[3] = {0, 0, 0};
    const std::string real_name =
        "ez_" + std::to_string(frequency) + ".r";
    const std::string imaginary_name =
        "ez_" + std::to_string(frequency) + ".i";
    std::unique_ptr<float[]> real_payload(static_cast<float *>(
        output->read(real_name.c_str(), &real_rank, real_dims, 3, true)));
    std::unique_ptr<float[]> imaginary_payload(static_cast<float *>(
        output->read(imaginary_name.c_str(), &imaginary_rank,
                     imaginary_dims, 3, true)));
    require(real_payload && imaginary_payload &&
                real_rank == reference_rank &&
                imaginary_rank == reference_rank,
            "recovered failure-atomic DFT output rank is invalid");
    for (int dim = 0; dim < reference_rank; ++dim)
      require(real_dims[dim] == reference_dims[dim] &&
                  imaginary_dims[dim] == reference_dims[dim],
              "recovered failure-atomic DFT output dimensions are invalid");
    for (size_t point = 0; point < reference_point_count; ++point) {
      const std::complex<double> observed(
          real_payload[point], imaginary_payload[point]);
      maximum_error = std::max(
          maximum_error,
          std::abs(reference[frequency][point] - observed));
    }
  }
  output.reset();
  require(maximum_error <= 2e-5,
          "recovered failure-atomic DFT output payload changed");

  const bool final_removed = unlink(output_path.c_str()) == 0;
  errno = 0;
  const bool temporary_absent =
      unlink(temporary_path.c_str()) != 0 && errno == ENOENT;
  require(final_removed && temporary_absent,
          "failure-atomic DFT output cleanup was incomplete");
  monitor.remove();
  if (am_master())
    std::cout
        << "PASS: public CUDA output_dft preserves an existing final after "
           "partial HDF5 failure, then truncates stale temp and atomically "
           "recovers; datasets=6 max_abs="
        << maximum_error << '\n';
  }
  gpu::set_backend(gpu::backend_mode::cpu);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "failure-atomic public DFT output leaked resident device buffers");
  require(gpu::detail::get_live_dft_output_pinned_buffers_for_testing() ==
              pinned_before,
          "failure-atomic public DFT output leaked pinned host buffers");
}

void require_public_dft_output_mpi_rename_failure_atomicity() {
  require(count_processors() == 2,
          "MPI DFT output rename-failure regression requires exactly two "
          "ranks");
  gpu::set_backend(gpu::backend_mode::cuda);
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  const std::uint64_t pinned_before =
      gpu::detail::get_live_dft_output_pinned_buffers_for_testing();
  size_t reported_point_count = 0;

  {
    const grid_volume gv = vol2d(1.6, 1.4, 12.0);
    const volume monitor_volume(vec(0.10, 0.10), vec(0.38, 0.34));
    const std::vector<double> frequencies = {0.23, 0.29, 0.35};
    component components[] = {Ez};
    structure s(gv, vacuum, no_pml(), identity(), 4);
    fields f(&s, 0.0, 0.0, true, 32, 32);
    f.use_real_fields();
    gaussian_src_time source(0.29, 0.10);
    f.add_point_source(Ez, source, vec(0.24, 0.22), 0.8);
    dft_fields monitor = f.add_dft_fields(
        components, 1, monitor_volume, frequencies, true, 1);
    for (int step = 0; step < 32; ++step) f.step();

    size_t local_monitor_points = 0;
    size_t local_monitor_chunks = 0;
    for (dft_chunk *chunk = monitor.chunks; chunk;
         chunk = chunk->next_in_dft)
      if (chunk->c == Ez) {
        ++local_monitor_chunks;
        local_monitor_points += chunk->N;
      }
    const bool owns_monitor = local_monitor_points > 0;
    const size_t owner_ranks =
        sum_to_all(static_cast<size_t>(owns_monitor));
    const size_t global_chunks = sum_to_all(local_monitor_chunks);
    require(owner_ranks == 1 && global_chunks >= 1,
            "MPI rename-failure monitor must have exactly one nonempty "
            "owner");

    // Capture the same snapshot through the independent public array
    // materialization path.  The output staging telemetry is reset below, so
    // these collectives do not weaken the owner/zero-work publication gates.
    std::vector<float> array_oracle;
    size_t oracle_point_count = 0;
    int oracle_rank = -1;
    size_t oracle_dims[3] = {0, 0, 0};
    bool array_oracle_ok = true;
    for (size_t frequency = 0; frequency < frequencies.size(); ++frequency) {
      int rank = -1;
      size_t dims[3] = {0, 0, 0};
      std::unique_ptr<std::complex<realnum>[]> raw(
          f.get_dft_array(monitor, Ez, static_cast<int>(frequency),
                          &rank, dims));
      size_t count = 1;
      bool geometry_ok = raw != nullptr && rank > 0 && rank <= 3;
      if (geometry_ok)
        for (int dim = 0; dim < rank; ++dim) {
          if (dims[dim] > std::numeric_limits<size_t>::max() / count) {
            geometry_ok = false;
            break;
          }
          count *= dims[dim];
        }
      if (!geometry_ok) {
        array_oracle_ok = false;
        continue;
      }
      if (frequency == 0) {
        oracle_rank = rank;
        std::copy(dims, dims + 3, oracle_dims);
        oracle_point_count = count;
      }
      else {
        array_oracle_ok =
            array_oracle_ok && rank == oracle_rank &&
            count == oracle_point_count;
        for (int dim = 0; dim < rank; ++dim)
          array_oracle_ok =
              array_oracle_ok && dims[dim] == oracle_dims[dim];
      }
      for (size_t point = 0; point < count; ++point)
        array_oracle.push_back(raw[point].real());
      for (size_t point = 0; point < count; ++point)
        array_oracle.push_back(raw[point].imag());
    }
    double oracle_maximum = 0.0;
    for (float value : array_oracle) {
      array_oracle_ok = array_oracle_ok && std::isfinite(value);
      oracle_maximum =
          std::max(oracle_maximum, std::abs(static_cast<double>(value)));
    }
    array_oracle_ok =
        array_oracle_ok && oracle_rank == 2 && oracle_dims[0] == 6 &&
        oracle_dims[1] == 5 && oracle_point_count == 30 &&
        array_oracle.size() ==
            2 * frequencies.size() * oracle_point_count &&
        oracle_maximum > 1e-8;
    const size_t valid_oracle_ranks =
        sum_to_all(static_cast<size_t>(array_oracle_ok));
    require(valid_oracle_ranks == static_cast<size_t>(count_processors()),
            "MPI rename-failure array oracle is invalid or rank-skewed");

    const char *output_name =
        "gpu-step-db-public-dft-output-mpi-rename-failure";
    const std::string output_path = f.h5file_name(output_name);
    const std::string temporary_name =
        std::string(output_name) + ".gpmeep-tmp";
    const std::string temporary_path =
        f.h5file_name(temporary_name.c_str());
    const std::string directory_marker_path =
        output_path + "/gpmeep-preserved-marker";
    const std::string directory_marker_payload =
        "gpmeep-directory-must-survive-rename-failure\n";
    struct stat directory_identity = {};
    bool setup_ok = true;
    if (am_master()) {
      errno = 0;
      if (unlink(directory_marker_path.c_str()) != 0 && errno != ENOENT &&
          errno != ENOTDIR)
        setup_ok = false;
      errno = 0;
      if (unlink(temporary_path.c_str()) != 0 && errno != ENOENT)
        setup_ok = false;
      errno = 0;
      if (unlink(output_path.c_str()) != 0 && errno != ENOENT &&
          errno != EISDIR)
        setup_ok = false;
      errno = 0;
      if (rmdir(output_path.c_str()) != 0 && errno != ENOENT)
        setup_ok = false;
      if (setup_ok && mkdir(output_path.c_str(), 0700) != 0)
        setup_ok = false;
      if (setup_ok) {
        std::ofstream marker(directory_marker_path,
                             std::ios::binary | std::ios::trunc);
        setup_ok = marker.good();
        if (setup_ok)
          marker.write(directory_marker_payload.data(),
                       static_cast<std::streamsize>(
                           directory_marker_payload.size()));
        marker.close();
        setup_ok = setup_ok && marker.good();
      }
      if (setup_ok &&
          (stat(output_path.c_str(), &directory_identity) != 0 ||
           !S_ISDIR(directory_identity.st_mode)))
        setup_ok = false;
    }
    setup_ok = broadcast(0, setup_ok);
    require(setup_ok,
            "could not create the MPI DFT publication blocker directory");
    all_wait();

    gpu::reset_dispatch_statistics();
    bool rename_failure_observed = false;
    try {
      f.output_dft(monitor, output_name);
    }
    catch (const std::runtime_error &error) {
      rename_failure_observed =
          std::string(error.what()).find(
              "could not atomically publish its HDF5 file") !=
          std::string::npos;
    }
    const size_t caught_ranks =
        sum_to_all(static_cast<size_t>(rename_failure_observed));
    const gpu::dft_materialization_statistics failed_statistics =
        gpu::get_dft_materialization_statistics();
    const std::uint64_t failed_live_device_buffers =
        gpu::get_live_resident_device_buffers();
    const std::uint64_t failed_live_pinned_buffers =
        gpu::detail::get_live_dft_output_pinned_buffers_for_testing();
    const std::uint64_t expected_local_d2h =
        static_cast<std::uint64_t>(2) * local_monitor_points *
        frequencies.size() * sizeof(float);
    const bool failed_public_telemetry_ok =
        failed_statistics.cuda_output_calls == 0 &&
        failed_statistics.cuda_output_points == 0 &&
        failed_statistics.cpu_output_calls == 0 &&
        failed_statistics.cpu_output_points == 0 &&
        failed_statistics.cpu_array_calls == 0 &&
        failed_statistics.cuda_array_calls == 0 &&
        failed_statistics.array_mpi_allreduce_calls == 0 &&
        failed_statistics.array_mpi_allreduce_bytes == 0 &&
        failed_live_pinned_buffers == pinned_before + (owns_monitor ? 2 : 0);
    const bool failed_local_staging_ok =
        owns_monitor
            ? (failed_statistics.cuda_output_staging_calls == 1 &&
               failed_statistics.cuda_output_staging_points ==
                   local_monitor_points &&
               failed_statistics.cuda_output_staging_frequencies ==
                   frequencies.size() &&
               failed_statistics
                       .cuda_output_staging_descriptor_uploads == 1 &&
               failed_statistics.cuda_output_staging_plan_reuses == 0 &&
               failed_statistics.cuda_output_staging_kernel_launches == 1 &&
               failed_statistics
                       .cuda_output_staging_result_device_to_host_bytes ==
                   expected_local_d2h &&
               failed_statistics
                       .cuda_output_staging_full_dft_device_to_host_bytes_avoided ==
                   expected_local_d2h &&
               failed_statistics.cuda_output_staging_workspace_ceiling_bytes >
                   0)
            : (failed_statistics.cuda_output_staging_calls == 0 &&
               failed_statistics.cuda_output_staging_points == 0 &&
               failed_statistics.cuda_output_staging_frequencies == 0 &&
               failed_statistics
                       .cuda_output_staging_descriptor_uploads == 0 &&
               failed_statistics.cuda_output_staging_plan_reuses == 0 &&
               failed_statistics.cuda_output_staging_kernel_launches == 0 &&
               failed_statistics
                       .cuda_output_staging_result_device_to_host_bytes == 0 &&
               failed_statistics
                       .cuda_output_staging_full_dft_device_to_host_bytes_avoided ==
                   0 &&
               failed_statistics.cuda_output_staging_workspace_ceiling_bytes ==
                   0);
    const size_t failed_public_valid_ranks = sum_to_all(
        static_cast<size_t>(failed_public_telemetry_ok));
    const size_t failed_staging_valid_ranks = sum_to_all(
        static_cast<size_t>(failed_local_staging_ok));
    const size_t failed_total_kernels = sum_to_all(static_cast<size_t>(
        failed_statistics.cuda_output_staging_kernel_launches));
    const size_t failed_d2h_ranks = sum_to_all(static_cast<size_t>(
        failed_statistics.cuda_output_staging_result_device_to_host_bytes >
        0));

    // Rank zero performs a serial inspection only after every rank has caught
    // the synchronized publication exception.  Keep all verifier failures as
    // data until after broadcast so a rank-local HDF5 issue cannot strand its
    // peer in a collective.
    std::vector<float> failed_payload;
    size_t global_point_count = 0;
    std::uint64_t observed_root_links = 0;
    int observed_artifact_rank = -1;
    size_t observed_artifact_dims[3] = {0, 0, 0};
    const auto read_complete_artifact =
        [&](const std::string &path, std::vector<float> *payload,
            size_t *point_count) {
          try {
            h5file file(path.c_str(), h5file::READONLY,
                        false /* parallel */, true /* local */);
            observed_root_links = file.root_object_count();
            if (observed_root_links != 6) return false;
            if (file.dataset_exists("gpmeep_stale_marker")) return false;
            int common_rank = -1;
            size_t common_dims[3] = {0, 0, 0};
            payload->clear();
            for (size_t frequency = 0; frequency < frequencies.size();
                 ++frequency)
              for (int reim = 0; reim < 2; ++reim) {
                const std::string dataset =
                    "ez_" + std::to_string(frequency) + "." +
                    (reim ? "i" : "r");
                if (!file.dataset_exists(dataset.c_str())) return false;
                int rank = -1;
                size_t dims[3] = {0, 0, 0};
                std::unique_ptr<float[]> values(static_cast<float *>(
                    file.read(dataset.c_str(), &rank, dims, 3, true)));
                if (!values || rank <= 0) return false;
                size_t count = 1;
                for (int dim = 0; dim < rank; ++dim) {
                  if (dims[dim] >
                      std::numeric_limits<size_t>::max() / count)
                    return false;
                  count *= dims[dim];
                }
                if (frequency == 0 && reim == 0) {
                  common_rank = rank;
                  std::copy(dims, dims + 3, common_dims);
                  observed_artifact_rank = rank;
                  std::copy(dims, dims + 3, observed_artifact_dims);
                  *point_count = count;
                }
                else {
                  if (rank != common_rank || count != *point_count)
                    return false;
                  for (int dim = 0; dim < rank; ++dim)
                    if (dims[dim] != common_dims[dim]) return false;
                }
                for (size_t point = 0; point < count; ++point) {
                  if (!std::isfinite(values[point])) return false;
                  payload->push_back(values[point]);
                }
              }
            double maximum = 0.0;
            for (float value : *payload)
              maximum = std::max(maximum,
                                 std::abs(static_cast<double>(value)));
            bool exact_geometry = common_rank == oracle_rank;
            for (int dim = 0; dim < common_rank; ++dim)
              exact_geometry =
                  exact_geometry && common_dims[dim] == oracle_dims[dim];
            return exact_geometry && *point_count == oracle_point_count &&
                   payload->size() ==
                       2 * frequencies.size() * *point_count &&
                   maximum > 1e-8;
          }
          catch (...) {
            return false;
          }
        };

    struct stat complete_temporary_identity = {};
    bool failed_artifact_ok = true;
    bool failed_namespace_ok = true;
    bool failed_hdf5_ok = true;
    bool failed_oracle_match = true;
    if (am_master()) {
      struct stat preserved_directory = {};
      std::ifstream marker(directory_marker_path, std::ios::binary);
      std::ostringstream marker_bytes;
      marker_bytes << marker.rdbuf();
      failed_namespace_ok =
          stat(output_path.c_str(), &preserved_directory) == 0 &&
          S_ISDIR(preserved_directory.st_mode) &&
          preserved_directory.st_dev == directory_identity.st_dev &&
          preserved_directory.st_ino == directory_identity.st_ino &&
          marker.is_open() && !marker.bad() &&
          marker_bytes.str() == directory_marker_payload &&
          stat(temporary_path.c_str(), &complete_temporary_identity) == 0 &&
          S_ISREG(complete_temporary_identity.st_mode) &&
          complete_temporary_identity.st_size > 0;
      if (failed_namespace_ok)
        failed_hdf5_ok = read_complete_artifact(
            temporary_path, &failed_payload, &global_point_count);
      failed_oracle_match =
          global_point_count == oracle_point_count &&
          failed_payload == array_oracle;
      failed_artifact_ok =
          failed_namespace_ok && failed_hdf5_ok && failed_oracle_match;
      std::ostringstream trace;
      trace << "DFT_OUTPUT_MPI_RENAME_TRACE rank=0 namespace="
            << failed_namespace_ok << " hdf5=" << failed_hdf5_ok
            << " oracle=" << failed_oracle_match
            << " root_links=" << observed_root_links
            << " artifact_rank=" << observed_artifact_rank
            << " artifact_dims=" << observed_artifact_dims[0] << 'x'
            << observed_artifact_dims[1]
            << " artifact_points=" << global_point_count
            << " payload_scalars=" << failed_payload.size()
            << " oracle_scalars=" << array_oracle.size();
      write_stderr_line_atomically(trace.str());
    }
    failed_artifact_ok = broadcast(0, failed_artifact_ok);
    broadcast(0, &global_point_count, 1);
    reported_point_count = global_point_count;
    {
      std::ostringstream trace;
      trace << "DFT_OUTPUT_MPI_RENAME_TRACE rank=" << my_rank()
            << " caught=" << rename_failure_observed
            << " public_ok=" << failed_public_telemetry_ok
            << " staging_ok=" << failed_local_staging_ok
            << " pinned=" << failed_live_pinned_buffers
            << " pinned_before=" << pinned_before;
      write_stderr_line_atomically(trace.str());
    }
    require(caught_ranks == static_cast<size_t>(count_processors()) &&
                failed_public_valid_ranks ==
                    static_cast<size_t>(count_processors()) &&
                failed_staging_valid_ranks ==
                    static_cast<size_t>(count_processors()) &&
                failed_total_kernels == 1 && failed_d2h_ranks == 1 &&
                failed_artifact_ok,
            "MPI CUDA output_dft rename failure was asymmetric, committed "
            "success telemetry, or lost its complete temporary artifact");

    // Add an out-of-inventory dataset after proving the failed temporary is
    // complete.  A correct retry opens its first dataset with WRITE and must
    // therefore truncate this marker before atomically publishing the file.
    bool stale_marker_written = true;
    if (am_master()) {
      try {
        {
          h5file stale(temporary_path.c_str(), h5file::READWRITE,
                       false /* parallel */, true /* local */);
          const size_t marker_dims[1] = {1};
          float marker_value = 42.25f;
          stale.write("gpmeep_stale_marker", 1, marker_dims,
                      &marker_value, true);
        }
        h5file verify(temporary_path.c_str(), h5file::READONLY,
                      false /* parallel */, true /* local */);
        int marker_rank = -1;
        size_t marker_dims[3] = {0, 0, 0};
        std::unique_ptr<float[]> marker(static_cast<float *>(
            verify.read("gpmeep_stale_marker", &marker_rank,
                        marker_dims, 3, true)));
        stale_marker_written =
            marker && marker_rank == 0 && marker_dims[0] == 1 &&
            marker[0] == 42.25f;
      }
      catch (...) {
        stale_marker_written = false;
      }
    }
    stale_marker_written = broadcast(0, stale_marker_written);
    require(stale_marker_written,
            "could not poison the completed MPI DFT temporary for retry");

    bool blocker_removed = true;
    if (am_master()) {
      blocker_removed = unlink(directory_marker_path.c_str()) == 0;
      blocker_removed =
          blocker_removed && rmdir(output_path.c_str()) == 0;
    }
    blocker_removed = broadcast(0, blocker_removed);
    require(blocker_removed,
            "could not remove the MPI DFT publication blocker directory");
    all_wait();

    gpu::reset_dispatch_statistics();
    f.output_dft(monitor, output_name);
    const gpu::dft_materialization_statistics recovered_statistics =
        gpu::get_dft_materialization_statistics();
    const std::uint64_t recovered_live_device_buffers =
        gpu::get_live_resident_device_buffers();
    const std::uint64_t recovered_live_pinned_buffers =
        gpu::detail::get_live_dft_output_pinned_buffers_for_testing();
    const std::uint64_t expected_public_points =
        static_cast<std::uint64_t>(global_point_count) * frequencies.size();
    const bool recovered_public_telemetry_ok =
        recovered_statistics.cuda_output_calls == 1 &&
        recovered_statistics.cuda_output_points == expected_public_points &&
        recovered_statistics.cpu_output_calls == 0 &&
        recovered_statistics.cpu_output_points == 0 &&
        recovered_statistics.cpu_array_calls == 0 &&
        recovered_statistics.cuda_array_calls == 0 &&
        recovered_statistics.array_mpi_allreduce_calls == 0 &&
        recovered_statistics.array_mpi_allreduce_bytes == 0 &&
        recovered_live_device_buffers == failed_live_device_buffers &&
        recovered_live_pinned_buffers == failed_live_pinned_buffers;
    const bool recovered_local_staging_ok =
        owns_monitor
            ? (recovered_statistics.cuda_output_staging_calls == 1 &&
               recovered_statistics.cuda_output_staging_points ==
                   local_monitor_points &&
               recovered_statistics.cuda_output_staging_frequencies ==
                   frequencies.size() &&
               recovered_statistics
                       .cuda_output_staging_descriptor_uploads == 0 &&
               recovered_statistics.cuda_output_staging_plan_reuses == 1 &&
               recovered_statistics.cuda_output_staging_kernel_launches ==
                   1 &&
               recovered_statistics
                       .cuda_output_staging_result_device_to_host_bytes ==
                   expected_local_d2h &&
               recovered_statistics
                       .cuda_output_staging_full_dft_device_to_host_bytes_avoided ==
                   expected_local_d2h &&
               recovered_statistics
                       .cuda_output_staging_workspace_ceiling_bytes ==
                   failed_statistics
                       .cuda_output_staging_workspace_ceiling_bytes)
            : (recovered_statistics.cuda_output_staging_calls == 0 &&
               recovered_statistics.cuda_output_staging_points == 0 &&
               recovered_statistics.cuda_output_staging_frequencies == 0 &&
               recovered_statistics
                       .cuda_output_staging_descriptor_uploads == 0 &&
               recovered_statistics.cuda_output_staging_plan_reuses == 0 &&
               recovered_statistics.cuda_output_staging_kernel_launches ==
                   0 &&
               recovered_statistics
                       .cuda_output_staging_result_device_to_host_bytes == 0 &&
               recovered_statistics
                       .cuda_output_staging_full_dft_device_to_host_bytes_avoided ==
                   0 &&
               recovered_statistics
                       .cuda_output_staging_workspace_ceiling_bytes == 0);
    const size_t recovered_public_valid_ranks = sum_to_all(
        static_cast<size_t>(recovered_public_telemetry_ok));
    const size_t recovered_staging_valid_ranks = sum_to_all(
        static_cast<size_t>(recovered_local_staging_ok));
    const size_t recovered_total_kernels = sum_to_all(static_cast<size_t>(
        recovered_statistics.cuda_output_staging_kernel_launches));
    const size_t recovered_d2h_ranks = sum_to_all(static_cast<size_t>(
        recovered_statistics
                .cuda_output_staging_result_device_to_host_bytes > 0));

    bool recovered_artifact_ok = true;
    if (am_master()) {
      struct stat final_status = {};
      std::vector<float> recovered_payload;
      size_t recovered_point_count = 0;
      recovered_artifact_ok =
          stat(output_path.c_str(), &final_status) == 0 &&
          S_ISREG(final_status.st_mode) &&
          final_status.st_dev == complete_temporary_identity.st_dev &&
          final_status.st_ino == complete_temporary_identity.st_ino &&
          access(temporary_path.c_str(), F_OK) != 0 &&
          read_complete_artifact(output_path, &recovered_payload,
                                 &recovered_point_count) &&
          recovered_point_count == global_point_count &&
          recovered_payload == failed_payload;
    }
    recovered_artifact_ok = broadcast(0, recovered_artifact_ok);
    require(recovered_public_valid_ranks ==
                static_cast<size_t>(count_processors()) &&
                recovered_staging_valid_ranks ==
                    static_cast<size_t>(count_processors()) &&
                recovered_total_kernels == 1 && recovered_d2h_ranks == 1 &&
                recovered_artifact_ok,
            "MPI CUDA output_dft did not reuse its staging plan or publish "
            "the exact completed temporary payload on retry");

    all_wait();
    bool cleanup_ok = true;
    if (am_master())
      cleanup_ok = unlink(output_path.c_str()) == 0 &&
                   access(temporary_path.c_str(), F_OK) != 0;
    cleanup_ok = broadcast(0, cleanup_ok);
    require(cleanup_ok,
            "MPI rename-failure regression could not clean its output");
    all_wait();
    monitor.remove();
  }

  gpu::set_backend(gpu::backend_mode::cpu);
  const size_t lifecycle_valid_ranks = sum_to_all(static_cast<size_t>(
      gpu::get_live_resident_device_buffers() == live_before));
  const size_t pinned_lifecycle_valid_ranks = sum_to_all(static_cast<size_t>(
      gpu::detail::get_live_dft_output_pinned_buffers_for_testing() ==
      pinned_before));
  require(lifecycle_valid_ranks == static_cast<size_t>(count_processors()) &&
              pinned_lifecycle_valid_ranks ==
                  static_cast<size_t>(count_processors()),
          "MPI rename-failure regression leaked resident device or pinned "
          "host buffers");
  if (am_master())
    std::cout
        << "PASS: MPI CUDA output_dft synchronously preserves a complete temp "
           "on rename failure and publishes its exact payload on retry; "
           "owner_kernels=1 peer_kernels=0 points="
        << reported_point_count << " frequencies=3 datasets=6\n";
}

void require_public_nonempty_dft_output_zero_work_mpi() {
  require(count_processors() == 2,
          "public nonempty DFT output regression requires exactly two MPI "
          "ranks");

  const grid_volume gv = vol2d(1.6, 1.4, 12.0);
  const volume monitor_volume(vec(0.10, 0.10), vec(0.38, 0.34));
  const std::vector<double> frequencies = {0.23, 0.29, 0.35};
  component components[] = {Ez};
  int reference_rank = -1;
  size_t reference_dims[3] = {0, 0, 0};
  size_t reference_point_count = 0;
  std::vector<std::vector<std::complex<double> > > reference_spectrum;

  // Build an independent CPU oracle.  The monitor has positive extent in
  // both axes, so output_dft must take the ordinary dense HDF5 path rather
  // than the degenerate-dimension implementation.
  gpu::set_backend(gpu::backend_mode::cpu);
  {
    structure s(gv, vacuum, no_pml(), identity(), 4);
    fields f(&s, 0.0, 0.0, true, 32, 32);
    f.use_real_fields();
    gaussian_src_time source(0.29, 0.10);
    f.add_point_source(Ez, source, vec(0.24, 0.22), 0.8);
    dft_fields monitor = f.add_dft_fields(
        components, 1, monitor_volume, frequencies, true, 1);
    for (int step = 0; step < 32; ++step) f.step();

    size_t local_monitor_points = 0;
    for (dft_chunk *chunk = monitor.chunks; chunk;
         chunk = chunk->next_in_dft)
      if (chunk->c == Ez) local_monitor_points += chunk->N;
    require(sum_to_all(local_monitor_points > 0 ? 1 : 0) == 1,
            "ordinary finite-area CPU DFT monitor was not owned by exactly "
            "one MPI rank");

    for (size_t frequency = 0; frequency < frequencies.size(); ++frequency) {
      int rank = -1;
      size_t dims[3] = {0, 0, 0};
      std::unique_ptr<std::complex<realnum>[]> raw(
          f.get_dft_array(monitor, Ez, static_cast<int>(frequency),
                          &rank, dims));
      require(raw != nullptr && rank > 0,
              "ordinary CPU DFT oracle was empty or scalar");
      size_t count = 1;
      for (int dim = 0; dim < rank; ++dim) {
        require(dims[dim] <=
                    std::numeric_limits<size_t>::max() / count,
                "ordinary CPU DFT oracle dimensions overflow");
        count *= dims[dim];
      }
      if (frequency == 0) {
        reference_rank = rank;
        std::copy(dims, dims + 3, reference_dims);
        reference_point_count = count;
      }
      else {
        require(rank == reference_rank && count == reference_point_count,
                "ordinary CPU DFT spectrum changed its dense geometry");
        for (int dim = 0; dim < rank; ++dim)
          require(dims[dim] == reference_dims[dim],
                  "ordinary CPU DFT spectrum changed a dense dimension");
      }
      std::vector<std::complex<double> > values(count);
      for (size_t point = 0; point < count; ++point)
        values[point] = std::complex<double>(raw[point]);
      reference_spectrum.push_back(std::move(values));
    }
    monitor.remove();
  }

  double reference_maximum = 0.0;
  for (const auto &frequency : reference_spectrum)
    for (const auto &value : frequency)
      reference_maximum = std::max(reference_maximum, std::abs(value));
  require(reference_point_count > 1 && reference_maximum > 1e-8,
          "ordinary CPU DFT oracle is too small to validate publication");

  gpu::set_backend(gpu::backend_mode::cuda);
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  {
    structure s(gv, vacuum, no_pml(), identity(), 4);
    fields f(&s, 0.0, 0.0, true, 32, 32);
    f.use_real_fields();
    gaussian_src_time source(0.29, 0.10);
    f.add_point_source(Ez, source, vec(0.24, 0.22), 0.8);
    dft_fields monitor = f.add_dft_fields(
        components, 1, monitor_volume, frequencies, true, 1);
    for (int step = 0; step < 32; ++step) f.step();

    size_t local_monitor_points = 0;
    size_t local_monitor_chunks = 0;
    for (dft_chunk *chunk = monitor.chunks; chunk;
         chunk = chunk->next_in_dft)
      if (chunk->c == Ez) {
        ++local_monitor_chunks;
        local_monitor_points += chunk->N;
      }
    const bool owns_monitor = local_monitor_points > 0;
    require(sum_to_all(owns_monitor ? 1 : 0) == 1,
            "ordinary finite-area CUDA DFT monitor was not owned by exactly "
            "one MPI rank");
    require(sum_to_all(local_monitor_chunks) >= 1,
            "ordinary finite-area CUDA DFT monitor has no chunks");

    // Capture the CUDA public array before resetting telemetry.  This is a
    // same-snapshot serialization oracle in addition to the independent CPU
    // run, and proves the HDF5 ordering rather than merely its norm.
    std::vector<std::vector<std::complex<double> > > cuda_spectrum;
    double cpu_cuda_array_max_error = 0.0;
    for (size_t frequency = 0; frequency < frequencies.size(); ++frequency) {
      int rank = -1;
      size_t dims[3] = {0, 0, 0};
      std::unique_ptr<std::complex<realnum>[]> raw(
          f.get_dft_array(monitor, Ez, static_cast<int>(frequency),
                          &rank, dims));
      require(raw != nullptr && rank == reference_rank,
              "ordinary CUDA DFT array rank differs from CPU");
      for (int dim = 0; dim < rank; ++dim)
        require(dims[dim] == reference_dims[dim],
                "ordinary CUDA DFT array dimensions differ from CPU");
      std::vector<std::complex<double> > values(reference_point_count);
      for (size_t point = 0; point < reference_point_count; ++point)
        values[point] = std::complex<double>(raw[point]);
      require_complex_vectors_close(
          reference_spectrum[frequency], values,
          "ordinary CUDA DFT array CPU oracle", 1.5e-3, 5e-6);
      for (size_t point = 0; point < reference_point_count; ++point)
        cpu_cuda_array_max_error = std::max(
            cpu_cuda_array_max_error,
            std::abs(reference_spectrum[frequency][point] - values[point]));
      cuda_spectrum.push_back(std::move(values));
    }

    const char *output_name =
        "gpu-step-db-public-dft-output-mpi-ordinary-nonempty";
    const std::string output_path = f.h5file_name(output_name);
    const std::string temporary_name =
        std::string(output_name) + ".gpmeep-tmp";
    const std::string temporary_path =
        f.h5file_name(temporary_name.c_str());
    if (am_master()) {
      unlink(output_path.c_str());
      unlink(temporary_path.c_str());
    }
    all_wait();

    gpu::reset_dispatch_statistics();
    std::cerr << "DFT_OUTPUT_MPI_TRACE rank=" << my_rank()
              << " stage=before-public-output path=" << output_path
              << " local_points=" << local_monitor_points << '\n'
              << std::flush;
    f.output_dft(monitor, output_name);
    std::cerr << "DFT_OUTPUT_MPI_TRACE rank=" << my_rank()
              << " stage=after-public-output\n"
              << std::flush;
    const gpu::dft_materialization_statistics statistics =
        gpu::get_dft_materialization_statistics();
    const std::uint64_t expected_public_points =
        static_cast<std::uint64_t>(reference_point_count) *
        frequencies.size();
    const std::uint64_t expected_local_d2h =
        static_cast<std::uint64_t>(2) * local_monitor_points *
        frequencies.size() * sizeof(float);
    std::cerr
        << "DFT_OUTPUT_MPI_TRACE rank=" << my_rank()
        << " stage=statistics owns=" << owns_monitor
        << " public_calls=" << statistics.cuda_output_calls
        << " public_points=" << statistics.cuda_output_points
        << " cpu_calls=" << statistics.cpu_output_calls
        << " staging_calls=" << statistics.cuda_output_staging_calls
        << " staging_points=" << statistics.cuda_output_staging_points
        << " staging_frequencies="
        << statistics.cuda_output_staging_frequencies
        << " uploads="
        << statistics.cuda_output_staging_descriptor_uploads
        << " reuses=" << statistics.cuda_output_staging_plan_reuses
        << " kernels="
        << statistics.cuda_output_staging_kernel_launches
        << " d2h="
        << statistics.cuda_output_staging_result_device_to_host_bytes
        << " avoided="
        << statistics
               .cuda_output_staging_full_dft_device_to_host_bytes_avoided
        << " workspace="
        << statistics.cuda_output_staging_workspace_ceiling_bytes
        << " array_allreduces=" << statistics.array_mpi_allreduce_calls
        << '\n' << std::flush;

    // Do not throw on one rank before peers enter the telemetry reductions
    // below.  In particular, the sole owner and zero-work peer intentionally
    // have different local staging counters; a rank-local assertion here can
    // strand the other rank in sum_to_all and obscure the actual mismatch.
    const bool public_telemetry_ok =
        statistics.cuda_output_calls == 1 &&
        statistics.cuda_output_points == expected_public_points &&
        statistics.cpu_output_calls == 0 &&
        statistics.cpu_output_points == 0;
    const bool dense_array_path_absent =
        statistics.array_mpi_allreduce_calls == 0 &&
        statistics.array_mpi_allreduce_bytes == 0 &&
        statistics.cpu_array_calls == 0 &&
        statistics.cuda_array_calls == 0;
    const bool local_staging_ok =
        owns_monitor
            ? (statistics.cuda_output_staging_calls == 1 &&
               statistics.cuda_output_staging_points ==
                   local_monitor_points &&
               statistics.cuda_output_staging_frequencies ==
                   frequencies.size() &&
               statistics.cuda_output_staging_descriptor_uploads == 1 &&
               statistics.cuda_output_staging_plan_reuses == 0 &&
               statistics.cuda_output_staging_kernel_launches == 1 &&
               statistics
                       .cuda_output_staging_result_device_to_host_bytes ==
                   expected_local_d2h &&
               statistics
                       .cuda_output_staging_full_dft_device_to_host_bytes_avoided >
                   0 &&
               statistics.cuda_output_staging_workspace_ceiling_bytes > 0)
            : (statistics.cuda_output_staging_calls == 0 &&
               statistics.cuda_output_staging_points == 0 &&
               statistics.cuda_output_staging_frequencies == 0 &&
               statistics.cuda_output_staging_descriptor_uploads == 0 &&
               statistics.cuda_output_staging_plan_reuses == 0 &&
               statistics.cuda_output_staging_kernel_launches == 0 &&
               statistics
                       .cuda_output_staging_result_device_to_host_bytes == 0 &&
               statistics
                       .cuda_output_staging_full_dft_device_to_host_bytes_avoided ==
                   0 &&
               statistics.cuda_output_staging_workspace_ceiling_bytes == 0);
    const size_t public_telemetry_valid_ranks =
        sum_to_all(static_cast<size_t>(public_telemetry_ok));
    const size_t dense_array_path_absent_ranks =
        sum_to_all(static_cast<size_t>(dense_array_path_absent));
    const size_t local_staging_valid_ranks =
        sum_to_all(static_cast<size_t>(local_staging_ok));
    const size_t total_staging_kernel_launches = sum_to_all(
        static_cast<size_t>(statistics.cuda_output_staging_kernel_launches));
    const size_t ranks_with_staging_d2h = sum_to_all(static_cast<size_t>(
        statistics.cuda_output_staging_result_device_to_host_bytes > 0));
    const size_t mpi_rank_count = static_cast<size_t>(count_processors());
    std::cerr << "DFT_OUTPUT_MPI_TRACE rank=" << my_rank()
              << " stage=after-telemetry-reductions"
              << " public_valid=" << public_telemetry_valid_ranks
              << " dense_absent=" << dense_array_path_absent_ranks
              << " local_valid=" << local_staging_valid_ranks
              << " total_kernels=" << total_staging_kernel_launches
              << " d2h_ranks=" << ranks_with_staging_d2h << '\n'
              << std::flush;

    require(public_telemetry_valid_ranks == mpi_rank_count,
            "public ordinary output_dft telemetry shows CPU fallback or an "
            "incorrect logical point count");
    require(dense_array_path_absent_ranks == mpi_rank_count,
            "public ordinary output_dft reused the dense get_dft_array "
            "all-reduce path");
    require(local_staging_valid_ranks == mpi_rank_count,
            "public ordinary output_dft did not stage exactly on its sole "
            "owner rank");
    require(total_staging_kernel_launches == 1 &&
                ranks_with_staging_d2h == 1,
            "public ordinary output_dft staged on more than its one owner "
            "rank");

    // The publication itself was parallel.  Have rank zero validate the
    // finished artifact through an independent serial read-only handle, then
    // broadcast the observed metadata and payload for collective assertions.
    // Opening another parallel-HDF5 handle here is unnecessary and some
    // HDF5/MPI stacks require collective dataset-transfer properties for its
    // whole-dataset read helper, which can turn a verifier-only read into a
    // false deadlock after the production operation has already completed.
    std::cerr << "DFT_OUTPUT_MPI_TRACE rank=" << my_rank()
              << " stage=before-master-readonly-open\n" << std::flush;
    std::unique_ptr<h5file> output;
    if (am_master())
      output.reset(new h5file(output_path.c_str(), h5file::READONLY,
                              false /* parallel */, true /* local */));
    std::cerr << "DFT_OUTPUT_MPI_TRACE rank=" << my_rank()
              << " stage=after-master-readonly-open\n" << std::flush;
    double cuda_hdf5_max_error = 0.0;
    double cpu_hdf5_max_error = 0.0;
    for (size_t frequency = 0; frequency < frequencies.size(); ++frequency) {
      int real_rank = -1;
      int imaginary_rank = -1;
      size_t real_dims[3] = {0, 0, 0};
      size_t imaginary_dims[3] = {0, 0, 0};
      const std::string real_name =
          "ez_" + std::to_string(frequency) + ".r";
      const std::string imaginary_name =
          "ez_" + std::to_string(frequency) + ".i";
      std::vector<float> real_values(reference_point_count);
      std::vector<float> imaginary_values(reference_point_count);
      bool payload_present = true;
      if (am_master()) {
        std::unique_ptr<float[]> real_payload(static_cast<float *>(
            output->read(real_name.c_str(), &real_rank, real_dims, 3,
                         true)));
        std::unique_ptr<float[]> imaginary_payload(static_cast<float *>(
            output->read(imaginary_name.c_str(), &imaginary_rank,
                         imaginary_dims, 3, true)));
        payload_present =
            real_payload != nullptr && imaginary_payload != nullptr;
        if (payload_present) {
          std::copy(real_payload.get(),
                    real_payload.get() + reference_point_count,
                    real_values.begin());
          std::copy(imaginary_payload.get(),
                    imaginary_payload.get() + reference_point_count,
                    imaginary_values.begin());
        }
      }
      payload_present = broadcast(0, payload_present);
      real_rank = broadcast(0, real_rank);
      imaginary_rank = broadcast(0, imaginary_rank);
      broadcast(0, real_dims, 3);
      broadcast(0, imaginary_dims, 3);
      broadcast(0, real_values.data(),
                static_cast<int>(real_values.size()));
      broadcast(0, imaginary_values.data(),
                static_cast<int>(imaginary_values.size()));
      require(payload_present && real_rank == reference_rank &&
                  imaginary_rank == reference_rank,
              "public ordinary output_dft HDF5 rank differs from its array "
              "oracle");
      for (int dim = 0; dim < reference_rank; ++dim)
        require(real_dims[dim] == reference_dims[dim] &&
                    imaginary_dims[dim] == reference_dims[dim],
                "public ordinary output_dft HDF5 dimensions differ from its "
                "array oracle");
      std::vector<std::complex<double> > observed(reference_point_count);
      for (size_t point = 0; point < reference_point_count; ++point)
        observed[point] = std::complex<double>(real_values[point],
                                               imaginary_values[point]);
      require_complex_vectors_close(
          cuda_spectrum[frequency], observed,
          "public ordinary output_dft same-snapshot HDF5 oracle",
          2e-5, 2e-6);
      require_complex_vectors_close(
          reference_spectrum[frequency], observed,
          "public ordinary output_dft independent CPU oracle",
          1.5e-3, 5e-6);
      for (size_t point = 0; point < reference_point_count; ++point) {
        cuda_hdf5_max_error = std::max(
            cuda_hdf5_max_error,
            std::abs(cuda_spectrum[frequency][point] - observed[point]));
        cpu_hdf5_max_error = std::max(
            cpu_hdf5_max_error,
            std::abs(reference_spectrum[frequency][point] -
                     observed[point]));
      }
    }
    output.reset();
    all_wait();
    bool atomic_temporary_absent = true;
    if (am_master()) {
      unlink(output_path.c_str());
      atomic_temporary_absent =
          access(temporary_path.c_str(), F_OK) != 0;
    }
    atomic_temporary_absent = broadcast(0, atomic_temporary_absent);
    require(atomic_temporary_absent,
            "public ordinary output_dft left its atomic temporary file");
    all_wait();

    if (am_master())
      std::cout
          << "PASS: public ordinary nonempty output_dft used one owner CUDA "
             "stage, zero peer work, no dense all-reduce, and CPU/array HDF5 "
             "oracles; points="
          << reference_point_count << " frequencies=" << frequencies.size()
          << " datasets=ez_0.r,ez_0.i,ez_1.r,ez_1.i,ez_2.r,ez_2.i"
          << " rank=" << reference_rank << " dims=" << reference_dims[0]
          << 'x' << reference_dims[1]
          << " cpu_cuda_array_max_abs=" << cpu_cuda_array_max_error
          << " cuda_hdf5_max_abs=" << cuda_hdf5_max_error
          << " cpu_hdf5_max_abs=" << cpu_hdf5_max_error
          << '\n';
    monitor.remove();
  }
  gpu::set_backend(gpu::backend_mode::cpu);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "public ordinary output_dft leaked resident device buffers");
}

void require_public_collapsed_dft_output_cuda() {
  require(count_processors() == 1,
          "public collapsed DFT output regression is singleton-only");
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  gpu::set_backend(gpu::backend_mode::cuda);

  {
    const grid_volume gv = vol2d(1.6, 1.4, 12.0);
    structure s(gv, vacuum, no_pml(), identity(), 4);
    fields f(&s, 0.0, 0.0, true, 32, 32);
    f.use_real_fields();
    {
      const char *checked_name =
          "gpu-step-db-checked-hdf5-write-failure";
      const std::string checked_path = f.h5file_name(checked_name);
      unlink(checked_path.c_str());
      h5file checked(checked_path.c_str(), h5file::WRITE,
                     false /* parallel */, true /* local */);
      const size_t checked_dims[1] = {1};
      const double checked_value[1] = {3.25};
      require(checked.ok() &&
                  checked.write_serial_checked(
                      "duplicate", 1, checked_dims, checked_value,
                      false /* single_precision */) &&
                  !checked.write_serial_checked(
                      "duplicate", 1, checked_dims, checked_value,
                      false /* single_precision */) &&
                  checked.flush_and_close_checked(),
              "checked serial HDF5 write did not return an actual dataset "
              "creation failure without aborting");
      require(unlink(checked_path.c_str()) == 0,
              "checked serial HDF5 failure fixture cleanup failed");
    }
    gaussian_src_time source(0.29, 0.10);
    f.add_point_source(Ez, source, vec(0.31, 0.27), 0.8);
    component components[] = {Ez};
    const std::vector<double> frequencies = {0.23, 0.29, 0.35};
    dft_fields point = f.add_dft_fields(
        components, 1,
        volume(vec(0.83, 0.67), vec(0.83, 0.67)), frequencies,
        true, 1);
    dft_fields thin_x = f.add_dft_fields(
        components, 1,
        volume(vec(0.83, 0.19), vec(0.83, 1.11)), frequencies,
        true, 1);
    dft_fields thin_y = f.add_dft_fields(
        components, 1,
        volume(vec(0.21, 0.67), vec(1.31, 0.67)), frequencies,
        true, 1);
    for (int step = 0; step < 32; ++step) f.step();

    const auto verify_monitor =
        [&](dft_fields &monitor, const char *label,
            int expected_rank, bool explicit_h5_suffix = false) {
          int oracle_rank = -1;
          size_t oracle_dims[3] = {0, 0, 0};
          size_t point_count = 0;
          std::vector<std::vector<std::complex<realnum>>> oracle;
          for (size_t frequency = 0; frequency < frequencies.size();
               ++frequency) {
            int rank = -1;
            size_t dims[3] = {0, 0, 0};
            std::unique_ptr<std::complex<realnum>[]> values(
                f.get_dft_array(monitor, Ez,
                                static_cast<int>(frequency), &rank, dims));
            require(values != nullptr && rank == expected_rank,
                    std::string(label) +
                        " CUDA get_dft_array oracle has wrong rank");
            size_t count = 1;
            for (int dim = 0; dim < rank; ++dim) {
              require(dims[dim] <=
                          std::numeric_limits<size_t>::max() / count,
                      std::string(label) + " oracle extent overflow");
              count *= dims[dim];
            }
            if (frequency == 0) {
              oracle_rank = rank;
              std::copy(dims, dims + 3, oracle_dims);
              point_count = count;
            }
            else {
              require(rank == oracle_rank && count == point_count,
                      std::string(label) +
                          " oracle geometry changed across frequencies");
              for (int dim = 0; dim < rank; ++dim)
                require(dims[dim] == oracle_dims[dim],
                        std::string(label) +
                            " oracle dimensions changed across frequencies");
            }
            oracle.emplace_back(values.get(), values.get() + count);
          }

          const std::string output_base =
              std::string("gpu-step-db-public-dft-output-collapsed-") +
              label;
          const std::string output_name =
              output_base + (explicit_h5_suffix ? ".h5" : "");
          const std::string output_path =
              f.h5file_name(output_base.c_str());
          const std::string temporary_name =
              output_base + ".gpmeep-tmp";
          const std::string temporary_path =
              f.h5file_name(temporary_name.c_str());
          unlink(output_path.c_str());
          unlink(temporary_path.c_str());

          gpu::reset_dispatch_statistics();
          f.output_dft(monitor, output_name.c_str());
          const gpu::dft_materialization_statistics statistics =
              gpu::get_dft_materialization_statistics();
          require(statistics.cuda_output_calls == 1 &&
                      statistics.cuda_output_points ==
                          point_count * frequencies.size() &&
                      statistics.cpu_output_calls == 0 &&
                      statistics.cpu_output_points == 0 &&
                      statistics.cpu_array_calls == 0 &&
                      statistics.cuda_array_calls == 0 &&
                      statistics.cuda_kernel_launches == 2 &&
                      statistics.cuda_result_device_to_host_bytes ==
                          point_count * frequencies.size() *
                              sizeof(std::complex<realnum>) &&
                      statistics.array_mpi_allreduce_calls ==
                          frequencies.size() &&
                      statistics.array_mpi_allreduce_bytes ==
                          point_count * frequencies.size() *
                              sizeof(std::complex<realnum>),
                  std::string(label) +
                      " collapsed CUDA output telemetry is not exact");

          h5file output(output_path.c_str(), h5file::READONLY,
                        false /* parallel */, true /* local */);
          require(output.root_object_count() == 2 * frequencies.size(),
                  std::string(label) +
                      " collapsed HDF5 inventory is not exact");
          double maximum_error = 0.0;
          bool bitwise_equal = true;
          bool finite_payload = true;
          for (size_t frequency = 0; frequency < frequencies.size();
               ++frequency) {
            int real_rank = -1;
            int imaginary_rank = -1;
            size_t real_dims[3] = {0, 0, 0};
            size_t imaginary_dims[3] = {0, 0, 0};
            const std::string real_name =
                "ez_" + std::to_string(frequency) + ".r";
            const std::string imaginary_name =
                "ez_" + std::to_string(frequency) + ".i";
            std::unique_ptr<double[]> real_values(static_cast<double *>(
                output.read(real_name.c_str(), &real_rank, real_dims, 3,
                            false)));
            std::unique_ptr<double[]> imaginary_values(static_cast<double *>(
                output.read(imaginary_name.c_str(), &imaginary_rank,
                            imaginary_dims, 3, false)));
            // h5file intentionally normalizes a stored rank-1 {1} dataset
            // back to public rank 0 while retaining dims[0]==1.
            const int expected_hdf5_rank = oracle_rank;
            require(real_values && imaginary_values &&
                        real_rank == expected_hdf5_rank &&
                        imaginary_rank == expected_hdf5_rank,
                    std::string(label) +
                        " collapsed HDF5 rank differs from its oracle");
            if (oracle_rank == 0)
              require(real_dims[0] == 1 && imaginary_dims[0] == 1,
                      std::string(label) +
                          " scalar HDF5 dataset is not shape (1,)");
            else
              for (int dim = 0; dim < oracle_rank; ++dim)
                require(real_dims[dim] == oracle_dims[dim] &&
                            imaginary_dims[dim] == oracle_dims[dim],
                        std::string(label) +
                            " collapsed HDF5 dimensions differ from oracle");
            for (size_t point_index = 0; point_index < point_count;
                 ++point_index) {
              const std::complex<double> observed(
                  real_values[point_index], imaginary_values[point_index]);
              const std::complex<double> expected(
                  oracle[frequency][point_index]);
              const double error = std::abs(observed - expected);
              finite_payload = finite_payload &&
                               std::isfinite(observed.real()) &&
                               std::isfinite(observed.imag()) &&
                               std::isfinite(expected.real()) &&
                               std::isfinite(expected.imag()) &&
                               std::isfinite(error);
              if (std::isfinite(error))
                maximum_error = std::max(maximum_error, error);
              const double expected_real = expected.real();
              const double expected_imaginary = expected.imag();
              bitwise_equal =
                  bitwise_equal &&
                  std::memcmp(&real_values[point_index], &expected_real,
                              sizeof(double)) == 0 &&
                  std::memcmp(&imaginary_values[point_index],
                              &expected_imaginary, sizeof(double)) == 0;
            }
          }
          require(finite_payload && maximum_error <= 2e-6 &&
                      bitwise_equal,
                  std::string(label) +
                      " device collapse differs from host collapse oracle");
          require(access(temporary_path.c_str(), F_OK) != 0,
                  std::string(label) +
                      " collapsed output left its atomic temporary file");
          require(unlink(output_path.c_str()) == 0,
                  std::string(label) +
                      " collapsed output cleanup failed");
          if (am_master())
            std::cout << "COLLAPSED_DFT_OUTPUT case=" << label
                      << " rank=" << oracle_rank
                      << " points=" << point_count
                      << " frequencies=" << frequencies.size()
                      << " kernels=" << statistics.cuda_kernel_launches
                      << " d2h="
                      << statistics.cuda_result_device_to_host_bytes
                      << " mpi_bytes="
                      << statistics.array_mpi_allreduce_bytes
                      << " max_abs=" << maximum_error
                      << " bitwise=" << (bitwise_equal ? 1 : 0) << '\n';
        };

    verify_monitor(point, "point", 0);
    verify_monitor(thin_x, "thin-x", 1);
    verify_monitor(thin_y, "thin-y", 1, true);

    const char *atomic_name =
        "gpu-step-db-public-dft-output-collapsed-atomic";
    const std::string atomic_path = f.h5file_name(atomic_name);
    const std::string atomic_temporary_name =
        std::string(atomic_name) + ".gpmeep-tmp";
    const std::string atomic_temporary_path =
        f.h5file_name(atomic_temporary_name.c_str());
    unlink(atomic_path.c_str());
    unlink(atomic_temporary_path.c_str());
    const std::string sentinel =
        "gpmeep-collapsed-final-must-survive-partial-output\n";
    {
      std::ofstream output(
          atomic_path, std::ios::binary | std::ios::trunc);
      require(output.good(),
              "could not create collapsed output sentinel");
      output.write(sentinel.data(),
                   static_cast<std::streamsize>(sentinel.size()));
      output.close();
      require(output.good(),
              "could not close collapsed output sentinel");
    }
    struct stat sentinel_status;
    require(stat(atomic_path.c_str(), &sentinel_status) == 0,
            "could not stat collapsed output sentinel");
    const auto read_bytes = [](const std::string &path) {
      std::ifstream input(path, std::ios::binary);
      if (!input.good())
        throw std::runtime_error(
            "could not read collapsed output artifact");
      std::ostringstream bytes;
      bytes << input.rdbuf();
      if (!input.good() && !input.eof())
        throw std::runtime_error(
            "failed while reading collapsed output artifact");
      return bytes.str();
    };

    gpu::reset_dispatch_statistics();
    gpu::detail::fail_next_cuda_dft_output_after_dataset_for_testing();
    bool injected_failure_observed = false;
    try { f.output_dft(point, atomic_name); }
    catch (const std::runtime_error &error) {
      injected_failure_observed =
          std::string(error.what()).find(
              "injected CUDA collapsed DFT output failure after an HDF5 "
              "dataset") != std::string::npos;
    }
    (void)gpu::detail::
        consume_cuda_dft_output_dataset_failure_for_testing();
    struct stat preserved_status;
    const gpu::dft_materialization_statistics failed_statistics =
        gpu::get_dft_materialization_statistics();
    require(injected_failure_observed &&
                stat(atomic_path.c_str(), &preserved_status) == 0 &&
                preserved_status.st_dev == sentinel_status.st_dev &&
                preserved_status.st_ino == sentinel_status.st_ino &&
                preserved_status.st_size == sentinel_status.st_size &&
                read_bytes(atomic_path) == sentinel &&
                access(atomic_temporary_path.c_str(), F_OK) == 0 &&
                failed_statistics.cuda_output_calls == 0 &&
                failed_statistics.cpu_output_calls == 0 &&
                failed_statistics.cuda_output_points == 0 &&
                failed_statistics.cpu_output_points == 0,
            "failed collapsed CUDA output replaced its final file or "
            "committed success telemetry");
    {
      h5file partial(atomic_temporary_path.c_str(), h5file::READONLY,
                     false /* parallel */, true /* local */);
      require(partial.root_object_count() == 2 &&
                  partial.dataset_exists("ez_0.r") &&
                  partial.dataset_exists("ez_0.i") &&
                  !partial.dataset_exists("ez_1.r"),
              "failed collapsed CUDA output temporary has the wrong "
              "dataset prefix");
    }
    {
      h5file stale(atomic_temporary_path.c_str(), h5file::READWRITE,
                   false /* parallel */, true /* local */);
      const size_t marker_dims[1] = {1};
      double marker = 42.25;
      stale.write("gpmeep_stale_marker", 1, marker_dims, &marker, false);
    }

    gpu::reset_dispatch_statistics();
    f.output_dft(point, atomic_name);
    const gpu::dft_materialization_statistics recovered_statistics =
        gpu::get_dft_materialization_statistics();
    struct stat recovered_status;
    require(stat(atomic_path.c_str(), &recovered_status) == 0 &&
                (recovered_status.st_dev != sentinel_status.st_dev ||
                 recovered_status.st_ino != sentinel_status.st_ino) &&
                access(atomic_temporary_path.c_str(), F_OK) != 0 &&
                read_bytes(atomic_path) != sentinel &&
                recovered_statistics.cuda_output_calls == 1 &&
                recovered_statistics.cuda_output_points ==
                    frequencies.size() &&
                recovered_statistics.cpu_output_calls == 0 &&
                recovered_statistics.cpu_output_points == 0,
            "collapsed CUDA output did not atomically recover from its "
            "partial temporary");
    {
      h5file recovered(atomic_path.c_str(), h5file::READONLY,
                       false /* parallel */, true /* local */);
      require(recovered.root_object_count() ==
                      2 * frequencies.size() &&
                  !recovered.dataset_exists("gpmeep_stale_marker"),
              "recovered collapsed CUDA output retained a stale dataset");
    }
    require(unlink(atomic_path.c_str()) == 0,
            "collapsed atomic output cleanup failed");
    point.remove();
    thin_x.remove();
    thin_y.remove();
  }

  {
    const grid_volume gv = vol2d(1.8, 1.6, 10.0);
    const symmetry reflection = mirror(X, gv);
    structure s(gv, vacuum, no_pml(), reflection, 4);
    fields f(&s, 0.0, 0.0, true, 64, 64);
    f.use_real_fields();
    gaussian_src_time source(0.27, 0.10);
    f.add_point_source(Ez, source, gv.center(), 0.8);
    const std::vector<double> frequencies = {0.23, 0.31};
    const volume line_volume(vec(1.30, 0.25), vec(1.30, 1.35));
    dft_flux flux = f.add_dft_flux_plane(line_volume, frequencies);
    volume_list force_region(line_volume, Sx, 0.75);
    dft_force force = f.add_dft_force(&force_region, frequencies, 1);
    volume_list near_region(line_volume, Sx, -1.25);
    dft_near2far near =
        f.add_dft_near2far(&near_region, frequencies, 1);
    for (int step = 0; step < 36; ++step) f.step();

    struct collapsed_wrapper_component_oracle {
      component c;
      int rank;
      size_t dims[3];
      size_t point_count;
      std::vector<std::vector<std::complex<realnum>>> values;
    };
    const auto verify_wrapper =
        [&](auto &monitor, const std::vector<dft_chunk *> &chunklists,
            const char *label) {
          std::vector<component> monitor_components;
          for (dft_chunk *chunks : chunklists)
            for (dft_chunk *chunk = chunks; chunk;
                 chunk = chunk->next_in_dft)
              if (std::find(monitor_components.begin(),
                            monitor_components.end(), chunk->c) ==
                  monitor_components.end())
                monitor_components.push_back(chunk->c);
          require(!monitor_components.empty(),
                  std::string(label) +
                      " collapsed output wrapper has no components");

          std::vector<collapsed_wrapper_component_oracle> oracle;
          size_t expected_point_terms = 0;
          for (component c : monitor_components) {
            collapsed_wrapper_component_oracle component_oracle;
            component_oracle.c = c;
            component_oracle.rank = -1;
            component_oracle.dims[0] = component_oracle.dims[1] =
                component_oracle.dims[2] = 0;
            component_oracle.point_count = 0;
            for (size_t frequency = 0; frequency < frequencies.size();
                 ++frequency) {
              int rank = -1;
              size_t dims[3] = {0, 0, 0};
              std::unique_ptr<std::complex<realnum>[]> values(
                  f.get_dft_array(monitor, c,
                                  static_cast<int>(frequency), &rank,
                                  dims));
              require(values != nullptr && rank == 1 && dims[0] > 1,
                      std::string(label) +
                          " collapsed output oracle is not a line");
              if (frequency == 0) {
                component_oracle.rank = rank;
                std::copy(dims, dims + 3, component_oracle.dims);
                component_oracle.point_count = dims[0];
              }
              else
                require(rank == component_oracle.rank &&
                            dims[0] == component_oracle.dims[0],
                        std::string(label) +
                            " oracle geometry changed across frequencies");
              component_oracle.values.emplace_back(
                  values.get(), values.get() + dims[0]);
            }
            expected_point_terms +=
                component_oracle.point_count * frequencies.size();
            oracle.push_back(std::move(component_oracle));
          }

          const std::string output_name =
              std::string("gpu-step-db-public-dft-output-collapsed-") +
              label;
          const std::string output_path =
              f.h5file_name(output_name.c_str());
          const std::string temporary_name =
              output_name + ".gpmeep-tmp";
          const std::string temporary_path =
              f.h5file_name(temporary_name.c_str());
          unlink(output_path.c_str());
          unlink(temporary_path.c_str());

          gpu::reset_dispatch_statistics();
          f.output_dft(monitor, output_name.c_str());
          const gpu::dft_materialization_statistics statistics =
              gpu::get_dft_materialization_statistics();
          const std::uint64_t expected_result_bytes =
              expected_point_terms * sizeof(std::complex<realnum>);
          require(statistics.cuda_output_calls == 1 &&
                      statistics.cuda_output_points ==
                          expected_point_terms &&
                      statistics.cpu_output_calls == 0 &&
                      statistics.cpu_output_points == 0 &&
                      statistics.cpu_array_calls == 0 &&
                      statistics.cuda_array_calls == 0 &&
                      statistics.cuda_kernel_launches > 0 &&
                      statistics.cuda_kernel_launches % 2 == 0 &&
                      statistics.cuda_result_device_to_host_bytes ==
                          expected_result_bytes &&
                      statistics.array_mpi_allreduce_bytes ==
                          expected_result_bytes &&
                      statistics.cuda_output_staging_calls == 0,
                  std::string(label) +
                      " collapsed output wrapper telemetry is invalid");

          h5file output(output_path.c_str(), h5file::READONLY,
                        false /* parallel */, true /* local */);
          require(output.root_object_count() ==
                      2 * oracle.size() * frequencies.size(),
                  std::string(label) +
                      " collapsed output wrapper inventory is invalid");
          double maximum_error = 0.0;
          bool finite_payload = true;
          bool bitwise_equal = true;
          for (const collapsed_wrapper_component_oracle &component_oracle :
               oracle)
            for (size_t frequency = 0; frequency < frequencies.size();
                 ++frequency) {
              int real_rank = -1;
              int imaginary_rank = -1;
              size_t real_dims[3] = {0, 0, 0};
              size_t imaginary_dims[3] = {0, 0, 0};
              const std::string prefix =
                  std::string(component_name(component_oracle.c)) + "_" +
                  std::to_string(frequency);
              std::unique_ptr<double[]> real_values(static_cast<double *>(
                  output.read((prefix + ".r").c_str(), &real_rank,
                              real_dims, 3, false)));
              std::unique_ptr<double[]> imaginary_values(
                  static_cast<double *>(
                      output.read((prefix + ".i").c_str(),
                                  &imaginary_rank, imaginary_dims, 3,
                                  false)));
              require(real_values && imaginary_values &&
                          real_rank == component_oracle.rank &&
                          imaginary_rank == component_oracle.rank &&
                          real_dims[0] == component_oracle.dims[0] &&
                          imaginary_dims[0] == component_oracle.dims[0],
                      std::string(label) +
                          " collapsed HDF5 component geometry is invalid");
              for (size_t point = 0;
                   point < component_oracle.point_count; ++point) {
                const std::complex<double> observed(
                    real_values[point], imaginary_values[point]);
                const std::complex<double> expected(
                    component_oracle.values[frequency][point]);
                const double error = std::abs(observed - expected);
                finite_payload = finite_payload &&
                                 std::isfinite(observed.real()) &&
                                 std::isfinite(observed.imag()) &&
                                 std::isfinite(expected.real()) &&
                                 std::isfinite(expected.imag()) &&
                                 std::isfinite(error);
                if (std::isfinite(error))
                  maximum_error = std::max(maximum_error, error);
                const double expected_real = expected.real();
                const double expected_imaginary = expected.imag();
                bitwise_equal =
                    bitwise_equal &&
                    std::memcmp(&real_values[point], &expected_real,
                                sizeof(double)) == 0 &&
                    std::memcmp(&imaginary_values[point],
                                &expected_imaginary, sizeof(double)) == 0;
              }
            }
          require(finite_payload && maximum_error == 0.0 &&
                      bitwise_equal &&
                      access(temporary_path.c_str(), F_OK) != 0,
                  std::string(label) +
                      " collapsed output differs from its exact oracle");
          require(unlink(output_path.c_str()) == 0,
                  std::string(label) +
                      " collapsed output cleanup failed");
          if (am_master())
            std::cout << "COLLAPSED_DFT_WRAPPER case=" << label
                      << " components=" << oracle.size()
                      << " point_terms=" << expected_point_terms
                      << " kernels=" << statistics.cuda_kernel_launches
                      << " d2h="
                      << statistics.cuda_result_device_to_host_bytes
                      << " max_abs=" << maximum_error << '\n';
        };

    verify_wrapper(flux, {flux.E, flux.H}, "flux-line");
    verify_wrapper(force,
                   {force.offdiag1, force.offdiag2, force.diag},
                   "force-line");
    verify_wrapper(near, {near.F}, "near2far-line");
    flux.remove();
    force.remove();
    near.remove();

    const std::vector<double> no_frequencies;
    dft_fields empty(nullptr, no_frequencies,
                     volume(vec(0.73, 0.61), vec(0.73, 0.61)));
    const char *empty_name =
        "gpu-step-db-public-dft-output-collapsed-empty";
    const std::string empty_path = f.h5file_name(empty_name);
    const std::string empty_temporary_name =
        std::string(empty_name) + ".gpmeep-tmp";
    const std::string empty_temporary_path =
        f.h5file_name(empty_temporary_name.c_str());
    unlink(empty_path.c_str());
    unlink(empty_temporary_path.c_str());

    gpu::set_backend(gpu::backend_mode::cpu);
    gpu::reset_dispatch_statistics();
    f.output_dft(empty, empty_name);
    const gpu::dft_materialization_statistics empty_cpu_statistics =
        gpu::get_dft_materialization_statistics();
    require(access(empty_path.c_str(), F_OK) != 0 &&
                access(empty_temporary_path.c_str(), F_OK) != 0 &&
                empty_cpu_statistics.cpu_output_calls == 1 &&
                empty_cpu_statistics.cpu_output_points == 0 &&
                empty_cpu_statistics.cuda_output_calls == 0,
            "zero-frequency CPU output no-op ABI changed");

    gpu::set_backend(gpu::backend_mode::cuda);
    gpu::reset_dispatch_statistics();
    f.output_dft(empty, empty_name);
    const gpu::dft_materialization_statistics empty_statistics =
        gpu::get_dft_materialization_statistics();
    require(access(empty_path.c_str(), F_OK) != 0 &&
                access(empty_temporary_path.c_str(), F_OK) != 0 &&
                empty_statistics.cuda_output_calls == 1 &&
                empty_statistics.cuda_output_points == 0 &&
                empty_statistics.cpu_output_calls == 0 &&
                empty_statistics.cuda_kernel_launches == 0 &&
                empty_statistics.cuda_result_device_to_host_bytes == 0 &&
                empty_statistics.array_mpi_allreduce_calls == 0,
            "zero-frequency collapsed CUDA output differs from the CPU "
            "no-op ABI or fabricated GPU work");
  }

  {
    const grid_volume gv = vol3d(1.4, 1.2, 1.0, 8.0);
    structure s(gv, vacuum, no_pml(), identity(), 4);
    fields f(&s, 0.0, 0.0, true, 32, 32);
    f.use_real_fields();
    gaussian_src_time source(0.31, 0.11);
    f.add_point_source(Ez, source, vec(0.29, 0.27, 0.23), 0.7);
    component components[] = {Ez};
    const std::vector<double> frequencies = {0.24, 0.31, 0.38};
    dft_fields plane = f.add_dft_fields(
        components, 1,
        volume(vec(0.71, 0.17, 0.13), vec(0.71, 1.03, 0.87)),
        frequencies, true, 1);
    dft_fields line = f.add_dft_fields(
        components, 1,
        volume(vec(0.19, 0.61, 0.43), vec(1.21, 0.61, 0.43)),
        frequencies, true, 1);
    for (int step = 0; step < 28; ++step) f.step();

    int oracle_rank = -1;
    size_t oracle_dims[3] = {0, 0, 0};
    size_t point_count = 0;
    std::vector<std::vector<std::complex<realnum>>> oracle;
    for (size_t frequency = 0; frequency < frequencies.size();
         ++frequency) {
      int rank = -1;
      size_t dims[3] = {0, 0, 0};
      std::unique_ptr<std::complex<realnum>[]> values(
          f.get_dft_array(plane, Ez, static_cast<int>(frequency),
                          &rank, dims));
      require(values != nullptr && rank == 2,
              "3D plane get_dft_array oracle is not rank 2");
      const size_t count = dims[0] * dims[1];
      if (frequency == 0) {
        oracle_rank = rank;
        std::copy(dims, dims + 3, oracle_dims);
        point_count = count;
      }
      else
        require(rank == oracle_rank && dims[0] == oracle_dims[0] &&
                    dims[1] == oracle_dims[1] && count == point_count,
                "3D plane oracle geometry changed across frequencies");
      oracle.emplace_back(values.get(), values.get() + count);
    }

    dft_chunk *chunklists[] = {plane.chunks};
    ivec full_min_corner, full_max_corner;
    size_t full_point_count = 0;
    size_t maximum_chunk_size = 0;
    int full_rank = -1;
    direction full_directions[3];
    size_t full_dims[3] = {0, 0, 0};
    f.get_dft_component_dims(
        chunklists, 1, Ez, full_min_corner, full_max_corner,
        full_point_count, maximum_chunk_size, full_rank, full_directions,
        full_dims);
    require(full_rank == 3 && full_point_count > point_count,
            "3D plane fixture did not expose an uncollapsed axis");

    const std::string output_name =
        "gpu-step-db-public-dft-output-collapsed-plane3d";
    const std::string output_path = f.h5file_name(output_name.c_str());
    const std::string temporary_name = output_name + ".gpmeep-tmp";
    const std::string temporary_path =
        f.h5file_name(temporary_name.c_str());
    unlink(output_path.c_str());
    unlink(temporary_path.c_str());
    gpu::reset_dispatch_statistics();
    f.output_dft(plane, output_name.c_str());
    const gpu::dft_materialization_statistics statistics =
        gpu::get_dft_materialization_statistics();
    const std::uint64_t expected_kernels =
        full_point_count <= 64 ? 2 : 2 * frequencies.size();
    require(statistics.cuda_output_calls == 1 &&
                statistics.cuda_output_points ==
                    point_count * frequencies.size() &&
                statistics.cpu_output_calls == 0 &&
                statistics.cuda_kernel_launches == expected_kernels &&
                statistics.cuda_result_device_to_host_bytes ==
                    point_count * frequencies.size() *
                        sizeof(std::complex<realnum>) &&
                statistics.array_mpi_allreduce_bytes ==
                    point_count * frequencies.size() *
                        sizeof(std::complex<realnum>),
            "3D plane collapsed CUDA output telemetry is not exact");

    h5file output(output_path.c_str(), h5file::READONLY,
                  false /* parallel */, true /* local */);
    require(output.root_object_count() == 2 * frequencies.size(),
            "3D plane collapsed HDF5 inventory is not exact");
    double maximum_error = 0.0;
    bool bitwise_equal = true;
    bool finite_payload = true;
    for (size_t frequency = 0; frequency < frequencies.size();
         ++frequency) {
      int real_rank = -1;
      int imaginary_rank = -1;
      size_t real_dims[3] = {0, 0, 0};
      size_t imaginary_dims[3] = {0, 0, 0};
      const std::string real_name =
          "ez_" + std::to_string(frequency) + ".r";
      const std::string imaginary_name =
          "ez_" + std::to_string(frequency) + ".i";
      std::unique_ptr<double[]> real_values(static_cast<double *>(
          output.read(real_name.c_str(), &real_rank, real_dims, 3, false)));
      std::unique_ptr<double[]> imaginary_values(static_cast<double *>(
          output.read(imaginary_name.c_str(), &imaginary_rank,
                      imaginary_dims, 3, false)));
      require(real_values && imaginary_values && real_rank == oracle_rank &&
                  imaginary_rank == oracle_rank &&
                  real_dims[0] == oracle_dims[0] &&
                  real_dims[1] == oracle_dims[1] &&
                  imaginary_dims[0] == oracle_dims[0] &&
                  imaginary_dims[1] == oracle_dims[1],
              "3D plane collapsed HDF5 geometry differs from oracle");
      for (size_t point = 0; point < point_count; ++point) {
        const std::complex<double> observed(real_values[point],
                                             imaginary_values[point]);
        const std::complex<double> expected(oracle[frequency][point]);
        const double error = std::abs(observed - expected);
        finite_payload = finite_payload &&
                         std::isfinite(observed.real()) &&
                         std::isfinite(observed.imag()) &&
                         std::isfinite(expected.real()) &&
                         std::isfinite(expected.imag()) &&
                         std::isfinite(error);
        if (std::isfinite(error))
          maximum_error = std::max(maximum_error, error);
        const double expected_real = expected.real();
        const double expected_imaginary = expected.imag();
        bitwise_equal =
            bitwise_equal &&
            std::memcmp(&real_values[point], &expected_real,
                        sizeof(double)) == 0 &&
            std::memcmp(&imaginary_values[point],
                        &expected_imaginary, sizeof(double)) == 0;
      }
    }
    require(finite_payload && maximum_error <= 2e-6 && bitwise_equal,
            "3D plane device collapse differs from host collapse oracle");
    require(access(temporary_path.c_str(), F_OK) != 0 &&
                unlink(output_path.c_str()) == 0,
            "3D plane collapsed output cleanup failed");
    if (am_master())
      std::cout << "COLLAPSED_DFT_OUTPUT case=plane3d rank="
                << oracle_rank << " points=" << point_count
                << " full_points=" << full_point_count
                << " frequencies=" << frequencies.size()
                << " kernels=" << statistics.cuda_kernel_launches
                << " d2h="
                << statistics.cuda_result_device_to_host_bytes
                << " mpi_bytes=" << statistics.array_mpi_allreduce_bytes
                << " max_abs=" << maximum_error
                << " bitwise=" << (bitwise_equal ? 1 : 0) << '\n';

    int line_rank = -1;
    size_t line_dims[3] = {0, 0, 0};
    size_t line_point_count = 0;
    std::vector<std::vector<std::complex<realnum>>> line_oracle;
    for (size_t frequency = 0; frequency < frequencies.size();
         ++frequency) {
      int rank = -1;
      size_t dims[3] = {0, 0, 0};
      std::unique_ptr<std::complex<realnum>[]> values(
          f.get_dft_array(line, Ez, static_cast<int>(frequency),
                          &rank, dims));
      require(values != nullptr && rank == 1 && dims[0] > 1,
              "3D line get_dft_array oracle is not rank 1");
      if (frequency == 0) {
        line_rank = rank;
        std::copy(dims, dims + 3, line_dims);
        line_point_count = dims[0];
      }
      else
        require(rank == line_rank && dims[0] == line_dims[0],
                "3D line oracle geometry changed across frequencies");
      line_oracle.emplace_back(values.get(), values.get() + dims[0]);
    }

    dft_chunk *line_chunklists[] = {line.chunks};
    ivec line_full_min_corner, line_full_max_corner;
    size_t line_full_point_count = 0;
    size_t line_maximum_chunk_size = 0;
    int line_full_rank = -1;
    direction line_full_directions[3];
    size_t line_full_dims[3] = {0, 0, 0};
    f.get_dft_component_dims(
        line_chunklists, 1, Ez, line_full_min_corner,
        line_full_max_corner, line_full_point_count,
        line_maximum_chunk_size, line_full_rank, line_full_directions,
        line_full_dims);
    require(line_full_rank == 3 &&
                line_full_point_count >= 4 * line_point_count,
            "3D line fixture did not expose two collapsed axes");

    const std::string line_output_name =
        "gpu-step-db-public-dft-output-collapsed-line3d";
    const std::string line_output_path =
        f.h5file_name(line_output_name.c_str());
    const std::string line_temporary_name =
        line_output_name + ".gpmeep-tmp";
    const std::string line_temporary_path =
        f.h5file_name(line_temporary_name.c_str());
    unlink(line_output_path.c_str());
    unlink(line_temporary_path.c_str());
    gpu::reset_dispatch_statistics();
    f.output_dft(line, line_output_name.c_str());
    const gpu::dft_materialization_statistics line_statistics =
        gpu::get_dft_materialization_statistics();
    const std::uint64_t line_reduced_bytes =
        line_point_count * frequencies.size() *
        sizeof(std::complex<realnum>);
    require(line_statistics.cuda_output_calls == 1 &&
                line_statistics.cuda_output_points ==
                    line_point_count * frequencies.size() &&
                line_statistics.cpu_output_calls == 0 &&
                line_statistics.cuda_output_staging_calls == 0 &&
                line_statistics.cuda_kernel_launches > 0 &&
                line_statistics.cuda_result_device_to_host_bytes ==
                    line_reduced_bytes &&
                line_statistics.array_mpi_allreduce_calls ==
                    frequencies.size() &&
                line_statistics.array_mpi_allreduce_bytes ==
                    line_reduced_bytes,
            "3D two-axis line collapse telemetry is not exact");

    bool line_valid = true;
    bool line_bitwise = true;
    double line_maximum_error = 0.0;
    {
      h5file line_output(line_output_path.c_str(), h5file::READONLY,
                         false /* parallel */, true /* local */);
      line_valid =
          line_output.root_object_count() == 2 * frequencies.size();
      for (size_t frequency = 0; frequency < frequencies.size();
           ++frequency) {
        int real_rank = -1;
        int imaginary_rank = -1;
        size_t real_dims[3] = {0, 0, 0};
        size_t imaginary_dims[3] = {0, 0, 0};
        const std::string real_name =
            "ez_" + std::to_string(frequency) + ".r";
        const std::string imaginary_name =
            "ez_" + std::to_string(frequency) + ".i";
        std::unique_ptr<double[]> real_values(static_cast<double *>(
            line_output.read(real_name.c_str(), &real_rank, real_dims, 3,
                             false)));
        std::unique_ptr<double[]> imaginary_values(static_cast<double *>(
            line_output.read(imaginary_name.c_str(), &imaginary_rank,
                             imaginary_dims, 3, false)));
        line_valid = line_valid && real_values && imaginary_values &&
                     real_rank == line_rank &&
                     imaginary_rank == line_rank &&
                     real_dims[0] == line_dims[0] &&
                     imaginary_dims[0] == line_dims[0];
        if (real_values && imaginary_values)
          for (size_t point = 0; point < line_point_count; ++point) {
            const std::complex<double> expected(
                line_oracle[frequency][point]);
            const std::complex<double> observed(
                real_values[point], imaginary_values[point]);
            const double error = std::abs(observed - expected);
            line_valid = line_valid &&
                         std::isfinite(observed.real()) &&
                         std::isfinite(observed.imag()) &&
                         std::isfinite(expected.real()) &&
                         std::isfinite(expected.imag()) &&
                         std::isfinite(error);
            if (std::isfinite(error))
              line_maximum_error =
                  std::max(line_maximum_error, error);
            const double expected_real = expected.real();
            const double expected_imaginary = expected.imag();
            line_bitwise =
                line_bitwise &&
                std::memcmp(&real_values[point], &expected_real,
                            sizeof(double)) == 0 &&
                std::memcmp(&imaginary_values[point],
                            &expected_imaginary, sizeof(double)) == 0;
          }
      }
    }
    require(line_valid && line_bitwise && line_maximum_error == 0.0 &&
                access(line_temporary_path.c_str(), F_OK) != 0 &&
                unlink(line_output_path.c_str()) == 0,
            "3D two-axis line collapse differs from its public oracle");
    if (am_master())
      std::cout << "COLLAPSED_DFT_OUTPUT case=line3d rank="
                << line_rank << " points=" << line_point_count
                << " full_points=" << line_full_point_count
                << " frequencies=" << frequencies.size()
                << " kernels=" << line_statistics.cuda_kernel_launches
                << " d2h="
                << line_statistics.cuda_result_device_to_host_bytes
                << " mpi_bytes="
                << line_statistics.array_mpi_allreduce_bytes
                << " max_abs=" << line_maximum_error
                << " bitwise=" << line_bitwise << '\n';
    line.remove();
    plane.remove();
  }

  gpu::set_backend(gpu::backend_mode::cpu);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "public collapsed DFT output leaked resident device buffers");
  if (am_master())
    std::cout
        << "PASS: public CUDA point/line output_dft used deterministic "
           "device collapse, reduced D2H/MPI, FP64 HDF5, and atomic "
           "publication\n";
}

void require_public_collapsed_dft_output_zero_owner_mpi() {
  require(count_processors() == 2,
          "collapsed DFT output zero-owner regression requires two ranks");
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  gpu::set_backend(gpu::backend_mode::cuda);

  {
    const grid_volume gv = vol2d(1.6, 1.4, 12.0);
    structure s(gv, vacuum, no_pml(), identity(), 4);
    fields f(&s, 0.0, 0.0, true, 32, 32);
    f.use_real_fields();
    gaussian_src_time source(0.29, 0.10);
    f.add_point_source(Ez, source, vec(0.31, 0.27), 0.8);
    component components[] = {Ez};
    const std::vector<double> frequencies = {0.23, 0.29, 0.35};
    dft_fields monitor = f.add_dft_fields(
        components, 1,
        volume(vec(1.29, 1.17), vec(1.29, 1.17)), frequencies,
        true, 1);
    dft_fields split_line = f.add_dft_fields(
        components, 1,
        volume(vec(0.05, 0.70), vec(1.55, 0.70)), frequencies,
        true, 1);
    const std::uint64_t local_owner =
        dft_chunk_count(monitor.chunks, true) > 0 ? 1u : 0u;
    require(global_sum(local_owner) == 1,
            "collapsed MPI point fixture must have exactly one owner");
    const std::uint64_t local_line_owner =
        dft_chunk_count(split_line.chunks, true) > 0 ? 1u : 0u;
    require(global_sum(local_line_owner) == 2,
            "collapsed MPI line fixture must span both ranks");
    for (int step = 0; step < 30; ++step) f.step();

    std::vector<std::complex<realnum>> oracle;
    for (size_t frequency = 0; frequency < frequencies.size();
         ++frequency) {
      int rank = -1;
      size_t dims[3] = {0, 0, 0};
      std::unique_ptr<std::complex<realnum>[]> values(
          f.get_dft_array(monitor, Ez, static_cast<int>(frequency),
                          &rank, dims));
      require(values != nullptr && rank == 0,
              "collapsed MPI point oracle is not scalar");
      oracle.push_back(values[0]);
    }

    const char *output_name =
        "gpu-step-db-public-dft-output-collapsed-mpi-point";
    const std::string output_path = f.h5file_name(output_name);
    const std::string temporary_name =
        std::string(output_name) + ".gpmeep-tmp";
    const std::string temporary_path =
        f.h5file_name(temporary_name.c_str());
    if (am_master()) {
      unlink(output_path.c_str());
      unlink(temporary_path.c_str());
    }
    all_wait();

    gpu::reset_dispatch_statistics();
    f.output_dft(monitor, output_name);
    const gpu::dft_materialization_statistics statistics =
        gpu::get_dft_materialization_statistics();
    const bool local_statistics_valid =
        statistics.cuda_output_calls == 1 &&
        statistics.cuda_output_points == frequencies.size() &&
        statistics.cpu_output_calls == 0 &&
        statistics.cpu_output_points == 0 &&
        statistics.cpu_array_calls == 0 &&
        statistics.cuda_array_calls == 0 &&
        statistics.array_mpi_allreduce_calls == frequencies.size() + 1 &&
        statistics.array_mpi_allreduce_bytes ==
            frequencies.size() * sizeof(std::complex<realnum>) +
                sizeof(float) &&
        (local_owner
             ? statistics.cuda_kernel_launches == 2 &&
                   statistics.cuda_result_device_to_host_bytes ==
                       frequencies.size() *
                           sizeof(std::complex<realnum>)
             : statistics.cuda_kernel_launches == 0 &&
                   statistics.cuda_result_device_to_host_bytes == 0);
    require(local_statistics_valid,
            "collapsed MPI point owner/zero-owner telemetry is not exact");

    bool artifact_valid = true;
    bool point_bitwise = true;
    double maximum_error = 0.0;
    if (am_master()) {
      try {
        h5file output(output_path.c_str(), h5file::READONLY,
                      false /* parallel */, true /* local */);
        artifact_valid =
            output.root_object_count() == 2 * frequencies.size();
        for (size_t frequency = 0; frequency < frequencies.size();
             ++frequency) {
          int real_rank = -1;
          int imaginary_rank = -1;
          size_t real_dims[3] = {0, 0, 0};
          size_t imaginary_dims[3] = {0, 0, 0};
          const std::string real_name =
              "ez_" + std::to_string(frequency) + ".r";
          const std::string imaginary_name =
              "ez_" + std::to_string(frequency) + ".i";
          std::unique_ptr<double[]> real_values(static_cast<double *>(
              output.read(real_name.c_str(), &real_rank, real_dims, 3,
                          false)));
          std::unique_ptr<double[]> imaginary_values(static_cast<double *>(
              output.read(imaginary_name.c_str(), &imaginary_rank,
                          imaginary_dims, 3, false)));
          artifact_valid =
              artifact_valid && real_values && imaginary_values &&
              real_rank == 0 && imaginary_rank == 0 &&
              real_dims[0] == 1 && imaginary_dims[0] == 1;
          if (real_values && imaginary_values) {
            const std::complex<double> observed(real_values[0],
                                                 imaginary_values[0]);
            const std::complex<double> expected(oracle[frequency]);
            const double expected_real = expected.real();
            const double expected_imaginary = expected.imag();
            const double error = std::abs(observed - expected);
            point_bitwise =
                point_bitwise &&
                std::memcmp(&real_values[0], &expected_real,
                            sizeof(expected_real)) == 0 &&
                std::memcmp(&imaginary_values[0], &expected_imaginary,
                            sizeof(expected_imaginary)) == 0;
            artifact_valid = artifact_valid &&
                             std::isfinite(observed.real()) &&
                             std::isfinite(observed.imag()) &&
                             std::isfinite(expected.real()) &&
                             std::isfinite(expected.imag()) &&
                             std::isfinite(error);
            if (std::isfinite(error))
              maximum_error = std::max(maximum_error, error);
          }
        }
        artifact_valid = artifact_valid && point_bitwise &&
                         maximum_error == 0.0 &&
                         access(temporary_path.c_str(), F_OK) != 0;
      }
      catch (...) { artifact_valid = false; }
    }
    require(artifact_valid,
            "collapsed MPI point HDF5 differs from scalar oracle");
    bool cleanup_valid = true;
    if (am_master()) cleanup_valid = unlink(output_path.c_str()) == 0;
    require(cleanup_valid, "collapsed MPI point output cleanup failed");
    all_wait();

    // Exercise a real H5Fcreate failure on rank zero while its peer waits in
    // the failure collective.  The checked serial HDF5 path must unwind both
    // ranks synchronously without reaching CUDA materialization or replacing
    // any final artifact.
    bool temporary_blocker_created = true;
    if (am_master())
      temporary_blocker_created =
          mkdir(temporary_path.c_str(), 0700) == 0;
    require(temporary_blocker_created,
            "could not create collapsed MPI temporary-file blocker");
    all_wait();
    gpu::reset_dispatch_statistics();
    bool open_failure_observed = false;
    try { f.output_dft(monitor, output_name); }
    catch (const std::runtime_error &error) {
      open_failure_observed =
          std::string(error.what()).find(
              "Meep CUDA collapsed DFT output could not create its "
              "temporary HDF5 file") != std::string::npos;
    }
    const gpu::dft_materialization_statistics open_failure_statistics =
        gpu::get_dft_materialization_statistics();
    require(open_failure_observed &&
                open_failure_statistics.cuda_output_calls == 0 &&
                open_failure_statistics.cuda_output_points == 0 &&
                open_failure_statistics.cpu_output_calls == 0 &&
                open_failure_statistics.cuda_kernel_launches == 0 &&
                open_failure_statistics.cuda_result_device_to_host_bytes ==
                    0 &&
                open_failure_statistics.array_mpi_allreduce_calls == 1 &&
                open_failure_statistics.array_mpi_allreduce_bytes ==
                    sizeof(float),
            "collapsed MPI HDF5-open failure was not synchronous and "
            "failure-atomic");
    bool open_failure_namespace_valid = true;
    if (am_master()) {
      struct stat blocker_status;
      open_failure_namespace_valid =
          access(output_path.c_str(), F_OK) != 0 &&
          stat(temporary_path.c_str(), &blocker_status) == 0 &&
          S_ISDIR(blocker_status.st_mode);
    }
    require(open_failure_namespace_valid,
            "collapsed MPI HDF5-open failure changed the output namespace");
    bool temporary_blocker_removed = true;
    if (am_master())
      temporary_blocker_removed = rmdir(temporary_path.c_str()) == 0;
    require(temporary_blocker_removed,
            "could not remove collapsed MPI temporary-file blocker");
    all_wait();

    bool blocker_created = true;
    if (am_master()) blocker_created = mkdir(output_path.c_str(), 0700) == 0;
    require(blocker_created,
            "could not create collapsed MPI rename blocker");
    all_wait();
    gpu::reset_dispatch_statistics();
    bool rename_failure_observed = false;
    try { f.output_dft(monitor, output_name); }
    catch (const std::runtime_error &error) {
      rename_failure_observed =
          std::string(error.what()).find(
              "Meep CUDA collapsed DFT output could not atomically publish "
              "its HDF5 file") != std::string::npos;
    }
    const gpu::dft_materialization_statistics failed_statistics =
        gpu::get_dft_materialization_statistics();
    require(rename_failure_observed &&
                failed_statistics.cuda_output_calls == 0 &&
                failed_statistics.cuda_output_points == 0 &&
                failed_statistics.cpu_output_calls == 0,
            "collapsed MPI rename failure was not synchronous or committed "
            "success telemetry");
    bool failure_namespace_valid = true;
    if (am_master()) {
      struct stat blocker_status;
      struct stat temporary_status;
      failure_namespace_valid =
          stat(output_path.c_str(), &blocker_status) == 0 &&
          S_ISDIR(blocker_status.st_mode) &&
          stat(temporary_path.c_str(), &temporary_status) == 0 &&
          S_ISREG(temporary_status.st_mode) && temporary_status.st_size > 0;
    }
    require(failure_namespace_valid,
            "collapsed MPI rename failure lost its blocker or complete "
            "temporary");
    bool blocker_removed = true;
    if (am_master()) blocker_removed = rmdir(output_path.c_str()) == 0;
    require(blocker_removed,
            "could not remove collapsed MPI rename blocker");
    all_wait();

    gpu::reset_dispatch_statistics();
    f.output_dft(monitor, output_name);
    const gpu::dft_materialization_statistics recovered_statistics =
        gpu::get_dft_materialization_statistics();
    bool recovery_valid =
        recovered_statistics.cuda_output_calls == 1 &&
        recovered_statistics.cuda_output_points == frequencies.size() &&
        recovered_statistics.cpu_output_calls == 0 &&
        recovered_statistics.array_mpi_allreduce_calls ==
            frequencies.size() + 1 &&
        recovered_statistics.array_mpi_allreduce_bytes ==
            frequencies.size() * sizeof(std::complex<realnum>) +
                sizeof(float);
    if (am_master()) {
      struct stat recovered_status;
      recovery_valid =
          recovery_valid && stat(output_path.c_str(), &recovered_status) == 0 &&
          S_ISREG(recovered_status.st_mode) &&
          access(temporary_path.c_str(), F_OK) != 0;
    }
    require(recovery_valid,
            "collapsed MPI output did not recover after rename failure");
    bool recovered_removed = true;
    if (am_master()) recovered_removed = unlink(output_path.c_str()) == 0;
    require(recovered_removed,
            "collapsed MPI recovered output cleanup failed");
    all_wait();

    int line_rank = -1;
    size_t line_dims[3] = {0, 0, 0};
    size_t line_point_count = 0;
    std::vector<std::vector<std::complex<realnum>>> line_oracle;
    for (size_t frequency = 0; frequency < frequencies.size();
         ++frequency) {
      int rank = -1;
      size_t dims[3] = {0, 0, 0};
      std::unique_ptr<std::complex<realnum>[]> values(
          f.get_dft_array(split_line, Ez, static_cast<int>(frequency),
                          &rank, dims));
      require(values != nullptr && rank == 1 && dims[0] > 1,
              "collapsed MPI split-line oracle is not a nontrivial vector");
      if (frequency == 0) {
        line_rank = rank;
        line_dims[0] = dims[0];
        line_point_count = dims[0];
      }
      else
        require(rank == line_rank && dims[0] == line_dims[0],
                "collapsed MPI split-line geometry changed across "
                "frequencies");
      line_oracle.emplace_back(values.get(), values.get() + dims[0]);
    }

    dft_chunk *line_chunklists[] = {split_line.chunks};
    ivec line_min_corner;
    ivec line_max_corner;
    size_t line_full_point_count = 0;
    size_t line_maximum_chunk_size = 0;
    int line_full_rank = 0;
    direction line_full_directions[3];
    size_t line_full_dims[3] = {1, 1, 1};
    f.get_dft_component_dims(
        line_chunklists, 1, Ez, line_min_corner, line_max_corner,
        line_full_point_count, line_maximum_chunk_size, line_full_rank,
        line_full_directions, line_full_dims);
    require(line_full_rank > line_rank &&
                line_full_point_count > line_point_count,
            "collapsed MPI split-line fixture has no dense dimension");

    const char *line_output_name =
        "gpu-step-db-public-dft-output-collapsed-mpi-line";
    const std::string line_output_path =
        f.h5file_name(line_output_name);
    const std::string line_temporary_name =
        std::string(line_output_name) + ".gpmeep-tmp";
    const std::string line_temporary_path =
        f.h5file_name(line_temporary_name.c_str());
    bool line_setup_valid = true;
    if (am_master()) {
      errno = 0;
      line_setup_valid =
          (unlink(line_output_path.c_str()) == 0 || errno == ENOENT);
      errno = 0;
      line_setup_valid =
          line_setup_valid &&
          (unlink(line_temporary_path.c_str()) == 0 || errno == ENOENT);
    }
    require(line_setup_valid,
            "collapsed MPI split-line output setup failed");
    all_wait();

    gpu::reset_dispatch_statistics();
    f.output_dft(split_line, line_output_name);
    const gpu::dft_materialization_statistics line_statistics =
        gpu::get_dft_materialization_statistics();
    const std::uint64_t line_result_bytes =
        line_full_point_count * frequencies.size() *
        sizeof(std::complex<realnum>);
    std::cout << "COLLAPSED_DFT_SHARED_FIBER rank=" << my_rank()
              << " reduced_points=" << line_point_count
              << " full_points=" << line_full_point_count
              << " kernels=" << line_statistics.cuda_kernel_launches
              << " d2h="
              << line_statistics.cuda_result_device_to_host_bytes
              << " mpi_calls="
              << line_statistics.array_mpi_allreduce_calls
              << " mpi_bytes="
              << line_statistics.array_mpi_allreduce_bytes << '\n';
    require(line_statistics.cuda_output_calls == 1 &&
                line_statistics.cuda_output_points ==
                    line_point_count * frequencies.size() &&
                line_statistics.cpu_output_calls == 0 &&
                line_statistics.cpu_output_points == 0 &&
                line_statistics.cpu_array_calls == 0 &&
                line_statistics.cuda_array_calls == 0 &&
                // The immediately preceding public get_dft_array oracle
                // populated the identical dense all-frequency cache.  The
                // shared-fiber output must reuse it without launching a new
                // materialization kernel, while still transferring and
                // reducing the full dense array before host collapse.
                line_statistics.cuda_kernel_launches == 0 &&
                line_statistics.cuda_result_device_to_host_bytes ==
                    line_result_bytes &&
                line_statistics.array_mpi_allreduce_calls ==
                    frequencies.size() + 1 &&
                line_statistics.array_mpi_allreduce_bytes ==
                    line_result_bytes + line_point_count * sizeof(float),
            "collapsed MPI shared-fiber dense-collective telemetry is not "
            "exact on both owner ranks");

    bool line_artifact_valid = true;
    bool line_bitwise = true;
    double line_maximum_error = 0.0;
    if (am_master()) {
      try {
        h5file output(line_output_path.c_str(), h5file::READONLY,
                      false /* parallel */, true /* local */);
        line_artifact_valid =
            output.root_object_count() == 2 * frequencies.size();
        for (size_t frequency = 0; frequency < frequencies.size();
             ++frequency) {
          int real_rank = -1;
          int imaginary_rank = -1;
          size_t real_dims[3] = {0, 0, 0};
          size_t imaginary_dims[3] = {0, 0, 0};
          const std::string real_name =
              "ez_" + std::to_string(frequency) + ".r";
          const std::string imaginary_name =
              "ez_" + std::to_string(frequency) + ".i";
          std::unique_ptr<double[]> real_values(static_cast<double *>(
              output.read(real_name.c_str(), &real_rank, real_dims, 3,
                          false)));
          std::unique_ptr<double[]> imaginary_values(static_cast<double *>(
              output.read(imaginary_name.c_str(), &imaginary_rank,
                          imaginary_dims, 3, false)));
          line_artifact_valid =
              line_artifact_valid && real_values && imaginary_values &&
              real_rank == line_rank && imaginary_rank == line_rank &&
              real_dims[0] == line_dims[0] &&
              imaginary_dims[0] == line_dims[0];
          if (real_values && imaginary_values)
            for (size_t point = 0; point < line_point_count; ++point) {
              const std::complex<double> observed(
                  real_values[point], imaginary_values[point]);
              const std::complex<double> expected(
                  line_oracle[frequency][point]);
              const double error = std::abs(observed - expected);
              const double expected_real = expected.real();
              const double expected_imaginary = expected.imag();
              line_artifact_valid = line_artifact_valid &&
                                    std::isfinite(observed.real()) &&
                                    std::isfinite(observed.imag()) &&
                                    std::isfinite(expected.real()) &&
                                    std::isfinite(expected.imag()) &&
                                    std::isfinite(error);
              line_bitwise =
                  line_bitwise &&
                  std::memcmp(&real_values[point], &expected_real,
                              sizeof(double)) == 0 &&
                  std::memcmp(&imaginary_values[point],
                              &expected_imaginary, sizeof(double)) == 0;
              if (std::isfinite(error))
                line_maximum_error =
                    std::max(line_maximum_error, error);
            }
        }
        line_artifact_valid =
            line_artifact_valid && line_maximum_error == 0.0 &&
            line_bitwise &&
            access(line_temporary_path.c_str(), F_OK) != 0;
      }
      catch (...) { line_artifact_valid = false; }
    }
    require(line_artifact_valid,
            "collapsed MPI split-line HDF5 differs from its global oracle");
    bool line_cleanup_valid = true;
    if (am_master())
      line_cleanup_valid = unlink(line_output_path.c_str()) == 0;
    require(line_cleanup_valid,
            "collapsed MPI split-line output cleanup failed");
    all_wait();
    split_line.remove();
    monitor.remove();

    std::cout << "COLLAPSED_DFT_OUTPUT_MPI rank=" << my_rank()
              << " owner=" << local_owner
              << " kernels=" << statistics.cuda_kernel_launches
              << " d2h="
              << statistics.cuda_result_device_to_host_bytes
              << " mpi_bytes=" << statistics.array_mpi_allreduce_bytes
              << '\n';
    if (am_master())
      std::cout
          << "PASS: 2-GPU collapsed point output used one owner, zero peer "
             "CUDA work, reduced collectives, an exact scalar HDF5 oracle, "
             "and synchronous rename recovery; a split line used both GPU "
             "owners with a CPU-order dense collective and exact global "
             "HDF5; point_max_abs="
          << maximum_error << " point_bitwise=" << point_bitwise
          << " line_points=" << line_point_count
          << " line_full_points=" << line_full_point_count
          << " line_max_abs=" << line_maximum_error
          << " line_bitwise=" << line_bitwise << '\n';
  }

  gpu::set_backend(gpu::backend_mode::cpu);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "collapsed MPI point output leaked resident device buffers");
}

void require_dft_array_materialization_equivalence() {
  require(count_processors() == 1,
          "focused DFT materialization oracle is singleton-only");
  require_low_level_dft_materialization_cache_invalidation();
  const dft_array_materialization_result cpu =
      run_dft_array_materialization_case(gpu::backend_mode::cpu);
  const dft_array_materialization_result cuda =
      run_dft_array_materialization_case(gpu::backend_mode::cuda);
  require(cpu.values.size() == cuda.values.size(),
          "CPU/CUDA point DFT array result counts differ");
  for (std::size_t index = 0; index < cpu.values.size(); ++index)
    require(std::abs(cpu.values[index] - cuda.values[index]) <=
                3e-4 * std::max(1.0, std::abs(cpu.values[index])),
            "CPU/CUDA point DFT arrays disagree");
  constexpr std::uint64_t expected_calls = 4 * 7;
  require(cpu.statistics.cpu_array_calls == expected_calls &&
              cpu.statistics.cpu_array_points == expected_calls &&
              cpu.statistics.cuda_array_calls == 0,
          "CPU scalar DFT array telemetry is incorrect");
  require(cuda.statistics.cuda_array_calls == expected_calls &&
              cuda.statistics.cuda_array_points == expected_calls &&
              cuda.statistics.cpu_array_calls == 0 &&
              cuda.statistics.cuda_kernel_launches == 3 &&
              cuda.statistics.cuda_result_device_to_host_bytes >=
                  expected_calls * sizeof(std::complex<realnum>) &&
              cuda.statistics.full_dft_device_to_host_bytes_avoided >
                  cuda.statistics.cuda_result_device_to_host_bytes &&
              cuda.statistics.array_mpi_allreduce_calls == expected_calls &&
              cuda.statistics.array_mpi_allreduce_bytes ==
                  cuda.statistics.cuda_result_device_to_host_bytes,
          "CUDA point DFT array did not use cached resident materialization: "
          "calls=" + std::to_string(cuda.statistics.cuda_array_calls) +
          " points=" + std::to_string(cuda.statistics.cuda_array_points) +
          " cpu_calls=" + std::to_string(cuda.statistics.cpu_array_calls) +
          " kernels=" + std::to_string(cuda.statistics.cuda_kernel_launches) +
          " d2h=" +
              std::to_string(
                  cuda.statistics.cuda_result_device_to_host_bytes) +
          " avoided=" +
              std::to_string(
                  cuda.statistics.full_dft_device_to_host_bytes_avoided) +
          " mpi_calls=" +
              std::to_string(cuda.statistics.array_mpi_allreduce_calls) +
          " mpi_bytes=" +
              std::to_string(cuda.statistics.array_mpi_allreduce_bytes));
  if (am_master())
    std::cout
        << "DFT_MATERIALIZATION_TELEMETRY cuda_calls="
        << cuda.statistics.cuda_array_calls
        << " cuda_points=" << cuda.statistics.cuda_array_points
        << " cpu_calls=" << cuda.statistics.cpu_array_calls
        << " kernels=" << cuda.statistics.cuda_kernel_launches
        << " result_d2h_bytes="
        << cuda.statistics.cuda_result_device_to_host_bytes
        << " full_dft_d2h_avoided_bytes="
        << cuda.statistics.full_dft_device_to_host_bytes_avoided
        << " mpi_allreduce_calls="
        << cuda.statistics.array_mpi_allreduce_calls
        << " mpi_allreduce_bytes="
        << cuda.statistics.array_mpi_allreduce_bytes << '\n';
}

void require_host_synthetic_material_query_without_dft_sync() {
  require(count_processors() == 1,
          "host synthetic material query regression is singleton-only");
  gpu::set_backend(gpu::backend_mode::cuda);
  const grid_volume gv = vol2d(1.6, 1.4, 12.0);
  structure s(gv, vacuum, no_pml(), identity(), 4);
  fields f(&s, 0.0, 0.0, true, 32, 32);
  f.use_real_fields();
  gaussian_src_time source(0.29, 0.10);
  f.add_point_source(Ez, source, vec(0.13, -0.17), 0.8);
  component components[] = {Ez};
  const volume point_monitor(vec(0.61, 0.53), vec(0.61, 0.53));
  dft_fields monitor = f.add_dft_fields(
      components, 1, point_monitor, 0.24, 0.24, 1, true, 1);
  for (int step = 0; step < 24; ++step) f.step();

  gpu::reset_dispatch_statistics();
  for (component material : {Dielectric, Permeability}) {
    int rank = -1;
    size_t dims[3] = {0, 0, 0};
    std::unique_ptr<std::complex<realnum>[]> values(
        f.get_dft_array(monitor, material, 0, &rank, dims));
    require(values != nullptr && rank == 0,
            "synthetic material point query did not collapse to a scalar");
    require(std::abs(std::complex<double>(values[0]) -
                     std::complex<double>(1.0, 0.0)) <= 2e-6,
            "synthetic vacuum material query returned the wrong value");
  }
  const gpu::dft_materialization_statistics statistics =
      gpu::get_dft_materialization_statistics();
  require(statistics.host_synthetic_material_array_calls == 2 &&
              statistics.host_synthetic_material_array_points == 2 &&
              statistics.cpu_array_calls == 0 &&
              statistics.cuda_array_calls == 0 &&
              statistics.cuda_kernel_launches == 0 &&
              statistics.cuda_result_device_to_host_bytes == 0,
          "synthetic material queries were misclassified as DFT work");
  require(gpu::get_dispatch_statistics().device_to_host_bytes == 0,
          "synthetic material query published a device-authoritative DFT "
          "cache");
  monitor.remove();
  gpu::set_backend(gpu::backend_mode::cpu);
}

void require_cylindrical_axis_zero_measure_materialization() {
  require(count_processors() == 1,
          "cylindrical-axis DFT materialization is singleton-only");
  gpu::set_backend(gpu::backend_mode::cpu);
  const grid_volume gv = volcyl(1.2, 1.0, 12.0);
  structure s(gv, vacuum, no_pml(), identity(), 2, 0.4);
  fields f(&s, 0.0, 0.0, true, 32, 32);
  f.use_real_fields();
  f.require_component(Ep);
  const volume axis_monitor(veccyl(0.0, 0.23), veccyl(0.0, 0.77));
  const std::vector<double> frequencies = {0.23, 0.27, 0.31};
  dft_chunk *chunks = f.add_dft(
      Ep, axis_monitor, frequencies, true, 1.0, nullptr, false, 1.0,
      true, 0, 1, false);
  dft_fields monitor(chunks, frequencies, axis_monitor);
  for (int step = 0; step < 4; ++step) f.step();

  int cpu_rank = -1;
  size_t cpu_dims[3] = {0, 0, 0};
  std::vector<std::complex<realnum> > cpu_values;
  size_t point_count = 0;
  for (size_t frequency = 0; frequency < frequencies.size(); ++frequency) {
    int rank = -1;
    size_t dims[3] = {0, 0, 0};
    std::unique_ptr<std::complex<realnum>[]> values(
        f.get_dft_array(monitor, Ep, static_cast<int>(frequency),
                        &rank, dims));
    require(values != nullptr,
            "cylindrical-axis CPU DFT array was empty");
    const size_t count =
        rank > 0
            ? dims[0] * (rank >= 2 ? dims[1] : 1) *
                  (rank == 3 ? dims[2] : 1)
            : 1;
    if (frequency == 0) {
      cpu_rank = rank;
      std::copy(dims, dims + 3, cpu_dims);
      point_count = count;
    }
    else {
      require(rank == cpu_rank && count == point_count,
              "cylindrical-axis CPU spectrum changed array geometry");
      for (int dim = 0; dim < rank; ++dim)
        require(dims[dim] == cpu_dims[dim],
                "cylindrical-axis CPU spectrum changed dimensions");
    }
    cpu_values.insert(cpu_values.end(), values.get(), values.get() + count);
  }

  gpu::set_backend(gpu::backend_mode::cuda);
  gpu::reset_dispatch_statistics();
  const auto capture_cuda_spectrum = [&]() {
    std::vector<std::complex<realnum> > captured;
    for (size_t frequency = 0; frequency < frequencies.size(); ++frequency) {
      int cuda_rank = -1;
      size_t cuda_dims[3] = {0, 0, 0};
      std::unique_ptr<std::complex<realnum>[]> values(
          f.get_dft_array(monitor, Ep, static_cast<int>(frequency),
                          &cuda_rank, cuda_dims));
      require(values != nullptr && cuda_rank == cpu_rank,
              "cylindrical-axis CUDA DFT rank differs from CPU");
      for (int dim = 0; dim < cpu_rank; ++dim)
        require(cuda_dims[dim] == cpu_dims[dim],
                "cylindrical-axis CUDA DFT dimensions differ from CPU");
      captured.insert(captured.end(), values.get(),
                      values.get() + point_count);
    }
    return captured;
  };
  const auto require_bitwise_cpu_oracle =
      [&](const std::vector<std::complex<realnum> > &observed,
          const char *message) {
        require(observed.size() == cpu_values.size(), message);
        require(std::memcmp(observed.data(), cpu_values.data(),
                            observed.size() * sizeof(observed[0])) == 0,
                message);
        for (const auto &value : observed)
          require(std::isfinite(value.real()) &&
                      std::isfinite(value.imag()),
                  message);
      };

  const std::vector<std::complex<realnum> > first =
      capture_cuda_spectrum();
  require_bitwise_cpu_oracle(
      first, "cylindrical-axis first CUDA sweep differs bitwise from CPU");
  const gpu::dft_materialization_statistics after_first =
      gpu::get_dft_materialization_statistics();
  require(after_first.cuda_kernel_launches == 1,
          "cylindrical-axis three-frequency first sweep did not use one "
          "all-frequency kernel");

  const std::vector<std::complex<realnum> > repeated =
      capture_cuda_spectrum();
  require_bitwise_cpu_oracle(
      repeated, "cylindrical-axis repeated CUDA sweep changed the result");
  require(gpu::get_dft_materialization_statistics()
                  .cuda_kernel_launches == 1,
          "identical cylindrical-axis spectrum relaunched its kernel");

  monitor.scale_dfts(std::complex<double>(0.75, -0.125));
  const std::vector<std::complex<realnum> > after_content_change =
      capture_cuda_spectrum();
  require_bitwise_cpu_oracle(
      after_content_change,
      "scaled zero cylindrical-axis spectrum differs bitwise from CPU");
  const gpu::dft_materialization_statistics statistics =
      gpu::get_dft_materialization_statistics();
  require(statistics.cuda_array_calls == 3 * frequencies.size() &&
              statistics.cuda_array_points ==
                  3 * frequencies.size() * point_count &&
              statistics.cpu_array_calls == 0 &&
              statistics.cuda_kernel_launches == 2,
          "cylindrical-axis all-frequency cache did not launch 1/+0/+1 "
          "kernels across first/repeated/content-changed sweeps");
  if (am_master())
    std::cout << "CYLINDRICAL_DFT_CACHE frequencies="
              << frequencies.size() << " points=" << point_count
              << " kernels=1/+0/+1 bitwise_cpu_oracle=yes\n";

  const char *output_name =
      "gpu-step-db-public-dft-output-collapsed-cylindrical-axis";
  const std::string output_path = f.h5file_name(output_name);
  const std::string temporary_name =
      std::string(output_name) + ".gpmeep-tmp";
  const std::string temporary_path =
      f.h5file_name(temporary_name.c_str());
  unlink(output_path.c_str());
  unlink(temporary_path.c_str());
  gpu::reset_dispatch_statistics();
  f.output_dft(monitor, output_name);
  const gpu::dft_materialization_statistics output_statistics =
      gpu::get_dft_materialization_statistics();
  const std::uint64_t output_bytes =
      point_count * frequencies.size() *
      sizeof(std::complex<realnum>);
  require(output_statistics.cuda_output_calls == 1 &&
              output_statistics.cuda_output_points ==
                  point_count * frequencies.size() &&
              output_statistics.cpu_output_calls == 0 &&
              output_statistics.cuda_output_staging_calls == 0 &&
              output_statistics.cuda_result_device_to_host_bytes ==
                  output_bytes &&
              output_statistics.array_mpi_allreduce_calls ==
                  frequencies.size() &&
              output_statistics.array_mpi_allreduce_bytes == output_bytes,
          "cylindrical-axis collapsed CUDA output telemetry is invalid");
  bool output_valid = true;
  bool output_bitwise = true;
  {
    h5file output(output_path.c_str(), h5file::READONLY,
                  false /* parallel */, true /* local */);
    output_valid =
        output.root_object_count() == 2 * frequencies.size();
    for (size_t frequency = 0; frequency < frequencies.size();
         ++frequency) {
      int real_rank = -1;
      int imaginary_rank = -1;
      size_t real_dims[3] = {0, 0, 0};
      size_t imaginary_dims[3] = {0, 0, 0};
      const std::string prefix =
          std::string(component_name(Ep)) + "_" +
          std::to_string(frequency);
      std::unique_ptr<double[]> real_values(static_cast<double *>(
          output.read((prefix + ".r").c_str(), &real_rank, real_dims, 3,
                      false)));
      std::unique_ptr<double[]> imaginary_values(static_cast<double *>(
          output.read((prefix + ".i").c_str(), &imaginary_rank,
                      imaginary_dims, 3, false)));
      const int expected_hdf5_rank = cpu_rank == 0 ? 1 : cpu_rank;
      output_valid = output_valid && real_values && imaginary_values &&
                     real_rank == expected_hdf5_rank &&
                     imaginary_rank == expected_hdf5_rank;
      if (real_values && imaginary_values)
        for (size_t point = 0; point < point_count; ++point) {
          const std::complex<double> expected(
              cpu_values[frequency * point_count + point]);
          const double expected_real = expected.real();
          const double expected_imaginary = expected.imag();
          output_valid = output_valid &&
                         std::isfinite(real_values[point]) &&
                         std::isfinite(imaginary_values[point]);
          output_bitwise =
              output_bitwise &&
              std::memcmp(&real_values[point], &expected_real,
                          sizeof(double)) == 0 &&
              std::memcmp(&imaginary_values[point],
                          &expected_imaginary, sizeof(double)) == 0;
        }
    }
  }
  require(output_valid && output_bitwise &&
              access(temporary_path.c_str(), F_OK) != 0 &&
              unlink(output_path.c_str()) == 0,
          "cylindrical-axis collapsed HDF5 differs bitwise from CPU");
  if (am_master())
    std::cout << "COLLAPSED_DFT_OUTPUT case=cylindrical-axis rank="
              << cpu_rank << " points=" << point_count
              << " frequencies=" << frequencies.size()
              << " kernels="
              << output_statistics.cuda_kernel_launches
              << " d2h="
              << output_statistics.cuda_result_device_to_host_bytes
              << " bitwise=" << output_bitwise << '\n';
  monitor.remove();
  gpu::set_backend(gpu::backend_mode::cpu);
}

void require_distributed_zero_work_dft_materialization() {
  require(count_processors() == 2,
          "distributed zero-work DFT materialization requires two ranks");
  gpu::set_backend(gpu::backend_mode::cpu);
  const grid_volume gv = vol2d(1.6, 1.4, 12.0);
  structure s(gv, vacuum, no_pml(), identity(), 4);
  fields f(&s, 0.0, 0.0, true, 32, 32);
  f.use_real_fields();
  gaussian_src_time source(0.29, 0.10);
  f.add_point_source(Ez, source, vec(-0.53, -0.47), 0.8);
  component components[] = {Ez};
  const volume point_monitor(vec(0.61, 0.53), vec(0.61, 0.53));
  dft_fields monitor = f.add_dft_fields(
      components, 1, point_monitor, 0.24, 0.24, 1, true, 1);
  for (int step = 0; step < 24; ++step) f.step();

  int local_monitor_chunks = 0;
  for (dft_chunk *chunk = monitor.chunks; chunk;
       chunk = chunk->next_in_dft)
    if (chunk->c == Ez) ++local_monitor_chunks;
  const int contributing_ranks =
      sum_to_all(local_monitor_chunks > 0 ? 1 : 0);
  require(contributing_ranks == 1,
          "point DFT monitor was not owned by exactly one MPI rank");

  int cpu_rank = -1;
  size_t cpu_dims[3] = {0, 0, 0};
  std::unique_ptr<std::complex<realnum>[]> cpu_values(
      f.get_dft_array(monitor, Ez, 0, &cpu_rank, cpu_dims));
  require(cpu_values != nullptr && cpu_rank == 0,
          "distributed CPU point DFT did not collapse to a scalar");

  gpu::set_backend(gpu::backend_mode::cuda);
  gpu::reset_dispatch_statistics();
  int cuda_rank = -1;
  size_t cuda_dims[3] = {0, 0, 0};
  std::unique_ptr<std::complex<realnum>[]> cuda_values(
      f.get_dft_array(monitor, Ez, 0, &cuda_rank, cuda_dims));
  require(cuda_values != nullptr && cuda_rank == 0,
          "distributed CUDA point DFT did not collapse to a scalar");
  require(std::abs(std::complex<double>(cpu_values[0]) -
                   std::complex<double>(cuda_values[0])) <=
              3e-4 * std::max(
                         1.0, std::abs(std::complex<double>(cpu_values[0]))),
          "distributed CUDA point DFT disagrees with the CPU oracle");

  const gpu::dft_materialization_statistics statistics =
      gpu::get_dft_materialization_statistics();
  const bool owns_monitor = local_monitor_chunks > 0;
  require(statistics.cuda_array_calls == 1 &&
              statistics.cuda_array_points == 1 &&
              statistics.cpu_array_calls == 0,
          "distributed CUDA logical DFT telemetry is incorrect");
  require(owns_monitor
              ? (statistics.cuda_kernel_launches == 1 &&
                 statistics.cuda_result_device_to_host_bytes > 0)
              : (statistics.cuda_kernel_launches == 0 &&
                 statistics.cuda_result_device_to_host_bytes == 0),
          "zero-work MPI rank performed a CUDA kernel or D2H transfer");
  require(statistics.array_mpi_allreduce_calls == 1 &&
              statistics.array_mpi_allreduce_bytes > 0,
          "distributed CUDA DFT array skipped its result collective");
  require(sum_to_all(
              static_cast<size_t>(statistics.cuda_kernel_launches)) == 1,
          "distributed one-owner DFT launched more than one CUDA kernel");
  require(sum_to_all(static_cast<size_t>(
              statistics.cuda_result_device_to_host_bytes > 0 ? 1 : 0)) ==
              1,
          "distributed one-owner DFT staged results on the wrong ranks");

  const std::uint64_t d2h_before_synthetic =
      gpu::get_dispatch_statistics().device_to_host_bytes;
  int material_rank = -1;
  size_t material_dims[3] = {0, 0, 0};
  std::unique_ptr<std::complex<realnum>[]> material_values(
      f.get_dft_array(monitor, Dielectric, 0, &material_rank,
                      material_dims));
  require(material_values != nullptr && material_rank == 0 &&
              std::abs(std::complex<double>(material_values[0]) -
                       std::complex<double>(1.0, 0.0)) <= 2e-6,
          "distributed synthetic Dielectric point query failed");
  const gpu::dft_materialization_statistics after_material =
      gpu::get_dft_materialization_statistics();
  require(after_material.host_synthetic_material_array_calls == 1 &&
              after_material.host_synthetic_material_array_points == 1 &&
              after_material.cuda_array_calls == 1 &&
              after_material.cpu_array_calls == 0 &&
              after_material.cuda_kernel_launches ==
                  statistics.cuda_kernel_launches &&
              after_material.cuda_result_device_to_host_bytes ==
                  statistics.cuda_result_device_to_host_bytes,
          "distributed synthetic material telemetry is incorrect");
  require(gpu::get_dispatch_statistics().device_to_host_bytes ==
              d2h_before_synthetic,
          "distributed synthetic material query synchronized resident DFT "
          "storage");

  monitor.remove();
  gpu::set_backend(gpu::backend_mode::cpu);
}

void require_resident_scalar_query(bool complex_fields) {
#if MEEP_SINGLE
  const grid_volume gv = vol2d(1.8, 1.6, 12.0);

  gpu::set_backend(gpu::backend_mode::cpu);
  structure cpu_structure(gv, vacuum, no_pml());
  cpu_structure.set_chi3(weak_chi3);
  fields cpu(&cpu_structure, 0.0, 0.0, true, 0, 0);
  if (complex_fields)
    cpu.use_bloch(vec(0.09, 0.04));
  else
    cpu.use_real_fields();
  continuous_src_time cpu_source(0.27);
  cpu.add_point_source(
      Ez, cpu_source, gv.center(),
      complex_fields ? std::complex<double>(0.8, -0.35)
                     : std::complex<double>(0.8, 0.0));
  for (int step = 0; step < 32; ++step)
    cpu.step();

  gpu::set_backend(gpu::backend_mode::cuda);
  structure cuda_structure(gv, vacuum, no_pml());
  cuda_structure.set_chi3(weak_chi3);
  fields cuda(&cuda_structure, 0.0, 0.0, true, 0, 0);
  if (complex_fields)
    cuda.use_bloch(vec(0.09, 0.04));
  else
    cuda.use_real_fields();
  continuous_src_time cuda_source(0.27);
  cuda.add_point_source(
      Ez, cuda_source, gv.center(),
      complex_fields ? std::complex<double>(0.8, -0.35)
                     : std::complex<double>(0.8, 0.0));
  for (int step = 0; step < 32; ++step)
    cuda.step();

  int chosen_chunk_index = -1;
  ivec chosen_location(gv.dim);
  std::size_t chosen_field_index = 0;
  std::complex<double> expected(0.0, 0.0);
  double maximum_magnitude = -1.0;
  for (int chunk_index = 0; chunk_index < cpu.num_chunks; ++chunk_index) {
    fields_chunk *chunk = cpu.chunks[chunk_index];
    if (!chunk->is_mine() || !chunk->f[Ez][0]) continue;
    const ivec is = chunk->gv.little_owned_corner0(Ez);
    const ivec ie = chunk->gv.big_corner();
    LOOP_OVER_IVECS(chunk->gv, is, ie, index) {
      const std::complex<double> value(
          chunk->f[Ez][0][index],
          chunk->f[Ez][1] ? chunk->f[Ez][1][index] : 0.0);
      if (std::abs(value) > maximum_magnitude) {
        IVEC_LOOP_ILOC(chunk->gv, location);
        chosen_chunk_index = chunk_index;
        chosen_location = location;
        chosen_field_index = static_cast<std::size_t>(index);
        expected = value;
        maximum_magnitude = std::abs(value);
      }
    }
  }
  require(chosen_chunk_index >= 0 && maximum_magnitude > 1e-7,
          "resident scalar regression could not find a nonzero local field");

  fields_chunk *cuda_chunk = cuda.chunks[chosen_chunk_index];
  require(cuda_chunk->is_mine() && cuda_chunk->f[Ez][0],
          "CPU/CUDA resident scalar chunks do not have matching ownership");
  require(static_cast<std::size_t>(
              cuda_chunk->gv.index(Ez, chosen_location)) ==
              chosen_field_index,
          "CPU/CUDA resident scalar chunks do not have matching indices");
  require((cuda_chunk->f[Ez][1] != nullptr) == complex_fields,
          "resident scalar complex-field allocation disagrees with test");

  const std::complex<double> stale_host(
      cuda_chunk->f[Ez][0][chosen_field_index],
      cuda_chunk->f[Ez][1]
          ? cuda_chunk->f[Ez][1][chosen_field_index]
          : 0.0);
  gpu::reset_dispatch_statistics();
  const gpu::dispatch_statistics before =
      gpu::get_dispatch_statistics();
  const std::complex<double> queried =
      cuda_chunk->get_field(Ez, chosen_location);
  const gpu::dispatch_statistics after =
      gpu::get_dispatch_statistics();
  const std::uint64_t expected_scalar_bytes =
      complex_fields ? 2 * sizeof(float) : sizeof(float);
  require(after.device_to_host_bytes - before.device_to_host_bytes ==
              expected_scalar_bytes,
          "resident field query copied more than its requested scalar(s)");
  require(std::complex<double>(
              cuda_chunk->f[Ez][0][chosen_field_index],
              cuda_chunk->f[Ez][1]
                  ? cuda_chunk->f[Ez][1][chosen_field_index]
                  : 0.0) == stale_host,
          "resident field query published a dirty mirror to the host array");
  require(std::abs(queried - expected) <=
              3e-6 + 3e-4 *
                          std::max(std::abs(queried),
                                   std::abs(expected)),
          "resident scalar query disagrees with the CPU field");
  require(std::abs(queried - stale_host) > 1e-7,
          "resident scalar regression did not retain a stale host value");

  float rejected_real = 123.0f;
  float rejected_imaginary = 456.0f;
  bool rejected_bounds = false;
  const gpu::dispatch_statistics before_rejected_bounds =
      gpu::get_dispatch_statistics();
  try {
    (void)gpu::detail::read_resident_field_point_fp32_for_owner(
        cuda_chunk, cuda_chunk->f[Ez][0], cuda_chunk->f[Ez][1],
        std::numeric_limits<std::size_t>::max(), &rejected_real,
        &rejected_imaginary);
  }
  catch (const std::out_of_range &) {
    rejected_bounds = true;
  }
  require(rejected_bounds && rejected_real == 123.0f &&
              rejected_imaginary == 456.0f,
          "resident scalar bounds rejection was not transactional");
  const gpu::dispatch_statistics after_rejected_bounds =
      gpu::get_dispatch_statistics();
  require(after_rejected_bounds.device_to_host_bytes ==
              before_rejected_bounds.device_to_host_bytes,
          "resident scalar bounds rejection accounted a partial transfer");

  gpu::detail::sync_resident_cache_for_owner(cuda_chunk);
  const gpu::dispatch_statistics synchronized =
      gpu::get_dispatch_statistics();
  const std::uint64_t full_sync_bytes =
      synchronized.device_to_host_bytes -
      after.device_to_host_bytes;
  require(full_sync_bytes >
              expected_scalar_bytes,
          "full resident synchronization was not larger than a scalar query");
  require(std::abs(std::complex<double>(
                       cuda_chunk->f[Ez][0][chosen_field_index],
                       cuda_chunk->f[Ez][1]
                           ? cuda_chunk->f[Ez][1][chosen_field_index]
                           : 0.0) -
                   queried) <= 1e-7,
          "full resident synchronization did not publish the queried value");
  rejected_real = 123.0f;
  rejected_imaginary = 456.0f;
  require(gpu::detail::read_resident_field_point_fp32_for_owner(
              cuda_chunk, cuda_chunk->f[Ez][0], cuda_chunk->f[Ez][1],
              chosen_field_index, &rejected_real,
              &rejected_imaginary) == 0 &&
              rejected_real == 123.0f &&
              rejected_imaginary == 456.0f,
          "clean resident mirror did not use the existing host-read path");
  if (complex_fields) {
    // Dirty only the imaginary mirror so the direct read must combine a
    // device scalar with an untouched clean host scalar.  Keep the failed
    // null-destination attempt fully transactional before the valid read.
    {
      gpu::detail::resident_curl_session mixed_session(cuda_chunk, true);
      require(mixed_session.active(),
              "mixed scalar query did not start a CUDA resident session");
      gpu::detail::resident_zero_span_fp32(
          mixed_session.cache(), cuda_chunk->f[Ez][1],
          cuda_chunk->gv.ntot(), chosen_field_index, 1);
      mixed_session.finish(false);
    }
    float mixed_real = 789.0f;
    bool rejected_null_destination = false;
    const gpu::dispatch_statistics before_rejected_null =
        gpu::get_dispatch_statistics();
    try {
      (void)gpu::detail::read_resident_field_point_fp32_for_owner(
          cuda_chunk, cuda_chunk->f[Ez][0], cuda_chunk->f[Ez][1],
          chosen_field_index, &mixed_real, nullptr);
    }
    catch (const std::invalid_argument &) {
      rejected_null_destination = true;
    }
    const gpu::dispatch_statistics after_rejected_null =
        gpu::get_dispatch_statistics();
    require(rejected_null_destination && mixed_real == 789.0f &&
                after_rejected_null.device_to_host_bytes ==
                    before_rejected_null.device_to_host_bytes,
            "resident scalar null-destination rejection was not transactional");

    float mixed_imaginary = 456.0f;
    const unsigned int mixed_components =
        gpu::detail::read_resident_field_point_fp32_for_owner(
            cuda_chunk, cuda_chunk->f[Ez][0], cuda_chunk->f[Ez][1],
            chosen_field_index, &mixed_real, &mixed_imaginary);
    const gpu::dispatch_statistics after_mixed =
        gpu::get_dispatch_statistics();
    require(mixed_components == 2u && mixed_real == 789.0f &&
                mixed_imaginary == 0.0f,
            "resident scalar mixed clean/dirty result was incorrect");
    require(after_mixed.device_to_host_bytes -
                after_rejected_null.device_to_host_bytes ==
                sizeof(float),
            "resident scalar mixed clean/dirty query copied extra data");
    require(cuda_chunk->f[Ez][0][chosen_field_index] ==
                static_cast<float>(queried.real()) &&
                cuda_chunk->f[Ez][1][chosen_field_index] ==
                    static_cast<float>(queried.imag()),
            "resident scalar mixed query modified a host mirror");
  }
  if (am_master())
    std::cout << "nonlinear-resident-scalar-query: "
              << (complex_fields ? "complex" : "real")
              << " scalar_d2h_bytes=" << expected_scalar_bytes
              << " full_sync_d2h_bytes=" << full_sync_bytes << '\n';
#else
  (void)complex_fields;
#endif
}

void require_finite_check_plan_lifecycle() {
#if MEEP_SINGLE
  gpu::set_backend(gpu::backend_mode::cuda);
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  int cold_owner = 0;
  {
    std::vector<float> cold(23, 0.75f);
    gpu::detail::resident_curl_session resident(
        &cold_owner, true);
    require(resident.active(),
            "cold finite-check lifecycle did not start a resident session");

    gpu::reset_dispatch_statistics();
    const std::uint64_t empty_live_before =
        gpu::get_live_resident_device_buffers();
    gpu::detail::resident_finite_check_session empty(
        resident.cache());
    empty.accumulate(
        resident.cache(), nullptr, nullptr, 0);
    require(empty.finish() && !empty.active(),
            "all-empty finite-check session did not finish true/inactive");

    gpu::detail::resident_finite_check_session outer(resident.cache());
    bool nested_rejected = false;
    try {
      gpu::detail::resident_finite_check_session inner(resident.cache());
    }
    catch (const std::logic_error &) {
      nested_rejected = true;
    }
    require(nested_rejected && outer.finish(),
            "nested finite-check result-owner session was not rejected");
    const gpu::dispatch_statistics empty_dispatch =
        gpu::get_dispatch_statistics();
    const gpu::detail::finite_check_transfer_statistics empty_finite =
        gpu::detail::get_finite_check_transfer_statistics();
    require(empty_dispatch.host_to_device_bytes == 0 &&
                empty_dispatch.device_to_host_bytes == 0 &&
                empty_finite.host_to_device_bytes == 0 &&
                empty_finite.device_to_host_bytes == 0 &&
                gpu::get_live_resident_device_buffers() ==
                    empty_live_before,
            "all-empty finite-check session allocated or transferred");

    // Build the first plan from a duplicate request.  This catches sizing the
    // descriptor upload from the pre-dedup array count (which would copy an
    // uninitialized vector-capacity slot even though only one span executes).
    float *cold_arrays[] = {cold.data(), cold.data()};
    std::size_t cold_counts[] = {cold.size(), cold.size()};
    gpu::detail::resident_finite_check_session check(
        resident.cache());
    check.accumulate(
        resident.cache(), cold_arrays, cold_counts, 2);
    require(check.finish(),
            "cold finite resident array was reported non-finite");
    const gpu::dispatch_statistics cold_dispatch =
        gpu::get_dispatch_statistics();
    const gpu::detail::finite_check_transfer_statistics cold_finite =
        gpu::detail::get_finite_check_transfer_statistics();
    const std::uint64_t expected_cold_h2d =
        static_cast<std::uint64_t>(cold.size() * sizeof(float)) +
        static_cast<std::uint64_t>(
            gpu::detail::finite_check_descriptor_bytes(1));
    require(cold_dispatch.host_to_device_bytes ==
                expected_cold_h2d &&
                cold_finite.host_to_device_bytes ==
                    expected_cold_h2d &&
                cold_dispatch.device_to_host_bytes == sizeof(int) &&
                cold_finite.device_to_host_bytes == sizeof(int),
            "cold finite check did not exactly account for field and "
            "descriptor uploads");
    gpu::detail::resident_finite_check_session reused(
        resident.cache());
    reused.accumulate(
        resident.cache(), cold_arrays, cold_counts, 2);
    require(reused.finish(),
            "reused cold finite plan reported a false failure");
    const gpu::dispatch_statistics cold_reused_dispatch =
        gpu::get_dispatch_statistics();
    const gpu::detail::finite_check_transfer_statistics
        cold_reused_finite =
            gpu::detail::get_finite_check_transfer_statistics();
    require(cold_reused_dispatch.host_to_device_bytes ==
                expected_cold_h2d &&
                cold_reused_dispatch.device_to_host_bytes ==
                    2 * sizeof(int) &&
                cold_reused_finite.host_to_device_bytes ==
                    expected_cold_h2d &&
                cold_reused_finite.device_to_host_bytes ==
                    2 * sizeof(int),
            "cold finite plan did not become a 0B-H2D reusable plan");
    resident.finish(false);
  }
  gpu::detail::destroy_resident_cache_for_owner(&cold_owner);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "cold finite-check lifecycle leaked a device buffer");

  // A cache reset that retains reusable allocations must also retain a
  // deferred finite-result generation.  Otherwise an intervening reset can
  // turn a recorded failure into a successful later scan.
  int reset_owner = 0;
  {
    std::vector<float> values(19, 0.5f);
    {
      gpu::detail::resident_curl_session resident(&reset_owner, true);
      gpu::detail::resident_ensure_mirror_fp32(
          resident.cache(), values.data(), values.size());
      const gpu::detail::indexed_value_fp32 make_nan = {
          0, std::numeric_limits<float>::quiet_NaN()};
      gpu::detail::resident_indexed_subtract_fp32(
          resident.cache(), values.data(), values.size(), &make_nan, 1);
      float *arrays[] = {values.data()};
      std::size_t counts[] = {values.size()};
      gpu::detail::resident_finite_check_session deferred(
          resident.cache());
      deferred.accumulate(resident.cache(), arrays, counts, 1);
      require(deferred.finish(false),
              "pre-reset deferred finite scan did not finish");
      resident.finish(false);
    }
    // Device/backend transitions must not release the only copy of a sticky
    // finish(false) verdict.  This preflight is transactional and is useful
    // even on a one-GPU host because CPU selection follows the same cache
    // invalidation path as an ordinal migration.
    bool pending_transition_rejected = false;
    try {
      gpu::set_backend(gpu::backend_mode::cpu);
    }
    catch (const std::logic_error &) {
      pending_transition_rejected = true;
    }
    require(pending_transition_rejected &&
                gpu::active_backend() == gpu::backend_mode::cuda,
            "backend transition discarded or accepted a deferred "
            "finite-check verdict");
    gpu::detail::reset_resident_cache_for_owner_reusing_allocations(
        &reset_owner);
    // reset publishes the dirty NaN before recycling the mirror.  Repair the
    // host value so only the retained deferred token can make the next
    // verdict fail.
    values[0] = 0.5f;
    {
      gpu::detail::resident_curl_session resident(&reset_owner, true);
      float *arrays[] = {values.data()};
      std::size_t counts[] = {values.size()};
      gpu::detail::resident_finite_check_session cumulative(
          resident.cache());
      cumulative.accumulate(resident.cache(), arrays, counts, 1);
      require(!cumulative.finish(),
              "cache reuse reset discarded a deferred non-finite verdict");
      gpu::detail::resident_finite_check_session recovered(
          resident.cache());
      recovered.accumulate(resident.cache(), arrays, counts, 1);
      require(recovered.finish(),
              "finite verdict did not recover after reset/readback");

      const gpu::detail::indexed_value_fp32 make_nan = {
          0, std::numeric_limits<float>::quiet_NaN()};
      gpu::detail::resident_indexed_subtract_fp32(
          resident.cache(), values.data(), values.size(), &make_nan, 1);
      gpu::detail::resident_finite_check_session deferred_empty(
          resident.cache());
      deferred_empty.accumulate(resident.cache(), arrays, counts, 1);
      require(deferred_empty.finish(false),
              "second deferred finite scan did not finish");
      gpu::detail::resident_finite_check_session empty_readback(
          resident.cache());
      require(!empty_readback.finish(),
              "zero-array session did not consume a deferred non-finite "
              "verdict");
      resident.finish(false);
    }
  }
  gpu::detail::destroy_resident_cache_for_owner(&reset_owner);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "reset finite-check lifecycle leaked a device buffer");

  // Reproduce a historical nested interval topology. A long dirty parent
  // can be hidden behind shorter, later interval starts, so finite checks,
  // prepared spans, and subtraction sources must not rely on only the
  // immediate predecessor. Boundary resolution uses the same audited range
  // resolver.
  int ambiguous_owner = 0;
  {
    std::vector<float> parent(100, 1.0f);
    gpu::detail::resident_curl_session resident(
        &ambiguous_owner, true);
    require(resident.active(),
            "ambiguous finite topology did not start a resident session");
    gpu::detail::resident_ensure_mirror_fp32(
        resident.cache(), parent.data(), parent.size());
    const gpu::detail::indexed_value_fp32 make_nan = {
        20, std::numeric_limits<float>::quiet_NaN()};
    gpu::detail::resident_indexed_subtract_fp32(
        resident.cache(), parent.data(), parent.size(),
        &make_nan, 1);
    gpu::detail::resident_ensure_mirror_fp32(
        resident.cache(), parent.data() + 5, 1);
    gpu::detail::resident_ensure_mirror_fp32(
        resident.cache(), parent.data() + 20, 1);

    gpu::reset_dispatch_statistics();
    const gpu::dispatch_statistics ambiguous_before =
        gpu::get_dispatch_statistics();
    const gpu::resident_statistics ambiguous_resident_before =
        gpu::get_resident_statistics();
    const gpu::detail::finite_check_transfer_statistics
        ambiguous_finite_before =
            gpu::detail::get_finite_check_transfer_statistics();
    const std::uint64_t ambiguous_live_before =
        gpu::get_live_resident_device_buffers();
    bool rejected_exact = false;
    try {
      float *arrays[] = {parent.data()};
      std::size_t counts[] = {parent.size()};
      gpu::detail::resident_finite_check_session check(
          resident.cache());
      check.accumulate(resident.cache(), arrays, counts, 1);
    }
    catch (const std::invalid_argument &) {
      rejected_exact = true;
    }
    bool rejected_nested_alias = false;
    try {
      float *arrays[] = {parent.data() + 4};
      std::size_t counts[] = {3};
      gpu::detail::resident_finite_check_session check(
          resident.cache());
      check.accumulate(resident.cache(), arrays, counts, 1);
    }
    catch (const std::invalid_argument &) {
      rejected_nested_alias = true;
    }
    bool rejected_hidden_parent = false;
    try {
      float *arrays[] = {parent.data() + 20};
      std::size_t counts[] = {1};
      gpu::detail::resident_finite_check_session check(
          resident.cache());
      check.accumulate(resident.cache(), arrays, counts, 1);
    }
    catch (const std::invalid_argument &) {
      rejected_hidden_parent = true;
    }
    bool rejected_ambiguous_span = false;
    try {
      gpu::detail::resident_ensure_span_fp32(
          resident.cache(), parent.data() + 20, 1);
    }
    catch (const std::invalid_argument &) {
      rejected_ambiguous_span = true;
    }
    std::vector<float> subtraction_destination(1, 3.0f);
    bool rejected_ambiguous_subtraction = false;
    try {
      gpu::detail::resident_subtract_fp32(
          resident.cache(), subtraction_destination.data(),
          parent.data() + 20, 1);
    }
    catch (const std::invalid_argument &) {
      rejected_ambiguous_subtraction = true;
    }
    const gpu::dispatch_statistics ambiguous_after =
        gpu::get_dispatch_statistics();
    const gpu::resident_statistics ambiguous_resident_after =
        gpu::get_resident_statistics();
    const gpu::detail::finite_check_transfer_statistics
        ambiguous_finite_after =
            gpu::detail::get_finite_check_transfer_statistics();
    require(
        rejected_exact && rejected_nested_alias &&
            rejected_hidden_parent && rejected_ambiguous_span &&
            rejected_ambiguous_subtraction &&
            subtraction_destination[0] == 3.0f &&
            ambiguous_after.host_to_device_bytes ==
                ambiguous_before.host_to_device_bytes &&
            ambiguous_after.device_to_host_bytes ==
                ambiguous_before.device_to_host_bytes &&
            ambiguous_resident_after.device_buffer_allocations ==
                ambiguous_resident_before.device_buffer_allocations &&
            ambiguous_resident_after.device_buffer_reuses ==
                ambiguous_resident_before.device_buffer_reuses &&
            ambiguous_finite_after.host_to_device_bytes ==
                ambiguous_finite_before.host_to_device_bytes &&
            ambiguous_finite_after.device_to_host_bytes ==
                ambiguous_finite_before.device_to_host_bytes &&
            gpu::get_live_resident_device_buffers() ==
                ambiguous_live_before,
        "ambiguous nested resident topology mutated state");
    resident.finish(false);
  }
  gpu::detail::destroy_resident_cache_for_owner(&ambiguous_owner);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "ambiguous resident topology leaked a device buffer");

  int owner = 0;
  std::vector<float> first(37, 1.0f);
  std::vector<float> second(19, -0.5f);
  std::vector<float> third(11, 0.25f);

  {
    gpu::detail::resident_curl_session resident(&owner, true);
    require(resident.active(),
            "finite-check lifecycle did not start a resident session");
    gpu::detail::resident_ensure_mirror_fp32(
        resident.cache(), first.data(), first.size());
    gpu::detail::resident_ensure_mirror_fp32(
        resident.cache(), second.data(), second.size());
    gpu::detail::resident_ensure_mirror_fp32(
        resident.cache(), third.data(), third.size());

    float *two_arrays[] = {first.data(), second.data()};
    std::size_t two_counts[] = {first.size(), second.size()};
    gpu::reset_dispatch_statistics();
    const gpu::dispatch_statistics fresh_before_malformed =
        gpu::get_dispatch_statistics();
    const gpu::resident_statistics fresh_resident_before_malformed =
        gpu::get_resident_statistics();
    const gpu::detail::finite_check_transfer_statistics
        fresh_finite_before_malformed =
            gpu::detail::get_finite_check_transfer_statistics();
    const std::uint64_t fresh_live_before_malformed =
        gpu::get_live_resident_device_buffers();
    bool fresh_rejected_size = false;
    try {
      std::size_t bad_counts[] = {
          first.size() + 1, second.size()};
      gpu::detail::resident_finite_check_session check(
          resident.cache());
      check.accumulate(
          resident.cache(), two_arrays, bad_counts, 2);
    }
    catch (const std::invalid_argument &) {
      fresh_rejected_size = true;
    }
    bool fresh_rejected_null = false;
    try {
      float *bad_arrays[] = {first.data(), nullptr};
      gpu::detail::resident_finite_check_session check(
          resident.cache());
      check.accumulate(
          resident.cache(), bad_arrays, two_counts, 2);
    }
    catch (const std::invalid_argument &) {
      fresh_rejected_null = true;
    }
    bool fresh_rejected_overflow = false;
    try {
      std::size_t bad_counts[] = {
          std::numeric_limits<std::size_t>::max(),
          second.size()};
      gpu::detail::resident_finite_check_session check(
          resident.cache());
      check.accumulate(
          resident.cache(), two_arrays, bad_counts, 2);
    }
    catch (const std::overflow_error &) {
      fresh_rejected_overflow = true;
    }
    const gpu::dispatch_statistics fresh_after_malformed =
        gpu::get_dispatch_statistics();
    const gpu::resident_statistics fresh_resident_after_malformed =
        gpu::get_resident_statistics();
    const gpu::detail::finite_check_transfer_statistics
        fresh_finite_after_malformed =
            gpu::detail::get_finite_check_transfer_statistics();
    require(
        fresh_rejected_size && fresh_rejected_null &&
            fresh_rejected_overflow &&
            fresh_after_malformed.host_to_device_bytes ==
                fresh_before_malformed.host_to_device_bytes &&
            fresh_after_malformed.device_to_host_bytes ==
                fresh_before_malformed.device_to_host_bytes &&
            fresh_resident_after_malformed.host_to_device_bytes_avoided ==
                fresh_resident_before_malformed
                    .host_to_device_bytes_avoided &&
            fresh_resident_after_malformed.device_to_host_bytes_avoided ==
                fresh_resident_before_malformed
                    .device_to_host_bytes_avoided &&
            fresh_resident_after_malformed.device_buffer_allocations ==
                fresh_resident_before_malformed
                    .device_buffer_allocations &&
            fresh_resident_after_malformed.device_buffer_reuses ==
                fresh_resident_before_malformed.device_buffer_reuses &&
            fresh_finite_after_malformed.host_to_device_bytes ==
                fresh_finite_before_malformed.host_to_device_bytes &&
            fresh_finite_after_malformed.device_to_host_bytes ==
                fresh_finite_before_malformed.device_to_host_bytes &&
            gpu::get_live_resident_device_buffers() ==
                fresh_live_before_malformed,
        "fresh malformed finite topology mutated result/plan state");

    {
      gpu::detail::resident_finite_check_session check(
          resident.cache());
      check.accumulate(
          resident.cache(), two_arrays, two_counts, 2);
      require(check.finish(),
              "finite resident arrays were reported non-finite");
    }
    const gpu::dispatch_statistics first_check =
        gpu::get_dispatch_statistics();
    const gpu::detail::finite_check_transfer_statistics
        first_finite_check =
            gpu::detail::get_finite_check_transfer_statistics();
    require(first_check.host_to_device_bytes > 0 &&
                first_check.device_to_host_bytes == sizeof(int) &&
                first_finite_check.host_to_device_bytes ==
                    first_check.host_to_device_bytes &&
                first_finite_check.device_to_host_bytes ==
                    sizeof(int),
            "first finite check did not upload one descriptor plan and "
            "read one result");
    const gpu::resident_statistics first_resident =
        gpu::get_resident_statistics();

    {
      gpu::detail::resident_finite_check_session check(
          resident.cache());
      check.accumulate(
          resident.cache(), two_arrays, two_counts, 2);
      require(check.finish(),
              "reused finite plan reported a false failure");
    }
    const gpu::dispatch_statistics reused_check =
        gpu::get_dispatch_statistics();
    const gpu::detail::finite_check_transfer_statistics
        reused_finite_check =
            gpu::detail::get_finite_check_transfer_statistics();
    const gpu::resident_statistics reused_resident =
        gpu::get_resident_statistics();
    require(reused_check.host_to_device_bytes ==
                first_check.host_to_device_bytes &&
                reused_check.device_to_host_bytes -
                        first_check.device_to_host_bytes ==
                    sizeof(int) &&
                reused_resident.device_buffer_allocations ==
                    first_resident.device_buffer_allocations &&
                reused_resident.device_buffer_reuses >
                    first_resident.device_buffer_reuses &&
                reused_finite_check.host_to_device_bytes ==
                    first_finite_check.host_to_device_bytes &&
                reused_finite_check.device_to_host_bytes -
                        first_finite_check.device_to_host_bytes ==
                    sizeof(int),
            "stable finite plan/result was not reused without H2D");

    float *duplicate_arrays[] = {
        first.data(), first.data(), second.data(), second.data()};
    std::size_t duplicate_counts[] = {
        first.size(), first.size(), second.size(), second.size()};
    {
      gpu::detail::resident_finite_check_session check(
          resident.cache());
      check.accumulate(
          resident.cache(), duplicate_arrays, duplicate_counts, 4);
      require(check.finish(),
              "deduplicated finite plan reported a false failure");
    }
    const gpu::dispatch_statistics duplicate_check =
        gpu::get_dispatch_statistics();
    const gpu::detail::finite_check_transfer_statistics
        duplicate_finite_check =
            gpu::detail::get_finite_check_transfer_statistics();
    require(
        duplicate_check.host_to_device_bytes ==
            reused_check.host_to_device_bytes &&
            duplicate_check.device_to_host_bytes -
                    reused_check.device_to_host_bytes ==
                sizeof(int) &&
            duplicate_finite_check.host_to_device_bytes ==
                reused_finite_check.host_to_device_bytes &&
            duplicate_finite_check.device_to_host_bytes -
                    reused_finite_check.device_to_host_bytes ==
                sizeof(int),
        "exact duplicate finite ranges were not removed before plan reuse");

    const gpu::dispatch_statistics before_malformed =
        gpu::get_dispatch_statistics();
    const std::uint64_t live_before_malformed =
        gpu::get_live_resident_device_buffers();
    bool rejected_size = false;
    try {
      std::size_t bad_counts[] = {
          first.size() + 1, second.size()};
      gpu::detail::resident_finite_check_session check(
          resident.cache());
      check.accumulate(
          resident.cache(), two_arrays, bad_counts, 2);
    }
    catch (const std::invalid_argument &) {
      rejected_size = true;
    }
    bool rejected_null = false;
    try {
      float *bad_arrays[] = {first.data(), nullptr};
      gpu::detail::resident_finite_check_session check(
          resident.cache());
      check.accumulate(
          resident.cache(), bad_arrays, two_counts, 2);
    }
    catch (const std::invalid_argument &) {
      rejected_null = true;
    }
    bool rejected_overflow = false;
    try {
      std::size_t bad_counts[] = {
          std::numeric_limits<std::size_t>::max(),
          second.size()};
      gpu::detail::resident_finite_check_session check(
          resident.cache());
      check.accumulate(
          resident.cache(), two_arrays, bad_counts, 2);
    }
    catch (const std::overflow_error &) {
      rejected_overflow = true;
    }
    const gpu::dispatch_statistics after_malformed =
        gpu::get_dispatch_statistics();
    require(rejected_size && rejected_null && rejected_overflow &&
                after_malformed.host_to_device_bytes ==
                    before_malformed.host_to_device_bytes &&
                after_malformed.device_to_host_bytes ==
                    before_malformed.device_to_host_bytes &&
                gpu::get_live_resident_device_buffers() ==
                    live_before_malformed,
            "malformed finite topology mutated transfers or allocations");

    {
      gpu::detail::resident_finite_check_session check(
          resident.cache());
      check.accumulate(
          resident.cache(), two_arrays, two_counts, 2);
      require(check.finish(),
              "malformed finite topology corrupted the old plan");
    }
    const gpu::dispatch_statistics after_recovery =
        gpu::get_dispatch_statistics();
    require(after_recovery.host_to_device_bytes ==
                after_malformed.host_to_device_bytes,
            "malformed finite topology invalidated a reusable plan");

    float *three_arrays[] = {
        second.data(), first.data(), third.data()};
    std::size_t three_counts[] = {
        second.size(), first.size(), third.size()};
    {
      gpu::detail::resident_finite_check_session check(
          resident.cache());
      check.accumulate(
          resident.cache(), three_arrays, three_counts, 3);
      require(check.finish(),
              "reordered/extended finite topology reported failure");
    }
    const gpu::dispatch_statistics rebuilt =
        gpu::get_dispatch_statistics();
    require(rebuilt.host_to_device_bytes >
                after_recovery.host_to_device_bytes,
            "changed finite topology uploaded no replacement descriptor");
    {
      gpu::detail::resident_finite_check_session check(
          resident.cache());
      check.accumulate(
          resident.cache(), three_arrays, three_counts, 3);
      require(check.finish(),
              "replacement finite plan was not reusable");
    }
    const gpu::dispatch_statistics rebuilt_reused =
        gpu::get_dispatch_statistics();
    require(rebuilt_reused.host_to_device_bytes ==
                rebuilt.host_to_device_bytes,
            "replacement finite descriptor was uploaded twice");

    const gpu::detail::indexed_value_fp32 make_nan = {
        0, std::numeric_limits<float>::quiet_NaN()};
    gpu::detail::resident_indexed_subtract_fp32(
        resident.cache(), first.data(), first.size(),
        &make_nan, 1);
    gpu::reset_dispatch_statistics();
    {
      gpu::detail::resident_finite_check_session check(
          resident.cache());
      check.accumulate(
          resident.cache(), three_arrays, three_counts, 3);
      require(!check.finish(),
              "NaN resident field passed the finite check");
    }
    const gpu::dispatch_statistics nan_check =
        gpu::get_dispatch_statistics();
    const gpu::detail::finite_check_transfer_statistics
        nan_finite_check =
            gpu::detail::get_finite_check_transfer_statistics();
    require(nan_check.host_to_device_bytes == 0 &&
                nan_check.device_to_host_bytes == sizeof(int) &&
                nan_finite_check.host_to_device_bytes == 0 &&
                nan_finite_check.device_to_host_bytes == sizeof(int),
            "NaN finite check did not reuse its resident topology");

    gpu::detail::resident_zero_span_fp32(
        resident.cache(), first.data(), first.size(), 0, 1);
    gpu::reset_dispatch_statistics();
    {
      gpu::detail::resident_finite_check_session check(
          resident.cache());
      check.accumulate(
          resident.cache(), three_arrays, three_counts, 3);
      require(check.finish(),
              "finite-result GPU reset did not recover after NaN");
    }
    const gpu::dispatch_statistics restored_check =
        gpu::get_dispatch_statistics();
    const gpu::detail::finite_check_transfer_statistics
        restored_finite_check =
            gpu::detail::get_finite_check_transfer_statistics();
    require(restored_check.host_to_device_bytes == 0 &&
                restored_check.device_to_host_bytes == sizeof(int) &&
                restored_finite_check.host_to_device_bytes == 0 &&
                restored_finite_check.device_to_host_bytes ==
                    sizeof(int),
            "restored finite check did not have 0B H2D/4B D2H");

    // Deferring a readback must preserve the cumulative failure bit even if
    // the offending field is repaired before the next scan.  This is the
    // exact contract used by the M8.5 synchronization experiment.
    gpu::detail::resident_indexed_subtract_fp32(
        resident.cache(), first.data(), first.size(), &make_nan, 1);
    gpu::reset_dispatch_statistics();
    {
      gpu::detail::resident_finite_check_session deferred(
          resident.cache());
      deferred.accumulate(
          resident.cache(), three_arrays, three_counts, 3);
      require(deferred.finish(false) && !deferred.active(),
              "deferred finite scan did not end without a readback");
    }
    const gpu::dispatch_statistics after_deferred =
        gpu::get_dispatch_statistics();
    require(after_deferred.host_to_device_bytes == 0 &&
                after_deferred.device_to_host_bytes == 0,
            "deferred finite scan unexpectedly transferred a scalar");
    gpu::detail::resident_zero_span_fp32(
        resident.cache(), first.data(), first.size(), 0, 1);
    {
      gpu::detail::resident_finite_check_session cumulative(
          resident.cache());
      cumulative.accumulate(
          resident.cache(), three_arrays, three_counts, 3);
      require(!cumulative.finish(),
              "deferred NaN verdict was lost after the field was repaired");
    }
    const gpu::dispatch_statistics after_cumulative =
        gpu::get_dispatch_statistics();
    require(after_cumulative.host_to_device_bytes == 0 &&
                after_cumulative.device_to_host_bytes == sizeof(int),
            "cumulative finite verdict did not use exactly one delayed D2H");
    {
      gpu::detail::resident_finite_check_session recovered(
          resident.cache());
      recovered.accumulate(
          resident.cache(), three_arrays, three_counts, 3);
      require(recovered.finish(),
              "finite result did not reset after cumulative readback");
    }

    // A subspan of a device-authoritative mirror must resolve to the
    // existing allocation plus an offset. It must never create an
    // overlapping mirror from stale host data.
    float *interior_arrays[] = {first.data() + 5};
    std::size_t interior_counts[] = {9};
    const std::uint64_t live_before_interior =
        gpu::get_live_resident_device_buffers();
    gpu::reset_dispatch_statistics();
    {
      gpu::detail::resident_finite_check_session check(
          resident.cache());
      check.accumulate(
          resident.cache(), interior_arrays, interior_counts, 1);
      require(check.finish(),
              "finite interior resident subspan reported failure");
    }
    const gpu::dispatch_statistics interior_built =
        gpu::get_dispatch_statistics();
    require(
        interior_built.host_to_device_bytes ==
            gpu::detail::finite_check_descriptor_bytes(1) &&
            interior_built.device_to_host_bytes == sizeof(int) &&
            gpu::get_live_resident_device_buffers() ==
                live_before_interior,
        "interior finite subspan uploaded stale field data or "
        "allocated an overlapping mirror");

    const gpu::detail::indexed_value_fp32 make_inf = {
        7, std::numeric_limits<float>::infinity()};
    gpu::detail::resident_indexed_subtract_fp32(
        resident.cache(), first.data(), first.size(),
        &make_inf, 1);
    gpu::reset_dispatch_statistics();
    {
      gpu::detail::resident_finite_check_session check(
          resident.cache());
      check.accumulate(
          resident.cache(), interior_arrays, interior_counts, 1);
      require(!check.finish(),
              "Inf in an interior device subspan passed the finite check");
    }
    const gpu::dispatch_statistics interior_inf =
        gpu::get_dispatch_statistics();
    const gpu::detail::finite_check_transfer_statistics
        interior_inf_finite =
            gpu::detail::get_finite_check_transfer_statistics();
    require(interior_inf.host_to_device_bytes == 0 &&
                interior_inf.device_to_host_bytes == sizeof(int) &&
                interior_inf_finite.host_to_device_bytes == 0 &&
                interior_inf_finite.device_to_host_bytes == sizeof(int) &&
                gpu::get_live_resident_device_buffers() ==
                    live_before_interior,
            "interior finite reuse uploaded stale host data or changed "
            "live allocations");
    gpu::detail::resident_zero_span_fp32(
        resident.cache(), first.data(), first.size(), 7, 1);

    gpu::reset_dispatch_statistics();
    const std::uint64_t live_before_bad_alias =
        gpu::get_live_resident_device_buffers();
    bool rejected_partial_alias = false;
    try {
      float *partial_arrays[] = {
          first.data() + first.size() - 2};
      std::size_t partial_counts[] = {4};
      gpu::detail::resident_finite_check_session check(
          resident.cache());
      check.accumulate(
          resident.cache(), partial_arrays, partial_counts, 1);
    }
    catch (const std::invalid_argument &) {
      rejected_partial_alias = true;
    }
    bool rejected_unaligned_alias = false;
    try {
      float *unaligned_arrays[] = {
          reinterpret_cast<float *>(
              reinterpret_cast<unsigned char *>(first.data()) + 1)};
      std::size_t unaligned_counts[] = {1};
      gpu::detail::resident_finite_check_session check(
          resident.cache());
      check.accumulate(
          resident.cache(), unaligned_arrays, unaligned_counts, 1);
    }
    catch (const std::invalid_argument &) {
      rejected_unaligned_alias = true;
    }
    const gpu::dispatch_statistics bad_alias_dispatch =
        gpu::get_dispatch_statistics();
    const gpu::detail::finite_check_transfer_statistics
        bad_alias_finite =
            gpu::detail::get_finite_check_transfer_statistics();
    require(rejected_partial_alias && rejected_unaligned_alias &&
                bad_alias_dispatch.host_to_device_bytes == 0 &&
                bad_alias_dispatch.device_to_host_bytes == 0 &&
                bad_alias_finite.host_to_device_bytes == 0 &&
                bad_alias_finite.device_to_host_bytes == 0 &&
                gpu::get_live_resident_device_buffers() ==
                    live_before_bad_alias,
            "malformed interior finite topology mutated cache state");

    gpu::reset_dispatch_statistics();
    require(gpu::detail::resident_primary_fields_are_finite_fp32(
                resident.cache(), nullptr, nullptr, 0),
            "legacy zero-array finite check did not return true");
    const gpu::dispatch_statistics empty_check =
        gpu::get_dispatch_statistics();
    require(empty_check.host_to_device_bytes == 0 &&
                empty_check.device_to_host_bytes == 0,
            "legacy zero-array finite check performed a transfer");

    bool rejected_null_cache = false;
    try {
      (void)gpu::detail::resident_primary_fields_are_finite_fp32(
          nullptr, nullptr, nullptr, 0);
    }
    catch (const std::logic_error &) {
      rejected_null_cache = true;
    }
    require(rejected_null_cache,
            "legacy zero-array finite check accepted a null cache");

    gpu::detail::resident_cache *inactive_cache = resident.cache();
    resident.finish(false);
    bool rejected_inactive_cache = false;
    try {
      (void)gpu::detail::resident_primary_fields_are_finite_fp32(
          inactive_cache, nullptr, nullptr, 0);
    }
    catch (const std::logic_error &) {
      rejected_inactive_cache = true;
    }
    const gpu::dispatch_statistics invalid_empty_checks =
        gpu::get_dispatch_statistics();
    require(rejected_inactive_cache &&
                invalid_empty_checks.host_to_device_bytes == 0 &&
                invalid_empty_checks.device_to_host_bytes == 0,
            "invalid legacy zero-array finite checks mutated transfers");
  }
  gpu::detail::destroy_resident_cache_for_owner(&owner);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "finite descriptor/result lifecycle leaked a device buffer");
#endif
}

void require_resident_host_write_transaction() {
#if MEEP_SINGLE
  gpu::set_backend(gpu::backend_mode::cuda);
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  int owner = 0;
  std::vector<float> field(37, 0.25f);

  {
    gpu::detail::resident_curl_session resident(&owner, true);
    gpu::detail::resident_ensure_mirror_fp32(
        resident.cache(), field.data(), field.size());

    const void *host_writes[] = {field.data() + 5};
    gpu::reset_dispatch_statistics();
    bool rejected_active = false;
    try {
      gpu::detail::prepare_resident_host_writes_for_owner(
          &owner, host_writes, 1);
    }
    catch (const std::logic_error &) {
      rejected_active = true;
    }
    const gpu::dispatch_statistics after_active_rejection =
        gpu::get_dispatch_statistics();
    require(
        rejected_active &&
            after_active_rejection.host_to_device_bytes == 0 &&
            after_active_rejection.device_to_host_bytes == 0,
        "active resident host-write rejection was not transactional");
    resident.finish(false);
  }

  // Rejection above must not have invalidated the existing clean mirror.
  gpu::reset_dispatch_statistics();
  {
    gpu::detail::resident_curl_session resident(&owner, true);
    gpu::detail::resident_ensure_span_fp32(
        resident.cache(), field.data() + 5, 1);
    resident.finish(false);
  }
  require(
      gpu::get_dispatch_statistics().host_to_device_bytes == 0,
      "active host-write rejection invalidated a resident mirror");

  // A clean but resident field can be invalidated by an interior pointer.
  // Its first later subspan use must refresh the complete owning allocation
  // once, preserve the allocation, and make the host write visible to a
  // device consumer.
  const void *host_writes[] = {field.data() + 5, field.data()};
  gpu::detail::prepare_resident_host_writes_for_owner(
      &owner, host_writes, 2);
  field[5] = 3.25f;

  gpu::reset_dispatch_statistics();
  const gpu::resident_statistics resident_before_refresh =
      gpu::get_resident_statistics();
  {
    gpu::detail::resident_curl_session resident(&owner, true);
    gpu::detail::resident_ensure_span_fp32(
        resident.cache(), field.data() + 5, 1);
    const gpu::dispatch_statistics after_first_span =
        gpu::get_dispatch_statistics();
    const gpu::resident_statistics resident_after_first_span =
        gpu::get_resident_statistics();
    require(
        after_first_span.host_to_device_bytes ==
                field.size() * sizeof(float) &&
            resident_after_first_span.device_buffer_allocations ==
                resident_before_refresh.device_buffer_allocations,
        "invalidated interior span did not refresh exactly one complete "
        "resident mirror without allocation");

    gpu::detail::resident_ensure_span_fp32(
        resident.cache(), field.data() + 6, 1);
    require(
        gpu::get_dispatch_statistics().host_to_device_bytes ==
            after_first_span.host_to_device_bytes,
        "second invalidated subspan replay uploaded the mirror twice");

    const gpu::detail::indexed_value_fp32 subtract = {5, 0.5f};
    gpu::detail::resident_indexed_subtract_fp32(
        resident.cache(), field.data(), field.size(), &subtract, 1);
    resident.finish(true);
  }
  require(
      std::abs(field[5] - 2.75f) <=
          4 * std::numeric_limits<float>::epsilon(),
      "device consumer observed stale data after a host-write refresh");

  // Device-authoritative data must reject host publication before any epoch
  // changes.  After an explicit sync, the same publication is legal.
  {
    gpu::detail::resident_curl_session resident(&owner, true);
    const gpu::detail::indexed_value_fp32 subtract = {7, 0.125f};
    gpu::detail::resident_indexed_subtract_fp32(
        resident.cache(), field.data(), field.size(), &subtract, 1);
    resident.finish(false);
  }
  const float stale_host = field[7];
  bool rejected_dirty = false;
  try {
    gpu::detail::prepare_resident_host_writes_for_owner(
        &owner, host_writes, 2);
  }
  catch (const std::logic_error &) {
    rejected_dirty = true;
  }
  require(rejected_dirty && field[7] == stale_host,
          "device-authoritative host-write rejection changed host data");
  gpu::detail::sync_resident_cache_for_owner(&owner);
  gpu::detail::prepare_resident_host_writes_for_owner(
      &owner, host_writes, 2);

  gpu::detail::destroy_resident_cache_for_owner(&owner);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "host-write transaction regression leaked a resident buffer");
#endif
}

void require_distributed_finite_verdict_aggregation() {
#if MEEP_SINGLE
  if (count_processors() < 2) return;
  gpu::set_backend(gpu::backend_mode::cuda);
  const std::uint64_t global_live_before =
      global_sum(gpu::get_live_resident_device_buffers());
  int owner = 0;
  {
    std::vector<float> fields(31, 0.5f);
    gpu::detail::resident_curl_session resident(&owner, true);
    require(resident.active(),
            "distributed finite verdict did not start a resident session");
    gpu::detail::resident_ensure_mirror_fp32(
        resident.cache(), fields.data(), fields.size());
    if (am_master()) {
      const gpu::detail::indexed_value_fp32 make_nan = {
          9, std::numeric_limits<float>::quiet_NaN()};
      gpu::detail::resident_indexed_subtract_fp32(
          resident.cache(), fields.data(), fields.size(),
          &make_nan, 1);
    }

    float *arrays[] = {fields.data()};
    std::size_t counts[] = {fields.size()};
    gpu::detail::resident_finite_check_session check(
        resident.cache());
    check.accumulate(resident.cache(), arrays, counts, 1);
    const bool local_finite = check.finish();
    require(and_to_all(local_finite == !am_master()),
            "rank-local distributed finite verdict was incorrect");
    const bool global_finite = and_to_all(local_finite);
    require(!global_finite,
            "rank-zero device NaN was not rejected on every rank");
    resident.finish(false);
  }
  gpu::detail::destroy_resident_cache_for_owner(&owner);
  require(global_sum(gpu::get_live_resident_device_buffers()) ==
              global_live_before,
          "distributed finite verdict leaked a resident device buffer");
  if (am_master())
    std::cout
        << "PASS: distributed finite verdict aggregation\n";
#endif
}

void require_boundary_descriptor_fast_replay_contract() {
#if MEEP_SINGLE
  require(count_processors() == 1,
          "boundary descriptor replay contract is singleton-only");
  scoped_environment_override enable_fast_replay(
      "MEEP_GPU_DISABLE_BOUNDARY_DESCRIPTOR_FAST_REPLAY", nullptr);
  gpu::set_backend(gpu::backend_mode::cuda);
  gpu::reset_boundary_descriptor_replay_statistics();
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();

  int boundary_owner = 0;
  int boundary_buffer_token = 0;
  int source_owner = 0;
  int destination_owner = 0;
  int topology_identity = 0;
  gpu::detail::boundary_exchange_buffer *buffer =
      gpu::detail::resident_boundary_exchange_buffer(
          &boundary_owner, &boundary_buffer_token, 3);
  std::vector<float> source = {0.25f, -1.5f, 2.75f, 4.5f};
  std::vector<float> extra_source(5, 0.125f);
  std::vector<float> destination(6, 0.0f);
  std::vector<float> extra_destination(7, -0.25f);

  {
    gpu::detail::resident_curl_session source_resident(
        &source_owner, true);
    gpu::detail::resident_curl_session destination_resident(
        &destination_owner, true);
    require(source_resident.active() && destination_resident.active(),
            "boundary descriptor replay did not start resident sessions");
    gpu::detail::resident_ensure_span_fp32(
        source_resident.cache(), source.data(), source.size());
    gpu::detail::resident_ensure_span_fp32(
        destination_resident.cache(), destination.data(),
        destination.size());

    gpu::detail::resident_cache *source_caches[] = {
        source_resident.cache(), source_resident.cache(),
        source_resident.cache()};
    const float *source_pointers[] = {
        source.data(), source.data() + 1, source.data() + 2};
    gpu::detail::boundary_descriptor_replay_token replay_token =
        gpu::detail::boundary_descriptor_replay_token_test_access::make(
            &topology_identity, 1);

    gpu::detail::resident_gather_boundary_fp32(
        source_caches, buffer, source_pointers, 3, &replay_token);
    gpu::detail::boundary_exchange_synchronize(buffer);
    gpu::boundary_descriptor_replay_statistics statistics =
        gpu::get_boundary_descriptor_replay_statistics();
    require(statistics.gather_full_validations == 1 &&
                statistics.gather_fast_replays == 0,
            "cold gather did not perform exactly one full validation");

    gpu::detail::resident_gather_boundary_fp32(
        source_caches, buffer, source_pointers, 3, &replay_token);
    gpu::detail::boundary_exchange_synchronize(buffer);
    statistics = gpu::get_boundary_descriptor_replay_statistics();
    require(statistics.gather_full_validations == 1 &&
                statistics.gather_fast_replays == 1,
            "stable gather topology did not take the fast replay path");

    // The descriptor array address is unchanged while one element changes.
    // A generic null-token call must compare the elements, rebuild the device
    // plan, and gather from the new address rather than trusting the pointer.
    source_pointers[1] = source.data() + 3;
    gpu::detail::resident_gather_boundary_fp32(
        source_caches, buffer, source_pointers, 3, nullptr);
    gpu::detail::boundary_exchange_synchronize(buffer);
    gpu::detail::boundary_exchange_copy_to_host(buffer);
    require(gpu::detail::boundary_exchange_host_data(buffer)[1] ==
                source[3],
            "null-token gather accepted a same-array descriptor mutation");
    statistics = gpu::get_boundary_descriptor_replay_statistics();
    require(statistics.gather_full_validations == 2 &&
                statistics.gather_fast_replays == 1,
            "null-token gather did not force full validation");

    // A full validation binds the stable token again; only the following
    // identical call may replay without comparing all scalar descriptors.
    gpu::detail::resident_gather_boundary_fp32(
        source_caches, buffer, source_pointers, 3, &replay_token);
    gpu::detail::boundary_exchange_synchronize(buffer);
    gpu::detail::resident_gather_boundary_fp32(
        source_caches, buffer, source_pointers, 3, &replay_token);
    gpu::detail::boundary_exchange_synchronize(buffer);
    statistics = gpu::get_boundary_descriptor_replay_statistics();
    require(statistics.gather_full_validations == 3 &&
                statistics.gather_fast_replays == 2,
            "gather token was not rebound safely after generic validation");

    const std::uint64_t gather_full_before_generation =
        statistics.gather_full_validations;
    gpu::detail::resident_ensure_span_fp32(
        source_resident.cache(), extra_source.data(),
        extra_source.size());
    gpu::detail::resident_gather_boundary_fp32(
        source_caches, buffer, source_pointers, 3, &replay_token);
    gpu::detail::boundary_exchange_synchronize(buffer);
    statistics = gpu::get_boundary_descriptor_replay_statistics();
    require(statistics.gather_full_validations ==
                gather_full_before_generation + 1,
            "gather ignored a resident allocation-generation change");

    // Mutation of the token value at the same token address models topology
    // invalidation without giving pointer identity a chance to hide it.
    replay_token =
        gpu::detail::boundary_descriptor_replay_token_test_access::make(
            &topology_identity,
            replay_token.topology_generation() + 1);
    const std::uint64_t gather_full_before_token =
        statistics.gather_full_validations;
    gpu::detail::resident_gather_boundary_fp32(
        source_caches, buffer, source_pointers, 3, &replay_token);
    gpu::detail::boundary_exchange_synchronize(buffer);
    statistics = gpu::get_boundary_descriptor_replay_statistics();
    require(statistics.gather_full_validations ==
                gather_full_before_token + 1,
            "gather ignored a same-address topology-token change");

    const gpu::detail::boundary_descriptor_replay_token saturated_token =
        gpu::detail::boundary_descriptor_replay_token_test_access::make(
            &topology_identity, 0);
    const std::uint64_t gather_full_before_saturation =
        statistics.gather_full_validations;
    const std::uint64_t gather_fast_before_saturation =
        statistics.gather_fast_replays;
    gpu::detail::resident_gather_boundary_fp32(
        source_caches, buffer, source_pointers, 3, &saturated_token);
    gpu::detail::boundary_exchange_synchronize(buffer);
    statistics = gpu::get_boundary_descriptor_replay_statistics();
    require(statistics.gather_full_validations ==
                gather_full_before_saturation + 1 &&
                statistics.gather_fast_replays ==
                    gather_fast_before_saturation,
            "generation-zero topology token was incorrectly replay eligible");

    const gpu::detail::remote_boundary_operation_fp32 initial_operations[] = {
        {destination_resident.cache(), destination.data(), nullptr,
         0, 1.0f, 0.0f},
        {destination_resident.cache(), destination.data() + 1, nullptr,
         1, 1.0f, 0.0f}};
    std::vector<gpu::detail::remote_boundary_operation_fp32> operations(
        initial_operations, initial_operations + 2);
    gpu::detail::resident_scatter_boundary_fp32(
        buffer, operations.data(), operations.size(), &replay_token);
    gpu::detail::boundary_exchange_synchronize(buffer);
    gpu::detail::resident_scatter_boundary_fp32(
        buffer, operations.data(), operations.size(), &replay_token);
    gpu::detail::boundary_exchange_synchronize(buffer);
    statistics = gpu::get_boundary_descriptor_replay_statistics();
    require(statistics.scatter_full_validations == 1 &&
                statistics.scatter_fast_replays == 1,
            "stable scatter topology did not transition to fast replay");

    operations[1].destination_real = destination.data() + 2;
    gpu::detail::resident_scatter_boundary_fp32(
        buffer, operations.data(), operations.size(), nullptr);
    gpu::detail::boundary_exchange_synchronize(buffer);
    statistics = gpu::get_boundary_descriptor_replay_statistics();
    require(statistics.scatter_full_validations == 2 &&
                statistics.scatter_fast_replays == 1,
            "null-token scatter did not detect a same-array mutation");
    // Publish immediately, before any later scatter can accidentally make a
    // stale device plan appear correct.  Only the mutated second descriptor
    // can write destination[2], and the untouched neighbor remains zero.
    destination_resident.finish(true);
    require(destination[2] == source[3] && destination[3] == 0.0f,
            "null-token scatter did not publish exactly the mutated target");

    // Retire the cache/plan that produced the mutation evidence, then reset
    // host state. Subsequent token/generation tests cannot contaminate that
    // already-published oracle or rely on its descriptor plan.
    gpu::detail::destroy_resident_cache_for_owner(&destination_owner);
    std::fill(destination.begin(), destination.end(), 0.0f);
    {
      gpu::detail::resident_curl_session replay_destination_resident(
          &destination_owner, true);
      require(replay_destination_resident.active(),
              "boundary scatter replay did not restart its resident cache");
      gpu::detail::resident_ensure_span_fp32(
          replay_destination_resident.cache(), destination.data(),
          destination.size());
      for (gpu::detail::remote_boundary_operation_fp32 &operation :
           operations)
        operation.destination_cache =
            replay_destination_resident.cache();

      gpu::detail::resident_scatter_boundary_fp32(
          buffer, operations.data(), operations.size(), &replay_token);
      gpu::detail::boundary_exchange_synchronize(buffer);
      gpu::detail::resident_scatter_boundary_fp32(
          buffer, operations.data(), operations.size(), &replay_token);
      gpu::detail::boundary_exchange_synchronize(buffer);
      statistics = gpu::get_boundary_descriptor_replay_statistics();
      require(statistics.scatter_full_validations == 3 &&
                  statistics.scatter_fast_replays == 2,
              "scatter token was not rebound safely after cache retirement");

      const std::uint64_t scatter_full_before_generation =
          statistics.scatter_full_validations;
      gpu::detail::resident_ensure_span_fp32(
          replay_destination_resident.cache(), extra_destination.data(),
          extra_destination.size());
      gpu::detail::resident_scatter_boundary_fp32(
          buffer, operations.data(), operations.size(), &replay_token);
      gpu::detail::boundary_exchange_synchronize(buffer);
      statistics = gpu::get_boundary_descriptor_replay_statistics();
      require(statistics.scatter_full_validations ==
                  scatter_full_before_generation + 1,
              "scatter ignored a resident allocation-generation change");

      replay_token =
          gpu::detail::boundary_descriptor_replay_token_test_access::make(
              &topology_identity,
              replay_token.topology_generation() + 1);
      const std::uint64_t scatter_full_before_token =
          statistics.scatter_full_validations;
      gpu::detail::resident_scatter_boundary_fp32(
          buffer, operations.data(), operations.size(), &replay_token);
      gpu::detail::boundary_exchange_synchronize(buffer);
      statistics = gpu::get_boundary_descriptor_replay_statistics();
      require(statistics.scatter_full_validations ==
                  scatter_full_before_token + 1,
              "scatter ignored a same-identity topology generation change");

      {
        scoped_environment_override disable_fast_replay(
            "MEEP_GPU_DISABLE_BOUNDARY_DESCRIPTOR_FAST_REPLAY", "1");
        const std::uint64_t gather_full_before_disable =
            statistics.gather_full_validations;
        const std::uint64_t scatter_full_before_disable =
            statistics.scatter_full_validations;
        const std::uint64_t gather_fast_before_disable =
            statistics.gather_fast_replays;
        const std::uint64_t scatter_fast_before_disable =
            statistics.scatter_fast_replays;
        gpu::detail::resident_gather_boundary_fp32(
            source_caches, buffer, source_pointers, 3, &replay_token);
        gpu::detail::boundary_exchange_synchronize(buffer);
        gpu::detail::resident_scatter_boundary_fp32(
            buffer, operations.data(), operations.size(), &replay_token);
        gpu::detail::boundary_exchange_synchronize(buffer);
        statistics = gpu::get_boundary_descriptor_replay_statistics();
        require(statistics.gather_full_validations ==
                    gather_full_before_disable + 1 &&
                    statistics.scatter_full_validations ==
                        scatter_full_before_disable + 1 &&
                    statistics.gather_fast_replays ==
                        gather_fast_before_disable &&
                    statistics.scatter_fast_replays ==
                        scatter_fast_before_disable,
                "descriptor fast-replay opt-out did not force full validation");
      }
      replay_destination_resident.finish(true);
    }

    source_resident.finish(true);
  }
  gpu::detail::destroy_resident_cache_for_owner(&source_owner);
  gpu::detail::destroy_resident_cache_for_owner(&destination_owner);
  gpu::detail::destroy_boundary_exchange_for_owner(&boundary_owner);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "boundary descriptor replay regression leaked a resident buffer");
#endif
}

void require_boundary_event_record_failure_fallback() {
#if MEEP_SINGLE
  require(count_processors() == 1,
          "boundary event-record failure fallback is singleton-only");
  gpu::set_backend(gpu::backend_mode::cuda);
  gpu::reset_dispatch_statistics();
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  int boundary_owner = 0;
  int boundary_token = 0;
  int destination_owner = 0;
  gpu::detail::boundary_exchange_buffer *buffer =
      gpu::detail::resident_boundary_exchange_buffer(
          &boundary_owner, &boundary_token, 4);
  float *host = gpu::detail::boundary_exchange_host_data(buffer);
  const float expected[] = {1.25f, -0.5f, 0.75f, 2.0f};
  std::copy(expected, expected + 4, host);

  bool copy_failure_observed = false;
  gpu::detail::set_boundary_event_record_failures_for_testing(1);
  try {
    gpu::detail::boundary_exchange_copy_to_device(buffer);
  }
  catch (const std::runtime_error &error) {
    copy_failure_observed =
        std::string(error.what()).find(
            "injected CUDA boundary event-record failure") !=
        std::string::npos;
  }
  gpu::detail::set_boundary_event_record_failures_for_testing(0);
  require(copy_failure_observed,
          "async H2D event-record failure injection was not observed");
  gpu::detail::boundary_lifetime_fallback_statistics fallback_statistics =
      gpu::detail::get_boundary_lifetime_fallback_statistics();
  require(fallback_statistics.device_synchronize_fallbacks == 1 &&
              fallback_statistics.lifetime_uncertain_leaks == 0,
          "async H2D event-record failure did not directly execute the "
          "device-synchronize lifetime fallback");
  gpu::detail::boundary_exchange_copy_to_host(buffer);
  for (std::size_t index = 0; index < 4; ++index)
    require(host[index] == expected[index],
            "event-record fallback returned before async H2D completed");

  std::vector<float> destination(4, 0.0f);
  std::copy(expected, expected + 4, host);
  gpu::detail::boundary_exchange_copy_to_device(buffer);
  bool scatter_failure_observed = false;
  {
    gpu::detail::resident_curl_session resident(
        &destination_owner, true);
    gpu::detail::resident_ensure_span_fp32(
        resident.cache(), destination.data(), destination.size());
    const gpu::detail::remote_boundary_operation_fp32 operation = {
        resident.cache(), destination.data(), nullptr, 0, 1.0f, 0.0f};
    gpu::detail::set_boundary_event_record_failures_for_testing(1);
    try {
      gpu::detail::resident_scatter_boundary_fp32(
          buffer, &operation, 1);
    }
    catch (const std::runtime_error &error) {
      scatter_failure_observed =
          std::string(error.what()).find(
              "injected CUDA boundary event-record failure") !=
          std::string::npos;
    }
    gpu::detail::set_boundary_event_record_failures_for_testing(0);
    resident.finish(true);
  }
  require(scatter_failure_observed,
          "scatter event-record failure injection was not observed");
  fallback_statistics =
      gpu::detail::get_boundary_lifetime_fallback_statistics();
  require(fallback_statistics.device_synchronize_fallbacks == 2 &&
              fallback_statistics.lifetime_uncertain_leaks == 0,
          "scatter event-record failure did not directly execute the "
          "device-synchronize lifetime fallback");
  require(destination[0] == expected[0],
          "event-record fallback returned before scatter completed");

  // Exercise the separate path where a valid completion event exists but its
  // synchronization reports an error. The device-wide fallback must run and
  // the original event diagnostic must still reach the caller.
  std::copy(expected, expected + 4, host);
  gpu::detail::boundary_exchange_copy_to_device(buffer);
  gpu::detail::set_boundary_event_synchronize_failures_for_testing(1);
  bool event_synchronize_failure_observed = false;
  try {
    gpu::detail::boundary_exchange_synchronize(buffer);
  }
  catch (const std::runtime_error &error) {
    event_synchronize_failure_observed =
        std::string(error.what()).find(
            "injected CUDA boundary event-synchronize failure") !=
        std::string::npos;
  }
  gpu::detail::set_boundary_event_synchronize_failures_for_testing(0);
  fallback_statistics =
      gpu::detail::get_boundary_lifetime_fallback_statistics();
  require(event_synchronize_failure_observed &&
              fallback_statistics.device_synchronize_fallbacks == 3 &&
              fallback_statistics.lifetime_uncertain_leaks == 0,
          "event-synchronize failure did not preserve its diagnostic and "
          "execute the device-synchronize fallback");

  gpu::detail::destroy_boundary_exchange_for_owner(&boundary_owner);
  gpu::detail::destroy_resident_cache_for_owner(&destination_owner);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "event-record fallback leaked a buffer after successful device sync");
#endif
}

void require_boundary_double_fence_failure_leaks_safely() {
#if MEEP_SINGLE
  require(count_processors() == 1,
          "boundary double-fence failure probe is singleton-only");
  gpu::set_backend(gpu::backend_mode::cuda);
  gpu::reset_dispatch_statistics();
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  int boundary_owner = 0;
  int boundary_token = 0;
  gpu::detail::boundary_exchange_buffer *buffer =
      gpu::detail::resident_boundary_exchange_buffer(
          &boundary_owner, &boundary_token, 4);
  float *host = gpu::detail::boundary_exchange_host_data(buffer);
  const float expected[] = {0.25f, -1.5f, 2.75f, 4.0f};
  std::copy(expected, expected + 4, host);
  gpu::detail::boundary_exchange_copy_to_device(buffer);

  gpu::detail::set_boundary_event_synchronize_failures_for_testing(1);
  gpu::detail::set_boundary_device_synchronize_failures_for_testing(1);
  bool double_failure_observed = false;
  try {
    gpu::detail::boundary_exchange_synchronize(buffer);
  }
  catch (const std::runtime_error &error) {
    const std::string diagnostic = error.what();
    double_failure_observed =
        diagnostic.find(
            "injected CUDA boundary event-synchronize failure") !=
            std::string::npos &&
        diagnostic.find(
            "injected CUDA boundary device-synchronize failure") !=
            std::string::npos;
  }
  gpu::detail::set_boundary_event_synchronize_failures_for_testing(0);
  gpu::detail::set_boundary_device_synchronize_failures_for_testing(0);
  require(double_failure_observed,
          "double lifetime-fence failure did not retain both diagnostics");

  // The failed device-wide fence makes the raw allocations potentially live.
  // Destruction must remove registry metadata without cudaFree/free_pinned or
  // event destruction; this dedicated process exits immediately afterwards.
  gpu::detail::destroy_boundary_exchange_for_owner(&boundary_owner);
  const gpu::detail::boundary_lifetime_fallback_statistics statistics =
      gpu::detail::get_boundary_lifetime_fallback_statistics();
  require(statistics.device_synchronize_fallbacks == 1 &&
              statistics.lifetime_uncertain_leaks == 1 &&
              gpu::get_live_resident_device_buffers() == live_before + 1,
          "double lifetime-fence failure freed or misaccounted a "
          "potentially live boundary allocation");
#endif
}

void run_distributed_step_finite_abort_probe() {
#if MEEP_SINGLE
  require(count_processors() >= 2,
          "distributed step finite-abort probe requires multiple MPI ranks");
  gpu::set_backend(gpu::backend_mode::cuda);
  const grid_volume gv = vol2d(2.0, 1.6, 14.0);
  structure s(gv, vacuum, no_pml());
  fields f(&s, 0.0, 0.0, true, 64, 64);
  f.use_real_fields();
  f.initialize_field(Ez, initialized_profile);

  // Materialize the ordinary persistent-step mirrors before injecting the
  // rank-local non-finite value through the same resident CUDA machinery.
  f.step();
  write_stderr_line_atomically(
      "gpmeep-finite-abort-probe:rank=" +
      std::to_string(my_global_rank()) + ",stage=resident-step-ready");
  if (am_master()) {
    bool injected = false;
    for (int chunk_index = 0; chunk_index < f.num_chunks; ++chunk_index) {
      fields_chunk *chunk = f.chunks[chunk_index];
      if (!chunk->is_mine() || !chunk->f[Ez][0]) continue;
      write_stderr_line_atomically(
          "gpmeep-finite-abort-probe:rank=" +
          std::to_string(my_global_rank()) +
          ",stage=before-resident-session,chunk=" +
          std::to_string(chunk_index));
      gpu::detail::resident_curl_session resident(chunk, true);
      write_stderr_line_atomically(
          "gpmeep-finite-abort-probe:rank=" +
          std::to_string(my_global_rank()) +
          ",stage=before-nan-launch,chunk=" +
          std::to_string(chunk_index));
      const gpu::detail::indexed_value_fp32 make_nan = {
          0, std::numeric_limits<float>::quiet_NaN()};
      gpu::detail::resident_indexed_subtract_fp32(
          resident.cache(), chunk->f[Ez][0], chunk->gv.ntot(),
          &make_nan, 1);
      write_stderr_line_atomically(
          "gpmeep-finite-abort-probe:rank=" +
          std::to_string(my_global_rank()) +
          ",stage=after-nan-launch,chunk=" +
          std::to_string(chunk_index));
      resident.finish(false);
      write_stderr_line_atomically(
          "gpmeep-finite-abort-probe:rank=" +
          std::to_string(my_global_rank()) +
          ",stage=after-resident-finish,chunk=" +
          std::to_string(chunk_index));
      injected = true;
      break;
    }
    if (!injected)
      meep::abort(
          "rank zero could not find a resident Ez field for NaN injection");
  }
  write_stderr_line_atomically(
      "gpmeep-finite-abort-probe:rank=" +
      std::to_string(my_global_rank()) + ",stage=before-injection-barrier");
  all_wait();
  write_stderr_line_atomically(
      "gpmeep-finite-abort-probe:rank=" +
      std::to_string(my_global_rank()) + ",stage=before-failing-step");

  // This call must not return on any rank. fields::step performs its real
  // distributed finite verdict and enters the established MPI_Abort path.
  f.step();
  std::cerr << "FAIL: fields::step returned after rank-local device NaN on "
            << "rank " << my_global_rank() << '\n';
  throw std::runtime_error(
      "distributed fields::step accepted a rank-local device NaN");
#else
  throw std::runtime_error(
      "distributed CUDA finite-abort probe requires an FP32 build");
#endif
}

run_result run_monitor_consumer_case(
    gpu::backend_mode mode, bool disable_dft_phase_sharing = false,
    gpu::detail::dft_batch_statistics *dft_batch_statistics = nullptr) {
  scoped_environment_override sharing_control(
      "MEEP_GPU_DISABLE_DFT_PHASE_SHARING",
      disable_dft_phase_sharing ? "1" : nullptr);
  gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();
  const grid_volume gv = vol2d(1.8, 1.6, 10.0);
  structure s(gv, vacuum, no_pml());
  fields f(&s, 0.0, 0.0, true, 64, 64);
  f.use_real_fields();
  continuous_src_time source(0.27);
  f.add_point_source(Ez, source, gv.center(), 0.8);

  const double frequency_min = 0.22;
  const double frequency_max = 0.32;
  const int frequency_count = 2;
  dft_flux flux = f.add_dft_flux(
      X, volume(vec(1.35, 0.30), vec(1.35, 1.30)), frequency_min,
      frequency_max, frequency_count);
  volume_list energy_region(
      volume(vec(0.30, 0.25), vec(1.50, 1.35)), Ex);
  dft_energy energy = f.add_dft_energy(
      &energy_region, frequency_min, frequency_max, frequency_count);
  volume_list force_region(
      volume(vec(1.30, 0.30), vec(1.30, 1.30)), Sx);
  dft_force force = f.add_dft_force(
      &force_region, frequency_min, frequency_max, frequency_count);
  volume_list near_region(
      volume(vec(0.30, 0.25), vec(1.50, 0.25)), Sy, -1.0,
      new volume_list(
          volume(vec(1.50, 0.25), vec(1.50, 1.35)), Sx, 1.0,
          new volume_list(
              volume(vec(0.30, 1.35), vec(1.50, 1.35)), Sy, 1.0,
              new volume_list(
                  volume(vec(0.30, 0.25), vec(0.30, 1.35)), Sx, -1.0))));
  dft_near2far near = f.add_dft_near2far(
      &near_region, frequency_min, frequency_max, frequency_count);
  dft_ldos ldos(frequency_min, frequency_max, frequency_count);

  for (int step = 0; step < 48; ++step) {
    f.step();
    ldos.update(f);
  }

  run_result result;
  double *flux_values = flux.flux();
  const std::vector<std::complex<double> > complex_flux =
      flux.complexflux();
  double *electric = energy.electric();
  double *magnetic = energy.magnetic();
  double *total = energy.total();
  double *force_values = force.force();
  std::complex<double> *farfield = near.farfield(vec(2.4, 0.8));
  double *ldos_values = ldos.ldos();
  std::complex<double> *ldos_fields = ldos.F();
  std::complex<double> *ldos_currents = ldos.J();
  for (int frequency = 0; frequency < frequency_count; ++frequency) {
    result.samples.push_back(
        std::complex<double>(flux_values[frequency], 0.0));
    result.samples.push_back(complex_flux[frequency]);
    result.samples.push_back(
        std::complex<double>(electric[frequency], 0.0));
    result.samples.push_back(
        std::complex<double>(magnetic[frequency], 0.0));
    result.samples.push_back(
        std::complex<double>(total[frequency], 0.0));
    result.samples.push_back(
        std::complex<double>(force_values[frequency], 0.0));
    for (int component_index = 0; component_index < 6; ++component_index)
      result.samples.push_back(
          farfield[6 * frequency + component_index]);
    result.samples.push_back(
        std::complex<double>(ldos_values[frequency], 0.0));
    result.samples.push_back(ldos_fields[frequency]);
    result.samples.push_back(ldos_currents[frequency]);
  }
  delete[] flux_values;
  delete[] electric;
  delete[] magnetic;
  delete[] total;
  delete[] force_values;
  delete[] farfield;
  delete[] ldos_values;
  delete[] ldos_fields;
  delete[] ldos_currents;
  result.dfts = gpu::get_dft_statistics();
  result.dft_reductions = gpu::get_dft_reduction_statistics();
  if (dft_batch_statistics)
    *dft_batch_statistics = gpu::detail::get_dft_batch_statistics();
  near.remove();
  force.remove();
  energy.remove();
  flux.remove();
  aggregate_statistics(result);
  return result;
}

struct near2far3d_result {
  std::vector<double> fields;
  gpu::near2far_statistics near2far;
  gpu::resident_statistics resident;
  gpu::dispatch_statistics dispatch;
  std::size_t output_points = 0;
  std::size_t frequency_count = 0;
};

near2far3d_result run_near2far3d_transform_case(
    gpu::backend_mode mode) {
  gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();
  const grid_volume gv = vol3d(2.0, 1.8, 1.6, 8.0);
  structure s(gv, vacuum, no_pml());
  fields f(&s, 0.0, 0.0, true, 64, 64);
  const vec center = gv.center();
  gaussian_src_time source(0.31, 0.12);
  f.add_point_source(Ez, source, center, 0.75);
  f.add_point_source(Hx, source, center + vec(0.08, -0.05, 0.03),
                     std::complex<double>(-0.21, 0.17));

  const double hx = 0.46;
  const double hy = 0.38;
  const double hz = 0.30;
  volume_list near_region(
      volume(vec(center.x() + hx, center.y() - hy, center.z() - hz),
             vec(center.x() + hx, center.y() + hy, center.z() + hz)),
      Sx, 1.0,
      new volume_list(
          volume(vec(center.x() - hx, center.y() - hy, center.z() - hz),
                 vec(center.x() - hx, center.y() + hy, center.z() + hz)),
          Sx, -1.0,
          new volume_list(
              volume(vec(center.x() - hx, center.y() + hy,
                         center.z() - hz),
                     vec(center.x() + hx, center.y() + hy,
                         center.z() + hz)),
              Sy, 1.0,
              new volume_list(
                  volume(vec(center.x() - hx, center.y() - hy,
                             center.z() - hz),
                         vec(center.x() + hx, center.y() - hy,
                             center.z() + hz)),
                  Sy, -1.0,
                  new volume_list(
                      volume(vec(center.x() - hx, center.y() - hy,
                                 center.z() + hz),
                             vec(center.x() + hx, center.y() + hy,
                                 center.z() + hz)),
                      Sz, 1.0,
                      new volume_list(
                          volume(vec(center.x() - hx, center.y() - hy,
                                     center.z() - hz),
                                 vec(center.x() + hx, center.y() + hy,
                                     center.z() - hz)),
                          Sz, -1.0))))));
  constexpr double frequencies[] = {0.23, 0.31, 0.41};
  static_assert(frequencies[0] > 0.0 && frequencies[1] > 0.0 &&
                    frequencies[2] > 0.0,
                "the 2D CUDA Near2Far qualification requires positive "
                "frequencies");
  dft_near2far near =
      f.add_dft_near2far(&near_region, frequencies, 3);
  for (int step = 0; step < 72; ++step) f.step();

  size_t dims[4] = {1, 1, 1, 1};
  int rank = 0;
  size_t output_points = 0;
  const volume targets(
      vec(center.x() + 1.4, center.y() - 0.35, center.z() - 0.25),
      vec(center.x() + 1.9, center.y() + 0.35, center.z() + 0.25));
  std::unique_ptr<double[]> raw(
      near.get_farfields_array(targets, rank, dims, output_points, 5.0));
  require(raw != nullptr && output_points > 1,
          "3D near-to-far transform produced no output grid");
  near2far3d_result result;
  result.output_points = output_points;
  result.frequency_count = 3;
  result.fields.assign(raw.get(), raw.get() + 12 * output_points * 3);
  result.near2far = gpu::get_near2far_statistics();
  result.resident = gpu::get_resident_statistics();
  result.dispatch = gpu::get_dispatch_statistics();
  near.remove();
  return result;
}

void require_near2far3d_transform_equivalence() {
  const near2far3d_result cpu =
      run_near2far3d_transform_case(gpu::backend_mode::cpu);
  const near2far3d_result cuda =
      run_near2far3d_transform_case(gpu::backend_mode::cuda);
  require(cpu.fields.size() == cuda.fields.size() &&
              cpu.output_points == cuda.output_points &&
              cpu.frequency_count == cuda.frequency_count,
          "CPU/CUDA 3D near-to-far output shapes differ");
  double difference_squared = 0.0;
  double reference_squared = 0.0;
  double maximum_error = 0.0;
  double maximum_reference = 0.0;
  for (std::size_t index = 0; index < cpu.fields.size(); ++index) {
    const double difference = cuda.fields[index] - cpu.fields[index];
    difference_squared += difference * difference;
    reference_squared += cpu.fields[index] * cpu.fields[index];
    maximum_error = std::max(maximum_error, std::abs(difference));
    maximum_reference =
        std::max(maximum_reference, std::abs(cpu.fields[index]));
  }
  const double nrmse =
      std::sqrt(difference_squared / std::max(reference_squared, 1.0e-300));
  const double normalized_maximum =
      maximum_error / std::max(maximum_reference, 1.0e-300);
  require(nrmse <= 5.0e-4 && normalized_maximum <= 3.0e-3,
          "CUDA 3D near-to-far differs from CPU: nrmse=" +
              std::to_string(nrmse) + " normalized_maximum=" +
              std::to_string(normalized_maximum));
  require(cpu.near2far.cpu_transform_calls > 0 &&
              cpu.near2far.cuda_transform_calls == 0,
          "CPU 3D near-to-far transform counters are inconsistent");
  require(cuda.near2far.cuda_transform_calls == 1 &&
              cuda.near2far.cpu_transform_calls == 0 &&
              cuda.near2far.cuda_terms > 0 &&
              cuda.near2far.cuda_submitted_chunks > 0 &&
              cuda.near2far.cuda_target_tiles > 0 &&
              cuda.near2far.cuda_fast_precision_calls +
                      cuda.near2far.cuda_mixed_precision_calls ==
                  1 + cuda.near2far.cuda_cancellation_retries &&
              cuda.near2far.cuda_kernel_launches ==
                  2 * cuda.near2far.cuda_operation_tiles &&
              cuda.near2far.cuda_result_device_to_host_bytes >=
                  12 * cuda.output_points * cuda.frequency_count *
                      sizeof(double) &&
              cuda.near2far.cuda_condition_device_to_host_bytes > 0 &&
              cuda.near2far.dft_device_to_host_bytes_avoided > 0,
          "required CUDA 3D near-to-far transform lacks strict execution "
          "or transfer evidence: fast=" +
              std::to_string(cuda.near2far.cuda_fast_precision_calls) +
              " mixed=" +
              std::to_string(cuda.near2far.cuda_mixed_precision_calls) +
              " retries=" +
              std::to_string(cuda.near2far.cuda_cancellation_retries) +
              " operation-tiles=" +
              std::to_string(cuda.near2far.cuda_operation_tiles) +
              " launches=" +
              std::to_string(cuda.near2far.cuda_kernel_launches) +
              " result-bytes=" +
              std::to_string(
                  cuda.near2far.cuda_result_device_to_host_bytes) +
              " condition-bytes=" +
              std::to_string(
                  cuda.near2far.cuda_condition_device_to_host_bytes));
}

struct near2far_adjoint_result {
  std::vector<std::complex<double> > amplitudes;
  std::vector<ptrdiff_t> indices;
  std::vector<component> components;
  gpu::near2far_statistics statistics;
  std::size_t source_points = 0;
  std::size_t chunks = 0;
  std::uint64_t live_before = 0;
  std::uint64_t live_after = 0;
  std::uint64_t first_host_to_device_bytes = 0;
  std::uint64_t repeated_host_to_device_bytes = 0;
};

near2far_adjoint_result run_near2far_adjoint_case(
    gpu::backend_mode mode, std::size_t workspace_ceiling = 0,
    bool repeat_identical_call = false) {
  gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();
  const grid_volume gv = vol3d(2.0, 1.8, 1.6, 8.0);
  structure s(gv, vacuum, no_pml());
  fields f(&s, 0.0, 0.0, true, 64, 64);
  const vec center = gv.center();
  gaussian_src_time source_time(0.31, 0.12);
  f.add_point_source(Ez, source_time, center, 0.75);
  const double hx = 0.46;
  const double hy = 0.38;
  const double hz = 0.30;
  volume_list near_region(
      volume(vec(center.x() + hx, center.y() - hy, center.z() - hz),
             vec(center.x() + hx, center.y() + hy, center.z() + hz)),
      Sx, 1.0,
      new volume_list(
          volume(vec(center.x() - hx, center.y() - hy,
                     center.z() - hz),
                 vec(center.x() - hx, center.y() + hy,
                     center.z() + hz)),
          Sx, -1.0,
          new volume_list(
              volume(vec(center.x() - hx, center.y() + hy,
                         center.z() - hz),
                     vec(center.x() + hx, center.y() + hy,
                         center.z() + hz)),
              Sy, 1.0,
              new volume_list(
                  volume(vec(center.x() - hx, center.y() - hy,
                             center.z() - hz),
                         vec(center.x() + hx, center.y() - hy,
                             center.z() + hz)),
                  Sy, -1.0,
                  new volume_list(
                      volume(vec(center.x() - hx, center.y() - hy,
                                 center.z() + hz),
                             vec(center.x() + hx, center.y() + hy,
                                 center.z() + hz)),
                      Sz, 1.0,
                      new volume_list(
                          volume(vec(center.x() - hx, center.y() - hy,
                                     center.z() - hz),
                                 vec(center.x() + hx, center.y() + hy,
                                     center.z() - hz)),
                          Sz, -1.0))))));
  constexpr double frequencies[] = {0.23, 0.41};
  dft_near2far near =
      f.add_dft_near2far(&near_region, frequencies, 2);
  const vec far_points[] = {
      vec(center.x() + 2.2, center.y() - 0.7, center.z() + 0.9),
      vec(center.x() - 1.8, center.y() + 2.4, center.z() - 1.1)};
  double far_point_list[] = {
      far_points[0].x(), far_points[0].y(), far_points[0].z(),
      far_points[1].x(), far_points[1].y(), far_points[1].z()};
  std::vector<std::complex<double> > gradient(2 * 2 * 6);
  for (size_t index = 0; index < gradient.size(); ++index)
    gradient[index] = std::complex<double>(
        0.13 + 0.007 * index, -0.09 + 0.005 * index);
  near2far_adjoint_result result;
  result.live_before = gpu::get_live_resident_device_buffers();
  const std::size_t previous_workspace_ceiling =
      workspace_ceiling
          ? gpu::detail::exchange_near2far_workspace_ceiling_for_testing(
                workspace_ceiling)
          : 0;
  std::vector<sourcedata> profiles;
  try {
    profiles = near.near_sourcedata(
        far_points[0], far_point_list, 2, gradient.data(), 1.0e-3);
    const gpu::near2far_statistics first_statistics =
        gpu::get_near2far_statistics();
    result.first_host_to_device_bytes =
        first_statistics.cuda_adjoint_host_to_device_bytes;
    if (repeat_identical_call) {
      const std::vector<sourcedata> repeated = near.near_sourcedata(
          far_points[0], far_point_list, 2, gradient.data(), 1.0e-3);
      require(repeated.size() == profiles.size(),
              "cached adjoint Near2Far changed source-profile count");
      for (size_t chunk = 0; chunk < profiles.size(); ++chunk)
        require(repeated[chunk].near_fd_comp == profiles[chunk].near_fd_comp &&
                    repeated[chunk].idx_arr == profiles[chunk].idx_arr &&
                    repeated[chunk].amp_arr == profiles[chunk].amp_arr,
                "cached adjoint Near2Far changed an identical result");
      const gpu::near2far_statistics repeated_statistics =
          gpu::get_near2far_statistics();
      result.repeated_host_to_device_bytes =
          repeated_statistics.cuda_adjoint_host_to_device_bytes -
          first_statistics.cuda_adjoint_host_to_device_bytes;
    }
  }
  catch (...) {
    if (workspace_ceiling)
      gpu::detail::exchange_near2far_workspace_ceiling_for_testing(
          previous_workspace_ceiling);
    throw;
  }
  if (workspace_ceiling)
    gpu::detail::exchange_near2far_workspace_ceiling_for_testing(
        previous_workspace_ceiling);
  for (const sourcedata &profile : profiles) {
    ++result.chunks;
    result.source_points += profile.idx_arr.size();
    result.components.push_back(profile.near_fd_comp);
    result.indices.insert(result.indices.end(), profile.idx_arr.begin(),
                          profile.idx_arr.end());
    result.amplitudes.insert(result.amplitudes.end(),
                             profile.amp_arr.begin(),
                             profile.amp_arr.end());
  }
  result.statistics = gpu::get_near2far_statistics();
  near.remove();
  result.live_after = gpu::get_live_resident_device_buffers();
  return result;
}

void require_near2far_adjoint_retained_plan_reuse() {
  const near2far_adjoint_result cached = run_near2far_adjoint_case(
      gpu::backend_mode::cuda, 0, true);
  const std::uint64_t attempts =
      2 + cached.statistics.cuda_adjoint_cancellation_retries;
  require(
      cached.statistics.cuda_adjoint_calls == 2 &&
          cached.statistics.cuda_adjoint_kernel_launches == attempts &&
          cached.first_host_to_device_bytes > 0 &&
          cached.repeated_host_to_device_bytes == 0 &&
          cached.statistics.cuda_adjoint_descriptor_uploads < attempts &&
          cached.live_before == cached.live_after,
      "identical adjoint Near2Far calls did not reuse retained CUDA "
      "descriptors/workspace: first_h2d=" +
          std::to_string(cached.first_host_to_device_bytes) +
          " repeated_h2d=" +
          std::to_string(cached.repeated_host_to_device_bytes) +
          " uploads=" +
          std::to_string(cached.statistics.cuda_adjoint_descriptor_uploads) +
          " launches=" +
          std::to_string(cached.statistics.cuda_adjoint_kernel_launches));
}

void require_near2far_adjoint_cancellation_retry() {
  gpu::set_backend(gpu::backend_mode::cuda);
  gpu::reset_dispatch_statistics();
  const gpu::detail::near2far_point_fp64 source_points[] = {
      {0.17, -0.23, 0.31}};
  const std::complex<double> source_amplitudes[] = {
      std::complex<double>(0.73, -0.19)};
  const gpu::detail::near2far_adjoint_request_fp64 request = {
      source_points, source_amplitudes, 1, 1, true};
  const gpu::detail::near2far_point_fp64 targets[] = {
      {2.1, -1.4, 0.9}, {2.1, -1.4, 0.9}};
  const double frequencies[] = {0.37};
  const gpu::detail::near2far_periodic_copy_fp64 copies[] = {
      {{0.0, 0.0, 0.0}, 1.0, 0.0}};
  std::complex<double> gradient[12];
  for (std::size_t component = 0; component < 6; ++component) {
    const std::complex<double> value(
        0.21 + 0.017 * component, -0.13 + 0.011 * component);
    gradient[component] = value;
    gradient[6 + component] = -value;
  }
  std::complex<double> output;
  bool used_mixed_precision = false;
  int plan_owner = 0;
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  gpu::detail::near2far_adjoint_cartesian_fp32(
      gpu::detail::near2far_cartesian_dimension::three, &plan_owner,
      &request, 1, targets, 2, frequencies, 1, copies, 1, 1.0, 1.0,
      gradient, &output, &used_mixed_precision);
  const gpu::near2far_statistics statistics =
      gpu::get_near2far_statistics();
  gpu::detail::destroy_resident_near2far_plan_for_owner(&plan_owner);
  const std::uint64_t live_after =
      gpu::get_live_resident_device_buffers();
  require(
      used_mixed_precision && std::isfinite(real(output)) &&
          std::isfinite(imag(output)) && std::abs(output) <= 1.0e-12 &&
          statistics.cuda_adjoint_calls == 1 &&
          statistics.cpu_adjoint_calls == 0 &&
          statistics.cuda_adjoint_fast_precision_calls == 1 &&
          statistics.cuda_adjoint_mixed_precision_calls == 1 &&
          statistics.cuda_adjoint_cancellation_retries == 1 &&
          statistics.cuda_adjoint_kernel_launches == 2 &&
          live_before == live_after,
      "destructive adjoint Near2Far cancellation did not retry on mixed "
      "CUDA or leaked its retained plan: output=" +
          std::to_string(std::abs(output)) + " fast=" +
          std::to_string(
              statistics.cuda_adjoint_fast_precision_calls) +
          " mixed=" +
          std::to_string(
              statistics.cuda_adjoint_mixed_precision_calls) +
          " retries=" +
          std::to_string(statistics.cuda_adjoint_cancellation_retries));
}

void require_near2far_adjoint_negative_radial_copy() {
#if MEEP_SINGLE && MEEP_HAVE_CUDA
  gpu::set_backend(gpu::backend_mode::cuda);
  gpu::reset_dispatch_statistics();
  const gpu::detail::near2far_point_fp64 source_points[] = {
      {0.15, 0.0, -0.10}};
  const std::complex<double> source_amplitudes[] = {
      std::complex<double>(0.73, -0.19)};
  const gpu::detail::near2far_adjoint_request_fp64 request = {
      source_points, source_amplitudes, 1, 1, false};
  const gpu::detail::near2far_point_fp64 target = {-2.2, 1.7, 0.7};
  // The periodic R image is negative. CPU greencyl rotates this signed
  // coordinate normally, so the CUDA selector must bound it as a ring of
  // radius |R| rather than reject it or use a non-conservative distance.
  const gpu::detail::near2far_periodic_copy_fp64 copy = {
      {-0.40, 0.0, 0.07}, std::cos(0.23), std::sin(0.23)};
  constexpr double frequency = -0.31;
  constexpr double mode = -1.5;

  // Exercise the ordinary forward transform as well as its adjoint. This is
  // intentionally a deferred fast call: it exposes the selector's bound for
  // the negative-R source ring and prevents a mixed retry from hiding a bad
  // fast-path distance envelope.
  float dft[] = {0.73f, -0.19f};
  int resident_owner = 0;
  int forward_plan_owner = 0;
  const gpu::detail::near2far_request_fp32 forward_request = {
      &resident_owner, dft, source_points, 1, 1, false};
  std::complex<double> forward_observed[6] = {};
  double forward_bound = std::numeric_limits<double>::quiet_NaN();
  bool forward_used_mixed = true;
  gpu::reset_dispatch_statistics();
  try {
    gpu::detail::resident_near2far_cartesian_fp32(
        gpu::detail::near2far_cartesian_dimension::cylindrical,
        &forward_plan_owner, &forward_request, 1, &target, 1,
        &frequency, 1, &copy, 1, 1.7, 0.9, forward_observed,
        &forward_bound, true, false, false, &forward_used_mixed, mode,
        1.0e-6);
  }
  catch (...) {
    gpu::detail::destroy_resident_near2far_plan_for_owner(
        &forward_plan_owner);
    gpu::detail::destroy_resident_cache_for_owner(&resident_owner);
    throw;
  }
  const gpu::near2far_statistics forward_statistics =
      gpu::get_near2far_statistics();

  std::complex<double> forward_reference[6] = {};
  greencyl(forward_reference, vec(target.x, target.y, target.z), frequency,
           1.7, 0.9,
           veccyl(source_points[0].x + copy.displacement.x,
                  source_points[0].z + copy.displacement.z),
           Hp, std::complex<double>(dft[0], dft[1]), mode, 1.0e-10);
  const std::complex<double> phase(copy.phase_real,
                                   copy.phase_imaginary);
  double forward_error = 0.0;
  double forward_reference_maximum = 0.0;
  double forward_observed_scalar_maximum = 0.0;
  bool forward_empirical_finite = true;
  for (std::size_t component = 0; component < 6; ++component) {
    forward_reference[component] *= phase;
    forward_empirical_finite =
        forward_empirical_finite &&
        std::isfinite(forward_observed[component].real()) &&
        std::isfinite(forward_observed[component].imag());
    forward_error = std::max(
        forward_error,
        std::abs(forward_observed[component] -
                 forward_reference[component]));
    forward_reference_maximum = std::max(
        forward_reference_maximum,
        std::abs(forward_reference[component]));
    forward_observed_scalar_maximum = std::max(
        forward_observed_scalar_maximum,
        std::max(std::abs(forward_observed[component].real()),
                 std::abs(forward_observed[component].imag())));
  }
  const double forward_accepted_error =
      3.0e-3 * forward_observed_scalar_maximum + 3.0e-8;
  const bool forward_bound_publishable =
      std::isfinite(forward_bound) && forward_bound >= 0.0 &&
      forward_bound <= forward_accepted_error;
  bool production_retry_valid = true;
  if (!forward_bound_publishable) {
    std::complex<double> production_observed[6] = {};
    bool production_used_mixed = false;
    gpu::reset_dispatch_statistics();
    try {
      gpu::detail::resident_near2far_cartesian_fp32(
          gpu::detail::near2far_cartesian_dimension::cylindrical,
          &forward_plan_owner, &forward_request, 1, &target, 1,
          &frequency, 1, &copy, 1, 1.7, 0.9, production_observed,
          nullptr, false, false, false, &production_used_mixed, mode,
          1.0e-6);
    }
    catch (...) {
      gpu::detail::destroy_resident_near2far_plan_for_owner(
          &forward_plan_owner);
      gpu::detail::destroy_resident_cache_for_owner(&resident_owner);
      throw;
    }
    double production_error = 0.0;
    bool production_finite = true;
    for (std::size_t component = 0; component < 6; ++component) {
      production_finite =
          production_finite &&
          std::isfinite(production_observed[component].real()) &&
          std::isfinite(production_observed[component].imag());
      production_error = std::max(
          production_error,
          std::abs(production_observed[component] -
                   forward_reference[component]));
    }
    const gpu::near2far_statistics production_statistics =
        gpu::get_near2far_statistics();
    production_retry_valid =
        production_used_mixed && production_finite &&
        production_error <=
            4.0e-3 * forward_reference_maximum + 3.0e-8 &&
        production_statistics.cuda_transform_calls == 1 &&
        production_statistics.cpu_transform_calls == 0 &&
        production_statistics.cuda_fast_precision_calls == 1 &&
        production_statistics.cuda_mixed_precision_calls == 1 &&
        production_statistics.cuda_cancellation_retries == 1;
  }
  gpu::detail::destroy_resident_near2far_plan_for_owner(
      &forward_plan_owner);
  gpu::detail::destroy_resident_cache_for_owner(&resident_owner);
  require(
      !forward_used_mixed && forward_empirical_finite &&
          (!std::isfinite(forward_bound) ||
           forward_error <=
               forward_bound * (1.0 + 1.0e-12) + 1.0e-15) &&
          forward_error <=
              4.0e-3 * forward_reference_maximum + 3.0e-8 &&
          (forward_bound_publishable || production_retry_valid) &&
          forward_statistics.cuda_transform_calls == 1 &&
          forward_statistics.cpu_transform_calls == 0 &&
          forward_statistics.cuda_fast_precision_calls == 1 &&
          forward_statistics.cuda_mixed_precision_calls == 0 &&
          forward_statistics.cuda_periodic_copies == 1,
      "cylindrical forward Near2Far rejected, under-bounded, or "
      "miscomputed a negative-R periodic image: error=" +
          std::to_string(forward_error) + " bound=" +
          std::to_string(forward_bound));

  std::complex<double> gradient[6];
  for (std::size_t component = 0; component < 6; ++component)
    gradient[component] = std::complex<double>(
        0.17 + 0.013 * component, -0.11 + 0.009 * component);
  std::complex<double> observed;
  int plan_owner = 0;
  gpu::reset_dispatch_statistics();
  try {
    gpu::detail::near2far_adjoint_cartesian_fp32(
        gpu::detail::near2far_cartesian_dimension::cylindrical,
        &plan_owner, &request, 1, &target, 1, &frequency, 1, &copy, 1,
        1.7, 0.9, gradient, &observed, nullptr, mode, 1.0e-6);
  }
  catch (...) {
    gpu::detail::destroy_resident_near2far_plan_for_owner(&plan_owner);
    throw;
  }
  const gpu::near2far_statistics statistics =
      gpu::get_near2far_statistics();
  gpu::detail::destroy_resident_near2far_plan_for_owner(&plan_owner);

  std::complex<double> fields[6] = {};
  greencyl(fields, vec(target.x, target.y, target.z), frequency, 1.7, 0.9,
           veccyl(source_points[0].x + copy.displacement.x,
                  source_points[0].z + copy.displacement.z),
           Hp, source_amplitudes[0], mode, 1.0e-10);
  std::complex<double> reference;
  for (std::size_t component = 0; component < 6; ++component)
    reference += fields[component] * phase * gradient[component];
  const double error = std::abs(observed - reference);
  require(
      std::isfinite(real(observed)) && std::isfinite(imag(observed)) &&
          error <= 4.0e-3 * std::abs(reference) + 3.0e-8 &&
          statistics.cuda_adjoint_calls == 1 &&
          statistics.cpu_adjoint_calls == 0 &&
          statistics.cuda_adjoint_periodic_copies == 1,
      "cylindrical adjoint Near2Far rejected or miscomputed a negative-R "
      "periodic image: error=" + std::to_string(error));
  if (am_master())
    std::cout << "PASS: cylindrical forward/adjoint arbitrary Cartesian "
                 "target, negative-R periodic image, negative frequency, "
                 "and noninteger m agree with CPU and honor the published "
                 "fast-error bound\n";
#endif
}

void require_near2far_adjoint_all_axis_tiling() {
#if MEEP_SINGLE && MEEP_HAVE_CUDA
  gpu::set_backend(gpu::backend_mode::cuda);
  constexpr std::size_t source_count = 4;
  constexpr std::size_t target_count = 3;
  constexpr std::size_t frequency_count = 2;
  constexpr std::size_t copy_count = 16;
  gpu::detail::near2far_point_fp64 source_points[source_count];
  std::complex<double> source_amplitudes[source_count];
  for (std::size_t source = 0; source < source_count; ++source) {
    source_points[source] = {
        -0.31 + 0.14 * source, 0.19 - 0.07 * source,
        -0.23 + 0.09 * source};
    source_amplitudes[source] = std::complex<double>(
        0.37 + 0.031 * source, -0.21 + 0.017 * source);
  }
  const gpu::detail::near2far_adjoint_request_fp64 request = {
      source_points, source_amplitudes, source_count, 0, true};
  const gpu::detail::near2far_point_fp64 targets[target_count] = {
      {2.3, -1.4, 1.1}, {-1.8, 2.7, -0.9}, {3.2, 1.3, 2.1}};
  const double frequencies[frequency_count] = {0.23, 0.41};
  gpu::detail::near2far_periodic_copy_fp64 copies[copy_count];
  for (std::size_t copy = 0; copy < copy_count; ++copy) {
    const double phase = 0.071 * static_cast<double>(copy);
    copies[copy] = {
        {0.013 * static_cast<double>(copy),
         -0.009 * static_cast<double>(copy),
         0.007 * static_cast<double>(copy)},
        std::cos(phase), std::sin(phase)};
  }
  std::complex<double> gradient[
      6 * target_count * frequency_count];
  for (std::size_t target = 0; target < target_count; ++target)
    for (std::size_t frequency = 0; frequency < frequency_count;
         ++frequency)
      for (std::size_t component = 0; component < 6; ++component)
        gradient[6 * (target * frequency_count + frequency) + component] =
            std::complex<double>(
                0.13 + 0.017 * target - 0.011 * frequency +
                    0.007 * component,
                -0.09 + 0.013 * target + 0.019 * frequency -
                    0.005 * component);

  std::complex<double> reference[source_count * frequency_count] = {};
  for (std::size_t source = 0; source < source_count; ++source)
    for (std::size_t frequency = 0; frequency < frequency_count;
         ++frequency)
      for (std::size_t target = 0; target < target_count; ++target)
        for (const auto &copy : copies) {
          std::complex<double> fields[6] = {};
          green3d(
              fields,
              vec(targets[target].x, targets[target].y,
                  targets[target].z),
              frequencies[frequency], 2.1, 1.3,
              vec(source_points[source].x + copy.displacement.x,
                  source_points[source].y + copy.displacement.y,
                  source_points[source].z + copy.displacement.z),
              Ex, source_amplitudes[source]);
          const std::complex<double> phase(copy.phase_real,
                                           copy.phase_imaginary);
          for (std::size_t component = 0; component < 6; ++component)
            reference[source * frequency_count + frequency] +=
                fields[component] * phase *
                gradient[6 * (target * frequency_count + frequency) +
                         component];
        }

  constexpr std::size_t forced_workspace_ceiling = 256;
  const std::size_t previous_workspace_ceiling =
      gpu::detail::exchange_near2far_workspace_ceiling_for_testing(
          forced_workspace_ceiling);
  std::complex<double> tiled[source_count * frequency_count] = {};
  int plan_owner = 0;
  gpu::near2far_statistics statistics = {};
  try {
    gpu::reset_dispatch_statistics();
    gpu::detail::near2far_adjoint_cartesian_fp32(
        gpu::detail::near2far_cartesian_dimension::three, &plan_owner,
        &request, 1, targets, target_count, frequencies, frequency_count,
        copies, copy_count, 2.1, 1.3, gradient, tiled);
    statistics = gpu::get_near2far_statistics();
  }
  catch (...) {
    gpu::detail::exchange_near2far_workspace_ceiling_for_testing(
        previous_workspace_ceiling);
    gpu::detail::destroy_resident_near2far_plan_for_owner(&plan_owner);
    throw;
  }
  gpu::detail::exchange_near2far_workspace_ceiling_for_testing(
      previous_workspace_ceiling);
  gpu::detail::destroy_resident_near2far_plan_for_owner(&plan_owner);

  double reference_scale = 1.0e-300;
  double maximum_error = 0.0;
  for (std::size_t work = 0; work < source_count * frequency_count;
       ++work) {
    reference_scale = std::max(reference_scale, std::abs(reference[work]));
    maximum_error =
        std::max(maximum_error, std::abs(tiled[work] - reference[work]));
  }
  // At 256 bytes even the smaller FP32 plan must split source 4->2,
  // target 3->1, frequency 2->1, and copy 16->4. Hence at least 48
  // cooperative launches are required for one fast attempt (mixed requires
  // still more). This proves the host dJ subset packing and every accumulate
  // axis, rather than merely observing that some tiling occurred.
  require(
      maximum_error <= 4.0e-3 * reference_scale + 3.0e-8 &&
          statistics.cuda_adjoint_calls == 1 &&
          statistics.cpu_adjoint_calls == 0 &&
          statistics.cuda_adjoint_source_points == source_count &&
          statistics.cuda_adjoint_far_points == target_count &&
          statistics.cuda_adjoint_frequencies == frequency_count &&
          statistics.cuda_adjoint_periodic_copies == copy_count &&
          statistics.cuda_adjoint_maximum_workspace_bytes > 0 &&
          statistics.cuda_adjoint_maximum_workspace_bytes <=
              forced_workspace_ceiling &&
          statistics.cuda_adjoint_kernel_launches >= 48,
      "adjoint Near2Far 256-byte all-axis tiling changed the CPU result or "
      "did not split source/target/frequency/copy axes: error=" +
          canonical_decimal(maximum_error) + " launches=" +
          std::to_string(statistics.cuda_adjoint_kernel_launches));
  if (am_master())
    std::cout << "PASS: adjoint Near2Far 256-byte workspace tiles source, "
                 "target, frequency/dJ, and periodic-copy axes with CPU "
                 "agreement\n";
#endif
}

void require_near2far_adjoint_two_device_migration() {
#if MEEP_SINGLE && MEEP_HAVE_CUDA
  gpu::set_backend(gpu::backend_mode::cuda);
  std::vector<int> compatible_ordinals;
  for (const gpu::device_info &device : gpu::enumerate_devices())
    if (device.compatible) compatible_ordinals.push_back(device.ordinal);
  if (compatible_ordinals.size() < 2) {
    if (am_master())
      std::cout << "SKIP: adjoint Near2Far retained-plan migration requires "
                   "two compatible GPUs\n";
    return;
  }

  const int original_ordinal = gpu::selected_device();
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  const gpu::detail::near2far_point_fp64 source_point =
      {0.17, -0.23, 0.31};
  const std::complex<double> source_amplitude(0.73, -0.19);
  const gpu::detail::near2far_adjoint_request_fp64 request = {
      &source_point, &source_amplitude, 1, 1, true};
  const gpu::detail::near2far_point_fp64 target = {2.1, -1.4, 0.9};
  constexpr double frequency = 0.37;
  const gpu::detail::near2far_periodic_copy_fp64 copy = {
      {0.0, 0.0, 0.0}, 1.0, 0.0};
  std::complex<double> gradient[6];
  for (std::size_t component = 0; component < 6; ++component)
    gradient[component] = std::complex<double>(
        0.21 + 0.017 * component, -0.13 + 0.011 * component);
  int plan_owner = 0;
  gpu::near2far_statistics device_statistics[3] = {};
  std::complex<double> results[3];
  const int ordinals[3] = {compatible_ordinals[0],
                           compatible_ordinals[1],
                           compatible_ordinals[0]};
  try {
    for (int pass = 0; pass < 3; ++pass) {
      gpu::select_device(ordinals[pass]);
      gpu::reset_dispatch_statistics();
      gpu::detail::near2far_adjoint_cartesian_fp32(
          gpu::detail::near2far_cartesian_dimension::three, &plan_owner,
          &request, 1, &target, 1, &frequency, 1, &copy, 1, 1.0, 1.0,
          gradient, &results[pass]);
      device_statistics[pass] = gpu::get_near2far_statistics();
      require(gpu::selected_device() == ordinals[pass],
              "adjoint Near2Far migration changed the selected device");
    }
  }
  catch (...) {
    gpu::detail::destroy_resident_near2far_plan_for_owner(&plan_owner);
    if (original_ordinal >= 0) gpu::select_device(original_ordinal);
    throw;
  }
  const std::uint64_t warm_live =
      gpu::get_live_resident_device_buffers();
  gpu::detail::destroy_resident_near2far_plan_for_owner(&plan_owner);
  if (original_ordinal >= 0) gpu::select_device(original_ordinal);

  const double scale = std::max(1.0, std::abs(results[0]));
  require(
      std::abs(results[1] - results[0]) <= 5.0e-6 * scale &&
          results[2] == results[0] &&
          device_statistics[1].cuda_adjoint_calls == 1 &&
          device_statistics[2].cuda_adjoint_calls == 1 &&
          device_statistics[1].cpu_adjoint_calls == 0 &&
          device_statistics[2].cpu_adjoint_calls == 0 &&
          device_statistics[1].cuda_adjoint_host_to_device_bytes > 0 &&
          device_statistics[2].cuda_adjoint_host_to_device_bytes > 0 &&
          warm_live > live_before &&
          gpu::get_live_resident_device_buffers() == live_before,
      "adjoint Near2Far retained plan did not migrate and rebuild across "
      "two GPUs without changing results or leaking buffers");

  const std::uint64_t forward_live_before =
      gpu::get_live_resident_device_buffers();
  int forward_resident_owner = 0;
  int forward_plan_owner = 0;
  float forward_dft[2] = {0.73f, -0.19f};
  const gpu::detail::near2far_request_fp32 forward_request = {
      &forward_resident_owner, forward_dft, &source_point, 1, 1, true};
  std::complex<double> forward_results[3][6] = {};
  gpu::near2far_statistics forward_statistics[3] = {};
  std::uint64_t forward_warm_live = 0;
  try {
    for (int pass = 0; pass < 3; ++pass) {
      gpu::select_device(ordinals[pass]);
      gpu::reset_dispatch_statistics();
      bool used_mixed = false;
      gpu::detail::resident_near2far_cartesian_fp32(
          gpu::detail::near2far_cartesian_dimension::three,
          &forward_plan_owner, &forward_request, 1, &target, 1,
          &frequency, 1, &copy, 1, 1.0, 1.0,
          forward_results[pass], nullptr, false, true, false,
          &used_mixed);
      forward_statistics[pass] = gpu::get_near2far_statistics();
      require(used_mixed && gpu::selected_device() == ordinals[pass],
              "forward Near2Far migration did not execute mixed CUDA on "
              "the selected device");
      const std::uint64_t current_live =
          gpu::get_live_resident_device_buffers();
      if (pass == 0)
        forward_warm_live = current_live;
      else
        require(current_live == forward_warm_live,
                "forward Near2Far migration changed its bounded live "
                "allocation count");
    }
  }
  catch (...) {
    gpu::detail::destroy_resident_near2far_plan_for_owner(
        &forward_plan_owner);
    gpu::detail::destroy_resident_cache_for_owner(
        &forward_resident_owner);
    if (original_ordinal >= 0) gpu::select_device(original_ordinal);
    throw;
  }
  gpu::detail::destroy_resident_near2far_plan_for_owner(
      &forward_plan_owner);
  gpu::detail::destroy_resident_cache_for_owner(
      &forward_resident_owner);
  if (original_ordinal >= 0) gpu::select_device(original_ordinal);

  std::complex<double> forward_reference[6] = {};
  green3d(forward_reference, vec(target.x, target.y, target.z), frequency,
          1.0, 1.0,
          vec(source_point.x, source_point.y, source_point.z), Ey,
          std::complex<double>(forward_dft[0], forward_dft[1]));
  double forward_scale = 1.0e-300;
  double forward_error = 0.0;
  double cross_device_error = 0.0;
  for (std::size_t component = 0; component < 6; ++component) {
    forward_scale =
        std::max(forward_scale, std::abs(forward_reference[component]));
    for (int pass = 0; pass < 3; ++pass)
      forward_error = std::max(
          forward_error,
          std::abs(forward_results[pass][component] -
                   forward_reference[component]));
    cross_device_error = std::max(
        cross_device_error,
        std::abs(forward_results[1][component] -
                 forward_results[0][component]));
  }
  require(
      forward_error <= 5.0e-10 * forward_scale + 1.0e-14 &&
          cross_device_error <= 5.0e-10 * forward_scale + 1.0e-14 &&
          std::equal(forward_results[0], forward_results[0] + 6,
                     forward_results[2]) &&
          forward_statistics[1].cuda_transform_calls == 1 &&
          forward_statistics[2].cuda_transform_calls == 1 &&
          forward_statistics[1].cpu_transform_calls == 0 &&
          forward_statistics[2].cpu_transform_calls == 0 &&
          forward_statistics[1].cuda_mixed_precision_calls == 1 &&
          forward_statistics[2].cuda_mixed_precision_calls == 1 &&
          forward_statistics[1].cuda_descriptor_uploads == 1 &&
          forward_statistics[2].cuda_descriptor_uploads == 1 &&
          forward_warm_live > forward_live_before &&
          gpu::get_live_resident_device_buffers() == forward_live_before,
      "forward Near2Far retained plan did not migrate GPU0->GPU1->GPU0 "
      "with CPU agreement, exact return, and lifecycle recovery");
  if (am_master())
    std::cout << "PASS: forward and adjoint Near2Far retained plans migrate "
                 "GPU0->GPU1->GPU0 with CPU agreement and lifecycle "
                 "recovery\n";
#endif
}

void require_near2far_adjoint_equivalence() {
  const near2far_adjoint_result cpu =
      run_near2far_adjoint_case(gpu::backend_mode::cpu);
  const near2far_adjoint_result cuda =
      run_near2far_adjoint_case(gpu::backend_mode::cuda);
  require(cpu.indices == cuda.indices &&
              cpu.components == cuda.components &&
              cpu.source_points == cuda.source_points &&
              cpu.chunks == cuda.chunks &&
              cpu.amplitudes.size() == cuda.amplitudes.size(),
          "CPU/CUDA adjoint Near2Far source-profile topology differs");
  double reference_squared = 0.0;
  double error_squared = 0.0;
  double reference_maximum = 0.0;
  double error_maximum = 0.0;
  for (size_t index = 0; index < cpu.amplitudes.size(); ++index) {
    const double error =
        std::abs(cuda.amplitudes[index] - cpu.amplitudes[index]);
    reference_squared += std::norm(cpu.amplitudes[index]);
    error_squared += error * error;
    reference_maximum =
        std::max(reference_maximum, std::abs(cpu.amplitudes[index]));
    error_maximum = std::max(error_maximum, error);
  }
  const double nrmse = std::sqrt(
      error_squared / std::max(reference_squared, 1.0e-300));
  require(nrmse <= 8.0e-4 &&
              error_maximum <= 4.0e-3 * reference_maximum + 3.0e-8,
          "CUDA adjoint Near2Far source profiles differ from CPU: nrmse=" +
              std::to_string(nrmse) + " maximum_error=" +
              std::to_string(error_maximum));
  const std::uint64_t expected_terms =
      static_cast<std::uint64_t>(cuda.source_points) * 2 * 2;
  const std::uint64_t attempts =
      1 + cuda.statistics.cuda_adjoint_cancellation_retries;
  require(cpu.statistics.cpu_adjoint_calls == 1 &&
              cpu.statistics.cuda_adjoint_calls == 0 &&
              cpu.statistics.cpu_adjoint_terms == expected_terms &&
              cuda.statistics.cuda_adjoint_calls == 1 &&
              cuda.statistics.cpu_adjoint_calls == 0 &&
              cuda.statistics.cuda_adjoint_terms == expected_terms &&
              cuda.statistics.cuda_adjoint_submitted_chunks == cuda.chunks &&
              cuda.statistics.cuda_adjoint_source_points ==
                  cuda.source_points &&
              cuda.statistics.cuda_adjoint_far_points == 2 &&
              cuda.statistics.cuda_adjoint_frequencies == 2 &&
              cuda.statistics.cuda_adjoint_periodic_copies == 1 &&
              cuda.statistics.cuda_adjoint_fast_precision_calls +
                      cuda.statistics.cuda_adjoint_mixed_precision_calls ==
                  attempts &&
              cuda.statistics.cuda_adjoint_kernel_launches == attempts &&
              cuda.statistics.cuda_adjoint_descriptor_uploads == attempts &&
              cuda.statistics.cuda_adjoint_host_to_device_bytes > 0 &&
              cuda.statistics.cuda_adjoint_result_device_to_host_bytes >=
                  2 * cuda.source_points * 2 * sizeof(double) &&
              cuda.live_before == cuda.live_after,
          "adjoint Near2Far CUDA execution/transfer/lifetime counters are "
          "inconsistent");

  constexpr std::size_t forced_workspace_ceiling = 512;
  const near2far_adjoint_result tiled = run_near2far_adjoint_case(
      gpu::backend_mode::cuda, forced_workspace_ceiling);
  require(cpu.indices == tiled.indices &&
              cpu.components == tiled.components &&
              cpu.source_points == tiled.source_points &&
              cpu.chunks == tiled.chunks &&
              cpu.amplitudes.size() == tiled.amplitudes.size(),
          "forced-tile CUDA adjoint Near2Far topology differs from CPU");
  double tiled_reference_squared = 0.0;
  double tiled_error_squared = 0.0;
  double tiled_reference_maximum = 0.0;
  double tiled_error_maximum = 0.0;
  for (size_t index = 0; index < cpu.amplitudes.size(); ++index) {
    const double error =
        std::abs(tiled.amplitudes[index] - cpu.amplitudes[index]);
    tiled_reference_squared += std::norm(cpu.amplitudes[index]);
    tiled_error_squared += error * error;
    tiled_reference_maximum =
        std::max(tiled_reference_maximum, std::abs(cpu.amplitudes[index]));
    tiled_error_maximum = std::max(tiled_error_maximum, error);
  }
  const double tiled_nrmse = std::sqrt(
      tiled_error_squared / std::max(tiled_reference_squared, 1.0e-300));
  const std::uint64_t tiled_attempts =
      1 + tiled.statistics.cuda_adjoint_cancellation_retries;
  require(
      tiled_nrmse <= 8.0e-4 &&
          tiled_error_maximum <=
              4.0e-3 * tiled_reference_maximum + 3.0e-8 &&
          tiled.statistics.cuda_adjoint_maximum_workspace_bytes > 0 &&
          tiled.statistics.cuda_adjoint_maximum_workspace_bytes <=
              forced_workspace_ceiling &&
          tiled.statistics.cuda_adjoint_kernel_launches > tiled_attempts &&
          tiled.statistics.cuda_adjoint_descriptor_uploads ==
              tiled.statistics.cuda_adjoint_kernel_launches &&
          tiled.live_before == tiled.live_after,
      "forced-workspace CUDA adjoint Near2Far did not tile within its "
      "ceiling or preserve the CPU result: nrmse=" +
          std::to_string(tiled_nrmse) + " maximum_error=" +
          std::to_string(tiled_error_maximum) + " workspace=" +
          std::to_string(
              tiled.statistics.cuda_adjoint_maximum_workspace_bytes) +
          " launches=" +
          std::to_string(tiled.statistics.cuda_adjoint_kernel_launches));
}

void require_near2far_adjoint_cpu_reference() {
  const near2far_adjoint_result cpu =
      run_near2far_adjoint_case(gpu::backend_mode::cpu);
  require(!cpu.amplitudes.empty() && cpu.source_points > 0 &&
              cpu.chunks > 0 &&
              cpu.amplitudes.size() == 2 * cpu.source_points &&
              cpu.statistics.cpu_adjoint_calls == 1 &&
              cpu.statistics.cuda_adjoint_calls == 0 &&
              cpu.statistics.cpu_adjoint_terms ==
                  4 * static_cast<std::uint64_t>(cpu.source_points),
          "CPU adjoint Near2Far source-profile/counter reference failed");
  bool local_amplitudes_finite = true;
  for (const std::complex<double> amplitude : cpu.amplitudes)
    local_amplitudes_finite =
        local_amplitudes_finite && std::isfinite(real(amplitude)) &&
        std::isfinite(imag(amplitude));
  // The monitor is partitioned across ranks, so the local amplitude count
  // may differ.  Never place a collective require() inside that rank-local
  // loop: doing so mismatches allreduce counts and deadlocks at MPI teardown.
  require(local_amplitudes_finite,
          "CPU adjoint Near2Far produced a non-finite source amplitude");
}

struct near2far3d_snapshot_capture {
  std::vector<double> grid;
  std::vector<std::complex<double> > batch;
  std::vector<std::complex<double> > scalar;
  gpu::near2far_statistics grid_statistics;
  gpu::near2far_statistics batch_statistics;
  gpu::near2far_statistics scalar_statistics;
  int rank = 0;
  size_t dims[3] = {1, 1, 1};
  size_t output_points = 0;
  std::uint64_t live_before_grid = 0;
  std::uint64_t live_after_grid = 0;
  std::uint64_t live_after_batch = 0;
  std::uint64_t live_after_scalar = 0;
};

near2far3d_snapshot_capture capture_near2far3d_snapshot(
    dft_near2far &near, const volume &targets,
    const std::vector<vec> &grid_points, gpu::backend_mode mode,
    bool allow_device_migration_cleanup = false) {
  near2far3d_snapshot_capture result;
  if (gpu::active_backend() != mode) gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();
  const std::uint64_t grid_live_before =
      gpu::get_live_resident_device_buffers();
  result.live_before_grid = grid_live_before;
  size_t dims[4] = {1, 1, 1, 1};
  std::unique_ptr<double[]> raw_grid(near.get_farfields_array(
      targets, result.rank, dims, result.output_points, 5.0));
  require(raw_grid && result.output_points == grid_points.size(),
          "same-snapshot 3D near-to-far grid shape changed: rank=" +
              std::to_string(result.rank) + " dims=" +
              std::to_string(dims[0]) + "x" +
              std::to_string(dims[1]) + "x" +
              std::to_string(dims[2]) + " output-points=" +
              std::to_string(result.output_points) + " expected-points=" +
              std::to_string(grid_points.size()));
  for (int axis = 0; axis < 3; ++axis) result.dims[axis] = dims[axis];
  result.grid.assign(
      raw_grid.get(),
      raw_grid.get() + 12 * result.output_points * near.freq.size());
  result.grid_statistics = gpu::get_near2far_statistics();
  result.live_after_grid = gpu::get_live_resident_device_buffers();
  require(allow_device_migration_cleanup ||
              result.live_after_grid >= grid_live_before,
          "same-snapshot grid transform corrupted the live-buffer gauge");

  gpu::reset_dispatch_statistics();
  const std::uint64_t batch_live_before =
      gpu::get_live_resident_device_buffers();
  std::unique_ptr<std::complex<double>[]> raw_batch(
      near.farfields(grid_points.data(), grid_points.size()));
  result.batch.assign(
      raw_batch.get(),
      raw_batch.get() + 6 * near.freq.size() * grid_points.size());
  result.batch_statistics = gpu::get_near2far_statistics();
  result.live_after_batch = gpu::get_live_resident_device_buffers();
  require(result.live_after_batch == batch_live_before,
          "same-snapshot batch transform leaked a call-scoped device buffer");

  gpu::reset_dispatch_statistics();
  const std::uint64_t scalar_live_before =
      gpu::get_live_resident_device_buffers();
  std::unique_ptr<std::complex<double>[]> raw_scalar(
      near.farfield(grid_points[0]));
  result.scalar.assign(
      raw_scalar.get(), raw_scalar.get() + 6 * near.freq.size());
  result.scalar_statistics = gpu::get_near2far_statistics();
  result.live_after_scalar = gpu::get_live_resident_device_buffers();
  require(result.live_after_scalar == scalar_live_before,
          "same-snapshot scalar transform leaked a call-scoped device buffer");
  return result;
}

void require_near2far_complex_agreement(
    const std::vector<std::complex<double> > &reference,
    const std::vector<std::complex<double> > &candidate,
    const std::string &label) {
  require(reference.size() == candidate.size(), label + " shape differs");
  double reference_squared = 0.0;
  double error_squared = 0.0;
  double reference_maximum = 0.0;
  double error_maximum = 0.0;
  for (size_t index = 0; index < reference.size(); ++index) {
    const double error = std::abs(candidate[index] - reference[index]);
    reference_squared += std::norm(reference[index]);
    error_squared += error * error;
    reference_maximum =
        std::max(reference_maximum, std::abs(reference[index]));
    error_maximum = std::max(error_maximum, error);
  }
  const double nrmse = std::sqrt(
      error_squared / std::max(reference_squared, 1.0e-300));
  require(nrmse <= 5.0e-4 &&
              error_maximum <= 3.0e-3 * reference_maximum + 3.0e-8,
          label + " differs: nrmse=" + std::to_string(nrmse) +
              " maximum_error=" + std::to_string(error_maximum));
}

void require_near2far_cuda_transform_statistics(
    const gpu::near2far_statistics &statistics, size_t target_count,
    size_t frequency_count, size_t chunk_count, size_t source_points,
    size_t periodic_copies, bool expect_descriptor_upload,
    bool require_dirty_dft_avoidance) {
  const std::uint64_t expected_terms =
      static_cast<std::uint64_t>(target_count) * frequency_count *
      source_points * periodic_copies;
  const std::uint64_t attempts =
      1 + statistics.cuda_cancellation_retries;
  // Different MPI partitions can make different conservative precision
  // decisions: one rank may start directly in mixed CUDA while a peer's fast
  // result triggers one collective mixed retry. The rank that was already
  // mixed participates in both result collectives without launching a second
  // local transform. Keep local kernel/copy evidence tied to local attempts
  // and collective traffic tied to the independently observed MPI attempts.
  const std::uint64_t collective_attempts =
      statistics.mpi_allreduce_calls;
  const std::uint64_t expected_result_bytes =
      12 * target_count * frequency_count * sizeof(double);
  const bool descriptor_uploads_are_consistent =
      statistics.cuda_cancellation_retries
          ? statistics.cuda_descriptor_uploads >= 1 &&
                statistics.cuda_descriptor_uploads <= 2
          : (statistics.cuda_descriptor_uploads <= 1 &&
             (!expect_descriptor_upload ||
              statistics.cuda_descriptor_uploads == 1));
  require(statistics.cuda_transform_calls == 1 &&
              statistics.cpu_transform_calls == 0 &&
              statistics.cuda_terms == expected_terms &&
              statistics.cuda_submitted_chunks == chunk_count &&
              statistics.cuda_source_points == source_points &&
              statistics.cuda_output_points == target_count &&
              statistics.cuda_frequencies == frequency_count &&
              statistics.cuda_periodic_copies == periodic_copies &&
              statistics.cuda_fast_precision_calls +
                      statistics.cuda_mixed_precision_calls ==
                  attempts &&
              statistics.cuda_cancellation_retries <= 1 &&
              (!statistics.cuda_cancellation_retries ||
               (statistics.cuda_fast_precision_calls == 1 &&
                statistics.cuda_mixed_precision_calls == 1)) &&
              statistics.cuda_target_tiles == attempts &&
              statistics.cuda_frequency_tiles == attempts &&
              statistics.cuda_operation_tiles == attempts &&
              statistics.cuda_maximum_workspace_bytes > 0 &&
              statistics.cuda_maximum_workspace_bytes <=
                  64u * 1024u * 1024u &&
              descriptor_uploads_are_consistent &&
              statistics.cuda_kernel_launches ==
                  2 * statistics.cuda_operation_tiles &&
              statistics.cuda_result_device_to_host_bytes ==
                  attempts * expected_result_bytes &&
              statistics.cuda_condition_device_to_host_bytes ==
                  statistics.cuda_fast_precision_calls *
                      target_count * frequency_count * sizeof(double) &&
              collective_attempts >= attempts &&
              collective_attempts <= attempts + 1 &&
              collective_attempts >= 1 && collective_attempts <= 2 &&
              statistics.mpi_allreduce_bytes ==
                  collective_attempts * 13 * target_count * frequency_count *
                      sizeof(double) &&
              (!require_dirty_dft_avoidance ||
               statistics.dft_device_to_host_bytes_avoided > 0),
          "same-snapshot CUDA near-to-far lacks exact execution evidence: "
          "calls=" + std::to_string(statistics.cuda_transform_calls) +
          " fast=" +
              std::to_string(statistics.cuda_fast_precision_calls) +
          " mixed=" +
              std::to_string(statistics.cuda_mixed_precision_calls) +
          " retries=" +
              std::to_string(statistics.cuda_cancellation_retries) +
          " target-tiles=" +
              std::to_string(statistics.cuda_target_tiles) +
          " frequency-tiles=" +
              std::to_string(statistics.cuda_frequency_tiles) +
          " operation-tiles=" +
              std::to_string(statistics.cuda_operation_tiles) +
          " workspace=" +
              std::to_string(statistics.cuda_maximum_workspace_bytes) +
          " uploads=" +
              std::to_string(statistics.cuda_descriptor_uploads) +
          " launches=" +
              std::to_string(statistics.cuda_kernel_launches) +
          " result-bytes=" +
              std::to_string(
                  statistics.cuda_result_device_to_host_bytes) +
          " condition-bytes=" +
              std::to_string(
                  statistics.cuda_condition_device_to_host_bytes) +
          " mpi-calls=" +
              std::to_string(statistics.mpi_allreduce_calls) +
          " mpi-bytes=" +
              std::to_string(statistics.mpi_allreduce_bytes));
}

void require_near2far_public_dimension_mismatch_rejected() {
  const direction periodic_d[2] = {NO_DIRECTION, NO_DIRECTION};
  const int periodic_n[2] = {0, 0};
  const double periodic_k[2] = {0.0, 0.0};
  const double period[2] = {0.0, 0.0};
  const double frequency = 0.31;
  dft_near2far monitor2d(
      nullptr, &frequency, 1, 1.0, 1.0,
      volume(vec(-0.4, -0.3), vec(0.4, 0.3)), periodic_d, periodic_n,
      periodic_k, period);
  dft_near2far monitor3d(
      nullptr, &frequency, 1, 1.0, 1.0,
      volume(vec(-0.4, -0.3, -0.2), vec(0.4, 0.3, 0.2)), periodic_d,
      periodic_n, periodic_k, period);

  const auto require_batch_rejection = [](dft_near2far &monitor,
                                          const vec &target,
                                          const char *label) {
    bool rejected = false;
    try {
      std::unique_ptr<std::complex<double>[]> ignored(
          monitor.farfields(&target, 1));
    }
    catch (const std::invalid_argument &error) {
      rejected = std::string(error.what()).find("incompatible") !=
                 std::string::npos;
    }
    require(rejected, std::string(label) +
                          " was not rejected before Green evaluation");
  };
  require_batch_rejection(monitor3d, vec(1.7, -0.2),
                          "3D monitor with 2D target");
  require_batch_rejection(monitor2d, vec(1.7, -0.2, 0.4),
                          "2D monitor with 3D target");

  int rank = 0;
  size_t dims[4] = {1, 1, 1, 1};
  size_t point_count = 0;
  bool grid_rejected = false;
  try {
    std::unique_ptr<double[]> ignored(monitor2d.get_farfields_array(
        volume(vec(1.2, -0.1, -0.2), vec(1.4, 0.1, 0.2)), rank, dims,
        point_count, 4.0));
  }
  catch (const std::invalid_argument &error) {
    grid_rejected = std::string(error.what()).find("compatible") !=
                    std::string::npos;
  }
  require(grid_rejected,
          "2D monitor with 3D output grid was not rejected atomically");
}

void require_near2far_cylindrical_inputs_rejected_without_device() {
#if MEEP_SINGLE && MEEP_HAVE_CUDA
  const gpu::detail::near2far_point_fp64 source = {0.4, 0.0, 0.1};
  const gpu::detail::near2far_point_fp64 target = {2.0, 0.0, -0.2};
  const gpu::detail::near2far_periodic_copy_fp64 periodic = {
      {0.0, 0.0, 0.0}, 1.0, 0.0};
  float dft[2] = {0.7f, -0.2f};
  double frequency = 0.31;
  int resident_owner = 0;
  int plan_owner = 0;
  const gpu::detail::near2far_request_fp32 request = {
      &resident_owner, dft, &source, 1, 0, true};
  std::complex<double> output[6] = {};
  const auto require_rejection = [](const auto &operation,
                                    const std::string &needle,
                                    const std::string &label) {
    bool rejected = false;
    try {
      operation();
    }
    catch (const std::invalid_argument &error) {
      rejected = std::string(error.what()).find(needle) !=
                 std::string::npos;
    }
    require(rejected, label + " was not rejected before touching CUDA");
  };
  require_rejection(
      [&]() {
        const double zero_frequency = 0.0;
        gpu::detail::resident_near2far_cartesian_fp32(
            gpu::detail::near2far_cartesian_dimension::cylindrical,
            &plan_owner, &request, 1, &target, 1, &zero_frequency, 1,
            &periodic, 1, 1.0, 1.0, output, nullptr, false, false, false,
            nullptr, 0.0, 1.0e-6);
      },
      "frequencies", "zero cylindrical frequency");
  require_rejection(
      [&]() {
        const gpu::detail::near2far_point_fp64 invalid_source =
            {source.x, 0.1, source.z};
        const gpu::detail::near2far_request_fp32 invalid_request = {
            &resident_owner, dft, &invalid_source, 1, 0, true};
        gpu::detail::resident_near2far_cartesian_fp32(
            gpu::detail::near2far_cartesian_dimension::cylindrical,
            &plan_owner, &invalid_request, 1, &target, 1, &frequency, 1,
            &periodic, 1, 1.0, 1.0, output, nullptr, false, false, false,
            nullptr, 0.0, 1.0e-6);
      },
      "source coordinates", "cylindrical source with a Cartesian-y coordinate");
  require_rejection(
      [&]() {
        gpu::detail::resident_near2far_cartesian_fp32(
            gpu::detail::near2far_cartesian_dimension::three,
            &plan_owner, &request, 1, &target, 1, &frequency, 1,
            &periodic, 1, 1.0, 1.0, output, nullptr, false, false, false,
            nullptr, 1.0, 1.0e-6);
      },
      "mode/tolerance", "cylindrical mode on a Cartesian transform");
  if (am_master())
    std::cout << "PASS: invalid cylindrical Near2Far zero frequency, geometry, "
                 "and Cartesian mode fail before CUDA\n";
#endif
}

void require_near2far2d_hankel_bound_sweep() {
#if MEEP_SINGLE && MEEP_HAVE_CUDA
  gpu::set_backend(gpu::backend_mode::cuda);
  const gpu::detail::near2far_point_fp64 source = {0.0, 0.0, 0.0};
  const gpu::detail::near2far_periodic_copy_fp64 periodic = {
      {0.0, 0.0, 0.0}, 1.0, 0.0};
  const double frequency = 1.0 / (2.0 * pi);
  constexpr double angle = 0.37;
  float dft[2] = {0.73f, -0.19f};
  int resident_owner = 0;
  int plan_owner = 0;
  const double kr_values[] = {
      0.2, 0.5, 1.0, 3.3842417671495935, 5.135622301840683,
      static_cast<double>(std::nextafter(8.0f, 0.0f)), 8.0,
      static_cast<double>(std::nextafter(8.0f, 16.0f)), 16.0, 64.0};
  double maximum_bound_ratio = 0.0;
  double maximum_observed_ratio = 0.0;
  std::size_t production_mixed_retries = 0;
  try {
    for (int source_case = 0; source_case < 6; ++source_case) {
      const int axis_index = source_case % 3;
      const bool electric = source_case < 3;
      const gpu::detail::near2far_request_fp32 request = {
          &resident_owner, dft, &source, 1, axis_index, electric};
      for (double desired_kr : kr_values) {
        const double k = 2.0 * pi * frequency;
        const double distance = desired_kr / k;
        const gpu::detail::near2far_point_fp64 target = {
            distance * std::cos(angle), distance * std::sin(angle), 0.0};
        std::complex<double> observed[6] = {};
        double published_bound[1] = {
            std::numeric_limits<double>::quiet_NaN()};
        bool used_mixed = true;
        gpu::reset_dispatch_statistics();
        gpu::detail::resident_near2far_cartesian_fp32(
            gpu::detail::near2far_cartesian_dimension::two,
            &plan_owner, &request, 1, &target, 1, &frequency, 1,
            &periodic, 1, 1.0, 1.0, observed, published_bound,
            true, false, false, &used_mixed);
        std::complex<double> reference[6] = {};
        green2d(reference, vec(target.x, target.y), frequency, 1.0, 1.0,
                vec(0.0, 0.0),
                direction_component(electric ? Ex : Hx,
                                    static_cast<direction>(axis_index)),
                std::complex<double>(dft[0], dft[1]));
        double work_maximum = 0.0;
        double scalar_error_maximum = 0.0;
        for (int component = 0; component < 6; ++component) {
          work_maximum = std::max(
              work_maximum,
              std::max(std::abs(reference[component].real()),
                       std::abs(reference[component].imag())));
          scalar_error_maximum = std::max(
              scalar_error_maximum,
              std::max(std::abs(observed[component].real() -
                                reference[component].real()),
                       std::abs(observed[component].imag() -
                                reference[component].imag())));
        }
        const gpu::near2far_statistics statistics =
            gpu::get_near2far_statistics();
        const double accepted_error =
            3.0e-3 * work_maximum + 3.0e-8;
        require(!used_mixed && std::isfinite(published_bound[0]) &&
                    published_bound[0] >= 0.0 &&
                    scalar_error_maximum <=
                        published_bound[0] * (1.0 + 1.0e-12) + 1.0e-30 &&
                    statistics.cuda_transform_calls == 1 &&
                    statistics.cuda_fast_precision_calls == 1 &&
                    statistics.cuda_mixed_precision_calls == 0,
                "2D Hankel fast error escaped its published bound at kr=" +
                    canonical_decimal(desired_kr) + " source-case=" +
                    std::to_string(source_case) + " actual=" +
                    canonical_decimal(scalar_error_maximum) + " bound=" +
                    canonical_decimal(published_bound[0]) + " accepted=" +
                    canonical_decimal(accepted_error));
        if (published_bound[0] > accepted_error) {
          // The deferred call above intentionally exposes the FP32 result and
          // its conservative bound.  A production call must not publish that
          // result when the bound exceeds the public error budget, even if the
          // sampled error happens to be small on this GPU/libm combination.
          std::complex<double> production_observed[6] = {};
          bool production_used_mixed = false;
          gpu::reset_dispatch_statistics();
          gpu::detail::resident_near2far_cartesian_fp32(
              gpu::detail::near2far_cartesian_dimension::two,
              &plan_owner, &request, 1, &target, 1, &frequency, 1,
              &periodic, 1, 1.0, 1.0, production_observed, nullptr,
              false, false, false, &production_used_mixed);
          double production_error_maximum = 0.0;
          for (int component = 0; component < 6; ++component)
            production_error_maximum = std::max(
                production_error_maximum,
                std::max(std::abs(production_observed[component].real() -
                                  reference[component].real()),
                         std::abs(production_observed[component].imag() -
                                  reference[component].imag())));
          const gpu::near2far_statistics production_statistics =
              gpu::get_near2far_statistics();
          require(production_used_mixed &&
                      production_error_maximum <=
                          5.0e-8 * work_maximum + 1.0e-30 &&
                      production_statistics.cuda_transform_calls == 1 &&
                      production_statistics.cuda_fast_precision_calls == 1 &&
                      production_statistics.cuda_mixed_precision_calls == 1 &&
                      production_statistics.cuda_cancellation_retries == 1,
                  "2D Hankel production selector published an FP32 result "
                  "whose bound exceeds the error budget at kr=" +
                      canonical_decimal(desired_kr) + " source-case=" +
                      std::to_string(source_case) + " used-mixed=" +
                      std::to_string(production_used_mixed) + " error=" +
                      canonical_decimal(production_error_maximum) +
                      " limit=" +
                      canonical_decimal(5.0e-8 * work_maximum + 1.0e-30) +
                      " calls=" +
                      std::to_string(
                          production_statistics.cuda_transform_calls) +
                      " fast=" +
                      std::to_string(
                          production_statistics.cuda_fast_precision_calls) +
                      " mixed=" +
                      std::to_string(
                          production_statistics.cuda_mixed_precision_calls) +
                      " retries=" +
                      std::to_string(
                          production_statistics.cuda_cancellation_retries));
          ++production_mixed_retries;
        }
        maximum_bound_ratio = std::max(
            maximum_bound_ratio,
            published_bound[0] / std::max(work_maximum, 1.0e-300));
        maximum_observed_ratio = std::max(
            maximum_observed_ratio,
            scalar_error_maximum / std::max(work_maximum, 1.0e-300));
      }
    }

    const auto require_direct_mixed = [
        &source, &periodic, &resident_owner, &plan_owner, &dft](
        double desired_kr, double eps, double mu,
        const std::string &label) {
      const double local_frequency = 0.31;
      const double k = 2.0 * pi * local_frequency * std::sqrt(eps * mu);
      const double distance = desired_kr / k;
      const gpu::detail::near2far_point_fp64 target = {
          distance * std::cos(angle), distance * std::sin(angle), 0.0};
      const gpu::detail::near2far_request_fp32 request = {
          &resident_owner, dft, &source, 1, 2, true};
      std::complex<double> observed[6] = {};
      bool used_mixed = false;
      gpu::reset_dispatch_statistics();
      gpu::detail::resident_near2far_cartesian_fp32(
          gpu::detail::near2far_cartesian_dimension::two,
          &plan_owner, &request, 1, &target, 1, &local_frequency, 1,
          &periodic, 1, eps, mu, observed, nullptr, false, false, false,
          &used_mixed);
      std::complex<double> reference[6] = {};
      green2d(reference, vec(target.x, target.y), local_frequency, eps, mu,
              vec(0.0, 0.0), Ez,
              std::complex<double>(dft[0], dft[1]));
      double reference_maximum = 0.0;
      double error_maximum = 0.0;
      for (int component = 0; component < 6; ++component) {
        reference_maximum =
            std::max(reference_maximum, std::abs(reference[component]));
        error_maximum = std::max(
            error_maximum,
            std::abs(observed[component] - reference[component]));
      }
      const gpu::near2far_statistics statistics =
          gpu::get_near2far_statistics();
      require(used_mixed && reference_maximum > 0.0 &&
                  error_maximum <= 5.0e-8 * reference_maximum + 1.0e-30 &&
                  statistics.cuda_transform_calls == 1 &&
                  statistics.cuda_fast_precision_calls == 0 &&
                  statistics.cuda_mixed_precision_calls == 1,
              label + " did not select accurate mixed CUDA: error=" +
                  canonical_decimal(error_maximum) + " reference=" +
                  canonical_decimal(reference_maximum));
    };
    require_direct_mixed(1.0e-6, 1.0, 1.0,
                         "kr-near-zero 2D Hankel selector");
    require_direct_mixed(
        static_cast<double>(std::nextafter(2048.0f, 0.0f)), 1.0, 1.0,
        "kr-below-2048 2D Hankel selector boundary");
    require_direct_mixed(2048.0, 1.0, 1.0,
                         "kr-at-2048 2D Hankel selector boundary");
    require_direct_mixed(
        static_cast<double>(std::nextafter(2048.0f, 4096.0f)), 1.0, 1.0,
        "kr-above-2048 2D Hankel selector boundary");
    require_direct_mixed(2.0, 1.0e-20, 1.0e20,
                         "extreme-impedance 2D Hankel selector");
    require_direct_mixed(2.0, 1.0e-30, 1.0e-30,
                         "underflow-index 2D Hankel selector");

    // Two nearly opposite Ez equivalent sources create a result whose fast
    // error is dominated by the pre-cancellation Hankel terms. The public
    // backend must discard that first result and retry on mixed CUDA.
    float positive[2] = {1.0f, 0.0f};
    float negative[2] = {-std::nextafter(1.0f, 0.0f), 0.0f};
    const gpu::detail::near2far_request_fp32 cancellation_requests[2] = {
        {&resident_owner, positive, &source, 1, 2, true},
        {&resident_owner, negative, &source, 1, 2, true}};
    const gpu::detail::near2far_point_fp64 cancellation_target =
        {2.0 * std::cos(angle), 2.0 * std::sin(angle), 0.0};
    std::complex<double> cancellation_observed[6] = {};
    gpu::reset_dispatch_statistics();
    gpu::detail::resident_near2far_cartesian_fp32(
        gpu::detail::near2far_cartesian_dimension::two,
        &plan_owner, cancellation_requests, 2, &cancellation_target, 1,
        &frequency, 1, &periodic, 1, 1.0, 1.0,
        cancellation_observed);
    const gpu::near2far_statistics cancellation_statistics =
        gpu::get_near2far_statistics();
    std::complex<double> cancellation_reference[6] = {};
    green2d(cancellation_reference,
            vec(cancellation_target.x, cancellation_target.y), frequency,
            1.0, 1.0, vec(0.0, 0.0), Ez,
            std::complex<double>(
                static_cast<double>(positive[0]) + negative[0], 0.0));
    double cancellation_reference_maximum = 0.0;
    double cancellation_error_maximum = 0.0;
    for (int component = 0; component < 6; ++component) {
      cancellation_reference_maximum = std::max(
          cancellation_reference_maximum,
          std::abs(cancellation_reference[component]));
      cancellation_error_maximum = std::max(
          cancellation_error_maximum,
          std::abs(cancellation_observed[component] -
                   cancellation_reference[component]));
    }
    require(cancellation_reference_maximum > 0.0 &&
                cancellation_error_maximum <=
                    5.0e-8 * cancellation_reference_maximum + 1.0e-30 &&
                cancellation_statistics.cuda_transform_calls == 1 &&
                cancellation_statistics.cuda_fast_precision_calls == 1 &&
                cancellation_statistics.cuda_mixed_precision_calls == 1 &&
                cancellation_statistics.cuda_cancellation_retries == 1,
            "2D Hankel cancellation did not retry once on mixed CUDA");
  }
  catch (...) {
    gpu::detail::destroy_resident_near2far_plan_for_owner(&plan_owner);
    gpu::detail::destroy_resident_cache_for_owner(&resident_owner);
    throw;
  }
  gpu::detail::destroy_resident_near2far_plan_for_owner(&plan_owner);
  gpu::detail::destroy_resident_cache_for_owner(&resident_owner);
  if (am_master())
    std::cout << "PASS: 2D Hankel log/boundary/H2 sweep stays within its "
                 "published bound and production retries every unsafe fast "
                 "result; max-actual="
              << maximum_observed_ratio << " max-bound="
              << maximum_bound_ratio << " mixed-retries="
              << production_mixed_retries << '\n';
#endif
}

void require_near2far_cartesian_dimension_cache_key() {
#if MEEP_SINGLE && MEEP_HAVE_CUDA
  gpu::set_backend(gpu::backend_mode::cuda);
  const gpu::detail::near2far_point_fp64 source = {0.1, -0.2, 0.0};
  const gpu::detail::near2far_point_fp64 target = {2.1, 0.7, 0.0};
  const gpu::detail::near2far_point_fp64 cylindrical_source =
      {0.1, 0.0, -0.2};
  const gpu::detail::near2far_point_fp64 cylindrical_target =
      {2.1, 0.0, 0.7};
  const gpu::detail::near2far_periodic_copy_fp64 periodic = {
      {0.0, 0.0, 0.0}, 1.0, 0.0};
  const double frequency = 0.31;
  float dft[2] = {0.63f, -0.21f};
  int resident_owner = 0;
  int plan_owner = 0;
  const gpu::detail::near2far_request_fp32 request = {
      &resident_owner, dft, &source, 1, 1, true};
  const gpu::detail::near2far_request_fp32 cylindrical_request = {
      &resident_owner, dft, &cylindrical_source, 1, 1, true};
  std::complex<double> observed2d[6] = {};
  std::complex<double> observed3d[6] = {};
  std::complex<double> repeated3d[6] = {};
  std::complex<double> observed_cylindrical[6] = {};
  std::complex<double> changed_mode_cylindrical[6] = {};
  gpu::near2far_statistics first = {};
  gpu::near2far_statistics switched = {};
  gpu::near2far_statistics reused = {};
  gpu::near2far_statistics switched_cylindrical = {};
  gpu::near2far_statistics reused_cylindrical = {};
  try {
    gpu::reset_dispatch_statistics();
    gpu::detail::resident_near2far_cartesian_fp32(
        gpu::detail::near2far_cartesian_dimension::two,
        &plan_owner, &request, 1, &target, 1, &frequency, 1, &periodic, 1,
        1.0, 1.0, observed2d, nullptr, false, true, false, nullptr);
    first = gpu::get_near2far_statistics();
    gpu::reset_dispatch_statistics();
    gpu::detail::resident_near2far_cartesian_fp32(
        gpu::detail::near2far_cartesian_dimension::three,
        &plan_owner, &request, 1, &target, 1, &frequency, 1, &periodic, 1,
        1.0, 1.0, observed3d, nullptr, false, true, false, nullptr);
    switched = gpu::get_near2far_statistics();
    gpu::reset_dispatch_statistics();
    gpu::detail::resident_near2far_cartesian_fp32(
        gpu::detail::near2far_cartesian_dimension::three,
        &plan_owner, &request, 1, &target, 1, &frequency, 1, &periodic, 1,
        1.0, 1.0, repeated3d, nullptr, false, true, false, nullptr);
    reused = gpu::get_near2far_statistics();
    gpu::reset_dispatch_statistics();
    gpu::detail::resident_near2far_cartesian_fp32(
        gpu::detail::near2far_cartesian_dimension::cylindrical,
        &plan_owner, &cylindrical_request, 1, &cylindrical_target, 1,
        &frequency, 1, &periodic, 1, 1.0, 1.0,
        observed_cylindrical, nullptr, false, true, false, nullptr,
        2.0, 1.0e-6);
    switched_cylindrical = gpu::get_near2far_statistics();
    gpu::reset_dispatch_statistics();
    gpu::detail::resident_near2far_cartesian_fp32(
        gpu::detail::near2far_cartesian_dimension::cylindrical,
        &plan_owner, &cylindrical_request, 1, &cylindrical_target, 1,
        &frequency, 1, &periodic, 1, 1.0, 1.0,
        changed_mode_cylindrical, nullptr, false, true, false, nullptr,
        3.0, 1.0e-6);
    reused_cylindrical = gpu::get_near2far_statistics();
  }
  catch (...) {
    gpu::detail::destroy_resident_near2far_plan_for_owner(&plan_owner);
    gpu::detail::destroy_resident_cache_for_owner(&resident_owner);
    throw;
  }
  std::complex<double> reference2d[6] = {};
  std::complex<double> reference3d[6] = {};
  std::complex<double> reference_cylindrical_m2[6] = {};
  std::complex<double> reference_cylindrical_m3[6] = {};
  green2d(reference2d, vec(target.x, target.y), frequency, 1.0, 1.0,
          vec(source.x, source.y), Ey,
          std::complex<double>(dft[0], dft[1]));
  green3d(reference3d, vec(target.x, target.y, target.z), frequency, 1.0,
          1.0, vec(source.x, source.y, source.z), Ey,
          std::complex<double>(dft[0], dft[1]));
  greencyl(reference_cylindrical_m2,
           veccyl(cylindrical_target.x, cylindrical_target.z), frequency,
           1.0, 1.0,
           veccyl(cylindrical_source.x, cylindrical_source.z), Ep,
           std::complex<double>(dft[0], dft[1]), 2.0, 1.0e-6);
  greencyl(reference_cylindrical_m3,
           veccyl(cylindrical_target.x, cylindrical_target.z), frequency,
           1.0, 1.0,
           veccyl(cylindrical_source.x, cylindrical_source.z), Ep,
           std::complex<double>(dft[0], dft[1]), 3.0, 1.0e-6);
  double error2d = 0.0;
  double error3d = 0.0;
  double cylindrical_m2_error = 0.0;
  double cylindrical_m3_error = 0.0;
  double difference_between_dimensions = 0.0;
  double difference_between_modes = 0.0;
  for (int component = 0; component < 6; ++component) {
    error2d = std::max(error2d,
                       std::abs(observed2d[component] -
                                reference2d[component]));
    error3d = std::max(error3d,
                       std::abs(observed3d[component] -
                                reference3d[component]));
    difference_between_dimensions = std::max(
        difference_between_dimensions,
        std::abs(observed2d[component] - observed3d[component]));
    cylindrical_m2_error = std::max(
        cylindrical_m2_error,
        std::abs(observed_cylindrical[component] -
                 reference_cylindrical_m2[component]));
    cylindrical_m3_error = std::max(
        cylindrical_m3_error,
        std::abs(changed_mode_cylindrical[component] -
                 reference_cylindrical_m3[component]));
    difference_between_modes = std::max(
        difference_between_modes,
        std::abs(observed_cylindrical[component] -
                 changed_mode_cylindrical[component]));
  }
  require(error2d <= 1.0e-10 && error3d <= 1.0e-10 &&
              cylindrical_m2_error <= 1.0e-10 &&
              cylindrical_m3_error <= 1.0e-10 &&
              difference_between_dimensions > 1.0e-6 &&
              difference_between_modes > 1.0e-6 &&
              std::equal(observed3d, observed3d + 6, repeated3d) &&
              first.cuda_descriptor_uploads == 1 &&
              switched.cuda_descriptor_uploads == 1 &&
              reused.cuda_descriptor_uploads == 0 &&
              switched_cylindrical.cuda_descriptor_uploads == 1 &&
              reused_cylindrical.cuda_descriptor_uploads == 0 &&
              first.cuda_mixed_precision_calls == 1 &&
              switched.cuda_mixed_precision_calls == 1 &&
              reused.cuda_mixed_precision_calls == 1 &&
              switched_cylindrical.cuda_mixed_precision_calls == 1 &&
              reused_cylindrical.cuda_mixed_precision_calls == 1,
          "retained Near2Far plan did not invalidate exactly once when its "
          "dimension changed or reuse descriptors when cylindrical m "
          "changed");
  gpu::detail::destroy_resident_near2far_plan_for_owner(&plan_owner);
  gpu::detail::destroy_resident_cache_for_owner(&resident_owner);
#endif
}

void require_near2far_cylindrical_bound_sweep() {
#if MEEP_SINGLE && MEEP_HAVE_CUDA
  gpu::set_backend(gpu::backend_mode::cuda);
  constexpr double eps = 1.7;
  constexpr double mu = 0.9;
  constexpr double reference_tolerance = 1.0e-10;
  const double tolerances[] = {1.0e-6, 1.0e-3};
  const double frequency = -0.31;
  const gpu::detail::near2far_point_fp64 source = {0.43, 0.0, -0.12};
  const gpu::detail::near2far_point_fp64 targets[] = {
      {3.8, 1.25, 0.2}, {-5.1, 3.15, -1.0},
      {-8.2, -5.7, 2.0}};
  const gpu::detail::near2far_periodic_copy_fp64 periodic = {
      {0.0, 0.0, 0.0}, 1.0, 0.0};
  const component source_components[] = {Er, Ep, Ez, Hr, Hp, Hz};
  const double modes[] = {-2.0, 0.0, 3.0};
  float dft[2] = {0.73f, -0.19f};
  int resident_owner = 0;
  int plan_owner = 0;
  double maximum_actual_to_bound = 0.0;
  std::size_t production_mixed_retries = 0;
  try {
    for (double tolerance : tolerances)
      for (double mode : modes)
        for (int source_case = 0; source_case < 6; ++source_case) {
        const gpu::detail::near2far_request_fp32 request = {
            &resident_owner, dft, &source, 1, source_case % 3,
            source_case < 3};
        std::complex<double> observed[18] = {};
        double published_bounds[3] = {
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()};
        bool used_mixed = true;
        gpu::reset_dispatch_statistics();
        gpu::detail::resident_near2far_cartesian_fp32(
            gpu::detail::near2far_cartesian_dimension::cylindrical,
            &plan_owner, &request, 1, targets, 3, &frequency, 1,
            &periodic, 1, eps, mu, observed, published_bounds,
            true, false, false, &used_mixed, mode, tolerance);
        const gpu::near2far_statistics statistics =
            gpu::get_near2far_statistics();
        require(!used_mixed &&
                    statistics.cuda_transform_calls == 1 &&
                    statistics.cuda_fast_precision_calls == 1 &&
                    statistics.cuda_mixed_precision_calls == 0,
                "ordinary cylindrical proof sweep did not execute fast "
                "CUDA for m=" + std::to_string(mode) + " source=" +
                    std::to_string(source_case));
        std::complex<double> references[18] = {};
        bool production_retry_required = false;
        double production_reference_maximum = 0.0;
        for (size_t target_index = 0; target_index < 3; ++target_index) {
          std::complex<double> *reference = references + 6 * target_index;
          greencyl(reference,
                   vec(targets[target_index].x,
                       targets[target_index].y,
                       targets[target_index].z),
                   frequency, eps, mu, veccyl(source.x, source.z),
                   source_components[source_case],
                   std::complex<double>(dft[0], dft[1]), mode,
                   reference_tolerance);
          double maximum_reference = 0.0;
          double maximum_observed_scalar = 0.0;
          double maximum_error = 0.0;
          for (int field_component = 0; field_component < 6;
               ++field_component) {
            maximum_reference = std::max(
                maximum_reference,
                std::abs(reference[field_component]));
            maximum_observed_scalar = std::max(
                maximum_observed_scalar,
                std::max(
                    std::abs(observed[6 * target_index + field_component]
                                 .real()),
                    std::abs(observed[6 * target_index + field_component]
                                 .imag())));
            maximum_error = std::max(
                maximum_error,
                std::abs(observed[6 * target_index + field_component] -
                         reference[field_component]));
          }
          const double published_bound = published_bounds[target_index];
          const double accepted_error =
              3.0e-3 * maximum_observed_scalar + 3.0e-8;
          production_retry_required =
              production_retry_required || published_bound > accepted_error;
          production_reference_maximum =
              std::max(production_reference_maximum, maximum_reference);
          require(std::isfinite(published_bound) &&
                      published_bound >= 0.0 &&
                      maximum_error <=
                          published_bound * (1.0 + 1.0e-12) + 1.0e-15 &&
                      maximum_error <=
                          3.0e-3 * maximum_reference + 3.0e-8,
                  "cylindrical fast CUDA error exceeds its published bound "
                  "for m=" + std::to_string(mode) + " source=" +
                      std::to_string(source_case) + " tol=" +
                      canonical_decimal(tolerance) + " target=" +
                      std::to_string(target_index) + " error=" +
                      canonical_decimal(maximum_error) + " bound=" +
                      canonical_decimal(published_bound));
          if (published_bound > 0.0)
            maximum_actual_to_bound = std::max(
                maximum_actual_to_bound,
                maximum_error / published_bound);
        }
        if (production_retry_required) {
          std::complex<double> production_observed[18] = {};
          bool production_used_mixed = false;
          gpu::reset_dispatch_statistics();
          gpu::detail::resident_near2far_cartesian_fp32(
              gpu::detail::near2far_cartesian_dimension::cylindrical,
              &plan_owner, &request, 1, targets, 3, &frequency, 1,
              &periodic, 1, eps, mu, production_observed, nullptr,
              false, false, false, &production_used_mixed, mode, tolerance);
          double production_error_maximum = 0.0;
          for (std::size_t value = 0; value < 18; ++value)
            production_error_maximum = std::max(
                production_error_maximum,
                std::abs(production_observed[value] - references[value]));
          const gpu::near2far_statistics production_statistics =
              gpu::get_near2far_statistics();
          require(production_used_mixed &&
                      production_error_maximum <=
                          3.0e-3 * production_reference_maximum + 3.0e-8 &&
                      production_statistics.cuda_transform_calls == 1 &&
                      production_statistics.cuda_fast_precision_calls == 1 &&
                      production_statistics.cuda_mixed_precision_calls == 1 &&
                      production_statistics.cuda_cancellation_retries == 1,
                  "cylindrical production selector published an FP32 result "
                  "whose bound exceeds the error budget for m=" +
                      std::to_string(mode) + " source=" +
                      std::to_string(source_case) + " tol=" +
                      canonical_decimal(tolerance));
          ++production_mixed_retries;
        }
      }
  }
  catch (...) {
    gpu::detail::destroy_resident_near2far_plan_for_owner(&plan_owner);
    gpu::detail::destroy_resident_cache_for_owner(&resident_owner);
    throw;
  }
  gpu::detail::destroy_resident_near2far_plan_for_owner(&plan_owner);
  gpu::detail::destroy_resident_cache_for_owner(&resident_owner);
  if (am_master())
    std::cout << "PASS: cylindrical fast CUDA m=-2,0,3 and all R/P/Z E/H "
                 "sources at strict/default tolerances remain inside their "
                 "published error bounds; "
                 "max-actual/bound="
              << maximum_actual_to_bound << " mixed-retries="
              << production_mixed_retries << '\n';
#endif
}

void require_near2far_cylindrical_default_tolerance_cancellation_retry() {
#if MEEP_SINGLE && MEEP_HAVE_CUDA
  gpu::set_backend(gpu::backend_mode::cuda);
  constexpr double tolerance = 1.0e-3;
  constexpr double reference_tolerance = 1.0e-10;
  constexpr double frequency = 0.31;
  const gpu::detail::near2far_point_fp64 source = {0.43, 0.0, -0.12};
  const gpu::detail::near2far_point_fp64 target = {4.0, 0.0, 0.2};
  const gpu::detail::near2far_periodic_copy_fp64 periodic = {
      {0.0, 0.0, 0.0}, 1.0, 0.0};
  float positive_dft[2] = {0.73f, -0.19f};
  float cancelling_dft[2] = {
      -std::nextafter(positive_dft[0], 0.0f),
      -std::nextafter(positive_dft[1], 0.0f)};
  int resident_owners[2] = {};
  int plan_owner = 0;
  const gpu::detail::near2far_request_fp32 requests[] = {
      {&resident_owners[0], positive_dft, &source, 1, 0, true},
      {&resident_owners[1], cancelling_dft, &source, 1, 0, true}};
  std::complex<double> observed[6] = {};
  double published_bound = std::numeric_limits<double>::quiet_NaN();
  bool used_mixed = false;
  try {
    gpu::reset_dispatch_statistics();
    gpu::detail::resident_near2far_cartesian_fp32(
        gpu::detail::near2far_cartesian_dimension::cylindrical,
        &plan_owner, requests, 2, &target, 1, &frequency, 1,
        &periodic, 1, 1.7, 0.9, observed, &published_bound,
        false, false, false, &used_mixed, 3.0, tolerance);
    const gpu::near2far_statistics statistics =
        gpu::get_near2far_statistics();
    std::complex<double> reference[6] = {};
    greencyl(reference, veccyl(target.x, target.z), frequency, 1.7, 0.9,
             veccyl(source.x, source.z), Er,
             std::complex<double>(positive_dft[0] + cancelling_dft[0],
                                  positive_dft[1] + cancelling_dft[1]),
             3.0, reference_tolerance);
    double scale = 1.0e-300;
    double error = 0.0;
    for (int component = 0; component < 6; ++component) {
      scale = std::max(scale, std::abs(reference[component]));
      error = std::max(error,
                       std::abs(observed[component] - reference[component]));
    }
    require(used_mixed &&
                statistics.cuda_transform_calls == 1 &&
                statistics.cuda_fast_precision_calls == 1 &&
                statistics.cuda_mixed_precision_calls == 1 &&
                statistics.cuda_cancellation_retries == 1 &&
                error <= 1.0e-10 * scale + 1.0e-13,
            "default-tolerance cylindrical cancellation did not retry on "
            "mixed CUDA and agree with the FP64 greencyl oracle");
  }
  catch (...) {
    gpu::detail::destroy_resident_near2far_plan_for_owner(&plan_owner);
    gpu::detail::destroy_resident_cache_for_owner(&resident_owners[0]);
    gpu::detail::destroy_resident_cache_for_owner(&resident_owners[1]);
    throw;
  }
  gpu::detail::destroy_resident_near2far_plan_for_owner(&plan_owner);
  gpu::detail::destroy_resident_cache_for_owner(&resident_owners[0]);
  gpu::detail::destroy_resident_cache_for_owner(&resident_owners[1]);
  if (am_master())
    std::cout << "PASS: default-tolerance cylindrical cancellation retries "
                 "on mixed CUDA\n";
#endif
}

void require_near2far2d_public_api_equivalence() {
  gpu::set_backend(gpu::backend_mode::cuda);
  gpu::reset_dispatch_statistics();
  const grid_volume gv = vol2d(2.0, 1.8, 8.0);
  structure s(gv, vacuum, no_pml());
  fields f(&s, 0.0, 0.0, true, 64, 64);
  const vec center = gv.center();
  gaussian_src_time source(0.31, 0.12);
  f.add_point_source(Ez, source, center, 0.75);
  f.add_point_source(Hx, source, center + vec(0.08, -0.05),
                     std::complex<double>(-0.21, 0.17));

  const double hx = 0.46;
  const double hy = 0.38;
  volume_list near_region(
      volume(vec(center.x() + hx, center.y() - hy),
             vec(center.x() + hx, center.y() + hy)),
      Sx, 1.0,
      new volume_list(
          volume(vec(center.x() - hx, center.y() - hy),
                 vec(center.x() - hx, center.y() + hy)),
          Sx, -1.0,
          new volume_list(
              volume(vec(center.x() - hx, center.y() + hy),
                     vec(center.x() + hx, center.y() + hy)),
              Sy, 1.0,
              new volume_list(
                  volume(vec(center.x() - hx, center.y() - hy),
                         vec(center.x() + hx, center.y() - hy)),
                  Sy, -1.0))));
  const double frequencies[] = {0.23, 0.31, 0.41};
  dft_near2far near = f.add_dft_near2far(&near_region, frequencies, 3);
  for (int step = 0; step < 72; ++step) f.step();

  size_t chunk_count = 0;
  size_t source_points = 0;
  for (dft_chunk *chunk = near.F; chunk; chunk = chunk->next_in_dft)
    if (chunk->N) {
      ++chunk_count;
      source_points += chunk->N;
    }
  require(chunk_count > 0 && source_points > 0,
          "2D Near2Far fixture owns no monitor samples");
  const size_t periodic_copies =
      (2 * static_cast<size_t>(near.periodic_n[0]) + 1) *
      (2 * static_cast<size_t>(near.periodic_n[1]) + 1);
  const volume targets(
      vec(center.x() + 1.3, center.y() - 0.3),
      vec(center.x() + 1.7, center.y() + 0.3));
  const size_t expected_dims[2] = {2, 3};
  std::vector<vec> grid_points;
  for (size_t i0 = 0; i0 < expected_dims[0]; ++i0)
    for (size_t i1 = 0; i1 < expected_dims[1]; ++i1)
      grid_points.push_back(vec(
          targets.in_direction_min(X) +
              i0 * targets.in_direction(X) / (expected_dims[0] - 1),
          targets.in_direction_min(Y) +
              i1 * targets.in_direction(Y) / (expected_dims[1] - 1)));

  // Capture once while the time stepping data is still device-authoritative
  // so the no-full-DFT-download contract is exercised.  Switching to the CPU
  // oracle intentionally clears device-specific Near2Far plans, so establish
  // a second CUDA baseline afterwards for the retained-workspace eviction
  // check below.
  const near2far3d_snapshot_capture dirty_cuda =
      capture_near2far3d_snapshot(
          near, targets, grid_points, gpu::backend_mode::cuda);
  const near2far3d_snapshot_capture cpu = capture_near2far3d_snapshot(
      near, targets, grid_points, gpu::backend_mode::cpu);
  const near2far3d_snapshot_capture cuda = capture_near2far3d_snapshot(
      near, targets, grid_points, gpu::backend_mode::cuda);
  require(cuda.rank == 2 && cuda.dims[0] == expected_dims[0] &&
              cuda.dims[1] == expected_dims[1] && cuda.dims[2] == 1,
          "CUDA 2D Near2Far grid shape is incorrect");
  require(cpu.grid.size() == cuda.grid.size(),
          "CPU/CUDA 2D Near2Far grid shapes differ");
  std::vector<std::complex<double> > cpu_grid;
  std::vector<std::complex<double> > cuda_grid;
  cpu_grid.reserve(cpu.grid.size() / 2);
  cuda_grid.reserve(cuda.grid.size() / 2);
  const size_t grid_point_count = grid_points.size();
  for (int component = 0; component < 6; ++component)
    for (size_t point = 0; point < grid_point_count; ++point)
      for (size_t frequency = 0; frequency < near.freq.size(); ++frequency) {
        const size_t real_index =
            ((2 * component) * grid_point_count + point) * near.freq.size() +
            frequency;
        const size_t imaginary_index =
            ((2 * component + 1) * grid_point_count + point) *
                near.freq.size() +
            frequency;
        cpu_grid.push_back(
            {cpu.grid[real_index], cpu.grid[imaginary_index]});
        cuda_grid.push_back(
            {cuda.grid[real_index], cuda.grid[imaginary_index]});
      }
  require_near2far_complex_agreement(
      cpu_grid, cuda_grid, "2D CUDA Near2Far grid");
  require_near2far_complex_agreement(
      cpu.batch, dirty_cuda.batch,
      "device-authoritative 2D CUDA Near2Far batch");
  require_near2far_complex_agreement(
      cpu.batch, cuda.batch, "2D CUDA Near2Far batch");
  require_near2far_complex_agreement(
      cpu.scalar, cuda.scalar, "2D CUDA Near2Far scalar");

  require_near2far_cuda_transform_statistics(
      dirty_cuda.grid_statistics, grid_points.size(), near.freq.size(),
      chunk_count, source_points, periodic_copies, true, true);
  require_near2far_cuda_transform_statistics(
      dirty_cuda.batch_statistics, grid_points.size(), near.freq.size(),
      chunk_count, source_points, periodic_copies, false, true);
  require_near2far_cuda_transform_statistics(
      dirty_cuda.scalar_statistics, 1, near.freq.size(), chunk_count,
      source_points, periodic_copies, false, true);
  require_near2far_cuda_transform_statistics(
      cuda.grid_statistics, grid_points.size(), near.freq.size(),
      chunk_count, source_points, periodic_copies, true, false);
  const std::uint64_t terms_per_target =
      static_cast<std::uint64_t>(near.freq.size()) * source_points *
      periodic_copies;
  require(cpu.grid_statistics.cpu_transform_calls == grid_points.size() &&
              cpu.grid_statistics.cpu_terms ==
                  terms_per_target * grid_points.size() &&
              cpu.batch_statistics.cpu_transform_calls ==
                  grid_points.size() &&
              cpu.batch_statistics.cpu_terms ==
                  terms_per_target * grid_points.size() &&
              cpu.scalar_statistics.cpu_transform_calls == 1 &&
              cpu.scalar_statistics.cpu_terms == terms_per_target,
          "2D CPU Near2Far reference counters are inconsistent");

  // Force the 2D path through nonzero target, frequency, and operation
  // offsets. A 256-byte ceiling admits mixed CUDA's minimum workspace but no
  // more than two fast operations for one target/frequency work item, so this
  // same public transform must tile all three axes. The already-live larger
  // scratch allocation must be evicted while static descriptors remain
  // reusable.
  constexpr std::size_t forced_workspace_ceiling = 256;
  require(chunk_count > 2,
          "2D forced-tiling fixture needs more than two monitor chunks");
  const std::size_t previous_workspace_ceiling =
      gpu::detail::exchange_near2far_workspace_ceiling_for_testing(
          forced_workspace_ceiling);
  near2far3d_snapshot_capture tiled_cuda;
  try {
    tiled_cuda = capture_near2far3d_snapshot(
        near, targets, grid_points, gpu::backend_mode::cuda);
  }
  catch (...) {
    gpu::detail::exchange_near2far_workspace_ceiling_for_testing(
        previous_workspace_ceiling);
    throw;
  }
  gpu::detail::exchange_near2far_workspace_ceiling_for_testing(
      previous_workspace_ceiling);
  require_near2far_complex_agreement(
      cpu.batch, tiled_cuda.batch,
      "forced target/frequency/operation-tiled 2D Near2Far batch");
  require_near2far_complex_agreement(
      cpu.scalar, tiled_cuda.scalar,
      "forced frequency/operation-tiled 2D Near2Far scalar");
  require(cpu.grid.size() == tiled_cuda.grid.size(),
          "forced tiled 2D Near2Far grid shape differs");
  double tiled_grid_reference = 0.0;
  double tiled_grid_error = 0.0;
  for (size_t index = 0; index < cpu.grid.size(); ++index) {
    tiled_grid_reference =
        std::max(tiled_grid_reference, std::abs(cpu.grid[index]));
    tiled_grid_error = std::max(
        tiled_grid_error,
        std::abs(tiled_cuda.grid[index] - cpu.grid[index]));
  }
  require(tiled_grid_error <= 3.0e-3 * tiled_grid_reference + 3.0e-8,
          "forced tiled 2D Near2Far grid differs from CPU");
  const auto require_2d_forced_tiling = [
      forced_workspace_ceiling](
      const gpu::near2far_statistics &statistics,
      bool require_target_tiling) {
    require(statistics.cuda_transform_calls == 1 &&
                statistics.cuda_fast_precision_calls == 1 &&
                statistics.cuda_mixed_precision_calls == 0 &&
                statistics.cuda_cancellation_retries == 0 &&
                statistics.cuda_target_tiles >= 1 &&
                (!require_target_tiling ||
                 statistics.cuda_target_tiles > 1) &&
                statistics.cuda_frequency_tiles >
                    statistics.cuda_target_tiles &&
                statistics.cuda_operation_tiles >
                    statistics.cuda_frequency_tiles &&
                statistics.cuda_maximum_workspace_bytes > 0 &&
                statistics.cuda_maximum_workspace_bytes <=
                    forced_workspace_ceiling &&
                statistics.cuda_kernel_launches ==
                    2 * statistics.cuda_operation_tiles &&
                statistics.cuda_result_device_to_host_bytes >
                    12 * statistics.cuda_output_points *
                        statistics.cuda_frequencies * sizeof(double) &&
                statistics.cuda_condition_device_to_host_bytes >=
                    statistics.cuda_output_points *
                        statistics.cuda_frequencies * sizeof(double),
            "2D Near2Far did not tile target/frequency/operation axes under "
            "the forced retained-workspace ceiling");
  };
  require_2d_forced_tiling(tiled_cuda.grid_statistics, true);
  require_2d_forced_tiling(tiled_cuda.batch_statistics, true);
  require_2d_forced_tiling(tiled_cuda.scalar_statistics, false);
  require(tiled_cuda.live_before_grid == cuda.live_after_scalar &&
              tiled_cuda.live_after_grid == tiled_cuda.live_before_grid &&
              tiled_cuda.live_after_batch == tiled_cuda.live_after_grid &&
              tiled_cuda.live_after_scalar == tiled_cuda.live_after_grid,
          "2D forced workspace eviction changed the live allocation count: "
          "baseline-after=" + std::to_string(cuda.live_after_scalar) +
              " tiled-before=" +
              std::to_string(tiled_cuda.live_before_grid) +
              " tiled-grid=" +
              std::to_string(tiled_cuda.live_after_grid) +
              " tiled-batch=" +
              std::to_string(tiled_cuda.live_after_batch) +
              " tiled-scalar=" +
              std::to_string(tiled_cuda.live_after_scalar));

  const near2far3d_snapshot_capture restored_cuda =
      capture_near2far3d_snapshot(
          near, targets, grid_points, gpu::backend_mode::cuda);
  require_near2far_complex_agreement(
      cpu.batch, restored_cuda.batch,
      "restored-workspace 2D Near2Far batch");
  require(restored_cuda.grid_statistics.cuda_descriptor_uploads == 0 &&
              restored_cuda.batch_statistics.cuda_descriptor_uploads == 0 &&
              restored_cuda.scalar_statistics.cuda_descriptor_uploads == 0 &&
              restored_cuda.live_before_grid == tiled_cuda.live_after_scalar &&
              restored_cuda.live_after_scalar ==
                  restored_cuda.live_before_grid,
          "2D retained descriptor cache was not reused after scratch resize");
  near.remove();
}

void require_near2far2d_public_periodic_constructor() {
  gpu::set_backend(gpu::backend_mode::cuda);
  const grid_volume gv = vol2d(2.0, 1.8, 8.0);
  structure s(gv, vacuum, no_pml());
  fields f(&s, 0.0, 0.0, true, 64, 64);
  const vec center = gv.center();
  constexpr double bloch_x = 0.11;
  f.use_bloch(vec(bloch_x, 0.0));
  gaussian_src_time source(0.31, 0.12);
  f.add_point_source(Ez, source, center + vec(0.03, -0.21), 0.8);

  volume_list near_region(
      volume(vec(center.x() - 1.0, center.y() + 0.37),
             vec(center.x() + 1.0, center.y() + 0.37)),
      Sy, 1.0);
  const double frequencies[] = {0.27, 0.35};
  constexpr int nperiods = 2;
  dft_near2far near =
      f.add_dft_near2far(&near_region, frequencies, 2, 1, nperiods);
  require(
      near.periodic_d[0] == NO_DIRECTION && near.periodic_d[1] == X &&
          near.periodic_n[0] == 0 && near.periodic_n[1] == nperiods &&
          near.period[1] > 0.0 &&
          std::abs(near.periodic_k[1] -
                   2.0 * pi * bloch_x * near.period[1]) <= 1.0e-12,
      "public 2D add_dft_near2far did not derive its periodic/Bloch axis");
  for (int step = 0; step < 56; ++step) f.step();

  size_t chunk_count = 0;
  size_t source_points = 0;
  for (dft_chunk *chunk = near.F; chunk; chunk = chunk->next_in_dft)
    if (chunk->N) {
      ++chunk_count;
      source_points += chunk->N;
    }
  require(chunk_count > 0 && source_points > 0,
          "public periodic 2D Near2Far monitor owns no source samples");
  const std::vector<vec> targets = {
      vec(2.4, 1.3), vec(-2.1, 0.6), vec(0.8, -2.3)};
  gpu::reset_dispatch_statistics();
  std::unique_ptr<std::complex<double>[]> raw_cuda(
      near.farfields(targets.data(), targets.size()));
  const std::vector<std::complex<double> > cuda(
      raw_cuda.get(),
      raw_cuda.get() + 6 * targets.size() * near.freq.size());
  const gpu::near2far_statistics cuda_statistics =
      gpu::get_near2far_statistics();
  gpu::set_backend(gpu::backend_mode::cpu);
  gpu::reset_dispatch_statistics();
  std::unique_ptr<std::complex<double>[]> raw_cpu(
      near.farfields(targets.data(), targets.size()));
  const std::vector<std::complex<double> > cpu(
      raw_cpu.get(), raw_cpu.get() + 6 * targets.size() * near.freq.size());
  const gpu::near2far_statistics cpu_statistics =
      gpu::get_near2far_statistics();

  require_near2far_complex_agreement(
      cpu, cuda, "public periodic/Bloch 2D CUDA Near2Far");
  constexpr size_t periodic_copies = 2 * nperiods + 1;
  require_near2far_cuda_transform_statistics(
      cuda_statistics, targets.size(), near.freq.size(), chunk_count,
      source_points, periodic_copies, true, true);
  require(
      cpu_statistics.cpu_transform_calls == targets.size() &&
          cpu_statistics.cuda_transform_calls == 0 &&
          cpu_statistics.cpu_terms ==
              static_cast<std::uint64_t>(targets.size()) * near.freq.size() *
                  source_points * periodic_copies,
      "public periodic 2D CPU Near2Far evidence is inconsistent");
  near.remove();
}

void require_near2far_adjoint_two_gpu_local_contract(
    dft_near2far &near, const std::vector<vec> &targets,
    std::size_t local_chunk_count, std::size_t local_source_points,
    std::size_t periodic_copy_count, const std::string &label);

void require_near2far_cylindrical_public_api_equivalence() {
#if MEEP_SINGLE && MEEP_HAVE_CUDA
  const int modes[] = {-2, 0, 3};
  for (int mode : modes) {
    gpu::set_backend(gpu::backend_mode::cuda);
    gpu::reset_dispatch_statistics();
    grid_volume gv = volcyl(1.5, 1.4, 9.0);
    gv.center_origin();
    structure s(gv, vacuum, no_pml(), identity(), 2);
    fields f(&s, static_cast<double>(mode), 0.0, true, 64, 64);
    f.use_bloch(0.0);
    gaussian_src_time source(0.31, 0.12);
    f.add_point_source(
        Ep, source, veccyl(0.43, -0.17),
        std::complex<double>(0.71, -0.13));
    f.add_point_source(
        Ez, source, veccyl(0.29, 0.23),
        std::complex<double>(-0.19, 0.07));
    f.add_point_source(
        Hr, source, veccyl(0.57, 0.09),
        std::complex<double>(0.11, 0.05));

    volume_list near_region(
        volume(veccyl(0.0, 0.40), veccyl(0.78, 0.40)), Sz, 1.0,
        new volume_list(
            volume(veccyl(0.78, 0.40), veccyl(0.78, -0.40)), Sr, 1.0,
            new volume_list(
                volume(veccyl(0.0, -0.40), veccyl(0.78, -0.40)),
                Sz, -1.0)));
    constexpr double frequencies[] = {0.23, -0.31};
    static_assert(frequencies[0] > 0.0 && frequencies[1] < 0.0,
                  "the cylindrical CUDA Near2Far qualification must retain "
                  "signed-frequency coverage");
    dft_near2far near =
        f.add_dft_near2far(&near_region, frequencies, 2);
    for (int step = 0; step < 72; ++step) f.step();

    size_t chunk_count = 0;
    size_t source_points = 0;
    for (dft_chunk *chunk = near.F; chunk; chunk = chunk->next_in_dft)
      if (chunk->N) {
        ++chunk_count;
        source_points += chunk->N;
      }
    require(chunk_count > 2 && source_points > 0,
            "cylindrical Near2Far fixture lacks real source work");

    const volume target_grid(
        veccyl(3.0, -0.25), veccyl(3.5, 0.25));
    std::vector<vec> targets;
    for (double radius : {3.0, 3.5})
      for (double axial : {-0.25, 0.25})
        targets.push_back(veccyl(radius, axial));
    const near2far3d_snapshot_capture cuda =
        capture_near2far3d_snapshot(
            near, target_grid, targets, gpu::backend_mode::cuda);
    const near2far3d_snapshot_capture cpu =
        capture_near2far3d_snapshot(
            near, target_grid, targets, gpu::backend_mode::cpu);
    require_near2far_complex_agreement(
        cpu.batch, cuda.batch,
        "cylindrical Near2Far batched points m=" + std::to_string(mode));
    require_near2far_complex_agreement(
        cpu.scalar, cuda.scalar,
        "cylindrical Near2Far scalar point m=" + std::to_string(mode));
    require(cpu.grid.size() == cuda.grid.size(),
            "cylindrical Near2Far grid shape differs");
    double grid_reference = 0.0;
    double grid_error = 0.0;
    for (size_t index = 0; index < cpu.grid.size(); ++index) {
      grid_reference = std::max(grid_reference, std::abs(cpu.grid[index]));
      grid_error = std::max(
          grid_error, std::abs(cuda.grid[index] - cpu.grid[index]));
    }
    require(grid_error <= 3.0e-3 * grid_reference + 3.0e-8,
            "cylindrical Near2Far output grid differs from CPU for m=" +
                std::to_string(mode));
    require_near2far_cuda_transform_statistics(
        cuda.grid_statistics, targets.size(), 2, chunk_count,
        source_points, 1, true, true);
    require_near2far_cuda_transform_statistics(
        cuda.batch_statistics, targets.size(), 2, chunk_count,
        source_points, 1, false, true);
    require_near2far_cuda_transform_statistics(
        cuda.scalar_statistics, 1, 2, chunk_count,
        source_points, 1, false, true);
    const auto require_fast_probe = [mode](
        const gpu::near2far_statistics &statistics) {
      require(statistics.cuda_fast_precision_calls == 1 &&
                  statistics.cuda_mixed_precision_calls ==
                      statistics.cuda_cancellation_retries &&
                  statistics.cuda_cancellation_retries <= 1,
              "ordinary cylindrical Near2Far did not start with a fast CUDA "
              "probe and use only an evidence-justified mixed retry "
              "for m=" + std::to_string(mode) + " fast=" +
                  std::to_string(
                      statistics.cuda_fast_precision_calls) +
                  " mixed=" +
                  std::to_string(
                      statistics.cuda_mixed_precision_calls) +
                  " retries=" +
                  std::to_string(
                      statistics.cuda_cancellation_retries));
    };
    require_fast_probe(cuda.grid_statistics);
    require_fast_probe(cuda.batch_statistics);
    require_fast_probe(cuda.scalar_statistics);
    require(cuda.live_after_grid >= cuda.live_before_grid &&
                cuda.live_after_batch == cuda.live_after_grid &&
                cuda.live_after_scalar == cuda.live_after_grid,
            "cylindrical Near2Far did not stabilize its retained live-buffer "
            "count after the first transform");

    // A cylindrical monitor is a 3D equivalent-current ring.  Exercise the
    // public scalar, batch, and grid APIs at true Cartesian observation
    // points rather than silently projecting them onto phi=0.  The grid is
    // deliberately in the negative-x/nonzero-y quadrant so either projection
    // bug changes every target.
    const volume cartesian_target_grid(
        vec(-2.4, -1.4, -0.25), vec(-1.9, -0.9, 0.25));
    std::vector<vec> cartesian_targets;
    for (double x : {-2.4, -1.9})
      for (double y : {-1.4, -0.9})
        for (double z : {-0.25, 0.25})
          cartesian_targets.push_back(vec(x, y, z));
    const near2far3d_snapshot_capture cartesian_cuda =
        capture_near2far3d_snapshot(
            near, cartesian_target_grid, cartesian_targets,
            gpu::backend_mode::cuda);
    const near2far3d_snapshot_capture cartesian_cpu =
        capture_near2far3d_snapshot(
            near, cartesian_target_grid, cartesian_targets,
            gpu::backend_mode::cpu);
    require_near2far_complex_agreement(
        cartesian_cpu.batch, cartesian_cuda.batch,
        "cylindrical monitor Cartesian batch m=" +
            std::to_string(mode));
    require_near2far_complex_agreement(
        cartesian_cpu.scalar, cartesian_cuda.scalar,
        "cylindrical monitor Cartesian scalar m=" +
            std::to_string(mode));
    require(cartesian_cpu.grid.size() == cartesian_cuda.grid.size(),
            "cylindrical monitor Cartesian grid shape differs");
    double cartesian_grid_reference = 0.0;
    double cartesian_grid_error = 0.0;
    for (size_t index = 0; index < cartesian_cpu.grid.size(); ++index) {
      cartesian_grid_reference = std::max(
          cartesian_grid_reference, std::abs(cartesian_cpu.grid[index]));
      cartesian_grid_error = std::max(
          cartesian_grid_error,
          std::abs(cartesian_cuda.grid[index] - cartesian_cpu.grid[index]));
    }
    require(cartesian_grid_error <=
                3.0e-3 * cartesian_grid_reference + 3.0e-8,
            "cylindrical monitor Cartesian grid differs from CPU for m=" +
                std::to_string(mode));
    require_near2far_cuda_transform_statistics(
        cartesian_cuda.grid_statistics, cartesian_targets.size(), 2,
        chunk_count, source_points, 1, true, false);
    require_near2far_cuda_transform_statistics(
        cartesian_cuda.batch_statistics, cartesian_targets.size(), 2,
        chunk_count, source_points, 1, false, false);
    require_near2far_cuda_transform_statistics(
        cartesian_cuda.scalar_statistics, 1, 2, chunk_count,
        source_points, 1, false, false);
    require_fast_probe(cartesian_cuda.grid_statistics);
    require_fast_probe(cartesian_cuda.batch_statistics);
    require_fast_probe(cartesian_cuda.scalar_statistics);

    // Independent cylindrical symmetry oracle. For integer m, rotating the
    // observation point by alpha must multiply the global Cartesian E/H
    // vectors by exp(i*m*alpha) and rotate their x/y components by alpha.
    constexpr double alpha = 2.17;
    constexpr double radius = 3.2;
    constexpr double axial = 0.17;
    const std::vector<vec> rotation_targets = {
        vec(radius, 0.0, axial),
        vec(radius * std::cos(alpha), radius * std::sin(alpha), axial)};
    const auto evaluate_rotation_targets = [&](gpu::backend_mode backend) {
      gpu::set_backend(backend);
      gpu::reset_dispatch_statistics();
      std::unique_ptr<std::complex<double>[]> raw(
          near.farfields(rotation_targets.data(), rotation_targets.size()));
      return std::make_pair(
          std::vector<std::complex<double> >(
              raw.get(), raw.get() +
                             6 * rotation_targets.size() * near.freq.size()),
          gpu::get_near2far_statistics());
    };
    const auto rotation_cuda =
        evaluate_rotation_targets(gpu::backend_mode::cuda);
    const auto rotation_cpu =
        evaluate_rotation_targets(gpu::backend_mode::cpu);
    require_near2far_complex_agreement(
        rotation_cpu.first, rotation_cuda.first,
        "cylindrical Cartesian rotation batch m=" +
            std::to_string(mode));
    const auto require_rotation_covariance = [&](const auto &values,
                                                  const char *backend) {
      const std::complex<double> phase =
          std::polar(1.0, static_cast<double>(mode) * alpha);
      double reference = 0.0;
      double error = 0.0;
      for (size_t frequency = 0; frequency < near.freq.size(); ++frequency)
        for (int family = 0; family < 2; ++family) {
          const size_t base = 6 * frequency + 3 * family;
          const size_t rotated =
              6 * (near.freq.size() + frequency) + 3 * family;
          const std::complex<double> expected[3] = {
              phase * (std::cos(alpha) * values[base] -
                       std::sin(alpha) * values[base + 1]),
              phase * (std::sin(alpha) * values[base] +
                       std::cos(alpha) * values[base + 1]),
              phase * values[base + 2]};
          for (int component = 0; component < 3; ++component) {
            reference = std::max(reference, std::abs(expected[component]));
            error = std::max(
                error,
                std::abs(values[rotated + component] - expected[component]));
          }
        }
      require(error <= 4.0e-3 * reference + 5.0e-8,
              std::string(backend) +
                  " cylindrical Cartesian rotation covariance failed for m=" +
                  std::to_string(mode) + " error=" +
                  std::to_string(error) + " reference=" +
                  std::to_string(reference));
    };
    require_rotation_covariance(rotation_cpu.first, "CPU");
    require_rotation_covariance(rotation_cuda.first, "CUDA");
    require_near2far_cuda_transform_statistics(
        rotation_cuda.second, rotation_targets.size(), 2, chunk_count,
        source_points, 1, false, false);

    const std::vector<vec> cartesian_adjoint_targets = {
        vec(-2.2, 1.7, -0.13), vec(2.4, -1.1, 0.29)};
    require_near2far_adjoint_two_gpu_local_contract(
        near, cartesian_adjoint_targets, chunk_count, source_points, 1,
        "cylindrical monitor Cartesian adjoint m=" +
            std::to_string(mode));

    if (mode == 0) {
      // Exercise nonzero target, frequency, and operation offsets in the
      // cylindrical kernel.  This ceiling is intentionally small enough to
      // force all three tiling axes while retaining a valid one-work-item
      // launch.  The static descriptor cache must survive both scratch
      // shrinkage and the subsequent restoration of the default ceiling.
      constexpr std::size_t forced_workspace_ceiling = 256;
      require(chunk_count > 2,
              "cylindrical forced-tiling fixture needs multiple chunks");
      const std::size_t previous_workspace_ceiling =
          gpu::detail::exchange_near2far_workspace_ceiling_for_testing(
              forced_workspace_ceiling);
      near2far3d_snapshot_capture tiled_cuda;
      try {
        tiled_cuda = capture_near2far3d_snapshot(
            near, target_grid, targets, gpu::backend_mode::cuda);
      }
      catch (...) {
        gpu::detail::exchange_near2far_workspace_ceiling_for_testing(
            previous_workspace_ceiling);
        throw;
      }
      gpu::detail::exchange_near2far_workspace_ceiling_for_testing(
          previous_workspace_ceiling);
      require_near2far_complex_agreement(
          cpu.batch, tiled_cuda.batch,
          "forced target/frequency/operation-tiled cylindrical batch");
      require_near2far_complex_agreement(
          cpu.scalar, tiled_cuda.scalar,
          "forced frequency/operation-tiled cylindrical scalar");
      require(cpu.grid.size() == tiled_cuda.grid.size(),
              "forced tiled cylindrical grid shape differs");
      double tiled_grid_reference = 0.0;
      double tiled_grid_error = 0.0;
      for (size_t index = 0; index < cpu.grid.size(); ++index) {
        tiled_grid_reference = std::max(
            tiled_grid_reference, std::abs(cpu.grid[index]));
        tiled_grid_error = std::max(
            tiled_grid_error,
            std::abs(tiled_cuda.grid[index] - cpu.grid[index]));
      }
      require(tiled_grid_error <=
                  3.0e-3 * tiled_grid_reference + 3.0e-8,
              "forced tiled cylindrical grid differs from CPU");
      const auto require_cylindrical_forced_tiling = [
          forced_workspace_ceiling](
          const gpu::near2far_statistics &statistics,
          bool require_target_tiling) {
        const std::uint64_t attempts =
            1 + statistics.cuda_cancellation_retries;
        require(
            statistics.cuda_transform_calls == 1 &&
                statistics.cuda_fast_precision_calls == 1 &&
                statistics.cuda_mixed_precision_calls ==
                    statistics.cuda_cancellation_retries &&
                statistics.cuda_cancellation_retries <= 1 &&
                statistics.cuda_target_tiles >= attempts &&
                (!require_target_tiling ||
                 statistics.cuda_target_tiles > attempts) &&
                statistics.cuda_frequency_tiles >
                    statistics.cuda_target_tiles &&
                statistics.cuda_operation_tiles >
                    statistics.cuda_frequency_tiles &&
                statistics.cuda_maximum_workspace_bytes > 0 &&
                statistics.cuda_maximum_workspace_bytes <=
                    forced_workspace_ceiling &&
                statistics.cuda_kernel_launches ==
                    2 * statistics.cuda_operation_tiles &&
                statistics.cuda_result_device_to_host_bytes >
                    12 * statistics.cuda_output_points *
                        statistics.cuda_frequencies * sizeof(double) &&
                statistics.cuda_condition_device_to_host_bytes >=
                    statistics.cuda_output_points *
                        statistics.cuda_frequencies * sizeof(double),
            "cylindrical Near2Far did not tile target/frequency/operation "
            "axes under the forced retained-workspace ceiling: fast=" +
                std::to_string(
                    statistics.cuda_fast_precision_calls) +
                " mixed=" +
                std::to_string(
                    statistics.cuda_mixed_precision_calls) +
                " retries=" +
                std::to_string(
                    statistics.cuda_cancellation_retries) +
                " target=" +
                std::to_string(statistics.cuda_target_tiles) +
                " frequency=" +
                std::to_string(statistics.cuda_frequency_tiles) +
                " operation=" +
                std::to_string(statistics.cuda_operation_tiles));
      };
      require_cylindrical_forced_tiling(
          tiled_cuda.grid_statistics, true);
      require_cylindrical_forced_tiling(
          tiled_cuda.batch_statistics, true);
      require_cylindrical_forced_tiling(
          tiled_cuda.scalar_statistics, false);
      require(tiled_cuda.live_after_grid >=
                  tiled_cuda.live_before_grid &&
                  tiled_cuda.live_after_batch ==
                      tiled_cuda.live_after_grid &&
                  tiled_cuda.live_after_scalar ==
                      tiled_cuda.live_after_grid,
              "cylindrical forced workspace changed the live allocation "
              "count after its first transform");

      const near2far3d_snapshot_capture restored_cuda =
          capture_near2far3d_snapshot(
              near, target_grid, targets, gpu::backend_mode::cuda);
      require_near2far_complex_agreement(
          cpu.batch, restored_cuda.batch,
          "restored-workspace cylindrical Near2Far batch");
      const auto descriptor_reuse_or_precision_retry = [](
          const gpu::near2far_statistics &statistics) {
        return statistics.cuda_cancellation_retries
                   ? statistics.cuda_descriptor_uploads >= 1 &&
                         statistics.cuda_descriptor_uploads <= 2
                   : statistics.cuda_descriptor_uploads == 0;
      };
      require(descriptor_reuse_or_precision_retry(
                  restored_cuda.grid_statistics) &&
                  descriptor_reuse_or_precision_retry(
                      restored_cuda.batch_statistics) &&
                  descriptor_reuse_or_precision_retry(
                      restored_cuda.scalar_statistics) &&
                  restored_cuda.live_before_grid ==
                      tiled_cuda.live_after_scalar &&
                  restored_cuda.live_after_scalar ==
                      restored_cuda.live_before_grid,
              "cylindrical retained descriptor cache was neither reused nor "
              "rebuilt exactly for an evidence-justified precision retry "
              "after scratch restoration");
    }
    near.remove();
  }
  if (am_master())
    std::cout << "PASS: cylindrical Near2Far scalar/batch/grid and adjoint "
                 "transforms for m=-2,0,3 preserve arbitrary Cartesian "
                 "targets, satisfy rotation covariance, use strict CUDA, "
                 "and agree with CPU\n";
#endif
}

void require_near2far_adjoint_two_gpu_local_contract(
    dft_near2far &near, const std::vector<vec> &targets,
    std::size_t local_chunk_count, std::size_t local_source_points,
    std::size_t periodic_copy_count, const std::string &label) {
  const bool stage_trace =
      std::getenv("MEEP_GPU_TEST_MPI_STAGE_TRACE") != nullptr;
  const auto trace_stage = [&](const char *stage) {
    if (stage_trace)
      std::cerr << "rank=" << my_rank() << " adjoint-stage=" << stage
                << std::endl;
  };
  std::vector<double> far_points(3 * targets.size());
  for (std::size_t target = 0; target < targets.size(); ++target) {
    far_points[3 * target] = targets[target].dim == Dcyl
                                 ? targets[target].in_direction(R)
                                 : targets[target].in_direction(X);
    far_points[3 * target + 1] = targets[target].dim == Dcyl
                                     ? 0.0
                                     : targets[target].in_direction(Y);
    far_points[3 * target + 2] = targets[target].in_direction(Z);
  }
  std::vector<std::complex<double> > gradient(
      6 * targets.size() * near.freq.size());
  for (std::size_t value = 0; value < gradient.size(); ++value)
    gradient[value] = std::complex<double>(
        0.11 + 0.003 * static_cast<double>(value),
        -0.07 + 0.002 * static_cast<double>(value));

  gpu::set_backend(gpu::backend_mode::cuda);
  gpu::reset_dispatch_statistics();
  trace_stage("before-cuda-near-sourcedata");
  const std::vector<sourcedata> cuda = near.near_sourcedata(
      targets.front(), far_points.data(), targets.size(), gradient.data(),
      1.0e-3);
  trace_stage("after-cuda-near-sourcedata");
  const gpu::near2far_statistics cuda_statistics =
      gpu::get_near2far_statistics();
  gpu::set_backend(gpu::backend_mode::cpu);
  gpu::reset_dispatch_statistics();
  trace_stage("before-cpu-near-sourcedata");
  const std::vector<sourcedata> cpu = near.near_sourcedata(
      targets.front(), far_points.data(), targets.size(), gradient.data(),
      1.0e-3);
  trace_stage("after-cpu-near-sourcedata");
  const gpu::near2far_statistics cpu_statistics =
      gpu::get_near2far_statistics();

  bool local_topology_pass = cpu.size() == cuda.size();
  double reference_squared = 0.0;
  double error_squared = 0.0;
  double reference_maximum = 0.0;
  double error_maximum = 0.0;
  std::size_t observed_source_points = 0;
  const std::size_t comparable_chunks = std::min(cpu.size(), cuda.size());
  for (std::size_t chunk = 0; chunk < comparable_chunks; ++chunk) {
    const bool chunk_topology_pass =
        cpu[chunk].near_fd_comp == cuda[chunk].near_fd_comp &&
        cpu[chunk].idx_arr == cuda[chunk].idx_arr &&
        cpu[chunk].amp_arr.size() == cuda[chunk].amp_arr.size();
    local_topology_pass = local_topology_pass && chunk_topology_pass;
    observed_source_points += cpu[chunk].idx_arr.size();
    const std::size_t comparable_values =
        std::min(cpu[chunk].amp_arr.size(), cuda[chunk].amp_arr.size());
    for (std::size_t value = 0; value < comparable_values; ++value) {
      const double error =
          std::abs(cuda[chunk].amp_arr[value] - cpu[chunk].amp_arr[value]);
      reference_squared += std::norm(cpu[chunk].amp_arr[value]);
      error_squared += error * error;
      reference_maximum =
          std::max(reference_maximum, std::abs(cpu[chunk].amp_arr[value]));
      error_maximum = std::max(error_maximum, error);
    }
  }
  const double nrmse = std::sqrt(
      error_squared / std::max(reference_squared, 1.0e-300));
  const std::uint64_t expected_terms =
      static_cast<std::uint64_t>(local_source_points) * targets.size() *
      near.freq.size() * periodic_copy_count;
  const bool local_pass =
      local_topology_pass && observed_source_points == local_source_points &&
      nrmse <= 1.0e-3 &&
      error_maximum <= 4.0e-3 * reference_maximum + 3.0e-8 &&
      cuda_statistics.cuda_adjoint_calls == 1 &&
      cuda_statistics.cpu_adjoint_calls == 0 &&
      cuda_statistics.cuda_adjoint_terms == expected_terms &&
      cuda_statistics.cuda_adjoint_submitted_chunks == local_chunk_count &&
      cuda_statistics.cuda_adjoint_source_points == local_source_points &&
      cuda_statistics.cuda_adjoint_far_points == targets.size() &&
      cuda_statistics.cuda_adjoint_frequencies == near.freq.size() &&
      cuda_statistics.cuda_adjoint_periodic_copies == periodic_copy_count &&
      cuda_statistics.cuda_adjoint_kernel_launches > 0 &&
      cuda_statistics.cuda_adjoint_maximum_workspace_bytes > 0 &&
      cuda_statistics.cuda_adjoint_maximum_workspace_bytes <=
          64u * 1024u * 1024u &&
      cpu_statistics.cpu_adjoint_calls == 1 &&
      cpu_statistics.cuda_adjoint_calls == 0 &&
      cpu_statistics.cpu_adjoint_terms == expected_terms;
  trace_stage("before-pass-allreduce");
  const bool global_pass = and_to_all(local_pass);
  trace_stage("after-pass-allreduce");
  if (!global_pass)
    throw std::runtime_error(
        label + " did not execute rank-local adjoint CUDA VJP on both "
                "GPUs with CPU agreement: nrmse=" +
        std::to_string(nrmse) + " maximum_error=" +
        std::to_string(error_maximum));
}

void require_near2far2d_two_gpu_mpi_contract() {
#if MEEP_SINGLE && MEEP_HAVE_CUDA
  const bool stage_trace =
      std::getenv("MEEP_GPU_TEST_MPI_STAGE_TRACE") != nullptr;
  const auto trace_stage = [&](const std::string &stage) {
    if (stage_trace)
      std::cerr << "rank=" << my_rank() << " n2f2d-stage=" << stage
                << std::endl;
  };
  trace_stage("entry");
  require(count_processors() == 2,
          "2D Near2Far multi-GPU regression requires two MPI ranks");
  gpu::set_backend(gpu::backend_mode::cuda);
  require(and_to_all(gpu::active_backend() == gpu::backend_mode::cuda),
          "2D Near2Far multi-GPU regression requires CUDA on every rank");

  const std::string local_identifier = gpu::selected_device_identifier();
  constexpr int identifier_capacity = 4096;
  char root_identifier[identifier_capacity] = {};
  if (my_rank() == 0)
    std::snprintf(root_identifier, sizeof(root_identifier), "%s",
                  local_identifier.c_str());
  broadcast(0, root_identifier, identifier_capacity);
  require(and_to_all(!local_identifier.empty() &&
                     (my_rank() == 0
                          ? local_identifier == root_identifier
                          : local_identifier != root_identifier)),
          "2D Near2Far two-rank regression did not claim two distinct "
          "physical GPU identifiers");
  trace_stage("distinct-gpus-confirmed");

  const grid_volume gv = vol2d(2.4, 2.0, 10.0);
  trace_stage("before-structure");
  structure s(gv, vacuum, no_pml(), identity(), 2);
  trace_stage("after-structure");
  fields f(&s, 0.0, 0.0, true, 64, 64);
  trace_stage("after-fields");
  const vec center = gv.center();
  constexpr double bloch_x = 0.09;
  f.use_bloch(vec(bloch_x, 0.0));
  trace_stage("after-use-bloch");
  gaussian_src_time source_time(0.31, 0.12);
  f.add_volume_source(Ez, source_time, f.v,
                      std::complex<double>(0.71, -0.13));
  trace_stage("after-ez-source");
  f.add_volume_source(Hx, source_time, f.v,
                      std::complex<double>(-0.19, 0.07));
  trace_stage("after-hx-source");
  volume_list near_region(
      volume(vec(center.x() - 1.2, center.y() + 0.41),
             vec(center.x() + 1.2, center.y() + 0.41)),
      Sy, 1.0);
  // Cartesian 2D uses the outgoing Hankel selector, whose CUDA contract is
  // intentionally limited to positive frequencies.  Signed-frequency
  // coverage belongs to the cylindrical greencyl fixture below.
  const double frequencies[] = {0.23, 0.31, 0.41};
  constexpr int nperiods = 2;
  dft_near2far near =
      f.add_dft_near2far(&near_region, frequencies, 3, 1, nperiods);
  trace_stage("after-add-dft");
  require(and_to_all(
              near.periodic_d[0] == NO_DIRECTION &&
              near.periodic_d[1] == X && near.periodic_n[0] == 0 &&
              near.periodic_n[1] == nperiods && near.period[1] > 0.0 &&
              std::abs(near.periodic_k[1] -
                       2.0 * pi * bloch_x * near.period[1]) <= 1.0e-12),
          "2D multi-GPU fixture did not retain its public periodic/Bloch "
          "descriptor on every rank");
  for (int step = 0; step < 72; ++step) {
    if (step % 8 == 0) trace_stage("before-step-" + std::to_string(step));
    f.step();
  }
  trace_stage("after-step-72");

  size_t local_chunk_count = 0;
  size_t local_source_points = 0;
  for (dft_chunk *chunk = near.F; chunk; chunk = chunk->next_in_dft)
    if (chunk->N) {
      ++local_chunk_count;
      local_source_points += chunk->N;
    }
  require(and_to_all(local_chunk_count > 0 && local_source_points > 0),
          "2D multi-GPU fixture did not place real monitor work on every "
          "rank");
  std::vector<vec> targets;
  for (size_t index = 0; index < 17; ++index) {
    const double azimuth =
        2.0 * pi * (static_cast<double>(index) + 0.37) / 17.0;
    const double radius = 2.7 + 0.13 * std::sin(5.0 * azimuth);
    targets.push_back(center +
                      vec(radius * std::cos(azimuth),
                          radius * std::sin(azimuth)));
  }
  const size_t field_count = 6 * targets.size() * near.freq.size();

  gpu::reset_dispatch_statistics();
  trace_stage("before-cuda-forward");
  std::unique_ptr<std::complex<double>[]> raw_cuda(
      near.farfields(targets.data(), targets.size()));
  trace_stage("after-cuda-forward");
  std::vector<std::complex<double> > cuda_fields(
      raw_cuda.get(), raw_cuda.get() + field_count);
  const gpu::near2far_statistics cuda_statistics =
      gpu::get_near2far_statistics();
  std::vector<std::complex<double> > root_cuda(cuda_fields);
  if (!am_master())
    std::fill(root_cuda.begin(), root_cuda.end(),
              std::complex<double>(0.0, 0.0));
  broadcast(0, root_cuda.data(), static_cast<int>(root_cuda.size()));
  const std::uint64_t global_cancellation_retries =
      static_cast<std::uint64_t>(max_to_all(static_cast<int>(
          cuda_statistics.cuda_cancellation_retries)));
  require(cuda_fields == root_cuda &&
              cuda_statistics.cuda_transform_calls == 1 &&
              cuda_statistics.cpu_transform_calls == 0 &&
              cuda_statistics.cuda_submitted_chunks == local_chunk_count &&
              cuda_statistics.cuda_source_points == local_source_points &&
              cuda_statistics.cuda_output_points == targets.size() &&
              cuda_statistics.cuda_frequencies == near.freq.size() &&
              cuda_statistics.cuda_periodic_copies == 2 * nperiods + 1 &&
              cuda_statistics.cuda_fast_precision_calls +
                      cuda_statistics.cuda_mixed_precision_calls ==
                  1 + cuda_statistics.cuda_cancellation_retries &&
              cuda_statistics.mpi_allreduce_calls ==
                  1 + global_cancellation_retries &&
              cuda_statistics.mpi_allreduce_bytes ==
                  (1 + global_cancellation_retries) * 13 *
                      targets.size() * near.freq.size() * sizeof(double) &&
              cuda_statistics.dft_device_to_host_bytes_avoided > 0,
          "2D two-GPU Near2Far did not execute rank-local CUDA work and the "
          "13-double result collective on every rank");

  gpu::set_backend(gpu::backend_mode::cpu);
  gpu::reset_dispatch_statistics();
  trace_stage("before-cpu-forward");
  std::unique_ptr<std::complex<double>[]> raw_cpu(
      near.farfields(targets.data(), targets.size()));
  trace_stage("after-cpu-forward");
  const std::vector<std::complex<double> > cpu_fields(
      raw_cpu.get(), raw_cpu.get() + field_count);
  require_near2far_complex_agreement(
      cpu_fields, cuda_fields,
      "2D two-GPU periodic/Bloch Near2Far CPU oracle");
  require_near2far_adjoint_two_gpu_local_contract(
      near, targets, local_chunk_count, local_source_points,
      2 * nperiods + 1, "2D two-GPU periodic/Bloch Near2Far adjoint");
  trace_stage("after-adjoint");

  // Remove all local monitor chunks from rank one. Both ranks must still
  // enter exactly one 13-channel collective; only rank zero may launch the
  // transform, and both must receive the same nonzero result.
  dft_chunk *saved_chunks = near.F;
  if (my_rank() == 1) near.F = nullptr;
  gpu::set_backend(gpu::backend_mode::cuda);
  gpu::reset_dispatch_statistics();
  trace_stage("before-chunkless-cuda-forward");
  std::unique_ptr<std::complex<double>[]> raw_chunkless_cuda(
      near.farfields(targets.data(), targets.size()));
  trace_stage("after-chunkless-cuda-forward");
  const std::vector<std::complex<double> > chunkless_cuda(
      raw_chunkless_cuda.get(), raw_chunkless_cuda.get() + field_count);
  const gpu::near2far_statistics chunkless_statistics =
      gpu::get_near2far_statistics();
  std::vector<std::complex<double> > chunkless_root(chunkless_cuda);
  if (!am_master())
    std::fill(chunkless_root.begin(), chunkless_root.end(),
              std::complex<double>(0.0, 0.0));
  broadcast(0, chunkless_root.data(),
            static_cast<int>(chunkless_root.size()));
  require(chunkless_cuda == chunkless_root &&
              chunkless_statistics.mpi_allreduce_calls >= 1 &&
              chunkless_statistics.mpi_allreduce_bytes >=
                  13 * targets.size() * near.freq.size() * sizeof(double) &&
              (my_rank() == 0
                   ? chunkless_statistics.cuda_transform_calls == 1 &&
                         chunkless_statistics.cuda_submitted_chunks ==
                             local_chunk_count
                   : chunkless_statistics.cuda_transform_calls == 0 &&
                         chunkless_statistics.cuda_submitted_chunks == 0),
          "2D chunkless rank did not participate in the production "
          "13-double CUDA collective");
  gpu::set_backend(gpu::backend_mode::cpu);
  trace_stage("before-chunkless-cpu-forward");
  std::unique_ptr<std::complex<double>[]> raw_chunkless_cpu(
      near.farfields(targets.data(), targets.size()));
  trace_stage("after-chunkless-cpu-forward");
  const std::vector<std::complex<double> > chunkless_cpu(
      raw_chunkless_cpu.get(), raw_chunkless_cpu.get() + field_count);
  require_near2far_complex_agreement(
      chunkless_cpu, chunkless_cuda,
      "2D chunkless-rank CUDA/CPU Near2Far result");
  if (my_rank() == 1) near.F = saved_chunks;
  near.remove();
  trace_stage("complete");
  if (am_master())
    std::cout << "PASS: 2D periodic/Bloch Near2Far uses two distinct GPUs, "
                 "rank-local CUDA, 13-double collectives, and a chunkless "
                 "rank with CPU agreement\n";
#else
  throw std::runtime_error(
      "2D Near2Far multi-GPU regression requires CUDA FP32 support");
#endif
}

void require_near2far_cylindrical_two_gpu_mpi_contract() {
#if MEEP_SINGLE && MEEP_HAVE_CUDA
  require(count_processors() == 2,
          "cylindrical Near2Far multi-GPU regression requires two MPI "
          "ranks");
  gpu::set_backend(gpu::backend_mode::cuda);
  require(and_to_all(gpu::active_backend() == gpu::backend_mode::cuda),
          "cylindrical Near2Far multi-GPU regression requires CUDA on "
          "every rank");

  const std::string local_identifier = gpu::selected_device_identifier();
  constexpr int identifier_capacity = 4096;
  char root_identifier[identifier_capacity] = {};
  if (my_rank() == 0)
    std::snprintf(root_identifier, sizeof(root_identifier), "%s",
                  local_identifier.c_str());
  broadcast(0, root_identifier, identifier_capacity);
  require(and_to_all(!local_identifier.empty() &&
                     (my_rank() == 0
                          ? local_identifier == root_identifier
                          : local_identifier != root_identifier)),
          "cylindrical Near2Far two-rank regression did not claim two "
          "distinct physical GPU identifiers");

  grid_volume gv = volcyl(1.6, 1.4, 10.0);
  gv.center_origin();
  structure s(gv, vacuum, no_pml(), identity(), 2);
  constexpr double azimuthal_mode = 2.0;
  fields f(&s, azimuthal_mode, 0.0, true, 64, 64);
  constexpr double bloch_z = 0.07;
  f.use_bloch(veccyl(0.0, bloch_z));
  gaussian_src_time source_time(0.31, 0.12);
  const volume source_region(veccyl(0.2, -0.6), veccyl(1.4, 0.6));
  f.add_volume_source(Ep, source_time, source_region,
                      std::complex<double>(0.71, -0.13));
  f.add_volume_source(Ez, source_time, source_region,
                      std::complex<double>(-0.19, 0.07));
  f.add_volume_source(Hr, source_time, source_region,
                      std::complex<double>(0.11, 0.05));
  // Default cylindrical chunking bisects the radial direction.  Put one
  // longitudinal monitor face on each side of that split so both ranks own
  // real DFT work rather than merely participating in the result collective.
  volume_list near_region(
      volume(veccyl(0.4, -0.7), veccyl(0.4, 0.7)), Sr, -1.0,
      new volume_list(
          volume(veccyl(1.2, -0.7), veccyl(1.2, 0.7)), Sr, 1.0));
  const double frequencies[] = {0.23, 0.31, 0.41};
  constexpr int nperiods = 2;
  dft_near2far near =
      f.add_dft_near2far(&near_region, frequencies, 3, 1, nperiods);
  require(and_to_all(
              near.periodic_d[0] == NO_DIRECTION &&
              near.periodic_d[1] == Z && near.periodic_n[0] == 0 &&
              near.periodic_n[1] == nperiods && near.period[1] > 0.0 &&
              std::abs(near.periodic_k[1] -
                       2.0 * pi * bloch_z * near.period[1]) <= 1.0e-12),
          "cylindrical multi-GPU fixture did not retain its public "
          "periodic-Z/Bloch descriptor on every rank");
  for (int step = 0; step < 72; ++step) f.step();

  size_t local_chunk_count = 0;
  size_t local_source_points = 0;
  bool local_modes_valid = true;
  for (dft_chunk *chunk = near.F; chunk; chunk = chunk->next_in_dft)
    if (chunk->N) {
      ++local_chunk_count;
      local_source_points += chunk->N;
      local_modes_valid =
          local_modes_valid && chunk->fc &&
          chunk->fc->m == azimuthal_mode;
    }
  require(and_to_all(local_chunk_count > 0 && local_source_points > 0 &&
                     local_modes_valid),
          "cylindrical multi-GPU fixture did not place real monitor work "
          "with the expected azimuthal mode on every rank");
  std::vector<vec> targets;
  for (size_t index = 0; index < 13; ++index) {
    const double phase =
        2.0 * pi * (static_cast<double>(index) + 0.37) / 13.0;
    const double radius = 2.35 + 0.11 * std::sin(3.0 * phase);
    const double axial = 0.31 * std::cos(2.0 * phase);
    targets.push_back(
        vec(radius * std::cos(phase), radius * std::sin(phase), axial));
  }
  const size_t field_count = 6 * targets.size() * near.freq.size();

  gpu::reset_dispatch_statistics();
  std::unique_ptr<std::complex<double>[]> raw_cuda(
      near.farfields(targets.data(), targets.size()));
  const std::vector<std::complex<double> > cuda_fields(
      raw_cuda.get(), raw_cuda.get() + field_count);
  const gpu::near2far_statistics cuda_statistics =
      gpu::get_near2far_statistics();
  std::vector<std::complex<double> > root_cuda(cuda_fields);
  if (!am_master())
    std::fill(root_cuda.begin(), root_cuda.end(),
              std::complex<double>(0.0, 0.0));
  broadcast(0, root_cuda.data(), static_cast<int>(root_cuda.size()));
  const std::uint64_t global_cancellation_retries =
      static_cast<std::uint64_t>(max_to_all(static_cast<int>(
          cuda_statistics.cuda_cancellation_retries)));
  if (std::getenv("MEEP_GPU_TEST_MPI_STAGE_TRACE"))
    std::cerr
        << "rank=" << my_rank() << " n2fcyl-forward-stats chunks="
        << local_chunk_count << '/' << cuda_statistics.cuda_submitted_chunks
        << " sources=" << local_source_points << '/'
        << cuda_statistics.cuda_source_points << " outputs="
        << cuda_statistics.cuda_output_points << " frequencies="
        << cuda_statistics.cuda_frequencies << " copies="
        << cuda_statistics.cuda_periodic_copies << " calls(cuda/cpu)="
        << cuda_statistics.cuda_transform_calls << '/'
        << cuda_statistics.cpu_transform_calls << " precision(fast/mixed/retry)="
        << cuda_statistics.cuda_fast_precision_calls << '/'
        << cuda_statistics.cuda_mixed_precision_calls << '/'
        << cuda_statistics.cuda_cancellation_retries << " allreduce(calls/bytes)="
        << cuda_statistics.mpi_allreduce_calls << '/'
        << cuda_statistics.mpi_allreduce_bytes << " dft-avoided="
        << cuda_statistics.dft_device_to_host_bytes_avoided << std::endl;
  require(cuda_fields == root_cuda &&
              cuda_statistics.cuda_transform_calls == 1 &&
              cuda_statistics.cpu_transform_calls == 0 &&
              cuda_statistics.cuda_submitted_chunks == local_chunk_count &&
              cuda_statistics.cuda_source_points == local_source_points &&
              cuda_statistics.cuda_output_points == targets.size() &&
              cuda_statistics.cuda_frequencies == near.freq.size() &&
              cuda_statistics.cuda_periodic_copies == 2 * nperiods + 1 &&
              cuda_statistics.cuda_fast_precision_calls +
                      cuda_statistics.cuda_mixed_precision_calls ==
                  1 + cuda_statistics.cuda_cancellation_retries &&
              cuda_statistics.mpi_allreduce_calls ==
                  1 + global_cancellation_retries &&
              cuda_statistics.mpi_allreduce_bytes ==
                  (1 + global_cancellation_retries) * 13 *
                      targets.size() * near.freq.size() * sizeof(double) &&
              cuda_statistics.dft_device_to_host_bytes_avoided > 0,
          "cylindrical two-GPU Near2Far did not execute rank-local CUDA "
          "work and the 13-double result collective on every rank");

  gpu::set_backend(gpu::backend_mode::cpu);
  gpu::reset_dispatch_statistics();
  std::unique_ptr<std::complex<double>[]> raw_cpu(
      near.farfields(targets.data(), targets.size()));
  const std::vector<std::complex<double> > cpu_fields(
      raw_cpu.get(), raw_cpu.get() + field_count);
  require_near2far_complex_agreement(
      cpu_fields, cuda_fields,
      "cylindrical two-GPU periodic/Bloch Near2Far CPU oracle");
  require_near2far_adjoint_two_gpu_local_contract(
      near, targets, local_chunk_count, local_source_points,
      2 * nperiods + 1,
      "cylindrical two-GPU periodic/Bloch Near2Far adjoint");

  // Remove all local monitor chunks from rank one.  The mode consensus must
  // be learned from rank zero before both ranks enter the same 13-channel
  // result collective; only the owning rank may launch a CUDA transform.
  dft_chunk *saved_chunks = near.F;
  if (my_rank() == 1) near.F = nullptr;
  gpu::set_backend(gpu::backend_mode::cuda);
  gpu::reset_dispatch_statistics();
  std::unique_ptr<std::complex<double>[]> raw_chunkless_cuda(
      near.farfields(targets.data(), targets.size()));
  const std::vector<std::complex<double> > chunkless_cuda(
      raw_chunkless_cuda.get(), raw_chunkless_cuda.get() + field_count);
  const gpu::near2far_statistics chunkless_statistics =
      gpu::get_near2far_statistics();
  std::vector<std::complex<double> > chunkless_root(chunkless_cuda);
  if (!am_master())
    std::fill(chunkless_root.begin(), chunkless_root.end(),
              std::complex<double>(0.0, 0.0));
  broadcast(0, chunkless_root.data(),
            static_cast<int>(chunkless_root.size()));
  require(chunkless_cuda == chunkless_root &&
              chunkless_statistics.mpi_allreduce_calls >= 1 &&
              chunkless_statistics.mpi_allreduce_bytes >=
                  13 * targets.size() * near.freq.size() * sizeof(double) &&
              (my_rank() == 0
                   ? chunkless_statistics.cuda_transform_calls == 1 &&
                         chunkless_statistics.cuda_submitted_chunks ==
                             local_chunk_count
                   : chunkless_statistics.cuda_transform_calls == 0 &&
                         chunkless_statistics.cuda_submitted_chunks == 0),
          "cylindrical chunkless rank did not participate in the production "
          "13-double CUDA collective");
  gpu::set_backend(gpu::backend_mode::cpu);
  std::unique_ptr<std::complex<double>[]> raw_chunkless_cpu(
      near.farfields(targets.data(), targets.size()));
  const std::vector<std::complex<double> > chunkless_cpu(
      raw_chunkless_cpu.get(), raw_chunkless_cpu.get() + field_count);
  require_near2far_complex_agreement(
      chunkless_cpu, chunkless_cuda,
      "cylindrical chunkless-rank CUDA/CPU Near2Far result");
  if (my_rank() == 1) near.F = saved_chunks;
  near.remove();
  if (am_master())
    std::cout << "PASS: cylindrical periodic-Z/Bloch Near2Far preserves "
                 "arbitrary Cartesian targets on two distinct GPUs, uses "
                 "rank-local CUDA and 13-double collectives, and handles a "
                 "chunkless rank with CPU agreement\n";
#else
  throw std::runtime_error(
      "cylindrical Near2Far multi-GPU regression requires CUDA FP32 "
      "support");
#endif
}

void require_near2far_channel_cancellation_retry() {
#if MEEP_SINGLE && MEEP_HAVE_CUDA
  gpu::set_backend(gpu::backend_mode::cuda);
  gpu::reset_dispatch_statistics();
  const gpu::detail::near2far_point_fp64 source = {0.1, 0.2, 0.3};
  const gpu::detail::near2far_point_fp64 perturbed_source = {
      static_cast<double>(std::nextafter(0.1f, 1.0f)), 0.2, 0.3};
  const gpu::detail::near2far_point_fp64 target = {11.0, 0.2, 0.3};
  const gpu::detail::near2far_periodic_copy_fp64 periodic = {
      {0.0, 0.0, 0.0}, 1.0, 0.0};
  const double frequency = 0.31;
  float x_positive[2] = {1.0f, 0.0f};
  float x_negative[2] = {
      -std::nextafter(1.0f, 0.0f), 0.0f};
  // The transverse channel is large compared with the residual canceled Ex
  // channel, but deliberately smaller than the two hidden Ex contributions.
  // This proves that the gate is channel-wise while still honoring the public
  // transform's small global absolute-error allowance.
  float y_control[2] = {1.0e-5f, 0.0f};
  int resident_owner = 0;
  int plan_owner = 0;
  const gpu::detail::near2far_request_fp32 requests[3] = {
      {&resident_owner, x_positive, &source, 1, 0, true},
      {&resident_owner, x_negative, &perturbed_source, 1, 0, true},
      {&resident_owner, y_control, &source, 1, 1, true}};
  std::complex<double> observed[6] = {};
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  try {
    gpu::detail::resident_near2far_3d_fp32(
        &plan_owner, requests, 3, &target, 1, &frequency, 1, &periodic, 1,
        1.0, 1.0, observed);
  }
  catch (...) {
    gpu::detail::destroy_resident_near2far_plan_for_owner(&plan_owner);
    gpu::detail::destroy_resident_cache_for_owner(&resident_owner);
    throw;
  }

  std::complex<double> reference[6] = {};
  const vec source_vec(source.x, source.y, source.z);
  const vec perturbed_source_vec(
      perturbed_source.x, perturbed_source.y, perturbed_source.z);
  const vec target_vec(target.x, target.y, target.z);
  const std::complex<double> amplitudes[3] = {
      std::complex<double>(x_positive[0], x_positive[1]),
      std::complex<double>(x_negative[0], x_negative[1]),
      std::complex<double>(y_control[0], y_control[1])};
  const component components[3] = {Ex, Ex, Ey};
  const vec operation_sources[3] = {
      source_vec, perturbed_source_vec, source_vec};
  for (int operation = 0; operation < 3; ++operation) {
    std::complex<double> contribution[6];
    green3d(contribution, target_vec, frequency, 1.0, 1.0,
            operation_sources[operation],
            components[operation], amplitudes[operation]);
    for (int component = 0; component < 6; ++component)
      reference[component] += contribution[component];
  }
  double maximum_error = 0.0;
  double maximum_reference = 0.0;
  for (int component = 0; component < 6; ++component) {
    maximum_error = std::max(
        maximum_error, std::abs(observed[component] - reference[component]));
    maximum_reference =
        std::max(maximum_reference, std::abs(reference[component]));
  }
  const gpu::near2far_statistics statistics =
      gpu::get_near2far_statistics();
  require(std::abs(reference[0]) < 1.0e-2 * maximum_reference &&
              maximum_reference > 0.0 &&
              maximum_error <= 5.0e-7 * maximum_reference + 1.0e-18 &&
              statistics.cpu_transform_calls == 0 &&
              statistics.cuda_transform_calls == 1 &&
              statistics.cuda_fast_precision_calls == 1 &&
              statistics.cuda_mixed_precision_calls == 1 &&
              statistics.cuda_cancellation_retries == 1 &&
              statistics.cuda_target_tiles == 2 &&
              statistics.cuda_frequency_tiles == 2 &&
              statistics.cuda_operation_tiles == 2 &&
              statistics.cuda_kernel_launches == 4 &&
              statistics.cuda_result_device_to_host_bytes ==
                  2 * 12 * sizeof(double) &&
              statistics.cuda_condition_device_to_host_bytes ==
                  sizeof(double),
          "bounded one-channel destructive cancellation did not rerun the "
          "public transform on mixed CUDA with CPU-double agreement: "
          "max-error=" + std::to_string(maximum_error) +
              " max-reference=" + std::to_string(maximum_reference) +
              " relative=" +
              std::to_string(maximum_error /
                             std::max(maximum_reference, 1.0e-300)) +
              " canceled=" + std::to_string(std::abs(reference[0])) +
              " calls=" +
              std::to_string(statistics.cuda_transform_calls) +
              " fast=" +
              std::to_string(statistics.cuda_fast_precision_calls) +
              " mixed=" +
              std::to_string(statistics.cuda_mixed_precision_calls) +
              " retries=" +
              std::to_string(statistics.cuda_cancellation_retries) +
              " target=" +
              std::to_string(statistics.cuda_target_tiles) +
              " frequency=" +
              std::to_string(statistics.cuda_frequency_tiles) +
              " operation=" +
              std::to_string(statistics.cuda_operation_tiles) +
              " launches=" +
              std::to_string(statistics.cuda_kernel_launches) +
              " result-bytes=" +
              std::to_string(
                  statistics.cuda_result_device_to_host_bytes) +
              " condition-bytes=" +
              std::to_string(
                  statistics.cuda_condition_device_to_host_bytes));
  gpu::detail::destroy_resident_near2far_plan_for_owner(&plan_owner);
  gpu::detail::destroy_resident_cache_for_owner(&resident_owner);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "Near2Far cancellation retry leaked resident allocations");

  // Cross the 64*256 partial coverage limit so every thread has multiple
  // serial interactions. The host gamma must grow with this depth and still
  // select mixed CUDA for a long alternating cancellation sequence.
  constexpr std::size_t large_point_count = 16386;
  std::vector<gpu::detail::near2far_point_fp64> large_points(
      large_point_count, source);
  std::vector<float> large_dft(2 * large_point_count, 0.0f);
  double summed_amplitude = 0.0;
  for (std::size_t point = 0; point < large_point_count; ++point) {
    large_dft[2 * point] =
        point % 2 == 0 ? 1.0f : -std::nextafter(1.0f, 0.0f);
    summed_amplitude += static_cast<double>(large_dft[2 * point]);
  }
  int large_resident_owner = 0;
  int large_plan_owner = 0;
  const gpu::detail::near2far_request_fp32 large_request = {
      &large_resident_owner, large_dft.data(), large_points.data(),
      large_point_count, 0, true};
  std::complex<double> large_observed[6] = {};
  gpu::reset_dispatch_statistics();
  gpu::detail::resident_near2far_3d_fp32(
      &large_plan_owner, &large_request, 1, &target, 1, &frequency, 1,
      &periodic, 1, 1.0, 1.0, large_observed);
  std::complex<double> large_reference[6];
  green3d(large_reference, target_vec, frequency, 1.0, 1.0, source_vec,
          Ex, std::complex<double>(summed_amplitude, 0.0));
  double large_reference_maximum = 0.0;
  double large_error_maximum = 0.0;
  for (int component = 0; component < 6; ++component) {
    large_reference_maximum = std::max(
        large_reference_maximum, std::abs(large_reference[component]));
    large_error_maximum = std::max(
        large_error_maximum,
        std::abs(large_observed[component] - large_reference[component]));
  }
  const gpu::near2far_statistics large_statistics =
      gpu::get_near2far_statistics();
  require(large_reference_maximum > 0.0 &&
              large_error_maximum <=
                  5.0e-7 * large_reference_maximum + 1.0e-18 &&
              large_statistics.cuda_transform_calls == 1 &&
              large_statistics.cuda_fast_precision_calls == 1 &&
              large_statistics.cuda_mixed_precision_calls == 1 &&
              large_statistics.cuda_cancellation_retries == 1 &&
              large_statistics.cuda_kernel_launches == 4,
          "large-point serial FP32 cancellation did not trigger one mixed "
          "CUDA retry with CPU-double agreement");
  gpu::detail::destroy_resident_near2far_plan_for_owner(&large_plan_owner);
  gpu::detail::destroy_resident_cache_for_owner(&large_resident_owner);
  require(gpu::get_live_resident_device_buffers() == live_before,
          "large-point Near2Far cancellation retry leaked allocations");
#endif
}

void require_near2far_radial_quantization_selector() {
#if MEEP_SINGLE && MEEP_HAVE_CUDA
  const gpu::detail::near2far_point_fp64 source = {0.0, 0.0, 0.0};
  const gpu::detail::near2far_periodic_copy_fp64 periodic = {
      {0.0, 0.0, 0.0}, 1.0, 0.0};
  float dft[2] = {0.73f, -0.19f};
  int resident_owner = 0;
  const gpu::detail::near2far_request_fp32 request = {
      &resident_owner, dft, &source, 1, 0, true};
  {
    int plan_owner = 0;
    const gpu::detail::near2far_point_fp64 target = {2.0, 0.0, 0.0};
    const double frequency = 0.31;
    std::complex<double> observed[6] = {};
    double fast_bounds[1] = {
        std::numeric_limits<double>::quiet_NaN()};
    bool used_mixed = false;
    gpu::detail::resident_near2far_3d_fp32(
        &plan_owner, &request, 1, &target, 1, &frequency, 1, &periodic, 1,
        1.0, 1.0, observed, fast_bounds, true, true, false, &used_mixed);
    require(used_mixed && fast_bounds[0] == 0.0,
            "direct mixed Near2Far did not deterministically zero its fast "
            "condition evidence");
    gpu::detail::destroy_resident_near2far_plan_for_owner(&plan_owner);
  }
  const auto require_mixed_oracle = [&](double frequency, double distance,
                                        const std::string &label) {
    int plan_owner = 0;
    const gpu::detail::near2far_point_fp64 target = {distance, 0.0, 0.0};
    std::complex<double> observed[6] = {};
    gpu::reset_dispatch_statistics();
    gpu::detail::resident_near2far_3d_fp32(
        &plan_owner, &request, 1, &target, 1, &frequency, 1, &periodic, 1,
        1.0, 1.0, observed);
    const gpu::near2far_statistics statistics =
        gpu::get_near2far_statistics();
    std::complex<double> reference[6] = {};
    green3d(reference, vec(distance, 0.0, 0.0), frequency, 1.0, 1.0,
            vec(0.0, 0.0, 0.0), Ex,
            std::complex<double>(dft[0], dft[1]));
    double reference_maximum = 0.0;
    double error_maximum = 0.0;
    for (int component = 0; component < 6; ++component) {
      require(std::isfinite(observed[component].real()) &&
                  std::isfinite(observed[component].imag()),
              label + " produced a non-finite CUDA field");
      reference_maximum =
          std::max(reference_maximum, std::abs(reference[component]));
      error_maximum = std::max(
          error_maximum, std::abs(observed[component] - reference[component]));
    }
    require(reference_maximum > 0.0 &&
                error_maximum <= 5.0e-7 * reference_maximum + 1.0e-18 &&
                statistics.cuda_transform_calls == 1 &&
                statistics.cuda_fast_precision_calls == 0 &&
                statistics.cuda_mixed_precision_calls == 1 &&
                statistics.cuda_cancellation_retries == 0,
            label + " admitted an unsafe FP32 radial Green transform");
    gpu::detail::destroy_resident_near2far_plan_for_owner(&plan_owner);
  };

  // The first frequency is finite as a double but below the normal FP32 range;
  // the second pairs a very large distance with small k, and the third creates
  // a tiny kr whose 1/kr^2 coefficient magnifies radial quantization. Every
  // case must go directly to mixed CUDA and agree with the FP64 CPU Green
  // oracle, proving the fast selector has no false negative at these limits.
  require_mixed_oracle(2.0e-39, 1.0,
                       "subnormal-frequency radial CUDA Near2Far selector");
  require_mixed_oracle(1.0e-20, 1.0e20,
                       "large-distance radial CUDA Near2Far selector");
  require_mixed_oracle(1.0e-12, 1.0e-3,
                       "small-kr reciprocal-square CUDA Near2Far selector");

  // This quantitative audit geometry has an exact polar coefficient of
  // 1.25*2^-149: nonzero and subnormal in FP32. A device which preserves
  // subnormals loses roughly 20% in the longitudinal field, while FTZ can
  // erase the coefficient. The current host selector's recently added k/r
  // family check rejects it before launch, so the public contract must go
  // directly to mixed CUDA and agree with the CPU oracle. The low-level CUDA
  // smoke test separately proves that the device evidence is +infinity.
  {
    constexpr double frequency = 8.5136750370755e-33;
    constexpr double distance = 2.4302246106229e12;
    float unit_dft[2] = {1.0f, 0.0f};
    const gpu::detail::near2far_request_fp32 unit_request = {
        &resident_owner, unit_dft, &source, 1, 0, true};
    const gpu::detail::near2far_point_fp64 target = {
        distance, 0.0, 0.0};
    int plan_owner = 0;
    std::complex<double> observed[6] = {};
    gpu::reset_dispatch_statistics();
    gpu::detail::resident_near2far_3d_fp32(
        &plan_owner, &unit_request, 1, &target, 1, &frequency, 1,
        &periodic, 1, 1.0, 1.0, observed);
    std::complex<double> reference[6] = {};
    green3d(reference, vec(distance, 0.0, 0.0), frequency, 1.0, 1.0,
            vec(0.0, 0.0, 0.0), Ex,
            std::complex<double>(1.0, 0.0));
    const gpu::near2far_statistics statistics =
        gpu::get_near2far_statistics();
    double reference_maximum = 0.0;
    double error_maximum = 0.0;
    for (int component = 0; component < 6; ++component) {
      reference_maximum =
          std::max(reference_maximum, std::abs(reference[component]));
      error_maximum = std::max(
          error_maximum, std::abs(observed[component] - reference[component]));
    }
    require(std::abs(reference[0].imag()) > 2.0e-7 &&
                std::abs(reference[0].imag()) < 2.2e-7 &&
                error_maximum <=
                    5.0e-7 * reference_maximum + 1.0e-18 &&
                statistics.cuda_transform_calls == 1 &&
                statistics.cuda_fast_precision_calls == 0 &&
                statistics.cuda_mixed_precision_calls == 1 &&
                statistics.cuda_cancellation_retries == 0,
            "subnormal polar Green coefficient was not rejected by the "
            "radial selector for direct mixed CUDA: reference-imag=" +
                std::to_string(reference[0].imag()) + " max-error=" +
                std::to_string(error_maximum) + " fast=" +
                std::to_string(statistics.cuda_fast_precision_calls) +
                " mixed=" +
                std::to_string(statistics.cuda_mixed_precision_calls) +
                " retries=" +
                std::to_string(statistics.cuda_cancellation_retries));
    gpu::detail::destroy_resident_near2far_plan_for_owner(&plan_owner);
  }

  // A second construction makes k/r exactly normal-scale while the extra
  // 1/(4*pi) polar factor is subnormal. It naturally passes the radial host
  // selector, reaches the fast kernel, and must be caught by the device's
  // finite-normal evidence before a single mixed-CUDA retry.
  {
    const double frequency =
        std::ldexp(1.0, -93) / (2.0 * pi);
    const double distance = std::ldexp(1.0, 31);
    float unit_dft[2] = {1.0f, 0.0f};
    const gpu::detail::near2far_request_fp32 unit_request = {
        &resident_owner, unit_dft, &source, 1, 0, true};
    const gpu::detail::near2far_point_fp64 target = {
        distance, 0.0, 0.0};
    int plan_owner = 0;
    std::complex<double> observed[6] = {};
    gpu::reset_dispatch_statistics();
    gpu::detail::resident_near2far_3d_fp32(
        &plan_owner, &unit_request, 1, &target, 1, &frequency, 1,
        &periodic, 1, 1.0, 1.0, observed);
    std::complex<double> reference[6] = {};
    green3d(reference, vec(distance, 0.0, 0.0), frequency, 1.0, 1.0,
            vec(0.0, 0.0, 0.0), Ex,
            std::complex<double>(1.0, 0.0));
    const gpu::near2far_statistics statistics =
        gpu::get_near2far_statistics();
    double reference_maximum = 0.0;
    double error_maximum = 0.0;
    for (int component = 0; component < 6; ++component) {
      reference_maximum =
          std::max(reference_maximum, std::abs(reference[component]));
      error_maximum = std::max(
          error_maximum, std::abs(observed[component] - reference[component]));
    }
    require(reference_maximum > 0.0 &&
                error_maximum <=
                    5.0e-7 * reference_maximum + 1.0e-18 &&
                statistics.cuda_transform_calls == 1 &&
                statistics.cuda_fast_precision_calls == 1 &&
                statistics.cuda_mixed_precision_calls == 1 &&
                statistics.cuda_cancellation_retries == 1,
            "normal k/r with a subnormal polar coefficient escaped device "
            "finite-normal evidence: max-error=" +
                std::to_string(error_maximum) + " fast=" +
                std::to_string(statistics.cuda_fast_precision_calls) +
                " mixed=" +
                std::to_string(statistics.cuda_mixed_precision_calls) +
                " retries=" +
                std::to_string(statistics.cuda_cancellation_retries));
    gpu::detail::destroy_resident_near2far_plan_for_owner(&plan_owner);
  }

  // Ordinary normal-scale arithmetic must retain the high-throughput path;
  // the finite-normal checks are evidence only, not a blanket mixed selector.
  {
    constexpr double frequency = 0.31;
    const gpu::detail::near2far_point_fp64 target = {0.0, 2.0, 0.0};
    int plan_owner = 0;
    std::complex<double> observed[6] = {};
    gpu::reset_dispatch_statistics();
    gpu::detail::resident_near2far_3d_fp32(
        &plan_owner, &request, 1, &target, 1, &frequency, 1, &periodic, 1,
        1.0, 1.0, observed);
    const gpu::near2far_statistics statistics =
        gpu::get_near2far_statistics();
    require(statistics.cuda_transform_calls == 1 &&
                statistics.cuda_fast_precision_calls == 1 &&
                statistics.cuda_mixed_precision_calls == 0 &&
                statistics.cuda_cancellation_retries == 0,
            "normal-scale Near2Far fixture did not remain on fast CUDA");
    gpu::detail::destroy_resident_near2far_plan_for_owner(&plan_owner);
  }

  const auto require_amplitude_evidence_retry = [
      &source, &periodic](
      float amplitude, double reference_amplitude,
      const std::string &label) {
    constexpr double frequency = 0.31;
    const gpu::detail::near2far_point_fp64 target = {0.1, 0.0, 0.0};
    float special_dft[2] = {amplitude, 0.0f};
    int special_resident_owner = 0;
    const gpu::detail::near2far_request_fp32 special_request = {
        &special_resident_owner, special_dft, &source, 1, 0, true};
    int plan_owner = 0;
    std::complex<double> observed[6] = {};
    gpu::reset_dispatch_statistics();
    gpu::detail::resident_near2far_3d_fp32(
        &plan_owner, &special_request, 1, &target, 1, &frequency, 1,
        &periodic, 1, 1.0, 1.0, observed);
    std::complex<double> reference[6] = {};
    green3d(reference, vec(0.1, 0.0, 0.0), frequency, 1.0, 1.0,
            vec(0.0, 0.0, 0.0), Ex,
            std::complex<double>(reference_amplitude, 0.0));
    const gpu::near2far_statistics statistics =
        gpu::get_near2far_statistics();
    double reference_maximum = 0.0;
    double error_maximum = 0.0;
    for (int component = 0; component < 6; ++component) {
      require(std::isfinite(observed[component].real()) &&
                  std::isfinite(observed[component].imag()),
              label + " mixed retry produced a non-finite field");
      reference_maximum =
          std::max(reference_maximum, std::abs(reference[component]));
      error_maximum = std::max(
          error_maximum, std::abs(observed[component] - reference[component]));
    }
    require(reference_maximum > 0.0 &&
                error_maximum <=
                    5.0e-7 * reference_maximum +
                        std::numeric_limits<double>::denorm_min() &&
                statistics.cuda_transform_calls == 1 &&
                statistics.cuda_fast_precision_calls == 1 &&
                statistics.cuda_mixed_precision_calls == 1 &&
                statistics.cuda_cancellation_retries == 1,
            label + " did not discard fast CUDA and retry once on mixed "
                    "CUDA with CPU-oracle agreement: max-error=" +
                canonical_decimal(error_maximum) + " reference=" +
                canonical_decimal(reference_maximum) + " fast=" +
                std::to_string(statistics.cuda_fast_precision_calls) +
                " mixed=" +
                std::to_string(statistics.cuda_mixed_precision_calls) +
                " retries=" +
                std::to_string(statistics.cuda_cancellation_retries));
    gpu::detail::destroy_resident_near2far_plan_for_owner(&plan_owner);
    gpu::detail::destroy_resident_cache_for_owner(&special_resident_owner);
  };

  // Resident DFT amplitudes are intentionally absent from the geometry-only
  // host selector. Device evidence must catch both FTZ-prone input and fast
  // signed-field overflow, then retain the FP32 payload while mixed CUDA does
  // all Green/accumulation arithmetic in FP64.
  require_amplitude_evidence_retry(
      std::numeric_limits<float>::denorm_min(),
      std::ldexp(1.0, -149),
      "subnormal DFT-amplitude Near2Far evidence");
  require_amplitude_evidence_retry(
      std::numeric_limits<float>::max(),
      static_cast<double>(std::numeric_limits<float>::max()),
      "FLT_MAX DFT-amplitude Near2Far evidence");

  // A longitudinal dipole at large-but-fast-eligible kr cancels the O(1)
  // real parts of term1 and term2 internally, leaving O(1/kr). The evidence
  // must measure those pre-cancellation Green terms and force one mixed retry.
  {
    constexpr double frequency = 0.31;
    constexpr double adversarial_kr = 64.0;
    const double distance = adversarial_kr / (2.0 * pi * frequency);
    const gpu::detail::near2far_point_fp64 target = {distance, 0.0, 0.0};
    int plan_owner = 0;
    std::complex<double> observed[6] = {};
    gpu::reset_dispatch_statistics();
    gpu::detail::resident_near2far_3d_fp32(
        &plan_owner, &request, 1, &target, 1, &frequency, 1, &periodic, 1,
        1.0, 1.0, observed);
    std::complex<double> reference[6] = {};
    green3d(reference, vec(distance, 0.0, 0.0), frequency, 1.0, 1.0,
            vec(0.0, 0.0, 0.0), Ex,
            std::complex<double>(dft[0], dft[1]));
    const gpu::near2far_statistics statistics =
        gpu::get_near2far_statistics();
    double reference_maximum = 0.0;
    double error_maximum = 0.0;
    for (int component = 0; component < 6; ++component) {
      reference_maximum =
          std::max(reference_maximum, std::abs(reference[component]));
      error_maximum = std::max(
          error_maximum, std::abs(observed[component] - reference[component]));
    }
    require(reference_maximum > 0.0 &&
                error_maximum <= 5.0e-7 * reference_maximum + 1.0e-18 &&
                statistics.cuda_transform_calls == 1 &&
                statistics.cuda_fast_precision_calls == 1 &&
                statistics.cuda_mixed_precision_calls == 1 &&
                statistics.cuda_cancellation_retries == 1,
            "longitudinal large-kr Green internal cancellation did not use "
            "the pre-assembly envelope for one mixed CUDA retry: error=" +
                std::to_string(error_maximum) + " reference=" +
                std::to_string(reference_maximum) + " fast=" +
                std::to_string(
                    statistics.cuda_fast_precision_calls) +
                " mixed=" +
                std::to_string(
                    statistics.cuda_mixed_precision_calls) +
                " retries=" +
                std::to_string(
                    statistics.cuda_cancellation_retries));
    gpu::detail::destroy_resident_near2far_plan_for_owner(&plan_owner);
  }
  gpu::detail::destroy_resident_cache_for_owner(&resident_owner);
#endif
}

void require_near2far3d_same_snapshot_equivalence() {
  require_near2far_channel_cancellation_retry();
  require_near2far_radial_quantization_selector();
  gpu::set_backend(gpu::backend_mode::cuda);
  gpu::reset_dispatch_statistics();
  const grid_volume gv = vol3d(2.0, 1.8, 1.6, 8.0);
  structure s(gv, vacuum, no_pml());
  fields f(&s, 0.0, 0.0, true, 64, 64);
  const vec center = gv.center();
  gaussian_src_time source(0.31, 0.12);
  f.add_point_source(Ez, source, center, 0.75);
  f.add_point_source(Hx, source, center + vec(0.08, -0.05, 0.03),
                     std::complex<double>(-0.21, 0.17));

  const double hx = 0.46;
  const double hy = 0.38;
  const double hz = 0.30;
  volume_list near_region(
      volume(vec(center.x() + hx, center.y() - hy, center.z() - hz),
             vec(center.x() + hx, center.y() + hy, center.z() + hz)),
      Sx, 1.0,
      new volume_list(
          volume(vec(center.x() - hx, center.y() - hy, center.z() - hz),
                 vec(center.x() - hx, center.y() + hy, center.z() + hz)),
          Sx, -1.0,
          new volume_list(
              volume(vec(center.x() - hx, center.y() + hy,
                         center.z() - hz),
                     vec(center.x() + hx, center.y() + hy,
                         center.z() + hz)),
              Sy, 1.0,
              new volume_list(
                  volume(vec(center.x() - hx, center.y() - hy,
                             center.z() - hz),
                         vec(center.x() + hx, center.y() - hy,
                             center.z() + hz)),
                  Sy, -1.0,
                  new volume_list(
                      volume(vec(center.x() - hx, center.y() - hy,
                                 center.z() + hz),
                             vec(center.x() + hx, center.y() + hy,
                                 center.z() + hz)),
                      Sz, 1.0,
                      new volume_list(
                          volume(vec(center.x() - hx, center.y() - hy,
                                     center.z() - hz),
                                 vec(center.x() + hx, center.y() + hy,
                                     center.z() - hz)),
                          Sz, -1.0))))));
  const double frequencies[] = {0.23, 0.31, 0.41};
  dft_near2far near =
      f.add_dft_near2far(&near_region, frequencies, 3);
  for (int step = 0; step < 72; ++step) f.step();

  size_t chunk_count = 0;
  size_t source_points = 0;
  for (dft_chunk *chunk = near.F; chunk; chunk = chunk->next_in_dft)
    if (chunk->N) {
      ++chunk_count;
      source_points += chunk->N;
    }
  const size_t periodic_copies =
      (2 * static_cast<size_t>(near.periodic_n[0]) + 1) *
      (2 * static_cast<size_t>(near.periodic_n[1]) + 1);
  const volume targets(
      vec(center.x() + 1.4, center.y() - 0.35, center.z() - 0.25),
      vec(center.x() + 1.9, center.y() + 0.35, center.z() + 0.25));
  const size_t expected_dims[3] = {2, 3, 2};
  std::vector<vec> grid_points;
  grid_points.reserve(12);
  for (size_t i0 = 0; i0 < expected_dims[0]; ++i0)
    for (size_t i1 = 0; i1 < expected_dims[1]; ++i1)
      for (size_t i2 = 0; i2 < expected_dims[2]; ++i2)
        grid_points.push_back(vec(
            targets.in_direction_min(X) +
                i0 * targets.in_direction(X) / (expected_dims[0] - 1),
            targets.in_direction_min(Y) +
                i1 * targets.in_direction(Y) / (expected_dims[1] - 1),
            targets.in_direction_min(Z) +
                i2 * targets.in_direction(Z) / (expected_dims[2] - 1)));

  // CUDA is deliberately captured first so the avoided-readback counter
  // proves that the transform consumes the device-authoritative DFT snapshot.
  const near2far3d_snapshot_capture cuda = capture_near2far3d_snapshot(
      near, targets, grid_points, gpu::backend_mode::cuda);

  // Exercise retained-plan invalidation from every public mutation family on
  // this exact device-authoritative snapshot. Scaling must release stale DFT
  // mirrors and rebuild descriptors, while preserving linearity.
  near.scale_dfts(2.0);
  const near2far3d_snapshot_capture scaled_cuda =
      capture_near2far3d_snapshot(
          near, targets, grid_points, gpu::backend_mode::cuda);
  std::vector<std::complex<double> > doubled_batch = cuda.batch;
  std::vector<std::complex<double> > doubled_scalar = cuda.scalar;
  for (std::complex<double> &value : doubled_batch) value *= 2.0;
  for (std::complex<double> &value : doubled_scalar) value *= 2.0;
  require_near2far_complex_agreement(
      doubled_batch, scaled_cuda.batch,
      "scaled live-plan CUDA Near2Far batch");
  require_near2far_complex_agreement(
      doubled_scalar, scaled_cuda.scalar,
      "scaled live-plan CUDA Near2Far scalar");
  double scaled_grid_reference = 0.0;
  double scaled_grid_error = 0.0;
  require(scaled_cuda.grid.size() == cuda.grid.size(),
          "scaled live-plan CUDA Near2Far grid shape differs");
  for (size_t index = 0; index < cuda.grid.size(); ++index) {
    const double expected = 2.0 * cuda.grid[index];
    scaled_grid_reference =
        std::max(scaled_grid_reference, std::abs(expected));
    scaled_grid_error = std::max(
        scaled_grid_error, std::abs(scaled_cuda.grid[index] - expected));
  }
  require(scaled_grid_error <=
              3.0e-3 * scaled_grid_reference + 3.0e-8 &&
              scaled_cuda.grid_statistics.cuda_descriptor_uploads ==
                  1 + scaled_cuda.grid_statistics.cuda_cancellation_retries,
          "scale_dfts did not invalidate and accurately rebuild a live CUDA "
          "Near2Far plan");

  near.scale_dfts(0.5);
  const near2far3d_snapshot_capture restored_after_scale =
      capture_near2far3d_snapshot(
          near, targets, grid_points, gpu::backend_mode::cuda);
  require_near2far_complex_agreement(
      cuda.batch, restored_after_scale.batch,
      "restored scaled CUDA Near2Far batch");
  require(restored_after_scale.grid_statistics.cuda_descriptor_uploads ==
              1 + restored_after_scale.grid_statistics
                      .cuda_cancellation_retries,
          "restoring scale did not rebuild the CUDA Near2Far plan");

  // Save while the retained CUDA plan is live, perturb the DFT state, then
  // load the authoritative snapshot and require a fresh descriptor plan.
  const char *lifecycle_file = "gpu-step-db-near2far-lifecycle";
  near.save_hdf5(f, lifecycle_file, "lifecycle");
  near.scale_dfts(std::complex<double>(-0.375, 0.125));
  near.load_hdf5(f, lifecycle_file, "lifecycle");
  const near2far3d_snapshot_capture loaded_cuda =
      capture_near2far3d_snapshot(
          near, targets, grid_points, gpu::backend_mode::cuda);
  require_near2far_complex_agreement(
      cuda.batch, loaded_cuda.batch,
      "HDF5-loaded live-plan CUDA Near2Far batch");
  require_near2far_complex_agreement(
      cuda.scalar, loaded_cuda.scalar,
      "HDF5-loaded live-plan CUDA Near2Far scalar");
  require(loaded_cuda.grid_statistics.cuda_descriptor_uploads ==
              1 + loaded_cuda.grid_statistics.cuda_cancellation_retries,
          "load_hdf5 did not rebuild the CUDA Near2Far plan");

  // A second compatible monitor loaded from the same file supplies an exact
  // subtraction operand. Both plans are made live before operator-=; the
  // destination must become zero and rebuild rather than replaying either
  // stale descriptor set.
  dft_near2far subtraction =
      f.add_dft_near2far(&near_region, frequencies, 3);
  subtraction.load_hdf5(f, lifecycle_file, "lifecycle");
  const near2far3d_snapshot_capture subtraction_cuda =
      capture_near2far3d_snapshot(
          subtraction, targets, grid_points, gpu::backend_mode::cuda);
  require_near2far_complex_agreement(
      cuda.batch, subtraction_cuda.batch,
      "loaded subtraction operand CUDA Near2Far batch");
  near -= subtraction;
  const near2far3d_snapshot_capture zero_cuda =
      capture_near2far3d_snapshot(
          near, targets, grid_points, gpu::backend_mode::cuda);
  double zero_maximum = 0.0;
  for (double value : zero_cuda.grid)
    zero_maximum = std::max(zero_maximum, std::abs(value));
  for (const std::complex<double> &value : zero_cuda.batch)
    zero_maximum = std::max(zero_maximum, std::abs(value));
  require(zero_maximum <= 1.0e-20 &&
              zero_cuda.grid_statistics.cuda_descriptor_uploads ==
                  1 + zero_cuda.grid_statistics.cuda_cancellation_retries,
          "operator-= replayed stale CUDA Near2Far descriptors or failed to "
          "produce zero");
  near.load_hdf5(f, lifecycle_file, "lifecycle");
  const near2far3d_snapshot_capture restored_after_subtract =
      capture_near2far3d_snapshot(
          near, targets, grid_points, gpu::backend_mode::cuda);
  require_near2far_complex_agreement(
      cuda.batch, restored_after_subtract.batch,
      "post-subtraction restored CUDA Near2Far batch");
  require(restored_after_subtract.grid_statistics.cuda_descriptor_uploads ==
              1 + restored_after_subtract.grid_statistics
                      .cuda_cancellation_retries,
          "post-subtraction load did not rebuild the CUDA Near2Far plan");

  // Re-establish the operand plan immediately before remove so the live
  // buffer gauge proves that remove itself releases all four retained
  // allocations, independent of a backend switch.
  const near2far3d_snapshot_capture subtraction_before_remove =
      capture_near2far3d_snapshot(
          subtraction, targets, grid_points, gpu::backend_mode::cuda);
  require(subtraction_before_remove.grid_statistics.cuda_transform_calls == 1,
          "subtraction operand did not establish a CUDA Near2Far plan");
  const std::uint64_t live_before_subtraction_remove =
      gpu::get_live_resident_device_buffers();
  subtraction.remove();
  const std::uint64_t live_after_subtraction_remove =
      gpu::get_live_resident_device_buffers();
  require(live_before_subtraction_remove >=
                  live_after_subtraction_remove + 4,
          "live CUDA Near2Far remove did not release its retained plan and "
          "scratch buffers");
  h5file *lifecycle_h5 =
      f.open_h5file(lifecycle_file, h5file::READONLY);
  lifecycle_h5->remove();
  delete lifecycle_h5;

  const near2far3d_snapshot_capture cpu = capture_near2far3d_snapshot(
      near, targets, grid_points, gpu::backend_mode::cpu);
  require(cuda.live_after_grid == cuda.live_before_grid + 4 &&
              cuda.live_after_batch == cuda.live_after_grid &&
              cuda.live_after_scalar == cuda.live_after_grid &&
              cpu.live_before_grid <= cuda.live_before_grid &&
              cpu.live_before_grid <= cuda.live_after_scalar - 4,
          "Near2Far retained plan/scratch buffers were not reused or "
          "released during the CUDA-to-CPU lifecycle transition: before=" +
              std::to_string(cuda.live_before_grid) + " grid=" +
              std::to_string(cuda.live_after_grid) + " batch=" +
              std::to_string(cuda.live_after_batch) + " scalar=" +
              std::to_string(cuda.live_after_scalar) + " cpu-before=" +
              std::to_string(cpu.live_before_grid));
  require(cuda.rank == 3 && cuda.dims[0] == expected_dims[0] &&
              cuda.dims[1] == expected_dims[1] &&
              cuda.dims[2] == expected_dims[2] &&
              cpu.rank == cuda.rank && cpu.dims[0] == cuda.dims[0] &&
              cpu.dims[1] == cuda.dims[1] &&
              cpu.dims[2] == cuda.dims[2],
          "3D near-to-far rank, dimensions, or singleton collapse changed");

  double global_reference = 0.0;
  for (double value : cpu.grid)
    global_reference = std::max(global_reference, std::abs(value));
  require(global_reference > 0.0,
          "same-snapshot CPU near-to-far reference is identically zero");
  for (int scalar_component = 0; scalar_component < 12;
       ++scalar_component)
    for (size_t frequency = 0; frequency < near.freq.size(); ++frequency) {
      double reference_squared = 0.0;
      double error_squared = 0.0;
      double reference_maximum = 0.0;
      double error_maximum = 0.0;
      for (size_t point = 0; point < grid_points.size(); ++point) {
        const size_t index =
            (static_cast<size_t>(scalar_component) * grid_points.size() +
             point) *
                near.freq.size() +
            frequency;
        const double error = std::abs(cuda.grid[index] - cpu.grid[index]);
        reference_squared += cpu.grid[index] * cpu.grid[index];
        error_squared += error * error;
        reference_maximum =
            std::max(reference_maximum, std::abs(cpu.grid[index]));
        error_maximum = std::max(error_maximum, error);
      }
      const double rms_reference = std::sqrt(
          reference_squared / static_cast<double>(grid_points.size()));
      const double rms_error = std::sqrt(
          error_squared / static_cast<double>(grid_points.size()));
      require(rms_error <=
                      5.0e-4 * rms_reference + 5.0e-6 * global_reference &&
                  error_maximum <= 3.0e-3 * reference_maximum +
                                       3.0e-5 * global_reference,
              "same-snapshot CUDA near-to-far component/frequency mismatch: "
              "component=" +
                  std::to_string(scalar_component) + " frequency=" +
                  std::to_string(frequency));
    }
  require_near2far_complex_agreement(
      cpu.batch, cuda.batch, "same-snapshot batched CUDA near-to-far");
  for (size_t point = 0; point < grid_points.size(); ++point)
    for (size_t frequency = 0; frequency < near.freq.size(); ++frequency)
      for (int component = 0; component < 6; ++component) {
        const std::complex<double> batch_value =
            cuda.batch[6 * (point * near.freq.size() + frequency) +
                       component];
        const std::complex<double> grid_value(
            cuda.grid[((2 * component + 0) * grid_points.size() + point) *
                          near.freq.size() +
                      frequency],
            cuda.grid[((2 * component + 1) * grid_points.size() + point) *
                          near.freq.size() +
                      frequency]);
        require(std::abs(batch_value - grid_value) <=
                    1.0e-12 * std::max(1.0, std::abs(grid_value)),
                "CUDA near-to-far grid/batch memory order differs");
      }
  std::vector<std::complex<double> > first_batch(
      cuda.batch.begin(), cuda.batch.begin() + 6 * near.freq.size());
  require_near2far_complex_agreement(
      first_batch, cuda.scalar,
      "CUDA scalar and batched near-to-far APIs");

  const std::uint64_t cpu_expected_terms =
      static_cast<std::uint64_t>(source_points) * periodic_copies *
      grid_points.size() * near.freq.size();
  require(cpu.grid_statistics.cpu_transform_calls == grid_points.size() &&
              cpu.grid_statistics.cuda_transform_calls == 0 &&
              cpu.grid_statistics.cpu_terms == cpu_expected_terms &&
              cpu.batch_statistics.cpu_transform_calls == grid_points.size() &&
              cpu.batch_statistics.cuda_transform_calls == 0 &&
              cpu.batch_statistics.cpu_terms == cpu_expected_terms &&
              cpu.scalar_statistics.cpu_transform_calls == 1 &&
              cpu.scalar_statistics.cuda_transform_calls == 0 &&
              cpu.grid_statistics.mpi_allreduce_calls == 1 &&
              cpu.grid_statistics.mpi_allreduce_bytes ==
                  12 * grid_points.size() * near.freq.size() *
                      sizeof(double) &&
              cpu.batch_statistics.mpi_allreduce_calls == 1 &&
              cpu.batch_statistics.mpi_allreduce_bytes ==
                  6 * grid_points.size() * near.freq.size() *
                      sizeof(std::complex<double>) &&
              cpu.scalar_statistics.mpi_allreduce_calls == 1 &&
              cpu.scalar_statistics.mpi_allreduce_bytes ==
                  6 * near.freq.size() * sizeof(std::complex<double>),
          "same-snapshot CPU near-to-far counters are inconsistent");
  require_near2far_cuda_transform_statistics(
      cuda.grid_statistics, grid_points.size(), near.freq.size(), chunk_count,
      source_points, periodic_copies, true, true);
  require_near2far_cuda_transform_statistics(
      cuda.batch_statistics, grid_points.size(), near.freq.size(), chunk_count,
      source_points, periodic_copies, false, false);
  require_near2far_cuda_transform_statistics(
      cuda.scalar_statistics, 1, near.freq.size(), chunk_count, source_points,
      periodic_copies, false, false);
  const auto valid_ordinary_precision_path = [](
      const gpu::near2far_statistics &statistics) {
    if (statistics.cuda_fast_precision_calls == 1)
      return statistics.cuda_mixed_precision_calls ==
             statistics.cuda_cancellation_retries;
    return count_processors() > 1 &&
           statistics.cuda_fast_precision_calls == 0 &&
           statistics.cuda_mixed_precision_calls == 1 &&
           statistics.cuda_cancellation_retries == 0;
  };
  require(valid_ordinary_precision_path(cuda.grid_statistics) &&
              valid_ordinary_precision_path(cuda.batch_statistics) &&
              valid_ordinary_precision_path(cuda.scalar_statistics),
          "ordinary bounded 3D Near2Far did not execute its FP32 probe and "
          "truthfully account for a selector-driven or cancellation-driven "
          "mixed CUDA path");

  // Migrate this exact physical DFT snapshot primary -> next -> primary in the
  // singleton suite when two compatible devices exist. CUDA ordinals are
  // rank-local under scheduler visibility remapping, so a multi-rank process
  // cannot safely infer a collision-free physical migration from local
  // ordinals alone. Dedicated distributed lanes separately prove unique
  // physical UUID mapping. Each singleton destination here must rebuild
  // bounded resident state, agree with the CPU snapshot, and return to
  // identical primary-device bits without increasing the process-wide live
  // allocation gauge.
  std::vector<int> compatible_ordinals;
  for (const gpu::device_info &device : gpu::enumerate_devices())
    if (device.compatible) compatible_ordinals.push_back(device.ordinal);
  if (count_processors() == 1 && compatible_ordinals.size() >= 2) {
    // The CPU reference capture above leaves the backend inactive, so
    // selected_device() alone would report -1 even though the process still
    // retains its rank-local CUDA ordinal. Reactivate that mapping before the
    // deliberate cross-device migration and restore it afterwards; otherwise
    // every MPI rank is left explicitly pinned to ordinal zero and the next
    // distributed fields owner correctly rejects the duplicate assignment.
    gpu::set_backend(gpu::backend_mode::cuda);
    const int original_ordinal = gpu::selected_device();
    const auto original_position = std::find(
        compatible_ordinals.begin(), compatible_ordinals.end(),
        original_ordinal);
    require(original_position != compatible_ordinals.end(),
            "Near2Far migration regression lost its rank-local CUDA ordinal");
    const std::size_t original_index = static_cast<std::size_t>(
        original_position - compatible_ordinals.begin());
    const int migration_ordinal = compatible_ordinals[
        (original_index + 1) % compatible_ordinals.size()];
    const auto capture_on = [&](int ordinal) {
      gpu::set_backend(gpu::backend_mode::cuda);
      gpu::select_device(ordinal);
      return capture_near2far3d_snapshot(
          near, targets, grid_points, gpu::backend_mode::cuda, true);
    };
    // Normalize any unrelated full-field mirrors left by the FDTD run. The
    // measured sequence below then contains the same DFT mirrors and four
    // retained Near2Far plan buffers on each physical device.
    (void)capture_on(migration_ordinal);
    const near2far3d_snapshot_capture primary =
        capture_on(original_ordinal);
    const std::uint64_t migrated_live = primary.live_after_scalar;
    const near2far3d_snapshot_capture migrated =
        capture_on(migration_ordinal);
    require_near2far_complex_agreement(
        cpu.batch, primary.batch, "primary-GPU migrated Near2Far batch");
    require_near2far_complex_agreement(
        cpu.batch, migrated.batch, "secondary-GPU migrated Near2Far batch");
    require_near2far_complex_agreement(
        cpu.scalar, primary.scalar, "primary-GPU migrated Near2Far scalar");
    require_near2far_complex_agreement(
        cpu.scalar, migrated.scalar,
        "secondary-GPU migrated Near2Far scalar");
    require(migrated.live_after_scalar == migrated_live,
            "Near2Far cross-device migration changed live allocation count");
    const near2far3d_snapshot_capture returned_primary =
        capture_on(original_ordinal);
    require(returned_primary.grid == primary.grid &&
                returned_primary.batch == primary.batch &&
                returned_primary.scalar == primary.scalar &&
                returned_primary.live_after_scalar == migrated_live,
            "Near2Far return to the primary GPU changed bits or leaked "
            "resident allocations");
    gpu::select_device(original_ordinal);
  }

  // Force every workspace dimension to tile using the same production
  // transform and DFT snapshot. At this resolution each logical operation has
  // one partial; 512 bytes admits multiple operations but cannot admit all
  // monitor chunks or more than one target/frequency work item. This catches
  // nonzero operation/frequency offsets and cross-operation accumulation
  // without a large synthetic allocation. Re-establish the ordinary retained
  // plan immediately before lowering the ceiling so this also proves that
  // already-live oversized scratch is evicted rather than merely hidden by a
  // smaller planned-byte counter.
  constexpr std::size_t forced_workspace_ceiling = 512;
  const near2far3d_snapshot_capture retained_large_cuda =
      capture_near2far3d_snapshot(
          near, targets, grid_points, gpu::backend_mode::cuda);
  require(retained_large_cuda.grid_statistics
                  .cuda_maximum_workspace_bytes >
              forced_workspace_ceiling,
          "ordinary CUDA Near2Far did not establish scratch larger than the "
          "forced workspace ceiling");
  const std::size_t previous_workspace_ceiling =
      gpu::detail::exchange_near2far_workspace_ceiling_for_testing(
          forced_workspace_ceiling);
  near2far3d_snapshot_capture tiled_cuda;
  try {
    tiled_cuda = capture_near2far3d_snapshot(
        near, targets, grid_points, gpu::backend_mode::cuda);
  }
  catch (...) {
    gpu::detail::exchange_near2far_workspace_ceiling_for_testing(
        previous_workspace_ceiling);
    throw;
  }
  gpu::detail::exchange_near2far_workspace_ceiling_for_testing(
      previous_workspace_ceiling);
  require_near2far_complex_agreement(
      cpu.batch, tiled_cuda.batch,
      "forced operation/frequency/target tiled CUDA near-to-far");
  require_near2far_complex_agreement(
      cpu.scalar, tiled_cuda.scalar,
      "forced operation/frequency tiled scalar CUDA near-to-far");
  require(tiled_cuda.grid.size() == cpu.grid.size(),
          "forced tiled CUDA Near2Far grid shape differs");
  double tiled_grid_reference = 0.0;
  double tiled_grid_error = 0.0;
  for (size_t index = 0; index < cpu.grid.size(); ++index) {
    tiled_grid_reference =
        std::max(tiled_grid_reference, std::abs(cpu.grid[index]));
    tiled_grid_error = std::max(
        tiled_grid_error,
        std::abs(tiled_cuda.grid[index] - cpu.grid[index]));
  }
  require(tiled_grid_error <= 3.0e-3 * tiled_grid_reference + 3.0e-8,
          "forced tiled CUDA Near2Far grid differs from CPU");
  const auto require_forced_tiling = [&](
      const gpu::near2far_statistics &statistics,
      bool require_multiple_targets) {
    require(statistics.cuda_transform_calls == 1 &&
                statistics.cuda_fast_precision_calls == 1 &&
                statistics.cuda_mixed_precision_calls ==
                    statistics.cuda_cancellation_retries &&
                statistics.cuda_target_tiles >= 1 &&
                (!require_multiple_targets ||
                 statistics.cuda_target_tiles > 1) &&
                statistics.cuda_frequency_tiles >
                    statistics.cuda_target_tiles &&
                statistics.cuda_operation_tiles >
                    statistics.cuda_frequency_tiles &&
                statistics.cuda_maximum_workspace_bytes > 0 &&
                statistics.cuda_maximum_workspace_bytes <=
                    forced_workspace_ceiling &&
                ((statistics.cuda_cancellation_retries &&
                  statistics.cuda_descriptor_uploads >= 1 &&
                 statistics.cuda_descriptor_uploads <= 2) ||
                 (!statistics.cuda_cancellation_retries &&
                  statistics.cuda_descriptor_uploads <= 1)) &&
                statistics.cuda_kernel_launches ==
                    2 * statistics.cuda_operation_tiles &&
                statistics.cuda_result_device_to_host_bytes >
                    12 * statistics.cuda_output_points *
                        statistics.cuda_frequencies * sizeof(double) &&
                statistics.cuda_condition_device_to_host_bytes >
                    statistics.cuda_output_points *
                        statistics.cuda_frequencies * sizeof(double),
            "forced Near2Far retained workspace did not shrink below the "
            "ceiling or tile every required axis: fast=" +
                std::to_string(statistics.cuda_fast_precision_calls) +
                " mixed=" +
                std::to_string(statistics.cuda_mixed_precision_calls) +
                " retries=" +
                std::to_string(statistics.cuda_cancellation_retries) +
                " target=" +
                std::to_string(statistics.cuda_target_tiles) +
                " frequency=" +
                std::to_string(statistics.cuda_frequency_tiles) +
                " operation=" +
                std::to_string(statistics.cuda_operation_tiles) +
                " workspace=" +
                std::to_string(statistics.cuda_maximum_workspace_bytes) +
                " uploads=" +
                std::to_string(statistics.cuda_descriptor_uploads) +
                " result-bytes=" +
                std::to_string(
                    statistics.cuda_result_device_to_host_bytes) +
                " condition-bytes=" +
                std::to_string(
                    statistics.cuda_condition_device_to_host_bytes));
  };
  require(tiled_cuda.live_before_grid == retained_large_cuda.live_after_scalar &&
              tiled_cuda.live_after_grid == tiled_cuda.live_before_grid,
          "shrinking retained Near2Far scratch changed the live allocation "
          "count");
  require_forced_tiling(tiled_cuda.grid_statistics, true);
  require_forced_tiling(tiled_cuda.batch_statistics, true);
  require_forced_tiling(tiled_cuda.scalar_statistics, false);

  // Prove that the integrated backend automatically selects the mixed CUDA
  // path for a far observation point. At this distance distinct source
  // coordinates collapse if geometry and phase are reduced to FP32.
  const std::vector<vec> far_targets = {
      vec(center.x() + 1.0e8, center.y(), center.z()),
      vec(center.x() + 1.0e8, center.y() + 0.375, center.z() - 0.25)};
  gpu::set_backend(gpu::backend_mode::cuda);
  gpu::reset_dispatch_statistics();
  std::unique_ptr<std::complex<double>[]> raw_far_cuda(
      near.farfields(far_targets.data(), far_targets.size()));
  const std::vector<std::complex<double> > far_cuda(
      raw_far_cuda.get(),
      raw_far_cuda.get() +
          6 * far_targets.size() * near.freq.size());
  const gpu::near2far_statistics far_cuda_statistics =
      gpu::get_near2far_statistics();
  gpu::set_backend(gpu::backend_mode::cpu);
  gpu::reset_dispatch_statistics();
  std::unique_ptr<std::complex<double>[]> raw_far_cpu(
      near.farfields(far_targets.data(), far_targets.size()));
  const std::vector<std::complex<double> > far_cpu(
      raw_far_cpu.get(),
      raw_far_cpu.get() +
          6 * far_targets.size() * near.freq.size());
  require_near2far_complex_agreement(
      far_cpu, far_cuda,
      "automatically selected mixed-precision far-distance CUDA Near2Far");
  double far_reference = 0.0;
  for (const std::complex<double> &value : far_cpu)
    far_reference = std::max(far_reference, std::abs(value));
  require(far_reference > 0.0 &&
              far_cuda_statistics.cuda_transform_calls == 1 &&
              far_cuda_statistics.cuda_fast_precision_calls == 0 &&
              far_cuda_statistics.cuda_mixed_precision_calls == 1 &&
              far_cuda_statistics.cuda_cancellation_retries == 0 &&
              far_cuda_statistics.cuda_target_tiles == 1 &&
              far_cuda_statistics.cuda_frequency_tiles == 1 &&
              far_cuda_statistics.cuda_operation_tiles == 1 &&
              far_cuda_statistics.cuda_maximum_workspace_bytes > 0 &&
              far_cuda_statistics.cuda_maximum_workspace_bytes <=
                  64u * 1024u * 1024u &&
              far_cuda_statistics.cuda_kernel_launches == 2 &&
              far_cuda_statistics.cuda_result_device_to_host_bytes ==
                  12 * far_targets.size() * near.freq.size() *
                      sizeof(double) &&
              far_cuda_statistics.cuda_condition_device_to_host_bytes == 0,
          "far-distance Near2Far did not execute exactly one mixed CUDA "
          "transform within the workspace ceiling");

  // Individually representable eps/mu values can still make the fast
  // kernel's derived material arithmetic invalid. The first case overflows
  // mu/eps and its impedance; the second underflows eps*mu, k*r, and
  // 1/(k*r)^2 intermediates. Both remain finite in FP64 and must therefore
  // select the mixed CUDA path rather than fail or silently lose a field.
  const double original_eps = near.eps;
  const double original_mu = near.mu;
  const auto require_extreme_material_mixed = [&near, &grid_points](
      double eps, double mu, const std::string &label) {
    near.eps = eps;
    near.mu = mu;
    gpu::set_backend(gpu::backend_mode::cuda);
    gpu::reset_dispatch_statistics();
    std::unique_ptr<std::complex<double>[]> raw_cuda(
        near.farfields(grid_points.data(), grid_points.size()));
    const std::vector<std::complex<double> > cuda_values(
        raw_cuda.get(), raw_cuda.get() +
                            6 * grid_points.size() * near.freq.size());
    const gpu::near2far_statistics cuda_statistics =
        gpu::get_near2far_statistics();
    gpu::set_backend(gpu::backend_mode::cpu);
    gpu::reset_dispatch_statistics();
    std::unique_ptr<std::complex<double>[]> raw_cpu(
        near.farfields(grid_points.data(), grid_points.size()));
    const std::vector<std::complex<double> > cpu_values(
        raw_cpu.get(), raw_cpu.get() +
                           6 * grid_points.size() * near.freq.size());
    for (const std::complex<double> &value : cuda_values)
      require(std::isfinite(value.real()) && std::isfinite(value.imag()),
              label + " produced a non-finite CUDA field");
    require_near2far_complex_agreement(cpu_values, cuda_values, label);
    require(cuda_statistics.cuda_transform_calls == 1 &&
                cuda_statistics.cuda_fast_precision_calls == 0 &&
                cuda_statistics.cuda_mixed_precision_calls == 1 &&
                cuda_statistics.cuda_cancellation_retries == 0 &&
                cuda_statistics.cuda_maximum_workspace_bytes > 0 &&
                cuda_statistics.cuda_maximum_workspace_bytes <=
                    64u * 1024u * 1024u,
            label + " did not select one bounded mixed CUDA transform");
  };
  require_extreme_material_mixed(
      1.0e-20, 1.0e20,
      "impedance-overflow guarded CUDA Near2Far");
  require_extreme_material_mixed(
      1.0e-30, 1.0e-30,
      "index-and-kr-underflow guarded CUDA Near2Far");
  near.eps = original_eps;
  near.mu = original_mu;

  // Exercise two independent periodic image axes and nontrivial Bloch phase
  // factors on the exact same DFT snapshot. The production constructor fills
  // these fields from nperiods and use_bloch; assigning them here isolates the
  // Green-transform contract from a second FDTD run.
  near.periodic_d[0] = X;
  near.periodic_n[0] = 1;
  near.periodic_k[0] = 0.37;
  near.period[0] = 2.0;
  near.periodic_d[1] = Y;
  near.periodic_n[1] = 2;
  near.periodic_k[1] = -0.23;
  near.period[1] = 1.8;
  constexpr size_t tiled_periodic_copies = 15;
  const near2far3d_snapshot_capture periodic_cuda =
      capture_near2far3d_snapshot(
          near, targets, grid_points, gpu::backend_mode::cuda);
  const near2far3d_snapshot_capture periodic_cpu =
      capture_near2far3d_snapshot(
          near, targets, grid_points, gpu::backend_mode::cpu);
  require(periodic_cuda.live_after_grid ==
                  periodic_cuda.live_before_grid + 4 &&
              periodic_cuda.live_after_batch ==
                  periodic_cuda.live_after_grid &&
              periodic_cuda.live_after_scalar ==
                  periodic_cuda.live_after_grid &&
              periodic_cpu.live_before_grid <=
                  periodic_cuda.live_before_grid &&
              periodic_cpu.live_before_grid <=
                  periodic_cuda.live_after_scalar - 4,
          "periodic Near2Far retained buffers were not bounded and released");
  require_near2far_complex_agreement(
      periodic_cpu.batch, periodic_cuda.batch,
      "two-axis periodic/Bloch batched CUDA near-to-far");
  require_near2far_complex_agreement(
      periodic_cpu.scalar, periodic_cuda.scalar,
      "two-axis periodic/Bloch scalar CUDA near-to-far");
  require_near2far_cuda_transform_statistics(
      periodic_cuda.grid_statistics, grid_points.size(), near.freq.size(),
      chunk_count, source_points, tiled_periodic_copies, true, false);
  require_near2far_cuda_transform_statistics(
      periodic_cuda.batch_statistics, grid_points.size(), near.freq.size(),
      chunk_count, source_points, tiled_periodic_copies, false, false);
  require_near2far_cuda_transform_statistics(
      periodic_cuda.scalar_statistics, 1, near.freq.size(), chunk_count,
      source_points, tiled_periodic_copies, false, false);
  const std::uint64_t periodic_cpu_terms =
      static_cast<std::uint64_t>(source_points) * tiled_periodic_copies *
      grid_points.size() * near.freq.size();
  require(periodic_cpu.grid_statistics.cpu_terms == periodic_cpu_terms &&
              periodic_cpu.batch_statistics.cpu_terms ==
                  periodic_cpu_terms,
          "two-axis periodic CPU Near2Far term evidence is inconsistent");
  near.remove();
}

void require_near2far3d_public_periodic_constructor() {
  gpu::set_backend(gpu::backend_mode::cuda);
  const grid_volume gv = vol3d(2.0, 1.8, 1.6, 8.0);
  structure s(gv, vacuum, no_pml());
  fields f(&s, 0.0, 0.0, true, 64, 64);
  const vec center = gv.center();
  constexpr double bloch_x = 0.11;
  constexpr double bloch_y = -0.07;
  f.use_bloch(vec(bloch_x, bloch_y, 0.0));
  gaussian_src_time source(0.31, 0.12);
  f.add_point_source(Ez, source, center + vec(0.03, -0.04, -0.31), 0.8);

  // A full-cell XY surface and nperiods>1 exercise the same public
  // constructor path used by Python Simulation.add_near2far. Both transverse
  // periodic axes and their Bloch phases must be derived by Meep rather than
  // injected into dft_near2far internals by this test.
  volume_list near_region(
      volume(vec(center.x() - 1.0, center.y() - 0.9,
                 center.z() + 0.37),
             vec(center.x() + 1.0, center.y() + 0.9,
                 center.z() + 0.37)),
      Sz, 1.0);
  const double frequencies[] = {0.27, 0.35};
  constexpr int nperiods = 2;
  dft_near2far near =
      f.add_dft_near2far(&near_region, frequencies, 2, 1, nperiods);
  require(
      near.periodic_d[0] == X && near.periodic_d[1] == Y &&
          near.periodic_n[0] == nperiods &&
          near.periodic_n[1] == nperiods && near.period[0] > 0.0 &&
          near.period[1] > 0.0 &&
          std::abs(near.periodic_k[0] -
                   2.0 * pi * bloch_x * near.period[0]) <= 1.0e-12 &&
          std::abs(near.periodic_k[1] -
                   2.0 * pi * bloch_y * near.period[1]) <= 1.0e-12,
      "public add_dft_near2far did not derive both periodic/Bloch axes");

  for (int step = 0; step < 56; ++step) f.step();
  size_t chunk_count = 0;
  size_t source_points = 0;
  for (dft_chunk *chunk = near.F; chunk; chunk = chunk->next_in_dft)
    if (chunk->N) {
      ++chunk_count;
      source_points += chunk->N;
    }
  require(chunk_count > 0 && source_points > 0,
          "public periodic Near2Far monitor owns no source samples");
  const std::vector<vec> targets = {
      vec(2.4, 1.3, 0.9), vec(-2.1, 0.6, 1.4),
      vec(0.8, -2.3, 1.1)};

  gpu::reset_dispatch_statistics();
  std::unique_ptr<std::complex<double>[]> raw_cuda(
      near.farfields(targets.data(), targets.size()));
  const std::vector<std::complex<double> > cuda(
      raw_cuda.get(),
      raw_cuda.get() + 6 * targets.size() * near.freq.size());
  const gpu::near2far_statistics cuda_statistics =
      gpu::get_near2far_statistics();
  gpu::set_backend(gpu::backend_mode::cpu);
  gpu::reset_dispatch_statistics();
  std::unique_ptr<std::complex<double>[]> raw_cpu(
      near.farfields(targets.data(), targets.size()));
  const std::vector<std::complex<double> > cpu(
      raw_cpu.get(), raw_cpu.get() + 6 * targets.size() * near.freq.size());
  const gpu::near2far_statistics cpu_statistics =
      gpu::get_near2far_statistics();

  require_near2far_complex_agreement(
      cpu, cuda, "public two-axis periodic/Bloch CUDA Near2Far");
  constexpr size_t periodic_copies =
      (2 * static_cast<size_t>(nperiods) + 1) *
      (2 * static_cast<size_t>(nperiods) + 1);
  require_near2far_cuda_transform_statistics(
      cuda_statistics, targets.size(), near.freq.size(), chunk_count,
      source_points, periodic_copies, true, true);
  require(
      cpu_statistics.cpu_transform_calls == targets.size() &&
          cpu_statistics.cuda_transform_calls == 0 &&
          cpu_statistics.cpu_terms ==
              static_cast<std::uint64_t>(targets.size()) *
                  near.freq.size() * source_points * periodic_copies,
      "public periodic CPU Near2Far reference evidence is inconsistent");
  near.remove();
}

void require_near2far3d_symmetry_equivalence() {
  gpu::set_backend(gpu::backend_mode::cuda);
  gpu::reset_dispatch_statistics();
  const grid_volume gv = vol3d(2.0, 1.8, 1.6, 8.0);
  const symmetry reflection = mirror(X, gv);
  structure s(gv, vacuum, no_pml(), reflection);
  fields f(&s, 0.0, 0.0, true, 64, 64);
  const vec center = gv.center();
  gaussian_src_time source(0.29, 0.10);
  f.add_point_source(Ez, source, center, 0.8);

  const double hx = 0.44;
  const double hy = 0.36;
  const double hz = 0.28;
  volume_list near_region(
      volume(vec(center.x() + hx, center.y() - hy, center.z() - hz),
             vec(center.x() + hx, center.y() + hy, center.z() + hz)),
      Sx, 1.0,
      new volume_list(
          volume(vec(center.x() - hx, center.y() - hy, center.z() - hz),
                 vec(center.x() - hx, center.y() + hy, center.z() + hz)),
          Sx, -1.0,
          new volume_list(
              volume(vec(center.x() - hx, center.y() + hy,
                         center.z() - hz),
                     vec(center.x() + hx, center.y() + hy,
                         center.z() + hz)),
              Sy, 1.0,
              new volume_list(
                  volume(vec(center.x() - hx, center.y() - hy,
                             center.z() - hz),
                         vec(center.x() + hx, center.y() - hy,
                             center.z() + hz)),
                  Sy, -1.0,
                  new volume_list(
                      volume(vec(center.x() - hx, center.y() - hy,
                                 center.z() + hz),
                             vec(center.x() + hx, center.y() + hy,
                                 center.z() + hz)),
                      Sz, 1.0,
                      new volume_list(
                          volume(vec(center.x() - hx, center.y() - hy,
                                     center.z() - hz),
                                 vec(center.x() + hx, center.y() + hy,
                                     center.z() - hz)),
                          Sz, -1.0))))));
  const double frequencies[] = {0.24, 0.34};
  dft_near2far near =
      f.add_dft_near2far(&near_region, frequencies, 2);
  for (int step = 0; step < 56; ++step) f.step();

  bool saw_nonidentity_symmetry = false;
  size_t chunk_count = 0;
  size_t source_points = 0;
  for (dft_chunk *chunk = near.F; chunk; chunk = chunk->next_in_dft)
    if (chunk->N) {
      ++chunk_count;
      source_points += chunk->N;
      saw_nonidentity_symmetry =
          saw_nonidentity_symmetry || chunk->sn != 0;
    }
  require(saw_nonidentity_symmetry,
          "3D Near2Far symmetry regression created no transformed DFT chunk");
  const volume targets(
      vec(center.x() + 1.35, center.y() - 0.30, center.z() - 0.375),
      vec(center.x() + 1.85, center.y() + 0.30, center.z() + 0.375));
  const size_t expected_dims[3] = {2, 3, 3};
  std::vector<vec> points;
  points.reserve(18);
  for (size_t i0 = 0; i0 < expected_dims[0]; ++i0)
    for (size_t i1 = 0; i1 < expected_dims[1]; ++i1)
      for (size_t i2 = 0; i2 < expected_dims[2]; ++i2)
        points.push_back(vec(
            targets.in_direction_min(X) +
                i0 * targets.in_direction(X) / (expected_dims[0] - 1),
            targets.in_direction_min(Y) +
                i1 * targets.in_direction(Y) / (expected_dims[1] - 1),
            targets.in_direction_min(Z) +
                i2 * targets.in_direction(Z) / (expected_dims[2] - 1)));
  const near2far3d_snapshot_capture cuda = capture_near2far3d_snapshot(
      near, targets, points, gpu::backend_mode::cuda);
  const near2far3d_snapshot_capture cpu = capture_near2far3d_snapshot(
      near, targets, points, gpu::backend_mode::cpu);
  require_near2far_complex_agreement(
      cpu.batch, cuda.batch,
      "mirror-symmetry batched CUDA 3D near-to-far");
  require_near2far_complex_agreement(
      cpu.scalar, cuda.scalar,
      "mirror-symmetry scalar CUDA 3D near-to-far");
  require_near2far_cuda_transform_statistics(
      cuda.grid_statistics, points.size(), near.freq.size(), chunk_count,
      source_points, 1, true, true);
  require_near2far_cuda_transform_statistics(
      cuda.batch_statistics, points.size(), near.freq.size(), chunk_count,
      source_points, 1, false, false);
  require_near2far_cuda_transform_statistics(
      cuda.scalar_statistics, 1, near.freq.size(), chunk_count,
      source_points, 1, false, false);
  require(cuda.live_after_grid == cuda.live_before_grid + 4 &&
              cuda.live_after_batch == cuda.live_after_grid &&
              cuda.live_after_scalar == cuda.live_after_grid &&
              cpu.live_before_grid <= cuda.live_before_grid &&
              cpu.live_before_grid <= cuda.live_after_scalar - 4,
          "symmetry Near2Far retained-buffer lifecycle is inconsistent");
  near.remove();
}

void write_near2far_mpi_evidence(const char *path,
                                 gpu::backend_mode mode) {
  if (!path || !*path)
    throw std::invalid_argument(
        "Near2Far MPI evidence path must be nonempty");
  const char *target_count_text =
      std::getenv("MEEP_GPU_TEST_NEAR2FAR_TARGET_COUNT");
  const char *repetition_text =
      std::getenv("MEEP_GPU_TEST_NEAR2FAR_REPETITIONS");
  const char *frequency_count_text =
      std::getenv("MEEP_GPU_TEST_NEAR2FAR_FREQUENCY_COUNT");
  const auto parse_positive = [](const char *text, size_t fallback,
                                 const char *label) {
    if (!text || !*text) return fallback;
    char *end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno || !end || *end || parsed == 0 ||
        parsed > std::numeric_limits<size_t>::max())
      throw std::invalid_argument(std::string(label) +
                                  " must be a positive size_t");
    return static_cast<size_t>(parsed);
  };
  const size_t target_count = parse_positive(
      target_count_text, 256, "Near2Far target count");
  const size_t repetitions = parse_positive(
      repetition_text, 5, "Near2Far repetition count");
  const size_t frequency_count = parse_positive(
      frequency_count_text, 12, "Near2Far frequency count");

  gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();
  const grid_volume gv = vol3d(2.4, 2.0, 1.8, 10.0);
  structure s(gv, vacuum, no_pml());
  fields f(&s, 0.0, 0.0, true, 64, 64);
  gaussian_src_time source(0.30, 0.12);
  f.add_volume_source(Ez, source, f.v,
                      std::complex<double>(0.72, -0.13));
  f.add_volume_source(Hy, source, f.v,
                      std::complex<double>(-0.19, 0.08));
  const vec center = gv.center();
  const double hx = 0.54;
  const double hy = 0.42;
  const double hz = 0.34;
  volume_list near_region(
      volume(vec(center.x() + hx, center.y() - hy, center.z() - hz),
             vec(center.x() + hx, center.y() + hy, center.z() + hz)),
      Sx, 1.0,
      new volume_list(
          volume(vec(center.x() - hx, center.y() - hy, center.z() - hz),
                 vec(center.x() - hx, center.y() + hy, center.z() + hz)),
          Sx, -1.0,
          new volume_list(
              volume(vec(center.x() - hx, center.y() + hy,
                         center.z() - hz),
                     vec(center.x() + hx, center.y() + hy,
                         center.z() + hz)),
              Sy, 1.0,
              new volume_list(
                  volume(vec(center.x() - hx, center.y() - hy,
                             center.z() - hz),
                         vec(center.x() + hx, center.y() - hy,
                             center.z() + hz)),
                  Sy, -1.0,
                  new volume_list(
                      volume(vec(center.x() - hx, center.y() - hy,
                                 center.z() + hz),
                             vec(center.x() + hx, center.y() + hy,
                                 center.z() + hz)),
                      Sz, 1.0,
                      new volume_list(
                          volume(vec(center.x() - hx, center.y() - hy,
                                     center.z() - hz),
                                 vec(center.x() + hx, center.y() + hy,
                                     center.z() - hz)),
                          Sz, -1.0))))));
  std::vector<double> frequencies;
  frequencies.reserve(frequency_count);
  for (size_t index = 0; index < frequency_count; ++index)
    frequencies.push_back(
        0.22 + 0.17 * static_cast<double>(index) /
                   static_cast<double>(std::max<size_t>(1, frequency_count - 1)));
  dft_near2far near =
      f.add_dft_near2far(
          &near_region, frequencies.data(), frequencies.size(), 3);
  for (int step = 0; step < 72; ++step) f.step();

  size_t local_chunk_count = 0;
  size_t local_source_points = 0;
  for (dft_chunk *chunk = near.F; chunk; chunk = chunk->next_in_dft)
    if (chunk->N) {
      ++local_chunk_count;
      local_source_points += chunk->N;
    }
  std::vector<vec> targets;
  targets.reserve(target_count);
  for (size_t target = 0; target < target_count; ++target) {
    const double fraction =
        (static_cast<double>(target) + 0.5) /
        static_cast<double>(target_count);
    const double azimuth = 2.0 * pi * fraction;
    const double polar =
        std::acos(1.0 - 2.0 * fraction);
    const double radius = 2.8 + 0.15 * std::sin(7.0 * azimuth);
    targets.push_back(
        center + vec(radius * std::sin(polar) * std::cos(azimuth),
                     radius * std::sin(polar) * std::sin(azimuth),
                     radius * std::cos(polar)));
  }

  if (const char *metadata_mismatch =
          std::getenv("MEEP_GPU_TEST_NEAR2FAR_METADATA_MISMATCH")) {
    require(count_processors() == 2,
            "Near2Far metadata mismatch probe requires two MPI ranks");
    bool caught = false;
    double removed_frequency = 0.0;
    const ndim saved_monitor_dimension = near.where.dim;
    try {
      if (std::strcmp(metadata_mismatch, "point-count") == 0) {
        const vec *local_targets = my_rank() == 0 ? targets.data() : nullptr;
        const size_t local_count = my_rank() == 0 ? targets.size() : 0;
        std::unique_ptr<std::complex<double>[]> ignored(
            near.farfields(local_targets, local_count));
      }
      else if (std::strcmp(metadata_mismatch, "null-points") == 0) {
        const vec *local_targets = my_rank() == 0 ? targets.data() : nullptr;
        std::unique_ptr<std::complex<double>[]> ignored(
            near.farfields(local_targets, targets.size()));
      }
      else if (std::strcmp(metadata_mismatch, "target-coordinates") == 0) {
        std::vector<vec> local_targets(targets);
        if (my_rank() == 1)
          local_targets[0] = local_targets[0] + vec(0.01, 0.0, 0.0);
        std::unique_ptr<std::complex<double>[]> ignored(
            near.farfields(local_targets.data(), local_targets.size()));
      }
      else if (std::strcmp(metadata_mismatch, "frequency-count") == 0) {
        if (my_rank() == 1) {
          removed_frequency = near.freq.back();
          near.freq.pop_back();
        }
        std::unique_ptr<std::complex<double>[]> ignored(
            near.farfields(targets.data(), targets.size()));
      }
      else if (std::strcmp(metadata_mismatch, "frequency-values") == 0) {
        if (my_rank() == 1) {
          removed_frequency = near.freq[0];
          near.freq[0] = std::nextafter(near.freq[0], 1.0);
        }
        std::unique_ptr<std::complex<double>[]> ignored(
            near.farfields(targets.data(), targets.size()));
      }
      else if (std::strcmp(metadata_mismatch, "material-periodic") == 0) {
        if (my_rank() == 1) near.eps = std::nextafter(near.eps, 2.0);
        std::unique_ptr<std::complex<double>[]> ignored(
            near.farfields(targets.data(), targets.size()));
      }
      else if (std::strcmp(metadata_mismatch, "greencyl-tol") == 0) {
        const double local_tol = my_rank() == 0 ? 1.0e-6 : 2.0e-6;
        std::unique_ptr<std::complex<double>[]> ignored(
            near.farfields(targets.data(), targets.size(), local_tol));
      }
      else if (std::strcmp(metadata_mismatch, "monitor-dimension") == 0) {
        if (my_rank() == 1) near.where.dim = Dcyl;
        std::unique_ptr<std::complex<double>[]> ignored(
            near.farfields(targets.data(), targets.size()));
      }
      else if (std::strcmp(metadata_mismatch, "grid-shape") == 0) {
        const double extent = my_rank() == 0 ? 0.2 : 0.4;
        int output_rank = 0;
        size_t output_dims[3] = {};
        size_t output_count = 0;
        std::unique_ptr<double[]> ignored(near.get_farfields_array(
            volume(center + vec(2.7, -extent, -extent),
                   center + vec(2.9, extent, extent)),
            output_rank, output_dims, output_count, 10.0));
      }
      else if (std::strcmp(metadata_mismatch,
                           "grid-monitor-dimension") == 0) {
        if (my_rank() == 1) near.where.dim = Dcyl;
        int output_rank = 0;
        size_t output_dims[3] = {};
        size_t output_count = 0;
        std::unique_ptr<double[]> ignored(near.get_farfields_array(
            volume(center + vec(2.7, -0.1, -0.1),
                   center + vec(2.9, 0.1, 0.1)),
            output_rank, output_dims, output_count, 10.0));
      }
      else if (std::strcmp(metadata_mismatch, "grid-frequency-count") == 0) {
        if (my_rank() == 1) {
          removed_frequency = near.freq.back();
          near.freq.pop_back();
        }
        int output_rank = 0;
        size_t output_dims[3] = {};
        size_t output_count = 0;
        std::unique_ptr<double[]> ignored(near.get_farfields_array(
            volume(center + vec(2.7, -0.1, -0.1),
                   center + vec(2.9, 0.1, 0.1)),
            output_rank, output_dims, output_count, 10.0));
      }
      else
        throw std::invalid_argument(
            "Near2Far metadata mismatch must be point-count, "
            "null-points, target-coordinates, frequency-count, "
            "frequency-values, material-periodic, monitor-dimension, "
            "grid-shape, grid-monitor-dimension, or "
            "grid-frequency-count/greencyl-tol");
    }
    catch (const std::runtime_error &error) {
      caught = std::string(error.what()).find("inconsistent Near2Far") !=
               std::string::npos;
    }
    if (my_rank() == 1 && removed_frequency != 0.0) {
      if (near.freq.size() + 1 == frequency_count)
        near.freq.push_back(removed_frequency);
      else
        near.freq[0] = removed_frequency;
    }
    if (my_rank() == 1 &&
        std::strcmp(metadata_mismatch, "material-periodic") == 0)
      near.eps = 1.0;
    if (my_rank() == 1) near.where.dim = saved_monitor_dimension;
    require(and_to_all(caught),
            "Near2Far rank-local metadata mismatch did not fail closed on "
            "every rank");
    gpu::set_backend(gpu::backend_mode::cpu);
    near.remove();
    if (am_master())
      std::cout << "PASS: Near2Far " << metadata_mismatch
                << " metadata mismatch fails closed before result "
                   "collectives\n";
    return;
  }

  if (const char *selection_mismatch =
          std::getenv("MEEP_GPU_TEST_NEAR2FAR_SELECTION_MISMATCH")) {
    require(count_processors() == 2,
            "Near2Far selection mismatch probe requires two MPI ranks");
    if (my_rank() == 1) {
      if (std::strcmp(selection_mismatch, "optout") == 0)
        setenv("MEEP_GPU_DISABLE_RESIDENT_NEAR2FAR", "1", 1);
      else if (std::strcmp(selection_mismatch, "backend") == 0)
        gpu::set_backend(gpu::backend_mode::cpu);
      else
        throw std::invalid_argument(
            "Near2Far selection mismatch must be optout or backend");
    }
    const char *api = std::getenv("MEEP_GPU_TEST_NEAR2FAR_SELECTION_API");
    bool caught = false;
    try {
      if (api && std::strcmp(api, "grid") == 0) {
        int output_rank = 0;
        size_t output_dims[3] = {};
        size_t output_count = 0;
        std::unique_ptr<double[]> ignored(near.get_farfields_array(
            volume(center + vec(2.7, -0.1, -0.1),
                   center + vec(2.9, 0.1, 0.1)),
            output_rank, output_dims, output_count, 5.0));
      }
      else if (!api || std::strcmp(api, "batch") == 0) {
        std::unique_ptr<std::complex<double>[]> ignored(
            near.farfields(targets.data(), targets.size()));
      }
      else
        throw std::invalid_argument(
            "Near2Far selection API must be batch or grid");
    }
    catch (const std::runtime_error &error) {
      caught = std::string(error.what()).find(
                   "inconsistent CUDA Near2Far selection across MPI ranks") !=
               std::string::npos;
    }
    require(and_to_all(caught),
            "Near2Far rank-local CUDA selection mismatch did not fail closed "
            "on every rank");
    if (my_rank() == 1 &&
        std::strcmp(selection_mismatch, "optout") == 0)
      unsetenv("MEEP_GPU_DISABLE_RESIDENT_NEAR2FAR");
    gpu::set_backend(gpu::backend_mode::cpu);
    near.remove();
    if (am_master())
      std::cout << "PASS: Near2Far " << (api ? api : "batch") << ' '
                << selection_mismatch
                << " mismatch fails closed on every MPI rank\n";
    return;
  }

  if (std::getenv("MEEP_GPU_TEST_NEAR2FAR_CHUNKLESS_RANK")) {
    require(count_processors() == 2,
            "Near2Far chunkless-rank probe requires two MPI ranks");
    require(and_to_all(local_chunk_count > 0 && local_source_points > 0),
            "Near2Far chunkless-rank fixture did not start with work on every "
            "rank");
    dft_chunk *saved_chunks = near.F;
    if (my_rank() == 1) near.F = nullptr;
    const size_t field_count = 6 * near.freq.size() * targets.size();
    gpu::reset_dispatch_statistics();
    std::unique_ptr<std::complex<double>[]> cuda_raw(
        near.farfields(targets.data(), targets.size()));
    std::vector<std::complex<double> > cuda_fields(
        cuda_raw.get(), cuda_raw.get() + field_count);
    const gpu::near2far_statistics cuda_statistics =
        gpu::get_near2far_statistics();
    std::vector<std::complex<double> > root_fields(cuda_fields);
    if (!am_master())
      std::fill(root_fields.begin(), root_fields.end(),
                std::complex<double>(0.0, 0.0));
    broadcast(0, root_fields.data(), static_cast<int>(root_fields.size()));
    double rank_error = 0.0;
    double field_maximum = 0.0;
    for (size_t index = 0; index < field_count; ++index) {
      rank_error =
          std::max(rank_error, std::abs(cuda_fields[index] - root_fields[index]));
      field_maximum = std::max(field_maximum, std::abs(cuda_fields[index]));
    }
    require(rank_error == 0.0 && field_maximum > 0.0,
            "Near2Far chunkless rank did not receive the identical nonzero "
            "public CUDA result");
    require(cuda_statistics.mpi_allreduce_calls == 1 &&
                cuda_statistics.mpi_allreduce_bytes ==
                    13 * targets.size() * near.freq.size() * sizeof(double) &&
                (my_rank() == 0
                     ? cuda_statistics.cuda_transform_calls == 1 &&
                           cuda_statistics.cuda_submitted_chunks ==
                               local_chunk_count
                     : cuda_statistics.cuda_transform_calls == 0 &&
                           cuda_statistics.cuda_submitted_chunks == 0),
            "Near2Far chunkless rank did not enter the production 13-double "
            "collective exactly once");

    gpu::set_backend(gpu::backend_mode::cpu);
    std::unique_ptr<std::complex<double>[]> cpu_raw(
        near.farfields(targets.data(), targets.size()));
    std::vector<std::complex<double> > cpu_fields(
        cpu_raw.get(), cpu_raw.get() + field_count);
    if (my_rank() == 1) near.F = saved_chunks;
    require_near2far_complex_agreement(
        cpu_fields, cuda_fields,
        "chunkless-rank public CUDA/CPU Near2Far result");
    near.remove();
    if (am_master())
      std::cout << "PASS: chunkless rank enters the public 13-double "
                   "Near2Far collective and receives the nonempty rank result\n";
    return;
  }

  if (std::getenv("MEEP_GPU_TEST_NEAR2FAR_LOWLEVEL_DISTRIBUTED")) {
    require(count_processors() == 2,
            "distributed Near2Far lowlevel probe requires two MPI ranks");
    require(and_to_all(local_chunk_count > 0 && local_source_points > 0),
            "distributed Near2Far lowlevel fixture requires local work on "
            "every rank");
    std::vector<std::complex<double> > cuda_local(6 * near.freq.size());
    std::vector<std::complex<double> > cuda_global(6 * near.freq.size());
    std::vector<std::complex<double> > cpu_local(6 * near.freq.size());
    std::vector<std::complex<double> > cpu_global(6 * near.freq.size());
    gpu::reset_dispatch_statistics();
    near.farfield_lowlevel(cuda_local.data(), targets.front());
    const gpu::near2far_statistics cuda_statistics =
        gpu::get_near2far_statistics();
    sum_to_all(cuda_local.data(), cuda_global.data(),
               static_cast<int>(cuda_global.size()));
    require(cuda_statistics.cuda_transform_calls == 1 &&
                cuda_statistics.cuda_fast_precision_calls == 0 &&
                cuda_statistics.cuda_mixed_precision_calls == 1 &&
                cuda_statistics.cuda_cancellation_retries == 0 &&
                cuda_statistics.mpi_allreduce_calls == 0,
            "distributed rank-local Near2Far lowlevel did not force mixed "
            "CUDA without owning the caller collective");
    gpu::set_backend(gpu::backend_mode::cpu);
    near.farfield_lowlevel(cpu_local.data(), targets.front());
    sum_to_all(cpu_local.data(), cpu_global.data(),
               static_cast<int>(cpu_global.size()));
    require_near2far_complex_agreement(
        cpu_global, cuda_global,
        "distributed mixed-CUDA rank-local lowlevel Near2Far result");
    near.remove();
    if (am_master())
      std::cout << "PASS: distributed Near2Far lowlevel preserves rank-local "
                   "ownership and forces mixed CUDA before explicit sum\n";
    return;
  }

  std::unique_ptr<std::complex<double>[]> warmup(
      near.farfields(targets.data(), targets.size()));
  const size_t field_count =
      6 * near.freq.size() * targets.size();
  std::vector<std::complex<double> > reference(
      warmup.get(), warmup.get() + field_count);
  gpu::reset_dispatch_statistics();
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  std::vector<double> elapsed;
  elapsed.reserve(repetitions);
  std::vector<std::complex<double> > final_fields;
  for (size_t repetition = 0; repetition < repetitions; ++repetition) {
    const double start = wall_time();
    std::unique_ptr<std::complex<double>[]> raw(
        near.farfields(targets.data(), targets.size()));
    const double local_elapsed = wall_time() - start;
    elapsed.push_back(max_to_all(local_elapsed));
    final_fields.assign(raw.get(), raw.get() + field_count);
    require(final_fields == reference,
            "Near2Far MPI evidence is not repeat deterministic");
  }
  const std::uint64_t live_after =
      gpu::get_live_resident_device_buffers();
  const gpu::near2far_statistics statistics =
      gpu::get_near2far_statistics();
  const gpu::dispatch_statistics dispatch =
      gpu::get_dispatch_statistics();
  const int selected_device =
      mode == gpu::backend_mode::cuda ? gpu::selected_device() : -1;
  const std::string identifier =
      selected_device >= 0 ? gpu::device_identifier(selected_device) : "cpu";

  std::ostringstream rank_path;
  rank_path << path << ".rank-" << my_global_rank() << ".txt";
  std::ofstream output(
      rank_path.str().c_str(), std::ios::binary | std::ios::trunc);
  if (!output)
    throw std::runtime_error(
        "cannot create Near2Far MPI evidence output");
  output.imbue(std::locale::classic());
  output << "schema=gpmeep-near2far-mpi-evidence-v2\n"
         << "backend="
         << (mode == gpu::backend_mode::cuda ? "cuda" : "cpu") << '\n'
         << "rank=" << my_global_rank() << '\n'
         << "world_size=" << count_processors() << '\n'
         << "selected_device=" << selected_device << '\n'
         << "device_identifier=" << identifier << '\n'
         << "local_chunk_count=" << local_chunk_count << '\n'
         << "local_source_points=" << local_source_points << '\n'
         << "target_count=" << targets.size() << '\n'
         << "frequency_count=" << near.freq.size() << '\n'
         << "repetitions=" << repetitions << '\n'
         << "live_buffers_before=" << live_before << '\n'
         << "live_buffers_after=" << live_after << '\n'
         << "cpu_transform_calls=" << statistics.cpu_transform_calls << '\n'
         << "cuda_transform_calls=" << statistics.cuda_transform_calls << '\n'
         << "cuda_submitted_chunks=" << statistics.cuda_submitted_chunks << '\n'
         << "cuda_source_points=" << statistics.cuda_source_points << '\n'
         << "cuda_output_points=" << statistics.cuda_output_points << '\n'
         << "cuda_frequencies=" << statistics.cuda_frequencies << '\n'
         << "cuda_periodic_copies=" << statistics.cuda_periodic_copies << '\n'
         << "cuda_fast_precision_calls="
         << statistics.cuda_fast_precision_calls << '\n'
         << "cuda_mixed_precision_calls="
         << statistics.cuda_mixed_precision_calls << '\n'
         << "cuda_cancellation_retries="
         << statistics.cuda_cancellation_retries << '\n'
         << "cuda_target_tiles=" << statistics.cuda_target_tiles << '\n'
         << "cuda_frequency_tiles=" << statistics.cuda_frequency_tiles << '\n'
         << "cuda_operation_tiles=" << statistics.cuda_operation_tiles << '\n'
         << "cuda_maximum_workspace_bytes="
         << statistics.cuda_maximum_workspace_bytes << '\n'
         << "cuda_descriptor_uploads=" << statistics.cuda_descriptor_uploads << '\n'
         << "cuda_kernel_launches=" << statistics.cuda_kernel_launches << '\n'
         << "cuda_result_device_to_host_bytes="
         << statistics.cuda_result_device_to_host_bytes << '\n'
         << "cuda_condition_device_to_host_bytes="
         << statistics.cuda_condition_device_to_host_bytes << '\n'
         << "dft_device_to_host_bytes_avoided="
         << statistics.dft_device_to_host_bytes_avoided << '\n'
         << "mpi_allreduce_calls=" << statistics.mpi_allreduce_calls << '\n'
         << "mpi_allreduce_bytes=" << statistics.mpi_allreduce_bytes << '\n'
         << "cpu_curl_calls=" << dispatch.cpu_curl_calls << '\n'
         << "cuda_curl_calls=" << dispatch.cuda_curl_calls << '\n';
  for (size_t repetition = 0; repetition < elapsed.size(); ++repetition)
    output << "elapsed_" << repetition << '=' << std::hexfloat
           << elapsed[repetition] << '\n';
  for (size_t index = 0; index < final_fields.size(); ++index)
    output << "field_" << index << "_real=" << std::hexfloat
           << final_fields[index].real() << '\n'
           << "field_" << index << "_imag=" << std::hexfloat
           << final_fields[index].imag() << '\n';
  output.flush();
  if (!output)
    throw std::runtime_error(
        "failed to flush Near2Far MPI evidence output");
  near.remove();
  if (am_master())
    std::cout << "PASS: wrote Near2Far MPI evidence for "
              << count_processors() << " rank(s), backend="
              << (mode == gpu::backend_mode::cuda ? "cuda" : "cpu")
              << ", targets=" << target_count
              << ", repetitions=" << repetitions << '\n';
}

void run_distributed_near2far_failure_abort_probe() {
  require(count_processors() > 1,
          "Near2Far failure-abort probe requires multiple MPI ranks");
  require(sizeof(realnum) == sizeof(float),
          "Near2Far CUDA failure-abort probe requires an FP32 build");
  gpu::set_backend(gpu::backend_mode::cuda);
  require(gpu::active_backend() == gpu::backend_mode::cuda,
          "Near2Far failure-abort probe requires the CUDA backend");
  const grid_volume gv = vol3d(1.4, 1.2, 1.0, 6.0);
  structure s(gv, vacuum, no_pml());
  fields f(&s);
  const vec center = gv.center();
  gaussian_src_time source(0.3, 0.1);
  f.add_point_source(Ez, source, center, 1.0);
  volume_list near_region(
      volume(vec(center.x() + 0.3, center.y() - 0.25,
                 center.z() - 0.2),
             vec(center.x() + 0.3, center.y() + 0.25,
                 center.z() + 0.2)),
      Sx, 1.0);
  dft_near2far near =
      f.add_dft_near2far(&near_region, 0.3, 0.3, 1);
  for (int step = 0; step < 8; ++step) f.step();
  const vec target(center.x() + 1.2, center.y() + 0.1,
                   center.z() - 0.05);
  if (my_global_rank() == 0)
    gpu::detail::set_near2far_execution_failures_for_testing(1);
  write_stderr_line_atomically(
      "gpmeep-near2far-abort-probe:rank=" +
      std::to_string(my_global_rank()) +
      ",stage=before-rank-local-cuda-failure");
  // The public farfields guard is armed on both ranks. Rank zero fails only
  // after selecting its CUDA device inside the resident Near2Far path, while
  // rank one performs the valid CUDA transform and proceeds toward the
  // matching allreduce. The guard must terminate through MPI_Abort instead
  // of leaving rank one hung in the collective.
  std::unique_ptr<std::complex<double>[]> ignored(
      near.farfields(&target, 1));
  write_stderr_line_atomically(
      "FAIL: Near2Far rank-local exception returned without communicator abort");
  throw std::runtime_error(
      "distributed Near2Far failure-abort probe unexpectedly returned");
}

ldos_workload_result run_full_volume_ldos_case(
    gpu::backend_mode mode) {
  gpu::set_backend(mode);
  const grid_volume gv = vol2d(2.0, 2.0, 12.0);
  structure s(gv, vacuum, no_pml());
  fields f(&s, 0.0, 0.0, true, 64, 64);
  continuous_src_time source(0.27);
  // Both source profiles span the full distributed volume. This makes every
  // rank own real electric and magnetic LDOS work instead of validating a
  // rank-zero-only point-source shortcut.
  f.add_volume_source(Ez, source, f.v,
                      std::complex<double>(0.80, -0.15));
  f.add_volume_source(Hz, source, f.v,
                      std::complex<double>(-0.35, 0.20));
  dft_ldos ldos(0.22, 0.32, 3);

  for (int step = 0; step < 8; ++step) f.step();
  gpu::reset_dispatch_statistics();
  constexpr int measured_updates = 32;
  for (int step = 0; step < measured_updates; ++step) {
    f.step();
    ldos.update(f);
  }

  ldos_workload_result result;
  double *values = ldos.ldos();
  std::complex<double> *fields_values = ldos.F();
  std::complex<double> *current_values = ldos.J();
  for (std::size_t frequency = 0; frequency < ldos.freq.size();
       ++frequency) {
    result.physics.samples.push_back(
        std::complex<double>(values[frequency], 0.0));
    result.physics.samples.push_back(fields_values[frequency]);
    result.physics.samples.push_back(current_values[frequency]);
  }
  delete[] values;
  delete[] fields_values;
  delete[] current_values;
  result.physics.statistics = gpu::get_dispatch_statistics();
  result.physics.resident = gpu::get_resident_statistics();
  result.physics.sources = gpu::get_source_statistics();
  result.ldos = gpu::get_ldos_statistics();
  return result;
}

void require_full_volume_ldos_determinism() {
  constexpr std::uint64_t measured_updates = 32;
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  const ldos_workload_result cpu =
      run_full_volume_ldos_case(gpu::backend_mode::cpu);
  require(
      and_to_all(cpu.ldos.cpu_reduction_calls == measured_updates &&
                 cpu.ldos.cpu_source_points > 0 &&
                 cpu.ldos.cuda_reduction_calls == 0),
      "full-volume CPU LDOS reference did not reduce electric and magnetic "
      "source profiles on every rank");
  require(gpu::get_live_resident_device_buffers() == live_before,
          "full-volume CPU LDOS reference changed resident allocation count");

  ldos_workload_result deterministic_reference;
  for (int repetition = 0; repetition < 4; ++repetition) {
    const ldos_workload_result cuda =
        run_full_volume_ldos_case(gpu::backend_mode::cuda);
    compare_results(cpu.physics, cuda.physics, two_dimensional_case);
    require(
        and_to_all(
            cuda.ldos.cpu_reduction_calls == 0 &&
            cuda.ldos.cuda_reduction_calls == measured_updates &&
            cuda.ldos.cuda_submitted_profiles >= 2 * measured_updates &&
            cuda.ldos.cuda_source_points > 0 &&
            cuda.ldos.cuda_descriptor_uploads == 1 &&
            cuda.ldos.cuda_kernel_launches == 2 * measured_updates &&
            cuda.ldos.cuda_result_device_to_host_bytes ==
                4 * sizeof(double) * measured_updates &&
            cuda.ldos.full_field_device_to_host_bytes_avoided > 0 &&
            cuda.physics.sources.cuda_update_calls > 0 &&
            cuda.physics.sources.cpu_update_calls == 0),
        "full-volume CUDA LDOS did not use deterministic resident reduction "
        "and exclusive CUDA source updates on every rank");
    if (repetition == 0)
      deterministic_reference = cuda;
    else
      require(
          cuda.physics.samples.size() ==
                  deterministic_reference.physics.samples.size() &&
              (cuda.physics.samples.empty() ||
               std::memcmp(
                   cuda.physics.samples.data(),
                   deterministic_reference.physics.samples.data(),
                   cuda.physics.samples.size() *
                       sizeof(std::complex<double>)) == 0),
          "full-volume CUDA LDOS F/J/value payload changed bits across "
          "independent simulations");
    require(gpu::get_live_resident_device_buffers() == live_before,
            "full-volume CUDA LDOS simulation leaked a resident allocation");
  }

  const gpu::ldos_statistics &statistics = deterministic_reference.ldos;
  const std::uint64_t global_profiles =
      global_sum(statistics.cuda_submitted_profiles);
  const std::uint64_t global_source_points =
      global_sum(statistics.cuda_source_points);
  const std::uint64_t global_kernels =
      global_sum(statistics.cuda_kernel_launches);
  const std::uint64_t global_result_bytes =
      global_sum(statistics.cuda_result_device_to_host_bytes);
  const std::uint64_t global_avoided_bytes =
      global_sum(statistics.full_field_device_to_host_bytes_avoided);
  if (am_master())
    std::cout
        << "ldos-deterministic-volume: ranks=" << count_processors()
        << " updates=" << measured_updates
        << " profiles=" << global_profiles
        << " source-points=" << global_source_points
        << " kernels=" << global_kernels
        << " result-d2h=" << global_result_bytes
        << " full-field-d2h-avoided=" << global_avoided_bytes
        << '\n';
}

ldos_benchmark_result run_ldos_transfer_benchmark_sample(
    bool disable_resident_ldos) {
  scoped_environment_override control(
      "MEEP_GPU_DISABLE_RESIDENT_LDOS",
      disable_resident_ldos ? "1" : nullptr);
  gpu::set_backend(gpu::backend_mode::cuda);
  constexpr int pixels = 64;
  constexpr double length = 4.0;
  const grid_volume gv =
      vol3d(length, length, length, pixels / length);
  structure s(gv, vacuum, no_pml());
  fields f(&s, 0.0, 0.0, true, 128, 128);
  f.use_real_fields();
  continuous_src_time source(0.23);
  f.add_volume_source(Ex, source, f.v, 0.75);
  f.add_volume_source(Hx, source, f.v, -0.40);
  for (int step = 0; step < 4; ++step) f.step();

  dft_ldos ldos(0.19, 0.29, 3);
  gpu::reset_dispatch_statistics();
  constexpr int measured_updates = 16;
  double local_seconds = 0.0;
  std::uint64_t voluntary_context_switches = 0;
  std::uint64_t involuntary_context_switches = 0;
  std::uint64_t major_faults = 0;
  std::uint64_t minor_faults = 0;
  std::vector<ldos_benchmark_result::timed_update_record> timed_updates;
  for (int step = 0; step < measured_updates; ++step) {
    f.step();
    all_wait();
    ldos_benchmark_result::timed_update_record timing;
    timing.affinity_before = current_cpu_affinity();
    timing.cpu_before = sched_getcpu();
    require(timing.cpu_before >= 0,
            "could not read pre-update LDOS worker CPU");
    struct rusage usage_before;
    require(getrusage(RUSAGE_THREAD, &usage_before) == 0,
            "could not read pre-update LDOS worker usage");
    struct timespec monotonic_start;
    require(clock_gettime(CLOCK_MONOTONIC, &monotonic_start) == 0 &&
                monotonic_start.tv_sec >= 0 && monotonic_start.tv_nsec >= 0 &&
                monotonic_start.tv_nsec < 1000000000L,
            "could not read pre-update CLOCK_MONOTONIC");
    timing.start_monotonic_ns =
        static_cast<std::uint64_t>(monotonic_start.tv_sec) * 1000000000ULL +
        static_cast<std::uint64_t>(monotonic_start.tv_nsec);
    ldos.update(f);
    struct timespec monotonic_stop;
    require(clock_gettime(CLOCK_MONOTONIC, &monotonic_stop) == 0 &&
                monotonic_stop.tv_sec >= 0 && monotonic_stop.tv_nsec >= 0 &&
                monotonic_stop.tv_nsec < 1000000000L,
            "could not read post-update CLOCK_MONOTONIC");
    timing.stop_monotonic_ns =
        static_cast<std::uint64_t>(monotonic_stop.tv_sec) * 1000000000ULL +
        static_cast<std::uint64_t>(monotonic_stop.tv_nsec);
    require(timing.stop_monotonic_ns > timing.start_monotonic_ns,
            "LDOS update CLOCK_MONOTONIC interval is not strictly positive");
    timing.elapsed_ns =
        timing.stop_monotonic_ns - timing.start_monotonic_ns;
    struct rusage usage_after;
    require(getrusage(RUSAGE_THREAD, &usage_after) == 0,
            "could not read post-update LDOS worker usage");
    require(usage_after.ru_nvcsw >= usage_before.ru_nvcsw &&
                usage_after.ru_nivcsw >= usage_before.ru_nivcsw &&
                usage_after.ru_minflt >= usage_before.ru_minflt &&
                usage_after.ru_majflt >= usage_before.ru_majflt,
            "LDOS worker usage counters moved backwards");
    timing.voluntary_context_switches = static_cast<std::uint64_t>(
        usage_after.ru_nvcsw - usage_before.ru_nvcsw);
    timing.involuntary_context_switches = static_cast<std::uint64_t>(
        usage_after.ru_nivcsw - usage_before.ru_nivcsw);
    timing.minor_faults = static_cast<std::uint64_t>(
        usage_after.ru_minflt - usage_before.ru_minflt);
    timing.major_faults = static_cast<std::uint64_t>(
        usage_after.ru_majflt - usage_before.ru_majflt);
    voluntary_context_switches += timing.voluntary_context_switches;
    involuntary_context_switches += timing.involuntary_context_switches;
    minor_faults += timing.minor_faults;
    major_faults += timing.major_faults;
    timing.cpu_after = sched_getcpu();
    require(timing.cpu_after >= 0,
            "could not read post-update LDOS worker CPU");
    timing.affinity_after = current_cpu_affinity();
    timed_updates.push_back(timing);
    local_seconds += static_cast<double>(
        timing.stop_monotonic_ns - timing.start_monotonic_ns) / 1e9;
  }

  ldos_benchmark_result result;
  result.reduction_seconds = max_to_all(local_seconds);
  result.voluntary_context_switches = voluntary_context_switches;
  result.involuntary_context_switches = involuntary_context_switches;
  result.major_faults = major_faults;
  result.minor_faults = minor_faults;
  result.timed_updates = timed_updates;
  double *values = ldos.ldos();
  std::complex<double> *fields_values = ldos.F();
  std::complex<double> *current_values = ldos.J();
  for (std::size_t frequency = 0; frequency < ldos.freq.size();
       ++frequency) {
    result.physics.samples.push_back(
        std::complex<double>(values[frequency], 0.0));
    result.physics.samples.push_back(fields_values[frequency]);
    result.physics.samples.push_back(current_values[frequency]);
  }
  delete[] values;
  delete[] fields_values;
  delete[] current_values;
  result.physics.statistics = gpu::get_dispatch_statistics();
  result.ldos = gpu::get_ldos_statistics();
  return result;
}

void require_ldos_transfer_speedup() {
  constexpr std::uint64_t measured_updates = 16;
  constexpr int repetitions = 5;
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  const std::vector<int> benchmark_cpu_affinity =
      wait_for_ldos_controller_gate();
  std::vector<double> speedups;
  std::vector<double> host_seconds;
  std::vector<double> resident_seconds;
  for (int repetition = 0; repetition < repetitions; ++repetition) {
    ldos_benchmark_result host;
    ldos_benchmark_result resident;
    if (repetition % 2 == 0) {
      host = run_ldos_transfer_benchmark_sample(true);
      resident = run_ldos_transfer_benchmark_sample(false);
    }
    else {
      resident = run_ldos_transfer_benchmark_sample(false);
      host = run_ldos_transfer_benchmark_sample(true);
    }
    const bool local_scheduling_gate =
        host.involuntary_context_switches == 0 &&
        resident.involuntary_context_switches == 0 &&
        host.major_faults == 0 && resident.major_faults == 0;
    const bool scheduling_gate = and_to_all(local_scheduling_gate);
    const numerical_error_summary errors = compare_sample_errors(
        host.physics.samples, resident.physics.samples);
    if (!scheduling_gate) {
      // Rejected physical attempts are never selected into the performance
      // sample. Emit every rank's complete timing/fault evidence first, then
      // fail the fixed five-attempt run. Natural minor faults remain
      // diagnostic and are deliberately not a rejection predicate.
      {
        const int rank = my_rank();
        std::ostringstream rejected;
        rejected << "gpmeep-ldos-rejected-v1:{\"schema_version\":1"
                 << ",\"rank\":" << rank
                 << ",\"repetition\":" << repetition
                 << ",\"order\":\""
                 << (repetition % 2 == 0 ? "host-resident" : "resident-host")
                 << "\",\"global_gate_accepted\":false"
                 << ",\"device_ordinal\":" << gpu::selected_device()
                 << ",\"device_uuid\":\""
                 << json_escape(gpu::selected_device_identifier()) << "\""
                 << ",\"host_seconds\":"
                 << canonical_decimal(host.reduction_seconds)
                 << ",\"resident_seconds\":"
                 << canonical_decimal(resident.reduction_seconds)
                 << ",\"host_d2h\":"
                 << host.physics.statistics.device_to_host_bytes
                 << ",\"resident_d2h\":"
                 << resident.physics.statistics.device_to_host_bytes
                 << ",\"host_cpu_reduction_calls\":"
                 << host.ldos.cpu_reduction_calls
                 << ",\"host_cuda_reduction_calls\":"
                 << host.ldos.cuda_reduction_calls
                 << ",\"resident_cpu_reduction_calls\":"
                 << resident.ldos.cpu_reduction_calls
                 << ",\"resident_cuda_reduction_calls\":"
                 << resident.ldos.cuda_reduction_calls
                 << ",\"resident_cuda_kernel_launches\":"
                 << resident.ldos.cuda_kernel_launches
                 << ",\"resident_result_d2h\":"
                 << resident.ldos.cuda_result_device_to_host_bytes
                 << ",\"max_absolute_error\":"
                 << canonical_decimal(errors.maximum_absolute)
                 << ",\"max_relative_error\":"
                 << canonical_decimal(errors.maximum_relative)
                 << ",\"host_digest\":\""
                 << hex_digest(sample_digest(host.physics.samples))
                 << "\",\"resident_digest\":\""
                 << hex_digest(sample_digest(resident.physics.samples))
                 << "\",\"host_samples\":"
                 << json_samples(host.physics.samples)
                 << ",\"resident_samples\":"
                 << json_samples(resident.physics.samples)
                 << ",\"cpu_affinity\":[";
        for (std::size_t cpu_index = 0;
             cpu_index < benchmark_cpu_affinity.size(); ++cpu_index) {
          if (cpu_index) rejected << ',';
          rejected << benchmark_cpu_affinity[cpu_index];
        }
        rejected << ']'
                 << ",\"host_voluntary_context_switches\":"
                 << host.voluntary_context_switches
                 << ",\"host_involuntary_context_switches\":"
                 << host.involuntary_context_switches
                 << ",\"resident_voluntary_context_switches\":"
                 << resident.voluntary_context_switches
                 << ",\"resident_involuntary_context_switches\":"
                 << resident.involuntary_context_switches
                 << ",\"host_major_faults\":" << host.major_faults
                 << ",\"resident_major_faults\":" << resident.major_faults
                 << ",\"host_minor_faults\":" << host.minor_faults
                 << ",\"resident_minor_faults\":" << resident.minor_faults
                 << ",\"reason_predicates\":{"
                 << "\"host_involuntary_context_switches_nonzero\":"
                 << (host.involuntary_context_switches ? "true" : "false")
                 << ",\"resident_involuntary_context_switches_nonzero\":"
                 << (resident.involuntary_context_switches ? "true" : "false")
                 << ",\"host_major_faults_nonzero\":"
                 << (host.major_faults ? "true" : "false")
                 << ",\"resident_major_faults_nonzero\":"
                 << (resident.major_faults ? "true" : "false")
                 << "},\"host_timed_updates\":";
        append_timed_updates_json(rejected, host.timed_updates);
        rejected << ",\"resident_timed_updates\":";
        append_timed_updates_json(rejected, resident.timed_updates);
        rejected << '}';
        emit_rank_records_one_writer(rejected.str());
      }
      all_wait();
    }
    require(
        scheduling_gate,
        "focused LDOS timing segment exceeded the scheduling/fault gate");
    compare_results(host.physics, resident.physics,
                    three_dimensional_case);
    require(
        and_to_all(
            host.ldos.cpu_reduction_calls == measured_updates &&
            host.ldos.cuda_reduction_calls == 0 &&
            resident.ldos.cpu_reduction_calls == 0 &&
            resident.ldos.cuda_reduction_calls == measured_updates &&
            resident.ldos.cuda_kernel_launches ==
                2 * measured_updates &&
            resident.ldos.cuda_result_device_to_host_bytes ==
                4 * sizeof(double) * measured_updates &&
            resident.physics.statistics.device_to_host_bytes <
                host.physics.statistics.device_to_host_bytes),
        "LDOS transfer benchmark did not compare full-field host fallback "
        "against the deterministic resident reduction on every rank");
    require(gpu::get_live_resident_device_buffers() == live_before,
            "LDOS transfer benchmark leaked a resident allocation");
    host_seconds.push_back(host.reduction_seconds);
    resident_seconds.push_back(resident.reduction_seconds);
    speedups.push_back(host.reduction_seconds /
                       resident.reduction_seconds);
    {
      const int rank = my_rank();
      std::ostringstream record;
      record << "gpmeep-ldos-rank-v1:{\"rank\":" << rank
             << ",\"repetition\":" << repetition
             << ",\"device_ordinal\":" << gpu::selected_device()
             << ",\"device_uuid\":\""
             << json_escape(gpu::selected_device_identifier()) << "\""
             << ",\"host_d2h\":"
             << host.physics.statistics.device_to_host_bytes
             << ",\"resident_d2h\":"
             << resident.physics.statistics.device_to_host_bytes
             << ",\"host_cpu_reduction_calls\":"
             << host.ldos.cpu_reduction_calls
             << ",\"host_cuda_reduction_calls\":"
             << host.ldos.cuda_reduction_calls
             << ",\"resident_cpu_reduction_calls\":"
             << resident.ldos.cpu_reduction_calls
             << ",\"resident_cuda_reduction_calls\":"
             << resident.ldos.cuda_reduction_calls
             << ",\"resident_cuda_kernel_launches\":"
             << resident.ldos.cuda_kernel_launches
             << ",\"resident_result_d2h\":"
             << resident.ldos.cuda_result_device_to_host_bytes
             << ",\"cpu_affinity\":[";
      for (std::size_t cpu_index = 0;
           cpu_index < benchmark_cpu_affinity.size(); ++cpu_index) {
        if (cpu_index) record << ',';
        record << benchmark_cpu_affinity[cpu_index];
      }
      record << ']'
             << ",\"host_voluntary_context_switches\":"
             << host.voluntary_context_switches
             << ",\"host_involuntary_context_switches\":"
             << host.involuntary_context_switches
             << ",\"resident_voluntary_context_switches\":"
             << resident.voluntary_context_switches
             << ",\"resident_involuntary_context_switches\":"
             << resident.involuntary_context_switches
             << ",\"host_major_faults\":" << host.major_faults
             << ",\"resident_major_faults\":" << resident.major_faults
             << ",\"host_minor_faults\":" << host.minor_faults
             << ",\"resident_minor_faults\":" << resident.minor_faults
             << ",\"host_timed_updates\":";
      append_timed_updates_json(record, host.timed_updates);
      record << ",\"resident_timed_updates\":";
      append_timed_updates_json(record, resident.timed_updates);
      record << '}';
      emit_rank_records_one_writer(record.str());
    }
    all_wait();
    if (am_master()) {
      std::cout
          << "gpmeep-ldos-pair-v1:{\"repetition\":" << repetition
          << ",\"order\":\""
          << (repetition % 2 == 0 ? "host-resident" : "resident-host")
          << "\",\"host_seconds\":"
          << canonical_decimal(host.reduction_seconds)
          << ",\"resident_seconds\":"
          << canonical_decimal(resident.reduction_seconds)
          << ",\"speedup\":"
          << canonical_decimal(host.reduction_seconds /
                               resident.reduction_seconds)
          << ",\"max_absolute_error\":"
          << canonical_decimal(errors.maximum_absolute)
          << ",\"max_relative_error\":"
          << canonical_decimal(errors.maximum_relative)
          << ",\"host_digest\":\""
          << hex_digest(sample_digest(host.physics.samples))
          << "\",\"resident_digest\":\""
          << hex_digest(sample_digest(resident.physics.samples))
          << "\",\"host_samples\":"
          << json_samples(host.physics.samples)
          << ",\"resident_samples\":"
          << json_samples(resident.physics.samples) << "}\n";
    }
  }
  std::vector<double> sorted_speedups = speedups;
  std::sort(sorted_speedups.begin(), sorted_speedups.end());
  const double minimum_speedup = sorted_speedups.front();
  const double median_speedup =
      sorted_speedups[sorted_speedups.size() / 2];
  require(minimum_speedup > 1.20,
          "deterministic resident LDOS did not beat full-field host fallback "
          "by more than 20% in every paired large-grid repetition");

  {
    all_wait();
    const int rank = my_rank();
    const runtime_mapping_inventory inventory = loaded_runtime_files();
    std::ostringstream record;
    record << "gpmeep-ldos-runtime-v1:{\"rank\":" << rank
           << ",\"device_ordinal\":" << gpu::selected_device()
           << ",\"device_uuid\":\""
           << json_escape(gpu::selected_device_identifier())
           << "\",\"libmeep\":\""
           << json_escape(loaded_libmeep_path()) << "\",\"mappings\":[";
    for (std::size_t index = 0; index < inventory.regular.size(); ++index) {
      if (index) record << ',';
      record << "{\"path\":\"" << json_escape(inventory.regular[index].path)
             << "\",\"device\":" << inventory.regular[index].device
             << ",\"inode\":" << inventory.regular[index].inode
             << ",\"size\":" << inventory.regular[index].size
             << ",\"mtime_ns\":" << inventory.regular[index].mtime_ns
             << ",\"ctime_ns\":" << inventory.regular[index].ctime_ns << '}';
    }
    record << "],\"special_mappings\":[";
    for (std::size_t index = 0; index < inventory.special.size(); ++index) {
      if (index) record << ',';
      const mapped_runtime_special &mapping = inventory.special[index];
      record << "{\"path\":\"" << json_escape(mapping.path)
             << "\",\"kind\":\"" << mapping.kind
             << "\",\"device\":" << mapping.device
             << ",\"inode\":" << mapping.inode
             << ",\"mode\":" << mapping.mode
             << ",\"rdev\":" << mapping.rdev
             << ",\"deleted\":" << (mapping.deleted ? "true" : "false")
             << ",\"executable\":false}";
    }
    record << "]}";
    emit_rank_records_one_writer(record.str());
  }
  all_wait();
  if (am_master()) {
    std::cout << "ldos-transfer-benchmark: ranks=" << count_processors()
              << " pixels=64 updates=" << measured_updates
              << " repetitions=" << repetitions
              << " minimum-speedup=" << canonical_decimal(minimum_speedup)
              << " median-speedup=" << canonical_decimal(median_speedup)
              << " paired-speedups=";
    for (std::size_t index = 0; index < speedups.size(); ++index)
      std::cout << (index ? "," : "")
                << canonical_decimal(speedups[index]);
    std::cout << '\n';
  }
}

void require_monitor_consumer_phase_sharing_equivalence(
    const run_result &cpu_reference) {
  gpu::detail::dft_batch_statistics sharing_statistics{};
  const run_result sharing = run_monitor_consumer_case(
      gpu::backend_mode::cuda, false, &sharing_statistics);
  gpu::detail::dft_batch_statistics control_statistics{};
  const run_result control = run_monitor_consumer_case(
      gpu::backend_mode::cuda, true, &control_statistics);
  compare_results(cpu_reference, sharing, two_dimensional_case);
  compare_results(cpu_reference, control, two_dimensional_case);

  const bool local_exact =
      sharing.samples.size() == control.samples.size() &&
      (sharing.samples.empty() ||
       std::memcmp(
           sharing.samples.data(), control.samples.data(),
           sharing.samples.size() * sizeof(std::complex<double>)) == 0);
  require(and_to_all(local_exact),
          "DFT phase sharing changed a flux, energy, force, near2far, or "
          "LDOS consumer result");
  const std::uint64_t sharing_updates =
      global_sum(sharing_statistics.submitted_updates);
  const std::uint64_t sharing_preparations =
      global_sum(sharing_statistics.phase_preparation_launches);
  const std::uint64_t sharing_reuses =
      global_sum(sharing_statistics.phase_reuses);
  const std::uint64_t control_updates =
      global_sum(control_statistics.submitted_updates);
  const std::uint64_t control_preparations =
      global_sum(control_statistics.phase_preparation_launches);
  const std::uint64_t control_reuses =
      global_sum(control_statistics.phase_reuses);
  require(
      sharing_updates > 0 && sharing_updates == control_updates &&
          sharing_preparations + sharing_reuses == sharing_updates &&
          sharing_reuses > 0 &&
          control_preparations == control_updates && control_reuses == 0,
      "representative DFT consumers did not exercise phase sharing and its "
      "disabled control");
  require(
      sharing.dfts.cuda_update_calls > 0 &&
          sharing.dfts.cpu_update_calls == 0 &&
          control.dfts.cuda_update_calls > 0 &&
          control.dfts.cpu_update_calls == 0,
      "representative monitor consumers silently used CPU DFT updates");
  require(
      cpu_reference.dft_reductions.cpu_reduction_calls > 0 &&
          cpu_reference.dft_reductions.cuda_reduction_calls == 0 &&
          sharing.dft_reductions.cuda_reduction_calls > 0 &&
          sharing.dft_reductions.cpu_reduction_calls == 0 &&
          control.dft_reductions.cuda_reduction_calls > 0 &&
          control.dft_reductions.cpu_reduction_calls == 0,
      "representative spectral consumers silently used the wrong CPU/CUDA "
      "reduction path");
  require(
      sharing.dft_reductions.cuda_submitted_pairs > 0 &&
          sharing.dft_reductions.cuda_point_frequency_terms > 0 &&
          sharing.dft_reductions.cuda_descriptor_uploads > 0 &&
          sharing.dft_reductions.cuda_plan_reuses > 0 &&
          sharing.dft_reductions.cuda_kernel_launches ==
              2 * sharing.dft_reductions.cuda_reduction_calls &&
          sharing.dft_reductions.cuda_result_device_to_host_bytes > 0 &&
          sharing.dft_reductions
                  .full_dft_device_to_host_bytes_avoided > 0 &&
          sharing.dft_reductions.mpi_allreduce_calls > 0 &&
          sharing.dft_reductions.mpi_allreduce_bytes > 0,
      "resident CUDA spectral consumers did not prove retained plans, "
      "two-kernel reductions, bounded result copies, or collectives");
}

double cw_benchmark_resolution() {
  const char *value = std::getenv("MEEP_GPU_TEST_CW_BENCHMARK_RESOLUTION");
  if (!value || !*value) return 32.0;
  errno = 0;
  char *end = nullptr;
  const double resolution = std::strtod(value, &end);
  require(!errno && end && !*end && std::isfinite(resolution) &&
              resolution >= 8.0 && resolution <= 256.0,
          "MEEP_GPU_TEST_CW_BENCHMARK_RESOLUTION must be finite and in "
          "[8,256]");
  return resolution;
}

double cw_benchmark_size() {
  const char *value = std::getenv("MEEP_GPU_TEST_CW_BENCHMARK_SIZE");
  if (!value || !*value) return 8.0;
  errno = 0;
  char *end = nullptr;
  const double size = std::strtod(value, &end);
  require(!errno && end && !*end && std::isfinite(size) &&
              size >= 4.0 && size <= 64.0,
          "MEEP_GPU_TEST_CW_BENCHMARK_SIZE must be finite and in [4,64]");
  return size;
}

run_result run_cw_case(gpu::backend_mode mode,
                       bool resident_solver = true,
                       bool benchmark_profile = false,
                       int fixed_iterations = 0,
                       bool capture_full_vector = true,
                       bool automatic_solver = false,
                       int requested_order = 0) {
  scoped_environment_override cw_solver(
      "MEEP_GPU_CW_SOLVER",
      automatic_solver
          ? "auto"
          : mode == gpu::backend_mode::cuda && resident_solver
                ? "resident"
                : "host");
  scoped_environment_override true_residual_diagnostic(
      "MEEP_GPU_CW_DIAGNOSTIC_TRUE_RESIDUAL",
      benchmark_profile ? "1" : nullptr);
  scoped_environment_override reliable_restart(
      "MEEP_GPU_CW_RELIABLE_RESTART",
      fixed_iterations > 0 ? "10" : benchmark_profile ? "20" : nullptr);
  gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();
  const double benchmark_size =
      benchmark_profile ? cw_benchmark_size() : 0.0;
  const double size_x = benchmark_profile ? benchmark_size : 1.8;
  const double size_y = benchmark_profile ? benchmark_size : 1.6;
  const double resolution =
      benchmark_profile ? cw_benchmark_resolution() : 8.0;
  const grid_volume gv = vol2d(size_x, size_y, resolution);
  structure s(gv, vacuum, pml(benchmark_profile ? 0.8 : 0.3));
  fields f(&s, 0.0, 0.0, true, 64, 64);
  continuous_src_time source(0.27);
  f.add_point_source(
      Ez, source,
      benchmark_profile ? vec(0.37 * size_x, 0.51 * size_y)
                        : vec(0.65, 0.8),
      0.8);
  all_wait();
  const double solve_start = wall_time();
  const int solver_order =
      requested_order > 0
          ? requested_order
          : fixed_iterations > 0 ? 2 : benchmark_profile ? 10 : 3;
  const bool converged = f.solve_cw(
      fixed_iterations > 0 ? 1e-7 : 1e-5,
      fixed_iterations > 0 ? fixed_iterations : 1500,
      solver_order);
  const double solve_seconds = wall_time() - solve_start;
  require(fixed_iterations > 0 ? !converged : converged,
          fixed_iterations > 0
              ? "fixed-work CW scaling probe converged unexpectedly"
              : "CW lifecycle regression failed to converge");

  run_result result;
  const vec points[] = {
      benchmark_profile ? vec(0.42 * size_x, 0.51 * size_y)
                        : vec(0.75, 0.8),
      benchmark_profile ? vec(0.58 * size_x, 0.43 * size_y)
                        : vec(1.05, 0.65),
      benchmark_profile ? vec(0.71 * size_x, 0.64 * size_y)
                        : vec(1.30, 1.05)};
  for (const vec &point : points) {
    result.samples.push_back(f.get_field(Ez, point));
    result.samples.push_back(f.get_field(Hx, point));
    result.samples.push_back(f.get_field(Hy, point));
  }
  if (capture_full_vector) {
    for (int chunk_index = 0; chunk_index < f.num_chunks; ++chunk_index)
      if (f.chunks[chunk_index]->is_mine())
        gpu::detail::sync_resident_cache_for_owner(f.chunks[chunk_index]);
    std::vector<gpu::detail::cw_field_vector_segment_fp32> cw_segments;
    std::vector<float> cw_packed;
    append_cw_test_segments(f, &cw_segments, &cw_packed);
    for (std::size_t index = 0; index < cw_packed.size(); index += 2)
      result.samples.push_back(std::complex<double>(
          cw_packed[index], cw_packed[index + 1]));
  }
  result.statistics = gpu::get_dispatch_statistics();
  result.resident = gpu::get_resident_statistics();
  result.field_updates = gpu::get_field_update_statistics();
  // A distributed workload completes at its slowest rank.  Rank-zero local
  // time can otherwise overstate multi-GPU scaling when another rank is the
  // straggler.
  result.workload_seconds = max_to_all(solve_seconds);
  aggregate_statistics(result);
  return result;
}

void require_resident_cw_breakdown_recovery() {
  scoped_environment_override cw_solver(
      "MEEP_GPU_CW_SOLVER", "resident");
  scoped_environment_override reliable_restart(
      "MEEP_GPU_CW_RELIABLE_RESTART", "10");
  gpu::set_backend(gpu::backend_mode::cuda);
  gpu::reset_dispatch_statistics();

  const grid_volume gv = vol2d(1.8, 1.6, 8.0);
  structure s(gv, vacuum, pml(0.3));
  fields f(&s, 0.0, 0.0, true, 64, 64);
  continuous_src_time source(0.27);
  f.add_point_source(Ez, source, vec(0.65, 0.8), 0.8);

  {
    // Only rank zero corrupts one reduction result. The vector backend's
    // finite consensus must turn this into the same -1 solver status on all
    // ranks without throwing through distributed solve_cw.
    scoped_environment_override injected_breakdown(
        "MEEP_GPU_TEST_CW_KRYLOV_RANK_BREAKDOWN", "1");
    require(!f.solve_cw(1e-5, 100, 10),
            "injected resident CW numerical breakdown reported success");
  }

  std::vector<gpu::detail::cw_field_vector_segment_fp32> segments;
  std::vector<float> values;
  for (int chunk_index = 0; chunk_index < f.num_chunks; ++chunk_index)
    if (f.chunks[chunk_index]->is_mine())
      gpu::detail::sync_resident_cache_for_owner(f.chunks[chunk_index]);
  append_cw_test_segments(f, &segments, &values);
  bool local_finite = true;
  for (float value : values)
    local_finite = local_finite && std::isfinite(value);
  require(and_to_all(local_finite),
          "resident CW breakdown published a nonfinite partial solution");

  // Reuse the same fields object. This catches stale resident sessions,
  // poisoned mirrors, or solve-state guards that merely make a fresh owner
  // appear healthy.
  require(f.solve_cw(1e-5, 1500, 2),
          "resident CW solver did not recover on the same fields object");
  segments.clear();
  values.clear();
  for (int chunk_index = 0; chunk_index < f.num_chunks; ++chunk_index)
    if (f.chunks[chunk_index]->is_mine())
      gpu::detail::sync_resident_cache_for_owner(f.chunks[chunk_index]);
  append_cw_test_segments(f, &segments, &values);
  local_finite = true;
  for (float value : values)
    local_finite = local_finite && std::isfinite(value);
  require(and_to_all(local_finite),
          "resident CW recovery produced a nonfinite solution");
  const gpu::dispatch_statistics statistics =
      gpu::get_dispatch_statistics();
  require(and_to_all(statistics.cuda_curl_calls > 0 &&
                     statistics.cpu_curl_calls == 0),
          "resident CW breakdown recovery silently used CPU curls");
}

run_result run_cylindrical_cw_case(gpu::backend_mode mode) {
  scoped_environment_override cw_solver(
      "MEEP_GPU_CW_SOLVER",
      mode == gpu::backend_mode::cuda ? "resident" : "host");
  gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();
  const grid_volume gv = volcyl(1.8, 1.6, 10.0);
  structure s(gv, vacuum, pml(0.25));
  fields f(&s, 0.0, 0.0, true, 32, 32);
  f.use_bloch(0.0);
  continuous_src_time source(0.27);
  f.add_point_source(
      Ez, source, veccyl(0.37, 0.83),
      std::complex<double>(0.8, -0.17));
  f.add_point_source(
      Ep, source, veccyl(0.53, 0.61),
      std::complex<double>(-0.21, 0.09));
  require(f.solve_cw(1e-5, 1500, 3),
          "cylindrical CW regression failed to converge");

  run_result result;
  const vec points[] = {
      veccyl(0.0, 0.47), veccyl(0.31, 0.73),
      veccyl(1.19, 1.08)};
  const component components[] = {Er, Ep, Ez};
  for (const vec &point : points)
    for (component c : components)
      result.samples.push_back(f.get_field(c, point));
  for (int chunk_index = 0; chunk_index < f.num_chunks; ++chunk_index)
    if (f.chunks[chunk_index]->is_mine())
      gpu::detail::sync_resident_cache_for_owner(f.chunks[chunk_index]);
  std::vector<gpu::detail::cw_field_vector_segment_fp32> segments;
  std::vector<float> packed;
  append_cw_test_segments(f, &segments, &packed);
  for (std::size_t index = 0; index < packed.size(); index += 2)
    result.samples.push_back(std::complex<double>(
        packed[index], packed[index + 1]));
  result.statistics = gpu::get_dispatch_statistics();
  result.resident = gpu::get_resident_statistics();
  result.field_updates = gpu::get_field_update_statistics();
  aggregate_statistics(result);
  return result;
}

void require_unsupported_cylindrical_bfast_is_atomic() {
  gpu::set_backend(gpu::backend_mode::cuda);
  gpu::reset_dispatch_statistics();

  const grid_volume gv = volcyl(2.0, 2.0, 10.0);
  structure s(gv, vacuum, no_pml());
  fields f(&s, 0.0, 0.0, true, 0, 0, {0.11, 0.0, 0.0});
  f.use_real_fields();
  gaussian_src_time source(0.3, 0.1);
  f.add_point_source(Ep, source, veccyl(0.7, 1.0), 1.0);

  bool threw = false;
  std::string message;
  try {
    f.step();
  }
  catch (const std::runtime_error &error) {
    threw = true;
    message = error.what();
  }
  require(
      threw,
      "required CUDA mode accepted unsupported cylindrical BFAST");
  require(
      message.find("cylindrical BFAST") != std::string::npos,
      "unsupported CUDA error did not identify cylindrical BFAST");
  require(f.time() == 0.0, "failed CUDA preflight advanced simulation time");
  const gpu::dispatch_statistics statistics = gpu::get_dispatch_statistics();
  require(statistics.cuda_curl_calls == 0 && statistics.cpu_curl_calls == 0,
          "failed CUDA preflight partially dispatched a curl");
}

void require_unsupported_polarization_is_atomic() {
  gpu::set_backend(gpu::backend_mode::cuda);
  gpu::reset_dispatch_statistics();

  const grid_volume gv = vol2d(1.6, 1.4, 10.0);
  structure s(gv, vacuum, no_pml());
  s.add_susceptibility(
      vacuum, E_stuff,
      noisy_lorentzian_susceptibility(0.05, 0.31, 0.07));
  fields f(&s);
  f.use_real_fields();
  gaussian_src_time source(0.3, 0.1);
  f.add_point_source(Ez, source, gv.center(), 1.0);

  bool threw = false;
  std::string message;
  try {
    f.step();
  }
  catch (const std::runtime_error &error) {
    threw = true;
    message = error.what();
  }
  require(threw,
          "required CUDA mode accepted an unsupported noisy polarization");
  require(message.find("polarization") != std::string::npos,
          "unsupported CUDA error did not identify polarization");
  require(f.time() == 0.0,
          "failed polarization preflight advanced simulation time");
  const gpu::dispatch_statistics statistics =
      gpu::get_dispatch_statistics();
  require(statistics.cuda_curl_calls == 0 &&
              statistics.cpu_curl_calls == 0,
          "failed polarization preflight partially dispatched a curl");
}

run_result run_automatic_polarization_fallback_case(
    gpu::backend_mode mode) {
  gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();

  const grid_volume gv = vol2d(1.8, 1.6, 10.0);
  structure s(gv, vacuum, no_pml());
  // add_susceptibility prepends states. Add the unsupported, deterministic
  // noisy subclass first so the standard Lorentzian performs CUDA work
  // before automatic mode must publish resident state for the CPU fallback.
  s.add_susceptibility(
      vacuum, E_stuff,
      noisy_lorentzian_susceptibility(0.0, 0.39, 0.06));
  s.add_susceptibility(
      vacuum, E_stuff,
      lorentzian_susceptibility(0.31, 0.05));
  fields f(&s, 0.0, 0.0, true, 64, 64);
  f.use_real_fields();
  gaussian_src_time source(0.29, 0.11);
  f.add_point_source(Ez, source, gv.center(), 0.8);
  for (int step = 0; step < 18; ++step)
    f.step();

  run_result result;
  const vec points[] = {
      gv.center(), vec(0.43, 0.57), vec(1.31, 1.07)};
  for (const vec &point : points)
    result.samples.push_back(f.get_field(Ez, point));
  result.samples.push_back(
      std::complex<double>(f.field_energy(), 0.0));
  result.statistics = gpu::get_dispatch_statistics();
  result.resident = gpu::get_resident_statistics();
  result.field_updates = gpu::get_field_update_statistics();
  result.polarizations = gpu::get_polarization_statistics();
  result.sources = gpu::get_source_statistics();
  result.boundaries = gpu::get_boundary_statistics();
  aggregate_statistics(result);
  return result;
}

void require_automatic_policy_boundaries() {
  constexpr std::uint64_t reference_free_memory =
      20ULL * 1024ULL * 1024ULL * 1024ULL;
  constexpr std::uint64_t reference_total_memory =
      24ULL * 1024ULL * 1024ULL * 1024ULL;
  const gpu::detail::automatic_device_policy_facts reference_device = {
      8, 6, 84, 1008000000000ULL,
      reference_free_memory, reference_total_memory};
  const gpu::detail::automatic_cuda_policy_decision below =
      gpu::detail::evaluate_automatic_cuda_policy(
          {524287u, 1u, reference_device, false, 0u});
  const gpu::detail::automatic_cuda_policy_decision at =
      gpu::detail::evaluate_automatic_cuda_policy(
          {524288u, 1u, reference_device, false, 0u});
  require(below.minimum_cells == 524288u && !below.use_cuda &&
              at.minimum_cells == 524288u && at.use_cuda,
          "automatic device policy reference crossover is not exact");

  const gpu::detail::automatic_cuda_policy_decision twice_cpu =
      gpu::detail::evaluate_automatic_cuda_policy(
          {1048576u, 2u, reference_device, false, 0u});
  const gpu::detail::automatic_cuda_policy_decision half_bandwidth =
      gpu::detail::evaluate_automatic_cuda_policy(
          {1048576u, 1u,
           {8, 6, 42, 504000000000ULL,
            reference_free_memory, reference_total_memory},
           false, 0u});
  require(twice_cpu.minimum_cells == 1048576u &&
              twice_cpu.use_cuda &&
              half_bandwidth.minimum_cells == 1048576u &&
              half_bandwidth.use_cuda,
          "automatic device policy did not scale with CPU threads and GPU "
          "memory bandwidth");

  const gpu::detail::automatic_cuda_policy_decision explicit_zero =
      gpu::detail::evaluate_automatic_cuda_policy(
          {0u, 64u,
           {6, 0, 1, 1u, reference_free_memory,
            reference_total_memory},
           true, 0u});
  const gpu::detail::automatic_cuda_policy_decision explicit_boundary =
      gpu::detail::evaluate_automatic_cuda_policy(
          {12345u, 1u, reference_device, true, 12345u});
  require(explicit_zero.explicit_minimum && explicit_zero.use_cuda &&
              explicit_zero.minimum_cells == 0u &&
              explicit_boundary.use_cuda &&
              explicit_boundary.minimum_cells == 12345u,
          "explicit automatic CUDA minimum did not override device policy");

  const gpu::detail::automatic_cuda_policy_decision sm_fallback =
      gpu::detail::evaluate_automatic_cuda_policy(
          {524288u, 1u,
           {8, 6, 84, 0u, reference_free_memory,
            reference_total_memory},
           false, 0u});
  require(sm_fallback.minimum_cells == 524288u &&
              sm_fallback.use_cuda,
          "automatic policy SM fallback changed the reference boundary");

  const gpu::detail::automatic_cuda_policy_decision saturated =
      gpu::detail::evaluate_automatic_cuda_policy(
          {std::numeric_limits<std::uint64_t>::max(),
           std::numeric_limits<std::uint64_t>::max(),
           {8, 6, 1, 1u, reference_free_memory,
            reference_total_memory},
           false, 0u});
  require(
      saturated.minimum_cells ==
              std::numeric_limits<std::uint64_t>::max() &&
          !saturated.use_cuda,
      "automatic device policy treated an unrepresentable threshold as "
      "reachable");

  const gpu::detail::automatic_cuda_policy_decision memory_pressure =
      gpu::detail::evaluate_automatic_cuda_policy(
          {524288u, 1u,
           {8, 6, 84, 1008000000000ULL,
            512ULL * 1024ULL * 1024ULL,
            1024ULL * 1024ULL * 1024ULL},
           false, 0u});
  require(!memory_pressure.memory_eligible &&
              !memory_pressure.use_cuda &&
              memory_pressure.estimated_required_memory_bytes >
                  memory_pressure.memory_available_after_reserve_bytes,
          "automatic policy admitted a workload without device-memory "
          "headroom");

  require(
      gpu::detail::automatic_cuda_host_only_minimum_cells(1u) ==
              524288u &&
          gpu::detail::automatic_cuda_host_only_minimum_cells(8u) ==
              4194304u &&
          gpu::detail::automatic_cuda_host_only_minimum_cells(16u) ==
              8388608u,
      "automatic host-only workload floor changed its reference scaling");

  bool rejected_zero_threads = false;
  bool rejected_zero_sms = false;
  bool rejected_invalid_memory = false;
  try {
    (void)gpu::detail::evaluate_automatic_cuda_policy(
        {1u, 0u, reference_device, false, 0u});
  }
  catch (const std::invalid_argument &) {
    rejected_zero_threads = true;
  }
  try {
    (void)gpu::detail::evaluate_automatic_cuda_policy(
        {1u, 1u,
         {8, 6, 0, 1u, reference_free_memory,
          reference_total_memory},
         false, 0u});
  }
  catch (const std::invalid_argument &) {
    rejected_zero_sms = true;
  }
  try {
    (void)gpu::detail::evaluate_automatic_cuda_policy(
        {1u, 1u, {8, 6, 1, 1u, 2u, 1u}, false, 0u});
  }
  catch (const std::invalid_argument &) {
    rejected_invalid_memory = true;
  }
  require(rejected_zero_threads && rejected_zero_sms &&
              rejected_invalid_memory,
          "automatic policy accepted invalid CPU/device facts");
}

void require_automatic_phase_batch_policy_boundaries() {
  constexpr std::size_t operations = 4;
  constexpr int multiprocessors = 84;
  constexpr std::size_t boundary_blocks =
      operations * static_cast<std::size_t>(multiprocessors) * 24u;
  require(
      gpu::detail::automatic_phase_batch_selected(
          operations, boundary_blocks, boundary_blocks / operations,
          multiprocessors) &&
          !gpu::detail::automatic_phase_batch_selected(
              operations, boundary_blocks + 1u,
              boundary_blocks / operations, multiprocessors),
      "automatic phase-batch block/SM boundary is not exact");
  require(
      !gpu::detail::automatic_phase_batch_selected(
          operations - 1u, 1u, 1u, multiprocessors) &&
          !gpu::detail::automatic_phase_batch_selected(
              operations, 0u, 1u, multiprocessors) &&
          !gpu::detail::automatic_phase_batch_selected(
              operations, 1u, 0u, multiprocessors) &&
          !gpu::detail::automatic_phase_batch_selected(
              operations, 1u, 1u, 0) &&
          !gpu::detail::automatic_phase_batch_selected(
              std::numeric_limits<std::size_t>::max(), 1u, 1u,
              multiprocessors),
      "automatic phase-batch policy accepted an underfilled or "
      "unrepresentable topology");
  const std::size_t per_operation_limit =
      static_cast<std::size_t>(multiprocessors) * 24u;
  require(
      !gpu::detail::automatic_phase_batch_selected(
          operations, boundary_blocks, per_operation_limit + 1u,
          multiprocessors),
      "automatic phase-batch policy accepted a pathologically skewed phase");
  require(
      !gpu::detail::automatic_dft_multi_monitor_batch_selected(0) &&
          !gpu::detail::automatic_dft_multi_monitor_batch_selected(3) &&
          !gpu::detail::automatic_dft_multi_monitor_batch_selected(4) &&
          !gpu::detail::automatic_dft_multi_monitor_batch_selected(4096) &&
          !gpu::detail::automatic_dft_multi_monitor_batch_selected(
              std::numeric_limits<std::size_t>::max()),
      "unqualified automatic multi-monitor DFT policy did not fail closed");

  {
    scoped_environment_override curl_enable(
        "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL", nullptr);
    scoped_environment_override curl_disable(
        "MEEP_GPU_DISABLE_PHASE_BATCHED_CURL", nullptr);
    scoped_environment_override update_enable(
        "MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH", nullptr);
    scoped_environment_override update_disable(
        "MEEP_GPU_DISABLE_PHASE_BATCHED_UPDATE_EH", nullptr);
    require(
        gpu::detail::phase_batched_curl_mode() ==
                gpu::detail::phase_batch_mode::automatic &&
            gpu::detail::phase_batched_update_eh_mode() ==
                gpu::detail::phase_batch_mode::automatic,
        "unset phase-batch controls did not select automatic mode");
  }
  {
    scoped_environment_override curl_enable(
        "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL", "1");
    scoped_environment_override curl_disable(
        "MEEP_GPU_DISABLE_PHASE_BATCHED_CURL", nullptr);
    require(
        gpu::detail::phase_batched_curl_mode() ==
            gpu::detail::phase_batch_mode::forced,
        "exact phase-batch opt-in did not force the kernel");
  }
  {
    scoped_environment_override curl_enable(
        "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL", "0");
    scoped_environment_override curl_disable(
        "MEEP_GPU_DISABLE_PHASE_BATCHED_CURL", nullptr);
    require(
        gpu::detail::phase_batched_curl_mode() ==
            gpu::detail::phase_batch_mode::disabled,
        "non-exact phase-batch opt-in was not disabled");
  }
  {
    scoped_environment_override curl_enable(
        "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL", "1");
    scoped_environment_override curl_disable(
        "MEEP_GPU_DISABLE_PHASE_BATCHED_CURL", "0");
    require(
        gpu::detail::phase_batched_curl_mode() ==
            gpu::detail::phase_batch_mode::disabled,
        "phase-batch disable control did not override forced mode");
  }
  {
    scoped_environment_override dft_enable(
        "MEEP_GPU_ENABLE_MULTI_MONITOR_DFT_BATCH", nullptr);
    scoped_environment_override dft_disable(
        "MEEP_GPU_DISABLE_MULTI_MONITOR_DFT_BATCH", nullptr);
    require(
        gpu::detail::dft_multi_monitor_batch_mode() ==
            gpu::detail::phase_batch_mode::automatic,
        "unset multi-monitor DFT controls did not select automatic mode");
  }
  {
    scoped_environment_override dft_enable(
        "MEEP_GPU_ENABLE_MULTI_MONITOR_DFT_BATCH", "1");
    scoped_environment_override dft_disable(
        "MEEP_GPU_DISABLE_MULTI_MONITOR_DFT_BATCH", nullptr);
    require(
        gpu::detail::dft_multi_monitor_batch_mode() ==
            gpu::detail::phase_batch_mode::forced,
        "exact multi-monitor DFT opt-in did not force the kernel");
  }
  {
    scoped_environment_override dft_enable(
        "MEEP_GPU_ENABLE_MULTI_MONITOR_DFT_BATCH", "true");
    scoped_environment_override dft_disable(
        "MEEP_GPU_DISABLE_MULTI_MONITOR_DFT_BATCH", nullptr);
    require(
        gpu::detail::dft_multi_monitor_batch_mode() ==
            gpu::detail::phase_batch_mode::disabled,
        "non-exact multi-monitor DFT opt-in was not disabled");
  }
  {
    scoped_environment_override dft_enable(
        "MEEP_GPU_ENABLE_MULTI_MONITOR_DFT_BATCH", "1");
    scoped_environment_override dft_disable(
        "MEEP_GPU_DISABLE_MULTI_MONITOR_DFT_BATCH", "0");
    require(
        gpu::detail::dft_multi_monitor_batch_mode() ==
            gpu::detail::phase_batch_mode::disabled,
        "multi-monitor DFT disable control did not override forced mode");
  }
}

void require_automatic_polarization_owner_fallback_is_atomic() {
  const run_result cpu = run_automatic_polarization_fallback_case(
      gpu::backend_mode::cpu);
  const char *saved_auto_minimum =
      std::getenv("MEEP_GPU_AUTO_MIN_CELLS");
  const bool had_saved_auto_minimum = saved_auto_minimum != nullptr;
  const std::string saved_auto_minimum_value =
      saved_auto_minimum ? saved_auto_minimum : "";
  // This regression exercises mixed phase fallback semantics, not the
  // automatic size policy, so make its intentionally tiny domain a CUDA
  // candidate. Whole-owner feature preflight must still select CPU before
  // any CUDA phase because the noisy susceptibility is unsupported.
  setenv("MEEP_GPU_AUTO_MIN_CELLS", "0", 1);
  const run_result automatic =
      run_automatic_polarization_fallback_case(
          gpu::backend_mode::automatic);
  if (had_saved_auto_minimum)
    setenv(
        "MEEP_GPU_AUTO_MIN_CELLS",
        saved_auto_minimum_value.c_str(), 1);
  else
    unsetenv("MEEP_GPU_AUTO_MIN_CELLS");
  compare_results(cpu, automatic, two_dimensional_case);
  require(gpu::active_backend() == gpu::backend_mode::cpu,
          "unsupported automatic owner did not select CPU");
  require(automatic.statistics.cpu_curl_calls > 0 &&
              automatic.statistics.cuda_curl_calls == 0 &&
              automatic.field_updates.cpu_update_eh_calls > 0 &&
              automatic.field_updates.cuda_update_eh_calls == 0 &&
              automatic.polarizations.cpu_update_calls > 0 &&
              automatic.polarizations.cuda_update_calls == 0 &&
              automatic.sources.cuda_update_calls == 0 &&
              automatic.boundaries.cuda_update_calls == 0 &&
              automatic.statistics.host_to_device_bytes == 0 &&
              automatic.statistics.device_to_host_bytes == 0 &&
              automatic.resident.device_buffer_allocations == 0,
          "automatic unsupported-feature preflight was not an atomic "
          "whole-owner CPU decision");
}

void require_automatic_owner_revalidates_environment() {
  require(count_processors() == 1,
          "automatic environment revalidation is singleton-only");
  scoped_environment_override initial_minimum(
      "MEEP_GPU_AUTO_MIN_CELLS", "999999999");
  gpu::set_backend(gpu::backend_mode::automatic);
  gpu::reset_dispatch_statistics();

  const grid_volume gv = vol2d(1.8, 1.6, 10.0);
  structure s(gv, vacuum, no_pml());
  fields f(&s, 0.0, 0.0, true, 64, 64);
  f.use_real_fields();
  gaussian_src_time source(0.29, 0.11);
  f.add_point_source(Ez, source, gv.center(), 0.8);
  f.step();
  require(!f.gpu_cuda_execution_selected() &&
              gpu::active_backend() == gpu::backend_mode::cpu,
          "large automatic threshold did not cache a CPU owner");
  const double time_before_invalid_configuration = f.time();

  bool invalid_configuration_rejected = false;
  {
    scoped_environment_override invalid_oversubscription(
        "MEEP_GPU_ALLOW_OVERSUBSCRIBE", "invalid-after-cache");
    require(
        std::string(f.gpu_execution_diagnostic()).find("invalidated") !=
            std::string::npos,
        "owner query did not invalidate a cached CPU decision after an "
        "environment change");
    try {
      f.step();
    }
    catch (const std::runtime_error &error) {
      invalid_configuration_rejected =
          std::string(error.what()).find(
              "MEEP_GPU_ALLOW_OVERSUBSCRIBE") != std::string::npos;
    }
  }
  require(invalid_configuration_rejected,
          "cached automatic CPU owner hid a malformed oversubscription "
          "configuration");
  require(f.time() == time_before_invalid_configuration,
          "rejected dynamic automatic configuration advanced time");

  bool invalid_device_rejected = false;
  {
    scoped_environment_override invalid_device(
        "MEEP_GPU_DEVICE", "invalid-after-cache");
    try {
      f.step();
    }
    catch (const std::runtime_error &error) {
      invalid_device_rejected =
          std::string(error.what()).find("MEEP_GPU_DEVICE") !=
          std::string::npos;
    }
  }
  require(invalid_device_rejected,
          "cached automatic CPU owner hid a malformed explicit device");
  require(f.time() == time_before_invalid_configuration,
          "rejected dynamic device configuration advanced time");

  gpu::reset_dispatch_statistics();
  {
    scoped_environment_override force_candidate(
        "MEEP_GPU_AUTO_MIN_CELLS", "0");
    f.step();
    const gpu::dispatch_statistics dispatch =
        gpu::get_dispatch_statistics();
    require(f.gpu_cuda_execution_selected() &&
                gpu::active_backend() == gpu::backend_mode::cuda &&
                dispatch.cuda_curl_calls > 0 &&
                dispatch.cpu_curl_calls == 0,
            "changed automatic minimum reused the stale CPU owner instead "
            "of selecting CUDA");

    const char *configured_transport =
        std::getenv("MEEP_GPU_MPI_TRANSPORT");
    const std::string changed_transport =
        configured_transport && std::string(configured_transport) == "pinned"
            ? "auto"
            : "pinned";
    const double time_before_changed_transport = f.time();
    bool changed_transport_rejected = false;
    gpu::reset_dispatch_statistics();
    {
      scoped_environment_override changed_transport_policy(
          "MEEP_GPU_MPI_TRANSPORT", changed_transport.c_str());
      try {
        f.step();
      }
      catch (const std::runtime_error &error) {
        changed_transport_rejected =
            std::string(error.what()).find("changed after") !=
            std::string::npos;
      }
    }
    const gpu::dispatch_statistics after_changed_transport =
        gpu::get_dispatch_statistics();
    require(changed_transport_rejected &&
                f.time() == time_before_changed_transport &&
                after_changed_transport.cpu_curl_calls == 0 &&
                after_changed_transport.cuda_curl_calls == 0,
            "cached automatic CUDA owner accepted a dynamic valid MPI "
            "transport change or dispatched a partial step");

    const char *configured_completion =
        std::getenv("MEEP_GPU_MPI_COMPLETION");
    const std::string changed_completion =
        configured_completion &&
                std::string(configured_completion) == "waitall"
            ? "waitsome"
            : "waitall";
    const double time_before_changed_completion = f.time();
    bool changed_completion_rejected = false;
    gpu::reset_dispatch_statistics();
    {
      scoped_environment_override changed_completion_policy(
          "MEEP_GPU_MPI_COMPLETION", changed_completion.c_str());
      try {
        f.step();
      }
      catch (const std::runtime_error &error) {
        changed_completion_rejected =
            std::string(error.what()).find("changed after") !=
            std::string::npos;
      }
    }
    const gpu::dispatch_statistics after_changed_completion =
        gpu::get_dispatch_statistics();
    require(changed_completion_rejected &&
                f.time() == time_before_changed_completion &&
                after_changed_completion.cpu_curl_calls == 0 &&
                after_changed_completion.cuda_curl_calls == 0,
            "cached automatic CUDA owner accepted a dynamic valid MPI "
            "completion-policy change or dispatched a partial step");

    gpu::reset_dispatch_statistics();
    f.step();
    const gpu::dispatch_statistics after_transport_restore =
        gpu::get_dispatch_statistics();
    require(f.gpu_cuda_execution_selected() &&
                after_transport_restore.cuda_curl_calls > 0 &&
                after_transport_restore.cpu_curl_calls == 0,
            "automatic CUDA owner did not recover after restoring MPI "
            "transport and completion policy");
  }
  gpu::set_backend(gpu::backend_mode::cpu);
}

void require_failed_dynamic_environment_preserves_device_claim() {
  const int world_size = count_processors();
  require(world_size == 2,
          "dynamic device-claim regression requires exactly two MPI ranks");

  scoped_environment_override force_candidate(
      "MEEP_GPU_AUTO_MIN_CELLS", "0");
  scoped_environment_override automatic_device(
      "MEEP_GPU_DEVICE", nullptr);
  scoped_environment_override exclusive_devices(
      "MEEP_GPU_ALLOW_OVERSUBSCRIBE", "0");
  scoped_environment_override pinned_transport(
      "MEEP_GPU_MPI_TRANSPORT", "pinned");
  scoped_environment_override waitsome_completion(
      "MEEP_GPU_MPI_COMPLETION", "waitsome");

  const int group = divide_parallel_processes(world_size);
  bool local_success = true;
  std::unique_ptr<structure> owner_structure;
  std::unique_ptr<fields> owner_fields;

  if (group == 0) {
    try {
      gpu::set_backend(gpu::backend_mode::automatic);
      const grid_volume gv = vol2d(1.8, 1.6, 10.0);
      owner_structure.reset(new structure(gv, vacuum, no_pml()));
      owner_fields.reset(
          new fields(owner_structure.get(), 0.0, 0.0, true, 64, 64));
      owner_fields->use_real_fields();
      gaussian_src_time source(0.29, 0.11);
      owner_fields->add_point_source(Ez, source, gv.center(), 0.8);
      owner_fields->step();
      local_success =
          owner_fields->gpu_cuda_execution_selected() &&
          gpu::selected_device() == 0;

      const double time_before_invalid_configuration = owner_fields->time();
      bool invalid_configuration_rejected = false;
      {
        scoped_environment_override invalid_oversubscription(
            "MEEP_GPU_ALLOW_OVERSUBSCRIBE", "invalid-after-cuda-cache");
        try {
          owner_fields->step();
        }
        catch (const std::runtime_error &error) {
          invalid_configuration_rejected =
              std::string(error.what()).find(
                  "MEEP_GPU_ALLOW_OVERSUBSCRIBE") != std::string::npos;
        }
      }
      local_success =
          local_success && invalid_configuration_rejected &&
          owner_fields->time() == time_before_invalid_configuration;
    }
    catch (...) {
      local_success = false;
    }
  }

  begin_global_communications();
  all_wait();
  end_global_communications();

  if (group == 0) {
    try {
      gpu::reset_dispatch_statistics();
      owner_fields->step();
      const gpu::dispatch_statistics restored =
          gpu::get_dispatch_statistics();
      local_success =
          local_success && owner_fields->gpu_cuda_execution_selected() &&
          restored.cuda_curl_calls > 0 && restored.cpu_curl_calls == 0;
    }
    catch (...) {
      local_success = false;
    }
  }
  else {
    bool competing_claim_rejected = false;
    try {
      gpu::set_backend(gpu::backend_mode::automatic);
      gpu::select_device(0);
      const grid_volume gv = vol2d(1.8, 1.6, 10.0);
      structure competing_structure(gv, vacuum, no_pml());
      fields competing_fields(
          &competing_structure, 0.0, 0.0, true, 64, 64);
      competing_fields.use_real_fields();
      gaussian_src_time source(0.31, 0.10);
      competing_fields.add_point_source(
          Ez, source, gv.center(), 0.7);
      competing_fields.step();
    }
    catch (const std::runtime_error &error) {
      competing_claim_rejected =
          std::string(error.what()).find(
              "already claimed by another Meep subcommunicator") !=
          std::string::npos;
    }
    catch (...) {}
    local_success = local_success && competing_claim_rejected;
  }

  begin_global_communications();
  all_wait();
  const bool every_rank_succeeded = and_to_all(local_success);
  end_global_communications();

  try {
    gpu::set_backend(gpu::backend_mode::cpu);
  }
  catch (...) {
    local_success = false;
  }
  owner_fields.reset();
  owner_structure.reset();
  end_divide_parallel();

  require(every_rank_succeeded && local_success,
          "failed dynamic CUDA configuration released the live owner's "
          "MPI-world device claim or the restored owner did not resume CUDA");
}

void require_automatic_feature_fallback_recovers() {
  require(count_processors() == 1,
          "automatic feature-fallback recovery is singleton-only");
  scoped_environment_override force_candidate(
      "MEEP_GPU_AUTO_MIN_CELLS", "0");
  gpu::set_backend(gpu::backend_mode::automatic);
  gpu::reset_dispatch_statistics();

  const grid_volume gv = vol2d(1.8, 1.6, 10.0);
  structure s(gv, vacuum, no_pml());
  s.add_susceptibility(
      vacuum, E_stuff,
      noisy_lorentzian_susceptibility(0.0, 0.39, 0.06));
  s.add_susceptibility(
      vacuum, E_stuff,
      lorentzian_susceptibility(0.31, 0.05));
  fields f(&s, 0.0, 0.0, true, 64, 64);
  f.use_real_fields();
  gaussian_src_time source(0.29, 0.11);
  f.add_point_source(Ez, source, gv.center(), 0.8);
  f.step();
  require(!f.gpu_cuda_execution_selected() &&
              gpu::active_backend() == gpu::backend_mode::cpu &&
              std::string(f.gpu_execution_diagnostic()).find(
                  "feature preflight failed") != std::string::npos,
          "unsupported automatic feature did not cache a feature-specific "
          "CPU fallback");

  struct detached_state_guard {
    struct entry {
      polarization_state **link;
      polarization_state *node;
    };
    std::vector<entry> entries;
    ~detached_state_guard() {
      for (auto item = entries.rbegin(); item != entries.rend(); ++item) {
        item->node->next = *item->link;
        *item->link = item->node;
      }
    }
  } detached;
  for (int chunk_index = 0; chunk_index < f.num_chunks; ++chunk_index) {
    fields_chunk *chunk = f.chunks[chunk_index];
    if (!chunk->is_mine()) continue;
    polarization_state **link = &chunk->pol[E_stuff];
    while (*link) {
      polarization_state *candidate = *link;
      if (candidate->s &&
          typeid(*candidate->s) ==
              typeid(noisy_lorentzian_susceptibility)) {
        *link = candidate->next;
        candidate->next = nullptr;
        detached.entries.push_back({link, candidate});
        break;
      }
      link = &candidate->next;
    }
  }
  require(!detached.entries.empty(),
          "feature-fallback recovery fixture found no noisy polarization");

  gpu::reset_dispatch_statistics();
  f.step();
  const gpu::dispatch_statistics dispatch =
      gpu::get_dispatch_statistics();
  const gpu::polarization_statistics polarizations =
      gpu::get_polarization_statistics();
  require(f.gpu_cuda_execution_selected() &&
              gpu::active_backend() == gpu::backend_mode::cuda &&
              dispatch.cuda_curl_calls > 0 &&
              dispatch.cpu_curl_calls == 0 &&
              polarizations.cuda_update_calls > 0 &&
              polarizations.cpu_update_calls == 0,
          "automatic owner remained stuck on CPU after its unsupported "
          "feature was removed");
  gpu::set_backend(gpu::backend_mode::cpu);
}

void require_interleaved_automatic_owner_decisions() {
  require(count_processors() == 1,
          "interleaved automatic-owner regression is singleton-only");
  const std::uint64_t live_before =
      gpu::get_live_resident_device_buffers();
  {
    const grid_volume small_gv = vol2d(1.8, 1.6, 10.0);
    const grid_volume large_gv = vol2d(8.0, 8.0, 12.0);
    require(large_gv.ntot() > small_gv.ntot() + 1,
            "automatic-owner test volumes are not ordered");
    const size_t threshold =
        small_gv.ntot() +
        (large_gv.ntot() - small_gv.ntot()) / 2;
    const std::string threshold_text = std::to_string(threshold);
    scoped_environment_override automatic_minimum(
        "MEEP_GPU_AUTO_MIN_CELLS", threshold_text.c_str());
    gpu::set_backend(gpu::backend_mode::automatic);

    structure small_structure(small_gv, vacuum, no_pml());
    structure large_structure(large_gv, vacuum, no_pml());
    fields small(&small_structure, 0.0, 0.0, true, 64, 64);
    fields large(&large_structure, 0.0, 0.0, true, 64, 64);
    small.use_real_fields();
    large.use_real_fields();
    gaussian_src_time small_source(0.27, 0.09);
    gaussian_src_time large_source(0.31, 0.11);
    small.add_point_source(Ez, small_source, small_gv.center(), 0.7);
    large.add_point_source(Ez, large_source, large_gv.center(), 0.8);

    small.step();
    require(gpu::active_backend() == gpu::backend_mode::cpu,
            "small automatic owner did not select CPU");
    require(!small.gpu_cuda_execution_selected() &&
                std::string(small.gpu_execution_diagnostic()).find(
                    "selected CPU") != std::string::npos,
            "small fields owner did not retain its CPU decision");
    for (int warmup = 0; warmup < 4; ++warmup)
      large.step();
    require(gpu::active_backend() == gpu::backend_mode::cuda,
            "large automatic owner did not select CUDA");
    require(large.gpu_cuda_execution_selected() &&
                std::string(large.gpu_execution_diagnostic()).find(
                    "selected CUDA") != std::string::npos,
            "large fields owner did not retain its CUDA decision");
    const std::uint64_t warm_live =
        gpu::get_live_resident_device_buffers();
    require(warm_live > live_before,
            "large automatic owner created no resident CUDA state");

    gpu::reset_dispatch_statistics();
    small.step();
    const gpu::dispatch_statistics after_small =
        gpu::get_dispatch_statistics();
    require(gpu::active_backend() == gpu::backend_mode::cpu &&
                after_small.cpu_curl_calls > 0 &&
                after_small.cuda_curl_calls == 0 &&
                after_small.host_to_device_bytes == 0 &&
                after_small.device_to_host_bytes == 0 &&
                gpu::get_live_resident_device_buffers() == warm_live,
            "cached small-owner CPU activation disturbed the live CUDA "
            "owner");
    require(large.gpu_cuda_execution_selected(),
            "CPU owner activation overwrote another owner's CUDA decision");

    large.step();
    const gpu::dispatch_statistics after_large =
        gpu::get_dispatch_statistics();
    require(gpu::active_backend() == gpu::backend_mode::cuda &&
                after_large.cuda_curl_calls > 0 &&
                gpu::get_live_resident_device_buffers() == warm_live,
            "cached large-owner CUDA activation rebuilt or lost resident "
            "state after an interleaved CPU step");

    small.step();
    require(gpu::active_backend() == gpu::backend_mode::cpu &&
                gpu::get_live_resident_device_buffers() == warm_live,
            "second small-owner activation invalidated another owner's "
            "resident state");

    // Keep two independent CUDA owners warm, then explicitly destroy one
    // chunk cache.  The peer owner must reuse all of its persistent phase
    // plans without an allocation or upload.  This catches the historical
    // global-plan clear hidden by a CPU owner which never owned a cache.
    structure peer_structure(large_gv, vacuum, no_pml());
    fields peer(&peer_structure, 0.0, 0.0, true, 64, 64);
    peer.use_real_fields();
    gaussian_src_time peer_source(0.33, 0.10);
    peer.add_point_source(Ez, peer_source, large_gv.center(), 0.6);
    for (int warmup = 0; warmup < 5; ++warmup)
      peer.step();
    require(peer.gpu_cuda_execution_selected(),
            "peer automatic owner did not select CUDA");
    gpu::reset_dispatch_statistics();
    peer.step();
    const gpu::resident_statistics stable_peer =
        gpu::get_resident_statistics();
    require(stable_peer.device_buffer_allocations == 0,
            "peer CUDA owner was not warm before cache-isolation test");

    gpu::detail::destroy_resident_cache_for_owner(large.chunks[0]);
    const std::uint64_t live_after_foreign_destroy =
        gpu::get_live_resident_device_buffers();
    gpu::reset_dispatch_statistics();
    peer.step();
    const gpu::dispatch_statistics isolated_peer_dispatch =
        gpu::get_dispatch_statistics();
    const gpu::resident_statistics isolated_peer_resident =
        gpu::get_resident_statistics();
    require(peer.gpu_cuda_execution_selected() &&
                isolated_peer_dispatch.cuda_curl_calls > 0 &&
                isolated_peer_dispatch.cpu_curl_calls == 0 &&
                isolated_peer_dispatch.host_to_device_bytes == 0 &&
                isolated_peer_resident.device_buffer_allocations == 0 &&
                gpu::get_live_resident_device_buffers() ==
                    live_after_foreign_destroy,
            "destroying one CUDA owner cache invalidated another owner's "
            "persistent execution plans");

    gpu::set_backend(gpu::backend_mode::cpu);
    require(!peer.gpu_cuda_execution_selected() &&
                std::string(peer.gpu_execution_diagnostic()).find(
                    "invalidated") != std::string::npos,
            "fields owner API exposed a stale decision after a process "
            "backend generation change");
  }
  require(gpu::get_live_resident_device_buffers() == live_before,
          "interleaved automatic owners leaked resident device buffers");
}

void require_resized_dft_preflight_is_atomic() {
  gpu::set_backend(gpu::backend_mode::cuda);

  const grid_volume gv = vol2d(1.6, 1.4, 10.0);
  structure s(gv, vacuum, no_pml());
  fields f(&s);
  f.use_real_fields();
  gaussian_src_time source(0.3, 0.1);
  f.add_point_source(Ez, source, gv.center(), 1.0);
  component components[] = {Ez};
  dft_fields monitor =
      f.add_dft_fields(components, 1, f.v, 0.18, 0.42, 3, true, 1);
  require(monitor.chunks,
          "DFT resize atomicity regression found no monitor chunks");
  const auto require_no_dispatch = [](const char *diagnostic) {
    const gpu::dispatch_statistics dispatch =
        gpu::get_dispatch_statistics();
    const gpu::field_update_statistics fields =
        gpu::get_field_update_statistics();
    const gpu::polarization_statistics polarizations =
        gpu::get_polarization_statistics();
    const gpu::source_statistics sources =
        gpu::get_source_statistics();
    const gpu::boundary_statistics boundaries =
        gpu::get_boundary_statistics();
    const gpu::dft_statistics dfts = gpu::get_dft_statistics();
    require(dispatch.cpu_curl_calls == 0 &&
                dispatch.cuda_curl_calls == 0 &&
                dispatch.host_to_device_bytes == 0 &&
                dispatch.device_to_host_bytes == 0 &&
                fields.cpu_update_eh_calls == 0 &&
                fields.cuda_update_eh_calls == 0 &&
                polarizations.cpu_update_calls == 0 &&
                polarizations.cuda_update_calls == 0 &&
                sources.cpu_update_calls == 0 &&
                sources.cuda_update_calls == 0 &&
                boundaries.cpu_update_calls == 0 &&
                boundaries.cuda_update_calls == 0 &&
                dfts.cpu_update_calls == 0 &&
                dfts.cuda_update_calls == 0,
            diagnostic);
  };

  // Warm the backend and feature-preflight caches before changing public DFT
  // metadata.  This makes the regression prove that a same-address monitor
  // whose omega size changes invalidates the cached eligibility decision.
  f.step();
  const double time_before = f.time();
  const double dft_before = f.dft_norm();
  const std::complex<double> field_before =
      f.get_field(Ez, gv.center());
  monitor.chunks->omega.push_back(0.57);
  gpu::reset_dispatch_statistics();

  bool threw = false;
  std::string message;
  try {
    f.step();
  }
  catch (const std::runtime_error &error) {
    threw = true;
    message = error.what();
  }
  require(threw,
          "required CUDA mode accepted a resized dft_chunk::omega");
  require(message.find("dft_chunk::omega") != std::string::npos,
          "DFT resize preflight error did not identify dft_chunk::omega");
  require(f.time() == time_before,
          "failed DFT resize preflight advanced simulation time");

  // Capture dispatch counters before the state-verification queries below;
  // those legitimate host consumers publish data from the warmed resident
  // cache and must not be attributed to the rejected step.
  require_no_dispatch(
      "failed DFT resize preflight performed a CPU/CUDA dispatch");

  monitor.chunks->omega.pop_back();
  require(f.dft_norm() == dft_before,
          "failed DFT resize preflight modified DFT state");
  require(f.get_field(Ez, gv.center()) == field_before,
          "failed DFT resize preflight modified field state");

  // The failed omega attempt deliberately clears preflight consensus. Run a
  // successful step first so this N mutation must invalidate a genuinely
  // warm same-address cache entry rather than merely exercising full
  // uncached preflight.
  f.step();
  const double n_time_before = f.time();
  const double n_dft_before = f.dft_norm();
  const std::complex<double> n_field_before =
      f.get_field(Ez, gv.center());
  const std::size_t original_N = monitor.chunks->N;
  monitor.chunks->N = original_N + 1;
  gpu::reset_dispatch_statistics();

  threw = false;
  message.clear();
  try {
    f.step();
  }
  catch (const std::runtime_error &error) {
    threw = true;
    message = error.what();
  }
  require(threw, "required CUDA mode accepted a changed dft_chunk::N");
  require(message.find("dft_chunk::N") != std::string::npos,
          "DFT N preflight error did not identify dft_chunk::N");
  require(f.time() == n_time_before,
          "failed cached DFT N preflight advanced time");
  require_no_dispatch(
      "failed cached DFT N preflight performed a CPU/CUDA dispatch");
  monitor.chunks->N = original_N;
  require(f.dft_norm() == n_dft_before &&
              f.get_field(Ez, gv.center()) == n_field_before,
          "failed cached DFT N preflight modified simulation state");

  // Independently prove ordering while phasing is pending. changed_materials
  // intentionally forces full preflight here; the assertion is that no H/E,
  // boundary, phase counter, or time state changes before that rejection.
  structure replacement(gv, vacuum, no_pml());
  const int requested_phase_steps =
      f.phase_in_material(&replacement, 2.0 * f.dt);
  require(requested_phase_steps > 0 && f.is_phasing(),
          "DFT N atomicity regression did not enter material phasing");
  const int phase_before = f.phasein_time;
  monitor.chunks->N = original_N + 1;
  gpu::reset_dispatch_statistics();
  threw = false;
  message.clear();
  try {
    f.step();
  }
  catch (const std::runtime_error &error) {
    threw = true;
    message = error.what();
  }
  require(threw && message.find("dft_chunk::N") != std::string::npos,
          "phasing preflight accepted or misreported changed dft_chunk::N");
  require(f.time() == n_time_before && f.phasein_time == phase_before,
          "failed phasing preflight advanced time or phase state");
  require_no_dispatch(
      "failed DFT N/phasing preflight performed a CPU/CUDA dispatch");
  monitor.chunks->N = original_N;
  require(f.dft_norm() == n_dft_before &&
              f.get_field(Ez, gv.center()) == n_field_before,
          "failed DFT N/phasing preflight modified simulation state");
}

void require_mutated_susceptibility_preflight_is_atomic() {
  gpu::set_backend(gpu::backend_mode::cuda);
  const grid_volume gv = vol2d(1.8, 1.6, 10.0);
  structure s(gv, vacuum, no_pml());
  s.add_susceptibility(
      gyrotropic_sigma, E_stuff,
      gyrotropic_susceptibility(
          vec(0.0, 0.0, 0.17), 0.74, 0.025, 0.0,
          GYROTROPIC_LORENTZIAN));
  fields f(&s, 0.0, 0.0, true, 64, 64);
  f.use_real_fields();
  gaussian_src_time source(0.29, 0.11);
  f.add_point_source(Ex, source, vec(0.61, 0.73), 0.8);
  f.add_point_source(Ey, source, vec(1.19, 0.91), -0.2);
  f.step();

  susceptibility *mutated = nullptr;
  component mutated_component = NO_COMPONENT;
  direction mutated_direction = NO_DIRECTION;
  realnum *aliased_diagonal = nullptr;
  for (int chunk_index = 0;
       chunk_index < f.num_chunks && !mutated; ++chunk_index) {
    fields_chunk *chunk = f.chunks[chunk_index];
    if (!chunk->is_mine()) continue;
    for (polarization_state *state = chunk->pol[E_stuff];
         state && !mutated; state = state->next) {
      susceptibility *candidate =
          const_cast<susceptibility *>(state->s);
      if (!candidate) continue;
      FOR_COMPONENTS(c) {
        const direction primary = component_direction(c);
        if (primary != X && primary != Y && primary != Z) continue;
        const direction off_diagonal =
            cycle_direction(chunk->gv.dim, primary, 1);
        if (candidate->sigma[c][primary] &&
            !candidate->sigma[c][off_diagonal]) {
          mutated = candidate;
          mutated_component = c;
          mutated_direction = off_diagonal;
          aliased_diagonal = candidate->sigma[c][primary];
          break;
        }
      }
    }
  }
  require(mutated && mutated_component != NO_COMPONENT &&
              mutated_direction != NO_DIRECTION && aliased_diagonal,
          "susceptibility mutation regression found no sigma topology");

  const double time_before = f.time();
  const std::complex<double> field_before =
      f.get_field(Ex, vec(0.61, 0.73));
  mutated->sigma[mutated_component][mutated_direction] =
      aliased_diagonal;
  gpu::reset_dispatch_statistics();
  bool threw = false;
  std::string message;
  try {
    f.step();
  }
  catch (const std::runtime_error &error) {
    threw = true;
    message = error.what();
  }
  catch (...) {
    mutated->sigma[mutated_component][mutated_direction] = nullptr;
    throw;
  }
  // Restore ownership before any assertion can throw; the temporary alias
  // must never reach susceptibility destruction and be deleted twice.
  mutated->sigma[mutated_component][mutated_direction] = nullptr;

  require(threw,
          "required CUDA mode accepted mutated gyrotropic sigma topology");
  require(message.find("anisotropic sigma") != std::string::npos,
          "susceptibility topology preflight reported the wrong reason");
  require(f.time() == time_before,
          "failed susceptibility preflight advanced simulation time");
  const gpu::dispatch_statistics dispatch =
      gpu::get_dispatch_statistics();
  const gpu::field_update_statistics fields =
      gpu::get_field_update_statistics();
  const gpu::polarization_statistics polarizations =
      gpu::get_polarization_statistics();
  const gpu::source_statistics sources =
      gpu::get_source_statistics();
  const gpu::boundary_statistics boundaries =
      gpu::get_boundary_statistics();
  const gpu::dft_statistics dfts = gpu::get_dft_statistics();
  require(dispatch.cpu_curl_calls == 0 &&
              dispatch.cuda_curl_calls == 0 &&
              dispatch.host_to_device_bytes == 0 &&
              dispatch.device_to_host_bytes == 0 &&
              fields.cpu_update_eh_calls == 0 &&
              fields.cuda_update_eh_calls == 0 &&
              polarizations.cpu_update_calls == 0 &&
              polarizations.cuda_update_calls == 0 &&
              sources.cpu_update_calls == 0 &&
              sources.cuda_update_calls == 0 &&
              boundaries.cpu_update_calls == 0 &&
              boundaries.cuda_update_calls == 0 &&
              dfts.cpu_update_calls == 0 &&
              dfts.cuda_update_calls == 0,
          "failed susceptibility topology preflight performed a dispatch");
  require(f.get_field(Ex, vec(0.61, 0.73)) == field_before,
          "failed susceptibility topology preflight modified fields");
}

void require_integrated_source_cache_lifecycle() {
  const auto run_combinable_source = [](gpu::backend_mode mode) {
    gpu::set_backend(mode);
    gpu::reset_dispatch_statistics();
    const grid_volume gv = vol2d(1.6, 1.4, 10.0);
    structure s(gv, vacuum, no_pml());
    fields f(&s, 0.0, 0.0, true, 64, 64);
    f.use_real_fields();

    gaussian_src_time source(0.27, 0.11);
    const vec source_point(0.47, 0.63);
    f.add_point_source(Ez, source, source_point, 0.4);
    f.require_source_components();
    f.step();

    std::size_t local_volume_count_before = 0;
    std::size_t local_point_count_before = 0;
    double local_amplitude_l1_before = 0.0;
    for (int chunk = 0; chunk < f.num_chunks; ++chunk)
      FOR_FIELD_TYPES(ft) for (const src_vol &profile :
                                  f.chunks[chunk]->get_sources(ft)) {
        if (profile.c != Ez) continue;
        ++local_volume_count_before;
        local_point_count_before += profile.num_points();
        for (std::size_t point = 0; point < profile.num_points(); ++point)
          local_amplitude_l1_before += std::abs(profile.amplitude(point));
      }
    require(global_sum(local_point_count_before) > 0,
            "combinable-source fixture created no source points");

    const gpu::dispatch_statistics dispatch_before_combine =
        gpu::get_dispatch_statistics();
    const gpu::resident_statistics resident_before_combine =
        gpu::get_resident_statistics();
    // An equal src_time at the same grid point takes src_vol::combinable's
    // in-place amplitude path after the first profile is already resident.
    f.add_point_source(Ez, source, source_point, 0.2);
    f.require_source_components();
    const gpu::dispatch_statistics dispatch_after_combine =
        gpu::get_dispatch_statistics();
    const gpu::resident_statistics resident_after_combine =
        gpu::get_resident_statistics();

    std::size_t local_volume_count_after = 0;
    std::size_t local_point_count_after = 0;
    double local_amplitude_l1_after = 0.0;
    for (int chunk = 0; chunk < f.num_chunks; ++chunk)
      FOR_FIELD_TYPES(ft) for (const src_vol &profile :
                                  f.chunks[chunk]->get_sources(ft)) {
        if (profile.c != Ez) continue;
        ++local_volume_count_after;
        local_point_count_after += profile.num_points();
        for (std::size_t point = 0; point < profile.num_points(); ++point)
          local_amplitude_l1_after += std::abs(profile.amplitude(point));
      }
    require(local_volume_count_after == local_volume_count_before &&
                local_point_count_after == local_point_count_before,
            "combinable source unexpectedly created a second profile");
    require(local_point_count_before == 0 ||
                std::abs(local_amplitude_l1_after -
                         1.5 * local_amplitude_l1_before) <=
                    32 * std::numeric_limits<double>::epsilon() *
                        std::max(1.0, local_amplitude_l1_after),
            "combinable source did not update amplitudes in place");
    require(dispatch_after_combine.host_to_device_bytes ==
                    dispatch_before_combine.host_to_device_bytes &&
                dispatch_after_combine.device_to_host_bytes ==
                    dispatch_before_combine.device_to_host_bytes &&
                resident_after_combine.device_buffer_allocations ==
                    resident_before_combine.device_buffer_allocations,
            "in-place source combination transferred or allocated unrelated "
            "resident state");

    for (int step = 0; step < 12; ++step) f.step();
    const gpu::source_statistics source_statistics =
        gpu::get_source_statistics();
    if (mode == gpu::backend_mode::cuda) {
      const std::uint64_t global_cuda_source_calls =
          global_sum(source_statistics.cuda_update_calls);
      const std::uint64_t global_cpu_source_calls =
          global_sum(source_statistics.cpu_update_calls);
      require(global_cuda_source_calls > 0 &&
                  global_cpu_source_calls == 0,
              "combinable source silently used the CPU source kernel");
    }
    return f.get_field(Ez, vec(0.71, 0.79));
  };

  const std::complex<double> combined_cpu =
      run_combinable_source(gpu::backend_mode::cpu);
  const std::complex<double> combined_cuda =
      run_combinable_source(gpu::backend_mode::cuda);
  const double combined_error = std::abs(combined_cpu - combined_cuda);
  const double combined_tolerance =
      2e-6 + 8e-4 * std::max(std::abs(combined_cpu),
                             std::abs(combined_cuda));
  require(combined_error <= combined_tolerance,
          "resident in-place source combination disagrees with CPU: " +
              std::to_string(combined_error) + " exceeded " +
              std::to_string(combined_tolerance));

  gpu::set_backend(gpu::backend_mode::cuda);
  gpu::reset_dispatch_statistics();
  const std::uint64_t live_before =
      global_sum(gpu::get_live_resident_device_buffers());

  {
    const grid_volume gv = vol2d(1.6, 1.4, 10.0);
    anisotropic_test_material anisotropic;
    structure s(gv, anisotropic, no_pml());
    fields f(&s, 0.0, 0.0, true, 64, 64);
    f.use_real_fields();

    const char *transport_environment =
        std::getenv("MEEP_GPU_MPI_TRANSPORT");
    const bool forced_pinned_transport =
        count_processors() > 1 && transport_environment &&
        (std::string(transport_environment) == "pinned" ||
         std::string(transport_environment) == "host");

    std::uint64_t warm_live_after_removal = 0;
    for (int cycle = 0; cycle < 4; ++cycle) {
      gaussian_src_time source(0.27, 0.11);
      require(source.is_integrated,
              "lifecycle regression requires an integrated electric source");
      const gpu::dispatch_statistics dispatch_before_source_change =
          gpu::get_dispatch_statistics();
      const gpu::resident_statistics resident_before_source_change =
          gpu::get_resident_statistics();
      f.add_point_source(Ez, source, gv.center(), 0.4);
      // Match Python Simulation.add_sources(), including the unconditional
      // boundary-source repair pass used by change_sources().
      f.require_source_components();
      const gpu::dispatch_statistics dispatch_after_source_change =
          gpu::get_dispatch_statistics();
      const gpu::resident_statistics resident_after_source_change =
          gpu::get_resident_statistics();
      require(dispatch_after_source_change.host_to_device_bytes ==
                  dispatch_before_source_change.host_to_device_bytes &&
                  dispatch_after_source_change.device_to_host_bytes ==
                      dispatch_before_source_change.device_to_host_bytes,
              "adding or repairing a source transferred unrelated resident "
              "fields");
      require(resident_after_source_change.device_buffer_allocations ==
                  resident_before_source_change.device_buffer_allocations,
              "adding or repairing a source allocated a replacement "
              "resident buffer");
      const gpu::dispatch_statistics dispatch_before_source_step =
          gpu::get_dispatch_statistics();
      const gpu::multi_gpu_statistics multi_gpu_before_source_step =
          gpu::get_multi_gpu_statistics();
      f.step();
      const gpu::dispatch_statistics dispatch_after_source_step =
          gpu::get_dispatch_statistics();
      const gpu::multi_gpu_statistics multi_gpu_after_source_step =
          gpu::get_multi_gpu_statistics();
      if (cycle > 0) {
        const std::uint64_t source_step_readback_bytes =
            dispatch_after_source_step.device_to_host_bytes -
            dispatch_before_source_step.device_to_host_bytes;
        const std::uint64_t pinned_mpi_readback_bytes =
            multi_gpu_after_source_step.pinned_device_to_host_bytes -
            multi_gpu_before_source_step.pinned_device_to_host_bytes;
        if (forced_pinned_transport)
          require(global_sum(pinned_mpi_readback_bytes) > 0,
                  "forced pinned MPI lifecycle recorded no D2H staging to "
                  "subtract from source-change accounting");
        require(source_step_readback_bytes >= pinned_mpi_readback_bytes,
                "pinned MPI readback accounting exceeded all CUDA D2H bytes");
        const std::uint64_t non_mpi_readback_bytes =
            source_step_readback_bytes - pinned_mpi_readback_bytes;
        // Pinned MPI transport legitimately stages halo cells through the
        // host.  Once those bytes are removed, the persistent-step finite
        // check returns one FP32 status scalar.  Anything larger indicates
        // that source replacement published device-authoritative fields.
        require(non_mpi_readback_bytes <= sizeof(float),
                "a no-op source-component requirement published all "
                "device-authoritative fields: " +
                    std::to_string(non_mpi_readback_bytes) +
                    " unexpected bytes");
      }
      const std::uint64_t live_with_source =
          global_sum(gpu::get_live_resident_device_buffers());
      require(live_with_source > live_before,
              "integrated-source step made no resident CUDA buffers");

      const gpu::dispatch_statistics dispatch_before_removal =
          gpu::get_dispatch_statistics();
      const gpu::resident_statistics resident_before_removal =
          gpu::get_resident_statistics();
      f.remove_sources();
      const gpu::dispatch_statistics dispatch_after_removal =
          gpu::get_dispatch_statistics();
      const gpu::resident_statistics resident_after_removal =
          gpu::get_resident_statistics();
      require(dispatch_after_removal.host_to_device_bytes ==
                  dispatch_before_removal.host_to_device_bytes &&
                  dispatch_after_removal.device_to_host_bytes ==
                      dispatch_before_removal.device_to_host_bytes,
              "removing a source transferred unrelated resident fields");
      require(resident_after_removal.device_buffer_allocations ==
                  resident_before_removal.device_buffer_allocations,
              "removing a source allocated a replacement resident buffer");

      f.step();
      const std::uint64_t live_after_removal =
          global_sum(gpu::get_live_resident_device_buffers());
      // Source input allocations are allowed to remain in the owner-scoped
      // exact-size recycle pool, but removal must never create more storage.
      // Warm-cycle stability and owner destruction below prove bounded reuse
      // and eventual release.
      require(live_after_removal <= live_with_source,
              "removing an integrated source increased resident CUDA storage");
      const gpu::dispatch_statistics dispatch_after_source_free_step =
          gpu::get_dispatch_statistics();
      const std::uint64_t source_free_upload_bytes =
          dispatch_after_source_free_step.host_to_device_bytes -
          dispatch_after_removal.host_to_device_bytes;
      // The first source-free step can lazily exercise a small set of
      // material/conductivity inputs that the source-driven step did not use.
      // Bound this by sixteen local field-sized arrays; the former full-cache
      // destruction republishes many more arrays and is already forbidden by
      // the zero-transfer removal check above.
      const std::uint64_t maximum_source_free_upload_bytes =
          16u * static_cast<std::uint64_t>(gv.ntot()) * sizeof(realnum);
      require(source_free_upload_bytes <= maximum_source_free_upload_bytes,
              "source removal forced a full resident-field reupload: " +
                  std::to_string(source_free_upload_bytes) +
                  " bytes exceeded " +
                  std::to_string(maximum_source_free_upload_bytes));
      // With full-cache destruction removed, distributed descriptor and
      // finite-check plans can be populated lazily while both the cold and
      // first no-op-material source/source-free paths are observed. They must
      // stabilize after those warm-up paths, and destruction below still
      // proves that no owner-scoped buffer survives the fields object.
      if (cycle == 2)
        warm_live_after_removal = live_after_removal;
      else if (cycle > 2)
        require(live_after_removal == warm_live_after_removal,
                "integrated source add/remove cycles changed resident CUDA "
                "buffer count after warm-up: " +
                    std::to_string(live_after_removal) + " versus " +
                    std::to_string(warm_live_after_removal));
    }

    // Vary the spatial point count long enough to overflow the bounded
    // source-profile recycle pool. Device allocations may churn for new
    // exact sizes, but retained live buffers must reach a fixed plateau.
    std::uint64_t bounded_live_after_varying_removal = 0;
    for (int cycle = 0; cycle < 12; ++cycle) {
      gaussian_src_time source(0.33, 0.12);
      const double half_extent = 0.05 * static_cast<double>(cycle + 1);
      const vec center = gv.center();
      f.add_volume_source(
          Ez, source,
          volume(center - vec(half_extent, half_extent),
                 center + vec(half_extent, half_extent)),
          0.25);
      f.require_source_components();
      f.step();
      f.remove_sources();
      const std::uint64_t live_after_removal =
          global_sum(gpu::get_live_resident_device_buffers());
      if (cycle == 7)
        bounded_live_after_varying_removal = live_after_removal;
      else if (cycle > 7)
        require(live_after_removal ==
                    bounded_live_after_varying_removal,
                "varying source profile sizes retained unbounded CUDA "
                "allocations after the recycle-pool plateau: " +
                    std::to_string(live_after_removal) + " versus " +
                    std::to_string(bounded_live_after_varying_removal));
    }

    const gpu::field_update_statistics updates =
        gpu::get_field_update_statistics();
    const gpu::source_statistics source_updates =
        gpu::get_source_statistics();
    require(updates.cuda_update_eh_calls > 0 &&
                updates.cpu_update_eh_calls == 0,
            "integrated-source lifecycle silently used CPU E/H updates");
    require(source_updates.cuda_update_calls > 0 &&
                source_updates.cpu_update_calls == 0,
            "varying-source lifecycle silently used CPU source updates");
  }

  require(global_sum(gpu::get_live_resident_device_buffers()) == live_before,
          "destroyed fields chunk retained resident CUDA buffers");
}

void require_distributed_component_allocation_consensus() {
  if (count_processors() < 2) return;
  const grid_volume gv = vol2d(2.0, 1.8, 10.0);
  structure s(gv, vacuum, no_pml(), identity(), count_processors());
  fields f(&s, 0.0, 0.0, true, 64, 64);
  f.use_real_fields();

  // Preallocate every valid real component only on rank zero. Requiring Ez
  // must therefore observe need_to_reconnect==0 on rank zero and >0 on its
  // peers while still entering figure_out_step_plan collectively.
  if (am_master()) {
    const component tm_components[] = {Hx, Hy, Bx, By, Ez, Dz};
    for (component c : tm_components)
      if (gv.has_field(c))
        for (int chunk = 0; chunk < f.num_chunks; ++chunk)
          f.chunks[chunk]->alloc_f(c);
  }

  bool local_owns_chunk = false;
  bool local_has_ez = true;
  for (int chunk = 0; chunk < f.num_chunks; ++chunk)
    if (f.chunks[chunk]->is_mine()) {
      local_owns_chunk = true;
      local_has_ez = local_has_ez &&
                     f.chunks[chunk]->have_component(Ez);
    }
  require(local_owns_chunk,
          "component-consensus fixture left an MPI rank without a chunk");
  require(am_master() ? local_has_ez : !local_has_ez,
          "component-consensus fixture did not create rank-skew allocation");

  f.require_component(Ez);
  bool local_required = true;
  for (int chunk = 0; chunk < f.num_chunks; ++chunk)
    if (f.chunks[chunk]->is_mine())
      local_required = local_required &&
                       f.chunks[chunk]->have_component(Ez);
  require(local_required,
          "collective component requirement missed an owning MPI rank");
  f.step();
  if (am_master())
    std::cout << "PASS: rank-skew component allocation remains collective\n";
}

std::vector<double> run_dft_reset_lifecycle(gpu::backend_mode mode,
                                            bool full_reset) {
  gpu::set_backend(mode);
  const grid_volume gv = vol2d(1.6, 1.4, 10.0);
  structure s(gv, vacuum, no_pml());
  fields f(&s, 0.0, 0.0, true, 64, 64);
  f.use_real_fields();
  continuous_src_time source(0.27);
  f.add_point_source(Ez, source, gv.center(), 0.8);
  component components[] = {Ez};
  dft_fields monitor =
      f.add_dft_fields(components, 1, f.v, 0.20, 0.34, 3);
  (void)monitor;

  for (int step = 0; step < 24; ++step)
    f.step();
  if (full_reset)
    f.reset();
  else
    f.zero_fields();

  std::vector<double> norms;
  norms.push_back(f.dft_norm());
  if (!full_reset) {
    for (int step = 0; step < 8; ++step)
      f.step();
    norms.push_back(f.dft_norm());
  }
  return norms;
}

void require_dft_reset_lifecycle() {
  for (bool full_reset : {false, true}) {
    const std::vector<double> cpu =
        run_dft_reset_lifecycle(gpu::backend_mode::cpu, full_reset);
    const std::vector<double> cuda =
        run_dft_reset_lifecycle(gpu::backend_mode::cuda, full_reset);
    require(cpu.size() == cuda.size(),
            "DFT reset lifecycle produced inconsistent result counts");
    for (std::size_t index = 0; index < cpu.size(); ++index) {
      require(cpu[index] > 0.0,
              "zero_fields/reset did not retain a CPU DFT reference signal");
      const double error = std::abs(cpu[index] - cuda[index]);
      const double tolerance =
          2e-5 + 8e-4 * std::max(std::abs(cpu[index]),
                                  std::abs(cuda[index]));
      require(error <= tolerance,
              "zero_fields/reset discarded a device-resident DFT value");
    }
  }
}

std::vector<double> run_shifted_dft_lifecycle(gpu::backend_mode mode,
                                              bool check_device_buffers) {
  gpu::set_backend(mode);
  const grid_volume gv = vol2d(1.8, 1.6, 12.0);
  structure s(gv, vacuum, no_pml());
  fields f(&s, 0.0, 0.0, true, 64, 64);
  f.use_real_fields();
  continuous_src_time source(0.27);
  f.add_point_source(Ez, source, gv.center(), 0.8);
  for (int step = 0; step < 32; ++step)
    f.step();

  component components[] = {Ez};
  const volume monitor_regions[] = {
      volume(vec(0.45, 0.55)), volume(vec(1.35, 1.05))};

  // With MPI, the two point monitors can materialize the same primary field
  // on different owned chunks.  Prewarm both locations before taking the
  // descriptor-mirror baseline so the lifecycle comparison measures only
  // same-size DFT replacement, not legitimate lazy primary-field residency.
  dft_fields warm_first = f.add_dft_fields(
      components, 1, monitor_regions[0], 0.20, 0.34, 3);
  dft_fields warm_second = f.add_dft_fields(
      components, 1, monitor_regions[1], 0.20, 0.34, 3);
  f.step();
  // dft_norm() also owns one lazily allocated scalar-reduction scratch value
  // per resident fields chunk.  Prewarm that bounded cache state on both MPI
  // owners before using the live-buffer count as a descriptor-lifetime oracle.
  (void)f.dft_norm();
  warm_first.remove();
  warm_second.remove();

  std::vector<double> norms;
  std::uint64_t live_after_first_removal = 0;
  for (int monitor_index = 0; monitor_index < 2; ++monitor_index) {
    dft_fields monitor = f.add_dft_fields(
        components, 1, monitor_regions[monitor_index], 0.20, 0.34, 3);
    for (int step = 0; step < 12; ++step)
      f.step();
    norms.push_back(f.dft_norm());
    require(norms.back() > 0.0,
            "shifted DFT lifecycle accumulated no reference signal");
    monitor.remove();
    if (check_device_buffers) {
      const std::uint64_t live_after_removal =
          global_sum(gpu::get_live_resident_device_buffers());
      if (monitor_index == 0)
        live_after_first_removal = live_after_removal;
      else
        require(live_after_removal == live_after_first_removal,
                "same-size DFT monitor replacement leaked descriptor mirrors");
    }
  }
  return norms;
}

void require_shifted_dft_lifecycle() {
  const std::vector<double> cpu =
      run_shifted_dft_lifecycle(gpu::backend_mode::cpu, false);
  const std::vector<double> cuda =
      run_shifted_dft_lifecycle(gpu::backend_mode::cuda, true);
  require(cpu.size() == cuda.size(),
          "shifted DFT lifecycle produced inconsistent result counts");
  for (std::size_t index = 0; index < cpu.size(); ++index) {
    const double error = std::abs(cpu[index] - cuda[index]);
    const double tolerance =
        2e-5 + 8e-4 * std::max(std::abs(cpu[index]),
                                std::abs(cuda[index]));
    require(error <= tolerance,
            "same-size shifted monitor reused a stale CUDA DFT descriptor");
  }
}

struct dft_norm_reduction_result {
  double norm = 0.0;
  std::uint64_t first_host_to_device_bytes = 0;
  std::uint64_t first_device_to_host_bytes = 0;
  std::uint64_t second_host_to_device_bytes = 0;
  std::uint64_t second_device_to_host_bytes = 0;
  std::size_t monitor_chunk_count = 0;
};

dft_norm_reduction_result run_dft_norm_reduction(
    gpu::backend_mode mode, bool persist) {
  gpu::set_backend(mode);
  const grid_volume gv = vol2d(2.0, 1.8, 16.0);
  structure s(gv, vacuum, no_pml());
  fields f(&s, 0.0, 0.0, true, 64, 64);
  f.use_real_fields();
  continuous_src_time source(0.27);
  f.add_point_source(Ez, source, gv.center(), 0.8);
  component components[] = {Ez};
  const volume monitor_region(
      vec(0.35, 0.30), vec(1.65, 1.50));
  dft_fields monitor = f.add_dft_fields(
      components, 1, monitor_region, 0.19, 0.35, 4, true, 1,
      persist);
  dft_norm_reduction_result result;
  for (dft_chunk *chunk = monitor.chunks; chunk;
       chunk = chunk->next_in_dft)
    ++result.monitor_chunk_count;
  require(result.monitor_chunk_count > 0,
          "DFT norm reduction regression created no monitor chunks");
  for (int step = 0; step < 40; ++step)
    f.step();

  gpu::reset_dispatch_statistics();
  const gpu::dispatch_statistics before =
      gpu::get_dispatch_statistics();
  result.norm = f.dft_norm();
  const gpu::dispatch_statistics after_first =
      gpu::get_dispatch_statistics();
  const double repeated_norm = f.dft_norm();
  const gpu::dispatch_statistics after_second =
      gpu::get_dispatch_statistics();
  require(
      std::abs(result.norm - repeated_norm) <=
          4096 * std::numeric_limits<double>::epsilon() *
              std::max(1.0, std::abs(result.norm)),
      "repeated resident DFT norm reduction differed beyond bounded FP64 "
      "atomic-order roundoff");
  result.first_host_to_device_bytes =
      after_first.host_to_device_bytes -
      before.host_to_device_bytes;
  result.first_device_to_host_bytes =
      after_first.device_to_host_bytes -
      before.device_to_host_bytes;
  result.second_host_to_device_bytes =
      after_second.host_to_device_bytes -
      after_first.host_to_device_bytes;
  result.second_device_to_host_bytes =
      after_second.device_to_host_bytes -
      after_first.device_to_host_bytes;
  return result;
}

void require_dft_norm_reduction() {
  for (bool persist : {false, true}) {
    const dft_norm_reduction_result cpu =
        run_dft_norm_reduction(gpu::backend_mode::cpu, persist);
    const dft_norm_reduction_result cuda =
        run_dft_norm_reduction(gpu::backend_mode::cuda, persist);
    require(cpu.norm > 0.0 && cuda.norm > 0.0,
            "DFT norm reduction accumulated no reference signal");
    const double error = std::abs(cpu.norm - cuda.norm);
    const double tolerance =
        2e-5 + 8e-4 * std::max(std::abs(cpu.norm),
                                std::abs(cuda.norm));
    require(error <= tolerance,
            persist
                ? "persistent indexed CUDA DFT norm disagrees with CPU"
                : "contiguous CUDA DFT norm disagrees with CPU");
    const std::uint64_t expected_scalar_bytes =
        static_cast<std::uint64_t>(cuda.monitor_chunk_count) *
        sizeof(double);
    require(
        cuda.first_device_to_host_bytes == expected_scalar_bytes &&
            cuda.second_device_to_host_bytes == expected_scalar_bytes,
        "resident CUDA DFT norm copied more than one FP64 scalar per "
        "monitor chunk");
    require(
        cuda.second_host_to_device_bytes == 0,
        "repeated resident CUDA DFT norm re-uploaded static metadata");
    if (!persist)
      require(
          cuda.first_host_to_device_bytes == 0,
          "contiguous resident CUDA DFT norm unexpectedly uploaded "
          "metadata");
  }
}

std::vector<std::complex<double> >
run_persistent_dft_after_fields_destruction(
    gpu::backend_mode mode, bool clear_before_destruction) {
  gpu::set_backend(mode);
  const grid_volume gv = vol2d(2.0, 1.8, 16.0);
  structure s(gv, vacuum, no_pml());
  std::unique_ptr<dft_fields> monitor;
  {
    fields f(&s, 0.0, 0.0, true, 64, 64);
    f.use_real_fields();
    continuous_src_time source(0.27);
    f.add_point_source(Ez, source, gv.center(), 0.8);
    component components[] = {Ez};
    monitor.reset(new dft_fields(f.add_dft_fields(
        components, 1, volume(vec(0.35, 0.30), vec(1.65, 1.50)),
        0.19, 0.35, 4, true, 1, true)));
    for (int step = 0; step < 40; ++step) f.step();
    if (clear_before_destruction) {
      f.clear_dft_monitors();
      for (dft_chunk *chunk = monitor->chunks; chunk;
           chunk = chunk->next_in_dft)
        require(chunk->fc != nullptr,
                "clear_dft_monitors prematurely detached a live adjoint "
                "monitor from its fields chunk");
    }
    // Do not synchronize the monitor here.  The fields_chunk destructor is
    // responsible for publishing a persistent, device-authoritative DFT.
  }

  std::vector<std::complex<double> > values;
  for (dft_chunk *chunk = monitor->chunks; chunk;
       chunk = chunk->next_in_dft) {
    require(chunk->persist && chunk->fc == nullptr,
            "persistent DFT was not detached from its destroyed fields "
            "chunk");
    const std::size_t count = chunk->N * chunk->omega.size();
    for (std::size_t index = 0; index < count; ++index)
      values.push_back(chunk->dft[index]);
  }
  require(!values.empty(),
          "persistent DFT destruction regression captured no values");
  monitor->remove();
  return values;
}

void require_persistent_dft_survives_fields_destruction() {
  for (bool clear_before_destruction : {false, true}) {
    const std::vector<std::complex<double> > cpu =
        run_persistent_dft_after_fields_destruction(
            gpu::backend_mode::cpu, clear_before_destruction);
    const std::vector<std::complex<double> > cuda =
        run_persistent_dft_after_fields_destruction(
            gpu::backend_mode::cuda, clear_before_destruction);
    require(cpu.size() == cuda.size(),
            "persistent DFT destruction changed result size");
    for (std::size_t index = 0; index < cpu.size(); ++index) {
      const double error = std::abs(cpu[index] - cuda[index]);
      const double tolerance =
          2e-5 + 8e-4 * std::max(std::abs(cpu[index]),
                                  std::abs(cuda[index]));
      require(error <= tolerance,
              clear_before_destruction
                  ? "clear_dft_monitors lost a device-authoritative "
                    "persistent CUDA DFT when fields was destroyed"
                  : "persistent CUDA DFT lost device-authoritative values "
                    "when fields was destroyed");
    }
  }
}

void trigger_distributed_dft_norm_rank_failure() {
  require(count_processors() > 1,
          "distributed DFT norm failure injection requires MPI");
  gpu::set_backend(gpu::backend_mode::cuda);
  const grid_volume gv = vol2d(2.0, 1.8, 16.0);
  structure s(gv, vacuum, no_pml());
  fields f(&s, 0.0, 0.0, true, 64, 64);
  f.use_real_fields();
  continuous_src_time source(0.27);
  f.add_point_source(Ez, source, gv.center(), 0.8);
  component components[] = {Ez};
  dft_fields monitor =
      f.add_dft_fields(components, 1, f.v, 0.19, 0.35, 4);
  for (int step = 0; step < 4; ++step)
    f.step();

  // Inject a deterministic rank-zero exception after dft_norm has armed its
  // distributed guard but before the local reduction reaches sum_to_all.
  gpu::detail::fail_next_dft_norm_on_master_for_testing();
  (void)f.dft_norm();
  meep::abort(
      "distributed DFT norm rank failure did not abort the communicator");
}

void trigger_distributed_cw_rank_failure() {
  require(count_processors() > 1,
          "distributed CW failure injection requires MPI");
  gpu::set_backend(gpu::backend_mode::cuda);
  const grid_volume gv = vol2d(1.8, 1.6, 8.0);
  structure s(gv, vacuum, pml(0.3));
  fields f(&s, 0.0, 0.0, true, 64, 64);
  continuous_src_time source(0.27);
  f.add_point_source(Ez, source, vec(0.65, 0.8), 0.8);
  (void)f.solve_cw(2e-4, 1500, 3);
  meep::abort(
      "distributed CW rank failure did not abort the communicator");
}

void trigger_distributed_cw_diagnostic_mismatch() {
  require(count_processors() > 1,
          "distributed CW diagnostic mismatch requires MPI");
  gpu::set_backend(gpu::backend_mode::cuda);
  const grid_volume gv = vol2d(1.8, 1.6, 8.0);
  structure s(gv, vacuum, pml(0.3));
  fields f(&s, 0.0, 0.0, true, 64, 64);
  continuous_src_time source(0.27);
  f.add_point_source(Ez, source, vec(0.65, 0.8), 0.8);
  if (am_master())
    setenv("MEEP_GPU_CW_DIAGNOSTIC_TRUE_RESIDUAL", "1", 1);
  else
    unsetenv("MEEP_GPU_CW_DIAGNOSTIC_TRUE_RESIDUAL");
  (void)f.solve_cw(2e-4, 1500, 3);
  meep::abort(
      "rank-mismatched CW diagnostic did not abort the communicator");
}

void trigger_distributed_cw_configuration_mismatch(
    const std::string &setting) {
  require(count_processors() > 1,
          "distributed CW configuration mismatch requires MPI");
  require(setting == "solver" || setting == "restart" ||
              setting == "order",
          "unknown distributed CW configuration mismatch fixture");
  gpu::set_backend(gpu::backend_mode::cuda);
  const grid_volume gv = vol2d(1.8, 1.6, 8.0);
  structure s(gv, vacuum, pml(0.3));
  fields f(&s, 0.0, 0.0, true, 64, 64);
  continuous_src_time source(0.27);
  f.add_point_source(Ez, source, vec(0.65, 0.8), 0.8);
  if (setting == "solver")
    setenv("MEEP_GPU_CW_SOLVER",
           am_master() ? "resident" : "host", 1);
  if (setting == "restart")
    setenv("MEEP_GPU_CW_RELIABLE_RESTART",
           am_master() ? "10" : "20", 1);
  const int order =
      setting == "order" && !am_master() ? 3 : 2;
  (void)f.solve_cw(2e-4, 1500, order);
  meep::abort(
      "rank-mismatched CW configuration did not abort the communicator");
}

double run_mutated_dft_frequencies(gpu::backend_mode mode,
                                   bool check_transfers) {
  gpu::set_backend(mode);
  gpu::reset_dispatch_statistics();
  const grid_volume gv = vol2d(1.8, 1.6, 12.0);
  structure s(gv, vacuum, no_pml());
  fields f(&s, 0.0, 0.0, true, 64, 64);
  f.use_real_fields();
  continuous_src_time source(0.29);
  f.add_point_source(Ez, source, gv.center(), 0.8);
  component components[] = {Ez};
  dft_fields monitor =
      f.add_dft_fields(components, 1, f.v, 0.18, 0.42, 3, true, 1);

  for (int step = 0; step < 6; ++step)
    f.step();
  std::vector<std::unique_ptr<gpu::detail::resident_curl_session> >
      outer_sessions;
  if (check_transfers)
    for (dft_chunk *chunk = monitor.chunks; chunk;
         chunk = chunk->next_in_dft)
      outer_sessions.emplace_back(
          new gpu::detail::resident_curl_session(chunk->fc, true));
  const auto update_monitor = [&monitor](double time) {
    for (dft_chunk *chunk = monitor.chunks; chunk;
         chunk = chunk->next_in_dft)
      chunk->update_dft(time);
  };
  const auto require_phase_cache = [&monitor](double time) {
    for (dft_chunk *chunk = monitor.chunks; chunk;
         chunk = chunk->next_in_dft)
      for (std::size_t frequency = 0; frequency < chunk->omega.size();
           ++frequency) {
        const std::complex<double> expected =
            std::polar(1.0, chunk->omega[frequency] * time) * chunk->scale;
        const std::complex<double> observed(
            chunk->dft_phase[frequency].real(),
            chunk->dft_phase[frequency].imag());
        require(std::abs(observed - expected) <=
                    2e-6 * std::max(1.0, std::abs(expected)),
                "dft_chunk::dft_phase did not retain its public cache "
                "semantics");
      }
  };
  const gpu::dispatch_statistics before_baseline =
      gpu::get_dispatch_statistics();
  const double baseline_time = f.time() + 0.013;
  update_monitor(baseline_time);
  require_phase_cache(baseline_time);
  const gpu::dispatch_statistics after_baseline =
      gpu::get_dispatch_statistics();
  const std::uint64_t baseline_h2d =
      after_baseline.host_to_device_bytes -
      before_baseline.host_to_device_bytes;

  std::uint64_t refreshed_frequency_bytes = 0;
  for (dft_chunk *chunk = monitor.chunks; chunk;
       chunk = chunk->next_in_dft) {
    require(chunk->omega.size() == 3,
            "DFT mutation regression found an unexpected frequency count");
    chunk->omega[0] += 0.173;
    refreshed_frequency_bytes +=
        static_cast<std::uint64_t>(chunk->omega.size() * sizeof(double));
  }
  require(refreshed_frequency_bytes > 0,
          "DFT mutation regression found no local monitor chunks");

  const gpu::dispatch_statistics before_mutation =
      gpu::get_dispatch_statistics();
  const double mutation_time = f.time() + 0.027;
  update_monitor(mutation_time);
  require_phase_cache(mutation_time);
  const gpu::dispatch_statistics after_mutation =
      gpu::get_dispatch_statistics();
  const std::uint64_t mutation_h2d =
      after_mutation.host_to_device_bytes -
      before_mutation.host_to_device_bytes;

  const gpu::dispatch_statistics before_reuse =
      gpu::get_dispatch_statistics();
  const double reuse_time = f.time() + 0.041;
  update_monitor(reuse_time);
  require_phase_cache(reuse_time);
  const gpu::dispatch_statistics after_reuse =
      gpu::get_dispatch_statistics();
  const std::uint64_t reuse_h2d =
      after_reuse.host_to_device_bytes -
      before_reuse.host_to_device_bytes;

  if (check_transfers) {
    if (am_master())
      std::cout << "dft-omega-refresh: baseline_h2d=" << baseline_h2d
                << " mutation_h2d=" << mutation_h2d
                << " reuse_h2d=" << reuse_h2d
                << " expected_frequency_bytes="
                << refreshed_frequency_bytes << '\n';
    require(baseline_h2d == 0 && reuse_h2d == 0,
            "steady CUDA DFT timestep performed an H2D transfer");
    require(
        mutation_h2d == refreshed_frequency_bytes,
        "CUDA DFT frequency mutation did not perform exactly one static "
        "omega refresh");
  }
  for (auto &session : outer_sessions)
    session->finish(false);
  return f.dft_norm();
}

void require_mutated_dft_frequencies() {
  const double cpu =
      run_mutated_dft_frequencies(gpu::backend_mode::cpu, false);
  const double cuda =
      run_mutated_dft_frequencies(gpu::backend_mode::cuda, true);
  require(cpu > 0.0, "mutated CPU DFT accumulated no reference signal");
  const double error = std::abs(cpu - cuda);
  const double tolerance =
      2e-5 + 8e-4 * std::max(std::abs(cpu), std::abs(cuda));
  require(error <= tolerance,
          "same-size dft_chunk::omega mutation was not reflected by CUDA");
}

std::unique_ptr<binary_partition> make_two_rank_partition(
    double split_position, int left_rank, int right_rank) {
  return std::unique_ptr<binary_partition>(new binary_partition(
      split_plane{X, split_position},
      std::unique_ptr<binary_partition>(
          new binary_partition(left_rank)),
      std::unique_ptr<binary_partition>(
          new binary_partition(right_rank))));
}

std::complex<double> integrate2_product(
    const std::complex<realnum> *values, const vec &, void *) {
  return values[0] * values[1];
}

void trigger_integrate2_swapped_ownership() {
  require(count_processors() == 2,
          "integrate2 swapped-ownership regression requires two MPI ranks");
  const grid_volume gv = vol2d(2.0, 1.8, 10.0);
  const double split_position = gv.center().x();
  std::unique_ptr<binary_partition> forward =
      make_two_rank_partition(split_position, 0, 1);
  std::unique_ptr<binary_partition> reversed =
      make_two_rank_partition(split_position, 1, 0);
  structure s1(gv, vacuum, no_pml(), identity(), 0, 0.5, false,
               DEFAULT_SUBPIXEL_TOL, DEFAULT_SUBPIXEL_MAXEVAL,
               forward.get());
  structure s2(gv, vacuum, no_pml(), identity(), 0, 0.5, false,
               DEFAULT_SUBPIXEL_TOL, DEFAULT_SUBPIXEL_MAXEVAL,
               reversed.get());
  fields f1(&s1);
  fields f2(&s2);
  require(f1.equal_layout(f2),
          "swapped-ownership fixture changed the geometric field layout");
  bool ownership_differs = false;
  for (int i = 0; i < f1.num_chunks; ++i)
    ownership_differs = ownership_differs ||
                        f1.chunks[i]->n_proc() !=
                            f2.chunks[i]->n_proc();
  require(ownership_differs,
          "swapped-ownership fixture did not change chunk owners");
  const component first_components[] = {Ez};
  const component second_components[] = {Ez};
  (void)f1.integrate2(
      f2, 1, first_components, 1, second_components,
      integrate2_product, nullptr, f1.v);
  meep::abort(
      "integrate2 accepted equal geometry with swapped MPI chunk ownership");
}

void trigger_integrate2_rank_local_sync_failure() {
  require(count_processors() == 2,
          "integrate2 rank-failure regression requires two MPI ranks");
  gpu::set_backend(gpu::backend_mode::cuda);
  const grid_volume gv = vol2d(2.0, 1.8, 10.0);
  structure s1(gv, vacuum, no_pml(), identity(), 2);
  structure s2(gv, vacuum, no_pml(), identity(), 2);
  fields f1(&s1);
  fields f2(&s2);

  std::unique_ptr<gpu::detail::resident_curl_session> active_session;
  if (my_rank() == 0)
    for (int i = 0; i < f2.num_chunks; ++i)
      if (f2.chunks[i]->is_mine()) {
        active_session.reset(
            new gpu::detail::resident_curl_session(f2.chunks[i], true));
        break;
      }
  require(and_to_all(my_rank() != 0 || active_session.get() != nullptr),
          "rank zero owns no chunk for integrate2 failure injection");

  const component first_components[] = {Ez};
  const component second_components[] = {Ez};
  (void)f1.integrate2(
      f2, 1, first_components, 1, second_components,
      integrate2_product, nullptr, f1.v);
  meep::abort(
      "rank-local integrate2 CUDA publication failure did not abort the "
      "communicator");
}

void require_distributed_boundary_timing() {
  require(count_processors() == 2,
          "boundary timing regression requires two MPI ranks");
  gpu::set_backend(gpu::backend_mode::cuda);
  const grid_volume gv = vol2d(2.5, 2.0, 12.0);
  structure s(gv, vacuum, no_pml(), identity(), 2);
  fields f(&s, 0.0, 0.0, true, 64);
  f.use_real_fields();
  gaussian_src_time electric_source(0.31, 0.12);
  gaussian_src_time magnetic_source(0.23, 0.10);
  f.add_volume_source(
      Ez, electric_source,
      volume(vec(0.71, 0.25), vec(0.71, 1.75)), 0.7);
  f.add_point_source(Hz, magnetic_source, vec(1.63, 1.27), -0.4);

  const time_sink detailed_sinks[] = {
      BoundarySteppingB, BoundarySteppingWH, BoundarySteppingPH,
      BoundarySteppingH, BoundarySteppingD,  BoundarySteppingWE,
      BoundarySteppingPE, BoundarySteppingE};
  struct timing_observation {
    double boundaries;
    double mpi;
    double detailed;
  };
  const char *transport_environment =
      std::getenv("MEEP_GPU_MPI_TRANSPORT");
  const std::string transport =
      transport_environment ? transport_environment : "auto";
  auto validate_window = [&](const char *label) {
    const gpu::boundary_statistics boundaries =
        gpu::get_boundary_statistics();
    const gpu::multi_gpu_statistics multi_gpu =
        gpu::get_multi_gpu_statistics();
    const double boundary_time = f.get_time_spent_on(Boundaries);
    const double mpi_time = f.get_time_spent_on(MpiOneTime);
    double detailed_time = 0.0;
    for (time_sink sink : detailed_sinks)
      detailed_time += f.get_time_spent_on(sink);
    const double timer_tolerance =
        1e-5 + 0.02 * std::max(detailed_time, 1e-6);
    const bool used_cuda_aware = multi_gpu.cuda_aware_bytes > 0;
    const bool used_pinned = multi_gpu.pinned_staging_bytes > 0;
    bool transport_matches = used_cuda_aware != used_pinned;
    if (transport == "pinned" || transport == "host")
      transport_matches = used_pinned && !used_cuda_aware;
    else if (transport == "cuda-aware" || transport == "device")
      transport_matches = used_cuda_aware && !used_pinned;

    const bool local_pass =
        boundaries.cuda_update_calls > 0 &&
        boundaries.cuda_update_points > 0 &&
        boundaries.cpu_update_calls == 0 &&
        multi_gpu.mpi_messages > 0 && multi_gpu.mpi_scalars > 0 &&
        transport_matches && boundary_time > 0.0 && mpi_time > 0.0 &&
        detailed_time > 0.0 &&
        boundary_time + mpi_time <= detailed_time + timer_tolerance;
    require(
        and_to_all(local_pass),
        std::string("distributed CUDA boundary timing ") + label +
            " window did not separate Boundaries and MpiOneTime without "
            "double-counting detailed boundary phases or violated the "
            "requested transport");
    return timing_observation{
        boundary_time, mpi_time, detailed_time};
  };

  // First cover plan construction and the full exchange path.
  f.reset_timers();
  gpu::reset_dispatch_statistics();
  f.step();
  const timing_observation full = validate_window("full-plan");

  // Then cover cached descriptor replay and repeated timer-stack restoration.
  f.reset_timers();
  gpu::reset_dispatch_statistics();
  for (int step = 0; step < 16; ++step)
    f.step();
  const timing_observation cached = validate_window("cached-plan");

  // A deliberately delayed receive callback proves the local H2D/scatter
  // work is nested in Boundaries instead of merely relying on a loose sum of
  // overlapping wall-clock timers.  Both ranks use the same delay, so it is
  // local unpack work rather than intentional peer imbalance.
  setenv("MEEP_GPU_TEST_BOUNDARY_CALLBACK_DELAY_US", "5000", 1);
  f.reset_timers();
  gpu::reset_dispatch_statistics();
  f.step();
  unsetenv("MEEP_GPU_TEST_BOUNDARY_CALLBACK_DELAY_US");
  const timing_observation delayed = validate_window("delayed-callback");
  const double cached_boundary_per_step = cached.boundaries / 16.0;
  const double cached_mpi_per_step = cached.mpi / 16.0;
  const double delayed_mpi_growth =
      std::max(0.0, delayed.mpi - cached_mpi_per_step);
  const bool local_callback_attribution =
      delayed.boundaries >= cached_boundary_per_step + 0.004 &&
      delayed.boundaries >= 2.0 * delayed.mpi &&
      delayed_mpi_growth <= 0.001;
  require(
      and_to_all(local_callback_attribution),
      "delayed CUDA receive callback was not attributed to Boundaries "
      "independently of MpiOneTime or leaked at least 1 ms into the MPI "
      "timer");
  if (am_master())
    std::cout
        << "PASS: distributed CUDA boundary timing separates MPI completion"
        << " transport=" << transport
        << " full-boundaries=" << full.boundaries
        << " full-mpi=" << full.mpi
        << " full-detailed=" << full.detailed
        << " cached-boundaries=" << cached.boundaries
        << " cached-mpi=" << cached.mpi
        << " cached-detailed=" << cached.detailed
        << " delayed-boundaries=" << delayed.boundaries
        << " delayed-mpi=" << delayed.mpi
        << " delayed-detailed=" << delayed.detailed << '\n';
}

void require_distributed_dft_decimation_consensus() {
  require(count_processors() == 2,
          "DFT decimation consensus regression requires two MPI ranks");
  const grid_volume gv = vol2d(2.5, 2.0, 12.0);
  structure s(gv, vacuum, no_pml(), identity(), 2);
  fields f(&s, 0.0, 0.0, true, 64);
  f.use_real_fields();
  gaussian_src_time source(0.31, 0.12);
  f.add_point_source(Ez, source, vec(0.31, 0.73), 1.0);

  // add_srcdata sources such as an adjoint IndexedSource can exist on only
  // one owning rank.  Reproduce that distribution explicitly while the DFT
  // monitor is constructed, then restore the fields-owned list for cleanup.
  src_time *saved_sources = f.sources;
  if (!am_master()) f.sources = nullptr;
  dft_chunk *monitor = f.add_dft(
      Ez, f.v, 0.29, 0.29, 1, true, 1.0, nullptr, false, 1.0,
      true, 0, 0, false);
  f.sources = saved_sources;

  require(and_to_all(monitor != nullptr),
          "distributed DFT monitor was not present on every rank");
  int local_minimum = std::numeric_limits<int>::max();
  int local_maximum = 0;
  for (dft_chunk *chunk = monitor; chunk; chunk = chunk->next_in_dft) {
    local_minimum = std::min(local_minimum, chunk->get_decimation_factor());
    local_maximum = std::max(local_maximum, chunk->get_decimation_factor());
  }
  const int global_minimum = min_to_all(local_minimum);
  const int global_maximum = max_to_all(local_maximum);
  require(global_minimum > 1 && global_minimum == global_maximum,
          "a rank without a local source forced automatic DFT decimation to "
          "one or ranks selected inconsistent factors");

  require(!am_master() || saved_sources != nullptr,
          "source-owning rank has no source-time metadata");
  if (am_master()) saved_sources->set_fwidth(0.0);
  if (!am_master()) f.sources = nullptr;
  dft_chunk *continuous_monitor = f.add_dft(
      Ez, f.v, 0.29, 0.29, 1, true, 1.0, nullptr, false, 1.0,
      true, 0, 0, false);
  f.sources = saved_sources;
  require(and_to_all(continuous_monitor != nullptr),
          "continuous-source DFT monitor was not present on every rank");
  int local_continuous_maximum = 0;
  for (dft_chunk *chunk = continuous_monitor; chunk;
       chunk = chunk->next_in_dft)
    local_continuous_maximum = std::max(
        local_continuous_maximum, chunk->get_decimation_factor());
  require(max_to_all(local_continuous_maximum) == 1,
          "a continuous source on one rank did not force unit DFT "
          "decimation globally");
  if (am_master())
    std::cout << "PASS: sparse MPI sources retain global DFT decimation factor="
              << global_minimum
              << " and continuous sources force factor=1\n";
}

void require_near2far_cross_rank_collective_cancellation() {
  require(count_processors() == 2,
          "cross-rank Near2Far cancellation probe requires two MPI ranks");
  double local[13] = {};
  double global[13] = {};
  local[0] = my_rank() == 0 ? 1.0 : -std::nextafter(1.0, 0.0);
  local[12] = 1.0e-5;
  require(local[12] <= 3.0e-3 * std::abs(local[0]) + 3.0e-8,
          "cross-rank cancellation probe is already unsafe rank-locally");
  gpu::reset_dispatch_statistics();
  const bool retry = gpu::detail::test_near2far_collective_requires_mixed(
      local, global, 1);
  const gpu::near2far_statistics statistics =
      gpu::get_near2far_statistics();
  require(retry && std::abs(global[0]) < 1.0e-12 &&
              std::abs(global[12] - 2.0e-5) < 1.0e-15 &&
              statistics.mpi_allreduce_calls == 1 &&
              statistics.mpi_allreduce_bytes == 13 * sizeof(double),
          "Near2Far collective gate missed cancellation created only by the "
          "MPI rank sum");
  if (am_master())
    std::cout << "PASS: Near2Far collective gate detects cross-rank "
                 "cancellation\n";
}

} // namespace

int main(int argc, char **argv) {
  preinitialize_ldos_cpu_affinity();
  initialize mpi(argc, argv);

  try {
    if (std::getenv("MEEP_GPU_TEST_NEAR2FAR_DIMENSION_ONLY")) {
      require(count_processors() == 1,
              "Near2Far dimension regression requires one MPI rank");
      require_near2far_public_dimension_mismatch_rejected();
      require_near2far_cylindrical_inputs_rejected_without_device();
      if (am_master())
        std::cout << "PASS: public Near2Far rejects 2D/3D monitor-target "
                     "dimension mismatches\n";
      return 0;
    }
    if (std::getenv("MEEP_GPU_TEST_NEAR2FAR_ADJOINT_CPU_ONLY")) {
      require(count_processors() == 1,
              "adjoint Near2Far CPU regression requires one MPI rank");
      require_near2far_adjoint_cpu_reference();
      if (am_master())
        std::cout << "PASS: CPU adjoint Near2Far source profiles and "
                     "execution counters\n";
      return 0;
    }
    if (std::getenv("MEEP_GPU_TEST_NEAR2FAR_ADJOINT_MPI_CPU_ONLY")) {
      require(count_processors() == 2,
              "MPI adjoint Near2Far CPU regression requires two ranks");
      require_near2far_adjoint_cpu_reference();
      if (am_master())
        std::cout << "PASS: distributed CPU adjoint Near2Far source profiles "
                     "and execution counters\n";
      return 0;
    }
    if (std::getenv("MEEP_GPU_TEST_NEAR2FAR_2D_MPI_ONLY")) {
      require_near2far2d_two_gpu_mpi_contract();
      gpu::set_backend(gpu::backend_mode::cpu);
      return 0;
    }
    if (std::getenv("MEEP_GPU_TEST_NEAR2FAR_CYL_MPI_ONLY")) {
      require_near2far_cylindrical_two_gpu_mpi_contract();
      gpu::set_backend(gpu::backend_mode::cpu);
      return 0;
    }
    if (std::getenv("MEEP_GPU_TEST_NEAR2FAR_CROSS_RANK_CANCELLATION")) {
      require_near2far_cross_rank_collective_cancellation();
      return 0;
    }
    if (std::getenv("MEEP_GPU_TEST_DENSE_MATERIALIZE_ONLY")) {
      if (am_master())
        std::cout << "PASS: materialized dense-oracle runtime ELF\n";
      return 0;
    }
    if (const char *near2far_evidence =
            std::getenv("MEEP_GPU_TEST_NEAR2FAR_EVIDENCE_WRITE")) {
      const char *backend =
          std::getenv("MEEP_GPU_TEST_NEAR2FAR_EVIDENCE_BACKEND");
      if (!backend || !*backend)
        throw std::runtime_error(
            "MEEP_GPU_TEST_NEAR2FAR_EVIDENCE_BACKEND must be cpu or cuda");
      gpu::backend_mode mode;
      if (std::strcmp(backend, "cpu") == 0)
        mode = gpu::backend_mode::cpu;
      else if (std::strcmp(backend, "cuda") == 0)
        mode = gpu::backend_mode::cuda;
      else
        throw std::runtime_error(
            "MEEP_GPU_TEST_NEAR2FAR_EVIDENCE_BACKEND must be cpu or cuda");
      write_near2far_mpi_evidence(near2far_evidence, mode);
      gpu::set_backend(gpu::backend_mode::cpu);
      return 0;
    }
    if (std::getenv("MEEP_GPU_TEST_NEAR2FAR_ABORT")) {
      run_distributed_near2far_failure_abort_probe();
      return 1;
    }
    if (std::getenv("MEEP_GPU_TEST_NEAR2FAR_REGRESSION_ONLY")) {
      require(count_processors() == 1,
              "focused Near2Far regression requires one MPI rank");
      std::string near2far_diagnostic;
      require(gpu::runtime_available(&near2far_diagnostic),
              "focused Near2Far regression requires CUDA: " +
                  near2far_diagnostic);
      require_near2far_public_dimension_mismatch_rejected();
      require_near2far_cylindrical_inputs_rejected_without_device();
      require_near2far2d_hankel_bound_sweep();
      require_near2far_cartesian_dimension_cache_key();
      require_near2far_cylindrical_bound_sweep();
      require_near2far_cylindrical_default_tolerance_cancellation_retry();
      require_near2far2d_public_api_equivalence();
      require_near2far2d_public_periodic_constructor();
      require_near2far_cylindrical_public_api_equivalence();
      require_near2far3d_transform_equivalence();
      require_near2far_adjoint_equivalence();
      require_near2far_adjoint_retained_plan_reuse();
      require_near2far_adjoint_cancellation_retry();
      require_near2far_adjoint_negative_radial_copy();
      require_near2far_adjoint_all_axis_tiling();
      require_near2far_adjoint_two_device_migration();
      require_near2far3d_same_snapshot_equivalence();
      if (am_master())
        std::cout
            << "PASS: focused Near2Far selector, retained-workspace, "
               "lifecycle, periodic, and numerical contracts\n";
      gpu::set_backend(gpu::backend_mode::cpu);
      return 0;
    }
    require_restarted_bicgstab_boundaries();
    require_saturating_boundary_topology_generation();
    if (const char *evidence_output =
            std::getenv("MEEP_GPU_TEST_DENSE_EVIDENCE_WRITE")) {
      require(count_processors() == 1,
              "dense long-horizon evidence generation is singleton-only");
      const char *backend =
          std::getenv("MEEP_GPU_TEST_DENSE_EVIDENCE_BACKEND");
      require(backend && *backend,
              "MEEP_GPU_TEST_DENSE_EVIDENCE_BACKEND must be cpu or cuda");
      gpu::backend_mode mode;
      if (std::strcmp(backend, "cpu") == 0)
        mode = gpu::backend_mode::cpu;
      else if (std::strcmp(backend, "cuda") == 0)
        mode = gpu::backend_mode::cuda;
      else
        throw std::runtime_error(
            "MEEP_GPU_TEST_DENSE_EVIDENCE_BACKEND must be cpu or cuda");
#if !MEEP_SINGLE
      require(mode == gpu::backend_mode::cpu,
              "the CPU-FP64 evidence lane cannot request CUDA");
#endif
      const dense_long_run_result evidence =
          run_dense_long_field_case(mode);
      write_dense_long_evidence(evidence_output, evidence, mode);
      if (am_master())
        std::cout << "PASS: wrote deterministic dense long-horizon "
                     "full-array evidence: backend="
                  << backend << " arrays=" << evidence.arrays.size()
                  << '\n';
      gpu::set_backend(gpu::backend_mode::cpu);
      return 0;
    }
#if MEEP_SINGLE
    if (std::getenv("MEEP_GPU_TEST_DENSE_FP64_WRITE"))
      throw std::runtime_error(
          "FP64 dense fixture generation requires a CPU double build");
#else
    if (const char *fp64_fixture_output =
            std::getenv("MEEP_GPU_TEST_DENSE_FP64_WRITE")) {
      require(count_processors() == 1,
              "FP64 dense fixture generation is singleton-only");
      const dense_long_run_result fp64 =
          run_dense_long_field_case(gpu::backend_mode::cpu);
      write_dense_fp64_fixture(fp64_fixture_output, fp64);
      if (am_master())
        std::cout << "PASS: wrote independent CPU-FP64 dense fixture: "
                  << "arrays=" << fp64.arrays.size()
                  << " fnv64=0x" << std::hex << dense_fixture_hash(fp64)
                  << std::dec << '\n';
      return 0;
    }
#endif
    if (std::getenv("MEEP_GPU_TEST_STEP_FINITE_ABORT")) {
      run_distributed_step_finite_abort_probe();
      return 1;
    }
    if (std::getenv("MEEP_GPU_TEST_BOUNDARY_LIFETIME_LEAK_ONLY")) {
      require_boundary_double_fence_failure_leaks_safely();
      if (am_master())
        std::cout
            << "PASS: double CUDA boundary fence failure preserves "
               "potentially live allocations until process teardown\n";
      return 0;
    }
    require_coalesced_transfer_batch_planner();
    require_phase_block_operation_index_planner();
    require_source_amplitude_l1_cache();
    require_halo_curl_partition_combinatorics();
    require_automatic_policy_boundaries();
    require_automatic_phase_batch_policy_boundaries();
#if MEEP_HAVE_CUDA
    // These malformed descriptors are rejected before device selection or
    // allocation, so keep the fail-closed contract executable even on a CUDA
    // build whose driver is temporarily unavailable.
    require_overlapping_dft_batch_rejected_before_prepare();
#endif
    if (am_master())
      std::cout << "PASS: coalesced MPI transfer batching preserves tag order; "
                   "phase block maps are exact; "
                   "8,000 halo/curl Yee-box partitions are exact and disjoint; "
                   "malformed DFT mirror aliases fail before CUDA access"
                << '\n';
    if (std::getenv("MEEP_GPU_TEST_POLICY_ONLY")) return 0;
    if (std::getenv("MEEP_GPU_TEST_DYNAMIC_CLAIM_ONLY")) {
      require_failed_dynamic_environment_preserves_device_claim();
      if (am_master())
        std::cout << "PASS: failed dynamic CUDA configuration preserves the "
                     "live MPI-world GPU claim and restored owner cache\n";
      return 0;
    }
    if (std::getenv("MEEP_GPU_TEST_SPLIT_PREFLIGHT")) {
      const int groups = count_processors();
      require(groups > 1,
              "split preflight regression requires multiple MPI ranks");
      divide_parallel_processes(groups);
    }
    if (std::getenv("MEEP_GPU_TEST_PREFLIGHT_ONLY")) {
      gpu::reset_dispatch_statistics();
      const grid_volume gv = vol2d(1.4, 1.2, 8.0);
      structure s(gv, vacuum, no_pml());
      fields f(&s);
      f.use_real_fields();
      gaussian_src_time source(0.27, 0.08);
      f.add_point_source(Ez, source, gv.center(), 1.0);
      if (std::getenv("MEEP_GPU_TEST_RANK_INVALID_DEVICE") &&
          my_rank() != 0)
        setenv("MEEP_GPU_DEVICE", "invalid", 1);
      if (std::getenv("MEEP_GPU_TEST_RANK_BACKEND_MISMATCH"))
        gpu::set_backend(
            my_rank() == 0 ? gpu::backend_mode::automatic
                           : gpu::backend_mode::cpu);
      if (std::getenv("MEEP_GPU_TEST_RANK_AUTO_THRESHOLD_MISMATCH")) {
        gpu::set_backend(gpu::backend_mode::automatic);
        setenv("MEEP_GPU_AUTO_MIN_CELLS",
               my_rank() == 0 ? "0" : "1", 1);
      }
      f.step();
      if (std::getenv("MEEP_GPU_TEST_OUTER_UNWIND_SUCCESS")) {
        bool step_completed = false;
        bool initialization_completed = false;
        bool outer_exception_caught = false;
        try {
          distributed_outer_unwind_probe probe(
              f, step_completed, initialization_completed);
          throw std::runtime_error("injected outer unwind");
        }
        catch (const std::runtime_error &error) {
          outer_exception_caught =
              std::string(error.what()) == "injected outer unwind";
        }
        require(and_to_all(step_completed && initialization_completed &&
                           outer_exception_caught),
                "distributed success guard misclassified an outer unwind");
      }
      if (std::getenv("MEEP_GPU_TEST_DEVICE_SWAP")) {
        const int current_device = gpu::selected_device();
        require(current_device == 0 || current_device == 1,
                "device-swap regression requires CUDA ordinals 0 and 1");
        gpu::select_device(current_device == 0 ? 1 : 0);
        f.step();
      }
      const gpu::dispatch_statistics statistics =
          gpu::get_dispatch_statistics();
      const gpu::field_update_statistics field_updates =
          gpu::get_field_update_statistics();
      const gpu::source_statistics source_updates =
          gpu::get_source_statistics();
      const gpu::boundary_statistics boundary_updates =
          gpu::get_boundary_statistics();
      const gpu::multi_gpu_statistics multi_gpu =
          gpu::get_multi_gpu_statistics();
      constexpr int diagnostic_capacity = 4096;
      char root_diagnostic[diagnostic_capacity] = {};
      if (my_rank() == 0)
        std::snprintf(root_diagnostic, sizeof(root_diagnostic), "%s",
                      gpu::backend_diagnostic().c_str());
      broadcast(0, root_diagnostic, diagnostic_capacity);
      require(gpu::backend_diagnostic() == root_diagnostic,
              "distributed automatic backend diagnostic is not canonical");
      const bool owner_cuda = f.gpu_cuda_execution_selected();
      const bool process_cuda =
          gpu::active_backend() == gpu::backend_mode::cuda;
      const std::string owner_diagnostic =
          f.gpu_execution_diagnostic();
      if (am_master())
        std::cout << "preflight-owner-query: owner_cuda=" << owner_cuda
                  << " process_cuda=" << process_cuda
                  << " owner_diagnostic='" << owner_diagnostic
                  << "' process_diagnostic='" << root_diagnostic << "'\n";
      require(owner_cuda == process_cuda &&
                  owner_diagnostic == root_diagnostic,
              "fields-owner backend query disagrees with preflight result");
      if (std::getenv("MEEP_GPU_TEST_EXPECT_CPU")) {
        require(gpu::active_backend() == gpu::backend_mode::cpu,
                "automatic distributed preflight did not fall back to CPU");
        require(statistics.cpu_curl_calls > 0 &&
                    statistics.cuda_curl_calls == 0,
                "automatic distributed preflight fallback did not use CPU");
        if (gpu::requested_backend() == gpu::backend_mode::automatic) {
          const gpu::runtime_touch_statistics runtime =
              gpu::get_runtime_touch_statistics();
          require(runtime.availability_probes == 0 &&
                      runtime.device_enumerations == 0 &&
                      runtime.device_selections == 0,
                  "small automatic MPI CPU selection touched CUDA");
        }
      }
      else {
        require(gpu::active_backend() == gpu::backend_mode::cuda,
                "distributed preflight selected no CUDA backend");
        require(statistics.cuda_curl_calls > 0 &&
                    statistics.cpu_curl_calls == 0,
                "distributed preflight did not use an exclusive CUDA curl");
        if (gpu::requested_backend() == gpu::backend_mode::automatic) {
          const gpu::runtime_touch_statistics runtime =
              gpu::get_runtime_touch_statistics();
          require(
              and_to_all(runtime.availability_probes > 0 &&
                         runtime.device_enumerations > 0 &&
                         runtime.device_selections > 0),
              "automatic CUDA MPI selection did not perform observable "
              "runtime discovery and device selection on every rank");
        }
      }
      if (std::getenv("MEEP_GPU_TEST_EXPECT_PINNED_TRANSPORT")) {
        require(
            and_to_all(multi_gpu.mpi_messages > 0 &&
                       multi_gpu.pinned_staging_bytes > 0 &&
                       multi_gpu.cuda_aware_bytes == 0),
            "pinned MPI preflight did not move real boundary bytes through "
            "host staging on every rank");
      }
      if (std::getenv("MEEP_GPU_TEST_EXPECT_CUDA_AWARE_TRANSPORT")) {
        require(
            and_to_all(multi_gpu.mpi_messages > 0 &&
                       multi_gpu.cuda_aware_bytes > 0 &&
                       multi_gpu.pinned_staging_bytes == 0),
            "CUDA-aware MPI preflight did not move real boundary bytes "
            "directly on every rank");
      }
      if (std::getenv("MEEP_GPU_TEST_EXPECT_WAITALL_COMPLETION")) {
        std::unique_ptr<comms_manager> completion_probe =
            create_comms_manager();
        require(
            and_to_all(
                comms_supports_cuda_device_buffers(completion_probe.get()) &&
                comms_uses_waitall_completion(completion_probe.get())),
            "automatic CUDA-aware preflight did not instantiate the "
            "validated waitall completion policy on every rank");
      }
      if (std::getenv(
              "MEEP_GPU_TEST_EXPECT_POSITIVE_OVERSUBSCRIPTION")) {
        require(count_processors() == 2,
                "positive GPU oversubscription regression requires two MPI "
                "ranks");
        const char *visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");
        const char *allow_oversubscription =
            std::getenv("MEEP_GPU_ALLOW_OVERSUBSCRIBE");
        require(visible_devices && std::string(visible_devices) == "0" &&
                    allow_oversubscription &&
                    std::string(allow_oversubscription) == "1",
                "positive GPU oversubscription regression requires exactly "
                "one visible GPU and explicit sharing opt-in");
        const std::string local_identifier =
            gpu::selected_device_identifier();
        constexpr int identifier_capacity = 4096;
        char root_identifier[identifier_capacity] = {};
        if (my_rank() == 0)
          std::snprintf(root_identifier, sizeof(root_identifier), "%s",
                        local_identifier.c_str());
        broadcast(0, root_identifier, identifier_capacity);
        require(
            and_to_all(gpu::selected_device() == 0 &&
                       !local_identifier.empty() &&
                       local_identifier == root_identifier),
            "oversubscribed MPI ranks did not share the same physical GPU "
            "identifier through their one-device CUDA namespace");
        const std::uint64_t global_cuda_curls =
            global_sum(statistics.cuda_curl_calls);
        const std::uint64_t global_cuda_field_updates =
            global_sum(field_updates.cuda_update_eh_calls);
        const std::uint64_t global_cuda_source_updates =
            global_sum(source_updates.cuda_update_calls);
        const std::uint64_t global_cuda_source_points =
            global_sum(source_updates.cuda_update_points);
        const std::uint64_t global_cpu_source_updates =
            global_sum(source_updates.cpu_update_calls);
        const std::uint64_t global_cuda_boundary_updates =
            global_sum(boundary_updates.cuda_update_calls);
        const std::uint64_t global_mpi_messages =
            global_sum(multi_gpu.mpi_messages);
        const std::uint64_t global_pinned_bytes =
            global_sum(multi_gpu.pinned_staging_bytes);
        require(
            and_to_all(statistics.cuda_curl_calls > 0 &&
                       statistics.cuda_curl_points > 0 &&
                       statistics.cpu_curl_calls == 0 &&
                       field_updates.cuda_update_eh_calls > 0 &&
                       field_updates.cuda_update_eh_points > 0 &&
                       field_updates.cpu_update_eh_calls == 0 &&
                       boundary_updates.cuda_update_calls > 0 &&
                       boundary_updates.cuda_update_points > 0 &&
                       boundary_updates.cpu_update_calls == 0 &&
                       multi_gpu.mpi_messages > 0 &&
                       multi_gpu.mpi_scalars > 0 &&
                       multi_gpu.pinned_staging_bytes > 0 &&
                       multi_gpu.cuda_aware_bytes == 0),
            "positive GPU oversubscription did not execute CUDA curl, field, "
            "boundary, and pinned MPI transport work on every rank");
        require(global_cuda_source_updates > 0 &&
                    global_cuda_source_points > 0 &&
                    global_cpu_source_updates == 0,
                "positive GPU oversubscription did not execute an exclusive "
                "CUDA source update across the MPI world");
        if (am_master())
          std::cout
              << "positive-oversubscription: ranks=2 visible-devices=1 "
                 "selected-ordinal=0 shared-physical-identifier='"
              << root_identifier << "' cuda-curls="
              << global_cuda_curls
              << " cuda-field-updates="
              << global_cuda_field_updates
              << " cuda-source-updates="
              << global_cuda_source_updates
              << " cuda-boundary-updates="
              << global_cuda_boundary_updates
              << " mpi-messages=" << global_mpi_messages
              << " pinned-bytes=" << global_pinned_bytes << '\n';
      }
      if (am_master()) {
        const gpu::runtime_touch_statistics runtime =
            gpu::get_runtime_touch_statistics();
        std::cout << "preflight-backend: " << root_diagnostic
                  << " runtime-probes=" << runtime.availability_probes
                  << " enumerations=" << runtime.device_enumerations
                  << " selections=" << runtime.device_selections << '\n';
      }
      if (am_master())
        std::cout << "PASS: distributed CUDA preflight regression\n";
      return 0;
    }

    if (std::getenv("MEEP_GPU_TEST_DFT_DECIMATION_ONLY")) {
      require_distributed_dft_decimation_consensus();
      return 0;
    }
    if (count_processors() == 2)
      require_distributed_dft_decimation_consensus();
    require_distributed_component_allocation_consensus();
    require_fields_tile_base_contract();

    std::string diagnostic;
    const bool local_runtime_available =
        gpu::runtime_available(&diagnostic);
    const bool any_runtime_available = or_to_all(local_runtime_available);
    const bool every_runtime_available =
        and_to_all(local_runtime_available);
    bool cuda_available = every_runtime_available;
    if (any_runtime_available != every_runtime_available)
      diagnostic = "CUDA runtime availability differs between MPI ranks";
    const bool automatic_fallback_expected =
        !any_runtime_available;
    if (cuda_available) {
      // Required CUDA performs eager device qualification.  Automatic mode
      // is deliberately lazy and must not touch CUDA until a fields owner
      // survives its host-only workload floor.
      gpu::set_backend(gpu::backend_mode::cuda);
      const bool local_cuda_active =
          gpu::active_backend() == gpu::backend_mode::cuda;
      const bool any_rank_cuda_active = or_to_all(local_cuda_active);
      const bool every_rank_cuda_active = and_to_all(local_cuda_active);
      if (any_rank_cuda_active != every_rank_cuda_active) {
        diagnostic =
            "not every MPI rank can acquire a dedicated compatible GPU";
        cuda_available = false;
      }
      else
        cuda_available = every_rank_cuda_active;
    }
    const bool ade_state_only =
        std::getenv("MEEP_GPU_TEST_ADE_STATE_ONLY") != nullptr;
    const bool mixed_dispersive_only =
        std::getenv("MEEP_GPU_TEST_MIXED_DISPERSIVE_ONLY") != nullptr;
    require(!ade_state_only || cuda_available,
            "production ADE state contract requires CUDA on every rank");
    require(!mixed_dispersive_only || cuda_available,
            "mixed dispersive multi-GPU regression requires CUDA on every "
            "rank");
    if (cuda_available) {
      const std::vector<gpu::device_info> devices = gpu::enumerate_devices();
      std::vector<int> compatible_ordinals;
      for (const gpu::device_info &device : devices)
        if (device.compatible)
          compatible_ordinals.push_back(device.ordinal);
      const char *device_override = std::getenv("MEEP_GPU_DEVICE");
      const int expected_device =
          device_override && *device_override
              ? static_cast<int>(std::strtol(device_override, nullptr, 10))
              : compatible_ordinals.size() == 1
                    ? compatible_ordinals.front()
                    : compatible_ordinals.at(
                          static_cast<std::size_t>(my_node_rank()));
      const bool local_mapping_correct =
          gpu::selected_device() == expected_device;
      require(and_to_all(local_mapping_correct),
              device_override && *device_override
                  ? "explicit per-rank MPI GPU mapping is incorrect"
                  : "default node-local MPI rank to GPU mapping is incorrect");

      if (std::getenv("MEEP_GPU_TEST_DFT_CHECKPOINT_D2H_RANK_FAILURE"))
        trigger_distributed_dft_checkpoint_rank_failure(false);
      if (std::getenv("MEEP_GPU_TEST_DFT_CHECKPOINT_H2D_RANK_FAILURE"))
        trigger_distributed_dft_checkpoint_rank_failure(true);
      if (std::getenv("MEEP_GPU_TEST_DFT_CHECKPOINT_ONLY")) {
        if (count_processors() == 1) {
          require_dft_checkpoint_residency_and_numerics();
          require_dft_checkpoint_failure_atomicity();
          require_zero_value_dft_checkpoint();
        }
        else if (count_processors() == 2)
          require_distributed_zero_work_dft_checkpoint();
        else
          require(false,
                  "focused DFT checkpoint regression requires one or two "
                  "ranks");
        if (am_master())
          std::cout
              << "PASS: DFT checkpoint targeted transfers, resident "
                 "load-minus scale, failure atomicity, empty-rank/zero-value "
                 "semantics, plan invalidation, and CPU oracle\n";
        gpu::set_backend(gpu::backend_mode::cpu);
        return 0;
      }

      if (std::getenv("MEEP_GPU_TEST_DFT_MATERIALIZATION_ONLY")) {
        if (count_processors() == 1)
          {
            require_dft_array_materialization_equivalence();
            require_public_dft_array_materialization_equivalence();
            require_host_synthetic_material_query_without_dft_sync();
            require_cylindrical_axis_zero_measure_materialization();
          }
        else if (count_processors() == 2)
          require_distributed_zero_work_dft_materialization();
        else
          require(false,
                  "focused DFT materialization requires one or two ranks");
        if (am_master())
          std::cout
              << "PASS: resident CUDA get_dft_array CPU oracle, scalar "
                 "telemetry, all-frequency cache, multi-owner discard, "
                 "owner-ABA invalidation, failure-atomic active-phase "
                 "preflight, and distributed zero-work collectives\n";
        gpu::set_backend(gpu::backend_mode::cpu);
        return 0;
      }

      if (std::getenv("MEEP_GPU_TEST_DFT_OUTPUT_STAGING_ONLY")) {
        require_backend_dft_output_staging_contract();
        gpu::set_backend(gpu::backend_mode::cpu);
        return 0;
      }

      if (std::getenv("MEEP_GPU_TEST_DFT_OUTPUT_ATOMICITY_ONLY")) {
        require_public_dft_output_failure_atomicity();
        return 0;
      }

      if (std::getenv("MEEP_GPU_TEST_DFT_OUTPUT_MPI_RENAME_FAILURE_ONLY")) {
        require_public_dft_output_mpi_rename_failure_atomicity();
        return 0;
      }

      if (std::getenv("MEEP_GPU_TEST_DFT_OUTPUT_MPI_ONLY")) {
        require_public_nonempty_dft_output_zero_work_mpi();
        return 0;
      }

      if (std::getenv("MEEP_GPU_TEST_DFT_OUTPUT_COLLAPSED_ONLY")) {
        if (count_processors() == 1) {
          require_public_collapsed_dft_output_cuda();
          require_cylindrical_axis_zero_measure_materialization();
        }
        else if (count_processors() == 2)
          require_public_collapsed_dft_output_zero_owner_mpi();
        else
          require(false,
                  "focused collapsed DFT output requires one or two ranks");
        return 0;
      }

      require_cuda_krylov_vector_primitives();
      require_cuda_cw_field_vector_roundtrip();
      const bool phase_source_only =
          std::getenv("MEEP_GPU_TEST_PHASE_SOURCE_ONLY") != nullptr;
      const bool ldos_reduction_only =
          std::getenv("MEEP_GPU_TEST_LDOS_REDUCTION_ONLY") != nullptr;
      const bool ldos_benchmark_only =
          std::getenv("MEEP_GPU_TEST_LDOS_BENCHMARK_ONLY") != nullptr;
      const bool ldos_migration_only =
          std::getenv("MEEP_GPU_TEST_LDOS_MIGRATION_ONLY") != nullptr;
      require(static_cast<int>(phase_source_only) +
                      static_cast<int>(ldos_reduction_only) +
                      static_cast<int>(ldos_benchmark_only) +
                      static_cast<int>(ldos_migration_only) <=
                  1,
              "focused phase-source, LDOS-only, LDOS benchmark, and LDOS "
              "migration modes are mutually exclusive");
      if (phase_source_only) {
        require_phase_batched_source_split_and_replay_contract();
        if (am_master())
          std::cout << "PASS: focused phase-batched CUDA source duplicate-"
                       "index, replay, replacement, and lifecycle contract\n";
        gpu::set_backend(gpu::backend_mode::cpu);
        return 0;
      }
      if (ldos_reduction_only) {
        require_resident_ldos_reduction_contract();
        require_full_volume_ldos_determinism();
        if (am_master())
          std::cout << "PASS: focused deterministic resident LDOS and full-"
                       "volume electric/magnetic CPU-CUDA contract\n";
        gpu::set_backend(gpu::backend_mode::cpu);
        return 0;
      }
      if (ldos_benchmark_only) {
        require_ldos_transfer_speedup();
        if (am_master())
          std::cout << "PASS: repeated deterministic resident LDOS transfer "
                       "speedup gate\n" << std::flush;
        all_wait();
        publish_ldos_results_ready_and_wait_release();
        all_wait();
        gpu::set_backend(gpu::backend_mode::cpu);
        return 0;
      }
      if (ldos_migration_only) {
        require_resident_ldos_two_device_migration_contract();
        gpu::set_backend(gpu::backend_mode::cpu);
        return 0;
      }
      if (ade_state_only) {
        require_integrated_ade_state_contract(true);
        if (am_master())
          std::cout
              << "PASS: production CPU/CUDA Lorentz/Drude state contract, "
                 "fine-grid conditioning, million-step neutral mode, and "
                 "seeded noisy semantics\n";
        gpu::set_backend(gpu::backend_mode::cpu);
        return 0;
      }
      if (mixed_dispersive_only) {
        require_mixed_dispersive_multi_gpu_contract();
        if (am_master())
          std::cout
              << "PASS: anisotropic Lorentz+Drude CPU/2-GPU agreement and "
                 "remote-boundary transport\n";
        gpu::set_backend(gpu::backend_mode::cpu);
        return 0;
      }
      if (std::getenv("MEEP_GPU_TEST_CW_VECTOR_ONLY")) {
        if (am_master())
          std::cout << "PASS: resident CUDA CW Krylov FP32 vectors, FP64 "
                       "reductions, and owned-field packed roundtrip\n";
        gpu::set_backend(gpu::backend_mode::cpu);
        return 0;
      }
      if (std::getenv("MEEP_GPU_TEST_CW_BREAKDOWN_ONLY")) {
        require_resident_cw_breakdown_recovery();
        if (am_master()) {
          if (count_processors() > 1)
            std::cout
                << "PASS: rank-asymmetric resident CW numerical breakdown "
                   "returns false, preserves finite fields, and recovers on "
                   "the same fields owner\n";
          else
            std::cout
                << "PASS: singleton resident CW numerical breakdown returns "
                   "false, preserves finite fields, and recovers on the same "
                   "fields owner\n";
        }
        gpu::set_backend(gpu::backend_mode::cpu);
        return 0;
      }
      if (std::getenv("MEEP_GPU_TEST_CW_SOLVER_ONLY")) {
        const run_result cpu_cw = run_cw_case(gpu::backend_mode::cpu);
        const run_result cuda_host_cw =
            run_cw_case(gpu::backend_mode::cuda, false);
        const run_result cuda_cw = run_cw_case(gpu::backend_mode::cuda);
        const run_result cuda_auto_cw = run_cw_case(
            gpu::backend_mode::cuda, true, false, 0, true, true);
        compare_cw_results(cpu_cw, cuda_host_cw);
        compare_cw_results(cpu_cw, cuda_cw);
        compare_cw_results(cpu_cw, cuda_auto_cw);
        require(cuda_cw.statistics.cuda_curl_calls > 0 &&
                    cuda_cw.statistics.cpu_curl_calls == 0,
                "resident CW solver probe silently used CPU curls");
        require(cuda_auto_cw.statistics.device_to_host_bytes <
                    cuda_host_cw.statistics.device_to_host_bytes,
                "automatic CUDA CW selection did not eliminate host-vector "
                "Krylov readbacks");
        const run_result cpu_cylindrical_cw =
            run_cylindrical_cw_case(gpu::backend_mode::cpu);
        const run_result cuda_cylindrical_cw =
            run_cylindrical_cw_case(gpu::backend_mode::cuda);
        compare_cw_results(cpu_cylindrical_cw, cuda_cylindrical_cw);
        if (am_master())
          std::cout << "PASS: Cartesian and cylindrical resident CUDA CW "
                       "BiCGSTAB-L agree with the host-vector CPU reference; "
                       "cpu-seconds="
                    << cpu_cw.workload_seconds
                    << " cuda-host-vector-seconds="
                    << cuda_host_cw.workload_seconds
                    << " cuda-resident-seconds="
                    << cuda_cw.workload_seconds
                    << " cuda-auto-seconds="
                    << cuda_auto_cw.workload_seconds << '\n';
        gpu::set_backend(gpu::backend_mode::cpu);
        return 0;
      }
      if (std::getenv("MEEP_GPU_TEST_CW_RESIDENT_SCALE_ONLY")) {
        // L=2 and 100 iterations retain roughly the same 400 Krylov
        // operator applications as the earlier L=10/20 probe while avoiding
        // a high-order unconverged polynomial's FP32 breakdown sensitivity.
        constexpr int fixed_iterations = 100;
        const run_result cuda_cw = run_cw_case(
            gpu::backend_mode::cuda, true, true, fixed_iterations, false);
        require(cuda_cw.every_rank_cuda_curl &&
                    cuda_cw.statistics.cuda_curl_calls > 0 &&
                    cuda_cw.statistics.cpu_curl_calls == 0,
                "resident CW fixed-work scaling probe silently used CPU "
                "curls");
        // This exact L=2/restart-10 fixture performs 420 field-operator
        // applications, including reliable-residual checks. Each produces
        // two CUDA curl dispatches plus the four setup/finalization pairs.
        // An early Krylov breakdown must not be reported as a faster run.
        const std::uint64_t expected_cuda_curl_calls =
            UINT64_C(848) * static_cast<std::uint64_t>(count_processors());
        require(cuda_cw.minimum_rank_cuda_curl_calls == UINT64_C(848) &&
                    cuda_cw.maximum_rank_cuda_curl_calls == UINT64_C(848) &&
                    cuda_cw.statistics.cuda_curl_calls ==
                        expected_cuda_curl_calls,
                "resident CW fixed-work scaling probe did not complete its "
                "declared operator count on every rank");
        const double benchmark_size = cw_benchmark_size();
        const double benchmark_resolution = cw_benchmark_resolution();
        if (am_master()) {
          std::cout << std::setprecision(17)
                    << "cw-resident-fixed-work: ranks="
                    << count_processors()
                    << " size=" << benchmark_size
                    << " resolution=" << benchmark_resolution
                    << " iterations=" << fixed_iterations
                    << " seconds=" << cuda_cw.workload_seconds
                    << " cuda-curl-calls="
                    << cuda_cw.statistics.cuda_curl_calls
                    << " cuda-curl-calls-per-rank-min="
                    << cuda_cw.minimum_rank_cuda_curl_calls
                    << " cuda-curl-calls-per-rank-max="
                    << cuda_cw.maximum_rank_cuda_curl_calls << '\n';
          std::cout << "cw-resident-fixed-work-probes:";
          for (std::size_t index = 0;
               index < std::min<std::size_t>(9, cuda_cw.samples.size());
               ++index)
            std::cout << ' ' << cuda_cw.samples[index].real() << ','
                      << cuda_cw.samples[index].imag();
          std::cout << '\n';
        }
        gpu::set_backend(gpu::backend_mode::cpu);
        return 0;
      }
      if (std::getenv("MEEP_GPU_TEST_CW_BENCHMARK_ONLY")) {
        const run_result cpu_cw =
            run_cw_case(gpu::backend_mode::cpu, false, true);
        const run_result cuda_host_cw =
            run_cw_case(gpu::backend_mode::cuda, false, true);
        const run_result cuda_cw =
            run_cw_case(gpu::backend_mode::cuda, true, true);
        if (am_master())
          std::cout << "cw-resident-benchmark: cpu-seconds="
                    << cpu_cw.workload_seconds
                    << " cuda-host-vector-seconds="
                    << cuda_host_cw.workload_seconds
                    << " cuda-resident-seconds="
                    << cuda_cw.workload_seconds
                    << " resident-vs-host-cuda="
                    << cuda_host_cw.workload_seconds /
                           cuda_cw.workload_seconds
                    << " resident-vs-serial-cpu="
                    << cpu_cw.workload_seconds /
                           cuda_cw.workload_seconds
                    << '\n';
        compare_cw_results(cuda_host_cw, cuda_cw);
        compare_cw_results(cpu_cw, cuda_host_cw);
        compare_cw_results(cpu_cw, cuda_cw);
        require(cuda_cw.workload_seconds < cuda_host_cw.workload_seconds,
                "resident CUDA CW benchmark did not beat host-vector CUDA");
        require(cuda_cw.workload_seconds < cpu_cw.workload_seconds,
                "resident CUDA CW benchmark did not beat serial CPU");
        gpu::set_backend(gpu::backend_mode::cpu);
        return 0;
      }

      if (std::getenv("MEEP_GPU_TEST_DENSE_LONG_ONLY")) {
        require(count_processors() == 1,
                "dense long-horizon whole-field oracle is singleton-only");
        require_dense_long_field_agreement();
        if (am_master())
          std::cout << "PASS: dense long-horizon CPU-FP32/CUDA-FP32 "
                       "whole-field oracle\n";
        gpu::set_backend(gpu::backend_mode::cpu);
        return 0;
      }

      if (std::getenv("MEEP_GPU_TEST_INTEGRATE2_SWAPPED_OWNERSHIP"))
        trigger_integrate2_swapped_ownership();
      if (std::getenv("MEEP_GPU_TEST_INTEGRATE2_RANK_FAILURE"))
        trigger_integrate2_rank_local_sync_failure();
      if (std::getenv("MEEP_GPU_TEST_BOUNDARY_TIMING_ONLY")) {
        require_distributed_boundary_timing();
        gpu::set_backend(gpu::backend_mode::cpu);
        return 0;
      }
      if (std::getenv("MEEP_GPU_TEST_DFT_PHASE_SHARING_ONLY")) {
        require_dft_phase_sharing_equivalence();
        const run_result monitor_cpu =
            run_monitor_consumer_case(gpu::backend_mode::cpu);
        require_monitor_consumer_phase_sharing_equivalence(monitor_cpu);
        if (am_master())
          std::cout << "PASS: complete CUDA DFT arrays and representative "
                       "monitor consumers are bitwise exact with phase "
                       "sharing and multi-monitor batching in automatic, "
                       "forced, and disabled modes\n";
        gpu::set_backend(gpu::backend_mode::cpu);
        return 0;
      }

      const char *saved_auto_minimum =
          std::getenv("MEEP_GPU_AUTO_MIN_CELLS");
      const bool had_saved_auto_minimum =
          saved_auto_minimum != nullptr;
      const std::string saved_auto_minimum_value =
          saved_auto_minimum ? saved_auto_minimum : "";
      setenv("MEEP_GPU_AUTO_MIN_CELLS", "65536", 1);
      const run_result automatic_initialization =
          run_initialize_field_before_step_case(
              gpu::backend_mode::automatic);
      require(
          gpu::active_backend() == gpu::backend_mode::cpu &&
              automatic_initialization.boundaries.cpu_update_calls > 0 &&
              automatic_initialization.boundaries.cuda_update_calls == 0 &&
              automatic_initialization.statistics.cuda_curl_calls == 0 &&
              automatic_initialization.field_updates.cuda_update_eh_calls ==
                  0 &&
              automatic_initialization.polarizations.cuda_update_calls == 0 &&
              automatic_initialization.sources.cuda_update_calls == 0 &&
              automatic_initialization.dfts.cuda_update_calls == 0 &&
              automatic_initialization.statistics.host_to_device_bytes ==
                  0 &&
              automatic_initialization.statistics.device_to_host_bytes ==
                  0 &&
              automatic_initialization.resident.device_buffer_allocations ==
                  0,
          "automatic CUDA workload policy touched CUDA during a small "
          "pre-step field initialization");
      const gpu::runtime_touch_statistics initialization_runtime =
          gpu::get_runtime_touch_statistics();
      require(initialization_runtime.availability_probes == 0 &&
                  initialization_runtime.device_enumerations == 0 &&
                  initialization_runtime.device_selections == 0,
              "automatic small initialization touched the CUDA runtime");
      const run_result automatic_small =
          run_case(gpu::backend_mode::automatic, two_dimensional_case);
      require(gpu::active_backend() == gpu::backend_mode::cpu,
              "automatic CUDA workload policy did not keep a small "
              "domain on the CPU");
      require(
          gpu::requested_backend() == gpu::backend_mode::automatic,
          "automatic CUDA workload policy permanently changed the "
          "process-level requested backend");
      require(automatic_small.statistics.cpu_curl_calls > 0 &&
                  automatic_small.statistics.cuda_curl_calls == 0,
              "automatic CUDA workload policy dispatched a small domain "
              "to CUDA");
      const gpu::runtime_touch_statistics small_runtime =
          gpu::get_runtime_touch_statistics();
      require(small_runtime.availability_probes == 0 &&
                  small_runtime.device_enumerations == 0 &&
                  small_runtime.device_selections == 0,
              "automatic small-domain CPU selection touched CUDA");

      setenv("MEEP_GPU_AUTO_MIN_CELLS", "0", 1);
      const run_result automatic_forced =
          run_case(gpu::backend_mode::automatic, two_dimensional_case,
                   false);
      compare_results(
          automatic_small, automatic_forced, two_dimensional_case);
      require(gpu::active_backend() == gpu::backend_mode::cuda,
              "a later automatic simulation did not re-probe CUDA after "
              "an earlier small-domain CPU fallback");
      require(
          gpu::requested_backend() == gpu::backend_mode::automatic,
          "automatic CUDA re-probe changed the process-level request");
      require(automatic_forced.statistics.cuda_curl_calls > 0 &&
                  automatic_forced.statistics.cpu_curl_calls == 0,
              "zero automatic CUDA threshold did not dispatch CUDA");

      if (count_processors() == 1) {
        // Distributed fields::step deliberately aborts MPI on any escaping
        // rank-local exception so a peer can never hang in a later boundary
        // collective. Catch-and-inspect validation belongs to the singleton
        // regression; a separate expected-failure launcher covers MPI aborts.
        setenv("MEEP_GPU_AUTO_MIN_CELLS", "-1", 1);
        bool invalid_auto_policy_rejected = false;
        try {
          (void)run_case(
              gpu::backend_mode::automatic, one_dimensional_case);
        }
        catch (const std::exception &) {
          invalid_auto_policy_rejected = true;
        }
        require(invalid_auto_policy_rejected,
                "invalid automatic CUDA threshold was not rejected");
      }
      if (had_saved_auto_minimum)
        setenv(
            "MEEP_GPU_AUTO_MIN_CELLS",
            saved_auto_minimum_value.c_str(), 1);
      else
        unsetenv("MEEP_GPU_AUTO_MIN_CELLS");

      if (std::getenv("MEEP_GPU_TEST_DFT_NORM_ONLY")) {
        require_dft_norm_reduction();
        require_persistent_dft_survives_fields_destruction();
        if (am_master())
          std::cout
              << "PASS: contiguous and persistent CUDA DFT norms agree "
                 "with CPU using scalar-only device transfers; persistent "
                 "DFTs survive direct and clear_dft_monitors fields "
                 "destruction paths\n";
        gpu::set_backend(gpu::backend_mode::cpu);
        return 0;
      }
      if (std::getenv("MEEP_GPU_TEST_DFT_NORM_RANK_FAILURE"))
        trigger_distributed_dft_norm_rank_failure();
      if (std::getenv("MEEP_GPU_TEST_CW_RANK_FAILURE"))
        trigger_distributed_cw_rank_failure();
      if (std::getenv("MEEP_GPU_TEST_CW_DIAGNOSTIC_MISMATCH"))
        trigger_distributed_cw_diagnostic_mismatch();
      if (std::getenv("MEEP_GPU_TEST_CW_SOLVER_MISMATCH"))
        trigger_distributed_cw_configuration_mismatch("solver");
      if (std::getenv("MEEP_GPU_TEST_CW_RESTART_MISMATCH"))
        trigger_distributed_cw_configuration_mismatch("restart");
      if (std::getenv("MEEP_GPU_TEST_CW_ORDER_MISMATCH"))
        trigger_distributed_cw_configuration_mismatch("order");
    }
    gpu::set_backend(gpu::backend_mode::cpu);
    const test_dimension cases[] = {
        one_dimensional_case, two_dimensional_case, three_dimensional_case};
    for (test_dimension which : cases) {
      const run_result cpu = run_case(gpu::backend_mode::cpu, which);
      require(cpu.statistics.cpu_curl_calls > 0, "CPU reference dispatched no curl updates");
      require(cpu.statistics.cuda_curl_calls == 0,
              "CPU reference unexpectedly dispatched CUDA");

      if (!cuda_available) {
        if (automatic_fallback_expected) {
          const run_result automatic =
              run_case(gpu::backend_mode::automatic, which);
          compare_results(cpu, automatic, which);
          require(automatic.statistics.cpu_curl_calls > 0,
                  "automatic fallback dispatched no CPU curl updates");
          require(automatic.statistics.cuda_curl_calls == 0,
                  "automatic fallback unexpectedly dispatched CUDA");
        }
        continue;
      }

      const run_result cuda = run_case(gpu::backend_mode::cuda, which);
      compare_results(cpu, cuda, which);
      require(cuda.statistics.cuda_curl_calls > 0, "CUDA mode dispatched no CUDA curl updates");
      require(cuda.statistics.cuda_curl_points > 0, "CUDA mode updated no grid points");
      require(cuda.statistics.cpu_curl_calls == 0,
              "required CUDA mode silently fell back to a CPU curl");
      require(cuda.every_rank_cuda_curl,
              "not every MPI rank used an exclusive CUDA curl path");
      require(cuda.statistics.host_to_device_bytes > 0 &&
                  cuda.statistics.device_to_host_bytes > 0,
              "resident CUDA path did not report host/device transfers");
      require(cuda.resident.device_buffer_allocations > 0 &&
                  cuda.resident.device_buffer_reuses >
                      cuda.resident.device_buffer_allocations,
              "resident CUDA path did not reuse device allocations");
      if (which == two_dimensional_case) {
        require(cuda.resident.host_to_device_bytes_avoided > 0 &&
                    cuda.resident.device_to_host_bytes_avoided > 0,
                "resident CUDA path did not eliminate tiled field transfers");
        if (count_processors() > 1) {
          require(cuda.every_rank_remote_exchange,
                  "not every MPI rank exchanged resident CUDA boundaries");
          require(cuda.multi_gpu.mpi_messages > 0 &&
                      cuda.multi_gpu.mpi_scalars > 0,
                  "multi-GPU CUDA path exchanged no remote boundary data");
          require(cuda.multi_gpu.cuda_aware_bytes +
                          cuda.multi_gpu.pinned_staging_bytes >
                      0,
                  "multi-GPU CUDA path used neither CUDA-aware MPI nor "
                  "pinned host staging");
          const std::string transport =
              std::getenv("MEEP_GPU_MPI_TRANSPORT")
                  ? std::getenv("MEEP_GPU_MPI_TRANSPORT")
                  : "auto";
          const bool any_forced_pinned =
              or_to_all(transport == "pinned" ||
                        transport == "host");
          const bool any_forced_device =
              or_to_all(transport == "cuda-aware" ||
                        transport == "device");
          if (any_forced_pinned)
            require(cuda.multi_gpu.pinned_staging_bytes > 0 &&
                        cuda.multi_gpu.cuda_aware_bytes == 0,
                    "forced pinned MPI transport did not exclusively use "
                    "pinned host staging");
          if (any_forced_device)
            require(cuda.multi_gpu.cuda_aware_bytes > 0 &&
                        cuda.multi_gpu.pinned_staging_bytes == 0,
                    "forced CUDA-aware MPI transport did not exclusively use "
                    "device buffers");
          if (am_master())
            std::cout << "multi-GPU transport="
                      << (cuda.multi_gpu.cuda_aware_bytes
                              ? "cuda-aware"
                              : "pinned")
                      << " messages=" << cuda.multi_gpu.mpi_messages
                      << " scalars=" << cuda.multi_gpu.mpi_scalars
                      << " cuda-aware-bytes="
                      << cuda.multi_gpu.cuda_aware_bytes
                      << " pinned-bytes="
                      << cuda.multi_gpu.pinned_staging_bytes << '\n';
        }
      }
    }

    if (cuda_available) {
      const gyrotropy_model gyrotropic_models[] = {
          GYROTROPIC_LORENTZIAN, GYROTROPIC_DRUDE,
          GYROTROPIC_SATURATED};
      const test_dimension gyrotropic_dimensions[] = {
          two_dimensional_case, three_dimensional_case};
      for (gyrotropy_model model : gyrotropic_models)
        for (test_dimension which : gyrotropic_dimensions) {
          const run_result cpu_gyrotropic = run_gyrotropic_case(
              gpu::backend_mode::cpu, model, which);
          const run_result cuda_gyrotropic = run_gyrotropic_case(
              gpu::backend_mode::cuda, model, which);
          compare_results(
              cpu_gyrotropic, cuda_gyrotropic, which);
          require(
              cuda_gyrotropic.statistics.cuda_curl_calls > 0 &&
                  cuda_gyrotropic.statistics.cpu_curl_calls == 0,
              "required CUDA gyrotropic case silently used CPU curls");
          require(
              cuda_gyrotropic.field_updates.cuda_update_eh_calls > 0 &&
                  cuda_gyrotropic.field_updates.cpu_update_eh_calls == 0,
              "required CUDA gyrotropic case silently used CPU E/H updates");
          require(
              cuda_gyrotropic.polarizations.cuda_update_calls > 0 &&
                  cuda_gyrotropic.polarizations.cpu_update_calls == 0,
              "required CUDA gyrotropic case silently used CPU "
              "polarizations");
          require(
              cuda_gyrotropic.sources.cuda_update_calls > 0 &&
                  cuda_gyrotropic.sources.cpu_update_calls == 0,
              "required CUDA gyrotropic case silently used CPU sources");
          require(
              cuda_gyrotropic.boundaries.cuda_update_calls > 0 &&
                  cuda_gyrotropic.boundaries.cpu_update_calls == 0,
              "required CUDA gyrotropic case silently used CPU boundaries");
          require(
              cuda_gyrotropic.resident.device_buffer_reuses > 0,
              "required CUDA gyrotropic case did not reuse resident mirrors");
          if (count_processors() > 1)
            require(
                cuda_gyrotropic.every_rank_remote_exchange,
                "not every MPI rank exchanged resident gyrotropic "
                "boundaries");
        }
    }

    const test_dimension multilevel_dimensions[] = {
        one_dimensional_case, two_dimensional_case,
        three_dimensional_case};
    for (test_dimension which : multilevel_dimensions) {
      const run_result cpu_multilevel = run_multilevel_case(
          gpu::backend_mode::cpu, which);
      require(
          cpu_multilevel.polarizations.cpu_update_calls > 0 &&
              cpu_multilevel.polarizations.cuda_update_calls == 0,
          "CPU multilevel reference did not use CPU polarizations");
      if (cuda_available) {
        const run_result cuda_multilevel = run_multilevel_case(
            gpu::backend_mode::cuda, which);
        compare_results(cpu_multilevel, cuda_multilevel, which);
        require(
            cuda_multilevel.statistics.cuda_curl_calls > 0 &&
                cuda_multilevel.statistics.cpu_curl_calls == 0,
            "required CUDA multilevel case silently used CPU curls");
        require(
            cuda_multilevel.field_updates.cuda_update_eh_calls > 0 &&
                cuda_multilevel.field_updates.cpu_update_eh_calls == 0,
            "required CUDA multilevel case silently used CPU E/H updates");
        require(
            cuda_multilevel.polarizations.cuda_update_calls > 0 &&
                cuda_multilevel.polarizations.cpu_update_calls == 0,
            "required CUDA multilevel case silently used CPU "
            "polarizations");
        require(
            cuda_multilevel.sources.cuda_update_calls > 0 &&
                cuda_multilevel.sources.cpu_update_calls == 0,
            "required CUDA multilevel case silently used CPU sources");
        require(
            cuda_multilevel.boundaries.cuda_update_calls > 0 &&
                cuda_multilevel.boundaries.cpu_update_calls == 0,
            "required CUDA multilevel case silently used CPU boundaries");
        require(
            cuda_multilevel.resident.device_buffer_reuses > 0,
            "required CUDA multilevel case did not reuse resident mirrors");
        if (count_processors() > 1)
          require(
              cuda_multilevel.every_rank_remote_exchange,
              "not every MPI rank exchanged resident multilevel "
              "boundaries");
      }
    }

    if (cuda_available && count_processors() == 1) {
      const std::uint64_t live_before_failed_plan =
          gpu::get_live_resident_device_buffers();
      gpu::detail::fail_next_multilevel_plan_commit_for_testing();
      bool rejected_plan_commit = false;
      try {
        (void)run_multilevel_case(
            gpu::backend_mode::cuda, one_dimensional_case);
      }
      catch (const std::bad_alloc &) {
        rejected_plan_commit = true;
      }
      require(
          rejected_plan_commit &&
              gpu::get_live_resident_device_buffers() ==
                  live_before_failed_plan,
          "failed multilevel descriptor-plan commit leaked a resident "
          "device buffer");
      const run_result recovered_multilevel =
          run_multilevel_case(
              gpu::backend_mode::cuda, one_dimensional_case);
      require(
          recovered_multilevel.polarizations.cuda_update_calls > 0 &&
              recovered_multilevel.polarizations.cpu_update_calls == 0,
          "multilevel descriptor plan did not recover after an injected "
          "commit failure");
    }

    if (std::getenv("MEEP_GPU_TEST_MULTI_GPU_ONLY")) {
      require_mixed_dispersive_multi_gpu_contract();
      if (am_master())
        std::cout << "PASS: focused multi-GPU mapping, numerical agreement, "
                     "mixed anisotropic Lorentz+Drude, and boundary "
                     "transport checks\n";
      gpu::set_backend(gpu::backend_mode::cpu);
      return 0;
    }

    const bool material_complex_cases[] = {false, true};
    for (bool complex_bloch : material_complex_cases) {
      const run_result cpu_material =
          run_material_case(gpu::backend_mode::cpu, complex_bloch);
      if (!cuda_available) {
        if (automatic_fallback_expected) {
          const run_result automatic_material =
              run_material_case(gpu::backend_mode::automatic, complex_bloch);
          compare_results(cpu_material, automatic_material,
                          two_dimensional_case);
          require(automatic_material.statistics.cpu_curl_calls > 0 &&
                      automatic_material.statistics.cuda_curl_calls == 0,
                  "automatic material fallback did not use CPU curls");
          require(automatic_material.field_updates.cpu_update_eh_calls > 0 &&
                      automatic_material.field_updates.cuda_update_eh_calls == 0,
                  "automatic material fallback did not use CPU E/H updates");
          require(automatic_material.polarizations.cpu_update_calls > 0 &&
                      automatic_material.polarizations.cuda_update_calls == 0,
                  "automatic material fallback did not use CPU polarizations");
          require(automatic_material.sources.cpu_update_calls > 0 &&
                      automatic_material.sources.cuda_update_calls == 0,
                  "automatic material fallback did not use CPU sources");
          require(automatic_material.boundaries.cpu_update_calls > 0 &&
                      automatic_material.boundaries.cuda_update_calls == 0,
                  "automatic material fallback did not use CPU boundaries");
          require(automatic_material.dfts.cpu_update_calls > 0 &&
                      automatic_material.dfts.cuda_update_calls == 0,
                  "automatic material fallback did not use CPU DFT updates");
        }
      }
      else {
        const run_result cuda_material =
            run_material_case(gpu::backend_mode::cuda, complex_bloch);
        compare_results(cpu_material, cuda_material, two_dimensional_case);
        require(cuda_material.statistics.cuda_curl_calls > 0 &&
                    cuda_material.statistics.cpu_curl_calls == 0,
                "required CUDA material phase silently used CPU curls");
        require(cuda_material.field_updates.cuda_update_eh_calls > 0 &&
                    cuda_material.field_updates.cpu_update_eh_calls == 0,
                "required CUDA material phase silently used CPU E/H updates");
        require(cuda_material.polarizations.cuda_update_calls > 0 &&
                    cuda_material.polarizations.cpu_update_calls == 0,
                "required CUDA material phase silently used CPU polarizations");
        require(cuda_material.sources.cuda_update_calls > 0 &&
                    cuda_material.sources.cpu_update_calls == 0,
                "required CUDA material phase silently used CPU sources");
        require(cuda_material.boundaries.cuda_update_calls > 0 &&
                    cuda_material.boundaries.cuda_update_points > 0 &&
                    cuda_material.boundaries.cpu_update_calls == 0,
                "required CUDA material phase silently used CPU boundaries");
        require(cuda_material.dfts.cuda_update_calls > 0 &&
                    cuda_material.dfts.cuda_update_points > 0 &&
                    cuda_material.dfts.cpu_update_calls == 0,
                "required CUDA material phase silently used CPU DFT updates");
        require(cuda_material.resident.device_buffer_allocations > 0 &&
                    cuda_material.resident.device_buffer_reuses > 0,
                "CUDA material phase did not use resident mirrors");
      }
    }
    if (cuda_available) {
      require_phase_batched_launch_equivalence();
      require_curl_replay_direct_transition_contract();
      require_curl_replay_invalidation_and_failure_contract();
      require_phase_batched_source_split_and_replay_contract();
      require_resident_ldos_reduction_contract();
      require_cuda_tile_coalescing_equivalence();
      require_dft_phase_sharing_equivalence();
      // Keep the end-to-end CPU/CUDA FDTD comparison and separately isolate
      // Green-transform error by reusing one device-produced DFT snapshot.
      require_near2far_public_dimension_mismatch_rejected();
      require_near2far_cylindrical_inputs_rejected_without_device();
      require_near2far2d_hankel_bound_sweep();
      require_near2far_cartesian_dimension_cache_key();
      require_near2far_cylindrical_bound_sweep();
      require_near2far_cylindrical_default_tolerance_cancellation_retry();
      require_near2far2d_public_api_equivalence();
      require_near2far2d_public_periodic_constructor();
      require_near2far_cylindrical_public_api_equivalence();
      require_near2far3d_transform_equivalence();
      require_near2far_adjoint_equivalence();
      require_near2far_adjoint_retained_plan_reuse();
      require_near2far_adjoint_cancellation_retry();
      require_near2far3d_same_snapshot_equivalence();
      require_near2far3d_public_periodic_constructor();
      require_near2far3d_symmetry_equivalence();
    }

    // These host consumers and the CW bridge are valid independently of a
    // physical CUDA device, so always exercise the CPU reference on build
    // hosts. A CUDA-capable host compares against these same results below.
    const run_result monitor_cpu =
        run_monitor_consumer_case(gpu::backend_mode::cpu);
    require(
        monitor_cpu.dft_reductions.cpu_reduction_calls > 0 &&
            monitor_cpu.dft_reductions.cpu_submitted_pairs > 0 &&
            monitor_cpu.dft_reductions.cpu_point_frequency_terms > 0 &&
            monitor_cpu.dft_reductions.cuda_reduction_calls == 0 &&
            monitor_cpu.dft_reductions.mpi_allreduce_calls > 0 &&
            monitor_cpu.dft_reductions.mpi_allreduce_bytes > 0,
        "CPU monitor-consumer reference did not prove spectral reductions "
        "and result collectives without CUDA fallback");
    require_near2far_adjoint_cpu_reference();
    const run_result cw_cpu = run_cw_case(
        gpu::backend_mode::cpu, true, false, 0, MEEP_SINGLE != 0);

    if (!cuda_available) {
      if (automatic_fallback_expected) {
        bool required_cuda_failed = false;
        try {
          gpu::set_backend(gpu::backend_mode::cuda);
        }
        catch (const std::runtime_error &) {
          required_cuda_failed = true;
        }
        require(required_cuda_failed,
                "required CUDA mode did not reject an unavailable runtime");
      }
      std::cout << "SKIP: CPU reference passed; uniform MPI CUDA unavailable: "
                << diagnostic << '\n';
      gpu::set_backend(gpu::backend_mode::cpu);
      return 77;
    }

    // A distributed CUDA step deliberately aborts the communicator on a
    // rank-local exception to prevent peers from hanging in boundary
    // collectives. These catch-and-inspect atomicity tests are therefore
    // meaningful only for a single rank.
    if (count_processors() == 1) {
      require_interleaved_automatic_owner_decisions();
      require_resident_scalar_query(false);
      require_resident_scalar_query(true);
      require_finite_check_plan_lifecycle();
      require_boundary_descriptor_fast_replay_contract();
      require_boundary_event_record_failure_fallback();
      require_resident_host_write_transaction();
      require_unsupported_cylindrical_bfast_is_atomic();
      require_unsupported_polarization_is_atomic();
      require_automatic_polarization_owner_fallback_is_atomic();
      require_automatic_owner_revalidates_environment();
      require_automatic_feature_fallback_recovers();
      require_resized_dft_preflight_is_atomic();
      require_mutated_susceptibility_preflight_is_atomic();
      require_dense_long_field_agreement();
    }
    else {
      require_distributed_finite_verdict_aggregation();
    }

    const run_result first_step_cpu =
        run_first_step_material_case(gpu::backend_mode::cpu);
    const run_result first_step_cuda =
        run_first_step_material_case(gpu::backend_mode::cuda);
    compare_results(first_step_cpu, first_step_cuda, two_dimensional_case);
    require(first_step_cuda.field_updates.cuda_update_eh_calls > 0 &&
                first_step_cuda.field_updates.cpu_update_eh_calls == 0,
            "first-step PML/nonlinear case did not use CUDA E/H updates");

    const double beta_values[] = {-0.17, 0.17};
    const bool complex_beta_cases[] = {false, true};
    for (double beta : beta_values)
      for (bool complex_fields : complex_beta_cases) {
        const run_result beta_cpu =
            run_beta_case(gpu::backend_mode::cpu, beta, complex_fields);
        const run_result beta_cuda =
            run_beta_case(gpu::backend_mode::cuda, beta, complex_fields);
        compare_results(beta_cpu, beta_cuda, two_dimensional_case);
        require(beta_cuda.statistics.cuda_curl_calls > 0 &&
                    beta_cuda.statistics.cuda_curl_points > 0 &&
                    beta_cuda.statistics.cpu_curl_calls == 0,
                "required CUDA beta case silently used CPU curls");
        require(beta_cuda.sources.cuda_update_calls > 0 &&
                    beta_cuda.sources.cpu_update_calls == 0,
                "required CUDA beta case silently used CPU sources");
        require(beta_cuda.resident.device_buffer_reuses > 0,
                "required CUDA beta case did not reuse resident mirrors");
      }

    const bool bfast_boolean_cases[] = {false, true};
    for (test_dimension which : cases)
      for (bool use_pml : bfast_boolean_cases)
        for (bool use_conductivity : bfast_boolean_cases)
          for (bool complex_fields : bfast_boolean_cases) {
            const run_result bfast_cpu = run_bfast_case(
                gpu::backend_mode::cpu, which, use_pml,
                use_conductivity, complex_fields);
            const run_result bfast_cuda = run_bfast_case(
                gpu::backend_mode::cuda, which, use_pml,
                use_conductivity, complex_fields);
            compare_results(bfast_cpu, bfast_cuda, which);
            require(
                bfast_cuda.statistics.cuda_curl_calls > 0 &&
                    bfast_cuda.statistics.cuda_curl_points > 0 &&
                    bfast_cuda.statistics.cpu_curl_calls == 0,
                "required CUDA BFAST case silently used CPU curls");
            require(
                bfast_cuda.statistics.cuda_curl_points >
                    bfast_cpu.statistics.cpu_curl_points,
                "required CUDA BFAST case dispatched no distinct BFAST "
                "device work");
            require(
                bfast_cuda.sources.cuda_update_calls > 0 &&
                    bfast_cuda.sources.cpu_update_calls == 0,
                "required CUDA BFAST case silently used CPU sources");
            require(
                bfast_cuda.resident.device_buffer_reuses > 0,
                "required CUDA BFAST case did not reuse resident mirrors");
          }

    struct cylindrical_configuration {
      int azimuthal_m;
      bool real_fields;
      bool zero_near_origin;
      bool dispersive;
      bool use_pml;
    };
    const cylindrical_configuration cylindrical_cases[] = {
        {0, true, true, false, true},    {0, false, true, true, true},
        {1, false, true, false, true},   {-1, false, true, false, true},
        {2, false, true, false, true},   {-2, false, false, false, true},
        {0, false, true, false, false},  {1, false, true, false, false}};
    for (const cylindrical_configuration &configuration :
         cylindrical_cases) {
      const run_result cylindrical_cpu = run_cylindrical_case(
          gpu::backend_mode::cpu, configuration.azimuthal_m,
          configuration.real_fields, configuration.zero_near_origin,
          configuration.dispersive, configuration.use_pml);
      const run_result cylindrical_cuda = run_cylindrical_case(
          gpu::backend_mode::cuda, configuration.azimuthal_m,
          configuration.real_fields, configuration.zero_near_origin,
          configuration.dispersive, configuration.use_pml);
      compare_results(
          cylindrical_cpu, cylindrical_cuda, two_dimensional_case);
      require(cylindrical_cuda.statistics.cuda_curl_calls > 0 &&
                  cylindrical_cuda.statistics.cuda_curl_points > 0 &&
                  cylindrical_cuda.statistics.cpu_curl_calls == 0,
              "required CUDA cylindrical case silently used CPU curls");
      require(cylindrical_cuda.field_updates.cuda_update_eh_calls > 0 &&
                  cylindrical_cuda.field_updates.cpu_update_eh_calls == 0,
              "required CUDA cylindrical case silently used CPU E/H updates");
      require(cylindrical_cuda.sources.cuda_update_calls > 0 &&
                  cylindrical_cuda.sources.cpu_update_calls == 0,
              "required CUDA cylindrical case silently used CPU sources");
      if (configuration.dispersive)
        require(
            cylindrical_cuda.polarizations.cuda_update_calls > 0 &&
                cylindrical_cuda.polarizations.cpu_update_calls == 0,
            "required CUDA cylindrical dispersion silently used CPU updates");
      require(cylindrical_cuda.boundaries.cuda_update_calls > 0 &&
                  cylindrical_cuda.boundaries.cpu_update_calls == 0,
              "required CUDA cylindrical case silently used CPU boundaries");
      require(cylindrical_cuda.dfts.cuda_update_calls > 0 &&
                  cylindrical_cuda.dfts.cpu_update_calls == 0,
              "required CUDA cylindrical case silently used CPU DFT updates");
      require(cylindrical_cuda.resident.device_buffer_reuses > 0,
              "required CUDA cylindrical case did not reuse resident mirrors");
    }

    const run_result transition_cpu = run_backend_transition_case(false);
    const run_result transition_cuda_cpu = run_backend_transition_case(true);
    compare_results(transition_cpu, transition_cuda_cpu,
                    two_dimensional_case);
    require(transition_cuda_cpu.statistics.cuda_curl_calls > 0 &&
                transition_cuda_cpu.statistics.cpu_curl_calls > 0,
            "backend-transition case did not exercise CUDA then CPU curls");

    const run_result initialize_cpu =
        run_initialize_field_case(gpu::backend_mode::cpu);
    const run_result initialize_cuda =
        run_initialize_field_case(gpu::backend_mode::cuda);
    compare_results(initialize_cpu, initialize_cuda, two_dimensional_case);
    require_fields_copy_lifecycle();

    require_integrated_source_cache_lifecycle();
    require_dft_reset_lifecycle();
    require_shifted_dft_lifecycle();
    require_dft_norm_reduction();
    require_persistent_dft_survives_fields_destruction();
    require_mutated_dft_frequencies();

    require_monitor_consumer_phase_sharing_equivalence(monitor_cpu);

    const run_result cw_cuda = run_cw_case(
        gpu::backend_mode::cuda, true, false, 0, true, true);
    compare_cw_results(cw_cpu, cw_cuda);
    require(cw_cuda.statistics.cuda_curl_calls > 0 &&
                cw_cuda.statistics.cpu_curl_calls == 0 &&
                cw_cuda.field_updates.cuda_update_eh_calls > 0 &&
                cw_cuda.field_updates.cpu_update_eh_calls == 0,
            "CW lifecycle case silently used CPU field updates");
    if (am_master())
      std::cout << "cw-resident-lifecycle: allocations="
                << cw_cuda.resident.device_buffer_allocations
                << " reuses=" << cw_cuda.resident.device_buffer_reuses
                << " cuda-curl-calls="
                << cw_cuda.statistics.cuda_curl_calls << '\n';
    const std::uint64_t maximum_bounded_cw_allocations =
        768u * static_cast<std::uint64_t>(count_processors());
    require(cw_cuda.resident.device_buffer_allocations > 0 &&
                cw_cuda.resident.device_buffer_allocations <=
                    maximum_bounded_cw_allocations &&
                cw_cuda.resident.device_buffer_reuses >
                    8 * cw_cuda.resident.device_buffer_allocations,
            "CW lifecycle allocations were not bounded independently of "
            "Krylov curl volume: allocations=" +
                std::to_string(
                    cw_cuda.resident.device_buffer_allocations) +
                " maximum_bounded_allocations=" +
                std::to_string(maximum_bounded_cw_allocations) +
                " reuses=" +
                std::to_string(cw_cuda.resident.device_buffer_reuses) +
                " cuda_curl_calls=" +
                std::to_string(cw_cuda.statistics.cuda_curl_calls));

    std::cout << "PASS: CPU and resident CUDA FP32 step_db agree in 1D, tiled 2D, "
                 "3D, PML/conductivity, complex Bloch, 1024-step dense "
                 "whole-field L-infinity/L2, integrated-source DFT "
                 "consumer/lifecycle, and CW cases\n";
    gpu::set_backend(gpu::backend_mode::cpu);
    return 0;
  }
  catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
