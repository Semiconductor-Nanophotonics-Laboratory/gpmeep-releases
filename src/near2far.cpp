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

/* Near-to-far field transformation: compute DFT of tangential fields on
   a "near" surface, and use these (via the equivalence principle) to
   compute the fields on a "far" surface via the homogeneous-medium Green's
   function in 2d or 3d. */

#include "meep_internals.hpp"
#include "gpu_backend_internal.hpp"
#include <assert.h>
#include "config.h"
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <limits>
#include <math.h>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace std;

namespace meep {

namespace {

bool near2far_exception_is_in_flight() noexcept {
#if __cplusplus >= 201703L
  return std::uncaught_exceptions() > 0;
#else
  return std::uncaught_exception();
#endif
}

class distributed_near2far_failure_guard {
public:
  explicit distributed_near2far_failure_guard(bool armed)
      : armed_(armed) {}

  ~distributed_near2far_failure_guard() {
    if (armed_ && near2far_exception_is_in_flight())
      meep::abort(
          "rank-local failure during distributed near-to-far processing; "
          "aborting the communicator to prevent an MPI collective deadlock");
  }

  void dismiss() noexcept { armed_ = false; }

private:
  bool armed_;
};

size_t checked_near2far_product(size_t left, size_t right,
                                const char *label) {
  if (left && right > std::numeric_limits<size_t>::max() / left)
    throw std::overflow_error(std::string("near-to-far ") + label +
                              " overflow");
  return left * right;
}

void sum_to_all_large(const std::complex<double> *input,
                      std::complex<double> *output, size_t count) {
  size_t offset = 0;
  while (offset < count) {
    // The complex overload converts the element count to twice as many
    // doubles in an int-valued MPI API.
    const size_t maximum_complex_batch =
        static_cast<size_t>(std::numeric_limits<int>::max()) / 2;
    const int batch = static_cast<int>(std::min<size_t>(
        count - offset, maximum_complex_batch));
    gpu::detail::record_near2far_mpi_allreduce(
        static_cast<size_t>(batch) * sizeof(std::complex<double>));
    sum_to_all(input + offset, output + offset, batch);
    offset += static_cast<size_t>(batch);
  }
}

void sum_to_all_large(const double *input, double *output, size_t count) {
  size_t offset = 0;
  while (offset < count) {
    const int batch = static_cast<int>(std::min<size_t>(
        count - offset, static_cast<size_t>(std::numeric_limits<int>::max())));
    gpu::detail::record_near2far_mpi_allreduce(
        static_cast<size_t>(batch) * sizeof(double));
    sum_to_all(input + offset, output + offset, batch);
    offset += static_cast<size_t>(batch);
  }
}

bool near2far_collective_requires_mixed(const double *local_records,
                                        double *global_records,
                                        size_t work_count) {
  const size_t scalar_count = checked_near2far_product(
      work_count, static_cast<size_t>(13),
      "CUDA collective result and condition count");
  sum_to_all_large(local_records, global_records, scalar_count);
  constexpr double relative_budget = 3.0e-3;
  constexpr double absolute_budget = 3.0e-8;
  bool retry_required = false;
  for (size_t work = 0; work < work_count; ++work) {
    double maximum_magnitude = 0.0;
    for (int channel = 0; channel < 12; ++channel)
      maximum_magnitude = std::max(
          maximum_magnitude,
          std::abs(global_records[13 * work + channel]));
    const double error_bound = global_records[13 * work + 12];
    if (std::isnan(error_bound) || error_bound < 0.0)
      throw std::runtime_error(
          "Meep CUDA near-to-far collective produced invalid cancellation "
          "evidence");
    retry_required = retry_required || !std::isfinite(error_bound) ||
                     error_bound > relative_budget * maximum_magnitude +
                                       absolute_budget;
  }
  return retry_required;
}

bool resident_near2far_opted_out() noexcept {
  const char *disabled = std::getenv("MEEP_GPU_DISABLE_RESIDENT_NEAR2FAR");
  return disabled && disabled[0] == '1' && disabled[1] == '\0';
}

bool near2far_dimension_pair_is_supported(ndim monitor_dimension,
                                          ndim observation_dimension) {
  const bool dimensions_are_known =
      (monitor_dimension == D2 || monitor_dimension == D3 ||
       monitor_dimension == Dcyl) &&
      (observation_dimension == D2 || observation_dimension == D3 ||
       observation_dimension == Dcyl);
  return dimensions_are_known &&
         (monitor_dimension == observation_dimension ||
          (monitor_dimension == Dcyl && observation_dimension == D3));
}

bool cuda_near2far_selected(ndim observation_dimension,
                            const dft_near2far &transform,
                            double greencyl_tol) {
  const ndim monitor_dimension = transform.where.dim;
  // A cylindrical monitor represents a rotated 3D equivalent-current ring,
  // so greencyl can evaluate either an rz-plane point or an arbitrary
  // Cartesian (x,y,z) point. Other monitor types retain their exact
  // dimensionality; projecting those sources would change the problem.
  if (!near2far_dimension_pair_is_supported(
          monitor_dimension, observation_dimension))
    return false;
  if (monitor_dimension == D2 &&
      std::any_of(transform.freq.begin(), transform.freq.end(),
                  [](double frequency) {
                    return !std::isfinite(frequency) || frequency <= 0.0;
                  }))
    return false;
  if (monitor_dimension == Dcyl &&
      std::any_of(transform.freq.begin(), transform.freq.end(),
                  [](double frequency) {
                    return !std::isfinite(frequency) || frequency == 0.0;
                  }))
    return false;
  if (monitor_dimension == Dcyl &&
      (!std::isfinite(greencyl_tol) || greencyl_tol <= 0.0))
    return false;
  return sizeof(realnum) == sizeof(float) &&
         !resident_near2far_opted_out() &&
         gpu::active_backend() == gpu::backend_mode::cuda;
}

enum class collective_cuda_selection { cpu, cuda, inconsistent };

collective_cuda_selection consensus_cuda_near2far_selection(
    ndim dimension, const dft_near2far &transform,
    double greencyl_tol) {
  // Public Near2Far APIs must enter one identical result collective on every
  // rank. A rank-local observation/monitor dimension, opt-out, or backend
  // mismatch would otherwise split the communicator between different
  // Green-function setup collectives or between the 13-double CUDA evidence
  // reduction and the 12-scalar CPU reduction. Fail closed before any of
  // those shapes is entered.
  int root_dimensions[2] = {
      static_cast<int>(dimension), static_cast<int>(transform.where.dim)};
  broadcast(0, root_dimensions, 2);
  const bool dimensions_are_consistent = and_to_all(
      static_cast<int>(dimension) == root_dimensions[0] &&
      static_cast<int>(transform.where.dim) == root_dimensions[1]);
  if (!dimensions_are_consistent)
    return collective_cuda_selection::inconsistent;
  const int selected_rank_count =
      sum_to_all(cuda_near2far_selected(
                     dimension, transform, greencyl_tol) ? 1 : 0);
  if (selected_rank_count == 0) return collective_cuda_selection::cpu;
  if (selected_rank_count == count_processors())
    return collective_cuda_selection::cuda;
  return collective_cuda_selection::inconsistent;
}

bool collective_farfields_metadata_is_consistent(
    const vec *points, size_t point_count, size_t frequency_count,
    const dft_near2far &transform, double greencyl_tol, ndim &dimension) {
  const bool pointers_valid =
      and_to_all(points != nullptr || point_count == 0);
  const bool local_pointer_valid = points != nullptr || point_count == 0;
  size_t root_sizes[2] = {point_count, frequency_count};
  int root_dimensions[2] = {
      point_count == 0 || !local_pointer_valid
          ? -1
          : static_cast<int>(points[0].dim),
      static_cast<int>(transform.where.dim)};
  broadcast(0, root_sizes, 2);
  broadcast(0, root_dimensions, 2);
  bool consistent = pointers_valid && point_count == root_sizes[0] &&
                    frequency_count == root_sizes[1] &&
                    static_cast<int>(transform.where.dim) ==
                        root_dimensions[1] &&
                    (point_count == 0 ||
                     static_cast<int>(points[0].dim) ==
                         root_dimensions[0]);
  if (point_count != 0 && local_pointer_valid) {
    const ndim local_dimension = points[0].dim;
    consistent = consistent &&
                 (local_dimension == D3 || local_dimension == D2 ||
                  local_dimension == Dcyl);
    for (size_t point = 1; point < point_count; ++point)
      consistent = consistent && points[point].dim == local_dimension;
  }
  consistent = and_to_all(consistent);
  if (consistent && point_count != 0) {
    dimension = static_cast<ndim>(root_dimensions[0]);
    std::vector<double> local_values;
    local_values.reserve(point_count * 3 + transform.freq.size() + 8);
    for (size_t point = 0; point < point_count; ++point) {
      if (dimension == Dcyl) {
        local_values.push_back(points[point].r());
        local_values.push_back(points[point].z());
      }
      else {
        local_values.push_back(points[point].x());
        local_values.push_back(points[point].y());
        if (dimension == D3) local_values.push_back(points[point].z());
      }
    }
    local_values.insert(local_values.end(), transform.freq.begin(),
                        transform.freq.end());
    local_values.push_back(transform.eps);
    local_values.push_back(transform.mu);
    local_values.push_back(greencyl_tol);
    for (int axis = 0; axis < 2; ++axis) {
      local_values.push_back(transform.periodic_k[axis]);
      local_values.push_back(transform.period[axis]);
    }
    std::vector<double> root_values(local_values);
    size_t offset = 0;
    while (offset < root_values.size()) {
      const int batch = static_cast<int>(std::min<size_t>(
          root_values.size() - offset,
          static_cast<size_t>(std::numeric_limits<int>::max())));
      broadcast(0, root_values.data() + offset, batch);
      offset += static_cast<size_t>(batch);
    }
    int local_integers[4] = {
        static_cast<int>(transform.periodic_d[0]), transform.periodic_n[0],
        static_cast<int>(transform.periodic_d[1]), transform.periodic_n[1]};
    int root_integers[4] = {local_integers[0], local_integers[1],
                            local_integers[2], local_integers[3]};
    broadcast(0, root_integers, 4);
    consistent = local_values == root_values &&
                 std::equal(local_integers, local_integers + 4,
                            root_integers);
    consistent = and_to_all(consistent);
  }
  return consistent;
}

bool collective_grid_metadata_is_consistent(
    const volume &where, double resolution, const size_t dims[3], size_t count,
    size_t frequency_count, int rank, const dft_near2far &transform,
    double greencyl_tol) {
  size_t root_sizes[5] =
      {count, frequency_count, dims[0], dims[1], dims[2]};
  int root_ints[3] = {rank, static_cast<int>(where.dim),
                      static_cast<int>(transform.where.dim)};
  double root_geometry[7] = {resolution, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const direction axes[3] = {X, Y, Z};
  int coordinate_count = where.dim == D3 ? 3 : where.dim == D2 ? 2 : 2;
  if (where.dim == Dcyl) {
    root_geometry[1] = where.in_direction_min(R);
    root_geometry[4] = where.in_direction_max(R);
    root_geometry[2] = where.in_direction_min(Z);
    root_geometry[5] = where.in_direction_max(Z);
  }
  else
    for (int axis = 0; axis < coordinate_count; ++axis) {
      root_geometry[1 + axis] = where.in_direction_min(axes[axis]);
      root_geometry[4 + axis] = where.in_direction_max(axes[axis]);
    }
  broadcast(0, root_sizes, 5);
  broadcast(0, root_ints, 3);
  broadcast(0, root_geometry, 7);
  bool consistent = count == root_sizes[0] &&
                    frequency_count == root_sizes[1] &&
                    dims[0] == root_sizes[2] && dims[1] == root_sizes[3] &&
                    dims[2] == root_sizes[4] && rank == root_ints[0] &&
                    static_cast<int>(where.dim) == root_ints[1] &&
                    static_cast<int>(transform.where.dim) == root_ints[2] &&
                    resolution == root_geometry[0];
  double local_geometry[6] = {};
  if (where.dim == Dcyl) {
    local_geometry[0] = where.in_direction_min(R);
    local_geometry[3] = where.in_direction_max(R);
    local_geometry[1] = where.in_direction_min(Z);
    local_geometry[4] = where.in_direction_max(Z);
  }
  else
    for (int axis = 0; axis < coordinate_count; ++axis) {
      local_geometry[axis] = where.in_direction_min(axes[axis]);
      local_geometry[3 + axis] = where.in_direction_max(axes[axis]);
    }
  for (int coordinate = 0; coordinate < 6; ++coordinate)
    consistent =
        consistent && local_geometry[coordinate] == root_geometry[1 + coordinate];
  consistent = and_to_all(consistent);
  if (!consistent) return false;
  std::vector<double> local_values(transform.freq.begin(),
                                   transform.freq.end());
  local_values.push_back(transform.eps);
  local_values.push_back(transform.mu);
  local_values.push_back(greencyl_tol);
  for (int axis = 0; axis < 2; ++axis) {
    local_values.push_back(transform.periodic_k[axis]);
    local_values.push_back(transform.period[axis]);
  }
  std::vector<double> root_values(local_values);
  size_t metadata_offset = 0;
  while (metadata_offset < root_values.size()) {
    const int batch = static_cast<int>(std::min<size_t>(
        root_values.size() - metadata_offset,
        static_cast<size_t>(std::numeric_limits<int>::max())));
    broadcast(0, root_values.data() + metadata_offset, batch);
    metadata_offset += static_cast<size_t>(batch);
  }
  int local_integers[4] = {
      static_cast<int>(transform.periodic_d[0]), transform.periodic_n[0],
      static_cast<int>(transform.periodic_d[1]), transform.periodic_n[1]};
  int root_integers[4] = {local_integers[0], local_integers[1],
                          local_integers[2], local_integers[3]};
  broadcast(0, root_integers, 4);
  consistent = consistent && local_values == root_values &&
               std::equal(local_integers, local_integers + 4, root_integers);
  return and_to_all(consistent);
}

int cartesian_axis(direction axis) {
  switch (axis) {
    case X: return 0;
    case Y: return 1;
    case Z: return 2;
    default: return -1;
  }
}

void set_cartesian_axis(gpu::detail::near2far_point_fp64 &point,
                        direction axis, double value) {
  switch (axis) {
    case X: point.x = value; break;
    case Y: point.y = value; break;
    case Z: point.z = value; break;
    default: break;
  }
}

gpu::detail::near2far_point_fp64 cartesian_near2far_point(
    const vec &point, ndim dimension) {
  if (dimension == D2) return {point.x(), point.y(), 0.0};
  if (dimension == D3) return {point.x(), point.y(), point.z()};
  if (dimension == Dcyl) return {point.r(), 0.0, point.z()};
  throw std::invalid_argument(
      "CUDA near-to-far point must be 2D, 3D, or cylindrical");
}

int near2far_direction_index(direction axis, ndim dimension) {
  if (dimension != Dcyl) return cartesian_axis(axis);
  switch (axis) {
    case R: return 0;
    case P: return 1;
    case Z: return 2;
    default: return -1;
  }
}

gpu::detail::near2far_cartesian_dimension near2far_runtime_dimension(
    ndim dimension) {
  if (dimension == D2)
    return gpu::detail::near2far_cartesian_dimension::two;
  if (dimension == D3)
    return gpu::detail::near2far_cartesian_dimension::three;
  if (dimension == Dcyl)
    return gpu::detail::near2far_cartesian_dimension::cylindrical;
  throw std::invalid_argument("unsupported CUDA Near2Far dimension");
}

std::vector<gpu::detail::near2far_periodic_copy_fp64>
build_near2far_periodic_copies(const dft_near2far &transform,
                               ndim dimension) {
  for (int axis = 0; axis < 2; ++axis) {
    if (transform.periodic_n[axis] < 0)
      throw std::invalid_argument(
          "near-to-far periodic image counts must be nonnegative");
    if (transform.periodic_n[axis] == std::numeric_limits<int>::max())
      throw std::overflow_error(
          "near-to-far periodic image count is too large");
    const direction periodic_direction = transform.periodic_d[axis];
    const bool valid_direction =
        periodic_direction == NO_DIRECTION ||
        (dimension == Dcyl
             ? (periodic_direction == R || periodic_direction == Z)
             : (cartesian_axis(periodic_direction) >= 0 &&
                (dimension != D2 || periodic_direction != Z)));
    if (!valid_direction)
      throw std::invalid_argument(
          "CUDA near-to-far periodic direction must belong to the selected "
          "dimension");
  }
  if (transform.periodic_d[0] != NO_DIRECTION &&
      transform.periodic_d[0] == transform.periodic_d[1])
    throw std::invalid_argument(
        "CUDA near-to-far requires distinct periodic directions");
  const size_t count0 = checked_near2far_product(
      static_cast<size_t>(transform.periodic_n[0]), 2,
      "periodic image count") + 1;
  const size_t count1 = checked_near2far_product(
      static_cast<size_t>(transform.periodic_n[1]), 2,
      "periodic image count") + 1;
  const size_t copy_count = checked_near2far_product(
      count0, count1, "periodic copy count");
  std::vector<gpu::detail::near2far_periodic_copy_fp64> copies;
  if (copy_count > copies.max_size())
    throw std::length_error("near-to-far periodic copy count is too large");
  copies.reserve(copy_count);
  const long long n0 = transform.periodic_n[0];
  const long long n1 = transform.periodic_n[1];
  for (long long image0 = -n0; image0 <= n0; ++image0)
    for (long long image1 = -n1; image1 <= n1; ++image1) {
      gpu::detail::near2far_point_fp64 displacement = {0.0, 0.0, 0.0};
      const auto set_displacement = [dimension, &displacement](
          direction axis, double value) {
        if (dimension == Dcyl) {
          if (axis == R) displacement.x = value;
          else if (axis == Z) displacement.z = value;
        }
        else
          set_cartesian_axis(displacement, axis, value);
      };
      if (transform.periodic_d[0] != NO_DIRECTION)
        set_displacement(transform.periodic_d[0],
                         static_cast<double>(image0) * transform.period[0]);
      if (transform.periodic_d[1] != NO_DIRECTION)
        set_displacement(transform.periodic_d[1],
                         static_cast<double>(image1) * transform.period[1]);
      const double phase =
          static_cast<double>(image0) * transform.periodic_k[0] +
          static_cast<double>(image1) * transform.periodic_k[1];
      const std::complex<double> multiplier = std::polar(1.0, phase);
      copies.push_back(
          {displacement, real(multiplier), imag(multiplier)});
    }
  return copies;
}

bool cuda_near2far_cartesian(
    const dft_near2far &transform,
    ndim observation_dimension,
    const std::vector<gpu::detail::near2far_point_fp64> &targets,
    std::vector<std::complex<double> > &output,
    bool reduce_collectively, double greencyl_tol,
    bool force_rank_local_mixed = false) {
  if (!cuda_near2far_selected(
          observation_dimension, transform, greencyl_tol))
    return false;
  const ndim monitor_dimension = transform.where.dim;
  if (targets.empty() || transform.freq.empty()) {
    output.clear();
    return true;
  }

  size_t chunk_count = 0;
  for (dft_chunk *chunk = transform.F; chunk;
       chunk = chunk->next_in_dft)
    if (chunk->N) ++chunk_count;
  const size_t work_count = checked_near2far_product(
      targets.size(), transform.freq.size(), "CUDA work count");
  output.assign(checked_near2far_product(
                    work_count, static_cast<size_t>(6),
                    "CUDA complex output count"),
                std::complex<double>(0.0, 0.0));
  static_assert(sizeof(realnum) != sizeof(float) ||
                    sizeof(std::complex<realnum>) == 2 * sizeof(float),
                "CUDA near-to-far requires interleaved FP32 DFT storage");

  std::vector<std::vector<gpu::detail::near2far_point_fp64> > coordinates;
  std::vector<gpu::detail::near2far_request_fp32> requests;
  coordinates.reserve(chunk_count);
  requests.reserve(chunk_count);
  const size_t frequency_count = transform.freq.size();
  double azimuthal_mode = 0.0;
  bool have_azimuthal_mode = false;
  for (dft_chunk *chunk = transform.F; chunk;
       chunk = chunk->next_in_dft) {
    if (chunk->N == 0) continue;
    if (chunk->omega.size() != frequency_count)
      throw std::invalid_argument(
          "near-to-far DFT chunk frequency count mismatch");
    component source_component = component(chunk->vc);
    const direction source_direction = component_direction(source_component);
    const int direction_index =
        near2far_direction_index(source_direction, monitor_dimension);
    if (direction_index < 0 ||
        (!is_electric(source_component) && !is_magnetic(source_component)))
      throw std::invalid_argument(
          "CUDA near-to-far equivalent source is not a supported E/H "
          "component");
    if (monitor_dimension == Dcyl) {
      const double chunk_mode = chunk->fc->m;
      if (!std::isfinite(chunk_mode) ||
          (have_azimuthal_mode && chunk_mode != azimuthal_mode))
        throw std::invalid_argument(
            "CUDA cylindrical Near2Far chunks have inconsistent azimuthal "
            "modes");
      azimuthal_mode = chunk_mode;
      have_azimuthal_mode = true;
    }

    coordinates.emplace_back();
    std::vector<gpu::detail::near2far_point_fp64> &points =
        coordinates.back();
    points.reserve(chunk->N);
    vec rshift(chunk->shift * (0.5 * chunk->fc->gv.inva));
    LOOP_OVER_IVECS(chunk->fc->gv, chunk->is, chunk->ie, idx) {
      IVEC_LOOP_LOC(chunk->fc->gv, source);
      source = chunk->S.transform(source, chunk->sn) + rshift;
      points.push_back(
          cartesian_near2far_point(source, monitor_dimension));
    }
    if (points.size() != chunk->N)
      throw std::logic_error(
          "near-to-far source-coordinate count differs from DFT storage");
    requests.push_back(
        {chunk->fc, reinterpret_cast<float *>(chunk->dft), points.data(),
         points.size(), direction_index, is_electric(source_component)});
  }

  // A rank with no local DFT chunks must still enter the cylindrical result
  // collective with the same azimuthal mode as the ranks that own monitor
  // work.  Derive the mode from the first owning rank, then require every
  // other owner to agree before any CUDA kernel or 13-double reduction is
  // entered.  This also prevents rank-local plan arguments from silently
  // disagreeing in a distributed transform.
  if (monitor_dimension == Dcyl && reduce_collectively) {
    const int rank_count = count_processors();
    const int mode_owner =
        min_to_all(have_azimuthal_mode ? my_rank() : rank_count);
    double collective_mode = have_azimuthal_mode ? azimuthal_mode : 0.0;
    if (mode_owner < rank_count) broadcast(mode_owner, &collective_mode, 1);
    const bool mode_is_consistent = and_to_all(
        !have_azimuthal_mode || azimuthal_mode == collective_mode);
    if (!mode_is_consistent)
      throw std::runtime_error(
          "inconsistent cylindrical Near2Far azimuthal mode across MPI "
          "ranks");
    azimuthal_mode = collective_mode;
  }

  const std::vector<gpu::detail::near2far_periodic_copy_fp64> copies =
      build_near2far_periodic_copies(transform, monitor_dimension);
  std::vector<double> local_error_bounds(work_count, 0.0);
  bool local_used_mixed_precision = false;
  if (chunk_count != 0)
    gpu::detail::resident_near2far_cartesian_fp32(
        near2far_runtime_dimension(monitor_dimension),
        transform.F, requests.data(), requests.size(), targets.data(),
        targets.size(), transform.freq.data(), transform.freq.size(),
        copies.data(), copies.size(), transform.eps, transform.mu,
        output.data(), reduce_collectively ? local_error_bounds.data() : nullptr,
        reduce_collectively, force_rank_local_mixed, false,
        &local_used_mixed_precision, azimuthal_mode,
        monitor_dimension == Dcyl ? greencyl_tol : 0.0);
  // farfield_lowlevel has historically returned rank-local contributions; its
  // callers own the MPI reduction. Preserve that contract while public batch
  // and grid APIs request the collective cancellation-safe path below.
  if (!reduce_collectively) return true;

  // Reduce the twelve signed result scalars and their conservative FP32 error
  // bound in one collective. Rank-local cancellation checks are insufficient:
  // individually well-conditioned rank contributions can cancel only after the
  // MPI sum. The thirteenth scalar makes that global condition observable while
  // adding one double per target/frequency to the normal communication path.
  const size_t collective_scalar_count = checked_near2far_product(
      work_count, static_cast<size_t>(13),
      "CUDA collective result and condition count");
  std::vector<double> local_collective(collective_scalar_count, 0.0);
  std::vector<double> global_collective(collective_scalar_count, 0.0);
  const auto pack_collective = [&]() {
    std::fill(local_collective.begin(), local_collective.end(), 0.0);
    for (size_t work = 0; work < work_count; ++work) {
      for (int component = 0; component < 6; ++component) {
        const std::complex<double> value = output[6 * work + component];
        local_collective[13 * work + 2 * component] = real(value);
        local_collective[13 * work + 2 * component + 1] = imag(value);
      }
      local_collective[13 * work + 12] = local_error_bounds[work];
    }
  };
  pack_collective();
  bool collective_retry_required = near2far_collective_requires_mixed(
      local_collective.data(), global_collective.data(), work_count);

  if (collective_retry_required) {
    if (chunk_count != 0 && !local_used_mixed_precision)
      gpu::detail::resident_near2far_cartesian_fp32(
          near2far_runtime_dimension(monitor_dimension),
          transform.F, requests.data(), requests.size(), targets.data(),
          targets.size(), transform.freq.data(), transform.freq.size(),
          copies.data(), copies.size(), transform.eps, transform.mu,
          output.data(), local_error_bounds.data(), false, true, true,
          &local_used_mixed_precision, azimuthal_mode,
          monitor_dimension == Dcyl ? greencyl_tol : 0.0);
    std::fill(local_error_bounds.begin(), local_error_bounds.end(), 0.0);
    pack_collective();
    sum_to_all_large(local_collective.data(), global_collective.data(),
                     collective_scalar_count);
  }

  for (size_t work = 0; work < work_count; ++work)
    for (int component = 0; component < 6; ++component)
      output[6 * work + component] = std::complex<double>(
          global_collective[13 * work + 2 * component],
          global_collective[13 * work + 2 * component + 1]);
  return true;
}

std::uint64_t cpu_near2far_term_count(const dft_near2far &transform,
                                      size_t target_count) noexcept {
  std::uint64_t source_points = 0;
  for (dft_chunk *chunk = transform.F; chunk;
       chunk = chunk->next_in_dft) {
    if (chunk->N >
        std::numeric_limits<std::uint64_t>::max() - source_points)
      return std::numeric_limits<std::uint64_t>::max();
    source_points += static_cast<std::uint64_t>(chunk->N);
  }
  std::uint64_t copies = 1;
  for (int axis = 0; axis < 2; ++axis) {
    if (transform.periodic_n[axis] < 0) return 0;
    const std::uint64_t count =
        2 * static_cast<std::uint64_t>(transform.periodic_n[axis]) + 1;
    if (copies > std::numeric_limits<std::uint64_t>::max() / count)
      return std::numeric_limits<std::uint64_t>::max();
    copies *= count;
  }
  const std::uint64_t factors[2] = {
      static_cast<std::uint64_t>(transform.freq.size()),
      static_cast<std::uint64_t>(target_count)};
  std::uint64_t terms = source_points;
  if (terms > std::numeric_limits<std::uint64_t>::max() / copies)
    return std::numeric_limits<std::uint64_t>::max();
  terms *= copies;
  for (std::uint64_t factor : factors) {
    if (factor && terms > std::numeric_limits<std::uint64_t>::max() / factor)
      return std::numeric_limits<std::uint64_t>::max();
    terms *= factor;
  }
  return terms;
}

} // namespace

namespace gpu {
namespace detail {

bool test_near2far_collective_requires_mixed(
    const double *local_records, double *global_records, size_t work_count) {
  if ((!local_records || !global_records) && work_count != 0)
    throw std::invalid_argument(
        "near-to-far collective test records must be non-null");
  return near2far_collective_requires_mixed(local_records, global_records,
                                            work_count);
}

} // namespace detail
} // namespace gpu

dft_near2far::dft_near2far(dft_chunk *F_, double fmin, double fmax, int Nf, double eps_, double mu_,
                           const volume &where_, const direction periodic_d_[2],
                           const int periodic_n_[2], const double periodic_k_[2],
                           const double period_[2])
    : F(F_), eps(eps_), mu(mu_), where(where_) {
  freq = meep::linspace(fmin, fmax, Nf);
  for (int i = 0; i < 2; ++i) {
    periodic_d[i] = periodic_d_[i];
    periodic_n[i] = periodic_n_[i];
    periodic_k[i] = periodic_k_[i];
    period[i] = period_[i];
  }
}

dft_near2far::dft_near2far(dft_chunk *F_, const std::vector<double> &freq_, double eps_, double mu_,
                           const volume &where_, const direction periodic_d_[2],
                           const int periodic_n_[2], const double periodic_k_[2],
                           const double period_[2])
    : F(F_), eps(eps_), mu(mu_), where(where_) {
  freq = freq_;
  for (int i = 0; i < 2; ++i) {
    periodic_d[i] = periodic_d_[i];
    periodic_n[i] = periodic_n_[i];
    periodic_k[i] = periodic_k_[i];
    period[i] = period_[i];
  }
}

dft_near2far::dft_near2far(dft_chunk *F_, const double *freq_, size_t Nfreq, double eps_,
                           double mu_, const volume &where_, const direction periodic_d_[2],
                           const int periodic_n_[2], const double periodic_k_[2],
                           const double period_[2])
    : F(F_), eps(eps_), mu(mu_), where(where_) {
  freq.resize(Nfreq);
  for (size_t i = 0; i < Nfreq; ++i)
    freq[i] = freq_[i];
  for (int i = 0; i < 2; ++i) {
    periodic_d[i] = periodic_d_[i];
    periodic_n[i] = periodic_n_[i];
    periodic_k[i] = periodic_k_[i];
    period[i] = period_[i];
  }
}

dft_near2far::dft_near2far(const dft_near2far &f) : F(f.F), eps(f.eps), mu(f.mu), where(f.where) {
  freq = f.freq;
  for (int i = 0; i < 2; ++i) {
    periodic_d[i] = f.periodic_d[i];
    periodic_n[i] = f.periodic_n[i];
    periodic_k[i] = f.periodic_k[i];
    period[i] = f.period[i];
  }
}

void dft_near2far::remove() {
  gpu::detail::destroy_resident_near2far_plan_for_owner(F);
  while (F) {
    dft_chunk *nxt = F->next_in_dft;
    delete F;
    F = nxt;
  }
}

void dft_near2far::operator-=(const dft_near2far &st) {
  if (F && st.F) *F -= *st.F;
}

void dft_near2far::save_hdf5(h5file *file, const char *dprefix) {
  save_dft_hdf5(F, "F", file, dprefix);
}

void dft_near2far::load_hdf5(h5file *file, const char *dprefix) {
  load_dft_hdf5(F, "F", file, dprefix);
}

void dft_near2far::save_hdf5(fields &f, const char *fname, const char *dprefix,
                             const char *prefix) {
  std::unique_ptr<h5file> ff(f.open_h5file(fname, h5file::WRITE, prefix));
  save_hdf5(ff.get(), dprefix);
}

void dft_near2far::load_hdf5(fields &f, const char *fname, const char *dprefix,
                             const char *prefix) {
  std::unique_ptr<h5file> ff(f.open_h5file(fname, h5file::READONLY, prefix));
  load_hdf5(ff.get(), dprefix);
}

void dft_near2far::scale_dfts(complex<double> scale) {
  if (F) F->scale_dft(scale);
}

typedef void (*greenfunc)(std::complex<double> *EH, const vec &x, double freq, double eps,
                          double mu, const vec &x0, component c0, std::complex<double> f0);

/* Given the field f0 correponding to current-source component c0 at
   x0, compute the E/H fields EH[6] (6 components) at x for a frequency
   freq in the homogeneous 3d medium eps and mu.

   Adapted from code by M. T. Homer Reid in his SCUFF-EM package
   (file scuff-em/src/libs/libIncField/PointSource.cc), which is GPL v2+. */
void green3d(std::complex<double> *EH, const vec &x, double freq, double eps, double mu,
             const vec &x0, component c0, std::complex<double> f0) {
  vec rhat = x - x0;
  double r = abs(rhat);
  rhat = rhat / r;

  if (rhat.dim != D3) meep::abort("wrong dimensionality in green3d");

  double n = sqrt(eps * mu);
  double k = 2 * pi * freq * n;
  std::complex<double> ikr = std::complex<double>(0.0, k * r);
  double ikr2 = -(k * r) * (k * r);
  /* note that SCUFF-EM computes the fields from the dipole moment p,
     whereas we need it from the current J = -i*omega*p, so our result
     is divided by -i*omega compared to SCUFF */
  std::complex<double> expfac = f0 * polar(k * n / (4 * pi * r), k * r + pi * 0.5);
  double Z = sqrt(mu / eps);

  vec p = zero_vec(rhat.dim);
  p.set_direction(component_direction(c0), 1);
  double pdotrhat = p & rhat;
  vec rhatcrossp = vec(rhat.y() * p.z() - rhat.z() * p.y(), rhat.z() * p.x() - rhat.x() * p.z(),
                       rhat.x() * p.y() - rhat.y() * p.x());

  /* compute the various scalar quantities in the point source formulae */
  std::complex<double> term1 = 1.0 - 1.0 / ikr + 1.0 / ikr2;
  std::complex<double> term2 = (-1.0 + 3.0 / ikr - 3.0 / ikr2) * pdotrhat;
  std::complex<double> term3 = (1.0 - 1.0 / ikr);

  /* now assemble everything based on source type */
  if (is_electric(c0)) {
    expfac /= eps;

    EH[0] = expfac * (term1 * p.x() + term2 * rhat.x());
    EH[1] = expfac * (term1 * p.y() + term2 * rhat.y());
    EH[2] = expfac * (term1 * p.z() + term2 * rhat.z());

    EH[3] = expfac * term3 * rhatcrossp.x() / Z;
    EH[4] = expfac * term3 * rhatcrossp.y() / Z;
    EH[5] = expfac * term3 * rhatcrossp.z() / Z;
  }
  else if (is_magnetic(c0)) {
    expfac /= mu;

    EH[0] = -expfac * term3 * rhatcrossp.x() * Z;
    EH[1] = -expfac * term3 * rhatcrossp.y() * Z;
    EH[2] = -expfac * term3 * rhatcrossp.z() * Z;

    EH[3] = expfac * (term1 * p.x() + term2 * rhat.x());
    EH[4] = expfac * (term1 * p.y() + term2 * rhat.y());
    EH[5] = expfac * (term1 * p.z() + term2 * rhat.z());
  }
  else
    meep::abort("unrecognized source type");
}

// hankel function J + iY
#if defined(HAVE_JN)
static std::complex<double> hankel(int n, double x) {
  return std::complex<double>(jn(n, x), yn(n, x));
}
#elif defined(HAVE_LIBGSL)
#include <gsl/gsl_sf_bessel.h>
static std::complex<double> hankel(int n, double x) {
  return std::complex<double>(gsl_sf_bessel_Jn(n, x), gsl_sf_bessel_Yn(n, x));
}
#else  /* !HAVE_LIBGSL */
static std::complex<double> hankel(int n, double x) {
  (void)n;
  (void)x; // unused
  meep::abort("GNU GSL library is required for Hankel functions");
}
#endif /* !HAVE_LIBGSL */

/* like green3d, but 2d Green's functions */
void green2d(std::complex<double> *EH, const vec &x, double freq, double eps, double mu,
             const vec &x0, component c0, std::complex<double> f0) {
  vec rhat = x - x0;
  double r = abs(rhat);
  rhat = rhat / r;

  if (rhat.dim != D2) meep::abort("wrong dimensionality in green2d");

  double omega = 2 * pi * freq;
  double k = omega * sqrt(eps * mu);
  std::complex<double> ik = std::complex<double>(0.0, k);
  double kr = k * r;
  double Z = sqrt(mu / eps);
  std::complex<double> H0 = hankel(0, kr) * f0;
  std::complex<double> H1 = hankel(1, kr) * f0;
  std::complex<double> ikH1 = 0.25 * ik * H1;

  if (component_direction(c0) == meep::Z) {
    if (is_electric(c0)) { // Ez source
      EH[0] = EH[1] = 0.0;
      EH[2] = (-0.25 * omega * mu) * H0;

      EH[3] = -rhat.y() * ikH1;
      EH[4] = rhat.x() * ikH1;
      EH[5] = 0.0;
    }
    else /* (is_magnetic(c0)) */ { // Hz source
      EH[0] = rhat.y() * ikH1;
      EH[1] = -rhat.x() * ikH1;
      EH[2] = 0.0;

      EH[3] = EH[4] = 0.0;
      EH[5] = (-0.25 * omega * eps) * H0;
    }
  }
  else { /* in-plane source */
    std::complex<double> H2 = hankel(2, kr) * f0;

    vec p = zero_vec(rhat.dim);
    p.set_direction(component_direction(c0), 1);

    double pdotrhat = p & rhat;
    double rhatcrossp = rhat.x() * p.y() - rhat.y() * p.x();

    if (is_electric(c0)) { // Exy source
      EH[0] = -(rhat.x() * (pdotrhat / r * 0.25 * Z)) * H1 +
              (rhat.y() * (rhatcrossp * omega * mu * 0.125)) * (H0 - H2);
      EH[1] = -(rhat.y() * (pdotrhat / r * 0.25 * Z)) * H1 -
              (rhat.x() * (rhatcrossp * omega * mu * 0.125)) * (H0 - H2);
      EH[2] = 0.0;

      EH[3] = EH[4] = 0.0;
      EH[5] = -rhatcrossp * ikH1;
    }
    else /* (is_magnetic(c0)) */ { // Hxy source
      EH[0] = EH[1] = 0.0;
      EH[2] = rhatcrossp * ikH1;

      EH[3] = -(rhat.x() * (pdotrhat / r * 0.25 / Z)) * H1 +
              (rhat.y() * (rhatcrossp * omega * eps * 0.125)) * (H0 - H2);
      EH[4] = -(rhat.y() * (pdotrhat / r * 0.25 / Z)) * H1 -
              (rhat.x() * (rhatcrossp * omega * eps * 0.125)) * (H0 - H2);
      EH[5] = 0.0;
    }
  }
}

// cylindrical Green's function constructed by integrating green3d as the source
// term rotates around the z axis with exp(im*phi) dependence, integrated to a tolerance tol.
// (note: this is the Green's function divided by 2pi*x0.r(), to compensate for a 2piR factor
//  in the near2far add_dft weight.)
void greencyl(std::complex<double> *EH, const vec &x, double freq, double eps, double mu,
              const vec &x0, component c0, std::complex<double> f0, double m, double tol) {
  if (x0.dim != Dcyl) meep::abort("wrong dimensionality in greencyl");
  vec x_3d(x.dim == Dcyl ? x.r() : x.x(), x.y(), x.z());
  direction d = component_direction(c0);
  component cx = direction_component(c0, X), cy = direction_component(c0, Y);
  for (int j = 0; j < 6; ++j)
    EH[j] = 0;

  /* Perform phi integral.  Since phi integrand is smooth, quadrature with equally spaced points
     should converge exponentially fast with the number N of quadrature points.  We
     repeatedly double N until convergence to tol is achieved, re-using previous points. */
  const int N0 = 16 + int(4 * abs(m));
  double dphi = 2.0 / N0; // factor of 2*pi*r is already included in add_dft weight
  double sumabs = 0;      // integral of L1-norm of integrand
  for (int N = N0; N <= 65536; N *= 2) {
    std::complex<double> EH_sum[6];
    dphi *= 0.5; // delta phi is halved because N doubles
    double dphi2pi = dphi * 2 * pi;
    for (int j = 0; j < 6; ++j)
      EH_sum[j] = EH[j] * 0.5; // re-use previous quadrature points (with halved dphi)
    sumabs *= 0.5;             // re-use previous quadrature (with halved dphi)
    /* N-point quadrature points i = 0..N-1.  After the first iteration (N==N0), we
       only need to sum over odd i, since the even i were summed for the previous N. */
    for (int i = (N > N0); i < N; i += 1 + (N > N0)) {
      double phi = i * dphi2pi, c = cos(phi), s = sin(phi);
      vec x0_phi(x0.r() * c, x0.r() * s, x0.z()); // source point rotated by phi
      std::complex<double> EH_phi[6], f0_exp_imphi = f0 * std::polar(1.0, m * phi) * dphi;
      /* if the source direction is in the r or phi directions, then we must rotate
        the direction of the source current in the xy plane as well */
      if (d == Z) { // source currents in z direction don't rotate
        green3d(EH_phi, x_3d, freq, eps, mu, x0_phi, c0, f0_exp_imphi);
        for (int j = 0; j < 6; ++j) {
          EH_sum[j] += EH_phi[j];
          sumabs += abs(EH_phi[j]);
        }
      }
      else if (d == R) { // r_hat = c x_hat + s y_hat
        green3d(EH_phi, x_3d, freq, eps, mu, x0_phi, cx, f0_exp_imphi * c);
        for (int j = 0; j < 6; ++j) {
          EH_sum[j] += EH_phi[j];
          sumabs += abs(EH_phi[j]);
        }
        green3d(EH_phi, x_3d, freq, eps, mu, x0_phi, cy, f0_exp_imphi * s);
        for (int j = 0; j < 6; ++j) {
          EH_sum[j] += EH_phi[j];
          sumabs += abs(EH_phi[j]);
        }
      }
      else { // (d == P):  phi_hat = c y_hat - s x_hat
        green3d(EH_phi, x_3d, freq, eps, mu, x0_phi, cx, f0_exp_imphi * (-s));
        for (int j = 0; j < 6; ++j) {
          EH_sum[j] += EH_phi[j];
          sumabs += abs(EH_phi[j]);
        }
        green3d(EH_phi, x_3d, freq, eps, mu, x0_phi, cy, f0_exp_imphi * c);
        for (int j = 0; j < 6; ++j) {
          EH_sum[j] += EH_phi[j];
          sumabs += abs(EH_phi[j]);
        }
      }
    }
    // accumulate the new and old sums and check how much the integral has changed in L1 norm
    double sumdiff = 0;
    for (int j = 0; j < 6; ++j) {
      sumdiff += abs(EH[j] - EH_sum[j]);
      EH[j] = EH_sum[j];
    }
    if (sumdiff <= sumabs * tol) break; // doubling N changed sum by less than tol
  }
}

void dft_near2far::farfield_lowlevel(std::complex<double> *EH, const vec &x, double greencyl_tol) {
  if (x.dim != D3 && x.dim != D2 && x.dim != Dcyl)
    meep::abort("only 2d or 3d or cylindrical far-field computation is supported");
  if (!near2far_dimension_pair_is_supported(where.dim, x.dim))
    throw std::invalid_argument(
        "near-to-far monitor and observation dimensions are incompatible");
  greenfunc green = x.dim == D2 ? green2d : green3d;

  const size_t Nfreq = freq.size();
  for (size_t i = 0; i < 6 * Nfreq; ++i)
    EH[i] = 0.0;

  if (x.dim == D3 || x.dim == D2 || x.dim == Dcyl) {
    const std::vector<gpu::detail::near2far_point_fp64> targets = {
        cartesian_near2far_point(x, x.dim)};
    std::vector<std::complex<double> > cuda_output;
    if (cuda_near2far_cartesian(
            *this, x.dim, targets, cuda_output, false, greencyl_tol,
            count_processors() > 1)) {
      std::copy(cuda_output.begin(), cuda_output.end(), EH);
      return;
    }
  }

  gpu::detail::record_cpu_near2far(cpu_near2far_term_count(*this, 1));

  // CUDA DFT updates can leave the monitor arrays device-authoritative.
  // Synchronize before the OpenMP reader loop.
  for (dft_chunk *f = F; f; f = f->next_in_dft)
    gpu::detail::sync_resident_cache_for_owner(f->fc);

  for (dft_chunk *f = F; f; f = f->next_in_dft) {
    assert(Nfreq == f->omega.size());

    component c0 = component(f->vc); /* equivalent source component */

    vec rshift(f->shift * (0.5 * f->fc->gv.inva));
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < Nfreq; ++i) {
      std::complex<double> EH6[6];
      size_t idx_dft = 0;
      LOOP_OVER_IVECS(f->fc->gv, f->is, f->ie, idx) {
        IVEC_LOOP_LOC(f->fc->gv, x0);
        x0 = f->S.transform(x0, f->sn) + rshift;
        vec xs(x0);
        for (int i0 = -periodic_n[0]; i0 <= periodic_n[0]; ++i0) {
          if (periodic_d[0] != NO_DIRECTION)
            xs.set_direction(periodic_d[0], x0.in_direction(periodic_d[0]) + i0 * period[0]);
          double phase0 = i0 * periodic_k[0];
          for (int i1 = -periodic_n[1]; i1 <= periodic_n[1]; ++i1) {
            if (periodic_d[1] != NO_DIRECTION)
              xs.set_direction(periodic_d[1], x0.in_direction(periodic_d[1]) + i1 * period[1]);
            double phase = phase0 + i1 * periodic_k[1];
            std::complex<double> cphase = std::polar(1.0, phase);
            if (where.dim == Dcyl)
              greencyl(EH6, x, freq[i], eps, mu, xs, c0, f->dft[Nfreq * idx_dft + i], f->fc->m,
                       greencyl_tol);
            else
              green(EH6, x, freq[i], eps, mu, xs, c0, f->dft[Nfreq * idx_dft + i]);
            for (int j = 0; j < 6; ++j)
              EH[i * 6 + j] += EH6[j] * cphase;
          }
        }
        idx_dft++;
      }
    }
  }
}

std::complex<double> *dft_near2far::farfield(const vec &x, double greencyl_tol) {
  return farfields(&x, 1, greencyl_tol);
}

std::complex<double> *dft_near2far::farfields(
    const vec *points, size_t point_count, double greencyl_tol) {
  distributed_near2far_failure_guard distributed_failure_guard(
      count_processors() > 1);
  const size_t Nfreq = freq.size();
  ndim dimensionality = D3;
  if (!collective_farfields_metadata_is_consistent(
          points, point_count, Nfreq, *this, greencyl_tol,
          dimensionality)) {
    distributed_failure_guard.dismiss();
    throw std::runtime_error(
        "inconsistent Near2Far batched point count, frequency count, "
        "dimension, or point-array validity across MPI ranks");
  }
  const bool dimension_pair_is_supported = and_to_all(
      point_count == 0 || near2far_dimension_pair_is_supported(
                              where.dim, dimensionality));
  if (!dimension_pair_is_supported) {
    distributed_failure_guard.dismiss();
    throw std::invalid_argument(
        "near-to-far monitor and observation dimensions are incompatible "
        "on an MPI rank");
  }
  const size_t scalar_count = checked_near2far_product(
      checked_near2far_product(point_count, Nfreq,
                              "point-frequency field count"),
      static_cast<size_t>(6), "batched field count");
  std::unique_ptr<std::complex<double>[]> EH_local(
      new std::complex<double>[scalar_count]);
  bool transformed_on_cuda = false;
  if (point_count != 0) {
    const collective_cuda_selection cuda_selection =
        consensus_cuda_near2far_selection(
            dimensionality, *this, greencyl_tol);
    if (cuda_selection == collective_cuda_selection::inconsistent) {
      distributed_failure_guard.dismiss();
      throw std::runtime_error(
          "inconsistent CUDA Near2Far selection across MPI ranks; backend "
          "mode and MEEP_GPU_DISABLE_RESIDENT_NEAR2FAR must agree");
    }

    if (cuda_selection == collective_cuda_selection::cuda) {
      std::vector<gpu::detail::near2far_point_fp64> targets;
      targets.reserve(point_count);
      for (size_t point = 0; point < point_count; ++point)
        targets.push_back(
            cartesian_near2far_point(points[point], dimensionality));
      std::vector<std::complex<double> > cuda_output;
      transformed_on_cuda =
          cuda_near2far_cartesian(
              *this, dimensionality, targets, cuda_output, true,
              greencyl_tol);
      if (transformed_on_cuda)
        std::copy(cuda_output.begin(), cuda_output.end(), EH_local.get());
    }
    if (!transformed_on_cuda)
      for (size_t point = 0; point < point_count; ++point)
        farfield_lowlevel(EH_local.get() + point * Nfreq * 6,
                          points[point], greencyl_tol);
  }
  std::unique_ptr<std::complex<double>[]> EH(
      new std::complex<double>[scalar_count]);
  if (transformed_on_cuda)
    std::copy(EH_local.get(), EH_local.get() + scalar_count, EH.get());
  else
    sum_to_all_large(EH_local.get(), EH.get(), scalar_count);
  distributed_failure_guard.dismiss();
  return EH.release();
}

double *dft_near2far::get_farfields_array(const volume &where, int &rank, size_t *dims, size_t &N,
                                          double resolution, double greencyl_tol) {
  distributed_near2far_failure_guard distributed_failure_guard(
      count_processors() > 1);
  const bool valid_grid_inputs = and_to_all(
      dims != nullptr && std::isfinite(resolution) && resolution > 0.0 &&
      (where.dim == D3 || where.dim == D2 || where.dim == Dcyl) &&
      near2far_dimension_pair_is_supported(
          this->where.dim, where.dim));
  if (!valid_grid_inputs) {
    distributed_failure_guard.dismiss();
    throw std::invalid_argument(
        "near-to-far grid dimensions, resolution, and dimensionality must be "
        "valid and compatible with the monitor on every MPI rank");
  }
  /* compute output grid size etc. */
  double dx[3] = {0, 0, 0};
  direction dirs[3] = {X, Y, Z};

  rank = 0;
  N = dims[0] = dims[1] = dims[2] = 1;

  bool local_extents_valid = true;
  bool local_shape_product_valid = true;
  LOOP_OVER_DIRECTIONS(where.dim, d) {
    const double scaled_extent = where.in_direction(d) * resolution;
    local_extents_valid =
        local_extents_valid && std::isfinite(scaled_extent) &&
        scaled_extent >= 0.0 &&
        scaled_extent <=
            static_cast<double>(std::numeric_limits<int>::max());
  }
  if (!and_to_all(local_extents_valid)) {
    distributed_failure_guard.dismiss();
    throw std::overflow_error(
        "near-to-far output grid extent is invalid or too large on an MPI "
        "rank");
  }
  LOOP_OVER_DIRECTIONS(where.dim, d) {
    const double scaled_extent = where.in_direction(d) * resolution;
    const double floored_extent = floor(scaled_extent);
    if (floored_extent <= 1.0)
      dims[rank] = 1;
    else {
      dims[rank] = static_cast<size_t>(floored_extent);
      dx[rank] = where.in_direction(d) / (dims[rank] - 1);
    }
    if (N && dims[rank] > std::numeric_limits<size_t>::max() / N) {
      local_shape_product_valid = false;
      N = 0;
    }
    else
      N *= dims[rank];
    dirs[rank++] = d;
  }
  if (!and_to_all(local_shape_product_valid)) {
    distributed_failure_guard.dismiss();
    throw std::overflow_error(
        "near-to-far output grid point count overflows on an MPI rank");
  }
  if (where.dim == Dcyl) dirs[2] = P; // otherwise Z is listed twice

  const size_t Nfreq = freq.size();
  if (!collective_grid_metadata_is_consistent(
          where, resolution, dims, N, Nfreq, rank, *this,
          greencyl_tol)) {
    distributed_failure_guard.dismiss();
    throw std::runtime_error(
        "inconsistent Near2Far grid shape, coordinates, resolution, "
        "frequency count, or dimension across MPI ranks");
  }
  if (N == 0 || Nfreq == 0) {
    distributed_failure_guard.dismiss();
    return NULL; /* nothing to output */
  }
  const size_t work_count = checked_near2far_product(
      N, Nfreq, "output point-frequency count");
  const size_t scalar_count = checked_near2far_product(
      work_count, static_cast<size_t>(12), "output scalar count");

  /* 6 x 2 x N x Nfreq array of fields in row-major order */
  std::unique_ptr<double[]> EH(new double[scalar_count]);
  std::unique_ptr<double[]> EH_local(new double[scalar_count]);

  bool transformed_on_cuda = false;
  const collective_cuda_selection cuda_selection =
      consensus_cuda_near2far_selection(where.dim, *this, greencyl_tol);
  if (cuda_selection == collective_cuda_selection::inconsistent) {
    distributed_failure_guard.dismiss();
    throw std::runtime_error(
        "inconsistent CUDA Near2Far selection across MPI ranks; backend mode "
        "and MEEP_GPU_DISABLE_RESIDENT_NEAR2FAR must agree");
  }
  if (cuda_selection == collective_cuda_selection::cuda) {
    std::vector<gpu::detail::near2far_point_fp64> targets;
    targets.reserve(N);
    vec x(where.dim);
    for (size_t i0 = 0; i0 < dims[0]; ++i0) {
      x.set_direction(dirs[0],
                      where.in_direction_min(dirs[0]) + i0 * dx[0]);
      for (size_t i1 = 0; i1 < dims[1]; ++i1) {
        x.set_direction(dirs[1],
                        where.in_direction_min(dirs[1]) + i1 * dx[1]);
        for (size_t i2 = 0; i2 < dims[2]; ++i2) {
          x.set_direction(dirs[2],
                          where.in_direction_min(dirs[2]) + i2 * dx[2]);
          targets.push_back(cartesian_near2far_point(x, where.dim));
        }
      }
    }
    if (targets.size() != N)
      throw std::logic_error(
          "near-to-far CUDA target grid count changed during construction");
    std::vector<std::complex<double> > cuda_output;
    transformed_on_cuda =
        cuda_near2far_cartesian(
            *this, where.dim, targets, cuda_output, true, greencyl_tol);
    if (transformed_on_cuda) {
      for (size_t idx = 0; idx < N; ++idx)
        for (size_t frequency = 0; frequency < Nfreq; ++frequency)
          for (int component = 0; component < 6; ++component) {
            const std::complex<double> value =
                cuda_output[6 * (idx * Nfreq + frequency) + component];
            EH_local[((component * 2 + 0) * N + idx) * Nfreq +
                     frequency] = real(value);
            EH_local[((component * 2 + 1) * N + idx) * Nfreq +
                     frequency] = imag(value);
          }
    }
  }

  if (!transformed_on_cuda) {
    std::vector<std::complex<double> > EH1(
        checked_near2far_product(Nfreq, static_cast<size_t>(6),
                                "single-point field count"));
    double start = wall_time();
    size_t last_point = 0;
    vec x(where.dim);
    for (size_t i0 = 0; i0 < dims[0]; ++i0) {
      x.set_direction(dirs[0],
                      where.in_direction_min(dirs[0]) + i0 * dx[0]);
      for (size_t i1 = 0; i1 < dims[1]; ++i1) {
        x.set_direction(dirs[1],
                        where.in_direction_min(dirs[1]) + i1 * dx[1]);
        for (size_t i2 = 0; i2 < dims[2]; ++i2) {
          x.set_direction(dirs[2],
                          where.in_direction_min(dirs[2]) + i2 * dx[2]);
          double t;
          if (verbosity > 0 &&
              (t = wall_time()) > start + MEEP_MIN_OUTPUT_TIME) {
            const size_t this_point =
                (dims[1] * i0 + i1) * dims[2] + i2 + 1;
            master_printf(
                "get_farfields_array working on point %zu of %zu (%d%% done), %g s/point\n",
                this_point, N, (int)((double)this_point / N * 100),
                (t - start) /
                    std::max(1.0, static_cast<double>(this_point - last_point)));
            start = t;
            last_point = this_point;
          }
          farfield_lowlevel(EH1.data(), x, greencyl_tol);
          if (verbosity > 1)
            all_wait(); // Allow consistent progress updates from master
          const size_t idx = (i0 * dims[1] + i1) * dims[2] + i2;
          for (size_t frequency = 0; frequency < Nfreq; ++frequency)
            for (int component = 0; component < 6; ++component) {
              const std::complex<double> value =
                  EH1[frequency * 6 + component];
              EH_local[((component * 2 + 0) * N + idx) * Nfreq +
                       frequency] = real(value);
              EH_local[((component * 2 + 1) * N + idx) * Nfreq +
                       frequency] = imag(value);
            }
        }
      }
    }
  }
  if (transformed_on_cuda)
    std::copy(EH_local.get(), EH_local.get() + scalar_count, EH.get());
  else
    sum_to_all_large(EH_local.get(), EH.get(), scalar_count);

  /* collapse singleton dimensions */
  int ireduced = 0;
  for (int i = 0; i < rank; ++i) {
    if (dims[i] > 1) dims[ireduced++] = dims[i];
  }
  rank = ireduced;

  distributed_failure_guard.dismiss();
  return EH.release();
}

void dft_near2far::save_farfields(const char *fname, const char *prefix, const volume &where,
                                  double resolution, double greencyl_tol) {
  size_t dims[4] = {1, 1, 1, 1};
  int rank = 0;
  size_t N = 1;

  double *EH = get_farfields_array(where, rank, dims, N, resolution, greencyl_tol);
  if (!EH) return; /* nothing to output */

  const size_t Nfreq = freq.size();
  /* frequencies are the last dimension */
  if (Nfreq > 1) dims[rank++] = Nfreq;

  /* output to a file with one dataset per component & real/imag part */
  if (am_master()) {
    const int buflen = 1024;
    static char filename[buflen];
    snprintf(filename, buflen, "%s%s%s.h5", prefix ? prefix : "", prefix && prefix[0] ? "-" : "",
             fname);
    h5file ff(filename, h5file::WRITE, false);
    component c[6] = {Ex, Ey, Ez, Hx, Hy, Hz};
    char dataname[128];
    for (int k = 0; k < 6; ++k)
      for (int reim = 0; reim < 2; ++reim) {
        snprintf(dataname, 128, "%s.%c", component_name(c[k]), "ri"[reim]);
        ff.write(dataname, rank, dims, EH + (k * 2 + reim) * N * Nfreq);
      }
  }

  delete[] EH;
}

double *dft_near2far::flux(direction df, const volume &where, double resolution) {
  if (coordinate_mismatch(where.dim, df) || where.dim == Dcyl)
    meep::abort("cannot get flux for near2far: co-ordinate mismatch");

  size_t dims[4] = {1, 1, 1, 1};
  int rank = 0;
  size_t N = 1;

  double *EH = get_farfields_array(where, rank, dims, N, resolution);

  const size_t Nfreq = freq.size();
  double *F = new double[Nfreq];
  std::complex<double> ff_EH[6];
  std::complex<double> cE[2], cH[2];

  for (size_t i = 0; i < Nfreq; ++i)
    F[i] = 0;

  for (size_t idx = 0; idx < N; ++idx) {
    for (size_t i = 0; i < Nfreq; ++i) {
      for (int k = 0; k < 6; ++k)
        ff_EH[k] = std::complex<double>(*(EH + ((k * 2 + 0) * N + idx) * Nfreq + i),
                                        *(EH + ((k * 2 + 1) * N + idx) * Nfreq + i));
      switch (df) {
        case X: cE[0] = ff_EH[1], cE[1] = ff_EH[2], cH[0] = ff_EH[5], cH[1] = ff_EH[4]; break;
        case Y: cE[0] = ff_EH[2], cE[1] = ff_EH[0], cH[0] = ff_EH[3], cH[1] = ff_EH[5]; break;
        case Z: cE[0] = ff_EH[0], cE[1] = ff_EH[1], cH[0] = ff_EH[4], cH[1] = ff_EH[3]; break;
        case R:
        case P:
        case NO_DIRECTION: meep::abort("invalid flux direction");
      }
      for (int j = 0; j < 2; ++j)
        F[i] += real(cE[j] * conj(cH[j])) * (1 - 2 * j);
    }
  }

  double dV = 1;
  LOOP_OVER_DIRECTIONS(where.dim, d) {
    int dim = int(floor(where.in_direction(d) * resolution));
    if (dim > 1) dV *= where.in_direction(d) / (dim - 1);
  }

  for (size_t i = 0; i < Nfreq; ++i)
    F[i] *= dV;

  delete[] EH;

  return F;
}

static double approxeq(double a, double b) { return fabs(a - b) < 0.5e-11 * (fabs(a) + fabs(b)); }

dft_near2far fields::add_dft_near2far(const volume_list *where, const double *freq, size_t Nfreq,
                                      int decimation_factor, int Nperiods) {

  dft_chunk *F = 0; /* E and H chunks*/
  double eps = 0, mu = 0;
  volume everywhere = where->v;

  direction periodic_d[2] = {NO_DIRECTION, NO_DIRECTION};
  int periodic_n[2] = {0, 0};
  double periodic_k[2] = {0, 0}, period[2] = {0, 0};

  for (const volume_list *w = where; w; w = w->next) {
    everywhere = everywhere | where->v;
    direction nd = component_direction(w->c);
    if (nd == NO_DIRECTION) nd = normal_direction(w->v);
    if (nd == NO_DIRECTION) meep::abort("unknown dft_near2far normal");
    direction fd[2];

    double weps = real(get_eps(w->v.center()));
    double wmu = real(get_mu(w->v.center()));
    if (w != where && !(approxeq(eps, weps) && approxeq(mu, wmu)))
      meep::abort("dft_near2far requires surfaces in a homogeneous medium");
    eps = weps;
    mu = wmu;

    /* two transverse directions to normal (in cyclic order to get
       correct sign s below) */
    switch (nd) {
      case X:
        fd[0] = Y;
        fd[1] = Z;
        break;
      case Y:
        fd[0] = Z;
        fd[1] = X;
        break;
      case R:
        fd[0] = P;
        fd[1] = Z;
        break;
      case P:
        fd[0] = Z;
        fd[1] = R;
        break;
      case Z:
        if (gv.dim == Dcyl)
          fd[0] = R, fd[1] = P;
        else
          fd[0] = X, fd[1] = Y;
        break;
      default: meep::abort("invalid normal direction in dft_near2far!");
    }

    if (Nperiods > 1) {
      for (int i = 0; i < 2; ++i) {
        double user_width = user_volume.num_direction(fd[i]) / a;
        if (has_direction(v.dim, fd[i]) && boundaries[High][fd[i]] == Periodic &&
            boundaries[Low][fd[i]] == Periodic &&
            float(w->v.in_direction(fd[i])) >= float(user_width)) {
          periodic_d[i] = fd[i];
          periodic_n[i] = Nperiods;
          period[i] = user_width;
          periodic_k[i] = 2 * pi * real(k[fd[i]]) * period[i];
        }
      }
    }

    for (int i = 0; i < 2; ++i) {   /* E or H */
      for (int j = 0; j < 2; ++j) { /* first or second component */
        component c = direction_component(i == 0 ? Ex : Hx, fd[j]);

        /* find equivalent source component c0 and sign s */
        component c0 = direction_component(i == 0 ? Hx : Ex, fd[1 - j]);
        double s = j == 0 ? 1 : -1; /* sign of n x c */
        if (is_electric(c)) s = -s;

        F = add_dft(c, w->v, freq, Nfreq, true, s * w->weight, F, false, 1.0, false, c0,
                    decimation_factor);
      }
    }
  }

  return dft_near2far(F, freq, Nfreq, eps, mu, everywhere, periodic_d, periodic_n, periodic_k,
                      period);
}

// Modified from farfield_lowlevel
std::vector<struct sourcedata> dft_near2far::near_sourcedata(const vec &x_0, double *farpt_list,
                                                             size_t nfar_pts,
                                                             const std::complex<double> *dJ,
                                                             double greencyl_tol) {
  if (x_0.dim != D3 && x_0.dim != D2 && x_0.dim != Dcyl)
    meep::abort("only 2d or 3d or cylindrical far-field computation is supported");
  const ndim observation_dimension = x_0.dim;
  const ndim monitor_dimension = where.dim;
  if (!near2far_dimension_pair_is_supported(
          monitor_dimension, observation_dimension))
    throw std::invalid_argument(
        "near-to-far adjoint monitor and observation dimensions are "
        "incompatible");
  if (nfar_pts != 0 && !farpt_list)
    throw std::invalid_argument(
        "near-to-far adjoint far-point coordinates must be non-null");
  greenfunc green = monitor_dimension == D2 ? green2d : green3d;

  const size_t Nfreq = freq.size();
  if (nfar_pts != 0 && Nfreq != 0 && !dJ)
    throw std::invalid_argument(
        "near-to-far adjoint dJ must be non-null");
  std::vector<struct sourcedata> temp;

  if (cuda_near2far_selected(observation_dimension, *this, greencyl_tol) &&
      nfar_pts != 0 && Nfreq != 0) {
    const std::vector<gpu::detail::near2far_periodic_copy_fp64> copies =
        build_near2far_periodic_copies(*this, monitor_dimension);
    std::vector<gpu::detail::near2far_point_fp64> targets;
    targets.reserve(nfar_pts);
    for (size_t point = 0; point < nfar_pts; ++point) {
      const gpu::detail::near2far_point_fp64 target =
          {farpt_list[3 * point], farpt_list[3 * point + 1],
           farpt_list[3 * point + 2]};
      targets.push_back(target);
    }

    size_t chunk_count = 0;
    for (dft_chunk *chunk = F; chunk; chunk = chunk->next_in_dft)
      ++chunk_count;
    std::vector<std::vector<gpu::detail::near2far_point_fp64> >
        source_coordinates;
    std::vector<std::vector<std::complex<double> > > source_amplitudes;
    std::vector<gpu::detail::near2far_adjoint_request_fp64> requests;
    std::vector<size_t> request_for_chunk;
    source_coordinates.reserve(chunk_count);
    source_amplitudes.reserve(chunk_count);
    requests.reserve(chunk_count);
    request_for_chunk.reserve(chunk_count);
    temp.reserve(chunk_count);
    size_t total_source_points = 0;
    double azimuthal_mode = 0.0;
    bool have_azimuthal_mode = false;

    for (dft_chunk *f = F; f; f = f->next_in_dft) {
      if (Nfreq != f->omega.size())
        throw std::invalid_argument(
            "near-to-far adjoint DFT chunk frequency count mismatch");
      std::vector<ptrdiff_t> idx_arr;
      std::vector<std::complex<double> > amp_arr;
      sourcedata temp_struct =
          {component(f->c), idx_arr, f->fc->chunk_idx, amp_arr};
      source_coordinates.emplace_back();
      source_amplitudes.emplace_back();
      std::vector<gpu::detail::near2far_point_fp64> &coordinates =
          source_coordinates.back();
      std::vector<std::complex<double> > &amplitudes =
          source_amplitudes.back();
      coordinates.reserve(f->N);
      amplitudes.reserve(f->N);

      const component c0 = component(f->vc);
      const direction source_direction = component_direction(c0);
      const int direction_index =
          near2far_direction_index(source_direction, monitor_dimension);
      if (direction_index < 0 ||
          (!is_electric(c0) && !is_magnetic(c0)))
        throw std::invalid_argument(
            "CUDA near-to-far adjoint equivalent source is not a "
            "supported E/H component");
      if (monitor_dimension == Dcyl) {
        const double chunk_mode = f->fc->m;
        if (!std::isfinite(chunk_mode) ||
            (have_azimuthal_mode && chunk_mode != azimuthal_mode))
          throw std::invalid_argument(
              "CUDA cylindrical near-to-far adjoint chunks have "
              "inconsistent azimuthal modes");
        azimuthal_mode = chunk_mode;
        have_azimuthal_mode = true;
      }

      vec rshift(f->shift * (0.5 * f->fc->gv.inva));
      LOOP_OVER_IVECS(f->fc->gv, f->is, f->ie, idx) {
        IVEC_LOOP_ILOC(f->fc->gv, ix0);
        IVEC_LOOP_LOC(f->fc->gv, source);
        source = f->S.transform(source, f->sn) + rshift;
        const double quadrature_weight =
            IVEC_LOOP_WEIGHT(f->s0, f->s1, f->e0, f->e1,
                             f->dV0 + f->dV1 * loop_i2);
        std::complex<double> amplitude =
            quadrature_weight * f->stored_weight;
        if (is_electric(temp_struct.near_fd_comp)) amplitude *= -1.0;
        amplitude /= f->S.multiplicity(ix0);
        if (monitor_dimension == Dcyl) {
          double final_radius = source.r();
          for (int axis = 0; axis < 2; ++axis)
            if (periodic_d[axis] == R)
              final_radius += periodic_n[axis] * period[axis];
          // Match the CPU loop exactly: its post-periodic Jacobian division
          // observes the last periodic image, while the usual cylindrical
          // Z-periodic case leaves the original source radius unchanged.
          if (final_radius != 0.0) amplitude /= final_radius;
        }
        temp_struct.idx_arr.push_back(idx);
        coordinates.push_back(
            cartesian_near2far_point(source, monitor_dimension));
        amplitudes.push_back(amplitude);
      }
      if (coordinates.size() != f->N || amplitudes.size() != f->N)
        throw std::logic_error(
            "near-to-far adjoint source-coordinate count differs from "
            "DFT storage");
      temp.push_back(std::move(temp_struct));
      if (f->N == 0) {
        request_for_chunk.push_back(std::numeric_limits<size_t>::max());
        continue;
      }
      request_for_chunk.push_back(requests.size());
      requests.push_back(
          {coordinates.data(), amplitudes.data(), coordinates.size(),
           direction_index, is_electric(c0)});
      if (total_source_points >
          std::numeric_limits<size_t>::max() - coordinates.size())
        throw std::overflow_error(
            "near-to-far adjoint source point count overflow");
      total_source_points += coordinates.size();
    }

    if (!requests.empty()) {
      std::vector<std::complex<double> > cuda_output(
          checked_near2far_product(total_source_points, Nfreq,
                                   "adjoint CUDA output count"));
      bool used_mixed_precision = false;
      gpu::detail::near2far_adjoint_cartesian_fp32(
          near2far_runtime_dimension(monitor_dimension), F, requests.data(),
          requests.size(), targets.data(), targets.size(), freq.data(),
          Nfreq, copies.data(), copies.size(), eps, mu, dJ,
          cuda_output.data(), &used_mixed_precision, azimuthal_mode,
          monitor_dimension == Dcyl ? greencyl_tol : 0.0);
      size_t output_offset = 0;
      for (size_t chunk = 0; chunk < temp.size(); ++chunk) {
        if (request_for_chunk[chunk] ==
            std::numeric_limits<size_t>::max())
          continue;
        const size_t point_count =
            requests[request_for_chunk[chunk]].point_count;
        const size_t value_count = checked_near2far_product(
            point_count, Nfreq, "adjoint chunk output count");
        temp[chunk].amp_arr.insert(
            temp[chunk].amp_arr.end(), cuda_output.begin() + output_offset,
            cuda_output.begin() + output_offset + value_count);
        output_offset += value_count;
      }
      if (output_offset != cuda_output.size())
        throw std::logic_error(
            "near-to-far adjoint CUDA output partition mismatch");
    }
    return temp;
  }

  gpu::detail::record_cpu_near2far_adjoint(
      cpu_near2far_term_count(*this, nfar_pts));

  for (dft_chunk *f = F; f; f = f->next_in_dft) {
    assert(Nfreq == f->omega.size());
    std::vector<ptrdiff_t> idx_arr;
    std::vector<std::complex<double> > amp_arr;
    component c0 = component(f->vc); /* equivalent source component */

    vec rshift(f->shift * (0.5 * f->fc->gv.inva));
    std::complex<double> EH6[6];
    size_t idx_dft = 0;
    sourcedata temp_struct = {component(f->c), idx_arr, f->fc->chunk_idx, amp_arr};

    LOOP_OVER_IVECS(f->fc->gv, f->is, f->ie, idx) {
      IVEC_LOOP_ILOC(f->fc->gv, ix0);
      IVEC_LOOP_LOC(f->fc->gv, x0);
      x0 = f->S.transform(x0, f->sn) + rshift;
      vec xs(x0);
      double w;
      w = IVEC_LOOP_WEIGHT(f->s0, f->s1, f->e0, f->e1, f->dV0 + f->dV1 * loop_i2);

      temp_struct.idx_arr.push_back(idx);
      for (size_t i = 0; i < Nfreq; ++i) {
        std::complex<double> EH0 = std::complex<double>(0, 0);
        for (int i0 = -periodic_n[0]; i0 <= periodic_n[0]; ++i0) {
          if (periodic_d[0] != NO_DIRECTION)
            xs.set_direction(periodic_d[0], x0.in_direction(periodic_d[0]) + i0 * period[0]);
          double phase0 = i0 * periodic_k[0];
          for (int i1 = -periodic_n[1]; i1 <= periodic_n[1]; ++i1) {
            if (periodic_d[1] != NO_DIRECTION)
              xs.set_direction(periodic_d[1], x0.in_direction(periodic_d[1]) + i1 * period[1]);
            double phase = phase0 + i1 * periodic_k[1];
            std::complex<double> cphase = std::polar(1.0, phase);
            for (size_t ipt = 0; ipt < nfar_pts; ++ipt) {
              vec x = vec(farpt_list[3 * ipt], farpt_list[3 * ipt + 1], farpt_list[3 * ipt + 2]);
              if (monitor_dimension == Dcyl)
                greencyl(EH6, x, freq[i], eps, mu, xs, c0, w, f->fc->m, greencyl_tol);
              else
                green(EH6, x, freq[i], eps, mu, xs, c0, w);
              for (int j = 0; j < 6; ++j)
                EH0 += EH6[j] * cphase * (f->stored_weight) * dJ[6 * Nfreq * ipt + 6 * i + j];
            }
          }
        }
        idx_dft++;
        if (is_electric(temp_struct.near_fd_comp)) EH0 *= -1;
        EH0 /= f->S.multiplicity(ix0);
        if (monitor_dimension == Dcyl) {
          if (xs.r() != 0) EH0 /= xs.r();
          // Somehow, a factor of (2pi)r for r of the near2far region was double counted.
          // 2pi is canceled out by the integral over design region, where an extra factor of 2pi*r'
          // (r' of the design region) is needed. See meepgeom.cpp
        }
        temp_struct.amp_arr.push_back(EH0);
      }
    }
    temp.push_back(temp_struct);
  }
  return temp;
}

} // namespace meep
