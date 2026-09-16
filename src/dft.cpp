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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <assert.h>
#include <atomic>
#include <exception>
#include <memory>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include "meep.hpp"
#include "meep_internals.hpp"
#include "gpu_backend_internal.hpp"

using namespace std;

namespace meep {

namespace {

std::atomic<bool> fail_next_dft_norm_on_master(false);
std::atomic<bool> fail_next_cuda_dft_output_after_dataset(false);

bool exception_is_in_flight() noexcept {
#if __cplusplus >= 201703L
  return std::uncaught_exceptions() > 0;
#else
  return std::uncaught_exception();
#endif
}

bool has_hdf5_suffix(const char *filename) {
  if (!filename) return false;
  const size_t length = strlen(filename);
  return length >= 3 &&
         memcmp(filename + length - 3, ".h5", 3) == 0;
}

void sync_dft_chunk_list(dft_chunk *chunks) {
  for (dft_chunk *chunk = chunks; chunk; chunk = chunk->next_in_dft)
    gpu::detail::sync_resident_cache_for_owner(chunk->fc);
}

class distributed_dft_norm_failure_guard {
public:
  explicit distributed_dft_norm_failure_guard(bool armed) : armed_(armed) {}

  ~distributed_dft_norm_failure_guard() {
    if (armed_ && exception_is_in_flight())
      meep::abort(
          "rank-local failure during a distributed DFT norm reduction; "
          "aborting the communicator to prevent an MPI collective deadlock");
  }

private:
  bool armed_;
};

class distributed_dft_reduction_failure_guard {
public:
  explicit distributed_dft_reduction_failure_guard(bool armed)
      : armed_(armed) {}

  ~distributed_dft_reduction_failure_guard() {
    if (armed_ && exception_is_in_flight())
      meep::abort(
          "rank-local failure during a distributed DFT spectral reduction; "
          "aborting the communicator to prevent an MPI collective deadlock");
  }

  void dismiss() noexcept { armed_ = false; }

private:
  bool armed_;
};

class distributed_dft_materialization_failure_guard {
public:
  explicit distributed_dft_materialization_failure_guard(bool armed)
      : armed_(armed) {}

  ~distributed_dft_materialization_failure_guard() {
    if (armed_ && exception_is_in_flight())
      meep::abort(
          "rank-local failure during a distributed DFT array "
          "materialization; aborting the communicator to prevent an MPI "
          "collective deadlock");
  }

  void dismiss() noexcept { armed_ = false; }

private:
  bool armed_;
};

class distributed_dft_output_failure_guard {
public:
  distributed_dft_output_failure_guard()
      : armed_(count_processors() > 1) {}

  ~distributed_dft_output_failure_guard() {
    if (armed_ && exception_is_in_flight())
      meep::abort(
          "rank-local failure during distributed CUDA DFT output; "
          "aborting the communicator to prevent an HDF5 collective "
          "deadlock");
  }

  void dismiss() noexcept { armed_ = false; }

private:
  bool armed_;
};

class distributed_dft_checkpoint_failure_guard {
public:
  explicit distributed_dft_checkpoint_failure_guard(const char *operation)
      : operation_(operation), armed_(count_processors() > 1) {}

  ~distributed_dft_checkpoint_failure_guard() {
    if (armed_ && exception_is_in_flight())
      meep::abort(
          "rank-local failure during distributed DFT checkpoint %s; "
          "aborting the communicator to prevent an HDF5 collective "
          "deadlock",
          operation_);
  }

  void dismiss() noexcept { armed_ = false; }

private:
  const char *operation_;
  bool armed_;
};

class distributed_eigenmode_overlap_failure_guard {
public:
  distributed_eigenmode_overlap_failure_guard()
      : armed_(count_processors() > 1) {}

  ~distributed_eigenmode_overlap_failure_guard() {
    if (armed_ && exception_is_in_flight())
      meep::abort(
          "rank-local failure during a distributed CUDA eigenmode "
          "overlap; aborting the communicator to prevent an MPI "
          "collective deadlock");
  }

  void dismiss() noexcept { armed_ = false; }

private:
  bool armed_;
};

constexpr int eigenmode_overlap_component_count = 4;
constexpr int eigenmode_overlap_coordinate_count = 5;
constexpr int eigenmode_overlap_consensus_flag_count = 3;
constexpr int eigenmode_overlap_extent_values_per_component =
    2 * eigenmode_overlap_coordinate_count;
constexpr int eigenmode_overlap_consensus_value_count =
    eigenmode_overlap_consensus_flag_count +
    eigenmode_overlap_component_count *
        eigenmode_overlap_extent_values_per_component;

struct eigenmode_overlap_consensus {
  bool every_rank_cuda = false;
  bool every_rank_cpu = false;
  bool channel_has_extent[eigenmode_overlap_component_count] = {};
};

eigenmode_overlap_consensus resolve_eigenmode_overlap_consensus(
    fields *field, const grid_volume &gv, const volume &whole_volume,
    dft_chunk *const chunklists[2], const component components[4]) {
  int local[eigenmode_overlap_consensus_value_count] = {};
  int global[eigenmode_overlap_consensus_value_count] = {};
  bool local_cuda_active = false;
  bool local_backend_query_ok = true;
  try {
    local_cuda_active = gpu::detail::cuda_active();
  }
  catch (...) {
    if (count_processors() == 1) throw;
    local_backend_query_ok = false;
  }
  local[0] = local_backend_query_ok ? 0 : 1;
  local[1] = local_backend_query_ok && local_cuda_active ? 1 : 0;
  local[2] = local_backend_query_ok && !local_cuda_active ? 1 : 0;

  const ivec empty_min =
      gv.round_vec(whole_volume.get_max_corner()) + one_ivec(gv.dim);
  const ivec empty_max =
      gv.round_vec(whole_volume.get_min_corner()) - one_ivec(gv.dim);
  for (int channel = 0; channel < eigenmode_overlap_component_count;
       ++channel) {
    const int base = eigenmode_overlap_consensus_flag_count +
                     channel *
                         eigenmode_overlap_extent_values_per_component;
    for (int coordinate = 0;
         coordinate < eigenmode_overlap_coordinate_count; ++coordinate) {
      const direction d = direction(coordinate);
      local[base + coordinate] = empty_max.in_direction(d);
      local[base + eigenmode_overlap_coordinate_count + coordinate] =
          -empty_min.in_direction(d);
    }
    for (int list_index = 0; list_index < 2; ++list_index)
      for (dft_chunk *chunk = chunklists[list_index]; chunk;
           chunk = chunk->next_in_dft) {
        if (chunk->c != components[channel]) continue;
        const ivec transformed_start =
            chunk->S.transform(chunk->is, chunk->sn) + chunk->shift;
        const ivec transformed_end =
            chunk->S.transform(chunk->ie, chunk->sn) + chunk->shift;
        for (int coordinate = 0;
             coordinate < eigenmode_overlap_coordinate_count;
             ++coordinate) {
          const direction d = direction(coordinate);
          local[base + coordinate] = std::max(
              local[base + coordinate],
              std::max(transformed_start.in_direction(d),
                       transformed_end.in_direction(d)));
          local[base + eigenmode_overlap_coordinate_count + coordinate] =
              std::max(
                  local[base + eigenmode_overlap_coordinate_count +
                        coordinate],
                  -std::min(transformed_start.in_direction(d),
                            transformed_end.in_direction(d)));
        }
      }
  }

  field->am_now_working_on(MpiAllTime);
  max_to_all(local, global, eigenmode_overlap_consensus_value_count);
  field->finished_working();
  if (global[0])
    meep::abort(
        "rank-local backend configuration failure before distributed "
        "eigenmode overlap");
  if (global[1] && global[2])
    meep::abort(
        "inconsistent CPU/CUDA backend selection across MPI ranks before "
        "distributed eigenmode overlap");
  if (!global[1] && !global[2])
    meep::abort(
        "distributed eigenmode overlap resolved no active backend");

  eigenmode_overlap_consensus result;
  result.every_rank_cuda = global[1] != 0;
  result.every_rank_cpu = global[2] != 0;
  for (int channel = 0; channel < eigenmode_overlap_component_count;
       ++channel) {
    const int base = eigenmode_overlap_consensus_flag_count +
                     channel *
                         eigenmode_overlap_extent_values_per_component;
    LOOP_OVER_DIRECTIONS(gv.dim, d) {
      const int maximum = global[base + static_cast<int>(d)];
      const int minimum =
          -global[base + eigenmode_overlap_coordinate_count +
                  static_cast<int>(d)];
      const int extent = std::max(0, (maximum - minimum) / 2 + 1);
      if (extent > 1) result.channel_has_extent[channel] = true;
    }
  }
  return result;
}

void compute_cpu_eigenmode_overlap(
    fields *field, void *mode1_data, void *mode2_data, dft_flux flux,
    int num_freq, const component cE[2], const component cH[2],
    complex<double> overlaps[2]) {
  gpu::detail::record_cpu_dft_overlap_call();
  dft_chunk *chunklists[2] = {flux.E, flux.H};
  const complex<double> ExHy = field->process_dft_component(
      chunklists, 2, num_freq, cE[0], 0, 0, 0, 0, 0, mode1_data,
      mode2_data, cH[0]);
  const complex<double> EyHx = field->process_dft_component(
      chunklists, 2, num_freq, cE[1], 0, 0, 0, 0, 0, mode1_data,
      mode2_data, cH[1]);
  const complex<double> HyEx = field->process_dft_component(
      chunklists, 2, num_freq, cH[0], 0, 0, 0, 0, 0, mode1_data,
      mode2_data, cE[0]);
  const complex<double> HxEy = field->process_dft_component(
      chunklists, 2, num_freq, cH[1], 0, 0, 0, 0, 0, mode1_data,
      mode2_data, cE[1]);
  overlaps[0] = ExHy - EyHx;
  overlaps[1] = HyEx - HxEy;
}

void sum_dft_reduction_to_all(const double *input, double *output,
                              size_t count) {
  size_t offset = 0;
  while (offset < count) {
    const int batch = static_cast<int>(std::min<size_t>(
        count - offset,
        static_cast<size_t>(std::numeric_limits<int>::max())));
    gpu::detail::record_dft_reduction_mpi_allreduce(
        static_cast<size_t>(batch) * sizeof(double));
    sum_to_all(input + offset, output + offset, batch);
    offset += static_cast<size_t>(batch);
  }
}

void sum_dft_reduction_to_all(
    const std::complex<double> *input,
    std::complex<double> *output, size_t count) {
  size_t offset = 0;
  while (offset < count) {
    const size_t maximum_complex_batch =
        static_cast<size_t>(std::numeric_limits<int>::max()) / 2;
    const int batch = static_cast<int>(std::min<size_t>(
        count - offset, maximum_complex_batch));
    gpu::detail::record_dft_reduction_mpi_allreduce(
        static_cast<size_t>(batch) * sizeof(std::complex<double>));
    sum_to_all(input + offset, output + offset, batch);
    offset += static_cast<size_t>(batch);
  }
}

} // namespace

namespace gpu {
namespace detail {

void fail_next_dft_norm_on_master_for_testing() noexcept {
  fail_next_dft_norm_on_master.store(true, std::memory_order_release);
}

bool consume_dft_norm_master_failure_for_testing() noexcept {
  return fail_next_dft_norm_on_master.exchange(
      false, std::memory_order_acq_rel);
}

void fail_next_cuda_dft_output_after_dataset_for_testing() noexcept {
  fail_next_cuda_dft_output_after_dataset.store(
      true, std::memory_order_release);
}

bool consume_cuda_dft_output_dataset_failure_for_testing() noexcept {
  return fail_next_cuda_dft_output_after_dataset.exchange(
      false, std::memory_order_acq_rel);
}

dft_pair_list_summary append_dft_pair_reduction_requests(
    std::vector<dft_pair_reduction_request_fp32> *destination,
    const dft_chunk *lhs, const dft_chunk *rhs,
    std::size_t frequency_count,
    std::complex<double> fixed_weight,
    bool use_lhs_extra_weight) {
  dft_pair_list_summary summary = {0, 0};
  if (!destination) {
    for (const dft_chunk *left = lhs, *right = rhs;
         left && right;
         left = left->next_in_dft, right = right->next_in_dft) {
      if (frequency_count &&
          left->N >
              std::numeric_limits<std::size_t>::max() / frequency_count)
        throw std::overflow_error(
            "DFT spectral reduction point-frequency count overflow");
      const std::size_t terms = left->N * frequency_count;
      if (summary.point_frequency_terms >
          std::numeric_limits<std::size_t>::max() - terms)
        throw std::overflow_error(
            "DFT spectral reduction total term count overflow");
      if (summary.pair_count ==
          std::numeric_limits<std::size_t>::max())
        throw std::overflow_error(
            "DFT spectral reduction pair count overflow");
      ++summary.pair_count;
      summary.point_frequency_terms += terms;
    }
    return summary;
  }

  if ((lhs == nullptr) != (rhs == nullptr))
    throw std::invalid_argument(
        "DFT spectral reduction chunk lists have different lengths");
  if (frequency_count == 0 && lhs)
    throw std::invalid_argument(
        "DFT spectral reduction chunks require frequencies");
  if (!std::isfinite(fixed_weight.real()) ||
      !std::isfinite(fixed_weight.imag()))
    throw std::invalid_argument(
        "DFT spectral reduction fixed weight must be finite");
  if (sizeof(realnum) != sizeof(float))
    throw std::logic_error(
        "CUDA DFT spectral reduction requires an FP32 Meep build");

  for (const dft_chunk *left = lhs, *right = rhs;
       left || right;
       left = left ? left->next_in_dft : nullptr,
       right = right ? right->next_in_dft : nullptr) {
    if (!left || !right)
      throw std::invalid_argument(
          "DFT spectral reduction chunk lists have different lengths");
    if (!left->dft || !right->dft || left->N == 0 || right->N == 0)
      throw std::invalid_argument(
          "DFT spectral reduction chunk is empty or unallocated");
    if (left->N > right->N)
      throw std::invalid_argument(
          "DFT spectral reduction lhs extent exceeds rhs storage");
    if (left->omega.size() != frequency_count ||
        right->omega.size() != frequency_count ||
        left->omega != right->omega)
      throw std::invalid_argument(
          "DFT spectral reduction frequencies differ between paired chunks");
    if (left->N >
        std::numeric_limits<std::size_t>::max() / frequency_count)
      throw std::overflow_error(
          "DFT spectral reduction point-frequency count overflow");
    const std::size_t terms = left->N * frequency_count;
    if (summary.point_frequency_terms >
        std::numeric_limits<std::size_t>::max() - terms)
      throw std::overflow_error(
          "DFT spectral reduction total term count overflow");
    if (summary.pair_count ==
        std::numeric_limits<std::size_t>::max())
      throw std::overflow_error(
          "DFT spectral reduction pair count overflow");
    const std::complex<double> weight =
        use_lhs_extra_weight
            ? std::complex<double>(left->extra_weight)
            : fixed_weight;
    if (!std::isfinite(weight.real()) || !std::isfinite(weight.imag()))
      throw std::invalid_argument(
          "DFT spectral reduction chunk weight must be finite");
    if (!left->fc || !right->fc)
      throw std::invalid_argument(
          "CUDA DFT spectral reduction chunk owner is missing");
    destination->push_back(
        {left->fc, right->fc,
         reinterpret_cast<const float *>(left->dft),
         reinterpret_cast<const float *>(right->dft),
         left->N, left->N, right->N,
         weight.real(), weight.imag()});
    ++summary.pair_count;
    summary.point_frequency_terms += terms;
  }
  return summary;
}

} // namespace detail
} // namespace gpu

#if MEEP_HAVE_CUDA && MEEP_SINGLE
namespace {

struct cuda_dft_descriptor {
  std::vector<std::ptrdiff_t> field_indices;
  std::vector<realnum> weights;
  std::vector<double> angular_frequencies;
  std::vector<std::ptrdiff_t> norm_point_indices;
  bool norm_point_indices_ready = false;
};

struct cuda_dft_descriptor_registry {
  std::mutex mutex;
  std::unordered_map<const dft_chunk *,
                     std::unique_ptr<cuda_dft_descriptor> >
      descriptors;
};

cuda_dft_descriptor_registry &get_cuda_dft_descriptor_registry() {
  static cuda_dft_descriptor_registry *registry =
      new cuda_dft_descriptor_registry;
  return *registry;
}

const cuda_dft_descriptor &get_cuda_dft_descriptor(
    const dft_chunk *chunk) {
  cuda_dft_descriptor_registry &registry =
      get_cuda_dft_descriptor_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto found = registry.descriptors.find(chunk);
  if (found == registry.descriptors.end())
    throw std::logic_error("CUDA DFT descriptor is missing");
  if (found->second->angular_frequencies.size() != chunk->omega.size())
    throw std::invalid_argument(
        "resizing dft_chunk::omega after monitor construction is not "
        "supported");
  if (found->second->field_indices.size() != chunk->N ||
      found->second->weights.size() != chunk->N)
    throw std::invalid_argument(
        "changing dft_chunk::N after monitor construction is not "
        "supported");
  const bool frequencies_match =
      chunk->omega.empty() ||
      memcmp(found->second->angular_frequencies.data(),
             chunk->omega.data(),
             chunk->omega.size() * sizeof(double)) == 0;
  if (!frequencies_match) {
    // An in-place edit of the historical public omega vector must not leave
    // the clean resident mirror stale. update_dft is not thread-safe, so this
    // registry/cache lock ordering does not introduce a supported concurrent
    // mutation case.
    gpu::detail::discard_resident_mirror_for_owner(
        chunk->fc, found->second->angular_frequencies.data());
    found->second->angular_frequencies = chunk->omega;
  }
  return *found->second;
}

bool cuda_dft_descriptor_size_is_valid(
    const dft_chunk *chunk, std::string *reason) {
  cuda_dft_descriptor_registry &registry =
      get_cuda_dft_descriptor_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto found = registry.descriptors.find(chunk);
  if (found == registry.descriptors.end()) {
    if (reason) *reason = "CUDA DFT descriptor is missing";
    return false;
  }
  if (found->second->angular_frequencies.size() != chunk->omega.size()) {
    if (reason)
      *reason =
          "resizing dft_chunk::omega after monitor construction is not "
          "supported";
    return false;
  }
  if (found->second->field_indices.size() != chunk->N ||
      found->second->weights.size() != chunk->N) {
    if (reason)
      *reason =
          "changing dft_chunk::N after monitor construction is not "
          "supported";
    return false;
  }
  return true;
}

std::unique_ptr<cuda_dft_descriptor>
release_cuda_dft_descriptor(const dft_chunk *chunk) noexcept {
  cuda_dft_descriptor_registry &registry =
      get_cuda_dft_descriptor_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto found = registry.descriptors.find(chunk);
  if (found == registry.descriptors.end()) return nullptr;
  std::unique_ptr<cuda_dft_descriptor> descriptor =
      std::move(found->second);
  registry.descriptors.erase(found);
  return descriptor;
}

const std::vector<std::ptrdiff_t> &get_cuda_dft_norm_point_indices(
    const dft_chunk *chunk, grid_volume fgv) {
  cuda_dft_descriptor_registry &registry =
      get_cuda_dft_descriptor_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto found = registry.descriptors.find(chunk);
  if (found == registry.descriptors.end())
    throw std::logic_error("CUDA DFT descriptor is missing");
  cuda_dft_descriptor &descriptor = *found->second;
  if (!descriptor.norm_point_indices_ready) {
    std::vector<std::ptrdiff_t> indices;
    grid_volume subgv =
        fgv.subvolume(chunk->is, chunk->ie, chunk->c);
    const ivec original_is = chunk->is_old;
    const ivec original_ie = chunk->ie_old;
    LOOP_OVER_IVECS(subgv, original_is, original_ie, idx) {
      if (idx < 0 || static_cast<std::size_t>(idx) >= chunk->N)
        throw std::out_of_range(
            "CUDA persistent DFT norm index is outside the monitor");
      indices.push_back(static_cast<std::ptrdiff_t>(idx));
    }
    descriptor.norm_point_indices.swap(indices);
    descriptor.norm_point_indices_ready = true;
  }
  return descriptor.norm_point_indices;
}

} // namespace
#endif

std::vector<double> linspace(double freq_min, double freq_max, size_t Nfreq) {
  double dfreq = Nfreq <= 1 ? 0.0 : (freq_max - freq_min) / (Nfreq - 1);
  std::vector<double> freq(Nfreq);
  if (Nfreq <= 1)
    freq[0] = (freq_min + freq_max) * 0.5;
  else
    for (size_t i = 0; i < Nfreq; ++i)
      freq[i] = freq_min + i * dfreq;

  return freq;
}

struct dft_chunk_data { // for passing to field::loop_in_chunks as void*
  component c;
  int vc;
  std::vector<double> omega;
  complex<double> stored_weight, extra_weight;
  double dt_factor;
  bool include_dV_and_interp_weights;
  bool sqrt_dV_and_interp_weights;
  bool empty_dim[5];
  dft_chunk *dft_chunks;
  int decimation_factor;
  bool persist;
};

dft_chunk::dft_chunk(fields_chunk *fc_, ivec is_, ivec ie_, vec s0_, vec s1_, vec e0_, vec e1_,
                     double dV0_, double dV1_, component c_, bool use_centered_grid,
                     complex<double> phase_factor, ivec shift_, const symmetry &S_, int sn_,
                     const void *data_) {
  dft_chunk_data *data = (dft_chunk_data *)data_;
  if (!fc_->f[c_][0]) meep::abort("invalid fields_chunk/component combination in dft_chunk");

  fc = fc_;
  is = is_;
  ie = ie_;
  s0 = s0_;
  s1 = s1_;
  e0 = e0_;
  e1 = e1_;
  dV0 = dV0_;
  dV1 = dV1_;

  persist = data->persist;

  c = c_;

  /* for adjoint calculations, we want to pad
  (or expand) the dimensions of the dft region
  to account for boundary effects. We will pad
  by 1 pixel in each dimension, while ensuring
  we don't step outside of the chunk loop itself
  */
  if (persist) {
    is_old = is_;
    ie_old = ie_;
    /* Clamp to this component's own grid, not to the centered grid. Component c
       runs from little_corner()+iyee_shift(c) to big_corner()+iyee_shift(c) (cf.
       LOOP_OVER_VOL), so clamping to the bare corners truncates the topmost node
       of a yee-shifted direction -- precisely the ghost node that the adjoint
       restriction stencil reaches for, and which step_boundaries() already keeps
       current. It also left `is` off the component lattice, which desynchronized
       the LOOP_OVER_IVECS counter from grid_volume::index(). */
    const ivec shift_c = fc->gv.iyee_shift(c);
    is = max(is - one_ivec(fc->gv.dim) * 2, fc->gv.little_corner() + shift_c);
    ie = min(ie + one_ivec(fc->gv.dim) * 2, fc->gv.big_corner() + shift_c);
  }

  if (use_centered_grid)
    fc->gv.yee2cent_offsets(c, avg1, avg2);
  else
    avg1 = avg2 = 0;

  stored_weight = data->stored_weight;
  extra_weight = data->extra_weight;
  scale = stored_weight * phase_factor * data->dt_factor;

  /* this is for e.g. computing E x H, where we don't want to
     multiply by the interpolation weights or the grid_volume twice. */
  include_dV_and_interp_weights = data->include_dV_and_interp_weights;

  /* an alternative way to avoid multipling by interpolation weights twice:
     multiply by square root of the weights */
  sqrt_dV_and_interp_weights = data->sqrt_dV_and_interp_weights;

  shift = shift_;
  S = S_;
  sn = sn_;
  vc = data->vc;
  decimation_factor = data->decimation_factor;

  const int Nomega = data->omega.size();
  omega = data->omega;
  std::unique_ptr<complex<realnum>[]> dft_phase_storage(
      new complex<realnum>[Nomega]);

  N = 1;
  LOOP_OVER_DIRECTIONS(is.dim, d) { N *= (ie.in_direction(d) - is.in_direction(d)) / 2 + 1; }
#if MEEP_HAVE_CUDA
  std::unique_ptr<cuda_dft_descriptor> cuda_descriptor(
      new cuda_dft_descriptor);
  cuda_descriptor->field_indices.reserve(N);
  cuda_descriptor->weights.reserve(N);
  // Keep a private copy for the resident CUDA mirror. get_cuda_dft_descriptor
  // refreshes it if the historical public dft_chunk::omega vector is edited.
  cuda_descriptor->angular_frequencies = data->omega;
  // Descriptor construction appends to two vectors and therefore must be
  // deterministic and single-threaded. The parallel loop macro would race
  // on push_back when MPI tests enable more than one OpenMP thread.
  LOOP_OVER_IVECS(fc->gv, is, ie, idx) {
    double weight;
    if (include_dV_and_interp_weights) {
      weight = IVEC_LOOP_WEIGHT(s0, s1, e0, e1,
                                dV0 + dV1 * loop_i2);
      if (sqrt_dV_and_interp_weights) weight = sqrt(weight);
    }
    else
      weight = 1.0;
    cuda_descriptor->field_indices.push_back(idx);
    cuda_descriptor->weights.push_back(static_cast<realnum>(weight));
  }
  if (cuda_descriptor->field_indices.size() != N ||
      cuda_descriptor->weights.size() != N)
    meep::abort("internal CUDA DFT descriptor size mismatch");
#endif
  std::unique_ptr<complex<realnum>[]> dft_storage(
      new complex<realnum>[N * Nomega]);
  for (size_t i = 0; i < N * Nomega; ++i)
    dft_storage[i] = 0.0;
#if MEEP_HAVE_CUDA
  {
    cuda_dft_descriptor_registry &registry =
        get_cuda_dft_descriptor_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto inserted =
        registry.descriptors.emplace(this, std::move(cuda_descriptor));
    if (!inserted.second)
      meep::abort("duplicate CUDA DFT descriptor registration");
  }
#endif
  dft_phase = dft_phase_storage.release();
  dft = dft_storage.release();
  for (int i = 0; i < 5; ++i)
    empty_dim[i] = data->empty_dim[i];

  next_in_chunk = fc->dft_chunks;
  fc->dft_chunks = this;
  next_in_dft = data->dft_chunks;
}

dft_chunk::~dft_chunk() {
  // Eigenmode retained plans use the first DFT chunk as their stable monitor
  // identity.  Invalidate that identity before any of its host/device inputs
  // are released so a later monitor cannot observe a recycled address.
  gpu::detail::destroy_resident_eigenmode_overlap_plans_for_owner(this);
  // The DFT allocation is disposable here, but the same owner cache can hold
  // device-authoritative field and polarization arrays. Forget only this
  // monitor mirror instead of tearing down the entire chunk cache.
  gpu::detail::discard_resident_mirror_for_owner(fc, dft);
#if MEEP_HAVE_CUDA
  // The descriptor vectors are themselves mirrored persistently. Discard
  // those mirrors before freeing the host vectors, otherwise a later
  // same-sized monitor can reuse the host addresses and observe stale
  // indices or weights on the device.
  std::unique_ptr<cuda_dft_descriptor> descriptor =
      release_cuda_dft_descriptor(this);
  if (descriptor) {
    gpu::detail::discard_resident_mirror_for_owner(
        fc, descriptor->field_indices.data());
    gpu::detail::discard_resident_mirror_for_owner(
        fc, descriptor->weights.data());
    gpu::detail::discard_resident_mirror_for_owner(
        fc, descriptor->angular_frequencies.data());
    if (descriptor->norm_point_indices_ready &&
        !descriptor->norm_point_indices.empty())
      gpu::detail::discard_resident_mirror_for_owner(
          fc, descriptor->norm_point_indices.data());
  }
#endif
  delete[] dft;
  delete[] dft_phase;

  // Persistent adjoint monitors are detached by fields_chunk::~fields_chunk
  // and intentionally outlive their owner.  Only a still-attached monitor
  // can be unlinked from the owner's list here.
  if (fc) {
    const auto unlink = [this](dft_chunk **head) {
      dft_chunk *cur = *head;
      if (cur == this) {
        *head = next_in_chunk;
        return true;
      }
      while (cur && cur->next_in_chunk && cur->next_in_chunk != this)
        cur = cur->next_in_chunk;
      if (!cur || cur->next_in_chunk != this) return false;
      cur->next_in_chunk = next_in_chunk;
      return true;
    };
    if (!unlink(&fc->dft_chunks))
      (void)unlink(&fc->detached_dft_chunks);
  }
}

void dft_flux::remove() {
  gpu::detail::destroy_resident_dft_reduction_plans_for_owner(this);
  invalidate_eigenmode_cache();
  while (E) {
    dft_chunk *nxt = E->next_in_dft;
    delete E;
    E = nxt;
  }
  while (H) {
    dft_chunk *nxt = H->next_in_dft;
    delete H;
    H = nxt;
  }
}

static void add_dft_chunkloop(fields_chunk *fc, int ichunk, component cgrid, ivec is, ivec ie,
                              vec s0, vec s1, vec e0, vec e1, double dV0, double dV1, ivec shift,
                              complex<double> shift_phase, const symmetry &S, int sn,
                              void *chunkloop_data) {
  dft_chunk_data *data = (dft_chunk_data *)chunkloop_data;
  (void)ichunk; // unused

  component c = S.transform(data->c, -sn);
  if (c >= NUM_FIELD_COMPONENTS || !fc->f[c][0]) return; // this chunk doesn't have component c

  data->dft_chunks =
      new dft_chunk(fc, is, ie, s0, s1, e0, e1, dV0, dV1, c, cgrid == Centered,
                    shift_phase * S.phase_shift(c, sn), shift, S, sn, chunkloop_data);
}

dft_chunk *fields::add_dft(component c, const volume &where, const double *freq, size_t Nfreq,
                           bool include_dV_and_interp_weights, complex<double> stored_weight,
                           dft_chunk *chunk_next, bool sqrt_dV_and_interp_weights,
                           complex<double> extra_weight, bool use_centered_grid, int vc,
                           int decimation_factor, bool persist) {
  if (coordinate_mismatch(gv.dim, c)) return NULL;

  /* If you call add_dft before adding sources, it will do nothing
     since no fields will be found.   This is almost certainly not
     what the user wants. */
  if (!components_allocated)
    meep::abort("allocate field components (by adding sources) before adding dft objects");
  if (!include_dV_and_interp_weights && sqrt_dV_and_interp_weights)
    meep::abort("include_dV_and_interp_weights must be true for sqrt_dV_and_interp_weights=true in "
                "add_dft");

  dft_chunk_data data;
  data.persist = persist;
  data.c = c;
  data.vc = vc;

  if (decimation_factor == 0) {
    double src_freq_max = 0;
    bool force_unit_decimation = has_nonlinearities(false);
    for (src_time *s = sources; s; s = s->next) {
      if (s->get_fwidth() == 0)
        force_unit_decimation = true;
      else
        src_freq_max =
            std::max(src_freq_max, std::abs(s->frequency().real()) + 0.5 * s->get_fwidth());
    }
    // add_srcdata can place a source on only a subset of MPI ranks.  An empty
    // local source list must not force decimation to one, while a high-bandwidth
    // or continuous source on any rank must constrain every rank.  Reduce the
    // physical inputs first so all ranks compute the same conservative value.
    src_freq_max = max_to_all(src_freq_max);
    force_unit_decimation = or_to_all(force_unit_decimation);
    double freq_max = 0;
    for (size_t i = 0; i < Nfreq; ++i)
      freq_max = std::max(freq_max, std::abs(freq[i]));
    if ((freq_max > 0) && (src_freq_max > 0) && !force_unit_decimation)
      decimation_factor = std::max(1, int(std::floor(1 / (dt * (freq_max + src_freq_max)))));
    else
      decimation_factor = 1;

    // Retain a final consensus reduction as a defensive guard against future
    // rank-local inputs being added to this calculation.
    decimation_factor = min_to_all(decimation_factor);
  }
  data.decimation_factor = decimation_factor;

  data.omega.resize(Nfreq);
  for (size_t i = 0; i < Nfreq; ++i)
    data.omega[i] = 2 * pi * freq[i];
  data.stored_weight = stored_weight;
  data.extra_weight = extra_weight;
  data.dt_factor = dt / sqrt(2.0 * pi) * decimation_factor;
  data.include_dV_and_interp_weights = include_dV_and_interp_weights;
  data.sqrt_dV_and_interp_weights = sqrt_dV_and_interp_weights;
  data.empty_dim[0] = data.empty_dim[1] = data.empty_dim[2] = data.empty_dim[3] =
      data.empty_dim[4] = false;
  LOOP_OVER_DIRECTIONS(where.dim, d) { data.empty_dim[d] = where.in_direction(d) == 0; }
  data.dft_chunks = chunk_next;
  loop_in_chunks(add_dft_chunkloop, (void *)&data, where, use_centered_grid ? Centered : c);

  return data.dft_chunks;
}

dft_chunk *fields::add_dft(const volume_list *where, const std::vector<double> &freq,
                           bool include_dV_and_interp_weights, bool persist) {
  dft_chunk *chunks = 0;
  while (where) {
    if (is_derived(where->c)) meep::abort("derived_component invalid for dft");
    complex<double> stored_weight = where->weight;
    chunks = add_dft(component(where->c), where->v, freq, include_dV_and_interp_weights,
                     stored_weight, chunks, persist);
    where = where->next;
  }
  return chunks;
}

static void update_public_dft_phase(dft_chunk *chunk, double time) {
  const int frequency_count = chunk->omega.size();
  for (int frequency = 0; frequency < frequency_count; ++frequency)
    chunk->dft_phase[frequency] =
        polar(1.0, chunk->omega[frequency] * time) * chunk->scale;
}

#if MEEP_HAVE_CUDA
static bool append_cuda_dft_requests(
    fields_chunk *chunk, gpu::detail::resident_cache *cache,
    double timeE, double timeH, int current_step,
    std::vector<gpu::detail::dft_update_request_fp32> &requests) {
  bool appended = false;
  for (dft_chunk *cur = chunk->dft_chunks; cur;
       cur = cur->next_in_chunk) {
    if ((current_step % cur->get_decimation_factor()) != 0 ||
        !cur->fc->f[cur->c][0])
      continue;
    const cuda_dft_descriptor &descriptor =
        get_cuda_dft_descriptor(cur);
    const double update_time =
        is_H_or_B(cur->c) ? timeH : timeE;
    update_public_dft_phase(cur, update_time);
    requests.push_back(
        {cache, reinterpret_cast<float *>(cur->dft),
         cur->fc->f[cur->c][0], cur->fc->f[cur->c][1],
         cur->fc->gv.ntot(), descriptor.field_indices.data(),
         descriptor.weights.data(), cur->N,
         descriptor.angular_frequencies.data(),
         descriptor.angular_frequencies.size(), update_time,
         cur->scale.real(), cur->scale.imag(), cur->avg1, cur->avg2});
    appended = true;
  }
  return appended;
}
#endif

void fields::update_dfts() {
  am_now_working_on(FourierTransforming);
#if MEEP_HAVE_CUDA
  const bool cuda_phase_eligible =
      gpu::detail::cuda_active() && sizeof(realnum) == sizeof(float);
  if (cuda_phase_eligible) {
    std::vector<std::unique_ptr<gpu::detail::resident_curl_session> >
        sessions(static_cast<std::size_t>(num_chunks));
    std::vector<gpu::detail::dft_update_request_fp32> requests;
    bool batch_path_ready = true;
    for (int chunk_index = 0; chunk_index < num_chunks; ++chunk_index) {
      fields_chunk *chunk = chunks[chunk_index];
      if (!chunk->is_mine() || chunk->doing_solve_cw) continue;
      bool has_active_update = false;
      for (dft_chunk *cur = chunk->dft_chunks; cur;
           cur = cur->next_in_chunk)
        has_active_update =
            has_active_update ||
            ((t % cur->get_decimation_factor()) == 0 &&
             cur->fc->f[cur->c][0]);
      if (!has_active_update) continue;
      sessions[static_cast<std::size_t>(chunk_index)].reset(
          new gpu::detail::resident_curl_session(chunk, true));
      gpu::detail::resident_curl_session *session =
          sessions[static_cast<std::size_t>(chunk_index)].get();
      if (!session->active()) {
        batch_path_ready = false;
        break;
      }
      append_cuda_dft_requests(
          chunk, session->cache(), time(), time() - 0.5 * dt, t,
          requests);
    }
    if (batch_path_ready) {
      if (!requests.empty())
        gpu::detail::resident_update_dft_batch_fp32(
            requests.data(), requests.size());
      for (auto &session : sessions)
        if (session) session->finish();
      finished_working();
      return;
    }
    for (auto &session : sessions)
      if (session) session->finish();
  }
#endif
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine())
      chunks[i]->update_dfts(time(), time() - 0.5 * dt, t);
  finished_working();
}

void fields_chunk::update_dfts(double timeE, double timeH, int current_step) {
  if (doing_solve_cw) return;
#if MEEP_HAVE_CUDA
  const bool cuda_phase_eligible =
      gpu::detail::cuda_active() && sizeof(realnum) == sizeof(float);
  if (cuda_phase_eligible) {
    std::vector<gpu::detail::dft_update_request_fp32> requests;
    gpu::detail::resident_curl_session cuda_session(this, true);
    if (cuda_session.active()) {
      append_cuda_dft_requests(
          this, cuda_session.cache(), timeE, timeH, current_step,
          requests);
      if (!requests.empty()) {
        gpu::detail::resident_update_dft_batch_fp32(
            requests.data(), requests.size());
      }
      cuda_session.finish();
      return;
    }
  }
#endif
  for (dft_chunk *cur = dft_chunks; cur; cur = cur->next_in_chunk) {
    if ((current_step % cur->get_decimation_factor()) == 0) {
      cur->update_dft(is_H_or_B(cur->c) ? timeH : timeE);
    }
  }
}

bool fields::cuda_dft_preflight(std::string *reason) const {
#if MEEP_HAVE_CUDA
  for (int chunk_index = 0; chunk_index < num_chunks; ++chunk_index)
    if (chunks[chunk_index]->is_mine())
      for (dft_chunk *chunk = chunks[chunk_index]->dft_chunks; chunk;
           chunk = chunk->next_in_chunk) {
        std::string chunk_reason;
        if (!cuda_dft_descriptor_size_is_valid(chunk, &chunk_reason)) {
          if (reason) {
            *reason = "DFT monitor in chunk " +
                      std::to_string(chunk_index) + ": " + chunk_reason;
          }
          return false;
        }
      }
#else
  (void)reason;
#endif
  return true;
}

void dft_chunk::update_dft(double time) {
  if (!fc->f[c][0]) return;

  const int Nomega = omega.size();
  const bool cuda_phase_eligible =
      gpu::detail::cuda_active() && sizeof(realnum) == sizeof(float);
  if (gpu::detail::cuda_active() && !cuda_phase_eligible)
    throw std::runtime_error(
        "preflighted CUDA DFT update requires FP32 fields");
#if MEEP_HAVE_CUDA
  const cuda_dft_descriptor *descriptor = nullptr;
  if (cuda_phase_eligible)
    descriptor = &get_cuda_dft_descriptor(this);
#endif

  // dft_phase is public API state and historically contains the phases from
  // the most recent update. Keep that observable state current on both CPU
  // and CUDA; the CUDA path constructs its own phases from resident omega and
  // does not transfer this host cache.
  update_public_dft_phase(this, time);

  gpu::detail::resident_curl_session cuda_session(fc,
                                                   cuda_phase_eligible);
#if MEEP_HAVE_CUDA
  if (cuda_session.active()) {
    static_assert(sizeof(std::complex<realnum>) == 2 * sizeof(realnum),
                  "CUDA DFT requires interleaved std::complex storage");
    gpu::detail::resident_update_dft_fp32(
        cuda_session.cache(), reinterpret_cast<float *>(dft), fc->f[c][0],
        fc->f[c][1], fc->gv.ntot(),
        descriptor->field_indices.data(), descriptor->weights.data(), N,
        descriptor->angular_frequencies.data(),
        descriptor->angular_frequencies.size(), time, scale.real(),
        scale.imag(), avg1, avg2);
    cuda_session.finish();
    return;
  }
#endif

  int numcmp = fc->f[c][1] ? 2 : 1;

  PLOOP_OVER_IVECS(fc->gv, is, ie, idx) {
    size_t idx_dft = IVEC_LOOP_COUNTER;
    double w;
    if (include_dV_and_interp_weights) {
      w = IVEC_LOOP_WEIGHT(s0, s1, e0, e1, dV0 + dV1 * loop_i2);
      if (sqrt_dV_and_interp_weights) w = sqrt(w);
    }
    else
      w = 1.0;
    realnum f[2]; // real/imag field value at epsilon point
    if (avg2)
      for (int cmp = 0; cmp < numcmp; ++cmp)
        f[cmp] = (w * 0.25) * (fc->f[c][cmp][idx] + fc->f[c][cmp][idx + avg1] +
                               fc->f[c][cmp][idx + avg2] + fc->f[c][cmp][idx + (avg1 + avg2)]);
    else if (avg1)
      for (int cmp = 0; cmp < numcmp; ++cmp)
        f[cmp] = (w * 0.5) * (fc->f[c][cmp][idx] + fc->f[c][cmp][idx + avg1]);
    else
      for (int cmp = 0; cmp < numcmp; ++cmp)
        f[cmp] = w * fc->f[c][cmp][idx];

    if (numcmp == 2) {
      complex<realnum> fc(f[0], f[1]);
      for (int i = 0; i < Nomega; ++i)
        dft[Nomega * idx_dft + i] += dft_phase[i] * fc;
    }
    else {
      realnum fr = f[0];
      for (int i = 0; i < Nomega; ++i)
        dft[Nomega * idx_dft + i] +=
            std::complex<realnum>{fr * dft_phase[i].real(), fr * dft_phase[i].imag()};
    }
  }
  gpu::detail::record_cpu_dft(N * static_cast<size_t>(Nomega));
}

/* Return the L2 norm of the DFTs themselves.  This is useful
   to check whether the simulation is finished (whether all relevant fields have decayed).
   (Collective operation.) */
double fields::dft_norm() {
  am_now_working_on(Other);
  // Local descriptor validation, CUDA allocation/launch, or a device-to-host
  // scalar copy can fail before sum_to_all.  A surviving rank would otherwise
  // wait forever in the collective.  dft_norm is collective even on the CPU,
  // so arm the guard for every distributed invocation before doing any local
  // work that may throw.
  distributed_dft_norm_failure_guard distributed_failure_guard(
      count_processors() > 1);
  if (am_master() &&
      gpu::detail::consume_dft_norm_master_failure_for_testing())
    throw std::runtime_error(
        "injected rank-local DFT norm failure for MPI deadlock regression");
  double sum = 0.0;
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) sum += chunks[i]->dft_norm2(gv);
  finished_working();
  return std::sqrt(sum_to_all(sum));
}

double fields_chunk::dft_norm2(grid_volume fgv) const {
  double sum = 0.0;
  for (dft_chunk *cur = dft_chunks; cur; cur = cur->next_in_chunk)
    sum += cur->norm2(fgv);
  return sum;
}

static double sqr(std::complex<realnum> x) {
  // Keep the CPU oracle independent of compiler contraction flags.  The
  // volatile stores force both FP32 products to round before the FP32 add,
  // exactly matching the CUDA __fmul_rn/__fadd_rn sequence used by the
  // resident DFT reduction.
  volatile realnum real_squared = x.real() * x.real();
  volatile realnum imaginary_squared = x.imag() * x.imag();
  const realnum magnitude_squared = real_squared + imaginary_squared;
  return static_cast<double>(magnitude_squared);
}

double dft_chunk::norm2(grid_volume fgv) const {
  if (!fc->f[c][0]) return 0.0;
  const size_t Nomega = omega.size();
#if MEEP_HAVE_CUDA
  // Keep an explicit diagnostic escape hatch so a device reduction can be
  // compared against the historical full DFT synchronization without
  // rebuilding a second binary. This is intentionally opt-in and is also
  // useful when diagnosing a new CUDA architecture.
  const bool disable_cuda_norm_reduction =
      getenv("MEEP_GPU_DISABLE_DFT_NORM_REDUCTION") != nullptr;
  if (gpu::detail::cuda_active() && !disable_cuda_norm_reduction) {
    const std::ptrdiff_t *point_indices = nullptr;
    size_t selected_point_count = N;
    if (persist) {
      const std::vector<std::ptrdiff_t> &indices =
          get_cuda_dft_norm_point_indices(this, fgv);
      point_indices = indices.data();
      selected_point_count = indices.size();
    }
    double device_sum = 0.0;
    if (gpu::detail::resident_dft_norm2_fp32_for_owner(
            fc, reinterpret_cast<const float *>(dft), N, Nomega,
            point_indices, selected_point_count, &device_sum))
      return device_sum;
  }
#endif
  gpu::detail::sync_resident_cache_for_owner(fc);
  double sum = 0.0;
  size_t idx_dft;
  /* looping over chunks that have been "expanded"
  for adjoint calculations requires some care. Namely,
  we want to make sure we don't double count the padding
  and can replicate results with different chunk combinations.
  */
  if (persist) {
    grid_volume subgv = fgv.subvolume(is, ie, c);
    LOOP_OVER_IVECS(subgv, is_old, ie_old, idx) {
      /* Index by position: the loop counter is not a valid dft[] index here,
         because the persistent pad must respect this component's Yee lattice
         and only grid_volume::index() accounts for that shift. */
      IVEC_LOOP_ILOC(subgv, ip);
      ptrdiff_t didx = subgv.index(c, ip);
      if (didx < 0 || (size_t)didx >= N) continue;
      for (size_t i = 0; i < Nomega; ++i)
        sum += sqr(dft[Nomega * didx + i]);
    }
  }
  /* note we place the if outside of the
  loop to avoid branching. This routine gets
  called a lot, so let's try to stay efficient
  (at the expense of uglier code).
   */
  else {
    LOOP_OVER_IVECS(fgv, is, ie, idx) {
      idx_dft = IVEC_LOOP_COUNTER;
      for (size_t i = 0; i < Nomega; ++i)
        sum += sqr(dft[Nomega * idx_dft + i]);
    }
  }

  return sum;
}

// return the maximum decimation factor across
// all dft regions
int fields::max_decimation() const {
  int maxdec = 1;
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) maxdec = std::max(maxdec, chunks[i]->max_decimation());
  return max_to_all(maxdec);
}

int fields_chunk::max_decimation() const {
  int maxdec = std::numeric_limits<int>::min();
  for (dft_chunk *cur = dft_chunks; cur; cur = cur->next_in_chunk)
    maxdec = std::max(maxdec, cur->get_decimation_factor());
  return maxdec;
}

// return the maximum abs(freq) over all DFT chunks
double fields::dft_maxfreq() const {
  double maxfreq = 0;
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) maxfreq = std::max(maxfreq, chunks[i]->dft_maxfreq());
  return max_to_all(maxfreq);
}

double fields_chunk::dft_maxfreq() const {
  double maxomega = 0;
  for (dft_chunk *cur = dft_chunks; cur; cur = cur->next_in_chunk)
    maxomega = std::max(maxomega, cur->maxomega());
  return maxomega / (2 * meep::pi);
}

double dft_chunk::maxomega() const {
  double maxomega = 0;
  for (const auto &o : omega)
    maxomega = std::max(maxomega, std::abs(o));
  return maxomega;
}

void dft_chunk::scale_dft(complex<double> scale) {
  if (!std::isfinite(scale.real()) || !std::isfinite(scale.imag()))
    throw invalid_argument("DFT scale must be finite");
  if (N && omega.size() > numeric_limits<size_t>::max() / N)
    throw overflow_error("DFT scale value count overflow");
  const size_t value_count = N * omega.size();
  if (gpu::detail::cuda_active()) {
    if (sizeof(realnum) != sizeof(float))
      throw runtime_error(
          "Meep CUDA DFT scale requires a single-precision build");
    const gpu::detail::resident_dft_scale_result result =
        gpu::detail::scale_resident_complex_dft_fp32_for_owner(
            fc, reinterpret_cast<float *>(dft), value_count,
            scale.real(), scale.imag());
    gpu::detail::record_cuda_dft_scale(
        value_count, result.kernel_launches,
        result.host_to_device_bytes);
  }
  else {
    gpu::detail::sync_resident_cache_for_owner(fc);
    for (size_t i = 0; i < value_count; ++i)
      dft[i] *= scale;
    gpu::detail::record_cpu_dft_scale(value_count);
    gpu::detail::discard_resident_mirror_for_owner(fc, dft);
  }
  if (next_in_dft) next_in_dft->scale_dft(scale);
}

void dft_chunk::operator-=(const dft_chunk &chunk) {
  gpu::detail::sync_resident_cache_for_owner(fc);
  gpu::detail::sync_resident_cache_for_owner(chunk.fc);
  if (c != chunk.c || N * omega.size() != chunk.N * chunk.omega.size())
    meep::abort("Mismatched chunks in dft_chunk::operator-=");

  for (size_t i = 0; i < N * omega.size(); ++i)
    dft[i] -= chunk.dft[i];

  if (next_in_dft) {
    if (!chunk.next_in_dft) meep::abort("Mismatched chunk lists in dft_chunk::operator-=");
    *next_in_dft -= *chunk.next_in_dft;
  }
  gpu::detail::discard_resident_mirror_for_owner(fc, dft);
}

size_t my_dft_chunks_Ntotal(dft_chunk *dft_chunks, size_t *my_start) {
  // When writing to a sharded file, we write out only the chunks we own.
  size_t n = 0;
  for (dft_chunk *cur = dft_chunks; cur; cur = cur->next_in_dft) {
    if (cur->N &&
        cur->omega.size() > numeric_limits<size_t>::max() / cur->N / 2)
      throw overflow_error("DFT checkpoint chunk value count overflow");
    const size_t chunk_values = cur->N * cur->omega.size() * 2;
    if (chunk_values > numeric_limits<size_t>::max() - n)
      throw overflow_error("DFT checkpoint total value count overflow");
    n += chunk_values;
  }

  *my_start = 0;
  return n;
}

size_t dft_chunks_Ntotal(dft_chunk *dft_chunks, size_t *my_start) {
  // If writing to a single parallel file, we are compute our chunks offset
  // into the single-parallel-file that has all the data.
  size_t n = my_dft_chunks_Ntotal(dft_chunks, my_start);
  *my_start = partial_sum_to_all(n) - n; // sum(n) for processes before this
  return sum_to_all(n);
}

size_t dft_chunks_Ntotal(dft_chunk *dft_chunks, size_t *my_start, bool single_parallel_file) {
  return single_parallel_file ? dft_chunks_Ntotal(dft_chunks, my_start)
                              : my_dft_chunks_Ntotal(dft_chunks, my_start);
}

namespace {

struct staged_checkpoint_chunk {
  dft_chunk *chunk;
  size_t scalar_count;
  vector<realnum> values;
};

struct staged_checkpoint_dataset {
  vector<staged_checkpoint_chunk> chunks;
  size_t local_values;
  size_t host_to_device_bytes;
};

void append_checkpoint_ranges(
    dft_chunk *chunks,
    vector<gpu::detail::resident_checkpoint_range_fp32> &ranges) {
  for (dft_chunk *cur = chunks; cur; cur = cur->next_in_dft) {
    if (cur->N &&
        cur->omega.size() > numeric_limits<size_t>::max() / cur->N / 2)
      throw overflow_error("DFT checkpoint range value count overflow");
    ranges.push_back(
        {cur->fc, reinterpret_cast<const float *>(cur->dft),
         cur->N * cur->omega.size() * 2});
  }
}

void save_dft_hdf5_dataset(
    dft_chunk *dft_chunks, const char *name, h5file *file,
    const char *dprefix, bool single_parallel_file, bool use_cuda,
    size_t full_cache_device_to_host_bytes_avoided) {
  size_t istart;
  size_t n = dft_chunks_Ntotal(dft_chunks, &istart, single_parallel_file);
  size_t ignored_start = 0;
  const size_t local_values =
      my_dft_chunks_Ntotal(dft_chunks, &ignored_start);

  char dataname[1024];
  snprintf(dataname, 1024,
           "%s%s"
           "%s_dft",
           dprefix ? dprefix : "", dprefix && dprefix[0] ? "_" : "", name);
  file->create_data(dataname, 1, &n);

  size_t staged_d2h_bytes = 0;
  vector<realnum> staging;
  for (dft_chunk *cur = dft_chunks; cur; cur = cur->next_in_dft) {
    size_t Nchunk = cur->N * cur->omega.size() * 2;
    const realnum *payload = reinterpret_cast<const realnum *>(cur->dft);
    if (use_cuda) {
      staging.resize(Nchunk);
      const gpu::detail::resident_range_readback readback =
          gpu::detail::stage_resident_range_fp32_for_owner(
              cur->fc, reinterpret_cast<const float *>(cur->dft),
              Nchunk, reinterpret_cast<float *>(staging.data()));
      payload = staging.data();
      if (readback.device_to_host_bytes >
          numeric_limits<size_t>::max() - staged_d2h_bytes)
        throw overflow_error("DFT checkpoint D2H byte count overflow");
      staged_d2h_bytes += readback.device_to_host_bytes;
    }
    else {
      gpu::detail::sync_resident_cache_for_owner(cur->fc);
    }
    if (Nchunk > numeric_limits<size_t>::max() - istart)
      throw overflow_error("DFT checkpoint save offset overflow");
    file->write_chunk(1, &istart, &Nchunk,
                      const_cast<realnum *>(payload));
    istart += Nchunk;
  }
  file->done_writing_chunks();
  if (use_cuda)
    gpu::detail::record_cuda_dft_checkpoint_save(
        local_values, staged_d2h_bytes,
        full_cache_device_to_host_bytes_avoided);
  else
    gpu::detail::record_cpu_dft_checkpoint_save(local_values);
}

void save_dft_hdf5_many_impl(
    const dft_hdf5_dataset *datasets, size_t dataset_count, h5file *file,
    const char *dprefix, bool single_parallel_file) {
  if (!datasets && dataset_count)
    throw invalid_argument("DFT checkpoint dataset list must be non-null");
  if (!file && dataset_count)
    throw invalid_argument("DFT checkpoint HDF5 file must be non-null");
  const bool use_cuda = gpu::detail::cuda_active();
  if (use_cuda && sizeof(realnum) != sizeof(float))
    throw std::runtime_error(
        "Meep CUDA DFT checkpoint save requires a single-precision build");

  vector<gpu::detail::resident_checkpoint_range_fp32> checkpoint_ranges;
  for (size_t index = 0; index < dataset_count; ++index) {
    if (!datasets[index].name)
      throw invalid_argument("DFT checkpoint dataset name must be non-null");
    append_checkpoint_ranges(datasets[index].chunks, checkpoint_ranges);
  }
  const gpu::detail::resident_checkpoint_traffic checkpoint_traffic =
      use_cuda
          ? gpu::detail::summarize_resident_checkpoint_ranges_fp32(
                checkpoint_ranges.data(), checkpoint_ranges.size())
          : gpu::detail::resident_checkpoint_traffic{0, 0, 0};
  for (size_t index = 0; index < dataset_count; ++index) {
    save_dft_hdf5_dataset(
        datasets[index].chunks, datasets[index].name, file, dprefix,
        single_parallel_file, use_cuda,
        index == 0
            ? checkpoint_traffic.full_cache_device_to_host_bytes_avoided
            : 0);
    if (index + 1 < dataset_count) file->prevent_deadlock();
  }
}

staged_checkpoint_dataset prepare_dft_hdf5_dataset(
    dft_chunk *dft_chunks, const char *name, h5file *file,
    const char *dprefix, bool single_parallel_file) {
  staged_checkpoint_dataset staged_dataset;
  staged_dataset.local_values = 0;
  staged_dataset.host_to_device_bytes = 0;
  size_t istart;
  size_t n = dft_chunks_Ntotal(dft_chunks, &istart, single_parallel_file);
  size_t ignored_start = 0;
  staged_dataset.local_values =
      my_dft_chunks_Ntotal(dft_chunks, &ignored_start);

  char dataname[1024];
  snprintf(dataname, 1024,
           "%s%s"
           "%s_dft",
           dprefix ? dprefix : "", dprefix && dprefix[0] ? "_" : "", name);
  int file_rank;
  size_t file_dims;
  file->read_size(dataname, &file_rank, &file_dims, 1);
  if (file_rank != 1 || file_dims != n)
    meep::abort("incorrect dataset size (%zd vs. %zd) in load_dft_hdf5 %s:%s", file_dims, n,
                file->file_name(), dataname);

  for (dft_chunk *cur = dft_chunks; cur; cur = cur->next_in_dft) {
    const size_t Nchunk = cur->N * cur->omega.size() * 2;
    staged_checkpoint_chunk staged = {cur, Nchunk, vector<realnum>(Nchunk)};
    if (Nchunk > numeric_limits<size_t>::max() - istart)
      throw overflow_error("DFT checkpoint load offset overflow");
    file->read_chunk(1, &istart, &staged.scalar_count,
                     staged.values.data());
    istart += Nchunk;
    staged_dataset.chunks.push_back(std::move(staged));
  }
  return staged_dataset;
}

void commit_dft_hdf5_datasets(
    vector<staged_checkpoint_dataset> &datasets, bool use_cuda) {
  // Commit the complete validated/read host checkpoint before changing any
  // mirror.  From this point onward an upload failure leaves every host DFT
  // at the new checkpoint.  All stale mirrors are removed before the first
  // upload, so a partial device commit can contain only matching mirrors or
  // no mirror at all, never old/new mixed authority.
  for (staged_checkpoint_dataset &dataset : datasets)
    for (staged_checkpoint_chunk &staged : dataset.chunks)
      if (staged.scalar_count)
        copy(staged.values.begin(), staged.values.end(),
             reinterpret_cast<realnum *>(staged.chunk->dft));
  if (use_cuda) {
    for (staged_checkpoint_dataset &dataset : datasets)
      for (staged_checkpoint_chunk &staged : dataset.chunks)
        gpu::detail::discard_resident_mirror_for_owner(
            staged.chunk->fc, staged.chunk->dft);
    for (staged_checkpoint_dataset &dataset : datasets)
      for (staged_checkpoint_chunk &staged : dataset.chunks) {
        const gpu::detail::resident_range_upload upload =
            gpu::detail::upload_resident_range_fp32_for_owner(
                staged.chunk->fc,
                reinterpret_cast<float *>(staged.chunk->dft),
                staged.scalar_count);
        if (!upload.resident)
          throw runtime_error(
              "Meep CUDA DFT checkpoint load did not create a resident mirror");
        if (upload.host_to_device_bytes >
            numeric_limits<size_t>::max() -
                dataset.host_to_device_bytes)
          throw overflow_error("DFT checkpoint H2D byte count overflow");
        dataset.host_to_device_bytes += upload.host_to_device_bytes;
      }
  }
  for (const staged_checkpoint_dataset &dataset : datasets)
    if (use_cuda)
      gpu::detail::record_cuda_dft_checkpoint_load(
          dataset.local_values, dataset.host_to_device_bytes);
    else
      gpu::detail::record_cpu_dft_checkpoint_load(dataset.local_values);
}

void load_dft_hdf5_many_impl(
    const dft_hdf5_dataset *datasets, size_t dataset_count, h5file *file,
    const char *dprefix, bool single_parallel_file) {
  if (!datasets && dataset_count)
    throw invalid_argument("DFT checkpoint dataset list must be non-null");
  if (!file && dataset_count)
    throw invalid_argument("DFT checkpoint HDF5 file must be non-null");
  const bool use_cuda = gpu::detail::cuda_active();
  if (use_cuda && sizeof(realnum) != sizeof(float))
    throw std::runtime_error(
        "Meep CUDA DFT checkpoint load requires a single-precision build");
  vector<staged_checkpoint_dataset> staged_datasets;
  staged_datasets.reserve(dataset_count);
  for (size_t index = 0; index < dataset_count; ++index) {
    if (!datasets[index].name)
      throw invalid_argument("DFT checkpoint dataset name must be non-null");
    staged_datasets.push_back(prepare_dft_hdf5_dataset(
        datasets[index].chunks, datasets[index].name, file, dprefix,
        single_parallel_file));
    if (index + 1 < dataset_count) file->prevent_deadlock();
  }
  commit_dft_hdf5_datasets(staged_datasets, use_cuda);
}

} // namespace

// Note: the file must have been created in parallel mode, typically via fields::open_h5file.
void save_dft_hdf5(dft_chunk *dft_chunks, const char *name, h5file *file, const char *dprefix,
                   bool single_parallel_file) {
  distributed_dft_checkpoint_failure_guard distributed_failure_guard("save");
  const dft_hdf5_dataset dataset = {dft_chunks, name};
  save_dft_hdf5_many_impl(&dataset, 1, file, dprefix,
                          single_parallel_file);
  distributed_failure_guard.dismiss();
}

void save_dft_hdf5(dft_chunk *dft_chunks, component c, h5file *file, const char *dprefix,
                   bool single_parallel_file) {
  save_dft_hdf5(dft_chunks, component_name(c), file, dprefix, single_parallel_file);
}

void save_dft_hdf5_many(const dft_hdf5_dataset *datasets,
                        size_t dataset_count, h5file *file,
                        const char *dprefix, bool single_parallel_file) {
  distributed_dft_checkpoint_failure_guard distributed_failure_guard("save");
  save_dft_hdf5_many_impl(datasets, dataset_count, file, dprefix,
                          single_parallel_file);
  distributed_failure_guard.dismiss();
}

void load_dft_hdf5(dft_chunk *dft_chunks, const char *name, h5file *file, const char *dprefix,
                   bool single_parallel_file) {
  distributed_dft_checkpoint_failure_guard distributed_failure_guard("load");
  const dft_hdf5_dataset dataset = {dft_chunks, name};
  load_dft_hdf5_many_impl(&dataset, 1, file, dprefix,
                          single_parallel_file);
  distributed_failure_guard.dismiss();
}

void load_dft_hdf5(dft_chunk *dft_chunks, component c, h5file *file, const char *dprefix,
                   bool single_parallel_file) {
  load_dft_hdf5(dft_chunks, component_name(c), file, dprefix, single_parallel_file);
}

void load_dft_hdf5_many(const dft_hdf5_dataset *datasets,
                        size_t dataset_count, h5file *file,
                        const char *dprefix, bool single_parallel_file) {
  distributed_dft_checkpoint_failure_guard distributed_failure_guard("load");
  load_dft_hdf5_many_impl(datasets, dataset_count, file, dprefix,
                          single_parallel_file);
  distributed_failure_guard.dismiss();
}

dft_flux::dft_flux(const component cE_, const component cH_, dft_chunk *E_, dft_chunk *H_,
                   double fmin, double fmax, int Nf, const volume &where_,
                   direction normal_direction_, bool use_symmetry_)
    : E(E_), H(H_), cE(cE_), cH(cH_), where(where_), normal_direction(normal_direction_),
      use_symmetry(use_symmetry_), eigenmode_cache(NULL), eigenmode_cache_dispersive(false),
      eigenmode_cache_frequency(0) {
  freq = meep::linspace(fmin, fmax, Nf);
}

dft_flux::dft_flux(const component cE_, const component cH_, dft_chunk *E_, dft_chunk *H_,
                   const std::vector<double> &freq_, const volume &where_,
                   direction normal_direction_, bool use_symmetry_)
    : E(E_), H(H_), cE(cE_), cH(cH_), where(where_), normal_direction(normal_direction_),
      use_symmetry(use_symmetry_), eigenmode_cache(NULL), eigenmode_cache_dispersive(false),
      eigenmode_cache_frequency(0) {
  freq = freq_;
}

dft_flux::dft_flux(const component cE_, const component cH_, dft_chunk *E_, dft_chunk *H_,
                   const double *freq_, size_t Nfreq, const volume &where_,
                   direction normal_direction_, bool use_symmetry_)
    : freq(Nfreq), E(E_), H(H_), cE(cE_), cH(cH_), where(where_),
      normal_direction(normal_direction_), use_symmetry(use_symmetry_), eigenmode_cache(NULL),
      eigenmode_cache_dispersive(false), eigenmode_cache_frequency(0) {
  for (size_t i = 0; i < Nfreq; ++i)
    freq[i] = freq_[i];
}

dft_flux::dft_flux(const dft_flux &f) : where(f.where) {
  freq = f.freq;
  E = f.E;
  H = f.H;
  cE = f.cE;
  cH = f.cH;
  normal_direction = f.normal_direction;
  use_symmetry = f.use_symmetry;
  eigenmode_cache = NULL; // don't share cache across copies
  eigenmode_cache_dispersive = false;
  eigenmode_cache_frequency = 0;
}

dft_flux::~dft_flux() {
  gpu::detail::destroy_resident_dft_reduction_plans_for_owner(this);
  invalidate_eigenmode_cache();
}

double *dft_flux::flux() {
  distributed_dft_reduction_failure_guard distributed_failure_guard(
      count_processors() > 1);
  const size_t Nfreq = freq.size();
  const bool use_cuda = gpu::detail::cuda_active();
  std::vector<gpu::detail::dft_pair_reduction_request_fp32> requests;
  const gpu::detail::dft_pair_list_summary summary =
      gpu::detail::append_dft_pair_reduction_requests(
          use_cuda ? &requests : nullptr, E, H, Nfreq,
          std::complex<double>(1.0, 0.0), false);
  std::vector<double> local(Nfreq, 0.0);
  if (use_cuda && !requests.empty()) {
    if (Nfreq > std::numeric_limits<size_t>::max() / 2)
      throw std::overflow_error("DFT flux result size overflow");
    std::vector<double> complex_result(2 * Nfreq, 0.0);
    gpu::detail::resident_reduce_dft_pairs_fp32(
        this, 0, requests.data(), requests.size(), Nfreq,
        complex_result.data());
    for (size_t i = 0; i < Nfreq; ++i)
      local[i] = complex_result[2 * i];
  }
  else if (!use_cuda) {
    sync_dft_chunk_list(E);
    sync_dft_chunk_list(H);
    for (dft_chunk *curE = E, *curH = H; curE && curH;
         curE = curE->next_in_dft, curH = curH->next_in_dft)
      for (size_t k = 0; k < curE->N; ++k)
        for (size_t i = 0; i < Nfreq; ++i)
          local[i] += real(curE->dft[k * Nfreq + i] *
                           conj(curH->dft[k * Nfreq + i]));
    gpu::detail::record_cpu_dft_reduction(
        summary.pair_count, summary.point_frequency_terms);
  }
  std::unique_ptr<double[]> global(new double[Nfreq]);
  sum_dft_reduction_to_all(local.data(), global.get(), Nfreq);
  distributed_failure_guard.dismiss();
  return global.release();
}

std::vector<std::complex<double> > dft_flux::complexflux() {
  distributed_dft_reduction_failure_guard distributed_failure_guard(
      count_processors() > 1);
  const size_t Nfreq = freq.size();
  const bool use_cuda = gpu::detail::cuda_active();
  std::vector<gpu::detail::dft_pair_reduction_request_fp32> requests;
  const gpu::detail::dft_pair_list_summary summary =
      gpu::detail::append_dft_pair_reduction_requests(
          use_cuda ? &requests : nullptr, E, H, Nfreq,
          std::complex<double>(1.0, 0.0), false);
  std::vector<std::complex<double> > local(Nfreq, 0.0);
  if (use_cuda && !requests.empty()) {
    if (Nfreq > std::numeric_limits<size_t>::max() / 2)
      throw std::overflow_error(
          "DFT complex-flux result size overflow");
    std::vector<double> staged(2 * Nfreq, 0.0);
    gpu::detail::resident_reduce_dft_pairs_fp32(
        this, 0, requests.data(), requests.size(), Nfreq,
        staged.data());
    for (size_t i = 0; i < Nfreq; ++i)
      local[i] = std::complex<double>(staged[2 * i],
                                      staged[2 * i + 1]);
  }
  else if (!use_cuda) {
    sync_dft_chunk_list(E);
    sync_dft_chunk_list(H);
    for (dft_chunk *curE = E, *curH = H; curE && curH;
         curE = curE->next_in_dft, curH = curH->next_in_dft)
      for (size_t k = 0; k < curE->N; ++k)
        for (size_t i = 0; i < Nfreq; ++i)
          local[i] += curE->dft[k * Nfreq + i] *
                      conj(curH->dft[k * Nfreq + i]);
    gpu::detail::record_cpu_dft_reduction(
        summary.pair_count, summary.point_frequency_terms);
  }
  std::vector<std::complex<double> > global(Nfreq);
  sum_dft_reduction_to_all(local.data(), global.data(), Nfreq);
  distributed_failure_guard.dismiss();
  return global;
}

void dft_flux::save_hdf5(h5file *file, const char *dprefix) {
  const dft_hdf5_dataset datasets[] = {
      {E, component_name(cE)}, {H, component_name(cH)}};
  save_dft_hdf5_many(datasets, 2, file, dprefix);
}

void dft_flux::load_hdf5(h5file *file, const char *dprefix) {
  const dft_hdf5_dataset datasets[] = {
      {E, component_name(cE)}, {H, component_name(cH)}};
  load_dft_hdf5_many(datasets, 2, file, dprefix);
}

void dft_flux::save_hdf5(fields &f, const char *fname, const char *dprefix, const char *prefix) {
  std::unique_ptr<h5file> ff(f.open_h5file(fname, h5file::WRITE, prefix));
  save_hdf5(ff.get(), dprefix);
}

void dft_flux::load_hdf5(fields &f, const char *fname, const char *dprefix, const char *prefix) {
  std::unique_ptr<h5file> ff(f.open_h5file(fname, h5file::READONLY, prefix));
  load_hdf5(ff.get(), dprefix);
}

void dft_flux::scale_dfts(complex<double> scale) {
  if (E) E->scale_dft(scale);
  if (H) H->scale_dft(scale);
}

dft_flux fields::add_dft_flux(const volume_list *where_, const double *freq, size_t Nfreq,
                              bool use_symmetry, bool centered_grid, int decimation_factor) {
  if (!where_) // handle empty list of volumes
    return dft_flux(Ex, Hy, NULL, NULL, freq, Nfreq, v, NO_DIRECTION, use_symmetry);

  dft_chunk *E = 0, *H = 0;
  component cE[2] = {Ex, Ey}, cH[2] = {Hy, Hx};

  // the dft_flux object needs to store the (unreduced) volume for
  // mode-coefficient computation in mpb.cpp, but this only works
  // when the volume_list consists of a single volume, so it suffices
  // to store the first volume in the list.
  volume firstvol(where_->v);

  volume_list *where = use_symmetry ? S.reduce(where_) : new volume_list(where_);
  volume_list *where_save = where;
  while (where) {
    derived_component c = derived_component(where->c);
    if (coordinate_mismatch(gv.dim, component_direction(c)))
      meep::abort("coordinate-type mismatch in add_dft_flux");

    switch (c) {
      case Sx: cE[0] = Ey, cE[1] = Ez, cH[0] = Hz, cH[1] = Hy; break;
      case Sy: cE[0] = Ez, cE[1] = Ex, cH[0] = Hx, cH[1] = Hz; break;
      case Sr: cE[0] = Ep, cE[1] = Ez, cH[0] = Hz, cH[1] = Hp; break;
      case Sp: cE[0] = Ez, cE[1] = Er, cH[0] = Hr, cH[1] = Hz; break;
      case Sz:
        if (gv.dim == Dcyl)
          cE[0] = Er, cE[1] = Ep, cH[0] = Hp, cH[1] = Hr;
        else
          cE[0] = Ex, cE[1] = Ey, cH[0] = Hy, cH[1] = Hx;
        break;
      default: meep::abort("invalid flux component!");
    }

    for (int i = 0; i < 2; ++i) {
      E = add_dft(cE[i], where->v, freq, Nfreq, true, where->weight * double(1 - 2 * i), E, false,
                  std::complex<double>(1.0, 0), centered_grid, 0, decimation_factor);
      H = add_dft(cH[i], where->v, freq, Nfreq, false, 1.0, H, false, std::complex<double>(1.0, 0),
                  centered_grid, 0, decimation_factor);
    }

    where = where->next;
  }
  delete where_save;

  // if the volume list has only one entry, store its component's direction.
  // if the volume list has > 1 entry, store NO_DIRECTION.
  direction flux_dir = (where_->next ? NO_DIRECTION : component_direction(where_->c));
  return dft_flux(cE[0], cH[0], E, H, freq, Nfreq, firstvol, flux_dir, use_symmetry);
}

dft_energy::dft_energy(dft_chunk *E_, dft_chunk *H_, dft_chunk *D_, dft_chunk *B_, double fmin,
                       double fmax, int Nf, const volume &where_)
    : E(E_), H(H_), D(D_), B(B_), where(where_) {
  freq = meep::linspace(fmin, fmax, Nf);
}

dft_energy::dft_energy(dft_chunk *E_, dft_chunk *H_, dft_chunk *D_, dft_chunk *B_,
                       const std::vector<double> &freq_, const volume &where_)
    : E(E_), H(H_), D(D_), B(B_), where(where_) {
  freq = freq_;
}

dft_energy::dft_energy(dft_chunk *E_, dft_chunk *H_, dft_chunk *D_, dft_chunk *B_,
                       const double *freq_, size_t Nfreq, const volume &where_)
    : freq(Nfreq), E(E_), H(H_), D(D_), B(B_), where(where_) {
  for (size_t i = 0; i < Nfreq; ++i)
    freq[i] = freq_[i];
}

dft_energy::dft_energy(const dft_energy &f) : where(f.where) {
  freq = f.freq;
  E = f.E;
  H = f.H;
  D = f.D;
  B = f.B;
}

dft_energy::~dft_energy() {
  gpu::detail::destroy_resident_dft_reduction_plans_for_owner(this);
}

double *dft_energy::electric() {
  distributed_dft_reduction_failure_guard distributed_failure_guard(
      count_processors() > 1);
  const size_t Nfreq = freq.size();
  const bool use_cuda = gpu::detail::cuda_active();
  std::vector<gpu::detail::dft_pair_reduction_request_fp32> requests;
  const gpu::detail::dft_pair_list_summary summary =
      gpu::detail::append_dft_pair_reduction_requests(
          use_cuda ? &requests : nullptr, E, D, Nfreq,
          std::complex<double>(0.5, 0.0), false);
  std::vector<double> local(Nfreq, 0.0);
  if (use_cuda && !requests.empty()) {
    if (Nfreq > std::numeric_limits<size_t>::max() / 2)
      throw std::overflow_error(
          "DFT electric-energy result size overflow");
    std::vector<double> staged(2 * Nfreq, 0.0);
    gpu::detail::resident_reduce_dft_pairs_fp32(
        this, 0, requests.data(), requests.size(), Nfreq,
        staged.data());
    for (size_t i = 0; i < Nfreq; ++i)
      local[i] = staged[2 * i];
  }
  else if (!use_cuda) {
    sync_dft_chunk_list(E);
    sync_dft_chunk_list(D);
    for (dft_chunk *curE = E, *curD = D; curE && curD;
         curE = curE->next_in_dft, curD = curD->next_in_dft)
      for (size_t k = 0; k < curE->N; ++k)
        for (size_t i = 0; i < Nfreq; ++i)
          local[i] +=
              0.5 * real(conj(curE->dft[k * Nfreq + i]) *
                         curD->dft[k * Nfreq + i]);
    gpu::detail::record_cpu_dft_reduction(
        summary.pair_count, summary.point_frequency_terms);
  }
  std::unique_ptr<double[]> global(new double[Nfreq]);
  sum_dft_reduction_to_all(local.data(), global.get(), Nfreq);
  distributed_failure_guard.dismiss();
  return global.release();
}

double *dft_energy::magnetic() {
  distributed_dft_reduction_failure_guard distributed_failure_guard(
      count_processors() > 1);
  const size_t Nfreq = freq.size();
  const bool use_cuda = gpu::detail::cuda_active();
  std::vector<gpu::detail::dft_pair_reduction_request_fp32> requests;
  const gpu::detail::dft_pair_list_summary summary =
      gpu::detail::append_dft_pair_reduction_requests(
          use_cuda ? &requests : nullptr, H, B, Nfreq,
          std::complex<double>(0.5, 0.0), false);
  std::vector<double> local(Nfreq, 0.0);
  if (use_cuda && !requests.empty()) {
    if (Nfreq > std::numeric_limits<size_t>::max() / 2)
      throw std::overflow_error(
          "DFT magnetic-energy result size overflow");
    std::vector<double> staged(2 * Nfreq, 0.0);
    gpu::detail::resident_reduce_dft_pairs_fp32(
        this, 1, requests.data(), requests.size(), Nfreq,
        staged.data());
    for (size_t i = 0; i < Nfreq; ++i)
      local[i] = staged[2 * i];
  }
  else if (!use_cuda) {
    sync_dft_chunk_list(H);
    sync_dft_chunk_list(B);
    for (dft_chunk *curH = H, *curB = B; curH && curB;
         curH = curH->next_in_dft, curB = curB->next_in_dft)
      for (size_t k = 0; k < curH->N; ++k)
        for (size_t i = 0; i < Nfreq; ++i)
          local[i] +=
              0.5 * real(conj(curH->dft[k * Nfreq + i]) *
                         curB->dft[k * Nfreq + i]);
    gpu::detail::record_cpu_dft_reduction(
        summary.pair_count, summary.point_frequency_terms);
  }
  std::unique_ptr<double[]> global(new double[Nfreq]);
  sum_dft_reduction_to_all(local.data(), global.get(), Nfreq);
  distributed_failure_guard.dismiss();
  return global.release();
}

double *dft_energy::total() {
  const size_t Nfreq = freq.size();
  double *Fe = electric();
  double *Fm = magnetic();
  double *F = new double[Nfreq];
  for (size_t i = 0; i < Nfreq; ++i)
    F[i] = Fe[i] + Fm[i];
  delete[] Fe;
  delete[] Fm;
  return F;
}

dft_energy fields::add_dft_energy(const volume_list *where_, const double *freq, size_t Nfreq,
                                  int decimation_factor) {

  if (!where_) // handle empty list of volumes
    return dft_energy(NULL, NULL, NULL, NULL, freq, Nfreq, v);

  dft_chunk *E = 0, *D = 0, *H = 0, *B = 0;
  volume firstvol(where_->v);
  volume_list *where = new volume_list(where_);
  volume_list *where_save = where;
  while (where) {
    LOOP_OVER_FIELD_DIRECTIONS(gv.dim, d) {
      E = add_dft(direction_component(Ex, d), where->v, freq, Nfreq, true, 1.0, E, false, 1.0, true,
                  0, decimation_factor);
      D = add_dft(direction_component(Dx, d), where->v, freq, Nfreq, false, 1.0, D, false, 1.0,
                  true, 0, decimation_factor);
      H = add_dft(direction_component(Hx, d), where->v, freq, Nfreq, true, 1.0, H, false, 1.0, true,
                  0, decimation_factor);
      B = add_dft(direction_component(Bx, d), where->v, freq, Nfreq, false, 1.0, B, false, 1.0,
                  true, 0, decimation_factor);
    }
    where = where->next;
  }
  delete where_save;

  return dft_energy(E, H, D, B, freq, Nfreq, firstvol);
}

void dft_energy::save_hdf5(h5file *file, const char *dprefix) {
  const dft_hdf5_dataset datasets[] = {
      {E, "E"}, {D, "D"}, {H, "H"}, {B, "B"}};
  save_dft_hdf5_many(datasets, 4, file, dprefix);
}

void dft_energy::load_hdf5(h5file *file, const char *dprefix) {
  const dft_hdf5_dataset datasets[] = {
      {E, "E"}, {D, "D"}, {H, "H"}, {B, "B"}};
  load_dft_hdf5_many(datasets, 4, file, dprefix);
}

void dft_energy::save_hdf5(fields &f, const char *fname, const char *dprefix, const char *prefix) {
  std::unique_ptr<h5file> ff(f.open_h5file(fname, h5file::WRITE, prefix));
  save_hdf5(ff.get(), dprefix);
}

void dft_energy::load_hdf5(fields &f, const char *fname, const char *dprefix, const char *prefix) {
  std::unique_ptr<h5file> ff(f.open_h5file(fname, h5file::READONLY, prefix));
  load_hdf5(ff.get(), dprefix);
}

void dft_energy::scale_dfts(complex<double> scale) {
  if (E) E->scale_dft(scale);
  if (D) D->scale_dft(scale);
  if (H) H->scale_dft(scale);
  if (B) B->scale_dft(scale);
}

void dft_energy::remove() {
  gpu::detail::destroy_resident_dft_reduction_plans_for_owner(this);
  while (E) {
    dft_chunk *nxt = E->next_in_dft;
    delete E;
    E = nxt;
  }
  while (D) {
    dft_chunk *nxt = D->next_in_dft;
    delete D;
    D = nxt;
  }
  while (H) {
    dft_chunk *nxt = H->next_in_dft;
    delete H;
    H = nxt;
  }
  while (B) {
    dft_chunk *nxt = B->next_in_dft;
    delete B;
    B = nxt;
  }
}

direction fields::normal_direction(const volume &where) const {
  direction d = where.normal_direction();
  if (d == NO_DIRECTION) {
    /* hack so that we still infer the normal direction correctly for
       volumes with empty dimensions */
    volume where_pad(where);
    LOOP_OVER_DIRECTIONS(where.dim, d1) {
      if (nosize_direction(d1) && where.in_direction(d1) == 0.0)
        where_pad.set_direction_max(d1, where.in_direction_min(d1) + 0.1);
    }
    d = where_pad.normal_direction();
    if (d == NO_DIRECTION && gv.dim == D2 && beta != 0 && where_pad.in_direction(X) > 0 &&
        where_pad.in_direction(Y) > 0)
      d = Z;
    if (d == NO_DIRECTION)
      meep::abort("Could not determine normal direction for given grid_volume.");
  }
  return d;
}

dft_flux fields::add_dft_flux(direction d, const volume &where, const double *freq, size_t Nfreq,
                              bool use_symmetry, bool centered_grid, int decimation_factor) {
  if (d == NO_DIRECTION) d = normal_direction(where);
  volume_list vl(where, direction_component(Sx, d));
  dft_flux flux = add_dft_flux(&vl, freq, Nfreq, use_symmetry, centered_grid, decimation_factor);
  flux.normal_direction = d;
  return flux;
}

dft_flux fields::add_mode_monitor(direction d, const volume &where, const double *freq,
                                  size_t Nfreq, bool centered_grid, int decimation_factor) {
  return add_dft_flux(d, where, freq, Nfreq, /*use_symmetry=*/false, centered_grid,
                      decimation_factor);
}

dft_flux fields::add_dft_flux_box(const volume &where, double freq_min, double freq_max,
                                  int Nfreq) {
  return add_dft_flux_box(where, meep::linspace(freq_min, freq_max, Nfreq));
}

dft_flux fields::add_dft_flux_box(const volume &where, const std::vector<double> &freq) {
  volume_list *faces = 0;
  LOOP_OVER_DIRECTIONS(where.dim, d) {
    if (where.in_direction(d) > 0) {
      volume face(where);
      derived_component c = direction_component(Sx, d);
      face.set_direction_min(d, where.in_direction_max(d));
      faces = new volume_list(face, c, +1, faces);
      face.set_direction_min(d, where.in_direction_min(d));
      face.set_direction_max(d, where.in_direction_min(d));
      faces = new volume_list(face, c, -1, faces);
    }
  }

  dft_flux flux = add_dft_flux(faces, freq);
  delete faces;
  return flux;
}

dft_flux fields::add_dft_flux_plane(const volume &where, double freq_min, double freq_max,
                                    int Nfreq) {
  return add_dft_flux_plane(where, meep::linspace(freq_min, freq_max, Nfreq));
}

dft_flux fields::add_dft_flux_plane(const volume &where, const std::vector<double> &freq) {
  return add_dft_flux(NO_DIRECTION, where, freq);
}

dft_fields::dft_fields(dft_chunk *chunks_, double freq_min, double freq_max, int Nf,
                       const volume &where_)
    : where(where_) {
  chunks = chunks_;
  freq = meep::linspace(freq_min, freq_max, Nf);
}

dft_fields::dft_fields(dft_chunk *chunks_, const std::vector<double> &freq_, const volume &where_)
    : where(where_) {
  chunks = chunks_;
  freq = freq_;
}

dft_fields::dft_fields(dft_chunk *chunks_, const double *freq_, size_t Nfreq, const volume &where_)
    : freq(Nfreq), where(where_) {
  chunks = chunks_;
  for (size_t i = 0; i < Nfreq; ++i)
    freq[i] = freq_[i];
}

void dft_fields::scale_dfts(complex<double> scale) {
  if (chunks) chunks->scale_dft(scale);
}

void dft_fields::remove() {
  while (chunks) {
    dft_chunk *nxt = chunks->next_in_dft;
    delete chunks;
    chunks = nxt;
  }
}

dft_fields fields::add_dft_fields(component *components, int num_components, const volume where,
                                  const double *freq, size_t Nfreq, bool use_centered_grid,
                                  int decimation_factor, bool persist) {
  bool include_dV_and_interp_weights = false;
  bool sqrt_dV_and_interp_weights = false; // default option from meep.hpp (expose to user?)
  std::complex<double> extra_weight = 1.0; // default option from meep.hpp (expose to user?)
  complex<double> stored_weight = 1.0;
  dft_chunk *chunks = NULL;
  for (int nc = 0; nc < num_components; nc++)
    chunks = add_dft(components[nc], where, freq, Nfreq, include_dV_and_interp_weights,
                     stored_weight, chunks, sqrt_dV_and_interp_weights, extra_weight,
                     use_centered_grid, 0, decimation_factor, persist);

  return dft_fields(chunks, freq, Nfreq, where);
}

/***************************************************************/
/* chunk-level processing for fields::process_dft_component.   */
/***************************************************************/
complex<double> dft_chunk::process_dft_component(int rank, direction *ds, ivec min_corner,
                                                 ivec max_corner, int num_freq, h5file *file,
                                                 realnum *buffer, int reim,
                                                 complex<realnum> *field_array, void *mode1_data,
                                                 void *mode2_data, int ic_conjugate,
                                                 bool retain_interp_weights, fields *parent) {

  const component sampled_component =
      component(ic_conjugate >= 0 ? ic_conjugate : -ic_conjugate);
  if (sampled_component != Dielectric && sampled_component != Permeability)
    gpu::detail::sync_resident_cache_for_owner(fc);
  if ((num_freq < 0) || (num_freq > static_cast<int>(omega.size()) - 1))
    meep::abort("process_dft_component: frequency index %d is outside the range of the frequency "
                "array of size %lu",
                num_freq, omega.size());

  /*****************************************************************/
  /* compute the size of the chunk we own and its strides etc.     */
  /*****************************************************************/
  size_t start[3] = {0, 0, 0};
  size_t file_count[3] = {1, 1, 1}, array_count[3] = {1, 1, 1};
  int file_offset[3] = {0, 0, 0};
  int file_stride[3] = {1, 1, 1};
  ivec isS = S.transform(is, sn) + shift;
  ivec ieS = S.transform(ie, sn) + shift;

  ivec permute(zero_ivec(fc->gv.dim));
  for (int i = 0; i < 3; ++i)
    permute.set_direction(fc->gv.yucky_direction(i), i);
  permute = S.transform_unshifted(permute, sn);
  LOOP_OVER_DIRECTIONS(permute.dim, d) { permute.set_direction(d, abs(permute.in_direction(d))); }

  for (int i = 0; i < rank; ++i) {
    direction d = ds[i];
    int isd = isS.in_direction(d), ied = ieS.in_direction(d);
    start[i] = (std::min(isd, ied) - min_corner.in_direction(d)) / 2;
    file_count[i] = abs(ied - isd) / 2 + 1;
    if (ied < isd) file_offset[permute.in_direction(d)] = file_count[i] - 1;
    array_count[i] = (max_corner.in_direction(d) - min_corner.in_direction(d)) / 2 + 1;
  }

  for (int i = 0; i < rank; ++i) {
    direction d = ds[i];
    int j = permute.in_direction(d);
    for (int k = i + 1; k < rank; ++k)
      file_stride[j] *= file_count[k];
    file_offset[j] *= file_stride[j];
    if (file_offset[j]) file_stride[j] *= -1;
  }

  /*****************************************************************/
  /* For collapsing empty dimensions, we want to retain interpolation
     weights for empty dimensions, but not interpolation weights for
     integration of edge pixels (for retain_interp_weights == true).
     All of the weights are stored in (s0, s1, e0, e1), so we make
     a copy of these with the weights for non-empty dimensions set to 1. */
  vec s0i(s0), s1i(s1), e0i(e0), e1i(e1);
  LOOP_OVER_DIRECTIONS(fc->gv.dim, d) {
    if (!empty_dim[d]) {
      s0i.set_direction(d, 1.0);
      s1i.set_direction(d, 1.0);
      e0i.set_direction(d, 1.0);
      e1i.set_direction(d, 1.0);
    }
  }

  /***************************************************************/
  /* loop over all grid points in our piece of the volume        */
  /***************************************************************/
  vec rshift(shift * (0.5 * fc->gv.inva));
  int chunk_idx = 0;
  complex<double> integral = 0.0;
  component c_conjugate = (component)(ic_conjugate >= 0 ? ic_conjugate : -ic_conjugate);
  LOOP_OVER_IVECS(fc->gv, is, ie, idx) {
    IVEC_LOOP_LOC(fc->gv, loc);
    loc = S.transform(loc, sn) + rshift;
    double w = IVEC_LOOP_WEIGHT(s0, s1, e0, e1, dV0 + dV1 * loop_i2);
    double interp_w = retain_interp_weights ? IVEC_LOOP_WEIGHT(s0i, s1i, e0i, e1i, 1.0) : 1.0;

    complex<double> dft_val =
        (c_conjugate == NO_COMPONENT ? w
         : c_conjugate == Dielectric ? parent->get_eps(loc)
         : c_conjugate == Permeability
             ? parent->get_mu(loc)
             : complex<double>(dft[omega.size() * (chunk_idx++) + num_freq]) / stored_weight);
    if (include_dV_and_interp_weights && dft_val != 0.0)
      dft_val /= (sqrt_dV_and_interp_weights ? sqrt(w) : w);

    complex<double> mode1val = 0.0, mode2val = 0.0;
    if (mode1_data) mode1val = eigenmode_amplitude(mode1_data, loc, S.transform(c_conjugate, sn));
    if (mode2_data) mode2val = eigenmode_amplitude(mode2_data, loc, S.transform(c, sn));

    if (file) {
      int idx2 = ((((file_offset[0] + file_offset[1] + file_offset[2]) + loop_i1 * file_stride[0]) +
                   loop_i2 * file_stride[1]) +
                  loop_i3 * file_stride[2]);

      dft_val *= interp_w;

      complex<double> val = (mode1_data ? mode1val : dft_val);
      buffer[idx2] = reim ? imag(val) : real(val);
    }
    else if (field_array) {
      IVEC_LOOP_ILOC(fc->gv, iloc);         // iloc <-- indices of parent point in Yee grid
      iloc = S.transform(iloc, sn) + shift; // iloc <-- indices of child point in Yee grid
      iloc -= min_corner;                   // iloc <-- 2*(indices of point in DFT array)

      // the index of point n1 or (n1,n2) or (n1,n2,n3) in a 1D, 2D, or 3D array is
      // (for a 1D array) n1
      // (for a 2D array) n2 + n1*N2
      // (for a 3D array) n3 + n2*N3 + n1*N2*N3
      // where NI = number of points in Ith direction.
      int idx2 = 0;
      for (int i = rank - 1, stride = 1; i >= 0; stride *= array_count[i--])
        idx2 += stride * (iloc.in_direction(ds[i]) / 2);
      field_array[idx2] = interp_w * dft_val;
    }
    else {
      mode1val = conj(mode1val); // conjugated inner product
      if (mode2_data)
        integral += w * mode1val * mode2val;
      else
        integral += w * mode1val * dft_val;
    }

  } // LOOP_OVER_IVECS(fc->gv, is, ie, idx)

  if (file) file->write_chunk(rank, start, file_count, buffer);

  return integral;
}

// get variables that are needed by complex<double> fields::process_dft_component
void fields::get_dft_component_dims(dft_chunk **chunklists, int num_chunklists, component c,
                                    ivec &min_corner, ivec &max_corner, size_t &array_size,
                                    size_t &bufsz, int &rank, direction *ds, size_t *dims,
                                    int *array_rank, size_t *array_dims, direction *array_dirs) {
  /***************************************************************/
  /* get statistics on the volume slice **************************/
  /***************************************************************/
  volume *where = &v; // use full volume of fields
  bufsz = 0;
  min_corner = gv.round_vec(where->get_max_corner()) + one_ivec(gv.dim);
  max_corner = gv.round_vec(where->get_min_corner()) - one_ivec(gv.dim);

  for (int ncl = 0; ncl < num_chunklists; ncl++)
    for (dft_chunk *chunk = chunklists[ncl]; chunk; chunk = chunk->next_in_dft) {
      if (chunk->c != c) continue;
      ivec isS = chunk->S.transform(chunk->is, chunk->sn) + chunk->shift;
      ivec ieS = chunk->S.transform(chunk->ie, chunk->sn) + chunk->shift;
      min_corner = min(min_corner, min(isS, ieS));
      max_corner = max(max_corner, max(isS, ieS));
      size_t this_bufsz = 1;
      LOOP_OVER_DIRECTIONS(chunk->fc->gv.dim, d) {
        this_bufsz *= (chunk->ie.in_direction(d) - chunk->is.in_direction(d)) / 2 + 1;
      }
      bufsz = std::max(bufsz, this_bufsz);
    }
  am_now_working_on(MpiAllTime);
  max_corner = max_to_all(max_corner);
  min_corner = -max_to_all(-min_corner); // i.e., min_to_all
  finished_working();

  /***************************************************************/
  /***************************************************************/
  /***************************************************************/
  rank = 0;
  array_size = 1;
  LOOP_OVER_DIRECTIONS(gv.dim, d) {
    if (rank >= 3) meep::abort("too many dimensions in process_dft_component");
    size_t n = std::max(0, (max_corner.in_direction(d) - min_corner.in_direction(d)) / 2 + 1);

    if (n > 1) {
      ds[rank] = d;
      dims[rank++] = n;
      array_size *= n;
    }
  }
  if (array_rank) {
    *array_rank = rank;
    for (int d = 0; d < rank; d++) {
      if (array_dims) array_dims[d] = dims[d];
      if (array_dirs) array_dirs[d] = ds[d];
    }
  }
}

namespace {

void materialize_synthetic_component_array_host(
    fields *parent, dft_chunk **chunklists, int num_chunklists, component c,
    component material, int rank, direction *ds, ivec min_corner,
    ivec max_corner, size_t array_size,
    complex<realnum> *field_array) {
  std::vector<double> local_weights(array_size, 0.0);
  std::vector<double> global_weights(array_size, 0.0);
  size_t array_count[3] = {1, 1, 1};
  for (int dim = 0; dim < rank; ++dim)
    array_count[dim] = static_cast<size_t>(
        (max_corner.in_direction(ds[dim]) -
         min_corner.in_direction(ds[dim])) /
            2 +
        1);

  for (int ncl = 0; ncl < num_chunklists; ++ncl)
    for (dft_chunk *chunk = chunklists[ncl]; chunk;
         chunk = chunk->next_in_dft)
      if (chunk->c == c) {
        vec s0i(chunk->s0), s1i(chunk->s1), e0i(chunk->e0),
            e1i(chunk->e1);
        LOOP_OVER_DIRECTIONS(chunk->fc->gv.dim, d) {
          if (!chunk->empty_dim[d]) {
            s0i.set_direction(d, 1.0);
            s1i.set_direction(d, 1.0);
            e0i.set_direction(d, 1.0);
            e1i.set_direction(d, 1.0);
          }
        }
        LOOP_OVER_IVECS(chunk->fc->gv, chunk->is, chunk->ie, idx) {
          IVEC_LOOP_ILOC(chunk->fc->gv, iloc);
          iloc = chunk->S.transform(iloc, chunk->sn) + chunk->shift;
          iloc -= min_corner;
          size_t destination = 0;
          size_t stride = 1;
          for (int dim = rank - 1; dim >= 0; --dim) {
            const ptrdiff_t coordinate = iloc.in_direction(ds[dim]) / 2;
            if (coordinate < 0 ||
                static_cast<size_t>(coordinate) >= array_count[dim])
              throw std::out_of_range(
                  "synthetic DFT material destination is outside the "
                  "dense array");
            destination += stride * static_cast<size_t>(coordinate);
            if (dim > 0) stride *= array_count[dim];
          }
          local_weights[destination] =
              IVEC_LOOP_WEIGHT(s0i, s1i, e0i, e1i, 1.0);
        }
      }

  constexpr size_t mpi_batch_size = static_cast<size_t>(1) << 20;
  for (size_t offset = 0; offset < array_size;) {
    const size_t batch =
        std::min(mpi_batch_size, array_size - offset);
    parent->am_now_working_on(MpiAllTime);
    sum_to_all(local_weights.data() + offset,
               global_weights.data() + offset,
               static_cast<int>(batch));
    parent->finished_working();
    gpu::detail::record_dft_array_mpi_allreduce(batch * sizeof(double));
    offset += batch;
  }

  for (size_t destination = 0; destination < array_size; ++destination) {
    if (global_weights[destination] == 0.0) {
      field_array[destination] = 0.0;
      continue;
    }
    size_t remainder = destination;
    ivec iloc(min_corner);
    for (int dim = rank - 1; dim >= 0; --dim) {
      const size_t coordinate = remainder % array_count[dim];
      remainder /= array_count[dim];
      iloc.set_direction(
          ds[dim], min_corner.in_direction(ds[dim]) +
                       2 * static_cast<int>(coordinate));
    }
    const vec location = parent->gv[iloc];
    const complex<double> value =
        material == Dielectric ? parent->get_eps(location)
                               : parent->get_mu(location);
    field_array[destination] = global_weights[destination] * value;
  }
}

} // namespace

#if MEEP_HAVE_CUDA
namespace {

struct cuda_dft_materialization_metadata {
  const dft_chunk *chunk;
  std::vector<std::ptrdiff_t> destination_indices;
  std::vector<float> point_weights;
  std::vector<std::uint8_t> zero_divisor_flags;
};

void materialize_dft_component_array_cuda(
    dft_chunk **chunklists, int num_chunklists, component c, int num_freq,
    int rank, const direction *ds, ivec min_corner, ivec max_corner,
    size_t array_size, complex<realnum> *field_array,
    const gpu::detail::dft_materialization_collapse_fp32 *collapse = nullptr,
    size_t reduced_array_size = 0) {
  static_assert(sizeof(complex<realnum>) == 2 * sizeof(float),
                "CUDA DFT materialization requires interleaved complex FP32");

  std::vector<cuda_dft_materialization_metadata> metadata;
  for (int ncl = 0; ncl < num_chunklists; ++ncl)
    for (dft_chunk *chunk = chunklists[ncl]; chunk;
         chunk = chunk->next_in_dft)
      if (chunk->c == c) {
        if (num_freq < 0 ||
            num_freq >= static_cast<int>(chunk->omega.size()))
          meep::abort(
              "CUDA DFT materialization: frequency index %d is outside "
              "the range of the frequency array of size %lu",
              num_freq, chunk->omega.size());
        if (chunk->stored_weight == complex<double>(0.0, 0.0))
          throw std::runtime_error(
              "Meep CUDA DFT materialization encountered a zero stored "
              "weight");

        cuda_dft_materialization_metadata entry;
        entry.chunk = chunk;
        entry.destination_indices.reserve(chunk->N);
        entry.point_weights.reserve(chunk->N);
        entry.zero_divisor_flags.reserve(chunk->N);

        size_t array_count[3] = {1, 1, 1};
        for (int dim = 0; dim < rank; ++dim) {
          const direction d = ds[dim];
          array_count[dim] =
              static_cast<size_t>(
                  (max_corner.in_direction(d) -
                   min_corner.in_direction(d)) /
                      2 +
                  1);
        }

        vec s0i(chunk->s0), s1i(chunk->s1), e0i(chunk->e0),
            e1i(chunk->e1);
        LOOP_OVER_DIRECTIONS(chunk->fc->gv.dim, d) {
          if (!chunk->empty_dim[d]) {
            s0i.set_direction(d, 1.0);
            s1i.set_direction(d, 1.0);
            e0i.set_direction(d, 1.0);
            e1i.set_direction(d, 1.0);
          }
        }

        LOOP_OVER_IVECS(chunk->fc->gv, chunk->is, chunk->ie, idx) {
          IVEC_LOOP_ILOC(chunk->fc->gv, iloc);
          iloc = chunk->S.transform(iloc, chunk->sn) + chunk->shift;
          iloc -= min_corner;

          std::ptrdiff_t destination = 0;
          std::ptrdiff_t stride = 1;
          for (int dim = rank - 1; dim >= 0; --dim) {
            const std::ptrdiff_t coordinate =
                iloc.in_direction(ds[dim]) / 2;
            if (coordinate < 0 ||
                static_cast<size_t>(coordinate) >= array_count[dim])
              throw std::out_of_range(
                  "Meep CUDA DFT materialization destination coordinate "
                  "is outside the dense array");
            destination += stride * coordinate;
            if (dim > 0) {
              if (array_count[dim] >
                  static_cast<size_t>(
                      std::numeric_limits<std::ptrdiff_t>::max() /
                      stride))
                throw std::overflow_error(
                    "Meep CUDA DFT materialization destination stride "
                    "overflow");
              stride *= static_cast<std::ptrdiff_t>(array_count[dim]);
            }
          }

          const double interp_weight = IVEC_LOOP_WEIGHT(
              s0i, s1i, e0i, e1i, 1.0);
          double publication_weight = interp_weight;
          if (chunk->include_dV_and_interp_weights) {
            const double stored_point_weight = IVEC_LOOP_WEIGHT(
                chunk->s0, chunk->s1, chunk->e0, chunk->e1,
                chunk->dV0 + chunk->dV1 * loop_i2);
            const double divisor = chunk->sqrt_dV_and_interp_weights
                                       ? std::sqrt(stored_point_weight)
                                       : stored_point_weight;
            if (!std::isfinite(divisor))
              throw std::runtime_error(
                  "Meep CUDA DFT materialization encountered a non-finite "
                  "stored point weight");
            // CPU process_dft_component deliberately leaves an exactly-zero
            // DFT sample unchanged when its cylindrical-axis dV is zero.  A
            // separate flag lets the device make the same value-dependent
            // decision without reading resident DFT storage on the host and
            // remains correct under fast-math/signed-zero transformations.
            entry.zero_divisor_flags.push_back(divisor == 0.0 ? 1u : 0u);
            if (divisor != 0.0) publication_weight /= divisor;
          }
          else
            entry.zero_divisor_flags.push_back(0u);
          if (!std::isfinite(publication_weight))
            throw std::runtime_error(
                "Meep CUDA DFT materialization encountered a non-finite "
                "publication weight");
          entry.destination_indices.push_back(destination);
          entry.point_weights.push_back(
              static_cast<float>(publication_weight));
        }
        if (entry.destination_indices.size() != chunk->N ||
            entry.point_weights.size() != chunk->N ||
            entry.zero_divisor_flags.size() != chunk->N)
          throw std::logic_error(
              "Meep CUDA DFT materialization metadata size does not match "
              "the DFT chunk");
        metadata.push_back(std::move(entry));
      }

  std::vector<gpu::detail::dft_materialization_request_fp32> requests;
  requests.reserve(metadata.size());
  for (const cuda_dft_materialization_metadata &entry : metadata) {
    const dft_chunk *chunk = entry.chunk;
    const complex<double> inverse_weight = 1.0 / chunk->stored_weight;
    requests.push_back(
        {chunk->fc, reinterpret_cast<const float *>(chunk->dft),
         entry.destination_indices.data(), entry.point_weights.data(),
         chunk->N, chunk->N, chunk->omega.size(),
         static_cast<size_t>(num_freq), 1,
         static_cast<float>(inverse_weight.real()),
         static_cast<float>(inverse_weight.imag()),
         entry.zero_divisor_flags.data()});
  }

  if (collapse)
    gpu::detail::resident_materialize_collapsed_dft_array_fp32(
        requests.empty() ? nullptr : requests.data(), requests.size(),
        reinterpret_cast<float *>(field_array), array_size,
        reduced_array_size, 1, *collapse);
  else
    gpu::detail::resident_materialize_dft_array_fp32(
        requests.empty() ? nullptr : requests.data(), requests.size(),
        reinterpret_cast<float *>(field_array), array_size, 1);
}

} // namespace
#endif

#if MEEP_HAVE_CUDA && MEEP_SINGLE
namespace {

struct cuda_dft_output_chunk_metadata {
  dft_chunk *chunk;
  size_t start[3];
  size_t count[3];
  size_t output_point_offset;
  std::vector<std::ptrdiff_t> destination_indices;
  std::vector<float> point_weights;
  std::vector<std::uint8_t> zero_divisor_flags;
};

struct cuda_dft_output_component_metadata {
  component c;
  int rank;
  size_t dims[3];
  size_t global_point_count;
  std::vector<cuda_dft_output_chunk_metadata> chunks;
};

size_t checked_dft_output_product(size_t left, size_t right,
                                  const char *label) {
  if (right && left > std::numeric_limits<size_t>::max() / right)
    throw std::overflow_error(std::string("Meep CUDA DFT output ") +
                              label + " size overflow");
  return left * right;
}

size_t checked_dft_output_sum(size_t left, size_t right,
                              const char *label) {
  if (left > std::numeric_limits<size_t>::max() - right)
    throw std::overflow_error(std::string("Meep CUDA DFT output ") +
                              label + " size overflow");
  return left + right;
}

bool append_cuda_dft_output_component_metadata(
    fields *parent, dft_chunk **chunklists, int num_chunklists, component c,
    size_t frequency_count, size_t *packed_point_count,
    cuda_dft_output_component_metadata *component_metadata) {
  if (!parent || !packed_point_count || !component_metadata)
    throw std::invalid_argument(
        "Meep CUDA DFT output metadata arguments must be non-null");

  ivec min_corner, max_corner;
  size_t array_size = 0;
  size_t maximum_chunk_size = 0;
  direction directions[3];
  int rank = 0;
  size_t dims[3] = {1, 1, 1};
  parent->get_dft_component_dims(
      chunklists, num_chunklists, c, min_corner, max_corner, array_size,
      maximum_chunk_size, rank, directions, dims);
  (void)maximum_chunk_size;
  if (rank == 0) return false;

  component_metadata->c = c;
  component_metadata->rank = rank;
  component_metadata->global_point_count = array_size;
  for (int dim = 0; dim < 3; ++dim)
    component_metadata->dims[dim] = dims[dim];

  for (int chunklist = 0; chunklist < num_chunklists; ++chunklist)
    for (dft_chunk *chunk = chunklists[chunklist]; chunk;
         chunk = chunk->next_in_dft) {
      if (chunk->c != c) continue;
      if (chunk->N == 0)
        throw std::logic_error(
            "Meep CUDA DFT output encountered an empty local chunk");
      if (chunk->omega.size() != frequency_count)
        throw std::runtime_error(
            "Meep CUDA DFT output frequency count differs between chunks");
      if (chunk->stored_weight == complex<double>(0.0, 0.0))
        throw std::runtime_error(
            "Meep CUDA DFT output encountered a zero stored weight");

      cuda_dft_output_chunk_metadata entry;
      entry.chunk = chunk;
      entry.output_point_offset = *packed_point_count;
      for (int dim = 0; dim < 3; ++dim) {
        entry.start[dim] = 0;
        entry.count[dim] = 1;
      }

      std::ptrdiff_t file_offset[3] = {0, 0, 0};
      std::ptrdiff_t file_stride[3] = {1, 1, 1};
      const ivec transformed_start =
          chunk->S.transform(chunk->is, chunk->sn) + chunk->shift;
      const ivec transformed_end =
          chunk->S.transform(chunk->ie, chunk->sn) + chunk->shift;
      ivec permute(zero_ivec(chunk->fc->gv.dim));
      for (int dim = 0; dim < 3; ++dim)
        permute.set_direction(chunk->fc->gv.yucky_direction(dim), dim);
      permute = chunk->S.transform_unshifted(permute, chunk->sn);
      LOOP_OVER_DIRECTIONS(permute.dim, d) {
        permute.set_direction(d, abs(permute.in_direction(d)));
      }

      for (int dim = 0; dim < rank; ++dim) {
        const direction d = directions[dim];
        const int start_coordinate = transformed_start.in_direction(d);
        const int end_coordinate = transformed_end.in_direction(d);
        entry.start[dim] =
            (std::min(start_coordinate, end_coordinate) -
             min_corner.in_direction(d)) /
            2;
        entry.count[dim] =
            static_cast<size_t>(abs(end_coordinate - start_coordinate) /
                                    2 +
                                1);
        const int storage_dimension = permute.in_direction(d);
        // Legacy process_dft_component uses the transformed yucky-grid slot
        // in fixed three-entry offset/stride arrays.  In 2D a valid active
        // output direction can map to slot 2 even though rank is only 2.
        if (storage_dimension < 0 || storage_dimension >= 3)
          throw std::out_of_range(
              "Meep CUDA DFT output permutation is outside its storage slots");
        if (end_coordinate < start_coordinate)
          file_offset[storage_dimension] =
              static_cast<std::ptrdiff_t>(entry.count[dim] - 1);
      }

      for (int dim = 0; dim < rank; ++dim) {
        const int storage_dimension =
            permute.in_direction(directions[dim]);
        for (int following = dim + 1; following < rank; ++following) {
          if (entry.count[following] >
              static_cast<size_t>(
                  std::numeric_limits<std::ptrdiff_t>::max() /
                  file_stride[storage_dimension]))
            throw std::overflow_error(
                "Meep CUDA DFT output file stride overflow");
          file_stride[storage_dimension] *=
              static_cast<std::ptrdiff_t>(entry.count[following]);
        }
        file_offset[storage_dimension] *= file_stride[storage_dimension];
        if (file_offset[storage_dimension])
          file_stride[storage_dimension] *= -1;
      }

      size_t chunk_point_count = 1;
      for (int dim = 0; dim < rank; ++dim)
        chunk_point_count = checked_dft_output_product(
            chunk_point_count, entry.count[dim], "chunk point");
      if (chunk_point_count != chunk->N)
        throw std::logic_error(
            "Meep CUDA DFT output chunk extent differs from DFT storage");

      entry.destination_indices.reserve(chunk->N);
      entry.point_weights.reserve(chunk->N);
      entry.zero_divisor_flags.reserve(chunk->N);
      LOOP_OVER_IVECS(chunk->fc->gv, chunk->is, chunk->ie, idx) {
        const std::ptrdiff_t destination =
            file_offset[0] + file_offset[1] + file_offset[2] +
            static_cast<std::ptrdiff_t>(loop_i1) * file_stride[0] +
            static_cast<std::ptrdiff_t>(loop_i2) * file_stride[1] +
            static_cast<std::ptrdiff_t>(loop_i3) * file_stride[2];
        if (destination < 0 ||
            static_cast<size_t>(destination) >= chunk_point_count)
          throw std::out_of_range(
              "Meep CUDA DFT output destination is outside its packed chunk");

        double publication_weight = 1.0;
        std::uint8_t zero_divisor = 0u;
        if (chunk->include_dV_and_interp_weights) {
          const double stored_point_weight = IVEC_LOOP_WEIGHT(
              chunk->s0, chunk->s1, chunk->e0, chunk->e1,
              chunk->dV0 + chunk->dV1 * loop_i2);
          const double divisor = chunk->sqrt_dV_and_interp_weights
                                     ? std::sqrt(stored_point_weight)
                                     : stored_point_weight;
          if (!std::isfinite(divisor))
            throw std::runtime_error(
                "Meep CUDA DFT output encountered a non-finite stored "
                "point weight");
          zero_divisor = divisor == 0.0 ? 1u : 0u;
          if (divisor != 0.0) publication_weight /= divisor;
        }
        const float converted_weight =
            static_cast<float>(publication_weight);
        if (!std::isfinite(converted_weight))
          throw std::runtime_error(
              "Meep CUDA DFT output point weight is outside the FP32 range");
        entry.destination_indices.push_back(destination);
        entry.point_weights.push_back(converted_weight);
        entry.zero_divisor_flags.push_back(zero_divisor);
      }
      if (entry.destination_indices.size() != chunk->N ||
          entry.point_weights.size() != chunk->N ||
          entry.zero_divisor_flags.size() != chunk->N)
        throw std::logic_error(
            "Meep CUDA DFT output metadata size differs from its chunk");

      *packed_point_count = checked_dft_output_sum(
          *packed_point_count, chunk_point_count, "packed point");
      component_metadata->chunks.push_back(std::move(entry));
    }
  return true;
}

void output_dft_components_cuda_nonempty(
    fields *parent, dft_chunk **chunklists, int num_chunklists,
    size_t frequency_count, const char *parallel_output_name) {
  if (!parent || !parallel_output_name)
    throw std::invalid_argument(
        "Meep CUDA DFT output requires a parent and filename");
  static_assert(sizeof(realnum) == sizeof(float),
                "CUDA DFT output requires an FP32 Meep build");

  // Publish only a completely written HDF5 file.  Keeping the temporary file
  // beside the destination makes the final POSIX rename atomic and also
  // preserves an existing user file if a later CUDA tile or HDF5 operation
  // fails.  Concurrent writers to the same Meep output name are unsupported;
  // a stale temporary file from an interrupted run is intentionally
  // truncated by the first WRITE below.
  const std::string temporary_output_name =
      std::string(parallel_output_name) + ".gpmeep-tmp";
  const std::string temporary_output_path =
      parent->h5file_name(temporary_output_name.c_str());
  const std::string final_output_path =
      parent->h5file_name(parallel_output_name);

  distributed_dft_output_failure_guard failure_guard;
  std::vector<cuda_dft_output_component_metadata> components;
  size_t local_packed_point_count = 0;
  size_t global_output_point_terms = 0;
  FOR_COMPONENTS(c) {
    cuda_dft_output_component_metadata metadata;
    if (append_cuda_dft_output_component_metadata(
            parent, chunklists, num_chunklists, c, frequency_count,
            &local_packed_point_count, &metadata)) {
      global_output_point_terms = checked_dft_output_sum(
          global_output_point_terms,
          checked_dft_output_product(metadata.global_point_count,
                                     frequency_count,
                                     "public point-frequency"),
          "public point-frequency");
      components.push_back(std::move(metadata));
    }
  }

  // Every rank executes the same component-dimension collectives above.  A
  // monitor that does not overlap this rank therefore still learns the exact
  // global dataset inventory before any HDF5 call.
  if (components.empty() || frequency_count == 0) {
    gpu::detail::record_cuda_dft_output_call();
    failure_guard.dismiss();
    return;
  }

  constexpr size_t exact_fp64_integer_limit =
      static_cast<size_t>(1) << 53;
  if (local_packed_point_count > exact_fp64_integer_limit)
    throw std::overflow_error(
        "Meep CUDA DFT output rank-local point count exceeds the exact "
        "collective range");
  parent->am_now_working_on(MpiAllTime);
  const double global_packed_as_double =
      max_to_all(static_cast<double>(local_packed_point_count));
  parent->finished_working();
  if (!std::isfinite(global_packed_as_double) ||
      global_packed_as_double < 1.0 ||
      global_packed_as_double >
          static_cast<double>(exact_fp64_integer_limit))
    throw std::runtime_error(
        "Meep CUDA DFT output has an invalid global packed point count");
  const size_t global_maximum_packed_points =
      static_cast<size_t>(global_packed_as_double);

  // Bound each retained planar buffer (one device output plus two pinned-host
  // publication slots) to a portable 64 MiB target whenever a single
  // frequency plane fits. All
  // ranks use the global maximum point count, hence choose identical tile
  // counts even when some ranks own no monitor chunks.
  constexpr size_t target_workspace_bytes =
      static_cast<size_t>(64) * 1024 * 1024;
  const size_t bytes_per_global_frequency = checked_dft_output_product(
      checked_dft_output_product(global_maximum_packed_points,
                                 static_cast<size_t>(2),
                                 "global planar scalar"),
      sizeof(float), "global planar byte");
  size_t frequency_capacity =
      bytes_per_global_frequency
          ? target_workspace_bytes / bytes_per_global_frequency
          : frequency_count;
  frequency_capacity = std::max<size_t>(1, frequency_capacity);
  frequency_capacity = std::min(frequency_capacity, frequency_count);

  std::vector<gpu::detail::dft_output_staging_request_fp32> requests;
  size_t local_request_count = 0;
  for (const cuda_dft_output_component_metadata &metadata : components)
    local_request_count = checked_dft_output_sum(
        local_request_count, metadata.chunks.size(), "local request");
  requests.reserve(local_request_count);
  for (const cuda_dft_output_component_metadata &metadata : components)
    for (const cuda_dft_output_chunk_metadata &entry : metadata.chunks) {
      const complex<double> inverse_weight =
          1.0 / entry.chunk->stored_weight;
      const float inverse_real =
          static_cast<float>(inverse_weight.real());
      const float inverse_imaginary =
          static_cast<float>(inverse_weight.imag());
      if (!std::isfinite(inverse_real) ||
          !std::isfinite(inverse_imaginary))
        throw std::runtime_error(
            "Meep CUDA DFT output inverse stored weight is outside the "
            "FP32 range");
      requests.push_back(
          {entry.chunk->fc,
           reinterpret_cast<const float *>(entry.chunk->dft),
           entry.destination_indices.data(), entry.point_weights.data(),
           entry.zero_divisor_flags.data(), entry.chunk->N, entry.chunk->N,
           entry.chunk->omega.size(), inverse_real, inverse_imaginary,
           entry.output_point_offset});
    }
  if ((local_packed_point_count == 0) != requests.empty())
    throw std::logic_error(
        "Meep CUDA DFT output request inventory differs from packed points");

  bool first_dataset = true;
  for (size_t frequency_start = 0; frequency_start < frequency_count;
       frequency_start += frequency_capacity) {
    const size_t selected_frequency_count =
        std::min(frequency_capacity, frequency_count - frequency_start);
    gpu::detail::resident_dft_output_staging_view_fp32 view = {
        nullptr, local_packed_point_count, frequency_start,
        selected_frequency_count, frequency_capacity};
    if (!requests.empty())
      view = gpu::detail::resident_stage_dft_output_fp32(
          requests.data(), requests.size(), local_packed_point_count,
          frequency_start, selected_frequency_count, frequency_capacity);
    if (view.output_point_count != local_packed_point_count ||
        view.frequency_start != frequency_start ||
        view.frequency_count != selected_frequency_count ||
        view.frequency_capacity != frequency_capacity ||
        (!requests.empty() && !view.planar))
      throw std::logic_error(
          "Meep CUDA DFT output backend returned an inconsistent view");

    // In particular, the first stage and D2H above complete before the first
    // WRITE open can truncate an existing user file.
    for (size_t tile_frequency = 0;
         tile_frequency < selected_frequency_count; ++tile_frequency) {
      const size_t public_frequency = frequency_start + tile_frequency;
      for (const cuda_dft_output_component_metadata &metadata : components)
        for (int reim = 0; reim < 2; ++reim) {
          std::unique_ptr<h5file> file(parent->open_h5file(
              temporary_output_name.c_str(),
              first_dataset ? h5file::WRITE : h5file::READWRITE));
          first_dataset = false;
          char dataname[100];
          snprintf(dataname, sizeof(dataname), "%s_%i.%c",
                   component_name(metadata.c),
                   static_cast<int>(public_frequency),
                   reim ? 'i' : 'r');
          file->create_or_extend_data(
              dataname, metadata.rank, metadata.dims,
              false /* append_data */, true /* single_precision */);
          for (const cuda_dft_output_chunk_metadata &entry :
               metadata.chunks) {
            const size_t plane_offset = checked_dft_output_sum(
                checked_dft_output_product(
                    checked_dft_output_sum(
                        checked_dft_output_product(tile_frequency,
                                                   static_cast<size_t>(2),
                                                   "tile plane"),
                        static_cast<size_t>(reim), "tile plane"),
                    local_packed_point_count, "tile plane"),
                entry.output_point_offset, "tile chunk offset");
            file->write_chunk(
                metadata.rank, entry.start, entry.count,
                const_cast<float *>(view.planar + plane_offset));
          }
          file->done_writing_chunks();
          file->prevent_deadlock();
          file.reset();
          const bool injected_failure =
              gpu::detail::
                  consume_cuda_dft_output_dataset_failure_for_testing();
          if (injected_failure)
            throw std::runtime_error(
                "injected CUDA DFT output failure after an HDF5 dataset");
        }
    }
  }

  // Every HDF5 handle is closed before publication.  Only rank zero renames
  // the shared file, then all ranks agree on the result before returning.
  all_wait();
  int publication_failed = 0;
  if (am_master() &&
      rename(temporary_output_path.c_str(), final_output_path.c_str()) != 0)
    publication_failed = 1;
  parent->am_now_working_on(MpiAllTime);
  publication_failed = sum_to_all(publication_failed);
  parent->finished_working();
  if (publication_failed) {
    // The failure result was communicated collectively, so unwinding is now
    // symmetric and does not require the rank-local emergency abort guard.
    failure_guard.dismiss();
    throw std::runtime_error(
        "Meep CUDA DFT output could not atomically publish its HDF5 file");
  }
  all_wait();

  gpu::detail::record_cuda_dft_output_call();
  gpu::detail::record_cuda_dft_output_points(global_output_point_terms);
  failure_guard.dismiss();
}

struct cuda_collapsed_dft_output_component_metadata {
  component c;
  ivec min_corner;
  ivec max_corner;
  int full_rank;
  direction full_directions[3];
  size_t full_dims[3];
  size_t full_point_count;
  int reduced_rank;
  size_t reduced_dims[3];
  size_t reduced_point_count;
  bool collapse_active;
  bool requires_dense_collective;
  gpu::detail::dft_materialization_collapse_fp32 collapse;
};

void mark_cuda_dft_component_dense_occupancy(
    dft_chunk **chunklists, int num_chunklists, component c, int rank,
    const direction *directions, ivec min_corner, ivec max_corner,
    size_t dense_point_count, std::vector<std::uint8_t> *occupancy) {
  if (!occupancy)
    throw std::invalid_argument(
        "Meep CUDA collapsed DFT occupancy output must be non-null");
  occupancy->assign(dense_point_count, 0u);
  size_t array_count[3] = {1, 1, 1};
  size_t verified_point_count = 1;
  for (int dim = 0; dim < rank; ++dim) {
    const direction d = directions[dim];
    const int extent =
        (max_corner.in_direction(d) - min_corner.in_direction(d)) / 2 + 1;
    if (extent <= 0)
      throw std::logic_error(
          "Meep CUDA collapsed DFT occupancy has a non-positive extent");
    array_count[dim] = static_cast<size_t>(extent);
    verified_point_count = checked_dft_output_product(
        verified_point_count, array_count[dim], "occupancy point");
  }
  if (verified_point_count != dense_point_count)
    throw std::logic_error(
        "Meep CUDA collapsed DFT occupancy extent is inconsistent");

  for (int chunklist = 0; chunklist < num_chunklists; ++chunklist)
    for (dft_chunk *chunk = chunklists[chunklist]; chunk;
         chunk = chunk->next_in_dft) {
      if (chunk->c != c) continue;
      LOOP_OVER_IVECS(chunk->fc->gv, chunk->is, chunk->ie, idx) {
        IVEC_LOOP_ILOC(chunk->fc->gv, iloc);
        iloc = chunk->S.transform(iloc, chunk->sn) + chunk->shift;
        iloc -= min_corner;
        size_t destination = 0;
        size_t stride = 1;
        for (int dim = rank - 1; dim >= 0; --dim) {
          const ptrdiff_t coordinate =
              iloc.in_direction(directions[dim]) / 2;
          if (coordinate < 0 ||
              static_cast<size_t>(coordinate) >= array_count[dim])
            throw std::out_of_range(
                "Meep CUDA collapsed DFT occupancy coordinate is outside "
                "the dense array");
          destination = checked_dft_output_sum(
              destination,
              checked_dft_output_product(
                  static_cast<size_t>(coordinate), stride,
                  "occupancy destination"),
              "occupancy destination");
          if (dim > 0)
            stride = checked_dft_output_product(
                stride, array_count[dim], "occupancy stride");
        }
        if (destination >= dense_point_count)
          throw std::out_of_range(
              "Meep CUDA collapsed DFT occupancy destination is outside "
              "the dense array");
        (*occupancy)[destination] = 1u;
      }
    }
}

size_t collapsed_dft_reduced_index(
    const cuda_collapsed_dft_output_component_metadata &metadata,
    size_t dense_index) {
  if (dense_index >= metadata.full_point_count)
    throw std::out_of_range(
        "Meep CUDA collapsed DFT dense index is outside the component");
  size_t coordinates[3] = {0, 0, 0};
  size_t remainder = dense_index;
  for (int dim = metadata.full_rank - 1; dim >= 0; --dim) {
    coordinates[dim] = remainder % metadata.full_dims[dim];
    remainder /= metadata.full_dims[dim];
  }
  if (remainder != 0)
    throw std::logic_error(
        "Meep CUDA collapsed DFT dense index decoding failed");

  size_t reduced_index = 0;
  size_t reduced_stride = 1;
  for (int dim = metadata.full_rank - 1; dim >= 0; --dim)
    if (!metadata.collapse.collapsed[dim]) {
      reduced_index = checked_dft_output_sum(
          reduced_index,
          checked_dft_output_product(coordinates[dim], reduced_stride,
                                     "collapsed destination"),
          "collapsed destination");
      reduced_stride = checked_dft_output_product(
          reduced_stride, metadata.full_dims[dim],
          "collapsed destination stride");
    }
  if (reduced_index >= metadata.reduced_point_count)
    throw std::out_of_range(
        "Meep CUDA collapsed DFT reduced index is outside the component");
  return reduced_index;
}

bool collapsed_dft_requires_dense_collective(
    fields *parent, dft_chunk **chunklists, int num_chunklists,
    const cuda_collapsed_dft_output_component_metadata &metadata) {
  if (count_processors() <= 1 || !metadata.collapse_active) return false;

  std::vector<std::uint8_t> dense_occupancy;
  mark_cuda_dft_component_dense_occupancy(
      chunklists, num_chunklists, metadata.c, metadata.full_rank,
      metadata.full_directions, metadata.min_corner, metadata.max_corner,
      metadata.full_point_count, &dense_occupancy);
  std::vector<float> local_owners(metadata.reduced_point_count, 0.0f);
  std::vector<float> global_owners(metadata.reduced_point_count, 0.0f);
  for (size_t dense_index = 0; dense_index < dense_occupancy.size();
       ++dense_index)
    if (dense_occupancy[dense_index])
      local_owners[collapsed_dft_reduced_index(metadata, dense_index)] =
          1.0f;

  constexpr size_t mpi_batch_size = static_cast<size_t>(1) << 20;
  for (size_t offset = 0; offset < metadata.reduced_point_count;) {
    const size_t batch = std::min(
        mpi_batch_size, metadata.reduced_point_count - offset);
    parent->am_now_working_on(MpiAllTime);
    sum_to_all(local_owners.data() + offset,
               global_owners.data() + offset,
               static_cast<int>(batch));
    parent->finished_working();
    gpu::detail::record_dft_array_mpi_allreduce(
        batch * sizeof(float));
    offset += batch;
  }
  return std::any_of(global_owners.begin(), global_owners.end(),
                     [](float owners) { return owners > 1.0f; });
}

std::vector<complex<realnum>> collapse_dft_values_host(
    const std::vector<complex<realnum>> &dense_values,
    const cuda_collapsed_dft_output_component_metadata &metadata) {
  if (dense_values.size() != metadata.full_point_count)
    throw std::invalid_argument(
        "Meep CUDA collapsed DFT host input has the wrong size");
  std::vector<complex<realnum>> reduced_values(
      metadata.reduced_point_count, complex<realnum>(0.0, 0.0));
  for (size_t dense_index = 0; dense_index < dense_values.size();
       ++dense_index)
    reduced_values[collapsed_dft_reduced_index(metadata, dense_index)] +=
        dense_values[dense_index];
  return reduced_values;
}

void output_dft_components_cuda_collapsed(
    fields *parent, dft_chunk **chunklists, int num_chunklists,
    volume dft_volume, size_t frequency_count,
    const char *parallel_output_name) {
  if (!parent || !parallel_output_name)
    throw std::invalid_argument(
        "Meep CUDA collapsed DFT output requires a parent and filename");
  static_assert(sizeof(realnum) == sizeof(float),
                "CUDA collapsed DFT output requires an FP32 Meep build");

  const std::string temporary_output_name =
      std::string(parallel_output_name) + ".gpmeep-tmp";
  const std::string temporary_output_path =
      parent->h5file_name(temporary_output_name.c_str());
  const std::string final_output_path =
      parent->h5file_name(parallel_output_name);
  distributed_dft_output_failure_guard failure_guard;

  std::vector<cuda_collapsed_dft_output_component_metadata> components;
  size_t global_output_point_terms = 0;
  FOR_COMPONENTS(c) {
    cuda_collapsed_dft_output_component_metadata metadata;
    metadata.c = c;
    metadata.full_rank = 0;
    metadata.full_dims[0] = metadata.full_dims[1] =
        metadata.full_dims[2] = 1;
    size_t maximum_chunk_size = 0;
    parent->get_dft_component_dims(
        chunklists, num_chunklists, c, metadata.min_corner,
        metadata.max_corner, metadata.full_point_count,
        maximum_chunk_size, metadata.full_rank,
        metadata.full_directions, metadata.full_dims);
    (void)maximum_chunk_size;
    if (metadata.full_rank == 0) continue;

    size_t full_strides[3];
    size_t reduced_strides[3];
    direction reduced_directions[3];
    reduce_array_dimensions(
        dft_volume, metadata.full_rank, metadata.full_dims,
        metadata.full_directions, full_strides, metadata.reduced_rank,
        metadata.reduced_dims, reduced_directions, reduced_strides);
    metadata.reduced_point_count = checked_dft_output_product(
        checked_dft_output_product(metadata.reduced_dims[0],
                                   metadata.reduced_dims[1],
                                   "collapsed component point"),
        metadata.reduced_dims[2], "collapsed component point");
    metadata.collapse.full_rank =
        static_cast<size_t>(metadata.full_rank);
    metadata.collapse_active = false;
    metadata.requires_dense_collective = false;
    for (int dim = 0; dim < 3; ++dim) {
      metadata.collapse.full_dims[dim] = metadata.full_dims[dim];
      metadata.collapse.collapsed[dim] =
          dim < metadata.full_rank &&
                  dft_volume.in_direction(
                      metadata.full_directions[dim]) == 0.0
              ? 1u
              : 0u;
      metadata.collapse_active =
          metadata.collapse_active ||
          metadata.collapse.collapsed[dim] != 0;
    }
    metadata.requires_dense_collective =
        collapsed_dft_requires_dense_collective(
            parent, chunklists, num_chunklists, metadata);
    global_output_point_terms = checked_dft_output_sum(
        global_output_point_terms,
        checked_dft_output_product(metadata.reduced_point_count,
                                   frequency_count,
                                   "collapsed public point-frequency"),
        "collapsed public point-frequency");
    components.push_back(std::move(metadata));
  }

  // Match the historical CPU/public behavior for an empty monitor or a
  // zero-frequency monitor: the call succeeds, records no point work, and
  // does not create an output artifact.  h5file opens lazily, so attempting
  // to publish a never-written temporary file would otherwise turn this
  // no-op into a spurious ENOENT rename failure.
  if (components.empty() || frequency_count == 0) {
    gpu::detail::record_cuda_dft_output_call();
    failure_guard.dismiss();
    return;
  }

  const auto any_rank_failed = [parent](int local_failure) {
    parent->am_now_working_on(MpiAllTime);
    const int failures = sum_to_all(local_failure);
    parent->finished_working();
    return failures != 0;
  };

  std::unique_ptr<h5file> file;
  int open_failed = 0;
  if (am_master()) {
    try {
      file.reset(new h5file(temporary_output_path.c_str(), h5file::WRITE,
                            false /* parallel */));
      if (!file->ok()) {
        file.reset();
        open_failed = 1;
      }
    }
    catch (...) { open_failed = 1; }
  }
  if (any_rank_failed(open_failed)) {
    file.reset();
    failure_guard.dismiss();
    throw std::runtime_error(
        "Meep CUDA collapsed DFT output could not create its temporary "
        "HDF5 file");
  }

  for (size_t frequency = 0; frequency < frequency_count; ++frequency)
    for (const cuda_collapsed_dft_output_component_metadata &metadata :
         components) {
      const size_t collective_point_count =
          metadata.requires_dense_collective
              ? metadata.full_point_count
              : metadata.reduced_point_count;
      std::vector<complex<realnum>> local_values(
          collective_point_count);
      materialize_dft_component_array_cuda(
          chunklists, num_chunklists, metadata.c,
          static_cast<int>(frequency), metadata.full_rank,
          metadata.full_directions,
          metadata.min_corner, metadata.max_corner,
          metadata.full_point_count, local_values.data(),
          metadata.collapse_active &&
                  !metadata.requires_dense_collective
              ? &metadata.collapse
              : nullptr,
          collective_point_count);

      std::vector<complex<realnum>> global_values(
          collective_point_count);
      constexpr size_t mpi_batch_size = static_cast<size_t>(1) << 20;
      for (size_t offset = 0; offset < collective_point_count;) {
        const size_t batch = std::min(
            mpi_batch_size, collective_point_count - offset);
        parent->am_now_working_on(MpiAllTime);
        sum_to_all(local_values.data() + offset,
                   global_values.data() + offset, batch);
        parent->finished_working();
        gpu::detail::record_dft_array_mpi_allreduce(
            batch * sizeof(complex<realnum>));
        offset += batch;
      }
      if (metadata.requires_dense_collective)
        global_values = collapse_dft_values_host(global_values, metadata);

      int write_failed = 0;
      if (am_master()) {
        try {
          const int hdf5_rank =
              metadata.reduced_rank == 0 ? 1 : metadata.reduced_rank;
          size_t hdf5_dims[3] = {metadata.reduced_dims[0],
                                 metadata.reduced_dims[1],
                                 metadata.reduced_dims[2]};
          if (metadata.reduced_rank == 0) hdf5_dims[0] = 1;
          std::vector<double> plane(metadata.reduced_point_count);
          for (int reim = 0; reim < 2; ++reim) {
            for (size_t point = 0;
                 point < metadata.reduced_point_count; ++point)
              plane[point] = reim == 0
                                 ? real(global_values[point])
                                 : imag(global_values[point]);
            char dataname[100];
            snprintf(dataname, sizeof(dataname), "%s_%i.%c",
                     component_name(metadata.c),
                     static_cast<int>(frequency), reim ? 'i' : 'r');
            if (!file->write_serial_checked(
                    dataname, hdf5_rank, hdf5_dims, plane.data(),
                    false /* single_precision */))
              throw std::runtime_error(
                  "checked collapsed DFT HDF5 dataset write failed");
          }
        }
        catch (...) {
          write_failed = 1;
          file.reset();
        }
      }
      if (any_rank_failed(write_failed)) {
        file.reset();
        failure_guard.dismiss();
        throw std::runtime_error(
            "Meep CUDA collapsed DFT output failed while writing an HDF5 "
            "dataset");
      }

      const int injected_failure =
          gpu::detail::
              consume_cuda_dft_output_dataset_failure_for_testing()
              ? 1
              : 0;
      if (any_rank_failed(injected_failure)) {
        file.reset();
        failure_guard.dismiss();
        throw std::runtime_error(
            "injected CUDA collapsed DFT output failure after an HDF5 "
            "dataset");
      }
    }

  int finalization_failed = 0;
  if (am_master()) {
    try {
      if (!file || !file->flush_and_close_checked())
        finalization_failed = 1;
    }
    catch (...) { finalization_failed = 1; }
  }
  if (any_rank_failed(finalization_failed)) {
    file.reset();
    failure_guard.dismiss();
    throw std::runtime_error(
        "Meep CUDA collapsed DFT output failed while flushing or closing "
        "its HDF5 file");
  }
  file.reset();
  all_wait();
  int publication_failed = 0;
  if (am_master() &&
      rename(temporary_output_path.c_str(), final_output_path.c_str()) != 0)
    publication_failed = 1;
  if (any_rank_failed(publication_failed)) {
    failure_guard.dismiss();
    throw std::runtime_error(
        "Meep CUDA collapsed DFT output could not atomically publish its "
        "HDF5 file");
  }
  all_wait();

  gpu::detail::record_cuda_dft_output_call();
  gpu::detail::record_cuda_dft_output_points(global_output_point_terms);
  failure_guard.dismiss();
}

} // namespace
#endif

/***************************************************************/
/* low-level [actually intermediate-level, since it calls      */
/* dft_chunk::process_dft_component(), which is the true       */
/* low-level function] workhorse routine that forms the common */
/* backend for several operations involving DFT fields.        */
/*                                                             */
/* looks through the given collection of dft_chunks and        */
/* processes only those chunks that store component c, using   */
/* only data for frequency #num_freq.                          */
/*                                                             */
/* the meaning of 'processes' depends on the arguments:        */
/*                                                             */
/*  1. if HDF5FileName is non-null: write to the given HDF5    */
/*     file a new dataset describing either                    */
/*      (A) DFT field component c (if mode_data1 is null), or  */
/*      (B) mode field component c for the eigenmode described */
/*          by mode_data1 (if it is non-null)                  */
/*                                                             */
/*  2. if HDF5FileName is null but pfield_array is non-null:   */
/*     set *pfield_array equal to a newly allocated buffer     */
/*     populated on return with values of DFT field component  */
/*     c, equivalent to writing the data to HDF5 and reading   */
/*     it back into field_array.                               */
/*                                                             */
/*  3. if both HDF5FileName and field_array are null: compute  */
/*     and return an  overlap integral between                 */
/*      (A) the DFT fields and the fields of the eigenmode     */
/*          described by mode_data1 (if mode_data2 is null)    */
/*      (B) the eigenmode fields described by mode_data1       */
/*          and the eigenmode fields described by mode_data2   */
/*          (if mode_data2 is non-null).                       */
/*     more specifically, the integral computed is             */
/*      < mode1_{c_conjugate} | dft_{c} >                      */
/*     in case (A) and                                         */
/*      < mode1_{c_conjugate} | mode2_{c} >                    */
/*     in case (B).                                            */
/*                                                             */
/* if where is non-null, only field components inside *where   */
/* are processed.                                              */
/***************************************************************/
complex<double> fields::process_dft_component(dft_chunk **chunklists, int num_chunklists,
                                              int num_freq, component c, const char *HDF5FileName,
                                              complex<realnum> **pfield_array, int *array_rank,
                                              size_t *array_dims, direction *array_dirs,
                                              void *mode1_data, void *mode2_data,
                                              component c_conjugate, bool *first_component,
                                              bool retain_interp_weights) {

  /***************************************************************/
  /***************************************************************/
  /***************************************************************/
  int ic_conjugate = (int)c_conjugate;
  const bool material_component_request = component_index(c) == -1;
  if (component_index(c) == -1) {
    ic_conjugate = -((int)c);
    num_chunklists = 1;
    // A small distributed monitor can be absent on this rank.  Resolve the
    // physical Yee component collectively instead of dereferencing a null
    // local list and leaving peers blocked in get_dft_component_dims.
    const int local_component =
        chunklists && chunklists[0] ? static_cast<int>(chunklists[0]->c) : -1;
    const int global_component = max_to_all(local_component);
    const bool components_agree = and_to_all(
        local_component < 0 || local_component == global_component);
    if (!components_agree)
      meep::abort(
          "synthetic DFT material query has inconsistent physical "
          "components across MPI ranks");
    if (global_component < 0) {
      if (pfield_array) *pfield_array = 0;
      if (array_rank) *array_rank = 0;
      return 0.0;
    }
    c = static_cast<component>(global_component);
  }

  ivec min_corner, max_corner;
  int rank;
  direction ds[3];
  size_t array_size, bufsz, dims[3];
  get_dft_component_dims(chunklists, num_chunklists, c, min_corner, max_corner, array_size, bufsz,
                         rank, ds, dims, array_rank, array_dims, array_dirs);

  if (rank == 0) {
    if (pfield_array) *pfield_array = 0;
    return 0.0; // no chunks with the specified component on this processor
  }

  if (material_component_request && pfield_array && !HDF5FileName &&
      !mode1_data && !mode2_data) {
    distributed_dft_materialization_failure_guard failure_guard(
        count_processors() > 1);
    std::unique_ptr<complex<realnum>[]> material_array(
        array_size ? new complex<realnum>[array_size]() : nullptr);
    materialize_synthetic_component_array_host(
        this, chunklists, num_chunklists, c,
        component(ic_conjugate >= 0 ? ic_conjugate : -ic_conjugate),
        rank, ds, min_corner, max_corner, array_size,
        material_array.get());
    // Keep the historical final scalar collective in the same position for
    // every rank; per-point material queries above are now also collective
    // on every rank in identical dense-index order.
    am_now_working_on(MpiAllTime);
    (void)sum_to_all(complex<double>(0.0, 0.0));
    finished_working();
    failure_guard.dismiss();
    *pfield_array = material_array.release();
    return 0.0;
  }

  // get_dft_array on an active CUDA backend is a strict resident operation.
  // It stages only the selected dense result and never publishes the owning
  // fields_chunk cache or traverses the DFT allocation on the CPU. HDF5 and
  // eigenmode paths remain separate operations with their own qualification.
  if (!material_component_request && pfield_array && !HDF5FileName &&
      !mode1_data && !mode2_data &&
      gpu::detail::cuda_active()) {
    if (sizeof(realnum) != sizeof(float))
      throw std::runtime_error(
          "Meep CUDA get_dft_array requires a single-precision build");
#if MEEP_HAVE_CUDA && MEEP_SINGLE
    distributed_dft_materialization_failure_guard failure_guard(
        count_processors() > 1);
    std::unique_ptr<complex<realnum>[]> cuda_field_array(
        array_size ? new complex<realnum>[array_size] : nullptr);
    materialize_dft_component_array_cuda(
        chunklists, num_chunklists, c, num_freq, rank, ds, min_corner,
        max_corner, array_size, cuda_field_array.get());

    constexpr size_t mpi_batch_size = static_cast<size_t>(1) << 20;
    std::unique_ptr<complex<realnum>[]> mpi_buffer(
        new complex<realnum>[mpi_batch_size]);
    size_t offset = 0;
    while (offset < array_size) {
      const size_t batch =
          std::min(mpi_batch_size, array_size - offset);
      am_now_working_on(MpiAllTime);
      sum_to_all(cuda_field_array.get() + offset, mpi_buffer.get(), batch);
      finished_working();
      gpu::detail::record_dft_array_mpi_allreduce(
          batch * sizeof(complex<realnum>));
      memcpy(cuda_field_array.get() + offset, mpi_buffer.get(),
             batch * sizeof(complex<realnum>));
      offset += batch;
    }
    // Preserve the historical collective sequence of
    // process_dft_component, whose final zero overlap reduction also occurs
    // for get_dft_array calls.
    am_now_working_on(MpiAllTime);
    (void)sum_to_all(complex<double>(0.0, 0.0));
    finished_working();
    failure_guard.dismiss();
    *pfield_array = cuda_field_array.release();
    return 0.0;
#else
    throw std::runtime_error(
        "Meep CUDA get_dft_array is unavailable in this build");
#endif
  }
  if (HDF5FileName)
    gpu::detail::record_cpu_dft_output_points(array_size);
  else if (!pfield_array && mode1_data)
    gpu::detail::record_cpu_dft_overlap_terms(array_size);

  /***************************************************************/
  /* buffer for process-local contributions to HDF5 output files,*/
  /* like h5_output_data::buf in h5fields.cpp                    */
  /***************************************************************/
  realnum *buffer = 0;
  complex<realnum> *field_array = 0;
  int reim_max = 0;
  if (HDF5FileName) {
    buffer = new realnum[bufsz];
    reim_max = 1;
  }
  else if (pfield_array)
    *pfield_array = field_array =
        (array_size ? new complex<realnum>[array_size]() : 0);

  complex<double> overlap = 0.0;
  for (int reim = 0; reim <= reim_max; reim++) {
    h5file *file = 0;
    if (HDF5FileName) {
      file = open_h5file(HDF5FileName, (*first_component) ? h5file::WRITE : h5file::READWRITE);
      *first_component = false;
      char dataname[100];
      snprintf(dataname, 100, "%s_%i.%c", component_name(c), num_freq, reim ? 'i' : 'r');
      file->create_or_extend_data(dataname, rank, dims, false /* append_data */,
                                  sizeof(realnum) == sizeof(float) /* single_precision */);
    }

    for (int ncl = 0; ncl < num_chunklists; ncl++)
      for (dft_chunk *chunk = chunklists[ncl]; chunk; chunk = chunk->next_in_dft)
        if (chunk->c == c)
          overlap += chunk->process_dft_component(rank, ds, min_corner, max_corner, num_freq, file,
                                                  buffer, reim, field_array, mode1_data, mode2_data,
                                                  ic_conjugate, retain_interp_weights, this);

    if (HDF5FileName) {
      file->done_writing_chunks();
      file->prevent_deadlock(); // hackery
      delete file;
    }
    else if (field_array) {
/***************************************************************/
/* repeatedly call sum_to_all to consolidate full field array  */
/* on all cores                                                */
/***************************************************************/
#define BUFSIZE 1 << 20 // use 1M element (16 MB) buffer
      complex<realnum> *buf = new complex<realnum>[BUFSIZE];
      ptrdiff_t offset = 0;
      size_t remaining = array_size;
      while (remaining != 0) {
        size_t size = (remaining > BUFSIZE ? BUFSIZE : remaining);
        am_now_working_on(MpiAllTime);
        sum_to_all(field_array + offset, buf, size);
        finished_working();
        gpu::detail::record_dft_array_mpi_allreduce(
            size * sizeof(complex<realnum>));
        memcpy(field_array + offset, buf, size * sizeof(complex<realnum>));
        remaining -= size;
        offset += size;
      }
      delete[] buf;
    }
  } // for(int reim=0; reim<=reim_max; reim++)

  if (HDF5FileName)
    delete[] buffer;
  else {
    am_now_working_on(MpiAllTime);
    overlap = sum_to_all(overlap);
    finished_working();
  }

  return overlap;
}

/***************************************************************/
/* routines for fetching arrays of dft fields                  */
/***************************************************************/
complex<realnum> *fields::get_dft_array(dft_flux flux, component c, int num_freq, int *rank,
                                        size_t dims[3]) {
  dft_chunk *chunklists[2];
  chunklists[0] = flux.E;
  chunklists[1] = flux.H;
  complex<realnum> *array;
  direction dirs[3];
  process_dft_component(chunklists, 2, num_freq, c, 0, &array, rank, dims, dirs);
  array = collapse_array(array, rank, dims, dirs, flux.where);
  const size_t points = array
                            ? (*rank > 0
                                   ? dims[0] * (*rank >= 2 ? dims[1] : 1) *
                                         (*rank == 3 ? dims[2] : 1)
                                   : 1)
                            : 0;
  if (gpu::detail::cuda_active()) {
    if (c == Dielectric || c == Permeability)
      gpu::detail::record_host_synthetic_material_array(points);
    else
      gpu::detail::record_cuda_dft_array_materialization(points);
  }
  else
    gpu::detail::record_cpu_dft_array_materialization(points);
  return array;
}

complex<realnum> *fields::get_dft_array(dft_force force, component c, int num_freq, int *rank,
                                        size_t dims[3]) {
  dft_chunk *chunklists[3];
  chunklists[0] = force.offdiag1;
  chunklists[1] = force.offdiag2;
  chunklists[2] = force.diag;
  complex<realnum> *array;
  direction dirs[3];
  process_dft_component(chunklists, 3, num_freq, c, 0, &array, rank, dims, dirs);
  array = collapse_array(array, rank, dims, dirs, force.where);
  const size_t points = array
                            ? (*rank > 0
                                   ? dims[0] * (*rank >= 2 ? dims[1] : 1) *
                                         (*rank == 3 ? dims[2] : 1)
                                   : 1)
                            : 0;
  if (gpu::detail::cuda_active()) {
    if (c == Dielectric || c == Permeability)
      gpu::detail::record_host_synthetic_material_array(points);
    else
      gpu::detail::record_cuda_dft_array_materialization(points);
  }
  else
    gpu::detail::record_cpu_dft_array_materialization(points);
  return array;
}

complex<realnum> *fields::get_dft_array(dft_near2far n2f, component c, int num_freq, int *rank,
                                        size_t dims[3]) {
  dft_chunk *chunklists[1];
  chunklists[0] = n2f.F;
  complex<realnum> *array;
  direction dirs[3];
  process_dft_component(chunklists, 1, num_freq, c, 0, &array, rank, dims, dirs);
  array = collapse_array(array, rank, dims, dirs, n2f.where);
  const size_t points = array
                            ? (*rank > 0
                                   ? dims[0] * (*rank >= 2 ? dims[1] : 1) *
                                         (*rank == 3 ? dims[2] : 1)
                                   : 1)
                            : 0;
  if (gpu::detail::cuda_active()) {
    if (c == Dielectric || c == Permeability)
      gpu::detail::record_host_synthetic_material_array(points);
    else
      gpu::detail::record_cuda_dft_array_materialization(points);
  }
  else
    gpu::detail::record_cpu_dft_array_materialization(points);
  return array;
}

complex<realnum> *fields::get_dft_array(dft_fields fdft, component c, int num_freq, int *rank,
                                        size_t dims[3]) {
  dft_chunk *chunklists[1];
  chunklists[0] = fdft.chunks;
  complex<realnum> *array;
  direction dirs[3];
  process_dft_component(chunklists, 1, num_freq, c, 0, &array, rank, dims, dirs);
  array = collapse_array(array, rank, dims, dirs, fdft.where);
  const size_t points = array
                            ? (*rank > 0
                                   ? dims[0] * (*rank >= 2 ? dims[1] : 1) *
                                         (*rank == 3 ? dims[2] : 1)
                                   : 1)
                            : 0;
  if (gpu::detail::cuda_active()) {
    if (c == Dielectric || c == Permeability)
      gpu::detail::record_host_synthetic_material_array(points);
    else
      gpu::detail::record_cuda_dft_array_materialization(points);
  }
  else
    gpu::detail::record_cpu_dft_array_materialization(points);
  return array;
}

/***************************************************************/
/* wrapper around process_dft_component that writes HDF5       */
/* datasets for all components at all frequencies stored in    */
/* the given collection of DFT chunks                          */
/***************************************************************/
void fields::output_dft_components(dft_chunk **chunklists, int num_chunklists, volume dft_volume,
                                   const char *HDF5FileName) {
  int NumFreqs = 0;
  for (int nc = 0; nc < num_chunklists && NumFreqs == 0; nc++)
    if (chunklists[nc]) NumFreqs = chunklists[nc]->omega.size();

  // If the volume has zero thickness in one or more directions, the DFT
  // grid is two pixels thick in those directions, but we want the HDF5 output
  // to be just one pixel thick in those directions. solution: first get the
  // fields in array form (as get_dft_array), then collapse degenerate dimensions
  // and export the collapsed array to HDF5.  Every geometry, not only an
  // empty-dimension geometry, must resolve NumFreqs collectively: a small
  // ordinary volume can live wholly on one MPI rank, and a rank-local loop
  // count would leave that owner inside HDF5 collectives while its peers skip
  // them.
  bool have_empty_dims = false;
  LOOP_OVER_DIRECTIONS(dft_volume.dim, d) {
    if (dft_volume.in_direction(d) == 0.0) have_empty_dims = true;
  }

  const std::string output_filename =
      std::string(HDF5FileName) +
      (has_hdf5_suffix(HDF5FileName) ? "" : ".h5");
  std::string parallel_output_name(HDF5FileName);
  if (has_hdf5_suffix(HDF5FileName))
    parallel_output_name.erase(parallel_output_name.size() - 3);

  am_now_working_on(MpiAllTime);
  NumFreqs = max_to_all(NumFreqs);
  finished_working();

  if (gpu::detail::cuda_active()) {
    if (sizeof(realnum) != sizeof(float))
      throw std::runtime_error(
          "Meep CUDA output_dft requires a single-precision build");
#if MEEP_HAVE_CUDA && MEEP_SINGLE
    if (have_empty_dims)
      output_dft_components_cuda_collapsed(
          this, chunklists, num_chunklists, dft_volume,
          static_cast<size_t>(NumFreqs), parallel_output_name.c_str());
    else
      output_dft_components_cuda_nonempty(
          this, chunklists, num_chunklists, static_cast<size_t>(NumFreqs),
          parallel_output_name.c_str());
    return;
#else
    throw std::runtime_error(
        "Meep CUDA output_dft is unavailable in this build");
#endif
  }

  gpu::detail::record_cpu_dft_output_call();

  h5file *file = 0;
  if (have_empty_dims && am_master())
    file = new h5file(output_filename.c_str(), h5file::WRITE,
                      false /*parallel*/);

  bool first_component = true;
  for (int num_freq = 0; num_freq < NumFreqs; num_freq++)
    FOR_COMPONENTS(c) {
      if (!have_empty_dims) {
        process_dft_component(chunklists, num_chunklists, num_freq, c,
                              parallel_output_name.c_str(), 0, 0, 0, 0,
                              0, 0, Ex, &first_component);
      }
      else {
        complex<realnum> *array = 0;
        int rank;
        size_t dims[3];
        direction dirs[3];
        process_dft_component(chunklists, num_chunklists, num_freq, c, 0, &array, &rank, dims,
                              dirs);
        if (rank > 0 && am_master()) {
          array = collapse_array(array, &rank, dims, dirs, dft_volume);
          if (rank == 0) {
            // h5file's scalar-dataset interface is represented as one
            // one-element dimension, matching Python get_dft_array's rank-0
            // scalar value without dropping point monitors.
            rank = 1;
            dims[0] = 1;
          }
          size_t array_size = dims[0] * (rank >= 2 ? dims[1] * (rank == 3 ? dims[2] : 1) : 1);
          gpu::detail::record_cpu_dft_output_points(array_size);
          double *real_array = new double[array_size];
          if (!real_array) meep::abort("%s:%i:out of memory(%lu)", __FILE__, __LINE__, array_size);
          for (int reim = 0; reim < 2; reim++) {
            for (size_t n = 0; n < array_size; n++)
              real_array[n] = (reim == 0 ? real(array[n]) : imag(array[n]));
            char dataname[100];
            snprintf(dataname, 100, "%s_%i.%c", component_name(c), num_freq, reim ? 'i' : 'r');
            file->write(dataname, rank, dims, real_array, false /* single_precision */);
          }
          delete[] real_array;
        }
        if (array) delete[] array;
      }
    }
  if (file) delete file;
}

void fields::output_dft(dft_flux flux, const char *HDF5FileName) {
  dft_chunk *chunklists[2];
  chunklists[0] = flux.E;
  chunklists[1] = flux.H;
  output_dft_components(chunklists, 2, flux.where, HDF5FileName);
}

void fields::output_dft(dft_force force, const char *HDF5FileName) {
  dft_chunk *chunklists[3];
  chunklists[0] = force.offdiag1;
  chunklists[1] = force.offdiag2;
  chunklists[2] = force.diag;
  output_dft_components(chunklists, 3, force.where, HDF5FileName);
}

void fields::output_dft(dft_near2far n2f, const char *HDF5FileName) {
  dft_chunk *chunklists[1];
  chunklists[0] = n2f.F;
  output_dft_components(chunklists, 1, n2f.where, HDF5FileName);
}

void fields::output_dft(dft_fields fdft, const char *HDF5FileName) {
  dft_chunk *chunklists[1];
  chunklists[0] = fdft.chunks;
  output_dft_components(chunklists, 1, fdft.where, HDF5FileName);
}

/***************************************************************/
/***************************************************************/
/***************************************************************/
void fields::get_overlap(void *mode1_data, void *mode2_data, dft_flux flux, int num_freq,
                         complex<double> overlaps[2]) {
  component cE[2], cH[2];
  switch (flux.normal_direction) {
    case X:
      cE[0] = Ey;
      cH[0] = Hz;
      cE[1] = Ez;
      cH[1] = Hy;
      break;
    case Y:
      cE[0] = Ez;
      cH[0] = Hx;
      cE[1] = Ex;
      cH[1] = Hz;
      break;
    case R:
      cE[0] = Ep;
      cH[0] = Hz;
      cE[1] = Ez;
      cH[1] = Hp;
      break;
    case P:
      cE[0] = Ez;
      cH[0] = Hr;
      cE[1] = Er;
      cH[1] = Hz;
      break;
    case Z:
      if (gv.dim == Dcyl)
        cE[0] = Er, cE[1] = Ep, cH[0] = Hp, cH[1] = Hr;
      else
        cE[0] = Ex, cE[1] = Ey, cH[0] = Hy, cH[1] = Hx;
      break;
    default: meep::abort("invalid normal_direction in get_overlap");
  };

  dft_chunk *chunklists[2] = {flux.E, flux.H};
  const component channel_components[4] = {
      cE[0], cE[1], cH[0], cH[1]};
  const eigenmode_overlap_consensus consensus =
      resolve_eigenmode_overlap_consensus(
          this, gv, v, chunklists, channel_components);
  gpu::detail::record_eigenmode_zero_rank_channels_skipped(
      static_cast<std::size_t>(std::count(
          consensus.channel_has_extent,
          consensus.channel_has_extent +
              eigenmode_overlap_component_count,
          false)));
  if (consensus.every_rank_cuda)
    gpu::detail::record_eigenmode_mpi_allreduce(
        eigenmode_overlap_consensus_value_count * sizeof(int));

  if (consensus.every_rank_cuda) {
    if (sizeof(realnum) != sizeof(float))
      throw std::runtime_error(
          "Meep CUDA eigenmode overlap requires a single-precision "
          "build");
#if MEEP_HAVE_CUDA && MEEP_SINGLE
    if (!mode1_data)
      throw std::invalid_argument(
          "Meep CUDA eigenmode overlap requires a mode-1 profile");
    const bool mode_mode = mode2_data != nullptr;
    distributed_eigenmode_overlap_failure_guard failure_guard;

    struct sampled_request {
      const dft_chunk *chunk = nullptr;
      std::uint32_t output_index = 0;
      std::vector<std::complex<double> > weighted_mode1;
      std::vector<std::complex<double> > mode2;
      std::vector<std::uint8_t> zero_normalization_divisors;
    };
    std::vector<sampled_request> sampled;
    std::size_t sampled_profile_points = 0;
    const auto append_component =
        [&](component c, component c_conjugate,
            std::uint32_t output_index) {
      for (int list_index = 0; list_index < 2; ++list_index)
        for (dft_chunk *chunk = chunklists[list_index]; chunk;
             chunk = chunk->next_in_dft) {
          if (chunk->c != c) continue;
          if (!chunk->fc || chunk->N == 0)
            throw std::invalid_argument(
                "Meep CUDA eigenmode overlap encountered an empty DFT "
                "chunk");
          if (!mode_mode) {
            if (!chunk->dft || chunk->omega.empty() || num_freq < 0 ||
                static_cast<std::size_t>(num_freq) >=
                    chunk->omega.size())
              throw std::out_of_range(
                  "Meep CUDA eigenmode frequency is outside DFT storage");
            if (chunk->stored_weight == std::complex<double>(0.0, 0.0) ||
                !std::isfinite(chunk->stored_weight.real()) ||
                !std::isfinite(chunk->stored_weight.imag()))
              throw std::invalid_argument(
                  "Meep CUDA eigenmode stored weight must be finite and "
                  "nonzero");
          }

          sampled_request request;
          request.chunk = chunk;
          request.output_index = output_index;
          request.weighted_mode1.reserve(chunk->N);
          if (mode_mode) request.mode2.reserve(chunk->N);
          if (!mode_mode && chunk->include_dV_and_interp_weights)
            request.zero_normalization_divisors.reserve(chunk->N);
          vec rshift(chunk->shift * (0.5 * chunk->fc->gv.inva));
          std::size_t point_count = 0;
          LOOP_OVER_IVECS(chunk->fc->gv, chunk->is, chunk->ie, idx) {
            IVEC_LOOP_LOC(chunk->fc->gv, loc);
            loc = chunk->S.transform(loc, chunk->sn) + rshift;
            const double quadrature_weight = IVEC_LOOP_WEIGHT(
                chunk->s0, chunk->s1, chunk->e0, chunk->e1,
                chunk->dV0 + chunk->dV1 * loop_i2);
            double profile_weight = quadrature_weight;
            if (!mode_mode && chunk->include_dV_and_interp_weights) {
              const bool zero_divisor = quadrature_weight == 0.0;
              request.zero_normalization_divisors.push_back(
                  zero_divisor ? 1u : 0u);
              profile_weight = chunk->sqrt_dV_and_interp_weights
                                   ? std::sqrt(quadrature_weight)
                                   : (zero_divisor ? 0.0 : 1.0);
            }
            const std::complex<double> mode1 = eigenmode_amplitude(
                mode1_data, loc,
                chunk->S.transform(c_conjugate, chunk->sn));
            request.weighted_mode1.push_back(
                profile_weight * std::conj(mode1));
            if (mode_mode)
              request.mode2.push_back(eigenmode_amplitude(
                  mode2_data, loc,
                  chunk->S.transform(c, chunk->sn)));
            ++point_count;
          }
          if (point_count != chunk->N)
            throw std::logic_error(
                "Meep CUDA eigenmode sampled profile extent differs from "
                "DFT storage");
          const std::size_t profile_multiplier = mode_mode ? 2 : 1;
          if (point_count >
              (std::numeric_limits<std::size_t>::max() -
               sampled_profile_points) /
                  profile_multiplier)
            throw std::overflow_error(
                "Meep CUDA eigenmode sampled profile counter overflow");
          sampled_profile_points += point_count * profile_multiplier;
          sampled.push_back(std::move(request));
        }
    };

    if (consensus.channel_has_extent[0])
      append_component(cE[0], cH[0], 0u);
    if (consensus.channel_has_extent[1])
      append_component(cE[1], cH[1], 1u);
    if (consensus.channel_has_extent[2])
      append_component(cH[0], cE[0], 2u);
    if (consensus.channel_has_extent[3])
      append_component(cH[1], cE[1], 3u);

    std::vector<gpu::detail::eigenmode_overlap_request_fp32> requests;
    requests.reserve(sampled.size());
    for (const sampled_request &entry : sampled) {
      const dft_chunk *chunk = entry.chunk;
      const std::complex<double> inverse_weight =
          mode_mode ? std::complex<double>(0.0, 0.0)
                    : 1.0 / chunk->stored_weight;
      requests.push_back(
          {chunk->fc,
           mode_mode ? nullptr
                     : reinterpret_cast<const float *>(chunk->dft),
           entry.weighted_mode1.data(),
           mode_mode ? entry.mode2.data() : nullptr,
           nullptr,
           entry.zero_normalization_divisors.empty()
               ? nullptr
               : entry.zero_normalization_divisors.data(),
           chunk->N,
           mode_mode ? 0 : chunk->N,
           mode_mode ? 0 : chunk->omega.size(),
           entry.output_index,
           inverse_weight.real(), inverse_weight.imag()});
    }
    gpu::detail::record_eigenmode_host_profile_sampling(
        sampled_profile_points);
    double local_real_imag[8] = {};
    const void *plan_owner =
        flux.E ? static_cast<const void *>(flux.E)
               : (flux.H ? static_cast<const void *>(flux.H)
                         : static_cast<const void *>(this));
    gpu::detail::resident_reduce_eigenmode_overlaps_fp32(
        plan_owner,
        mode_mode
            ? gpu::detail::eigenmode_overlap_batch_kind::mode_mode
            : gpu::detail::eigenmode_overlap_batch_kind::mode_flux,
        requests.data(),
        requests.size(), mode_mode ? 0u
                                  : static_cast<std::size_t>(num_freq),
        local_real_imag);
    std::complex<double> local[4];
    std::complex<double> global[4];
    for (std::size_t channel = 0; channel < 4; ++channel)
      local[channel] = std::complex<double>(
          local_real_imag[2 * channel],
          local_real_imag[2 * channel + 1]);
    am_now_working_on(MpiAllTime);
    sum_to_all(local, global, 4);
    finished_working();
    gpu::detail::record_eigenmode_mpi_allreduce(
        sizeof(global));
    overlaps[0] = global[0] - global[1];
    overlaps[1] = global[2] - global[3];
    failure_guard.dismiss();
    return;
#else
    throw std::runtime_error(
        "Meep CUDA eigenmode overlap is unavailable in this build");
#endif
  }

  compute_cpu_eigenmode_overlap(
      this, mode1_data, mode2_data, flux, num_freq, cE, cH,
      overlaps);
}

void fields::get_mode_flux_overlap(void *mode_data, dft_flux flux, int num_freq,
                                   std::complex<double> overlaps[2]) {
  get_overlap(mode_data, 0, flux, num_freq, overlaps);
}

void fields::get_mode_mode_overlap(void *mode1_data, void *mode2_data, dft_flux flux,
                                   std::complex<double> overlaps[2]) {
  get_overlap(mode1_data, mode2_data, flux, 0, overlaps);
}

void fields::get_mode_flux_and_mode_mode_overlaps(
    void *mode_data, dft_flux flux, int num_freq,
    std::complex<double> mode_flux_overlaps[2],
    std::complex<double> mode_mode_overlaps[2]) {
  component cE[2], cH[2];
  switch (flux.normal_direction) {
    case X:
      cE[0] = Ey;
      cH[0] = Hz;
      cE[1] = Ez;
      cH[1] = Hy;
      break;
    case Y:
      cE[0] = Ez;
      cH[0] = Hx;
      cE[1] = Ex;
      cH[1] = Hz;
      break;
    case R:
      cE[0] = Ep;
      cH[0] = Hz;
      cE[1] = Ez;
      cH[1] = Hp;
      break;
    case P:
      cE[0] = Ez;
      cH[0] = Hr;
      cE[1] = Er;
      cH[1] = Hz;
      break;
    case Z:
      if (gv.dim == Dcyl)
        cE[0] = Er, cE[1] = Ep, cH[0] = Hp, cH[1] = Hr;
      else
        cE[0] = Ex, cE[1] = Ey, cH[0] = Hy, cH[1] = Hx;
      break;
    default:
      meep::abort(
          "invalid normal_direction in fused eigenmode overlap");
  }

  dft_chunk *chunklists[2] = {flux.E, flux.H};
  const component channel_components[4] = {
      cE[0], cE[1], cH[0], cH[1]};
  const eigenmode_overlap_consensus consensus =
      resolve_eigenmode_overlap_consensus(
          this, gv, v, chunklists, channel_components);
  gpu::detail::record_eigenmode_zero_rank_channels_skipped(
      static_cast<std::size_t>(std::count(
          consensus.channel_has_extent,
          consensus.channel_has_extent +
              eigenmode_overlap_component_count,
          false)));
  if (consensus.every_rank_cpu) {
    compute_cpu_eigenmode_overlap(
        this, mode_data, 0, flux, num_freq, cE, cH,
        mode_flux_overlaps);
    compute_cpu_eigenmode_overlap(
        this, mode_data, mode_data, flux, 0, cE, cH,
        mode_mode_overlaps);
    return;
  }
  gpu::detail::record_eigenmode_mpi_allreduce(
      eigenmode_overlap_consensus_value_count * sizeof(int));

  if (sizeof(realnum) != sizeof(float))
    throw std::runtime_error(
        "Meep CUDA fused eigenmode overlap requires a single-precision "
        "build");
#if MEEP_HAVE_CUDA && MEEP_SINGLE
  if (!mode_data || !mode_flux_overlaps || !mode_mode_overlaps)
    throw std::invalid_argument(
        "Meep CUDA fused eigenmode overlap requires mode and result "
        "storage");
  distributed_eigenmode_overlap_failure_guard failure_guard;

  struct sampled_request {
    const dft_chunk *chunk = nullptr;
    std::uint32_t output_index = 0;
    std::vector<std::complex<double> > mode_flux_weighted_mode1;
    std::vector<std::complex<double> > mode_mode_weighted_mode1;
    std::vector<std::complex<double> > mode2;
    std::vector<std::uint8_t> zero_normalization_divisors;
  };
  std::vector<sampled_request> sampled;
  std::size_t sampled_profile_points = 0;
  const auto append_component =
      [&](component c, component c_conjugate,
          std::uint32_t output_index) {
    for (int list_index = 0; list_index < 2; ++list_index)
      for (dft_chunk *chunk = chunklists[list_index]; chunk;
           chunk = chunk->next_in_dft) {
        if (chunk->c != c) continue;
        if (!chunk->fc || chunk->N == 0 || !chunk->dft ||
            chunk->omega.empty() || num_freq < 0 ||
            static_cast<std::size_t>(num_freq) >= chunk->omega.size())
          throw std::out_of_range(
              "Meep CUDA fused eigenmode overlap extent is outside DFT "
              "storage");
        if (chunk->stored_weight == std::complex<double>(0.0, 0.0) ||
            !std::isfinite(chunk->stored_weight.real()) ||
            !std::isfinite(chunk->stored_weight.imag()))
          throw std::invalid_argument(
              "Meep CUDA fused eigenmode stored weight must be finite "
              "and nonzero");

        sampled_request request;
        request.chunk = chunk;
        request.output_index = output_index;
        request.mode_flux_weighted_mode1.reserve(chunk->N);
        request.mode_mode_weighted_mode1.reserve(chunk->N);
        request.mode2.reserve(chunk->N);
        if (chunk->include_dV_and_interp_weights)
          request.zero_normalization_divisors.reserve(chunk->N);
        vec rshift(chunk->shift * (0.5 * chunk->fc->gv.inva));
        std::size_t point_count = 0;
        LOOP_OVER_IVECS(chunk->fc->gv, chunk->is, chunk->ie, idx) {
          IVEC_LOOP_LOC(chunk->fc->gv, loc);
          loc = chunk->S.transform(loc, chunk->sn) + rshift;
          const double quadrature_weight = IVEC_LOOP_WEIGHT(
              chunk->s0, chunk->s1, chunk->e0, chunk->e1,
              chunk->dV0 + chunk->dV1 * loop_i2);
          double mode_flux_profile_weight = quadrature_weight;
          if (chunk->include_dV_and_interp_weights) {
            const bool zero_divisor = quadrature_weight == 0.0;
            request.zero_normalization_divisors.push_back(
                zero_divisor ? 1u : 0u);
            mode_flux_profile_weight =
                chunk->sqrt_dV_and_interp_weights
                    ? std::sqrt(quadrature_weight)
                    : (zero_divisor ? 0.0 : 1.0);
          }
          const std::complex<double> conjugate_mode1 = std::conj(
              eigenmode_amplitude(
                  mode_data, loc,
                  chunk->S.transform(c_conjugate, chunk->sn)));
          request.mode_flux_weighted_mode1.push_back(
              mode_flux_profile_weight * conjugate_mode1);
          request.mode_mode_weighted_mode1.push_back(
              quadrature_weight * conjugate_mode1);
          request.mode2.push_back(eigenmode_amplitude(
              mode_data, loc, chunk->S.transform(c, chunk->sn)));
          ++point_count;
        }
        if (point_count != chunk->N)
          throw std::logic_error(
              "Meep CUDA fused eigenmode sampled profile extent differs "
              "from DFT storage");
        if (point_count >
            (std::numeric_limits<std::size_t>::max() -
             sampled_profile_points) /
                3u)
          throw std::overflow_error(
              "Meep CUDA fused eigenmode profile counter overflow");
        sampled_profile_points += 3u * point_count;
        sampled.push_back(std::move(request));
      }
  };

  if (consensus.channel_has_extent[0])
    append_component(cE[0], cH[0], 0u);
  if (consensus.channel_has_extent[1])
    append_component(cE[1], cH[1], 1u);
  if (consensus.channel_has_extent[2])
    append_component(cH[0], cE[0], 2u);
  if (consensus.channel_has_extent[3])
    append_component(cH[1], cE[1], 3u);

  std::vector<gpu::detail::eigenmode_overlap_request_fp32> requests;
  requests.reserve(sampled.size());
  for (const sampled_request &entry : sampled) {
    const dft_chunk *chunk = entry.chunk;
    const std::complex<double> inverse_weight =
        1.0 / chunk->stored_weight;
    requests.push_back(
        {chunk->fc,
         reinterpret_cast<const float *>(chunk->dft),
         entry.mode_flux_weighted_mode1.data(), entry.mode2.data(),
         entry.mode_mode_weighted_mode1.data(),
         entry.zero_normalization_divisors.empty()
             ? nullptr
             : entry.zero_normalization_divisors.data(),
         chunk->N, chunk->N, chunk->omega.size(), entry.output_index,
         inverse_weight.real(), inverse_weight.imag()});
  }
  gpu::detail::record_eigenmode_host_profile_sampling(
      sampled_profile_points);
  double local_real_imag[16] = {};
  const void *plan_owner =
      flux.E ? static_cast<const void *>(flux.E)
             : (flux.H ? static_cast<const void *>(flux.H)
                       : static_cast<const void *>(this));
  gpu::detail::resident_reduce_eigenmode_overlaps_fp32(
      plan_owner, gpu::detail::eigenmode_overlap_batch_kind::both,
      requests.data(), requests.size(),
      static_cast<std::size_t>(num_freq), local_real_imag);
  std::complex<double> local[8];
  std::complex<double> global[8];
  for (std::size_t channel = 0; channel < 8; ++channel)
    local[channel] = std::complex<double>(
        local_real_imag[2 * channel],
        local_real_imag[2 * channel + 1]);
  am_now_working_on(MpiAllTime);
  sum_to_all(local, global, 8);
  finished_working();
  gpu::detail::record_eigenmode_mpi_allreduce(sizeof(global));
  mode_flux_overlaps[0] = global[0] - global[1];
  mode_flux_overlaps[1] = global[2] - global[3];
  mode_mode_overlaps[0] = global[4] - global[5];
  mode_mode_overlaps[1] = global[6] - global[7];
  failure_guard.dismiss();
#else
  throw std::runtime_error(
      "Meep CUDA fused eigenmode overlap is unavailable in this build");
#endif
}

/* deregister all of the remaining dft monitors
from the fields object. Note that this does not
delete the underlying dft_chunks! (useful for
adjoint calculations, where we want to keep
the chunk data around) */
void fields::clear_dft_monitors() {
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine() && chunks[i]->dft_chunks) {
      // Remove the monitors from step-time updates while retaining an owner
      // lifecycle list.  The live fc pointer remains available to adjoint
      // consumers; fields_chunk destruction later publishes the cache and
      // nulls fc before the detached monitor can outlive its owner.
      dft_chunk *tail = chunks[i]->dft_chunks;
      while (tail->next_in_chunk) tail = tail->next_in_chunk;
      tail->next_in_chunk = chunks[i]->detached_dft_chunks;
      chunks[i]->detached_dft_chunks = chunks[i]->dft_chunks;
      chunks[i]->dft_chunks = NULL;
    }
}

// return the size of the dft monitor
std::vector<size_t> fields::dft_monitor_size(dft_fields fdft, const volume &where, component c) {
  ivec min_corner, max_corner;
  int rank, reduced_rank;
  direction dirs[3], reduced_dirs[3];
  size_t array_size, bufsz, dims[3], reduced_dims[3], reduced_stride[3], stride[3];
  dft_chunk *chunklists[1];
  chunklists[0] = fdft.chunks;

  get_dft_component_dims(chunklists, 1, c, min_corner, max_corner, array_size, bufsz, rank, dirs,
                         dims);
  reduce_array_dimensions(where, rank, dims, dirs, stride, reduced_rank, reduced_dims, reduced_dirs,
                          reduced_stride);
  std::vector<size_t> reduced_dims_vec = {reduced_dims[0], reduced_dims[1], reduced_dims[2]};

  return reduced_dims_vec;
}

std::vector<struct sourcedata> dft_fields::fourier_sourcedata(const volume &where, component c,
                                                              fields &f,
                                                              const std::complex<double> *dJ) {
  const size_t Nfreq = freq.size();

  ivec min_corner, max_corner;
  int rank, reduced_rank;
  direction dirs[3], reduced_dirs[3];
  size_t array_size, bufsz, dims[3], reduced_dims[3], reduced_stride[3], stride[3];
  dft_chunk *chunklists[1];
  chunklists[0] = chunks;

  f.get_dft_component_dims(chunklists, 1, c, min_corner, max_corner, array_size, bufsz, rank, dirs,
                           dims);
  reduce_array_dimensions(where, rank, dims, dirs, stride, reduced_rank, reduced_dims, reduced_dirs,
                          reduced_stride);
  size_t reduced_grid_size =
      reduced_dims[0] * reduced_dims[1] * reduced_dims[2]; // total number of points in the monitor

  std::vector<struct sourcedata> temp;

  for (dft_chunk *f = chunks; f; f = f->next_in_dft) {
    assert(Nfreq == f->omega.size());
    vec rshift(f->shift * (0.5 * f->fc->gv.inva));

    std::vector<ptrdiff_t> idx_arr;
    std::vector<std::complex<double> > amp_arr;
    std::complex<double> EH0 = std::complex<double>(0, 0);
    component c = component(f->c);
    direction cd = component_direction(c);
    sourcedata temp_struct = {c, idx_arr, f->fc->chunk_idx, amp_arr};

    int position_array[3] = {0, 0, 0}; // array indicating the position of a point relative to the
                                       // minimum corner of the monitor

    LOOP_OVER_IVECS(f->fc->gv, f->is, f->ie, idx) {
      IVEC_LOOP_LOC(f->fc->gv, x0);
      IVEC_LOOP_ILOC(f->fc->gv, ix0);
      x0 = f->S.transform(x0, f->sn) + rshift;
      ix0 = f->S.transform(ix0, f->sn) + f->shift;

      double dJ_weight = 1; // weight for linear interpolation
      int nd = 0;
      LOOP_OVER_DIRECTIONS(f->fc->gv.dim, d) {
        if (where.in_direction(d) > 0)
          position_array[nd++] = int((ix0.in_direction(d) - min_corner.in_direction(d)) / 2);
        else
          dJ_weight *= (1 - abs(x0.in_direction(d) - where.in_direction_min(d)) /
                                (f->fc->gv.inva)); // based on distances
      }

      // index when dJ is flattened to a one-dimenional array
      size_t idx_1d = (position_array[0] * reduced_dims[1] + position_array[1]) * reduced_dims[2] +
                      position_array[2];

      if (f->avg1 == 0 && f->avg2 == 0) { // yee_grid = true
        temp_struct.idx_arr.push_back(idx);
        for (size_t i = 0; i < Nfreq; ++i) {
          EH0 = dJ_weight * dJ[reduced_grid_size * i + idx_1d];

          if (is_electric(c)) EH0 *= -1;
          if (is_D(c) && f->fc->s->chi1inv[c - Dx + Ex][cd])
            EH0 /= -f->fc->s->chi1inv[c - Dx + Ex][cd][idx];
          if (is_B(c) && f->fc->s->chi1inv[c - Bx + Hx][cd])
            EH0 /= f->fc->s->chi1inv[c - Bx + Hx][cd][idx];

          EH0 /= f->S.multiplicity(ix0);
          temp_struct.amp_arr.push_back(EH0);
        }
      }
      else { // yee_grid = false
        // four or two neighbouring points in the yee lattice are involved in calculating the value
        // at the center of a voxel
        ptrdiff_t site_ind[4] = {idx, idx + f->avg1, idx + f->avg2, idx + f->avg1 + f->avg2};
        for (size_t j = 0; j < 4; ++j) {
          temp_struct.idx_arr.push_back(site_ind[j]);
          for (size_t i = 0; i < Nfreq; ++i) {
            EH0 = dJ_weight * dJ[reduced_grid_size * i + idx_1d] *
                  0.25; // split the amplitude of the adjoint source into four parts

            if (is_electric(c)) EH0 *= -1;
            if (is_D(c) && f->fc->s->chi1inv[c - Dx + Ex][cd])
              EH0 /= -f->fc->s->chi1inv[c - Dx + Ex][cd][idx];
            if (is_B(c) && f->fc->s->chi1inv[c - Bx + Hx][cd])
              EH0 /= f->fc->s->chi1inv[c - Bx + Hx][cd][idx];

            EH0 /= f->S.multiplicity(ix0);
            temp_struct.amp_arr.push_back(EH0);
          }
        }
      }
    }
    temp.push_back(temp_struct);
  }
  return temp;
}

} // namespace meep
