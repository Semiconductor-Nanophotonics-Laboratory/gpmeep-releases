/* Copyright (C) 2005-2026 Massachusetts Institute of Technology
%
%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2, or (at your option)
%  any later version.
%
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
%
%  You should have received a copy of the GNU General Public License
%  along with this program; if not, write to the Free Software Foundation,
%  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
*/

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#include "meep.hpp"
#include "meep_internals.hpp"
#include "gpu_backend_internal.hpp"
#include "gpu_grid_index.hpp"

#define RESTRICT

#include <limits>
#include <cstdint>
#include <cstring>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <typeinfo>

using namespace std;

namespace meep {
namespace {

std::uint64_t curl_phase_hash_mix(std::uint64_t hash,
                                  std::uint64_t value) noexcept {
  hash ^= value + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) +
          (hash >> 2);
  return hash;
}

template <typename T>
std::uint64_t curl_phase_hash_pointer(std::uint64_t hash,
                                      const T *pointer) noexcept {
  return curl_phase_hash_mix(
      hash, static_cast<std::uint64_t>(
                reinterpret_cast<std::uintptr_t>(pointer)));
}

std::uint64_t curl_phase_hash_double(std::uint64_t hash,
                                     double value) noexcept {
  static_assert(sizeof(double) == sizeof(std::uint64_t),
                "curl topology fingerprint requires binary64 double");
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return curl_phase_hash_mix(hash, bits);
}

std::uint64_t curl_phase_hash_ivec(std::uint64_t hash,
                                   const ivec &value) noexcept {
  for (int axis = 0; axis < 3; ++axis)
    hash = curl_phase_hash_mix(
        hash, static_cast<std::uint64_t>(value.yucky_val(axis)));
  return hash;
}

struct curl_phase_replay_topology {
  std::uint64_t fingerprint;
  bool replay_supported;
};

curl_phase_replay_topology current_curl_phase_replay_topology(
    fields &owner, field_type ft) {
  std::uint64_t hash = UINT64_C(0xcbf29ce484222325);
  hash = curl_phase_hash_mix(hash, static_cast<std::uint64_t>(ft));
  hash = curl_phase_hash_mix(
      hash, static_cast<std::uint64_t>(owner.num_chunks));
  hash = curl_phase_hash_double(hash, owner.beta);
  hash = curl_phase_hash_double(hash, owner.m);
  hash = curl_phase_hash_mix(
      hash, owner.bfast_scaled_k.size());
  for (double value : owner.bfast_scaled_k)
    hash = curl_phase_hash_double(hash, value);
  const bool tile_coalescing_disabled =
      std::getenv("MEEP_GPU_DISABLE_TILE_COALESCING") != nullptr;
  const bool batched_tiles_disabled =
      std::getenv("MEEP_GPU_DISABLE_BATCHED_TILES") != nullptr;
  hash = curl_phase_hash_mix(hash, tile_coalescing_disabled ? 1u : 0u);
  hash = curl_phase_hash_mix(hash, batched_tiles_disabled ? 1u : 0u);

  bool replay_supported = true;
  for (int chunk_index = 0; chunk_index < owner.num_chunks;
       ++chunk_index) {
    fields_chunk *chunk = owner.chunks[chunk_index];
    hash = curl_phase_hash_pointer(hash, chunk);
    const bool mine = chunk && chunk->is_mine();
    hash = curl_phase_hash_mix(hash, mine ? 1u : 0u);
    if (!mine) continue;

    hash = curl_phase_hash_mix(
        hash, static_cast<std::uint64_t>(chunk->gv.dim));
    hash = curl_phase_hash_mix(hash, chunk->gv.ntot());
    for (int axis = 0; axis < 3; ++axis)
      hash = curl_phase_hash_mix(
          hash, static_cast<std::uint64_t>(
                    chunk->gv.stride(static_cast<direction>(axis))));
    hash = curl_phase_hash_double(hash, chunk->Courant);
    hash = curl_phase_hash_double(hash, chunk->dt);
    hash = curl_phase_hash_double(hash, chunk->beta);
    hash = curl_phase_hash_double(hash, chunk->m);
    hash = curl_phase_hash_mix(hash, chunk->is_real);
    hash = curl_phase_hash_mix(
        hash, chunk->zero_fields_near_cylorigin ? 1u : 0u);
    hash = curl_phase_hash_mix(hash, chunk->bfast_scaled_k.size());
    bool uses_bfast = chunk->bfast_scaled_k.size() != 3;
    for (double value : chunk->bfast_scaled_k) {
      hash = curl_phase_hash_double(hash, value);
      uses_bfast = uses_bfast || value != 0.0;
    }
    // Cartesian material curls are the only phases covered by this replay
    // plan. The direct beta/BFAST/cylindrical corrections must be observed
    // by the ordinary chunk walk on every invocation.
    if (chunk->gv.dim == Dcyl || chunk->beta != 0.0 || uses_bfast)
      replay_supported = false;

    hash = curl_phase_hash_pointer(hash, chunk->s);
    hash = curl_phase_hash_mix(hash, chunk->gvs_tiled.size());
    for (const grid_volume &sub_gv : chunk->gvs_tiled) {
      hash = curl_phase_hash_mix(
          hash, static_cast<std::uint64_t>(sub_gv.dim));
      hash = curl_phase_hash_mix(hash, sub_gv.ntot());
      hash = curl_phase_hash_ivec(hash, sub_gv.big_corner());
      FOR_FT_COMPONENTS(ft, cc)
        hash = curl_phase_hash_ivec(
            hash, sub_gv.little_owned_corner0(cc));
    }

    const int complex_parts = 2 - chunk->is_real;
    // Input components are selected by a private, setup-time curl plan.
    // Hashing every allocated primary field is compact and also detects any
    // public allocation/pointer mutation that could add or replace an input.
    for (int component_index = 0;
         component_index < NUM_FIELD_COMPONENTS; ++component_index)
      for (int complex_part = 0; complex_part < complex_parts;
           ++complex_part)
        hash = curl_phase_hash_pointer(
            hash, chunk->f[component_index][complex_part]);
    FOR_FT_COMPONENTS(ft, cc) {
      const direction d_c = component_direction(cc);
      const direction dsig = cycle_direction(chunk->gv.dim, d_c, 1);
      const direction dsigu = cycle_direction(chunk->gv.dim, d_c, 2);
      for (int complex_part = 0; complex_part < complex_parts;
           ++complex_part) {
        hash = curl_phase_hash_pointer(hash, chunk->f_u[cc][complex_part]);
        hash = curl_phase_hash_pointer(
            hash, chunk->f_cond[cc][complex_part]);
        hash = curl_phase_hash_pointer(
            hash, chunk->f_bfast[cc][complex_part]);
      }
      hash = curl_phase_hash_pointer(
          hash, chunk->s->conductivity[cc][d_c]);
      hash = curl_phase_hash_pointer(hash, chunk->s->condinv[cc][d_c]);
      const direction material_directions[] = {dsig, dsigu};
      for (direction material_direction : material_directions) {
        hash = curl_phase_hash_mix(
            hash, static_cast<std::uint64_t>(
                      chunk->s->sigsize[material_direction]));
        hash = curl_phase_hash_pointer(
            hash, chunk->s->sig[material_direction]);
        hash = curl_phase_hash_pointer(
            hash, chunk->s->kap[material_direction]);
        hash = curl_phase_hash_pointer(
            hash, chunk->s->siginv[material_direction]);
      }
    }
  }
  return {hash, replay_supported};
}

void record_replayed_curl_tile_coalescing(fields &owner,
                                          field_type ft) noexcept {
  if (std::getenv("MEEP_GPU_DISABLE_TILE_COALESCING") != nullptr) return;
  for (int chunk_index = 0; chunk_index < owner.num_chunks;
       ++chunk_index) {
    fields_chunk *chunk = owner.chunks[chunk_index];
    if (!chunk || !chunk->is_mine() || chunk->gvs_tiled.size() <= 1)
      continue;
    bool has_curl_work = false;
    const int complex_parts = 2 - chunk->is_real;
    for (int complex_part = 0;
         complex_part < complex_parts && !has_curl_work; ++complex_part)
      FOR_FT_COMPONENTS(ft, component)
        if (chunk->f[component][complex_part]) {
          has_curl_work = true;
          break;
        }
    if (has_curl_work)
      gpu::detail::record_curl_tile_coalescing(
          static_cast<std::uint64_t>(chunk->gvs_tiled.size()));
  }
}

size_t loop_point_count(const ivec &is, const ivec &ie) {
  size_t count = 1;
  for (int axis = 0; axis < 3; ++axis) {
    const ptrdiff_t extent = (ie.yucky_val(axis) - is.yucky_val(axis)) / 2 + 1;
    if (extent <= 0) return 0;
    if (count > std::numeric_limits<size_t>::max() / static_cast<size_t>(extent))
      throw std::overflow_error("Meep curl loop point count overflow");
    count *= static_cast<size_t>(extent);
  }
  return count;
}

void checked_add_points(size_t value, size_t *total) {
  if (!total) return;
  if (value > std::numeric_limits<size_t>::max() - *total)
    throw std::overflow_error("Meep curl partition point count overflow");
  *total += value;
}

bool partition_curl_box(const ivec &full_is, const ivec &full_ie,
                        const bool active_directions[3],
                        ivec *interior_is, ivec *interior_ie) {
  *interior_is = full_is;
  *interior_ie = full_ie;
  for (int axis = 0; axis < 3; ++axis) {
    if (!active_directions[axis]) continue;
    // Bounds are inclusive and adjacent Yee output points differ by two.
    // Keep one output plane away from both faces: although each individual
    // stencil is one-sided, staggering can place its halo on either face.
    if (full_ie.yucky_val(axis) - full_is.yucky_val(axis) < 4)
      return false;
    const direction d = static_cast<direction>(axis);
    interior_is->set_direction(d, full_is.in_direction(d) + 2);
    interior_ie->set_direction(d, full_ie.in_direction(d) - 2);
  }
  return loop_point_count(*interior_is, *interior_ie) != 0;
}

using curl_box = std::pair<ivec, ivec>;

std::vector<curl_box> curl_region_boxes(const ivec &full_is,
                                        const ivec &full_ie,
                                        const ivec &global_interior_is,
                                        const ivec &global_interior_ie,
                                        bool have_global_interior,
                                        cuda_curl_region region) {
  if (region == cuda_curl_region::full)
    return std::vector<curl_box>(1, curl_box(full_is, full_ie));

  if (!have_global_interior)
    return region == cuda_curl_region::interior
               ? std::vector<curl_box>()
               : std::vector<curl_box>(1, curl_box(full_is, full_ie));
  if (region == cuda_curl_region::interior)
    return std::vector<curl_box>(
        1, curl_box(global_interior_is, global_interior_ie));

  std::vector<curl_box> boxes;
  ivec remaining_is = full_is;
  ivec remaining_ie = full_ie;
  for (int axis = 0; axis < 3; ++axis) {
    const direction d = static_cast<direction>(axis);
    if (remaining_is.in_direction(d) <
        global_interior_is.in_direction(d)) {
      ivec low_ie = remaining_ie;
      low_ie.set_direction(
          d, global_interior_is.in_direction(d) - 2);
      boxes.push_back(curl_box(remaining_is, low_ie));
    }
    if (global_interior_ie.in_direction(d) <
        remaining_ie.in_direction(d)) {
      ivec high_is = remaining_is;
      high_is.set_direction(
          d, global_interior_ie.in_direction(d) + 2);
      boxes.push_back(curl_box(high_is, remaining_ie));
    }
    remaining_is.set_direction(
        d, global_interior_is.in_direction(d));
    remaining_ie.set_direction(
        d, global_interior_ie.in_direction(d));
  }
  return boxes;
}

size_t halo_curl_minimum_interior_points() {
  const char *value =
      std::getenv("MEEP_GPU_HALO_CURL_MIN_INTERIOR_POINTS");
  if (!value || !*value) return 65536;
  for (const char *digit = value; *digit; ++digit)
    if (*digit < '0' || *digit > '9')
      throw std::invalid_argument(
          "MEEP_GPU_HALO_CURL_MIN_INTERIOR_POINTS must be a non-negative integer");
  errno = 0;
  char *end = NULL;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (errno || !end || *end != '\0' ||
      parsed > std::numeric_limits<size_t>::max())
    throw std::invalid_argument(
        "MEEP_GPU_HALO_CURL_MIN_INTERIOR_POINTS must be a non-negative integer");
  return static_cast<size_t>(parsed);
}

bool automatic_cuda_minimum_cells_override(size_t *minimum_cells) {
  if (!minimum_cells)
    throw std::invalid_argument(
        "automatic CUDA minimum output must be non-null");
  const char *value = std::getenv("MEEP_GPU_AUTO_MIN_CELLS");
  if (!value || !*value) return false;
  for (const char *digit = value; *digit; ++digit)
    if (*digit < '0' || *digit > '9')
      throw std::invalid_argument(
          "MEEP_GPU_AUTO_MIN_CELLS must be a non-negative integer");
  errno = 0;
  char *end = NULL;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (errno || !end || *end != '\0' ||
      parsed > std::numeric_limits<size_t>::max())
    throw std::invalid_argument(
        "MEEP_GPU_AUTO_MIN_CELLS must be a non-negative integer");
  *minimum_cells = static_cast<size_t>(parsed);
  return true;
}

std::uint64_t automatic_policy_cpu_threads() {
  // The core CPU Yee curl, constitutive update, and polarization loops in
  // this build are serial within one MPI rank. OMP_NUM_THREADS affects some
  // auxiliary host routines but does not create concurrent FDTD stencil
  // workers. Creating an empty OpenMP probe team here would therefore
  // overstate the CPU alternative and can strand a production owner on one
  // CPU core. MPI parallelism is already represented by each rank's smaller
  // local_cells value, so one rank contributes one actual CPU FDTD worker.
  return 1u;
}

std::string first_collective_failure_diagnostic(
    bool local_failure, const std::string &local_diagnostic) {
  const int no_failure = count_processors();
  const int source_rank = min_to_all(
      local_failure ? my_rank() : no_failure);
  if (source_rank == no_failure) return std::string();

  // A fixed-size collective avoids a second fallible allocation/length
  // protocol while ranks are already handling a preflight error.  CUDA and
  // policy diagnostics are short; truncation remains explicit and safely
  // NUL-terminated if a future provider emits an excessive message.
  constexpr int diagnostic_capacity = 4096;
  char diagnostic[diagnostic_capacity] = {};
  if (my_rank() == source_rank)
    std::snprintf(diagnostic, sizeof(diagnostic), "%s",
                  local_diagnostic.c_str());
  broadcast(source_rank, diagnostic, diagnostic_capacity);
  return std::string(diagnostic);
}

std::string rank_zero_collective_diagnostic(
    const std::string &local_diagnostic) {
  constexpr int diagnostic_capacity = 4096;
  char diagnostic[diagnostic_capacity] = {};
  if (my_rank() == 0)
    std::snprintf(diagnostic, sizeof(diagnostic), "%s",
                  local_diagnostic.c_str());
  broadcast(0, diagnostic, diagnostic_capacity);
  return std::string(diagnostic);
}

} // namespace

std::uint64_t gpu::detail::automatic_cuda_host_only_minimum_cells(
    std::uint64_t cpu_threads) noexcept {
  if (cpu_threads == 0) return std::numeric_limits<std::uint64_t>::max();
  constexpr long double reference_cells = 524288.0L;
  constexpr long double reference_cpu_threads = 1.0L;
  const long double scaled =
      reference_cells * static_cast<long double>(cpu_threads) /
      reference_cpu_threads;
  const long double maximum_cells = static_cast<long double>(
      std::numeric_limits<std::uint64_t>::max());
  if (!std::isfinite(scaled) || scaled >= maximum_cells)
    return std::numeric_limits<std::uint64_t>::max();
  return static_cast<std::uint64_t>(std::ceil(
      std::max<long double>(65536.0L, scaled)));
}

gpu::detail::automatic_cuda_policy_decision
gpu::detail::evaluate_automatic_cuda_policy(
    const automatic_cuda_policy_input &input) {
  if (input.cpu_threads == 0)
    throw std::invalid_argument(
        "automatic CUDA policy requires at least one CPU thread");
  if (input.device.total_memory_bytes == 0 ||
      input.device.free_memory_bytes > input.device.total_memory_bytes)
    throw std::invalid_argument(
        "automatic CUDA policy requires a valid free/total device-memory "
        "observation");
  std::uint64_t minimum_cells = input.explicit_minimum_cells;
  bool threshold_unreachable = false;
  if (!input.explicit_minimum) {
    if (input.device.multiprocessor_count <= 0)
      throw std::invalid_argument(
          "automatic CUDA policy requires a positive multiprocessor count");
    // The reference point is the conservative 524,288-cell crossover floor
    // calibrated with one serial CPU FDTD worker per MPI rank and a
    // 1.008-TB/s RTX 3090 Ti.  A 754,810-cell 3D periodic diffraction case
    // measured 7.66x faster than an eight-physical-core MPI CPU baseline;
    // the previous 1,048,576-cell floor was therefore a material false
    // negative.  Launch-dominated validation cases remain at or below
    // 148,877 local cells.
    // FDTD is predominantly memory-bandwidth bound, so scale the floor by
    // CPU team size and inversely by the selected device's theoretical memory
    // bandwidth.  SM count supplies a deterministic fallback for runtimes
    // that do not expose a usable memory clock/bus width.
    constexpr long double reference_cells = 524288.0L;
    constexpr long double reference_cpu_threads = 1.0L;
    constexpr long double reference_bandwidth = 1008000000000.0L;
    constexpr long double reference_sms = 84.0L;
    long double device_bandwidth = static_cast<long double>(
        input.device.memory_bandwidth_bytes_per_second);
    if (device_bandwidth <= 0.0L)
      device_bandwidth =
          reference_bandwidth *
          static_cast<long double>(input.device.multiprocessor_count) /
          reference_sms;
    const long double device_scaled =
        reference_cells * static_cast<long double>(input.cpu_threads) /
        reference_cpu_threads * reference_bandwidth / device_bandwidth;
    const long double host_floor = static_cast<long double>(
        automatic_cuda_host_only_minimum_cells(input.cpu_threads));
    const long double scaled = std::max(host_floor, device_scaled);
    const long double maximum_cells = static_cast<long double>(
        std::numeric_limits<std::uint64_t>::max());
    if (!std::isfinite(scaled) || scaled >= maximum_cells) {
      minimum_cells = std::numeric_limits<std::uint64_t>::max();
      threshold_unreachable = true;
    }
    else
      minimum_cells = static_cast<std::uint64_t>(std::ceil(
          std::max<long double>(65536.0L, scaled)));
  }
  // Reject memory pressure before a CUDA field mirror or kernel plan is
  // allocated.  The estimate deliberately covers substantially more than
  // the ordinary 6-component FP32 Yee state (materials, PML auxiliaries,
  // sources, DFT plans, and exchange buffers) and retains both a fixed and a
  // proportional reserve for the runtime and other processes.  This is a
  // conservative automatic-mode admission policy; required CUDA mode still
  // reports an allocator error instead of silently changing backends.
  constexpr std::uint64_t bytes_per_local_cell = 1024u;
  constexpr std::uint64_t fixed_working_bytes = 64u * 1024u * 1024u;
  constexpr std::uint64_t minimum_reserve_bytes = 512u * 1024u * 1024u;
  const std::uint64_t proportional_reserve =
      input.device.total_memory_bytes / 10u;
  const std::uint64_t reserve =
      std::max(minimum_reserve_bytes, proportional_reserve);
  const std::uint64_t available_after_reserve =
      input.device.free_memory_bytes > reserve
          ? input.device.free_memory_bytes - reserve
          : 0u;
  std::uint64_t estimated_required =
      std::numeric_limits<std::uint64_t>::max();
  if (input.local_cells <=
      (std::numeric_limits<std::uint64_t>::max() -
       fixed_working_bytes) /
          bytes_per_local_cell)
    estimated_required =
        input.local_cells * bytes_per_local_cell + fixed_working_bytes;
  const bool memory_eligible =
      estimated_required <= available_after_reserve;
  const bool size_eligible =
      !threshold_unreachable && input.local_cells >= minimum_cells;
  return {input.local_cells,
          minimum_cells,
          input.cpu_threads,
          input.device,
          estimated_required,
          available_after_reserve,
          input.explicit_minimum,
          memory_eligible,
          size_eligible && memory_eligible};
}

gpu::detail::curl_partition_validation
gpu::detail::validate_curl_partition_for_testing(
    const int lower[3], const int upper[3],
    const bool active_directions[3]) {
  const ivec full_is(lower[0], lower[1], lower[2]);
  const ivec full_ie(upper[0], upper[1], upper[2]);
  ivec interior_is, interior_ie;
  const bool have_interior = partition_curl_box(
      full_is, full_ie, active_directions, &interior_is, &interior_ie);
  const std::vector<curl_box> interior_boxes = curl_region_boxes(
      full_is, full_ie, interior_is, interior_ie, have_interior,
      cuda_curl_region::interior);
  const std::vector<curl_box> shell_boxes = curl_region_boxes(
      full_is, full_ie, interior_is, interior_ie, have_interior,
      cuda_curl_region::shell);
  using point = std::tuple<int, int, int>;
  const auto insert_boxes = [](const std::vector<curl_box> &boxes,
                               std::set<point> *points) {
    bool unique = true;
    for (const curl_box &box : boxes)
      for (int x = box.first.x(); x <= box.second.x(); x += 2)
        for (int y = box.first.y(); y <= box.second.y(); y += 2)
          for (int z = box.first.z(); z <= box.second.z(); z += 2)
            unique = points->insert(point(x, y, z)).second && unique;
    return unique;
  };
  std::set<point> full;
  for (int x = full_is.x(); x <= full_ie.x(); x += 2)
    for (int y = full_is.y(); y <= full_ie.y(); y += 2)
      for (int z = full_is.z(); z <= full_ie.z(); z += 2)
        full.insert(point(x, y, z));
  std::set<point> interior;
  std::set<point> shell;
  const bool interior_unique = insert_boxes(interior_boxes, &interior);
  const bool shell_unique = insert_boxes(shell_boxes, &shell);
  bool disjoint = interior_unique && shell_unique;
  for (const point &value : interior)
    if (shell.count(value)) {
      disjoint = false;
      break;
    }
  std::set<point> covered = interior;
  covered.insert(shell.begin(), shell.end());
  return {static_cast<std::uint64_t>(full.size()),
          static_cast<std::uint64_t>(interior.size()),
          static_cast<std::uint64_t>(shell.size()), have_interior,
          disjoint, covered == full};
}

void fields::cuda_backend_preflight() const {
  if (gpu_backend_preflight_is_cached()) return;
  gpu::detail::backend_preflight_status local =
      gpu::detail::probe_backend_for_distributed_step();
  const bool any_configuration_error =
      or_to_all(!local.configuration_ok);
  // Preserve the actual configuration error before interpreting the mode
  // bits.  A malformed mode intentionally has no valid bit and must not be
  // misreported as an MPI mode mismatch.
  if (any_configuration_error) {
    const std::string collective_diagnostic =
        first_collective_failure_diagnostic(
            !local.configuration_ok, local.diagnostic);
    std::ostringstream message;
    message << "distributed CUDA backend configuration failed";
    if (!collective_diagnostic.empty())
      message << ": " << collective_diagnostic;
    throw std::runtime_error(message.str());
  }

  const bool every_cpu_requested = and_to_all(local.cpu_requested);
  const bool every_automatic_requested =
      and_to_all(local.automatic_requested);
  const bool every_cuda_required = and_to_all(local.cuda_required);
  if (static_cast<int>(every_cpu_requested) +
          static_cast<int>(every_automatic_requested) +
          static_cast<int>(every_cuda_required) !=
      1)
    throw std::runtime_error(
        "distributed Meep backend requests differ across MPI ranks");

  if (every_cpu_requested) {
    const std::string diagnostic = "fields owner selected requested CPU";
    gpu::detail::activate_backend_for_owner(
        gpu::backend_mode::cpu, diagnostic);
    cache_gpu_backend_preflight(false, diagnostic);
    return;
  }

  size_t local_cells = 0;
  size_t explicit_minimum_cells = 0;
  bool explicit_minimum = false;
  std::uint64_t cpu_threads = 1;
  if (every_automatic_requested) {
    bool local_policy_valid = true;
    std::string local_policy_error;
    try {
      explicit_minimum = automatic_cuda_minimum_cells_override(
          &explicit_minimum_cells);
      for (int i = 0; i < num_chunks; ++i)
        if (chunks[i]->is_mine()) {
          const size_t chunk_cells = chunks[i]->gv.ntot();
          if (chunk_cells >
              std::numeric_limits<size_t>::max() - local_cells)
            throw std::overflow_error(
                "automatic CUDA local cell count overflow");
          local_cells += chunk_cells;
        }
      cpu_threads = automatic_policy_cpu_threads();
      if (cpu_threads == 0)
        throw std::runtime_error(
            "automatic CUDA CPU-team observation returned zero threads");
    }
    catch (const std::exception &error) {
      local_policy_valid = false;
      local_policy_error = error.what();
    }
    if (or_to_all(!local_policy_valid)) {
      const std::string collective_diagnostic =
          first_collective_failure_diagnostic(
              !local_policy_valid, local_policy_error);
      std::ostringstream message;
      message << "distributed automatic CUDA workload policy failed";
      if (!collective_diagnostic.empty())
        message << ": " << collective_diagnostic;
      throw std::runtime_error(message.str());
    }

    const bool root_explicit_minimum = broadcast(0, explicit_minimum);
    size_t root_explicit_minimum_cells = explicit_minimum_cells;
    broadcast(0, &root_explicit_minimum_cells, 1);
    const bool local_minimum_mismatch =
        explicit_minimum != root_explicit_minimum ||
        (explicit_minimum &&
         explicit_minimum_cells != root_explicit_minimum_cells);
    if (or_to_all(local_minimum_mismatch))
      throw std::runtime_error(
          "MEEP_GPU_AUTO_MIN_CELLS differs across MPI ranks");

    // This first-stage floor uses only the observed CPU team and workload.
    // If any rank is below it, every rank chooses CPU without querying the
    // CUDA runtime.  The second-stage device policy can only raise this
    // conservative floor for a slower GPU.
    const std::uint64_t host_only_minimum =
        explicit_minimum
            ? static_cast<std::uint64_t>(explicit_minimum_cells)
            : gpu::detail::automatic_cuda_host_only_minimum_cells(
                  cpu_threads);
    const bool local_can_reach_cuda =
        static_cast<std::uint64_t>(local_cells) >= host_only_minimum;
    if (!and_to_all(local_can_reach_cuda)) {
      std::ostringstream local_diagnostic;
      local_diagnostic
          << "automatic fields owner selected CPU before CUDA runtime: "
          << "local_cells=" << local_cells
          << ", host_only_minimum_cells=" << host_only_minimum
          << ", cpu_fdtd_workers_per_rank=" << cpu_threads
          << ", policy="
          << (explicit_minimum ? "explicit-minimum" : "host-floor-v2");
      const std::string diagnostic =
          first_collective_failure_diagnostic(
              !local_can_reach_cuda, local_diagnostic.str());
      prepare_gpu_owner_for_cpu();
      gpu::detail::activate_backend_for_owner(
          gpu::backend_mode::cpu, diagnostic);
      cache_gpu_backend_preflight(false, diagnostic);
      return;
    }

    // Only owners which survived the host-only floor are allowed to discover
    // and select a CUDA device.
    local = gpu::detail::prepare_cuda_candidate_for_distributed_step();
    if (or_to_all(!local.configuration_ok)) {
      const std::string collective_diagnostic =
          first_collective_failure_diagnostic(
              !local.configuration_ok, local.diagnostic);
      std::ostringstream message;
      message << "distributed automatic CUDA candidate configuration failed";
      if (!collective_diagnostic.empty())
        message << ": " << collective_diagnostic;
      throw std::runtime_error(message.str());
    }
  }

  const bool any_explicit_device_requested =
      or_to_all(local.explicit_device_requested);
  const bool any_cuda_active = or_to_all(local.cuda_active);
  const bool every_cuda_active = and_to_all(local.cuda_active);
  if (any_cuda_active != every_cuda_active) {
    if (every_cuda_required)
      throw std::runtime_error(
          "distributed CUDA preflight found inconsistent active backends");
    const std::string diagnostic =
        "automatic fields owner selected CPU because CUDA availability "
        "differs across MPI ranks";
    prepare_gpu_owner_for_cpu();
    gpu::detail::activate_backend_for_owner(
        gpu::backend_mode::cpu, diagnostic);
    cache_gpu_backend_preflight(false, diagnostic);
    return;
  }
  if (!any_cuda_active) {
    if (every_cuda_required)
      throw std::runtime_error(
          "required CUDA backend has no compatible active device");
    const std::string diagnostic =
        "automatic fields owner selected CPU because CUDA is unavailable";
    prepare_gpu_owner_for_cpu();
    gpu::detail::activate_backend_for_owner(
        gpu::backend_mode::cpu, diagnostic);
    cache_gpu_backend_preflight(false, diagnostic);
    return;
  }

  std::string decision_diagnostic =
      "required CUDA fields owner passed backend preflight";
  if (every_automatic_requested) {
    bool local_policy_valid = true;
    std::string local_policy_error;
    gpu::detail::automatic_cuda_policy_decision decision{};
    try {
      decision = gpu::detail::evaluate_automatic_cuda_policy(
          {static_cast<std::uint64_t>(local_cells),
           cpu_threads,
           gpu::detail::selected_automatic_device_policy_facts(),
           explicit_minimum,
           static_cast<std::uint64_t>(explicit_minimum_cells)});
    }
    catch (const std::exception &error) {
      local_policy_valid = false;
      local_policy_error = error.what();
    }
    if (or_to_all(!local_policy_valid)) {
      const std::string collective_diagnostic =
          first_collective_failure_diagnostic(
              !local_policy_valid, local_policy_error);
      std::ostringstream message;
      message << "distributed automatic CUDA workload policy failed";
      if (!collective_diagnostic.empty())
        message << ": " << collective_diagnostic;
      throw std::runtime_error(message.str());
    }
    if (!and_to_all(decision.use_cuda)) {
      std::ostringstream diagnostic;
      diagnostic
          << "automatic fields owner selected CPU: local_cells="
          << decision.local_cells << ", minimum_cells="
          << decision.minimum_cells << ", cpu_fdtd_workers_per_rank="
          << decision.cpu_threads << ", device_bandwidth_Bps="
          << decision.device.memory_bandwidth_bytes_per_second
          << ", estimated_device_memory_B="
          << decision.estimated_required_memory_bytes
          << ", available_device_memory_after_reserve_B="
          << decision.memory_available_after_reserve_bytes
          << ", memory_eligible="
          << (decision.memory_eligible ? "yes" : "no")
          << ", policy="
          << (decision.explicit_minimum ? "explicit-minimum"
                                        : "host-device-memory-v3");
      const std::string collective_diagnostic =
          first_collective_failure_diagnostic(
              !decision.use_cuda, diagnostic.str());
      prepare_gpu_owner_for_cpu();
      gpu::detail::activate_backend_for_owner(
          gpu::backend_mode::cpu, collective_diagnostic);
      cache_gpu_backend_preflight(false, collective_diagnostic);
      return;
    }
    std::ostringstream diagnostic;
    diagnostic << "automatic fields owner selected CUDA: local_cells="
               << decision.local_cells << ", minimum_cells="
               << decision.minimum_cells << ", cpu_fdtd_workers_per_rank="
               << decision.cpu_threads << ", device_bandwidth_Bps="
               << decision.device.memory_bandwidth_bytes_per_second
               << ", estimated_device_memory_B="
               << decision.estimated_required_memory_bytes
               << ", available_device_memory_after_reserve_B="
               << decision.memory_available_after_reserve_bytes
               << ", memory_eligible="
               << (decision.memory_eligible ? "yes" : "no")
               << ", policy="
               << (decision.explicit_minimum ? "explicit-minimum"
                                             : "host-device-memory-v3");
    decision_diagnostic =
        rank_zero_collective_diagnostic(diagnostic.str());
  }

  gpu::detail::activate_backend_for_owner(
      gpu::backend_mode::cuda, decision_diagnostic);

  // A transport mismatch is a wire-protocol error, not a CUDA performance
  // fallback. Fail every rank uniformly before any manager posts a request.
  validate_gpu_mpi_transport();

  try {
    validate_gpu_device_assignment();
  }
  catch (const std::exception &error) {
    if (every_cuda_required || any_explicit_device_requested) throw;
    const std::string diagnostic = rank_zero_collective_diagnostic(
        std::string("automatic fields owner selected CPU after CUDA ") +
        "device-assignment preflight failed: " + error.what());
    prepare_gpu_owner_for_cpu();
    gpu::detail::activate_backend_for_owner(
        gpu::backend_mode::cpu, diagnostic);
    cache_gpu_backend_preflight(false, diagnostic);
    return;
  }
  cache_gpu_backend_preflight(true, decision_diagnostic);
}

void fields::cuda_step_preflight() const {
  if (!gpu::detail::cuda_active()) return;
  if (gpu_step_preflight_is_cached()) return;
  bool local_supported = true;
  std::string local_reason;
  const field_type curl_types[] = {B_stuff, D_stuff};
  for (field_type ft : curl_types)
    for (int i = 0; i < num_chunks; ++i)
      if (chunks[i]->is_mine()) {
        std::string reason;
        if (!chunks[i]->cuda_step_db_eligible(ft, &reason)) {
          local_supported = false;
          if (local_reason.empty()) {
            std::ostringstream message;
            message << (ft == B_stuff ? "B" : "D")
                    << " curl in chunk " << i << ": " << reason;
            local_reason = message.str();
          }
        }
      }

  const field_type polarization_types[] = {H_stuff, E_stuff};
  for (field_type ft : polarization_types)
    for (int i = 0; i < num_chunks; ++i)
      if (chunks[i]->is_mine())
        for (polarization_state *p = chunks[i]->pol[ft]; p; p = p->next) {
          const bool standard_lorentzian =
              p->s &&
              typeid(*p->s) == typeid(lorentzian_susceptibility);
          const bool standard_gyrotropic =
              p->s &&
              typeid(*p->s) == typeid(gyrotropic_susceptibility);
          const bool standard_multilevel =
              p->s &&
              typeid(*p->s) == typeid(multilevel_susceptibility);
          if (!standard_lorentzian && !standard_gyrotropic &&
              !standard_multilevel) {
            local_supported = false;
            if (local_reason.empty()) {
              std::ostringstream message;
              message << "polarization in chunk " << i
                      << ": only standard Lorentzian/Drude, gyrotropic, "
                         "and multilevel susceptibilities are currently "
                         "supported";
              local_reason = message.str();
            }
          }
          else if (standard_gyrotropic || standard_multilevel) {
            realnum *w[NUM_FIELD_COMPONENTS][2];
            FOR_COMPONENTS(c) DOCMP2 {
              w[c][cmp] = chunks[i]->f_w[c][cmp]
                              ? chunks[i]->f_w[c][cmp]
                              : chunks[i]->f[c][cmp];
            }
            std::string reason;
            const bool eligible =
                standard_gyrotropic
                    ? static_cast<const gyrotropic_susceptibility *>(p->s)
                          ->cuda_update_P_eligible(
                              w, chunks[i]->gv, &reason)
                    : static_cast<const multilevel_susceptibility *>(p->s)
                          ->cuda_update_P_eligible(
                              w, chunks[i]->f_w_prev, chunks[i]->gv,
                              &reason);
            if (!eligible) {
              local_supported = false;
              if (local_reason.empty()) {
                std::ostringstream message;
                message
                    << (standard_gyrotropic ? "gyrotropic"
                                           : "multilevel")
                    << " polarization in chunk " << i << ": " << reason;
                local_reason = message.str();
              }
            }
          }
        }
  std::string dft_reason;
  if (!cuda_dft_preflight(&dft_reason)) {
    local_supported = false;
    if (local_reason.empty()) local_reason = dft_reason;
  }
  if (!and_to_all(local_supported)) {
    const std::string collective_reason =
        first_collective_failure_diagnostic(
            !local_supported, local_reason);
    std::ostringstream message;
    message << "distributed CUDA time-step feature preflight failed";
    if (!collective_reason.empty()) message << ": " << collective_reason;
    if (gpu::detail::cuda_required())
      throw std::runtime_error(message.str());
    prepare_gpu_owner_for_cpu();
    const std::string diagnostic =
        "automatic fields owner selected CPU: " + message.str();
    gpu::detail::activate_backend_for_owner(
        gpu::backend_mode::cpu, diagnostic);
    cache_gpu_backend_preflight(false, diagnostic, true);
    return;
  }
  cache_gpu_step_preflight();
}

void fields::step_db(field_type ft) {
  if (ft != B_stuff && ft != D_stuff) meep::abort("step_db only works with B/D");

  // A required CUDA backend is all-or-nothing for this curl phase. Preflight
  // every local chunk before updating any of them, so an unsupported chunk
  // cannot leave a partially advanced time step.
  if (gpu::detail::cuda_active()) {
    for (int i = 0; i < num_chunks; ++i)
      if (chunks[i]->is_mine()) {
        std::string reason;
        if (!chunks[i]->cuda_step_db_eligible(ft, &reason)) {
          std::ostringstream message;
          message << "CUDA step_db preflight failed for chunk " << i << ": " << reason;
          throw std::runtime_error(message.str());
        }
      }
  }

  gpu::detail::phase_batch_mode phase_batch_mode =
      gpu::detail::cuda_active()
          ? gpu::detail::phase_batched_curl_mode()
          : gpu::detail::phase_batch_mode::disabled;
  for (int i = 0;
       i < num_chunks &&
       phase_batch_mode != gpu::detail::phase_batch_mode::disabled;
       ++i)
    if (chunks[i]->is_mine() &&
        !gpu::detail::resident_phase_is_active_for_owner(chunks[i]))
      phase_batch_mode = gpu::detail::phase_batch_mode::disabled;
  const curl_phase_replay_topology replay_topology =
      phase_batch_mode != gpu::detail::phase_batch_mode::disabled
          ? current_curl_phase_replay_topology(*this, ft)
          : curl_phase_replay_topology{0, false};
  gpu::detail::resident_curl_phase_batch phase_batch(
      this, static_cast<int>(ft), phase_batch_mode,
      replay_topology.fingerprint, replay_topology.replay_supported);

  const bool replayed = phase_batch.replay_if_ready();
  if (!replayed)
    for (int i = 0; i < num_chunks; i++)
      if (chunks[i]->is_mine())
        if (chunks[i]->step_db(ft)) {
          chunk_connections_valid = false;
          assert(changed_materials);
        }
  if (replayed) record_replayed_curl_tile_coalescing(*this, ft);
  phase_batch.finish();
}

bool fields_chunk::cuda_step_db_eligible(field_type ft, std::string *reason) const {
  const auto reject = [reason](const std::string &message) {
    if (reason) *reason = message;
    return false;
  };

  if (sizeof(realnum) != sizeof(float))
    return reject("the current CUDA curl backend requires FP32 (--enable-single)");
  if (bfast_scaled_k.size() != 3)
    return reject("BFAST wavevector must contain exactly three components");
  const bool use_bfast =
      bfast_scaled_k[0] || bfast_scaled_k[1] || bfast_scaled_k[2];
  if (use_bfast && gv.dim == Dcyl)
    return reject(
        "cylindrical BFAST semantics are not supported by the CUDA curl");

  DOCMP FOR_FT_COMPONENTS(ft, cc) {
    if (!f[cc][cmp]) continue;
    const bool have_p = have_plus_deriv[cc] && f[plus_component[cc]][cmp];
    const bool have_m = have_minus_deriv[cc] && f[minus_component[cc]][cmp];
    if (!have_p && !have_m) {
      std::ostringstream message;
      message << "no curl operand is allocated for " << component_name(cc);
      return reject(message.str());
    }
  }

  if (reason) reason->clear();
  return true;
}

bool fields_chunk::cuda_halo_curl_overlap_eligible(
    field_type ft, size_t *full_points, size_t *interior_points,
    bool *rejected_small, std::string *reason) const {
  const auto reject = [reason](const std::string &message) {
    if (reason) *reason = message;
    return false;
  };
  if (rejected_small) *rejected_small = false;
  if (ft != D_stuff)
    return reject("halo/curl overlap only partitions the D curl");
  if (gv.dim != D3)
    return reject("halo/curl overlap currently requires a 3D grid");

  std::string base_reason;
  if (!cuda_step_db_eligible(ft, &base_reason)) return reject(base_reason);
  if (bfast_scaled_k[0] || bfast_scaled_k[1] || bfast_scaled_k[2])
    return reject("BFAST curl partitioning is not supported");
  if (gvs_tiled.size() != 1)
    return reject(
        "explicit loop-tiled D curls are not yet safely coalesced");
  if (!gpu::detail::resident_phase_is_active_for_owner(this))
    return reject("no resident CUDA phase is active for the chunk");

  size_t local_full = 0;
  size_t local_interior = 0;
  for (const auto &sub_gv : gvs_tiled) {
    DOCMP FOR_FT_COMPONENTS(ft, cc) {
      if (!f[cc][cmp]) continue;
      const component c_p = plus_component[cc];
      const component c_m = minus_component[cc];
      const bool have_p =
          have_plus_deriv[cc] && f[c_p][cmp] != nullptr;
      const bool have_m =
          have_minus_deriv[cc] && f[c_m][cmp] != nullptr;
      bool active_directions[3] = {false, false, false};
      if (have_p)
        active_directions[static_cast<int>(plus_deriv_direction[cc])] =
            true;
      if (have_m)
        active_directions[static_cast<int>(minus_deriv_direction[cc])] =
            true;

      const direction d_c = component_direction(cc);
      const direction dsig0 = cycle_direction(gv.dim, d_c, 1);
      const direction dsig =
          s->sigsize[dsig0] > 1 ? dsig0 : NO_DIRECTION;
      const direction dsigu0 = cycle_direction(gv.dim, d_c, 2);
      const direction dsigu =
          s->sigsize[dsigu0] > 1 ? dsigu0 : NO_DIRECTION;
      if (dsig != NO_DIRECTION && s->conductivity[cc][d_c] &&
          !f_cond[cc][cmp])
        return reject("PML/conductivity auxiliary is not warm");
      if (dsigu != NO_DIRECTION && !f_u[cc][cmp])
        return reject("PML auxiliary is not warm");

      const ivec full_is = sub_gv.little_owned_corner0(cc);
      const ivec full_ie = sub_gv.big_corner();
      ivec global_interior_is, global_interior_ie;
      const bool have_global_interior = partition_curl_box(
          gv.little_owned_corner0(cc), gv.big_corner(),
          active_directions, &global_interior_is,
          &global_interior_ie);
      if (!have_global_interior) {
        if (rejected_small) *rejected_small = true;
        return reject("curl chunk is too thin for a disjoint interior");
      }
      checked_add_points(loop_point_count(full_is, full_ie), &local_full);
      const std::vector<curl_box> interior_boxes = curl_region_boxes(
          full_is, full_ie, global_interior_is, global_interior_ie,
          have_global_interior, cuda_curl_region::interior);
      for (const curl_box &box : interior_boxes)
        checked_add_points(loop_point_count(box.first, box.second),
                           &local_interior);
    }
  }
  if (local_full == 0 || local_interior == 0) {
    if (rejected_small) *rejected_small = true;
    return reject("curl partition contains no field points");
  }
  checked_add_points(local_full, full_points);
  checked_add_points(local_interior, interior_points);
  if (reason) reason->clear();
  return true;
}

bool fields::cuda_halo_curl_overlap_eligible(
    field_type ft, size_t *full_points, size_t *interior_points,
    bool *rejected_small, std::string *reason) const {
  const auto reject = [reason](const std::string &message) {
    if (reason) *reason = message;
    return false;
  };
  if (full_points) *full_points = 0;
  if (interior_points) *interior_points = 0;
  if (rejected_small) *rejected_small = false;
  if (!gpu::detail::cuda_active())
    return reject("CUDA backend is not active");

  bool have_local_chunk = false;
  for (int i = 0; i < num_chunks; ++i)
    if (chunks[i]->is_mine()) {
      have_local_chunk = true;
      bool chunk_small = false;
      std::string chunk_reason;
      if (!chunks[i]->cuda_halo_curl_overlap_eligible(
              ft, full_points, interior_points, &chunk_small,
              &chunk_reason)) {
        if (rejected_small) *rejected_small = chunk_small;
        return reject(chunk_reason);
      }
    }
  if (!have_local_chunk) return reject("rank owns no field chunk");
  const size_t minimum = halo_curl_minimum_interior_points();
  const size_t interior = interior_points ? *interior_points : 0;
  if (interior < minimum) {
    if (rejected_small) *rejected_small = true;
    std::ostringstream message;
    message << "interior curl workload " << interior
            << " is below minimum " << minimum;
    return reject(message.str());
  }
  if (reason) reason->clear();
  return true;
}

#if MEEP_SINGLE
bool fields_chunk::step_db_cuda_region(field_type ft,
                                       cuda_curl_region region) {
  if (ft != D_stuff || gv.dim != D3)
    throw std::logic_error(
        "CUDA halo/curl region dispatch requires a 3D D curl");
  if (bfast_scaled_k[0] || bfast_scaled_k[1] || bfast_scaled_k[2])
    throw std::logic_error(
        "CUDA halo/curl region dispatch does not support BFAST");

  gpu::detail::resident_curl_session cuda_session(this, true);
  for (const auto &sub_gv : gvs_tiled) {
    DOCMP FOR_FT_COMPONENTS(ft, cc) {
      if (!f[cc][cmp]) continue;
      const component c_p = plus_component[cc];
      const component c_m = minus_component[cc];
      const direction d_deriv_p = plus_deriv_direction[cc];
      const direction d_deriv_m = minus_deriv_direction[cc];
      const bool have_p = have_plus_deriv[cc] && f[c_p][cmp];
      const bool have_m = have_minus_deriv[cc] && f[c_m][cmp];
      realnum *f_p = have_p ? f[c_p][cmp] : nullptr;
      realnum *f_m = have_m ? f[c_m][cmp] : nullptr;
      ptrdiff_t stride_p = have_p ? -gv.stride(d_deriv_p) : 0;
      ptrdiff_t stride_m = have_m ? -gv.stride(d_deriv_m) : 0;
      bool active_directions[3] = {false, false, false};
      if (have_p)
        active_directions[static_cast<int>(d_deriv_p)] = true;
      if (have_m)
        active_directions[static_cast<int>(d_deriv_m)] = true;

      const direction d_c = component_direction(cc);
      const direction dsig0 = cycle_direction(gv.dim, d_c, 1);
      const direction dsig =
          s->sigsize[dsig0] > 1 ? dsig0 : NO_DIRECTION;
      const direction dsigu0 = cycle_direction(gv.dim, d_c, 2);
      const direction dsigu =
          s->sigsize[dsigu0] > 1 ? dsigu0 : NO_DIRECTION;
      const bool pml_f = dsig != NO_DIRECTION;
      const bool pml_u = dsigu != NO_DIRECTION;
      const realnum *conductivity = s->conductivity[cc][d_c];
      if (pml_f && conductivity && !f_cond[cc][cmp])
        throw std::logic_error(
            "CUDA halo/curl PML/conductivity auxiliary became cold");
      if (pml_u && !f_u[cc][cmp])
        throw std::logic_error(
            "CUDA halo/curl PML auxiliary became cold");
      const gpu::detail::curl_material_fp32 material = {
          pml_f ? s->sig[dsig] : nullptr,
          pml_f ? s->kap[dsig] : nullptr,
          pml_f ? s->siginv[dsig] : nullptr,
          pml_f ? static_cast<size_t>(s->sigsize[dsig]) : 0,
          pml_u ? f_u[cc][cmp] : nullptr,
          pml_u ? s->sig[dsigu] : nullptr,
          pml_u ? s->kap[dsigu] : nullptr,
          pml_u ? s->siginv[dsigu] : nullptr,
          pml_u ? static_cast<size_t>(s->sigsize[dsigu]) : 0,
          static_cast<float>(dt),
          conductivity,
          conductivity ? s->condinv[cc][d_c] : nullptr,
          pml_f && conductivity ? f_cond[cc][cmp] : nullptr};

      ivec global_interior_is, global_interior_ie;
      const bool have_global_interior = partition_curl_box(
          gv.little_owned_corner0(cc), gv.big_corner(),
          active_directions, &global_interior_is,
          &global_interior_ie);
      const std::vector<curl_box> boxes = curl_region_boxes(
          sub_gv.little_owned_corner0(cc), sub_gv.big_corner(),
          global_interior_is, global_interior_ie,
          have_global_interior, region);
      for (const curl_box &box : boxes) {
        const gpu::detail::index_space_fp32 index_space =
            gpu::detail::make_index_space_fp32(
                gv, box.first, box.second, dsig, dsigu);
        gpu::detail::resident_step_curl_material_fp32(
            cuda_session.cache(), f[cc][cmp], f_p, f_m, gv.ntot(),
            index_space, stride_p, stride_m,
            static_cast<float>(Courant), material);
      }
    }
  }
  cuda_session.finish();
  return false;
}

void fields::step_db_cuda_region(field_type ft, cuda_curl_region region) {
  if (ft != D_stuff || region == cuda_curl_region::full)
    throw std::logic_error("invalid CUDA halo/curl region phase");
  gpu::detail::phase_batch_mode phase_batch_mode =
      gpu::detail::cuda_active()
          ? gpu::detail::phase_batched_curl_mode()
          : gpu::detail::phase_batch_mode::disabled;
  for (int i = 0;
       i < num_chunks &&
       phase_batch_mode != gpu::detail::phase_batch_mode::disabled;
       ++i)
    if (chunks[i]->is_mine() &&
        !gpu::detail::resident_phase_is_active_for_owner(chunks[i]))
      phase_batch_mode = gpu::detail::phase_batch_mode::disabled;
  const int phase_key =
      (region == cuda_curl_region::interior ? NUM_FIELD_TYPES
                                            : 2 * NUM_FIELD_TYPES) +
      static_cast<int>(ft);
  const curl_phase_replay_topology replay_topology =
      phase_batch_mode != gpu::detail::phase_batch_mode::disabled
          ? current_curl_phase_replay_topology(*this, ft)
          : curl_phase_replay_topology{0, false};
  gpu::detail::resident_curl_phase_batch phase_batch(
      this, phase_key, phase_batch_mode,
      replay_topology.fingerprint, replay_topology.replay_supported);
  if (!phase_batch.replay_if_ready())
    for (int i = 0; i < num_chunks; ++i)
      if (chunks[i]->is_mine())
        if (chunks[i]->step_db_cuda_region(ft, region)) {
          chunk_connections_valid = false;
          assert(changed_materials);
        }
  phase_batch.finish();
}
#endif

bool fields_chunk::step_db(field_type ft) {
  bool allocated_u = false;
  const bool cuda_backend_active = gpu::detail::cuda_active();
  std::string cuda_ineligible_reason;
  const bool cuda_phase_eligible =
      cuda_backend_active && cuda_step_db_eligible(ft, &cuda_ineligible_reason);
  if (cuda_backend_active && !cuda_phase_eligible)
    throw std::runtime_error("preflighted CUDA chunk curl is unsupported: " +
                             cuda_ineligible_reason);
  gpu::detail::resident_curl_session cuda_session(this, cuda_phase_eligible);

  // CPU cache tiles partition one chunk's owned Yee points. They improve host
  // locality, but thousands of tiny CUDA descriptors make launch scheduling
  // dominate the stencil. CUDA therefore executes the equivalent full chunk
  // volume once while retaining gvs_tiled unchanged for CPU fallback and the
  // diagnostic legacy path.
  const bool coalesce_cuda_tiles =
      cuda_session.active() && gvs_tiled.size() > 1 &&
      std::getenv("MEEP_GPU_DISABLE_TILE_COALESCING") == nullptr;
  const size_t execution_volume_count =
      coalesce_cuda_tiles ? 1 : gvs_tiled.size();
#if MEEP_HAVE_CUDA
  bool recorded_tile_coalescing = false;
  std::vector<gpu::detail::index_space_fp32>
      cuda_index_spaces[NUM_FIELD_COMPONENTS][2];
#endif
  for (size_t volume_index = 0; volume_index < execution_volume_count;
       ++volume_index) {
    const grid_volume &sub_gv =
        coalesce_cuda_tiles ? gv : gvs_tiled[volume_index];
    DOCMP FOR_FT_COMPONENTS(ft, cc) {
      if (f[cc][cmp]) {
        const component c_p = plus_component[cc], c_m = minus_component[cc];
        const direction d_deriv_p = plus_deriv_direction[cc];
        const direction d_deriv_m = minus_deriv_direction[cc];
        const direction d_c = component_direction(cc);
        const bool have_p = have_plus_deriv[cc];
        const bool have_m = have_minus_deriv[cc];
        const direction dsig0 = cycle_direction(gv.dim, d_c, 1);
        const direction dsig = s->sigsize[dsig0] > 1 ? dsig0 : NO_DIRECTION;
        const direction dsigu0 = cycle_direction(gv.dim, d_c, 2);
        const direction dsigu = s->sigsize[dsigu0] > 1 ? dsigu0 : NO_DIRECTION;
        ptrdiff_t stride_p = have_p ? gv.stride(d_deriv_p) : 0;
        ptrdiff_t stride_m = have_m ? gv.stride(d_deriv_m) : 0;
        realnum *f_p = have_p ? f[c_p][cmp] : NULL;
        realnum *f_m = have_m ? f[c_m][cmp] : NULL;
        realnum *the_f = f[cc][cmp];
        const bool use_bfast =
            bfast_scaled_k[0] || bfast_scaled_k[1] ||
            bfast_scaled_k[2];

        if (dsig != NO_DIRECTION && s->conductivity[cc][d_c] && !f_cond[cc][cmp]) {
          f_cond[cc][cmp] = new realnum[gv.ntot()];
          memset(f_cond[cc][cmp], 0, sizeof(realnum) * gv.ntot());
        }
        if (dsigu != NO_DIRECTION && !f_u[cc][cmp]) {
          f_u[cc][cmp] = new realnum[gv.ntot()];
          memcpy(f_u[cc][cmp], the_f, gv.ntot() * sizeof(realnum));
          allocated_u = true;
        }
        if (use_bfast && !f_bfast[cc][cmp]) {
          f_bfast[cc][cmp] = new realnum[gv.ntot()];
          memset(f_bfast[cc][cmp], 0, sizeof(realnum) * gv.ntot());
        }

        if (ft == D_stuff) { // strides are opposite sign for H curl
          stride_p = -stride_p;
          stride_m = -stride_m;
        }

        if (gv.dim == Dcyl) switch (d_c) {
            case R:
              f_p = NULL; // im/r Fz term will be handled separately
              break;
            case P: break; // curl works normally for phi component
            case Z: {
              f_m = NULL; // im/r Fr term will be handled separately

              /* Here we do a somewhat cool hack: the update of the z
                 component gives a 1/r d(r Fp)/dr term, rather than
                 just the derivative dg/dr expected in step_curl.
                 Rather than duplicating all of step_curl to handle
                 this bloody derivative, however, we define a new
                 array f_rderiv_int which is the integral of 1/r d(r Fp)/dr,
                 so that we can pass it to the unmodified step_curl
                 and get the correct derivative.  (More precisely,
                 the derivative and integral are replaced by differences
                 and sums, but you get the idea). */
              if (!cuda_session.active()) {
                if (!f_rderiv_int)
                  f_rderiv_int = new realnum[gv.ntot()];
                realnum ir0 =
                    gv.origin_r() * gv.a +
                    0.5 * gv.iyee_shift(c_p).in_direction(R);
                memset(
                    f_rderiv_int, 0,
                    sizeof(realnum) * (gv.nz() + 1));
                int sr = gv.nz() + 1;
                for (int ir = 1; ir <= gv.nr(); ++ir) {
                  realnum rinv = 1.0 / ((ir + ir0) - 0.5);
                  IVDEP
                  for (int iz = 0; iz <= gv.nz(); ++iz) {
                    ptrdiff_t idx = ir * sr + iz;
                    f_rderiv_int[idx] =
                        f_rderiv_int[idx - sr] +
                        rinv *
                            (f_p[idx] * (ir + ir0) -
                             f_p[idx - sr] * ((ir - 1) + ir0));
                  }
                }
                f_p = f_rderiv_int;
              }
              break;
            }
            default: meep::abort("bug - non-cylindrical field component in Dcyl");
          }

        const ivec curl_is = sub_gv.little_owned_corner0(cc);
        const ivec curl_ie = sub_gv.big_corner();
        const size_t curl_points = loop_point_count(curl_is, curl_ie);
        bool cuda_dispatched = false;

#if MEEP_HAVE_CUDA
        const bool cuda_eligible = f_p || f_m;
        const bool batch_tiles =
            execution_volume_count > 1 &&
            std::getenv("MEEP_GPU_DISABLE_BATCHED_TILES") == nullptr;
        if (cuda_session.active() && !cuda_eligible)
          throw std::logic_error("Meep CUDA phase eligibility changed during a curl update");
        if (cuda_eligible && cuda_session.active()) {
          const gpu::detail::index_space_fp32 index_space =
              gpu::detail::make_index_space_fp32(
                  gv, curl_is, curl_ie, dsig, dsigu);
          const bool pml_f = dsig != NO_DIRECTION;
          const bool pml_u = dsigu != NO_DIRECTION;
          const realnum *conductivity = s->conductivity[cc][d_c];
          const gpu::detail::curl_material_fp32 material = {
              pml_f ? s->sig[dsig] : nullptr,
              pml_f ? s->kap[dsig] : nullptr,
              pml_f ? s->siginv[dsig] : nullptr,
              pml_f ? static_cast<size_t>(s->sigsize[dsig]) : 0,
              pml_u ? f_u[cc][cmp] : nullptr,
              pml_u ? s->sig[dsigu] : nullptr,
              pml_u ? s->kap[dsigu] : nullptr,
              pml_u ? s->siginv[dsigu] : nullptr,
              pml_u ? static_cast<size_t>(s->sigsize[dsigu]) : 0,
              static_cast<float>(dt),
              conductivity,
              conductivity ? s->condinv[cc][d_c] : nullptr,
              pml_f && conductivity ? f_cond[cc][cmp] : nullptr};
          if (gv.dim == Dcyl && d_c == Z) {
            const float radial_origin_offset =
                static_cast<float>(
                    gv.origin_r() * gv.a +
                    0.5 *
                        gv.iyee_shift(c_p).in_direction(R));
            gpu::detail::resident_step_cylindrical_radial_curl_fp32(
                cuda_session.cache(), the_f, f_p, gv.ntot(), index_space,
                gv.stride(R), radial_origin_offset,
                ft == D_stuff ? -1 : 1,
                static_cast<float>(Courant), material);
          }
          else {
            std::vector<gpu::detail::index_space_fp32> &component_spaces =
                cuda_index_spaces[cc][cmp];
            component_spaces.push_back(index_space);
            if (!batch_tiles)
              gpu::detail::resident_step_curl_material_fp32(
                  cuda_session.cache(), the_f, f_p, f_m, gv.ntot(),
                  index_space, stride_p, stride_m,
                  static_cast<float>(Courant), material);
            else if (component_spaces.size() == execution_volume_count)
              gpu::detail::resident_step_curl_material_batched_fp32(
                  cuda_session.cache(), the_f, f_p, f_m, gv.ntot(),
                  component_spaces.data(), component_spaces.size(),
                  stride_p, stride_m, static_cast<float>(Courant),
                  material);
          }
          if (coalesce_cuda_tiles && !recorded_tile_coalescing) {
            gpu::detail::record_curl_tile_coalescing(
                static_cast<std::uint64_t>(gvs_tiled.size()));
            recorded_tile_coalescing = true;
          }
          cuda_dispatched = true;
        }
#endif

        if (!cuda_dispatched) {
          STEP_CURL(the_f, cc, f_p, f_m, stride_p, stride_m, gv, curl_is, curl_ie, Courant, dsig,
                    s->sig[dsig], s->kap[dsig], s->siginv[dsig], f_u[cc][cmp], dsigu,
                    s->sig[dsigu], s->kap[dsigu], s->siginv[dsigu], dt,
                    s->conductivity[cc][d_c], s->condinv[cc][d_c], f_cond[cc][cmp]);
          gpu::detail::record_cpu_curl(curl_points);
        }

        if (use_bfast) {
          realnum k1 =
              have_m ? bfast_scaled_k[component_index(c_m)] : 0; // puts k1 in direction of g2
          realnum k2 =
              have_p ? bfast_scaled_k[component_index(c_p)] : 0; // puts k2 in direction of g1
          if (ft == D_stuff) {
            k1 = -k1;
            k2 = -k2;
          }
#if MEEP_HAVE_CUDA
          if (cuda_session.active()) {
            const bool pml_f = dsig != NO_DIRECTION;
            const bool pml_u = dsigu != NO_DIRECTION;
            const realnum *conductivity =
                s->conductivity[cc][d_c];
            const gpu::detail::bfast_material_fp32 material = {
                pml_f ? s->siginv[dsig] : nullptr,
                pml_f ? static_cast<size_t>(s->sigsize[dsig]) : 0,
                pml_u ? f_u[cc][cmp] : nullptr,
                pml_u ? s->siginv[dsigu] : nullptr,
                pml_u ? static_cast<size_t>(s->sigsize[dsigu]) : 0,
                conductivity ? s->condinv[cc][d_c] : nullptr,
                pml_f && conductivity ? f_cond[cc][cmp] : nullptr};
            if (batch_tiles) {
              const std::vector<gpu::detail::index_space_fp32>
                  &component_spaces = cuda_index_spaces[cc][cmp];
              if (component_spaces.size() == execution_volume_count)
                gpu::detail::resident_step_bfast_batched_fp32(
                    cuda_session.cache(), the_f, f_p, f_m,
                    f_bfast[cc][cmp], gv.ntot(),
                    component_spaces.data(), component_spaces.size(),
                    stride_p, stride_m, static_cast<float>(k1),
                    static_cast<float>(k2), material);
            }
            else
              gpu::detail::resident_step_bfast_fp32(
                  cuda_session.cache(), the_f, f_p, f_m,
                  f_bfast[cc][cmp], gv.ntot(),
                  gpu::detail::make_index_space_fp32(
                      gv, sub_gv.little_owned_corner0(cc),
                      sub_gv.big_corner(), dsig, dsigu),
                  stride_p, stride_m, static_cast<float>(k1),
                  static_cast<float>(k2), material);
          }
          else
#endif
          {
            STEP_BFAST(
                the_f, cc, f_p, f_m, stride_p, stride_m, gv,
                sub_gv.little_owned_corner0(cc), sub_gv.big_corner(),
                Courant, dsig, s->sig[dsig], s->kap[dsig],
                s->siginv[dsig], f_u[cc][cmp], dsigu, s->sig[dsigu],
                s->kap[dsigu], s->siginv[dsigu], dt,
                s->conductivity[cc][d_c], s->condinv[cc][d_c],
                f_cond[cc][cmp], f_bfast[cc][cmp], k1, k2);
          }
        }
      }
    }
  }

  /* In 2d with beta != 0, add beta terms.  This is a trick to model
     an exp(i beta z) z-dependence but without requiring a "3d"
     calculation and without requiring complex fields.  Looking at the
     z=0 2d cross-section, the exp(i beta z) term adds an i \beta
     \hat{z} \times cross-product to the curls, which couples the TE
     and TM polarizations.  However, to avoid complex fields, in the
     case of real fields we implicitly store i*(TM fields) rather than
     the TM fields, in which case the i's cancel in the update
     equations.  (Mathematically, this is equivalent to looking at the
     superposition of the fields at beta and the timereversed fields
     at -beta.)  The nice thing about this is that most calculations
     of flux, energy, etcetera, are insensitive to this implicit "i"
     factor.   For complex fields, we implement i*beta directly. */
  if (gv.dim == D2 && beta != 0) DOCMP for (direction d_c = X; d_c <= Y; d_c = direction(d_c + 1)) {
      component cc = direction_component(first_field_component(ft), d_c);
      component c_g = direction_component(ft == D_stuff ? Hx : Ex, d_c == X ? Y : X);
      realnum *the_f = f[cc][cmp];
      const realnum *g = f[c_g][1 - cmp] ? f[c_g][1 - cmp] : f[c_g][cmp];
      const direction dsig0 = cycle_direction(gv.dim, d_c, 1);
      const direction dsig = s->sigsize[dsig0] > 1 ? dsig0 : NO_DIRECTION;
      const direction dsigu0 = cycle_direction(gv.dim, d_c, 2);
      const direction dsigu = s->sigsize[dsigu0] > 1 ? dsigu0 : NO_DIRECTION;
      const realnum betadt = 2 * pi * beta * dt * (d_c == X ? +1 : -1) *
                             (f[c_g][1 - cmp] ? (ft == D_stuff ? -1 : +1) * (2 * cmp - 1) : 1);
#if MEEP_HAVE_CUDA
      if (cuda_session.active() && the_f && g) {
        const bool pml_f = dsig != NO_DIRECTION;
        const bool pml_u = dsigu != NO_DIRECTION;
        const realnum *conductivity_inverse = s->condinv[cc][d_c];
        const gpu::detail::index_space_fp32 index_space =
            gpu::detail::make_index_space_fp32(
                gv, gv.little_owned_corner0(cc), gv.big_corner(), dsig,
                dsigu);
        const gpu::detail::beta_material_fp32 material = {
            pml_f ? s->siginv[dsig] : nullptr,
            pml_f ? static_cast<size_t>(s->sigsize[dsig]) : 0,
            pml_u ? f_u[cc][cmp] : nullptr,
            pml_u ? s->siginv[dsigu] : nullptr,
            pml_u ? static_cast<size_t>(s->sigsize[dsigu]) : 0,
            conductivity_inverse,
            pml_f && conductivity_inverse ? f_cond[cc][cmp] : nullptr};
        gpu::detail::resident_step_beta_fp32(
            cuda_session.cache(), the_f, g, gv.ntot(), index_space,
            static_cast<float>(betadt), material);
      }
      else
#endif
        STEP_BETA(the_f, cc, g, gv, gv.little_owned_corner0(cc),
                  gv.big_corner(), betadt, dsig, s->siginv[dsig],
                  f_u[cc][cmp], dsigu, s->siginv[dsigu],
                  s->condinv[cc][d_c], f_cond[cc][cmp]);
    }

  // in cylindrical coordinates, we now have to add the i*m/r terms... */
  if (gv.dim == Dcyl && m != 0) DOCMP FOR_FT_COMPONENTS(ft, cc) {
      const direction d_c = component_direction(cc);
      if (f[cc][cmp] && (d_c == R || d_c == Z)) {
        const component c_g = d_c == R ? plus_component[cc] : minus_component[cc];
        const realnum *g = f[c_g][1 - cmp];
        realnum *the_f = f[cc][cmp];
        const realnum *cndinv = s->condinv[cc][d_c];
        realnum *fcnd = f_cond[cc][cmp];
        realnum *fu = f_u[cc][cmp];
        const direction dsig = cycle_direction(gv.dim, d_c, 1);
        const realnum *siginv = s->sigsize[dsig] > 1 ? s->siginv[dsig] : 0;
        const direction dsigu = cycle_direction(gv.dim, d_c, 2);
        const realnum *siginvu = s->sigsize[dsigu] > 1 ? s->siginv[dsigu] : 0;
        const ivec is = gv.little_owned_corner0(cc);

        // Constant factor for the i*m component of the i*m/r term.
        // A factor of 2 is included because in LOOP_OVER_VOL_OWNED0 each
        // increment of the array index in the grid_volume in the R direction
        // corresponds to a change of 0.5*Δr in real space.
        const realnum the_m =
            2 * m * (1 - 2 * cmp) * (1 - 2 * (ft == B_stuff)) * (1 - 2 * (d_c == R)) * Courant;

#if MEEP_HAVE_CUDA
        if (cuda_session.active()) {
          if (!g)
            throw std::logic_error(
                "Meep CUDA cylindrical i*m/r operand is missing");
          const bool pml_f = siginv != nullptr;
          const bool pml_u = siginvu != nullptr;
          const gpu::detail::index_space_fp32 index_space =
              gpu::detail::make_index_space_fp32(
                  gv, is, gv.big_corner(),
                  pml_f ? dsig : NO_DIRECTION,
                  pml_u ? dsigu : NO_DIRECTION);
          const gpu::detail::beta_material_fp32 material = {
              pml_f ? siginv : nullptr,
              pml_f ? static_cast<size_t>(s->sigsize[dsig]) : 0,
              pml_u ? fu : nullptr,
              pml_u ? siginvu : nullptr,
              pml_u ? static_cast<size_t>(s->sigsize[dsigu]) : 0,
              cndinv,
              pml_f && cndinv ? fcnd : nullptr};
          gpu::detail::resident_step_cylindrical_imr_fp32(
              cuda_session.cache(), the_f, g, gv.ntot(), index_space,
              is.in_direction(R), static_cast<float>(the_m), material);
        }
        else
#endif
        {
        // 8 special cases of the same loop (sigh):
        if (siginv) { // PML in f update
          KSTRIDE_DEF(dsig, k, is, gv);
          if (siginvu) { // PML + fu
            KSTRIDE_DEF(dsigu, ku, is, gv);
            if (cndinv) // PML + fu + conductivity
              //////////////////// MOST GENERAL CASE //////////////////////
              PLOOP_OVER_VOL_OWNED0(gv, cc, i) {
                realnum rinv = the_m / (loop_is2 + 2 * loop_i2);
                DEF_k;
                DEF_ku;
                realnum df, dfcnd = rinv * g[i] * cndinv[i];
                fcnd[i] += dfcnd;
                fu[i] += (df = dfcnd * siginv[k]);
                the_f[i] += siginvu[ku] * df;
              }
            /////////////////////////////////////////////////////////////
            else // PML + fu - conductivity
              PLOOP_OVER_VOL_OWNED0(gv, cc, i) {
                realnum rinv = the_m / (loop_is2 + 2 * loop_i2);
                DEF_k;
                DEF_ku;
                realnum df, dfcnd = rinv * g[i];
                fu[i] += (df = dfcnd * siginv[k]);
                the_f[i] += siginvu[ku] * df;
              }
          }
          else {        // PML - fu
            if (cndinv) // PML - fu + conductivity
              PLOOP_OVER_VOL_OWNED0(gv, cc, i) {
                realnum rinv = the_m / (loop_is2 + 2 * loop_i2);
                DEF_k;
                realnum dfcnd = rinv * g[i] * cndinv[i];
                fcnd[i] += dfcnd;
                the_f[i] += dfcnd * siginv[k];
              }
            else // PML - fu - conductivity
              PLOOP_OVER_VOL_OWNED0(gv, cc, i) {
                realnum rinv = the_m / (loop_is2 + 2 * loop_i2);
                DEF_k;
                realnum dfcnd = rinv * g[i];
                the_f[i] += dfcnd * siginv[k];
              }
          }
        }
        else {           // no PML in f update
          if (siginvu) { // no PML + fu
            KSTRIDE_DEF(dsigu, ku, is, gv);
            if (cndinv) // no PML + fu + conductivity
              PLOOP_OVER_VOL_OWNED0(gv, cc, i) {
                realnum rinv = the_m / (loop_is2 + 2 * loop_i2);
                DEF_ku;
                realnum df = rinv * g[i] * cndinv[i];
                fu[i] += df;
                the_f[i] += siginvu[ku] * df;
              }
            else // no PML + fu - conductivity
              PLOOP_OVER_VOL_OWNED0(gv, cc, i) {
                realnum rinv = the_m / (loop_is2 + 2 * loop_i2);
                DEF_ku;
                realnum df = rinv * g[i];
                fu[i] += df;
                the_f[i] += siginvu[ku] * df;
              }
          }
          else {        // no PML - fu
            if (cndinv) // no PML - fu + conductivity
              PLOOP_OVER_VOL_OWNED0(gv, cc, i) {
                realnum rinv = the_m / (loop_is2 + 2 * loop_i2);
                the_f[i] += rinv * g[i] * cndinv[i];
              }
            else // no PML - fu - conductivity
              PLOOP_OVER_VOL_OWNED0(gv, cc, i) {
                realnum rinv = the_m / (loop_is2 + 2 * loop_i2);
                the_f[i] += rinv * g[i];
              }
          }
        }
        }
      }
    }

#if MEEP_HAVE_CUDA
  if (gv.dim == Dcyl && gv.origin_r() == 0.0 &&
      cuda_session.active()) {
    const int nz = gv.nz();
    const size_t radial_row_size = static_cast<size_t>(nz + 1);
    const auto zero_rows = [&](realnum *array, size_t row_count) {
      if (array && row_count)
        gpu::detail::resident_zero_span_fp32(
            cuda_session.cache(), array, gv.ntot(), 0,
            row_count * radial_row_size);
    };
    DOCMP {
      if (m == 0 && ft == D_stuff && f[Dz][cmp]) {
        const direction d_c = Z;
        const direction dsig0 = cycle_direction(gv.dim, d_c, 1);
        const direction dsig =
            s->sigsize[dsig0] > 1 ? dsig0 : NO_DIRECTION;
        const direction dsigu0 = cycle_direction(gv.dim, d_c, 2);
        const direction dsigu =
            s->sigsize[dsigu0] > 1 ? dsigu0 : NO_DIRECTION;
        ivec is = gv.little_owned_corner(Dz);
        ivec ie = gv.big_owned_corner(Dz);
        ie.set_direction(R, 0);
        const bool pml_f = dsig != NO_DIRECTION;
        const bool pml_u = dsigu != NO_DIRECTION;
        // Match the historical r=0 recurrence exactly: it applies
        // conductivity only through the simultaneous PML/conductivity
        // auxiliary, rather than as a conductivity-only update.
        const realnum *conductivity =
            f_cond[Dz][cmp] ? s->conductivity[Dz][Z] : nullptr;
        const gpu::detail::curl_material_fp32 material = {
            pml_f ? s->sig[dsig] : nullptr,
            pml_f ? s->kap[dsig] : nullptr,
            pml_f ? s->siginv[dsig] : nullptr,
            pml_f ? static_cast<size_t>(s->sigsize[dsig]) : 0,
            pml_u ? f_u[Dz][cmp] : nullptr,
            pml_u ? s->sig[dsigu] : nullptr,
            pml_u ? s->kap[dsigu] : nullptr,
            pml_u ? s->siginv[dsigu] : nullptr,
            pml_u ? static_cast<size_t>(s->sigsize[dsigu]) : 0,
            static_cast<float>(dt),
            conductivity,
            conductivity ? s->condinv[Dz][Z] : nullptr,
            pml_f && conductivity ? f_cond[Dz][cmp] : nullptr};
        const gpu::detail::index_space_fp32 index_space =
            gpu::detail::make_index_space_fp32(
                gv, is, ie, dsig, dsigu);
        gpu::detail::resident_step_cylindrical_axis_fp32(
            cuda_session.cache(), f[Dz][cmp], f[Hp][cmp], nullptr,
            gv.ntot(), index_space, 0, 0, 0.0f,
            static_cast<float>(4 * Courant), material);
        zero_rows(f[Dp][cmp], 1);
        zero_rows(f_cond[Dp][cmp], 1);
        zero_rows(f_u[Dp][cmp], 1);
      }
      else if (m == 0 && ft == B_stuff && f[Br][cmp]) {
        zero_rows(f[Br][cmp], 1);
        zero_rows(f_cond[Br][cmp], 1);
        zero_rows(f_u[Br][cmp], 1);
      }
      else if (fabs(m) == 1) {
        const component cc = ft == D_stuff ? Dp : Br;
        const direction d_c = component_direction(cc);
        if (!f[cc][cmp]) continue;
        const realnum *primary = f[ft == D_stuff ? Hr : Ep][cmp];
        const realnum *secondary =
            f[ft == D_stuff ? Hz : Ez][ft == D_stuff ? cmp : 1 - cmp];
        if (!primary || !secondary)
          throw std::logic_error(
              "Meep CUDA cylindrical axis operand is missing");
        const int sd = ft == D_stuff ? +1 : -1;
        const realnum secondary_scale =
            ft == D_stuff ? 2 : (1 - 2 * cmp) * m;
        const direction dsig0 = cycle_direction(gv.dim, d_c, 1);
        const direction dsig =
            s->sigsize[dsig0] > 1 ? dsig0 : NO_DIRECTION;
        const direction dsigu0 = cycle_direction(gv.dim, d_c, 2);
        const direction dsigu =
            s->sigsize[dsigu0] > 1 ? dsigu0 : NO_DIRECTION;
        ivec is = gv.little_owned_corner(cc);
        ivec ie = gv.big_owned_corner(cc);
        ie.set_direction(R, 0);
        const bool pml_f = dsig != NO_DIRECTION;
        const bool pml_u = dsigu != NO_DIRECTION;
        const realnum *conductivity =
            f_cond[cc][cmp] ? s->conductivity[cc][d_c] : nullptr;
        const gpu::detail::curl_material_fp32 material = {
            pml_f ? s->sig[dsig] : nullptr,
            pml_f ? s->kap[dsig] : nullptr,
            pml_f ? s->siginv[dsig] : nullptr,
            pml_f ? static_cast<size_t>(s->sigsize[dsig]) : 0,
            pml_u ? f_u[cc][cmp] : nullptr,
            pml_u ? s->sig[dsigu] : nullptr,
            pml_u ? s->kap[dsigu] : nullptr,
            pml_u ? s->siginv[dsigu] : nullptr,
            pml_u ? static_cast<size_t>(s->sigsize[dsigu]) : 0,
            static_cast<float>(dt),
            conductivity,
            conductivity ? s->condinv[cc][d_c] : nullptr,
            pml_f && conductivity ? f_cond[cc][cmp] : nullptr};
        const gpu::detail::index_space_fp32 index_space =
            gpu::detail::make_index_space_fp32(
                gv, is, ie, dsig, dsigu);
        gpu::detail::resident_step_cylindrical_axis_fp32(
            cuda_session.cache(), f[cc][cmp], primary, secondary,
            gv.ntot(), index_space, -sd,
            ft == D_stuff ? 0 : gv.nz() + 1,
            static_cast<float>(secondary_scale),
            static_cast<float>(sd * Courant), material);
        if (ft == D_stuff) {
          zero_rows(f[Dz][cmp], 1);
          zero_rows(f_cond[Dz][cmp], 1);
          zero_rows(f_u[Dz][cmp], 1);
        }
      }
      else if (m != 0) {
        size_t row_count = 1;
        if (zero_fields_near_cylorigin) {
          row_count = 0;
          const double rmax =
              fabs(m) - int(gv.origin_r() * gv.a + 0.5);
          while (row_count <= static_cast<size_t>(gv.nr()) &&
                 static_cast<double>(row_count) < rmax)
            ++row_count;
        }
        if (ft == D_stuff) {
          zero_rows(f[Dr][cmp], row_count);
          zero_rows(f[Dp][cmp], row_count);
          zero_rows(f[Dz][cmp], row_count);
          zero_rows(f_cond[Dr][cmp], row_count);
          zero_rows(f_cond[Dp][cmp], row_count);
          zero_rows(f_cond[Dz][cmp], row_count);
          zero_rows(f_u[Dr][cmp], row_count);
          zero_rows(f_u[Dp][cmp], row_count);
          zero_rows(f_u[Dz][cmp], row_count);
        }
        else {
          zero_rows(f[Br][cmp], row_count);
          zero_rows(f[Bp][cmp], row_count);
          zero_rows(f[Bz][cmp], row_count);
          zero_rows(f_cond[Br][cmp], row_count);
          zero_rows(f_cond[Bp][cmp], row_count);
          zero_rows(f_cond[Bz][cmp], row_count);
          zero_rows(f_u[Br][cmp], row_count);
          zero_rows(f_u[Bp][cmp], row_count);
          zero_rows(f_u[Bz][cmp], row_count);
        }
      }
    }
  }
#endif

#define ZERO_Z(array) memset(array, 0, sizeof(realnum) * (nz + 1));

  // deal with annoying r=0 boundary conditions for m=0 and m=1
  if (gv.dim == Dcyl && gv.origin_r() == 0.0 &&
      !cuda_session.active()) DOCMP {
      const int nz = gv.nz();
      if (m == 0 && ft == D_stuff && f[Dz][cmp]) {
        // d(Dz)/dt = (1/r) * d(r*Hp)/dr
        const realnum *g = f[Hp][cmp];
        const realnum *cndinv = s->condinv[Dz][Z];
        const realnum *cnd = s->conductivity[Dz][Z];
        realnum *fcnd = f_cond[Dz][cmp];
        const direction dsig = cycle_direction(gv.dim, Z, 1);
        const realnum *siginv = s->sigsize[dsig] > 1 ? s->siginv[dsig] : 0;
        const realnum *sig = s->sigsize[dsig] > 1 ? s->sig[dsig] : 0;
        const realnum *kap = s->sigsize[dsig] > 1 ? s->kap[dsig] : 0;
        const direction dsigu = cycle_direction(gv.dim, Z, 2);
        const realnum *siginvu = s->sigsize[dsigu] > 1 ? s->siginv[dsigu] : 0;
        const realnum *sigu = s->sigsize[dsigu] > 1 ? s->sig[dsigu] : 0;
        const realnum *kapu = s->sigsize[dsigu] > 1 ? s->kap[dsigu] : 0;
        realnum *fu = siginvu && f_u[Dz][cmp] ? f[Dz][cmp] : 0;
        realnum *the_f = fu ? f_u[Dz][cmp] : f[Dz][cmp];
        realnum dt2 = dt * 0.5;

        ivec is = gv.little_owned_corner(Dz);
        ivec ie = gv.big_owned_corner(Dz);
        ie.set_direction(R, 0);
        LOOP_OVER_IVECS(gv, is, ie, i) {
          realnum fprev = the_f[i];
          realnum dfcnd = g[i] * (Courant * 4);
          if (fcnd) {
            realnum fcnd_prev = fcnd[i];
            fcnd[i] = ((1 - dt2 * cnd[i]) * fcnd[i] + dfcnd) * cndinv[i];
            dfcnd = fcnd[i] - fcnd_prev;
          }
          KSTRIDE_DEF(dsig, k, is, gv);
          DEF_k;
          KSTRIDE_DEF(dsigu, ku, is, gv);
          DEF_ku;
          the_f[i] = ((kap ? kap[k] - sig[k] : 1) * the_f[i] + dfcnd) * (siginv ? siginv[k] : 1);
          if (fu)
            fu[i] = siginvu[ku] * ((kapu ? kapu[ku] - sigu[ku] : 1) * fu[i] + the_f[i] - fprev);
        }
        ZERO_Z(f[Dp][cmp]);
        if (f_cond[Dp][cmp]) ZERO_Z(f_cond[Dp][cmp]);
        if (f_u[Dp][cmp]) ZERO_Z(f_u[Dp][cmp]);
      }
      else if (m == 0 && ft == B_stuff && f[Br][cmp]) {
        ZERO_Z(f[Br][cmp]);
        if (f_cond[Br][cmp]) ZERO_Z(f_cond[Br][cmp]);
        if (f_u[Br][cmp]) ZERO_Z(f_u[Br][cmp]);
      }
      else if (fabs(m) == 1) {
        // D_stuff: d(Dp)/dt = d(Hr)/dz - d(Hz)/dr
        // B_stuff: d(Br)/dt = d(Ep)/dz - i*m*Ez/r
        component cc = ft == D_stuff ? Dp : Br;
        direction d_c = component_direction(cc);
        if (!f[cc][cmp]) continue;
        const realnum *f_p = f[ft == D_stuff ? Hr : Ep][cmp];
        const realnum *f_m = ft == D_stuff ? f[Hz][cmp] : (f[Ez][1 - cmp] + (nz + 1));
        const realnum *cndinv = s->condinv[cc][d_c];
        const realnum *cnd = s->conductivity[cc][d_c];
        realnum *fcnd = f_cond[cc][cmp];
        const direction dsig = cycle_direction(gv.dim, d_c, 1);
        const realnum *siginv = s->sigsize[dsig] > 1 ? s->siginv[dsig] : 0;
        const realnum *sig = s->sigsize[dsig] > 1 ? s->sig[dsig] : 0;
        const realnum *kap = s->sigsize[dsig] > 1 ? s->kap[dsig] : 0;
        const direction dsigu = cycle_direction(gv.dim, d_c, 2);
        const realnum *siginvu = s->sigsize[dsigu] > 1 ? s->siginv[dsigu] : 0;
        const realnum *sigu = s->sigsize[dsigu] > 1 ? s->sig[dsigu] : 0;
        const realnum *kapu = s->sigsize[dsigu] > 1 ? s->kap[dsigu] : 0;
        realnum *fu = siginvu && f_u[cc][cmp] ? f[cc][cmp] : 0;
        realnum *the_f = fu ? f_u[cc][cmp] : f[cc][cmp];
        int sd = ft == D_stuff ? +1 : -1;
        realnum f_m_mult = ft == D_stuff ? 2 : (1 - 2 * cmp) * m;
        realnum dt2 = dt * 0.5;

        ivec is = gv.little_owned_corner(cc);
        ivec ie = gv.big_owned_corner(cc);
        ie.set_direction(R, 0);
        LOOP_OVER_IVECS(gv, is, ie, i) {
          realnum fprev = the_f[i];
          realnum dfcnd = (sd * Courant) * (f_p[i] - f_p[i - sd] - f_m_mult * f_m[i]);
          if (fcnd) {
            realnum fcnd_prev = fcnd[i];
            fcnd[i] = ((1 - dt2 * cnd[i]) * fcnd[i] + dfcnd) * cndinv[i];
            dfcnd = fcnd[i] - fcnd_prev;
          }
          KSTRIDE_DEF(dsig, k, is, gv);
          DEF_k;
          KSTRIDE_DEF(dsigu, ku, is, gv);
          DEF_ku;
          the_f[i] = ((kap ? kap[k] - sig[k] : 1) * the_f[i] + dfcnd) * (siginv ? siginv[k] : 1);
          if (fu)
            fu[i] = siginvu[ku] * ((kapu ? kapu[ku] - sigu[ku] : 1) * fu[i] + the_f[i] - fprev);
        }
        if (ft == D_stuff) {
          ZERO_Z(f[Dz][cmp]);
          if (f_cond[Dz][cmp]) ZERO_Z(f_cond[Dz][cmp]);
          if (f_u[Dz][cmp]) ZERO_Z(f_u[Dz][cmp]);
        }
      }
      else if (m != 0) {                  // m != {0,+1,-1}
        if (zero_fields_near_cylorigin) { /* default behavior */
          /* I seem to recall David telling me that this was for numerical
             stability of some sort - the larger m is, the farther from
             the origin we need to be before we can use nonzero fields
             ... note that this is a fixed number of pixels for a given m,
             so it should still converge.  Still, this is weird...

             Update: experimentally, this seems to indeed be important
             for stability.  Setting these fields to zero, it seems to be
             stable with a Courant number < 0.62 or so for all m.  Without
             this, it becomes unstable unless we set the Courant number to
             about 1 / (|m| + 0.5) or less.

             Cons: setting fields near the origin to identically zero is
             somewhat unexpected for users, and probably spoils 2nd-order
             accuracy, and may not fix all stability issues anyway (based
             on anecdotal evidence from Alex M. of having to reduce Courant
             for large m). */
          double rmax = fabs(m) - int(gv.origin_r() * gv.a + 0.5);
          if (ft == D_stuff)
            for (int r = 0; r <= gv.nr() && r < rmax; r++) {
              const int ir = r * (nz + 1);
              if (f[Dr][cmp]) ZERO_Z(f[Dr][cmp] + ir);
              ZERO_Z(f[Dp][cmp] + ir);
              ZERO_Z(f[Dz][cmp] + ir);
              if (f_cond[Dr][cmp]) ZERO_Z(f_cond[Dr][cmp] + ir);
              if (f_cond[Dp][cmp]) ZERO_Z(f_cond[Dp][cmp] + ir);
              if (f_cond[Dz][cmp]) ZERO_Z(f_cond[Dz][cmp] + ir);
              if (f_u[Dr][cmp]) ZERO_Z(f_u[Dr][cmp] + ir);
              if (f_u[Dp][cmp]) ZERO_Z(f_u[Dp][cmp] + ir);
              if (f_u[Dz][cmp]) ZERO_Z(f_u[Dz][cmp] + ir);
            }
          else
            for (int r = 0; r <= gv.nr() && r < rmax; r++) {
              const int ir = r * (nz + 1);
              ZERO_Z(f[Br][cmp] + ir);
              if (f[Bp][cmp]) ZERO_Z(f[Bp][cmp] + ir);
              if (f[Bz][cmp]) ZERO_Z(f[Bz][cmp] + ir);
              if (f_cond[Br][cmp]) ZERO_Z(f_cond[Br][cmp] + ir);
              if (f_cond[Bp][cmp]) ZERO_Z(f_cond[Bp][cmp] + ir);
              if (f_cond[Bz][cmp]) ZERO_Z(f_cond[Bz][cmp] + ir);
              if (f_u[Br][cmp]) ZERO_Z(f_u[Br][cmp] + ir);
              if (f_u[Bp][cmp]) ZERO_Z(f_u[Bp][cmp] + ir);
              if (f_u[Bz][cmp]) ZERO_Z(f_u[Bz][cmp] + ir);
            }
        }
        else {
          /* Without David's hack: just set boundary conditions at r=0.
             This seems to be unstable unless we make the Courant number
             around 1 / (|m| + 0.5) or smaller.  Pros: probably maintains
             2nd-order accuracy, is more sane for r near zero.  Cons:
             1/(|m|+0.5) is purely empirical (no theory yet), and I'm not
             sure how universal it is.  Makes higher m's more expensive. */
          if (ft == D_stuff) {
            if (f[Dr][cmp]) ZERO_Z(f[Dr][cmp]);
            ZERO_Z(f[Dp][cmp]);
            ZERO_Z(f[Dz][cmp]);
            if (f_cond[Dr][cmp]) ZERO_Z(f_cond[Dr][cmp]);
            if (f_cond[Dp][cmp]) ZERO_Z(f_cond[Dp][cmp]);
            if (f_cond[Dz][cmp]) ZERO_Z(f_cond[Dz][cmp]);
            if (f_u[Dr][cmp]) ZERO_Z(f_u[Dr][cmp]);
            if (f_u[Dp][cmp]) ZERO_Z(f_u[Dp][cmp]);
            if (f_u[Dz][cmp]) ZERO_Z(f_u[Dz][cmp]);
          }
          else {
            ZERO_Z(f[Br][cmp]);
            if (f[Bp][cmp]) ZERO_Z(f[Bp][cmp]);
            if (f[Bz][cmp]) ZERO_Z(f[Bz][cmp]);
            if (f_cond[Br][cmp]) ZERO_Z(f_cond[Br][cmp]);
            if (f_cond[Bp][cmp]) ZERO_Z(f_cond[Bp][cmp]);
            if (f_cond[Bz][cmp]) ZERO_Z(f_cond[Bz][cmp]);
            if (f_u[Br][cmp]) ZERO_Z(f_u[Br][cmp]);
            if (f_u[Bp][cmp]) ZERO_Z(f_u[Bp][cmp]);
            if (f_u[Bz][cmp]) ZERO_Z(f_u[Bz][cmp]);
          }
        }
      }
    }

  cuda_session.finish();

  return allocated_u;
}

} // namespace meep
