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

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "meep.hpp"
#include "meep_internals.hpp"
#include "gpu_backend_internal.hpp"

#include "config.h"

#define RESTRICT

using namespace std;

namespace meep {

namespace {

void append_owner_environment_value(
    std::string *signature, const char *name) {
  const char *value = std::getenv(name);
  signature->append(name);
  signature->push_back('=');
  if (!value) {
    signature->append("unset;");
    return;
  }
  const std::size_t length = std::strlen(value);
  signature->append(std::to_string(length));
  signature->push_back(':');
  signature->append(value, length);
  signature->push_back(';');
}

std::string gpu_owner_environment_signature() {
  std::string signature;
  append_owner_environment_value(&signature, "MEEP_GPU_DEVICE");
  append_owner_environment_value(
      &signature, "MEEP_GPU_ALLOW_OVERSUBSCRIBE");
  append_owner_environment_value(
      &signature, "MEEP_GPU_AUTO_MIN_CELLS");
  append_owner_environment_value(
      &signature, "MEEP_GPU_MPI_TRANSPORT");
  append_owner_environment_value(
      &signature, "MEEP_GPU_MPI_COMPLETION");
  return signature;
}

std::string gpu_transport_environment_signature() {
  std::string signature;
  append_owner_environment_value(
      &signature, "MEEP_GPU_MPI_TRANSPORT");
  append_owner_environment_value(
      &signature, "MEEP_GPU_MPI_COMPLETION");
  return signature;
}

class distributed_step_failure_guard {
public:
  explicit distributed_step_failure_guard(bool armed)
      : armed_(armed), completed_(false) {}

  void dismiss() noexcept { completed_ = true; }

  ~distributed_step_failure_guard() {
    // An explicit completion flag works in C++11 and does not mistake a
    // successful step invoked during an outer stack unwind for a new
    // rank-local failure.
    if (armed_ && !completed_)
      meep::abort(
          "rank-local failure during a distributed time step; aborting the "
          "communicator to prevent an MPI collective or boundary deadlock");
  }

private:
  bool armed_;
  bool completed_;
};

// step_boundaries enters the legacy Boundaries timing sink before selecting
// the resident CUDA path.  Match the CPU path while a comms_manager completes
// its outstanding messages, then restore Boundaries before any post-exchange
// CUDA work or exception reaches the outer cleanup handler.
class boundary_mpi_completion_scope {
public:
  explicit boundary_mpi_completion_scope(fields &owner) : owner_(owner) {
    owner_.finished_working();
    owner_.am_now_working_on(MpiOneTime);
  }

  ~boundary_mpi_completion_scope() {
    owner_.finished_working();
    owner_.am_now_working_on(Boundaries);
  }

  boundary_mpi_completion_scope(const boundary_mpi_completion_scope &) = delete;
  boundary_mpi_completion_scope &operator=(const boundary_mpi_completion_scope &) = delete;

private:
  fields &owner_;
};

bool boundary_eh_overlap_opted_in() noexcept {
  // This optimization is intentionally fail-closed: authoritative A/B did
  // not show a material, stable improvement.  Only the exact value "1"
  // opts in, and the legacy disable switch always wins on conflicts.
  const char *enabled =
      std::getenv("MEEP_GPU_ENABLE_BOUNDARY_EH_OVERLAP");
  return enabled && enabled[0] == '1' && enabled[1] == '\0' &&
         std::getenv("MEEP_GPU_DISABLE_BOUNDARY_EH_OVERLAP") == nullptr;
}

bool halo_curl_overlap_opted_in() noexcept {
  // Keep the new path strictly opt-in until paired correctness/performance
  // evidence meets its promotion gate. Any disable-variable presence wins.
  const char *enabled =
      std::getenv("MEEP_GPU_ENABLE_HALO_CURL_OVERLAP");
  return enabled && enabled[0] == '1' && enabled[1] == '\0' &&
         std::getenv("MEEP_GPU_DISABLE_HALO_CURL_OVERLAP") == nullptr;
}

#if MEEP_HAVE_CUDA
void boundary_callback_test_delay() {
  const char *value =
      std::getenv("MEEP_GPU_TEST_BOUNDARY_CALLBACK_DELAY_US");
  if (!value || !*value) return;
  char *end = nullptr;
  const long delay_us = std::strtol(value, &end, 10);
  if (!end || *end != '\0' || delay_us < 0 || delay_us > 1000000)
    throw std::invalid_argument(
        "MEEP_GPU_TEST_BOUNDARY_CALLBACK_DELAY_US must be an integer from "
        "0 through 1000000");
  if (delay_us > 0)
    std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
}

#endif

} // namespace

#if MEEP_HAVE_CUDA
struct boundary_array_span {
  std::uintptr_t begin;
  std::uintptr_t end;
};

static std::vector<boundary_array_span> prepare_boundary_arrays(
    fields_chunk *chunk, gpu::detail::resident_cache *cache) {
  if (!chunk || !cache)
    throw std::invalid_argument(
        "Meep CUDA boundary chunk and cache must be non-null");
  const size_t ntot = chunk->gv.ntot();
  std::vector<realnum *> arrays;
  auto add_array = [&arrays](realnum *base) {
    if (base) arrays.push_back(base);
  };

  FOR_COMPONENTS(c) for (int cmp = 0; cmp < 2; ++cmp) {
    realnum *const field_arrays[] = {
        chunk->f[c][cmp],          chunk->f_u[c][cmp],
        chunk->f_w[c][cmp],        chunk->f_cond[c][cmp],
        chunk->f_bfast[c][cmp],    chunk->f_minus_p[c][cmp],
        chunk->f_w_prev[c][cmp],   chunk->f_backup[c][cmp],
        chunk->f_u_backup[c][cmp], chunk->f_w_backup[c][cmp],
        chunk->f_cond_backup[c][cmp],
        chunk->f_bfast_backup[c][cmp]};
    for (realnum *base : field_arrays)
      add_array(base);
  }

  FOR_FIELD_TYPES(pft) {
    for (polarization_state *state = chunk->pol[pft]; state;
         state = state->next) {
      if (!state->data || !state->s) continue;
      const multilevel_susceptibility *multilevel =
          dynamic_cast<const multilevel_susceptibility *>(state->s);
      if (multilevel &&
          typeid(*state->s) == typeid(multilevel_susceptibility) &&
          !multilevel->prepare_boundary_cuda(cache, state->data))
        throw std::logic_error(
            "preflighted CUDA multilevel boundary state was not prepared");
      FOR_COMPONENTS(c) {
        const int internal_count =
            state->s->num_internal_notowned_needed(c, state->data);
        for (int k = 0; k < internal_count; ++k) {
          realnum *base =
              state->s->internal_notowned_ptr(k, c, 0, state->data);
          add_array(base);
        }
        const int complex_internal_count =
            state->s->num_cinternal_notowned_needed(c, state->data);
        for (int k = 0; k < complex_internal_count; ++k)
          for (int cmp = 0; cmp < 2; ++cmp) {
            realnum *base = state->s->cinternal_notowned_ptr(
                k, c, cmp, 0, state->data);
            add_array(base);
          }
      }
    }
  }
  std::sort(arrays.begin(), arrays.end(), std::less<realnum *>());
  arrays.erase(std::unique(arrays.begin(), arrays.end()), arrays.end());

  if (ntot >
      std::numeric_limits<std::uintptr_t>::max() / sizeof(realnum))
    throw std::overflow_error("Meep CUDA boundary array size overflow");
  const std::uintptr_t bytes =
      static_cast<std::uintptr_t>(ntot * sizeof(realnum));
  std::vector<boundary_array_span> spans;
  spans.reserve(arrays.size());
  for (realnum *base : arrays) {
    const std::uintptr_t begin =
        reinterpret_cast<std::uintptr_t>(base);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin)
      throw std::overflow_error("Meep CUDA boundary array address overflow");
    gpu::detail::resident_ensure_span_fp32(cache, base, ntot);
    spans.push_back({begin, begin + bytes});
  }
  std::sort(
      spans.begin(), spans.end(),
      [](const boundary_array_span &left,
         const boundary_array_span &right) {
        return left.begin < right.begin;
      });
  return spans;
}

static bool boundary_pointer_is_prepared(
    const std::vector<boundary_array_span> &spans,
    const realnum *point) {
  if (!point) return false;
  const std::uintptr_t address =
      reinterpret_cast<std::uintptr_t>(point);
  const auto span = std::upper_bound(
      spans.begin(), spans.end(), address,
      [](std::uintptr_t value, const boundary_array_span &candidate) {
        return value < candidate.begin;
      });
  if (span == spans.begin()) return false;
  const boundary_array_span &candidate = *(span - 1);
  return address >= candidate.begin && address < candidate.end &&
         (address - candidate.begin) % sizeof(realnum) == 0;
}
#endif

struct fields::gpu_boundary_topology {
  struct remote_send {
    int other_proc_id;
    int tag;
    size_t transfer_size;
    gpu::detail::boundary_exchange_buffer *buffer;
    std::vector<gpu::detail::resident_cache *> source_caches;
    std::vector<const float *> sources;
  };

  struct remote_receive {
    int other_proc_id;
    int tag;
    size_t transfer_size;
    const void *buffer_token;
    gpu::detail::boundary_exchange_buffer *primary_buffer;
    gpu::detail::boundary_exchange_buffer *secondary_buffer;
    gpu::detail::boundary_exchange_buffer *buffer;
    bool select_secondary_next;
    std::vector<gpu::detail::remote_boundary_operation_fp32> operations;
  };

  std::vector<gpu::detail::resident_cache *> chunk_caches;
  std::vector<remote_send> remote_sends;
  std::vector<remote_receive> remote_receives;
  std::unique_ptr<comms_manager> manager;
  gpu::detail::boundary_phase_graph *phase_graph = nullptr;
  bool phase_graph_creation_attempted = false;
  const std::uint64_t descriptor_replay_generation;

  gpu_boundary_topology()
      : descriptor_replay_generation(
            next_descriptor_replay_generation()) {}

  gpu::detail::boundary_descriptor_replay_token
  descriptor_replay_token() const noexcept {
    return gpu::detail::boundary_descriptor_replay_token(
        this, descriptor_replay_generation);
  }

  ~gpu_boundary_topology() {
    gpu::detail::destroy_boundary_phase_graph(phase_graph);
  }

private:
  static std::uint64_t next_descriptor_replay_generation() noexcept {
    static std::atomic<std::uint64_t> generation(0);
    // Once UINT64_MAX is issued, the CAS allocator remains saturated and all
    // later topologies receive zero. Zero is permanently full-validation-only
    // instead of wrapping to a forgeable earlier generation.
    return gpu::detail::next_saturating_boundary_topology_generation(
        &generation);
  }
};

struct gpu_step_preflight_token {
  const void *identity;
  std::size_t metadata;
  std::size_t secondary_metadata;

  bool operator==(const gpu_step_preflight_token &other) const {
    return identity == other.identity && metadata == other.metadata &&
           secondary_metadata == other.secondary_metadata;
  }
};

struct fields::gpu_boundary_topology_registry {
  struct owner_state {
    std::array<gpu_boundary_topology *, NUM_FIELD_TYPES> topologies{};
    int validated_device = -1;
    std::uint64_t validated_device_generation =
        std::numeric_limits<std::uint64_t>::max();
    std::uint64_t validated_backend_generation =
        std::numeric_limits<std::uint64_t>::max();
    std::string validated_backend_environment_signature;
    gpu::backend_mode execution_backend = gpu::backend_mode::cpu;
    std::string execution_diagnostic =
        "fields owner has not completed backend preflight";
    std::uint64_t validated_step_backend_generation =
        std::numeric_limits<std::uint64_t>::max();
    std::vector<gpu_step_preflight_token> validated_step_signature;
    bool cpu_feature_fallback = false;
    std::vector<gpu_step_preflight_token>
        feature_fallback_signature;
    bool step_preflight_consensus = false;
    bool transport_validated = false;
    std::string validated_transport_environment_signature;
  };
  std::mutex mutex;
  std::unordered_map<const fields *, owner_state> owners;
};

static std::vector<gpu_step_preflight_token> gpu_step_preflight_signature(
    const fields *owner) {
  const char *failure_rank =
      std::getenv("MEEP_GPU_TEST_PREFLIGHT_SIGNATURE_FAILURE_RANK");
  if (failure_rank && *failure_rank) {
    char *end = nullptr;
    const long parsed = std::strtol(failure_rank, &end, 10);
    if (!end || *end != '\0' || parsed < 0 ||
        parsed > std::numeric_limits<int>::max())
      throw std::invalid_argument(
          "MEEP_GPU_TEST_PREFLIGHT_SIGNATURE_FAILURE_RANK must be a "
          "nonnegative MPI rank");
    if (my_global_rank() == static_cast<int>(parsed))
      throw std::runtime_error(
          "injected rank-local CUDA preflight signature failure");
  }
  typedef gpu_step_preflight_token token;
  std::vector<token> signature;
  signature.push_back(token{owner->fluxes, 0, 0});
  for (int chunk_index = 0; chunk_index < owner->num_chunks;
       ++chunk_index) {
    fields_chunk *chunk = owner->chunks[chunk_index];
    if (!chunk->is_mine()) continue;
    signature.push_back(token{chunk, 0, 0});
    for (dft_chunk *dft = chunk->dft_chunks; dft;
         dft = dft->next_in_chunk)
      // omega is historical public state.  A user can resize it without
      // changing the dft_chunk address, so include the size checked by
      // cuda_dft_preflight instead of treating pointer identity as a
      // sufficient eligibility signature.
      signature.push_back(token{dft, dft->omega.size(), dft->N});
    FOR_FIELD_TYPES(ft)
      for (polarization_state *state = chunk->pol[ft]; state;
           state = state->next) {
        signature.push_back(
            token{state, static_cast<std::size_t>(ft), 0});
        signature.push_back(token{state->s, 0, 0});
        if (state->s)
          FOR_COMPONENTS(c) FOR_DIRECTIONS(d)
            // sigma and trivial_sigma are historical public state used by
            // gyrotropic/multilevel eligibility.  Their topology can change
            // without replacing the susceptibility object, so pointer
            // identity alone is not a sufficient cache key.
            signature.push_back(token{
                state->s->sigma[c][d], static_cast<std::size_t>(c),
                (static_cast<std::size_t>(d) << 1) |
                    static_cast<std::size_t>(
                        state->s->trivial_sigma[c][d])});
      }
  }
  return signature;
}

fields::gpu_boundary_topology_registry &
fields::gpu_boundary_topology_registry_instance() {
  // Fields instances can have static storage duration. Keep this ABI-private
  // registry process-long so late destruction never observes a dead registry.
  static gpu_boundary_topology_registry *registry =
      new gpu_boundary_topology_registry;
  return *registry;
}

fields::gpu_boundary_topology *
fields::get_gpu_boundary_topology(field_type ft) const {
  gpu_boundary_topology_registry &registry =
      gpu_boundary_topology_registry_instance();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto owner = registry.owners.find(this);
  return owner == registry.owners.end()
             ? nullptr
             : owner->second.topologies[static_cast<std::size_t>(ft)];
}

void fields::replace_gpu_boundary_topology(
    field_type ft, gpu_boundary_topology *topology) {
  gpu_boundary_topology_registry &registry =
      gpu_boundary_topology_registry_instance();
  gpu_boundary_topology *replaced = nullptr;
  {
    std::lock_guard<std::mutex> lock(registry.mutex);
    std::array<gpu_boundary_topology *, NUM_FIELD_TYPES> &entries =
        registry.owners[this].topologies;
    gpu_boundary_topology *&entry =
        entries[static_cast<std::size_t>(ft)];
    replaced = entry;
    entry = topology;
  }
  // Graph teardown may synchronize a CUDA event. Never hold the topology
  // registry mutex across that blocking runtime call.
  delete replaced;
}

bool fields::gpu_backend_preflight_is_cached() const {
  const std::uint64_t generation =
      gpu::detail::backend_generation();
  const std::string environment_signature =
      gpu_owner_environment_signature();
  gpu_boundary_topology_registry &registry =
      gpu_boundary_topology_registry_instance();
  bool locally_cached = false;
  bool local_cuda_execution = false;
  bool local_feature_fallback = false;
  bool locally_step_cached = true;
  std::string local_diagnostic;
  std::vector<gpu_step_preflight_token> step_signature;
  {
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto owner = registry.owners.find(this);
    locally_cached =
        owner != registry.owners.end() &&
        owner->second.validated_backend_generation == generation &&
        owner->second.validated_backend_environment_signature ==
            environment_signature;
    if (locally_cached) {
      local_cuda_execution =
          owner->second.execution_backend == gpu::backend_mode::cuda;
      local_diagnostic = owner->second.execution_diagnostic;
      local_feature_fallback = owner->second.cpu_feature_fallback;
    }
  }
  if (locally_cached &&
      (local_cuda_execution || local_feature_fallback))
    step_signature = gpu_step_preflight_signature(this);
  {
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto owner = registry.owners.find(this);
    if (locally_cached && local_cuda_execution)
      locally_step_cached =
          owner != registry.owners.end() &&
          !changed_materials &&
          owner->second.validated_step_backend_generation == generation &&
          owner->second.validated_step_signature == step_signature;
    else if (locally_cached && local_feature_fallback)
      locally_step_cached =
          owner != registry.owners.end() &&
          !changed_materials &&
          owner->second.feature_fallback_signature == step_signature;
  }
  // Preserve the exact distributed cache/backend-consensus contract with a
  // single collective.  The previous implementation issued three scalar
  // Allreduces on every time step (cache-valid, any-CUDA, every-CUDA), which
  // made an otherwise cached preflight a global three-rendezvous fast path.
  // Logical-AND of both the CUDA bit and its complement distinguishes the
  // all-CUDA, all-CPU, and mixed-backend states without a separate OR.
  const int local_consensus[] = {
      locally_cached && locally_step_cached ? 1 : 0,
      local_cuda_execution ? 1 : 0,
      local_cuda_execution ? 0 : 1};
  int distributed_consensus[3] = {0, 0, 0};
  and_to_all(local_consensus, distributed_consensus, 3);
  const bool every_cached = distributed_consensus[0] != 0;
  if (every_cached) {
    const bool every_cuda = distributed_consensus[1] != 0;
    const bool every_cpu = distributed_consensus[2] != 0;
    if (every_cuda == every_cpu)
      throw std::runtime_error(
          "cached fields-owner backend decisions differ across MPI ranks");
    gpu::detail::activate_backend_for_owner(
        every_cuda ? gpu::backend_mode::cuda : gpu::backend_mode::cpu,
        local_diagnostic);
  }
  {
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.owners[this].step_preflight_consensus =
        every_cached && locally_step_cached && local_cuda_execution;
  }
  return every_cached;
}

void fields::cache_gpu_backend_preflight(
    bool cuda_execution,
    const std::string &diagnostic,
    bool feature_fallback) const {
  const gpu::backend_mode execution_backend =
      cuda_execution ? gpu::backend_mode::cuda
                     : gpu::backend_mode::cpu;
  const std::uint64_t generation =
      gpu::detail::backend_generation();
  const std::string environment_signature =
      gpu_owner_environment_signature();
  std::vector<gpu_step_preflight_token> feature_signature;
  if (!cuda_execution && feature_fallback)
    feature_signature = gpu_step_preflight_signature(this);
  gpu_boundary_topology_registry &registry =
      gpu_boundary_topology_registry_instance();
  std::lock_guard<std::mutex> lock(registry.mutex);
  gpu_boundary_topology_registry::owner_state &state =
      registry.owners[this];
  state.validated_backend_generation = generation;
  state.validated_backend_environment_signature =
      environment_signature;
  state.execution_backend = execution_backend;
  state.execution_diagnostic = diagnostic;
  state.cpu_feature_fallback =
      execution_backend == gpu::backend_mode::cpu && feature_fallback;
  if (state.cpu_feature_fallback)
    state.feature_fallback_signature.swap(feature_signature);
  else
    state.feature_fallback_signature.clear();
  if (execution_backend == gpu::backend_mode::cpu) {
    state.validated_step_backend_generation =
        std::numeric_limits<std::uint64_t>::max();
    state.validated_step_signature.clear();
  }
  state.step_preflight_consensus = false;
}

bool fields::gpu_cuda_execution_selected() const {
  const std::uint64_t generation = gpu::detail::backend_generation();
  const std::string environment_signature =
      gpu_owner_environment_signature();
  gpu_boundary_topology_registry &registry =
      gpu_boundary_topology_registry_instance();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto owner = registry.owners.find(this);
  return owner != registry.owners.end() &&
         owner->second.validated_backend_generation ==
             generation &&
         owner->second.validated_backend_environment_signature ==
             environment_signature &&
         owner->second.execution_backend == gpu::backend_mode::cuda;
}

const char *fields::gpu_execution_diagnostic() const {
  static thread_local std::string diagnostic;
  const std::uint64_t generation = gpu::detail::backend_generation();
  const std::string environment_signature =
      gpu_owner_environment_signature();
  gpu_boundary_topology_registry &registry =
      gpu_boundary_topology_registry_instance();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto owner = registry.owners.find(this);
  diagnostic =
      owner == registry.owners.end()
          ? std::string("fields owner has not completed backend preflight")
          : owner->second.validated_backend_generation != generation ||
                    owner->second.validated_backend_environment_signature !=
                        environment_signature
                ? std::string(
                      "fields owner backend decision was invalidated by a "
                      "process backend, device, or automatic-policy change")
                : owner->second.execution_diagnostic;
  return diagnostic.c_str();
}

void fields::prepare_gpu_owner_for_cpu() const {
  for (int i = 0; i < num_chunks; ++i)
    if (chunks[i]->is_mine()) {
      // Publish this owner's authoritative device state transactionally
      // before the noexcept destruction path releases its mirrors.  Never
      // synchronize or invalidate another live fields owner's cache merely
      // because this owner selected CPU.
      gpu::detail::sync_resident_cache_for_owner(chunks[i]);
      gpu::detail::destroy_resident_cache_for_owner(chunks[i]);
    }
}

bool fields::gpu_step_preflight_is_cached() const {
  const std::uint64_t generation = gpu::detail::backend_generation();
  const std::vector<gpu_step_preflight_token> signature =
      gpu_step_preflight_signature(this);
  gpu_boundary_topology_registry &registry =
      gpu_boundary_topology_registry_instance();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto owner = registry.owners.find(this);
  return owner != registry.owners.end() &&
         owner->second.step_preflight_consensus &&
         owner->second.validated_step_backend_generation == generation &&
         owner->second.validated_step_signature == signature;
}

void fields::cache_gpu_step_preflight() const {
  const std::uint64_t generation = gpu::detail::backend_generation();
  std::vector<gpu_step_preflight_token> signature =
      gpu_step_preflight_signature(this);
  gpu_boundary_topology_registry &registry =
      gpu_boundary_topology_registry_instance();
  std::lock_guard<std::mutex> lock(registry.mutex);
  gpu_boundary_topology_registry::owner_state &state =
      registry.owners[this];
  state.validated_step_backend_generation = generation;
  state.validated_step_signature.swap(signature);
  state.step_preflight_consensus = true;
}

void fields::validate_gpu_mpi_transport() const {
  const std::string environment_signature =
      gpu_transport_environment_signature();
  gpu_boundary_topology_registry &registry =
      gpu_boundary_topology_registry_instance();
  bool locally_validated = false;
  bool local_environment_changed_after_validation = false;
  {
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto owner = registry.owners.find(this);
    local_environment_changed_after_validation =
        owner != registry.owners.end() &&
        owner->second.transport_validated &&
        owner->second.validated_transport_environment_signature !=
            environment_signature;
    locally_validated =
        owner != registry.owners.end() &&
        owner->second.transport_validated &&
        owner->second.validated_transport_environment_signature ==
            environment_signature;
  }
  if (and_to_all(locally_validated)) return;
  if (or_to_all(local_environment_changed_after_validation))
    throw std::runtime_error(
        "MEEP_GPU_MPI_TRANSPORT or MEEP_GPU_MPI_COMPLETION changed after "
        "fields-owner transport preflight; dynamic MPI transport changes "
        "are not supported");

  validate_distributed_mpi_transport();

  std::lock_guard<std::mutex> lock(registry.mutex);
  registry.owners[this].transport_validated = true;
  registry.owners[this].validated_transport_environment_signature =
      environment_signature;
}

void fields::validate_gpu_device_assignment() const {
  const int device = gpu::selected_device();
  const std::uint64_t generation =
      gpu::detail::backend_generation();
  gpu_boundary_topology_registry &registry =
      gpu_boundary_topology_registry_instance();
  bool locally_validated = false;
  {
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto owner = registry.owners.find(this);
    locally_validated =
        owner != registry.owners.end() &&
        owner->second.validated_device == device &&
        owner->second.validated_device_generation == generation;
  }
  if (and_to_all(locally_validated)) return;

  // This is intentionally the explicit distributed-fields preflight, where
  // every participating rank is in the same call. Generic GPU discovery and
  // backend queries remain rank-local and cannot deadlock a rank subset.
  gpu::detail::validate_distributed_device_assignment();

  std::lock_guard<std::mutex> lock(registry.mutex);
  registry.owners[this].validated_device = device;
  registry.owners[this].validated_device_generation = generation;
}

void fields::destroy_gpu_boundary_topologies(bool erase_owner) {
  gpu_boundary_topology_registry &registry =
      gpu_boundary_topology_registry_instance();
  std::array<gpu_boundary_topology *, NUM_FIELD_TYPES> topologies{};
  {
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto owner = registry.owners.find(this);
    if (owner == registry.owners.end()) return;
    topologies = owner->second.topologies;
    if (erase_owner)
      registry.owners.erase(owner);
    else {
      owner->second.topologies.fill(nullptr);
      owner->second.validated_step_backend_generation =
          std::numeric_limits<std::uint64_t>::max();
      owner->second.validated_step_signature.clear();
      if (owner->second.cpu_feature_fallback)
        owner->second.feature_fallback_signature.clear();
      owner->second.step_preflight_consensus = false;
      owner->second.transport_validated = false;
      owner->second.validated_transport_environment_signature.clear();
    }
  }
  for (gpu_boundary_topology *topology : topologies)
    delete topology;
}

void fields::step() {
  // Arm the rank-local failure guard before any allocation, CUDA query, or
  // collective in backend/feature preflight.  Otherwise one rank can unwind
  // while a peer waits forever in the matching collective or boundary phase.
  // Distributed CPU steps can deadlock on the same rank-local exception
  // pattern once peers enter a collective or boundary exchange, so this
  // guard intentionally covers every backend rather than changing policy
  // after a potentially-throwing backend query.
  distributed_step_failure_guard distributed_failure_guard(
      count_processors() > 1);

  // No rank may query a throwing CUDA configuration before all ranks have
  // agreed on backend state and communication/device assignments.  Perform
  // both backend and feature validation before even restoring synchronized
  // magnetic fields so a rejected step is state-atomic.
  try {
    cuda_backend_preflight();
    cuda_step_preflight();
  }
  catch (const std::exception &error) {
    if (count_processors() > 1) {
      // Preserve the actual fail-closed contract in MPI qualification logs.
      // The guard's generic fallback is still needed for exceptions after
      // preflight, where one rank may already be inside a boundary exchange.
      distributed_failure_guard.dismiss();
      meep::abort(
          "distributed time-step preflight failed before dispatch: %s",
          error.what());
    }
    throw;
  }

  // however many times the fields have been synched, we want to restore now
  int save_synchronized_magnetic_fields = synchronized_magnetic_fields;
  if (synchronized_magnetic_fields) {
    synchronized_magnetic_fields = 1; // reset synchronization count
    restore_magnetic_fields();
  }

  am_now_working_on(Stepping);

  // Structure phasing/material replacement mutates host-side coefficient
  // arrays. Preserve device-authoritative fields, then invalidate all mirrors
  // before those coefficients can be reused by a later CUDA kernel.
  if (gpu::detail::cuda_active() && (changed_materials || is_phasing()))
    for (int i = 0; i < num_chunks; ++i)
      if (chunks[i]->is_mine()) {
        gpu::detail::sync_resident_cache_for_owner(chunks[i]);
        gpu::detail::destroy_resident_cache_for_owner(chunks[i]);
      }

  if (!t) {
    last_step_output_wall_time = wall_time();
    last_step_output_t = t;
  }
  if (verbosity > 0 && wall_time() > last_step_output_wall_time + MEEP_MIN_OUTPUT_TIME) {
    master_printf("on time step %d (time=%g), %g s/step\n", t, time(),
                  (wall_time() - last_step_output_wall_time) / (t - last_step_output_t));
    if (save_synchronized_magnetic_fields)
      master_printf("  (doing expensive timestepping of synched fields)\n");
    last_step_output_wall_time = wall_time();
    last_step_output_t = t;
  }

  phase_material();

  // update cached conductivity-inverse array, if needed
  for (int i = 0; i < num_chunks; i++)
    chunks[i]->s->update_condinv();

  const bool persistent_cuda_step =
      gpu::detail::cuda_active() && !fluxes;
  std::vector<std::unique_ptr<gpu::detail::resident_curl_session> >
      resident_step_sessions;
  if (persistent_cuda_step) {
    resident_step_sessions.reserve(num_chunks);
    for (int i = 0; i < num_chunks; ++i)
      if (chunks[i]->is_mine())
        resident_step_sessions.emplace_back(
            new gpu::detail::resident_curl_session(chunks[i], true));
  }

  calc_sources(time()); // for B sources
  {
    auto step_timer = with_timing_scope(FieldUpdateB);
    step_db(B_stuff);
  }
  step_source(B_stuff);
  bool h_updated = false;
  {
    auto step_timer = with_timing_scope(BoundarySteppingB);
    h_updated = step_boundaries_impl(B_stuff, boundary_eh_overlap::H);
  }
  calc_sources(time() + 0.5 * dt); // for integrated H sources
  if (!h_updated) {
    auto update_timer = with_timing_scope(FieldUpdateH);
    update_eh(H_stuff);
  }
  {
    auto step_timer = with_timing_scope(BoundarySteppingWH);
    step_boundaries(WH_stuff);
  }
  update_pols(H_stuff);
  {
    auto step_timer = with_timing_scope(BoundarySteppingPH);
    step_boundaries(PH_stuff);
  }
  bool d_curl_completed = false;
  {
    auto step_timer = with_timing_scope(BoundarySteppingH);
    d_curl_completed =
        step_boundaries_impl(H_stuff, boundary_eh_overlap::D_curl);
  }

  if (fluxes) fluxes->update_half();

  calc_sources(time() + 0.5 * dt); // for D sources
  if (!d_curl_completed) {
    auto step_timer = with_timing_scope(FieldUpdateD);
    step_db(D_stuff);
  }
  step_source(D_stuff);
  bool e_updated = false;
  {
    auto step_timer = with_timing_scope(BoundarySteppingD);
    e_updated = step_boundaries_impl(D_stuff, boundary_eh_overlap::E);
  }
  calc_sources(time() + dt); // for integrated E sources
  if (!e_updated) {
    auto update_timer = with_timing_scope(FieldUpdateE);
    update_eh(E_stuff);
  }
  {
    auto step_timer = with_timing_scope(BoundarySteppingWE);
    step_boundaries(WE_stuff);
  }
  update_pols(E_stuff);
  {
    auto step_timer = with_timing_scope(BoundarySteppingPE);
    step_boundaries(PE_stuff);
  }
  {
    auto step_timer = with_timing_scope(BoundarySteppingE);
    step_boundaries(E_stuff);
  }

  if (fluxes) fluxes->update();
  t += 1;
  update_dfts();

  bool device_fields_finite = true;
  if (persistent_cuda_step) {
#if MEEP_HAVE_CUDA
    if (!resident_step_sessions.empty()) {
      gpu::detail::resident_finite_check_session finite_check(
          resident_step_sessions.front()->cache());
      size_t session_index = 0;
      for (int i = 0; i < num_chunks; ++i)
        if (chunks[i]->is_mine()) {
          std::array<float *, NUM_FIELD_COMPONENTS * 2> arrays{};
          std::array<size_t, NUM_FIELD_COMPONENTS * 2> counts{};
          size_t array_count = 0;
          FOR_COMPONENTS(c) for (int cmp = 0; cmp < 2; ++cmp)
            if (chunks[i]->f[c][cmp]) {
              arrays[array_count] = chunks[i]->f[c][cmp];
              counts[array_count] = chunks[i]->gv.ntot();
              ++array_count;
            }
          finite_check.accumulate(
              resident_step_sessions[session_index]->cache(),
              arrays.data(), counts.data(), array_count);
          ++session_index;
        }
      // Preserve the historical per-step NaN/Inf failure contract. Deferring
      // this scalar readback can permanently hide a failure when a run ends
      // before the next readback boundary.
      device_fields_finite = finite_check.finish();
    }
#endif
    // Every rank must learn the same finite-field verdict before any rank can
    // return from step().  A local MPI_Abort alone eventually terminates the
    // job, but permits healthy ranks to observe a transient successful step.
    device_fields_finite = and_to_all(device_fields_finite);
  }
  if (persistent_cuda_step)
    for (auto &session : resident_step_sessions)
      session->finish(false);
  finished_working();

  // re-synch magnetic fields if they were previously synchronized
  if (save_synchronized_magnetic_fields) {
    synchronize_magnetic_fields();
    synchronized_magnetic_fields = save_synchronized_magnetic_fields;
  }

  changed_materials = false; // any material changes were handled in connect_chunks()

  if ((persistent_cuda_step && !device_fields_finite) ||
      (!persistent_cuda_step &&
       !std::isfinite(get_field(D_EnergyDensity, gv.center(), false))))
    meep::abort("simulation fields are NaN or Inf");
  distributed_failure_guard.dismiss();
}

void fields::phase_material() {
  bool changed = false;
  if (is_phasing()) {
#ifdef _OPENMP
#pragma omp parallel for reduction(|| : changed)
#endif
    for (int i = 0; i < num_chunks; i++)
      if (chunks[i]->is_mine()) {
        chunks[i]->phase_material(phasein_time);
        changed = changed || chunks[i]->new_s;
      }
    phasein_time--;
    am_now_working_on(MpiAllTime);
    bool changed_mpi = or_to_all(changed);
    finished_working();
    if (changed_mpi) {
      calc_sources(time() + 0.5 * dt); // for integrated H sources
      update_eh(H_stuff);              // ensure H = 1/mu * B
      step_boundaries(H_stuff);
      calc_sources(time() + dt); // for integrated E sources
      update_eh(E_stuff);        // ensure E = 1/eps * D
      step_boundaries(E_stuff);
    }
  }
}

void fields_chunk::phase_material(int phasein_time) {
  if (new_s && phasein_time > 0) {
    changing_structure();
    s->mix_with(new_s, 1.0 / phasein_time);
  }
}

void fields::process_incoming_chunk_data(field_type ft, const chunk_pair &comm_pair) {
  am_now_working_on(Boundaries);
  int this_chunk_idx = comm_pair.second;
  const int pair_idx = chunk_pair_to_index(comm_pair);
  const realnum *pair_comm_block = static_cast<realnum *>(comm_blocks[ft][pair_idx]);

  {
    const comms_key key = {ft, CONNECT_PHASE, comm_pair};
    size_t num_transfers = get_comm_size(key) / 2; // Two realnums per complex
    if (num_transfers) {
      const std::vector<realnum *> &incoming_connection =
          chunks[this_chunk_idx]->connections_in.at(key);
      const std::vector<std::complex<realnum> > &connection_phase_for_ft =
          chunks[this_chunk_idx]->connection_phases[key];

      for (size_t n = 0; n < num_transfers; ++n) {
        std::complex<realnum> temp =
            connection_phase_for_ft[n] *
            std::complex<realnum>(pair_comm_block[2 * n], pair_comm_block[2 * n + 1]);
        *(incoming_connection[2 * n]) = temp.real();
        *(incoming_connection[2 * n + 1]) = temp.imag();
      }
      pair_comm_block += 2 * num_transfers;
    }
  }

  {
    const comms_key key = {ft, CONNECT_NEGATE, comm_pair};
    const size_t num_transfers = get_comm_size(key);
    if (num_transfers) {
      const std::vector<realnum *> &incoming_connection =
          chunks[this_chunk_idx]->connections_in.at(key);
      for (size_t n = 0; n < num_transfers; ++n) {
        *(incoming_connection[n]) = -pair_comm_block[n];
      }
      pair_comm_block += num_transfers;
    }
  }

  {
    const comms_key key = {ft, CONNECT_COPY, comm_pair};
    const size_t num_transfers = get_comm_size(key);
    if (num_transfers) {
      const std::vector<realnum *> &incoming_connection =
          chunks[this_chunk_idx]->connections_in.at(key);
      for (size_t n = 0; n < num_transfers; ++n) {
        *(incoming_connection[n]) = pair_comm_block[n];
      }
    }
  }
  finished_working();
}

bool fields::cuda_whole_update_eh_overlap_eligible(
    field_type ft, std::string *reason) const {
  const auto reject = [reason](const char *message) {
    if (reason) *reason = message;
    return false;
  };
  if (ft != E_stuff && ft != H_stuff)
    return reject("overlap target is not E/H");
  if (!gpu::detail::cuda_active() ||
      sizeof(realnum) != sizeof(float))
    return reject("required FP32 CUDA is inactive");
  if (fluxes)
    return reject("flux monitor disables the persistent CUDA step");
  if (changed_materials || !chunk_connections_valid)
    return reject("material or chunk topology is changing");

  bool have_local_chunk = false;
  for (int i = 0; i < num_chunks; ++i)
    if (chunks[i]->is_mine()) {
      have_local_chunk = true;
      if (!gpu::detail::resident_phase_is_active_for_owner(chunks[i]))
        return reject("no persistent resident CUDA phase");
      if (!chunks[i]->cuda_whole_update_eh_overlap_eligible(
              ft, reason))
        return false;
    }
  if (!have_local_chunk)
    return reject("rank owns no field chunk");
  if (reason) reason->clear();
  return true;
}

void fields::step_boundaries(field_type ft) {
  (void)step_boundaries_impl(ft, boundary_eh_overlap::none);
}

bool fields::step_boundaries_impl(field_type ft,
                                  boundary_eh_overlap overlap) {
  connect_chunks(); // re-connect if !chunk_connections_valid

  const bool eh_overlap_requested =
      overlap == boundary_eh_overlap::H ||
      overlap == boundary_eh_overlap::E;
  const bool curl_overlap_requested =
      overlap == boundary_eh_overlap::D_curl;
  const bool overlap_requested =
      eh_overlap_requested || curl_overlap_requested;
  const field_type overlap_ft =
      overlap == boundary_eh_overlap::H ? H_stuff : E_stuff;
  bool overlap_classified = !overlap_requested;
  if (eh_overlap_requested) {
    gpu::detail::record_boundary_eh_overlap_check();
    if (!boundary_eh_overlap_opted_in()) {
      gpu::detail::record_boundary_eh_overlap_disabled();
      overlap_classified = true;
    }
  }
  else if (curl_overlap_requested) {
    gpu::detail::record_halo_curl_overlap_check();
    if (!halo_curl_overlap_opted_in()) {
      gpu::detail::record_halo_curl_overlap_disabled();
      overlap_classified = true;
    }
  }

  const auto record_overlap_no_remote = [&]() {
    if (curl_overlap_requested)
      gpu::detail::record_halo_curl_overlap_no_remote();
    else
      gpu::detail::record_boundary_eh_overlap_no_remote();
  };
  const auto record_overlap_unsupported_schedule = [&]() {
    if (curl_overlap_requested)
      gpu::detail::record_halo_curl_overlap_unsupported_schedule();
    else
      gpu::detail::record_boundary_eh_overlap_unsupported_schedule();
  };
  const auto record_overlap_cold_topology = [&]() {
    if (curl_overlap_requested)
      gpu::detail::record_halo_curl_overlap_cold_topology();
    else
      gpu::detail::record_boundary_eh_overlap_cold_topology();
  };
  const auto record_overlap_rejected = [&](bool small) {
    if (curl_overlap_requested)
      gpu::detail::record_halo_curl_overlap_rejected(small);
    else
      gpu::detail::record_boundary_eh_overlap_rejected();
  };

#if MEEP_HAVE_CUDA
  bool local_cuda_eligible =
      gpu::detail::cuda_active() && sizeof(realnum) == sizeof(float);
  bool have_local_chunk = false;
  for (int i = 0; i < num_chunks && local_cuda_eligible; ++i)
    if (chunks[i]->is_mine()) {
      have_local_chunk = true;
    }
  local_cuda_eligible = local_cuda_eligible && have_local_chunk;

  std::vector<std::unique_ptr<gpu::detail::resident_curl_session> >
      sessions(static_cast<std::size_t>(num_chunks));
  if (local_cuda_eligible) {
    try {
      for (int i = 0; i < num_chunks; ++i)
        if (chunks[i]->is_mine())
          sessions[static_cast<std::size_t>(i)].reset(
              new gpu::detail::resident_curl_session(chunks[i], true));
    }
    catch (const std::exception &error) {
      if (count_processors() > 1)
        meep::abort(
            "CUDA resident boundary session failed on MPI rank %d: %s",
            my_global_rank(), error.what());
      throw;
    }
  }

  if (local_cuda_eligible) {
    // Match the original CPU boundary timing stack.  Every successful CUDA
    // return below already calls finished_working(); entering the Boundaries
    // sink here makes that call balanced and preserves both the aggregate and
    // detailed BoundaryStepping* timing APIs.
    am_now_working_on(Boundaries);
    bool boundary_launch_started = false;
    try {
      const size_t zero_plan_token =
          2u * static_cast<size_t>(ft);
      const size_t copy_plan_token = zero_plan_token + 1u;

      // With no remote MPI peers, a connected boundary topology is immutable
      // until disconnect_chunks() explicitly invalidates its cached plans.
      // Re-launch those resident plans directly instead of rebuilding and
      // validating hundreds of thousands of host pointer operations on every
      // time step.
      if (count_processors() == 1 &&
          std::getenv("MEEP_GPU_DISABLE_BOUNDARY_PLAN_REPLAY") == nullptr &&
          gpu::detail::resident_boundary_plan_ready(
              this, zero_plan_token) &&
          gpu::detail::resident_boundary_plan_ready(
              this, copy_plan_token)) {
        // Re-establish every boundary-capable mirror before replay. Other
        // field phases can create a new mirror or make a clean host array
        // authoritative without changing the connected-boundary topology.
        // Skipping this preparation can therefore replay valid descriptors
        // against stale device contents.
        for (int i = 0; i < num_chunks; ++i)
          if (chunks[i]->is_mine())
            (void)prepare_boundary_arrays(
                chunks[i],
                sessions[static_cast<std::size_t>(i)]->cache());

        // Preparation can allocate a mirror and invalidate the descriptor
        // addresses. In that case fall through to the full rebuild below.
        if (gpu::detail::resident_boundary_plan_ready(
                this, zero_plan_token) &&
            gpu::detail::resident_boundary_plan_ready(
                this, copy_plan_token)) {
          boundary_launch_started =
              gpu::detail::resident_launch_boundary_plan(
                  this, zero_plan_token);
          boundary_launch_started =
              gpu::detail::resident_launch_boundary_plan(
                  this, copy_plan_token) ||
              boundary_launch_started;
          for (auto &session : sessions)
            if (session) session->finish();
          if (!overlap_classified) {
            record_overlap_no_remote();
            overlap_classified = true;
          }
          finished_working();
          return false;
        }
      }

      using remote_send = gpu_boundary_topology::remote_send;
      using remote_receive = gpu_boundary_topology::remote_receive;

      gpu_boundary_topology *cached_topology =
          get_gpu_boundary_topology(ft);
      bool cached_topology_matches =
          count_processors() > 1 && cached_topology &&
          cached_topology->chunk_caches.size() ==
              static_cast<std::size_t>(num_chunks);
      for (int i = 0; i < num_chunks && cached_topology_matches; ++i)
        if (chunks[i]->is_mine() &&
            cached_topology->chunk_caches[static_cast<std::size_t>(i)] !=
                sessions[static_cast<std::size_t>(i)]->cache())
          cached_topology_matches = false;

      if (cached_topology_matches &&
          gpu::detail::resident_boundary_plan_ready(
              this, zero_plan_token) &&
          gpu::detail::resident_boundary_plan_ready(
              this, copy_plan_token)) {
        for (int i = 0; i < num_chunks; ++i)
          if (chunks[i]->is_mine())
            (void)prepare_boundary_arrays(
                chunks[i],
                sessions[static_cast<std::size_t>(i)]->cache());

        if (gpu::detail::resident_boundary_plan_ready(
                this, zero_plan_token) &&
            gpu::detail::resident_boundary_plan_ready(
                this, copy_plan_token)) {
          const gpu::detail::boundary_descriptor_replay_token
              descriptor_replay_token =
                  cached_topology->descriptor_replay_token();
          if (!cached_topology->manager)
            cached_topology->manager = create_comms_manager();
          comms_manager *manager = cached_topology->manager.get();
          const bool cuda_aware =
              comms_supports_cuda_device_buffers(manager);
          const bool has_remote_messages =
              !cached_topology->remote_sends.empty() ||
              !cached_topology->remote_receives.empty();
          const bool eager_cuda_aware =
              cuda_aware && has_remote_messages &&
              std::getenv("MEEP_GPU_DISABLE_EAGER_MPI") == nullptr;
          bool overlap_predicate_pass = false;
          size_t overlap_full_points = 0;
          size_t overlap_interior_points = 0;
          bool overlap_rejected_small = false;
          if (!overlap_classified) {
            if (!has_remote_messages) {
              record_overlap_no_remote();
              overlap_classified = true;
            }
            else if (!eager_cuda_aware) {
              // Pinned MPI and the explicitly disabled eager reference path
              // both defer request posting until comms_finish(). Moving E/H
              // ahead of that call would only reorder work; it cannot overlap
              // communication.
              record_overlap_unsupported_schedule();
              overlap_classified = true;
            }
            else {
              std::string overlap_reason;
              if (curl_overlap_requested)
                overlap_predicate_pass =
                    cuda_halo_curl_overlap_eligible(
                        D_stuff, &overlap_full_points,
                        &overlap_interior_points,
                        &overlap_rejected_small, &overlap_reason);
              else
                overlap_predicate_pass =
                    cuda_whole_update_eh_overlap_eligible(
                        overlap_ft, &overlap_reason);
              if (!overlap_predicate_pass) {
                if (std::getenv(
                        curl_overlap_requested
                            ? "MEEP_GPU_DEBUG_HALO_CURL_OVERLAP"
                            : "MEEP_GPU_DEBUG_BOUNDARY_EH_OVERLAP"))
                  std::fprintf(
                      stderr,
                      "gpmeep-%s-overlap-reject-v1:rank=%d,field=%s,reason=%s\n",
                      curl_overlap_requested ? "halo-curl" : "boundary-eh",
                      my_global_rank(),
                      curl_overlap_requested
                          ? "D"
                          : (overlap_ft == H_stuff ? "H" : "E"),
                      overlap_reason.c_str());
                record_overlap_rejected(overlap_rejected_small);
                overlap_classified = true;
              }
            }
          }
          const bool pingpong_cuda_receive =
              cuda_aware &&
              std::getenv("MEEP_GPU_DISABLE_RECEIVE_PINGPONG") == nullptr;
          std::shared_ptr<std::exception_ptr> exchange_error(
              new std::exception_ptr);
          std::vector<const void *> send_buffers;
          send_buffers.reserve(
              cached_topology->remote_sends.size());
          auto queue_remote_receives = [&]() {
            for (remote_receive &receive :
                 cached_topology->remote_receives) {
              remote_receive *receive_pointer = &receive;
              comms_manager::receive_callback callback =
                  [this, receive_pointer, cuda_aware,
                   exchange_error, descriptor_replay_token]() {
                    if (*exchange_error) return;
                    // Completion callbacks run only from comms_finish(),
                    // after graph/direct local work has been enqueued. This
                    // is the host ordering gate for an eagerly posted Irecv.
                    am_now_working_on(Boundaries);
                    try {
                      boundary_callback_test_delay();
                      if (!cuda_aware)
                        gpu::detail::boundary_exchange_copy_to_device(
                            receive_pointer->buffer);
                      gpu::detail::resident_scatter_boundary_fp32(
                          receive_pointer->buffer,
                          receive_pointer->operations.data(),
                          receive_pointer->operations.size(),
                          &descriptor_replay_token);
                      gpu::detail::record_mpi_boundary_transfer(
                          receive_pointer->transfer_size,
                          cuda_aware, true);
                    }
                    catch (...) {
                      *exchange_error = std::current_exception();
                    }
                    finished_working();
                  };
              void *receive_buffer =
                  cuda_aware
                      ? static_cast<void *>(
                            gpu::detail::boundary_exchange_device_data(
                                receive.buffer))
                      : static_cast<void *>(
                            gpu::detail::boundary_exchange_host_data(
                                receive.buffer));
              manager->receive_real_async(
                  receive_buffer, receive.transfer_size,
                  receive.other_proc_id, receive.tag, callback);
            }
          };
          auto queue_remote_sends = [&]() {
            for (std::size_t index = 0;
                 index < cached_topology->remote_sends.size(); ++index) {
              remote_send &send = cached_topology->remote_sends[index];
              manager->send_real_async(
                  send_buffers[index], send.transfer_size,
                  send.other_proc_id, send.tag);
              gpu::detail::record_mpi_boundary_transfer(
                  send.transfer_size, cuda_aware, false);
            }
          };

          if (cuda_aware) {
            for (remote_receive &receive :
                 cached_topology->remote_receives) {
              if (pingpong_cuda_receive) {
                if (!receive.secondary_buffer)
                  receive.secondary_buffer =
                      gpu::detail::resident_boundary_exchange_buffer_slot(
                          this, receive.buffer_token,
                          receive.transfer_size, 1);
                for (const remote_send &send :
                     cached_topology->remote_sends)
                  if (receive.secondary_buffer == send.buffer)
                    throw std::logic_error(
                        "Meep CUDA receive and send boundary buffers alias");
                const bool secondary =
                    receive.select_secondary_next;
                receive.buffer = secondary ? receive.secondary_buffer
                                           : receive.primary_buffer;
                receive.select_secondary_next =
                    !receive.select_secondary_next;
                gpu::detail::record_boundary_receive_pingpong_selection(
                    secondary);
              }
              else
                receive.buffer = receive.primary_buffer;
            }
          }

          // A pinned Irecv may reuse host bytes still consumed by the prior
          // asynchronous H2D, while CUDA-aware MPI may reuse the device bytes
          // consumed by the prior scatter. The scatter completion event is
          // the lifetime fence for both transports.
          for (remote_receive &receive :
               cached_topology->remote_receives)
            gpu::detail::boundary_exchange_synchronize(
                receive.buffer);

          if (cuda_aware)
            for (remote_send &send : cached_topology->remote_sends)
              send_buffers.push_back(
                  gpu::detail::boundary_exchange_device_data(
                      send.buffer));

          if (eager_cuda_aware) {
            queue_remote_receives();
            queue_remote_sends();
            // Every callback/request allocation is complete before the first
            // Irecv. Any later distributed CUDA failure takes the existing
            // fail-fast MPI abort path rather than abandoning a posted peer.
            comms_start_cuda_device_receives(manager);
          }

          // When eager posting is active the CUDA-aware receive is already
          // live here. A launch/copy failure therefore cannot fall back
          // locally: the outer distributed exception path aborts the MPI job
          // so no peer is stranded.
          boundary_launch_started =
              !cached_topology->remote_sends.empty() ||
              !cached_topology->remote_receives.empty();
          bool phase_graph_launched = false;
          if (cuda_aware && cached_topology->phase_graph)
            phase_graph_launched =
                gpu::detail::resident_launch_boundary_phase_graph(
                    this, cached_topology->phase_graph);
          // launch() performs the structural/generation validation while
          // holding the backend registry lock. Only pay for a second read-only
          // check after the rare false return, to distinguish a stale graph
          // from a temporarily disabled optimization.
          if (!phase_graph_launched && cached_topology->phase_graph &&
              !gpu::detail::resident_boundary_phase_graph_ready(
                  this, cached_topology->phase_graph)) {
            gpu::detail::destroy_boundary_phase_graph(
                cached_topology->phase_graph);
            cached_topology->phase_graph = nullptr;
            cached_topology->phase_graph_creation_attempted = false;
          }

          if (!phase_graph_launched) {
            boundary_launch_started =
                gpu::detail::resident_launch_boundary_plan(
                    this, zero_plan_token) ||
                boundary_launch_started;
            for (remote_send &send :
                 cached_topology->remote_sends)
              gpu::detail::resident_gather_boundary_fp32(
                  send.source_caches.data(), send.buffer,
                  send.sources.data(),
                  send.sources.size(), &descriptor_replay_token);
            boundary_launch_started =
                gpu::detail::resident_launch_boundary_plan(
                    this, copy_plan_token) ||
                boundary_launch_started;

            // Inspect the now-warm immutable plans and prepare a graph for the
            // next replay. Never launch both direct work and the graph in one
            // boundary phase.
            if (cuda_aware &&
                !cached_topology->phase_graph_creation_attempted) {
              std::vector<gpu::detail::boundary_exchange_buffer *>
                  send_phase_buffers;
              send_phase_buffers.reserve(
                  cached_topology->remote_sends.size());
              for (remote_send &send : cached_topology->remote_sends)
                send_phase_buffers.push_back(send.buffer);
              cached_topology->phase_graph =
                  gpu::detail::resident_create_boundary_phase_graph(
                      this, zero_plan_token, copy_plan_token,
                      send_phase_buffers.data(),
                      send_phase_buffers.size());
              cached_topology->phase_graph_creation_attempted = true;
            }
          }
          else {
            boundary_launch_started = true;
          }

          if (phase_graph_launched)
            gpu::detail::resident_synchronize_boundary_phase_graph_send_ready(
                cached_topology->phase_graph);

          for (remote_send &send :
               cached_topology->remote_sends) {
            if (cuda_aware) {
              if (!phase_graph_launched)
                gpu::detail::boundary_exchange_synchronize(
                    send.buffer);
            }
            else {
              gpu::detail::boundary_exchange_copy_to_host(
                  send.buffer);
              send_buffers.push_back(
                  gpu::detail::boundary_exchange_host_data(
                      send.buffer));
            }
          }

          if (eager_cuda_aware)
            // Ordinary CUDA-aware MPI is not assumed stream-aware. The graph
            // send-ready event or each direct gather event was synchronized
            // above before the first MPI_Isend can inspect its device buffer.
            comms_start_cuda_device_sends(manager);
          else if (cuda_aware) {
            queue_remote_receives();
            queue_remote_sends();
          }
          else {
            queue_remote_receives();
            queue_remote_sends();
          }

          const size_t physical_messages =
              comms_physical_message_count(manager);
          bool overlapped_update = false;
          bool curl_interior_launched = false;
          if (overlap_predicate_pass) {
            if (physical_messages == 0) {
              record_overlap_no_remote();
              overlap_classified = true;
            }
            else if (curl_overlap_requested) {
              if (overlap_interior_points > overlap_full_points)
                throw std::logic_error(
                    "CUDA halo/curl interior exceeds full curl work");
              gpu::detail::record_halo_curl_overlap_eligible();
              {
                auto update_timer = with_timing_scope(FieldUpdateD);
                step_db_cuda_region(D_stuff,
                                    cuda_curl_region::interior);
              }
              overlap_classified = true;
              curl_interior_launched = true;
            }
            else {
              gpu::detail::record_boundary_eh_overlap_eligible();
              const bool connections_were_valid = chunk_connections_valid;
              {
                auto update_timer = with_timing_scope(
                    overlap_ft == H_stuff ? FieldUpdateH : FieldUpdateE);
                update_eh(overlap_ft);
              }
              if (!connections_were_valid || !chunk_connections_valid)
                throw std::logic_error(
                    "overlapped CUDA E/H update changed the chunk topology");
              gpu::detail::record_boundary_eh_overlap_launch(
                  overlap_ft == E_stuff);
              overlap_classified = true;
              overlapped_update = true;
            }
          }
          if (physical_messages > 0) {
            gpu::detail::record_mpi_completion(
                comms_uses_waitall_completion(manager));
            boundary_mpi_completion_scope mpi_timing(*this);
            comms_finish(manager);
          }
          else
            comms_finish(manager);
          gpu::detail::record_mpi_boundary_messages(
              physical_messages);
          if (*exchange_error)
            std::rethrow_exception(*exchange_error);
          if (curl_interior_launched) {
            {
              // Receive callbacks have enqueued the H scatter on the same
              // CUDA stream. The shell launch is therefore ordered after the
              // halo is visible without a device-wide synchronization.
              auto update_timer = with_timing_scope(FieldUpdateD);
              step_db_cuda_region(D_stuff, cuda_curl_region::shell);
            }
            const size_t shell_points =
                overlap_full_points - overlap_interior_points;
            gpu::detail::record_halo_curl_overlap_launch(
                overlap_full_points, overlap_interior_points,
                shell_points);
            overlapped_update = true;
          }
          for (auto &session : sessions)
            if (session) session->finish();
          finished_working();
          return overlapped_update;
        }
      }

      if (!overlap_classified) {
        record_overlap_cold_topology();
        overlap_classified = true;
      }

      std::vector<gpu::detail::boundary_operation_fp32> zero_operations;
      std::vector<gpu::detail::boundary_operation_fp32> copy_operations;
      std::vector<remote_send> remote_sends;
      std::vector<remote_receive> remote_receives;
      std::map<int, std::vector<const comms_operation *> >
          remote_receive_groups;
      std::map<int, std::vector<const comms_operation *> >
          remote_send_groups;
      std::vector<std::vector<boundary_array_span> >
          boundary_spans(static_cast<std::size_t>(num_chunks));
      size_t scalar_points = 0;

      // Prepare every pointer before launching anything, so automatic mode
      // can fall back to the original CPU path without a partially-applied
      // boundary update.
      std::exception_ptr preparation_error;
      try {
        for (int i = 0; i < num_chunks; ++i)
          if (chunks[i]->is_mine())
            boundary_spans[static_cast<std::size_t>(i)] =
                prepare_boundary_arrays(
                    chunks[i],
                    sessions[static_cast<std::size_t>(i)]->cache());
        for (int i = 0; i < num_chunks; ++i) {
        if (!chunks[i]->is_mine()) continue;
        fields_chunk *chunk = chunks[i];
        gpu::detail::resident_cache *cache =
            sessions[static_cast<std::size_t>(i)]->cache();
        zero_operations.reserve(zero_operations.size() +
                                chunk->num_zeroes[ft]);
        for (size_t n = 0; n < chunk->num_zeroes[ft]; ++n) {
          realnum *destination = chunk->zeroes[ft][n];
          if (!boundary_pointer_is_prepared(
                  boundary_spans[static_cast<std::size_t>(i)],
                  destination))
            throw std::out_of_range(
                "Meep CUDA metal boundary pointer is not resident-capable");
          zero_operations.push_back(
              {nullptr, cache, destination, nullptr, nullptr, nullptr, 0.0f,
               0.0f});
          ++scalar_points;
        }
        }

        const auto &sequence = comms_sequence_for_field[ft];
        for (const comms_operation &op : sequence.receive_ops) {
        const int source_index = op.other_chunk_idx;
        const int destination_index = op.my_chunk_idx;
        if (!chunks[destination_index]->is_mine())
          throw std::logic_error(
              "Meep CUDA boundary receive destination is not local");
        fields_chunk *source_chunk = chunks[source_index];
        fields_chunk *destination_chunk = chunks[destination_index];
        gpu::detail::resident_cache *destination_cache =
            sessions[static_cast<std::size_t>(destination_index)]->cache();
        const chunk_pair pair{source_index, destination_index};

        if (source_chunk->is_mine()) {
          gpu::detail::resident_cache *source_cache =
              sessions[static_cast<std::size_t>(source_index)]->cache();
          for (connect_phase phase : all_connect_phases) {
            const comms_key key = {ft, phase, pair};
            const size_t transfer_size = get_comm_size(key);
            if (!transfer_size) continue;
            const std::vector<realnum *> &outgoing =
                source_chunk->connections_out.at(key);
            const std::vector<realnum *> &incoming =
                destination_chunk->connections_in.at(key);
            if (outgoing.size() != transfer_size ||
                incoming.size() != transfer_size)
              throw std::logic_error(
                  "Meep CUDA local boundary connection size mismatch");

            if (phase == CONNECT_PHASE) {
              if (transfer_size % 2)
                throw std::logic_error(
                    "Meep CUDA phased boundary has an odd scalar count");
              const std::vector<std::complex<realnum> > &phases =
                  destination_chunk->connection_phases.at(key);
              if (phases.size() != transfer_size / 2)
                throw std::logic_error(
                    "Meep CUDA phased boundary phase-count mismatch");
              for (size_t n = 0; n < transfer_size / 2; ++n) {
                realnum *source_real = outgoing[2 * n];
                realnum *source_imag = outgoing[2 * n + 1];
                realnum *destination_real = incoming[2 * n];
                realnum *destination_imag = incoming[2 * n + 1];
                if (!boundary_pointer_is_prepared(
                        boundary_spans[
                            static_cast<std::size_t>(source_index)],
                        source_real) ||
                    !boundary_pointer_is_prepared(
                        boundary_spans[
                            static_cast<std::size_t>(source_index)],
                        source_imag) ||
                    !boundary_pointer_is_prepared(
                        boundary_spans[
                            static_cast<std::size_t>(destination_index)],
                        destination_real) ||
                    !boundary_pointer_is_prepared(
                        boundary_spans[
                            static_cast<std::size_t>(destination_index)],
                        destination_imag))
                  throw std::out_of_range(
                      "Meep CUDA phased boundary pointer is not "
                      "resident-capable");
                copy_operations.push_back(
                    {source_cache, destination_cache, destination_real,
                     destination_imag, source_real, source_imag,
                     static_cast<float>(phases[n].real()),
                     static_cast<float>(phases[n].imag())});
                scalar_points += 2;
              }
            }
            else {
              const float sign =
                  phase == CONNECT_NEGATE ? -1.0f : 1.0f;
              for (size_t n = 0; n < transfer_size; ++n) {
                realnum *source = outgoing[n];
                realnum *destination = incoming[n];
                if (!boundary_pointer_is_prepared(
                        boundary_spans[
                            static_cast<std::size_t>(source_index)],
                        source) ||
                    !boundary_pointer_is_prepared(
                        boundary_spans[
                            static_cast<std::size_t>(destination_index)],
                        destination))
                  throw std::out_of_range(
                      "Meep CUDA scalar boundary pointer is not "
                      "resident-capable");
                copy_operations.push_back(
                    {source_cache, destination_cache, destination, nullptr,
                     source, nullptr, sign, 0.0f});
                ++scalar_points;
              }
            }
          }
        }
        else {
          remote_receive_groups[op.other_proc_id].push_back(&op);
        }
        }

        for (auto &entry : remote_receive_groups) {
          std::vector<const comms_operation *> &operations = entry.second;
          std::sort(
              operations.begin(), operations.end(),
              [](const comms_operation *left,
                 const comms_operation *right) {
                return left->tag < right->tag;
              });
          std::vector<size_t> operation_sizes;
          operation_sizes.reserve(operations.size());
          for (const comms_operation *operation : operations)
            operation_sizes.push_back(operation->transfer_size);
          const std::vector<gpu::detail::coalesced_transfer_batch> batches =
              gpu::detail::plan_coalesced_transfer_batches(
                  operation_sizes.data(), operation_sizes.size(),
                  static_cast<size_t>(std::numeric_limits<int>::max()));
          for (const gpu::detail::coalesced_transfer_batch &batch : batches) {
            const size_t group_transfer_size = batch.scalar_count;
            const comms_operation &first = *operations[batch.begin];
            remote_receive receive = {
              entry.first, first.tag, group_transfer_size,
              comm_blocks[ft][first.pair_idx],
              nullptr, nullptr, nullptr, true, {}};
            receive.primary_buffer =
              gpu::detail::resident_boundary_exchange_buffer_slot(
                  this, comm_blocks[ft][first.pair_idx],
                  group_transfer_size, 0);
            receive.buffer = receive.primary_buffer;
            receive.operations.reserve(group_transfer_size);
            size_t group_source_offset = 0;
            for (size_t operation_index = batch.begin;
                 operation_index < batch.end; ++operation_index) {
            const comms_operation *operation = operations[operation_index];
            const int source_index = operation->other_chunk_idx;
            const int destination_index = operation->my_chunk_idx;
            if (!chunks[destination_index]->is_mine() ||
                chunks[source_index]->is_mine())
              throw std::logic_error(
                  "Meep CUDA coalesced receive topology is inconsistent");
            fields_chunk *destination_chunk = chunks[destination_index];
            gpu::detail::resident_cache *destination_cache =
                sessions[static_cast<std::size_t>(destination_index)]
                    ->cache();
            const chunk_pair pair{source_index, destination_index};
            size_t operation_source_offset = 0;
            for (connect_phase phase : all_connect_phases) {
              const comms_key key = {ft, phase, pair};
              const size_t transfer_size = get_comm_size(key);
              if (!transfer_size) continue;
              const std::vector<realnum *> &incoming =
                  destination_chunk->connections_in.at(key);
              if (incoming.size() != transfer_size)
                throw std::logic_error(
                    "Meep CUDA coalesced receive size mismatch");
              if (phase == CONNECT_PHASE) {
                if (transfer_size % 2)
                  throw std::logic_error(
                      "Meep CUDA coalesced phased receive has an odd size");
                const std::vector<std::complex<realnum> > &phases =
                    destination_chunk->connection_phases.at(key);
                if (phases.size() != transfer_size / 2)
                  throw std::logic_error(
                      "Meep CUDA coalesced receive phase-count mismatch");
                for (size_t n = 0; n < transfer_size / 2; ++n) {
                  realnum *destination_real = incoming[2 * n];
                  realnum *destination_imag = incoming[2 * n + 1];
                  if (!boundary_pointer_is_prepared(
                          boundary_spans[static_cast<std::size_t>(
                              destination_index)],
                          destination_real) ||
                      !boundary_pointer_is_prepared(
                          boundary_spans[static_cast<std::size_t>(
                              destination_index)],
                          destination_imag))
                    throw std::out_of_range(
                        "Meep CUDA coalesced phased destination is not "
                        "resident-capable");
                  receive.operations.push_back(
                      {destination_cache, destination_real,
                       destination_imag,
                       group_source_offset + operation_source_offset +
                           2 * n,
                       static_cast<float>(phases[n].real()),
                       static_cast<float>(phases[n].imag())});
                }
              }
              else {
                const float sign =
                    phase == CONNECT_NEGATE ? -1.0f : 1.0f;
                for (size_t n = 0; n < transfer_size; ++n) {
                  realnum *destination = incoming[n];
                  if (!boundary_pointer_is_prepared(
                          boundary_spans[static_cast<std::size_t>(
                              destination_index)],
                          destination))
                    throw std::out_of_range(
                        "Meep CUDA coalesced destination is not "
                        "resident-capable");
                  receive.operations.push_back(
                      {destination_cache, destination, nullptr,
                       group_source_offset + operation_source_offset + n,
                       sign, 0.0f});
                }
              }
              operation_source_offset += transfer_size;
            }
            if (operation_source_offset != operation->transfer_size)
              throw std::logic_error(
                  "Meep CUDA coalesced receive operation size mismatch");
            group_source_offset += operation->transfer_size;
            }
            if (group_source_offset != group_transfer_size)
              throw std::logic_error(
                  "Meep CUDA coalesced receive total size mismatch");
            remote_receives.push_back(std::move(receive));
          }
        }

        for (const comms_operation &op : sequence.send_ops) {
          if (!chunks[op.my_chunk_idx]->is_mine())
            throw std::logic_error(
                "Meep CUDA boundary send source is not local");
          if (chunks[op.other_chunk_idx]->is_mine()) continue;
          remote_send_groups[op.other_proc_id].push_back(&op);
        }

        for (auto &entry : remote_send_groups) {
          std::vector<const comms_operation *> &operations = entry.second;
          std::sort(
              operations.begin(), operations.end(),
              [](const comms_operation *left,
                 const comms_operation *right) {
                return left->tag < right->tag;
              });
          std::vector<size_t> operation_sizes;
          operation_sizes.reserve(operations.size());
          for (const comms_operation *operation : operations)
            operation_sizes.push_back(operation->transfer_size);
          const std::vector<gpu::detail::coalesced_transfer_batch> batches =
              gpu::detail::plan_coalesced_transfer_batches(
                  operation_sizes.data(), operation_sizes.size(),
                  static_cast<size_t>(std::numeric_limits<int>::max()));
          for (const gpu::detail::coalesced_transfer_batch &batch : batches) {
            const size_t group_transfer_size = batch.scalar_count;
            const comms_operation &first = *operations[batch.begin];
            remote_send send = {
              entry.first, first.tag, group_transfer_size,
              gpu::detail::resident_boundary_exchange_buffer(
                  this, comm_blocks[ft][first.pair_idx],
                  group_transfer_size),
              {}, {}};
            send.source_caches.reserve(group_transfer_size);
            send.sources.reserve(group_transfer_size);
            size_t group_source_size = 0;
            for (size_t operation_index = batch.begin;
                 operation_index < batch.end; ++operation_index) {
            const comms_operation *operation = operations[operation_index];
            const int source_index = operation->my_chunk_idx;
            const int destination_index = operation->other_chunk_idx;
            if (!chunks[source_index]->is_mine() ||
                chunks[destination_index]->is_mine())
              throw std::logic_error(
                  "Meep CUDA coalesced send topology is inconsistent");
            fields_chunk *source_chunk = chunks[source_index];
            gpu::detail::resident_cache *source_cache =
                sessions[static_cast<std::size_t>(source_index)]->cache();
            const chunk_pair pair{source_index, destination_index};
            size_t operation_source_size = 0;
            for (connect_phase phase : all_connect_phases) {
              const comms_key key = {ft, phase, pair};
              const size_t transfer_size = get_comm_size(key);
              if (!transfer_size) continue;
              const std::vector<realnum *> &outgoing =
                  source_chunk->connections_out.at(key);
              if (outgoing.size() != transfer_size)
                throw std::logic_error(
                    "Meep CUDA coalesced send size mismatch");
              for (realnum *source : outgoing) {
                if (!boundary_pointer_is_prepared(
                        boundary_spans[static_cast<std::size_t>(
                            source_index)],
                        source))
                  throw std::out_of_range(
                      "Meep CUDA coalesced source is not resident-capable");
                send.source_caches.push_back(source_cache);
                send.sources.push_back(source);
              }
              operation_source_size += transfer_size;
            }
            if (operation_source_size != operation->transfer_size)
              throw std::logic_error(
                  "Meep CUDA coalesced send operation size mismatch");
            group_source_size += operation->transfer_size;
            }
            if (group_source_size != group_transfer_size ||
                send.source_caches.size() != group_transfer_size ||
                send.sources.size() != group_transfer_size)
              throw std::logic_error(
                  "Meep CUDA coalesced send total size mismatch");
            remote_sends.push_back(std::move(send));
          }
        }
      }
      catch (const std::exception &error) {
        if (count_processors() > 1)
          meep::abort(
              "CUDA boundary preparation failed on MPI rank %d: %s",
              my_global_rank(), error.what());
        preparation_error = std::current_exception();
      }
      if (preparation_error) std::rethrow_exception(preparation_error);

      std::unique_ptr<comms_manager> manager = create_comms_manager();
      const bool cuda_aware =
          comms_supports_cuda_device_buffers(manager.get());
      const bool has_remote_messages =
          !remote_sends.empty() || !remote_receives.empty();
      const bool eager_cuda_aware =
          cuda_aware && has_remote_messages &&
          std::getenv("MEEP_GPU_DISABLE_EAGER_MPI") == nullptr;
      const bool pingpong_cuda_receive =
          cuda_aware &&
          std::getenv("MEEP_GPU_DISABLE_RECEIVE_PINGPONG") == nullptr;
      if (pingpong_cuda_receive)
        for (remote_receive &receive : remote_receives)
          receive.secondary_buffer =
              gpu::detail::resident_boundary_exchange_buffer_slot(
                  this, receive.buffer_token,
                  receive.transfer_size, 1);

      // Every active boundary exchange object owns a distinct live CUDA
      // allocation. In particular, an eagerly posted receive must never
      // overwrite a graph send buffer while gather/local work is running.
      for (const remote_receive &receive : remote_receives)
        for (const remote_send &send : remote_sends)
          if (receive.primary_buffer == send.buffer ||
              (receive.secondary_buffer &&
               receive.secondary_buffer == send.buffer))
            throw std::logic_error(
                "Meep CUDA receive and send boundary buffers alias");
      std::shared_ptr<std::exception_ptr> exchange_error(
          new std::exception_ptr);
      std::vector<const void *> send_buffers;
      send_buffers.reserve(remote_sends.size());
      auto queue_remote_transfers = [&]() {
        for (remote_receive &receive : remote_receives) {
          remote_receive *receive_pointer = &receive;
          comms_manager::receive_callback callback =
              [this, receive_pointer, cuda_aware, exchange_error]() {
                if (*exchange_error) return;
                // Callbacks are deferred until comms_finish(), after local
                // graph/direct work has been enqueued.
                am_now_working_on(Boundaries);
                try {
                  boundary_callback_test_delay();
                  if (!cuda_aware)
                    gpu::detail::boundary_exchange_copy_to_device(
                        receive_pointer->buffer);
                  gpu::detail::resident_scatter_boundary_fp32(
                      receive_pointer->buffer,
                      receive_pointer->operations.data(),
                      receive_pointer->operations.size());
                  gpu::detail::record_mpi_boundary_transfer(
                      receive_pointer->transfer_size,
                      cuda_aware, true);
                }
                catch (...) {
                  *exchange_error = std::current_exception();
                }
                finished_working();
              };
          void *receive_buffer =
              cuda_aware
                  ? static_cast<void *>(
                        gpu::detail::boundary_exchange_device_data(
                            receive.buffer))
                  : static_cast<void *>(
                        gpu::detail::boundary_exchange_host_data(
                            receive.buffer));
          manager->receive_real_async(
              receive_buffer, receive.transfer_size,
              receive.other_proc_id, receive.tag, callback);
        }

        for (std::size_t index = 0; index < remote_sends.size(); ++index) {
          remote_send &send = remote_sends[index];
          manager->send_real_async(
              send_buffers[index], send.transfer_size,
              send.other_proc_id, send.tag);
          gpu::detail::record_mpi_boundary_transfer(
              send.transfer_size, cuda_aware, false);
        }
      };

      // Protect either the device receive bytes (CUDA-aware) or page-locked
      // host bytes still consumed by the previous async H2D (pinned).
      for (remote_receive &receive : remote_receives)
        gpu::detail::boundary_exchange_synchronize(receive.buffer);

      if (cuda_aware)
        for (remote_send &send : remote_sends)
          send_buffers.push_back(
              gpu::detail::boundary_exchange_device_data(send.buffer));

      if (eager_cuda_aware) {
        queue_remote_transfers();
        comms_start_cuda_device_receives(manager.get());
      }

      // Match the CPU ordering: zero metals, snapshot every outgoing remote
      // value, then apply local transfers. Once an eager receive is posted, a
      // fallible CUDA operation takes the distributed fail-fast path below.
      boundary_launch_started =
          !zero_operations.empty() || !copy_operations.empty() ||
          !remote_sends.empty() || !remote_receives.empty();
      gpu::detail::resident_apply_boundary_fp32(
          this, zero_plan_token,
          zero_operations.data(), zero_operations.size());
      for (remote_send &send : remote_sends)
        gpu::detail::resident_gather_boundary_fp32(
            send.source_caches.data(), send.buffer,
            send.sources.data(),
            send.sources.size());
      gpu::detail::resident_apply_boundary_fp32(
          this, copy_plan_token,
          copy_operations.data(), copy_operations.size());

      for (remote_send &send : remote_sends) {
        if (cuda_aware) {
          gpu::detail::boundary_exchange_synchronize(send.buffer);
        }
        else {
          gpu::detail::boundary_exchange_copy_to_host(send.buffer);
          send_buffers.push_back(
              gpu::detail::boundary_exchange_host_data(send.buffer));
        }
      }

      if (eager_cuda_aware)
        comms_start_cuda_device_sends(manager.get());
      else
        queue_remote_transfers();

      // Manager destruction completes all messages and invokes receive
      // callbacks while the persistent buffers and resident sessions live.
      const size_t physical_messages =
          comms_physical_message_count(manager.get());
      if (physical_messages > 0) {
        gpu::detail::record_mpi_completion(
            comms_uses_waitall_completion(manager.get()));
        boundary_mpi_completion_scope mpi_timing(*this);
        comms_finish(manager.get());
      }
      else
        comms_finish(manager.get());
      manager.reset();
      gpu::detail::record_mpi_boundary_messages(
          physical_messages);
      if (*exchange_error) std::rethrow_exception(*exchange_error);

      if (count_processors() > 1) {
        std::unique_ptr<gpu_boundary_topology> topology(
            new gpu_boundary_topology);
        topology->chunk_caches.resize(
            static_cast<std::size_t>(num_chunks), nullptr);
        for (int i = 0; i < num_chunks; ++i)
          if (chunks[i]->is_mine())
            topology->chunk_caches[static_cast<std::size_t>(i)] =
                sessions[static_cast<std::size_t>(i)]->cache();
        topology->remote_sends = std::move(remote_sends);
        topology->remote_receives = std::move(remote_receives);
        topology->manager = create_comms_manager();
        std::vector<gpu::detail::boundary_exchange_buffer *>
            send_phase_buffers;
        send_phase_buffers.reserve(topology->remote_sends.size());
        for (remote_send &send : topology->remote_sends)
          send_phase_buffers.push_back(send.buffer);
        if (comms_supports_cuda_device_buffers(topology->manager.get()))
          topology->phase_graph =
              gpu::detail::resident_create_boundary_phase_graph(
                  this, zero_plan_token, copy_plan_token,
                  send_phase_buffers.data(), send_phase_buffers.size());
        topology->phase_graph_creation_attempted = true;
        replace_gpu_boundary_topology(ft, topology.release());
      }

      for (auto &session : sessions)
        if (session) session->finish();
      (void)scalar_points;
      finished_working();
      return false;
    }
    catch (...) {
      // A rank-local CUDA failure cannot safely fall back while its peers are
      // entering the matching boundary exchange. Terminate the MPI job
      // collectively instead of allowing those peers to wait indefinitely.
      if (count_processors() > 1) {
        try {
          throw;
        }
        catch (const std::exception &error) {
          meep::abort(
              "CUDA boundary exchange failed on MPI rank %d: %s",
              my_global_rank(), error.what());
        }
        catch (...) {
          meep::abort(
              "CUDA boundary exchange failed on MPI rank %d",
              my_global_rank());
        }
      }
      // Once a kernel has been submitted, CUDA failure is not safely
      // recoverable by replaying the boundary on potentially half-synchronized
      // host mirrors. Let session destruction discard unsynchronized mirrors
      // and report the error in both automatic and required modes.
      if (boundary_launch_started) {
        finished_working();
        throw;
      }
      for (auto &session : sessions) {
        if (!session) continue;
        try {
          session->finish();
        }
        catch (...) {
          if (gpu::detail::cuda_active()) {
            finished_working();
            throw;
          }
        }
      }
      if (gpu::detail::cuda_active()) {
        finished_working();
        throw;
      }
      finished_working();
      // Automatic mode deliberately resumes through the original CPU path.
    }
  }
#endif

  if (!overlap_classified) {
    record_overlap_rejected(false);
    overlap_classified = true;
  }

  size_t cpu_boundary_points = 0;
  for (int i = 0; i < num_chunks; ++i)
    if (chunks[i]->is_mine())
      cpu_boundary_points += chunks[i]->num_zeroes[ft];
  for (const comms_operation &op : comms_sequence_for_field[ft].receive_ops)
    if (chunks[op.my_chunk_idx]->is_mine())
      cpu_boundary_points += op.transfer_size;

  {
    // Initiate receive operations as early as possible.
    std::unique_ptr<comms_manager> manager = create_comms_manager();

    const auto &sequence = comms_sequence_for_field[ft];
    for (const comms_operation &op : sequence.receive_ops) {
      if (chunks[op.other_chunk_idx]->is_mine()) { continue; }
      chunk_pair comm_pair{op.other_chunk_idx, op.my_chunk_idx};
      comms_manager::receive_callback cb = [this, ft, comm_pair]() {
        process_incoming_chunk_data(ft, comm_pair);
      };
      manager->receive_real_async(comm_blocks[ft][op.pair_idx], static_cast<int>(op.transfer_size),
                                  op.other_proc_id, op.tag, cb);
    }

    // Do the metals first!
    for (int i = 0; i < num_chunks; i++)
      if (chunks[i]->is_mine()) chunks[i]->zero_metal(ft);

    // Copy outgoing data into buffers while following the predefined sequence of comms operations.
    // Trigger the asynchronous send immediately once the outgoing comms buffer has been filled.
    am_now_working_on(Boundaries);

    for (const comms_operation &op : sequence.send_ops) {
      const std::pair<int, int> comm_pair{op.my_chunk_idx, op.other_chunk_idx};
      const int pair_idx = op.pair_idx;

      realnum *outgoing_comm_block = comm_blocks[ft][pair_idx];
      for (connect_phase ip : all_connect_phases) {
        const comms_key key = {ft, ip, comm_pair};
        const size_t pair_comm_size = get_comm_size(key);
        if (pair_comm_size) {
          const std::vector<realnum *> &outgoing_connection =
              chunks[op.my_chunk_idx]->connections_out.at(key);
          for (size_t n = 0; n < pair_comm_size; ++n) {
            outgoing_comm_block[n] = *(outgoing_connection[n]);
          }
          outgoing_comm_block += pair_comm_size;
        }
      }
      if (chunks[op.other_chunk_idx]->is_mine()) { continue; }
      manager->send_real_async(comm_blocks[ft][pair_idx], static_cast<int>(op.transfer_size),
                               op.other_proc_id, op.tag);
    }

    // Process local transfers, which do not depend on a communication mechanism across nodes.
    for (const comms_operation &op : sequence.receive_ops) {
      if (chunks[op.other_chunk_idx]->is_mine()) {
        process_incoming_chunk_data(ft, {op.other_chunk_idx, op.my_chunk_idx});
      }
    }
    finished_working();

    am_now_working_on(MpiOneTime);
    // Let the communication manager drop out of scope to complete all outstanding requests.
    // As data is received, the installed callback handles copying the data from the comm buffer
    // back into the chunk field array.
  }
  gpu::detail::record_cpu_boundary(cpu_boundary_points);
  finished_working();
  return false;
}

void fields::step_source(field_type ft, bool including_integrated) {
  if (ft != D_stuff && ft != B_stuff) meep::abort("only step_source(D/B) is okay");
  bool phase_batch_enabled =
      gpu::detail::cuda_active() &&
      gpu::detail::phase_batched_source_opted_in();
  for (int i = 0; i < num_chunks && phase_batch_enabled; ++i)
    if (chunks[i]->is_mine() &&
        !gpu::detail::resident_phase_is_active_for_owner(chunks[i]))
      phase_batch_enabled = false;
  const int phase_key =
      2 * static_cast<int>(ft) + (including_integrated ? 1 : 0);
  gpu::detail::resident_source_phase_batch phase_batch(
      this, phase_key, phase_batch_enabled);
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) chunks[i]->step_source(ft, including_integrated);
  phase_batch.finish();
}

void fields_chunk::step_source(field_type ft, bool including_integrated) {
  if (doing_solve_cw && !including_integrated) return;
  const bool cuda_phase_eligible =
      gpu::detail::cuda_active() && sizeof(realnum) == sizeof(float);
  gpu::detail::resident_curl_session cuda_session(this,
                                                   cuda_phase_eligible);
  for (const src_vol &sv : sources[ft]) {
    component c = direction_component(first_field_component(ft), component_direction(sv.c));
    const realnum *cndinv = s->condinv[c][component_direction(sv.c)];
    if ((including_integrated || !sv.t()->is_integrated) && f[c][0] &&
        ((ft == D_stuff && is_electric(sv.c)) || (ft == B_stuff && is_magnetic(sv.c)))) {
      if (cuda_session.active()) {
#if MEEP_HAVE_CUDA
        const complex<double> time_scale = sv.t()->current() * dt;
        bool phase_batched = false;
        DOCMP {
          phase_batched =
              gpu::detail::resident_indexed_source_subtract_fp32(
                  cuda_session.cache(), f[c][cmp], gv.ntot(),
                  sv.indices_data(), sv.amplitudes_fp32_data(),
                  sv.num_points(), cndinv,
                  {static_cast<float>(time_scale.real()),
                   static_cast<float>(time_scale.imag())},
                  cmp != 0) ||
              phase_batched;
        }
        if (!phase_batched)
          gpu::detail::record_cuda_source(
              sv.num_points() * static_cast<size_t>(is_real ? 1 : 2));
#endif
      }
      else {
        if (cndinv)
          for (size_t j = 0; j < sv.num_points(); j++) {
            const ptrdiff_t i = sv.index_at(j);
            const complex<double> A = sv.current(j) * dt * double(cndinv[i]);
            f[c][0][i] -= real(A);
            if (!is_real) f[c][1][i] -= imag(A);
          }
        else
          for (size_t j = 0; j < sv.num_points(); j++) {
            const complex<double> A = sv.current(j) * dt;
            const ptrdiff_t i = sv.index_at(j);
            f[c][0][i] -= real(A);
            if (!is_real) f[c][1][i] -= imag(A);
          }
        gpu::detail::record_cpu_source(
            sv.num_points() * static_cast<size_t>(is_real ? 1 : 2));
      }
    }
  }
  cuda_session.finish();
}

void fields::calc_sources(double tim) {
  for (src_time *s = sources; s; s = s->next)
    s->update(tim, dt);
  for (int i = 0; i < num_chunks; i++)
    if (chunks[i]->is_mine()) chunks[i]->calc_sources(tim);
}

void fields_chunk::calc_sources(double time) {
  (void)time; // unused;
}

} // namespace meep
