#include "meep/gpu.hpp"
#include "meep/meep-config.h"
#include "meep/mympi.hpp"
#include "gpu_backend_internal.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#if MEEP_HAVE_CUDA
#include "meep_cuda/detail/bfast.hpp"
#include "meep_cuda/detail/curl.hpp"
#include "meep_cuda/detail/polarization.hpp"
#include "meep_cuda/detail/update_eh.hpp"
#include "meep_cuda/runtime.hpp"
#endif

namespace meep {
namespace gpu {

namespace detail {

namespace {

bool exact_opt_in_without_disable(const char *enable_name,
                                  const char *disable_name) noexcept {
  const char *enabled = std::getenv(enable_name);
  return enabled && enabled[0] == '1' && enabled[1] == '\0' &&
         std::getenv(disable_name) == nullptr;
}

phase_batch_mode phase_batch_mode_from_environment(
    const char *enable_name, const char *disable_name) noexcept {
  if (std::getenv(disable_name) != nullptr)
    return phase_batch_mode::disabled;
  const char *enabled = std::getenv(enable_name);
  if (!enabled) return phase_batch_mode::automatic;
  return enabled[0] == '1' && enabled[1] == '\0'
             ? phase_batch_mode::forced
             : phase_batch_mode::disabled;
}

#if MEEP_HAVE_CUDA
bool inject_eigenmode_failure_for_testing(const char *stage) noexcept {
  const char *requested =
      std::getenv("MEEP_GPU_TEST_EIGENMODE_FAIL_STAGE");
  return requested && std::strcmp(requested, stage) == 0;
}
#endif

} // namespace

phase_batch_mode phase_batched_curl_mode() noexcept {
  return phase_batch_mode_from_environment(
      "MEEP_GPU_ENABLE_PHASE_BATCHED_CURL",
      "MEEP_GPU_DISABLE_PHASE_BATCHED_CURL");
}

phase_batch_mode phase_batched_update_eh_mode() noexcept {
  return phase_batch_mode_from_environment(
      "MEEP_GPU_ENABLE_PHASE_BATCHED_UPDATE_EH",
      "MEEP_GPU_DISABLE_PHASE_BATCHED_UPDATE_EH");
}

phase_batch_mode dft_multi_monitor_batch_mode() noexcept {
  return phase_batch_mode_from_environment(
      "MEEP_GPU_ENABLE_MULTI_MONITOR_DFT_BATCH",
      "MEEP_GPU_DISABLE_MULTI_MONITOR_DFT_BATCH");
}

bool automatic_dft_multi_monitor_batch_selected(
    std::size_t operation_count) noexcept {
  // Fail closed until the sealed cross-size, one/two-GPU crossover matrix is
  // available. Request count alone cannot account for descriptor pressure,
  // 64-bit tile decoding, or heterogeneous monitor shapes. Forced mode keeps
  // the implementation measurable without exposing unproved default policy.
  (void)operation_count;
  return false;
}

bool phase_batched_source_opted_in() noexcept {
  return exact_opt_in_without_disable(
      "MEEP_GPU_ENABLE_PHASE_BATCHED_SOURCE",
      "MEEP_GPU_DISABLE_PHASE_BATCHED_SOURCE");
}

bool automatic_phase_batch_selected(
    std::size_t operation_count, std::size_t logical_block_count,
    std::size_t maximum_operation_block_count,
    int multiprocessor_count) noexcept {
  // The phase kernel saves one launch per operation but performs one block-map
  // lookup for every logical block. Repeated 1/2-GPU crossover measurements
  // place the conservative transition near 24 logical blocks per SM per
  // operation. Require at least four operations so descriptor dispatch cannot
  // outweigh a negligible launch saving. Scaling by SM count makes the rule
  // independent of a particular GPU UUID, grid dimension, or pixel count.
  constexpr std::size_t minimum_operations = 4;
  constexpr std::size_t maximum_blocks_per_sm_per_operation = 24;
  if (operation_count < minimum_operations || logical_block_count == 0 ||
      maximum_operation_block_count == 0 || multiprocessor_count <= 0)
    return false;
  const std::size_t sms = static_cast<std::size_t>(multiprocessor_count);
  if (sms > std::numeric_limits<std::size_t>::max() /
                maximum_blocks_per_sm_per_operation)
    return false;
  const std::size_t per_operation_limit =
      sms * maximum_blocks_per_sm_per_operation;
  if (operation_count >
      std::numeric_limits<std::size_t>::max() / per_operation_limit)
    return false;
  return maximum_operation_block_count <= per_operation_limit &&
         logical_block_count <= operation_count * per_operation_limit;
}

std::uint64_t next_saturating_boundary_topology_generation(
    std::atomic<std::uint64_t> *generation) noexcept {
  if (!generation) return 0;
  std::uint64_t current =
      generation->load(std::memory_order_relaxed);
  while (current != std::numeric_limits<std::uint64_t>::max()) {
    const std::uint64_t next = current + 1;
    if (generation->compare_exchange_weak(
            current, next, std::memory_order_relaxed,
            std::memory_order_relaxed))
      return next;
  }
  return 0;
}

std::vector<coalesced_transfer_batch> plan_coalesced_transfer_batches(
    const std::size_t *transfer_sizes, std::size_t transfer_count,
    std::size_t maximum_batch_scalars) {
  if (transfer_count > 0 && !transfer_sizes)
    throw std::invalid_argument(
        "Meep CUDA coalesced transfer sizes must be non-null");
  if (maximum_batch_scalars == 0)
    throw std::invalid_argument(
        "Meep CUDA coalesced transfer limit must be nonzero");

  std::vector<coalesced_transfer_batch> batches;
  std::size_t begin = 0;
  while (begin < transfer_count) {
    std::size_t end = begin;
    std::size_t scalar_count = 0;
    while (end < transfer_count) {
      const std::size_t next = transfer_sizes[end];
      if (next == 0)
        throw std::invalid_argument(
            "Meep CUDA coalesced transfer size must be nonzero");
      if (next > maximum_batch_scalars)
        throw std::length_error(
            "Meep CUDA logical transfer exceeds the MPI count limit");
      if (scalar_count > maximum_batch_scalars - next) break;
      scalar_count += next;
      ++end;
    }
    if (end == begin || scalar_count == 0)
      throw std::logic_error(
          "Meep CUDA coalesced transfer planner made no progress");
    batches.push_back({begin, end, scalar_count});
    begin = end;
  }
  return batches;
}

std::vector<std::uint32_t> plan_phase_block_operation_indices(
    const std::size_t *block_starts, std::size_t operation_count,
    std::size_t total_block_count) {
  if (operation_count > 0 && !block_starts)
    throw std::invalid_argument(
        "Meep CUDA phase-batched block starts must be non-null");
  if (operation_count >
      static_cast<std::size_t>(
          std::numeric_limits<std::uint32_t>::max()))
    throw std::overflow_error(
        "Meep CUDA phase-batched operation index overflow");
  if (operation_count == 0) {
    if (total_block_count != 0)
      throw std::logic_error(
          "Meep CUDA phase-batched block map has no operations");
    return {};
  }

  std::vector<std::uint32_t> indices;
  if (total_block_count > indices.max_size())
    throw std::overflow_error(
        "Meep CUDA phase-batched block map length overflow");
  indices.reserve(total_block_count);
  for (std::size_t operation_index = 0;
       operation_index < operation_count; ++operation_index) {
    if (block_starts[operation_index] != indices.size())
      throw std::logic_error(
          "Meep CUDA phase-batched operation prefix is not contiguous");
    const std::size_t next_block_start =
        operation_index + 1 < operation_count
            ? block_starts[operation_index + 1]
            : total_block_count;
    if (next_block_start < indices.size() ||
        next_block_start > total_block_count)
      throw std::logic_error(
          "Meep CUDA phase-batched operation prefix is invalid");
    indices.insert(
        indices.end(), next_block_start - indices.size(),
        static_cast<std::uint32_t>(operation_index));
  }
  if (indices.size() != total_block_count)
    throw std::logic_error(
        "Meep CUDA phase-batched block map is incomplete");
  return indices;
}

#if MEEP_HAVE_CUDA
std::uint64_t next_resident_allocation_generation() noexcept {
  static std::atomic<std::uint64_t> generation(0);
  return generation.fetch_add(1, std::memory_order_relaxed) + 1;
}

std::uint64_t next_resident_content_generation() noexcept {
  static std::atomic<std::uint64_t> generation(0);
  return generation.fetch_add(1, std::memory_order_relaxed) + 1;
}
#endif

struct resident_cache {
#if MEEP_HAVE_CUDA
  struct mirror {
    void *device_pointer;
    std::size_t bytes;
    std::uint64_t uploaded_epoch;
    std::uint64_t content_generation;
    bool device_dirty;
    std::vector<float> host_staging;

    mirror()
        : device_pointer(nullptr), bytes(0), uploaded_epoch(0),
          content_generation(0), device_dirty(false) {}
  };

  std::unordered_map<const float *, mirror> mirrors;
  std::map<std::uintptr_t, const float *> mirror_intervals;
  std::vector<std::pair<std::size_t, void *> > recycled_mirror_allocations;
  // Source profiles are replaced far more often than ordinary field and
  // material mirrors. Keep their exact-size reuse pool separate so changing
  // source extents cannot retain an unbounded number of device allocations.
  std::vector<std::pair<std::size_t, void *> >
      recycled_source_profile_allocations;
  // Source indices are immutable for the lifetime of an src_vol. Cache their
  // bounds validation so an extended source does not incur an O(points) host
  // scan on every time step.
  std::unordered_map<const std::ptrdiff_t *,
                     std::pair<std::size_t, std::size_t> >
      validated_source_profiles;
  struct structured_curl_batch {
    void *device_spaces = nullptr;
    std::size_t descriptor_count = 0;
    std::size_t maximum_point_count = 0;
  };
  struct finite_check_entry {
    const float *host_base = nullptr;
    const float *mirror_host_base = nullptr;
    const float *device_base = nullptr;
    std::size_t scalar_count = 0;
    std::size_t mirror_bytes = 0;
    std::size_t byte_offset = 0;

    finite_check_entry() {}
    finite_check_entry(const float *host, const float *mirror_host,
                       const float *device, std::size_t scalars,
                       std::size_t mirror_size, std::size_t offset)
        : host_base(host), mirror_host_base(mirror_host),
          device_base(device), scalar_count(scalars),
          mirror_bytes(mirror_size), byte_offset(offset) {}
  };
  struct finite_check_plan {
    void *device_spans = nullptr;
    std::vector<finite_check_entry> entries;
    std::size_t maximum_span_count = 0;
    std::uint64_t allocation_generation = 0;
  };
  struct multilevel_plan {
    void *device_population_channels = nullptr;
    void *device_polarization_channels = nullptr;
    void *device_transitions = nullptr;
    std::vector<multilevel_population_channel_fp32> population_channels;
    std::vector<multilevel_polarization_channel_fp32>
        polarization_channels;
    std::vector<multilevel_transition_fp32> transitions;
    std::size_t maximum_polarization_point_count = 0;
    std::uint64_t allocation_generation = 0;
  };
  std::unordered_map<const float *, structured_curl_batch>
      structured_curl_batches;
  std::unordered_map<const float *, multilevel_plan> multilevel_plans;
  finite_check_plan finite_check;
  std::uint32_t *finite_result;
  std::uint32_t finite_result_generation;
  bool finite_result_pending;
  bool finite_session_active;
  double *dft_norm_result;
  struct dft_batch_plan {
    void *device_pointer;
    std::size_t capacity_bytes;
    std::size_t descriptor_offset;
    std::size_t block_map_offset;
    std::size_t total_block_count;
    std::vector<meep_cuda::dft_update_operation_fp32> host_operations;

    dft_batch_plan()
        : device_pointer(nullptr), capacity_bytes(0),
          descriptor_offset(0), block_map_offset(0),
          total_block_count(0) {}
  };
  // Electric and magnetic DFT phases can alternate between two stable
  // descriptor topologies. Retain both instead of replacing one plan every
  // half step. The phase vectors at the beginning of each allocation are
  // refreshed every update; descriptors and the logical-block map remain
  // resident while that topology is unchanged.
  dft_batch_plan dft_batch_plans[2];
  int dft_batch_next_replacement;
  struct dft_materialization_request_signature {
    const void *owner = nullptr;
    const float *host_dft = nullptr;
    std::size_t storage_point_count = 0;
    std::size_t point_count = 0;
    std::size_t source_frequency_count = 0;
    std::size_t source_frequency_start = 0;
    std::size_t selected_frequency_count = 0;
    float inverse_stored_weight_real = 0.0f;
    float inverse_stored_weight_imaginary = 0.0f;
  };
  struct dft_materialization_dependency {
    const void *owner = nullptr;
    resident_cache *cache = nullptr;
    const float *host_dft = nullptr;
    const void *device_dft = nullptr;
    std::size_t bytes = 0;
    std::uint64_t allocation_generation = 0;
    std::uint64_t content_generation = 0;
  };
  struct dft_materialization_plan {
    void *device_metadata = nullptr;
    void *device_output = nullptr;
    std::size_t metadata_bytes = 0;
    std::size_t output_bytes = 0;
    std::size_t descriptor_offset = 0;
    std::size_t block_map_offset = 0;
    std::size_t total_block_count = 0;
    std::size_t output_point_count = 0;
    std::size_t output_frequency_count = 0;
    bool all_frequency_cache = false;
    bool collapse_active = false;
    std::size_t collapsed_output_offset = 0;
    std::size_t collapsed_output_point_count = 0;
    std::size_t collapse_full_rank = 0;
    std::size_t collapse_full_dims[3] = {1, 1, 1};
    std::uint8_t collapse_flags[3] = {0, 0, 0};
    std::vector<dft_materialization_request_signature> requests;
    std::vector<std::ptrdiff_t> destinations;
    std::vector<float> weights;
    std::vector<std::uint8_t> zero_divisor_flags;
    std::vector<std::uint8_t> publication_flags;
    std::vector<dft_materialization_dependency> dependencies;
  };
  // get_dft_array normally walks one component's frequencies consecutively.
  // Retaining one exact plan is therefore sufficient to turn a point-spectrum
  // loop into one all-frequency kernel followed by scalar slice readbacks,
  // while bounding cache memory independently of the number of monitors.
  dft_materialization_plan dft_materialization;
  struct dft_output_staging_request_signature {
    const void *owner = nullptr;
    const float *host_dft = nullptr;
    std::size_t storage_point_count = 0;
    std::size_t point_count = 0;
    std::size_t source_frequency_count = 0;
    float inverse_stored_weight_real = 0.0f;
    float inverse_stored_weight_imaginary = 0.0f;
    std::size_t output_point_offset = 0;
  };
  struct dft_output_staging_dependency {
    const void *owner = nullptr;
    resident_cache *cache = nullptr;
    const float *host_dft = nullptr;
    const void *device_dft = nullptr;
    std::size_t bytes = 0;
    std::uint64_t allocation_generation = 0;
    std::uint64_t content_generation = 0;
  };
  struct dft_output_staging_plan {
    void *device_metadata = nullptr;
    void *device_output = nullptr;
    // Returned views share ownership of this allocation so dependency-cache
    // invalidation cannot leave a dangling planar pointer.
    std::shared_ptr<float> pinned_host;
    // CUDA D2H never targets the published allocation directly. A failed or
    // partially completed transfer may modify this private scratch buffer;
    // after the complete D2H succeeds, a shared_ptr swap publishes it without
    // an output-sized host copy.
    std::shared_ptr<float> pinned_transfer;
    std::size_t metadata_bytes = 0;
    std::size_t output_bytes = 0;
    std::size_t descriptor_offset = 0;
    std::size_t block_map_offset = 0;
    std::size_t total_block_count = 0;
    std::size_t output_point_count = 0;
    std::size_t frequency_capacity = 0;
    std::vector<dft_output_staging_request_signature> requests;
    std::vector<std::ptrdiff_t> destinations;
    std::vector<float> weights;
    std::vector<std::uint8_t> zero_divisor_flags;
    std::vector<std::uint32_t> block_operation_indices;
    std::vector<dft_output_staging_dependency> dependencies;
  };
  dft_output_staging_plan dft_output_staging;
  struct ldos_plan_buffer {
    void *device_pointer;
    std::size_t capacity_bytes;

    ldos_plan_buffer() : device_pointer(nullptr), capacity_bytes(0) {}
  };
  // LDOS descriptors contain device addresses owned by several resident
  // caches.  Never rewrite the active slot: a topology replacement is first
  // uploaded to the inactive slot and becomes visible only at the final,
  // allocation-free metadata commit.  Two retained high-water slots bound
  // replacement churn without weakening failure atomicity.
  ldos_plan_buffer ldos_plan_buffers[2];
  int ldos_active_plan_slot;
  bool ldos_plan_snapshot_valid;
  std::size_t ldos_plan_descriptor_bytes;
  std::size_t ldos_plan_total_block_count;
  std::vector<meep_cuda::indexed_ldos_operation_fp32>
      ldos_host_operations;
  double *ldos_reduction_workspace;
  std::size_t ldos_reduction_partial_capacity;
  void *index_pointer;
  std::size_t index_capacity;
  int device_ordinal;
  std::uint64_t epoch;
  std::uint64_t allocation_generation;
  bool phase_active;

  resident_cache()
      : finite_result(nullptr), finite_result_generation(0),
        finite_result_pending(false), finite_session_active(false),
        dft_norm_result(nullptr), dft_batch_next_replacement(0),
        ldos_active_plan_slot(-1),
        ldos_plan_snapshot_valid(false), ldos_plan_descriptor_bytes(0),
        ldos_plan_total_block_count(0), ldos_reduction_workspace(nullptr),
        ldos_reduction_partial_capacity(0),
        index_pointer(nullptr), index_capacity(0),
        device_ordinal(-1), epoch(0),
        allocation_generation(next_resident_allocation_generation()),
        phase_active(false) {}
#endif
};

struct boundary_exchange_buffer {
#if MEEP_HAVE_CUDA
  struct transfer_plan {
    void *device_pointer;
    std::size_t operation_count;
    std::size_t scalar_count;
    std::uint64_t generation;
    std::vector<resident_cache *> gather_caches;
    std::vector<const float *> gather_sources;
    std::vector<remote_boundary_operation_fp32> scatter_operations;
    std::vector<std::pair<resident_cache *, std::uint64_t> >
        cache_generations;
    std::vector<std::pair<resident_cache *, const float *> >
        writable_mirrors;
    const void *replay_topology_identity;
    std::uint64_t replay_topology_generation;

    transfer_plan()
        : device_pointer(nullptr), operation_count(0), scalar_count(0),
          generation(0),
          replay_topology_identity(nullptr), replay_topology_generation(0) {}
  };

  void *device_pointer;
  void *host_pointer;
  void *completion_event;
  bool completion_event_recorded;
  bool lifetime_uncertain;
  std::size_t scalar_count;
  int device_ordinal;
  transfer_plan gather_plan;
  transfer_plan scatter_plan;

  boundary_exchange_buffer()
      : device_pointer(nullptr), host_pointer(nullptr),
        completion_event(nullptr), completion_event_recorded(false),
        lifetime_uncertain(false),
        scalar_count(0), device_ordinal(-1) {}
#endif
};

struct boundary_phase_graph {
#if MEEP_HAVE_CUDA
  meep_cuda::boundary_phase_graph_fp32 *execution_graph;
  const void *owner;
  std::size_t zero_plan_token;
  std::size_t copy_plan_token;
  const void *zero_plan_snapshot;
  const void *copy_plan_snapshot;
  std::uint64_t zero_plan_generation;
  std::uint64_t copy_plan_generation;
  std::vector<boundary_exchange_buffer *> send_buffers;
  std::vector<std::uint64_t> gather_plan_generations;
  void *send_ready_event;
  void *completion_event;
  bool send_ready_event_recorded;
  bool completion_event_recorded;
  int device_ordinal;

  boundary_phase_graph()
      : execution_graph(nullptr), owner(nullptr), zero_plan_token(0),
        copy_plan_token(0), zero_plan_snapshot(nullptr),
        copy_plan_snapshot(nullptr), zero_plan_generation(0),
        copy_plan_generation(0), send_ready_event(nullptr),
        completion_event(nullptr), send_ready_event_recorded(false),
        completion_event_recorded(false), device_ordinal(-1) {}
#endif
};

struct resident_cw_vector_plan {
#if MEEP_HAVE_CUDA
  struct cache_dependency {
    const void *owner;
    resident_cache *cache;
  };
  struct mirror_dependency {
    resident_cache *cache;
    const float *host_pointer;
    const void *device_pointer;
    std::size_t bytes;
  };

  void *device_segments;
  std::size_t segment_count;
  std::size_t maximum_point_count;
  std::size_t complex_count;
  int device_ordinal;
  std::vector<cache_dependency> cache_dependencies;
  std::vector<mirror_dependency> mirror_dependencies;
  std::vector<std::pair<resident_cache *, float *> > writable_mirrors;

  resident_cw_vector_plan()
      : device_segments(nullptr), segment_count(0),
        maximum_point_count(0), complex_count(0), device_ordinal(-1) {}
#endif
};

} // namespace detail

namespace {

void append_environment_value(std::string *signature, const char *name) {
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

std::string backend_configuration_environment_signature() {
  std::string signature;
  append_environment_value(&signature, "MEEP_GPU_DEVICE");
  append_environment_value(
      &signature, "MEEP_GPU_ALLOW_OVERSUBSCRIBE");
  return signature;
}

struct backend_state {
  std::mutex mutex;
  bool configured;
  bool device_ready;
  backend_mode requested;
  backend_mode active;
  int selected_device;
  bool explicit_device_selection;
  std::uint64_t generation;
  std::string configured_environment_signature;
  std::string diagnostic;

  backend_state()
      : configured(false), device_ready(false),
        requested(backend_mode::cpu), active(backend_mode::cpu),
        selected_device(-1), explicit_device_selection(false), generation(0) {}
};

backend_state state;

std::atomic<std::uint64_t> cpu_curl_calls(0);
std::atomic<std::uint64_t> cpu_curl_points(0);
std::atomic<std::uint64_t> cuda_curl_calls(0);
std::atomic<std::uint64_t> cuda_curl_points(0);
std::atomic<std::uint64_t> cpu_update_eh_calls(0);
std::atomic<std::uint64_t> cpu_update_eh_points(0);
std::atomic<std::uint64_t> cuda_update_eh_calls(0);
std::atomic<std::uint64_t> cuda_update_eh_points(0);
std::atomic<std::uint64_t> cpu_polarization_calls(0);
std::atomic<std::uint64_t> cpu_polarization_points(0);
std::atomic<std::uint64_t> cuda_polarization_calls(0);
std::atomic<std::uint64_t> cuda_polarization_points(0);
std::atomic<std::uint64_t> cpu_source_calls(0);
std::atomic<std::uint64_t> cpu_source_points(0);
std::atomic<std::uint64_t> cuda_source_calls(0);
std::atomic<std::uint64_t> cuda_source_points(0);
std::atomic<std::uint64_t> cpu_boundary_calls(0);
std::atomic<std::uint64_t> cpu_boundary_points(0);
std::atomic<std::uint64_t> cuda_boundary_calls(0);
std::atomic<std::uint64_t> cuda_boundary_points(0);
std::atomic<std::uint64_t> boundary_gather_fast_replays(0);
std::atomic<std::uint64_t> boundary_gather_full_validations(0);
std::atomic<std::uint64_t> boundary_scatter_fast_replays(0);
std::atomic<std::uint64_t> boundary_scatter_full_validations(0);
std::atomic<std::uint64_t> cpu_dft_calls(0);
std::atomic<std::uint64_t> cpu_dft_points(0);
std::atomic<std::uint64_t> cuda_dft_calls(0);
std::atomic<std::uint64_t> cuda_dft_points(0);
std::atomic<std::uint64_t> cpu_ldos_calls(0);
std::atomic<std::uint64_t> cpu_ldos_source_points(0);
std::atomic<std::uint64_t> dft_batch_calls(0);
std::atomic<std::uint64_t> dft_batch_submitted_updates(0);
std::atomic<std::uint64_t> dft_phase_preparation_launches(0);
std::atomic<std::uint64_t> dft_phase_reuses(0);
std::atomic<std::uint64_t> dft_update_kernel_launches(0);
std::atomic<std::uint64_t> dft_maximum_batch_size(0);
std::atomic<std::uint64_t> dft_multi_monitor_automatic_checks(0);
std::atomic<std::uint64_t> dft_multi_monitor_automatic_selected(0);
std::atomic<std::uint64_t> dft_multi_monitor_automatic_rejected(0);
std::atomic<std::uint64_t> dft_multi_monitor_forced_batches(0);
std::atomic<std::uint64_t> dft_multi_monitor_batched_updates(0);
std::atomic<std::uint64_t> dft_multi_monitor_unbatched_updates(0);
std::atomic<std::uint64_t> dft_multi_monitor_plan_uploads(0);
std::atomic<std::uint64_t> dft_multi_monitor_plan_reuses(0);
std::atomic<std::uint64_t> dft_multi_monitor_metadata_h2d_bytes(0);
std::atomic<std::uint64_t> cpu_dft_reduction_calls(0);
std::atomic<std::uint64_t> cpu_dft_reduction_pairs(0);
std::atomic<std::uint64_t> cpu_dft_reduction_terms(0);
std::atomic<std::uint64_t> cuda_dft_reduction_calls(0);
std::atomic<std::uint64_t> cuda_dft_reduction_pairs(0);
std::atomic<std::uint64_t> cuda_dft_reduction_terms(0);
std::atomic<std::uint64_t> cuda_dft_reduction_descriptor_uploads(0);
std::atomic<std::uint64_t> cuda_dft_reduction_plan_reuses(0);
std::atomic<std::uint64_t> cuda_dft_reduction_kernel_launches(0);
std::atomic<std::uint64_t> cuda_dft_reduction_result_d2h_bytes(0);
std::atomic<std::uint64_t> dft_reduction_full_dft_d2h_bytes_avoided(0);
std::atomic<std::uint64_t> dft_reduction_mpi_allreduce_calls(0);
std::atomic<std::uint64_t> dft_reduction_mpi_allreduce_bytes(0);
std::atomic<std::uint64_t> cpu_dft_array_materialization_calls(0);
std::atomic<std::uint64_t> cpu_dft_array_materialization_points(0);
std::atomic<std::uint64_t> cuda_dft_array_materialization_calls(0);
std::atomic<std::uint64_t> cuda_dft_array_materialization_points(0);
std::atomic<std::uint64_t> host_synthetic_material_array_calls(0);
std::atomic<std::uint64_t> host_synthetic_material_array_points(0);
std::atomic<std::uint64_t> cpu_dft_output_calls(0);
std::atomic<std::uint64_t> cpu_dft_output_points(0);
std::atomic<std::uint64_t> cuda_dft_output_calls(0);
std::atomic<std::uint64_t> cuda_dft_output_points(0);
std::atomic<std::uint64_t> cuda_dft_output_staging_calls(0);
std::atomic<std::uint64_t> cuda_dft_output_staging_points(0);
std::atomic<std::uint64_t> cuda_dft_output_staging_frequencies(0);
std::atomic<std::uint64_t> cuda_dft_output_staging_descriptor_uploads(0);
std::atomic<std::uint64_t> cuda_dft_output_staging_plan_reuses(0);
std::atomic<std::uint64_t> cuda_dft_output_staging_kernel_launches(0);
std::atomic<std::uint64_t> cuda_dft_output_staging_result_d2h_bytes(0);
std::atomic<std::uint64_t>
    cuda_dft_output_staging_full_dft_d2h_bytes_avoided(0);
std::atomic<std::uint64_t> cuda_dft_output_staging_workspace_ceiling_bytes(0);
std::atomic<std::uint64_t> cpu_dft_overlap_calls(0);
std::atomic<std::uint64_t> cpu_dft_overlap_terms(0);
std::atomic<std::uint64_t> cuda_dft_overlap_calls(0);
std::atomic<std::uint64_t> cuda_dft_overlap_terms(0);
std::atomic<std::uint64_t> cuda_eigenmode_mode_flux_calls(0);
std::atomic<std::uint64_t> cuda_eigenmode_mode_mode_calls(0);
std::atomic<std::uint64_t> cuda_eigenmode_submitted_pairs(0);
std::atomic<std::uint64_t> cuda_eigenmode_descriptor_uploads(0);
std::atomic<std::uint64_t> cuda_eigenmode_plan_reuses(0);
std::atomic<std::uint64_t> cuda_eigenmode_kernel_launches(0);
std::atomic<std::uint64_t> cuda_eigenmode_result_d2h_bytes(0);
std::atomic<std::uint64_t> eigenmode_full_dft_d2h_bytes_avoided(0);
std::atomic<std::uint64_t> host_mode_profile_sampling_calls(0);
std::atomic<std::uint64_t> host_mode_profile_sampling_points(0);
std::atomic<std::uint64_t> eigenmode_zero_rank_channels_skipped(0);
std::atomic<std::uint64_t> host_mode_profile_h2d_bytes(0);
std::atomic<std::uint64_t> eigenmode_mpi_allreduce_calls(0);
std::atomic<std::uint64_t> eigenmode_mpi_allreduce_bytes(0);
std::atomic<std::uint64_t> cuda_dft_materialization_kernel_launches(0);
std::atomic<std::uint64_t> cuda_dft_materialization_result_d2h_bytes(0);
std::atomic<std::uint64_t> dft_materialization_full_dft_d2h_bytes_avoided(0);
std::atomic<std::uint64_t> dft_array_mpi_allreduce_calls(0);
std::atomic<std::uint64_t> dft_array_mpi_allreduce_bytes(0);
std::atomic<std::uint64_t> cpu_dft_checkpoint_save_calls(0);
std::atomic<std::uint64_t> cpu_dft_checkpoint_save_values(0);
std::atomic<std::uint64_t> cuda_dft_checkpoint_save_calls(0);
std::atomic<std::uint64_t> cuda_dft_checkpoint_save_values(0);
std::atomic<std::uint64_t> cuda_dft_checkpoint_save_d2h_bytes(0);
std::atomic<std::uint64_t> cuda_dft_checkpoint_save_full_cache_d2h_avoided(0);
std::atomic<std::uint64_t> cpu_dft_checkpoint_load_calls(0);
std::atomic<std::uint64_t> cpu_dft_checkpoint_load_values(0);
std::atomic<std::uint64_t> cuda_dft_checkpoint_load_calls(0);
std::atomic<std::uint64_t> cuda_dft_checkpoint_load_values(0);
std::atomic<std::uint64_t> cuda_dft_checkpoint_load_h2d_bytes(0);
std::atomic<std::uint64_t> cpu_dft_scale_calls(0);
std::atomic<std::uint64_t> cpu_dft_scale_values(0);
std::atomic<std::uint64_t> cuda_dft_scale_calls(0);
std::atomic<std::uint64_t> cuda_dft_scale_values(0);
std::atomic<std::uint64_t> cuda_dft_scale_kernel_launches(0);
std::atomic<std::uint64_t> cuda_dft_scale_h2d_bytes(0);
std::atomic<std::int64_t> dft_checkpoint_d2h_failure_after_for_testing(-1);
std::atomic<std::int64_t> dft_checkpoint_h2d_failure_after_for_testing(-1);
std::atomic<std::int64_t>
    dft_output_staging_d2h_failure_after_for_testing(-1);
std::atomic<std::uint64_t> live_dft_output_pinned_buffers(0);
std::atomic<std::uint64_t> ldos_batch_calls(0);
std::atomic<std::uint64_t> ldos_submitted_profiles(0);
std::atomic<std::uint64_t> ldos_source_points(0);
std::atomic<std::uint64_t> ldos_descriptor_uploads(0);
std::atomic<std::uint64_t> ldos_kernel_launches(0);
std::atomic<std::uint64_t> ldos_result_device_to_host_bytes(0);
std::atomic<std::uint64_t> ldos_full_field_device_to_host_bytes_avoided(0);
std::atomic<std::uint64_t> cpu_near2far_calls(0);
std::atomic<std::uint64_t> cpu_near2far_terms(0);
std::atomic<std::uint64_t> cuda_near2far_calls(0);
std::atomic<std::uint64_t> cuda_near2far_terms(0);
std::atomic<std::uint64_t> cuda_near2far_submitted_chunks(0);
std::atomic<std::uint64_t> cuda_near2far_source_points(0);
std::atomic<std::uint64_t> cuda_near2far_output_points(0);
std::atomic<std::uint64_t> cuda_near2far_frequencies(0);
std::atomic<std::uint64_t> cuda_near2far_periodic_copies(0);
std::atomic<std::uint64_t> cuda_near2far_fast_precision_calls(0);
std::atomic<std::uint64_t> cuda_near2far_mixed_precision_calls(0);
std::atomic<std::uint64_t> cuda_near2far_cancellation_retries(0);
std::atomic<std::uint64_t> cuda_near2far_target_tiles(0);
std::atomic<std::uint64_t> cuda_near2far_frequency_tiles(0);
std::atomic<std::uint64_t> cuda_near2far_operation_tiles(0);
std::atomic<std::uint64_t> cuda_near2far_maximum_workspace_bytes(0);
std::atomic<std::uint64_t> cuda_near2far_descriptor_uploads(0);
std::atomic<std::uint64_t> cuda_near2far_kernel_launches(0);
std::atomic<std::uint64_t> cuda_near2far_result_device_to_host_bytes(0);
std::atomic<std::uint64_t> cuda_near2far_condition_device_to_host_bytes(0);
std::atomic<std::uint64_t> cuda_near2far_dft_device_to_host_bytes_avoided(0);
std::atomic<std::uint64_t> near2far_mpi_allreduce_calls(0);
std::atomic<std::uint64_t> near2far_mpi_allreduce_bytes(0);
std::atomic<std::uint64_t> cpu_near2far_adjoint_calls(0);
std::atomic<std::uint64_t> cpu_near2far_adjoint_terms(0);
std::atomic<std::uint64_t> cuda_near2far_adjoint_calls(0);
std::atomic<std::uint64_t> cuda_near2far_adjoint_terms(0);
std::atomic<std::uint64_t> cuda_near2far_adjoint_submitted_chunks(0);
std::atomic<std::uint64_t> cuda_near2far_adjoint_source_points(0);
std::atomic<std::uint64_t> cuda_near2far_adjoint_far_points(0);
std::atomic<std::uint64_t> cuda_near2far_adjoint_frequencies(0);
std::atomic<std::uint64_t> cuda_near2far_adjoint_periodic_copies(0);
std::atomic<std::uint64_t> cuda_near2far_adjoint_fast_precision_calls(0);
std::atomic<std::uint64_t> cuda_near2far_adjoint_mixed_precision_calls(0);
std::atomic<std::uint64_t> cuda_near2far_adjoint_cancellation_retries(0);
std::atomic<std::uint64_t> cuda_near2far_adjoint_maximum_workspace_bytes(0);
std::atomic<std::uint64_t> cuda_near2far_adjoint_descriptor_uploads(0);
std::atomic<std::uint64_t> cuda_near2far_adjoint_kernel_launches(0);
std::atomic<std::uint64_t> cuda_near2far_adjoint_host_to_device_bytes(0);
std::atomic<std::uint64_t> cuda_near2far_adjoint_result_device_to_host_bytes(0);
std::atomic<std::uint64_t> cuda_near2far_adjoint_condition_device_to_host_bytes(0);
std::atomic<std::uint64_t> mpi_boundary_messages(0);
std::atomic<std::uint64_t> mpi_boundary_scalars(0);
std::atomic<std::uint64_t> cuda_aware_mpi_bytes(0);
std::atomic<std::uint64_t> pinned_mpi_bytes(0);
std::atomic<std::uint64_t> pinned_mpi_device_to_host_bytes(0);
std::atomic<std::uint64_t> pinned_mpi_host_to_device_bytes(0);
std::atomic<std::uint64_t> mpi_waitsome_executions(0);
std::atomic<std::uint64_t> mpi_waitall_executions(0);
std::atomic<std::uint64_t> host_to_device_bytes(0);
std::atomic<std::uint64_t> device_to_host_bytes(0);
std::atomic<std::uint64_t> finite_check_host_to_device_bytes(0);
std::atomic<std::uint64_t> finite_check_device_to_host_bytes(0);
std::atomic<std::uint64_t> host_to_device_bytes_avoided(0);
std::atomic<std::uint64_t> device_to_host_bytes_avoided(0);
std::atomic<std::uint64_t> device_buffer_allocations(0);
std::atomic<std::uint64_t> device_buffer_reuses(0);
std::atomic<std::uint64_t> boundary_phase_graph_creations(0);
std::atomic<std::uint64_t> boundary_phase_graph_launches(0);
std::atomic<std::uint64_t> boundary_receive_secondary_allocations(0);
std::atomic<std::uint64_t> boundary_receive_pingpong_selections(0);
std::atomic<std::uint64_t> boundary_receive_secondary_selections(0);
std::atomic<std::uint64_t> boundary_event_record_failures_for_testing(0);
std::atomic<std::uint64_t>
    boundary_event_synchronize_failures_for_testing(0);
std::atomic<std::uint64_t>
    boundary_device_synchronize_failures_for_testing(0);
std::atomic<std::uint64_t> boundary_device_synchronize_fallbacks(0);
std::atomic<std::uint64_t> boundary_lifetime_uncertain_leaks(0);
std::atomic<std::uint64_t> near2far_execution_failures_for_testing(0);
std::atomic<std::uint64_t> boundary_eh_overlap_checks(0);
std::atomic<std::uint64_t> boundary_eh_overlap_eligible(0);
std::atomic<std::uint64_t> boundary_eh_overlap_launched_h(0);
std::atomic<std::uint64_t> boundary_eh_overlap_launched_e(0);
std::atomic<std::uint64_t> boundary_eh_overlap_skipped_disabled(0);
std::atomic<std::uint64_t>
    boundary_eh_overlap_skipped_unsupported_schedule(0);
std::atomic<std::uint64_t> boundary_eh_overlap_skipped_no_remote(0);
std::atomic<std::uint64_t> boundary_eh_overlap_skipped_cold_topology(0);
std::atomic<std::uint64_t> boundary_eh_overlap_rejected(0);
std::atomic<std::uint64_t> halo_curl_overlap_checks(0);
std::atomic<std::uint64_t> halo_curl_overlap_eligible(0);
std::atomic<std::uint64_t> halo_curl_overlap_launches(0);
std::atomic<std::uint64_t> halo_curl_overlap_skipped_disabled(0);
std::atomic<std::uint64_t> halo_curl_overlap_skipped_unsupported_schedule(0);
std::atomic<std::uint64_t> halo_curl_overlap_skipped_no_remote(0);
std::atomic<std::uint64_t> halo_curl_overlap_skipped_cold_topology(0);
std::atomic<std::uint64_t> halo_curl_overlap_rejected_feature(0);
std::atomic<std::uint64_t> halo_curl_overlap_rejected_small(0);
std::atomic<std::uint64_t> halo_curl_overlap_full_points(0);
std::atomic<std::uint64_t> halo_curl_overlap_interior_points(0);
std::atomic<std::uint64_t> halo_curl_overlap_shell_points(0);
std::atomic<std::uint64_t> tile_coalesced_curl_chunk_phases(0);
std::atomic<std::uint64_t> tile_coalesced_curl_input_tiles(0);
std::atomic<std::uint64_t> tile_coalesced_update_eh_chunk_phases(0);
std::atomic<std::uint64_t> tile_coalesced_update_eh_input_tiles(0);
std::atomic<std::uint64_t> phase_curl_automatic_checks(0);
std::atomic<std::uint64_t> phase_curl_automatic_selected(0);
std::atomic<std::uint64_t> phase_curl_automatic_rejected(0);
std::atomic<std::uint64_t> phase_curl_forced_batches(0);
std::atomic<std::uint64_t> phase_curl_batched_operations(0);
std::atomic<std::uint64_t> phase_curl_unbatched_operations(0);
std::atomic<std::uint64_t> phase_curl_replay_checks(0);
std::atomic<std::uint64_t> phase_curl_replay_hits(0);
std::atomic<std::uint64_t> phase_curl_replay_unready(0);
std::atomic<std::uint64_t> phase_curl_replay_generation_misses(0);
std::atomic<std::uint64_t> phase_curl_replay_mirror_misses(0);
std::atomic<std::uint64_t>
    curl_phase_replay_launch_failures_for_testing(0);
std::atomic<std::uint64_t> phase_update_eh_automatic_checks(0);
std::atomic<std::uint64_t> phase_update_eh_automatic_selected(0);
std::atomic<std::uint64_t> phase_update_eh_automatic_rejected(0);
std::atomic<std::uint64_t> phase_update_eh_forced_batches(0);
std::atomic<std::uint64_t> phase_update_eh_batched_operations(0);
std::atomic<std::uint64_t> phase_update_eh_unbatched_operations(0);
std::atomic<bool> fail_multilevel_plan_commit_for_testing(false);
std::atomic<bool> fail_ldos_plan_commit_for_testing(false);
std::atomic<std::size_t> near2far_workspace_ceiling_bytes(
    64u * 1024u * 1024u);

std::atomic<std::uint64_t> &live_device_buffer_count() {
  // Resident owners may have static storage duration and be destroyed after
  // ordinary namespace-scope objects. Match the process-long registry
  // lifetime so late cache teardown never touches a destroyed gauge.
  static std::atomic<std::uint64_t> *count =
      new std::atomic<std::uint64_t>(0);
  return *count;
}

std::uint64_t next_boundary_plan_generation() noexcept {
  static std::atomic<std::uint64_t> generation(0);
  return generation.fetch_add(1, std::memory_order_relaxed) + 1;
}

#if MEEP_HAVE_CUDA
struct resident_registry {
  std::mutex mutex;
  std::unordered_map<const void *, detail::resident_cache *> caches;
};

resident_registry &get_resident_registry() {
  // Owners may have static storage duration and be destroyed after ordinary
  // function-local statics. Keep the registry process-long so their
  // destructors can always remove caches safely during static teardown.
  static resident_registry *registry = new resident_registry;
  return *registry;
}

void release_dft_materialization_plan(
    detail::resident_cache *cache) noexcept {
  if (!cache) return;
  detail::resident_cache::dft_materialization_plan &plan =
      cache->dft_materialization;
  if (cache->device_ordinal >= 0) {
    try {
      meep_cuda::select_device(cache->device_ordinal);
    }
    catch (...) {}
  }
  const std::uint64_t released =
      (plan.device_metadata ? 1u : 0u) +
      (plan.device_output ? 1u : 0u);
  meep_cuda::free_device(plan.device_metadata);
  meep_cuda::free_device(plan.device_output);
  if (released)
    live_device_buffer_count().fetch_sub(
        released, std::memory_order_relaxed);
  plan = detail::resident_cache::dft_materialization_plan();
}

void release_dft_output_staging_plan(
    detail::resident_cache *cache) noexcept {
  if (!cache) return;
  detail::resident_cache::dft_output_staging_plan &plan =
      cache->dft_output_staging;
  if (cache->device_ordinal >= 0) {
    try {
      meep_cuda::select_device(cache->device_ordinal);
    }
    catch (...) {}
  }
  const std::uint64_t released =
      (plan.device_metadata ? 1u : 0u) +
      (plan.device_output ? 1u : 0u);
  meep_cuda::free_device(plan.device_metadata);
  meep_cuda::free_device(plan.device_output);
  if (released)
    live_device_buffer_count().fetch_sub(
        released, std::memory_order_relaxed);
  plan = detail::resident_cache::dft_output_staging_plan();
}

// Caller owns registry.mutex. This is used when a source mirror/cache is
// discarded so plans retained by any other fields_chunk cannot keep a stale
// nested device address. The generation checks in the steady-state lookup are
// a second independent ABA defense.
void clear_dft_materialization_plans_for_cache_locked(
    resident_registry &registry,
    detail::resident_cache *dependency) noexcept {
  for (const auto &entry : registry.caches) {
    detail::resident_cache *const plan_cache = entry.second;
    const auto &materialization_dependencies =
        plan_cache->dft_materialization.dependencies;
    const bool materialization_depends =
        plan_cache == dependency ||
        std::any_of(
            materialization_dependencies.begin(),
            materialization_dependencies.end(),
            [dependency](const auto &candidate) {
              return candidate.cache == dependency;
            });
    if (materialization_depends)
      release_dft_materialization_plan(plan_cache);
    const auto &output_dependencies =
        plan_cache->dft_output_staging.dependencies;
    const bool output_depends =
        plan_cache == dependency ||
        std::any_of(
            output_dependencies.begin(), output_dependencies.end(),
            [dependency](const auto &candidate) {
              return candidate.cache == dependency;
            });
    if (output_depends) release_dft_output_staging_plan(plan_cache);
  }
}

void clear_dft_materialization_plans_for_cache(
    detail::resident_cache *dependency) noexcept {
  if (!dependency) return;
  resident_registry &registry = get_resident_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  clear_dft_materialization_plans_for_cache_locked(
      registry, dependency);
}

std::mutex &get_ldos_reduction_mutex() {
  // LDOS result/plan storage is shared by all profiles in one process. Keep
  // launches and source-profile descriptor invalidation serialized without
  // imposing any synchronization on the ordinary field-update hot path.
  static std::mutex *mutex = new std::mutex;
  return *mutex;
}

struct near2far_request_snapshot {
  detail::resident_cache *cache = nullptr;
  std::uint64_t cache_generation = 0;
  const float *device_dft = nullptr;
  std::size_t point_count = 0;
  int direction = 0;
  bool electric = false;
  std::vector<meep_cuda::cartesian_point_fp64> coordinates;
};

struct near2far_retained_plan {
  int device_ordinal = -1;
  detail::near2far_cartesian_dimension dimension =
      detail::near2far_cartesian_dimension::three;
  bool mixed_precision = false;
  void *static_storage = nullptr;
  std::size_t static_capacity_bytes = 0;
  const void *device_operations = nullptr;
  const void *device_frequencies = nullptr;
  const void *device_periodic_copies = nullptr;
  std::vector<near2far_request_snapshot> requests;
  std::vector<double> frequencies;
  std::vector<meep_cuda::near2far_periodic_copy_fp64> periodic_copies;
  void *target_buffer = nullptr;
  std::size_t target_capacity_bytes = 0;
  void *partial_buffer = nullptr;
  std::size_t partial_capacity_bytes = 0;
  void *output_buffer = nullptr;
  std::size_t output_capacity_bytes = 0;
  std::vector<detail::resident_cache *> cache_dependencies;
};

struct near2far_plan_registry {
  std::mutex mutex;
  std::unordered_map<const void *, near2far_retained_plan> plans;
};

near2far_plan_registry &get_near2far_plan_registry() {
  static near2far_plan_registry *registry = new near2far_plan_registry;
  return *registry;
}

struct near2far_adjoint_retained_plan {
  int device_ordinal = -1;
  bool mixed_precision = false;
  detail::near2far_cartesian_dimension dimension =
      detail::near2far_cartesian_dimension::three;
  double eps = 0.0;
  double mu = 0.0;
  double azimuthal_mode = 0.0;
  double greencyl_tolerance = 0.0;
  void *buffers[7] = {};
  std::size_t capacities[7] = {};
  bool sources_valid = false;
  bool targets_valid = false;
  bool frequencies_valid = false;
  bool copies_valid = false;
  bool gradient_valid = false;
  std::vector<meep_cuda::near2far_adjoint_source_fp64> sources;
  std::vector<meep_cuda::cartesian_point_fp64> targets;
  std::vector<double> frequencies;
  std::vector<meep_cuda::near2far_periodic_copy_fp64> copies;
  std::vector<meep_cuda::complex_value_fp64> gradient;
};

struct near2far_adjoint_plan_registry {
  std::mutex mutex;
  std::unordered_map<const void *, near2far_adjoint_retained_plan> plans;
};

near2far_adjoint_plan_registry &get_near2far_adjoint_plan_registry() {
  static near2far_adjoint_plan_registry *registry =
      new near2far_adjoint_plan_registry;
  return *registry;
}

bool same_near2far_point(
    const meep_cuda::cartesian_point_fp64 &left,
    const meep_cuda::cartesian_point_fp64 &right) noexcept {
  return left.x == right.x && left.y == right.y && left.z == right.z;
}

bool same_near2far_copy(
    const meep_cuda::near2far_periodic_copy_fp64 &left,
    const meep_cuda::near2far_periodic_copy_fp64 &right) noexcept {
  return same_near2far_point(left.displacement, right.displacement) &&
         left.phase.real == right.phase.real &&
         left.phase.imag == right.phase.imag;
}

bool same_near2far_adjoint_source(
    const meep_cuda::near2far_adjoint_source_fp64 &left,
    const meep_cuda::near2far_adjoint_source_fp64 &right) noexcept {
  return same_near2far_point(left.point, right.point) &&
         left.amplitude.real == right.amplitude.real &&
         left.amplitude.imag == right.amplitude.imag &&
         left.direction == right.direction && left.electric == right.electric;
}

bool same_near2far_adjoint_sources(
    const std::vector<meep_cuda::near2far_adjoint_source_fp64> &left,
    const std::vector<meep_cuda::near2far_adjoint_source_fp64> &right) noexcept {
  if (left.size() != right.size()) return false;
  for (std::size_t index = 0; index < left.size(); ++index)
    if (!same_near2far_adjoint_source(left[index], right[index])) return false;
  return true;
}

bool same_near2far_points(
    const std::vector<meep_cuda::cartesian_point_fp64> &left,
    const std::vector<meep_cuda::cartesian_point_fp64> &right) noexcept {
  if (left.size() != right.size()) return false;
  for (std::size_t index = 0; index < left.size(); ++index)
    if (!same_near2far_point(left[index], right[index])) return false;
  return true;
}

bool same_near2far_copies(
    const std::vector<meep_cuda::near2far_periodic_copy_fp64> &left,
    const std::vector<meep_cuda::near2far_periodic_copy_fp64> &right) noexcept {
  if (left.size() != right.size()) return false;
  for (std::size_t index = 0; index < left.size(); ++index)
    if (!same_near2far_copy(left[index], right[index])) return false;
  return true;
}

bool same_near2far_gradient(
    const std::vector<meep_cuda::complex_value_fp64> &left,
    const std::vector<meep_cuda::complex_value_fp64> &right) noexcept {
  if (left.size() != right.size()) return false;
  for (std::size_t index = 0; index < left.size(); ++index)
    if (left[index].real != right[index].real ||
        left[index].imag != right[index].imag)
      return false;
  return true;
}

void release_near2far_adjoint_plan(
    near2far_adjoint_retained_plan &plan) noexcept {
  if (plan.device_ordinal >= 0) {
    try {
      meep_cuda::select_device(plan.device_ordinal);
    }
    catch (...) {}
  }
  std::uint64_t released = 0;
  for (std::size_t buffer = 0; buffer < 7; ++buffer) {
    meep_cuda::free_device(plan.buffers[buffer]);
    if (plan.buffers[buffer]) ++released;
  }
  if (released) live_device_buffer_count().fetch_sub(released);
  plan = near2far_adjoint_retained_plan();
}

void clear_all_near2far_adjoint_plans() noexcept {
  near2far_adjoint_plan_registry &registry =
      get_near2far_adjoint_plan_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (auto &entry : registry.plans)
    release_near2far_adjoint_plan(entry.second);
  registry.plans.clear();
}

void clear_near2far_adjoint_plan_for_owner(const void *owner) noexcept {
  if (!owner) return;
  near2far_adjoint_plan_registry &registry =
      get_near2far_adjoint_plan_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto found = registry.plans.find(owner);
  if (found == registry.plans.end()) return;
  release_near2far_adjoint_plan(found->second);
  registry.plans.erase(found);
}

bool same_near2far_request(
    const near2far_request_snapshot &left,
    const near2far_request_snapshot &right) noexcept {
  if (left.cache != right.cache ||
      left.cache_generation != right.cache_generation ||
      left.device_dft != right.device_dft ||
      left.point_count != right.point_count ||
      left.direction != right.direction || left.electric != right.electric ||
      left.coordinates.size() != right.coordinates.size())
    return false;
  for (std::size_t index = 0; index < left.coordinates.size(); ++index)
    if (!same_near2far_point(left.coordinates[index],
                             right.coordinates[index]))
      return false;
  return true;
}

bool same_near2far_static_plan(
    const near2far_retained_plan &plan,
    const std::vector<near2far_request_snapshot> &requests,
    const std::vector<double> &frequencies,
    const std::vector<meep_cuda::near2far_periodic_copy_fp64> &copies,
    int device_ordinal, detail::near2far_cartesian_dimension dimension,
    bool mixed_precision) noexcept {
  if (!plan.static_storage || plan.device_ordinal != device_ordinal ||
      plan.dimension != dimension ||
      plan.mixed_precision != mixed_precision ||
      plan.requests.size() != requests.size() ||
      plan.frequencies != frequencies ||
      plan.periodic_copies.size() != copies.size())
    return false;
  for (std::size_t index = 0; index < requests.size(); ++index)
    if (!same_near2far_request(plan.requests[index], requests[index]))
      return false;
  for (std::size_t index = 0; index < copies.size(); ++index)
    if (!same_near2far_copy(plan.periodic_copies[index], copies[index]))
      return false;
  return true;
}

void release_near2far_plan(near2far_retained_plan &plan) noexcept {
  if (plan.device_ordinal >= 0) {
    try {
      meep_cuda::select_device(plan.device_ordinal);
    }
    catch (...) {}
  }
  const std::uint64_t released =
      (plan.static_storage ? 1u : 0u) +
      (plan.target_buffer ? 1u : 0u) +
      (plan.partial_buffer ? 1u : 0u) +
      (plan.output_buffer ? 1u : 0u);
  meep_cuda::free_device(plan.static_storage);
  meep_cuda::free_device(plan.target_buffer);
  meep_cuda::free_device(plan.partial_buffer);
  meep_cuda::free_device(plan.output_buffer);
  if (released) live_device_buffer_count().fetch_sub(released);
  plan = near2far_retained_plan();
}

void clear_all_near2far_plans() noexcept {
  near2far_plan_registry &registry = get_near2far_plan_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (auto &entry : registry.plans) release_near2far_plan(entry.second);
  registry.plans.clear();
  clear_all_near2far_adjoint_plans();
}

void clear_near2far_plans_for_cache(
    detail::resident_cache *cache) noexcept {
  near2far_plan_registry &registry = get_near2far_plan_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (auto entry = registry.plans.begin(); entry != registry.plans.end();) {
    const bool depends_on_cache =
        std::find(entry->second.cache_dependencies.begin(),
                  entry->second.cache_dependencies.end(), cache) !=
        entry->second.cache_dependencies.end();
    if (depends_on_cache) {
      release_near2far_plan(entry->second);
      entry = registry.plans.erase(entry);
    }
    else
      ++entry;
  }
}

void clear_near2far_plan_for_owner(const void *owner) noexcept {
  if (!owner) return;
  near2far_plan_registry &registry = get_near2far_plan_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto found = registry.plans.find(owner);
  if (found == registry.plans.end()) return;
  release_near2far_plan(found->second);
  registry.plans.erase(found);
}

struct dft_reduction_plan_key {
  const void *owner = nullptr;
  unsigned int lane = 0;

  bool operator==(const dft_reduction_plan_key &other) const noexcept {
    return owner == other.owner && lane == other.lane;
  }
};

struct dft_reduction_plan_key_hash {
  std::size_t operator()(const dft_reduction_plan_key &key) const noexcept {
    const std::size_t pointer_hash =
        std::hash<const void *>()(key.owner);
    const std::size_t lane_hash =
        std::hash<unsigned int>()(key.lane);
    return pointer_hash ^
           (lane_hash + static_cast<std::size_t>(0x9e3779b9u) +
            (pointer_hash << 6) + (pointer_hash >> 2));
  }
};

struct dft_reduction_request_snapshot {
  detail::resident_cache *lhs_cache = nullptr;
  detail::resident_cache *rhs_cache = nullptr;
  std::uint64_t lhs_cache_generation = 0;
  std::uint64_t rhs_cache_generation = 0;
  const float *lhs_host = nullptr;
  const float *rhs_host = nullptr;
  const float *lhs_device = nullptr;
  const float *rhs_device = nullptr;
  std::size_t point_count = 0;
  float weight_real = 0.0f;
  float weight_imaginary = 0.0f;
};

struct dft_reduction_retained_plan {
  int device_ordinal = -1;
  void *static_storage = nullptr;
  std::size_t static_capacity_bytes = 0;
  const meep_cuda::dft_pair_reduction_operation_fp32 *device_operations =
      nullptr;
  const std::uint32_t *device_block_map = nullptr;
  double *partial_workspace = nullptr;
  double *device_result = nullptr;
  std::size_t partial_capacity = 0;
  std::size_t frequency_count = 0;
  std::size_t total_spatial_block_count = 0;
  std::vector<dft_reduction_request_snapshot> requests;
  std::vector<detail::resident_cache *> cache_dependencies;
};

struct dft_reduction_plan_registry {
  std::mutex mutex;
  std::unordered_map<dft_reduction_plan_key,
                     dft_reduction_retained_plan,
                     dft_reduction_plan_key_hash>
      plans;
};

dft_reduction_plan_registry &get_dft_reduction_plan_registry() {
  static dft_reduction_plan_registry *registry =
      new dft_reduction_plan_registry;
  return *registry;
}

bool same_dft_reduction_request(
    const dft_reduction_request_snapshot &left,
    const dft_reduction_request_snapshot &right) noexcept {
  return left.lhs_cache == right.lhs_cache &&
         left.rhs_cache == right.rhs_cache &&
         left.lhs_cache_generation == right.lhs_cache_generation &&
         left.rhs_cache_generation == right.rhs_cache_generation &&
         left.lhs_host == right.lhs_host &&
         left.rhs_host == right.rhs_host &&
         left.lhs_device == right.lhs_device &&
         left.rhs_device == right.rhs_device &&
         left.point_count == right.point_count &&
         std::memcmp(&left.weight_real, &right.weight_real,
                     sizeof(float)) == 0 &&
         std::memcmp(&left.weight_imaginary, &right.weight_imaginary,
                     sizeof(float)) == 0;
}

bool same_dft_reduction_plan(
    const dft_reduction_retained_plan &plan,
    const std::vector<dft_reduction_request_snapshot> &requests,
    int device_ordinal, std::size_t frequency_count,
    std::size_t total_spatial_block_count) noexcept {
  if (!plan.static_storage || !plan.partial_workspace ||
      !plan.device_result || plan.device_ordinal != device_ordinal ||
      plan.frequency_count != frequency_count ||
      plan.total_spatial_block_count != total_spatial_block_count ||
      plan.requests.size() != requests.size())
    return false;
  for (std::size_t index = 0; index < requests.size(); ++index)
    if (!same_dft_reduction_request(plan.requests[index], requests[index]))
      return false;
  return true;
}

void release_dft_reduction_plan(
    dft_reduction_retained_plan &plan) noexcept {
  if (plan.device_ordinal >= 0) {
    try {
      meep_cuda::select_device(plan.device_ordinal);
    }
    catch (...) {}
  }
  const std::uint64_t released =
      (plan.static_storage ? 1u : 0u) +
      (plan.partial_workspace ? 1u : 0u) +
      (plan.device_result ? 1u : 0u);
  meep_cuda::free_device(plan.static_storage);
  meep_cuda::free_device(plan.partial_workspace);
  meep_cuda::free_device(plan.device_result);
  if (released) live_device_buffer_count().fetch_sub(released);
  plan = dft_reduction_retained_plan();
}

void clear_all_dft_reduction_plans() noexcept {
  dft_reduction_plan_registry &registry =
      get_dft_reduction_plan_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (auto &entry : registry.plans)
    release_dft_reduction_plan(entry.second);
  registry.plans.clear();
}

void clear_dft_reduction_plans_for_cache(
    detail::resident_cache *cache) noexcept {
  dft_reduction_plan_registry &registry =
      get_dft_reduction_plan_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (auto entry = registry.plans.begin(); entry != registry.plans.end();) {
    const bool depends_on_cache =
        std::find(entry->second.cache_dependencies.begin(),
                  entry->second.cache_dependencies.end(), cache) !=
        entry->second.cache_dependencies.end();
    if (depends_on_cache) {
      release_dft_reduction_plan(entry->second);
      entry = registry.plans.erase(entry);
    }
    else
      ++entry;
  }
}

void clear_dft_reduction_plans_for_owner(const void *owner) noexcept {
  if (!owner) return;
  dft_reduction_plan_registry &registry =
      get_dft_reduction_plan_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (auto entry = registry.plans.begin(); entry != registry.plans.end();) {
    if (entry->first.owner == owner) {
      release_dft_reduction_plan(entry->second);
      entry = registry.plans.erase(entry);
    }
    else
      ++entry;
  }
}

struct eigenmode_overlap_plan_key {
  const void *owner = nullptr;
  unsigned int lane = 0;

  bool operator==(const eigenmode_overlap_plan_key &other) const noexcept {
    return owner == other.owner && lane == other.lane;
  }
};

struct eigenmode_overlap_plan_key_hash {
  std::size_t operator()(
      const eigenmode_overlap_plan_key &key) const noexcept {
    const std::size_t pointer_hash =
        std::hash<const void *>()(key.owner);
    const std::size_t lane_hash =
        std::hash<unsigned int>()(key.lane);
    return pointer_hash ^
           (lane_hash + static_cast<std::size_t>(0x9e3779b9u) +
            (pointer_hash << 6) + (pointer_hash >> 2));
  }
};

struct eigenmode_overlap_request_snapshot {
  detail::resident_cache *cache = nullptr;
  std::uint64_t cache_generation = 0;
  const float *dft_host = nullptr;
  const float *dft_device = nullptr;
  std::size_t point_count = 0;
  std::size_t dft_storage_point_count = 0;
  std::size_t dft_frequency_count = 0;
  std::size_t mode1_offset = 0;
  std::size_t mode2_offset = 0;
  std::size_t zero_flag_offset = 0;
  bool has_zero_flags = false;
  std::uint32_t output_index = 0;
  double inverse_weight_real = 0.0;
  double inverse_weight_imaginary = 0.0;
  bool mode_mode = false;
};

struct eigenmode_overlap_retained_plan {
  int device_ordinal = -1;
  void *static_storage = nullptr;
  meep_cuda::complex_value_fp64 *profile_storage = nullptr;
  double *partial_workspace = nullptr;
  double *device_result = nullptr;
  const meep_cuda::eigenmode_overlap_operation_fp32 *device_operations =
      nullptr;
  const std::uint32_t *device_block_map = nullptr;
  std::size_t profile_value_count = 0;
  std::size_t zero_flag_count = 0;
  std::size_t output_count = 0;
  std::size_t partial_capacity = 0;
  std::size_t total_block_count = 0;
  std::vector<eigenmode_overlap_request_snapshot> requests;
  std::vector<detail::resident_cache *> cache_dependencies;
};

struct eigenmode_overlap_plan_registry {
  std::mutex mutex;
  std::unordered_map<eigenmode_overlap_plan_key,
                     eigenmode_overlap_retained_plan,
                     eigenmode_overlap_plan_key_hash>
      plans;
};

eigenmode_overlap_plan_registry &get_eigenmode_overlap_plan_registry() {
  static eigenmode_overlap_plan_registry *registry =
      new eigenmode_overlap_plan_registry;
  return *registry;
}

bool same_eigenmode_overlap_request(
    const eigenmode_overlap_request_snapshot &left,
    const eigenmode_overlap_request_snapshot &right) noexcept {
  return left.cache == right.cache &&
         left.cache_generation == right.cache_generation &&
         left.dft_host == right.dft_host &&
         left.dft_device == right.dft_device &&
         left.point_count == right.point_count &&
         left.dft_storage_point_count == right.dft_storage_point_count &&
         left.dft_frequency_count == right.dft_frequency_count &&
         left.mode1_offset == right.mode1_offset &&
         left.mode2_offset == right.mode2_offset &&
         left.zero_flag_offset == right.zero_flag_offset &&
         left.has_zero_flags == right.has_zero_flags &&
         left.output_index == right.output_index &&
         std::memcmp(&left.inverse_weight_real,
                     &right.inverse_weight_real, sizeof(double)) == 0 &&
         std::memcmp(&left.inverse_weight_imaginary,
                     &right.inverse_weight_imaginary,
                     sizeof(double)) == 0 &&
         left.mode_mode == right.mode_mode;
}

bool same_eigenmode_overlap_plan(
    const eigenmode_overlap_retained_plan &plan,
    const std::vector<eigenmode_overlap_request_snapshot> &requests,
    int device_ordinal, std::size_t profile_value_count,
    std::size_t zero_flag_count,
    std::size_t output_count,
    std::size_t total_block_count) noexcept {
  if (!plan.static_storage || !plan.profile_storage ||
      !plan.partial_workspace || !plan.device_result ||
      plan.device_ordinal != device_ordinal ||
      plan.profile_value_count != profile_value_count ||
      plan.zero_flag_count != zero_flag_count ||
      plan.output_count != output_count ||
      plan.total_block_count != total_block_count ||
      plan.requests.size() != requests.size())
    return false;
  for (std::size_t index = 0; index < requests.size(); ++index)
    if (!same_eigenmode_overlap_request(plan.requests[index],
                                        requests[index]))
      return false;
  return true;
}

void release_eigenmode_overlap_plan(
    eigenmode_overlap_retained_plan &plan) noexcept {
  if (plan.device_ordinal >= 0) {
    try {
      meep_cuda::select_device(plan.device_ordinal);
    }
    catch (...) {}
  }
  const std::uint64_t released =
      (plan.static_storage ? 1u : 0u) +
      (plan.profile_storage ? 1u : 0u) +
      (plan.partial_workspace ? 1u : 0u) +
      (plan.device_result ? 1u : 0u);
  meep_cuda::free_device(plan.static_storage);
  meep_cuda::free_device(plan.profile_storage);
  meep_cuda::free_device(plan.partial_workspace);
  meep_cuda::free_device(plan.device_result);
  if (released) live_device_buffer_count().fetch_sub(released);
  plan = eigenmode_overlap_retained_plan();
}

void clear_all_eigenmode_overlap_plans() noexcept {
  eigenmode_overlap_plan_registry &registry =
      get_eigenmode_overlap_plan_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (auto &entry : registry.plans)
    release_eigenmode_overlap_plan(entry.second);
  registry.plans.clear();
}

void clear_eigenmode_overlap_plans_for_cache(
    detail::resident_cache *cache) noexcept {
  eigenmode_overlap_plan_registry &registry =
      get_eigenmode_overlap_plan_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (auto entry = registry.plans.begin(); entry != registry.plans.end();) {
    const bool depends_on_cache =
        std::find(entry->second.cache_dependencies.begin(),
                  entry->second.cache_dependencies.end(), cache) !=
        entry->second.cache_dependencies.end();
    if (depends_on_cache) {
      release_eigenmode_overlap_plan(entry->second);
      entry = registry.plans.erase(entry);
    }
    else
      ++entry;
  }
}

void clear_eigenmode_overlap_plans_for_owner(
    const void *owner) noexcept {
  if (!owner) return;
  eigenmode_overlap_plan_registry &registry =
      get_eigenmode_overlap_plan_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (auto entry = registry.plans.begin(); entry != registry.plans.end();) {
    if (entry->first.owner == owner) {
      release_eigenmode_overlap_plan(entry->second);
      entry = registry.plans.erase(entry);
    }
    else
      ++entry;
  }
}

void invalidate_ldos_plan_snapshot(
    detail::resident_cache *cache) noexcept {
  if (!cache) return;
  // Keep the active-slot marker so the next transactional rebuild targets
  // the other slot.  Although its nested pointers may now be stale, it is
  // never launched while snapshot_valid is false and remains untouched until
  // a complete replacement has committed.
  cache->ldos_plan_snapshot_valid = false;
  cache->ldos_plan_descriptor_bytes = 0;
  cache->ldos_plan_total_block_count = 0;
  cache->ldos_host_operations.clear();
}

void invalidate_all_ldos_plan_snapshots_locked(
    resident_registry &registry) noexcept {
  for (auto &entry : registry.caches)
    invalidate_ldos_plan_snapshot(entry.second);
}

struct boundary_exchange_key {
  const void *token;
  std::size_t slot;

  bool operator==(const boundary_exchange_key &other) const noexcept {
    return token == other.token && slot == other.slot;
  }
};

struct boundary_exchange_key_hash {
  std::size_t operator()(const boundary_exchange_key &key) const noexcept {
    const std::size_t token_hash = std::hash<const void *>()(key.token);
    return token_hash ^
           (key.slot + static_cast<std::size_t>(0x9e3779b9U) +
            (token_hash << 6) + (token_hash >> 2));
  }
};

struct boundary_exchange_owner {
  std::unordered_map<boundary_exchange_key,
                     detail::boundary_exchange_buffer *,
                     boundary_exchange_key_hash> buffers;
  struct operation_plan {
    void *device_pointer = nullptr;
    meep_cuda::boundary_graph_fp32 *execution_graph = nullptr;
    std::size_t operation_count = 0;
    std::uint64_t generation = 0;
    std::size_t scalar_count = 0;
    bool all_sources_null = false;
    bool nonalias_safe = false;
    int device_ordinal = -1;
    std::size_t stable_reuses = 0;
    std::vector<detail::boundary_operation_fp32> host_operations;
    std::vector<std::pair<detail::resident_cache *, std::uint64_t> >
        cache_generations;
    std::vector<std::pair<detail::resident_cache *, const float *> >
        writable_mirrors;
  };
  std::unordered_map<std::size_t, operation_plan *> operation_plans;
};

struct boundary_exchange_registry {
  std::mutex mutex;
  std::unordered_map<const void *, boundary_exchange_owner> owners;
};

boundary_exchange_registry &get_boundary_exchange_registry() {
  static boundary_exchange_registry *registry =
      new boundary_exchange_registry;
  return *registry;
}

struct curl_phase_plan {
  struct mirror_dependency {
    detail::resident_cache *cache = nullptr;
    float *host_pointer = nullptr;
    std::size_t bytes = 0;

    bool operator==(const mirror_dependency &other) const noexcept {
      return cache == other.cache && host_pointer == other.host_pointer &&
             bytes == other.bytes;
    }
  };

  void *device_operations = nullptr;
  std::size_t capacity_bytes = 0;
  int device_ordinal = -1;
  std::size_t descriptor_bytes = 0;
  std::size_t operation_count = 0;
  std::size_t total_block_count = 0;
  std::uint64_t point_count = 0;
  std::uint64_t topology_fingerprint = 0;
  std::size_t stable_reuses = 0;
  bool complete_phase = false;
  detail::phase_batch_mode mode = detail::phase_batch_mode::disabled;
  std::vector<meep_cuda::curl_phase_operation_fp32> host_operations;
  std::vector<detail::resident_cache *> cache_dependencies;
  std::vector<std::pair<detail::resident_cache *, std::uint64_t> >
      cache_generations;
  std::vector<mirror_dependency> mirror_dependencies;
  std::vector<mirror_dependency> writable_mirrors;
};

struct curl_phase_registry {
  std::mutex mutex;
  std::map<std::pair<const void *, int>, curl_phase_plan> plans;
};

curl_phase_registry &get_curl_phase_registry() {
  static curl_phase_registry *registry = new curl_phase_registry;
  return *registry;
}

struct curl_phase_collection {
  bool active = false;
  detail::phase_batch_mode mode = detail::phase_batch_mode::disabled;
  const void *owner = nullptr;
  int phase_key = 0;
  int device_ordinal = -1;
  std::uint64_t point_count = 0;
  std::uint64_t topology_fingerprint = 0;
  std::vector<meep_cuda::curl_phase_operation_fp32> operations;
  std::vector<detail::resident_cache *> cache_dependencies;
  std::vector<curl_phase_plan::mirror_dependency> mirror_dependencies;
  std::vector<curl_phase_plan::mirror_dependency> writable_mirrors;
  std::size_t flush_count = 0;
  std::size_t batched_flush_count = 0;
  bool replayed = false;
  bool replay_forbidden = false;
  bool replay_supported = false;
};

thread_local curl_phase_collection pending_curl_phase;

struct update_eh_phase_plan {
  void *device_operations = nullptr;
  std::size_t capacity_bytes = 0;
  int device_ordinal = -1;
  std::vector<meep_cuda::update_eh_phase_operation_fp32>
      host_operations;
  std::vector<detail::resident_cache *> cache_dependencies;
};

struct update_eh_phase_registry {
  std::mutex mutex;
  std::map<std::pair<const void *, int>, update_eh_phase_plan> plans;
};

update_eh_phase_registry &get_update_eh_phase_registry() {
  static update_eh_phase_registry *registry =
      new update_eh_phase_registry;
  return *registry;
}

struct update_eh_phase_collection {
  bool active = false;
  detail::phase_batch_mode mode = detail::phase_batch_mode::disabled;
  const void *owner = nullptr;
  int phase_key = 0;
  int device_ordinal = -1;
  std::uint64_t point_count = 0;
  std::vector<meep_cuda::update_eh_phase_operation_fp32> operations;
  std::vector<detail::resident_cache *> cache_dependencies;
};

thread_local update_eh_phase_collection pending_update_eh_phase;

int phase_batch_multiprocessor_count(int device_ordinal) {
  static thread_local int cached_ordinal = -1;
  static thread_local int cached_multiprocessor_count = 0;
  if (cached_ordinal == device_ordinal &&
      cached_multiprocessor_count > 0)
    return cached_multiprocessor_count;
  const std::vector<meep_cuda::device_info> devices =
      meep_cuda::enumerate_devices();
  for (const meep_cuda::device_info &device : devices)
    if (device.ordinal == device_ordinal) {
      if (device.multiprocessor_count <= 0)
        throw std::runtime_error(
            "selected CUDA device reports no multiprocessors");
      cached_ordinal = device_ordinal;
      cached_multiprocessor_count = device.multiprocessor_count;
      return cached_multiprocessor_count;
    }
  throw std::runtime_error(
      "selected CUDA device disappeared during phase-batch policy");
}

struct source_phase_plan {
  void *device_operations = nullptr;
  std::size_t capacity_bytes = 0;
  int device_ordinal = -1;
  std::vector<meep_cuda::indexed_source_phase_operation_fp32>
      host_operations;
  std::vector<detail::resident_cache *> cache_dependencies;
};

struct source_phase_registry {
  std::mutex mutex;
  std::map<std::tuple<const void *, int, std::size_t>, source_phase_plan>
      plans;
};

source_phase_registry &get_source_phase_registry() {
  static source_phase_registry *registry = new source_phase_registry;
  return *registry;
}

struct source_phase_collection {
  bool active = false;
  const void *owner = nullptr;
  int phase_key = 0;
  std::size_t group_index = 0;
  int device_ordinal = -1;
  std::uint64_t point_count = 0;
  std::vector<meep_cuda::indexed_source_phase_operation_fp32>
      operations;
  std::vector<meep_cuda::complex_value_fp32> time_scales;
  std::vector<detail::resident_cache *> cache_dependencies;
};

thread_local source_phase_collection pending_source_phase;

std::size_t checked_phase_product(std::size_t count,
                                  std::size_t element_size,
                                  const char *label) {
  if (element_size &&
      count > std::numeric_limits<std::size_t>::max() / element_size)
    throw std::overflow_error(std::string("Meep CUDA ") + label +
                              " byte count overflow");
  return count * element_size;
}

std::size_t checked_phase_sum(std::size_t left, std::size_t right,
                              const char *label) {
  if (left > std::numeric_limits<std::size_t>::max() - right)
    throw std::overflow_error(std::string("Meep CUDA ") + label +
                              " storage overflow");
  return left + right;
}

template <typename Operation>
std::vector<std::uint32_t> make_phase_block_operation_indices(
    const std::vector<Operation> &operations,
    std::size_t total_block_count) {
  static_assert(
      sizeof(Operation) % alignof(std::uint32_t) == 0,
      "phase descriptor storage must preserve block-map alignment");
  std::vector<std::size_t> block_starts;
  block_starts.reserve(operations.size());
  for (const auto &operation : operations)
    block_starts.push_back(operation.block_start);
  return detail::plan_phase_block_operation_indices(
      block_starts.data(), block_starts.size(), total_block_count);
}

bool same_index_space(const meep_cuda::index_space_fp32 &left,
                      const meep_cuda::index_space_fp32 &right) {
  return left.field_start == right.field_start &&
         left.extent1 == right.extent1 &&
         left.extent2 == right.extent2 &&
         left.extent3 == right.extent3 &&
         left.field_stride1 == right.field_stride1 &&
         left.field_stride2 == right.field_stride2 &&
         left.field_stride3 == right.field_stride3 &&
         left.coefficient_start == right.coefficient_start &&
         left.coefficient_stride1 == right.coefficient_stride1 &&
         left.coefficient_stride2 == right.coefficient_stride2 &&
         left.coefficient_stride3 == right.coefficient_stride3 &&
         left.coefficient2_start == right.coefficient2_start &&
         left.coefficient2_stride1 == right.coefficient2_stride1 &&
         left.coefficient2_stride2 == right.coefficient2_stride2 &&
         left.coefficient2_stride3 == right.coefficient2_stride3;
}

bool same_curl_material(const meep_cuda::curl_material_fp32 &left,
                        const meep_cuda::curl_material_fp32 &right) {
  return left.sigma == right.sigma && left.kappa == right.kappa &&
         left.sigma_inverse == right.sigma_inverse &&
         left.field_u == right.field_u &&
         left.sigma_u == right.sigma_u &&
         left.kappa_u == right.kappa_u &&
         left.sigma_u_inverse == right.sigma_u_inverse &&
         left.dt == right.dt &&
         left.conductivity == right.conductivity &&
         left.conductivity_inverse == right.conductivity_inverse &&
         left.field_conductivity == right.field_conductivity;
}

bool same_curl_phase_operation(
    const meep_cuda::curl_phase_operation_fp32 &left,
    const meep_cuda::curl_phase_operation_fp32 &right) {
  return left.field == right.field && left.g1 == right.g1 &&
         left.g2 == right.g2 &&
         same_index_space(left.inline_space, right.inline_space) &&
         left.point_count == right.point_count &&
         left.block_start == right.block_start &&
         left.stride1 == right.stride1 &&
         left.stride2 == right.stride2 && left.dtdx == right.dtdx &&
         same_curl_material(left.material, right.material);
}

bool same_curl_phase_operations(
    const std::vector<meep_cuda::curl_phase_operation_fp32> &left,
    const std::vector<meep_cuda::curl_phase_operation_fp32> &right) {
  if (left.size() != right.size()) return false;
  for (std::size_t index = 0; index < left.size(); ++index)
    if (!same_curl_phase_operation(left[index], right[index]))
      return false;
  return true;
}

void release_curl_phase_plan(curl_phase_plan &plan) noexcept {
  if (plan.device_ordinal >= 0) {
    try {
      meep_cuda::select_device(plan.device_ordinal);
    }
    catch (...) {}
  }
  meep_cuda::free_device(plan.device_operations);
  if (plan.device_operations) live_device_buffer_count().fetch_sub(1);
  plan.device_operations = nullptr;
  plan.capacity_bytes = 0;
  plan.device_ordinal = -1;
  plan.descriptor_bytes = 0;
  plan.operation_count = 0;
  plan.total_block_count = 0;
  plan.point_count = 0;
  plan.stable_reuses = 0;
  plan.complete_phase = false;
  plan.mode = detail::phase_batch_mode::disabled;
  plan.host_operations.clear();
  plan.cache_dependencies.clear();
  plan.cache_generations.clear();
  plan.mirror_dependencies.clear();
  plan.writable_mirrors.clear();
}

void clear_all_curl_phase_plans() noexcept {
  curl_phase_registry &registry = get_curl_phase_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (auto &entry : registry.plans)
    release_curl_phase_plan(entry.second);
  registry.plans.clear();
}

void clear_curl_phase_plans_for_owner(const void *owner) noexcept {
  curl_phase_registry &registry = get_curl_phase_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (auto entry = registry.plans.begin();
       entry != registry.plans.end();) {
    if (entry->first.first == owner) {
      release_curl_phase_plan(entry->second);
      entry = registry.plans.erase(entry);
    }
    else
      ++entry;
  }
}

void clear_curl_phase_plans_for_cache(
    detail::resident_cache *cache) noexcept {
  curl_phase_registry &registry = get_curl_phase_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (auto entry = registry.plans.begin();
       entry != registry.plans.end();) {
    const bool depends_on_cache =
        std::find(entry->second.cache_dependencies.begin(),
                  entry->second.cache_dependencies.end(), cache) !=
        entry->second.cache_dependencies.end();
    if (depends_on_cache) {
      release_curl_phase_plan(entry->second);
      entry = registry.plans.erase(entry);
    }
    else
      ++entry;
  }
}

void mark_curl_phase_plan_incomplete(const void *owner, int phase_key) {
  curl_phase_registry &registry = get_curl_phase_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto found = registry.plans.find(std::make_pair(owner, phase_key));
  if (found != registry.plans.end()) found->second.complete_phase = false;
}

void mark_curl_phase_plan_incomplete_noexcept(
    const void *owner, int phase_key) noexcept {
  try {
    mark_curl_phase_plan_incomplete(owner, phase_key);
  }
  catch (...) {
    // The plan was already made incomplete before ordinary collection began.
    // This helper is used only by unwinding paths where throwing is forbidden.
  }
}

void certify_complete_curl_phase_noexcept(
    const curl_phase_collection &collection) noexcept {
  try {
    curl_phase_registry &registry = get_curl_phase_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.plans.find(
        std::make_pair(collection.owner, collection.phase_key));
    if (found == registry.plans.end()) return;
    curl_phase_plan &plan = found->second;
    plan.complete_phase = collection.flush_count == 1 &&
                          collection.batched_flush_count == 1 &&
                          !collection.replay_forbidden;
    if (!plan.complete_phase) plan.stable_reuses = 0;
  }
  catch (...) {
    // Collection makes a prior plan incomplete before any kernel is issued.
    // Failure to certify therefore leaves replay disabled, not unsafe.
  }
}

bool collect_curl_phase_operation(
    detail::resident_cache *cache, int device_ordinal,
    const meep_cuda::curl_phase_operation_fp32 &operation,
    std::uint64_t point_count) {
  curl_phase_collection &collection = pending_curl_phase;
  if (!collection.active) return false;
  if (!cache)
    throw std::invalid_argument(
        "Meep CUDA phase-batched curl cache must be non-null");
  if (collection.device_ordinal < 0)
    collection.device_ordinal = device_ordinal;
  else if (collection.device_ordinal != device_ordinal)
    throw std::invalid_argument(
        "Meep CUDA phase-batched curls must use one device");
  if (collection.operations.empty() && collection.flush_count == 0)
    mark_curl_phase_plan_incomplete(
        collection.owner, collection.phase_key);
  collection.operations.push_back(operation);
  if (std::find(collection.cache_dependencies.begin(),
                collection.cache_dependencies.end(), cache) ==
      collection.cache_dependencies.end())
    collection.cache_dependencies.push_back(cache);
  if (collection.point_count >
      std::numeric_limits<std::uint64_t>::max() - point_count)
    throw std::overflow_error(
        "Meep CUDA phase-batched curl point count overflow");
  collection.point_count += point_count;
  return true;
}

void collect_curl_phase_mirror(
    std::vector<curl_phase_plan::mirror_dependency> *mirrors,
    detail::resident_cache *cache, const float *host_pointer,
    std::size_t bytes) {
  if (!host_pointer) return;
  curl_phase_collection &collection = pending_curl_phase;
  if (!collection.active)
    throw std::logic_error(
        "Meep CUDA curl mirror recorded outside phase collection");
  const curl_phase_plan::mirror_dependency dependency = {
      cache, const_cast<float *>(host_pointer), bytes};
  if (std::find(mirrors->begin(), mirrors->end(), dependency) ==
      mirrors->end())
    mirrors->push_back(dependency);
}

void collect_curl_phase_dependency(detail::resident_cache *cache,
                                   const float *host_pointer,
                                   std::size_t bytes) {
  collect_curl_phase_mirror(
      &pending_curl_phase.mirror_dependencies, cache, host_pointer, bytes);
}

void collect_curl_phase_writable(detail::resident_cache *cache,
                                 float *host_pointer,
                                 std::size_t bytes) {
  collect_curl_phase_mirror(
      &pending_curl_phase.writable_mirrors, cache, host_pointer, bytes);
}

void flush_pending_curl_phase() {
  curl_phase_collection &collection = pending_curl_phase;
  if (!collection.active || collection.operations.empty()) return;
  ++collection.flush_count;
  if (collection.device_ordinal < 0)
    throw std::logic_error(
        "Meep CUDA phase-batched curl has no device");
  meep_cuda::select_device(collection.device_ordinal);
  // A singleton cannot save a launch. Preserve the lower-overhead structured
  // kernel instead of paying for a resident descriptor and block map.
  if (collection.operations.size() == 1) {
    const auto &operation = collection.operations.front();
    meep_cuda::step_curl_material_structured_fp32(
        operation.field, operation.g1, operation.g2,
        operation.inline_space, operation.point_count, operation.stride1,
        operation.stride2, operation.dtdx, operation.material);
    cuda_curl_calls.fetch_add(1);
    cuda_curl_points.fetch_add(collection.point_count);
    phase_curl_unbatched_operations.fetch_add(
        1, std::memory_order_relaxed);
    collection.operations.clear();
    collection.cache_dependencies.clear();
    collection.mirror_dependencies.clear();
    collection.writable_mirrors.clear();
    collection.point_count = 0;
    return;
  }
  constexpr std::size_t phase_threads = 256;
  std::size_t total_block_count = 0;
  std::size_t maximum_operation_block_count = 0;
  for (auto &operation : collection.operations) {
    operation.block_start = total_block_count;
    const std::size_t operation_blocks =
        operation.point_count / phase_threads +
        (operation.point_count % phase_threads != 0);
    if (total_block_count >
        std::numeric_limits<std::size_t>::max() - operation_blocks)
      throw std::overflow_error(
          "Meep CUDA phase-batched curl block count overflow");
    total_block_count += operation_blocks;
    maximum_operation_block_count =
        std::max(maximum_operation_block_count, operation_blocks);
  }
  const bool use_phase_kernel =
      collection.mode == detail::phase_batch_mode::forced ||
      (collection.mode == detail::phase_batch_mode::automatic &&
       detail::automatic_phase_batch_selected(
           collection.operations.size(), total_block_count,
           maximum_operation_block_count,
           phase_batch_multiprocessor_count(
               collection.device_ordinal)));
  if (collection.mode == detail::phase_batch_mode::automatic) {
    phase_curl_automatic_checks.fetch_add(1, std::memory_order_relaxed);
    (use_phase_kernel ? phase_curl_automatic_selected
                      : phase_curl_automatic_rejected)
        .fetch_add(1, std::memory_order_relaxed);
  }
  else if (collection.mode == detail::phase_batch_mode::forced)
    phase_curl_forced_batches.fetch_add(1, std::memory_order_relaxed);
  if (!use_phase_kernel) {
    for (const auto &operation : collection.operations)
      meep_cuda::step_curl_material_structured_fp32(
          operation.field, operation.g1, operation.g2,
          operation.inline_space, operation.point_count,
          operation.stride1, operation.stride2, operation.dtdx,
          operation.material);
    cuda_curl_calls.fetch_add(
        static_cast<std::uint64_t>(collection.operations.size()));
    cuda_curl_points.fetch_add(collection.point_count);
    phase_curl_unbatched_operations.fetch_add(
        static_cast<std::uint64_t>(collection.operations.size()),
        std::memory_order_relaxed);
    collection.operations.clear();
    collection.cache_dependencies.clear();
    collection.mirror_dependencies.clear();
    collection.writable_mirrors.clear();
    collection.point_count = 0;
    return;
  }
  const std::size_t descriptor_bytes = checked_phase_product(
      collection.operations.size(),
      sizeof(meep_cuda::curl_phase_operation_fp32),
      "phase-batched curl descriptor");
  const std::size_t block_map_bytes = checked_phase_product(
      total_block_count, sizeof(std::uint32_t),
      "phase-batched curl block map");
  const std::size_t storage_bytes = checked_phase_sum(
      descriptor_bytes, block_map_bytes, "phase-batched curl");
  curl_phase_registry &registry = get_curl_phase_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  curl_phase_plan &plan = registry.plans[
      std::make_pair(collection.owner, collection.phase_key)];
  if (plan.device_operations &&
      plan.device_ordinal != collection.device_ordinal)
    release_curl_phase_plan(plan);

  const bool operations_match =
      plan.device_operations &&
      same_curl_phase_operations(plan.host_operations,
                                 collection.operations);
  bool generations_match =
      plan.cache_generations.size() ==
      collection.cache_dependencies.size();
  for (std::size_t index = 0;
       generations_match && index < collection.cache_dependencies.size();
       ++index)
    generations_match =
        plan.cache_generations[index].first ==
            collection.cache_dependencies[index] &&
        plan.cache_generations[index].second ==
            collection.cache_dependencies[index]->allocation_generation;
  const bool stable_topology =
      operations_match && generations_match &&
      plan.mode == collection.mode &&
      plan.topology_fingerprint == collection.topology_fingerprint &&
      plan.mirror_dependencies == collection.mirror_dependencies &&
      plan.writable_mirrors == collection.writable_mirrors;
  if (!operations_match) {
    const std::vector<std::uint32_t> block_operation_indices =
        make_phase_block_operation_indices(
            collection.operations, total_block_count);
    const auto copy_plan_to_device = [&](void *destination) {
      meep_cuda::copy_to_device(
          destination, collection.operations.data(), descriptor_bytes);
      if (block_map_bytes)
        meep_cuda::copy_to_device(
            static_cast<unsigned char *>(destination) + descriptor_bytes,
            block_operation_indices.data(), block_map_bytes);
    };
    if (!plan.device_operations ||
        plan.capacity_bytes < storage_bytes) {
      void *replacement =
          meep_cuda::allocate_device_bytes(storage_bytes);
      try {
        copy_plan_to_device(replacement);
      }
      catch (...) {
        meep_cuda::free_device(replacement);
        throw;
      }
      void *old = plan.device_operations;
      plan.device_operations = replacement;
      plan.capacity_bytes = storage_bytes;
      plan.device_ordinal = collection.device_ordinal;
      meep_cuda::free_device(old);
      if (!old) live_device_buffer_count().fetch_add(1);
      device_buffer_allocations.fetch_add(1);
    }
    else
      copy_plan_to_device(plan.device_operations);
    plan.host_operations = collection.operations;
    host_to_device_bytes.fetch_add(
        static_cast<std::uint64_t>(storage_bytes));
  }
  else
    device_buffer_reuses.fetch_add(1);
  plan.cache_dependencies = collection.cache_dependencies;
  plan.cache_generations.clear();
  plan.cache_generations.reserve(collection.cache_dependencies.size());
  for (detail::resident_cache *cache : collection.cache_dependencies)
    plan.cache_generations.push_back(
        std::make_pair(cache, cache->allocation_generation));
  plan.mirror_dependencies = collection.mirror_dependencies;
  plan.writable_mirrors = collection.writable_mirrors;
  plan.descriptor_bytes = descriptor_bytes;
  plan.operation_count = collection.operations.size();
  plan.total_block_count = total_block_count;
  plan.point_count = collection.point_count;
  plan.topology_fingerprint = collection.topology_fingerprint;
  plan.complete_phase = false;
  plan.mode = collection.mode;
  if (stable_topology) {
    if (plan.stable_reuses != std::numeric_limits<std::size_t>::max())
      ++plan.stable_reuses;
  }
  else
    plan.stable_reuses = 0;

  meep_cuda::step_curl_material_phase_batched_fp32(
      static_cast<const meep_cuda::curl_phase_operation_fp32 *>(
          plan.device_operations),
      reinterpret_cast<const std::uint32_t *>(
          static_cast<const unsigned char *>(plan.device_operations) +
          descriptor_bytes),
      collection.operations.size(), total_block_count);
  cuda_curl_calls.fetch_add(1);
  cuda_curl_points.fetch_add(collection.point_count);
  phase_curl_batched_operations.fetch_add(
      static_cast<std::uint64_t>(collection.operations.size()),
      std::memory_order_relaxed);
  ++collection.batched_flush_count;
  collection.operations.clear();
  collection.cache_dependencies.clear();
  collection.mirror_dependencies.clear();
  collection.writable_mirrors.clear();
  collection.point_count = 0;
}

void flush_pending_curl_phase_before_direct_operation() {
  curl_phase_collection &collection = pending_curl_phase;
  if (collection.active) {
    mark_curl_phase_plan_incomplete(
        collection.owner, collection.phase_key);
    collection.replay_forbidden = true;
  }
  flush_pending_curl_phase();
}

bool same_update_eh_material(
    const meep_cuda::update_eh_material_fp32 &left,
    const meep_cuda::update_eh_material_fp32 &right) {
  return left.inverse_susceptibility == right.inverse_susceptibility &&
         left.offdiagonal1 == right.offdiagonal1 &&
         left.offdiagonal2 == right.offdiagonal2 &&
         left.chi2 == right.chi2 && left.chi3 == right.chi3 &&
         left.field_w == right.field_w && left.sigma == right.sigma &&
         left.kappa == right.kappa;
}

bool same_update_eh_phase_operation(
    const meep_cuda::update_eh_phase_operation_fp32 &left,
    const meep_cuda::update_eh_phase_operation_fp32 &right) {
  return left.field == right.field && left.g == right.g &&
         left.g1 == right.g1 && left.g2 == right.g2 &&
         same_index_space(left.index_space, right.index_space) &&
         left.point_count == right.point_count &&
         left.block_start == right.block_start &&
         left.field_stride == right.field_stride &&
         left.stride1 == right.stride1 &&
         left.stride2 == right.stride2 &&
         same_update_eh_material(left.material, right.material);
}

bool same_update_eh_phase_operations(
    const std::vector<meep_cuda::update_eh_phase_operation_fp32> &left,
    const std::vector<meep_cuda::update_eh_phase_operation_fp32> &right) {
  if (left.size() != right.size()) return false;
  for (std::size_t index = 0; index < left.size(); ++index)
    if (!same_update_eh_phase_operation(left[index], right[index]))
      return false;
  return true;
}

void release_update_eh_phase_plan(update_eh_phase_plan &plan) noexcept {
  if (plan.device_ordinal >= 0) {
    try {
      meep_cuda::select_device(plan.device_ordinal);
    }
    catch (...) {}
  }
  meep_cuda::free_device(plan.device_operations);
  if (plan.device_operations) live_device_buffer_count().fetch_sub(1);
  plan.device_operations = nullptr;
  plan.capacity_bytes = 0;
  plan.device_ordinal = -1;
  plan.host_operations.clear();
  plan.cache_dependencies.clear();
}

void clear_all_update_eh_phase_plans() noexcept {
  update_eh_phase_registry &registry = get_update_eh_phase_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (auto &entry : registry.plans)
    release_update_eh_phase_plan(entry.second);
  registry.plans.clear();
}

void clear_update_eh_phase_plans_for_owner(const void *owner) noexcept {
  update_eh_phase_registry &registry = get_update_eh_phase_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (auto entry = registry.plans.begin();
       entry != registry.plans.end();) {
    if (entry->first.first == owner) {
      release_update_eh_phase_plan(entry->second);
      entry = registry.plans.erase(entry);
    }
    else
      ++entry;
  }
}

void clear_update_eh_phase_plans_for_cache(
    detail::resident_cache *cache) noexcept {
  update_eh_phase_registry &registry = get_update_eh_phase_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (auto entry = registry.plans.begin();
       entry != registry.plans.end();) {
    const bool depends_on_cache =
        std::find(entry->second.cache_dependencies.begin(),
                  entry->second.cache_dependencies.end(), cache) !=
        entry->second.cache_dependencies.end();
    if (depends_on_cache) {
      release_update_eh_phase_plan(entry->second);
      entry = registry.plans.erase(entry);
    }
    else
      ++entry;
  }
}

void flush_pending_update_eh_phase() {
  update_eh_phase_collection &collection = pending_update_eh_phase;
  if (!collection.active || collection.operations.empty()) return;
  if (collection.device_ordinal < 0)
    throw std::logic_error(
        "Meep CUDA phase-batched E/H update has no device");
  meep_cuda::select_device(collection.device_ordinal);
  if (collection.operations.size() == 1) {
    const auto &operation = collection.operations.front();
    meep_cuda::update_eh_structured_fp32(
        operation.field, operation.g, operation.g1, operation.g2,
        operation.index_space, operation.point_count,
        operation.field_stride, operation.stride1, operation.stride2,
        operation.material);
    cuda_update_eh_calls.fetch_add(1);
    cuda_update_eh_points.fetch_add(collection.point_count);
    phase_update_eh_unbatched_operations.fetch_add(
        1, std::memory_order_relaxed);
    collection.operations.clear();
    collection.cache_dependencies.clear();
    collection.point_count = 0;
    return;
  }
  constexpr std::size_t phase_threads = 256;
  std::size_t total_block_count = 0;
  std::size_t maximum_operation_block_count = 0;
  for (auto &operation : collection.operations) {
    operation.block_start = total_block_count;
    const std::size_t operation_blocks =
        operation.point_count / phase_threads +
        (operation.point_count % phase_threads != 0);
    if (total_block_count >
        std::numeric_limits<std::size_t>::max() - operation_blocks)
      throw std::overflow_error(
          "Meep CUDA phase-batched E/H block count overflow");
    total_block_count += operation_blocks;
    maximum_operation_block_count =
        std::max(maximum_operation_block_count, operation_blocks);
  }
  const bool use_phase_kernel =
      collection.mode == detail::phase_batch_mode::forced ||
      (collection.mode == detail::phase_batch_mode::automatic &&
       detail::automatic_phase_batch_selected(
           collection.operations.size(), total_block_count,
           maximum_operation_block_count,
           phase_batch_multiprocessor_count(
               collection.device_ordinal)));
  if (collection.mode == detail::phase_batch_mode::automatic) {
    phase_update_eh_automatic_checks.fetch_add(
        1, std::memory_order_relaxed);
    (use_phase_kernel ? phase_update_eh_automatic_selected
                      : phase_update_eh_automatic_rejected)
        .fetch_add(1, std::memory_order_relaxed);
  }
  else if (collection.mode == detail::phase_batch_mode::forced)
    phase_update_eh_forced_batches.fetch_add(
        1, std::memory_order_relaxed);
  if (!use_phase_kernel) {
    for (const auto &operation : collection.operations)
      meep_cuda::update_eh_structured_fp32(
          operation.field, operation.g, operation.g1, operation.g2,
          operation.index_space, operation.point_count,
          operation.field_stride, operation.stride1,
          operation.stride2, operation.material);
    cuda_update_eh_calls.fetch_add(
        static_cast<std::uint64_t>(collection.operations.size()));
    cuda_update_eh_points.fetch_add(collection.point_count);
    phase_update_eh_unbatched_operations.fetch_add(
        static_cast<std::uint64_t>(collection.operations.size()),
        std::memory_order_relaxed);
    collection.operations.clear();
    collection.cache_dependencies.clear();
    collection.point_count = 0;
    return;
  }
  const std::size_t descriptor_bytes = checked_phase_product(
      collection.operations.size(),
      sizeof(meep_cuda::update_eh_phase_operation_fp32),
      "phase-batched E/H descriptor");
  const std::size_t block_map_bytes = checked_phase_product(
      total_block_count, sizeof(std::uint32_t),
      "phase-batched E/H block map");
  const std::size_t storage_bytes = checked_phase_sum(
      descriptor_bytes, block_map_bytes, "phase-batched E/H");
  update_eh_phase_registry &registry = get_update_eh_phase_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  update_eh_phase_plan &plan = registry.plans[
      std::make_pair(collection.owner, collection.phase_key)];
  if (plan.device_operations &&
      plan.device_ordinal != collection.device_ordinal)
    release_update_eh_phase_plan(plan);
  const bool operations_match =
      plan.device_operations &&
      same_update_eh_phase_operations(
          plan.host_operations, collection.operations);
  if (!operations_match) {
    const std::vector<std::uint32_t> block_operation_indices =
        make_phase_block_operation_indices(
            collection.operations, total_block_count);
    const auto copy_plan_to_device = [&](void *destination) {
      meep_cuda::copy_to_device(
          destination, collection.operations.data(), descriptor_bytes);
      if (block_map_bytes)
        meep_cuda::copy_to_device(
            static_cast<unsigned char *>(destination) + descriptor_bytes,
            block_operation_indices.data(), block_map_bytes);
    };
    if (!plan.device_operations ||
        plan.capacity_bytes < storage_bytes) {
      void *replacement =
          meep_cuda::allocate_device_bytes(storage_bytes);
      try {
        copy_plan_to_device(replacement);
      }
      catch (...) {
        meep_cuda::free_device(replacement);
        throw;
      }
      void *old = plan.device_operations;
      plan.device_operations = replacement;
      plan.capacity_bytes = storage_bytes;
      plan.device_ordinal = collection.device_ordinal;
      meep_cuda::free_device(old);
      if (!old) live_device_buffer_count().fetch_add(1);
      device_buffer_allocations.fetch_add(1);
    }
    else
      copy_plan_to_device(plan.device_operations);
    plan.host_operations = collection.operations;
    host_to_device_bytes.fetch_add(
        static_cast<std::uint64_t>(storage_bytes));
  }
  else
    device_buffer_reuses.fetch_add(1);
  plan.cache_dependencies = collection.cache_dependencies;

  meep_cuda::update_eh_phase_batched_fp32(
      static_cast<const meep_cuda::update_eh_phase_operation_fp32 *>(
          plan.device_operations),
      reinterpret_cast<const std::uint32_t *>(
          static_cast<const unsigned char *>(plan.device_operations) +
          descriptor_bytes),
      collection.operations.size(), total_block_count);
  cuda_update_eh_calls.fetch_add(1);
  cuda_update_eh_points.fetch_add(collection.point_count);
  phase_update_eh_batched_operations.fetch_add(
      static_cast<std::uint64_t>(collection.operations.size()),
      std::memory_order_relaxed);
  collection.operations.clear();
  collection.cache_dependencies.clear();
  collection.point_count = 0;
}

bool collect_update_eh_phase_operation(
    detail::resident_cache *cache, int device_ordinal,
    const meep_cuda::update_eh_phase_operation_fp32 &operation,
    std::uint64_t point_count) {
  update_eh_phase_collection &collection = pending_update_eh_phase;
  if (!collection.active) return false;
  if (!cache)
    throw std::invalid_argument(
        "Meep CUDA phase-batched E/H cache must be non-null");
  if (collection.device_ordinal < 0)
    collection.device_ordinal = device_ordinal;
  else if (collection.device_ordinal != device_ordinal)
    throw std::invalid_argument(
        "Meep CUDA phase-batched E/H updates must use one device");
  for (const auto &pending : collection.operations)
    if (pending.field == operation.field) {
      flush_pending_update_eh_phase();
      break;
    }
  collection.operations.push_back(operation);
  if (std::find(collection.cache_dependencies.begin(),
                collection.cache_dependencies.end(), cache) ==
      collection.cache_dependencies.end())
    collection.cache_dependencies.push_back(cache);
  if (collection.point_count >
      std::numeric_limits<std::uint64_t>::max() - point_count)
    throw std::overflow_error(
        "Meep CUDA phase-batched E/H point count overflow");
  collection.point_count += point_count;
  return true;
}

bool same_source_phase_operation(
    const meep_cuda::indexed_source_phase_operation_fp32 &left,
    const meep_cuda::indexed_source_phase_operation_fp32 &right) {
  return left.destination == right.destination &&
         left.indices == right.indices &&
         left.amplitudes == right.amplitudes &&
         left.conductivity_inverse == right.conductivity_inverse &&
         left.point_count == right.point_count &&
         left.block_start == right.block_start &&
         left.imaginary_component == right.imaginary_component;
}

bool same_source_phase_operations(
    const std::vector<meep_cuda::indexed_source_phase_operation_fp32> &left,
    const std::vector<meep_cuda::indexed_source_phase_operation_fp32> &right) {
  if (left.size() != right.size()) return false;
  for (std::size_t index = 0; index < left.size(); ++index)
    if (!same_source_phase_operation(left[index], right[index]))
      return false;
  return true;
}

void release_source_phase_plan(source_phase_plan &plan) noexcept {
  if (plan.device_ordinal >= 0) {
    try {
      meep_cuda::select_device(plan.device_ordinal);
    }
    catch (...) {}
  }
  meep_cuda::free_device(plan.device_operations);
  if (plan.device_operations) live_device_buffer_count().fetch_sub(1);
  plan.device_operations = nullptr;
  plan.capacity_bytes = 0;
  plan.device_ordinal = -1;
  plan.host_operations.clear();
  plan.cache_dependencies.clear();
}

void clear_all_source_phase_plans() noexcept {
  source_phase_registry &registry = get_source_phase_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (auto &entry : registry.plans)
    release_source_phase_plan(entry.second);
  registry.plans.clear();
}

void clear_source_phase_plans_for_owner(const void *owner) noexcept {
  source_phase_registry &registry = get_source_phase_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (auto entry = registry.plans.begin(); entry != registry.plans.end();) {
    if (std::get<0>(entry->first) == owner) {
      release_source_phase_plan(entry->second);
      entry = registry.plans.erase(entry);
    }
    else
      ++entry;
  }
}

void prune_source_phase_plans_after_group(
    const void *owner, int phase_key,
    std::size_t completed_group_count) noexcept {
  source_phase_registry &registry = get_source_phase_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (auto entry = registry.plans.begin(); entry != registry.plans.end();) {
    const bool obsolete_group =
        std::get<0>(entry->first) == owner &&
        std::get<1>(entry->first) == phase_key &&
        std::get<2>(entry->first) >= completed_group_count;
    if (obsolete_group) {
      release_source_phase_plan(entry->second);
      entry = registry.plans.erase(entry);
    }
    else
      ++entry;
  }
}

void clear_source_phase_plan_for_group(
    const void *owner, int phase_key, std::size_t group_index) noexcept {
  source_phase_registry &registry = get_source_phase_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto key = std::make_tuple(owner, phase_key, group_index);
  const auto found = registry.plans.find(key);
  if (found == registry.plans.end()) return;
  release_source_phase_plan(found->second);
  registry.plans.erase(found);
}

void clear_source_phase_plans_for_cache(
    detail::resident_cache *cache) noexcept {
  source_phase_registry &registry = get_source_phase_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (auto entry = registry.plans.begin(); entry != registry.plans.end();) {
    const bool depends_on_cache =
        std::find(entry->second.cache_dependencies.begin(),
                  entry->second.cache_dependencies.end(), cache) !=
        entry->second.cache_dependencies.end();
    if (depends_on_cache) {
      release_source_phase_plan(entry->second);
      entry = registry.plans.erase(entry);
    }
    else
      ++entry;
  }
}

void invalidate_source_phase_plan_snapshots_for_cache(
    detail::resident_cache *cache) noexcept {
  source_phase_registry &registry = get_source_phase_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (auto &entry : registry.plans)
    if (std::find(entry.second.cache_dependencies.begin(),
                  entry.second.cache_dependencies.end(), cache) !=
        entry.second.cache_dependencies.end())
      entry.second.host_operations.clear();
}

void flush_pending_source_phase() {
  source_phase_collection &collection = pending_source_phase;
  if (!collection.active || collection.operations.empty()) return;
  if (collection.operations.size() != collection.time_scales.size() ||
      collection.operations.size() >
          meep_cuda::indexed_source_phase_max_operations)
    throw std::logic_error(
        "Meep CUDA source batch parameter state is inconsistent");
  if (collection.device_ordinal < 0)
    throw std::logic_error(
        "Meep CUDA phase-batched source has no device");
  meep_cuda::select_device(collection.device_ordinal);
  const auto complete_group = [&collection] {
    collection.operations.clear();
    collection.time_scales.clear();
    collection.cache_dependencies.clear();
    collection.point_count = 0;
    if (collection.group_index ==
        std::numeric_limits<std::size_t>::max())
      throw std::overflow_error(
          "Meep CUDA source batch group count overflow");
    ++collection.group_index;
  };
  // A singleton cannot save a launch. Keep the lower-overhead legacy kernel
  // and avoid creating a resident descriptor plan merely to binary-search a
  // one-element operation table.
  if (collection.operations.size() == 1) {
    const auto &operation = collection.operations.front();
    const auto time_scale = collection.time_scales.front();
    clear_source_phase_plan_for_group(
        collection.owner, collection.phase_key, collection.group_index);
    meep_cuda::indexed_source_subtract_fp32(
        operation.destination, operation.indices, operation.amplitudes,
        operation.conductivity_inverse, operation.point_count, time_scale,
        operation.imaginary_component);
    cuda_source_calls.fetch_add(1);
    cuda_source_points.fetch_add(collection.point_count);
    complete_group();
    return;
  }
  constexpr std::size_t phase_threads = 256;
  std::size_t total_block_count = 0;
  for (auto &operation : collection.operations) {
    operation.block_start = total_block_count;
    const std::size_t operation_blocks =
        operation.point_count / phase_threads +
        (operation.point_count % phase_threads != 0);
    if (total_block_count >
        std::numeric_limits<std::size_t>::max() - operation_blocks)
      throw std::overflow_error(
          "Meep CUDA phase-batched source block count overflow");
    total_block_count += operation_blocks;
  }
  const std::size_t descriptor_bytes = checked_phase_product(
      collection.operations.size(),
      sizeof(meep_cuda::indexed_source_phase_operation_fp32),
      "phase-batched source descriptor");
  const std::size_t block_map_bytes = checked_phase_product(
      total_block_count, sizeof(std::uint32_t),
      "phase-batched source block map");
  const std::size_t storage_bytes = checked_phase_sum(
      descriptor_bytes, block_map_bytes, "phase-batched source");
  source_phase_registry &registry = get_source_phase_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  source_phase_plan &plan = registry.plans[
      std::make_tuple(collection.owner, collection.phase_key,
                      collection.group_index)];
  if (plan.device_operations &&
      plan.device_ordinal != collection.device_ordinal)
    release_source_phase_plan(plan);
  const bool operations_match =
      plan.device_operations &&
      same_source_phase_operations(
          plan.host_operations, collection.operations);
  if (!operations_match) {
    const std::vector<std::uint32_t> block_operation_indices =
        make_phase_block_operation_indices(
            collection.operations, total_block_count);
    const auto copy_plan_to_device = [&](void *destination) {
      meep_cuda::copy_to_device(
          destination, collection.operations.data(), descriptor_bytes);
      if (block_map_bytes)
        meep_cuda::copy_to_device(
            static_cast<unsigned char *>(destination) + descriptor_bytes,
            block_operation_indices.data(), block_map_bytes);
    };
    if (!plan.device_operations ||
        plan.capacity_bytes < storage_bytes) {
      void *replacement =
          meep_cuda::allocate_device_bytes(storage_bytes);
      try {
        copy_plan_to_device(replacement);
      }
      catch (...) {
        meep_cuda::free_device(replacement);
        throw;
      }
      void *old = plan.device_operations;
      plan.device_operations = replacement;
      plan.capacity_bytes = storage_bytes;
      plan.device_ordinal = collection.device_ordinal;
      meep_cuda::free_device(old);
      if (!old) live_device_buffer_count().fetch_add(1);
      device_buffer_allocations.fetch_add(1);
    }
    else {
      copy_plan_to_device(plan.device_operations);
    }
    plan.host_operations = collection.operations;
    host_to_device_bytes.fetch_add(
        static_cast<std::uint64_t>(storage_bytes));
  }
  else
    device_buffer_reuses.fetch_add(1);
  plan.cache_dependencies = collection.cache_dependencies;

  meep_cuda::indexed_source_phase_time_scales_fp32 time_scales = {};
  std::copy(collection.time_scales.begin(), collection.time_scales.end(),
            time_scales.values);
  meep_cuda::indexed_source_subtract_phase_batched_fp32(
      static_cast<
          const meep_cuda::indexed_source_phase_operation_fp32 *>(
              plan.device_operations),
      reinterpret_cast<const std::uint32_t *>(
          static_cast<const unsigned char *>(plan.device_operations) +
          descriptor_bytes),
      collection.operations.size(), total_block_count, time_scales);
  cuda_source_calls.fetch_add(1);
  cuda_source_points.fetch_add(collection.point_count);
  complete_group();
}

bool collect_source_phase_operation(
    detail::resident_cache *cache, int device_ordinal,
    const meep_cuda::indexed_source_phase_operation_fp32 &operation,
    meep_cuda::complex_value_fp32 time_scale,
    std::uint64_t point_count) {
  source_phase_collection &collection = pending_source_phase;
  if (!collection.active) return false;
  if (!cache)
    throw std::invalid_argument(
        "Meep CUDA phase-batched source cache must be non-null");
  if (collection.device_ordinal < 0)
    collection.device_ordinal = device_ordinal;
  else if (collection.device_ordinal != device_ordinal)
    throw std::invalid_argument(
        "Meep CUDA phase-batched sources must use one device");
  // A later source targeting the same field array may overlap the earlier
  // profile. Launch the preceding independent group before accepting it so
  // source application order and FP32 arithmetic match the legacy path.
  const bool repeated_destination =
      std::any_of(collection.operations.begin(), collection.operations.end(),
                  [&](const auto &candidate) {
                    return candidate.destination == operation.destination;
                  });
  if (repeated_destination ||
      collection.operations.size() ==
          meep_cuda::indexed_source_phase_max_operations)
    flush_pending_source_phase();
  collection.operations.push_back(operation);
  collection.time_scales.push_back(time_scale);
  if (std::find(collection.cache_dependencies.begin(),
                collection.cache_dependencies.end(), cache) ==
      collection.cache_dependencies.end())
    collection.cache_dependencies.push_back(cache);
  if (collection.point_count >
      std::numeric_limits<std::uint64_t>::max() - point_count)
    throw std::overflow_error(
        "Meep CUDA phase-batched source point count overflow");
  collection.point_count += point_count;
  return true;
}

void release_boundary_transfer_plan(
    detail::boundary_exchange_buffer::transfer_plan &plan) noexcept {
  meep_cuda::free_device(plan.device_pointer);
  if (plan.device_pointer) live_device_buffer_count().fetch_sub(1);
  plan.device_pointer = nullptr;
  plan.operation_count = 0;
  plan.scalar_count = 0;
  plan.generation = 0;
  plan.gather_caches.clear();
  plan.gather_sources.clear();
  plan.scatter_operations.clear();
  plan.cache_generations.clear();
  plan.writable_mirrors.clear();
  plan.replay_topology_identity = nullptr;
  plan.replay_topology_generation = 0;
}

bool consume_failure_for_testing(
    std::atomic<std::uint64_t> &failures) noexcept {
  std::uint64_t remaining = failures.load(std::memory_order_relaxed);
  while (remaining != 0)
    if (failures.compare_exchange_weak(
            remaining, remaining - 1, std::memory_order_relaxed,
            std::memory_order_relaxed))
      return true;
  return false;
}

bool consume_countdown_failure_for_testing(
    std::atomic<std::int64_t> &countdown) noexcept {
  std::int64_t remaining = countdown.load(std::memory_order_relaxed);
  while (remaining >= 0) {
    const std::int64_t replacement = remaining == 0 ? -1 : remaining - 1;
    if (countdown.compare_exchange_weak(
            remaining, replacement, std::memory_order_relaxed,
            std::memory_order_relaxed))
      return remaining == 0;
  }
  return false;
}

void synchronize_boundary_event_for_lifetime(void *event) {
  if (consume_failure_for_testing(
          boundary_event_synchronize_failures_for_testing))
    throw std::runtime_error(
        "injected CUDA boundary event-synchronize failure");
  meep_cuda::synchronize_event(event);
}

void synchronize_boundary_device_fallback() {
  boundary_device_synchronize_fallbacks.fetch_add(
      1, std::memory_order_relaxed);
  if (consume_failure_for_testing(
          boundary_device_synchronize_failures_for_testing))
    throw std::runtime_error(
        "injected CUDA boundary device-synchronize failure");
  meep_cuda::synchronize();
}

std::string boundary_exception_message(
    const std::exception_ptr &error) noexcept {
  try {
    if (error) std::rethrow_exception(error);
  }
  catch (const std::exception &exception) {
    return exception.what();
  }
  catch (...) {
    return "non-standard exception";
  }
  return "unknown exception";
}

[[noreturn]] void throw_boundary_double_fence_failure(
    const char *operation, const std::exception_ptr &primary,
    const std::exception_ptr &fallback) {
  throw std::runtime_error(
      std::string(operation) + ": " +
      boundary_exception_message(primary) +
      "; CUDA device-synchronize fallback also failed: " +
      boundary_exception_message(fallback));
}

void record_boundary_completion_event(
    detail::boundary_exchange_buffer *buffer) {
  if (!buffer || !buffer->completion_event)
    throw std::invalid_argument(
        "Meep CUDA boundary completion event must be non-null");
  try {
    if (consume_failure_for_testing(
            boundary_event_record_failures_for_testing))
      throw std::runtime_error(
          "injected CUDA boundary event-record failure");
    meep_cuda::record_event(buffer->completion_event);
    buffer->completion_event_recorded = true;
  }
  catch (...) {
    const std::exception_ptr record_error = std::current_exception();
    // Work was already submitted before cudaEventRecord.  Establish a
    // device-wide lifetime fence before propagating the original error.  If
    // even that fails, release_boundary_exchange_buffer deliberately leaks
    // the possibly-live raw allocations instead of risking a UAF.
    try {
      synchronize_boundary_device_fallback();
      buffer->completion_event_recorded = false;
    }
    catch (...) {
      buffer->lifetime_uncertain = true;
      throw_boundary_double_fence_failure(
          "CUDA boundary event record failed", record_error,
          std::current_exception());
    }
    std::rethrow_exception(record_error);
  }
}

void release_boundary_exchange_buffer(
    detail::boundary_exchange_buffer *buffer) noexcept {
  if (!buffer) return;
  if (buffer->device_ordinal >= 0) {
    try {
      meep_cuda::select_device(buffer->device_ordinal);
    }
    catch (...) {
      buffer->lifetime_uncertain = true;
    }
  }
  // A recorded completion event can still reference the transfer plans and
  // device buffer.  Finish it before releasing those resources; cudaFree's
  // incidental synchronization is not a lifetime contract.
  if (!buffer->lifetime_uncertain &&
      buffer->completion_event_recorded)
    try {
      synchronize_boundary_event_for_lifetime(
          buffer->completion_event);
      buffer->completion_event_recorded = false;
    }
    catch (...) {
      try {
        synchronize_boundary_device_fallback();
        buffer->completion_event_recorded = false;
      }
      catch (...) {
        buffer->lifetime_uncertain = true;
      }
    }
  if (buffer->lifetime_uncertain) {
    // Raw device/pinned/event allocations may still be referenced by CUDA.
    // Keep them alive until process teardown; deleting this metadata is safe
    // because submitted work never dereferences the C++ owner object itself.
    boundary_lifetime_uncertain_leaks.fetch_add(
        1, std::memory_order_relaxed);
    delete buffer;
    return;
  }
  release_boundary_transfer_plan(buffer->gather_plan);
  release_boundary_transfer_plan(buffer->scatter_plan);
  meep_cuda::free_device(buffer->device_pointer);
  meep_cuda::free_pinned(buffer->host_pointer);
  meep_cuda::destroy_event(buffer->completion_event);
  if (buffer->device_pointer) live_device_buffer_count().fetch_sub(1);
  delete buffer;
}

void release_boundary_operation_plan(
    boundary_exchange_owner::operation_plan *plan) noexcept {
  if (!plan) return;
  if (plan->device_ordinal >= 0) {
    try {
      meep_cuda::select_device(plan->device_ordinal);
    }
    catch (...) {}
  }
  // A direct kernel or graph launch can still reference both the graph and
  // its device descriptor.  Plan invalidation is rare, so synchronize the
  // selected device explicitly before releasing either resource rather than
  // relying on cudaFree's incidental synchronization behavior.
  try {
    meep_cuda::synchronize();
  }
  catch (...) {}
  meep_cuda::destroy_boundary_graph_fp32(plan->execution_graph);
  meep_cuda::free_device(plan->device_pointer);
  if (plan->device_pointer) live_device_buffer_count().fetch_sub(1);
  delete plan;
}

bool boundary_operation_plan_matches(
    const boundary_exchange_owner::operation_plan *plan,
    const detail::boundary_operation_fp32 *operations,
    std::size_t operation_count) {
  if (!plan || plan->host_operations.size() != operation_count)
    return false;
  for (std::size_t index = 0; index < operation_count; ++index) {
    const detail::boundary_operation_fp32 &left =
        plan->host_operations[index];
    const detail::boundary_operation_fp32 &right = operations[index];
    if (left.source_cache != right.source_cache ||
        left.destination_cache != right.destination_cache ||
        left.destination_real != right.destination_real ||
        left.destination_imag != right.destination_imag ||
        left.source_real != right.source_real ||
        left.source_imag != right.source_imag ||
        left.phase_real != right.phase_real ||
        left.phase_imag != right.phase_imag)
      return false;
  }
  return true;
}

bool boundary_operation_plan_generations_match(
    const boundary_exchange_owner::operation_plan *plan) {
  if (!plan) return false;
  for (const auto &entry : plan->cache_generations)
    if (!entry.first ||
        entry.first->allocation_generation != entry.second)
      return false;
  return true;
}

void clear_all_boundary_exchange_buffers() noexcept {
  boundary_exchange_registry &registry = get_boundary_exchange_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (auto &owner : registry.owners) {
    for (auto &entry : owner.second.buffers)
      release_boundary_exchange_buffer(entry.second);
    for (auto &entry : owner.second.operation_plans)
      release_boundary_operation_plan(entry.second);
  }
  registry.owners.clear();
}

bool boundary_transfer_plan_depends_on_cache(
    const detail::boundary_exchange_buffer::transfer_plan &plan,
    detail::resident_cache *cache) noexcept {
  if (std::find(plan.gather_caches.begin(), plan.gather_caches.end(),
                cache) != plan.gather_caches.end())
    return true;
  for (const auto &entry : plan.cache_generations)
    if (entry.first == cache) return true;
  for (const auto &entry : plan.writable_mirrors)
    if (entry.first == cache) return true;
  for (const detail::remote_boundary_operation_fp32 &operation :
       plan.scatter_operations)
    if (operation.destination_cache == cache) return true;
  return false;
}

bool boundary_operation_plan_depends_on_cache(
    const boundary_exchange_owner::operation_plan *plan,
    detail::resident_cache *cache) noexcept {
  if (!plan) return false;
  for (const auto &entry : plan->cache_generations)
    if (entry.first == cache) return true;
  for (const auto &entry : plan->writable_mirrors)
    if (entry.first == cache) return true;
  for (const detail::boundary_operation_fp32 &operation :
       plan->host_operations)
    if (operation.source_cache == cache ||
        operation.destination_cache == cache)
      return true;
  return false;
}

void clear_boundary_exchange_plans_for_cache(
    detail::resident_cache *cache) noexcept {
  boundary_exchange_registry &registry = get_boundary_exchange_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (auto &owner_entry : registry.owners) {
    boundary_exchange_owner &owner = owner_entry.second;
    for (auto &buffer_entry : owner.buffers) {
      detail::boundary_exchange_buffer *buffer = buffer_entry.second;
      if (!buffer) continue;
      if (boundary_transfer_plan_depends_on_cache(
              buffer->gather_plan, cache))
        release_boundary_transfer_plan(buffer->gather_plan);
      if (boundary_transfer_plan_depends_on_cache(
              buffer->scatter_plan, cache))
        release_boundary_transfer_plan(buffer->scatter_plan);
    }
    for (auto plan_entry = owner.operation_plans.begin();
         plan_entry != owner.operation_plans.end();) {
      if (boundary_operation_plan_depends_on_cache(
              plan_entry->second, cache)) {
        release_boundary_operation_plan(plan_entry->second);
        plan_entry = owner.operation_plans.erase(plan_entry);
      }
      else
        ++plan_entry;
    }
  }
}

detail::resident_cache *resident_cache_for_owner(const void *owner) {
  if (!owner) throw std::invalid_argument("Meep CUDA resident cache owner must be non-null");
  resident_registry &registry = get_resident_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto found = registry.caches.find(owner);
  if (found != registry.caches.end()) return found->second;

  detail::resident_cache *created = new detail::resident_cache;
  try {
    registry.caches.emplace(owner, created);
  }
  catch (...) {
    delete created;
    throw;
  }
  return created;
}

detail::resident_cache *find_resident_cache_for_owner(
    const void *owner) noexcept {
  if (!owner) return nullptr;
  resident_registry &registry = get_resident_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto found = registry.caches.find(owner);
  return found == registry.caches.end() ? nullptr : found->second;
}

detail::resident_cache *remove_resident_cache_for_owner(const void *owner) noexcept {
  resident_registry &registry = get_resident_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto found = registry.caches.find(owner);
  if (found == registry.caches.end()) return nullptr;
  detail::resident_cache *cache = found->second;
  registry.caches.erase(found);
  return cache;
}
#endif

backend_mode parse_mode(const char *value) {
  if (!value || !*value || std::strcmp(value, "cpu") == 0) return backend_mode::cpu;
  if (std::strcmp(value, "auto") == 0 || std::strcmp(value, "automatic") == 0)
    return backend_mode::automatic;
  if (std::strcmp(value, "cuda") == 0) return backend_mode::cuda;
  throw std::invalid_argument(std::string("invalid MEEP_GPU_BACKEND='") + value +
                              "' (expected cpu, auto, or cuda)");
}

#if MEEP_HAVE_CUDA
int parse_device_ordinal(const char *value) {
  if (!value || !*value) return -1;
  errno = 0;
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (errno != 0 || !end || *end != '\0' || parsed < 0 || parsed > INT_MAX)
    throw std::invalid_argument(std::string("invalid MEEP_GPU_DEVICE='") + value +
                                "' (expected a non-negative CUDA runtime ordinal)");
  return static_cast<int>(parsed);
}

bool parse_allow_oversubscription() {
  const char *value = std::getenv("MEEP_GPU_ALLOW_OVERSUBSCRIBE");
  if (!value || !*value || std::strcmp(value, "0") == 0 ||
      std::strcmp(value, "false") == 0 || std::strcmp(value, "no") == 0)
    return false;
  if (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
      std::strcmp(value, "yes") == 0)
    return true;
  throw std::invalid_argument(
      std::string("invalid MEEP_GPU_ALLOW_OVERSUBSCRIBE='") + value +
      "' (expected 0/1, false/true, or no/yes)");
}

bool cuda_visibility_is_rank_isolated() {
  const char *value = std::getenv("CUDA_VISIBLE_DEVICES");
  if (!value || !*value) return false;
  // A single visible token is the standard scheduler/container convention
  // for exposing one distinct physical GPU per rank. Ordinal zero is then
  // correct in every rank-local CUDA namespace.
  return std::strchr(value, ',') == nullptr;
}

int rank_compatible_device() {
  const std::vector<meep_cuda::device_info> devices = meep_cuda::enumerate_devices();
  std::vector<int> compatible_ordinals;
  for (const auto &device : devices)
    if (meep_cuda::device_compatible(device))
      compatible_ordinals.push_back(device.ordinal);
  const bool allow_oversubscription = parse_allow_oversubscription();
  const std::size_t node_size =
      static_cast<std::size_t>(count_node_processors());
  if (!allow_oversubscription && node_size > compatible_ordinals.size() &&
      !(compatible_ordinals.size() == 1 &&
        cuda_visibility_is_rank_isolated())) {
    std::ostringstream message;
    message << "Meep CUDA requires one dedicated visible GPU per node-local "
               "MPI rank: "
            << compatible_ordinals.size() << " compatible devices are visible "
            << "to " << node_size
            << " ranks; assign one CUDA_VISIBLE_DEVICES token per rank or set "
               "MEEP_GPU_ALLOW_OVERSUBSCRIBE=1 explicitly";
    throw std::runtime_error(message.str());
  }
  return detail::choose_rank_device_ordinal(
      compatible_ordinals.data(), compatible_ordinals.size(),
      my_node_rank(), allow_oversubscription);
}
#endif

void configure_locked(backend_mode requested) {
  if (requested != backend_mode::cpu && requested != backend_mode::automatic &&
      requested != backend_mode::cuda)
    throw std::invalid_argument("invalid Meep GPU backend mode");

  state.requested = requested;
  state.active = backend_mode::cpu;
  state.device_ready = false;
  state.diagnostic.clear();

  if (requested == backend_mode::cpu) {
    state.diagnostic = "CPU backend requested";
    state.configured = true;
    return;
  }

#if MEEP_HAVE_CUDA
  // User-authored CUDA policy is configuration, not availability.  Validate
  // it before probing the runtime so a typo or explicit device request cannot
  // be silently reinterpreted as permission for automatic CPU execution.
  (void)parse_allow_oversubscription();
  const char *device_value = std::getenv("MEEP_GPU_DEVICE");
  const bool explicit_device_environment =
      device_value && *device_value;
  if (explicit_device_environment) (void)parse_device_ordinal(device_value);
  const bool explicit_device =
      explicit_device_environment || state.explicit_device_selection;

  // Automatic selection is intentionally two-stage.  Workload policy runs
  // before this process calls cudaGetDeviceCount/cudaSetDevice, so a small
  // CPU-selected owner does not initialize or otherwise touch CUDA.
  if (requested == backend_mode::automatic && !explicit_device) {
    state.diagnostic =
        "automatic backend awaiting host-only workload preflight";
    state.configured = true;
    return;
  }

  std::string runtime_diagnostic;
  const int environment_device =
      explicit_device_environment ? parse_device_ordinal(device_value) : -1;
  const bool runtime_is_available =
      meep_cuda::runtime_available(&runtime_diagnostic);
  int ordinal = state.selected_device;
  std::string selection_error;

  if (!runtime_is_available) {
    if (explicit_device)
      throw std::runtime_error(
          "explicit CUDA device selection failed: " + runtime_diagnostic);
    selection_error = runtime_diagnostic;
  }
  else {
    try {
      if (explicit_device_environment)
        ordinal = environment_device;
      if (ordinal < 0) ordinal = rank_compatible_device();
      if (ordinal < 0)
        throw std::runtime_error(
            "CUDA runtime found no compatible device");
      meep_cuda::select_device(ordinal);
    }
    catch (const std::exception &error) {
      if (explicit_device)
        throw std::runtime_error(
            "explicit CUDA device selection failed: " +
            std::string(error.what()));
      selection_error = error.what();
    }
  }

  if (!selection_error.empty()) {
    state.diagnostic = selection_error;
    state.configured = true;
    if (requested == backend_mode::cuda)
      throw std::runtime_error(
          "CUDA backend device selection failed: " + state.diagnostic);
    return;
  }

  state.selected_device = ordinal;
  state.device_ready = true;
  state.active = backend_mode::cuda;
  std::ostringstream message;
  message << "CUDA device " << ordinal << " selected for global rank "
          << my_global_rank() << " (node rank " << my_node_rank() << "/"
          << count_node_processors() << ")";
  state.diagnostic = message.str();
  state.configured = true;
#else
  state.diagnostic = "Meep was built without CUDA support";
  state.configured = true;
  if (requested == backend_mode::cuda)
    throw std::runtime_error("CUDA backend requested but Meep was built without CUDA support");
#endif
}

void configure_transaction_locked(backend_mode requested) {
  const backend_mode previous_requested = state.requested;
  const backend_mode previous_active = state.active;
  const bool previous_configured = state.configured;
  const bool previous_device_ready = state.device_ready;
  const int previous_selected_device = state.selected_device;
  const std::string previous_environment_signature =
      state.configured_environment_signature;
  const std::string previous_diagnostic = state.diagnostic;
  try {
    configure_locked(requested);
    state.configured_environment_signature =
        backend_configuration_environment_signature();
    ++state.generation;
  }
  catch (...) {
    state.requested = previous_requested;
    state.active = previous_active;
    state.configured = previous_configured;
    state.device_ready = previous_device_ready;
    state.selected_device = previous_selected_device;
    state.configured_environment_signature =
        previous_environment_signature;
    state.diagnostic = previous_diagnostic;
    throw;
  }
}

void ensure_configured_locked() {
  if (!state.configured)
    configure_transaction_locked(parse_mode(std::getenv("MEEP_GPU_BACKEND")));
}

#if MEEP_HAVE_CUDA
std::size_t checked_bytes(std::size_t count, std::size_t element_size, const char *what) {
  if (element_size && count > static_cast<std::size_t>(-1) / element_size)
    throw std::overflow_error(std::string(what) + " byte count overflow");
  return count * element_size;
}

std::size_t checked_product(std::size_t left, std::size_t right,
                            const char *what) {
  if (left && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error(std::string("Meep CUDA ") + what +
                              " size overflow");
  return left * right;
}

struct index_space_bounds {
  std::ptrdiff_t field_min;
  std::ptrdiff_t field_max;
  int coefficient_min;
  int coefficient_max;
  int coefficient2_min;
  int coefficient2_max;
  std::size_t point_count;
};

index_space_bounds validate_index_space(
    const detail::index_space_fp32 &space, std::size_t array_count) {
  if (!space.extent1 || !space.extent2 || !space.extent3)
    throw std::invalid_argument(
        "Meep CUDA structured index extents must be nonzero");
  if (space.extent1 >
      std::numeric_limits<std::size_t>::max() / space.extent2)
    throw std::overflow_error("Meep CUDA structured index count overflow");
  const std::size_t first_two = space.extent1 * space.extent2;
  if (first_two >
      std::numeric_limits<std::size_t>::max() / space.extent3)
    throw std::overflow_error("Meep CUDA structured index count overflow");
  if (array_count >
      static_cast<std::size_t>(
          std::numeric_limits<std::ptrdiff_t>::max()))
    throw std::overflow_error(
        "Meep CUDA field array is too large for ptrdiff_t indexing");
  if (space.field_start < 0 || space.field_stride1 < 0 ||
      space.field_stride2 < 0 || space.field_stride3 < 0)
    throw std::invalid_argument(
        "Meep CUDA structured field start and strides must be nonnegative");

  std::ptrdiff_t field_max = space.field_start;
  const std::size_t extents[] = {
      space.extent1, space.extent2, space.extent3};
  const std::ptrdiff_t field_strides[] = {
      space.field_stride1, space.field_stride2, space.field_stride3};
  for (int axis = 0; axis < 3; ++axis) {
    const std::size_t steps = extents[axis] - 1;
    if (field_strides[axis] &&
        steps > static_cast<std::size_t>(
                    std::numeric_limits<std::ptrdiff_t>::max() /
                    field_strides[axis]))
      throw std::overflow_error(
          "Meep CUDA structured field index overflow");
    const std::ptrdiff_t offset =
        static_cast<std::ptrdiff_t>(steps) * field_strides[axis];
    if (offset >
        std::numeric_limits<std::ptrdiff_t>::max() - field_max)
      throw std::overflow_error(
          "Meep CUDA structured field index overflow");
    field_max += offset;
  }
  if (field_max >= static_cast<std::ptrdiff_t>(array_count))
    throw std::out_of_range(
        "Meep CUDA structured field index is outside the field array");

  const auto coefficient_maximum =
      [&extents](int start, int stride1, int stride2, int stride3) {
        if (start < 0) {
          if (stride1 || stride2 || stride3)
            throw std::invalid_argument(
                "Meep CUDA absent coefficient index has nonzero strides");
          return -1;
        }
        if (stride1 < 0 || stride2 < 0 || stride3 < 0)
          throw std::invalid_argument(
              "Meep CUDA coefficient strides must be nonnegative");
        std::int64_t maximum = start;
        const int strides[] = {stride1, stride2, stride3};
        for (int axis = 0; axis < 3; ++axis) {
          const std::size_t steps = extents[axis] - 1;
          if (strides[axis] &&
              steps > static_cast<std::size_t>(
                          std::numeric_limits<int>::max() /
                          strides[axis]))
            throw std::overflow_error(
                "Meep CUDA structured coefficient index overflow");
          maximum += static_cast<std::int64_t>(steps) * strides[axis];
        }
        if (maximum > std::numeric_limits<int>::max())
          throw std::overflow_error(
              "Meep CUDA structured coefficient index overflow");
        return static_cast<int>(maximum);
      };

  return {
      space.field_start,
      field_max,
      space.coefficient_start,
      coefficient_maximum(
          space.coefficient_start, space.coefficient_stride1,
          space.coefficient_stride2, space.coefficient_stride3),
      space.coefficient2_start,
      coefficient_maximum(
          space.coefficient2_start, space.coefficient2_stride1,
          space.coefficient2_stride2, space.coefficient2_stride3),
      first_two * space.extent3};
}

meep_cuda::index_space_fp32 device_index_space(
    const detail::index_space_fp32 &space) {
  return {
      space.field_start,
      space.extent1,
      space.extent2,
      space.extent3,
      space.field_stride1,
      space.field_stride2,
      space.field_stride3,
      space.coefficient_start,
      space.coefficient_stride1,
      space.coefficient_stride2,
      space.coefficient_stride3,
      space.coefficient2_start,
      space.coefficient2_stride1,
      space.coefficient2_stride2,
      space.coefficient2_stride3};
}

bool shifted_index_is_valid(std::ptrdiff_t index, std::ptrdiff_t shift,
                            std::ptrdiff_t limit) {
  if (shift >= 0) return shift < limit - index;
  return shift >= -index;
}

void validate_neighbor_range(std::ptrdiff_t first, std::ptrdiff_t last,
                             std::ptrdiff_t field_stride,
                             std::ptrdiff_t neighbor_stride,
                             std::ptrdiff_t limit, const char *which) {
  if (neighbor_stride == std::numeric_limits<std::ptrdiff_t>::min())
    throw std::out_of_range(std::string(which) +
                            " neighbor stride is not representable");
  if (!shifted_index_is_valid(first, field_stride, limit) ||
      !shifted_index_is_valid(last, field_stride, limit) ||
      !shifted_index_is_valid(first, -neighbor_stride, limit) ||
      !shifted_index_is_valid(last, -neighbor_stride, limit))
    throw std::out_of_range(std::string(which) +
                            " neighbor index is outside the field array");
  const std::ptrdiff_t shifted_first = first + field_stride;
  const std::ptrdiff_t shifted_last = last + field_stride;
  if (!shifted_index_is_valid(shifted_first, -neighbor_stride, limit) ||
      !shifted_index_is_valid(shifted_last, -neighbor_stride, limit))
    throw std::out_of_range(std::string(which) +
                            " diagonal neighbor is outside the field array");
}

void validate_four_point_range(std::ptrdiff_t first, std::ptrdiff_t last,
                               std::ptrdiff_t offset1,
                               std::ptrdiff_t offset2,
                               std::ptrdiff_t limit, const char *which) {
  if (!shifted_index_is_valid(first, offset1, limit) ||
      !shifted_index_is_valid(last, offset1, limit) ||
      !shifted_index_is_valid(first, offset2, limit) ||
      !shifted_index_is_valid(last, offset2, limit))
    throw std::out_of_range(std::string(which) +
                            " neighbor index is outside the field array");
  const std::ptrdiff_t shifted_first = first + offset1;
  const std::ptrdiff_t shifted_last = last + offset1;
  if (!shifted_index_is_valid(shifted_first, offset2, limit) ||
      !shifted_index_is_valid(shifted_last, offset2, limit))
    throw std::out_of_range(std::string(which) +
                            " diagonal neighbor is outside the field array");
}

index_space_bounds validate_curl_index_space(
    const detail::index_space_fp32 &space, std::size_t array_count,
    const float *g1, const float *g2, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2,
    const detail::curl_material_fp32 &material) {
  const index_space_bounds bounds =
      validate_index_space(space, array_count);
  const bool pml_f = material.sigma || material.kappa || material.sigma_inverse;
  if (pml_f &&
      !(material.sigma && material.kappa && material.sigma_inverse &&
        material.sigma_count))
    throw std::invalid_argument("Meep CUDA PML-f arrays and size must be all present");
  const bool pml_u = material.field_u || material.sigma_u || material.kappa_u ||
                     material.sigma_u_inverse;
  if (pml_u &&
      !(material.field_u && material.sigma_u && material.kappa_u &&
        material.sigma_u_inverse && material.sigma_u_count))
    throw std::invalid_argument(
        "Meep CUDA PML-u auxiliary field, arrays, and size must be all present");
  const bool conductivity =
      material.conductivity || material.conductivity_inverse;
  if (conductivity &&
      !(material.conductivity && material.conductivity_inverse))
    throw std::invalid_argument(
        "Meep CUDA conductivity and inverse arrays must be both present");
  if (conductivity && pml_f && !material.field_conductivity)
    throw std::invalid_argument(
        "Meep CUDA simultaneous PML-f and conductivity requires an auxiliary field");
  if (material.field_conductivity && !(conductivity && pml_f))
    throw std::invalid_argument(
        "Meep CUDA conductivity auxiliary field requires PML-f and conductivity");

  const std::ptrdiff_t limit = static_cast<std::ptrdiff_t>(array_count);
  if (g1 &&
      (!shifted_index_is_valid(bounds.field_min, stride1, limit) ||
       !shifted_index_is_valid(bounds.field_max, stride1, limit)))
    throw std::out_of_range(
        "Meep CUDA curl first operand index is outside the field array");
  if (g2 &&
      (!shifted_index_is_valid(bounds.field_min, stride2, limit) ||
       !shifted_index_is_valid(bounds.field_max, stride2, limit)))
    throw std::out_of_range(
        "Meep CUDA curl second operand index is outside the field array");
  if (pml_f &&
      (bounds.coefficient_min < 0 ||
       static_cast<std::size_t>(bounds.coefficient_max) >=
           material.sigma_count))
    throw std::out_of_range(
        "Meep CUDA curl PML-f index is outside the sigma array");
  if (!pml_f && bounds.coefficient_min != -1)
    throw std::invalid_argument(
        "Meep CUDA curl has a PML-f index without PML-f arrays");
  if (pml_u &&
      (bounds.coefficient2_min < 0 ||
       static_cast<std::size_t>(bounds.coefficient2_max) >=
           material.sigma_u_count))
    throw std::out_of_range(
        "Meep CUDA curl PML-u index is outside the sigma array");
  if (!pml_u && bounds.coefficient2_min != -1)
    throw std::invalid_argument(
        "Meep CUDA curl has a PML-u index without PML-u arrays");
  return bounds;
}

index_space_bounds validate_beta_index_space(
    const detail::index_space_fp32 &space, std::size_t array_count,
    const detail::beta_material_fp32 &material) {
  const index_space_bounds bounds =
      validate_index_space(space, array_count);
  const bool pml_f = material.sigma_inverse != nullptr;
  const bool pml_u =
      material.field_u || material.sigma_u_inverse;
  if (pml_f && !material.sigma_count)
    throw std::invalid_argument(
        "Meep CUDA beta PML-f inverse array requires its size");
  if (pml_u &&
      !(material.field_u && material.sigma_u_inverse &&
        material.sigma_u_count))
    throw std::invalid_argument(
        "Meep CUDA beta PML-u auxiliary field, inverse array, and size "
        "must be all present");
  const bool conductivity = material.conductivity_inverse != nullptr;
  if (conductivity && pml_f && !material.field_conductivity)
    throw std::invalid_argument(
        "Meep CUDA beta conductivity with PML-f requires an auxiliary "
        "field");
  if (material.field_conductivity && !(conductivity && pml_f))
    throw std::invalid_argument(
        "Meep CUDA beta conductivity auxiliary field requires "
        "conductivity and PML-f");
  if (pml_f &&
      (bounds.coefficient_min < 0 ||
       static_cast<std::size_t>(bounds.coefficient_max) >=
           material.sigma_count))
    throw std::out_of_range(
        "Meep CUDA beta PML-f index is outside the inverse array");
  if (!pml_f && bounds.coefficient_min != -1)
    throw std::invalid_argument(
        "Meep CUDA beta has a PML-f index without an inverse array");
  if (pml_u &&
      (bounds.coefficient2_min < 0 ||
       static_cast<std::size_t>(bounds.coefficient2_max) >=
           material.sigma_u_count))
    throw std::out_of_range(
        "Meep CUDA beta PML-u index is outside the inverse array");
  if (!pml_u && bounds.coefficient2_min != -1)
    throw std::invalid_argument(
        "Meep CUDA beta has a PML-u index without an inverse array");
  return bounds;
}

index_space_bounds validate_bfast_index_space(
    const detail::index_space_fp32 &space, std::size_t array_count,
    const float *g1, const float *g2, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, float k1, float k2,
    const detail::bfast_material_fp32 &material) {
  const detail::beta_material_fp32 additive_material = {
      material.sigma_inverse,
      material.sigma_count,
      material.field_u,
      material.sigma_u_inverse,
      material.sigma_u_count,
      material.conductivity_inverse,
      material.field_conductivity};
  const index_space_bounds bounds =
      validate_beta_index_space(space, array_count, additive_material);
  const meep_cuda::detail::bfast_operands_fp32 operands =
      meep_cuda::detail::normalize_bfast_operands(
          g1, g2, stride1, stride2, k1, k2);
  if (!operands.g1)
    throw std::invalid_argument(
        "Meep CUDA BFAST requires at least one curl operand");
  const std::ptrdiff_t limit =
      static_cast<std::ptrdiff_t>(array_count);
  if (!shifted_index_is_valid(
          bounds.field_min, operands.stride1, limit) ||
      !shifted_index_is_valid(
          bounds.field_max, operands.stride1, limit))
    throw std::out_of_range(
        "Meep CUDA BFAST first operand index is outside the field array");
  if (operands.g2 &&
      (!shifted_index_is_valid(
           bounds.field_min, operands.stride2, limit) ||
       !shifted_index_is_valid(
           bounds.field_max, operands.stride2, limit)))
    throw std::out_of_range(
        "Meep CUDA BFAST second operand index is outside the field array");
  return bounds;
}

index_space_bounds validate_update_eh_index_space(
    const detail::index_space_fp32 &space, std::size_t array_count,
    const float *g1, const float *g2,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2,
    const detail::update_eh_material_fp32 &material) {
  const index_space_bounds bounds =
      validate_index_space(space, array_count);

  const meep_cuda::detail::update_eh_operands_fp32 operands =
      meep_cuda::detail::normalize_update_eh_operands(
          g1, g2, material.offdiagonal1, material.offdiagonal2, stride1,
          stride2);
  if (operands.u1 && !operands.g1)
    throw std::invalid_argument(
        "Meep CUDA first off-diagonal coefficient requires its input field");
  if (operands.u2 && !operands.g2)
    throw std::invalid_argument(
        "Meep CUDA second off-diagonal coefficient requires its input field");
  if (operands.u2 && !operands.u1)
    throw std::invalid_argument(
        "Meep CUDA second off-diagonal coefficient requires the first coefficient");
  if (material.chi3 && !material.chi2)
    throw std::invalid_argument("Meep CUDA chi3 requires a chi2 array");

  const bool pml =
      material.field_w || material.sigma || material.kappa ||
      material.sigma_count;
  if (pml && !(material.field_w && material.sigma && material.kappa &&
               material.sigma_count))
    throw std::invalid_argument(
        "Meep CUDA E/H PML field, arrays, and size must be all present");

  const std::ptrdiff_t limit = static_cast<std::ptrdiff_t>(array_count);
  const bool read_neighbors1 =
      operands.g1 && (operands.u1 || material.chi3);
  const bool read_neighbors2 =
      operands.g2 && (operands.u2 || material.chi3);
  if (read_neighbors1)
    validate_neighbor_range(
        bounds.field_min, bounds.field_max, field_stride, operands.stride1,
        limit, "first E/H input");
  if (read_neighbors2)
    validate_neighbor_range(
        bounds.field_min, bounds.field_max, field_stride, operands.stride2,
        limit, "second E/H input");
  if (pml &&
      (bounds.coefficient_min < 0 ||
       static_cast<std::size_t>(bounds.coefficient_max) >=
           material.sigma_count))
    throw std::out_of_range(
        "Meep CUDA E/H PML index is outside the sigma array");
  if (!pml && bounds.coefficient_min != -1)
    throw std::invalid_argument(
        "Meep CUDA E/H update has a PML index without PML arrays");
  if (bounds.coefficient2_min != -1)
    throw std::invalid_argument(
        "Meep CUDA E/H update has an unused second coefficient index");
  return bounds;
}

index_space_bounds validate_lorentzian_index_space(
    const detail::index_space_fp32 &space, std::size_t array_count,
    const float *field1, const float *field2,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2,
    const detail::lorentzian_material_fp32 &material) {
  const index_space_bounds bounds =
      validate_index_space(space, array_count);
  if (!material.sigma)
    throw std::invalid_argument(
        "Meep CUDA Lorentzian update requires diagonal sigma");

  const meep_cuda::detail::lorentzian_operands_fp32 operands =
      meep_cuda::detail::normalize_lorentzian_operands(
          field1, field2, material.offdiagonal1, material.offdiagonal2,
          stride1, stride2);
  if (operands.sigma1 && !operands.w1)
    throw std::invalid_argument(
        "Meep CUDA first Lorentzian off-diagonal sigma requires its field");
  if (operands.sigma2 && !operands.w2)
    throw std::invalid_argument(
        "Meep CUDA second Lorentzian off-diagonal sigma requires its field");
  if (operands.sigma2 && !operands.sigma1)
    throw std::invalid_argument(
        "Meep CUDA second Lorentzian sigma requires the first coefficient");

  const std::ptrdiff_t limit = static_cast<std::ptrdiff_t>(array_count);
  if (operands.sigma1)
    validate_neighbor_range(
        bounds.field_min, bounds.field_max, field_stride, operands.stride1,
        limit, "first Lorentzian input");
  if (operands.sigma2)
    validate_neighbor_range(
        bounds.field_min, bounds.field_max, field_stride, operands.stride2,
        limit, "second Lorentzian input");
  if (bounds.coefficient_min != -1 || bounds.coefficient2_min != -1)
    throw std::invalid_argument(
        "Meep CUDA Lorentzian update has unused coefficient indices");
  return bounds;
}

index_space_bounds validate_gyrotropic_index_space(
    const detail::index_space_fp32 &space, std::size_t array_count,
    const float *field1, const float *field2,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2,
    const detail::gyrotropic_material_fp32 &material) {
  const index_space_bounds bounds =
      validate_index_space(space, array_count);
  if (!material.sigma)
    throw std::invalid_argument(
        "Meep CUDA gyrotropic update requires sigma");
  if (material.model <
          meep_cuda::detail::gyrotropic_lorentzian_fp32 ||
      material.model >
          meep_cuda::detail::gyrotropic_saturated_fp32)
    throw std::invalid_argument(
        "Meep CUDA gyrotropic update has an invalid model");

  const std::ptrdiff_t limit =
      static_cast<std::ptrdiff_t>(array_count);
  if (field1)
    validate_neighbor_range(
        bounds.field_min, bounds.field_max, field_stride, stride1,
        limit, "first gyrotropic transverse field");
  if (field2)
    validate_neighbor_range(
        bounds.field_min, bounds.field_max, field_stride, stride2,
        limit, "second gyrotropic transverse field");
  if (bounds.coefficient_min != -1 || bounds.coefficient2_min != -1)
    throw std::invalid_argument(
        "Meep CUDA gyrotropic update has unused coefficient indices");
  return bounds;
}

int active_cuda_device_ordinal() {
  std::lock_guard<std::mutex> lock(state.mutex);
  ensure_configured_locked();
  if (state.active != backend_mode::cuda || state.selected_device < 0)
    throw std::runtime_error("Meep CUDA resident cache requested without an active CUDA device");
  return state.selected_device;
}

void clear_device_buffers(detail::resident_cache *cache) noexcept {
  std::uint64_t released =
      static_cast<std::uint64_t>(cache->mirrors.size()) +
      static_cast<std::uint64_t>(
          cache->recycled_mirror_allocations.size()) +
      static_cast<std::uint64_t>(
          cache->recycled_source_profile_allocations.size()) +
      static_cast<std::uint64_t>(cache->structured_curl_batches.size()) +
      (cache->finite_check.device_spans ? 1u : 0u) +
      (cache->finite_result ? 1u : 0u) +
      (cache->dft_norm_result ? 1u : 0u) +
      (cache->dft_batch_plans[0].device_pointer ? 1u : 0u) +
      (cache->dft_batch_plans[1].device_pointer ? 1u : 0u) +
      (cache->dft_materialization.device_metadata ? 1u : 0u) +
      (cache->dft_materialization.device_output ? 1u : 0u) +
      (cache->dft_output_staging.device_metadata ? 1u : 0u) +
      (cache->dft_output_staging.device_output ? 1u : 0u) +
      (cache->ldos_reduction_workspace ? 1u : 0u) +
      (cache->ldos_plan_buffers[0].device_pointer ? 1u : 0u) +
      (cache->ldos_plan_buffers[1].device_pointer ? 1u : 0u) +
      (cache->index_pointer ? 1u : 0u);
  for (auto &entry : cache->mirrors)
    meep_cuda::free_device(entry.second.device_pointer);
  cache->mirrors.clear();
  cache->mirror_intervals.clear();
  for (const auto &entry : cache->recycled_mirror_allocations)
    meep_cuda::free_device(entry.second);
  cache->recycled_mirror_allocations.clear();
  for (const auto &entry : cache->recycled_source_profile_allocations)
    meep_cuda::free_device(entry.second);
  cache->recycled_source_profile_allocations.clear();
  cache->validated_source_profiles.clear();
  for (auto &entry : cache->structured_curl_batches)
    meep_cuda::free_device(entry.second.device_spaces);
  cache->structured_curl_batches.clear();
  for (auto &entry : cache->multilevel_plans) {
    if (entry.second.device_population_channels) ++released;
    if (entry.second.device_polarization_channels) ++released;
    if (entry.second.device_transitions) ++released;
    meep_cuda::free_device(entry.second.device_population_channels);
    meep_cuda::free_device(entry.second.device_polarization_channels);
    meep_cuda::free_device(entry.second.device_transitions);
  }
  cache->multilevel_plans.clear();
  meep_cuda::free_device(cache->finite_check.device_spans);
  cache->finite_check.device_spans = nullptr;
  cache->finite_check.entries.clear();
  cache->finite_check.maximum_span_count = 0;
  cache->finite_check.allocation_generation = 0;
  meep_cuda::free_device(cache->finite_result);
  cache->finite_result = nullptr;
  cache->finite_result_generation = 0;
  cache->finite_result_pending = false;
  meep_cuda::free_device(cache->dft_norm_result);
  cache->dft_norm_result = nullptr;
  for (detail::resident_cache::dft_batch_plan &plan :
       cache->dft_batch_plans) {
    meep_cuda::free_device(plan.device_pointer);
    plan = detail::resident_cache::dft_batch_plan();
  }
  cache->dft_batch_next_replacement = 0;
  meep_cuda::free_device(
      cache->dft_materialization.device_metadata);
  meep_cuda::free_device(
      cache->dft_materialization.device_output);
  cache->dft_materialization =
      detail::resident_cache::dft_materialization_plan();
  meep_cuda::free_device(
      cache->dft_output_staging.device_metadata);
  meep_cuda::free_device(
      cache->dft_output_staging.device_output);
  cache->dft_output_staging =
      detail::resident_cache::dft_output_staging_plan();
  meep_cuda::free_device(cache->ldos_reduction_workspace);
  cache->ldos_reduction_workspace = nullptr;
  cache->ldos_reduction_partial_capacity = 0;
  for (detail::resident_cache::ldos_plan_buffer &slot :
       cache->ldos_plan_buffers) {
    meep_cuda::free_device(slot.device_pointer);
    slot.device_pointer = nullptr;
    slot.capacity_bytes = 0;
  }
  cache->ldos_active_plan_slot = -1;
  invalidate_ldos_plan_snapshot(cache);
  meep_cuda::free_device(cache->index_pointer);
  cache->index_pointer = nullptr;
  cache->index_capacity = 0;
  cache->allocation_generation =
      detail::next_resident_allocation_generation();
  live_device_buffer_count().fetch_sub(released);
}

void sync_resident_mirrors_to_host(detail::resident_cache *cache);

void select_cache_device(detail::resident_cache *cache) {
  const int ordinal = active_cuda_device_ordinal();
  if (cache->device_ordinal >= 0 && cache->device_ordinal != ordinal) {
    // A plan in another result cache may contain a nested pointer into this
    // cache.  Serialize migration with LDOS and invalidate every such host
    // snapshot before releasing device-specific allocations.  Lock order is
    // process-wide LDOS mutex followed by the resident registry mutex.
    std::lock_guard<std::mutex> ldos_lock(get_ldos_reduction_mutex());
    resident_registry &registry = get_resident_registry();
    std::lock_guard<std::mutex> registry_lock(registry.mutex);
    invalidate_all_ldos_plan_snapshots_locked(registry);
    clear_dft_reduction_plans_for_cache(cache);
    clear_eigenmode_overlap_plans_for_cache(cache);
    clear_near2far_plans_for_cache(cache);
    meep_cuda::select_device(cache->device_ordinal);
    // Device-dirty mirrors may be the only authoritative copy. Migrate them
    // transactionally through the host before releasing old-device storage.
    sync_resident_mirrors_to_host(cache);
    clear_device_buffers(cache);
  }
  meep_cuda::select_device(ordinal);
  cache->device_ordinal = ordinal;
}

void begin_resident_phase(detail::resident_cache *cache) {
  if (cache->phase_active)
    throw std::logic_error("nested Meep CUDA resident curl phases are not supported");
  select_cache_device(cache);
  if (cache->epoch == std::numeric_limits<std::uint64_t>::max()) {
    for (auto &entry : cache->mirrors)
      entry.second.uploaded_epoch = 0;
    cache->epoch = 1;
  }
  else {
    ++cache->epoch;
  }
  cache->phase_active = true;
}

void *ensure_resident_mirror(
    detail::resident_cache *cache, const float *host_pointer,
    std::size_t bytes,
    std::atomic<std::uint64_t> *operation_upload_bytes = nullptr,
    bool invalidates_address_plans = true,
    std::atomic<std::int64_t> *failure_countdown_for_testing = nullptr) {
  auto found = cache->mirrors.find(host_pointer);
  if (found != cache->mirrors.end() && found->second.bytes != bytes) {
    meep_cuda::free_device(found->second.device_pointer);
    cache->mirror_intervals.erase(
        reinterpret_cast<std::uintptr_t>(host_pointer));
    cache->mirrors.erase(found);
    live_device_buffer_count().fetch_sub(1);
    found = cache->mirrors.end();
  }

  if (found == cache->mirrors.end()) {
    detail::resident_cache::mirror created;
    using recycled_pool =
        std::vector<std::pair<std::size_t, void *> >;
    recycled_pool *selected_pool = nullptr;
    std::size_t recycled_index = 0;
    const auto select_exact_size =
        [&selected_pool, &recycled_index, bytes](recycled_pool &pool) {
          for (std::size_t index = 0; index < pool.size(); ++index)
            if (pool[index].first == bytes) {
              selected_pool = &pool;
              recycled_index = index;
              return true;
            }
          return false;
        };
    if (!invalidates_address_plans)
      select_exact_size(cache->recycled_source_profile_allocations);
    if (!selected_pool)
      select_exact_size(cache->recycled_mirror_allocations);
    const bool recycled = selected_pool != nullptr;
    created.device_pointer =
        recycled
            ? (*selected_pool)[recycled_index].second
            : meep_cuda::allocate_device_bytes(bytes);
    created.bytes = bytes;
    try {
      found = cache->mirrors.emplace(host_pointer, created).first;
      cache->mirror_intervals.emplace(
          reinterpret_cast<std::uintptr_t>(host_pointer), host_pointer);
    }
    catch (...) {
      cache->mirrors.erase(host_pointer);
      cache->mirror_intervals.erase(
          reinterpret_cast<std::uintptr_t>(host_pointer));
      if (!recycled)
        meep_cuda::free_device(created.device_pointer);
      throw;
    }
    if (recycled) {
      (*selected_pool)[recycled_index] = selected_pool->back();
      selected_pool->pop_back();
      device_buffer_reuses.fetch_add(1);
    }
    else {
      device_buffer_allocations.fetch_add(1);
      live_device_buffer_count().fetch_add(1);
    }
    if (invalidates_address_plans)
      cache->allocation_generation =
          detail::next_resident_allocation_generation();
  }
  else {
    device_buffer_reuses.fetch_add(1);
  }

  detail::resident_cache::mirror &mirror = found->second;
  // Resident mirrors remain valid across phases until an explicit host-side
  // mutation destroys or forgets them. Device-dirty data is authoritative;
  // clean uploaded mirrors are immutable inputs such as material tensors.
  if (mirror.device_dirty || mirror.uploaded_epoch != 0) {
    host_to_device_bytes_avoided.fetch_add(static_cast<std::uint64_t>(bytes));
  }
  else {
    if (failure_countdown_for_testing &&
        consume_countdown_failure_for_testing(
            *failure_countdown_for_testing))
      throw std::runtime_error(
          "injected CUDA DFT checkpoint H2D failure");
    meep_cuda::copy_to_device(mirror.device_pointer, host_pointer, bytes);
    mirror.uploaded_epoch = cache->epoch ? cache->epoch : 1;
    mirror.content_generation =
        detail::next_resident_content_generation();
    host_to_device_bytes.fetch_add(static_cast<std::uint64_t>(bytes));
    if (operation_upload_bytes)
      operation_upload_bytes->fetch_add(
          static_cast<std::uint64_t>(bytes));
  }
  return mirror.device_pointer;
}

void *ensure_index_buffer(detail::resident_cache *cache, std::size_t bytes) {
  if (bytes > cache->index_capacity) {
    std::size_t capacity = 256;
    while (capacity < bytes && capacity <= std::numeric_limits<std::size_t>::max() / 2)
      capacity *= 2;
    if (capacity < bytes) capacity = bytes;

    void *replacement = meep_cuda::allocate_device_bytes(capacity);
    const bool had_index_buffer = cache->index_pointer != nullptr;
    meep_cuda::free_device(cache->index_pointer);
    cache->index_pointer = replacement;
    cache->index_capacity = capacity;
    device_buffer_allocations.fetch_add(1);
    if (!had_index_buffer) live_device_buffer_count().fetch_add(1);
  }
  else {
    device_buffer_reuses.fetch_add(1);
  }
  return cache->index_pointer;
}

void mark_device_dirty(detail::resident_cache *cache, float *host_pointer,
                       std::size_t bytes) {
  const auto found = cache->mirrors.find(host_pointer);
  if (found == cache->mirrors.end())
    throw std::logic_error("Meep CUDA writable mirror was not made resident");
  if (found->second.device_dirty)
    device_to_host_bytes_avoided.fetch_add(static_cast<std::uint64_t>(bytes));
  found->second.device_dirty = true;
  found->second.content_generation =
      detail::next_resident_content_generation();
}

struct resolved_resident_range {
  const float *host_base;
  detail::resident_cache::mirror *mirror;
  std::uintptr_t byte_offset;

  resolved_resident_range()
      : host_base(nullptr), mirror(nullptr), byte_offset(0) {}
};

resolved_resident_range resolve_resident_range(
    detail::resident_cache *cache, const float *host_address,
    std::size_t bytes, bool allow_missing, const char *what) {
  if (!cache || !host_address || bytes == 0)
    throw std::invalid_argument(
        std::string(what) + " requires a non-null, nonempty range");
  if (cache->mirror_intervals.size() != cache->mirrors.size())
    throw std::logic_error(
        "Meep CUDA resident mirror interval count is inconsistent");

  const std::uintptr_t request_begin =
      reinterpret_cast<std::uintptr_t>(host_address);
  if (request_begin % alignof(float))
    throw std::invalid_argument(
        std::string(what) + " is not float-aligned");
  if (bytes >
      std::numeric_limits<std::uintptr_t>::max() - request_begin)
    throw std::overflow_error(
        std::string(what) + " address range overflow");
  const std::uintptr_t request_end = request_begin + bytes;

  resolved_resident_range resolved;
  for (const auto &interval : cache->mirror_intervals) {
    const auto found = cache->mirrors.find(interval.second);
    if (found == cache->mirrors.end() ||
        reinterpret_cast<std::uintptr_t>(interval.second) !=
            interval.first)
      throw std::logic_error(
          "Meep CUDA resident interval index is inconsistent");
    if (found->second.bytes >
        std::numeric_limits<std::uintptr_t>::max() - interval.first)
      throw std::overflow_error(
          "Meep CUDA resident mirror address overflow");
    const std::uintptr_t mirror_end =
        interval.first + found->second.bytes;
    if (request_begin >= mirror_end ||
        interval.first >= request_end)
      continue;
    if (request_begin < interval.first || request_end > mirror_end)
      throw std::invalid_argument(
          std::string(what) +
          " partially overlaps a resident mirror");
    if (resolved.mirror)
      throw std::invalid_argument(
          std::string(what) +
          " has ambiguous overlapping resident mirrors");
    resolved.host_base = interval.second;
    resolved.mirror = &found->second;
    resolved.byte_offset = request_begin - interval.first;
  }
  if (!resolved.mirror && !allow_missing)
    throw std::out_of_range(
        std::string(what) + " has no prepared resident mirror");
  return resolved;
}

void *resident_device_address(detail::resident_cache *cache,
                              const float *host_address, bool writable) {
  const resolved_resident_range resolved = resolve_resident_range(
      cache, host_address, sizeof(float), false,
      "Meep CUDA boundary pointer");
  if (writable) {
    if (resolved.mirror->device_dirty)
      device_to_host_bytes_avoided.fetch_add(
          static_cast<std::uint64_t>(resolved.mirror->bytes));
    resolved.mirror->device_dirty = true;
    resolved.mirror->content_generation =
        detail::next_resident_content_generation();
  }
  return static_cast<unsigned char *>(resolved.mirror->device_pointer) +
         resolved.byte_offset;
}

const float *resident_mirror_base_for_address(
    detail::resident_cache *cache, const float *host_address) {
  return resolve_resident_range(
             cache, host_address, sizeof(float), false,
             "Meep CUDA boundary pointer")
      .host_base;
}

void mark_boundary_plan_destinations_dirty(
    boundary_exchange_owner::operation_plan *plan) {
  if (!plan) return;
  for (const auto &entry : plan->writable_mirrors) {
    // The globally unique allocation generation is checked before this
    // function, so the mirror base cannot be an ABA-reused cache entry.
    (void)resident_device_address(entry.first, entry.second, true);
  }
}

void sync_resident_mirrors_to_host(detail::resident_cache *cache) {
  if (!cache) return;
  meep_cuda::synchronize();

  std::vector<std::pair<const float *, detail::resident_cache::mirror *> > dirty;
  dirty.reserve(cache->mirrors.size());
  for (auto &entry : cache->mirrors) {
    detail::resident_cache::mirror &mirror = entry.second;
    if (!mirror.device_dirty) continue;
    mirror.host_staging.resize(mirror.bytes / sizeof(float));
    dirty.emplace_back(entry.first, &mirror);
  }

  // Stage every device-to-host transfer before changing a Meep field. If a
  // CUDA copy fails, the session can discard all device-dirty state while the
  // authoritative host fields remain at their pre-phase values.
  for (const auto &entry : dirty) {
    meep_cuda::copy_to_host(entry.second->host_staging.data(),
                            entry.second->device_pointer, entry.second->bytes);
    device_to_host_bytes.fetch_add(static_cast<std::uint64_t>(entry.second->bytes));
  }

  for (const auto &entry : dirty) {
    detail::resident_cache::mirror &mirror = *entry.second;
    std::memcpy(const_cast<float *>(entry.first), mirror.host_staging.data(), mirror.bytes);
    mirror.device_dirty = false;
  }
}

void finish_resident_phase(detail::resident_cache *cache,
                           bool synchronize_to_host) {
  if (!cache || !cache->phase_active) return;
  if (synchronize_to_host) {
    sync_resident_mirrors_to_host(cache);
    // Once an owned phase returns control to host code, automatic mixed
    // CPU/CUDA execution must assume every clean host mirror can change.
    for (auto &entry : cache->mirrors)
      entry.second.uploaded_epoch = 0;
  }
  cache->phase_active = false;
}

void sync_and_invalidate_all_resident_caches() {
  std::lock_guard<std::mutex> ldos_lock(get_ldos_reduction_mutex());
  resident_registry &registry = get_resident_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  // Validate every cache before synchronizing any of them so backend/device
  // selection remains transactional.  A finish(false) finite scan owns a
  // device-resident sticky verdict; migrating or disabling that device would
  // otherwise free the only copy and silently turn a recorded NaN/Inf into a
  // later success.
  for (const auto &entry : registry.caches) {
    const detail::resident_cache *cache = entry.second;
    if (cache->phase_active)
      throw std::logic_error(
          "cannot change the Meep CUDA backend or device during an active "
          "resident phase");
    if (cache->finite_result_pending)
      throw std::logic_error(
          "cannot change the Meep CUDA backend or device while a deferred "
          "finite-check verdict is pending");
  }
  for (auto &entry : registry.caches) {
    detail::resident_cache *cache = entry.second;
    if (cache->device_ordinal >= 0) {
      meep_cuda::select_device(cache->device_ordinal);
      sync_resident_mirrors_to_host(cache);
    }
    for (auto &mirror : cache->mirrors)
      mirror.second.uploaded_epoch = 0;
  }
  invalidate_all_ldos_plan_snapshots_locked(registry);
  // MPI exchange buffers are device-specific scratch, never authoritative.
  // Drop them before a process changes its selected runtime ordinal.
  clear_all_boundary_exchange_buffers();
  clear_all_curl_phase_plans();
  clear_all_update_eh_phase_plans();
  clear_all_source_phase_plans();
  clear_all_dft_reduction_plans();
  clear_all_eigenmode_overlap_plans();
  clear_all_near2far_plans();
}

void discard_resident_phase(detail::resident_cache *cache) noexcept {
  if (!cache || !cache->phase_active) return;
  try {
    meep_cuda::synchronize();
  }
  catch (...) {}
  // Never clear device_dirty here: a retained mirror from an earlier time
  // step may be the only authoritative copy. Runtime failure is propagated,
  // and a later explicit sync can still recover successfully completed work.
  cache->phase_active = false;
}
#endif

} // namespace

bool backend_compiled() noexcept {
#if MEEP_HAVE_CUDA
  return true;
#else
  return false;
#endif
}

bool runtime_available(std::string *diagnostic) {
#if MEEP_HAVE_CUDA
  return meep_cuda::runtime_available(diagnostic);
#else
  if (diagnostic) *diagnostic = "Meep was built without CUDA support";
  return false;
#endif
}

std::vector<device_info> enumerate_devices() {
#if MEEP_HAVE_CUDA
  const std::vector<meep_cuda::device_info> cuda_devices = meep_cuda::enumerate_devices();
  std::vector<device_info> devices;
  devices.reserve(cuda_devices.size());
  for (const auto &device : cuda_devices) {
    devices.push_back({device.ordinal,
                       device.compute_major,
                       device.compute_minor,
                       device.multiprocessor_count,
                       device.max_threads_per_block,
                       device.global_memory_bytes,
                       device.memory_bandwidth_bytes_per_second,
                       meep_cuda::device_uuid(device.ordinal),
                       device.name,
                       meep_cuda::device_compatible(device)});
  }
  return devices;
#else
  return {};
#endif
}

std::string device_identifier(int ordinal) {
#if MEEP_HAVE_CUDA
  return meep_cuda::device_uuid(ordinal);
#else
  (void)ordinal;
  return {};
#endif
}

std::string compiled_architectures() {
#if MEEP_HAVE_CUDA
  return meep_cuda::compiled_architectures();
#else
  return "none";
#endif
}

void select_device(int ordinal) {
#if MEEP_HAVE_CUDA
  sync_and_invalidate_all_resident_caches();
  meep_cuda::select_device(ordinal);
  std::lock_guard<std::mutex> lock(state.mutex);
  state.selected_device = ordinal;
  state.explicit_device_selection = true;
  state.device_ready = true;
  ++state.generation;
  if (state.configured && state.requested != backend_mode::cpu) {
    state.active = backend_mode::cuda;
    std::ostringstream message;
    message << "CUDA device " << ordinal << " selected";
    state.diagnostic = message.str();
  }
#else
  (void)ordinal;
  throw std::runtime_error("Meep was built without CUDA support");
#endif
}

void set_backend(backend_mode mode) {
#if MEEP_HAVE_CUDA
  // Preserve device-authoritative simulation state before CPU execution or a
  // different selected device can observe the host arrays.
  sync_and_invalidate_all_resident_caches();
#endif
  if (mode == backend_mode::cpu)
    release_distributed_device_identifier();
  std::lock_guard<std::mutex> lock(state.mutex);
  configure_transaction_locked(mode);
}

backend_mode requested_backend() {
  std::lock_guard<std::mutex> lock(state.mutex);
  ensure_configured_locked();
  return state.requested;
}

backend_mode active_backend() {
  std::lock_guard<std::mutex> lock(state.mutex);
  ensure_configured_locked();
  return state.active;
}

int selected_device() {
  std::lock_guard<std::mutex> lock(state.mutex);
  ensure_configured_locked();
  return state.active == backend_mode::cuda ? state.selected_device : -1;
}

std::string selected_device_identifier() {
#if MEEP_HAVE_CUDA
  int ordinal = -1;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    ensure_configured_locked();
    if (state.active != backend_mode::cuda) return {};
    ordinal = state.selected_device;
  }
  return ordinal >= 0 ? device_identifier(ordinal) : std::string();
#else
  return {};
#endif
}

std::string backend_diagnostic() {
  std::lock_guard<std::mutex> lock(state.mutex);
  ensure_configured_locked();
  return state.diagnostic;
}

dispatch_statistics get_dispatch_statistics() noexcept {
  return {cpu_curl_calls.load(),
          cpu_curl_points.load(),
          cuda_curl_calls.load(),
          cuda_curl_points.load(),
          host_to_device_bytes.load(),
          device_to_host_bytes.load()};
}

resident_statistics get_resident_statistics() noexcept {
  return {host_to_device_bytes_avoided.load(),
          device_to_host_bytes_avoided.load(),
          device_buffer_allocations.load(),
          device_buffer_reuses.load()};
}

std::uint64_t get_live_resident_device_buffers() noexcept {
  return live_device_buffer_count().load();
}

field_update_statistics get_field_update_statistics() noexcept {
  return {cpu_update_eh_calls.load(), cpu_update_eh_points.load(),
          cuda_update_eh_calls.load(), cuda_update_eh_points.load()};
}

polarization_statistics get_polarization_statistics() noexcept {
  return {cpu_polarization_calls.load(), cpu_polarization_points.load(),
          cuda_polarization_calls.load(), cuda_polarization_points.load()};
}

source_statistics get_source_statistics() noexcept {
  return {cpu_source_calls.load(), cpu_source_points.load(),
          cuda_source_calls.load(), cuda_source_points.load()};
}

boundary_statistics get_boundary_statistics() noexcept {
  return {cpu_boundary_calls.load(), cpu_boundary_points.load(),
          cuda_boundary_calls.load(), cuda_boundary_points.load()};
}

dft_statistics get_dft_statistics() noexcept {
  return {cpu_dft_calls.load(), cpu_dft_points.load(),
          cuda_dft_calls.load(), cuda_dft_points.load()};
}

dft_batch_statistics get_dft_batch_statistics() noexcept {
  return {dft_batch_calls.load(std::memory_order_relaxed),
          dft_batch_submitted_updates.load(std::memory_order_relaxed),
          dft_phase_preparation_launches.load(std::memory_order_relaxed),
          dft_phase_reuses.load(std::memory_order_relaxed),
          dft_update_kernel_launches.load(std::memory_order_relaxed),
          dft_maximum_batch_size.load(std::memory_order_relaxed),
          dft_multi_monitor_automatic_checks.load(
              std::memory_order_relaxed),
          dft_multi_monitor_automatic_selected.load(
              std::memory_order_relaxed),
          dft_multi_monitor_automatic_rejected.load(
              std::memory_order_relaxed),
          dft_multi_monitor_forced_batches.load(
              std::memory_order_relaxed),
          dft_multi_monitor_batched_updates.load(
              std::memory_order_relaxed),
          dft_multi_monitor_unbatched_updates.load(
              std::memory_order_relaxed),
          dft_multi_monitor_plan_uploads.load(std::memory_order_relaxed),
          dft_multi_monitor_plan_reuses.load(std::memory_order_relaxed),
          dft_multi_monitor_metadata_h2d_bytes.load(
              std::memory_order_relaxed)};
}

dft_reduction_statistics get_dft_reduction_statistics() noexcept {
  return {
      cpu_dft_reduction_calls.load(std::memory_order_relaxed),
      cpu_dft_reduction_pairs.load(std::memory_order_relaxed),
      cpu_dft_reduction_terms.load(std::memory_order_relaxed),
      cuda_dft_reduction_calls.load(std::memory_order_relaxed),
      cuda_dft_reduction_pairs.load(std::memory_order_relaxed),
      cuda_dft_reduction_terms.load(std::memory_order_relaxed),
      cuda_dft_reduction_descriptor_uploads.load(
          std::memory_order_relaxed),
      cuda_dft_reduction_plan_reuses.load(std::memory_order_relaxed),
      cuda_dft_reduction_kernel_launches.load(
          std::memory_order_relaxed),
      cuda_dft_reduction_result_d2h_bytes.load(
          std::memory_order_relaxed),
      dft_reduction_full_dft_d2h_bytes_avoided.load(
          std::memory_order_relaxed),
      dft_reduction_mpi_allreduce_calls.load(
          std::memory_order_relaxed),
      dft_reduction_mpi_allreduce_bytes.load(
          std::memory_order_relaxed)};
}

dft_materialization_statistics
get_dft_materialization_statistics() noexcept {
  return {
      cpu_dft_array_materialization_calls.load(std::memory_order_relaxed),
      cpu_dft_array_materialization_points.load(std::memory_order_relaxed),
      cuda_dft_array_materialization_calls.load(std::memory_order_relaxed),
      cuda_dft_array_materialization_points.load(std::memory_order_relaxed),
      host_synthetic_material_array_calls.load(std::memory_order_relaxed),
      host_synthetic_material_array_points.load(std::memory_order_relaxed),
      cpu_dft_output_calls.load(std::memory_order_relaxed),
      cpu_dft_output_points.load(std::memory_order_relaxed),
      cuda_dft_output_calls.load(std::memory_order_relaxed),
      cuda_dft_output_points.load(std::memory_order_relaxed),
      cuda_dft_output_staging_calls.load(std::memory_order_relaxed),
      cuda_dft_output_staging_points.load(std::memory_order_relaxed),
      cuda_dft_output_staging_frequencies.load(std::memory_order_relaxed),
      cuda_dft_output_staging_descriptor_uploads.load(
          std::memory_order_relaxed),
      cuda_dft_output_staging_plan_reuses.load(std::memory_order_relaxed),
      cuda_dft_output_staging_kernel_launches.load(
          std::memory_order_relaxed),
      cuda_dft_output_staging_result_d2h_bytes.load(
          std::memory_order_relaxed),
      cuda_dft_output_staging_full_dft_d2h_bytes_avoided.load(
          std::memory_order_relaxed),
      cuda_dft_output_staging_workspace_ceiling_bytes.load(
          std::memory_order_relaxed),
      cuda_dft_materialization_kernel_launches.load(
          std::memory_order_relaxed),
      cuda_dft_materialization_result_d2h_bytes.load(
          std::memory_order_relaxed),
      dft_materialization_full_dft_d2h_bytes_avoided.load(
          std::memory_order_relaxed),
      dft_array_mpi_allreduce_calls.load(std::memory_order_relaxed),
      dft_array_mpi_allreduce_bytes.load(std::memory_order_relaxed)};
}

dft_checkpoint_statistics get_dft_checkpoint_statistics() noexcept {
  return {
      cpu_dft_checkpoint_save_calls.load(std::memory_order_relaxed),
      cpu_dft_checkpoint_save_values.load(std::memory_order_relaxed),
      cuda_dft_checkpoint_save_calls.load(std::memory_order_relaxed),
      cuda_dft_checkpoint_save_values.load(std::memory_order_relaxed),
      cuda_dft_checkpoint_save_d2h_bytes.load(std::memory_order_relaxed),
      cuda_dft_checkpoint_save_full_cache_d2h_avoided.load(
          std::memory_order_relaxed),
      cpu_dft_checkpoint_load_calls.load(std::memory_order_relaxed),
      cpu_dft_checkpoint_load_values.load(std::memory_order_relaxed),
      cuda_dft_checkpoint_load_calls.load(std::memory_order_relaxed),
      cuda_dft_checkpoint_load_values.load(std::memory_order_relaxed),
      cuda_dft_checkpoint_load_h2d_bytes.load(std::memory_order_relaxed)};
}

dft_scale_statistics get_dft_scale_statistics() noexcept {
  return {
      cpu_dft_scale_calls.load(std::memory_order_relaxed),
      cpu_dft_scale_values.load(std::memory_order_relaxed),
      cuda_dft_scale_calls.load(std::memory_order_relaxed),
      cuda_dft_scale_values.load(std::memory_order_relaxed),
      cuda_dft_scale_kernel_launches.load(std::memory_order_relaxed),
      cuda_dft_scale_h2d_bytes.load(std::memory_order_relaxed)};
}

eigenmode_overlap_statistics get_eigenmode_overlap_statistics() noexcept {
  return {
      cpu_dft_overlap_calls.load(std::memory_order_relaxed),
      cpu_dft_overlap_terms.load(std::memory_order_relaxed),
      cuda_dft_overlap_calls.load(std::memory_order_relaxed),
      cuda_dft_overlap_terms.load(std::memory_order_relaxed),
      cuda_eigenmode_mode_flux_calls.load(std::memory_order_relaxed),
      cuda_eigenmode_mode_mode_calls.load(std::memory_order_relaxed),
      cuda_eigenmode_submitted_pairs.load(std::memory_order_relaxed),
      cuda_eigenmode_descriptor_uploads.load(std::memory_order_relaxed),
      cuda_eigenmode_plan_reuses.load(std::memory_order_relaxed),
      cuda_eigenmode_kernel_launches.load(std::memory_order_relaxed),
      cuda_eigenmode_result_d2h_bytes.load(std::memory_order_relaxed),
      eigenmode_full_dft_d2h_bytes_avoided.load(std::memory_order_relaxed),
      host_mode_profile_sampling_calls.load(std::memory_order_relaxed),
      host_mode_profile_sampling_points.load(std::memory_order_relaxed),
      eigenmode_zero_rank_channels_skipped.load(
          std::memory_order_relaxed),
      host_mode_profile_h2d_bytes.load(std::memory_order_relaxed),
      eigenmode_mpi_allreduce_calls.load(std::memory_order_relaxed),
      eigenmode_mpi_allreduce_bytes.load(std::memory_order_relaxed)};
}

ldos_statistics get_ldos_statistics() noexcept {
  return {
      cpu_ldos_calls.load(std::memory_order_relaxed),
      cpu_ldos_source_points.load(std::memory_order_relaxed),
      ldos_batch_calls.load(std::memory_order_relaxed),
      ldos_submitted_profiles.load(std::memory_order_relaxed),
      ldos_source_points.load(std::memory_order_relaxed),
      ldos_descriptor_uploads.load(std::memory_order_relaxed),
      ldos_kernel_launches.load(std::memory_order_relaxed),
      ldos_result_device_to_host_bytes.load(std::memory_order_relaxed),
      ldos_full_field_device_to_host_bytes_avoided.load(
          std::memory_order_relaxed)};
}

near2far_statistics get_near2far_statistics() noexcept {
  return {
      cpu_near2far_calls.load(std::memory_order_relaxed),
      cpu_near2far_terms.load(std::memory_order_relaxed),
      cuda_near2far_calls.load(std::memory_order_relaxed),
      cuda_near2far_terms.load(std::memory_order_relaxed),
      cuda_near2far_submitted_chunks.load(std::memory_order_relaxed),
      cuda_near2far_source_points.load(std::memory_order_relaxed),
      cuda_near2far_output_points.load(std::memory_order_relaxed),
      cuda_near2far_frequencies.load(std::memory_order_relaxed),
      cuda_near2far_periodic_copies.load(std::memory_order_relaxed),
      cuda_near2far_fast_precision_calls.load(std::memory_order_relaxed),
      cuda_near2far_mixed_precision_calls.load(std::memory_order_relaxed),
      cuda_near2far_cancellation_retries.load(std::memory_order_relaxed),
      cuda_near2far_target_tiles.load(std::memory_order_relaxed),
      cuda_near2far_frequency_tiles.load(std::memory_order_relaxed),
      cuda_near2far_operation_tiles.load(std::memory_order_relaxed),
      cuda_near2far_maximum_workspace_bytes.load(
          std::memory_order_relaxed),
      cuda_near2far_descriptor_uploads.load(std::memory_order_relaxed),
      cuda_near2far_kernel_launches.load(std::memory_order_relaxed),
      cuda_near2far_result_device_to_host_bytes.load(
          std::memory_order_relaxed),
      cuda_near2far_condition_device_to_host_bytes.load(
          std::memory_order_relaxed),
      cuda_near2far_dft_device_to_host_bytes_avoided.load(
          std::memory_order_relaxed),
      near2far_mpi_allreduce_calls.load(std::memory_order_relaxed),
      near2far_mpi_allreduce_bytes.load(std::memory_order_relaxed),
      cpu_near2far_adjoint_calls.load(std::memory_order_relaxed),
      cpu_near2far_adjoint_terms.load(std::memory_order_relaxed),
      cuda_near2far_adjoint_calls.load(std::memory_order_relaxed),
      cuda_near2far_adjoint_terms.load(std::memory_order_relaxed),
      cuda_near2far_adjoint_submitted_chunks.load(
          std::memory_order_relaxed),
      cuda_near2far_adjoint_source_points.load(std::memory_order_relaxed),
      cuda_near2far_adjoint_far_points.load(std::memory_order_relaxed),
      cuda_near2far_adjoint_frequencies.load(std::memory_order_relaxed),
      cuda_near2far_adjoint_periodic_copies.load(
          std::memory_order_relaxed),
      cuda_near2far_adjoint_fast_precision_calls.load(
          std::memory_order_relaxed),
      cuda_near2far_adjoint_mixed_precision_calls.load(
          std::memory_order_relaxed),
      cuda_near2far_adjoint_cancellation_retries.load(
          std::memory_order_relaxed),
      cuda_near2far_adjoint_maximum_workspace_bytes.load(
          std::memory_order_relaxed),
      cuda_near2far_adjoint_descriptor_uploads.load(
          std::memory_order_relaxed),
      cuda_near2far_adjoint_kernel_launches.load(
          std::memory_order_relaxed),
      cuda_near2far_adjoint_host_to_device_bytes.load(
          std::memory_order_relaxed),
      cuda_near2far_adjoint_result_device_to_host_bytes.load(
          std::memory_order_relaxed),
      cuda_near2far_adjoint_condition_device_to_host_bytes.load(
          std::memory_order_relaxed)};
}

multi_gpu_statistics get_multi_gpu_statistics() noexcept {
  return {mpi_boundary_messages.load(), mpi_boundary_scalars.load(),
          cuda_aware_mpi_bytes.load(), pinned_mpi_bytes.load(),
          pinned_mpi_device_to_host_bytes.load(),
          pinned_mpi_host_to_device_bytes.load()};
}

mpi_completion_statistics get_mpi_completion_statistics() noexcept {
  return {mpi_waitsome_executions.load(), mpi_waitall_executions.load()};
}

boundary_eh_overlap_statistics
get_boundary_eh_overlap_statistics() noexcept {
  return {
      boundary_eh_overlap_checks.load(std::memory_order_relaxed),
      boundary_eh_overlap_eligible.load(std::memory_order_relaxed),
      boundary_eh_overlap_launched_h.load(std::memory_order_relaxed),
      boundary_eh_overlap_launched_e.load(std::memory_order_relaxed),
      boundary_eh_overlap_skipped_disabled.load(std::memory_order_relaxed),
      boundary_eh_overlap_skipped_unsupported_schedule.load(
          std::memory_order_relaxed),
      boundary_eh_overlap_skipped_no_remote.load(std::memory_order_relaxed),
      boundary_eh_overlap_skipped_cold_topology.load(
          std::memory_order_relaxed),
      boundary_eh_overlap_rejected.load(std::memory_order_relaxed)};
}

halo_curl_overlap_statistics get_halo_curl_overlap_statistics() noexcept {
  return {
      halo_curl_overlap_checks.load(std::memory_order_relaxed),
      halo_curl_overlap_eligible.load(std::memory_order_relaxed),
      halo_curl_overlap_launches.load(std::memory_order_relaxed),
      halo_curl_overlap_skipped_disabled.load(std::memory_order_relaxed),
      halo_curl_overlap_skipped_unsupported_schedule.load(
          std::memory_order_relaxed),
      halo_curl_overlap_skipped_no_remote.load(std::memory_order_relaxed),
      halo_curl_overlap_skipped_cold_topology.load(
          std::memory_order_relaxed),
      halo_curl_overlap_rejected_feature.load(std::memory_order_relaxed),
      halo_curl_overlap_rejected_small.load(std::memory_order_relaxed),
      halo_curl_overlap_full_points.load(std::memory_order_relaxed),
      halo_curl_overlap_interior_points.load(std::memory_order_relaxed),
      halo_curl_overlap_shell_points.load(std::memory_order_relaxed)};
}

tile_coalescing_statistics get_tile_coalescing_statistics() noexcept {
  return {
      tile_coalesced_curl_chunk_phases.load(std::memory_order_relaxed),
      tile_coalesced_curl_input_tiles.load(std::memory_order_relaxed),
      tile_coalesced_update_eh_chunk_phases.load(
          std::memory_order_relaxed),
      tile_coalesced_update_eh_input_tiles.load(
          std::memory_order_relaxed)};
}

phase_batch_policy_statistics
get_phase_batch_policy_statistics() noexcept {
  return {
      phase_curl_automatic_checks.load(std::memory_order_relaxed),
      phase_curl_automatic_selected.load(std::memory_order_relaxed),
      phase_curl_automatic_rejected.load(std::memory_order_relaxed),
      phase_curl_forced_batches.load(std::memory_order_relaxed),
      phase_curl_batched_operations.load(std::memory_order_relaxed),
      phase_curl_unbatched_operations.load(std::memory_order_relaxed),
      phase_update_eh_automatic_checks.load(std::memory_order_relaxed),
      phase_update_eh_automatic_selected.load(std::memory_order_relaxed),
      phase_update_eh_automatic_rejected.load(std::memory_order_relaxed),
      phase_update_eh_forced_batches.load(std::memory_order_relaxed),
      phase_update_eh_batched_operations.load(std::memory_order_relaxed),
      phase_update_eh_unbatched_operations.load(
          std::memory_order_relaxed)};
}

curl_phase_replay_statistics
get_curl_phase_replay_statistics() noexcept {
  return {
      phase_curl_replay_checks.load(std::memory_order_relaxed),
      phase_curl_replay_hits.load(std::memory_order_relaxed),
      phase_curl_replay_unready.load(std::memory_order_relaxed),
      phase_curl_replay_generation_misses.load(std::memory_order_relaxed),
      phase_curl_replay_mirror_misses.load(std::memory_order_relaxed)};
}

boundary_descriptor_replay_statistics
get_boundary_descriptor_replay_statistics() noexcept {
  return {
      boundary_gather_fast_replays.load(std::memory_order_relaxed),
      boundary_gather_full_validations.load(std::memory_order_relaxed),
      boundary_scatter_fast_replays.load(std::memory_order_relaxed),
      boundary_scatter_full_validations.load(std::memory_order_relaxed)};
}

void reset_boundary_descriptor_replay_statistics() noexcept {
  boundary_gather_fast_replays.store(0, std::memory_order_relaxed);
  boundary_gather_full_validations.store(0, std::memory_order_relaxed);
  boundary_scatter_fast_replays.store(0, std::memory_order_relaxed);
  boundary_scatter_full_validations.store(0, std::memory_order_relaxed);
}

runtime_touch_statistics get_runtime_touch_statistics() noexcept {
#if MEEP_HAVE_CUDA
  const meep_cuda::runtime_touch_statistics statistics =
      meep_cuda::get_runtime_touch_statistics();
  return {statistics.availability_probes,
          statistics.device_enumerations,
          statistics.device_selections};
#else
  return {0, 0, 0};
#endif
}

void reset_dispatch_statistics() noexcept {
#if MEEP_HAVE_CUDA
  meep_cuda::reset_runtime_touch_statistics();
#endif
  cpu_curl_calls.store(0);
  cpu_curl_points.store(0);
  cuda_curl_calls.store(0);
  cuda_curl_points.store(0);
  cpu_update_eh_calls.store(0);
  cpu_update_eh_points.store(0);
  cuda_update_eh_calls.store(0);
  cuda_update_eh_points.store(0);
  cpu_polarization_calls.store(0);
  cpu_polarization_points.store(0);
  cuda_polarization_calls.store(0);
  cuda_polarization_points.store(0);
  cpu_source_calls.store(0);
  cpu_source_points.store(0);
  cuda_source_calls.store(0);
  cuda_source_points.store(0);
  cpu_boundary_calls.store(0);
  cpu_boundary_points.store(0);
  cuda_boundary_calls.store(0);
  cuda_boundary_points.store(0);
  reset_boundary_descriptor_replay_statistics();
  cpu_dft_calls.store(0);
  cpu_dft_points.store(0);
  cuda_dft_calls.store(0);
  cuda_dft_points.store(0);
  cpu_ldos_calls.store(0, std::memory_order_relaxed);
  cpu_ldos_source_points.store(0, std::memory_order_relaxed);
  dft_batch_calls.store(0, std::memory_order_relaxed);
  dft_batch_submitted_updates.store(0, std::memory_order_relaxed);
  dft_phase_preparation_launches.store(0, std::memory_order_relaxed);
  dft_phase_reuses.store(0, std::memory_order_relaxed);
  dft_update_kernel_launches.store(0, std::memory_order_relaxed);
  dft_maximum_batch_size.store(0, std::memory_order_relaxed);
  dft_multi_monitor_automatic_checks.store(0, std::memory_order_relaxed);
  dft_multi_monitor_automatic_selected.store(0, std::memory_order_relaxed);
  dft_multi_monitor_automatic_rejected.store(0, std::memory_order_relaxed);
  dft_multi_monitor_forced_batches.store(0, std::memory_order_relaxed);
  dft_multi_monitor_batched_updates.store(0, std::memory_order_relaxed);
  dft_multi_monitor_unbatched_updates.store(0, std::memory_order_relaxed);
  dft_multi_monitor_plan_uploads.store(0, std::memory_order_relaxed);
  dft_multi_monitor_plan_reuses.store(0, std::memory_order_relaxed);
  dft_multi_monitor_metadata_h2d_bytes.store(
      0, std::memory_order_relaxed);
  cpu_dft_reduction_calls.store(0, std::memory_order_relaxed);
  cpu_dft_reduction_pairs.store(0, std::memory_order_relaxed);
  cpu_dft_reduction_terms.store(0, std::memory_order_relaxed);
  cuda_dft_reduction_calls.store(0, std::memory_order_relaxed);
  cuda_dft_reduction_pairs.store(0, std::memory_order_relaxed);
  cuda_dft_reduction_terms.store(0, std::memory_order_relaxed);
  cuda_dft_reduction_descriptor_uploads.store(
      0, std::memory_order_relaxed);
  cuda_dft_reduction_plan_reuses.store(0, std::memory_order_relaxed);
  cuda_dft_reduction_kernel_launches.store(
      0, std::memory_order_relaxed);
  cuda_dft_reduction_result_d2h_bytes.store(
      0, std::memory_order_relaxed);
  dft_reduction_full_dft_d2h_bytes_avoided.store(
      0, std::memory_order_relaxed);
  dft_reduction_mpi_allreduce_calls.store(
      0, std::memory_order_relaxed);
  dft_reduction_mpi_allreduce_bytes.store(
      0, std::memory_order_relaxed);
  cpu_dft_array_materialization_calls.store(0, std::memory_order_relaxed);
  cpu_dft_array_materialization_points.store(0, std::memory_order_relaxed);
  cuda_dft_array_materialization_calls.store(0, std::memory_order_relaxed);
  cuda_dft_array_materialization_points.store(0, std::memory_order_relaxed);
  host_synthetic_material_array_calls.store(0, std::memory_order_relaxed);
  host_synthetic_material_array_points.store(0, std::memory_order_relaxed);
  cpu_dft_output_calls.store(0, std::memory_order_relaxed);
  cpu_dft_output_points.store(0, std::memory_order_relaxed);
  cuda_dft_output_calls.store(0, std::memory_order_relaxed);
  cuda_dft_output_points.store(0, std::memory_order_relaxed);
  cuda_dft_output_staging_calls.store(0, std::memory_order_relaxed);
  cuda_dft_output_staging_points.store(0, std::memory_order_relaxed);
  cuda_dft_output_staging_frequencies.store(0, std::memory_order_relaxed);
  cuda_dft_output_staging_descriptor_uploads.store(
      0, std::memory_order_relaxed);
  cuda_dft_output_staging_plan_reuses.store(0, std::memory_order_relaxed);
  cuda_dft_output_staging_kernel_launches.store(
      0, std::memory_order_relaxed);
  cuda_dft_output_staging_result_d2h_bytes.store(
      0, std::memory_order_relaxed);
  cuda_dft_output_staging_full_dft_d2h_bytes_avoided.store(
      0, std::memory_order_relaxed);
  cuda_dft_output_staging_workspace_ceiling_bytes.store(
      0, std::memory_order_relaxed);
  cpu_dft_overlap_calls.store(0, std::memory_order_relaxed);
  cpu_dft_overlap_terms.store(0, std::memory_order_relaxed);
  cuda_dft_overlap_calls.store(0, std::memory_order_relaxed);
  cuda_dft_overlap_terms.store(0, std::memory_order_relaxed);
  cuda_eigenmode_mode_flux_calls.store(0, std::memory_order_relaxed);
  cuda_eigenmode_mode_mode_calls.store(0, std::memory_order_relaxed);
  cuda_eigenmode_submitted_pairs.store(0, std::memory_order_relaxed);
  cuda_eigenmode_descriptor_uploads.store(0, std::memory_order_relaxed);
  cuda_eigenmode_plan_reuses.store(0, std::memory_order_relaxed);
  cuda_eigenmode_kernel_launches.store(0, std::memory_order_relaxed);
  cuda_eigenmode_result_d2h_bytes.store(0, std::memory_order_relaxed);
  eigenmode_full_dft_d2h_bytes_avoided.store(0, std::memory_order_relaxed);
  host_mode_profile_sampling_calls.store(0, std::memory_order_relaxed);
  host_mode_profile_sampling_points.store(0, std::memory_order_relaxed);
  eigenmode_zero_rank_channels_skipped.store(
      0, std::memory_order_relaxed);
  host_mode_profile_h2d_bytes.store(0, std::memory_order_relaxed);
  eigenmode_mpi_allreduce_calls.store(0, std::memory_order_relaxed);
  eigenmode_mpi_allreduce_bytes.store(0, std::memory_order_relaxed);
  cuda_dft_materialization_kernel_launches.store(
      0, std::memory_order_relaxed);
  cuda_dft_materialization_result_d2h_bytes.store(
      0, std::memory_order_relaxed);
  dft_materialization_full_dft_d2h_bytes_avoided.store(
      0, std::memory_order_relaxed);
  dft_array_mpi_allreduce_calls.store(0, std::memory_order_relaxed);
  dft_array_mpi_allreduce_bytes.store(0, std::memory_order_relaxed);
  cpu_dft_checkpoint_save_calls.store(0, std::memory_order_relaxed);
  cpu_dft_checkpoint_save_values.store(0, std::memory_order_relaxed);
  cuda_dft_checkpoint_save_calls.store(0, std::memory_order_relaxed);
  cuda_dft_checkpoint_save_values.store(0, std::memory_order_relaxed);
  cuda_dft_checkpoint_save_d2h_bytes.store(0, std::memory_order_relaxed);
  cuda_dft_checkpoint_save_full_cache_d2h_avoided.store(
      0, std::memory_order_relaxed);
  cpu_dft_checkpoint_load_calls.store(0, std::memory_order_relaxed);
  cpu_dft_checkpoint_load_values.store(0, std::memory_order_relaxed);
  cuda_dft_checkpoint_load_calls.store(0, std::memory_order_relaxed);
  cuda_dft_checkpoint_load_values.store(0, std::memory_order_relaxed);
  cuda_dft_checkpoint_load_h2d_bytes.store(0, std::memory_order_relaxed);
  cpu_dft_scale_calls.store(0, std::memory_order_relaxed);
  cpu_dft_scale_values.store(0, std::memory_order_relaxed);
  cuda_dft_scale_calls.store(0, std::memory_order_relaxed);
  cuda_dft_scale_values.store(0, std::memory_order_relaxed);
  cuda_dft_scale_kernel_launches.store(0, std::memory_order_relaxed);
  cuda_dft_scale_h2d_bytes.store(0, std::memory_order_relaxed);
  dft_checkpoint_d2h_failure_after_for_testing.store(
      -1, std::memory_order_relaxed);
  dft_checkpoint_h2d_failure_after_for_testing.store(
      -1, std::memory_order_relaxed);
  dft_output_staging_d2h_failure_after_for_testing.store(
      -1, std::memory_order_relaxed);
  ldos_batch_calls.store(0, std::memory_order_relaxed);
  ldos_submitted_profiles.store(0, std::memory_order_relaxed);
  ldos_source_points.store(0, std::memory_order_relaxed);
  ldos_descriptor_uploads.store(0, std::memory_order_relaxed);
  ldos_kernel_launches.store(0, std::memory_order_relaxed);
  ldos_result_device_to_host_bytes.store(0, std::memory_order_relaxed);
  ldos_full_field_device_to_host_bytes_avoided.store(
      0, std::memory_order_relaxed);
  cpu_near2far_calls.store(0, std::memory_order_relaxed);
  cpu_near2far_terms.store(0, std::memory_order_relaxed);
  cuda_near2far_calls.store(0, std::memory_order_relaxed);
  cuda_near2far_terms.store(0, std::memory_order_relaxed);
  cuda_near2far_submitted_chunks.store(0, std::memory_order_relaxed);
  cuda_near2far_source_points.store(0, std::memory_order_relaxed);
  cuda_near2far_output_points.store(0, std::memory_order_relaxed);
  cuda_near2far_frequencies.store(0, std::memory_order_relaxed);
  cuda_near2far_periodic_copies.store(0, std::memory_order_relaxed);
  cuda_near2far_fast_precision_calls.store(0, std::memory_order_relaxed);
  cuda_near2far_mixed_precision_calls.store(0, std::memory_order_relaxed);
  cuda_near2far_cancellation_retries.store(0, std::memory_order_relaxed);
  cuda_near2far_target_tiles.store(0, std::memory_order_relaxed);
  cuda_near2far_frequency_tiles.store(0, std::memory_order_relaxed);
  cuda_near2far_operation_tiles.store(0, std::memory_order_relaxed);
  cuda_near2far_maximum_workspace_bytes.store(
      0, std::memory_order_relaxed);
  cuda_near2far_descriptor_uploads.store(0, std::memory_order_relaxed);
  cuda_near2far_kernel_launches.store(0, std::memory_order_relaxed);
  cuda_near2far_result_device_to_host_bytes.store(
      0, std::memory_order_relaxed);
  cuda_near2far_condition_device_to_host_bytes.store(
      0, std::memory_order_relaxed);
  cuda_near2far_dft_device_to_host_bytes_avoided.store(
      0, std::memory_order_relaxed);
  near2far_mpi_allreduce_calls.store(0, std::memory_order_relaxed);
  near2far_mpi_allreduce_bytes.store(0, std::memory_order_relaxed);
  cpu_near2far_adjoint_calls.store(0, std::memory_order_relaxed);
  cpu_near2far_adjoint_terms.store(0, std::memory_order_relaxed);
  cuda_near2far_adjoint_calls.store(0, std::memory_order_relaxed);
  cuda_near2far_adjoint_terms.store(0, std::memory_order_relaxed);
  cuda_near2far_adjoint_submitted_chunks.store(
      0, std::memory_order_relaxed);
  cuda_near2far_adjoint_source_points.store(0, std::memory_order_relaxed);
  cuda_near2far_adjoint_far_points.store(0, std::memory_order_relaxed);
  cuda_near2far_adjoint_frequencies.store(0, std::memory_order_relaxed);
  cuda_near2far_adjoint_periodic_copies.store(0,
                                              std::memory_order_relaxed);
  cuda_near2far_adjoint_fast_precision_calls.store(
      0, std::memory_order_relaxed);
  cuda_near2far_adjoint_mixed_precision_calls.store(
      0, std::memory_order_relaxed);
  cuda_near2far_adjoint_cancellation_retries.store(
      0, std::memory_order_relaxed);
  cuda_near2far_adjoint_maximum_workspace_bytes.store(
      0, std::memory_order_relaxed);
  cuda_near2far_adjoint_descriptor_uploads.store(
      0, std::memory_order_relaxed);
  cuda_near2far_adjoint_kernel_launches.store(
      0, std::memory_order_relaxed);
  cuda_near2far_adjoint_host_to_device_bytes.store(
      0, std::memory_order_relaxed);
  cuda_near2far_adjoint_result_device_to_host_bytes.store(
      0, std::memory_order_relaxed);
  cuda_near2far_adjoint_condition_device_to_host_bytes.store(
      0, std::memory_order_relaxed);
  mpi_boundary_messages.store(0);
  mpi_boundary_scalars.store(0);
  cuda_aware_mpi_bytes.store(0);
  pinned_mpi_bytes.store(0);
  pinned_mpi_device_to_host_bytes.store(0);
  pinned_mpi_host_to_device_bytes.store(0);
  mpi_waitsome_executions.store(0);
  mpi_waitall_executions.store(0);
  host_to_device_bytes.store(0);
  device_to_host_bytes.store(0);
  finite_check_host_to_device_bytes.store(0);
  finite_check_device_to_host_bytes.store(0);
  host_to_device_bytes_avoided.store(0);
  device_to_host_bytes_avoided.store(0);
  device_buffer_allocations.store(0);
  device_buffer_reuses.store(0);
  boundary_phase_graph_creations.store(0);
  boundary_phase_graph_launches.store(0);
  boundary_receive_secondary_allocations.store(0);
  boundary_event_record_failures_for_testing.store(
      0, std::memory_order_relaxed);
  boundary_event_synchronize_failures_for_testing.store(
      0, std::memory_order_relaxed);
  boundary_device_synchronize_failures_for_testing.store(
      0, std::memory_order_relaxed);
  boundary_device_synchronize_fallbacks.store(
      0, std::memory_order_relaxed);
  boundary_lifetime_uncertain_leaks.store(
      0, std::memory_order_relaxed);
  near2far_execution_failures_for_testing.store(
      0, std::memory_order_relaxed);
  boundary_receive_pingpong_selections.store(0);
  boundary_receive_secondary_selections.store(0);
  boundary_eh_overlap_checks.store(0, std::memory_order_relaxed);
  boundary_eh_overlap_eligible.store(0, std::memory_order_relaxed);
  boundary_eh_overlap_launched_h.store(0, std::memory_order_relaxed);
  boundary_eh_overlap_launched_e.store(0, std::memory_order_relaxed);
  boundary_eh_overlap_skipped_disabled.store(0,
                                             std::memory_order_relaxed);
  boundary_eh_overlap_skipped_unsupported_schedule.store(
      0, std::memory_order_relaxed);
  boundary_eh_overlap_skipped_no_remote.store(0,
                                              std::memory_order_relaxed);
  boundary_eh_overlap_skipped_cold_topology.store(
      0, std::memory_order_relaxed);
  boundary_eh_overlap_rejected.store(0, std::memory_order_relaxed);
  halo_curl_overlap_checks.store(0, std::memory_order_relaxed);
  halo_curl_overlap_eligible.store(0, std::memory_order_relaxed);
  halo_curl_overlap_launches.store(0, std::memory_order_relaxed);
  halo_curl_overlap_skipped_disabled.store(0, std::memory_order_relaxed);
  halo_curl_overlap_skipped_unsupported_schedule.store(
      0, std::memory_order_relaxed);
  halo_curl_overlap_skipped_no_remote.store(0, std::memory_order_relaxed);
  halo_curl_overlap_skipped_cold_topology.store(0,
                                                std::memory_order_relaxed);
  halo_curl_overlap_rejected_feature.store(0, std::memory_order_relaxed);
  halo_curl_overlap_rejected_small.store(0, std::memory_order_relaxed);
  halo_curl_overlap_full_points.store(0, std::memory_order_relaxed);
  halo_curl_overlap_interior_points.store(0, std::memory_order_relaxed);
  halo_curl_overlap_shell_points.store(0, std::memory_order_relaxed);
  tile_coalesced_curl_chunk_phases.store(0, std::memory_order_relaxed);
  tile_coalesced_curl_input_tiles.store(0, std::memory_order_relaxed);
  tile_coalesced_update_eh_chunk_phases.store(
      0, std::memory_order_relaxed);
  tile_coalesced_update_eh_input_tiles.store(
      0, std::memory_order_relaxed);
  phase_curl_automatic_checks.store(0, std::memory_order_relaxed);
  phase_curl_automatic_selected.store(0, std::memory_order_relaxed);
  phase_curl_automatic_rejected.store(0, std::memory_order_relaxed);
  phase_curl_forced_batches.store(0, std::memory_order_relaxed);
  phase_curl_batched_operations.store(0, std::memory_order_relaxed);
  phase_curl_unbatched_operations.store(0, std::memory_order_relaxed);
  phase_curl_replay_checks.store(0, std::memory_order_relaxed);
  phase_curl_replay_hits.store(0, std::memory_order_relaxed);
  phase_curl_replay_unready.store(0, std::memory_order_relaxed);
  phase_curl_replay_generation_misses.store(0, std::memory_order_relaxed);
  phase_curl_replay_mirror_misses.store(0, std::memory_order_relaxed);
  curl_phase_replay_launch_failures_for_testing.store(
      0, std::memory_order_relaxed);
  phase_update_eh_automatic_checks.store(0, std::memory_order_relaxed);
  phase_update_eh_automatic_selected.store(0, std::memory_order_relaxed);
  phase_update_eh_automatic_rejected.store(0, std::memory_order_relaxed);
  phase_update_eh_forced_batches.store(0, std::memory_order_relaxed);
  phase_update_eh_batched_operations.store(0, std::memory_order_relaxed);
  phase_update_eh_unbatched_operations.store(
      0, std::memory_order_relaxed);
}

namespace detail {

curl_phase_replay_plan_snapshot
get_curl_phase_replay_plan_snapshot_for_testing(
    const void *owner, int phase_key) noexcept {
#if MEEP_HAVE_CUDA
  try {
    curl_phase_registry &registry = get_curl_phase_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.plans.find(
        std::make_pair(owner, phase_key));
    if (found == registry.plans.end()) return {false, false, 0};
    return {true, found->second.complete_phase,
            found->second.stable_reuses};
  }
  catch (...) {
    return {false, false, 0};
  }
#else
  (void)owner;
  (void)phase_key;
  return {false, false, 0};
#endif
}

void set_curl_phase_replay_launch_failures_for_testing(
    std::uint64_t count) noexcept {
  curl_phase_replay_launch_failures_for_testing.store(
      count, std::memory_order_relaxed);
}

void set_boundary_event_record_failures_for_testing(
    std::uint64_t count) noexcept {
  boundary_event_record_failures_for_testing.store(
      count, std::memory_order_relaxed);
}

void set_boundary_event_synchronize_failures_for_testing(
    std::uint64_t count) noexcept {
  boundary_event_synchronize_failures_for_testing.store(
      count, std::memory_order_relaxed);
}

void set_boundary_device_synchronize_failures_for_testing(
    std::uint64_t count) noexcept {
  boundary_device_synchronize_failures_for_testing.store(
      count, std::memory_order_relaxed);
}

void set_near2far_execution_failures_for_testing(
    std::uint64_t count) noexcept {
  near2far_execution_failures_for_testing.store(
      count, std::memory_order_relaxed);
}

boundary_phase_graph_statistics
get_boundary_phase_graph_statistics() noexcept {
  return {boundary_phase_graph_creations.load(),
          boundary_phase_graph_launches.load()};
}

boundary_receive_pingpong_statistics
get_boundary_receive_pingpong_statistics() noexcept {
  return {boundary_receive_secondary_allocations.load(),
          boundary_receive_pingpong_selections.load(),
          boundary_receive_secondary_selections.load()};
}

boundary_lifetime_fallback_statistics
get_boundary_lifetime_fallback_statistics() noexcept {
  return {
      boundary_device_synchronize_fallbacks.load(
          std::memory_order_relaxed),
      boundary_lifetime_uncertain_leaks.load(
          std::memory_order_relaxed)};
}

void record_boundary_eh_overlap_check() noexcept {
  boundary_eh_overlap_checks.fetch_add(1, std::memory_order_relaxed);
}

void record_boundary_eh_overlap_eligible() noexcept {
  boundary_eh_overlap_eligible.fetch_add(1, std::memory_order_relaxed);
}

void record_boundary_eh_overlap_launch(bool electric) noexcept {
  if (electric)
    boundary_eh_overlap_launched_e.fetch_add(
        1, std::memory_order_relaxed);
  else
    boundary_eh_overlap_launched_h.fetch_add(
        1, std::memory_order_relaxed);
}

void record_boundary_eh_overlap_disabled() noexcept {
  boundary_eh_overlap_skipped_disabled.fetch_add(
      1, std::memory_order_relaxed);
}

void record_boundary_eh_overlap_unsupported_schedule() noexcept {
  boundary_eh_overlap_skipped_unsupported_schedule.fetch_add(
      1, std::memory_order_relaxed);
}

void record_boundary_eh_overlap_no_remote() noexcept {
  boundary_eh_overlap_skipped_no_remote.fetch_add(
      1, std::memory_order_relaxed);
}

void record_boundary_eh_overlap_cold_topology() noexcept {
  boundary_eh_overlap_skipped_cold_topology.fetch_add(
      1, std::memory_order_relaxed);
}

void record_boundary_eh_overlap_rejected() noexcept {
  boundary_eh_overlap_rejected.fetch_add(1, std::memory_order_relaxed);
}

void record_halo_curl_overlap_check() noexcept {
  halo_curl_overlap_checks.fetch_add(1, std::memory_order_relaxed);
}

void record_halo_curl_overlap_eligible() noexcept {
  halo_curl_overlap_eligible.fetch_add(1, std::memory_order_relaxed);
}

void record_halo_curl_overlap_launch(std::uint64_t full_points,
                                     std::uint64_t interior_points,
                                     std::uint64_t shell_points) noexcept {
  halo_curl_overlap_launches.fetch_add(1, std::memory_order_relaxed);
  halo_curl_overlap_full_points.fetch_add(full_points,
                                          std::memory_order_relaxed);
  halo_curl_overlap_interior_points.fetch_add(interior_points,
                                              std::memory_order_relaxed);
  halo_curl_overlap_shell_points.fetch_add(shell_points,
                                           std::memory_order_relaxed);
}

void record_halo_curl_overlap_disabled() noexcept {
  halo_curl_overlap_skipped_disabled.fetch_add(1,
                                                std::memory_order_relaxed);
}

void record_halo_curl_overlap_unsupported_schedule() noexcept {
  halo_curl_overlap_skipped_unsupported_schedule.fetch_add(
      1, std::memory_order_relaxed);
}

void record_halo_curl_overlap_no_remote() noexcept {
  halo_curl_overlap_skipped_no_remote.fetch_add(1,
                                                 std::memory_order_relaxed);
}

void record_halo_curl_overlap_cold_topology() noexcept {
  halo_curl_overlap_skipped_cold_topology.fetch_add(
      1, std::memory_order_relaxed);
}

void record_halo_curl_overlap_rejected(bool small) noexcept {
  (small ? halo_curl_overlap_rejected_small
         : halo_curl_overlap_rejected_feature)
      .fetch_add(1, std::memory_order_relaxed);
}

void record_curl_tile_coalescing(std::uint64_t input_tiles) noexcept {
  tile_coalesced_curl_chunk_phases.fetch_add(
      1, std::memory_order_relaxed);
  tile_coalesced_curl_input_tiles.fetch_add(
      input_tiles, std::memory_order_relaxed);
}

void record_update_eh_tile_coalescing(
    std::uint64_t input_tiles) noexcept {
  tile_coalesced_update_eh_chunk_phases.fetch_add(
      1, std::memory_order_relaxed);
  tile_coalesced_update_eh_input_tiles.fetch_add(
      input_tiles, std::memory_order_relaxed);
}

dft_batch_statistics get_dft_batch_statistics() noexcept {
  return ::meep::gpu::get_dft_batch_statistics();
}

dft_reduction_statistics get_dft_reduction_statistics() noexcept {
  return ::meep::gpu::get_dft_reduction_statistics();
}

dft_materialization_statistics
get_dft_materialization_statistics() noexcept {
  return ::meep::gpu::get_dft_materialization_statistics();
}

dft_checkpoint_statistics get_dft_checkpoint_statistics() noexcept {
  return ::meep::gpu::get_dft_checkpoint_statistics();
}

dft_scale_statistics get_dft_scale_statistics() noexcept {
  return ::meep::gpu::get_dft_scale_statistics();
}

eigenmode_overlap_statistics get_eigenmode_overlap_statistics() noexcept {
  return ::meep::gpu::get_eigenmode_overlap_statistics();
}

ldos_reduction_statistics get_ldos_reduction_statistics() noexcept {
  return {
      ldos_batch_calls.load(std::memory_order_relaxed),
      ldos_submitted_profiles.load(std::memory_order_relaxed),
      ldos_source_points.load(std::memory_order_relaxed),
      ldos_descriptor_uploads.load(std::memory_order_relaxed),
      ldos_kernel_launches.load(std::memory_order_relaxed),
      ldos_result_device_to_host_bytes.load(std::memory_order_relaxed),
      ldos_full_field_device_to_host_bytes_avoided.load(
          std::memory_order_relaxed)};
}

void resident_reduce_ldos_reentrant_for_testing(
    const ldos_reduction_request_fp32 *requests,
    std::size_t request_count, double result[4]) {
#if MEEP_HAVE_CUDA
  // The test-only caller owns the exact production mutex before entering the
  // real reduction.  Its try-lock therefore rejects deterministically without
  // a flag, helper thread, timing window, or branch in the production path.
  std::lock_guard<std::mutex> ldos_lock(get_ldos_reduction_mutex());
  resident_reduce_ldos_fp32(requests, request_count, result);
#else
  (void)requests;
  (void)request_count;
  (void)result;
  throw std::runtime_error(
      "Meep CUDA LDOS reentry probe called in a CPU-only build");
#endif
}

void fail_next_ldos_plan_commit_for_testing() noexcept {
  fail_ldos_plan_commit_for_testing.store(
      true, std::memory_order_release);
}

std::size_t resident_ldos_plan_buffer_count_for_testing(
    const resident_cache *cache) noexcept {
#if MEEP_HAVE_CUDA
  if (!cache) return 0;
  return (cache->ldos_plan_buffers[0].device_pointer ? 1u : 0u) +
         (cache->ldos_plan_buffers[1].device_pointer ? 1u : 0u);
#else
  (void)cache;
  return 0;
#endif
}

void record_boundary_receive_pingpong_selection(bool secondary) noexcept {
  boundary_receive_pingpong_selections.fetch_add(1);
  if (secondary)
    boundary_receive_secondary_selections.fetch_add(1);
}

void fail_next_multilevel_plan_commit_for_testing() noexcept {
  fail_multilevel_plan_commit_for_testing.store(
      true, std::memory_order_release);
}

finite_check_transfer_statistics
get_finite_check_transfer_statistics() noexcept {
  return {finite_check_host_to_device_bytes.load(),
          finite_check_device_to_host_bytes.load()};
}

std::size_t finite_check_descriptor_bytes(
    std::size_t descriptor_count) noexcept {
#if MEEP_HAVE_CUDA
  return descriptor_count *
         sizeof(meep_cuda::array_span_fp32);
#else
  (void)descriptor_count;
  return 0;
#endif
}

bool cuda_active() { return active_backend() == backend_mode::cuda; }

bool cuda_required() { return requested_backend() == backend_mode::cuda; }

std::uint64_t backend_generation() {
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.generation;
}

backend_preflight_status probe_backend_for_distributed_step() {
  backend_preflight_status status =
      {false, false, false, false, false, false, std::string()};
  try {
#if MEEP_HAVE_CUDA
    // set_backend(auto) validates the environment transactionally, but users
    // can legitimately change policy variables before a later fields step.
    // An owner cache compares the same raw values before entering this probe;
    // refresh the process configuration here so a malformed new value cannot
    // be hidden by an old device selection. Preserve device-authoritative
    // state before changing or clearing that selection.
    const std::string current_environment_signature =
        backend_configuration_environment_signature();
    bool refresh_configuration = false;
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      refresh_configuration =
          state.configured && state.requested != backend_mode::cpu &&
          state.configured_environment_signature !=
              current_environment_signature;
    }
    if (refresh_configuration) {
      sync_and_invalidate_all_resident_caches();
      // Retain the current MPI-world claim until the new configuration has
      // passed both parsing and device selection.  A failed refresh restores
      // the previous backend generation/state transactionally; releasing the
      // claim here would let the corresponding fields-owner cache become
      // valid again after the environment is restored without ever reclaiming
      // its physical GPU.  A successful refresh invalidates that owner cache
      // through the generation bump, and the later distributed assignment
      // preflight releases/replaces the old claim as one collective set.
      std::lock_guard<std::mutex> lock(state.mutex);
      if (state.configured && state.requested != backend_mode::cpu &&
          state.configured_environment_signature !=
              current_environment_signature)
        configure_transaction_locked(state.requested);
    }
#endif
    std::lock_guard<std::mutex> lock(state.mutex);
    backend_mode requested = state.requested;
    if (!state.configured)
      requested = parse_mode(std::getenv("MEEP_GPU_BACKEND"));
    status.cuda_required = requested == backend_mode::cuda;
    status.automatic_requested =
        requested == backend_mode::automatic;
    status.cpu_requested = requested == backend_mode::cpu;
    const char *device_value = std::getenv("MEEP_GPU_DEVICE");
    status.explicit_device_requested =
        (device_value && *device_value) || state.explicit_device_selection;
    ensure_configured_locked();
    status.configuration_ok = true;
    // Automatic owner decisions may publish CPU as the last active backend,
    // but the already-selected device remains a CUDA candidate for another
    // owner.  Candidate availability and current execution are deliberately
    // separate here.
    status.cuda_active =
        state.requested != backend_mode::cpu && state.device_ready;
    status.cuda_required = state.requested == backend_mode::cuda;
    status.automatic_requested =
        state.requested == backend_mode::automatic;
    status.cpu_requested = state.requested == backend_mode::cpu;
    status.explicit_device_requested =
        (device_value && *device_value) || state.explicit_device_selection;
    status.diagnostic = state.diagnostic;
  }
  catch (const std::exception &error) {
    status.diagnostic = error.what();
  }
  catch (...) {
    status.diagnostic = "unknown CUDA backend configuration error";
  }
  return status;
}

backend_preflight_status prepare_cuda_candidate_for_distributed_step() {
  backend_preflight_status status =
      {false, false, false, false, false, false, std::string()};
  try {
    std::lock_guard<std::mutex> lock(state.mutex);
    ensure_configured_locked();
    status.cuda_required = state.requested == backend_mode::cuda;
    status.automatic_requested =
        state.requested == backend_mode::automatic;
    status.cpu_requested = state.requested == backend_mode::cpu;
    const char *device_value = std::getenv("MEEP_GPU_DEVICE");
    status.explicit_device_requested =
        (device_value && *device_value) || state.explicit_device_selection;
    if (state.requested == backend_mode::automatic &&
        !state.device_ready) {
#if MEEP_HAVE_CUDA
      std::string runtime_diagnostic;
      const bool explicit_device_environment = device_value && *device_value;
      const bool explicit_device =
          explicit_device_environment || state.explicit_device_selection;
      const int environment_device = explicit_device_environment
                                         ? parse_device_ordinal(device_value)
                                         : -1;
      const bool runtime_is_available =
          meep_cuda::runtime_available(&runtime_diagnostic);
      int ordinal = state.selected_device;
      std::string selection_error;
      if (!runtime_is_available) {
        if (explicit_device)
          throw std::runtime_error(
              "explicit CUDA device selection failed: " +
              runtime_diagnostic);
        selection_error = runtime_diagnostic;
      }
      else {
        try {
          if (explicit_device_environment) ordinal = environment_device;
          if (ordinal < 0) ordinal = rank_compatible_device();
          if (ordinal < 0)
            throw std::runtime_error(
                "CUDA runtime found no compatible device");
          meep_cuda::select_device(ordinal);
        }
        catch (const std::exception &error) {
          if (explicit_device)
            throw std::runtime_error(
                "explicit CUDA device selection failed: " +
                std::string(error.what()));
          selection_error = error.what();
        }
      }
      if (selection_error.empty()) {
        state.selected_device = ordinal;
        state.device_ready = true;
        std::ostringstream message;
        message << "CUDA device " << ordinal
                << " prepared for automatic fields preflight on global rank "
                << my_global_rank() << " (node rank " << my_node_rank()
                << "/" << count_node_processors() << ")";
        state.diagnostic = message.str();
      }
      else
        state.diagnostic = selection_error;
#else
      state.diagnostic = "Meep was built without CUDA support";
#endif
    }
    status.configuration_ok = true;
    status.cuda_active =
        state.requested != backend_mode::cpu && state.device_ready;
    status.diagnostic = state.diagnostic;
  }
  catch (const std::exception &error) {
    status.diagnostic = error.what();
  }
  catch (...) {
    status.diagnostic = "unknown CUDA candidate configuration error";
  }
  return status;
}

automatic_device_policy_facts selected_automatic_device_policy_facts() {
#if MEEP_HAVE_CUDA
  int ordinal = -1;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    ensure_configured_locked();
    if (!state.device_ready || state.selected_device < 0)
      throw std::runtime_error(
          "automatic CUDA policy requires a selected compatible device");
    ordinal = state.selected_device;
  }
  const std::vector<meep_cuda::device_info> devices =
      meep_cuda::enumerate_devices();
  for (const meep_cuda::device_info &device : devices)
    if (device.ordinal == ordinal) {
      const meep_cuda::device_memory_info memory =
          meep_cuda::selected_device_memory_info();
      return {device.compute_major, device.compute_minor,
              device.multiprocessor_count,
              device.memory_bandwidth_bytes_per_second,
              memory.free_bytes, memory.total_bytes};
    }
  throw std::runtime_error(
      "selected CUDA device disappeared during automatic policy preflight");
#else
  throw std::runtime_error(
      "automatic CUDA policy is unavailable in a CPU-only build");
#endif
}

void activate_backend_for_owner(backend_mode execution,
                                const std::string &diagnostic) {
  if (execution != backend_mode::cpu && execution != backend_mode::cuda)
    throw std::invalid_argument(
        "fields owner execution backend must be CPU or CUDA");
  std::lock_guard<std::mutex> lock(state.mutex);
  ensure_configured_locked();
  if (state.requested == backend_mode::cpu &&
      execution != backend_mode::cpu)
    throw std::logic_error(
        "CPU-requested fields owner cannot activate CUDA");
  if (state.requested == backend_mode::cuda &&
      execution != backend_mode::cuda)
    throw std::logic_error(
        "required CUDA fields owner cannot activate CPU");
  if (execution == backend_mode::cuda && !state.device_ready)
    throw std::logic_error(
        "fields owner cannot activate an unconfigured CUDA device");
  state.active = execution;
  state.diagnostic = diagnostic.empty()
                         ? (execution == backend_mode::cuda
                                ? "fields owner selected CUDA"
                                : "fields owner selected CPU")
                         : diagnostic;
}

void validate_distributed_device_assignment() {
#if MEEP_HAVE_CUDA
  int ordinal = -1;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    ensure_configured_locked();
    if (state.active != backend_mode::cuda ||
        state.selected_device < 0)
      throw std::runtime_error(
          "distributed CUDA validation requires an active CUDA device");
    ordinal = state.selected_device;
  }
  bool allow_duplicates = false;
  bool local_assignment_valid = true;
  std::string identifier;
  try {
    allow_duplicates = parse_allow_oversubscription();
  }
  catch (...) {
    local_assignment_valid = false;
  }
  try {
    identifier = meep_cuda::device_uuid(ordinal);
  }
  catch (...) {
    local_assignment_valid = false;
  }
  // Every participating rank must reach the identifier allgather. An empty
  // value makes a local UUID/opt-in parse failure fail uniformly.
  if (!local_assignment_valid) identifier.clear();
  if (!distributed_device_identifiers_are_unique(
          identifier.c_str(), allow_duplicates)) {
    release_distributed_device_identifier();
    throw std::runtime_error(
        "MPI ranks did not select distinct physical CUDA/MIG devices or a "
        "GPU assignment could not be validated; assign unique "
        "CUDA_VISIBLE_DEVICES mappings or distinct MEEP_GPU_DEVICE ordinals, "
        "or opt in to sharing with "
        "MEEP_GPU_ALLOW_OVERSUBSCRIBE=1");
  }

  // Release this communicator's previous assignments as a group before
  // claiming the desired set. This permits a valid 0<->1 device permutation
  // without either rank observing the peer's stale claim. Unrelated
  // subcommunicators retain their claims and can still reject the new set.
  release_distributed_device_identifier();
  all_wait();
  const bool claim_succeeded =
      claim_distributed_device_identifier(identifier.c_str(),
                                          allow_duplicates);
  if (!and_to_all(claim_succeeded)) {
    release_distributed_device_identifier();
    throw std::runtime_error(
        "a physical CUDA/MIG device is already claimed by another Meep "
        "subcommunicator; assign one unique GPU per MPI rank or set "
        "MEEP_GPU_ALLOW_OVERSUBSCRIBE=1 explicitly");
  }
#else
  throw std::runtime_error(
      "distributed CUDA validation called in a CPU-only build");
#endif
}

int choose_rank_device_ordinal(const int *compatible_ordinals,
                               std::size_t compatible_count,
                               int node_rank, bool allow_oversubscription) {
  if (node_rank < 0)
    throw std::invalid_argument("Meep CUDA node-local rank must be non-negative");
  if (!compatible_ordinals && compatible_count)
    throw std::invalid_argument(
        "Meep CUDA compatible-device list must be non-null");
  if (compatible_count == 0) return -1;

  // Some schedulers expose exactly one distinct GPU to each rank. Every rank
  // then correctly selects runtime ordinal zero even though its node rank is
  // greater than zero.
  if (compatible_count == 1) return compatible_ordinals[0];
  if (static_cast<std::size_t>(node_rank) < compatible_count)
    return compatible_ordinals[node_rank];
  if (allow_oversubscription)
    return compatible_ordinals[
        static_cast<std::size_t>(node_rank) % compatible_count];

  std::ostringstream message;
  message << "Meep CUDA node rank " << node_rank << " has no dedicated GPU: "
          << compatible_count
          << " compatible CUDA devices are visible; set CUDA_VISIBLE_DEVICES "
             "per rank, MEEP_GPU_DEVICE explicitly, or "
             "MEEP_GPU_ALLOW_OVERSUBSCRIBE=1";
  throw std::runtime_error(message.str());
}

void record_cpu_curl(std::size_t points) noexcept {
  cpu_curl_calls.fetch_add(1);
  cpu_curl_points.fetch_add(static_cast<std::uint64_t>(points));
}

void record_cpu_update_eh(std::size_t points) noexcept {
  cpu_update_eh_calls.fetch_add(1);
  cpu_update_eh_points.fetch_add(static_cast<std::uint64_t>(points));
}

void record_cpu_polarization(std::size_t points) noexcept {
  cpu_polarization_calls.fetch_add(1);
  cpu_polarization_points.fetch_add(static_cast<std::uint64_t>(points));
}

void record_cpu_source(std::size_t points) noexcept {
  cpu_source_calls.fetch_add(1);
  cpu_source_points.fetch_add(static_cast<std::uint64_t>(points));
}

void record_cuda_source(std::size_t points) noexcept {
  cuda_source_calls.fetch_add(1);
  cuda_source_points.fetch_add(static_cast<std::uint64_t>(points));
}

void record_cpu_boundary(std::size_t points) noexcept {
  cpu_boundary_calls.fetch_add(1);
  cpu_boundary_points.fetch_add(static_cast<std::uint64_t>(points));
}

void record_cpu_dft(std::size_t points) noexcept {
  cpu_dft_calls.fetch_add(1);
  cpu_dft_points.fetch_add(static_cast<std::uint64_t>(points));
}

void record_cpu_dft_reduction(
    std::size_t pair_count,
    std::size_t point_frequency_terms) noexcept {
  cpu_dft_reduction_calls.fetch_add(1, std::memory_order_relaxed);
  cpu_dft_reduction_pairs.fetch_add(
      static_cast<std::uint64_t>(pair_count),
      std::memory_order_relaxed);
  cpu_dft_reduction_terms.fetch_add(
      static_cast<std::uint64_t>(point_frequency_terms),
      std::memory_order_relaxed);
}

void record_cpu_dft_array_materialization(std::size_t points) noexcept {
  cpu_dft_array_materialization_calls.fetch_add(
      1, std::memory_order_relaxed);
  cpu_dft_array_materialization_points.fetch_add(
      static_cast<std::uint64_t>(points), std::memory_order_relaxed);
}

void record_cuda_dft_array_materialization(std::size_t points) noexcept {
  cuda_dft_array_materialization_calls.fetch_add(
      1, std::memory_order_relaxed);
  cuda_dft_array_materialization_points.fetch_add(
      static_cast<std::uint64_t>(points), std::memory_order_relaxed);
}

void record_host_synthetic_material_array(std::size_t points) noexcept {
  host_synthetic_material_array_calls.fetch_add(
      1, std::memory_order_relaxed);
  host_synthetic_material_array_points.fetch_add(
      static_cast<std::uint64_t>(points), std::memory_order_relaxed);
}

void record_cpu_dft_output_call() noexcept {
  cpu_dft_output_calls.fetch_add(1, std::memory_order_relaxed);
}

void record_cpu_dft_output_points(std::size_t points) noexcept {
  cpu_dft_output_points.fetch_add(
      static_cast<std::uint64_t>(points), std::memory_order_relaxed);
}

void record_cuda_dft_output_call() noexcept {
  cuda_dft_output_calls.fetch_add(1, std::memory_order_relaxed);
}

void record_cuda_dft_output_points(std::size_t points) noexcept {
  cuda_dft_output_points.fetch_add(
      static_cast<std::uint64_t>(points), std::memory_order_relaxed);
}

void record_cpu_dft_overlap_call() noexcept {
  cpu_dft_overlap_calls.fetch_add(1, std::memory_order_relaxed);
}

void record_cpu_dft_overlap_terms(std::size_t terms) noexcept {
  cpu_dft_overlap_terms.fetch_add(
      static_cast<std::uint64_t>(terms), std::memory_order_relaxed);
}

void record_eigenmode_host_profile_sampling(
    std::size_t points) noexcept {
  host_mode_profile_sampling_calls.fetch_add(
      1, std::memory_order_relaxed);
  host_mode_profile_sampling_points.fetch_add(
      static_cast<std::uint64_t>(points), std::memory_order_relaxed);
}

void record_eigenmode_zero_rank_channels_skipped(
    std::size_t channels) noexcept {
  eigenmode_zero_rank_channels_skipped.fetch_add(
      static_cast<std::uint64_t>(channels),
      std::memory_order_relaxed);
}

void record_eigenmode_mpi_allreduce(std::size_t bytes) noexcept {
  eigenmode_mpi_allreduce_calls.fetch_add(
      1, std::memory_order_relaxed);
  eigenmode_mpi_allreduce_bytes.fetch_add(
      static_cast<std::uint64_t>(bytes), std::memory_order_relaxed);
}

void record_dft_array_mpi_allreduce(std::size_t bytes) noexcept {
  dft_array_mpi_allreduce_calls.fetch_add(1, std::memory_order_relaxed);
  dft_array_mpi_allreduce_bytes.fetch_add(
      static_cast<std::uint64_t>(bytes), std::memory_order_relaxed);
}

void record_cpu_dft_checkpoint_save(std::size_t values) noexcept {
  cpu_dft_checkpoint_save_calls.fetch_add(1, std::memory_order_relaxed);
  cpu_dft_checkpoint_save_values.fetch_add(
      static_cast<std::uint64_t>(values), std::memory_order_relaxed);
}

void record_cuda_dft_checkpoint_save(
    std::size_t values, std::size_t device_to_host_bytes,
    std::size_t full_cache_device_to_host_bytes_avoided) noexcept {
  cuda_dft_checkpoint_save_calls.fetch_add(1, std::memory_order_relaxed);
  cuda_dft_checkpoint_save_values.fetch_add(
      static_cast<std::uint64_t>(values), std::memory_order_relaxed);
  cuda_dft_checkpoint_save_d2h_bytes.fetch_add(
      static_cast<std::uint64_t>(device_to_host_bytes),
      std::memory_order_relaxed);
  cuda_dft_checkpoint_save_full_cache_d2h_avoided.fetch_add(
      static_cast<std::uint64_t>(
          full_cache_device_to_host_bytes_avoided),
      std::memory_order_relaxed);
}

void record_cpu_dft_checkpoint_load(std::size_t values) noexcept {
  cpu_dft_checkpoint_load_calls.fetch_add(1, std::memory_order_relaxed);
  cpu_dft_checkpoint_load_values.fetch_add(
      static_cast<std::uint64_t>(values), std::memory_order_relaxed);
}

void record_cuda_dft_checkpoint_load(
    std::size_t values, std::size_t host_to_device_bytes) noexcept {
  cuda_dft_checkpoint_load_calls.fetch_add(1, std::memory_order_relaxed);
  cuda_dft_checkpoint_load_values.fetch_add(
      static_cast<std::uint64_t>(values), std::memory_order_relaxed);
  cuda_dft_checkpoint_load_h2d_bytes.fetch_add(
      static_cast<std::uint64_t>(host_to_device_bytes),
      std::memory_order_relaxed);
}

void record_cpu_dft_scale(std::size_t values) noexcept {
  cpu_dft_scale_calls.fetch_add(1, std::memory_order_relaxed);
  cpu_dft_scale_values.fetch_add(
      static_cast<std::uint64_t>(values), std::memory_order_relaxed);
}

void record_cuda_dft_scale(std::size_t values,
                           std::size_t kernel_launches,
                           std::size_t host_to_device_bytes) noexcept {
  cuda_dft_scale_calls.fetch_add(1, std::memory_order_relaxed);
  cuda_dft_scale_values.fetch_add(
      static_cast<std::uint64_t>(values), std::memory_order_relaxed);
  cuda_dft_scale_kernel_launches.fetch_add(
      static_cast<std::uint64_t>(kernel_launches),
      std::memory_order_relaxed);
  cuda_dft_scale_h2d_bytes.fetch_add(
      static_cast<std::uint64_t>(host_to_device_bytes),
      std::memory_order_relaxed);
}

void set_dft_checkpoint_d2h_failure_after_for_testing(
    std::int64_t successful_transfers) noexcept {
  dft_checkpoint_d2h_failure_after_for_testing.store(
      successful_transfers < -1 ? -1 : successful_transfers,
      std::memory_order_relaxed);
}

void set_dft_checkpoint_h2d_failure_after_for_testing(
    std::int64_t successful_transfers) noexcept {
  dft_checkpoint_h2d_failure_after_for_testing.store(
      successful_transfers < -1 ? -1 : successful_transfers,
      std::memory_order_relaxed);
}

void set_dft_output_staging_d2h_failure_after_for_testing(
    std::int64_t successful_transfers) noexcept {
  dft_output_staging_d2h_failure_after_for_testing.store(
      successful_transfers < -1 ? -1 : successful_transfers,
      std::memory_order_relaxed);
}

std::uint64_t get_live_dft_output_pinned_buffers_for_testing() noexcept {
  return live_dft_output_pinned_buffers.load(std::memory_order_relaxed);
}

void record_cpu_ldos(std::size_t points) noexcept {
  cpu_ldos_calls.fetch_add(1, std::memory_order_relaxed);
  cpu_ldos_source_points.fetch_add(
      static_cast<std::uint64_t>(points), std::memory_order_relaxed);
}

void record_cpu_near2far(std::uint64_t terms) noexcept {
  cpu_near2far_calls.fetch_add(1, std::memory_order_relaxed);
  cpu_near2far_terms.fetch_add(terms, std::memory_order_relaxed);
}

void record_cpu_near2far_adjoint(std::uint64_t terms) noexcept {
  cpu_near2far_adjoint_calls.fetch_add(1, std::memory_order_relaxed);
  cpu_near2far_adjoint_terms.fetch_add(terms, std::memory_order_relaxed);
}

void record_near2far_mpi_allreduce(std::size_t bytes) noexcept {
  near2far_mpi_allreduce_calls.fetch_add(1, std::memory_order_relaxed);
  near2far_mpi_allreduce_bytes.fetch_add(
      static_cast<std::uint64_t>(bytes), std::memory_order_relaxed);
}

void record_dft_reduction_mpi_allreduce(std::size_t bytes) noexcept {
  dft_reduction_mpi_allreduce_calls.fetch_add(
      1, std::memory_order_relaxed);
  dft_reduction_mpi_allreduce_bytes.fetch_add(
      static_cast<std::uint64_t>(bytes),
      std::memory_order_relaxed);
}

void destroy_resident_dft_reduction_plans_for_owner(
    const void *plan_owner) noexcept {
#if MEEP_HAVE_CUDA
  clear_dft_reduction_plans_for_owner(plan_owner);
#else
  (void)plan_owner;
#endif
}

void destroy_resident_eigenmode_overlap_plans_for_owner(
    const void *plan_owner) noexcept {
#if MEEP_HAVE_CUDA
  clear_eigenmode_overlap_plans_for_owner(plan_owner);
#else
  (void)plan_owner;
#endif
}

void destroy_resident_near2far_plan_for_owner(
    const void *plan_owner) noexcept {
#if MEEP_HAVE_CUDA
  clear_near2far_plan_for_owner(plan_owner);
  clear_near2far_adjoint_plan_for_owner(plan_owner);
#else
  (void)plan_owner;
#endif
}

std::size_t exchange_near2far_workspace_ceiling_for_testing(
    std::size_t bytes) {
  if (bytes < 256)
    throw std::invalid_argument(
        "Meep CUDA Near2Far test workspace ceiling must be at least 256 "
        "bytes");
  return near2far_workspace_ceiling_bytes.exchange(
      bytes, std::memory_order_relaxed);
}

resident_curl_session::resident_curl_session(const void *owner, bool enabled)
    : cache_(nullptr), active_(false), owns_phase_(false) {
#if MEEP_HAVE_CUDA
  if (!enabled) return;
  cache_ = resident_cache_for_owner(owner);
  if (!cache_->phase_active) {
    begin_resident_phase(cache_);
    owns_phase_ = true;
  }
  active_ = true;
#else
  (void)owner;
  if (enabled)
    throw std::runtime_error("Meep CUDA resident curl phase called in a CPU-only build");
#endif
}

resident_curl_session::~resident_curl_session() {
#if MEEP_HAVE_CUDA
  if (active_ && owns_phase_) discard_resident_phase(cache_);
#endif
}

resident_curl_session::resident_curl_session(
    resident_curl_session &&other) noexcept
    : cache_(other.cache_), active_(other.active_),
      owns_phase_(other.owns_phase_) {
  other.cache_ = nullptr;
  other.active_ = false;
  other.owns_phase_ = false;
}

void resident_curl_session::finish(bool synchronize_to_host) {
#if MEEP_HAVE_CUDA
  if (!active_) return;
  if (!owns_phase_) {
    active_ = false;
    return;
  }
  try {
    finish_resident_phase(cache_, synchronize_to_host);
    active_ = false;
  }
  catch (...) {
    discard_resident_phase(cache_);
    active_ = false;
    throw;
  }
#else
  (void)synchronize_to_host;
  if (active_)
    throw std::runtime_error("Meep CUDA resident curl phase called in a CPU-only build");
#endif
}

resident_curl_phase_batch::resident_curl_phase_batch(
    const void *owner, int phase_key, phase_batch_mode mode,
    std::uint64_t topology_fingerprint, bool replay_supported)
    : active_(false) {
#if MEEP_HAVE_CUDA
  if (mode == phase_batch_mode::disabled) return;
  if (!owner)
    throw std::invalid_argument(
        "Meep CUDA phase-batched curl owner must be non-null");
  curl_phase_collection &collection = pending_curl_phase;
  if (collection.active)
    throw std::logic_error(
        "nested Meep CUDA phase-batched curls are not supported");
  collection.active = true;
  collection.mode = mode;
  collection.owner = owner;
  collection.phase_key = phase_key;
  collection.device_ordinal = -1;
  collection.point_count = 0;
  collection.topology_fingerprint = topology_fingerprint;
  collection.operations.clear();
  collection.cache_dependencies.clear();
  collection.mirror_dependencies.clear();
  collection.writable_mirrors.clear();
  collection.flush_count = 0;
  collection.batched_flush_count = 0;
  collection.replayed = false;
  collection.replay_forbidden = false;
  collection.replay_supported = replay_supported;
  active_ = true;
#else
  (void)owner;
  (void)phase_key;
  (void)topology_fingerprint;
  (void)replay_supported;
  if (mode != phase_batch_mode::disabled)
    throw std::runtime_error(
        "Meep CUDA phase-batched curl called in a CPU-only build");
#endif
}

resident_curl_phase_batch::~resident_curl_phase_batch() {
#if MEEP_HAVE_CUDA
  if (!active_) return;
  curl_phase_collection &collection = pending_curl_phase;
  if (!collection.replayed)
    mark_curl_phase_plan_incomplete_noexcept(
        collection.owner, collection.phase_key);
  collection.operations.clear();
  collection.cache_dependencies.clear();
  collection.mirror_dependencies.clear();
  collection.writable_mirrors.clear();
  collection.active = false;
  collection.mode = phase_batch_mode::disabled;
  collection.owner = nullptr;
  collection.device_ordinal = -1;
  collection.point_count = 0;
  collection.topology_fingerprint = 0;
  collection.flush_count = 0;
  collection.batched_flush_count = 0;
  collection.replayed = false;
  collection.replay_forbidden = false;
  collection.replay_supported = false;
#endif
}

bool resident_curl_phase_batch::replay_if_ready() {
#if MEEP_HAVE_CUDA
  if (!active_ ||
      std::getenv("MEEP_GPU_DISABLE_CURL_PHASE_REPLAY") != nullptr)
    return false;
  phase_curl_replay_checks.fetch_add(1, std::memory_order_relaxed);
  curl_phase_collection &collection = pending_curl_phase;
  if (!collection.active || !collection.operations.empty() ||
      !collection.replay_supported) {
    phase_curl_replay_unready.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  curl_phase_registry &registry = get_curl_phase_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto found = registry.plans.find(
      std::make_pair(collection.owner, collection.phase_key));
  if (found == registry.plans.end()) {
    phase_curl_replay_unready.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  curl_phase_plan &plan = found->second;
  if (!plan.complete_phase || plan.mode != collection.mode ||
      plan.topology_fingerprint != collection.topology_fingerprint ||
      !plan.device_operations ||
      plan.device_ordinal < 0 ||
      plan.stable_reuses < 2 || plan.operation_count < 2 ||
      plan.total_block_count == 0 || plan.descriptor_bytes == 0 ||
      plan.cache_generations.empty() ||
      plan.mirror_dependencies.empty() || plan.writable_mirrors.empty()) {
    phase_curl_replay_unready.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  for (const auto &generation : plan.cache_generations)
    if (!generation.first || !generation.first->phase_active ||
        generation.first->device_ordinal != plan.device_ordinal ||
        generation.first->allocation_generation != generation.second) {
      phase_curl_replay_generation_misses.fetch_add(
          1, std::memory_order_relaxed);
      return false;
    }
  for (const curl_phase_plan::mirror_dependency &dependency :
       plan.mirror_dependencies) {
    if (!dependency.cache || !dependency.host_pointer ||
        dependency.bytes == 0) {
      phase_curl_replay_mirror_misses.fetch_add(
          1, std::memory_order_relaxed);
      return false;
    }
    const auto mirror =
        dependency.cache->mirrors.find(dependency.host_pointer);
    if (mirror == dependency.cache->mirrors.end() ||
        mirror->second.bytes != dependency.bytes ||
        (!mirror->second.device_dirty &&
         mirror->second.uploaded_epoch == 0)) {
      phase_curl_replay_mirror_misses.fetch_add(
          1, std::memory_order_relaxed);
      return false;
    }
  }
  // Complete the entire validation before marking any output authoritative.
  // This keeps a rejected replay side-effect-free and lets the ordinary
  // collection path refresh a host-invalidated mirror.
  for (const curl_phase_plan::mirror_dependency &writable :
       plan.writable_mirrors) {
    if (!writable.cache || !writable.host_pointer || writable.bytes == 0) {
      phase_curl_replay_mirror_misses.fetch_add(
          1, std::memory_order_relaxed);
      return false;
    }
    const auto mirror = writable.cache->mirrors.find(writable.host_pointer);
    if (mirror == writable.cache->mirrors.end() ||
        mirror->second.bytes != writable.bytes) {
      phase_curl_replay_mirror_misses.fetch_add(
          1, std::memory_order_relaxed);
      return false;
    }
  }

  try {
    meep_cuda::select_device(plan.device_ordinal);
    if (consume_failure_for_testing(
            curl_phase_replay_launch_failures_for_testing))
      throw std::runtime_error(
          "injected CUDA curl phase replay launch failure");
    meep_cuda::step_curl_material_phase_batched_fp32(
        static_cast<const meep_cuda::curl_phase_operation_fp32 *>(
            plan.device_operations),
        reinterpret_cast<const std::uint32_t *>(
            static_cast<const unsigned char *>(plan.device_operations) +
            plan.descriptor_bytes),
        plan.operation_count, plan.total_block_count);
  }
  catch (...) {
    plan.complete_phase = false;
    plan.stable_reuses = 0;
    collection.replay_forbidden = true;
    phase_curl_replay_unready.fetch_add(1, std::memory_order_relaxed);
    throw;
  }
  for (const curl_phase_plan::mirror_dependency &writable :
       plan.writable_mirrors)
    mark_device_dirty(
        writable.cache, writable.host_pointer, writable.bytes);
  cuda_curl_calls.fetch_add(1);
  cuda_curl_points.fetch_add(plan.point_count);
  phase_curl_batched_operations.fetch_add(
      static_cast<std::uint64_t>(plan.operation_count),
      std::memory_order_relaxed);
  if (collection.mode == phase_batch_mode::automatic) {
    phase_curl_automatic_checks.fetch_add(1, std::memory_order_relaxed);
    phase_curl_automatic_selected.fetch_add(1, std::memory_order_relaxed);
  }
  else if (collection.mode == phase_batch_mode::forced)
    phase_curl_forced_batches.fetch_add(1, std::memory_order_relaxed);
  device_buffer_reuses.fetch_add(1);
  phase_curl_replay_hits.fetch_add(1, std::memory_order_relaxed);
  if (plan.stable_reuses != std::numeric_limits<std::size_t>::max())
    ++plan.stable_reuses;
  collection.replayed = true;
  return true;
#else
  if (active_)
    throw std::runtime_error(
        "Meep CUDA phase-batched curl called in a CPU-only build");
  return false;
#endif
}

void resident_curl_phase_batch::finish() {
#if MEEP_HAVE_CUDA
  if (!active_) return;
  try {
    flush_pending_curl_phase();
    if (!pending_curl_phase.replayed)
      certify_complete_curl_phase_noexcept(pending_curl_phase);
  }
  catch (...) {
    curl_phase_collection &collection = pending_curl_phase;
    mark_curl_phase_plan_incomplete_noexcept(
        collection.owner, collection.phase_key);
    collection.operations.clear();
    collection.cache_dependencies.clear();
    collection.mirror_dependencies.clear();
    collection.writable_mirrors.clear();
    collection.active = false;
    collection.mode = phase_batch_mode::disabled;
    collection.owner = nullptr;
    collection.device_ordinal = -1;
    collection.point_count = 0;
    collection.flush_count = 0;
    collection.batched_flush_count = 0;
    collection.replayed = false;
    collection.replay_forbidden = false;
    active_ = false;
    throw;
  }
  pending_curl_phase.active = false;
  pending_curl_phase.mode = phase_batch_mode::disabled;
  pending_curl_phase.owner = nullptr;
  pending_curl_phase.device_ordinal = -1;
  pending_curl_phase.cache_dependencies.clear();
  pending_curl_phase.mirror_dependencies.clear();
  pending_curl_phase.writable_mirrors.clear();
  pending_curl_phase.flush_count = 0;
  pending_curl_phase.batched_flush_count = 0;
  pending_curl_phase.replayed = false;
  pending_curl_phase.replay_forbidden = false;
  active_ = false;
#else
  if (active_)
    throw std::runtime_error(
        "Meep CUDA phase-batched curl called in a CPU-only build");
#endif
}

void destroy_resident_curl_phase_batches_for_owner(
    const void *owner) noexcept {
#if MEEP_HAVE_CUDA
  clear_curl_phase_plans_for_owner(owner);
#else
  (void)owner;
#endif
}

resident_update_eh_phase_batch::resident_update_eh_phase_batch(
    const void *owner, int phase_key, phase_batch_mode mode)
    : active_(false) {
#if MEEP_HAVE_CUDA
  if (mode == phase_batch_mode::disabled) return;
  if (!owner)
    throw std::invalid_argument(
        "Meep CUDA phase-batched E/H owner must be non-null");
  update_eh_phase_collection &collection = pending_update_eh_phase;
  if (collection.active)
    throw std::logic_error(
        "nested Meep CUDA phase-batched E/H updates are not supported");
  collection.active = true;
  collection.mode = mode;
  collection.owner = owner;
  collection.phase_key = phase_key;
  collection.device_ordinal = -1;
  collection.point_count = 0;
  collection.operations.clear();
  collection.cache_dependencies.clear();
  active_ = true;
#else
  (void)owner;
  (void)phase_key;
  if (mode != phase_batch_mode::disabled)
    throw std::runtime_error(
        "Meep CUDA phase-batched E/H called in a CPU-only build");
#endif
}

resident_update_eh_phase_batch::~resident_update_eh_phase_batch() {
#if MEEP_HAVE_CUDA
  if (!active_) return;
  update_eh_phase_collection &collection = pending_update_eh_phase;
  collection.operations.clear();
  collection.cache_dependencies.clear();
  collection.active = false;
  collection.mode = phase_batch_mode::disabled;
  collection.owner = nullptr;
  collection.device_ordinal = -1;
  collection.point_count = 0;
#endif
}

void resident_update_eh_phase_batch::finish() {
#if MEEP_HAVE_CUDA
  if (!active_) return;
  try {
    flush_pending_update_eh_phase();
  }
  catch (...) {
    update_eh_phase_collection &collection = pending_update_eh_phase;
    collection.operations.clear();
    collection.cache_dependencies.clear();
    collection.active = false;
    collection.mode = phase_batch_mode::disabled;
    collection.owner = nullptr;
    collection.device_ordinal = -1;
    collection.point_count = 0;
    active_ = false;
    throw;
  }
  pending_update_eh_phase.active = false;
  pending_update_eh_phase.mode = phase_batch_mode::disabled;
  pending_update_eh_phase.owner = nullptr;
  pending_update_eh_phase.device_ordinal = -1;
  pending_update_eh_phase.cache_dependencies.clear();
  active_ = false;
#else
  if (active_)
    throw std::runtime_error(
        "Meep CUDA phase-batched E/H called in a CPU-only build");
#endif
}

void destroy_resident_update_eh_phase_batches_for_owner(
    const void *owner) noexcept {
#if MEEP_HAVE_CUDA
  clear_update_eh_phase_plans_for_owner(owner);
#else
  (void)owner;
#endif
}

resident_source_phase_batch::resident_source_phase_batch(
    const void *owner, int phase_key, bool enabled)
    : active_(false) {
#if MEEP_HAVE_CUDA
  if (!enabled) return;
  if (!owner)
    throw std::invalid_argument(
        "Meep CUDA phase-batched source owner must be non-null");
  source_phase_collection &collection = pending_source_phase;
  if (collection.active)
    throw std::logic_error(
        "nested Meep CUDA phase-batched sources are not supported");
  collection.active = true;
  collection.owner = owner;
  collection.phase_key = phase_key;
  collection.group_index = 0;
  collection.device_ordinal = -1;
  collection.point_count = 0;
  collection.operations.clear();
  collection.time_scales.clear();
  collection.cache_dependencies.clear();
  active_ = true;
#else
  (void)owner;
  (void)phase_key;
  if (enabled)
    throw std::runtime_error(
        "Meep CUDA phase-batched source called in a CPU-only build");
#endif
}

resident_source_phase_batch::~resident_source_phase_batch() {
#if MEEP_HAVE_CUDA
  if (!active_) return;
  source_phase_collection &collection = pending_source_phase;
  collection.operations.clear();
  collection.time_scales.clear();
  collection.cache_dependencies.clear();
  collection.active = false;
  collection.owner = nullptr;
  collection.group_index = 0;
  collection.device_ordinal = -1;
  collection.point_count = 0;
#endif
}

void resident_source_phase_batch::finish() {
#if MEEP_HAVE_CUDA
  if (!active_) return;
  const void *const owner = pending_source_phase.owner;
  const int phase_key = pending_source_phase.phase_key;
  try {
    flush_pending_source_phase();
    prune_source_phase_plans_after_group(
        owner, phase_key, pending_source_phase.group_index);
  }
  catch (...) {
    source_phase_collection &collection = pending_source_phase;
    collection.operations.clear();
    collection.time_scales.clear();
    collection.cache_dependencies.clear();
    collection.active = false;
    collection.owner = nullptr;
    collection.group_index = 0;
    collection.device_ordinal = -1;
    collection.point_count = 0;
    active_ = false;
    throw;
  }
  pending_source_phase.active = false;
  pending_source_phase.owner = nullptr;
  pending_source_phase.device_ordinal = -1;
  pending_source_phase.group_index = 0;
  pending_source_phase.time_scales.clear();
  pending_source_phase.cache_dependencies.clear();
  active_ = false;
#else
  if (active_)
    throw std::runtime_error(
        "Meep CUDA phase-batched source called in a CPU-only build");
#endif
}

void destroy_resident_source_phase_batches_for_owner(
    const void *owner) noexcept {
#if MEEP_HAVE_CUDA
  clear_source_phase_plans_for_owner(owner);
#else
  (void)owner;
#endif
}

bool resident_phase_is_active_for_owner(const void *owner) noexcept {
#if MEEP_HAVE_CUDA
  if (!owner) return false;
  resident_registry &registry = get_resident_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto found = registry.caches.find(owner);
  return found != registry.caches.end() && found->second->phase_active;
#else
  (void)owner;
  return false;
#endif
}

resident_cw_vector_plan *create_resident_cw_vector_plan(
    const cw_field_vector_segment_fp32 *segments,
    std::size_t segment_count, std::size_t complex_count) {
#if MEEP_HAVE_CUDA
  if (!segments || segment_count == 0 || complex_count == 0)
    throw std::invalid_argument(
        "Meep CUDA CW vector plan requires nonempty segments");
  if (segment_count > 65535)
    throw std::overflow_error(
        "Meep CUDA CW vector plan has too many segments");

  std::unique_ptr<resident_cw_vector_plan> plan(
      new resident_cw_vector_plan);
  std::vector<meep_cuda::cw_field_vector_segment_fp32> device_segments;
  device_segments.reserve(segment_count);
  std::size_t next_complex_offset = 0;

  for (std::size_t index = 0; index < segment_count; ++index) {
    const cw_field_vector_segment_fp32 &segment = segments[index];
    if (!segment.owner || !segment.field_real ||
        !segment.field_imaginary || segment.array_count == 0 ||
        segment.point_count == 0)
      throw std::invalid_argument(
          "Meep CUDA CW vector segment is incomplete");
    if (segment.field_real == segment.field_imaginary)
      throw std::invalid_argument(
          "Meep CUDA CW real and imaginary fields must not alias");
    const index_space_bounds bounds =
        validate_index_space(segment.index_space, segment.array_count);
    if (bounds.point_count != segment.point_count)
      throw std::invalid_argument(
          "Meep CUDA CW vector segment point count is inconsistent");
    if (segment.complex_offset != next_complex_offset ||
        next_complex_offset > complex_count ||
        segment.point_count > complex_count - next_complex_offset)
      throw std::invalid_argument(
          "Meep CUDA CW vector segments are not a contiguous partition");
    next_complex_offset += segment.point_count;
    plan->maximum_point_count =
        std::max(plan->maximum_point_count, segment.point_count);

    resident_cache *cache = resident_cache_for_owner(segment.owner);
    if (!cache->phase_active || cache->device_ordinal < 0)
      throw std::logic_error(
          "Meep CUDA CW vector plan requires an active resident phase");
    if (plan->device_ordinal < 0)
      plan->device_ordinal = cache->device_ordinal;
    else if (plan->device_ordinal != cache->device_ordinal)
      throw std::logic_error(
          "Meep CUDA CW vector plan spans multiple devices in one rank");

    const auto dependency = std::find_if(
        plan->cache_dependencies.begin(),
        plan->cache_dependencies.end(),
        [&segment](const resident_cw_vector_plan::cache_dependency &entry) {
          return entry.owner == segment.owner;
        });
    if (dependency == plan->cache_dependencies.end())
      plan->cache_dependencies.push_back({segment.owner, cache});
    else if (dependency->cache != cache)
      throw std::logic_error(
          "Meep CUDA CW owner resolved to inconsistent resident caches");

    const std::size_t array_bytes = checked_bytes(
        segment.array_count, sizeof(float), "CW field mirror");
    float *device_real = static_cast<float *>(ensure_resident_mirror(
        cache, segment.field_real, array_bytes));
    float *device_imaginary = static_cast<float *>(ensure_resident_mirror(
        cache, segment.field_imaginary, array_bytes));
    plan->mirror_dependencies.push_back(
        {cache, segment.field_real, device_real, array_bytes});
    plan->mirror_dependencies.push_back(
        {cache, segment.field_imaginary, device_imaginary, array_bytes});
    device_segments.push_back(
        {device_real, device_imaginary,
         device_index_space(segment.index_space), segment.point_count,
         segment.complex_offset});
    const std::pair<resident_cache *, float *> real_destination(
        cache, segment.field_real);
    const std::pair<resident_cache *, float *> imaginary_destination(
        cache, segment.field_imaginary);
    if (std::find(plan->writable_mirrors.begin(),
                  plan->writable_mirrors.end(),
                  real_destination) == plan->writable_mirrors.end())
      plan->writable_mirrors.push_back(real_destination);
    if (std::find(plan->writable_mirrors.begin(),
                  plan->writable_mirrors.end(),
                  imaginary_destination) == plan->writable_mirrors.end())
      plan->writable_mirrors.push_back(imaginary_destination);
  }
  if (next_complex_offset != complex_count)
    throw std::invalid_argument(
        "Meep CUDA CW vector segments do not cover the packed vector");

  meep_cuda::select_device(plan->device_ordinal);
  const std::size_t descriptor_bytes = checked_bytes(
      device_segments.size(),
      sizeof(meep_cuda::cw_field_vector_segment_fp32),
      "CW field-vector descriptors");
  plan->device_segments = meep_cuda::allocate_device_bytes(
      descriptor_bytes);
  try {
    meep_cuda::copy_to_device(
        plan->device_segments, device_segments.data(), descriptor_bytes);
  }
  catch (...) {
    meep_cuda::free_device(plan->device_segments);
    plan->device_segments = nullptr;
    throw;
  }
  device_buffer_allocations.fetch_add(1);
  live_device_buffer_count().fetch_add(1);
  host_to_device_bytes.fetch_add(
      static_cast<std::uint64_t>(descriptor_bytes));
  plan->segment_count = segment_count;
  plan->complex_count = complex_count;
  return plan.release();
#else
  (void)segments;
  (void)segment_count;
  (void)complex_count;
  throw std::runtime_error(
      "Meep CUDA CW vector plan called in a CPU-only build");
#endif
}

namespace {

#if MEEP_HAVE_CUDA
void validate_resident_cw_vector_plan(resident_cw_vector_plan *plan) {
  if (!plan || !plan->device_segments || plan->segment_count == 0 ||
      plan->maximum_point_count == 0 || plan->complex_count == 0 ||
      plan->device_ordinal < 0)
    throw std::invalid_argument(
        "Meep CUDA CW vector plan is incomplete");
  resident_registry &registry = get_resident_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  for (const auto &dependency : plan->cache_dependencies) {
    const auto found = registry.caches.find(dependency.owner);
    if (found == registry.caches.end() ||
        found->second != dependency.cache ||
        !dependency.cache->phase_active ||
        dependency.cache->device_ordinal != plan->device_ordinal)
      throw std::logic_error(
          "Meep CUDA CW vector plan was invalidated during the solve");
  }
  for (const auto &dependency : plan->mirror_dependencies) {
    const auto found =
        dependency.cache->mirrors.find(dependency.host_pointer);
    if (found == dependency.cache->mirrors.end() ||
        found->second.device_pointer != dependency.device_pointer ||
        found->second.bytes != dependency.bytes)
      throw std::logic_error(
          "Meep CUDA CW field mirror changed during the solve");
  }
  meep_cuda::select_device(plan->device_ordinal);
}
#endif

} // namespace

void destroy_resident_cw_vector_plan(
    resident_cw_vector_plan *plan) noexcept {
  if (!plan) return;
#if MEEP_HAVE_CUDA
  if (plan->device_ordinal >= 0) {
    try {
      meep_cuda::select_device(plan->device_ordinal);
    }
    catch (...) {}
  }
  if (plan->device_segments) {
    meep_cuda::free_device(plan->device_segments);
    live_device_buffer_count().fetch_sub(1);
  }
#endif
  delete plan;
}

void gather_resident_cw_vector_fp32(
    resident_cw_vector_plan *plan, float *device_packed_real_imag) {
#if MEEP_HAVE_CUDA
  if (!device_packed_real_imag)
    throw std::invalid_argument(
        "Meep CUDA CW gather output must be non-null");
  validate_resident_cw_vector_plan(plan);
  meep_cuda::gather_cw_field_vector_fp32(
      static_cast<const meep_cuda::cw_field_vector_segment_fp32 *>(
          plan->device_segments),
      plan->segment_count, plan->maximum_point_count,
      device_packed_real_imag);
#else
  (void)plan;
  (void)device_packed_real_imag;
  throw std::runtime_error(
      "Meep CUDA CW gather called in a CPU-only build");
#endif
}

void scatter_resident_cw_vector_fp32(
    resident_cw_vector_plan *plan,
    const float *device_packed_real_imag) {
#if MEEP_HAVE_CUDA
  if (!device_packed_real_imag)
    throw std::invalid_argument(
        "Meep CUDA CW scatter input must be non-null");
  validate_resident_cw_vector_plan(plan);
  for (const auto &destination : plan->writable_mirrors)
    (void)resident_device_address(
        destination.first, destination.second, true);
  meep_cuda::scatter_cw_field_vector_fp32(
      static_cast<const meep_cuda::cw_field_vector_segment_fp32 *>(
          plan->device_segments),
      plan->segment_count, plan->maximum_point_count,
      device_packed_real_imag);
#else
  (void)plan;
  (void)device_packed_real_imag;
  throw std::runtime_error(
      "Meep CUDA CW scatter called in a CPU-only build");
#endif
}

void gather_resident_cw_field_operator_fp32(
    resident_cw_vector_plan *plan,
    const float *device_input_real_imag,
    float *device_output_real_imag, float dt_inverse,
    complex_value_fp32 iomega) {
#if MEEP_HAVE_CUDA
  if (!device_input_real_imag || !device_output_real_imag)
    throw std::invalid_argument(
        "Meep CUDA CW field operator vectors must be non-null");
  validate_resident_cw_vector_plan(plan);
  meep_cuda::gather_cw_field_operator_fp32(
      static_cast<const meep_cuda::cw_field_vector_segment_fp32 *>(
          plan->device_segments),
      plan->segment_count, plan->maximum_point_count,
      device_input_real_imag, device_output_real_imag, dt_inverse,
      iomega.real, iomega.imag);
#else
  (void)plan;
  (void)device_input_real_imag;
  (void)device_output_real_imag;
  (void)dt_inverse;
  (void)iomega;
  throw std::runtime_error(
      "Meep CUDA CW field operator called in a CPU-only build");
#endif
}

namespace {

void destroy_resident_cache_for_owner_impl(const void *owner,
                                           bool publish_to_host) noexcept {
#if MEEP_HAVE_CUDA
  // Result-cache LDOS plans may retain device pointers owned by the cache
  // being removed.  Exclude reductions for the complete remove/invalidate/
  // free transaction, always before taking the resident registry mutex.
  std::lock_guard<std::mutex> ldos_lock(get_ldos_reduction_mutex());
  resident_cache *cache = remove_resident_cache_for_owner(owner);
  if (!cache) return;
  {
    resident_registry &registry = get_resident_registry();
    std::lock_guard<std::mutex> registry_lock(registry.mutex);
    invalidate_all_ldos_plan_snapshots_locked(registry);
  }
  // Boundary operation and remote gather/scatter plans keep cache pointers
  // only as non-owning descriptor identities. Drop every such plan while the
  // cache is still alive so later generation validation can never dereference
  // a destroyed cache after zero_fields(), source repair, initialization, or
  // another host-side lifecycle transition.
  if (cache->device_ordinal >= 0) {
    try {
      meep_cuda::select_device(cache->device_ordinal);
      if (publish_to_host) sync_resident_mirrors_to_host(cache);
    }
    catch (...) {}
  }
  // Invalidate only persistent plans which captured this cache.  Other live
  // fields owners retain their boundary/curl/EH plans and device residency;
  // an owner choosing CPU must never flush an unrelated CUDA simulation.
  clear_boundary_exchange_plans_for_cache(cache);
  clear_curl_phase_plans_for_cache(cache);
  clear_update_eh_phase_plans_for_cache(cache);
  clear_source_phase_plans_for_cache(cache);
  clear_dft_reduction_plans_for_cache(cache);
  clear_eigenmode_overlap_plans_for_cache(cache);
  clear_near2far_plans_for_cache(cache);
  clear_dft_materialization_plans_for_cache(cache);
  cache->phase_active = false;
  // free_device is noexcept and the live-buffer bookkeeping must run even if
  // selecting a device failed (for example during CUDA runtime teardown).
  clear_device_buffers(cache);
  delete cache;
#else
  (void)owner;
  (void)publish_to_host;
#endif
}

} // namespace

resident_cache_snapshot_for_testing get_resident_cache_snapshot_for_testing(
    const void *owner) noexcept {
  resident_cache_snapshot_for_testing snapshot = {
      false, false, -1, 0, 0};
#if MEEP_HAVE_CUDA
  if (!owner) return snapshot;
  resident_registry &registry = get_resident_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto found = registry.caches.find(owner);
  if (found == registry.caches.end() || !found->second) return snapshot;
  snapshot.exists = true;
  snapshot.phase_active = found->second->phase_active;
  snapshot.device_ordinal = found->second->device_ordinal;
  snapshot.epoch = found->second->epoch;
  snapshot.mirror_count = found->second->mirrors.size();
#else
  (void)owner;
#endif
  return snapshot;
}

void destroy_resident_cache_for_owner(const void *owner) noexcept {
  destroy_resident_cache_for_owner_impl(owner, true);
}

void discard_resident_cache_for_owner(const void *owner) noexcept {
  destroy_resident_cache_for_owner_impl(owner, false);
}

void sync_resident_cache_for_owner(const void *owner) {
#if MEEP_HAVE_CUDA
  resident_cache *cache = nullptr;
  {
    resident_registry &registry = get_resident_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.caches.find(owner);
    if (found != registry.caches.end()) cache = found->second;
  }
  if (!cache || cache->device_ordinal < 0) return;
  if (cache->phase_active)
    throw std::logic_error(
        "cannot synchronize a Meep CUDA cache during an active resident "
        "phase");
  meep_cuda::select_device(cache->device_ordinal);
  sync_resident_mirrors_to_host(cache);
#else
  (void)owner;
#endif
}

void prepare_resident_host_writes_for_owner(
    const void *owner, const void *const *host_pointers,
    std::size_t pointer_count) {
#if MEEP_HAVE_CUDA
  if (!owner || pointer_count == 0) return;
  if (!host_pointers)
    throw std::invalid_argument(
        "Meep CUDA host-write pointer list must be non-null");
  resident_registry &registry = get_resident_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto cache_entry = registry.caches.find(owner);
  if (cache_entry == registry.caches.end()) return;
  resident_cache *cache = cache_entry->second;
  if (cache->phase_active)
    throw std::logic_error(
        "cannot prepare host writes during an active Meep CUDA phase");

  // Resolve and validate the complete set before changing any epoch.  This
  // makes a rejected publication transactional even when a later pointer in
  // the list aliases a device-authoritative mirror.
  std::vector<resident_cache::mirror *> mirrors;
  mirrors.reserve(pointer_count);
  for (std::size_t index = 0; index < pointer_count; ++index) {
    if (!host_pointers[index]) continue;
    const resolved_resident_range resolved = resolve_resident_range(
        cache, static_cast<const float *>(host_pointers[index]),
        sizeof(float), true, "Meep CUDA host-write pointer");
    if (!resolved.mirror) continue;
    if (resolved.mirror->device_dirty)
      throw std::logic_error(
          "cannot overwrite a device-authoritative Meep CUDA mirror");
    if (std::find(mirrors.begin(), mirrors.end(), resolved.mirror) ==
        mirrors.end())
      mirrors.push_back(resolved.mirror);
  }
  for (resident_cache::mirror *mirror : mirrors)
    mirror->uploaded_epoch = 0;
#else
  (void)owner;
  (void)host_pointers;
  (void)pointer_count;
#endif
}

void reset_resident_cache_for_owner_reusing_allocations(
    const void *owner) {
#if MEEP_HAVE_CUDA
  std::lock_guard<std::mutex> ldos_lock(get_ldos_reduction_mutex());
  resident_registry &registry = get_resident_registry();
  std::lock_guard<std::mutex> registry_lock(registry.mutex);
  const auto found = registry.caches.find(owner);
  resident_cache *const cache =
      found == registry.caches.end() ? nullptr : found->second;
  if (!cache) return;
  if (cache->phase_active)
    throw std::logic_error(
        "cannot reset a Meep CUDA cache during an active resident "
        "phase");
  if (cache->device_ordinal >= 0) {
    meep_cuda::select_device(cache->device_ordinal);
    sync_resident_mirrors_to_host(cache);
  }

  // Boundary and structured descriptors carry host/device addresses. Rebuild
  // those plans exactly as a full cache destruction would, while pooling only
  // the plain mirrors whose byte sizes are sufficient for safe reuse.
  clear_all_boundary_exchange_buffers();
  clear_all_curl_phase_plans();
  clear_all_update_eh_phase_plans();
  clear_all_source_phase_plans();
  clear_dft_reduction_plans_for_cache(cache);
  clear_eigenmode_overlap_plans_for_cache(cache);
  clear_near2far_plans_for_cache(cache);
  cache->recycled_mirror_allocations.reserve(
      cache->recycled_mirror_allocations.size() +
      cache->mirrors.size());
  for (auto &entry : cache->mirrors)
    cache->recycled_mirror_allocations.push_back(
        std::make_pair(entry.second.bytes,
                       entry.second.device_pointer));
  cache->mirrors.clear();
  cache->mirror_intervals.clear();
  cache->validated_source_profiles.clear();
  clear_dft_materialization_plans_for_cache_locked(registry, cache);
  invalidate_all_ldos_plan_snapshots_locked(registry);

  std::uint64_t released = 0;
  for (auto &entry : cache->structured_curl_batches) {
    if (entry.second.device_spaces) ++released;
    meep_cuda::free_device(entry.second.device_spaces);
  }
  cache->structured_curl_batches.clear();
  for (auto &entry : cache->multilevel_plans) {
    if (entry.second.device_population_channels) ++released;
    if (entry.second.device_polarization_channels) ++released;
    if (entry.second.device_transitions) ++released;
    meep_cuda::free_device(entry.second.device_population_channels);
    meep_cuda::free_device(entry.second.device_polarization_channels);
    meep_cuda::free_device(entry.second.device_transitions);
  }
  cache->multilevel_plans.clear();
  if (cache->finite_check.device_spans) {
    ++released;
    meep_cuda::free_device(cache->finite_check.device_spans);
  }
  cache->finite_check.device_spans = nullptr;
  cache->finite_check.entries.clear();
  cache->finite_check.maximum_span_count = 0;
  cache->finite_check.allocation_generation = 0;
  // finite_result is deliberately retained across this allocation-pooling
  // reset.  Preserve its pending generation as well: finish(false) promises
  // that a previously observed NaN/Inf remains sticky until a later readback,
  // even if address-bearing descriptor plans are rebuilt in between. The
  // scalar LDOS reduction workspace and both bounded LDOS plan buffers
  // are likewise retained. Their address-bearing snapshot is invalid, so the
  // next reduction rewrites only the inactive slot before committing it.
  live_device_buffer_count().fetch_sub(released);
  cache->allocation_generation =
      detail::next_resident_allocation_generation();
#else
  (void)owner;
#endif
}

unsigned int read_resident_field_point_fp32_for_owner(
    const void *owner, const float *real_host_base,
    const float *imaginary_host_base, std::size_t index,
    float *real_value, float *imaginary_value) {
#if MEEP_HAVE_CUDA
  if (!owner) return 0;

  resident_registry &registry = get_resident_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto cache_entry = registry.caches.find(owner);
  if (cache_entry == registry.caches.end()) return 0;
  resident_cache *cache = cache_entry->second;

  const auto real_mirror =
      real_host_base ? cache->mirrors.find(real_host_base)
                     : cache->mirrors.end();
  const auto imaginary_mirror =
      imaginary_host_base ? cache->mirrors.find(imaginary_host_base)
                          : cache->mirrors.end();
  const bool real_dirty =
      real_mirror != cache->mirrors.end() &&
      real_mirror->second.device_dirty;
  const bool imaginary_dirty =
      imaginary_mirror != cache->mirrors.end() &&
      imaginary_mirror->second.device_dirty;
  if (!real_dirty && !imaginary_dirty) return 0;
  if ((real_dirty && !real_value) ||
      (imaginary_dirty && !imaginary_value))
    throw std::invalid_argument(
        "Meep CUDA resident scalar destination must be non-null");
  if (cache->device_ordinal < 0)
    throw std::logic_error(
        "Meep CUDA dirty resident scalar has no device assignment");

  auto checked_device_scalar =
      [index](const resident_cache::mirror &mirror) -> const void * {
    if (mirror.bytes % sizeof(float))
      throw std::logic_error(
          "Meep CUDA FP32 resident mirror has a partial scalar");
    const std::size_t scalar_count = mirror.bytes / sizeof(float);
    if (index >= scalar_count)
      throw std::out_of_range(
          "Meep CUDA resident scalar index is outside its mirror");
    const std::size_t byte_offset = index * sizeof(float);
    return static_cast<const unsigned char *>(mirror.device_pointer) +
           byte_offset;
  };

  // Validate both addresses before initiating either transfer, so even a
  // malformed shorter imaginary mirror cannot leave a partial read/counter.
  const void *real_device_scalar =
      real_dirty ? checked_device_scalar(real_mirror->second) : nullptr;
  const void *imaginary_device_scalar =
      imaginary_dirty
          ? checked_device_scalar(imaginary_mirror->second)
          : nullptr;

  // cudaMemcpy D2H is synchronous.  Select the cache's rank-local device so
  // this remains correct when different MPI processes own different GPUs.
  meep_cuda::select_device(cache->device_ordinal);
  float staged_real = 0.0f;
  float staged_imaginary = 0.0f;
  if (real_dirty) {
    meep_cuda::copy_to_host(
        &staged_real, real_device_scalar, sizeof(staged_real));
    device_to_host_bytes.fetch_add(sizeof(staged_real));
  }
  if (imaginary_dirty) {
    meep_cuda::copy_to_host(
        &staged_imaginary, imaginary_device_scalar,
        sizeof(staged_imaginary));
    device_to_host_bytes.fetch_add(sizeof(staged_imaginary));
  }

  unsigned int copied = 0;
  if (real_dirty) {
    *real_value = staged_real;
    copied |= 1u;
  }
  if (imaginary_dirty) {
    *imaginary_value = staged_imaginary;
    copied |= 2u;
  }
  // In particular, do not clear device_dirty or write either host array:
  // the full resident mirror remains authoritative for subsequent steps.
  return copied;
#else
  (void)owner;
  (void)real_host_base;
  (void)imaginary_host_base;
  (void)index;
  (void)real_value;
  (void)imaginary_value;
  return 0;
#endif
}

resident_range_readback stage_resident_range_fp32_for_owner(
    const void *owner, const float *host_pointer, std::size_t count,
    float *destination) {
#if MEEP_HAVE_CUDA
  if (count == 0) return {false, 0};
  if (!owner || !host_pointer || !destination)
    throw std::invalid_argument(
        "Meep CUDA resident checkpoint readback requires non-null "
        "pointers");
  const std::size_t bytes = checked_bytes(
      count, sizeof(float), "DFT checkpoint readback");

  resident_registry &registry = get_resident_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto cache_entry = registry.caches.find(owner);
  if (cache_entry == registry.caches.end()) {
    std::memcpy(destination, host_pointer, bytes);
    return {false, 0};
  }
  resident_cache *cache = cache_entry->second;
  if (cache->phase_active)
    throw std::logic_error(
        "cannot stage a DFT checkpoint during an active Meep CUDA phase");
  const resolved_resident_range resolved = resolve_resident_range(
      cache, host_pointer, bytes, true,
      "Meep CUDA DFT checkpoint range");
  if (!resolved.mirror) {
    std::memcpy(destination, host_pointer, bytes);
    return {false, 0};
  }

  std::size_t transferred = 0;
  if (resolved.mirror->device_dirty) {
    if (consume_countdown_failure_for_testing(
            dft_checkpoint_d2h_failure_after_for_testing))
      throw std::runtime_error(
          "injected CUDA DFT checkpoint D2H failure");
    meep_cuda::select_device(cache->device_ordinal);
    const unsigned char *device_source =
        static_cast<const unsigned char *>(
            resolved.mirror->device_pointer) + resolved.byte_offset;
    meep_cuda::copy_to_host(destination, device_source, bytes);
    device_to_host_bytes.fetch_add(
        static_cast<std::uint64_t>(bytes), std::memory_order_relaxed);
    transferred = bytes;
  }
  else {
    std::memcpy(destination, host_pointer, bytes);
  }
  // Keep the complete mirror device-authoritative.  The staging buffer is a
  // point-in-time HDF5 payload, not a publication of the host field array.
  return {true, transferred};
#else
  if (count && (!host_pointer || !destination))
    throw std::invalid_argument(
        "Meep checkpoint readback requires non-null pointers");
  if (count)
    std::memcpy(destination, host_pointer, count * sizeof(float));
  (void)owner;
  return {false, 0};
#endif
}

resident_checkpoint_traffic summarize_resident_checkpoint_ranges_fp32(
    const resident_checkpoint_range_fp32 *ranges,
    std::size_t range_count) {
#if MEEP_HAVE_CUDA
  if (!ranges && range_count)
    throw std::invalid_argument(
        "Meep CUDA checkpoint range list must be non-null");
  resident_registry &registry = get_resident_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  std::set<resident_cache *, std::less<resident_cache *> > owners;
  using mirror_key =
      std::pair<resident_cache *, const float *>;
  std::map<mirror_key,
           std::vector<std::pair<std::size_t, std::size_t> > > intervals;
  std::size_t dirty_resident_bytes = 0;
  for (std::size_t index = 0; index < range_count; ++index) {
    const resident_checkpoint_range_fp32 &range = ranges[index];
    if (range.count == 0) continue;
    if (!range.owner || !range.host_pointer)
      throw std::invalid_argument(
          "Meep CUDA checkpoint range requires non-null pointers");
    const std::size_t bytes = checked_bytes(
        range.count, sizeof(float), "DFT checkpoint range summary");
    const auto cache_entry = registry.caches.find(range.owner);
    if (cache_entry == registry.caches.end()) continue;
    resident_cache *const cache = cache_entry->second;
    if (cache->phase_active)
      throw std::logic_error(
          "cannot summarize a DFT checkpoint during an active Meep CUDA "
          "phase");
    if (owners.insert(cache).second)
      for (const auto &entry : cache->mirrors) {
        if (!entry.second.device_dirty) continue;
        if (entry.second.bytes >
            std::numeric_limits<std::size_t>::max() -
                dirty_resident_bytes)
          throw std::overflow_error(
              "Meep CUDA checkpoint dirty resident byte count overflow");
        dirty_resident_bytes += entry.second.bytes;
      }
    const resolved_resident_range resolved = resolve_resident_range(
        cache, range.host_pointer, bytes, true,
        "Meep CUDA DFT checkpoint summary range");
    if (!resolved.mirror || !resolved.mirror->device_dirty) continue;
    const std::size_t begin =
        static_cast<std::size_t>(resolved.byte_offset);
    if (begin > std::numeric_limits<std::size_t>::max() - bytes)
      throw std::overflow_error(
          "Meep CUDA checkpoint interval end overflow");
    intervals[mirror_key(cache, resolved.host_base)].push_back(
        std::make_pair(begin, begin + bytes));
  }

  std::size_t staged_dirty_union_bytes = 0;
  for (auto &entry : intervals) {
    auto &spans = entry.second;
    std::sort(spans.begin(), spans.end());
    std::size_t begin = spans.front().first;
    std::size_t end = spans.front().second;
    for (std::size_t index = 1; index < spans.size(); ++index) {
      if (spans[index].first <= end) {
        end = std::max(end, spans[index].second);
        continue;
      }
      const std::size_t span = end - begin;
      if (span > std::numeric_limits<std::size_t>::max() -
                     staged_dirty_union_bytes)
        throw std::overflow_error(
            "Meep CUDA checkpoint staged dirty byte count overflow");
      staged_dirty_union_bytes += span;
      begin = spans[index].first;
      end = spans[index].second;
    }
    const std::size_t span = end - begin;
    if (span > std::numeric_limits<std::size_t>::max() -
                   staged_dirty_union_bytes)
      throw std::overflow_error(
          "Meep CUDA checkpoint staged dirty byte count overflow");
    staged_dirty_union_bytes += span;
  }
  if (staged_dirty_union_bytes > dirty_resident_bytes)
    throw std::logic_error(
        "Meep CUDA checkpoint staged bytes exceed dirty resident bytes");
  return {dirty_resident_bytes, staged_dirty_union_bytes,
          dirty_resident_bytes - staged_dirty_union_bytes};
#else
  if (!ranges && range_count)
    throw std::invalid_argument(
        "Meep checkpoint range list must be non-null");
  (void)ranges;
  (void)range_count;
  return {0, 0, 0};
#endif
}

resident_range_upload upload_resident_range_fp32_for_owner(
    const void *owner, float *host_pointer, std::size_t count) {
#if MEEP_HAVE_CUDA
  if (count == 0) return {true, 0};
  if (!owner || !host_pointer)
    throw std::invalid_argument(
        "Meep CUDA resident checkpoint upload requires non-null pointers");
  const std::size_t bytes = checked_bytes(
      count, sizeof(float), "DFT checkpoint upload");
  resident_cache *cache = resident_cache_for_owner(owner);
  if (cache->phase_active)
    throw std::logic_error(
        "cannot upload a DFT checkpoint during an active Meep CUDA phase");
  select_cache_device(cache);
  std::atomic<std::uint64_t> operation_h2d_bytes(0);
  try {
    (void)ensure_resident_mirror(
        cache, host_pointer, bytes, &operation_h2d_bytes, true,
        &dft_checkpoint_h2d_failure_after_for_testing);
  }
  catch (...) {
    // ensure_resident_mirror registers a newly allocated mirror before the
    // H2D copy.  Roll it back on either injected or real CUDA copy failure so
    // a failed checkpoint leaves this range matching the committed host data
    // or absent, never partially initialized.
    discard_resident_mirror_for_owner(owner, host_pointer);
    throw;
  }
  return {true, static_cast<std::size_t>(
                    operation_h2d_bytes.load(std::memory_order_relaxed))};
#else
  (void)owner;
  (void)host_pointer;
  (void)count;
  return {false, 0};
#endif
}

resident_dft_scale_result scale_resident_complex_dft_fp32_for_owner(
    const void *owner, float *host_real_imag, std::size_t count,
    double scale_real, double scale_imaginary) {
#if MEEP_HAVE_CUDA
  if (!std::isfinite(scale_real) || !std::isfinite(scale_imaginary))
    throw std::invalid_argument(
        "Meep CUDA DFT scale must be finite");
  if (count == 0) return {0, 0};
  if (!owner || !host_real_imag)
    throw std::invalid_argument(
        "Meep CUDA DFT scale requires non-null pointers");
  if (count > std::numeric_limits<std::size_t>::max() / 2)
    throw std::overflow_error("Meep CUDA DFT scale scalar count overflow");
  const std::size_t bytes = checked_bytes(
      2 * count, sizeof(float), "DFT complex scale");
  resident_cache *cache = resident_cache_for_owner(owner);
  if (cache->phase_active)
    throw std::logic_error(
        "cannot scale a DFT during an active Meep CUDA phase");
  select_cache_device(cache);
  const resolved_resident_range existing = resolve_resident_range(
      cache, host_real_imag, bytes, true,
      "Meep CUDA DFT scale range");
  if (existing.mirror &&
      (existing.host_base != host_real_imag || existing.byte_offset != 0 ||
       existing.mirror->bytes != bytes))
    throw std::invalid_argument(
        "Meep CUDA DFT scale must cover one complete resident mirror");
  std::atomic<std::uint64_t> operation_h2d_bytes(0);
  float *device = static_cast<float *>(ensure_resident_mirror(
      cache, host_real_imag, bytes, &operation_h2d_bytes));
  meep_cuda::complex_scale_inplace_fp32(
      device, count, scale_real, scale_imaginary);
  mark_device_dirty(cache, host_real_imag, bytes);

  // Reduction and Near2Far descriptors contain this address and would still
  // execute correctly, but their immutable snapshots describe the pre-scale
  // scientific state.  Drop them explicitly, and also drop materialization's
  // cached output, so every dependent consumer recomputes from the scaled
  // resident values.
  clear_dft_reduction_plans_for_cache(cache);
  clear_eigenmode_overlap_plans_for_cache(cache);
  clear_near2far_plans_for_cache(cache);
  clear_dft_materialization_plans_for_cache(cache);
  return {static_cast<std::size_t>(
              operation_h2d_bytes.load(std::memory_order_relaxed)),
          1};
#else
  (void)owner;
  (void)host_real_imag;
  (void)count;
  (void)scale_real;
  (void)scale_imaginary;
  throw std::runtime_error(
      "Meep CUDA DFT scale called in a CPU-only build");
#endif
}

void discard_resident_mirror_for_owner(const void *owner,
                                       const void *host_pointer) noexcept {
#if MEEP_HAVE_CUDA
  if (!host_pointer) return;
  std::lock_guard<std::mutex> ldos_lock(get_ldos_reduction_mutex());
  resident_registry &registry = get_resident_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto cache_entry = registry.caches.find(owner);
  if (cache_entry == registry.caches.end()) return;
  resident_cache *cache = cache_entry->second;
  const auto mirror =
      cache->mirrors.find(static_cast<const float *>(host_pointer));
  if (mirror == cache->mirrors.end()) return;
  // Any result cache may retain this device address as a nested LDOS
  // descriptor pointer, including when the removed buffer is later recycled
  // at the same address (the ABA case).
  invalidate_all_ldos_plan_snapshots_locked(registry);
  clear_dft_reduction_plans_for_cache(cache);
  clear_eigenmode_overlap_plans_for_cache(cache);
  clear_near2far_plans_for_cache(cache);
  clear_dft_materialization_plans_for_cache_locked(registry, cache);
  if (cache->device_ordinal >= 0) {
    try {
      meep_cuda::select_device(cache->device_ordinal);
    }
    catch (...) {}
  }
  meep_cuda::free_device(mirror->second.device_pointer);
  cache->mirror_intervals.erase(
      reinterpret_cast<std::uintptr_t>(host_pointer));
  cache->mirrors.erase(mirror);
  cache->allocation_generation =
      detail::next_resident_allocation_generation();
  live_device_buffer_count().fetch_sub(1);
#else
  (void)owner;
  (void)host_pointer;
#endif
}

void discard_resident_source_profile_for_owner(
    const void *owner, const std::ptrdiff_t *indices,
    const float *amplitudes) {
#if MEEP_HAVE_CUDA
  if (!owner || (!indices && !amplitudes)) return;
  std::lock_guard<std::mutex> ldos_lock(get_ldos_reduction_mutex());
  resident_registry &registry = get_resident_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto cache_entry = registry.caches.find(owner);
  if (cache_entry == registry.caches.end()) return;
  resident_cache *cache = cache_entry->second;
  if (cache->phase_active)
    throw std::logic_error(
        "cannot replace a Meep CUDA source profile during an active "
        "resident phase");

  const float *const index_key =
      reinterpret_cast<const float *>(indices);
  const float *const amplitude_key = amplitudes;
  const auto index_mirror = index_key
                                ? cache->mirrors.find(index_key)
                                : cache->mirrors.end();
  const auto amplitude_mirror =
      amplitude_key ? cache->mirrors.find(amplitude_key)
                    : cache->mirrors.end();
  if ((index_mirror != cache->mirrors.end() &&
       index_mirror->second.device_dirty) ||
      (amplitude_mirror != cache->mirrors.end() &&
       amplitude_mirror->second.device_dirty))
    throw std::logic_error(
        "Meep CUDA source profile unexpectedly became device-authoritative");

  if ((index_mirror != cache->mirrors.end() ||
       amplitude_mirror != cache->mirrors.end()) &&
      cache->device_ordinal >= 0)
    meep_cuda::select_device(cache->device_ordinal);

  // Persistent source-phase and LDOS descriptors may retain nested pointers
  // to either profile mirror. Invalidate only their host identity snapshots;
  // retained device storage is rewritten transactionally on the next launch.
  invalidate_source_phase_plan_snapshots_for_cache(cache);
  invalidate_all_ldos_plan_snapshots_locked(registry);

  // Bounds validation is keyed by the host index allocation and must not
  // survive removal even when no device mirror was created.
  if (indices) cache->validated_source_profiles.erase(indices);
  const auto recycle_clean_mirror = [cache](const float *key) {
    if (!key) return;
    const auto mirror = cache->mirrors.find(key);
    if (mirror == cache->mirrors.end()) return;
    // Retain enough exact-size buffers for common multi-profile replacement,
    // but bound both count and bytes for applications that change a volume
    // source's extent indefinitely. Oversized profiles are freed immediately.
    static constexpr std::size_t maximum_recycled_buffers = 8;
    static constexpr std::size_t maximum_recycled_bytes =
        64u * 1024u * 1024u;
    auto &pool = cache->recycled_source_profile_allocations;
    const std::size_t bytes = mirror->second.bytes;
    if (bytes <= maximum_recycled_bytes) {
      std::size_t pooled_bytes = 0;
      for (const auto &entry : pool) pooled_bytes += entry.first;
      while (!pool.empty() &&
             (pool.size() >= maximum_recycled_buffers ||
              pooled_bytes > maximum_recycled_bytes - bytes)) {
        pooled_bytes -= pool.front().first;
        meep_cuda::free_device(pool.front().second);
        pool.erase(pool.begin());
        live_device_buffer_count().fetch_sub(1);
      }
      pool.push_back(
          std::make_pair(bytes, mirror->second.device_pointer));
    }
    else {
      meep_cuda::free_device(mirror->second.device_pointer);
      live_device_buffer_count().fetch_sub(1);
    }
    cache->mirror_intervals.erase(
        reinterpret_cast<std::uintptr_t>(key));
    cache->mirrors.erase(mirror);
  };
  recycle_clean_mirror(index_key);
  if (amplitude_key != index_key)
    recycle_clean_mirror(amplitude_key);

  // Unrelated curl, E/H, finite-check, DFT, and boundary plans never capture
  // source-profile mirrors. Keep allocation_generation unchanged so those
  // plans remain reusable while the invalidated source/LDOS descriptors are
  // refreshed against the replacement profiles.
#else
  (void)owner;
  (void)indices;
  (void)amplitudes;
#endif
}

void destroy_boundary_exchange_for_owner(const void *owner) noexcept {
#if MEEP_HAVE_CUDA
  if (!owner) return;
  boundary_exchange_registry &registry = get_boundary_exchange_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto found = registry.owners.find(owner);
  if (found == registry.owners.end()) return;
  for (auto &entry : found->second.buffers)
    release_boundary_exchange_buffer(entry.second);
  for (auto &entry : found->second.operation_plans)
    release_boundary_operation_plan(entry.second);
  registry.owners.erase(found);
#else
  (void)owner;
#endif
}

boundary_exchange_buffer *resident_boundary_exchange_buffer(
    const void *owner, const void *token, std::size_t scalar_count) {
  return resident_boundary_exchange_buffer_slot(
      owner, token, scalar_count, 0);
}

boundary_exchange_buffer *resident_boundary_exchange_buffer_slot(
    const void *owner, const void *token, std::size_t scalar_count,
    std::size_t slot) {
#if MEEP_HAVE_CUDA
  if (!owner || !token)
    throw std::invalid_argument(
        "Meep CUDA boundary exchange owner and token must be non-null");
  if (scalar_count == 0)
    throw std::invalid_argument(
        "Meep CUDA boundary exchange size must be nonzero");
  const int ordinal = active_cuda_device_ordinal();
  const std::size_t bytes =
      checked_bytes(scalar_count, sizeof(float), "boundary exchange");

  boundary_exchange_registry &registry = get_boundary_exchange_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  boundary_exchange_owner &owner_buffers = registry.owners[owner];
  const boundary_exchange_key key = {token, slot};
  auto found = owner_buffers.buffers.find(key);
  if (found != owner_buffers.buffers.end() &&
      (found->second->scalar_count != scalar_count ||
       found->second->device_ordinal != ordinal)) {
    release_boundary_exchange_buffer(found->second);
    owner_buffers.buffers.erase(found);
    meep_cuda::select_device(ordinal);
    found = owner_buffers.buffers.end();
  }
  if (found != owner_buffers.buffers.end()) {
    device_buffer_reuses.fetch_add(1);
    return found->second;
  }

  meep_cuda::select_device(ordinal);
  std::unique_ptr<boundary_exchange_buffer> created(
      new boundary_exchange_buffer);
  try {
    created->device_pointer = meep_cuda::allocate_device_bytes(bytes);
    created->completion_event = meep_cuda::create_event();
  }
  catch (...) {
    meep_cuda::destroy_event(created->completion_event);
    meep_cuda::free_device(created->device_pointer);
    throw;
  }
  created->scalar_count = scalar_count;
  created->device_ordinal = ordinal;
  boundary_exchange_buffer *result = created.release();
  try {
    owner_buffers.buffers.emplace(key, result);
  }
  catch (...) {
    release_boundary_exchange_buffer(result);
    throw;
  }
  device_buffer_allocations.fetch_add(1);
  if (slot > 0)
    boundary_receive_secondary_allocations.fetch_add(1);
  live_device_buffer_count().fetch_add(1);
  return result;
#else
  (void)owner;
  (void)token;
  (void)scalar_count;
  (void)slot;
  throw std::runtime_error(
      "Meep CUDA boundary exchange called in a CPU-only build");
#endif
}

float *boundary_exchange_host_data(boundary_exchange_buffer *buffer) {
#if MEEP_HAVE_CUDA
  if (!buffer || !buffer->device_pointer || buffer->scalar_count == 0)
    throw std::invalid_argument(
        "Meep CUDA boundary exchange buffer must be non-null");
  if (!buffer->host_pointer) {
    meep_cuda::select_device(buffer->device_ordinal);
    const std::size_t bytes = checked_bytes(
        buffer->scalar_count, sizeof(float), "boundary exchange");
    buffer->host_pointer = meep_cuda::allocate_pinned_bytes(bytes);
  }
  return static_cast<float *>(buffer->host_pointer);
#else
  (void)buffer;
  throw std::runtime_error(
      "Meep CUDA boundary exchange called in a CPU-only build");
#endif
}

float *boundary_exchange_device_data(boundary_exchange_buffer *buffer) {
#if MEEP_HAVE_CUDA
  if (!buffer || !buffer->device_pointer)
    throw std::invalid_argument(
        "Meep CUDA boundary device buffer must be non-null");
  return static_cast<float *>(buffer->device_pointer);
#else
  (void)buffer;
  throw std::runtime_error(
      "Meep CUDA boundary exchange called in a CPU-only build");
#endif
}

std::size_t boundary_exchange_scalar_count(
    const boundary_exchange_buffer *buffer) noexcept {
#if MEEP_HAVE_CUDA
  return buffer ? buffer->scalar_count : 0;
#else
  (void)buffer;
  return 0;
#endif
}

void resident_gather_boundary_fp32(
    resident_cache *const *source_caches,
    boundary_exchange_buffer *buffer,
    const float *const *source_pointers, std::size_t scalar_count,
    const boundary_descriptor_replay_token *replay_token) {
#if MEEP_HAVE_CUDA
  if (!buffer || !source_caches || !source_pointers || scalar_count == 0 ||
      buffer->scalar_count != scalar_count)
    throw std::invalid_argument(
        "Meep CUDA remote-boundary gather arguments are inconsistent");

  const std::size_t operation_bytes = checked_bytes(
      scalar_count, sizeof(meep_cuda::boundary_operation_fp32),
      "remote boundary gather operations");
  if (scalar_count > std::numeric_limits<std::size_t>::max() / 2)
    throw std::overflow_error(
        "Meep CUDA remote-boundary gather staging count overflow");
  const std::size_t staging_bytes = checked_bytes(
      2 * scalar_count, sizeof(float), "remote boundary gather staging");
  if (operation_bytes >
      std::numeric_limits<std::size_t>::max() - staging_bytes)
    throw std::overflow_error(
        "Meep CUDA remote-boundary gather scratch overflow");

  boundary_exchange_buffer::transfer_plan &plan = buffer->gather_plan;
  const bool replay_enabled =
      replay_token && replay_token->topology_identity() &&
      replay_token->topology_generation() != 0 &&
      std::getenv(
          "MEEP_GPU_DISABLE_BOUNDARY_DESCRIPTOR_FAST_REPLAY") == nullptr;
  bool fast_replay =
      replay_enabled && plan.device_pointer &&
      plan.operation_count == scalar_count &&
      plan.scalar_count == scalar_count &&
      plan.gather_caches.size() == scalar_count &&
      plan.gather_sources.size() == scalar_count &&
      plan.replay_topology_identity == replay_token->topology_identity() &&
      plan.replay_topology_generation ==
          replay_token->topology_generation() &&
      !plan.cache_generations.empty();
  if (fast_replay)
    for (const auto &entry : plan.cache_generations)
      if (!entry.first ||
          !entry.first->phase_active ||
          entry.first->device_ordinal != buffer->device_ordinal ||
          entry.first->allocation_generation != entry.second)
        fast_replay = false;

  if (fast_replay)
    boundary_gather_fast_replays.fetch_add(1,
                                           std::memory_order_relaxed);
  else {
    boundary_gather_full_validations.fetch_add(
        1, std::memory_order_relaxed);
    for (std::size_t index = 0; index < scalar_count; ++index)
      if (!source_caches[index] || !source_caches[index]->phase_active ||
          source_caches[index]->device_ordinal != buffer->device_ordinal)
        throw std::invalid_argument(
            "Meep CUDA remote-boundary gather sources must share one active "
            "device");

    bool plan_matches =
        plan.device_pointer && plan.operation_count == scalar_count &&
        plan.scalar_count == scalar_count &&
        plan.gather_caches.size() == scalar_count &&
        plan.gather_sources.size() == scalar_count &&
        !plan.cache_generations.empty();
    for (std::size_t index = 0; index < scalar_count && plan_matches;
         ++index)
      plan_matches =
          plan.gather_caches[index] == source_caches[index] &&
          plan.gather_sources[index] == source_pointers[index];
    if (plan_matches)
      for (const auto &entry : plan.cache_generations)
        if (!entry.first || !entry.first->phase_active ||
            entry.first->device_ordinal != buffer->device_ordinal ||
            entry.first->allocation_generation != entry.second)
          plan_matches = false;

    if (!plan_matches && plan.device_pointer)
      release_boundary_transfer_plan(plan);

    const bool create_plan = !plan.device_pointer;
    if (create_plan) {
      std::vector<meep_cuda::boundary_operation_fp32> operations;
      operations.reserve(scalar_count);
      float *device_values = static_cast<float *>(buffer->device_pointer);
      for (std::size_t index = 0; index < scalar_count; ++index) {
        if (!source_pointers[index])
          throw std::invalid_argument(
              "Meep CUDA remote-boundary gather source must be non-null");
        operations.push_back(
            {device_values + index, nullptr,
             static_cast<const float *>(
                 resident_device_address(
                     source_caches[index], source_pointers[index], false)),
             nullptr, 1.0f, 0.0f});
      }

      plan.device_pointer =
          meep_cuda::allocate_device_bytes(operation_bytes + staging_bytes);
      try {
        meep_cuda::copy_to_device(
            plan.device_pointer, operations.data(), operation_bytes);
        plan.operation_count = scalar_count;
        plan.scalar_count = scalar_count;
        plan.gather_caches.assign(
            source_caches, source_caches + scalar_count);
        plan.gather_sources.assign(
            source_pointers, source_pointers + scalar_count);
        for (std::size_t index = 0; index < scalar_count; ++index)
          if (std::find_if(
                  plan.cache_generations.begin(),
                  plan.cache_generations.end(),
                  [source_caches, index](
                      const std::pair<resident_cache *, std::uint64_t>
                          &entry) {
                    return entry.first == source_caches[index];
                  }) == plan.cache_generations.end())
            plan.cache_generations.push_back(std::make_pair(
                source_caches[index],
                source_caches[index]->allocation_generation));
        plan.generation = next_boundary_plan_generation();
      }
      catch (...) {
        meep_cuda::free_device(plan.device_pointer);
        plan.device_pointer = nullptr;
        plan.operation_count = 0;
        plan.scalar_count = 0;
        plan.gather_caches.clear();
        plan.gather_sources.clear();
        plan.cache_generations.clear();
        throw;
      }
      host_to_device_bytes.fetch_add(
          static_cast<std::uint64_t>(operation_bytes));
      device_buffer_allocations.fetch_add(1);
      live_device_buffer_count().fetch_add(1);
    }
    else
      device_buffer_reuses.fetch_add(1);

    plan.replay_topology_identity =
        replay_enabled ? replay_token->topology_identity() : nullptr;
    plan.replay_topology_generation =
        replay_enabled ? replay_token->topology_generation() : 0;
  }

  if (fast_replay) device_buffer_reuses.fetch_add(1);

  if (std::getenv("MEEP_GPU_DISABLE_DIRECT_REMOTE_BOUNDARY") == nullptr)
    meep_cuda::apply_nonalias_boundary_fp32(
        static_cast<const meep_cuda::boundary_operation_fp32 *>(
            plan.device_pointer),
        scalar_count);
  else
    meep_cuda::apply_boundary_fp32(
        static_cast<const meep_cuda::boundary_operation_fp32 *>(
            plan.device_pointer),
        reinterpret_cast<float *>(
            static_cast<unsigned char *>(plan.device_pointer) +
            operation_bytes),
        scalar_count);
  record_boundary_completion_event(buffer);
#else
  (void)source_caches;
  (void)buffer;
  (void)source_pointers;
  (void)scalar_count;
  (void)replay_token;
  throw std::runtime_error(
      "Meep CUDA remote-boundary gather called in a CPU-only build");
#endif
}

#if MEEP_HAVE_CUDA
bool boundary_phase_graph_algorithms_enabled() noexcept {
  return std::getenv("MEEP_GPU_DISABLE_BOUNDARY_PHASE_GRAPH") == nullptr &&
         std::getenv("MEEP_GPU_DISABLE_BOUNDARY_GRAPH") == nullptr &&
         std::getenv("MEEP_GPU_DISABLE_DIRECT_BOUNDARY_ZERO") == nullptr &&
         std::getenv("MEEP_GPU_DISABLE_DIRECT_LOCAL_BOUNDARY") == nullptr &&
         std::getenv("MEEP_GPU_DISABLE_DIRECT_REMOTE_BOUNDARY") == nullptr;
}

bool boundary_transfer_plan_generations_match(
    const boundary_exchange_buffer::transfer_plan &plan) noexcept {
  if (!plan.device_pointer || plan.generation == 0) return false;
  for (const auto &entry : plan.cache_generations)
    if (!entry.first ||
        entry.first->allocation_generation != entry.second)
      return false;
  return true;
}

bool boundary_owner_contains_buffer(
    const boundary_exchange_owner &owner,
    const boundary_exchange_buffer *buffer) noexcept {
  for (const auto &entry : owner.buffers)
    if (entry.second == buffer) return true;
  return false;
}

bool boundary_phase_graph_matches_locked(
    const boundary_exchange_owner &owner,
    const boundary_phase_graph *graph) noexcept {
  if (!graph || !graph->execution_graph || !graph->owner ||
      !graph->send_ready_event || !graph->completion_event ||
      graph->device_ordinal < 0 ||
      graph->send_buffers.empty() ||
      graph->send_buffers.size() !=
          graph->gather_plan_generations.size())
    return false;
  const auto zero = owner.operation_plans.find(graph->zero_plan_token);
  const auto copy = owner.operation_plans.find(graph->copy_plan_token);
  if (zero == owner.operation_plans.end() ||
      copy == owner.operation_plans.end() ||
      zero->second != graph->zero_plan_snapshot ||
      copy->second != graph->copy_plan_snapshot ||
      zero->second->generation != graph->zero_plan_generation ||
      copy->second->generation != graph->copy_plan_generation ||
      zero->second->device_ordinal != graph->device_ordinal ||
      copy->second->device_ordinal != graph->device_ordinal ||
      !boundary_operation_plan_generations_match(zero->second) ||
      !boundary_operation_plan_generations_match(copy->second))
    return false;
  for (std::size_t index = 0; index < graph->send_buffers.size(); ++index) {
    const boundary_exchange_buffer *buffer = graph->send_buffers[index];
    if (!buffer || !boundary_owner_contains_buffer(owner, buffer) ||
        buffer->device_ordinal != graph->device_ordinal ||
        buffer->gather_plan.scalar_count != buffer->scalar_count ||
        buffer->gather_plan.generation !=
            graph->gather_plan_generations[index] ||
        !boundary_transfer_plan_generations_match(buffer->gather_plan))
      return false;
  }
  return true;
}
#endif

boundary_phase_graph *resident_create_boundary_phase_graph(
    const void *owner, std::size_t zero_plan_token,
    std::size_t copy_plan_token,
    boundary_exchange_buffer *const *send_buffers,
    std::size_t send_buffer_count) noexcept {
#if MEEP_HAVE_CUDA
  if (!owner || !send_buffers || send_buffer_count == 0 ||
      !boundary_phase_graph_algorithms_enabled())
    return nullptr;
  std::unique_ptr<boundary_phase_graph> result(new (std::nothrow)
                                                   boundary_phase_graph);
  if (!result) return nullptr;
  try {
    boundary_exchange_registry &registry = get_boundary_exchange_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto owner_entry = registry.owners.find(owner);
    if (owner_entry == registry.owners.end()) return nullptr;
    const boundary_exchange_owner &plans = owner_entry->second;
    const auto zero = plans.operation_plans.find(zero_plan_token);
    const auto copy = plans.operation_plans.find(copy_plan_token);
    if (zero == plans.operation_plans.end() ||
        copy == plans.operation_plans.end() ||
        zero->second->generation == 0 || copy->second->generation == 0 ||
        !boundary_operation_plan_generations_match(zero->second) ||
        !boundary_operation_plan_generations_match(copy->second) ||
        zero->second->device_ordinal != copy->second->device_ordinal ||
        (zero->second->operation_count &&
         !zero->second->all_sources_null))
      return nullptr;

    result->owner = owner;
    result->zero_plan_token = zero_plan_token;
    result->copy_plan_token = copy_plan_token;
    result->zero_plan_snapshot = zero->second;
    result->copy_plan_snapshot = copy->second;
    result->zero_plan_generation = zero->second->generation;
    result->copy_plan_generation = copy->second->generation;
    result->device_ordinal = zero->second->device_ordinal;
    meep_cuda::select_device(result->device_ordinal);
    result->send_ready_event = meep_cuda::create_event();
    result->completion_event = meep_cuda::create_event();
    result->send_buffers.assign(
        send_buffers, send_buffers + send_buffer_count);
    result->gather_plan_generations.reserve(send_buffer_count);

    std::vector<meep_cuda::boundary_phase_stage_fp32> stages;
    stages.reserve(send_buffer_count + 2);
    if (zero->second->operation_count)
      stages.push_back(
          {meep_cuda::boundary_phase_stage_kind::zero,
           static_cast<const meep_cuda::boundary_operation_fp32 *>(
               zero->second->device_pointer),
           nullptr, zero->second->operation_count, nullptr});

    for (std::size_t index = 0; index < send_buffer_count; ++index) {
      boundary_exchange_buffer *buffer = send_buffers[index];
      if (!buffer || !boundary_owner_contains_buffer(plans, buffer) ||
          buffer->device_ordinal != result->device_ordinal ||
          buffer->gather_plan.operation_count != buffer->scalar_count ||
          buffer->gather_plan.scalar_count != buffer->scalar_count ||
          !buffer->completion_event ||
          !boundary_transfer_plan_generations_match(buffer->gather_plan))
        throw std::runtime_error(
            "boundary phase graph send buffer validation failed");
      result->gather_plan_generations.push_back(
          buffer->gather_plan.generation);
      stages.push_back(
          {meep_cuda::boundary_phase_stage_kind::nonalias,
           static_cast<const meep_cuda::boundary_operation_fp32 *>(
               buffer->gather_plan.device_pointer),
           nullptr, buffer->gather_plan.operation_count,
           index + 1 == send_buffer_count
               ? result->send_ready_event
               : nullptr});
    }

    if (copy->second->operation_count) {
      meep_cuda::boundary_phase_stage_kind kind =
          copy->second->all_sources_null
              ? meep_cuda::boundary_phase_stage_kind::zero
              : (copy->second->nonalias_safe
                     ? meep_cuda::boundary_phase_stage_kind::nonalias
                     : meep_cuda::boundary_phase_stage_kind::ordered);
      const std::size_t operation_bytes = checked_bytes(
          copy->second->operation_count,
          sizeof(meep_cuda::boundary_operation_fp32),
          "boundary phase graph copy operations");
      stages.push_back(
          {kind,
           static_cast<const meep_cuda::boundary_operation_fp32 *>(
               copy->second->device_pointer),
           kind == meep_cuda::boundary_phase_stage_kind::ordered
               ? reinterpret_cast<float *>(
                     static_cast<unsigned char *>(
                         copy->second->device_pointer) + operation_bytes)
               : nullptr,
           copy->second->operation_count, nullptr});
    }
    if (stages.empty())
      throw std::runtime_error(
          "boundary phase graph contains no executable stages");

    result->execution_graph = meep_cuda::create_boundary_phase_graph_fp32(
        stages.data(), stages.size(), result->completion_event);
    boundary_phase_graph_creations.fetch_add(1);
    return result.release();
  }
  catch (...) {
    meep_cuda::destroy_boundary_phase_graph_fp32(result->execution_graph);
    meep_cuda::destroy_event(result->send_ready_event);
    meep_cuda::destroy_event(result->completion_event);
    return nullptr;
  }
#else
  (void)owner;
  (void)zero_plan_token;
  (void)copy_plan_token;
  (void)send_buffers;
  (void)send_buffer_count;
  return nullptr;
#endif
}

bool resident_boundary_phase_graph_ready(
    const void *owner, const boundary_phase_graph *graph) noexcept {
#if MEEP_HAVE_CUDA
  if (!owner || !graph || graph->owner != owner) return false;
  try {
    boundary_exchange_registry &registry = get_boundary_exchange_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto owner_entry = registry.owners.find(owner);
    return owner_entry != registry.owners.end() &&
           boundary_phase_graph_matches_locked(owner_entry->second, graph);
  }
  catch (...) {
    return false;
  }
#else
  (void)owner;
  (void)graph;
  return false;
#endif
}

bool resident_launch_boundary_phase_graph(
    const void *owner, boundary_phase_graph *graph) {
#if MEEP_HAVE_CUDA
  if (!owner || !graph || graph->owner != owner ||
      !boundary_phase_graph_algorithms_enabled())
    return false;
  boundary_exchange_registry &registry = get_boundary_exchange_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto owner_entry = registry.owners.find(owner);
  if (owner_entry == registry.owners.end() ||
      !boundary_phase_graph_matches_locked(owner_entry->second, graph))
    return false;
  boundary_exchange_owner &plans = owner_entry->second;
  boundary_exchange_owner::operation_plan *zero =
      plans.operation_plans.at(graph->zero_plan_token);
  boundary_exchange_owner::operation_plan *copy =
      plans.operation_plans.at(graph->copy_plan_token);

  mark_boundary_plan_destinations_dirty(zero);
  mark_boundary_plan_destinations_dirty(copy);
  meep_cuda::select_device(graph->device_ordinal);
  meep_cuda::launch_boundary_phase_graph_fp32(graph->execution_graph);
  boundary_phase_graph_launches.fetch_add(1);
  graph->send_ready_event_recorded = true;
  graph->completion_event_recorded = true;

  if (zero->operation_count) {
    cuda_boundary_calls.fetch_add(1);
    cuda_boundary_points.fetch_add(
        static_cast<std::uint64_t>(zero->scalar_count));
    device_buffer_reuses.fetch_add(1);
  }
  for (std::size_t index = 0; index < graph->send_buffers.size(); ++index)
    device_buffer_reuses.fetch_add(1);
  if (copy->operation_count) {
    cuda_boundary_calls.fetch_add(1);
    cuda_boundary_points.fetch_add(
        static_cast<std::uint64_t>(copy->scalar_count));
    device_buffer_reuses.fetch_add(1);
  }
  return true;
#else
  (void)owner;
  (void)graph;
  throw std::runtime_error(
      "Meep CUDA boundary phase graph called in a CPU-only build");
#endif
}

void resident_synchronize_boundary_phase_graph_send_ready(
    boundary_phase_graph *graph) {
#if MEEP_HAVE_CUDA
  if (!graph || !graph->execution_graph || !graph->send_ready_event ||
      !graph->send_ready_event_recorded || graph->device_ordinal < 0)
    throw std::invalid_argument(
        "Meep CUDA boundary phase graph has no pending send-ready event");
  meep_cuda::select_device(graph->device_ordinal);
  meep_cuda::synchronize_event(graph->send_ready_event);
#else
  (void)graph;
  throw std::runtime_error(
      "Meep CUDA boundary phase graph called in a CPU-only build");
#endif
}

void destroy_boundary_phase_graph(boundary_phase_graph *graph) noexcept {
  if (!graph) return;
#if MEEP_HAVE_CUDA
  if (graph->device_ordinal >= 0) {
    try {
      meep_cuda::select_device(graph->device_ordinal);
      if (graph->completion_event_recorded)
        meep_cuda::synchronize_event(graph->completion_event);
    }
    catch (...) {}
  }
  meep_cuda::destroy_boundary_phase_graph_fp32(graph->execution_graph);
  meep_cuda::destroy_event(graph->send_ready_event);
  meep_cuda::destroy_event(graph->completion_event);
#endif
  delete graph;
}

void boundary_exchange_copy_to_host(boundary_exchange_buffer *buffer) {
#if MEEP_HAVE_CUDA
  if (!buffer || !buffer->device_pointer)
    throw std::invalid_argument(
        "Meep CUDA boundary exchange buffer must be non-null");
  meep_cuda::select_device(buffer->device_ordinal);
  void *host_pointer = boundary_exchange_host_data(buffer);
  const std::size_t bytes = checked_bytes(
      buffer->scalar_count, sizeof(float), "boundary exchange");
  meep_cuda::copy_to_host(host_pointer, buffer->device_pointer, bytes);
  device_to_host_bytes.fetch_add(static_cast<std::uint64_t>(bytes));
#else
  (void)buffer;
  throw std::runtime_error(
      "Meep CUDA boundary exchange called in a CPU-only build");
#endif
}

void boundary_exchange_copy_to_device(boundary_exchange_buffer *buffer) {
#if MEEP_HAVE_CUDA
  if (!buffer || !buffer->device_pointer)
    throw std::invalid_argument(
        "Meep CUDA boundary exchange buffer must be non-null");
  meep_cuda::select_device(buffer->device_ordinal);
  const void *host_pointer = boundary_exchange_host_data(buffer);
  const std::size_t bytes = checked_bytes(
      buffer->scalar_count, sizeof(float), "boundary exchange");
  // host_pointer is page-locked storage owned by this persistent buffer.
  // Queue the copy on the same default stream as the following scatter so
  // MPI completion callbacks do not serialize MPI progress on cudaMemcpy.
  // Record a fence immediately: if scatter preparation later throws, the
  // next receive still cannot overwrite host_pointer while CUDA reads it.
  meep_cuda::copy_to_device_async(
      buffer->device_pointer, host_pointer, bytes);
  record_boundary_completion_event(buffer);
  host_to_device_bytes.fetch_add(static_cast<std::uint64_t>(bytes));
#else
  (void)buffer;
  throw std::runtime_error(
      "Meep CUDA boundary exchange called in a CPU-only build");
#endif
}

void boundary_exchange_synchronize(boundary_exchange_buffer *buffer) {
#if MEEP_HAVE_CUDA
  if (!buffer)
    throw std::invalid_argument(
        "Meep CUDA boundary exchange buffer must be non-null");
  meep_cuda::select_device(buffer->device_ordinal);
  if (buffer->completion_event_recorded)
    try {
      synchronize_boundary_event_for_lifetime(
          buffer->completion_event);
      buffer->completion_event_recorded = false;
    }
    catch (...) {
      const std::exception_ptr event_error = std::current_exception();
      try {
        synchronize_boundary_device_fallback();
        buffer->completion_event_recorded = false;
      }
      catch (...) {
        buffer->lifetime_uncertain = true;
        throw_boundary_double_fence_failure(
            "CUDA boundary event synchronization failed", event_error,
            std::current_exception());
      }
      std::rethrow_exception(event_error);
    }
#else
  (void)buffer;
  throw std::runtime_error(
      "Meep CUDA boundary exchange called in a CPU-only build");
#endif
}

void resident_scatter_boundary_fp32(
    boundary_exchange_buffer *buffer,
    const remote_boundary_operation_fp32 *operations,
    std::size_t operation_count,
    const boundary_descriptor_replay_token *replay_token) {
#if MEEP_HAVE_CUDA
  if (!buffer || !operations || operation_count == 0)
    throw std::invalid_argument(
        "Meep CUDA remote-boundary scatter arguments must be non-null");
  resident_cache *cache = operations[0].destination_cache;
  if (!cache || !cache->phase_active)
    throw std::logic_error(
        "Meep CUDA remote-boundary scatter requires an active resident phase");
  if (cache->device_ordinal != buffer->device_ordinal)
    throw std::invalid_argument(
        "Meep CUDA remote-boundary scatter buffer is on another device");

  const float *device_values =
      static_cast<const float *>(buffer->device_pointer);
  const std::size_t operation_bytes = checked_bytes(
      operation_count, sizeof(meep_cuda::boundary_operation_fp32),
      "remote boundary scatter operations");
  if (operation_count > std::numeric_limits<std::size_t>::max() / 2)
    throw std::overflow_error(
        "Meep CUDA remote-boundary scatter staging count overflow");
  const std::size_t staging_bytes = checked_bytes(
      2 * operation_count, sizeof(float), "remote boundary scatter staging");
  if (operation_bytes >
      std::numeric_limits<std::size_t>::max() - staging_bytes)
    throw std::overflow_error(
        "Meep CUDA remote-boundary scatter scratch overflow");

  boundary_exchange_buffer::transfer_plan &plan = buffer->scatter_plan;
  const bool replay_enabled =
      replay_token && replay_token->topology_identity() &&
      replay_token->topology_generation() != 0 &&
      std::getenv(
          "MEEP_GPU_DISABLE_BOUNDARY_DESCRIPTOR_FAST_REPLAY") == nullptr;
  bool fast_replay =
      replay_enabled && plan.device_pointer &&
      plan.operation_count == operation_count &&
      plan.scalar_count != 0 &&
      plan.scatter_operations.size() == operation_count &&
      plan.replay_topology_identity == replay_token->topology_identity() &&
      plan.replay_topology_generation ==
          replay_token->topology_generation() &&
      !plan.cache_generations.empty();
  if (fast_replay)
    for (const auto &entry : plan.cache_generations)
      if (!entry.first ||
          !entry.first->phase_active ||
          entry.first->device_ordinal != buffer->device_ordinal ||
          entry.first->allocation_generation != entry.second)
        fast_replay = false;

  if (fast_replay)
    boundary_scatter_fast_replays.fetch_add(1,
                                            std::memory_order_relaxed);
  else {
    boundary_scatter_full_validations.fetch_add(
        1, std::memory_order_relaxed);
    bool plan_matches =
        plan.device_pointer && plan.operation_count == operation_count &&
        plan.scalar_count != 0 &&
        plan.scatter_operations.size() == operation_count;
    for (std::size_t index = 0; index < operation_count && plan_matches;
         ++index) {
      const remote_boundary_operation_fp32 &left =
          plan.scatter_operations[index];
      const remote_boundary_operation_fp32 &right = operations[index];
      plan_matches =
          left.destination_cache == right.destination_cache &&
          left.destination_real == right.destination_real &&
          left.destination_imag == right.destination_imag &&
          left.source_offset == right.source_offset &&
          left.phase_real == right.phase_real &&
          left.phase_imag == right.phase_imag;
    }
    if (plan_matches)
      for (const auto &entry : plan.cache_generations)
        if (!entry.first || !entry.first->phase_active ||
            entry.first->device_ordinal != buffer->device_ordinal ||
            entry.first->allocation_generation != entry.second)
          plan_matches = false;

    if (!plan_matches && plan.device_pointer)
      release_boundary_transfer_plan(plan);
  }

  std::size_t scalar_count = 0;
  if (!plan.device_pointer) {
    std::vector<meep_cuda::boundary_operation_fp32> device_operations;
    std::vector<resident_cache *> referenced_caches;
    std::vector<std::pair<resident_cache *, std::uint64_t> >
        cache_generations;
    std::vector<std::pair<resident_cache *, const float *> >
        writable_mirrors;
    std::vector<remote_boundary_operation_fp32> scatter_operations(
        operations, operations + operation_count);
    device_operations.reserve(operation_count);
    referenced_caches.reserve(operation_count);
    cache_generations.reserve(operation_count);
    writable_mirrors.reserve(2 * operation_count);
    for (std::size_t index = 0; index < operation_count; ++index) {
      const remote_boundary_operation_fp32 &operation = operations[index];
      if (!operation.destination_cache ||
          !operation.destination_cache->phase_active ||
          operation.destination_cache->device_ordinal !=
              buffer->device_ordinal)
        throw std::invalid_argument(
            "Meep CUDA remote-boundary destinations must share one active "
            "device");
      if (!operation.destination_real ||
          operation.source_offset >= buffer->scalar_count ||
          (operation.destination_imag &&
           operation.source_offset + 1 >= buffer->scalar_count))
        throw std::out_of_range(
            "Meep CUDA remote-boundary scatter index is out of range");
      device_operations.push_back(
          {static_cast<float *>(resident_device_address(
               operation.destination_cache, operation.destination_real,
               false)),
           operation.destination_imag
               ? static_cast<float *>(resident_device_address(
                     operation.destination_cache,
                     operation.destination_imag, false))
               : nullptr,
           device_values + operation.source_offset,
           operation.destination_imag
               ? device_values + operation.source_offset + 1
               : nullptr,
           operation.phase_real, operation.phase_imag});
      scalar_count += operation.destination_imag ? 2u : 1u;

      if (std::find(referenced_caches.begin(), referenced_caches.end(),
                    operation.destination_cache) ==
          referenced_caches.end())
        referenced_caches.push_back(operation.destination_cache);
      const float *destination_bases[] = {
          resident_mirror_base_for_address(
              operation.destination_cache, operation.destination_real),
          operation.destination_imag
              ? resident_mirror_base_for_address(
                    operation.destination_cache,
                    operation.destination_imag)
              : nullptr};
      for (const float *base : destination_bases) {
        if (!base) continue;
        const auto writable =
            std::make_pair(operation.destination_cache, base);
        if (std::find(writable_mirrors.begin(), writable_mirrors.end(),
                      writable) == writable_mirrors.end())
          writable_mirrors.push_back(writable);
      }
    }
    for (resident_cache *destination_cache : referenced_caches)
      cache_generations.push_back(
          std::make_pair(destination_cache,
                         destination_cache->allocation_generation));

    void *created_device_pointer = nullptr;
    try {
      created_device_pointer =
          meep_cuda::allocate_device_bytes(operation_bytes + staging_bytes);
      meep_cuda::copy_to_device(
          created_device_pointer, device_operations.data(), operation_bytes);
    }
    catch (...) {
      meep_cuda::free_device(created_device_pointer);
      throw;
    }
    // Commit only after all host allocations and the descriptor upload have
    // succeeded.  A failed first attempt therefore leaves a canonical empty
    // plan that can be retried without stale cache/base metadata.
    plan.device_pointer = created_device_pointer;
    plan.operation_count = operation_count;
    plan.scalar_count = scalar_count;
    plan.generation = next_boundary_plan_generation();
    plan.scatter_operations.swap(scatter_operations);
    plan.cache_generations.swap(cache_generations);
    plan.writable_mirrors.swap(writable_mirrors);
    host_to_device_bytes.fetch_add(
        static_cast<std::uint64_t>(operation_bytes));
    device_buffer_allocations.fetch_add(1);
    live_device_buffer_count().fetch_add(1);
  }
  else {
    scalar_count = plan.scalar_count;
    device_buffer_reuses.fetch_add(1);
  }

  if (!fast_replay) {
    plan.replay_topology_identity =
        replay_enabled ? replay_token->topology_identity() : nullptr;
    plan.replay_topology_generation =
        replay_enabled ? replay_token->topology_generation() : 0;
  }

  for (const auto &entry : plan.writable_mirrors)
    (void)resident_device_address(entry.first, entry.second, true);

  if (std::getenv("MEEP_GPU_DISABLE_DIRECT_REMOTE_BOUNDARY") == nullptr)
    meep_cuda::apply_nonalias_boundary_fp32(
        static_cast<const meep_cuda::boundary_operation_fp32 *>(
            plan.device_pointer),
        operation_count);
  else
    meep_cuda::apply_boundary_fp32(
        static_cast<const meep_cuda::boundary_operation_fp32 *>(
            plan.device_pointer),
        reinterpret_cast<float *>(
            static_cast<unsigned char *>(plan.device_pointer) +
            operation_bytes),
        operation_count);
  record_boundary_completion_event(buffer);
  cuda_boundary_calls.fetch_add(1);
  cuda_boundary_points.fetch_add(
      static_cast<std::uint64_t>(scalar_count));
#else
  (void)buffer;
  (void)operations;
  (void)operation_count;
  (void)replay_token;
  throw std::runtime_error(
      "Meep CUDA remote-boundary scatter called in a CPU-only build");
#endif
}

void record_mpi_boundary_transfer(std::size_t scalar_count,
                                  bool cuda_aware,
                                  bool receive) noexcept {
  mpi_boundary_scalars.fetch_add(static_cast<std::uint64_t>(scalar_count));
  const std::uint64_t bytes =
      static_cast<std::uint64_t>(scalar_count) * sizeof(float);
  if (cuda_aware)
    cuda_aware_mpi_bytes.fetch_add(bytes);
  else {
    pinned_mpi_bytes.fetch_add(bytes);
    if (receive)
      pinned_mpi_host_to_device_bytes.fetch_add(bytes);
    else
      pinned_mpi_device_to_host_bytes.fetch_add(bytes);
  }
}

void record_mpi_boundary_messages(std::size_t message_count) noexcept {
  mpi_boundary_messages.fetch_add(
      static_cast<std::uint64_t>(message_count));
}

void record_mpi_completion(bool waitall) noexcept {
  (waitall ? mpi_waitall_executions : mpi_waitsome_executions).fetch_add(1);
}

#if MEEP_HAVE_CUDA
struct prepared_curl_material_fp32 {
  meep_cuda::curl_material_fp32 device;
  std::size_t field_bytes;
};

prepared_curl_material_fp32 prepare_curl_material_fp32(
    resident_cache *cache, std::size_t array_count,
    const curl_material_fp32 &material, const char *label) {
  const std::size_t field_bytes =
      checked_bytes(array_count, sizeof(float), label);
  const std::size_t sigma_bytes =
      checked_bytes(material.sigma_count, sizeof(float), "cylindrical PML-f");
  const std::size_t sigma_u_bytes =
      checked_bytes(material.sigma_u_count, sizeof(float), "cylindrical PML-u");
  void *device_sigma =
      material.sigma
          ? ensure_resident_mirror(cache, material.sigma, sigma_bytes)
          : nullptr;
  void *device_kappa =
      material.kappa
          ? ensure_resident_mirror(cache, material.kappa, sigma_bytes)
          : nullptr;
  void *device_sigma_inverse =
      material.sigma_inverse
          ? ensure_resident_mirror(
                cache, material.sigma_inverse, sigma_bytes)
          : nullptr;
  void *device_field_u =
      material.field_u
          ? ensure_resident_mirror(cache, material.field_u, field_bytes)
          : nullptr;
  void *device_sigma_u =
      material.sigma_u
          ? ensure_resident_mirror(cache, material.sigma_u, sigma_u_bytes)
          : nullptr;
  void *device_kappa_u =
      material.kappa_u
          ? ensure_resident_mirror(cache, material.kappa_u, sigma_u_bytes)
          : nullptr;
  void *device_sigma_u_inverse =
      material.sigma_u_inverse
          ? ensure_resident_mirror(
                cache, material.sigma_u_inverse, sigma_u_bytes)
          : nullptr;
  void *device_conductivity =
      material.conductivity
          ? ensure_resident_mirror(
                cache, material.conductivity, field_bytes)
          : nullptr;
  void *device_conductivity_inverse =
      material.conductivity_inverse
          ? ensure_resident_mirror(
                cache, material.conductivity_inverse, field_bytes)
          : nullptr;
  void *device_field_conductivity =
      material.field_conductivity
          ? ensure_resident_mirror(
                cache, material.field_conductivity, field_bytes)
          : nullptr;
  return {{
              static_cast<const float *>(device_sigma),
              static_cast<const float *>(device_kappa),
              static_cast<const float *>(device_sigma_inverse),
              static_cast<float *>(device_field_u),
              static_cast<const float *>(device_sigma_u),
              static_cast<const float *>(device_kappa_u),
              static_cast<const float *>(device_sigma_u_inverse),
              material.dt,
              static_cast<const float *>(device_conductivity),
              static_cast<const float *>(device_conductivity_inverse),
              static_cast<float *>(device_field_conductivity),
          },
          field_bytes};
}

void mark_curl_material_dirty(
    resident_cache *cache, const curl_material_fp32 &material,
    std::size_t field_bytes) {
  if (material.field_u)
    mark_device_dirty(cache, material.field_u, field_bytes);
  if (material.field_conductivity)
    mark_device_dirty(
        cache, material.field_conductivity, field_bytes);
}

void collect_curl_phase_dependencies(
    resident_cache *cache, float *field, const float *g1, const float *g2,
    std::size_t field_bytes, std::size_t sigma_bytes,
    std::size_t sigma_u_bytes, const curl_material_fp32 &material) {
  collect_curl_phase_dependency(cache, field, field_bytes);
  collect_curl_phase_dependency(cache, g1, field_bytes);
  collect_curl_phase_dependency(cache, g2, field_bytes);
  collect_curl_phase_dependency(cache, material.sigma, sigma_bytes);
  collect_curl_phase_dependency(cache, material.kappa, sigma_bytes);
  collect_curl_phase_dependency(
      cache, material.sigma_inverse, sigma_bytes);
  collect_curl_phase_dependency(cache, material.field_u, field_bytes);
  collect_curl_phase_dependency(cache, material.sigma_u, sigma_u_bytes);
  collect_curl_phase_dependency(cache, material.kappa_u, sigma_u_bytes);
  collect_curl_phase_dependency(
      cache, material.sigma_u_inverse, sigma_u_bytes);
  collect_curl_phase_dependency(
      cache, material.conductivity, field_bytes);
  collect_curl_phase_dependency(
      cache, material.conductivity_inverse, field_bytes);
  collect_curl_phase_dependency(
      cache, material.field_conductivity, field_bytes);
}
#endif

void resident_step_curl_material_fp32(
    resident_cache *cache, float *field, const float *g1, const float *g2,
    std::size_t array_count, const index_space_fp32 &index_space,
    std::ptrdiff_t stride1, std::ptrdiff_t stride2, float dtdx,
    const curl_material_fp32 &material) {
#if MEEP_HAVE_CUDA
  if (!cache || !cache->phase_active)
    throw std::logic_error("Meep CUDA curl requires an active resident phase");
  if (!field) throw std::invalid_argument("Meep CUDA curl field must be non-null");
  if (!g1 && !g2) throw std::invalid_argument("Meep CUDA curl requires at least one operand");
  const index_space_bounds bounds = validate_curl_index_space(
      index_space, array_count, g1, g2, stride1, stride2, material);
  const std::size_t index_count = bounds.point_count;

  static_assert(
      sizeof(index_space_fp32) == sizeof(meep_cuda::index_space_fp32) &&
          alignof(index_space_fp32) ==
              alignof(meep_cuda::index_space_fp32) &&
          offsetof(index_space_fp32, field_start) ==
              offsetof(meep_cuda::index_space_fp32, field_start) &&
          offsetof(index_space_fp32, coefficient2_stride3) ==
              offsetof(meep_cuda::index_space_fp32,
                       coefficient2_stride3),
      "Meep/CUDA structured index-space layout mismatch");

  const std::size_t field_bytes = checked_bytes(array_count, sizeof(float), "field");
  const std::size_t sigma_bytes =
      checked_bytes(material.sigma_count, sizeof(float), "PML-f");
  const std::size_t sigma_u_bytes =
      checked_bytes(material.sigma_u_count, sizeof(float), "PML-u");
  void *device_field = ensure_resident_mirror(cache, field, field_bytes);
  void *device_g1 = g1 ? ensure_resident_mirror(cache, g1, field_bytes) : nullptr;
  void *device_g2 = g2 ? ensure_resident_mirror(cache, g2, field_bytes) : nullptr;
  void *device_sigma =
      material.sigma ? ensure_resident_mirror(cache, material.sigma, sigma_bytes) : nullptr;
  void *device_kappa =
      material.kappa ? ensure_resident_mirror(cache, material.kappa, sigma_bytes) : nullptr;
  void *device_sigma_inverse =
      material.sigma_inverse
          ? ensure_resident_mirror(cache, material.sigma_inverse, sigma_bytes)
          : nullptr;
  void *device_field_u =
      material.field_u ? ensure_resident_mirror(cache, material.field_u, field_bytes) : nullptr;
  void *device_sigma_u =
      material.sigma_u
          ? ensure_resident_mirror(cache, material.sigma_u, sigma_u_bytes)
          : nullptr;
  void *device_kappa_u =
      material.kappa_u
          ? ensure_resident_mirror(cache, material.kappa_u, sigma_u_bytes)
          : nullptr;
  void *device_sigma_u_inverse =
      material.sigma_u_inverse
          ? ensure_resident_mirror(cache, material.sigma_u_inverse, sigma_u_bytes)
          : nullptr;
  void *device_conductivity =
      material.conductivity
          ? ensure_resident_mirror(cache, material.conductivity, field_bytes)
          : nullptr;
  void *device_conductivity_inverse =
      material.conductivity_inverse
          ? ensure_resident_mirror(cache, material.conductivity_inverse, field_bytes)
          : nullptr;
  void *device_field_conductivity =
      material.field_conductivity
          ? ensure_resident_mirror(cache, material.field_conductivity, field_bytes)
          : nullptr;
  const meep_cuda::curl_material_fp32 device_material = {
      static_cast<const float *>(device_sigma),
      static_cast<const float *>(device_kappa),
      static_cast<const float *>(device_sigma_inverse),
      static_cast<float *>(device_field_u),
      static_cast<const float *>(device_sigma_u),
      static_cast<const float *>(device_kappa_u),
      static_cast<const float *>(device_sigma_u_inverse),
      material.dt,
      static_cast<const float *>(device_conductivity),
      static_cast<const float *>(device_conductivity_inverse),
      static_cast<float *>(device_field_conductivity)};
  const meep_cuda::detail::curl_operands_fp32 operands =
      meep_cuda::detail::normalize_curl_operands(
          g1 ? static_cast<const float *>(device_g1) : nullptr,
          g2 ? static_cast<const float *>(device_g2) : nullptr,
          stride1, stride2, dtdx);
  meep_cuda::curl_phase_operation_fp32 operation{};
  operation.field = static_cast<float *>(device_field);
  operation.g1 = operands.g1;
  operation.g2 = operands.g2;
  operation.inline_space = device_index_space(index_space);
  operation.point_count = index_count;
  operation.stride1 = operands.stride1;
  operation.stride2 = operands.stride2;
  operation.dtdx = operands.dtdx;
  operation.material = device_material;
  const bool collected = collect_curl_phase_operation(
      cache, cache->device_ordinal, operation,
      static_cast<std::uint64_t>(index_count));
  if (collected) {
    collect_curl_phase_dependencies(
        cache, field, g1, g2, field_bytes, sigma_bytes, sigma_u_bytes,
        material);
    collect_curl_phase_writable(cache, field, field_bytes);
    collect_curl_phase_writable(cache, material.field_u, field_bytes);
    collect_curl_phase_writable(
        cache, material.field_conductivity, field_bytes);
  }
  if (!collected)
    meep_cuda::step_curl_material_structured_fp32(
        static_cast<float *>(device_field),
        g1 ? static_cast<const float *>(device_g1) : nullptr,
        g2 ? static_cast<const float *>(device_g2) : nullptr,
        device_index_space(index_space), index_count, stride1, stride2,
        dtdx, device_material);
  mark_device_dirty(cache, field, field_bytes);
  if (material.field_u) mark_device_dirty(cache, material.field_u, field_bytes);
  if (material.field_conductivity)
    mark_device_dirty(cache, material.field_conductivity, field_bytes);

  if (!collected) {
    cuda_curl_calls.fetch_add(1);
    cuda_curl_points.fetch_add(static_cast<std::uint64_t>(index_count));
  }
#else
  (void)cache;
  (void)field;
  (void)g1;
  (void)g2;
  (void)array_count;
  (void)index_space;
  (void)stride1;
  (void)stride2;
  (void)dtdx;
  (void)material;
  throw std::runtime_error("Meep CUDA resident curl called in a CPU-only build");
#endif
}

void resident_step_curl_material_batched_fp32(
    resident_cache *cache, float *field, const float *g1, const float *g2,
    std::size_t array_count, const index_space_fp32 *index_spaces,
    std::size_t index_space_count, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, float dtdx,
    const curl_material_fp32 &material) {
#if MEEP_HAVE_CUDA
  if (!cache || !cache->phase_active)
    throw std::logic_error(
        "Meep CUDA batched curl requires an active resident phase");
  if (!field || !index_spaces || index_space_count == 0)
    throw std::invalid_argument(
        "Meep CUDA batched curl field and spaces must be non-null");
  if (!g1 && !g2)
    throw std::invalid_argument(
        "Meep CUDA batched curl requires at least one operand");

  std::size_t point_count = 0;
  std::size_t maximum_point_count = 0;
  for (std::size_t index = 0; index < index_space_count; ++index) {
    const index_space_bounds bounds = validate_curl_index_space(
        index_spaces[index], array_count, g1, g2, stride1, stride2,
        material);
    if (point_count >
        std::numeric_limits<std::size_t>::max() - bounds.point_count)
      throw std::overflow_error("Meep CUDA batched curl point overflow");
    point_count += bounds.point_count;
    maximum_point_count =
        std::max(maximum_point_count, bounds.point_count);
  }

  static_assert(
      sizeof(index_space_fp32) == sizeof(meep_cuda::index_space_fp32) &&
          alignof(index_space_fp32) ==
              alignof(meep_cuda::index_space_fp32),
      "Meep/CUDA batched index-space layout mismatch");

  const std::size_t field_bytes =
      checked_bytes(array_count, sizeof(float), "batched curl field");
  const std::size_t sigma_bytes =
      checked_bytes(material.sigma_count, sizeof(float), "batched PML-f");
  const std::size_t sigma_u_bytes =
      checked_bytes(material.sigma_u_count, sizeof(float), "batched PML-u");
  void *device_field = ensure_resident_mirror(cache, field, field_bytes);
  void *device_g1 =
      g1 ? ensure_resident_mirror(cache, g1, field_bytes) : nullptr;
  void *device_g2 =
      g2 ? ensure_resident_mirror(cache, g2, field_bytes) : nullptr;
  void *device_sigma =
      material.sigma
          ? ensure_resident_mirror(cache, material.sigma, sigma_bytes)
          : nullptr;
  void *device_kappa =
      material.kappa
          ? ensure_resident_mirror(cache, material.kappa, sigma_bytes)
          : nullptr;
  void *device_sigma_inverse =
      material.sigma_inverse
          ? ensure_resident_mirror(
                cache, material.sigma_inverse, sigma_bytes)
          : nullptr;
  void *device_field_u =
      material.field_u
          ? ensure_resident_mirror(cache, material.field_u, field_bytes)
          : nullptr;
  void *device_sigma_u =
      material.sigma_u
          ? ensure_resident_mirror(cache, material.sigma_u, sigma_u_bytes)
          : nullptr;
  void *device_kappa_u =
      material.kappa_u
          ? ensure_resident_mirror(cache, material.kappa_u, sigma_u_bytes)
          : nullptr;
  void *device_sigma_u_inverse =
      material.sigma_u_inverse
          ? ensure_resident_mirror(
                cache, material.sigma_u_inverse, sigma_u_bytes)
          : nullptr;
  void *device_conductivity =
      material.conductivity
          ? ensure_resident_mirror(
                cache, material.conductivity, field_bytes)
          : nullptr;
  void *device_conductivity_inverse =
      material.conductivity_inverse
          ? ensure_resident_mirror(
                cache, material.conductivity_inverse, field_bytes)
          : nullptr;
  void *device_field_conductivity =
      material.field_conductivity
          ? ensure_resident_mirror(
                cache, material.field_conductivity, field_bytes)
          : nullptr;

  const std::size_t descriptor_bytes = checked_bytes(
      index_space_count, sizeof(meep_cuda::index_space_fp32),
      "batched curl descriptors");
  resident_cache::structured_curl_batch *batch = nullptr;
  auto found = cache->structured_curl_batches.find(field);
  if (found != cache->structured_curl_batches.end() &&
      found->second.descriptor_count != index_space_count) {
    meep_cuda::free_device(found->second.device_spaces);
    cache->structured_curl_batches.erase(found);
    live_device_buffer_count().fetch_sub(1);
    found = cache->structured_curl_batches.end();
  }
  if (found == cache->structured_curl_batches.end()) {
    resident_cache::structured_curl_batch created;
    created.device_spaces =
        meep_cuda::allocate_device_bytes(descriptor_bytes);
    try {
      meep_cuda::copy_to_device(
          created.device_spaces, index_spaces, descriptor_bytes);
      created.descriptor_count = index_space_count;
      created.maximum_point_count = maximum_point_count;
      const auto inserted =
          cache->structured_curl_batches.emplace(field, created);
      batch = &inserted.first->second;
    }
    catch (...) {
      meep_cuda::free_device(created.device_spaces);
      throw;
    }
    host_to_device_bytes.fetch_add(
        static_cast<std::uint64_t>(descriptor_bytes));
    device_buffer_allocations.fetch_add(1);
    live_device_buffer_count().fetch_add(1);
  }
  else {
    batch = &found->second;
    if (batch->maximum_point_count != maximum_point_count)
      throw std::logic_error(
          "Meep CUDA batched curl topology changed without invalidation");
    device_buffer_reuses.fetch_add(1);
  }

  const meep_cuda::curl_material_fp32 device_material = {
      static_cast<const float *>(device_sigma),
      static_cast<const float *>(device_kappa),
      static_cast<const float *>(device_sigma_inverse),
      static_cast<float *>(device_field_u),
      static_cast<const float *>(device_sigma_u),
      static_cast<const float *>(device_kappa_u),
      static_cast<const float *>(device_sigma_u_inverse),
      material.dt,
      static_cast<const float *>(device_conductivity),
      static_cast<const float *>(device_conductivity_inverse),
      static_cast<float *>(device_field_conductivity)};
  const meep_cuda::detail::curl_operands_fp32 operands =
      meep_cuda::detail::normalize_curl_operands(
          g1 ? static_cast<const float *>(device_g1) : nullptr,
          g2 ? static_cast<const float *>(device_g2) : nullptr,
          stride1, stride2, dtdx);
  bool collected = false;
  if (pending_curl_phase.active) {
    for (std::size_t index = 0; index < index_space_count; ++index) {
      const index_space_fp32 &space = index_spaces[index];
      const std::size_t space_points =
          space.extent1 * space.extent2 * space.extent3;
      meep_cuda::curl_phase_operation_fp32 operation{};
      operation.field = static_cast<float *>(device_field);
      operation.g1 = operands.g1;
      operation.g2 = operands.g2;
      operation.inline_space = device_index_space(space);
      operation.point_count = space_points;
      operation.stride1 = operands.stride1;
      operation.stride2 = operands.stride2;
      operation.dtdx = operands.dtdx;
      operation.material = device_material;
      if (!collect_curl_phase_operation(
              cache, cache->device_ordinal, operation,
              static_cast<std::uint64_t>(space_points)))
        throw std::logic_error(
            "Meep CUDA curl phase collection ended unexpectedly");
    }
    collected = true;
  }
  if (collected) {
    collect_curl_phase_dependencies(
        cache, field, g1, g2, field_bytes, sigma_bytes, sigma_u_bytes,
        material);
    collect_curl_phase_writable(cache, field, field_bytes);
    collect_curl_phase_writable(cache, material.field_u, field_bytes);
    collect_curl_phase_writable(
        cache, material.field_conductivity, field_bytes);
  }
  if (!collected)
    meep_cuda::step_curl_material_batched_structured_fp32(
        static_cast<float *>(device_field),
        g1 ? static_cast<const float *>(device_g1) : nullptr,
        g2 ? static_cast<const float *>(device_g2) : nullptr,
        static_cast<const meep_cuda::index_space_fp32 *>(
            batch->device_spaces),
        batch->descriptor_count, batch->maximum_point_count, stride1,
        stride2, dtdx, device_material);
  mark_device_dirty(cache, field, field_bytes);
  if (material.field_u)
    mark_device_dirty(cache, material.field_u, field_bytes);
  if (material.field_conductivity)
    mark_device_dirty(
        cache, material.field_conductivity, field_bytes);
  if (!collected) {
    cuda_curl_calls.fetch_add(1);
    cuda_curl_points.fetch_add(static_cast<std::uint64_t>(point_count));
  }
#else
  (void)cache;
  (void)field;
  (void)g1;
  (void)g2;
  (void)array_count;
  (void)index_spaces;
  (void)index_space_count;
  (void)stride1;
  (void)stride2;
  (void)dtdx;
  (void)material;
  throw std::runtime_error(
      "Meep CUDA resident batched curl called in a CPU-only build");
#endif
}

void resident_step_beta_fp32(
    resident_cache *cache, float *field, const float *g,
    std::size_t array_count, const index_space_fp32 &index_space,
    float betadt, const beta_material_fp32 &material) {
#if MEEP_HAVE_CUDA
  if (!cache || !cache->phase_active)
    throw std::logic_error(
        "Meep CUDA beta update requires an active resident phase");
  if (!field || !g)
    throw std::invalid_argument(
        "Meep CUDA beta field and operand must be non-null");
  const index_space_bounds bounds =
      validate_beta_index_space(index_space, array_count, material);
  const std::size_t index_count = bounds.point_count;
  const std::size_t field_bytes =
      checked_bytes(array_count, sizeof(float), "beta field");
  const std::size_t sigma_bytes =
      checked_bytes(material.sigma_count, sizeof(float), "beta PML-f");
  const std::size_t sigma_u_bytes =
      checked_bytes(material.sigma_u_count, sizeof(float), "beta PML-u");

  void *device_field =
      ensure_resident_mirror(cache, field, field_bytes);
  void *device_g = ensure_resident_mirror(cache, g, field_bytes);
  void *device_sigma_inverse =
      material.sigma_inverse
          ? ensure_resident_mirror(
                cache, material.sigma_inverse, sigma_bytes)
          : nullptr;
  void *device_field_u =
      material.field_u
          ? ensure_resident_mirror(cache, material.field_u, field_bytes)
          : nullptr;
  void *device_sigma_u_inverse =
      material.sigma_u_inverse
          ? ensure_resident_mirror(
                cache, material.sigma_u_inverse, sigma_u_bytes)
          : nullptr;
  void *device_conductivity_inverse =
      material.conductivity_inverse
          ? ensure_resident_mirror(
                cache, material.conductivity_inverse, field_bytes)
          : nullptr;
  void *device_field_conductivity =
      material.field_conductivity
          ? ensure_resident_mirror(
                cache, material.field_conductivity, field_bytes)
          : nullptr;

  const meep_cuda::beta_material_fp32 device_material = {
      static_cast<const float *>(device_sigma_inverse),
      static_cast<float *>(device_field_u),
      static_cast<const float *>(device_sigma_u_inverse),
      static_cast<const float *>(device_conductivity_inverse),
      static_cast<float *>(device_field_conductivity)};
  flush_pending_curl_phase_before_direct_operation();
  meep_cuda::step_beta_structured_fp32(
      static_cast<float *>(device_field),
      static_cast<const float *>(device_g),
      device_index_space(index_space), index_count, betadt,
      device_material);
  mark_device_dirty(cache, field, field_bytes);
  if (material.field_u)
    mark_device_dirty(cache, material.field_u, field_bytes);
  if (material.field_conductivity)
    mark_device_dirty(
        cache, material.field_conductivity, field_bytes);
  // Beta is part of the B/D curl phase; account for it explicitly so
  // dispatch statistics include the additional device work.
  cuda_curl_calls.fetch_add(1);
  cuda_curl_points.fetch_add(
      static_cast<std::uint64_t>(index_count));
#else
  (void)cache;
  (void)field;
  (void)g;
  (void)array_count;
  (void)index_space;
  (void)betadt;
  (void)material;
  throw std::runtime_error(
      "Meep CUDA resident beta update called in a CPU-only build");
#endif
}

void resident_step_bfast_fp32(
    resident_cache *cache, float *field, const float *g1, const float *g2,
    float *bfast_field, std::size_t array_count,
    const index_space_fp32 &index_space, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, float k1, float k2,
    const bfast_material_fp32 &material) {
#if MEEP_HAVE_CUDA
  if (!cache || !cache->phase_active)
    throw std::logic_error(
        "Meep CUDA BFAST update requires an active resident phase");
  if (!field || !bfast_field)
    throw std::invalid_argument(
        "Meep CUDA BFAST field and auxiliary field must be non-null");
  const index_space_bounds bounds = validate_bfast_index_space(
      index_space, array_count, g1, g2, stride1, stride2, k1, k2,
      material);
  const meep_cuda::detail::bfast_operands_fp32 operands =
      meep_cuda::detail::normalize_bfast_operands(
          g1, g2, stride1, stride2, k1, k2);
  const std::size_t field_bytes =
      checked_bytes(array_count, sizeof(float), "BFAST field");
  const std::size_t sigma_bytes =
      checked_bytes(material.sigma_count, sizeof(float), "BFAST PML-f");
  const std::size_t sigma_u_bytes =
      checked_bytes(material.sigma_u_count, sizeof(float), "BFAST PML-u");

  void *device_field =
      ensure_resident_mirror(cache, field, field_bytes);
  void *device_g1 = operands.g1
                        ? ensure_resident_mirror(
                              cache, operands.g1, field_bytes)
                        : nullptr;
  void *device_g2 = operands.g2
                        ? ensure_resident_mirror(
                              cache, operands.g2, field_bytes)
                        : nullptr;
  void *device_bfast =
      ensure_resident_mirror(cache, bfast_field, field_bytes);
  void *device_sigma_inverse =
      material.sigma_inverse
          ? ensure_resident_mirror(
                cache, material.sigma_inverse, sigma_bytes)
          : nullptr;
  void *device_field_u =
      material.field_u
          ? ensure_resident_mirror(cache, material.field_u, field_bytes)
          : nullptr;
  void *device_sigma_u_inverse =
      material.sigma_u_inverse
          ? ensure_resident_mirror(
                cache, material.sigma_u_inverse, sigma_u_bytes)
          : nullptr;
  void *device_conductivity_inverse =
      material.conductivity_inverse
          ? ensure_resident_mirror(
                cache, material.conductivity_inverse, field_bytes)
          : nullptr;
  void *device_field_conductivity =
      material.field_conductivity
          ? ensure_resident_mirror(
                cache, material.field_conductivity, field_bytes)
          : nullptr;

  const meep_cuda::bfast_material_fp32 device_material = {
      static_cast<const float *>(device_sigma_inverse),
      static_cast<float *>(device_field_u),
      static_cast<const float *>(device_sigma_u_inverse),
      static_cast<const float *>(device_conductivity_inverse),
      static_cast<float *>(device_field_conductivity)};
  flush_pending_curl_phase_before_direct_operation();
  meep_cuda::step_bfast_structured_fp32(
      static_cast<float *>(device_field),
      static_cast<const float *>(device_g1),
      device_g2 ? static_cast<const float *>(device_g2) : nullptr,
      static_cast<float *>(device_bfast), device_index_space(index_space),
      bounds.point_count, operands.stride1, operands.stride2, operands.k1,
      operands.k2, device_material);

  mark_device_dirty(cache, field, field_bytes);
  mark_device_dirty(cache, bfast_field, field_bytes);
  if (material.field_u)
    mark_device_dirty(cache, material.field_u, field_bytes);
  if (material.field_conductivity)
    mark_device_dirty(
        cache, material.field_conductivity, field_bytes);
  cuda_curl_calls.fetch_add(1);
  cuda_curl_points.fetch_add(
      static_cast<std::uint64_t>(bounds.point_count));
#else
  (void)cache;
  (void)field;
  (void)g1;
  (void)g2;
  (void)bfast_field;
  (void)array_count;
  (void)index_space;
  (void)stride1;
  (void)stride2;
  (void)k1;
  (void)k2;
  (void)material;
  throw std::runtime_error(
      "Meep CUDA resident BFAST update called in a CPU-only build");
#endif
}

void resident_step_bfast_batched_fp32(
    resident_cache *cache, float *field, const float *g1, const float *g2,
    float *bfast_field, std::size_t array_count,
    const index_space_fp32 *index_spaces, std::size_t index_space_count,
    std::ptrdiff_t stride1, std::ptrdiff_t stride2, float k1, float k2,
    const bfast_material_fp32 &material) {
#if MEEP_HAVE_CUDA
  if (!cache || !cache->phase_active)
    throw std::logic_error(
        "Meep CUDA batched BFAST requires an active resident phase");
  if (!field || !bfast_field || !index_spaces ||
      index_space_count == 0)
    throw std::invalid_argument(
        "Meep CUDA batched BFAST field, auxiliary, and spaces must be non-null");

  std::size_t point_count = 0;
  std::size_t maximum_point_count = 0;
  for (std::size_t index = 0; index < index_space_count; ++index) {
    const index_space_bounds bounds = validate_bfast_index_space(
        index_spaces[index], array_count, g1, g2, stride1, stride2, k1,
        k2, material);
    if (point_count >
        std::numeric_limits<std::size_t>::max() - bounds.point_count)
      throw std::overflow_error(
          "Meep CUDA batched BFAST point overflow");
    point_count += bounds.point_count;
    maximum_point_count =
        std::max(maximum_point_count, bounds.point_count);
  }
  const meep_cuda::detail::bfast_operands_fp32 operands =
      meep_cuda::detail::normalize_bfast_operands(
          g1, g2, stride1, stride2, k1, k2);

  auto found = cache->structured_curl_batches.find(field);
  if (found == cache->structured_curl_batches.end() ||
      found->second.descriptor_count != index_space_count ||
      found->second.maximum_point_count != maximum_point_count)
    throw std::logic_error(
        "Meep CUDA batched BFAST requires the matching curl descriptor batch");
  const resident_cache::structured_curl_batch &batch = found->second;

  const std::size_t field_bytes =
      checked_bytes(array_count, sizeof(float), "batched BFAST field");
  const std::size_t sigma_bytes =
      checked_bytes(material.sigma_count, sizeof(float), "batched BFAST PML-f");
  const std::size_t sigma_u_bytes =
      checked_bytes(material.sigma_u_count, sizeof(float), "batched BFAST PML-u");
  void *device_field =
      ensure_resident_mirror(cache, field, field_bytes);
  void *device_g1 =
      ensure_resident_mirror(cache, operands.g1, field_bytes);
  void *device_g2 = operands.g2
                        ? ensure_resident_mirror(
                              cache, operands.g2, field_bytes)
                        : nullptr;
  void *device_bfast =
      ensure_resident_mirror(cache, bfast_field, field_bytes);
  void *device_sigma_inverse =
      material.sigma_inverse
          ? ensure_resident_mirror(
                cache, material.sigma_inverse, sigma_bytes)
          : nullptr;
  void *device_field_u =
      material.field_u
          ? ensure_resident_mirror(cache, material.field_u, field_bytes)
          : nullptr;
  void *device_sigma_u_inverse =
      material.sigma_u_inverse
          ? ensure_resident_mirror(
                cache, material.sigma_u_inverse, sigma_u_bytes)
          : nullptr;
  void *device_conductivity_inverse =
      material.conductivity_inverse
          ? ensure_resident_mirror(
                cache, material.conductivity_inverse, field_bytes)
          : nullptr;
  void *device_field_conductivity =
      material.field_conductivity
          ? ensure_resident_mirror(
                cache, material.field_conductivity, field_bytes)
          : nullptr;
  const meep_cuda::bfast_material_fp32 device_material = {
      static_cast<const float *>(device_sigma_inverse),
      static_cast<float *>(device_field_u),
      static_cast<const float *>(device_sigma_u_inverse),
      static_cast<const float *>(device_conductivity_inverse),
      static_cast<float *>(device_field_conductivity)};
  flush_pending_curl_phase_before_direct_operation();
  meep_cuda::step_bfast_batched_structured_fp32(
      static_cast<float *>(device_field),
      static_cast<const float *>(device_g1),
      device_g2 ? static_cast<const float *>(device_g2) : nullptr,
      static_cast<float *>(device_bfast),
      static_cast<const meep_cuda::index_space_fp32 *>(
          batch.device_spaces),
      batch.descriptor_count, batch.maximum_point_count,
      operands.stride1, operands.stride2, operands.k1, operands.k2,
      device_material);
  mark_device_dirty(cache, field, field_bytes);
  mark_device_dirty(cache, bfast_field, field_bytes);
  if (material.field_u)
    mark_device_dirty(cache, material.field_u, field_bytes);
  if (material.field_conductivity)
    mark_device_dirty(
        cache, material.field_conductivity, field_bytes);
  cuda_curl_calls.fetch_add(1);
  cuda_curl_points.fetch_add(
      static_cast<std::uint64_t>(point_count));
#else
  (void)cache;
  (void)field;
  (void)g1;
  (void)g2;
  (void)bfast_field;
  (void)array_count;
  (void)index_spaces;
  (void)index_space_count;
  (void)stride1;
  (void)stride2;
  (void)k1;
  (void)k2;
  (void)material;
  throw std::runtime_error(
      "Meep CUDA resident batched BFAST called in a CPU-only build");
#endif
}

void resident_step_cylindrical_radial_curl_fp32(
    resident_cache *cache, float *field, const float *radial_operand,
    std::size_t array_count, const index_space_fp32 &index_space,
    std::ptrdiff_t radial_stride, float radial_origin_offset,
    int radial_difference_sign, float dtdx,
    const curl_material_fp32 &material) {
#if MEEP_HAVE_CUDA
  if (!cache || !cache->phase_active)
    throw std::logic_error(
        "Meep CUDA cylindrical curl requires an active resident phase");
  if (!field || !radial_operand)
    throw std::invalid_argument(
        "Meep CUDA cylindrical radial field and operand must be non-null");
  if (radial_stride <= 0 ||
      (radial_difference_sign != -1 && radial_difference_sign != 1))
    throw std::invalid_argument(
        "Meep CUDA cylindrical radial stride/sign is invalid");
  const std::ptrdiff_t neighbor_stride =
      radial_difference_sign * radial_stride;
  const index_space_bounds bounds = validate_curl_index_space(
      index_space, array_count, radial_operand, nullptr, neighbor_stride, 0,
      material);
  const float denominator_min =
      static_cast<float>(bounds.field_min / radial_stride) +
      radial_origin_offset + 0.5f * radial_difference_sign;
  const float denominator_max =
      static_cast<float>(bounds.field_max / radial_stride) +
      radial_origin_offset + 0.5f * radial_difference_sign;
  if (!(denominator_min > 0.0f) || !(denominator_max > 0.0f))
    throw std::out_of_range(
        "Meep CUDA cylindrical radial denominator is nonpositive");

  const prepared_curl_material_fp32 prepared =
      prepare_curl_material_fp32(
          cache, array_count, material, "cylindrical curl field");
  void *device_field =
      ensure_resident_mirror(cache, field, prepared.field_bytes);
  void *device_operand =
      ensure_resident_mirror(
          cache, radial_operand, prepared.field_bytes);
  flush_pending_curl_phase_before_direct_operation();
  meep_cuda::step_cylindrical_radial_curl_structured_fp32(
      static_cast<float *>(device_field),
      static_cast<const float *>(device_operand),
      device_index_space(index_space), bounds.point_count, radial_stride,
      radial_origin_offset, radial_difference_sign, dtdx, prepared.device);
  mark_device_dirty(cache, field, prepared.field_bytes);
  mark_curl_material_dirty(cache, material, prepared.field_bytes);
  cuda_curl_calls.fetch_add(1);
  cuda_curl_points.fetch_add(
      static_cast<std::uint64_t>(bounds.point_count));
#else
  (void)cache;
  (void)field;
  (void)radial_operand;
  (void)array_count;
  (void)index_space;
  (void)radial_stride;
  (void)radial_origin_offset;
  (void)radial_difference_sign;
  (void)dtdx;
  (void)material;
  throw std::runtime_error(
      "Meep CUDA cylindrical radial curl called in a CPU-only build");
#endif
}

void resident_step_cylindrical_imr_fp32(
    resident_cache *cache, float *field, const float *g,
    std::size_t array_count, const index_space_fp32 &index_space,
    int radial_coordinate_start, float coefficient,
    const beta_material_fp32 &material) {
#if MEEP_HAVE_CUDA
  if (!cache || !cache->phase_active)
    throw std::logic_error(
        "Meep CUDA cylindrical i*m/r requires an active resident phase");
  if (!field || !g)
    throw std::invalid_argument(
        "Meep CUDA cylindrical i*m/r field and operand must be non-null");
  if (radial_coordinate_start <= 0)
    throw std::out_of_range(
        "Meep CUDA cylindrical i*m/r radial coordinate is nonpositive");
  const index_space_bounds bounds =
      validate_beta_index_space(index_space, array_count, material);
  const std::size_t field_bytes =
      checked_bytes(array_count, sizeof(float), "cylindrical i*m/r field");
  const std::size_t sigma_bytes =
      checked_bytes(material.sigma_count, sizeof(float), "i*m/r PML-f");
  const std::size_t sigma_u_bytes =
      checked_bytes(material.sigma_u_count, sizeof(float), "i*m/r PML-u");
  void *device_field =
      ensure_resident_mirror(cache, field, field_bytes);
  void *device_g = ensure_resident_mirror(cache, g, field_bytes);
  void *device_sigma_inverse =
      material.sigma_inverse
          ? ensure_resident_mirror(
                cache, material.sigma_inverse, sigma_bytes)
          : nullptr;
  void *device_field_u =
      material.field_u
          ? ensure_resident_mirror(cache, material.field_u, field_bytes)
          : nullptr;
  void *device_sigma_u_inverse =
      material.sigma_u_inverse
          ? ensure_resident_mirror(
                cache, material.sigma_u_inverse, sigma_u_bytes)
          : nullptr;
  void *device_conductivity_inverse =
      material.conductivity_inverse
          ? ensure_resident_mirror(
                cache, material.conductivity_inverse, field_bytes)
          : nullptr;
  void *device_field_conductivity =
      material.field_conductivity
          ? ensure_resident_mirror(
                cache, material.field_conductivity, field_bytes)
          : nullptr;
  const meep_cuda::beta_material_fp32 device_material = {
      static_cast<const float *>(device_sigma_inverse),
      static_cast<float *>(device_field_u),
      static_cast<const float *>(device_sigma_u_inverse),
      static_cast<const float *>(device_conductivity_inverse),
      static_cast<float *>(device_field_conductivity)};
  flush_pending_curl_phase_before_direct_operation();
  meep_cuda::step_cylindrical_imr_structured_fp32(
      static_cast<float *>(device_field),
      static_cast<const float *>(device_g),
      device_index_space(index_space), bounds.point_count,
      radial_coordinate_start, coefficient, device_material);
  mark_device_dirty(cache, field, field_bytes);
  if (material.field_u)
    mark_device_dirty(cache, material.field_u, field_bytes);
  if (material.field_conductivity)
    mark_device_dirty(
        cache, material.field_conductivity, field_bytes);
  cuda_curl_calls.fetch_add(1);
  cuda_curl_points.fetch_add(
      static_cast<std::uint64_t>(bounds.point_count));
#else
  (void)cache;
  (void)field;
  (void)g;
  (void)array_count;
  (void)index_space;
  (void)radial_coordinate_start;
  (void)coefficient;
  (void)material;
  throw std::runtime_error(
      "Meep CUDA cylindrical i*m/r called in a CPU-only build");
#endif
}

void resident_step_cylindrical_axis_fp32(
    resident_cache *cache, float *field, const float *primary,
    const float *secondary, std::size_t array_count,
    const index_space_fp32 &index_space, std::ptrdiff_t neighbor_shift,
    std::ptrdiff_t secondary_offset, float secondary_scale,
    float drive_scale, const curl_material_fp32 &material) {
#if MEEP_HAVE_CUDA
  if (!cache || !cache->phase_active)
    throw std::logic_error(
        "Meep CUDA cylindrical axis requires an active resident phase");
  if (!field || !primary)
    throw std::invalid_argument(
        "Meep CUDA cylindrical axis field and primary must be non-null");
  const index_space_bounds bounds = validate_curl_index_space(
      index_space, array_count, secondary ? primary : nullptr, nullptr,
      neighbor_shift, 0, material);
  if (secondary) {
    const std::ptrdiff_t limit =
        static_cast<std::ptrdiff_t>(array_count);
    if (!shifted_index_is_valid(
            bounds.field_min, secondary_offset, limit) ||
        !shifted_index_is_valid(
            bounds.field_max, secondary_offset, limit))
      throw std::out_of_range(
          "Meep CUDA cylindrical axis secondary index is outside the field");
  }
  else if (neighbor_shift || secondary_offset || secondary_scale)
    throw std::invalid_argument(
        "Meep CUDA single-operand cylindrical axis has secondary parameters");

  const prepared_curl_material_fp32 prepared =
      prepare_curl_material_fp32(
          cache, array_count, material, "cylindrical axis field");
  void *device_field =
      ensure_resident_mirror(cache, field, prepared.field_bytes);
  void *device_primary =
      ensure_resident_mirror(cache, primary, prepared.field_bytes);
  void *device_secondary =
      secondary
          ? ensure_resident_mirror(cache, secondary, prepared.field_bytes)
          : nullptr;
  flush_pending_curl_phase_before_direct_operation();
  meep_cuda::step_cylindrical_axis_structured_fp32(
      static_cast<float *>(device_field),
      static_cast<const float *>(device_primary),
      static_cast<const float *>(device_secondary),
      device_index_space(index_space), bounds.point_count, neighbor_shift,
      secondary_offset, secondary_scale, drive_scale, prepared.device);
  mark_device_dirty(cache, field, prepared.field_bytes);
  mark_curl_material_dirty(cache, material, prepared.field_bytes);
  cuda_curl_calls.fetch_add(1);
  cuda_curl_points.fetch_add(
      static_cast<std::uint64_t>(bounds.point_count));
#else
  (void)cache;
  (void)field;
  (void)primary;
  (void)secondary;
  (void)array_count;
  (void)index_space;
  (void)neighbor_shift;
  (void)secondary_offset;
  (void)secondary_scale;
  (void)drive_scale;
  (void)material;
  throw std::runtime_error(
      "Meep CUDA cylindrical axis called in a CPU-only build");
#endif
}

void resident_zero_span_fp32(
    resident_cache *cache, float *field, std::size_t array_count,
    std::size_t offset, std::size_t count) {
#if MEEP_HAVE_CUDA
  if (!cache || !cache->phase_active)
    throw std::logic_error(
        "Meep CUDA zero span requires an active resident phase");
  if (!field)
    throw std::invalid_argument(
        "Meep CUDA zero span field must be non-null");
  if (offset > array_count || count > array_count - offset)
    throw std::out_of_range(
        "Meep CUDA zero span is outside the field array");
  if (!count) return;
  const std::size_t field_bytes =
      checked_bytes(array_count, sizeof(float), "zero-span field");
  float *device_field = static_cast<float *>(
      ensure_resident_mirror(cache, field, field_bytes));
  flush_pending_curl_phase_before_direct_operation();
  meep_cuda::zero_fp32(device_field + offset, count);
  mark_device_dirty(cache, field, field_bytes);
  cuda_curl_calls.fetch_add(1);
  cuda_curl_points.fetch_add(static_cast<std::uint64_t>(count));
#else
  (void)cache;
  (void)field;
  (void)array_count;
  (void)offset;
  (void)count;
  throw std::runtime_error(
      "Meep CUDA zero span called in a CPU-only build");
#endif
}

void resident_update_eh_fp32(
    resident_cache *cache, float *field, const float *g, const float *g1,
    const float *g2, std::size_t array_count,
    const index_space_fp32 &index_space,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, const update_eh_material_fp32 &material) {
#if MEEP_HAVE_CUDA
  if (!cache || !cache->phase_active)
    throw std::logic_error(
        "Meep CUDA E/H update requires an active resident phase");
  if (!field)
    throw std::invalid_argument("Meep CUDA E/H destination must be non-null");
  if (!g)
    throw std::invalid_argument(
        "Meep CUDA E/H constitutive input must be non-null");
  const index_space_bounds bounds = validate_update_eh_index_space(
      index_space, array_count, g1, g2, field_stride, stride1, stride2,
      material);
  const std::size_t index_count = bounds.point_count;

  const std::size_t field_bytes =
      checked_bytes(array_count, sizeof(float), "E/H field");
  const std::size_t sigma_bytes =
      checked_bytes(material.sigma_count, sizeof(float), "E/H PML");

  void *device_field = ensure_resident_mirror(cache, field, field_bytes);
  void *device_g = ensure_resident_mirror(cache, g, field_bytes);
  void *device_g1 =
      g1 ? ensure_resident_mirror(cache, g1, field_bytes) : nullptr;
  void *device_g2 =
      g2 ? ensure_resident_mirror(cache, g2, field_bytes) : nullptr;
  void *device_inverse =
      material.inverse_susceptibility
          ? ensure_resident_mirror(cache, material.inverse_susceptibility,
                                   field_bytes)
          : nullptr;
  void *device_offdiagonal1 =
      material.offdiagonal1
          ? ensure_resident_mirror(cache, material.offdiagonal1, field_bytes)
          : nullptr;
  void *device_offdiagonal2 =
      material.offdiagonal2
          ? ensure_resident_mirror(cache, material.offdiagonal2, field_bytes)
          : nullptr;
  void *device_chi2 =
      material.chi2
          ? ensure_resident_mirror(cache, material.chi2, field_bytes)
          : nullptr;
  void *device_chi3 =
      material.chi3
          ? ensure_resident_mirror(cache, material.chi3, field_bytes)
          : nullptr;
  void *device_field_w =
      material.field_w
          ? ensure_resident_mirror(cache, material.field_w, field_bytes)
          : nullptr;
  void *device_sigma =
      material.sigma
          ? ensure_resident_mirror(cache, material.sigma, sigma_bytes)
          : nullptr;
  void *device_kappa =
      material.kappa
          ? ensure_resident_mirror(cache, material.kappa, sigma_bytes)
          : nullptr;
  const meep_cuda::update_eh_material_fp32 device_material = {
      static_cast<const float *>(device_inverse),
      static_cast<const float *>(device_offdiagonal1),
      static_cast<const float *>(device_offdiagonal2),
      static_cast<const float *>(device_chi2),
      static_cast<const float *>(device_chi3),
      static_cast<float *>(device_field_w),
      static_cast<const float *>(device_sigma),
      static_cast<const float *>(device_kappa)};
  meep_cuda::update_eh_phase_operation_fp32 operation{};
  operation.field = static_cast<float *>(device_field);
  operation.g = static_cast<const float *>(device_g);
  operation.g1 = static_cast<const float *>(device_g1);
  operation.g2 = static_cast<const float *>(device_g2);
  operation.index_space = device_index_space(index_space);
  operation.point_count = index_count;
  operation.field_stride = field_stride;
  operation.stride1 = stride1;
  operation.stride2 = stride2;
  operation.material = device_material;
  const bool collected = collect_update_eh_phase_operation(
      cache, cache->device_ordinal, operation,
      static_cast<std::uint64_t>(index_count));
  if (!collected)
    meep_cuda::update_eh_structured_fp32(
        static_cast<float *>(device_field),
        static_cast<const float *>(device_g),
        static_cast<const float *>(device_g1),
        static_cast<const float *>(device_g2),
        device_index_space(index_space), index_count, field_stride,
        stride1, stride2, device_material);
  mark_device_dirty(cache, field, field_bytes);
  if (material.field_w)
    mark_device_dirty(cache, material.field_w, field_bytes);
  if (!collected) {
    cuda_update_eh_calls.fetch_add(1);
    cuda_update_eh_points.fetch_add(
        static_cast<std::uint64_t>(index_count));
  }
#else
  (void)cache;
  (void)field;
  (void)g;
  (void)g1;
  (void)g2;
  (void)array_count;
  (void)index_space;
  (void)field_stride;
  (void)stride1;
  (void)stride2;
  (void)material;
  throw std::runtime_error(
      "Meep CUDA resident E/H update called in a CPU-only build");
#endif
}

void resident_update_lorentzian_fp32(
    resident_cache *cache, float *polarization,
    float *polarization_auxiliary, const float *field, const float *field1,
    const float *field2, std::size_t array_count,
    const index_space_fp32 &index_space,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, const lorentzian_material_fp32 &material,
    lorentzian_state_kind state_kind) {
#if MEEP_HAVE_CUDA
  if (!cache || !cache->phase_active)
    throw std::logic_error(
        "Meep CUDA Lorentzian update requires an active resident phase");
  if (!polarization || !polarization_auxiliary)
    throw std::invalid_argument(
        "Meep CUDA Lorentzian polarization arrays must be non-null");
  if (!field)
    throw std::invalid_argument(
        "Meep CUDA Lorentzian field must be non-null");
  const index_space_bounds bounds = validate_lorentzian_index_space(
      index_space, array_count, field1, field2, field_stride, stride1,
      stride2, material);
  const std::size_t index_count = bounds.point_count;

  const std::size_t field_bytes =
      checked_bytes(array_count, sizeof(float), "Lorentzian field");
  void *device_polarization =
      ensure_resident_mirror(cache, polarization, field_bytes);
  void *device_auxiliary =
      ensure_resident_mirror(cache, polarization_auxiliary, field_bytes);
  void *device_field = ensure_resident_mirror(cache, field, field_bytes);
  void *device_field1 =
      field1 ? ensure_resident_mirror(cache, field1, field_bytes) : nullptr;
  void *device_field2 =
      field2 ? ensure_resident_mirror(cache, field2, field_bytes) : nullptr;
  void *device_sigma =
      ensure_resident_mirror(cache, material.sigma, field_bytes);
  void *device_sigma1 =
      material.offdiagonal1
          ? ensure_resident_mirror(cache, material.offdiagonal1, field_bytes)
          : nullptr;
  void *device_sigma2 =
      material.offdiagonal2
          ? ensure_resident_mirror(cache, material.offdiagonal2, field_bytes)
          : nullptr;
  const meep_cuda::lorentzian_material_fp32 device_material = {
      static_cast<const float *>(device_sigma),
      static_cast<const float *>(device_sigma1),
      static_cast<const float *>(device_sigma2),
      material.gamma_inverse,
      material.gamma_previous,
      material.omega_dt_squared,
      material.omega_dt_squared_denominator};
  if (state_kind == lorentzian_state_kind::increment)
    meep_cuda::update_lorentzian_increment_structured_fp32(
        static_cast<float *>(device_polarization),
        static_cast<float *>(device_auxiliary),
        static_cast<const float *>(device_field),
        static_cast<const float *>(device_field1),
        static_cast<const float *>(device_field2),
        device_index_space(index_space), index_count, field_stride, stride1,
        stride2, device_material);
  else
    meep_cuda::update_lorentzian_structured_fp32(
        static_cast<float *>(device_polarization),
        static_cast<float *>(device_auxiliary),
        static_cast<const float *>(device_field),
        static_cast<const float *>(device_field1),
        static_cast<const float *>(device_field2),
        device_index_space(index_space), index_count, field_stride, stride1,
        stride2, device_material);
  mark_device_dirty(cache, polarization, field_bytes);
  mark_device_dirty(cache, polarization_auxiliary, field_bytes);
  cuda_polarization_calls.fetch_add(1);
  cuda_polarization_points.fetch_add(
      static_cast<std::uint64_t>(index_count));
#else
  (void)cache;
  (void)polarization;
  (void)polarization_auxiliary;
  (void)field;
  (void)field1;
  (void)field2;
  (void)array_count;
  (void)index_space;
  (void)field_stride;
  (void)stride1;
  (void)stride2;
  (void)material;
  (void)state_kind;
  throw std::runtime_error(
      "Meep CUDA resident Lorentzian update called in a CPU-only build");
#endif
}

void resident_update_gyrotropic_fp32(
    resident_cache *cache, float *polarization0, float *polarization1,
    float *polarization2, float *previous0, float *previous1,
    float *previous2, const float *field0, const float *field1,
    const float *field2, std::size_t array_count,
    const index_space_fp32 &index_space, std::ptrdiff_t field_stride,
    std::ptrdiff_t stride1, std::ptrdiff_t stride2,
    const gyrotropic_material_fp32 &material) {
#if MEEP_HAVE_CUDA
  if (!cache || !cache->phase_active)
    throw std::logic_error(
        "Meep CUDA gyrotropic update requires an active resident phase");
  if (!polarization0 || !polarization1 || !polarization2 ||
      !previous0 || !previous1 || !previous2)
    throw std::invalid_argument(
        "Meep CUDA gyrotropic polarization arrays must be non-null");
  if (!field0)
    throw std::invalid_argument(
        "Meep CUDA gyrotropic primary field must be non-null");
  const index_space_bounds bounds =
      validate_gyrotropic_index_space(
          index_space, array_count, field1, field2, field_stride, stride1,
          stride2, material);
  const std::size_t index_count = bounds.point_count;
  const std::size_t field_bytes =
      checked_bytes(array_count, sizeof(float), "gyrotropic field");

  void *device_polarization0 =
      ensure_resident_mirror(cache, polarization0, field_bytes);
  void *device_polarization1 =
      ensure_resident_mirror(cache, polarization1, field_bytes);
  void *device_polarization2 =
      ensure_resident_mirror(cache, polarization2, field_bytes);
  void *device_previous0 =
      ensure_resident_mirror(cache, previous0, field_bytes);
  void *device_previous1 =
      ensure_resident_mirror(cache, previous1, field_bytes);
  void *device_previous2 =
      ensure_resident_mirror(cache, previous2, field_bytes);
  void *device_field0 =
      ensure_resident_mirror(cache, field0, field_bytes);
  void *device_field1 =
      field1 ? ensure_resident_mirror(cache, field1, field_bytes) : nullptr;
  void *device_field2 =
      field2 ? ensure_resident_mirror(cache, field2, field_bytes) : nullptr;
  void *device_sigma =
      ensure_resident_mirror(cache, material.sigma, field_bytes);

  meep_cuda::gyrotropic_material_fp32 device_material = {
      static_cast<const float *>(device_sigma),
      material.model,
      {},
      {},
      material.diagonal,
      material.gamma_previous,
      material.omega_dt_squared,
      material.precession_half_dt,
      material.omega_dt,
      material.gamma_dt,
      material.alpha_half,
      material.drive_dt};
  for (int entry = 0; entry < 9; ++entry) {
    device_material.inverse[entry] = material.inverse[entry];
    device_material.gyro[entry] = material.gyro[entry];
  }
  meep_cuda::update_gyrotropic_structured_fp32(
      static_cast<float *>(device_polarization0),
      static_cast<float *>(device_polarization1),
      static_cast<float *>(device_polarization2),
      static_cast<float *>(device_previous0),
      static_cast<float *>(device_previous1),
      static_cast<float *>(device_previous2),
      static_cast<const float *>(device_field0),
      static_cast<const float *>(device_field1),
      static_cast<const float *>(device_field2),
      device_index_space(index_space), index_count, field_stride, stride1,
      stride2, device_material);
  mark_device_dirty(cache, polarization0, field_bytes);
  mark_device_dirty(cache, polarization1, field_bytes);
  mark_device_dirty(cache, polarization2, field_bytes);
  mark_device_dirty(cache, previous0, field_bytes);
  mark_device_dirty(cache, previous1, field_bytes);
  mark_device_dirty(cache, previous2, field_bytes);
  cuda_polarization_calls.fetch_add(1);
  cuda_polarization_points.fetch_add(
      static_cast<std::uint64_t>(index_count));
#else
  (void)cache;
  (void)polarization0;
  (void)polarization1;
  (void)polarization2;
  (void)previous0;
  (void)previous1;
  (void)previous2;
  (void)field0;
  (void)field1;
  (void)field2;
  (void)array_count;
  (void)index_space;
  (void)field_stride;
  (void)stride1;
  (void)stride2;
  (void)material;
  throw std::runtime_error(
      "Meep CUDA resident gyrotropic update called in a CPU-only build");
#endif
}

void resident_update_multilevel_fp32(
    resident_cache *cache, float *population, float *population_scratch,
    const float *gamma, const float *gamma_inverse, const float *alpha,
    std::size_t level_count, std::size_t transition_count,
    std::size_t array_count, float half_dt,
    const index_space_fp32 &centered_space,
    const multilevel_population_channel_fp32 *population_channels,
    std::size_t population_channel_count,
    const multilevel_polarization_channel_fp32 *polarization_channels,
    std::size_t polarization_channel_count,
    const multilevel_transition_fp32 *transitions) {
#if MEEP_HAVE_CUDA
  if (!cache || !cache->phase_active)
    throw std::logic_error(
        "Meep CUDA multilevel update requires an active resident phase");
  if (!population || !population_scratch || !gamma || !gamma_inverse ||
      !alpha || !transitions)
    throw std::invalid_argument(
        "Meep CUDA multilevel state and material arrays must be non-null");
  if (!level_count || !transition_count || !array_count)
    throw std::invalid_argument(
        "Meep CUDA multilevel dimensions must be nonzero");
  if (population_channel_count && !population_channels)
    throw std::invalid_argument(
        "Meep CUDA multilevel population channels must be non-null");
  if (polarization_channel_count && !polarization_channels)
    throw std::invalid_argument(
        "Meep CUDA multilevel polarization channels must be non-null");

  const std::size_t population_count =
      checked_product(array_count, level_count, "multilevel population");
  const std::size_t matrix_count =
      checked_product(level_count, level_count, "multilevel matrix");
  const std::size_t alpha_count =
      checked_product(level_count, transition_count, "multilevel alpha");
  const std::size_t transition_pair_count =
      checked_product(transition_count, 2, "multilevel transition pair");
  const std::size_t polarization_count =
      checked_product(array_count, transition_pair_count,
                      "multilevel polarization");
  const std::size_t field_bytes =
      checked_bytes(array_count, sizeof(float), "multilevel field");
  const std::size_t population_bytes =
      checked_bytes(population_count, sizeof(float),
                    "multilevel population");
  const std::size_t matrix_bytes =
      checked_bytes(matrix_count, sizeof(float), "multilevel matrix");
  const std::size_t alpha_bytes =
      checked_bytes(alpha_count, sizeof(float), "multilevel alpha");
  const std::size_t polarization_bytes =
      checked_bytes(polarization_count, sizeof(float),
                    "multilevel polarization");

  const index_space_bounds centered_bounds =
      validate_index_space(centered_space, array_count);
  if (centered_bounds.coefficient_min != -1 ||
      centered_bounds.coefficient2_min != -1)
    throw std::invalid_argument(
        "Meep CUDA multilevel centered space has unused coefficient "
        "indices");
  const std::ptrdiff_t field_limit =
      static_cast<std::ptrdiff_t>(array_count);

  for (std::size_t index = 0; index < population_channel_count; ++index) {
    const multilevel_population_channel_fp32 &channel =
        population_channels[index];
    if (!channel.field || !channel.previous_field ||
        !channel.polarization)
      throw std::invalid_argument(
          "Meep CUDA multilevel population channel pointers must be "
          "non-null");
    validate_four_point_range(
        centered_bounds.field_min, centered_bounds.field_max,
        channel.centered_offset1, channel.centered_offset2, field_limit,
        "multilevel population channel");
  }

  std::size_t maximum_polarization_point_count = 0;
  for (std::size_t index = 0; index < polarization_channel_count; ++index) {
    const multilevel_polarization_channel_fp32 &channel =
        polarization_channels[index];
    if (!channel.polarization || !channel.field || !channel.sigma)
      throw std::invalid_argument(
          "Meep CUDA multilevel polarization channel pointers must be "
          "non-null");
    if (channel.direction < 0 || channel.direction >= 5)
      throw std::invalid_argument(
          "Meep CUDA multilevel polarization direction is invalid");
    const index_space_bounds bounds =
        validate_index_space(channel.index_space, array_count);
    if (bounds.point_count != channel.point_count)
      throw std::invalid_argument(
          "Meep CUDA multilevel polarization point count does not match "
          "its index space");
    if (bounds.coefficient_min != -1 ||
        bounds.coefficient2_min != -1)
      throw std::invalid_argument(
          "Meep CUDA multilevel polarization space has unused coefficient "
          "indices");
    validate_four_point_range(
        bounds.field_min, bounds.field_max,
        channel.population_offset1, channel.population_offset2, field_limit,
        "multilevel population inversion");
    maximum_polarization_point_count =
        std::max(maximum_polarization_point_count, bounds.point_count);
  }
  for (std::size_t transition = 0; transition < transition_count;
       ++transition) {
    if (transitions[transition].upper_level < 0 ||
        transitions[transition].lower_level < 0 ||
        static_cast<std::size_t>(
            transitions[transition].upper_level) >= level_count ||
        static_cast<std::size_t>(
            transitions[transition].lower_level) >= level_count)
      throw std::invalid_argument(
          "Meep CUDA multilevel transition level is out of range");
  }
  std::size_t polarization_points = 0;
  for (std::size_t index = 0; index < polarization_channel_count; ++index) {
    const std::size_t channel_points =
        checked_product(polarization_channels[index].point_count,
                        transition_count, "multilevel point");
    if (polarization_points >
        std::numeric_limits<std::size_t>::max() - channel_points)
      throw std::overflow_error(
          "Meep CUDA multilevel point count overflow");
    polarization_points += channel_points;
  }
  if (centered_bounds.point_count >
      std::numeric_limits<std::size_t>::max() - polarization_points)
    throw std::overflow_error(
        "Meep CUDA multilevel point count overflow");
  const std::size_t total_operation_points =
      centered_bounds.point_count + polarization_points;

  void *device_population =
      ensure_resident_mirror(cache, population, population_bytes);
  void *device_population_scratch =
      ensure_resident_mirror(cache, population_scratch, population_bytes);
  void *device_gamma =
      ensure_resident_mirror(cache, gamma, matrix_bytes);
  void *device_gamma_inverse =
      ensure_resident_mirror(cache, gamma_inverse, matrix_bytes);
  void *device_alpha =
      ensure_resident_mirror(cache, alpha, alpha_bytes);

  std::vector<meep_cuda::multilevel_population_channel_fp32>
      device_population_channels;
  device_population_channels.reserve(population_channel_count);
  for (std::size_t index = 0; index < population_channel_count; ++index) {
    const multilevel_population_channel_fp32 &channel =
        population_channels[index];
    device_population_channels.push_back(
        {static_cast<const float *>(
             ensure_resident_mirror(cache, channel.field, field_bytes)),
         static_cast<const float *>(ensure_resident_mirror(
             cache, channel.previous_field, field_bytes)),
         static_cast<const float *>(ensure_resident_mirror(
             cache, channel.polarization, polarization_bytes)),
         channel.centered_offset1, channel.centered_offset2});
  }

  std::vector<meep_cuda::multilevel_polarization_channel_fp32>
      device_polarization_channels;
  device_polarization_channels.reserve(polarization_channel_count);
  for (std::size_t index = 0; index < polarization_channel_count; ++index) {
    const multilevel_polarization_channel_fp32 &channel =
        polarization_channels[index];
    device_polarization_channels.push_back(
        {static_cast<float *>(ensure_resident_mirror(
             cache, channel.polarization, polarization_bytes)),
         static_cast<const float *>(
             ensure_resident_mirror(cache, channel.field, field_bytes)),
         static_cast<const float *>(
             ensure_resident_mirror(cache, channel.sigma, field_bytes)),
         device_index_space(channel.index_space), channel.point_count,
         channel.population_offset1, channel.population_offset2,
         channel.direction});
  }

  std::vector<meep_cuda::multilevel_transition_fp32> device_transitions;
  device_transitions.reserve(transition_count);
  for (std::size_t transition = 0; transition < transition_count;
       ++transition) {
    meep_cuda::multilevel_transition_fp32 converted = {
        transitions[transition].upper_level,
        transitions[transition].lower_level,
        transitions[transition].population_damping,
        transitions[transition].diagonal,
        transitions[transition].gamma_previous,
        transitions[transition].gamma_inverse,
        {}};
    for (int direction = 0; direction < 5; ++direction)
      converted.drive_scale[direction] =
          transitions[transition].drive_scale[direction];
    device_transitions.push_back(converted);
  }

  const auto same_space = [](const index_space_fp32 &left,
                             const index_space_fp32 &right) {
    return left.field_start == right.field_start &&
           left.extent1 == right.extent1 &&
           left.extent2 == right.extent2 &&
           left.extent3 == right.extent3 &&
           left.field_stride1 == right.field_stride1 &&
           left.field_stride2 == right.field_stride2 &&
           left.field_stride3 == right.field_stride3 &&
           left.coefficient_start == right.coefficient_start &&
           left.coefficient_stride1 == right.coefficient_stride1 &&
           left.coefficient_stride2 == right.coefficient_stride2 &&
           left.coefficient_stride3 == right.coefficient_stride3 &&
           left.coefficient2_start == right.coefficient2_start &&
           left.coefficient2_stride1 == right.coefficient2_stride1 &&
           left.coefficient2_stride2 == right.coefficient2_stride2 &&
           left.coefficient2_stride3 == right.coefficient2_stride3;
  };
  const auto same_population_channel =
      [](const multilevel_population_channel_fp32 &left,
         const multilevel_population_channel_fp32 &right) {
        return left.field == right.field &&
               left.previous_field == right.previous_field &&
               left.polarization == right.polarization &&
               left.centered_offset1 == right.centered_offset1 &&
               left.centered_offset2 == right.centered_offset2;
      };
  const auto same_polarization_channel =
      [&same_space](const multilevel_polarization_channel_fp32 &left,
                    const multilevel_polarization_channel_fp32 &right) {
        return left.polarization == right.polarization &&
               left.field == right.field && left.sigma == right.sigma &&
               same_space(left.index_space, right.index_space) &&
               left.point_count == right.point_count &&
               left.population_offset1 == right.population_offset1 &&
               left.population_offset2 == right.population_offset2 &&
               left.direction == right.direction;
      };
  const auto same_transition =
      [](const multilevel_transition_fp32 &left,
         const multilevel_transition_fp32 &right) {
        if (left.upper_level != right.upper_level ||
            left.lower_level != right.lower_level ||
            left.population_damping != right.population_damping ||
            left.diagonal != right.diagonal ||
            left.gamma_previous != right.gamma_previous ||
            left.gamma_inverse != right.gamma_inverse)
          return false;
        for (int direction = 0; direction < 5; ++direction)
          if (left.drive_scale[direction] !=
              right.drive_scale[direction])
            return false;
        return true;
      };

  auto found = cache->multilevel_plans.find(population);
  bool plan_matches =
      found != cache->multilevel_plans.end() &&
      found->second.allocation_generation ==
          cache->allocation_generation &&
      found->second.maximum_polarization_point_count ==
          maximum_polarization_point_count &&
      found->second.population_channels.size() ==
          population_channel_count &&
      found->second.polarization_channels.size() ==
          polarization_channel_count &&
      found->second.transitions.size() == transition_count;
  for (std::size_t index = 0; index < population_channel_count &&
                              plan_matches;
       ++index)
    plan_matches = same_population_channel(
        found->second.population_channels[index],
        population_channels[index]);
  for (std::size_t index = 0; index < polarization_channel_count &&
                              plan_matches;
       ++index)
    plan_matches = same_polarization_channel(
        found->second.polarization_channels[index],
        polarization_channels[index]);
  for (std::size_t index = 0; index < transition_count && plan_matches;
       ++index)
    plan_matches = same_transition(
        found->second.transitions[index], transitions[index]);

  if (!plan_matches) {
    // Insert an allocation-free placeholder before acquiring any raw CUDA
    // buffers. unordered_map insertion/rehash can allocate and throw; doing it
    // first prevents a host allocation failure from orphaning device memory.
    const bool replacing_existing =
        found != cache->multilevel_plans.end();
    if (!replacing_existing)
      found = cache->multilevel_plans
                  .emplace(population,
                           resident_cache::multilevel_plan())
                  .first;
    resident_cache::multilevel_plan created;
    const std::size_t population_descriptor_bytes = checked_bytes(
        population_channel_count,
        sizeof(meep_cuda::multilevel_population_channel_fp32),
        "multilevel population descriptors");
    const std::size_t polarization_descriptor_bytes = checked_bytes(
        polarization_channel_count,
        sizeof(meep_cuda::multilevel_polarization_channel_fp32),
        "multilevel polarization descriptors");
    const std::size_t transition_descriptor_bytes = checked_bytes(
        transition_count, sizeof(meep_cuda::multilevel_transition_fp32),
        "multilevel transition descriptors");
    std::uint64_t created_allocation_count = 0;
    try {
      if (population_descriptor_bytes) {
        created.device_population_channels =
            meep_cuda::allocate_device_bytes(
                population_descriptor_bytes);
        ++created_allocation_count;
        device_buffer_allocations.fetch_add(1);
        live_device_buffer_count().fetch_add(1);
        meep_cuda::copy_to_device(
            created.device_population_channels,
            device_population_channels.data(),
            population_descriptor_bytes);
      }
      if (polarization_descriptor_bytes) {
        created.device_polarization_channels =
            meep_cuda::allocate_device_bytes(
                polarization_descriptor_bytes);
        ++created_allocation_count;
        device_buffer_allocations.fetch_add(1);
        live_device_buffer_count().fetch_add(1);
        meep_cuda::copy_to_device(
            created.device_polarization_channels,
            device_polarization_channels.data(),
            polarization_descriptor_bytes);
      }
      created.device_transitions =
          meep_cuda::allocate_device_bytes(transition_descriptor_bytes);
      ++created_allocation_count;
      device_buffer_allocations.fetch_add(1);
      live_device_buffer_count().fetch_add(1);
      meep_cuda::copy_to_device(
          created.device_transitions, device_transitions.data(),
          transition_descriptor_bytes);
      if (population_channel_count)
        created.population_channels.assign(
            population_channels,
            population_channels + population_channel_count);
      if (polarization_channel_count)
        created.polarization_channels.assign(
            polarization_channels,
            polarization_channels + polarization_channel_count);
      created.transitions.assign(
          transitions, transitions + transition_count);
      created.maximum_polarization_point_count =
          maximum_polarization_point_count;
      created.allocation_generation = cache->allocation_generation;
      if (fail_multilevel_plan_commit_for_testing.exchange(
              false, std::memory_order_acq_rel))
        throw std::bad_alloc();
    }
    catch (...) {
      meep_cuda::free_device(created.device_population_channels);
      meep_cuda::free_device(created.device_polarization_channels);
      meep_cuda::free_device(created.device_transitions);
      live_device_buffer_count().fetch_sub(created_allocation_count);
      if (!replacing_existing)
        cache->multilevel_plans.erase(found);
      throw;
    }

    if (replacing_existing) {
      if (found->second.device_population_channels)
        live_device_buffer_count().fetch_sub(1);
      if (found->second.device_polarization_channels)
        live_device_buffer_count().fetch_sub(1);
      if (found->second.device_transitions)
        live_device_buffer_count().fetch_sub(1);
      meep_cuda::free_device(found->second.device_population_channels);
      meep_cuda::free_device(found->second.device_polarization_channels);
      meep_cuda::free_device(found->second.device_transitions);
    }
    found->second.device_population_channels =
        created.device_population_channels;
    found->second.device_polarization_channels =
        created.device_polarization_channels;
    found->second.device_transitions = created.device_transitions;
    created.device_population_channels = nullptr;
    created.device_polarization_channels = nullptr;
    created.device_transitions = nullptr;
    found->second.population_channels.swap(
        created.population_channels);
    found->second.polarization_channels.swap(
        created.polarization_channels);
    found->second.transitions.swap(created.transitions);
    found->second.maximum_polarization_point_count =
        created.maximum_polarization_point_count;
    found->second.allocation_generation =
        created.allocation_generation;
    host_to_device_bytes.fetch_add(
        static_cast<std::uint64_t>(
            population_descriptor_bytes +
            polarization_descriptor_bytes +
            transition_descriptor_bytes));
  }
  else {
    device_buffer_reuses.fetch_add(
        (found->second.device_population_channels ? 1u : 0u) +
        (found->second.device_polarization_channels ? 1u : 0u) +
        (found->second.device_transitions ? 1u : 0u));
  }

  const resident_cache::multilevel_plan &plan = found->second;
  meep_cuda::update_multilevel_structured_fp32(
      static_cast<float *>(device_population),
      static_cast<float *>(device_population_scratch),
      static_cast<const float *>(device_gamma),
      static_cast<const float *>(device_gamma_inverse),
      static_cast<const float *>(device_alpha), level_count,
      transition_count, array_count, half_dt,
      device_index_space(centered_space), centered_bounds.point_count,
      static_cast<const meep_cuda::multilevel_population_channel_fp32 *>(
          plan.device_population_channels),
      population_channel_count,
      static_cast<
          const meep_cuda::multilevel_polarization_channel_fp32 *>(
          plan.device_polarization_channels),
      polarization_channel_count,
      static_cast<const meep_cuda::multilevel_transition_fp32 *>(
          plan.device_transitions),
      plan.maximum_polarization_point_count);

  mark_device_dirty(cache, population, population_bytes);
  mark_device_dirty(cache, population_scratch, population_bytes);
  for (std::size_t index = 0; index < polarization_channel_count; ++index)
    mark_device_dirty(
        cache, polarization_channels[index].polarization,
        polarization_bytes);
  cuda_polarization_calls.fetch_add(1);
  cuda_polarization_points.fetch_add(
      static_cast<std::uint64_t>(total_operation_points));
#else
  (void)cache;
  (void)population;
  (void)population_scratch;
  (void)gamma;
  (void)gamma_inverse;
  (void)alpha;
  (void)level_count;
  (void)transition_count;
  (void)array_count;
  (void)half_dt;
  (void)centered_space;
  (void)population_channels;
  (void)population_channel_count;
  (void)polarization_channels;
  (void)polarization_channel_count;
  (void)transitions;
  throw std::runtime_error(
      "Meep CUDA resident multilevel update called in a CPU-only build");
#endif
}

void resident_copy_fp32(resident_cache *cache, float *destination,
                        const float *source, std::size_t count) {
#if MEEP_HAVE_CUDA
  if (!cache || !cache->phase_active)
    throw std::logic_error(
        "Meep CUDA copy requires an active resident phase");
  if (!destination || !source)
    throw std::invalid_argument(
        "Meep CUDA copy pointers must be non-null");
  if (count == 0) return;
  const std::size_t bytes = checked_bytes(count, sizeof(float), "field copy");
  void *device_destination =
      ensure_resident_mirror(cache, destination, bytes);
  void *device_source = ensure_resident_mirror(cache, source, bytes);
  meep_cuda::copy_fp32(static_cast<float *>(device_destination),
                      static_cast<const float *>(device_source), count);
  mark_device_dirty(cache, destination, bytes);
#else
  (void)cache;
  (void)destination;
  (void)source;
  (void)count;
  throw std::runtime_error(
      "Meep CUDA resident copy called in a CPU-only build");
#endif
}

void resident_subtract_fp32(resident_cache *cache, float *destination,
                            const float *source, std::size_t count) {
#if MEEP_HAVE_CUDA
  if (!cache || !cache->phase_active)
    throw std::logic_error(
        "Meep CUDA subtraction requires an active resident phase");
  if (!destination || !source)
    throw std::invalid_argument(
        "Meep CUDA subtraction pointers must be non-null");
  if (count == 0) return;
  const std::size_t bytes =
      checked_bytes(count, sizeof(float), "field subtraction");
  const resolved_resident_range source_resolved =
      resolve_resident_range(
          cache, source, bytes, true,
          "Meep CUDA subtraction source");
  const resolved_resident_range destination_resolved =
      resolve_resident_range(
          cache, destination, bytes, true,
          "Meep CUDA subtraction destination");
  const std::uintptr_t source_begin =
      reinterpret_cast<std::uintptr_t>(source);
  const std::uintptr_t destination_begin =
      reinterpret_cast<std::uintptr_t>(destination);
  if (source_begin != destination_begin &&
      source_begin < destination_begin + bytes &&
      destination_begin < source_begin + bytes)
    throw std::invalid_argument(
        "Meep CUDA subtraction source and destination ranges overlap");
  if (destination_resolved.mirror &&
      (destination_resolved.host_base != destination ||
       destination_resolved.mirror->bytes != bytes))
    throw std::invalid_argument(
        "Meep CUDA subtraction destination is a subspan of another "
        "resident mirror");
  void *device_destination =
      ensure_resident_mirror(cache, destination, bytes);
  void *device_source = source_resolved.mirror
                            ? static_cast<unsigned char *>(
                                  source_resolved.mirror->device_pointer) +
                                  source_resolved.byte_offset
                            : nullptr;
  if (device_source) {
    device_buffer_reuses.fetch_add(1);
    host_to_device_bytes_avoided.fetch_add(
        static_cast<std::uint64_t>(bytes));
  }
  else
    device_source = ensure_resident_mirror(cache, source, bytes);
  meep_cuda::subtract_fp32(static_cast<float *>(device_destination),
                          static_cast<const float *>(device_source), count);
  mark_device_dirty(cache, destination, bytes);
#else
  (void)cache;
  (void)destination;
  (void)source;
  (void)count;
  throw std::runtime_error(
      "Meep CUDA resident subtraction called in a CPU-only build");
#endif
}

void resident_indexed_subtract_fp32(
    resident_cache *cache, float *destination, std::size_t array_count,
    const indexed_value_fp32 *updates, std::size_t update_count) {
#if MEEP_HAVE_CUDA
  if (!cache || !cache->phase_active)
    throw std::logic_error(
        "Meep CUDA indexed subtraction requires an active resident phase");
  if (!destination)
    throw std::invalid_argument(
        "Meep CUDA indexed subtraction destination must be non-null");
  if (!updates && update_count)
    throw std::invalid_argument(
        "Meep CUDA indexed subtraction updates must be non-null");
  if (array_count >
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
    throw std::overflow_error(
        "Meep CUDA indexed subtraction array is too large");
  for (std::size_t j = 0; j < update_count; ++j)
    if (updates[j].index < 0 ||
        static_cast<std::size_t>(updates[j].index) >= array_count)
      throw std::out_of_range(
          "Meep CUDA indexed subtraction index is outside the field array");
  if (update_count == 0) return;

  static_assert(sizeof(indexed_value_fp32) ==
                    sizeof(meep_cuda::indexed_value_fp32),
                "Meep/CUDA indexed-value size mismatch");
  static_assert(alignof(indexed_value_fp32) ==
                    alignof(meep_cuda::indexed_value_fp32),
                "Meep/CUDA indexed-value alignment mismatch");
  static_assert(
      offsetof(indexed_value_fp32, index) ==
              offsetof(meep_cuda::indexed_value_fp32, index) &&
          offsetof(indexed_value_fp32, value) ==
              offsetof(meep_cuda::indexed_value_fp32, value),
      "Meep/CUDA indexed-value layout mismatch");

  const std::size_t field_bytes =
      checked_bytes(array_count, sizeof(float), "indexed destination");
  const std::size_t update_bytes =
      checked_bytes(update_count, sizeof(indexed_value_fp32),
                    "indexed updates");
  void *device_destination =
      ensure_resident_mirror(cache, destination, field_bytes);
  void *device_updates = ensure_index_buffer(cache, update_bytes);
  meep_cuda::copy_to_device(device_updates, updates, update_bytes);
  host_to_device_bytes.fetch_add(static_cast<std::uint64_t>(update_bytes));
  meep_cuda::indexed_subtract_fp32(
      static_cast<float *>(device_destination),
      static_cast<const meep_cuda::indexed_value_fp32 *>(device_updates),
      update_count);
  mark_device_dirty(cache, destination, field_bytes);
#else
  (void)cache;
  (void)destination;
  (void)array_count;
  (void)updates;
  (void)update_count;
  throw std::runtime_error(
      "Meep CUDA resident indexed subtraction called in a CPU-only build");
#endif
}

bool resident_indexed_source_subtract_fp32(
    resident_cache *cache, float *destination, std::size_t array_count,
    const std::ptrdiff_t *indices, const float *amplitudes_real_imag,
    std::size_t source_count, const float *conductivity_inverse,
    complex_value_fp32 time_scale, bool imaginary_component) {
#if MEEP_HAVE_CUDA
  if (!cache || !cache->phase_active)
    throw std::logic_error(
        "Meep CUDA indexed source requires an active resident phase");
  if (!destination)
    throw std::invalid_argument(
        "Meep CUDA indexed source destination must be non-null");
  if ((!indices || !amplitudes_real_imag) && source_count)
    throw std::invalid_argument(
        "Meep CUDA indexed source profile must be non-null");
  if (array_count >
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
    throw std::overflow_error(
        "Meep CUDA indexed source array is too large");
  if (source_count == 0) return false;

  const auto validated = cache->validated_source_profiles.find(indices);
  if (validated == cache->validated_source_profiles.end() ||
      validated->second.first != source_count ||
      validated->second.second != array_count) {
    for (std::size_t j = 0; j < source_count; ++j)
      if (indices[j] < 0 ||
          static_cast<std::size_t>(indices[j]) >= array_count)
        throw std::out_of_range(
            "Meep CUDA indexed source is outside the field array");
    cache->validated_source_profiles[indices] =
        std::make_pair(source_count, array_count);
  }

  static_assert(sizeof(complex_value_fp32) ==
                    sizeof(meep_cuda::complex_value_fp32),
                "Meep/CUDA source amplitude size mismatch");
  static_assert(alignof(complex_value_fp32) ==
                    alignof(meep_cuda::complex_value_fp32),
                "Meep/CUDA source amplitude alignment mismatch");

  const std::size_t field_bytes =
      checked_bytes(array_count, sizeof(float), "source destination");
  const std::size_t index_bytes =
      checked_bytes(source_count, sizeof(std::ptrdiff_t), "source indices");
  if (source_count > std::numeric_limits<std::size_t>::max() / 2)
    throw std::overflow_error("Meep CUDA source amplitude count overflow");
  const std::size_t amplitude_bytes =
      checked_bytes(2 * source_count, sizeof(float), "source amplitudes");

  void *device_destination =
      ensure_resident_mirror(cache, destination, field_bytes);
  void *device_indices = ensure_resident_mirror(
      cache, reinterpret_cast<const float *>(indices), index_bytes, nullptr,
      false);
  void *device_amplitudes =
      ensure_resident_mirror(cache, amplitudes_real_imag, amplitude_bytes,
                             nullptr, false);
  void *device_conductivity =
      conductivity_inverse
          ? ensure_resident_mirror(cache, conductivity_inverse, field_bytes)
          : nullptr;
  const meep_cuda::indexed_source_phase_operation_fp32 operation = {
      static_cast<float *>(device_destination),
      static_cast<const std::ptrdiff_t *>(device_indices),
      static_cast<const meep_cuda::complex_value_fp32 *>(device_amplitudes),
      static_cast<const float *>(device_conductivity), source_count, 0,
      imaginary_component};
  const bool collected = collect_source_phase_operation(
      cache, cache->device_ordinal, operation,
      {time_scale.real, time_scale.imag},
      static_cast<std::uint64_t>(source_count));
  if (!collected)
    meep_cuda::indexed_source_subtract_fp32(
        operation.destination, operation.indices, operation.amplitudes,
        operation.conductivity_inverse, operation.point_count,
        {time_scale.real, time_scale.imag},
        operation.imaginary_component);
  mark_device_dirty(cache, destination, field_bytes);
  return collected;
#else
  (void)cache;
  (void)destination;
  (void)array_count;
  (void)indices;
  (void)amplitudes_real_imag;
  (void)source_count;
  (void)conductivity_inverse;
  (void)time_scale;
  (void)imaginary_component;
  throw std::runtime_error(
      "Meep CUDA resident indexed source called in a CPU-only build");
#endif
}

void resident_ensure_mirror_fp32(resident_cache *cache, float *host_pointer,
                                 std::size_t count) {
#if MEEP_HAVE_CUDA
  if (!cache || !cache->phase_active)
    throw std::logic_error(
        "Meep CUDA mirror preparation requires an active resident phase");
  if (!host_pointer)
    throw std::invalid_argument(
        "Meep CUDA mirror preparation pointer must be non-null");
  if (count == 0) return;
  const std::size_t bytes =
      checked_bytes(count, sizeof(float), "prepared resident mirror");
  (void)ensure_resident_mirror(cache, host_pointer, bytes);
#else
  (void)cache;
  (void)host_pointer;
  (void)count;
  throw std::runtime_error(
      "Meep CUDA mirror preparation called in a CPU-only build");
#endif
}

void resident_ensure_span_fp32(resident_cache *cache, float *host_pointer,
                               std::size_t count) {
#if MEEP_HAVE_CUDA
  if (!cache || !cache->phase_active)
    throw std::logic_error(
        "Meep CUDA span preparation requires an active resident phase");
  if (!host_pointer)
    throw std::invalid_argument(
        "Meep CUDA span preparation pointer must be non-null");
  if (count == 0) return;
  const std::size_t bytes =
      checked_bytes(count, sizeof(float), "prepared resident span");
  const resolved_resident_range resolved = resolve_resident_range(
      cache, host_pointer, bytes, true,
      "Meep CUDA prepared resident span");
  if (resolved.mirror) {
    device_buffer_reuses.fetch_add(1);
    if (!resolved.mirror->device_dirty &&
        resolved.mirror->uploaded_epoch == 0) {
      // An in-place host algorithm can invalidate a full resident field while
      // a boundary request addresses only one scalar inside it. Refresh the
      // complete owning mirror before a cached device-address plan is replayed.
      meep_cuda::copy_to_device(
          resolved.mirror->device_pointer, resolved.host_base,
          resolved.mirror->bytes);
      resolved.mirror->uploaded_epoch = cache->epoch ? cache->epoch : 1;
      host_to_device_bytes.fetch_add(
          static_cast<std::uint64_t>(resolved.mirror->bytes));
    }
    else {
      host_to_device_bytes_avoided.fetch_add(
          static_cast<std::uint64_t>(bytes));
    }
    return;
  }
  (void)ensure_resident_mirror(cache, host_pointer, bytes);
#else
  (void)cache;
  (void)host_pointer;
  (void)count;
  throw std::runtime_error(
      "Meep CUDA span preparation called in a CPU-only build");
#endif
}

bool resident_boundary_plan_ready(const void *owner,
                                  std::size_t plan_token) noexcept {
#if MEEP_HAVE_CUDA
  if (!owner) return false;
  boundary_exchange_registry &registry = get_boundary_exchange_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto owner_entry = registry.owners.find(owner);
  if (owner_entry == registry.owners.end()) return false;
  const auto plan =
      owner_entry->second.operation_plans.find(plan_token);
  return plan != owner_entry->second.operation_plans.end() &&
         plan->second->stable_reuses >= 2 &&
         boundary_operation_plan_generations_match(plan->second);
#else
  (void)owner;
  (void)plan_token;
  return false;
#endif
}

bool resident_launch_boundary_plan(const void *owner,
                                   std::size_t plan_token) {
#if MEEP_HAVE_CUDA
  if (!owner)
    throw std::invalid_argument(
        "Meep CUDA boundary-plan owner must be non-null");
  boundary_exchange_registry &registry = get_boundary_exchange_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto owner_entry = registry.owners.find(owner);
  if (owner_entry == registry.owners.end())
    throw std::logic_error("Meep CUDA boundary plan is not prepared");
  const auto found =
      owner_entry->second.operation_plans.find(plan_token);
  if (found == owner_entry->second.operation_plans.end())
    throw std::logic_error("Meep CUDA boundary plan is not prepared");
  boundary_exchange_owner::operation_plan *plan = found->second;
  if (!boundary_operation_plan_generations_match(plan))
    throw std::logic_error(
        "Meep CUDA boundary plan references stale resident storage");
  if (plan->operation_count == 0) return false;
  mark_boundary_plan_destinations_dirty(plan);

  const std::size_t operation_bytes =
      checked_bytes(plan->operation_count,
                    sizeof(meep_cuda::boundary_operation_fp32),
                    "boundary operations");
  meep_cuda::select_device(plan->device_ordinal);
  if (plan->all_sources_null &&
      std::getenv("MEEP_GPU_DISABLE_DIRECT_BOUNDARY_ZERO") == nullptr)
    meep_cuda::zero_boundary_fp32(
        static_cast<const meep_cuda::boundary_operation_fp32 *>(
            plan->device_pointer),
        plan->operation_count);
  else if (plan->nonalias_safe &&
           std::getenv("MEEP_GPU_DISABLE_DIRECT_LOCAL_BOUNDARY") == nullptr)
    meep_cuda::apply_nonalias_boundary_fp32(
        static_cast<const meep_cuda::boundary_operation_fp32 *>(
            plan->device_pointer),
        plan->operation_count);
  else if (plan->execution_graph &&
      std::getenv("MEEP_GPU_DISABLE_BOUNDARY_GRAPH") == nullptr)
    meep_cuda::launch_boundary_graph_fp32(plan->execution_graph);
  else
    meep_cuda::apply_boundary_fp32(
        static_cast<const meep_cuda::boundary_operation_fp32 *>(
            plan->device_pointer),
        reinterpret_cast<float *>(
            static_cast<unsigned char *>(plan->device_pointer) +
            operation_bytes),
        plan->operation_count);
  cuda_boundary_calls.fetch_add(1);
  cuda_boundary_points.fetch_add(
      static_cast<std::uint64_t>(plan->scalar_count));
  device_buffer_reuses.fetch_add(1);
  return true;
#else
  (void)owner;
  (void)plan_token;
  throw std::runtime_error(
      "Meep CUDA resident boundary called in a CPU-only build");
#endif
}

void resident_apply_boundary_fp32(
    const void *owner, std::size_t plan_token,
    const boundary_operation_fp32 *operations,
    std::size_t operation_count) {
#if MEEP_HAVE_CUDA
  if (!owner)
    throw std::invalid_argument(
        "Meep CUDA boundary-plan owner must be non-null");
  if (!operations && operation_count)
    throw std::invalid_argument(
        "Meep CUDA boundary operations must be non-null");
  if (operation_count == 0) {
    const int ordinal = active_cuda_device_ordinal();
    boundary_exchange_registry &registry = get_boundary_exchange_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    boundary_exchange_owner &plans = registry.owners[owner];
    auto found = plans.operation_plans.find(plan_token);
    if (found != plans.operation_plans.end() &&
        found->second->operation_count != 0) {
      release_boundary_operation_plan(found->second);
      plans.operation_plans.erase(found);
      found = plans.operation_plans.end();
    }
    if (found == plans.operation_plans.end()) {
      std::unique_ptr<boundary_exchange_owner::operation_plan> created(
          new boundary_exchange_owner::operation_plan);
      created->device_ordinal = ordinal;
      created->stable_reuses = 2;
      created->generation = next_boundary_plan_generation();
      boundary_exchange_owner::operation_plan *plan = created.get();
      plans.operation_plans.emplace(plan_token, plan);
      created.release();
    }
    return;
  }

  resident_cache *buffer_cache = operations[0].destination_cache;
  if (!buffer_cache || !buffer_cache->phase_active)
    throw std::logic_error(
        "Meep CUDA boundary destination requires an active resident phase");
  std::size_t scalar_count = 0;

  const std::size_t operation_bytes =
      checked_bytes(operation_count,
                    sizeof(meep_cuda::boundary_operation_fp32),
                    "boundary operations");
  if (operation_count > std::numeric_limits<std::size_t>::max() / 2)
    throw std::overflow_error("Meep CUDA boundary staging count overflow");
  const std::size_t staging_bytes =
      checked_bytes(2 * operation_count, sizeof(float),
                    "boundary staging values");
  if (operation_bytes > std::numeric_limits<std::size_t>::max() -
                            staging_bytes)
    throw std::overflow_error("Meep CUDA boundary scratch size overflow");

  boundary_exchange_registry &registry = get_boundary_exchange_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  boundary_exchange_owner &plans = registry.owners[owner];
  auto found = plans.operation_plans.find(plan_token);
  if (found != plans.operation_plans.end() &&
      std::getenv("MEEP_GPU_DISABLE_BOUNDARY_PLAN_CACHE") != nullptr) {
    release_boundary_operation_plan(found->second);
    plans.operation_plans.erase(found);
    meep_cuda::select_device(buffer_cache->device_ordinal);
    found = plans.operation_plans.end();
  }
  if (found != plans.operation_plans.end() &&
      !boundary_operation_plan_matches(
          found->second, operations, operation_count)) {
    release_boundary_operation_plan(found->second);
    plans.operation_plans.erase(found);
    meep_cuda::select_device(buffer_cache->device_ordinal);
    found = plans.operation_plans.end();
  }
  if (found != plans.operation_plans.end() &&
      !boundary_operation_plan_generations_match(found->second)) {
    release_boundary_operation_plan(found->second);
    plans.operation_plans.erase(found);
    meep_cuda::select_device(buffer_cache->device_ordinal);
    found = plans.operation_plans.end();
  }
  if (found != plans.operation_plans.end() &&
      (found->second->operation_count != operation_count ||
       found->second->device_ordinal != buffer_cache->device_ordinal)) {
    release_boundary_operation_plan(found->second);
    plans.operation_plans.erase(found);
    meep_cuda::select_device(buffer_cache->device_ordinal);
    found = plans.operation_plans.end();
  }

  boundary_exchange_owner::operation_plan *plan = nullptr;
  if (found != plans.operation_plans.end()) {
    plan = found->second;
    ++plan->stable_reuses;
    scalar_count = plan->scalar_count;
    mark_boundary_plan_destinations_dirty(plan);
    device_buffer_reuses.fetch_add(1);
  }
  else {
    std::vector<meep_cuda::boundary_operation_fp32> device_operations;
    std::vector<resident_cache *> referenced_caches;
    device_operations.reserve(operation_count);
    referenced_caches.reserve(2 * operation_count);
    bool all_sources_null = true;
    for (std::size_t j = 0; j < operation_count; ++j) {
      const boundary_operation_fp32 &operation = operations[j];
      resident_cache *source_cache = operation.source_cache;
      resident_cache *destination_cache = operation.destination_cache;
      if (!destination_cache || !destination_cache->phase_active)
        throw std::logic_error(
            "Meep CUDA boundary destination requires an active resident phase");
      if (destination_cache->device_ordinal != buffer_cache->device_ordinal)
        throw std::invalid_argument(
            "Meep CUDA local boundary caches must use the same device");
      if (!operation.destination_real)
        throw std::invalid_argument(
            "Meep CUDA boundary destination must be non-null");
      if (!operation.source_real && operation.source_imag)
        throw std::invalid_argument(
            "Meep CUDA boundary imaginary source requires a real source");
      if (operation.source_real) all_sources_null = false;
      if (operation.source_real &&
          ((!operation.source_imag) != (!operation.destination_imag)))
        throw std::invalid_argument(
            "Meep CUDA complex boundary source/destination pairs must match");
      if (operation.source_real &&
          (!source_cache || !source_cache->phase_active))
        throw std::logic_error(
            "Meep CUDA boundary source requires an active resident phase");
      if (operation.source_real &&
          source_cache->device_ordinal != destination_cache->device_ordinal)
        throw std::invalid_argument(
            "Meep CUDA local boundary caches must use the same device");
      if (scalar_count >
          std::numeric_limits<std::size_t>::max() -
              (operation.destination_imag ? 2u : 1u))
        throw std::overflow_error(
            "Meep CUDA boundary point count overflow");
      scalar_count += operation.destination_imag ? 2u : 1u;

      if (std::find(referenced_caches.begin(), referenced_caches.end(),
                    destination_cache) == referenced_caches.end())
        referenced_caches.push_back(destination_cache);
      if (source_cache &&
          std::find(referenced_caches.begin(), referenced_caches.end(),
                    source_cache) == referenced_caches.end())
        referenced_caches.push_back(source_cache);

      device_operations.push_back(
          {static_cast<float *>(resident_device_address(
               destination_cache, operation.destination_real, true)),
           operation.destination_imag
               ? static_cast<float *>(resident_device_address(
                     destination_cache, operation.destination_imag, true))
               : nullptr,
           operation.source_real
               ? static_cast<const float *>(resident_device_address(
                     source_cache, operation.source_real, false))
               : nullptr,
           operation.source_imag
               ? static_cast<const float *>(resident_device_address(
                     source_cache, operation.source_imag, false))
               : nullptr,
           operation.phase_real,
           operation.phase_imag});
    }

    std::unique_ptr<boundary_exchange_owner::operation_plan> created(
        new boundary_exchange_owner::operation_plan);
    created->operation_count = operation_count;
    created->scalar_count = scalar_count;
    created->all_sources_null = all_sources_null;
    std::vector<std::uintptr_t> source_addresses;
    std::vector<std::uintptr_t> destination_addresses;
    source_addresses.reserve(2 * operation_count);
    destination_addresses.reserve(2 * operation_count);
    for (const meep_cuda::boundary_operation_fp32 &operation :
         device_operations) {
      if (operation.source_real)
        source_addresses.push_back(
            reinterpret_cast<std::uintptr_t>(operation.source_real));
      if (operation.source_imag)
        source_addresses.push_back(
            reinterpret_cast<std::uintptr_t>(operation.source_imag));
      destination_addresses.push_back(
          reinterpret_cast<std::uintptr_t>(operation.destination_real));
      if (operation.destination_imag)
        destination_addresses.push_back(
            reinterpret_cast<std::uintptr_t>(operation.destination_imag));
    }
    std::sort(source_addresses.begin(), source_addresses.end());
    source_addresses.erase(
        std::unique(source_addresses.begin(), source_addresses.end()),
        source_addresses.end());
    std::sort(destination_addresses.begin(), destination_addresses.end());
    destination_addresses.erase(
        std::unique(destination_addresses.begin(),
                    destination_addresses.end()),
        destination_addresses.end());
    created->nonalias_safe = true;
    std::size_t source_index = 0;
    std::size_t destination_index = 0;
    while (source_index < source_addresses.size() &&
           destination_index < destination_addresses.size()) {
      if (source_addresses[source_index] ==
          destination_addresses[destination_index]) {
        created->nonalias_safe = false;
        break;
      }
      if (source_addresses[source_index] <
          destination_addresses[destination_index])
        ++source_index;
      else
        ++destination_index;
    }
    created->device_ordinal = buffer_cache->device_ordinal;
    created->host_operations.assign(
        operations, operations + operation_count);
    created->cache_generations.reserve(referenced_caches.size());
    for (resident_cache *cache : referenced_caches)
      created->cache_generations.push_back(
          std::make_pair(cache, cache->allocation_generation));
    for (const boundary_operation_fp32 &operation :
         created->host_operations) {
      const float *destination_bases[] = {
          resident_mirror_base_for_address(
              operation.destination_cache, operation.destination_real),
          operation.destination_imag
              ? resident_mirror_base_for_address(
                    operation.destination_cache,
                    operation.destination_imag)
              : nullptr};
      for (const float *base : destination_bases) {
        if (!base) continue;
        const auto writable =
            std::make_pair(operation.destination_cache, base);
        if (std::find(created->writable_mirrors.begin(),
                      created->writable_mirrors.end(),
                      writable) == created->writable_mirrors.end())
          created->writable_mirrors.push_back(writable);
      }
    }
    // Finish every potentially-throwing host metadata allocation before
    // acquiring raw CUDA storage. The copy/registry insertion below has an
    // explicit device-pointer rollback guard.
    created->device_pointer =
        meep_cuda::allocate_device_bytes(operation_bytes + staging_bytes);
    try {
      meep_cuda::copy_to_device(created->device_pointer,
                                device_operations.data(), operation_bytes);
      const bool direct_nonalias_enabled =
          created->nonalias_safe &&
          std::getenv("MEEP_GPU_DISABLE_DIRECT_LOCAL_BOUNDARY") == nullptr;
      const bool boundary_graph_enabled =
          std::getenv("MEEP_GPU_DISABLE_BOUNDARY_GRAPH") == nullptr;
      if (!created->all_sources_null && !direct_nonalias_enabled &&
          boundary_graph_enabled)
        created->execution_graph = meep_cuda::create_boundary_graph_fp32(
            static_cast<const meep_cuda::boundary_operation_fp32 *>(
                created->device_pointer),
            reinterpret_cast<float *>(
                static_cast<unsigned char *>(created->device_pointer) +
                operation_bytes),
            operation_count);
      created->generation = next_boundary_plan_generation();
      plan = created.get();
      plans.operation_plans.emplace(plan_token, plan);
      created.release();
    }
    catch (...) {
      meep_cuda::destroy_boundary_graph_fp32(created->execution_graph);
      created->execution_graph = nullptr;
      meep_cuda::free_device(created->device_pointer);
      created->device_pointer = nullptr;
      throw;
    }
    host_to_device_bytes.fetch_add(
        static_cast<std::uint64_t>(operation_bytes));
    device_buffer_allocations.fetch_add(1);
    live_device_buffer_count().fetch_add(1);
  }

  meep_cuda::select_device(plan->device_ordinal);
  if (plan->all_sources_null &&
      std::getenv("MEEP_GPU_DISABLE_DIRECT_BOUNDARY_ZERO") == nullptr)
    meep_cuda::zero_boundary_fp32(
        static_cast<const meep_cuda::boundary_operation_fp32 *>(
            plan->device_pointer),
        operation_count);
  else if (plan->nonalias_safe &&
           std::getenv("MEEP_GPU_DISABLE_DIRECT_LOCAL_BOUNDARY") == nullptr)
    meep_cuda::apply_nonalias_boundary_fp32(
        static_cast<const meep_cuda::boundary_operation_fp32 *>(
            plan->device_pointer),
        operation_count);
  else if (plan->execution_graph &&
      std::getenv("MEEP_GPU_DISABLE_BOUNDARY_GRAPH") == nullptr)
    meep_cuda::launch_boundary_graph_fp32(plan->execution_graph);
  else
    meep_cuda::apply_boundary_fp32(
        static_cast<const meep_cuda::boundary_operation_fp32 *>(
            plan->device_pointer),
        reinterpret_cast<float *>(
            static_cast<unsigned char *>(plan->device_pointer) +
            operation_bytes),
        operation_count);
  cuda_boundary_calls.fetch_add(1);
  cuda_boundary_points.fetch_add(
      static_cast<std::uint64_t>(scalar_count));
#else
  (void)owner;
  (void)plan_token;
  (void)operations;
  (void)operation_count;
  throw std::runtime_error(
      "Meep CUDA resident boundary called in a CPU-only build");
#endif
}

namespace {
#if MEEP_HAVE_CUDA

struct finite_check_request {
  float *host_base;
  const float *mirror_base;
  std::size_t scalar_count;
  std::size_t bytes;
  std::size_t mirror_bytes;
  std::size_t byte_offset;
};

bool finite_check_plan_matches(
    const resident_cache *cache,
    const std::vector<finite_check_request> &requests,
    std::size_t maximum_span_count) {
  const resident_cache::finite_check_plan &plan = cache->finite_check;
  if (!plan.device_spans ||
      plan.allocation_generation != cache->allocation_generation ||
      plan.maximum_span_count != maximum_span_count ||
      plan.entries.size() != requests.size())
    return false;

  for (std::size_t index = 0; index < requests.size(); ++index) {
    const finite_check_request &request = requests[index];
    const resident_cache::finite_check_entry &entry =
        plan.entries[index];
    if (entry.host_base != request.host_base ||
        entry.scalar_count != request.scalar_count)
      return false;
    const auto mirror =
        cache->mirrors.find(entry.mirror_host_base);
    if (mirror == cache->mirrors.end() ||
        mirror->second.bytes != entry.mirror_bytes ||
        (!mirror->second.device_dirty &&
         mirror->second.uploaded_epoch == 0))
      return false;
    if (entry.byte_offset > entry.mirror_bytes ||
        request.bytes > entry.mirror_bytes - entry.byte_offset)
      return false;
    const std::uintptr_t mirror_host_begin =
        reinterpret_cast<std::uintptr_t>(entry.mirror_host_base);
    if (entry.byte_offset >
        std::numeric_limits<std::uintptr_t>::max() -
            mirror_host_begin)
      return false;
    if (mirror_host_begin + entry.byte_offset !=
        reinterpret_cast<std::uintptr_t>(request.host_base))
      return false;
    const float *device_base = reinterpret_cast<const float *>(
        static_cast<const unsigned char *>(
            mirror->second.device_pointer) +
        entry.byte_offset);
    if (entry.device_base != device_base)
      return false;
  }
  return true;
}

const resident_cache::finite_check_plan &prepare_finite_check_plan(
    resident_cache *cache, float *const *host_arrays,
    const std::size_t *array_counts, std::size_t array_count) {
  if (!cache || !cache->phase_active)
    throw std::logic_error(
        "Meep CUDA finite check requires an active resident phase");
  if ((!host_arrays || !array_counts) && array_count)
    throw std::invalid_argument(
        "Meep CUDA finite-check arrays and counts must be non-null");
  if (array_count == 0)
    throw std::invalid_argument(
        "Meep CUDA finite check requires at least one array");

  // Complete every validation and potentially-throwing host allocation
  // before ensure_resident_mirror can mutate the cache.  In particular,
  // never let a caller-supplied size mismatch resize a dirty authoritative
  // mirror.
  std::vector<finite_check_request> requests;
  std::vector<resident_cache::finite_check_entry> resolved_entries;
  std::vector<meep_cuda::array_span_fp32> resolved_spans;
  requests.reserve(array_count);
  resolved_entries.reserve(array_count);
  resolved_spans.reserve(array_count);
  std::size_t maximum_span_count = 0;
  for (std::size_t index = 0; index < array_count; ++index) {
    if (!host_arrays[index] || array_counts[index] == 0)
      throw std::invalid_argument(
          "Meep CUDA finite-check array must be non-null and nonempty");
    const std::size_t bytes =
        checked_bytes(array_counts[index], sizeof(float),
                      "finite-check field");
    const std::uintptr_t request_begin =
        reinterpret_cast<std::uintptr_t>(host_arrays[index]);
    if (request_begin % alignof(float))
      throw std::invalid_argument(
          "Meep CUDA finite-check host array is not float-aligned");
    if (bytes >
        std::numeric_limits<std::uintptr_t>::max() - request_begin)
      throw std::overflow_error(
          "Meep CUDA finite-check host range overflow");
    const std::uintptr_t request_end = request_begin + bytes;
    bool exact_duplicate = false;
    for (const finite_check_request &prior : requests) {
      const std::uintptr_t prior_begin =
          reinterpret_cast<std::uintptr_t>(prior.host_base);
      const std::uintptr_t prior_end =
          prior_begin + prior.bytes;
      const bool same_range =
          prior_begin == request_begin &&
          prior_end == request_end;
      if (same_range) {
        exact_duplicate = true;
        break;
      }
      if (!same_range &&
          request_begin < prior_end &&
          prior_begin < request_end)
        throw std::invalid_argument(
            "Meep CUDA finite-check array ranges overlap");
    }
    if (exact_duplicate) continue;

    requests.push_back(
        {host_arrays[index], nullptr, array_counts[index], bytes, 0, 0});
    maximum_span_count =
        std::max(maximum_span_count, array_counts[index]);
  }

  // Exact duplicates have already been removed.  Size both the device
  // allocation and the host copy from the constructed descriptor count,
  // rather than from the caller's pre-dedup array_count: vector capacity is
  // not initialized storage and must never be copied as a descriptor.
  const std::size_t spans_bytes =
      checked_bytes(requests.size(), sizeof(meep_cuda::array_span_fp32),
                    "finite-check spans");
  if (finite_check_plan_matches(
          cache, requests, maximum_span_count)) {
    device_buffer_reuses.fetch_add(1);
    return cache->finite_check;
  }

  // Mirror insertion/removal changes allocation_generation, so the stable
  // path above never needs to rescan the interval topology.  On rebuild,
  // audit every interval: overlapping mirrors are historically possible,
  // and an immediate-predecessor lookup cannot detect a long outer mirror
  // hidden behind a shorter nested one.
  if (cache->mirror_intervals.size() != cache->mirrors.size())
    throw std::logic_error(
        "Meep CUDA resident mirror interval count is inconsistent");
  for (finite_check_request &request : requests) {
    const std::uintptr_t request_begin =
        reinterpret_cast<std::uintptr_t>(request.host_base);
    const std::uintptr_t request_end =
        request_begin + request.bytes;
    std::size_t overlap_count = 0;
    for (const auto &interval : cache->mirror_intervals) {
      const auto mirror = cache->mirrors.find(interval.second);
      if (mirror == cache->mirrors.end() ||
          reinterpret_cast<std::uintptr_t>(interval.second) !=
              interval.first)
        throw std::logic_error(
            "Meep CUDA resident interval index is inconsistent");
      if (mirror->second.bytes >
          std::numeric_limits<std::uintptr_t>::max() -
              interval.first)
        throw std::overflow_error(
            "Meep CUDA resident mirror address overflow");
      const std::uintptr_t mirror_end =
          interval.first + mirror->second.bytes;
      if (request_begin >= mirror_end ||
          interval.first >= request_end)
        continue;
      ++overlap_count;
      if (overlap_count != 1)
        throw std::invalid_argument(
            "Meep CUDA finite-check range has ambiguous overlapping "
            "resident mirrors");
      if (request_begin < interval.first ||
          request_end > mirror_end)
        throw std::invalid_argument(
            "Meep CUDA finite-check range partially overlaps a "
            "resident mirror");
      if (request_begin == interval.first &&
          request_end != mirror_end)
        throw std::invalid_argument(
            "Meep CUDA finite-check array size disagrees with its "
            "authoritative resident mirror");
      request.mirror_base = interval.second;
      request.mirror_bytes = mirror->second.bytes;
      request.byte_offset = static_cast<std::size_t>(
          request_begin - interval.first);
    }
    if (overlap_count == 0) {
      request.mirror_base = request.host_base;
      request.mirror_bytes = request.bytes;
      request.byte_offset = 0;
    }
  }

  // Resolve every mirror only after the read-only preflight.  A clean mirror
  // with uploaded_epoch==0 is refreshed from its authoritative host array.
  for (const finite_check_request &request : requests) {
    const unsigned char *device_mirror =
        static_cast<const unsigned char *>(
            ensure_resident_mirror(
                cache, request.mirror_base, request.mirror_bytes,
                &finite_check_host_to_device_bytes));
    const float *device_base =
        reinterpret_cast<const float *>(
            device_mirror + request.byte_offset);
    resolved_entries.push_back(
        {request.host_base, request.mirror_base, device_base,
         request.scalar_count, request.mirror_bytes,
         request.byte_offset});
    resolved_spans.push_back(
        {device_base, request.scalar_count});
  }

  // An invalidated generation is never executed, even when an allocator
  // happens to reuse the same raw device address.
  const std::uint64_t resolved_generation =
      cache->allocation_generation;
  bool resolved_matches =
      cache->finite_check.device_spans &&
      cache->finite_check.allocation_generation ==
          resolved_generation &&
      cache->finite_check.maximum_span_count ==
          maximum_span_count &&
      cache->finite_check.entries.size() ==
          resolved_entries.size();
  if (resolved_matches)
    for (std::size_t index = 0;
         index < resolved_entries.size(); ++index) {
      const resident_cache::finite_check_entry &expected =
          cache->finite_check.entries[index];
      const resident_cache::finite_check_entry &resolved =
          resolved_entries[index];
      if (expected.host_base != resolved.host_base ||
          expected.mirror_host_base != resolved.mirror_host_base ||
          expected.device_base != resolved.device_base ||
          expected.scalar_count != resolved.scalar_count ||
          expected.mirror_bytes != resolved.mirror_bytes ||
          expected.byte_offset != resolved.byte_offset) {
        resolved_matches = false;
        break;
      }
    }
  if (resolved_matches) {
    device_buffer_reuses.fetch_add(1);
    return cache->finite_check;
  }

  void *replacement = meep_cuda::allocate_device_bytes(spans_bytes);
  try {
    meep_cuda::copy_to_device(
        replacement, resolved_spans.data(), spans_bytes);
  }
  catch (...) {
    meep_cuda::free_device(replacement);
    throw;
  }

  // All operations after the successful descriptor copy are noexcept.
  // Commit by swapping host metadata and the device allocation, then release
  // the old (generation-invalid) descriptor storage.
  void *old_device_spans = cache->finite_check.device_spans;
  cache->finite_check.device_spans = replacement;
  cache->finite_check.entries.swap(resolved_entries);
  cache->finite_check.maximum_span_count =
      maximum_span_count;
  cache->finite_check.allocation_generation =
      resolved_generation;
  meep_cuda::free_device(old_device_spans);

  host_to_device_bytes.fetch_add(
      static_cast<std::uint64_t>(spans_bytes));
  finite_check_host_to_device_bytes.fetch_add(
      static_cast<std::uint64_t>(spans_bytes));
  device_buffer_allocations.fetch_add(1);
  if (!old_device_spans)
    live_device_buffer_count().fetch_add(1);
  return cache->finite_check;
}

#endif
} // namespace

resident_finite_check_session::resident_finite_check_session(
    resident_cache *result_owner)
    : result_owner_(result_owner), active_(false),
      initialized_(false), generation_(0) {
#if MEEP_HAVE_CUDA
  if (!result_owner_ || !result_owner_->phase_active)
    throw std::logic_error(
        "Meep CUDA finite-check result owner requires an active "
        "resident phase");
  if (result_owner_->finite_session_active)
    throw std::logic_error(
        "nested Meep CUDA finite-check sessions for one result owner are "
        "not supported");
  meep_cuda::select_device(result_owner_->device_ordinal);
  result_owner_->finite_session_active = true;
  active_ = true;
#else
  (void)result_owner_;
  throw std::runtime_error(
      "Meep CUDA finite-check session called in a CPU-only build");
#endif
}

resident_finite_check_session::~resident_finite_check_session() {
#if MEEP_HAVE_CUDA
  if (active_ && result_owner_)
    result_owner_->finite_session_active = false;
#endif
}

void resident_finite_check_session::initialize_result() {
#if MEEP_HAVE_CUDA
  if (initialized_) return;
  meep_cuda::select_device(result_owner_->device_ordinal);
  const bool reused_result = result_owner_->finite_result != nullptr;
  if (!result_owner_->finite_result) {
    std::uint32_t *created = static_cast<std::uint32_t *>(
        meep_cuda::allocate_device_bytes(sizeof(std::uint32_t)));
    try {
      meep_cuda::clear_finite_generation_result(created);
    }
    catch (...) {
      meep_cuda::free_device(created);
      throw;
    }
    result_owner_->finite_result = created;
    result_owner_->finite_result_generation = 0;
    result_owner_->finite_result_pending = false;
    device_buffer_allocations.fetch_add(1);
    live_device_buffer_count().fetch_add(1);
  }
  if (!result_owner_->finite_result_pending) {
    if (result_owner_->finite_result_generation ==
        std::numeric_limits<std::uint32_t>::max()) {
      meep_cuda::clear_finite_generation_result(
          result_owner_->finite_result);
      result_owner_->finite_result_generation = 1;
    }
    else {
      ++result_owner_->finite_result_generation;
    }
    if (reused_result) device_buffer_reuses.fetch_add(1);
  }
  generation_ = result_owner_->finite_result_generation;
  initialized_ = true;
#else
  throw std::runtime_error(
      "Meep CUDA finite-check initialization called in a CPU-only build");
#endif
}

void resident_finite_check_session::accumulate(
    resident_cache *cache, float *const *host_arrays,
    const std::size_t *array_counts, std::size_t array_count) {
#if MEEP_HAVE_CUDA
  if (!active_)
    throw std::logic_error(
        "Meep CUDA finite-check session is not active");
  if (!cache || !cache->phase_active)
    throw std::logic_error(
        "Meep CUDA finite-check cache requires an active resident phase");
  if (cache->device_ordinal != result_owner_->device_ordinal)
    throw std::invalid_argument(
        "Meep CUDA finite-check caches must use the same device");
  if (array_count == 0) return;
  const resident_cache::finite_check_plan &plan =
      prepare_finite_check_plan(
          cache, host_arrays, array_counts, array_count);
  initialize_result();
  meep_cuda::all_finite_generation_fp32(
      static_cast<const meep_cuda::array_span_fp32 *>(
          plan.device_spans),
      plan.entries.size(), plan.maximum_span_count,
      generation_, result_owner_->finite_result);
  result_owner_->finite_result_pending = true;
#else
  (void)cache;
  (void)host_arrays;
  (void)array_counts;
  (void)array_count;
  throw std::runtime_error(
      "Meep CUDA finite-check accumulation called in a CPU-only build");
#endif
}

bool resident_finite_check_session::finish(bool readback) {
#if MEEP_HAVE_CUDA
  if (!active_)
    throw std::logic_error(
        "Meep CUDA finite-check session is not active");
  if (!initialized_) {
    if (readback && result_owner_->finite_result_pending) {
      // A zero-array session is still a valid request to consume a verdict
      // deferred by finish(false).  Bind to the sticky generation without
      // launching a new scan.
      generation_ = result_owner_->finite_result_generation;
      initialized_ = true;
    }
    else {
      result_owner_->finite_session_active = false;
      active_ = false;
      return true;
    }
  }
  if (!readback) {
    result_owner_->finite_session_active = false;
    active_ = false;
    return true;
  }
  std::uint32_t failed_generation = 0;
  meep_cuda::copy_to_host(
      &failed_generation, result_owner_->finite_result,
      sizeof(failed_generation));
  device_to_host_bytes.fetch_add(sizeof(failed_generation));
  finite_check_device_to_host_bytes.fetch_add(
      sizeof(failed_generation));
  result_owner_->finite_result_pending = false;
  result_owner_->finite_session_active = false;
  active_ = false;
  return failed_generation != generation_;
#else
  (void)readback;
  throw std::runtime_error(
      "Meep CUDA finite-check finish called in a CPU-only build");
#endif
}

bool resident_primary_fields_are_finite_fp32(
    resident_cache *cache, float *const *host_arrays,
    const std::size_t *array_counts, std::size_t array_count) {
#if MEEP_HAVE_CUDA
  if (!cache || !cache->phase_active)
    throw std::logic_error(
        "Meep CUDA finite check requires an active resident phase");
  if ((!host_arrays || !array_counts) && array_count)
    throw std::invalid_argument(
        "Meep CUDA finite-check arrays and counts must be non-null");
  if (array_count == 0) return true;
#endif
  resident_finite_check_session session(cache);
  session.accumulate(
      cache, host_arrays, array_counts, array_count);
  return session.finish();
}

void resident_update_dft_batch_fp32(
    const dft_update_request_fp32 *requests, std::size_t request_count) {
#if MEEP_HAVE_CUDA
  if (!requests && request_count)
    throw std::invalid_argument(
        "Meep CUDA DFT batch requests must be non-null");
  if (request_count == 0) return;

  // Ordinary dft_chunk storage is disjoint. Reject an overlapping internal
  // request before preparing any mirrors: distinct host aliases cannot be
  // represented by independent resident allocations, and resizing one
  // repeated-base mirror could invalidate a pointer prepared for an earlier
  // request. Outputs must also remain disjoint from every read-only input;
  // the CUDA kernel intentionally provides no in-place alias semantics.
  std::vector<std::pair<std::uintptr_t, std::uintptr_t> > output_intervals;
  std::vector<std::pair<std::uintptr_t, std::uintptr_t> > input_intervals;
  output_intervals.reserve(request_count);
  if (request_count > input_intervals.max_size() / 5)
    throw std::overflow_error(
        "Meep CUDA DFT input interval count overflow");
  input_intervals.reserve(5 * request_count);
  struct mirror_extent {
    std::uintptr_t cache_address;
    std::uintptr_t host_address;
    std::size_t bytes;
  };
  std::vector<mirror_extent> input_mirror_extents;
  if (request_count > input_mirror_extents.max_size() / 5)
    throw std::overflow_error(
        "Meep CUDA DFT mirror extent count overflow");
  input_mirror_extents.reserve(5 * request_count);
  const auto append_interval = [](
      std::vector<std::pair<std::uintptr_t, std::uintptr_t> > *intervals,
      const void *pointer, std::size_t bytes, const char *label) {
    if (!pointer || bytes == 0) return;
    const std::uintptr_t begin =
        reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin)
      throw std::overflow_error(
          std::string("Meep CUDA DFT ") + label +
          " address overflow");
    intervals->push_back({begin, begin + bytes});
  };
  const auto append_input_mirror_extent = [&input_mirror_extents](
      resident_cache *cache, const void *pointer, std::size_t bytes) {
    if (!pointer || bytes == 0) return;
    const std::uintptr_t host_address =
        reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes >
        std::numeric_limits<std::uintptr_t>::max() - host_address)
      throw std::overflow_error(
          "Meep CUDA DFT input mirror address overflow");
    input_mirror_extents.push_back(
        {reinterpret_cast<std::uintptr_t>(cache),
         host_address, bytes});
  };
  for (std::size_t request_index = 0; request_index < request_count;
       ++request_index) {
    const dft_update_request_fp32 &request = requests[request_index];
    if (!request.dft_real_imag)
      throw std::invalid_argument(
          "Meep CUDA DFT output must be non-null");
    if (request.point_count == 0 || request.frequency_count == 0)
      throw std::invalid_argument(
          "Meep CUDA DFT batch requests must contain nonzero work");
    if (request.point_count >
        std::numeric_limits<std::size_t>::max() /
            request.frequency_count)
      throw std::overflow_error("Meep CUDA DFT output count overflow");
    const std::size_t complex_count =
        request.point_count * request.frequency_count;
    if (complex_count > std::numeric_limits<std::size_t>::max() / 2)
      throw std::overflow_error("Meep CUDA DFT scalar count overflow");
    const std::size_t output_bytes = checked_bytes(
        2 * complex_count, sizeof(float), "DFT output");
    append_interval(
        &output_intervals, request.dft_real_imag, output_bytes,
        "output");
    const std::size_t field_bytes = checked_bytes(
        request.field_array_count, sizeof(float), "DFT field");
    const std::size_t index_bytes = checked_bytes(
        request.point_count, sizeof(std::ptrdiff_t), "DFT indices");
    const std::size_t weight_bytes = checked_bytes(
        request.point_count, sizeof(float), "DFT weights");
    const std::size_t frequency_bytes = checked_bytes(
        request.frequency_count, sizeof(double),
        "DFT angular frequencies");
    append_interval(
        &input_intervals, request.field_real, field_bytes,
        "real-field input");
    append_interval(
        &input_intervals, request.field_imag, field_bytes,
        "imaginary-field input");
    append_interval(
        &input_intervals, request.field_indices, index_bytes,
        "index input");
    append_interval(
        &input_intervals, request.weights, weight_bytes,
        "weight input");
    append_interval(
        &input_intervals, request.angular_frequencies, frequency_bytes,
        "frequency input");
    append_input_mirror_extent(
        request.cache, request.field_real, field_bytes);
    append_input_mirror_extent(
        request.cache, request.field_imag, field_bytes);
    append_input_mirror_extent(
        request.cache, request.field_indices, index_bytes);
    append_input_mirror_extent(
        request.cache, request.weights, weight_bytes);
    append_input_mirror_extent(
        request.cache, request.angular_frequencies, frequency_bytes);
  }
  std::sort(output_intervals.begin(), output_intervals.end());
  for (std::size_t index = 1; index < output_intervals.size(); ++index)
    if (output_intervals[index - 1].second >
        output_intervals[index].first)
      throw std::invalid_argument(
          "Meep CUDA DFT batch outputs must not overlap");
  std::sort(
      input_mirror_extents.begin(), input_mirror_extents.end(),
      [](const mirror_extent &left, const mirror_extent &right) {
        if (left.cache_address != right.cache_address)
          return left.cache_address < right.cache_address;
        if (left.host_address != right.host_address)
          return left.host_address < right.host_address;
        return left.bytes < right.bytes;
      });
  std::uintptr_t active_cache = 0;
  std::uintptr_t active_end = 0;
  bool have_active_cache = false;
  for (std::size_t index = 0; index < input_mirror_extents.size(); ++index) {
    const mirror_extent &current = input_mirror_extents[index];
    if (index > 0) {
      const mirror_extent &previous = input_mirror_extents[index - 1];
      if (previous.cache_address == current.cache_address &&
          previous.host_address == current.host_address &&
          previous.bytes != current.bytes)
        throw std::invalid_argument(
            "Meep CUDA DFT cannot mirror one input base with different "
            "byte extents in the same resident cache");
    }
    if (!have_active_cache || current.cache_address != active_cache) {
      active_cache = current.cache_address;
      active_end = current.host_address + current.bytes;
      have_active_cache = true;
      continue;
    }
    const bool exact_duplicate =
        index > 0 &&
        input_mirror_extents[index - 1].cache_address ==
            current.cache_address &&
        input_mirror_extents[index - 1].host_address ==
            current.host_address &&
        input_mirror_extents[index - 1].bytes == current.bytes;
    if (!exact_duplicate && current.host_address < active_end)
      throw std::invalid_argument(
          "Meep CUDA DFT input mirrors must not partially overlap in one "
          "resident cache");
    active_end = std::max(
        active_end, current.host_address + current.bytes);
  }

  // Both collections are half-open address intervals.  Sorting once and
  // sweeping them avoids an O(outputs * inputs) validation cost on every DFT
  // timestep while preserving exact overlap rejection for nested and
  // duplicate input ranges.
  std::sort(input_intervals.begin(), input_intervals.end());
  std::size_t output_index = 0;
  std::size_t input_index = 0;
  while (output_index < output_intervals.size() &&
         input_index < input_intervals.size()) {
    const auto &output = output_intervals[output_index];
    const auto &input = input_intervals[input_index];
    if (output.second <= input.first)
      ++output_index;
    else if (input.second <= output.first)
      ++input_index;
    else
      throw std::invalid_argument(
          "Meep CUDA DFT batch outputs must not overlap inputs");
  }

  struct prepared_dft_update {
    dft_update_request_fp32 request;
    resident_cache *cache;
    float *device_output;
    const float *device_field_real;
    const float *device_field_imag;
    const std::ptrdiff_t *device_indices;
    const float *device_weights;
    const double *device_frequencies;
    std::size_t output_bytes;
    std::size_t complex_count;
  };
  std::vector<prepared_dft_update> prepared;
  prepared.reserve(request_count);
  std::uint64_t total_complex_count = 0;

  // Validate and prepare every mirror before the first kernel launch.  A
  // malformed later request therefore cannot leave an otherwise valid batch
  // partially accumulated, and the shared scratch allocation cannot move
  // while an earlier update still consumes it on the default stream.
  for (std::size_t request_index = 0; request_index < request_count;
       ++request_index) {
    const dft_update_request_fp32 &request = requests[request_index];
    if (!request.cache || !request.cache->phase_active)
      throw std::logic_error(
          "Meep CUDA DFT batch requires an active resident phase for "
          "every request");
    if (!prepared.empty() &&
        request.cache->device_ordinal != prepared.front().cache->device_ordinal)
      throw std::invalid_argument(
          "Meep CUDA DFT batch cannot span CUDA devices");
    if (!request.dft_real_imag || !request.field_real ||
        !request.field_indices || !request.weights ||
        !request.angular_frequencies)
      throw std::invalid_argument(
          "Meep CUDA DFT output, field, indices, weights, and angular "
          "frequencies must be non-null");
    if (request.point_count == 0 || request.frequency_count == 0)
      throw std::invalid_argument(
          "Meep CUDA DFT batch requests must contain nonzero work");
    if (request.point_count >
        std::numeric_limits<std::size_t>::max() /
            request.frequency_count)
      throw std::overflow_error("Meep CUDA DFT output count overflow");
    const std::size_t complex_count =
        request.point_count * request.frequency_count;
    if (complex_count > std::numeric_limits<std::size_t>::max() / 2)
      throw std::overflow_error("Meep CUDA DFT scalar count overflow");
    if (request.field_array_count >
        static_cast<std::size_t>(
            std::numeric_limits<std::ptrdiff_t>::max()))
      throw std::overflow_error(
          "Meep CUDA DFT field count exceeds the index range");
    if (complex_count >
        std::numeric_limits<std::uint64_t>::max() - total_complex_count)
      throw std::overflow_error("Meep CUDA DFT batch point count overflow");
    total_complex_count += static_cast<std::uint64_t>(complex_count);

    const std::size_t field_bytes = checked_bytes(
        request.field_array_count, sizeof(float), "DFT field");
    const std::size_t output_bytes = checked_bytes(
        2 * complex_count, sizeof(float), "DFT output");
    const std::size_t index_bytes = checked_bytes(
        request.point_count, sizeof(std::ptrdiff_t), "DFT indices");
    const std::size_t weight_bytes = checked_bytes(
        request.point_count, sizeof(float), "DFT weights");
    const std::size_t frequency_bytes = checked_bytes(
        request.frequency_count, sizeof(double),
        "DFT angular frequencies");
    const std::ptrdiff_t limit =
        static_cast<std::ptrdiff_t>(request.field_array_count);
    for (std::size_t point = 0; point < request.point_count; ++point) {
      const std::ptrdiff_t index = request.field_indices[point];
      if (index < 0 || index >= limit)
        throw std::out_of_range(
            "Meep CUDA DFT field index is outside the field array");
      if (request.average_offset1 &&
          !shifted_index_is_valid(
              index, request.average_offset1, limit))
        throw std::out_of_range(
            "Meep CUDA DFT first average offset is outside the field array");
      if (request.average_offset2 &&
          (!shifted_index_is_valid(
               index, request.average_offset2, limit) ||
           !shifted_index_is_valid(
               index, request.average_offset1, limit) ||
           !shifted_index_is_valid(
               index + request.average_offset1,
               request.average_offset2, limit)))
        throw std::out_of_range(
            "Meep CUDA DFT second average offset is outside the field array");
    }

    prepared.push_back(
        {request, request.cache,
         static_cast<float *>(ensure_resident_mirror(
             request.cache, request.dft_real_imag, output_bytes)),
         static_cast<const float *>(ensure_resident_mirror(
             request.cache, request.field_real, field_bytes)),
         request.field_imag
             ? static_cast<const float *>(ensure_resident_mirror(
                   request.cache, request.field_imag, field_bytes))
             : nullptr,
         static_cast<const std::ptrdiff_t *>(ensure_resident_mirror(
             request.cache,
             reinterpret_cast<const float *>(request.field_indices),
             index_bytes)),
         static_cast<const float *>(ensure_resident_mirror(
             request.cache, request.weights, weight_bytes)),
         static_cast<const double *>(ensure_resident_mirror(
             request.cache,
             reinterpret_cast<const float *>(request.angular_frequencies),
             frequency_bytes)),
         output_bytes,
         complex_count});
  }

  static_assert(sizeof(complex_value_fp32) ==
                    sizeof(meep_cuda::complex_value_fp32),
                "Meep/CUDA DFT phase size mismatch");
  static_assert(alignof(complex_value_fp32) ==
                    alignof(meep_cuda::complex_value_fp32),
                "Meep/CUDA DFT phase alignment mismatch");

  auto same_scalar_bits = [](double left, double right) {
    return std::memcmp(&left, &right, sizeof(double)) == 0;
  };
  auto same_phase_key = [&](const prepared_dft_update &left,
                            const prepared_dft_update &right) {
    const dft_update_request_fp32 &a = left.request;
    const dft_update_request_fp32 &b = right.request;
    return a.frequency_count == b.frequency_count &&
           same_scalar_bits(a.time, b.time) &&
           same_scalar_bits(a.scale_real, b.scale_real) &&
           same_scalar_bits(a.scale_imag, b.scale_imag) &&
           std::memcmp(a.angular_frequencies, b.angular_frequencies,
                       a.frequency_count * sizeof(double)) == 0;
  };

  const bool sharing_disabled =
      std::getenv("MEEP_GPU_DISABLE_DFT_PHASE_SHARING") != nullptr;

  // Each ordinary dft_chunk owns a disjoint accumulation array, so operations
  // with the same phase key may be grouped across fields_chunk resident
  // caches without changing a numerical dependency. Hash each complete key
  // once and use exact comparison inside the matching hash bucket. This
  // preserves bitwise key semantics while avoiding an O(monitors^2 *
  // frequencies) scan when many monitors have distinct spectra.
  std::vector<std::vector<std::size_t> > phase_groups;
  phase_groups.reserve(prepared.size());
  if (sharing_disabled) {
    for (std::size_t index = 0; index < prepared.size(); ++index)
      phase_groups.push_back(std::vector<std::size_t>(1, index));
  }
  else {
    struct phase_signature {
      std::size_t frequency_count;
      std::uint64_t time_bits;
      std::uint64_t scale_real_bits;
      std::uint64_t scale_imag_bits;
      std::uint64_t frequency_hash;
    };
    struct phase_signature_hash {
      std::size_t operator()(const phase_signature &signature) const {
        std::size_t result = signature.frequency_count;
        const auto combine = [&result](std::uint64_t value) {
          result ^= static_cast<std::size_t>(
                        value + UINT64_C(0x9e3779b97f4a7c15) +
                        (static_cast<std::uint64_t>(result) << 6) +
                        (static_cast<std::uint64_t>(result) >> 2));
        };
        combine(signature.time_bits);
        combine(signature.scale_real_bits);
        combine(signature.scale_imag_bits);
        combine(signature.frequency_hash);
        return result;
      }
    };
    struct phase_signature_equal {
      bool operator()(const phase_signature &left,
                      const phase_signature &right) const {
        return left.frequency_count == right.frequency_count &&
               left.time_bits == right.time_bits &&
               left.scale_real_bits == right.scale_real_bits &&
               left.scale_imag_bits == right.scale_imag_bits &&
               left.frequency_hash == right.frequency_hash;
      }
    };
    const auto scalar_bits = [](double value) {
      std::uint64_t bits = 0;
      static_assert(sizeof(bits) == sizeof(value),
                    "DFT phase scalar size mismatch");
      std::memcpy(&bits, &value, sizeof(bits));
      return bits;
    };
    const auto frequency_hash = [](const double *values,
                                   std::size_t count) {
      const unsigned char *bytes =
          reinterpret_cast<const unsigned char *>(values);
      const std::size_t byte_count = count * sizeof(double);
      std::uint64_t result = UINT64_C(14695981039346656037);
      for (std::size_t index = 0; index < byte_count; ++index) {
        result ^= bytes[index];
        result *= UINT64_C(1099511628211);
      }
      return result;
    };
    std::unordered_map<
        phase_signature, std::vector<std::size_t>, phase_signature_hash,
        phase_signature_equal>
        candidate_groups;
    candidate_groups.reserve(prepared.size());
    for (std::size_t index = 0; index < prepared.size(); ++index) {
      const dft_update_request_fp32 &request = prepared[index].request;
      const phase_signature signature = {
          request.frequency_count, scalar_bits(request.time),
          scalar_bits(request.scale_real), scalar_bits(request.scale_imag),
          frequency_hash(
              request.angular_frequencies, request.frequency_count)};
      std::vector<std::size_t> &candidates = candidate_groups[signature];
      bool grouped = false;
      // A hash collision must never merge unequal phase vectors.
      for (std::size_t group_index : candidates)
        if (same_phase_key(
                prepared[phase_groups[group_index].front()],
                prepared[index])) {
          phase_groups[group_index].push_back(index);
          grouped = true;
          break;
        }
      if (!grouped) {
        candidates.push_back(phase_groups.size());
        phase_groups.push_back(std::vector<std::size_t>(1, index));
      }
    }
  }

  std::uint64_t phase_preparations = 0;
  std::uint64_t phase_reuse_count = 0;
  const phase_batch_mode multi_monitor_mode =
      dft_multi_monitor_batch_mode();
  const bool use_multi_monitor_kernel =
      multi_monitor_mode == phase_batch_mode::forced ||
      (multi_monitor_mode == phase_batch_mode::automatic &&
       automatic_dft_multi_monitor_batch_selected(request_count));
  if (multi_monitor_mode == phase_batch_mode::automatic) {
    dft_multi_monitor_automatic_checks.fetch_add(
        1, std::memory_order_relaxed);
    (use_multi_monitor_kernel
         ? dft_multi_monitor_automatic_selected
         : dft_multi_monitor_automatic_rejected)
        .fetch_add(1, std::memory_order_relaxed);
  }
  else if (multi_monitor_mode == phase_batch_mode::forced)
    dft_multi_monitor_forced_batches.fetch_add(
        1, std::memory_order_relaxed);

  std::uint64_t update_launches = 0;
  if (!use_multi_monitor_kernel) {
    std::size_t maximum_phase_scratch_bytes = 0;
    for (const std::vector<std::size_t> &group : phase_groups) {
      const std::size_t bytes = checked_bytes(
          prepared[group.front()].request.frequency_count,
          sizeof(complex_value_fp32), "DFT phase scratch");
      maximum_phase_scratch_bytes =
          std::max(maximum_phase_scratch_bytes, bytes);
    }
    void *device_phase_scratch = ensure_index_buffer(
        prepared.front().cache, maximum_phase_scratch_bytes);
    for (const std::vector<std::size_t> &group : phase_groups)
      for (std::size_t group_index = 0; group_index < group.size();
           ++group_index) {
        const prepared_dft_update &operation =
            prepared[group[group_index]];
        const dft_update_request_fp32 &request = operation.request;
        if (group_index == 0) {
          meep_cuda::update_dft_from_omega_fp32(
              operation.device_output, operation.device_field_real,
              operation.device_field_imag, operation.device_indices,
              operation.device_weights, request.point_count,
              operation.device_frequencies,
              static_cast<meep_cuda::complex_value_fp32 *>(
                  device_phase_scratch),
              request.frequency_count, request.time, request.scale_real,
              request.scale_imag, request.average_offset1,
              request.average_offset2);
          ++phase_preparations;
        }
        else {
          meep_cuda::update_dft_fp32(
              operation.device_output, operation.device_field_real,
              operation.device_field_imag, operation.device_indices,
              operation.device_weights, request.point_count,
              static_cast<const meep_cuda::complex_value_fp32 *>(
                  device_phase_scratch),
              request.frequency_count, request.average_offset1,
              request.average_offset2);
          ++phase_reuse_count;
        }
        mark_device_dirty(
            operation.cache, request.dft_real_imag,
            operation.output_bytes);
      }
    update_launches = static_cast<std::uint64_t>(request_count);
    dft_multi_monitor_unbatched_updates.fetch_add(
        static_cast<std::uint64_t>(request_count),
        std::memory_order_relaxed);
  }
  else {
    constexpr std::size_t threads_per_block =
        meep_cuda::dft_batch_threads_per_block_fp32;
    constexpr std::size_t maximum_frequency_threads = 32;

    const auto checked_sum = [](std::size_t left, std::size_t right,
                                const char *label) {
      if (left > std::numeric_limits<std::size_t>::max() - right)
        throw std::overflow_error(
            std::string("Meep CUDA ") + label + " storage overflow");
      return left + right;
    };
    const auto checked_product = [](std::size_t left, std::size_t right,
                                    const char *label) {
      if (right && left > std::numeric_limits<std::size_t>::max() / right)
        throw std::overflow_error(
            std::string("Meep CUDA ") + label + " count overflow");
      return left * right;
    };
    const auto align_up = [&](std::size_t value, std::size_t alignment,
                              const char *label) {
      const std::size_t remainder = value % alignment;
      return remainder == 0
                 ? value
                 : checked_sum(value, alignment - remainder, label);
    };

    std::vector<std::size_t> phase_group_offsets;
    phase_group_offsets.reserve(phase_groups.size());
    std::vector<std::size_t> operation_phase_groups(
        request_count, std::numeric_limits<std::size_t>::max());
    std::size_t phase_count = 0;
    for (std::size_t group_index = 0;
         group_index < phase_groups.size(); ++group_index) {
      const std::vector<std::size_t> &group = phase_groups[group_index];
      phase_group_offsets.push_back(phase_count);
      phase_count = checked_sum(
          phase_count,
          prepared[group.front()].request.frequency_count,
          "DFT phase scratch");
      for (std::size_t operation_index : group)
        operation_phase_groups[operation_index] = group_index;
    }
    for (std::size_t group_index : operation_phase_groups)
      if (group_index >= phase_group_offsets.size())
        throw std::logic_error(
            "Meep CUDA DFT operation has no phase group");

    const std::size_t phase_bytes = checked_product(
        phase_count, sizeof(meep_cuda::complex_value_fp32),
        "DFT phase scratch");
    const std::size_t descriptor_offset = align_up(
        phase_bytes, alignof(meep_cuda::dft_update_operation_fp32),
        "DFT descriptor alignment");
    const std::size_t descriptor_bytes = checked_product(
        request_count, sizeof(meep_cuda::dft_update_operation_fp32),
        "DFT descriptor");
    const std::size_t descriptor_end = checked_sum(
        descriptor_offset, descriptor_bytes, "DFT descriptor");
    const std::size_t block_map_offset = align_up(
        descriptor_end, alignof(std::uint32_t),
        "DFT block-map alignment");

    std::vector<meep_cuda::dft_update_operation_fp32> host_operations;
    host_operations.reserve(request_count);
    std::size_t total_block_count = 0;
    for (std::size_t operation_index = 0;
         operation_index < request_count; ++operation_index) {
      const prepared_dft_update &operation = prepared[operation_index];
      const dft_update_request_fp32 &request = operation.request;
      const std::size_t frequency_threads = std::min(
          request.frequency_count, maximum_frequency_threads);
      const std::size_t point_threads = std::min(
          request.point_count, threads_per_block / frequency_threads);
      const std::size_t frequency_blocks =
          request.frequency_count / frequency_threads +
          (request.frequency_count % frequency_threads != 0);
      const std::size_t point_blocks =
          request.point_count / point_threads +
          (request.point_count % point_threads != 0);
      const std::size_t operation_blocks = checked_product(
          frequency_blocks, point_blocks, "DFT logical block");
      const std::size_t block_start = total_block_count;
      total_block_count = checked_sum(
          total_block_count, operation_blocks, "DFT logical block");
      host_operations.push_back(
          {operation.device_output, operation.device_field_real,
           operation.device_field_imag, operation.device_indices,
           operation.device_weights, request.point_count, nullptr,
           request.frequency_count, request.average_offset1,
           request.average_offset2, block_start, frequency_blocks,
           point_blocks, static_cast<std::uint32_t>(frequency_threads),
           static_cast<std::uint32_t>(point_threads)});
    }
    const std::size_t block_map_bytes = checked_product(
        total_block_count, sizeof(std::uint32_t), "DFT block map");
    const std::size_t storage_bytes = checked_sum(
        block_map_offset, block_map_bytes, "DFT batch plan");

    resident_cache *const plan_cache = prepared.front().cache;
    const auto bind_phase_pointers = [&](void *device_storage) {
      auto *phase_base =
          static_cast<meep_cuda::complex_value_fp32 *>(device_storage);
      for (std::size_t operation_index = 0;
           operation_index < host_operations.size(); ++operation_index) {
        const std::size_t group_index =
            operation_phase_groups[operation_index];
        host_operations[operation_index].phases =
            phase_base + phase_group_offsets[group_index];
      }
    };
    const auto same_operation = [](
        const meep_cuda::dft_update_operation_fp32 &left,
        const meep_cuda::dft_update_operation_fp32 &right) {
      return left.dft_real_imag == right.dft_real_imag &&
             left.field_real == right.field_real &&
             left.field_imag == right.field_imag &&
             left.field_indices == right.field_indices &&
             left.weights == right.weights &&
             left.point_count == right.point_count &&
             left.phases == right.phases &&
             left.frequency_count == right.frequency_count &&
             left.average_offset1 == right.average_offset1 &&
             left.average_offset2 == right.average_offset2 &&
             left.block_start == right.block_start &&
             left.frequency_block_count == right.frequency_block_count &&
             left.point_block_count == right.point_block_count &&
             left.frequency_threads == right.frequency_threads &&
             left.point_threads == right.point_threads;
    };

    const auto plan_matches = [&](
        const resident_cache::dft_batch_plan &candidate) {
      bool matches =
          candidate.device_pointer &&
          candidate.capacity_bytes >= storage_bytes &&
          candidate.descriptor_offset == descriptor_offset &&
          candidate.block_map_offset == block_map_offset &&
          candidate.total_block_count == total_block_count &&
          candidate.host_operations.size() == host_operations.size();
      if (!matches) return false;
      bind_phase_pointers(candidate.device_pointer);
      // Validate the exact descriptor image after binding its resident phase
      // pointers.  The construction above intentionally leaves those
      // pointers null until the plan storage address is known.
      meep_cuda::validate_dft_batch_operations_fp32(
          host_operations.data(), host_operations.size(), total_block_count);
      for (std::size_t operation_index = 0;
           operation_index < host_operations.size(); ++operation_index)
        if (!same_operation(
                candidate.host_operations[operation_index],
                host_operations[operation_index]))
          return false;
      return true;
    };

    resident_cache::dft_batch_plan *selected_plan = nullptr;
    for (resident_cache::dft_batch_plan &candidate :
         plan_cache->dft_batch_plans)
      if (plan_matches(candidate)) {
        selected_plan = &candidate;
        break;
      }

    if (!selected_plan) {
      // Construct the O(logical-block-count) host map only when its device
      // image must change. A steady-state plan comparison is O(monitors) and
      // performs neither this allocation nor the map fill.
      const std::vector<std::uint32_t> block_operation_indices =
          make_phase_block_operation_indices(
              host_operations, total_block_count);
      void *replacement = meep_cuda::allocate_device_bytes(storage_bytes);
      bind_phase_pointers(replacement);
      try {
        // Validate precisely the host image copied below, including phase
        // pointers into the replacement allocation.  Keeping this inside the
        // guarded region also releases replacement if validation ever fails.
        meep_cuda::validate_dft_batch_operations_fp32(
            host_operations.data(), host_operations.size(),
            total_block_count);
        meep_cuda::copy_to_device(
            static_cast<unsigned char *>(replacement) + descriptor_offset,
            host_operations.data(), descriptor_bytes);
        meep_cuda::copy_to_device(
            static_cast<unsigned char *>(replacement) + block_map_offset,
            block_operation_indices.data(), block_map_bytes);
      }
      catch (...) {
        meep_cuda::free_device(replacement);
        throw;
      }
      int replacement_slot = -1;
      for (int slot = 0; slot < 2; ++slot)
        if (!plan_cache->dft_batch_plans[slot].device_pointer) {
          replacement_slot = slot;
          break;
        }
      if (replacement_slot < 0)
        replacement_slot = plan_cache->dft_batch_next_replacement;
      resident_cache::dft_batch_plan &plan =
          plan_cache->dft_batch_plans[replacement_slot];
      void *old = plan.device_pointer;
      plan.device_pointer = replacement;
      plan.capacity_bytes = storage_bytes;
      plan.descriptor_offset = descriptor_offset;
      plan.block_map_offset = block_map_offset;
      plan.total_block_count = total_block_count;
      plan.host_operations.swap(host_operations);
      plan_cache->dft_batch_next_replacement = 1 - replacement_slot;
      selected_plan = &plan;
      meep_cuda::free_device(old);
      device_buffer_allocations.fetch_add(1, std::memory_order_relaxed);
      if (!old)
        live_device_buffer_count().fetch_add(1, std::memory_order_relaxed);
      const std::size_t metadata_bytes = checked_sum(
          descriptor_bytes, block_map_bytes, "DFT metadata upload");
      host_to_device_bytes.fetch_add(
          static_cast<std::uint64_t>(metadata_bytes),
          std::memory_order_relaxed);
      dft_multi_monitor_plan_uploads.fetch_add(
          1, std::memory_order_relaxed);
      dft_multi_monitor_metadata_h2d_bytes.fetch_add(
          static_cast<std::uint64_t>(metadata_bytes),
          std::memory_order_relaxed);
    }
    else {
      device_buffer_reuses.fetch_add(1, std::memory_order_relaxed);
      dft_multi_monitor_plan_reuses.fetch_add(
          1, std::memory_order_relaxed);
    }

    resident_cache::dft_batch_plan &plan = *selected_plan;
    auto *phase_base = static_cast<meep_cuda::complex_value_fp32 *>(
        plan.device_pointer);
    for (std::size_t group_index = 0;
         group_index < phase_groups.size(); ++group_index) {
      const prepared_dft_update &operation =
          prepared[phase_groups[group_index].front()];
      const dft_update_request_fp32 &request = operation.request;
      meep_cuda::prepare_dft_phases_fp32(
          operation.device_frequencies,
          phase_base + phase_group_offsets[group_index],
          request.frequency_count, request.time, request.scale_real,
          request.scale_imag);
      ++phase_preparations;
      phase_reuse_count += phase_groups[group_index].size() - 1;
    }
    meep_cuda::update_dft_batch_fp32(
        static_cast<const meep_cuda::dft_update_operation_fp32 *>(
            static_cast<const void *>(
                static_cast<const unsigned char *>(plan.device_pointer) +
                plan.descriptor_offset)),
        reinterpret_cast<const std::uint32_t *>(
            static_cast<const unsigned char *>(plan.device_pointer) +
            plan.block_map_offset),
        request_count, total_block_count);
    for (const prepared_dft_update &operation : prepared)
      mark_device_dirty(
          operation.cache, operation.request.dft_real_imag,
          operation.output_bytes);
    update_launches = 1;
    dft_multi_monitor_batched_updates.fetch_add(
        static_cast<std::uint64_t>(request_count),
        std::memory_order_relaxed);
  }

  dft_batch_calls.fetch_add(1, std::memory_order_relaxed);
  dft_batch_submitted_updates.fetch_add(
      static_cast<std::uint64_t>(request_count),
      std::memory_order_relaxed);
  dft_phase_preparation_launches.fetch_add(
      phase_preparations, std::memory_order_relaxed);
  dft_phase_reuses.fetch_add(
      phase_reuse_count, std::memory_order_relaxed);
  dft_update_kernel_launches.fetch_add(
      update_launches, std::memory_order_relaxed);
  std::uint64_t observed_maximum =
      dft_maximum_batch_size.load(std::memory_order_relaxed);
  while (observed_maximum < request_count &&
         !dft_maximum_batch_size.compare_exchange_weak(
             observed_maximum, static_cast<std::uint64_t>(request_count),
             std::memory_order_relaxed, std::memory_order_relaxed)) {}
  cuda_dft_calls.fetch_add(
      static_cast<std::uint64_t>(request_count),
      std::memory_order_relaxed);
  cuda_dft_points.fetch_add(
      total_complex_count, std::memory_order_relaxed);
#else
  (void)requests;
  (void)request_count;
  throw std::runtime_error(
      "Meep CUDA DFT batch called in a CPU-only build");
#endif
}

void resident_update_dft_fp32(
    resident_cache *cache, float *dft_real_imag, float *field_real,
    float *field_imag, std::size_t field_array_count,
    const std::ptrdiff_t *field_indices, const float *weights,
    std::size_t point_count, const double *angular_frequencies,
    std::size_t frequency_count, double time, double scale_real,
    double scale_imag, std::ptrdiff_t average_offset1,
    std::ptrdiff_t average_offset2) {
#if MEEP_HAVE_CUDA
  if (!cache || !cache->phase_active)
    throw std::logic_error(
        "Meep CUDA DFT update requires an active resident phase");
  if (!dft_real_imag || !field_real || !field_indices || !weights ||
      !angular_frequencies)
    throw std::invalid_argument(
        "Meep CUDA DFT output, field, indices, weights, and angular "
        "frequencies must be non-null");
  if (point_count == 0 || frequency_count == 0) return;
  const dft_update_request_fp32 request = {
      cache, dft_real_imag, field_real, field_imag, field_array_count,
      field_indices, weights, point_count, angular_frequencies,
      frequency_count, time, scale_real, scale_imag, average_offset1,
      average_offset2};
  resident_update_dft_batch_fp32(&request, 1);
#else
  (void)cache;
  (void)dft_real_imag;
  (void)field_real;
  (void)field_imag;
  (void)field_array_count;
  (void)field_indices;
  (void)weights;
  (void)point_count;
  (void)angular_frequencies;
  (void)frequency_count;
  (void)time;
  (void)scale_real;
  (void)scale_imag;
  (void)average_offset1;
  (void)average_offset2;
  throw std::runtime_error(
      "Meep CUDA DFT update called in a CPU-only build");
#endif
}

static void resident_materialize_dft_array_fp32_impl(
    const dft_materialization_request_fp32 *requests,
    std::size_t request_count, float *host_output_real_imag,
    std::size_t output_point_count,
    std::size_t output_frequency_count,
    const dft_materialization_collapse_fp32 *collapse,
    std::size_t reduced_output_point_count) {
#if MEEP_HAVE_CUDA
  if (!host_output_real_imag)
    throw std::invalid_argument(
        "Meep CUDA DFT materialization output must be non-null");
  if (!requests && request_count)
    throw std::invalid_argument(
        "Meep CUDA DFT materialization requests must be non-null");
  if (request_count >
      static_cast<std::size_t>(
          std::numeric_limits<std::uint32_t>::max()))
    throw std::overflow_error(
        "Meep CUDA DFT materialization has too many requests");
  if (output_point_count == 0 || output_frequency_count == 0)
    throw std::invalid_argument(
        "Meep CUDA DFT materialization dimensions must be nonzero");

  const auto checked_sum = [](std::size_t left, std::size_t right,
                              const char *label) {
    if (left > std::numeric_limits<std::size_t>::max() - right)
      throw std::overflow_error(
          std::string("Meep CUDA ") + label + " size overflow");
    return left + right;
  };
  const auto align_up = [&checked_sum](std::size_t value,
                                       std::size_t alignment,
                                       const char *label) {
    const std::size_t remainder = value % alignment;
    return remainder == 0
               ? value
               : checked_sum(value, alignment - remainder, label);
  };

  const bool collapse_active = collapse != nullptr;
  meep_cuda::dft_collapse_layout_fp32 runtime_collapse = {
      0, {1, 1, 1}, {0, 0, 0}};
  std::size_t publication_point_count = output_point_count;
  if (collapse_active) {
    if (collapse->full_rank == 0 || collapse->full_rank > 3)
      throw std::invalid_argument(
          "Meep CUDA DFT collapse rank must be in [1, 3]");
    runtime_collapse.full_rank = collapse->full_rank;
    std::size_t full_product = 1;
    std::size_t reduced_product = 1;
    bool has_collapsed_dimension = false;
    for (std::size_t dim = 0; dim < 3; ++dim) {
      runtime_collapse.full_dims[dim] = collapse->full_dims[dim];
      runtime_collapse.collapsed[dim] = collapse->collapsed[dim];
      if (collapse->collapsed[dim] > 1)
        throw std::invalid_argument(
            "Meep CUDA DFT collapse flags must be zero or one");
      if (dim >= collapse->full_rank) {
        if (collapse->full_dims[dim] != 1 ||
            collapse->collapsed[dim] != 0)
          throw std::invalid_argument(
              "Meep CUDA DFT collapse inactive dimensions must be unit "
              "and retained");
        continue;
      }
      if (collapse->full_dims[dim] == 0)
        throw std::invalid_argument(
            "Meep CUDA DFT collapse dimensions must be nonzero");
      full_product = checked_product(
          full_product, collapse->full_dims[dim],
          "DFT collapse full extent");
      if (collapse->collapsed[dim] != 0)
        has_collapsed_dimension = true;
      else
        reduced_product = checked_product(
            reduced_product, collapse->full_dims[dim],
            "DFT collapse reduced extent");
    }
    if (!has_collapsed_dimension)
      throw std::invalid_argument(
          "Meep CUDA DFT collapse requires a collapsed dimension");
    if (full_product != output_point_count ||
        reduced_product != reduced_output_point_count)
      throw std::invalid_argument(
          "Meep CUDA DFT collapse layout differs from output extents");
    publication_point_count = reduced_output_point_count;
  }

  const std::size_t output_complex_count = checked_product(
      publication_point_count, output_frequency_count,
      "DFT materialization output");
  const std::size_t output_scalar_count = checked_product(
      output_complex_count, static_cast<std::size_t>(2),
      "DFT materialization output scalar");
  const std::size_t output_bytes = checked_bytes(
      output_scalar_count, sizeof(float),
      "DFT materialization output");

  if (request_count == 0) {
    // A distributed monitor may have no local chunks on this rank while
    // another rank owns the complete contribution.  Produce the additive
    // identity on the host without pretending that useful GPU work occurred:
    // the public call is still accounted by fields::get_dft_array, but this
    // local path performs no allocation, kernel launch, or D2H transfer.
    std::fill(host_output_real_imag,
              host_output_real_imag + output_scalar_count, 0.0f);
    return;
  }

  // A point-spectrum consumer usually asks for one frequency at a time. For
  // small spatial outputs, materialize every frequency once and retain that
  // device result. Subsequent calls copy only the selected frequency slice;
  // they do not upload metadata or launch a kernel. The 16 MiB cap prevents a
  // large spatial monitor from turning this latency optimization into a
  // second full-size DFT allocation.
  bool all_frequency_cache = output_frequency_count == 1;
  const std::size_t common_source_frequency_count =
      requests[0].source_frequency_count;
  const std::size_t requested_frequency_start =
      requests[0].source_frequency_start;
  for (std::size_t index = 0; index < request_count; ++index) {
    const dft_materialization_request_fp32 &request = requests[index];
    all_frequency_cache =
        all_frequency_cache && request.selected_frequency_count == 1 &&
        request.source_frequency_count == common_source_frequency_count &&
        request.source_frequency_start == requested_frequency_start;
  }
  std::size_t dense_cached_output_bytes = checked_bytes(
      checked_product(
          checked_product(output_point_count, output_frequency_count,
                          "DFT dense materialization output"),
          static_cast<std::size_t>(2),
          "DFT dense materialization output scalar"),
      sizeof(float), "DFT dense materialization output");
  if (all_frequency_cache) {
    const std::size_t cached_complex_count = checked_product(
        output_point_count, common_source_frequency_count,
        "DFT all-frequency cache");
    const std::size_t cached_scalar_count = checked_product(
        cached_complex_count, static_cast<std::size_t>(2),
        "DFT all-frequency cache scalar");
    dense_cached_output_bytes = checked_bytes(
        cached_scalar_count, sizeof(float),
        "DFT all-frequency cache");
    std::size_t candidate_output_bytes = dense_cached_output_bytes;
    if (collapse_active) {
      const std::size_t candidate_collapsed_bytes = checked_bytes(
          checked_product(
              checked_product(reduced_output_point_count,
                              common_source_frequency_count,
                              "DFT collapsed all-frequency cache"),
              static_cast<std::size_t>(2),
              "DFT collapsed all-frequency cache scalar"),
          sizeof(float), "DFT collapsed all-frequency cache");
      candidate_output_bytes = checked_sum(
          candidate_output_bytes, candidate_collapsed_bytes,
          "DFT combined all-frequency cache");
    }
    all_frequency_cache =
        output_point_count <= 64 &&
        candidate_output_bytes <=
            static_cast<std::size_t>(16) * 1024 * 1024;
  }
  const std::size_t materialized_frequency_count =
      all_frequency_cache ? common_source_frequency_count
                          : output_frequency_count;
  if (!all_frequency_cache)
    dense_cached_output_bytes = checked_bytes(
        checked_product(
            checked_product(output_point_count, output_frequency_count,
                            "DFT dense selected output"),
            static_cast<std::size_t>(2),
            "DFT dense selected output scalar"),
        sizeof(float), "DFT dense selected output");
  const std::size_t collapsed_cached_output_bytes =
      collapse_active
          ? checked_bytes(
                checked_product(
                    checked_product(reduced_output_point_count,
                                    materialized_frequency_count,
                                    "DFT collapsed cache"),
                    static_cast<std::size_t>(2),
                    "DFT collapsed cache scalar"),
                sizeof(float), "DFT collapsed cache")
          : 0;
  const std::size_t collapsed_output_offset =
      collapse_active ? dense_cached_output_bytes : 0;
  const std::size_t device_output_bytes = checked_sum(
      dense_cached_output_bytes, collapsed_cached_output_bytes,
      "DFT materialization device output");

  std::vector<meep_cuda::dft_materialization_operation_fp32>
      host_operations;
  host_operations.reserve(request_count);
  std::vector<resident_cache::dft_materialization_request_signature>
      request_signatures;
  request_signatures.reserve(request_count);
  std::vector<resident_cache::dft_materialization_dependency>
      dependencies;
  dependencies.reserve(request_count);
  std::vector<std::uint32_t> block_operation_indices;
  std::size_t total_point_count = 0;
  std::size_t total_block_count = 0;
  std::uint64_t full_dft_d2h_bytes_avoided = 0;
  std::set<std::pair<const resident_cache *, const void *> >
      avoided_source_dependencies;
  resident_cache *plan_cache = nullptr;

  for (std::size_t request_index = 0; request_index < request_count;
       ++request_index) {
    const dft_materialization_request_fp32 &request =
        requests[request_index];
    if (!request.owner || !request.dft_real_imag ||
        !request.destination_indices || !request.point_weights)
      throw std::invalid_argument(
          "Meep CUDA DFT materialization request is incomplete");
    if (request.storage_point_count == 0 || request.point_count == 0 ||
        request.source_frequency_count == 0 ||
        request.selected_frequency_count == 0)
      throw std::invalid_argument(
          "Meep CUDA DFT materialization request has empty work");
    if (request.point_count > request.storage_point_count)
      throw std::invalid_argument(
          "Meep CUDA DFT materialization point count exceeds storage");
    if (request.selected_frequency_count != output_frequency_count)
      throw std::invalid_argument(
          "Meep CUDA DFT materialization frequency extent is inconsistent");
    if (request.source_frequency_start >=
            request.source_frequency_count ||
        request.selected_frequency_count >
            request.source_frequency_count -
                request.source_frequency_start)
      throw std::out_of_range(
          "Meep CUDA DFT materialization source frequency is out of range");

    const std::size_t source_complex_count = checked_product(
        request.storage_point_count, request.source_frequency_count,
        "DFT materialization source");
    const std::size_t source_scalar_count = checked_product(
        source_complex_count, static_cast<std::size_t>(2),
        "DFT materialization source scalar");
    const std::size_t source_bytes = checked_bytes(
        source_scalar_count, sizeof(float),
        "DFT materialization source");
    // Reject overlapping aliases inside this request batch before any owner
    // cache can be created or resized. Exact reuse of one complete source
    // allocation is valid; an interior or straddling range is not.
    for (const auto &dependency : dependencies)
      if (dependency.owner == request.owner) {
        const std::uintptr_t existing_begin =
            reinterpret_cast<std::uintptr_t>(dependency.host_dft);
        const std::uintptr_t request_begin =
            reinterpret_cast<std::uintptr_t>(request.dft_real_imag);
        if (dependency.bytes >
                std::numeric_limits<std::uintptr_t>::max() -
                    existing_begin ||
            source_bytes >
                std::numeric_limits<std::uintptr_t>::max() - request_begin)
          throw std::overflow_error(
              "Meep CUDA DFT materialization source address overflow");
        const std::uintptr_t existing_end =
            existing_begin + dependency.bytes;
        const std::uintptr_t request_end = request_begin + source_bytes;
        const bool overlaps = request_begin < existing_end &&
                              existing_begin < request_end;
        const bool exact = request_begin == existing_begin &&
                           source_bytes == dependency.bytes;
        if (overlaps && !exact)
          throw std::invalid_argument(
              "Meep CUDA DFT materialization source ranges overlap without "
              "matching one complete allocation");
      }

    const std::size_t operation_source_frequency_start =
        all_frequency_cache ? 0 : request.source_frequency_start;
    const std::size_t operation_frequency_count =
        all_frequency_cache ? request.source_frequency_count
                            : request.selected_frequency_count;

    const std::size_t operation_work = checked_product(
        request.point_count, operation_frequency_count,
        "DFT materialization operation");
    const std::size_t threads = static_cast<std::size_t>(
        meep_cuda::dft_materialization_threads_per_block_fp32);
    const std::size_t operation_blocks =
        operation_work / threads + (operation_work % threads != 0);
    const std::size_t block_start = total_block_count;
    total_block_count = checked_sum(
        total_block_count, operation_blocks,
        "DFT materialization logical block");
    total_point_count = checked_sum(
        total_point_count, request.point_count,
        "DFT materialization metadata point");
    host_operations.push_back(
        {request.dft_real_imag, request.destination_indices,
         request.point_weights,
         request.storage_point_count, request.point_count,
         request.source_frequency_count,
         operation_source_frequency_start, operation_frequency_count,
         {request.inverse_stored_weight_real,
          request.inverse_stored_weight_imaginary},
         block_start, request.zero_divisor_flags, nullptr});
    request_signatures.push_back(
        {request.owner, request.dft_real_imag,
         request.storage_point_count, request.point_count,
         request.source_frequency_count, operation_source_frequency_start,
         operation_frequency_count, request.inverse_stored_weight_real,
         request.inverse_stored_weight_imaginary});
    dependencies.push_back(
        {request.owner, nullptr, request.dft_real_imag, nullptr,
         source_bytes, 0, 0});
  }

  if (total_block_count > block_operation_indices.max_size())
    throw std::overflow_error(
        "Meep CUDA DFT materialization block map is too large");
  block_operation_indices.resize(total_block_count);
  for (std::size_t operation_index = 0;
       operation_index < host_operations.size(); ++operation_index) {
    const std::size_t block_end =
        operation_index + 1 < host_operations.size()
            ? host_operations[operation_index + 1].block_start
            : total_block_count;
    for (std::size_t block = host_operations[operation_index].block_start;
         block < block_end; ++block)
      block_operation_indices[block] =
          static_cast<std::uint32_t>(operation_index);
  }
  std::vector<std::ptrdiff_t> flat_destinations;
  std::vector<float> flat_weights;
  std::vector<std::uint8_t> flat_zero_divisor_flags;
  std::vector<std::uint8_t> flat_publication_flags;
  flat_destinations.reserve(total_point_count);
  flat_weights.reserve(total_point_count);
  flat_zero_divisor_flags.reserve(total_point_count);
  for (const auto &operation : host_operations) {
    flat_destinations.insert(
        flat_destinations.end(), operation.destination_indices,
        operation.destination_indices + operation.point_count);
    flat_weights.insert(
        flat_weights.end(), operation.point_weights,
        operation.point_weights + operation.point_count);
    if (operation.zero_divisor_flags)
      flat_zero_divisor_flags.insert(
          flat_zero_divisor_flags.end(), operation.zero_divisor_flags,
          operation.zero_divisor_flags + operation.point_count);
    else
      flat_zero_divisor_flags.insert(
          flat_zero_divisor_flags.end(), operation.point_count, 0u);
  }
  // process_dft_component publishes in chunk-list, chunk, then point order
  // using assignment rather than accumulation. Closed surfaces and symmetry
  // images can therefore map multiple valid samples to one dense point. Mark
  // only the final occurrence so the parallel scatter exactly preserves the
  // CPU last-writer contract without a race or a serial kernel.
  flat_publication_flags.assign(total_point_count, 1u);
  std::unordered_map<std::ptrdiff_t, std::size_t> final_publication;
  final_publication.reserve(total_point_count);
  for (std::size_t point = 0; point < total_point_count; ++point)
    final_publication[flat_destinations[point]] = point;
  for (std::size_t point = 0; point < total_point_count; ++point)
    if (final_publication[flat_destinations[point]] != point)
      flat_publication_flags[point] = 0u;
  std::size_t host_point_offset = 0;
  for (auto &operation : host_operations) {
    operation.publication_flags =
        flat_publication_flags.data() + host_point_offset;
    host_point_offset += operation.point_count;
  }

  meep_cuda::validate_dft_materialization_operations_fp32(
      host_operations.empty() ? nullptr : host_operations.data(),
      block_operation_indices.empty()
          ? nullptr
          : block_operation_indices.data(),
      host_operations.size(), total_block_count, output_point_count,
      materialized_frequency_count);

  // Finish the complete metadata layout, including every byte/offset
  // product, before consulting or creating an owner cache.  Keeping this
  // pure calculation in preflight makes even an arithmetic rejection
  // failure-atomic for every request in the batch.
  const std::size_t destination_bytes = checked_bytes(
      flat_destinations.size(), sizeof(std::ptrdiff_t),
      "DFT materialization destination metadata");
  const std::size_t weight_offset = align_up(
      destination_bytes, alignof(float),
      "DFT materialization weight alignment");
  const std::size_t weight_bytes = checked_bytes(
      flat_weights.size(), sizeof(float),
      "DFT materialization weight metadata");
  const std::size_t zero_divisor_offset = checked_sum(
      weight_offset, weight_bytes,
      "DFT materialization zero-divisor metadata offset");
  const std::size_t zero_divisor_bytes = checked_bytes(
      flat_zero_divisor_flags.size(), sizeof(std::uint8_t),
      "DFT materialization zero-divisor metadata");
  const std::size_t publication_offset = checked_sum(
      zero_divisor_offset, zero_divisor_bytes,
      "DFT materialization publication metadata offset");
  const std::size_t publication_bytes = checked_bytes(
      flat_publication_flags.size(), sizeof(std::uint8_t),
      "DFT materialization publication metadata");
  const std::size_t descriptor_offset = align_up(
      checked_sum(publication_offset, publication_bytes,
                  "DFT materialization publication metadata"),
      alignof(meep_cuda::dft_materialization_operation_fp32),
      "DFT materialization descriptor alignment");
  const std::size_t descriptor_bytes = checked_bytes(
      host_operations.size(),
      sizeof(meep_cuda::dft_materialization_operation_fp32),
      "DFT materialization descriptor metadata");
  const std::size_t block_map_offset = align_up(
      checked_sum(descriptor_offset, descriptor_bytes,
                  "DFT materialization descriptor metadata"),
      alignof(std::uint32_t),
      "DFT materialization block-map alignment");
  const std::size_t block_map_bytes = checked_bytes(
      block_operation_indices.size(), sizeof(std::uint32_t),
      "DFT materialization block-map metadata");
  const std::size_t metadata_bytes = checked_sum(
      block_map_offset, block_map_bytes,
      "DFT materialization metadata");

  // All host metadata and products are valid. Before opening or creating a
  // cache, reject a range that is not one exact resident mirror and reject
  // owners already bound to a different CUDA device. This keeps malformed
  // calls failure-atomic even when the prior mirror is device-authoritative.
  const int selected_device = active_cuda_device_ordinal();
  for (const auto &dependency : dependencies) {
    resident_cache *const existing_cache =
        find_resident_cache_for_owner(dependency.owner);
    if (!existing_cache) continue;
    if (existing_cache->phase_active)
      throw std::logic_error(
          "cannot materialize a DFT array during an active Meep CUDA phase");
    const resolved_resident_range existing = resolve_resident_range(
        existing_cache, dependency.host_dft, dependency.bytes, true,
        "Meep CUDA DFT materialization source");
    if (existing.mirror &&
        (existing.host_base != dependency.host_dft ||
         existing.byte_offset != 0 ||
         existing.mirror->bytes != dependency.bytes))
      throw std::invalid_argument(
          "Meep CUDA DFT materialization source must match one complete "
          "resident mirror");
    if (existing_cache->device_ordinal >= 0 &&
        existing_cache->device_ordinal != selected_device)
      throw std::invalid_argument(
          "Meep CUDA DFT materialization owners must use the selected "
          "CUDA device");
  }

  meep_cuda::select_device(selected_device);
  for (std::size_t index = 0; index < dependencies.size(); ++index) {
    resident_cache::dft_materialization_dependency &dependency =
        dependencies[index];
    dependency.cache = resident_cache_for_owner(dependency.owner);
    if (!dependency.cache || dependency.cache->phase_active)
      throw std::logic_error(
          "cannot materialize a DFT array during an active Meep CUDA phase");
    if (dependency.cache->device_ordinal >= 0 &&
        dependency.cache->device_ordinal != selected_device)
      throw std::invalid_argument(
          "Meep CUDA DFT materialization owners must use the selected "
          "CUDA device");
    dependency.cache->device_ordinal = selected_device;
    if (!plan_cache)
      plan_cache = dependency.cache;
    const float *const device_dft = static_cast<const float *>(
        ensure_resident_mirror(dependency.cache, dependency.host_dft,
                               dependency.bytes));
    const auto source_mirror =
        dependency.cache->mirrors.find(dependency.host_dft);
    if (source_mirror == dependency.cache->mirrors.end() ||
        source_mirror->second.device_pointer != device_dft ||
        source_mirror->second.bytes != dependency.bytes)
      throw std::logic_error(
          "Meep CUDA DFT materialization source mirror is inconsistent");
    host_operations[index].dft_real_imag = device_dft;
    dependency.device_dft = device_dft;
    if (source_mirror->second.device_dirty &&
        avoided_source_dependencies
            .insert(std::make_pair(dependency.cache,
                                   dependency.host_dft))
            .second) {
      if (dependency.bytes >
          std::numeric_limits<std::uint64_t>::max() -
              full_dft_d2h_bytes_avoided)
        throw std::overflow_error(
            "Meep CUDA DFT materialization avoided-byte count overflow");
      full_dft_d2h_bytes_avoided +=
          static_cast<std::uint64_t>(dependency.bytes);
    }
  }

  // A later source in the same owner cache may advance its cache-wide
  // allocation generation. Snapshot dependencies only after every mirror is
  // prepared so an identical second call is a true topology hit.
  for (resident_cache::dft_materialization_dependency &dependency :
       dependencies) {
    const auto mirror =
        dependency.cache->mirrors.find(dependency.host_dft);
    if (mirror == dependency.cache->mirrors.end() ||
        mirror->second.device_pointer != dependency.device_dft ||
        mirror->second.bytes != dependency.bytes)
      throw std::logic_error(
          "Meep CUDA DFT materialization dependency changed while "
          "preparing its plan");
    dependency.allocation_generation =
        dependency.cache->allocation_generation;
    dependency.content_generation = mirror->second.content_generation;
  }

  if (!plan_cache)
    throw std::logic_error(
        "Meep CUDA DFT materialization has no plan cache");
  resident_cache::dft_materialization_plan &plan =
      plan_cache->dft_materialization;
  const auto same_float_bits = [](float left, float right) {
    std::uint32_t left_bits = 0;
    std::uint32_t right_bits = 0;
    std::memcpy(&left_bits, &left, sizeof(left_bits));
    std::memcpy(&right_bits, &right, sizeof(right_bits));
    return left_bits == right_bits;
  };
  const auto same_float_vector_bits = [&same_float_bits](
      const std::vector<float> &left,
      const std::vector<float> &right) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index)
      if (!same_float_bits(left[index], right[index])) return false;
    return true;
  };
  const auto same_signature = [&same_float_bits](
      const resident_cache::dft_materialization_request_signature &left,
      const resident_cache::dft_materialization_request_signature &right) {
    return left.owner == right.owner &&
           left.host_dft == right.host_dft &&
           left.storage_point_count == right.storage_point_count &&
           left.point_count == right.point_count &&
           left.source_frequency_count == right.source_frequency_count &&
           left.source_frequency_start == right.source_frequency_start &&
           left.selected_frequency_count == right.selected_frequency_count &&
           same_float_bits(left.inverse_stored_weight_real,
                           right.inverse_stored_weight_real) &&
           same_float_bits(left.inverse_stored_weight_imaginary,
                           right.inverse_stored_weight_imaginary);
  };
  const auto same_dependency_topology = [](
      const resident_cache::dft_materialization_dependency &left,
      const resident_cache::dft_materialization_dependency &right) {
    return left.owner == right.owner && left.cache == right.cache &&
           left.host_dft == right.host_dft &&
           left.device_dft == right.device_dft &&
           left.bytes == right.bytes &&
           left.allocation_generation == right.allocation_generation;
  };
  bool collapse_layout_matches =
      plan.collapse_active == collapse_active &&
      plan.collapsed_output_offset == collapsed_output_offset &&
      plan.collapsed_output_point_count ==
          (collapse_active ? reduced_output_point_count : 0) &&
      plan.collapse_full_rank ==
          (collapse_active ? collapse->full_rank : 0);
  if (collapse_layout_matches && collapse_active)
    for (std::size_t dim = 0; dim < 3; ++dim)
      if (plan.collapse_full_dims[dim] != collapse->full_dims[dim] ||
          plan.collapse_flags[dim] != collapse->collapsed[dim]) {
        collapse_layout_matches = false;
        break;
      }
  bool topology_matches =
      plan.device_metadata && plan.device_output &&
      plan.metadata_bytes == metadata_bytes &&
      plan.output_bytes == device_output_bytes &&
      plan.descriptor_offset == descriptor_offset &&
      plan.block_map_offset == block_map_offset &&
      plan.total_block_count == total_block_count &&
      plan.output_point_count == output_point_count &&
      plan.output_frequency_count == materialized_frequency_count &&
      plan.all_frequency_cache == all_frequency_cache &&
      collapse_layout_matches &&
      plan.requests.size() == request_signatures.size() &&
      plan.destinations == flat_destinations &&
      same_float_vector_bits(plan.weights, flat_weights) &&
      plan.zero_divisor_flags == flat_zero_divisor_flags &&
      plan.publication_flags == flat_publication_flags &&
      plan.dependencies.size() == dependencies.size();
  if (topology_matches)
    for (std::size_t index = 0; index < request_signatures.size(); ++index)
      if (!same_signature(plan.requests[index], request_signatures[index]) ||
          !same_dependency_topology(
              plan.dependencies[index], dependencies[index])) {
        topology_matches = false;
        break;
      }

  bool content_matches = topology_matches;
  if (content_matches)
    for (std::size_t index = 0; index < dependencies.size(); ++index)
      if (plan.dependencies[index].content_generation !=
          dependencies[index].content_generation) {
        content_matches = false;
        break;
      }

  void *active_metadata = plan.device_metadata;
  void *active_output = plan.device_output;
  void *replacement_metadata = nullptr;
  void *replacement_output = nullptr;
  bool launched = false;
  bool replacement_counted = false;
  try {
    if (!topology_matches) {
      replacement_metadata =
          meep_cuda::allocate_device_bytes(metadata_bytes);
      replacement_output =
          meep_cuda::allocate_device_bytes(device_output_bytes);
      device_buffer_allocations.fetch_add(2, std::memory_order_relaxed);
      live_device_buffer_count().fetch_add(2, std::memory_order_relaxed);
      replacement_counted = true;
      active_metadata = replacement_metadata;
      active_output = replacement_output;

      unsigned char *const metadata =
          static_cast<unsigned char *>(active_metadata);
      if (destination_bytes)
        meep_cuda::copy_to_device(
            metadata, flat_destinations.data(), destination_bytes);
      if (weight_bytes)
        meep_cuda::copy_to_device(
            metadata + weight_offset, flat_weights.data(), weight_bytes);
      if (zero_divisor_bytes)
        meep_cuda::copy_to_device(
            metadata + zero_divisor_offset,
            flat_zero_divisor_flags.data(), zero_divisor_bytes);
      if (publication_bytes)
        meep_cuda::copy_to_device(
            metadata + publication_offset,
            flat_publication_flags.data(), publication_bytes);
      std::size_t point_offset = 0;
      for (auto &operation : host_operations) {
        operation.destination_indices =
            reinterpret_cast<const std::ptrdiff_t *>(metadata) +
            point_offset;
        operation.point_weights =
            reinterpret_cast<const float *>(metadata + weight_offset) +
            point_offset;
        operation.zero_divisor_flags =
            reinterpret_cast<const std::uint8_t *>(
                metadata + zero_divisor_offset) + point_offset;
        operation.publication_flags =
            reinterpret_cast<const std::uint8_t *>(
                metadata + publication_offset) + point_offset;
        point_offset += operation.point_count;
      }
      meep_cuda::copy_to_device(
          metadata + descriptor_offset, host_operations.data(),
          descriptor_bytes);
      meep_cuda::copy_to_device(
          metadata + block_map_offset, block_operation_indices.data(),
          block_map_bytes);
      const std::size_t uploaded_bytes = checked_sum(
          checked_sum(
              checked_sum(destination_bytes, weight_bytes,
                          "DFT materialization metadata upload"),
              zero_divisor_bytes,
              "DFT materialization metadata upload"),
          checked_sum(
              publication_bytes,
              checked_sum(descriptor_bytes, block_map_bytes,
                          "DFT materialization metadata upload"),
              "DFT materialization metadata upload"),
          "DFT materialization metadata upload");
      host_to_device_bytes.fetch_add(
          static_cast<std::uint64_t>(uploaded_bytes),
          std::memory_order_relaxed);
    }
    else {
      device_buffer_reuses.fetch_add(2, std::memory_order_relaxed);
    }

    if (!content_matches) {
      const auto *device_operations = reinterpret_cast<const
          meep_cuda::dft_materialization_operation_fp32 *>(
          static_cast<const unsigned char *>(active_metadata) +
          descriptor_offset);
      const auto *device_block_map =
          reinterpret_cast<const std::uint32_t *>(
              static_cast<const unsigned char *>(active_metadata) +
              block_map_offset);
      meep_cuda::materialize_dft_fp32(
          device_operations, device_block_map, host_operations.size(),
          total_block_count, static_cast<float *>(active_output),
          output_point_count, materialized_frequency_count);
      if (collapse_active)
        meep_cuda::collapse_dft_array_fp32(
            static_cast<const float *>(active_output),
            reinterpret_cast<float *>(
                static_cast<unsigned char *>(active_output) +
                collapsed_output_offset),
            output_point_count, reduced_output_point_count,
            materialized_frequency_count, runtime_collapse);
      launched = true;
    }

    const std::size_t frequency_offset =
        all_frequency_cache ? requested_frequency_start : 0;
    const std::size_t scalar_offset = checked_product(
        checked_product(frequency_offset, publication_point_count,
                        "DFT materialization result offset"),
        static_cast<std::size_t>(2),
        "DFT materialization result scalar offset");
    const float *const publication_output =
        collapse_active
            ? reinterpret_cast<const float *>(
                  static_cast<const unsigned char *>(active_output) +
                  collapsed_output_offset)
            : static_cast<const float *>(active_output);
    meep_cuda::copy_to_host(
        host_output_real_imag, publication_output + scalar_offset,
        output_bytes);
  }
  catch (...) {
    meep_cuda::free_device(replacement_output);
    meep_cuda::free_device(replacement_metadata);
    if (replacement_counted)
      live_device_buffer_count().fetch_sub(2, std::memory_order_relaxed);
    throw;
  }

  if (!topology_matches) {
    const std::uint64_t released =
        (plan.device_metadata ? 1u : 0u) +
        (plan.device_output ? 1u : 0u);
    meep_cuda::free_device(plan.device_metadata);
    meep_cuda::free_device(plan.device_output);
    if (released)
      live_device_buffer_count().fetch_sub(
          released, std::memory_order_relaxed);
    plan.device_metadata = replacement_metadata;
    plan.device_output = replacement_output;
    plan.metadata_bytes = metadata_bytes;
    plan.output_bytes = device_output_bytes;
    plan.descriptor_offset = descriptor_offset;
    plan.block_map_offset = block_map_offset;
    plan.total_block_count = total_block_count;
    plan.output_point_count = output_point_count;
    plan.output_frequency_count = materialized_frequency_count;
    plan.all_frequency_cache = all_frequency_cache;
    plan.collapse_active = collapse_active;
    plan.collapsed_output_offset = collapsed_output_offset;
    plan.collapsed_output_point_count =
        collapse_active ? reduced_output_point_count : 0;
    plan.collapse_full_rank = collapse_active ? collapse->full_rank : 0;
    for (std::size_t dim = 0; dim < 3; ++dim) {
      plan.collapse_full_dims[dim] =
          collapse_active ? collapse->full_dims[dim] : 1;
      plan.collapse_flags[dim] =
          collapse_active ? collapse->collapsed[dim] : 0;
    }
    plan.requests.swap(request_signatures);
    plan.destinations.swap(flat_destinations);
    plan.weights.swap(flat_weights);
    plan.zero_divisor_flags.swap(flat_zero_divisor_flags);
    plan.publication_flags.swap(flat_publication_flags);
    plan.dependencies.swap(dependencies);
    replacement_metadata = nullptr;
    replacement_output = nullptr;
  }
  else if (launched) {
    for (std::size_t index = 0; index < dependencies.size(); ++index)
      plan.dependencies[index].content_generation =
          dependencies[index].content_generation;
  }

  device_to_host_bytes.fetch_add(
      static_cast<std::uint64_t>(output_bytes),
      std::memory_order_relaxed);
  cuda_dft_materialization_result_d2h_bytes.fetch_add(
      static_cast<std::uint64_t>(output_bytes),
      std::memory_order_relaxed);
  dft_materialization_full_dft_d2h_bytes_avoided.fetch_add(
      full_dft_d2h_bytes_avoided, std::memory_order_relaxed);
  if (launched)
    cuda_dft_materialization_kernel_launches.fetch_add(
        collapse_active ? 2 : 1, std::memory_order_relaxed);
#else
  (void)requests;
  (void)request_count;
  (void)host_output_real_imag;
  (void)output_point_count;
  (void)output_frequency_count;
  (void)collapse;
  (void)reduced_output_point_count;
  throw std::runtime_error(
      "Meep CUDA DFT materialization called in a CPU-only build");
#endif
}

void resident_materialize_dft_array_fp32(
    const dft_materialization_request_fp32 *requests,
    std::size_t request_count, float *host_output_real_imag,
    std::size_t output_point_count,
    std::size_t output_frequency_count) {
  resident_materialize_dft_array_fp32_impl(
      requests, request_count, host_output_real_imag, output_point_count,
      output_frequency_count, nullptr, 0);
}

void resident_materialize_collapsed_dft_array_fp32(
    const dft_materialization_request_fp32 *requests,
    std::size_t request_count, float *host_output_real_imag,
    std::size_t full_output_point_count,
    std::size_t reduced_output_point_count,
    std::size_t output_frequency_count,
    const dft_materialization_collapse_fp32 &collapse) {
  resident_materialize_dft_array_fp32_impl(
      requests, request_count, host_output_real_imag,
      full_output_point_count, output_frequency_count, &collapse,
      reduced_output_point_count);
}

resident_dft_output_staging_view_fp32 resident_stage_dft_output_fp32(
    const dft_output_staging_request_fp32 *requests,
    std::size_t request_count, std::size_t output_point_count,
    std::size_t frequency_start, std::size_t selected_frequency_count,
    std::size_t frequency_capacity) {
#if MEEP_HAVE_CUDA
#if !MEEP_SINGLE
    throw std::runtime_error(
        "Meep CUDA DFT output staging requires a single-precision build");
#endif
  if (!requests && request_count)
    throw std::invalid_argument(
        "Meep CUDA DFT output staging requests must be non-null");
  if (request_count == 0)
    throw std::invalid_argument(
        "Meep CUDA DFT output staging requires local requests; empty "
        "MPI ranks must be handled by the output layer");
  if (request_count >
      static_cast<std::size_t>(
          std::numeric_limits<std::uint32_t>::max()))
    throw std::overflow_error(
        "Meep CUDA DFT output staging has too many requests");
  if (output_point_count == 0 || selected_frequency_count == 0 ||
      frequency_capacity == 0)
    throw std::invalid_argument(
        "Meep CUDA DFT output staging dimensions must be nonzero");
  if (selected_frequency_count > frequency_capacity)
    throw std::invalid_argument(
        "Meep CUDA DFT output staging tile exceeds its capacity");

  const auto checked_sum = [](std::size_t left, std::size_t right,
                              const char *label) {
    if (left > std::numeric_limits<std::size_t>::max() - right)
      throw std::overflow_error(
          std::string("Meep CUDA ") + label + " size overflow");
    return left + right;
  };
  const auto align_up = [&checked_sum](std::size_t value,
                                       std::size_t alignment,
                                       const char *label) {
    const std::size_t remainder = value % alignment;
    return remainder == 0
               ? value
               : checked_sum(value, alignment - remainder, label);
  };

  const std::size_t capacity_values = checked_product(
      checked_product(output_point_count, frequency_capacity,
                      "DFT output staging capacity"),
      static_cast<std::size_t>(2),
      "DFT output staging planar capacity");
  const std::size_t output_bytes = checked_bytes(
      capacity_values, sizeof(float), "DFT output staging workspace");
  const std::size_t selected_values = checked_product(
      checked_product(output_point_count, selected_frequency_count,
                      "DFT output staging tile"),
      static_cast<std::size_t>(2), "DFT output staging planar tile");
  const std::size_t selected_bytes = checked_bytes(
      selected_values, sizeof(float), "DFT output staging tile");

  // Collect unique owners without opening resident phases yet.  Every host
  // pointer, extent, permutation, weight, tile, and workspace product is
  // validated below before an invalid call may create a cache, advance its
  // epoch, or migrate device-authoritative storage between GPUs.
  std::vector<const void *> unique_owners;
  unique_owners.reserve(request_count);
  for (std::size_t request_index = 0; request_index < request_count;
       ++request_index) {
    const void *const owner = requests[request_index].owner;
    if (!owner)
      throw std::invalid_argument(
          "Meep CUDA DFT output staging owner must be non-null");
    if (std::find(unique_owners.begin(), unique_owners.end(), owner) !=
        unique_owners.end())
      continue;
    unique_owners.push_back(owner);
  }

  std::vector<meep_cuda::dft_output_staging_operation_fp32>
      host_operations;
  std::vector<resident_cache::dft_output_staging_request_signature>
      request_signatures;
  std::vector<resident_cache::dft_output_staging_dependency>
      dependencies;
  std::vector<std::ptrdiff_t> flat_destinations;
  std::vector<float> flat_weights;
  std::vector<std::uint8_t> flat_zero_divisor_flags;
  std::vector<std::uint32_t> block_operation_indices;
  host_operations.reserve(request_count);
  request_signatures.reserve(request_count);
  dependencies.reserve(request_count);
  flat_destinations.reserve(output_point_count);
  flat_weights.reserve(output_point_count);
  flat_zero_divisor_flags.reserve(output_point_count);

  resident_cache *plan_cache = nullptr;
  std::size_t total_block_count = 0;
  std::size_t flattened_point_count = 0;
  std::uint64_t full_dft_d2h_bytes_avoided = 0;
  std::set<std::pair<const resident_cache *, const void *> >
      avoided_source_dependencies;

  for (std::size_t request_index = 0; request_index < request_count;
       ++request_index) {
    const dft_output_staging_request_fp32 &request =
        requests[request_index];
    if (!request.owner || !request.dft_real_imag ||
        !request.destination_indices || !request.point_weights)
      throw std::invalid_argument(
          "Meep CUDA DFT output staging request is incomplete");
    if (request.storage_point_count == 0 || request.point_count == 0 ||
        request.source_frequency_count == 0)
      throw std::invalid_argument(
          "Meep CUDA DFT output staging request has empty work");
    if (request.point_count > request.storage_point_count)
      throw std::invalid_argument(
          "Meep CUDA DFT output staging point count exceeds storage");
    if (!std::isfinite(request.inverse_stored_weight_real) ||
        !std::isfinite(request.inverse_stored_weight_imaginary))
      throw std::invalid_argument(
          "Meep CUDA DFT output staging inverse weight must be finite");
    if (frequency_start >= request.source_frequency_count ||
        selected_frequency_count >
            request.source_frequency_count - frequency_start)
      throw std::invalid_argument(
          "Meep CUDA DFT output staging source frequency range is invalid");

    const std::size_t source_complex_count = checked_product(
        request.storage_point_count, request.source_frequency_count,
        "DFT output staging source");
    const std::size_t source_scalar_count = checked_product(
        source_complex_count, static_cast<std::size_t>(2),
        "DFT output staging source scalar");
    const std::size_t source_bytes = checked_bytes(
        source_scalar_count, sizeof(float), "DFT output staging source");
    for (const auto &dependency : dependencies)
      if (dependency.owner == request.owner) {
        const std::uintptr_t existing_begin =
            reinterpret_cast<std::uintptr_t>(dependency.host_dft);
        const std::uintptr_t request_begin =
            reinterpret_cast<std::uintptr_t>(request.dft_real_imag);
        if (dependency.bytes >
                std::numeric_limits<std::uintptr_t>::max() -
                    existing_begin ||
            source_bytes >
                std::numeric_limits<std::uintptr_t>::max() - request_begin)
          throw std::overflow_error(
              "Meep CUDA DFT output staging source address overflow");
        const std::uintptr_t existing_end =
            existing_begin + dependency.bytes;
        const std::uintptr_t request_end = request_begin + source_bytes;
        const bool overlaps = request_begin < existing_end &&
                              existing_begin < request_end;
        const bool exact = request_begin == existing_begin &&
                           source_bytes == dependency.bytes;
        if (overlaps && !exact)
          throw std::invalid_argument(
              "Meep CUDA DFT output staging source ranges overlap without "
              "matching one complete allocation");
      }

    const std::size_t operation_work = checked_product(
        request.point_count, frequency_capacity,
        "DFT output staging operation");
    const std::size_t threads = static_cast<std::size_t>(
        meep_cuda::dft_output_staging_threads_per_block_fp32);
    const std::size_t operation_blocks =
        operation_work / threads + (operation_work % threads != 0);
    const std::size_t block_start = total_block_count;
    total_block_count = checked_sum(
        total_block_count, operation_blocks,
        "DFT output staging logical block");
    flattened_point_count = checked_sum(
        flattened_point_count, request.point_count,
        "DFT output staging metadata point");

    host_operations.push_back(
        {request.dft_real_imag, request.destination_indices,
         request.point_weights,
         request.zero_divisor_flags, request.storage_point_count,
         request.point_count, request.source_frequency_count,
         {request.inverse_stored_weight_real,
          request.inverse_stored_weight_imaginary},
         request.output_point_offset, block_start});
    request_signatures.push_back(
        {request.owner, request.dft_real_imag,
         request.storage_point_count, request.point_count,
         request.source_frequency_count,
         request.inverse_stored_weight_real,
         request.inverse_stored_weight_imaginary,
         request.output_point_offset});
    dependencies.push_back(
        {request.owner, nullptr, request.dft_real_imag, nullptr,
         source_bytes, 0, 0});
    flat_destinations.insert(
        flat_destinations.end(), request.destination_indices,
        request.destination_indices + request.point_count);
    flat_weights.insert(
        flat_weights.end(), request.point_weights,
        request.point_weights + request.point_count);
    if (request.zero_divisor_flags)
      flat_zero_divisor_flags.insert(
          flat_zero_divisor_flags.end(), request.zero_divisor_flags,
          request.zero_divisor_flags + request.point_count);
    else
      flat_zero_divisor_flags.insert(
          flat_zero_divisor_flags.end(), request.point_count, 0u);
  }

  if (flattened_point_count != output_point_count)
    throw std::invalid_argument(
        "Meep CUDA DFT output staging requests do not cover the packed "
        "output exactly");

  std::vector<std::size_t> block_starts;
  block_starts.reserve(host_operations.size());
  for (const auto &operation : host_operations)
    block_starts.push_back(operation.block_start);
  block_operation_indices = plan_phase_block_operation_indices(
      block_starts.data(), block_starts.size(), total_block_count);

  meep_cuda::validate_dft_output_staging_operations_fp32(
      host_operations.data(), block_operation_indices.data(),
      host_operations.size(), total_block_count, output_point_count,
      frequency_capacity);
  meep_cuda::validate_dft_output_staging_tile_fp32(
      host_operations.data(), host_operations.size(), frequency_start,
      selected_frequency_count, frequency_capacity);

  const std::size_t destination_bytes = checked_bytes(
      flat_destinations.size(), sizeof(std::ptrdiff_t),
      "DFT output staging destination metadata");
  const std::size_t weight_offset = align_up(
      destination_bytes, alignof(float),
      "DFT output staging weight alignment");
  const std::size_t weight_bytes = checked_bytes(
      flat_weights.size(), sizeof(float),
      "DFT output staging weight metadata");
  const std::size_t zero_divisor_offset = checked_sum(
      weight_offset, weight_bytes,
      "DFT output staging zero-divisor metadata offset");
  const std::size_t zero_divisor_bytes = checked_bytes(
      flat_zero_divisor_flags.size(), sizeof(std::uint8_t),
      "DFT output staging zero-divisor metadata");
  const std::size_t descriptor_offset = align_up(
      checked_sum(zero_divisor_offset, zero_divisor_bytes,
                  "DFT output staging zero-divisor metadata"),
      alignof(meep_cuda::dft_output_staging_operation_fp32),
      "DFT output staging descriptor alignment");
  const std::size_t descriptor_bytes = checked_bytes(
      host_operations.size(),
      sizeof(meep_cuda::dft_output_staging_operation_fp32),
      "DFT output staging descriptor metadata");
  const std::size_t block_map_offset = align_up(
      checked_sum(descriptor_offset, descriptor_bytes,
                  "DFT output staging descriptor metadata"),
      alignof(std::uint32_t),
      "DFT output staging block-map alignment");
  const std::size_t block_map_bytes = checked_bytes(
      block_operation_indices.size(), sizeof(std::uint32_t),
      "DFT output staging block-map metadata");
  const std::size_t metadata_bytes = checked_sum(
      block_map_offset, block_map_bytes,
      "DFT output staging metadata");
  const std::size_t retained_workspace = checked_sum(
      metadata_bytes,
      checked_sum(output_bytes,
                  checked_sum(output_bytes, output_bytes,
                              "DFT output staging retained workspace"),
                  "DFT output staging retained workspace"),
      "DFT output staging retained workspace");

  // Host pointers, extents, permutation, weights, flags, tile bounds, block
  // topology, and every workspace product have now been validated. Reject an
  // interior or straddling range against an already-resident allocation
  // before opening a phase: ensure_resident_mirror keys only by exact base and
  // must never create a second overlapping device mirror from stale host
  // bytes. Exact full-allocation reuse is the only accepted overlap.
  for (const auto &dependency : dependencies) {
    resident_cache *const cache =
        find_resident_cache_for_owner(dependency.owner);
    if (!cache) continue;
    const resolved_resident_range existing = resolve_resident_range(
        cache, dependency.host_dft, dependency.bytes, true,
        "Meep CUDA DFT output staging source");
    if (existing.mirror &&
        (existing.host_base != dependency.host_dft ||
         existing.byte_offset != 0 ||
         existing.mirror->bytes != dependency.bytes))
      throw std::invalid_argument(
          "Meep CUDA DFT output staging source must match one complete "
          "resident mirror");
  }

  const int selected_device = active_cuda_device_ordinal();
  meep_cuda::select_device(selected_device);

  // output_dft callers deal only in fields_chunk owners, never raw cache
  // handles. Open one phase per unique owner only after pure validation and
  // retain it across mirror preparation, launch, D2H, and publication.
  std::vector<std::unique_ptr<resident_curl_session> > owner_sessions;
  owner_sessions.reserve(unique_owners.size());
  for (const void *owner : unique_owners) {
    owner_sessions.emplace_back(new resident_curl_session(owner, true));
    if (!owner_sessions.back()->active())
      throw std::logic_error(
          "Meep CUDA DFT output staging could not open a resident phase");
  }
  const auto cache_for_owner = [&unique_owners, &owner_sessions](
                                   const void *owner) -> resident_cache * {
    const auto found =
        std::find(unique_owners.begin(), unique_owners.end(), owner);
    if (found == unique_owners.end())
      throw std::logic_error(
          "Meep CUDA DFT output staging owner disappeared during "
          "preparation");
    return owner_sessions[static_cast<std::size_t>(
                              found - unique_owners.begin())]
        ->cache();
  };

  for (auto &dependency : dependencies) {
    dependency.cache = cache_for_owner(dependency.owner);
    if (!dependency.cache || !dependency.cache->phase_active)
      throw std::logic_error(
          "Meep CUDA DFT output staging lost an owner resident phase");
    if (dependency.cache->device_ordinal != selected_device)
      throw std::invalid_argument(
          "Meep CUDA DFT output staging owners must use the selected "
          "CUDA device");
    if (!plan_cache) plan_cache = dependency.cache;
  }
  if (!plan_cache)
    throw std::logic_error(
        "Meep CUDA DFT output staging has no resident plan cache");

  // Only now may source mirrors be created or reused. This ordering makes
  // malformed requests side-effect free and keeps overlapping or
  // differently-sized device-dirty sources authoritative.
  for (std::size_t index = 0; index < dependencies.size(); ++index) {
    resident_cache::dft_output_staging_dependency &dependency =
        dependencies[index];
    const float *const device_dft = static_cast<const float *>(
        ensure_resident_mirror(dependency.cache, dependency.host_dft,
                               dependency.bytes));
    const auto source_mirror =
        dependency.cache->mirrors.find(dependency.host_dft);
    if (source_mirror == dependency.cache->mirrors.end() ||
        source_mirror->second.device_pointer != device_dft ||
        source_mirror->second.bytes != dependency.bytes)
      throw std::logic_error(
          "Meep CUDA DFT output staging source mirror is inconsistent");
    host_operations[index].dft_real_imag = device_dft;
    dependency.device_dft = device_dft;
    if (source_mirror->second.device_dirty &&
        avoided_source_dependencies
            .insert(std::make_pair(dependency.cache,
                                   dependency.host_dft))
            .second) {
      if (dependency.bytes >
          std::numeric_limits<std::uint64_t>::max() -
              full_dft_d2h_bytes_avoided)
        throw std::overflow_error(
            "Meep CUDA DFT output staging avoided-byte count overflow");
      full_dft_d2h_bytes_avoided +=
          static_cast<std::uint64_t>(dependency.bytes);
    }
  }

  // A later ensure_resident_mirror in the same cache can advance the
  // cache-wide allocation generation. Snapshot every dependency only after
  // all source mirrors exist, and verify its exact address/content identity.
  for (resident_cache::dft_output_staging_dependency &dependency :
       dependencies) {
    const auto mirror =
        dependency.cache->mirrors.find(dependency.host_dft);
    if (mirror == dependency.cache->mirrors.end() ||
        mirror->second.device_pointer != dependency.device_dft ||
        mirror->second.bytes != dependency.bytes)
      throw std::logic_error(
          "Meep CUDA DFT output staging dependency changed during plan "
          "preparation");
    dependency.allocation_generation =
        dependency.cache->allocation_generation;
    dependency.content_generation = mirror->second.content_generation;
  }

  resident_cache::dft_output_staging_plan &plan =
      plan_cache->dft_output_staging;
  const auto same_float_bits = [](float left, float right) {
    std::uint32_t left_bits = 0;
    std::uint32_t right_bits = 0;
    std::memcpy(&left_bits, &left, sizeof(left_bits));
    std::memcpy(&right_bits, &right, sizeof(right_bits));
    return left_bits == right_bits;
  };
  const auto same_float_vector_bits = [&same_float_bits](
      const std::vector<float> &left, const std::vector<float> &right) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index)
      if (!same_float_bits(left[index], right[index])) return false;
    return true;
  };
  const auto same_signature = [&same_float_bits](
      const resident_cache::dft_output_staging_request_signature &left,
      const resident_cache::dft_output_staging_request_signature &right) {
    return left.owner == right.owner &&
           left.host_dft == right.host_dft &&
           left.storage_point_count == right.storage_point_count &&
           left.point_count == right.point_count &&
           left.source_frequency_count == right.source_frequency_count &&
           same_float_bits(left.inverse_stored_weight_real,
                           right.inverse_stored_weight_real) &&
           same_float_bits(left.inverse_stored_weight_imaginary,
                           right.inverse_stored_weight_imaginary) &&
           left.output_point_offset == right.output_point_offset;
  };
  const auto same_dependency_topology = [](
      const resident_cache::dft_output_staging_dependency &left,
      const resident_cache::dft_output_staging_dependency &right) {
    return left.owner == right.owner && left.cache == right.cache &&
           left.host_dft == right.host_dft &&
           left.device_dft == right.device_dft &&
           left.bytes == right.bytes &&
           left.allocation_generation == right.allocation_generation;
  };
  bool topology_matches =
      plan.device_metadata && plan.device_output && plan.pinned_host &&
      plan.pinned_transfer &&
      plan.metadata_bytes == metadata_bytes &&
      plan.output_bytes == output_bytes &&
      plan.descriptor_offset == descriptor_offset &&
      plan.block_map_offset == block_map_offset &&
      plan.total_block_count == total_block_count &&
      plan.output_point_count == output_point_count &&
      plan.frequency_capacity == frequency_capacity &&
      plan.requests.size() == request_signatures.size() &&
      plan.destinations == flat_destinations &&
      same_float_vector_bits(plan.weights, flat_weights) &&
      plan.zero_divisor_flags == flat_zero_divisor_flags &&
      plan.block_operation_indices == block_operation_indices &&
      plan.dependencies.size() == dependencies.size();
  if (topology_matches)
    for (std::size_t index = 0; index < request_signatures.size(); ++index)
      if (!same_signature(plan.requests[index],
                          request_signatures[index]) ||
          !same_dependency_topology(plan.dependencies[index],
                                    dependencies[index])) {
        topology_matches = false;
        break;
      }

  // Content changes are expected after every DFT update, but explicitly
  // compare generations so a reused topology never implies a reused result.
  // Every call below launches again and publishes the new generations only
  // after the tile D2H succeeds.
  bool content_matches = topology_matches;
  if (content_matches)
    for (std::size_t index = 0; index < dependencies.size(); ++index)
      if (plan.dependencies[index].content_generation !=
          dependencies[index].content_generation) {
        content_matches = false;
        break;
      }
  (void)content_matches;

  void *active_metadata = plan.device_metadata;
  void *active_output = plan.device_output;
  float *active_transfer = plan.pinned_transfer.get();
  void *replacement_metadata = nullptr;
  void *replacement_output = nullptr;
  std::shared_ptr<float> replacement_pinned;
  std::shared_ptr<float> replacement_transfer;
  std::uint64_t replacement_device_buffers = 0;
  try {
    if (!topology_matches) {
      replacement_metadata =
          meep_cuda::allocate_device_bytes(metadata_bytes);
      ++replacement_device_buffers;
      device_buffer_allocations.fetch_add(1, std::memory_order_relaxed);
      live_device_buffer_count().fetch_add(1, std::memory_order_relaxed);
      replacement_output = meep_cuda::allocate_device_bytes(output_bytes);
      ++replacement_device_buffers;
      device_buffer_allocations.fetch_add(1, std::memory_order_relaxed);
      live_device_buffer_count().fetch_add(1, std::memory_order_relaxed);
      const auto allocate_pinned = [output_bytes]() {
        float *const pointer = static_cast<float *>(
            meep_cuda::allocate_pinned_bytes(output_bytes));
        // Count before constructing the shared_ptr.  The standard shared_ptr
        // constructor invokes its supplied deleter if allocating the control
        // block throws, so this also avoids a catch-path double free and keeps
        // the live allocation gauge exact under std::bad_alloc.
        live_dft_output_pinned_buffers.fetch_add(
            1, std::memory_order_relaxed);
        return std::shared_ptr<float>(pointer, [](float *allocated) {
          meep_cuda::free_pinned(allocated);
          live_dft_output_pinned_buffers.fetch_sub(
              1, std::memory_order_relaxed);
        });
      };
      replacement_pinned = allocate_pinned();
      replacement_transfer = allocate_pinned();
      active_metadata = replacement_metadata;
      active_output = replacement_output;
      active_transfer = replacement_transfer.get();

      unsigned char *const metadata =
          static_cast<unsigned char *>(active_metadata);
      meep_cuda::copy_to_device(
          metadata, flat_destinations.data(), destination_bytes);
      meep_cuda::copy_to_device(
          metadata + weight_offset, flat_weights.data(), weight_bytes);
      meep_cuda::copy_to_device(
          metadata + zero_divisor_offset,
          flat_zero_divisor_flags.data(), zero_divisor_bytes);
      std::size_t point_offset = 0;
      for (auto &operation : host_operations) {
        operation.destination_indices =
            reinterpret_cast<const std::ptrdiff_t *>(metadata) +
            point_offset;
        operation.point_weights =
            reinterpret_cast<const float *>(metadata + weight_offset) +
            point_offset;
        operation.zero_divisor_flags =
            reinterpret_cast<const std::uint8_t *>(
                metadata + zero_divisor_offset) + point_offset;
        point_offset += operation.point_count;
      }
      meep_cuda::copy_to_device(
          metadata + descriptor_offset, host_operations.data(),
          descriptor_bytes);
      meep_cuda::copy_to_device(
          metadata + block_map_offset, block_operation_indices.data(),
          block_map_bytes);
      const std::size_t uploaded_bytes = checked_sum(
          checked_sum(destination_bytes, weight_bytes,
                      "DFT output staging metadata upload"),
          checked_sum(
              zero_divisor_bytes,
              checked_sum(descriptor_bytes, block_map_bytes,
                          "DFT output staging metadata upload"),
              "DFT output staging metadata upload"),
          "DFT output staging metadata upload");
      host_to_device_bytes.fetch_add(
          static_cast<std::uint64_t>(uploaded_bytes),
          std::memory_order_relaxed);
    }
    else {
      device_buffer_reuses.fetch_add(2, std::memory_order_relaxed);
    }

    const auto *const device_operations = reinterpret_cast<const
        meep_cuda::dft_output_staging_operation_fp32 *>(
        static_cast<const unsigned char *>(active_metadata) +
        descriptor_offset);
    const auto *const device_block_map =
        reinterpret_cast<const std::uint32_t *>(
            static_cast<const unsigned char *>(active_metadata) +
            block_map_offset);
    meep_cuda::stage_dft_output_fp32(
        device_operations, device_block_map, host_operations.size(),
        total_block_count, static_cast<float *>(active_output),
        output_point_count, frequency_start, selected_frequency_count,
        frequency_capacity);
    meep_cuda::copy_to_host(
        active_transfer, active_output, selected_bytes);
    if (consume_countdown_failure_for_testing(
            dft_output_staging_d2h_failure_after_for_testing))
      throw std::runtime_error(
          "injected CUDA DFT output staging D2H publication failure");
  }
  catch (...) {
    replacement_transfer.reset();
    replacement_pinned.reset();
    meep_cuda::free_device(replacement_output);
    meep_cuda::free_device(replacement_metadata);
    if (replacement_device_buffers)
      live_device_buffer_count().fetch_sub(
          replacement_device_buffers, std::memory_order_relaxed);
    throw;
  }

  if (!topology_matches) {
    void *const old_metadata = plan.device_metadata;
    void *const old_output = plan.device_output;
    std::shared_ptr<float> old_pinned = plan.pinned_host;
    std::shared_ptr<float> old_transfer = plan.pinned_transfer;
    const std::uint64_t old_device_buffers =
        (old_metadata ? 1u : 0u) + (old_output ? 1u : 0u);
    plan.device_metadata = replacement_metadata;
    plan.device_output = replacement_output;
    // Publish the allocation that received the complete D2H and retain the
    // untouched peer as scratch for the next call. This pointer swap is
    // allocation-free and avoids an output-sized host memcpy.
    plan.pinned_host = std::move(replacement_transfer);
    plan.pinned_transfer = std::move(replacement_pinned);
    plan.metadata_bytes = metadata_bytes;
    plan.output_bytes = output_bytes;
    plan.descriptor_offset = descriptor_offset;
    plan.block_map_offset = block_map_offset;
    plan.total_block_count = total_block_count;
    plan.output_point_count = output_point_count;
    plan.frequency_capacity = frequency_capacity;
    plan.requests.swap(request_signatures);
    plan.destinations.swap(flat_destinations);
    plan.weights.swap(flat_weights);
    plan.zero_divisor_flags.swap(flat_zero_divisor_flags);
    plan.block_operation_indices.swap(block_operation_indices);
    plan.dependencies.swap(dependencies);
    replacement_metadata = nullptr;
    replacement_output = nullptr;
    old_transfer.reset();
    old_pinned.reset();
    meep_cuda::free_device(old_output);
    meep_cuda::free_device(old_metadata);
    if (old_device_buffers)
      live_device_buffer_count().fetch_sub(
          old_device_buffers, std::memory_order_relaxed);
  }
  else {
    // The active published view was never a D2H target. Once the transfer has
    // succeeded, exchanging the two owners is the atomic publication point.
    plan.pinned_host.swap(plan.pinned_transfer);
    for (std::size_t index = 0; index < dependencies.size(); ++index)
      plan.dependencies[index].content_generation =
          dependencies[index].content_generation;
  }

  device_to_host_bytes.fetch_add(
      static_cast<std::uint64_t>(selected_bytes),
      std::memory_order_relaxed);
  cuda_dft_output_staging_calls.fetch_add(1, std::memory_order_relaxed);
  cuda_dft_output_staging_points.fetch_add(
      static_cast<std::uint64_t>(output_point_count),
      std::memory_order_relaxed);
  cuda_dft_output_staging_frequencies.fetch_add(
      static_cast<std::uint64_t>(selected_frequency_count),
      std::memory_order_relaxed);
  if (topology_matches)
    cuda_dft_output_staging_plan_reuses.fetch_add(
        1, std::memory_order_relaxed);
  else
    cuda_dft_output_staging_descriptor_uploads.fetch_add(
        1, std::memory_order_relaxed);
  cuda_dft_output_staging_kernel_launches.fetch_add(
      1, std::memory_order_relaxed);
  cuda_dft_output_staging_result_d2h_bytes.fetch_add(
      static_cast<std::uint64_t>(selected_bytes),
      std::memory_order_relaxed);
  cuda_dft_output_staging_full_dft_d2h_bytes_avoided.fetch_add(
      full_dft_d2h_bytes_avoided, std::memory_order_relaxed);
  std::uint64_t observed_workspace =
      cuda_dft_output_staging_workspace_ceiling_bytes.load(
          std::memory_order_relaxed);
  const std::uint64_t candidate_workspace =
      static_cast<std::uint64_t>(retained_workspace);
  while (observed_workspace < candidate_workspace &&
         !cuda_dft_output_staging_workspace_ceiling_bytes
              .compare_exchange_weak(
                  observed_workspace, candidate_workspace,
                  std::memory_order_relaxed,
                  std::memory_order_relaxed)) {}

  for (auto &session : owner_sessions) session->finish(false);

  return {plan.pinned_host.get(), output_point_count, frequency_start,
          selected_frequency_count, frequency_capacity,
          std::shared_ptr<const float>(plan.pinned_host)};
#else
  (void)requests;
  (void)request_count;
  (void)output_point_count;
  (void)frequency_start;
  (void)selected_frequency_count;
  (void)frequency_capacity;
  throw std::runtime_error(
      "Meep CUDA DFT output staging called in a CPU-only build");
#endif
}

void resident_reduce_dft_pairs_fp32(
    const void *plan_owner, unsigned int plan_lane,
    const dft_pair_reduction_request_fp32 *requests,
    std::size_t request_count, std::size_t frequency_count,
    double *result_real_imag) {
#if MEEP_HAVE_CUDA
  if (!plan_owner || !result_real_imag)
    throw std::invalid_argument(
        "Meep CUDA DFT reduction owner and result must be non-null");
  if (!requests && request_count)
    throw std::invalid_argument(
        "Meep CUDA DFT reduction requests must be non-null");
  if (frequency_count == 0) {
    if (request_count != 0)
      throw std::invalid_argument(
          "Meep CUDA DFT reduction requests require frequencies");
    return;
  }
  const std::size_t result_scalar_count = checked_product(
      frequency_count, static_cast<std::size_t>(2),
      "DFT reduction result scalar");
  const std::size_t result_bytes = checked_bytes(
      result_scalar_count, sizeof(double), "DFT reduction result");
  if (request_count == 0) {
    std::fill(result_real_imag,
              result_real_imag + result_scalar_count, 0.0);
    return;
  }
  if (request_count >
      static_cast<std::size_t>(
          std::numeric_limits<std::uint32_t>::max()))
    throw std::overflow_error(
        "Meep CUDA DFT reduction request count exceeds block-map capacity");

  const auto convert_weight = [](double value, const char *label) {
    if (!std::isfinite(value))
      throw std::invalid_argument(
          std::string("Meep CUDA DFT reduction ") + label +
          " must be finite");
    const float converted = static_cast<float>(value);
    if (!std::isfinite(converted))
      throw std::invalid_argument(
          std::string("Meep CUDA DFT reduction ") + label +
          " is outside the FP32 range");
    return converted;
  };

  std::vector<const void *> owners;
  owners.reserve(checked_product(
      request_count, static_cast<std::size_t>(2),
      "DFT reduction owner reserve"));
  std::vector<float> weight_reals;
  std::vector<float> weight_imaginaries;
  weight_reals.reserve(request_count);
  weight_imaginaries.reserve(request_count);
  std::size_t total_spatial_block_count = 0;
  std::uint64_t total_terms = 0;
  for (std::size_t index = 0; index < request_count; ++index) {
    const dft_pair_reduction_request_fp32 &request = requests[index];
    if (!request.lhs_owner || !request.rhs_owner ||
        !request.lhs_real_imag || !request.rhs_real_imag ||
        request.point_count == 0 ||
        request.lhs_storage_point_count == 0 ||
        request.rhs_storage_point_count == 0)
      throw std::invalid_argument(
          "Meep CUDA DFT reduction request is incomplete");
    if (request.point_count > request.lhs_storage_point_count ||
        request.point_count > request.rhs_storage_point_count)
      throw std::out_of_range(
          "Meep CUDA DFT reduction extent exceeds monitor storage");
    const std::size_t complex_count = checked_product(
        request.point_count, frequency_count,
        "DFT reduction point-frequency count");
    (void)checked_bytes(
        checked_product(
            checked_product(request.lhs_storage_point_count,
                            frequency_count,
                            "DFT reduction lhs point-frequency count"),
            static_cast<std::size_t>(2),
            "DFT reduction lhs complex scalar count"),
        sizeof(float), "DFT reduction lhs monitor values");
    (void)checked_bytes(
        checked_product(
            checked_product(request.rhs_storage_point_count,
                            frequency_count,
                            "DFT reduction rhs point-frequency count"),
            static_cast<std::size_t>(2),
            "DFT reduction rhs complex scalar count"),
        sizeof(float), "DFT reduction rhs monitor values");
    if (complex_count >
        std::numeric_limits<std::uint64_t>::max() - total_terms)
      throw std::overflow_error(
          "Meep CUDA DFT reduction term counter overflow");
    total_terms += static_cast<std::uint64_t>(complex_count);
    const std::size_t operation_blocks =
        request.point_count /
            meep_cuda::dft_pair_reduction_point_threads +
        (request.point_count %
             meep_cuda::dft_pair_reduction_point_threads !=
         0);
    if (total_spatial_block_count >
        std::numeric_limits<std::size_t>::max() - operation_blocks)
      throw std::overflow_error(
          "Meep CUDA DFT reduction block count overflow");
    total_spatial_block_count += operation_blocks;
    weight_reals.push_back(
        convert_weight(request.weight_real, "real weight"));
    weight_imaginaries.push_back(
        convert_weight(request.weight_imaginary, "imaginary weight"));
    if (std::find(owners.begin(), owners.end(), request.lhs_owner) ==
        owners.end())
      owners.push_back(request.lhs_owner);
    if (std::find(owners.begin(), owners.end(), request.rhs_owner) ==
        owners.end())
      owners.push_back(request.rhs_owner);
  }
  if (total_spatial_block_count == 0)
    throw std::logic_error(
        "Meep CUDA DFT reduction block plan is empty");

  std::vector<resident_curl_session> sessions;
  sessions.reserve(owners.size());
  for (const void *owner : owners)
    sessions.emplace_back(owner, true);
  if (sessions.empty() || !sessions[0].active() ||
      !sessions[0].cache())
    throw std::logic_error(
        "Meep CUDA DFT reduction has no active resident cache");
  const int device_ordinal = sessions[0].cache()->device_ordinal;
  for (const resident_curl_session &session : sessions)
    if (!session.active() || !session.cache() ||
        session.cache()->device_ordinal != device_ordinal)
      throw std::logic_error(
          "Meep CUDA DFT reduction owners must use one device");
  meep_cuda::select_device(device_ordinal);

  const auto cache_for_owner =
      [&owners, &sessions](const void *owner) -> resident_cache * {
    const auto found = std::find(owners.begin(), owners.end(), owner);
    if (found == owners.end())
      throw std::logic_error(
          "Meep CUDA DFT reduction owner disappeared during preparation");
    return sessions[static_cast<std::size_t>(
                        found - owners.begin())]
        .cache();
  };

  struct prepared_mirror {
    resident_cache *cache = nullptr;
    const float *host = nullptr;
    std::size_t bytes = 0;
    const float *device = nullptr;
    bool was_device_dirty = false;
  };
  std::vector<prepared_mirror> prepared_mirrors;
  prepared_mirrors.reserve(checked_product(
      request_count, static_cast<std::size_t>(2),
      "DFT reduction mirror reserve"));
  std::uint64_t avoided_dft_bytes = 0;
  const auto prepare_mirror =
      [&prepared_mirrors, &avoided_dft_bytes](
          resident_cache *cache, const float *host,
          std::size_t bytes) -> const float * {
    for (const prepared_mirror &prepared : prepared_mirrors)
      if (prepared.cache == cache && prepared.host == host &&
          prepared.bytes == bytes)
        return prepared.device;
    const resolved_resident_range existing = resolve_resident_range(
        cache, host, bytes, true, "Meep CUDA DFT reduction monitor");
    if (existing.mirror &&
        (existing.host_base != host || existing.byte_offset != 0 ||
         existing.mirror->bytes != bytes))
      throw std::invalid_argument(
          "Meep CUDA DFT reduction monitor must match one complete mirror");
    const bool was_device_dirty =
        existing.mirror && existing.mirror->device_dirty;
    const float *device = static_cast<const float *>(
        ensure_resident_mirror(cache, host, bytes));
    prepared_mirror prepared;
    prepared.cache = cache;
    prepared.host = host;
    prepared.bytes = bytes;
    prepared.device = device;
    prepared.was_device_dirty = was_device_dirty;
    prepared_mirrors.push_back(prepared);
    if (was_device_dirty) {
      if (bytes >
          std::numeric_limits<std::uint64_t>::max() - avoided_dft_bytes)
        throw std::overflow_error(
            "Meep CUDA DFT reduction avoided-byte counter overflow");
      avoided_dft_bytes += static_cast<std::uint64_t>(bytes);
    }
    return device;
  };

  std::vector<dft_reduction_request_snapshot> snapshots;
  snapshots.reserve(request_count);
  std::vector<meep_cuda::dft_pair_reduction_operation_fp32>
      host_operations;
  host_operations.reserve(request_count);
  std::vector<resident_cache *> cache_dependencies;
  cache_dependencies.reserve(owners.size());
  std::size_t block_start = 0;
  for (std::size_t index = 0; index < request_count; ++index) {
    const dft_pair_reduction_request_fp32 &request = requests[index];
    resident_cache *lhs_cache = cache_for_owner(request.lhs_owner);
    resident_cache *rhs_cache = cache_for_owner(request.rhs_owner);
    const auto monitor_bytes =
        [frequency_count](std::size_t storage_point_count,
                          const char *label) {
          return checked_bytes(
              checked_product(
                  checked_product(storage_point_count, frequency_count,
                                  label),
                  static_cast<std::size_t>(2), label),
              sizeof(float), label);
        };
    const std::size_t lhs_monitor_bytes = monitor_bytes(
        request.lhs_storage_point_count,
        "DFT reduction lhs monitor values");
    const std::size_t rhs_monitor_bytes = monitor_bytes(
        request.rhs_storage_point_count,
        "DFT reduction rhs monitor values");
    const float *lhs_device = prepare_mirror(
        lhs_cache, request.lhs_real_imag, lhs_monitor_bytes);
    const float *rhs_device = prepare_mirror(
        rhs_cache, request.rhs_real_imag, rhs_monitor_bytes);
    dft_reduction_request_snapshot snapshot;
    snapshot.lhs_cache = lhs_cache;
    snapshot.rhs_cache = rhs_cache;
    snapshot.lhs_host = request.lhs_real_imag;
    snapshot.rhs_host = request.rhs_real_imag;
    snapshot.lhs_device = lhs_device;
    snapshot.rhs_device = rhs_device;
    snapshot.point_count = request.point_count;
    snapshot.weight_real = weight_reals[index];
    snapshot.weight_imaginary = weight_imaginaries[index];
    snapshots.push_back(snapshot);
    host_operations.push_back(
        {lhs_device, rhs_device, request.point_count, block_start,
         {weight_reals[index], weight_imaginaries[index]}});
    block_start +=
        request.point_count /
            meep_cuda::dft_pair_reduction_point_threads +
        (request.point_count %
             meep_cuda::dft_pair_reduction_point_threads !=
         0);
    if (std::find(cache_dependencies.begin(), cache_dependencies.end(),
                  lhs_cache) == cache_dependencies.end())
      cache_dependencies.push_back(lhs_cache);
    if (std::find(cache_dependencies.begin(), cache_dependencies.end(),
                  rhs_cache) == cache_dependencies.end())
      cache_dependencies.push_back(rhs_cache);
  }
  if (block_start != total_spatial_block_count)
    throw std::logic_error(
        "Meep CUDA DFT reduction block plan changed during preparation");
  for (dft_reduction_request_snapshot &snapshot : snapshots) {
    snapshot.lhs_cache_generation =
        snapshot.lhs_cache->allocation_generation;
    snapshot.rhs_cache_generation =
        snapshot.rhs_cache->allocation_generation;
  }

  const std::vector<std::uint32_t> block_operation_indices =
      make_phase_block_operation_indices(
          host_operations, total_spatial_block_count);
  const std::size_t descriptor_bytes = checked_phase_product(
      host_operations.size(),
      sizeof(meep_cuda::dft_pair_reduction_operation_fp32),
      "DFT reduction descriptors");
  const std::size_t block_map_bytes = checked_phase_product(
      block_operation_indices.size(), sizeof(std::uint32_t),
      "DFT reduction block map");
  const std::size_t static_bytes = checked_phase_sum(
      descriptor_bytes, block_map_bytes, "DFT reduction static plan");
  const std::size_t partial_capacity = std::min(
      total_spatial_block_count,
      meep_cuda::dft_pair_reduction_partial_capacity);
  const std::size_t partial_value_count = checked_product(
      checked_product(frequency_count, partial_capacity,
                      "DFT reduction partial count"),
      static_cast<std::size_t>(2),
      "DFT reduction partial complex scalar");
  const std::size_t partial_bytes = checked_bytes(
      partial_value_count, sizeof(double),
      "DFT reduction partial workspace");

  dft_reduction_plan_registry &registry =
      get_dft_reduction_plan_registry();
  std::unique_lock<std::mutex> plan_lock(registry.mutex);
  dft_reduction_plan_key key;
  key.owner = plan_owner;
  key.lane = plan_lane;
  dft_reduction_retained_plan &plan = registry.plans[key];
  const bool plan_matches = same_dft_reduction_plan(
      plan, snapshots, device_ordinal, frequency_count,
      total_spatial_block_count);
  bool uploaded_plan = false;
  if (!plan_matches) {
    dft_reduction_retained_plan replacement;
    replacement.device_ordinal = device_ordinal;
    replacement.partial_capacity = partial_capacity;
    replacement.frequency_count = frequency_count;
    replacement.total_spatial_block_count =
        total_spatial_block_count;
    replacement.requests = snapshots;
    replacement.cache_dependencies = cache_dependencies;
    try {
      replacement.static_storage =
          meep_cuda::allocate_device_bytes(static_bytes);
      replacement.partial_workspace = static_cast<double *>(
          meep_cuda::allocate_device_bytes(partial_bytes));
      replacement.device_result = static_cast<double *>(
          meep_cuda::allocate_device_bytes(result_bytes));
      replacement.static_capacity_bytes = static_bytes;
      replacement.device_operations = static_cast<
          const meep_cuda::dft_pair_reduction_operation_fp32 *>(
              replacement.static_storage);
      replacement.device_block_map =
          reinterpret_cast<const std::uint32_t *>(
              static_cast<const unsigned char *>(
                  replacement.static_storage) + descriptor_bytes);
      meep_cuda::copy_to_device(
          replacement.static_storage, host_operations.data(),
          descriptor_bytes);
      host_to_device_bytes.fetch_add(
          static_cast<std::uint64_t>(descriptor_bytes),
          std::memory_order_relaxed);
      meep_cuda::copy_to_device(
          static_cast<unsigned char *>(replacement.static_storage) +
              descriptor_bytes,
          block_operation_indices.data(), block_map_bytes);
      host_to_device_bytes.fetch_add(
          static_cast<std::uint64_t>(block_map_bytes),
          std::memory_order_relaxed);
    }
    catch (...) {
      meep_cuda::free_device(replacement.static_storage);
      meep_cuda::free_device(replacement.partial_workspace);
      meep_cuda::free_device(replacement.device_result);
      replacement.static_storage = nullptr;
      replacement.partial_workspace = nullptr;
      replacement.device_result = nullptr;
      throw;
    }
    device_buffer_allocations.fetch_add(3, std::memory_order_relaxed);
    live_device_buffer_count().fetch_add(3, std::memory_order_relaxed);
    release_dft_reduction_plan(plan);
    plan = std::move(replacement);
    meep_cuda::select_device(device_ordinal);
    uploaded_plan = true;
  }
  else {
    device_buffer_reuses.fetch_add(3, std::memory_order_relaxed);
  }

  meep_cuda::dft_pair_reduce_fp32(
      plan.device_operations, plan.device_block_map, request_count,
      total_spatial_block_count, frequency_count,
      plan.partial_workspace, plan.partial_capacity,
      plan.device_result);
  std::vector<double> staged(result_scalar_count);
  meep_cuda::copy_to_host(
      staged.data(), plan.device_result, result_bytes);
  device_to_host_bytes.fetch_add(
      static_cast<std::uint64_t>(result_bytes),
      std::memory_order_relaxed);
  cuda_dft_reduction_result_d2h_bytes.fetch_add(
      static_cast<std::uint64_t>(result_bytes),
      std::memory_order_relaxed);
  for (double value : staged)
    if (!std::isfinite(value))
      throw std::runtime_error(
          "Meep CUDA DFT reduction produced a non-finite result");
  std::copy(staged.begin(), staged.end(), result_real_imag);

  cuda_dft_reduction_calls.fetch_add(1, std::memory_order_relaxed);
  cuda_dft_reduction_pairs.fetch_add(
      static_cast<std::uint64_t>(request_count),
      std::memory_order_relaxed);
  cuda_dft_reduction_terms.fetch_add(
      total_terms, std::memory_order_relaxed);
  if (uploaded_plan)
    cuda_dft_reduction_descriptor_uploads.fetch_add(
        1, std::memory_order_relaxed);
  else
    cuda_dft_reduction_plan_reuses.fetch_add(
        1, std::memory_order_relaxed);
  cuda_dft_reduction_kernel_launches.fetch_add(
      2, std::memory_order_relaxed);
  dft_reduction_full_dft_d2h_bytes_avoided.fetch_add(
      avoided_dft_bytes, std::memory_order_relaxed);
#else
  (void)plan_owner;
  (void)plan_lane;
  (void)requests;
  (void)request_count;
  (void)frequency_count;
  (void)result_real_imag;
  throw std::runtime_error(
      "Meep CUDA DFT reduction called in a CPU-only build");
#endif
}

void resident_reduce_eigenmode_overlaps_fp32(
    const void *plan_owner, eigenmode_overlap_batch_kind batch_kind,
    const eigenmode_overlap_request_fp32 *requests,
    std::size_t request_count, std::size_t selected_frequency_index,
    double result_real_imag[16]) {
#if MEEP_HAVE_CUDA
  const bool mode_flux =
      batch_kind != eigenmode_overlap_batch_kind::mode_mode;
  const bool mode_mode =
      batch_kind != eigenmode_overlap_batch_kind::mode_flux;
  const bool combined =
      batch_kind == eigenmode_overlap_batch_kind::both;
  const std::size_t output_count =
      combined ? meep_cuda::eigenmode_overlap_max_output_count
               : meep_cuda::eigenmode_overlap_output_count;
  const std::size_t logical_contraction_count = combined ? 2u : 1u;
  const std::size_t result_scalar_count = 2 * output_count;
  const std::size_t result_bytes =
      result_scalar_count * sizeof(double);
  if (!plan_owner || !result_real_imag)
    throw std::invalid_argument(
        "Meep CUDA eigenmode overlap owner and result must be non-null");
  if (!requests && request_count)
    throw std::invalid_argument(
        "Meep CUDA eigenmode overlap requests must be non-null");
  if (request_count == 0) {
    std::fill(result_real_imag,
              result_real_imag + result_scalar_count, 0.0);
    cuda_dft_overlap_calls.fetch_add(
        logical_contraction_count, std::memory_order_relaxed);
    if (mode_flux)
      cuda_eigenmode_mode_flux_calls.fetch_add(
          1, std::memory_order_relaxed);
    if (mode_mode)
      cuda_eigenmode_mode_mode_calls.fetch_add(
          1, std::memory_order_relaxed);
    return;
  }
  if (request_count >
      static_cast<std::size_t>(
          std::numeric_limits<std::uint32_t>::max()))
    throw std::overflow_error(
        "Meep CUDA eigenmode overlap request count exceeds block-map "
        "capacity");

  std::vector<const void *> owners;
  owners.reserve(request_count);
  std::vector<meep_cuda::complex_value_fp64> packed_profiles;
  std::vector<std::uint8_t> packed_zero_flags;
  std::vector<std::size_t> mode1_offsets;
  std::vector<std::size_t> mode2_offsets;
  std::vector<std::size_t> mode_mode1_offsets;
  std::vector<std::size_t> zero_flag_offsets;
  mode1_offsets.reserve(request_count);
  mode2_offsets.reserve(request_count);
  mode_mode1_offsets.reserve(request_count);
  zero_flag_offsets.reserve(request_count);
  std::size_t profile_value_count = 0;
  std::size_t total_block_count = 0;
  std::uint64_t total_terms = 0;
  for (std::size_t index = 0; index < request_count; ++index) {
    const eigenmode_overlap_request_fp32 &request = requests[index];
    if (!request.owner || !request.weighted_conjugate_mode ||
        request.point_count == 0 ||
        request.output_index >=
            meep_cuda::eigenmode_overlap_output_count)
      throw std::invalid_argument(
          "Meep CUDA eigenmode overlap request is incomplete");
    if (mode_mode && !request.mode2_real_imag)
      throw std::invalid_argument(
          "Meep CUDA mode-mode overlap requires a mode-2 profile");
    if (!mode_mode &&
        (request.mode2_real_imag ||
         request.mode_mode_weighted_conjugate_mode))
      throw std::invalid_argument(
          "Meep CUDA mode-flux overlap must not carry mode-mode profiles");
    if (combined && !request.mode_mode_weighted_conjugate_mode)
      throw std::invalid_argument(
          "Meep CUDA fused eigenmode overlap requires both weighted "
          "mode-1 profiles");
    if (!combined && request.mode_mode_weighted_conjugate_mode)
      throw std::invalid_argument(
          "Meep CUDA single eigenmode overlap has an unexpected second "
          "weighted mode-1 profile");
    if (!mode_flux) {
      if (request.dft_real_imag ||
          request.zero_normalization_divisors ||
          request.dft_storage_point_count != 0 ||
          request.dft_frequency_count != 0)
        throw std::invalid_argument(
            "Meep CUDA mode-mode overlap must not consume DFT storage");
    }
    else {
      if (!request.dft_real_imag ||
          request.dft_storage_point_count == 0 ||
          request.point_count > request.dft_storage_point_count ||
          request.dft_frequency_count == 0 ||
          selected_frequency_index >= request.dft_frequency_count)
        throw std::out_of_range(
            "Meep CUDA mode-flux overlap extent is outside DFT storage");
      if (!std::isfinite(request.inverse_stored_weight_real) ||
          !std::isfinite(request.inverse_stored_weight_imag))
        throw std::invalid_argument(
            "Meep CUDA eigenmode inverse stored weight must be finite");
    }
    const std::size_t operation_profile_count = checked_product(
        request.point_count,
        static_cast<std::size_t>(combined ? 3 : (mode_mode ? 2 : 1)),
        "eigenmode sampled profile count");
    if (profile_value_count >
        std::numeric_limits<std::size_t>::max() -
            operation_profile_count)
      throw std::overflow_error(
          "Meep CUDA eigenmode sampled profile total overflow");
    profile_value_count += operation_profile_count;
    const std::size_t blocks_per_contraction =
        request.point_count /
            meep_cuda::eigenmode_overlap_threads_per_block +
        (request.point_count %
             meep_cuda::eigenmode_overlap_threads_per_block !=
         0);
    const std::size_t operation_blocks = checked_product(
        blocks_per_contraction, logical_contraction_count,
        "eigenmode fused overlap block count");
    if (total_block_count >
        std::numeric_limits<std::size_t>::max() - operation_blocks)
      throw std::overflow_error(
          "Meep CUDA eigenmode overlap block count overflow");
    total_block_count += operation_blocks;
    const std::size_t logical_terms = checked_product(
        request.point_count, logical_contraction_count,
        "eigenmode logical overlap term count");
    if (logical_terms >
        std::numeric_limits<std::uint64_t>::max() - total_terms)
      throw std::overflow_error(
          "Meep CUDA eigenmode overlap term counter overflow");
    total_terms += static_cast<std::uint64_t>(logical_terms);
    if (std::find(owners.begin(), owners.end(), request.owner) ==
        owners.end())
      owners.push_back(request.owner);
  }
  packed_profiles.reserve(profile_value_count);
  for (std::size_t index = 0; index < request_count; ++index) {
    const eigenmode_overlap_request_fp32 &request = requests[index];
    mode1_offsets.push_back(packed_profiles.size());
    for (std::size_t point = 0; point < request.point_count; ++point) {
      const std::complex<double> value =
          request.weighted_conjugate_mode[point];
      if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        throw std::runtime_error(
            "Meep CUDA eigenmode mode-1 profile is non-finite");
      packed_profiles.push_back({value.real(), value.imag()});
    }
    mode2_offsets.push_back(packed_profiles.size());
    if (mode_mode)
      for (std::size_t point = 0; point < request.point_count; ++point) {
        const std::complex<double> value = request.mode2_real_imag[point];
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
          throw std::runtime_error(
              "Meep CUDA eigenmode mode-2 profile is non-finite");
        packed_profiles.push_back({value.real(), value.imag()});
      }
    mode_mode1_offsets.push_back(packed_profiles.size());
    if (combined)
      for (std::size_t point = 0; point < request.point_count; ++point) {
        const std::complex<double> value =
            request.mode_mode_weighted_conjugate_mode[point];
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
          throw std::runtime_error(
              "Meep CUDA eigenmode second mode-1 profile is non-finite");
        packed_profiles.push_back({value.real(), value.imag()});
      }
    zero_flag_offsets.push_back(packed_zero_flags.size());
    if (request.zero_normalization_divisors)
      packed_zero_flags.insert(
          packed_zero_flags.end(),
          request.zero_normalization_divisors,
          request.zero_normalization_divisors + request.point_count);
  }
  if (packed_profiles.size() != profile_value_count ||
      total_block_count == 0)
    throw std::logic_error(
        "Meep CUDA eigenmode profile/block planning changed");
  const std::size_t profile_bytes = checked_bytes(
      profile_value_count,
      sizeof(meep_cuda::complex_value_fp64),
      "eigenmode sampled profiles");
  const std::size_t zero_flag_bytes = checked_bytes(
      packed_zero_flags.size(), sizeof(std::uint8_t),
      "eigenmode zero-normalization flags");
  const std::size_t dynamic_profile_bytes = checked_phase_sum(
      profile_bytes, zero_flag_bytes,
      "eigenmode sampled profiles and normalization flags");

  std::vector<resident_curl_session> sessions;
  sessions.reserve(owners.size());
  for (const void *owner : owners)
    sessions.emplace_back(owner, true);
  if (sessions.empty() || !sessions[0].active() ||
      !sessions[0].cache())
    throw std::logic_error(
        "Meep CUDA eigenmode overlap has no active resident cache");
  const int device_ordinal = sessions[0].cache()->device_ordinal;
  for (const resident_curl_session &session : sessions)
    if (!session.active() || !session.cache() ||
        session.cache()->device_ordinal != device_ordinal)
      throw std::logic_error(
          "Meep CUDA eigenmode overlap owners must use one device");
  meep_cuda::select_device(device_ordinal);

  const auto cache_for_owner =
      [&owners, &sessions](const void *owner) -> resident_cache * {
    const auto found = std::find(owners.begin(), owners.end(), owner);
    if (found == owners.end())
      throw std::logic_error(
          "Meep CUDA eigenmode owner disappeared during preparation");
    return sessions[static_cast<std::size_t>(found - owners.begin())]
        .cache();
  };

  struct prepared_dft_mirror {
    resident_cache *cache = nullptr;
    const float *host = nullptr;
    std::size_t bytes = 0;
    const float *device = nullptr;
  };
  std::vector<prepared_dft_mirror> prepared_mirrors;
  prepared_mirrors.reserve(mode_flux ? request_count : 0);
  const auto prepare_dft_mirror =
      [&prepared_mirrors](
          resident_cache *cache, const float *host,
          std::size_t bytes) -> const float * {
    for (const prepared_dft_mirror &prepared : prepared_mirrors)
      if (prepared.cache == cache && prepared.host == host &&
          prepared.bytes == bytes)
        return prepared.device;
    const resolved_resident_range existing = resolve_resident_range(
        cache, host, bytes, true, "Meep CUDA eigenmode DFT monitor");
    if (existing.mirror &&
        (existing.host_base != host || existing.byte_offset != 0 ||
         existing.mirror->bytes != bytes))
      throw std::invalid_argument(
          "Meep CUDA eigenmode DFT monitor must match one complete mirror");
    const bool was_device_dirty =
        existing.mirror && existing.mirror->device_dirty;
    const float *device = static_cast<const float *>(
        ensure_resident_mirror(cache, host, bytes));
    prepared_mirrors.push_back({cache, host, bytes, device});
    if (was_device_dirty)
      eigenmode_full_dft_d2h_bytes_avoided.fetch_add(
          static_cast<std::uint64_t>(bytes),
          std::memory_order_relaxed);
    return device;
  };

  std::vector<eigenmode_overlap_request_snapshot> snapshots;
  snapshots.reserve(request_count * logical_contraction_count);
  std::vector<resident_cache *> cache_dependencies;
  cache_dependencies.reserve(owners.size());
  for (std::size_t index = 0; index < request_count; ++index) {
    const eigenmode_overlap_request_fp32 &request = requests[index];
    resident_cache *cache = cache_for_owner(request.owner);
    const float *dft_device = nullptr;
    if (mode_flux) {
      const std::size_t dft_values = checked_product(
          checked_product(request.dft_storage_point_count,
                          request.dft_frequency_count,
                          "eigenmode DFT point-frequency count"),
          static_cast<std::size_t>(2),
          "eigenmode DFT scalar count");
      const std::size_t dft_bytes = checked_bytes(
          dft_values, sizeof(float), "eigenmode DFT values");
      dft_device = prepare_dft_mirror(
          cache, request.dft_real_imag, dft_bytes);
    }
    const auto append_snapshot =
        [&](bool snapshot_mode_mode, std::size_t mode1_offset,
            std::uint32_t output_index) {
      eigenmode_overlap_request_snapshot snapshot;
      snapshot.cache = cache;
      snapshot.dft_host =
          snapshot_mode_mode ? nullptr : request.dft_real_imag;
      snapshot.dft_device = snapshot_mode_mode ? nullptr : dft_device;
      snapshot.point_count = request.point_count;
      snapshot.dft_storage_point_count =
          snapshot_mode_mode ? 0 : request.dft_storage_point_count;
      snapshot.dft_frequency_count =
          snapshot_mode_mode ? 0 : request.dft_frequency_count;
      snapshot.mode1_offset = mode1_offset;
      snapshot.mode2_offset = mode2_offsets[index];
      snapshot.zero_flag_offset = zero_flag_offsets[index];
      snapshot.has_zero_flags =
          !snapshot_mode_mode &&
          request.zero_normalization_divisors != nullptr;
      snapshot.output_index = output_index;
      snapshot.inverse_weight_real =
          snapshot_mode_mode ? 0.0
                             : request.inverse_stored_weight_real;
      snapshot.inverse_weight_imaginary =
          snapshot_mode_mode ? 0.0
                             : request.inverse_stored_weight_imag;
      snapshot.mode_mode = snapshot_mode_mode;
      snapshots.push_back(snapshot);
    };
    if (mode_flux)
      append_snapshot(false, mode1_offsets[index], request.output_index);
    if (mode_mode)
      append_snapshot(
          true,
          combined ? mode_mode1_offsets[index] : mode1_offsets[index],
          request.output_index + (combined ? 4u : 0u));
    if (std::find(cache_dependencies.begin(), cache_dependencies.end(),
                  cache) == cache_dependencies.end())
      cache_dependencies.push_back(cache);
  }
  for (eigenmode_overlap_request_snapshot &snapshot : snapshots)
    snapshot.cache_generation = snapshot.cache->allocation_generation;

  const std::size_t descriptor_bytes = checked_phase_product(
      snapshots.size(),
      sizeof(meep_cuda::eigenmode_overlap_operation_fp32),
      "eigenmode overlap descriptors");
  const std::size_t block_map_bytes = checked_phase_product(
      total_block_count, sizeof(std::uint32_t),
      "eigenmode overlap block map");
  const std::size_t static_bytes = checked_phase_sum(
      descriptor_bytes, block_map_bytes,
      "eigenmode overlap static plan");
  const std::size_t partial_capacity = std::min(
      total_block_count,
      meep_cuda::eigenmode_overlap_partial_capacity);
  const std::size_t partial_value_count = checked_product(
      checked_product(
          static_cast<std::size_t>(
              2 * output_count),
          partial_capacity,
          "eigenmode overlap partial count"),
      sizeof(double), "eigenmode overlap partial bytes");
  const std::size_t partial_bytes = partial_value_count;

  eigenmode_overlap_plan_registry &registry =
      get_eigenmode_overlap_plan_registry();
  std::unique_lock<std::mutex> plan_lock(registry.mutex);
  eigenmode_overlap_plan_key key;
  key.owner = plan_owner;
  key.lane = static_cast<unsigned int>(batch_kind);
  eigenmode_overlap_retained_plan &plan = registry.plans[key];
  const bool plan_matches = same_eigenmode_overlap_plan(
      plan, snapshots, device_ordinal, profile_value_count,
      packed_zero_flags.size(),
      output_count,
      total_block_count);
  const auto make_operations =
      [&](meep_cuda::complex_value_fp64 *profile_storage) {
    std::vector<meep_cuda::eigenmode_overlap_operation_fp32> operations;
    operations.reserve(snapshots.size());
    std::size_t block_start = 0;
    for (std::size_t index = 0; index < snapshots.size(); ++index) {
      const eigenmode_overlap_request_snapshot &snapshot = snapshots[index];
      operations.push_back(
          {snapshot.dft_device,
           profile_storage + snapshot.mode1_offset,
           snapshot.mode_mode
               ? profile_storage + snapshot.mode2_offset
               : nullptr,
           snapshot.has_zero_flags
               ? reinterpret_cast<const std::uint8_t *>(profile_storage) +
                     profile_bytes + snapshot.zero_flag_offset
               : nullptr,
           snapshot.point_count, snapshot.dft_frequency_count,
           block_start, snapshot.output_index,
           {snapshot.inverse_weight_real,
            snapshot.inverse_weight_imaginary}});
      block_start +=
          snapshot.point_count /
              meep_cuda::eigenmode_overlap_threads_per_block +
          (snapshot.point_count %
               meep_cuda::eigenmode_overlap_threads_per_block !=
           0);
    }
    if (block_start != total_block_count)
      throw std::logic_error(
          "Meep CUDA eigenmode overlap topology changed");
    return operations;
  };

  if (!plan_matches) {
    eigenmode_overlap_retained_plan replacement;
    std::uint64_t replacement_allocation_count = 0;
    replacement.device_ordinal = device_ordinal;
    replacement.profile_value_count = profile_value_count;
    replacement.zero_flag_count = packed_zero_flags.size();
    replacement.output_count = output_count;
    replacement.partial_capacity = partial_capacity;
    replacement.total_block_count = total_block_count;
    replacement.requests = snapshots;
    replacement.cache_dependencies = cache_dependencies;
    try {
      replacement.static_storage =
          meep_cuda::allocate_device_bytes(static_bytes);
      device_buffer_allocations.fetch_add(1, std::memory_order_relaxed);
      live_device_buffer_count().fetch_add(1, std::memory_order_relaxed);
      ++replacement_allocation_count;
      replacement.profile_storage =
          static_cast<meep_cuda::complex_value_fp64 *>(
              meep_cuda::allocate_device_bytes(dynamic_profile_bytes));
      device_buffer_allocations.fetch_add(1, std::memory_order_relaxed);
      live_device_buffer_count().fetch_add(1, std::memory_order_relaxed);
      ++replacement_allocation_count;
      replacement.partial_workspace = static_cast<double *>(
          meep_cuda::allocate_device_bytes(partial_bytes));
      device_buffer_allocations.fetch_add(1, std::memory_order_relaxed);
      live_device_buffer_count().fetch_add(1, std::memory_order_relaxed);
      ++replacement_allocation_count;
      replacement.device_result = static_cast<double *>(
          meep_cuda::allocate_device_bytes(result_bytes));
      device_buffer_allocations.fetch_add(1, std::memory_order_relaxed);
      live_device_buffer_count().fetch_add(1, std::memory_order_relaxed);
      ++replacement_allocation_count;
      replacement.device_operations = static_cast<
          const meep_cuda::eigenmode_overlap_operation_fp32 *>(
              replacement.static_storage);
      replacement.device_block_map =
          reinterpret_cast<const std::uint32_t *>(
              static_cast<const unsigned char *>(
                  replacement.static_storage) + descriptor_bytes);
      const std::vector<meep_cuda::eigenmode_overlap_operation_fp32>
          operations = make_operations(replacement.profile_storage);
      const std::vector<std::uint32_t> block_map =
          make_phase_block_operation_indices(
              operations, total_block_count);
      meep_cuda::validate_eigenmode_overlap_operations_fp32(
          operations.data(), block_map.data(), operations.size(),
          block_map.size(), selected_frequency_index, output_count);
      meep_cuda::copy_to_device(
          replacement.static_storage, operations.data(),
          descriptor_bytes);
      host_to_device_bytes.fetch_add(
          static_cast<std::uint64_t>(descriptor_bytes),
          std::memory_order_relaxed);
      meep_cuda::copy_to_device(
          static_cast<unsigned char *>(replacement.static_storage) +
              descriptor_bytes,
          block_map.data(), block_map_bytes);
      host_to_device_bytes.fetch_add(
          static_cast<std::uint64_t>(block_map_bytes),
          std::memory_order_relaxed);
      cuda_eigenmode_descriptor_uploads.fetch_add(
          1, std::memory_order_relaxed);
      meep_cuda::copy_to_device(
          replacement.profile_storage, packed_profiles.data(),
          profile_bytes);
      host_to_device_bytes.fetch_add(
          static_cast<std::uint64_t>(profile_bytes),
          std::memory_order_relaxed);
      host_mode_profile_h2d_bytes.fetch_add(
          static_cast<std::uint64_t>(profile_bytes),
          std::memory_order_relaxed);
      if (zero_flag_bytes) {
        meep_cuda::copy_to_device(
            reinterpret_cast<unsigned char *>(
                replacement.profile_storage) + profile_bytes,
            packed_zero_flags.data(), zero_flag_bytes);
        host_to_device_bytes.fetch_add(
            static_cast<std::uint64_t>(zero_flag_bytes),
            std::memory_order_relaxed);
        host_mode_profile_h2d_bytes.fetch_add(
            static_cast<std::uint64_t>(zero_flag_bytes),
            std::memory_order_relaxed);
      }
      if (inject_eigenmode_failure_for_testing("profile_h2d"))
        throw std::runtime_error(
            "injected CUDA eigenmode profile H2D failure");
    }
    catch (...) {
      meep_cuda::free_device(replacement.static_storage);
      meep_cuda::free_device(replacement.profile_storage);
      meep_cuda::free_device(replacement.partial_workspace);
      meep_cuda::free_device(replacement.device_result);
      replacement.static_storage = nullptr;
      replacement.profile_storage = nullptr;
      replacement.partial_workspace = nullptr;
      replacement.device_result = nullptr;
      if (replacement_allocation_count)
        live_device_buffer_count().fetch_sub(
            replacement_allocation_count,
            std::memory_order_relaxed);
      throw;
    }
    release_eigenmode_overlap_plan(plan);
    plan = std::move(replacement);
    meep_cuda::select_device(device_ordinal);
  }
  else {
    const std::vector<meep_cuda::eigenmode_overlap_operation_fp32>
        operations = make_operations(plan.profile_storage);
    const std::vector<std::uint32_t> block_map =
        make_phase_block_operation_indices(operations, total_block_count);
    meep_cuda::validate_eigenmode_overlap_operations_fp32(
        operations.data(), block_map.data(), operations.size(),
        block_map.size(), selected_frequency_index, output_count);
    device_buffer_reuses.fetch_add(4, std::memory_order_relaxed);
    cuda_eigenmode_plan_reuses.fetch_add(
        1, std::memory_order_relaxed);
    meep_cuda::copy_to_device(
        plan.profile_storage, packed_profiles.data(), profile_bytes);
    host_to_device_bytes.fetch_add(
        static_cast<std::uint64_t>(profile_bytes),
        std::memory_order_relaxed);
    host_mode_profile_h2d_bytes.fetch_add(
        static_cast<std::uint64_t>(profile_bytes),
        std::memory_order_relaxed);
    if (zero_flag_bytes) {
      meep_cuda::copy_to_device(
          reinterpret_cast<unsigned char *>(plan.profile_storage) +
              profile_bytes,
          packed_zero_flags.data(), zero_flag_bytes);
      host_to_device_bytes.fetch_add(
          static_cast<std::uint64_t>(zero_flag_bytes),
          std::memory_order_relaxed);
      host_mode_profile_h2d_bytes.fetch_add(
          static_cast<std::uint64_t>(zero_flag_bytes),
          std::memory_order_relaxed);
    }
    if (inject_eigenmode_failure_for_testing("profile_h2d"))
      throw std::runtime_error(
          "injected CUDA eigenmode profile H2D failure");
  }

  if (inject_eigenmode_failure_for_testing("kernel"))
    throw std::runtime_error(
        "injected CUDA eigenmode kernel failure");
  meep_cuda::eigenmode_overlap_reduce_fp32(
      plan.device_operations, plan.device_block_map, snapshots.size(),
      total_block_count, selected_frequency_index,
      plan.partial_workspace, plan.partial_capacity,
      output_count,
      plan.device_result);
  cuda_eigenmode_kernel_launches.fetch_add(
      2, std::memory_order_relaxed);
  std::vector<double> staged(result_scalar_count);
  meep_cuda::copy_to_host(
      staged.data(), plan.device_result, result_bytes);
  device_to_host_bytes.fetch_add(
      static_cast<std::uint64_t>(result_bytes),
      std::memory_order_relaxed);
  cuda_eigenmode_result_d2h_bytes.fetch_add(
      static_cast<std::uint64_t>(result_bytes),
      std::memory_order_relaxed);
  if (inject_eigenmode_failure_for_testing("d2h"))
    throw std::runtime_error(
        "injected CUDA eigenmode result D2H failure");
  std::copy(staged.begin(), staged.end(), result_real_imag);

  cuda_dft_overlap_calls.fetch_add(
      logical_contraction_count, std::memory_order_relaxed);
  cuda_dft_overlap_terms.fetch_add(
      total_terms, std::memory_order_relaxed);
  if (mode_flux)
    cuda_eigenmode_mode_flux_calls.fetch_add(
        1, std::memory_order_relaxed);
  if (mode_mode)
    cuda_eigenmode_mode_mode_calls.fetch_add(
        1, std::memory_order_relaxed);
  cuda_eigenmode_submitted_pairs.fetch_add(
      static_cast<std::uint64_t>(request_count) *
          static_cast<std::uint64_t>(logical_contraction_count),
      std::memory_order_relaxed);
  for (resident_curl_session &session : sessions)
    session.finish(false);
#else
  (void)plan_owner;
  (void)batch_kind;
  (void)requests;
  (void)request_count;
  (void)selected_frequency_index;
  (void)result_real_imag;
  throw std::runtime_error(
      "Meep CUDA eigenmode overlap called in a CPU-only build");
#endif
}

void resident_reduce_ldos_fp32(
    const ldos_reduction_request_fp32 *requests,
    std::size_t request_count, double result[4]) {
#if MEEP_HAVE_CUDA
  if (!result)
    throw std::invalid_argument(
        "Meep CUDA LDOS reduction result must be non-null");
  if (!requests && request_count)
    throw std::invalid_argument(
        "Meep CUDA LDOS reduction requests must be non-null");
  if (request_count == 0) {
    std::fill(result, result + 4, 0.0);
    return;
  }

  std::unique_lock<std::mutex> ldos_lock(
      get_ldos_reduction_mutex(), std::try_to_lock);
  if (!ldos_lock.owns_lock())
    throw std::logic_error(
        "a Meep CUDA LDOS reduction or profile replacement is already active");
  constexpr std::size_t threads = 256;
  constexpr std::size_t partial_capacity =
      meep_cuda::indexed_ldos_reduction_partial_capacity;
  constexpr std::size_t channel_count =
      meep_cuda::indexed_ldos_channel_count;
  static_assert(
      sizeof(meep_cuda::complex_value_fp32) == 2 * sizeof(float),
      "CUDA LDOS complex amplitude must remain two interleaved FP32 values");
  static_assert(
      alignof(meep_cuda::complex_value_fp32) == alignof(float),
      "CUDA LDOS complex amplitude alignment must match its FP32 profile");
  static_assert(
      channel_count == 4,
      "Meep LDOS public result has exactly four channels");

  resident_cache *const result_cache = requests[0].cache;
  if (!result_cache || !result_cache->phase_active ||
      result_cache->device_ordinal < 0)
    throw std::logic_error(
        "Meep CUDA LDOS reduction requires active resident caches");
  const int device_ordinal = result_cache->device_ordinal;

  // Validate the complete batch before preparing any mirror. A malformed
  // later request must not allocate or upload state for an earlier request.
  std::set<resident_cache *, std::less<resident_cache *> > cache_set;
  std::size_t total_block_count = 0;
  std::uint64_t total_source_points = 0;
  for (std::size_t request_index = 0;
       request_index < request_count; ++request_index) {
    const ldos_reduction_request_fp32 &request =
        requests[request_index];
    if (!request.cache || !request.cache->phase_active ||
        request.cache->device_ordinal != device_ordinal)
      throw std::logic_error(
          "Meep CUDA LDOS reduction caches must be active on one device");
    if (!request.field_real || !request.indices ||
        !request.amplitudes_real_imag || request.field_array_count == 0 ||
        request.source_count == 0)
      throw std::invalid_argument(
          "Meep CUDA LDOS reduction request is incomplete");
    if (request.field_array_count >
        static_cast<std::size_t>(
            std::numeric_limits<std::ptrdiff_t>::max()))
      throw std::overflow_error(
          "Meep CUDA LDOS field array is too large");
    if (request.source_count >
        std::numeric_limits<std::size_t>::max() / 2)
      throw std::overflow_error(
          "Meep CUDA LDOS amplitude count overflow");

    (void)checked_bytes(
        request.field_array_count, sizeof(float), "LDOS field mirror");
    (void)checked_bytes(
        request.source_count, sizeof(std::ptrdiff_t),
        "LDOS source indices");
    (void)checked_bytes(
        2 * request.source_count, sizeof(float),
        "LDOS source amplitudes");

    const auto validated =
        request.cache->validated_source_profiles.find(request.indices);
    if (validated ==
            request.cache->validated_source_profiles.end() ||
        validated->second.first != request.source_count ||
        validated->second.second != request.field_array_count)
      for (std::size_t point = 0; point < request.source_count; ++point)
        if (request.indices[point] < 0 ||
            static_cast<std::size_t>(request.indices[point]) >=
                request.field_array_count)
          throw std::out_of_range(
              "Meep CUDA LDOS source index is outside the field array");

    const std::size_t operation_blocks =
        request.source_count / threads +
        (request.source_count % threads != 0);
    if (total_block_count >
        std::numeric_limits<std::size_t>::max() - operation_blocks)
      throw std::overflow_error(
          "Meep CUDA LDOS block count overflow");
    total_block_count += operation_blocks;
    if (request.source_count >
        std::numeric_limits<std::uint64_t>::max() -
            total_source_points)
      throw std::overflow_error(
          "Meep CUDA LDOS source point counter overflow");
    total_source_points +=
        static_cast<std::uint64_t>(request.source_count);
    cache_set.insert(request.cache);
  }

  std::uint64_t full_field_bytes_avoided = 0;
  for (const resident_cache *cache : cache_set)
    for (const auto &entry : cache->mirrors)
      if (entry.second.device_dirty) {
        if (entry.second.bytes >
            std::numeric_limits<std::uint64_t>::max() -
                full_field_bytes_avoided)
          throw std::overflow_error(
              "Meep CUDA LDOS avoided-byte counter overflow");
        full_field_bytes_avoided +=
            static_cast<std::uint64_t>(entry.second.bytes);
      }

  // Commit bounds-validation cache entries only after every request has
  // passed. This is host metadata and precedes all CUDA-side preparation.
  for (std::size_t request_index = 0;
       request_index < request_count; ++request_index) {
    const ldos_reduction_request_fp32 &request =
        requests[request_index];
    request.cache->validated_source_profiles[request.indices] =
        std::make_pair(request.source_count,
                       request.field_array_count);
  }

  std::vector<meep_cuda::indexed_ldos_operation_fp32> operations;
  operations.reserve(request_count);
  std::size_t built_block_count = 0;
  for (std::size_t request_index = 0;
       request_index < request_count; ++request_index) {
    const ldos_reduction_request_fp32 &request =
        requests[request_index];
    const std::size_t field_bytes = checked_bytes(
        request.field_array_count, sizeof(float), "LDOS field mirror");
    const std::size_t index_bytes = checked_bytes(
        request.source_count, sizeof(std::ptrdiff_t),
        "LDOS source indices");
    const std::size_t amplitude_bytes = checked_bytes(
        2 * request.source_count, sizeof(float),
        "LDOS source amplitudes");
    const float *device_real = static_cast<const float *>(
        ensure_resident_mirror(
            request.cache, request.field_real, field_bytes));
    const float *device_imaginary =
        request.field_imaginary
            ? static_cast<const float *>(ensure_resident_mirror(
                  request.cache, request.field_imaginary, field_bytes))
            : nullptr;
    const std::ptrdiff_t *device_indices =
        static_cast<const std::ptrdiff_t *>(ensure_resident_mirror(
            request.cache,
            reinterpret_cast<const float *>(request.indices),
            index_bytes, nullptr, false));
    const meep_cuda::complex_value_fp32 *device_amplitudes =
        static_cast<const meep_cuda::complex_value_fp32 *>(
            ensure_resident_mirror(
                request.cache, request.amplitudes_real_imag,
                amplitude_bytes, nullptr, false));
    operations.push_back(
        {device_real, device_imaginary, device_indices,
         device_amplitudes, request.source_count, built_block_count,
         request.magnetic});
    built_block_count +=
        request.source_count / threads +
        (request.source_count % threads != 0);
  }
  if (built_block_count != total_block_count)
    throw std::logic_error(
        "Meep CUDA LDOS block planning changed during preparation");

  const std::vector<std::uint32_t> block_operation_indices =
      make_phase_block_operation_indices(operations, total_block_count);
  const std::size_t descriptor_bytes = checked_phase_product(
      operations.size(),
      sizeof(meep_cuda::indexed_ldos_operation_fp32),
      "LDOS operation descriptor");
  const std::size_t block_map_bytes = checked_phase_product(
      block_operation_indices.size(), sizeof(std::uint32_t),
      "LDOS block map");
  const std::size_t plan_bytes = checked_phase_sum(
      descriptor_bytes, block_map_bytes, "LDOS plan");
  const std::size_t workspace_value_count = checked_phase_sum(
      checked_phase_product(
          channel_count, partial_capacity, "LDOS reduction partial"),
      channel_count, "LDOS reduction workspace");
  const std::size_t workspace_bytes = checked_phase_product(
      workspace_value_count, sizeof(double), "LDOS reduction workspace");

  select_cache_device(result_cache);
  const auto same_operation = [](
      const meep_cuda::indexed_ldos_operation_fp32 &left,
      const meep_cuda::indexed_ldos_operation_fp32 &right) {
    return left.field_real == right.field_real &&
           left.field_imaginary == right.field_imaginary &&
           left.indices == right.indices &&
           left.amplitudes == right.amplitudes &&
           left.point_count == right.point_count &&
           left.block_start == right.block_start &&
           left.magnetic == right.magnetic;
  };
  const int active_plan_slot = result_cache->ldos_active_plan_slot;
  const bool active_plan_present =
      (active_plan_slot == 0 || active_plan_slot == 1) &&
      result_cache->ldos_plan_buffers[active_plan_slot].device_pointer;
  bool operations_match =
      result_cache->ldos_plan_snapshot_valid && active_plan_present &&
      result_cache->ldos_plan_descriptor_bytes == descriptor_bytes &&
      result_cache->ldos_plan_total_block_count == total_block_count &&
      result_cache->ldos_host_operations.size() == operations.size();
  if (operations_match)
    for (std::size_t index = 0; index < operations.size(); ++index)
      if (!same_operation(result_cache->ldos_host_operations[index],
                          operations[index])) {
        operations_match = false;
        break;
      }

  std::vector<meep_cuda::indexed_ldos_operation_fp32>
      replacement_snapshot;
  if (!operations_match) replacement_snapshot = operations;
  int replacement_plan_slot = -1;
  void *new_plan_allocation = nullptr;
  bool reused_plan_buffer = false;
  double *replacement_workspace = nullptr;
  bool replacement_workspace_counted_live = false;
  try {
    if (!operations_match) {
      // A valid or invalidated active marker protects that slot until a new
      // descriptor has committed.  With no marker, prefer an existing slot
      // that already fits, then an empty slot, then replace the smaller
      // high-water allocation.
      if (active_plan_slot == 0 || active_plan_slot == 1) {
        replacement_plan_slot = 1 - active_plan_slot;
      }
      else if (result_cache->ldos_plan_buffers[0].device_pointer &&
               result_cache->ldos_plan_buffers[0].capacity_bytes >=
                   plan_bytes) {
        replacement_plan_slot = 0;
      }
      else if (result_cache->ldos_plan_buffers[1].device_pointer &&
               result_cache->ldos_plan_buffers[1].capacity_bytes >=
                   plan_bytes) {
        replacement_plan_slot = 1;
      }
      else if (!result_cache->ldos_plan_buffers[0].device_pointer) {
        replacement_plan_slot = 0;
      }
      else if (!result_cache->ldos_plan_buffers[1].device_pointer) {
        replacement_plan_slot = 1;
      }
      else {
        replacement_plan_slot =
            result_cache->ldos_plan_buffers[0].capacity_bytes <=
                    result_cache->ldos_plan_buffers[1].capacity_bytes
                ? 0
                : 1;
      }

      resident_cache::ldos_plan_buffer &target =
          result_cache->ldos_plan_buffers[replacement_plan_slot];
      void *upload_destination = target.device_pointer;
      if (!upload_destination || target.capacity_bytes < plan_bytes) {
        new_plan_allocation =
            meep_cuda::allocate_device_bytes(plan_bytes);
        device_buffer_allocations.fetch_add(1);
        live_device_buffer_count().fetch_add(1);
        upload_destination = new_plan_allocation;
      }
      else {
        reused_plan_buffer = true;
      }
      meep_cuda::copy_to_device(
          upload_destination, operations.data(), descriptor_bytes);
      meep_cuda::copy_to_device(
          static_cast<unsigned char *>(upload_destination) +
              descriptor_bytes,
          block_operation_indices.data(), block_map_bytes);
      // This is physical traffic even if the deterministic test probe rejects
      // publication below. Descriptor-upload statistics remain semantic and
      // are incremented only after a complete reduction.
      host_to_device_bytes.fetch_add(
          static_cast<std::uint64_t>(plan_bytes));
    }
    if (!result_cache->ldos_reduction_workspace ||
        result_cache->ldos_reduction_partial_capacity < partial_capacity) {
      replacement_workspace = static_cast<double *>(
          meep_cuda::allocate_device_bytes(workspace_bytes));
      device_buffer_allocations.fetch_add(1);
      live_device_buffer_count().fetch_add(1);
      replacement_workspace_counted_live = true;
    }
    if (!operations_match &&
        fail_ldos_plan_commit_for_testing.exchange(
            false, std::memory_order_acq_rel))
      throw std::bad_alloc();
  }
  catch (...) {
    meep_cuda::free_device(new_plan_allocation);
    if (new_plan_allocation)
      live_device_buffer_count().fetch_sub(1);
    meep_cuda::free_device(replacement_workspace);
    if (replacement_workspace_counted_live)
      live_device_buffer_count().fetch_sub(1);
    throw;
  }

  const bool uploaded_plan = !operations_match;
  if (!operations_match) {
    resident_cache::ldos_plan_buffer &target =
        result_cache->ldos_plan_buffers[replacement_plan_slot];
    if (new_plan_allocation) {
      void *const previous = target.device_pointer;
      target.device_pointer = new_plan_allocation;
      target.capacity_bytes = plan_bytes;
      new_plan_allocation = nullptr;
      meep_cuda::free_device(previous);
      if (previous) live_device_buffer_count().fetch_sub(1);
    }
    result_cache->ldos_plan_descriptor_bytes = descriptor_bytes;
    result_cache->ldos_plan_total_block_count = total_block_count;
    result_cache->ldos_host_operations.swap(replacement_snapshot);
    result_cache->ldos_active_plan_slot = replacement_plan_slot;
    result_cache->ldos_plan_snapshot_valid = true;
    if (reused_plan_buffer) device_buffer_reuses.fetch_add(1);
  }
  else
    device_buffer_reuses.fetch_add(1);

  if (replacement_workspace) {
    double *const previous = result_cache->ldos_reduction_workspace;
    result_cache->ldos_reduction_workspace = replacement_workspace;
    result_cache->ldos_reduction_partial_capacity = partial_capacity;
    meep_cuda::free_device(previous);
    if (previous) live_device_buffer_count().fetch_sub(1);
  }
  else
    device_buffer_reuses.fetch_add(1);

  double *const device_partials =
      result_cache->ldos_reduction_workspace;
  double *const device_result =
      device_partials + channel_count * partial_capacity;
  const auto *const device_operations =
      static_cast<const meep_cuda::indexed_ldos_operation_fp32 *>(
          result_cache->ldos_plan_buffers[
              result_cache->ldos_active_plan_slot].device_pointer);
  const auto *const device_block_map =
      reinterpret_cast<const std::uint32_t *>(
          static_cast<const unsigned char *>(
              result_cache->ldos_plan_buffers[
                  result_cache->ldos_active_plan_slot].device_pointer) +
          result_cache->ldos_plan_descriptor_bytes);
  meep_cuda::indexed_ldos_reduce_fp32(
      device_operations, device_block_map, operations.size(),
      total_block_count, device_partials,
      result_cache->ldos_reduction_partial_capacity, device_result,
      meep_cuda::indexed_ldos_result_mode::replace);

  double staged[4] = {};
  meep_cuda::copy_to_host(staged, device_result, sizeof(staged));
  std::copy(staged, staged + channel_count, result);

  constexpr std::uint64_t result_bytes = channel_count * sizeof(double);
  device_to_host_bytes.fetch_add(result_bytes);
  ldos_batch_calls.fetch_add(1, std::memory_order_relaxed);
  ldos_submitted_profiles.fetch_add(
      static_cast<std::uint64_t>(request_count),
      std::memory_order_relaxed);
  ldos_source_points.fetch_add(
      total_source_points, std::memory_order_relaxed);
  if (uploaded_plan)
    ldos_descriptor_uploads.fetch_add(1, std::memory_order_relaxed);
  ldos_kernel_launches.fetch_add(2, std::memory_order_relaxed);
  ldos_result_device_to_host_bytes.fetch_add(
      result_bytes, std::memory_order_relaxed);
  ldos_full_field_device_to_host_bytes_avoided.fetch_add(
      full_field_bytes_avoided, std::memory_order_relaxed);
#else
  (void)requests;
  (void)request_count;
  (void)result;
  throw std::runtime_error(
      "Meep CUDA LDOS reduction called in a CPU-only build");
#endif
}

void resident_near2far_cartesian_fp32(
    near2far_cartesian_dimension dimension,
    const void *plan_owner, const near2far_request_fp32 *requests,
    std::size_t request_count,
    const near2far_point_fp64 *targets, std::size_t target_count,
    const double *frequencies, std::size_t frequency_count,
    const near2far_periodic_copy_fp64 *periodic_copies,
    std::size_t periodic_copy_count, double eps, double mu,
    std::complex<double> *output,
    double *fast_error_bounds, bool defer_cancellation_retry,
    bool force_mixed_for_cancellation, bool collective_retry_attempt,
    bool *used_mixed_precision, double azimuthal_mode,
    double greencyl_tolerance) {
#if MEEP_HAVE_CUDA
  if (dimension != near2far_cartesian_dimension::two &&
      dimension != near2far_cartesian_dimension::three &&
      dimension != near2far_cartesian_dimension::cylindrical)
    throw std::invalid_argument(
        "Meep CUDA near-to-far dimension must be 2D, 3D, or cylindrical");
  const bool is_2d = dimension == near2far_cartesian_dimension::two;
  const bool is_cylindrical =
      dimension == near2far_cartesian_dimension::cylindrical;
  if (!plan_owner || !requests || !targets || !frequencies ||
      !periodic_copies || !output)
    throw std::invalid_argument(
        "Meep CUDA near-to-far inputs and output must be non-null");
  if (request_count == 0 || target_count == 0 || frequency_count == 0 ||
      periodic_copy_count == 0)
    throw std::invalid_argument(
        "Meep CUDA near-to-far counts must be nonzero");
  if (!std::isfinite(eps) || !std::isfinite(mu) || eps <= 0.0 || mu <= 0.0)
    throw std::invalid_argument(
        "Meep CUDA near-to-far epsilon and mu must be finite and positive");
  if (!std::isfinite(azimuthal_mode) ||
      (is_cylindrical
           ? (!std::isfinite(greencyl_tolerance) ||
              greencyl_tolerance <= 0.0 ||
              std::abs(azimuthal_mode) > 16380.0)
           : (azimuthal_mode != 0.0 || greencyl_tolerance != 0.0)))
    throw std::invalid_argument(
        "Meep CUDA cylindrical Near2Far mode/tolerance is invalid for the "
        "selected dimension");
  const auto to_fp32 = [](double value, const char *label) {
    const float converted = static_cast<float>(value);
    if (!std::isfinite(converted))
      throw std::invalid_argument(std::string("Meep CUDA near-to-far ") +
                                  label + " is outside the FP32 range");
    return converted;
  };
  const std::size_t work_count = checked_product(
      target_count, frequency_count, "near-to-far work");
  const std::size_t output_scalar_count = checked_product(
      work_count, static_cast<std::size_t>(12),
      "near-to-far output scalar");
  (void)checked_bytes(
      output_scalar_count, sizeof(double), "near-to-far output");
  std::vector<std::complex<double> > staged_output(
      checked_product(work_count, static_cast<std::size_t>(6),
                      "near-to-far complex output"));
  std::vector<double> staged_fast_error_bounds(work_count, 0.0);
  if (fast_error_bounds)
    std::fill(fast_error_bounds, fast_error_bounds + work_count, 0.0);
  for (std::size_t target = 0; target < target_count; ++target)
    if (!std::isfinite(targets[target].x) ||
        !std::isfinite(targets[target].y) ||
        !std::isfinite(targets[target].z) ||
        (is_2d && targets[target].z != 0.0))
      throw std::invalid_argument(
          "Meep CUDA near-to-far targets must be finite and dimensionally "
          "valid");
  for (std::size_t frequency = 0; frequency < frequency_count; ++frequency)
    if (!std::isfinite(frequencies[frequency]) ||
        (is_2d ? frequencies[frequency] <= 0.0
               : frequencies[frequency] == 0.0))
      throw std::invalid_argument(
          "Meep CUDA near-to-far frequencies must be finite and valid for "
          "the selected Green function");
  for (std::size_t copy = 0; copy < periodic_copy_count; ++copy)
    if (!std::isfinite(periodic_copies[copy].displacement.x) ||
        !std::isfinite(periodic_copies[copy].displacement.y) ||
        !std::isfinite(periodic_copies[copy].displacement.z) ||
        !std::isfinite(periodic_copies[copy].phase_real) ||
        !std::isfinite(periodic_copies[copy].phase_imaginary) ||
        (is_2d && periodic_copies[copy].displacement.z != 0.0) ||
        (is_cylindrical && periodic_copies[copy].displacement.y != 0.0))
      throw std::invalid_argument(
          "Meep CUDA near-to-far periodic copies must be finite and "
          "dimensionally valid");

  std::vector<const void *> owners;
  owners.reserve(request_count);
  std::size_t source_point_count = 0;
  std::size_t maximum_interactions = 0;
  for (std::size_t request_index = 0; request_index < request_count;
       ++request_index) {
    const near2far_request_fp32 &request = requests[request_index];
    if (!request.owner || !request.dft_real_imag ||
        !request.source_points || request.point_count == 0 ||
        request.direction < 0 || request.direction > 2)
      throw std::invalid_argument(
          "Meep CUDA near-to-far request is incomplete");
    const std::size_t complex_count = checked_product(
        request.point_count, frequency_count,
        "near-to-far DFT complex");
    (void)checked_bytes(
        checked_product(complex_count, static_cast<std::size_t>(2),
                        "near-to-far DFT scalar"),
        sizeof(float), "near-to-far DFT");
    (void)checked_bytes(request.point_count, sizeof(near2far_point_fp64),
                        "near-to-far source coordinates");
    const std::size_t interactions = checked_product(
        request.point_count, periodic_copy_count,
        "near-to-far source-copy interactions");
    maximum_interactions = std::max(maximum_interactions, interactions);
    if (source_point_count >
        std::numeric_limits<std::size_t>::max() - request.point_count)
      throw std::overflow_error(
          "Meep CUDA near-to-far source-point count overflow");
    source_point_count += request.point_count;
    for (std::size_t point = 0; point < request.point_count; ++point)
      if (!std::isfinite(request.source_points[point].x) ||
          !std::isfinite(request.source_points[point].y) ||
          !std::isfinite(request.source_points[point].z) ||
          (is_2d && request.source_points[point].z != 0.0) ||
          (is_cylindrical &&
           (request.source_points[point].x < 0.0 ||
            request.source_points[point].y != 0.0)))
        throw std::invalid_argument(
            "Meep CUDA near-to-far source coordinates must be finite and "
            "dimensionally valid");
    if (std::find(owners.begin(), owners.end(), request.owner) == owners.end())
      owners.push_back(request.owner);
  }

  // FP32 Green arithmetic is the high-throughput path. Select the CUDA
  // mixed-precision companion whenever a conservative bound says that
  // coordinate subtraction, direction normalization, or phase reduction can
  // no longer preserve the requested transform. This decision is made once
  // per public transform, never by falling back to the CPU.
  const float eps_fp32 = static_cast<float>(eps);
  const float mu_fp32 = static_cast<float>(mu);
  bool fp32_representable =
      std::isfinite(eps_fp32) && std::isfinite(mu_fp32) &&
      eps_fp32 > 0.0f && mu_fp32 > 0.0f;
  const double fp64_infinity =
      std::numeric_limits<double>::infinity();
  double target_min[3] =
      {fp64_infinity, fp64_infinity, fp64_infinity};
  double target_max[3] =
      {-fp64_infinity, -fp64_infinity, -fp64_infinity};
  double source_min[3] =
      {fp64_infinity, fp64_infinity, fp64_infinity};
  double source_max[3] =
      {-fp64_infinity, -fp64_infinity, -fp64_infinity};
  struct source_surface_bounds {
    double lower[3];
    double upper[3];
    std::size_t request_index;
    std::size_t copy_index;
  };
  std::vector<source_surface_bounds> source_surfaces;
  source_surfaces.reserve(checked_product(
      request_count, periodic_copy_count,
      "near-to-far source surface bound"));
  double maximum_input_coordinate = 1.0;
  const auto update_bounds = [](const double value[3], double lower[3],
                                double upper[3]) {
    for (int axis = 0; axis < 3; ++axis) {
      lower[axis] = std::min(lower[axis], value[axis]);
      upper[axis] = std::max(upper[axis], value[axis]);
    }
  };
  for (std::size_t target = 0; target < target_count; ++target) {
    const double cartesian_value[3] =
        {targets[target].x, targets[target].y, targets[target].z};
    const double selector_value[3] = {
        is_cylindrical
            ? std::hypot(cartesian_value[0], cartesian_value[1])
            : cartesian_value[0],
        is_cylindrical ? 0.0 : cartesian_value[1], cartesian_value[2]};
    if (!std::isfinite(selector_value[0]))
      throw std::invalid_argument(
          "Meep CUDA near-to-far target radial coordinate is invalid");
    update_bounds(selector_value, target_min, target_max);
    for (double coordinate : cartesian_value)
      {
        fp32_representable =
            fp32_representable &&
            std::isfinite(static_cast<float>(coordinate));
        maximum_input_coordinate =
            std::max(maximum_input_coordinate, std::abs(coordinate));
      }
    maximum_input_coordinate =
        std::max(maximum_input_coordinate, selector_value[0]);
  }
  for (std::size_t request_index = 0; request_index < request_count;
       ++request_index)
    for (std::size_t copy = 0; copy < periodic_copy_count; ++copy) {
      source_surface_bounds surface = {
          {fp64_infinity, fp64_infinity, fp64_infinity},
          {-fp64_infinity, -fp64_infinity, -fp64_infinity},
          request_index, copy};
      for (std::size_t point = 0;
           point < requests[request_index].point_count; ++point) {
        const near2far_point_fp64 &source =
            requests[request_index].source_points[point];
        const near2far_periodic_copy_fp64 &periodic =
            periodic_copies[copy];
        const double shifted[3] =
            {source.x + periodic.displacement.x,
             source.y + periodic.displacement.y,
             source.z + periodic.displacement.z};
        for (double coordinate : shifted)
          if (!std::isfinite(coordinate))
            throw std::invalid_argument(
                "Meep CUDA near-to-far shifted source is non-finite");
        // greencyl deliberately accepts a signed source radius.  A periodic
        // displacement can therefore move an otherwise valid monitor sample
        // through R=0.  Keep that signed coordinate in the runtime
        // descriptor (it changes the azimuthal phase), but use |R| for the
        // selector's radial distance envelope: the Cartesian source ring has
        // radius |R| regardless of that coordinate representation.
        const double bounded_shifted[3] = {
            is_cylindrical ? std::abs(shifted[0]) : shifted[0],
            is_cylindrical ? 0.0 : shifted[1], shifted[2]};
        update_bounds(bounded_shifted, source_min, source_max);
        update_bounds(bounded_shifted, surface.lower, surface.upper);
        const double source_values[3] = {source.x, source.y, source.z};
        const double displacement_values[3] =
            {periodic.displacement.x, periodic.displacement.y,
             periodic.displacement.z};
        for (int axis = 0; axis < 3; ++axis)
          {
            fp32_representable =
                fp32_representable &&
                std::isfinite(static_cast<float>(source_values[axis])) &&
                std::isfinite(static_cast<float>(displacement_values[axis]));
            maximum_input_coordinate = std::max(
                maximum_input_coordinate,
                std::max(std::abs(source_values[axis]),
                         std::max(std::abs(displacement_values[axis]),
                                  std::abs(shifted[axis]))));
          }
      }
      source_surfaces.push_back(surface);
    }
  for (std::size_t copy = 0; copy < periodic_copy_count; ++copy)
    fp32_representable =
        fp32_representable &&
        std::isfinite(static_cast<float>(periodic_copies[copy].phase_real)) &&
        std::isfinite(
            static_cast<float>(periodic_copies[copy].phase_imaginary));

  // The fast kernel consumes several derived material quantities rather than
  // eps and mu in isolation.  Two individually representable inputs can still
  // overflow their ratio (impedance), underflow their product (index/k), or
  // lose a reciprocal used by the electric/magnetic branches.  Compare the
  // values produced by the exact FP32 expression order against the FP64
  // reference before allowing that kernel.
  constexpr double maximum_derived_relative_error = 2.0e-5;
  const auto positive_fp32_derived_value_is_safe = [](
      double reference, float candidate) {
    if (!std::isfinite(reference) || reference <= 0.0 ||
        !std::isfinite(candidate) || candidate <= 0.0f)
      return false;
    return std::abs(static_cast<double>(candidate) - reference) <=
           maximum_derived_relative_error * reference;
  };
  const double material_product_fp64 = eps * mu;
  const double material_ratio_fp64 = mu / eps;
  const double reciprocal_eps_fp64 = 1.0 / eps;
  const double reciprocal_mu_fp64 = 1.0 / mu;
  const double refractive_index_fp64 = std::sqrt(material_product_fp64);
  const double impedance_fp64 = std::sqrt(material_ratio_fp64);
  const float material_product_fp32 = eps_fp32 * mu_fp32;
  const float material_ratio_fp32 = mu_fp32 / eps_fp32;
  const float reciprocal_eps_fp32 = 1.0f / eps_fp32;
  const float reciprocal_mu_fp32 = 1.0f / mu_fp32;
  const float refractive_index_fp32 = std::sqrt(material_product_fp32);
  const float impedance_fp32 = std::sqrt(material_ratio_fp32);
  double maximum_material_relative_error = 0.0;
  const auto record_material_relative_error = [
      &maximum_material_relative_error](double reference, float candidate) {
    if (std::isfinite(reference) && reference > 0.0 &&
        std::isfinite(candidate) && candidate > 0.0f)
      maximum_material_relative_error = std::max(
          maximum_material_relative_error,
          std::abs(static_cast<double>(candidate) - reference) / reference);
  };
  record_material_relative_error(material_product_fp64,
                                 material_product_fp32);
  record_material_relative_error(material_ratio_fp64,
                                 material_ratio_fp32);
  record_material_relative_error(reciprocal_eps_fp64,
                                 reciprocal_eps_fp32);
  record_material_relative_error(reciprocal_mu_fp64,
                                 reciprocal_mu_fp32);
  record_material_relative_error(refractive_index_fp64,
                                 refractive_index_fp32);
  record_material_relative_error(impedance_fp64, impedance_fp32);
  fp32_representable =
      fp32_representable &&
      positive_fp32_derived_value_is_safe(material_product_fp64,
                                          material_product_fp32) &&
      positive_fp32_derived_value_is_safe(material_ratio_fp64,
                                          material_ratio_fp32) &&
      positive_fp32_derived_value_is_safe(reciprocal_eps_fp64,
                                          reciprocal_eps_fp32) &&
      positive_fp32_derived_value_is_safe(reciprocal_mu_fp64,
                                          reciprocal_mu_fp32) &&
      positive_fp32_derived_value_is_safe(refractive_index_fp64,
                                          refractive_index_fp32) &&
      positive_fp32_derived_value_is_safe(impedance_fp64, impedance_fp32);

  double minimum_distance_squared = fp64_infinity;
  double maximum_distance_squared = 0.0;
  double maximum_coordinate = maximum_input_coordinate;
  for (std::size_t target = 0; target < target_count; ++target) {
    const double point[3] = {
        is_cylindrical
            ? std::hypot(targets[target].x, targets[target].y)
            : targets[target].x,
        is_cylindrical ? 0.0 : targets[target].y, targets[target].z};
    for (const source_surface_bounds &surface : source_surfaces) {
      double distance_squared = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        const double gap =
            point[axis] < surface.lower[axis]
                ? surface.lower[axis] - point[axis]
                : (point[axis] > surface.upper[axis]
                       ? point[axis] - surface.upper[axis]
                       : 0.0);
        distance_squared += gap * gap;
      }
      // A surface AABB is exact in its normal direction but deliberately
      // treats the sampled tangential grid as continuous. If the target lies
      // inside that box, refine only this ambiguous surface against its
      // actual source points. This avoids false mixed-precision selection for
      // periodic images without adding O(target*source*copy) work to ordinary
      // transforms whose boxes are separated.
      if (distance_squared == 0.0) {
        distance_squared = fp64_infinity;
        const near2far_request_fp32 &request =
            requests[surface.request_index];
        const near2far_periodic_copy_fp64 &periodic =
            periodic_copies[surface.copy_index];
        for (std::size_t source_index = 0;
             source_index < request.point_count; ++source_index) {
          const near2far_point_fp64 &source =
              request.source_points[source_index];
          const double shifted_radius =
              source.x + periodic.displacement.x;
          const double dx =
              point[0] -
              (is_cylindrical ? std::abs(shifted_radius)
                              : shifted_radius);
          const double dy = is_cylindrical
                                ? 0.0
                                : point[1] -
                                      (source.y + periodic.displacement.y);
          const double dz =
              point[2] - (source.z + periodic.displacement.z);
          distance_squared = std::min(
              distance_squared, dx * dx + dy * dy + dz * dz);
        }
      }
      minimum_distance_squared =
          std::min(minimum_distance_squared, distance_squared);
    }
  }
  for (int axis = 0; axis < 3; ++axis) {
    double farthest = std::max(
        std::abs(target_min[axis] - source_max[axis]),
        std::abs(target_max[axis] - source_min[axis]));
    if (is_cylindrical && axis == 0)
      farthest = target_max[0] + source_max[0];
    if (is_cylindrical && axis == 1) farthest = 0.0;
    maximum_distance_squared += farthest * farthest;
    maximum_coordinate = std::max(
        maximum_coordinate,
        std::max(std::max(std::abs(target_min[axis]),
                          std::abs(target_max[axis])),
                 std::max(std::abs(source_min[axis]),
                          std::abs(source_max[axis]))));
  }
  const double minimum_distance = std::sqrt(minimum_distance_squared);
  const double maximum_distance = std::sqrt(maximum_distance_squared);
  constexpr double pi_fp64 =
      3.141592653589793238462643383279502884;
  constexpr double pi_fp32 = 3.14159265358979323846f;
  double maximum_k = 0.0;
  double minimum_k = fp64_infinity;
  double maximum_k_error = 0.0;
  double maximum_k_relative_error = 0.0;
  double maximum_kr = 0.0;
  for (std::size_t frequency = 0; frequency < frequency_count; ++frequency) {
    const double k = 2.0 * pi_fp64 * frequencies[frequency] *
                     refractive_index_fp64;
    maximum_k = std::max(maximum_k, std::abs(k));
    minimum_k = std::min(minimum_k, std::abs(k));
    maximum_kr =
        std::max(maximum_kr, std::abs(k) * maximum_distance);
    const float frequency_fp32 = static_cast<float>(frequencies[frequency]);
    fp32_representable =
        fp32_representable && std::isfinite(frequency_fp32) &&
        frequency_fp32 != 0.0f;
    if (fp32_representable) {
      const float k_fp32 =
          2.0f * static_cast<float>(pi_fp32) * frequency_fp32 *
          refractive_index_fp32;
      maximum_k_error = std::max(
          maximum_k_error,
          std::abs(k - static_cast<double>(k_fp32)));
      const double absolute_k = std::abs(k);
      const float absolute_k_fp32 = std::abs(k_fp32);
      if (absolute_k > 0.0 && std::isfinite(absolute_k_fp32))
        maximum_k_relative_error = std::max(
            maximum_k_relative_error,
            std::abs(static_cast<double>(absolute_k_fp32) - absolute_k) /
                absolute_k);
      fp32_representable =
          fp32_representable &&
          positive_fp32_derived_value_is_safe(absolute_k,
                                              absolute_k_fp32);
      // Green's near-field terms evaluate both 1/kr and 1/(kr*kr) in
      // FP32. Check the smallest and largest possible source distance so the
      // complete monotone interval is protected against product/reciprocal
      // overflow, underflow, and excessive relative error.
      const auto fp32_kr_family_is_safe = [
          &positive_fp32_derived_value_is_safe, absolute_k,
          absolute_k_fp32](double distance) {
        const float distance_fp32 = static_cast<float>(distance);
        const double kr_fp64 = absolute_k * distance;
        const float kr_fp32 = absolute_k_fp32 * distance_fp32;
        if (!positive_fp32_derived_value_is_safe(kr_fp64, kr_fp32))
          return false;
        const double reciprocal_kr_fp64 = 1.0 / kr_fp64;
        const float reciprocal_kr_fp32 = 1.0f / kr_fp32;
        const double kr_squared_fp64 = kr_fp64 * kr_fp64;
        const float kr_squared_fp32 = kr_fp32 * kr_fp32;
        const double reciprocal_kr_squared_fp64 = 1.0 / kr_squared_fp64;
        const float reciprocal_kr_squared_fp32 = 1.0f / kr_squared_fp32;
        const double radial_prefactor_fp64 = absolute_k / distance;
        const float radial_prefactor_fp32 = absolute_k_fp32 / distance_fp32;
        const double inverse_distance_fp64 = 1.0 / distance;
        const float inverse_distance_fp32 = 1.0f / distance_fp32;
        const bool safe = positive_fp32_derived_value_is_safe(
                              reciprocal_kr_fp64, reciprocal_kr_fp32) &&
                          positive_fp32_derived_value_is_safe(
                              kr_squared_fp64, kr_squared_fp32) &&
                          positive_fp32_derived_value_is_safe(
                              reciprocal_kr_squared_fp64,
                              reciprocal_kr_squared_fp32) &&
                          positive_fp32_derived_value_is_safe(
                              radial_prefactor_fp64,
                              radial_prefactor_fp32) &&
                          positive_fp32_derived_value_is_safe(
                              inverse_distance_fp64,
                              inverse_distance_fp32);
        return safe;
      };
      fp32_representable =
          fp32_representable &&
          fp32_kr_family_is_safe(minimum_distance) &&
          fp32_kr_family_is_safe(maximum_distance);
    }
  }
  const double fp32_unit_roundoff_for_selector =
      static_cast<double>(std::numeric_limits<float>::epsilon());
  const float azimuthal_mode_fp32 = static_cast<float>(azimuthal_mode);
  const float greencyl_tolerance_fp32 =
      static_cast<float>(greencyl_tolerance);
  const double azimuthal_phase_error_bound = is_cylindrical
      ? (2.0 * pi_fp64 *
             std::abs(azimuthal_mode -
                      static_cast<double>(azimuthal_mode_fp32)) +
         32.0 * fp32_unit_roundoff_for_selector * pi_fp64 *
             (std::abs(azimuthal_mode) + 1.0))
      : 0.0;
  fp32_representable =
      fp32_representable &&
      (!is_cylindrical ||
       (std::isfinite(azimuthal_mode_fp32) &&
        std::isfinite(greencyl_tolerance_fp32) &&
        greencyl_tolerance_fp32 > 0.0f));
  const double radial_rotation_coordinate_error = is_cylindrical
      ? source_max[0] *
            32.0 * fp32_unit_roundoff_for_selector * pi_fp64
      : 0.0;
  const double coordinate_error_bound =
      16.0 * static_cast<double>(std::numeric_limits<float>::epsilon()) *
          maximum_coordinate +
      radial_rotation_coordinate_error;
  const double direction_error_bound =
      minimum_distance > 0.0
          ? coordinate_error_bound / minimum_distance
          : fp64_infinity;
  // Let the computed kr be kr*(1+e). The k conversion and the FP32 distance
  // computation give |e| <= dk + dr + dk*dr, plus a guarded allowance for the
  // multiply/sqrt rounding already represented by unit roundoff. Therefore
  // the exact relative perturbations of 1/kr and 1/kr^2 are bounded by
  // 1/(1-d)-1 and 1/(1-d)^2-1 respectively. The latter dominates; multiply by
  // three for term2's largest 3/kr^2 coefficient and its assembly. This is an
  // interval proof and does not assume that FP32 quantization error is
  // monotonic between the minimum and maximum source distances.
  const double kr_relative_error_bound =
      maximum_k_relative_error + direction_error_bound +
      maximum_k_relative_error * direction_error_bound +
      8.0 * fp32_unit_roundoff_for_selector;
  const double reciprocal_kr_squared_relative_error_bound =
      kr_relative_error_bound < 0.5
          ? 1.0 /
                    ((1.0 - kr_relative_error_bound) *
                     (1.0 - kr_relative_error_bound)) -
                1.0 + 4.0 * fp32_unit_roundoff_for_selector
          : fp64_infinity;
  const double maximum_green_radial_relative_error =
      3.0 * reciprocal_kr_squared_relative_error_bound;
  const double phase_error_bound =
      (maximum_k + maximum_k_error) * coordinate_error_bound +
      maximum_k_error * maximum_distance +
      8.0 * static_cast<double>(std::numeric_limits<float>::epsilon()) *
          maximum_kr;
  const double minimum_kr = minimum_k * minimum_distance;
  const bool mixed_precision =
      force_mixed_for_cancellation || !fp32_representable ||
      maximum_kr > 2048.0 ||
      direction_error_bound > 2.0e-5 || phase_error_bound > 2.5e-4 ||
      azimuthal_phase_error_bound > 2.5e-4 ||
      (is_2d && (!(minimum_kr > 0.0) ||
                 phase_error_bound >= 0.25 * minimum_kr));
  if (used_mixed_precision) *used_mixed_precision = mixed_precision;
  const float runtime_eps =
      mixed_precision ? 1.0f : to_fp32(eps, "epsilon");
  const float runtime_mu =
      mixed_precision ? 1.0f : to_fp32(mu, "mu");

  const std::size_t term_count = checked_product(
      checked_product(source_point_count, periodic_copy_count,
                      "near-to-far source-copy term"),
      work_count, "near-to-far total term");

  // The cylindrical quadrature has a substantially larger live-register set
  // than the Cartesian Green kernels.  A 128-thread block exposes more
  // independent source interactions per SM without spilling, while the
  // compile-time-specialized 2D/3D kernels retain their established
  // 256-thread launch shape.
  const std::size_t threads = is_cylindrical ? 128u : 256u;
  constexpr std::size_t maximum_partials = 64;
  const std::size_t workspace_ceiling =
      near2far_workspace_ceiling_bytes.load(std::memory_order_relaxed);
  std::size_t partial_count =
      maximum_interactions / threads +
      (maximum_interactions % threads != 0);
  partial_count = std::max<std::size_t>(
      1, std::min(partial_count, maximum_partials));
  // Convert the selector's actual (not merely threshold) geometry, phase, and
  // material errors into a contribution-level bound, then include FP32 Green
  // arithmetic and the publication into a block partial. The kernel's single
  // measured L1 value upper-bounds every channel, so applying this bound to
  // each signed channel is conservative even though it avoids twelve extra
  // evidence reductions.
  const std::size_t interactions_per_partial_wave = checked_product(
      partial_count, threads,
      "near-to-far interactions per partial wave");
  const std::size_t maximum_serial_interactions =
      maximum_interactions / interactions_per_partial_wave +
      (maximum_interactions % interactions_per_partial_wave != 0);
  // Includes Green internal-envelope construction before channel assembly in
  // addition to the signed field arithmetic and serial partial accumulation.
  const double fast_cancellation_rounding_steps =
      (is_2d ? 224.0 : 176.0) +
      8.0 * static_cast<double>(maximum_serial_interactions);
  const double fp32_unit_roundoff =
      static_cast<double>(std::numeric_limits<float>::epsilon());
  const double fast_cancellation_gamma_numerator =
      fast_cancellation_rounding_steps * fp32_unit_roundoff;
  const double fast_cancellation_gamma =
      fast_cancellation_gamma_numerator < 0.5
          ? fast_cancellation_gamma_numerator /
                (1.0 - fast_cancellation_gamma_numerator)
          : fp64_infinity;
  const double fast_contribution_relative_error_bound =
      fast_cancellation_gamma + phase_error_bound +
      azimuthal_phase_error_bound +
      8.0 * direction_error_bound +
      4.0 * maximum_material_relative_error +
      maximum_green_radial_relative_error +
      ((is_2d || is_cylindrical) ? 1.0e-5 : 0.0) +
      // greencyl stops on the sum of the six complex-component refinement
      // deltas, while its compact evidence channel stores one maximum-
      // component Green envelope per rotated source call.  The six-component
      // sum is therefore bounded by six times that envelope.  The fast kernel
      // runs at one quarter of the public tolerance below, so publish
      // 6*(tol/4) = 1.5*tol rather than under-reporting the quadrature term.
      (is_cylindrical ? 1.5 * greencyl_tolerance : 0.0);
  const double fast_true_l1_scale =
      fast_contribution_relative_error_bound < 0.5
          ? 1.0 / (1.0 - fast_contribution_relative_error_bound)
          : fp64_infinity;
  constexpr double fast_cancellation_relative_budget = 3.0e-3;
  constexpr double fast_cancellation_absolute_budget = 3.0e-8;
  const std::size_t partial_scalar_bytes =
      mixed_precision ? sizeof(double) : sizeof(float);
  const std::size_t partial_channel_count = mixed_precision ? 12u : 13u;
  const std::size_t bytes_per_operation_work = checked_product(
      checked_product(partial_count, partial_channel_count,
                      "near-to-far partial channels"),
      partial_scalar_bytes, "near-to-far partial bytes");
  const std::size_t output_bytes_per_work =
      (mixed_precision ? 12u : 13u) * sizeof(double);
  const std::size_t target_bytes_per_work =
      mixed_precision ? sizeof(meep_cuda::cartesian_point_fp64)
                      : sizeof(meep_cuda::cartesian_point_fp32);
  if (bytes_per_operation_work + output_bytes_per_work +
          target_bytes_per_work >
      workspace_ceiling)
    throw std::logic_error(
        "Meep CUDA near-to-far minimum workspace exceeds its ceiling");
  const std::size_t operation_tile_capacity = std::min(
      request_count,
      std::max<std::size_t>(
          1, (workspace_ceiling - output_bytes_per_work -
              target_bytes_per_work) / bytes_per_operation_work));
  const std::size_t bytes_per_work = checked_phase_sum(
      checked_product(operation_tile_capacity, bytes_per_operation_work,
                      "near-to-far per-work partial workspace"),
      output_bytes_per_work + target_bytes_per_work,
      "near-to-far total per-work workspace");
  const std::size_t work_tile_capacity =
      std::max<std::size_t>(1, workspace_ceiling / bytes_per_work);
  const std::size_t frequency_tile_capacity =
      std::min(frequency_count, work_tile_capacity);
  const std::size_t target_tile_capacity = std::min(
      target_count,
      std::max<std::size_t>(1, work_tile_capacity /
                                   frequency_tile_capacity));
  const std::size_t tile_work_capacity = checked_product(
      target_tile_capacity, frequency_tile_capacity,
      "near-to-far tile work");
  const std::size_t partial_scalar_capacity = checked_product(
      checked_product(
          checked_product(operation_tile_capacity, tile_work_capacity,
                          "near-to-far tile request-work"),
          partial_count, "near-to-far tile partial"),
      partial_channel_count,
      "near-to-far tile partial scalar");
  const std::size_t partial_bytes = checked_bytes(
      partial_scalar_capacity, partial_scalar_bytes,
      "near-to-far partial workspace");
  const std::size_t tile_output_scalar_capacity = checked_product(
      tile_work_capacity, static_cast<std::size_t>(12),
      "near-to-far tile output scalar");
  const std::size_t tile_output_bytes = checked_bytes(
      tile_output_scalar_capacity, sizeof(double),
      "near-to-far tile output");
  const std::size_t tile_condition_bytes =
      mixed_precision
          ? 0
          : checked_bytes(tile_work_capacity, sizeof(double),
                          "near-to-far tile condition evidence");
  const std::size_t tile_output_allocation_bytes = checked_phase_sum(
      tile_output_bytes, tile_condition_bytes,
      "near-to-far output and condition workspace");
  const std::size_t target_buffer_bytes = checked_bytes(
      target_tile_capacity, target_bytes_per_work,
      "near-to-far target tile");
  const std::size_t total_workspace_bytes = checked_phase_sum(
      checked_phase_sum(partial_bytes, tile_output_allocation_bytes,
                        "near-to-far partial and output workspace"),
      target_buffer_bytes, "near-to-far total device workspace");
  if (total_workspace_bytes > workspace_ceiling)
    throw std::logic_error(
        "Meep CUDA near-to-far workspace planner exceeded its ceiling");

  std::vector<resident_curl_session> sessions;
  sessions.reserve(owners.size());
  for (const void *owner : owners) sessions.emplace_back(owner, true);
  const int device_ordinal = sessions[0].cache()->device_ordinal;
  for (const resident_curl_session &session : sessions)
    if (!session.active() || !session.cache() ||
        session.cache()->device_ordinal != device_ordinal)
      throw std::logic_error(
          "Meep CUDA near-to-far resident owners must use one device");
  meep_cuda::select_device(device_ordinal);
  if (consume_failure_for_testing(
          near2far_execution_failures_for_testing))
    throw std::runtime_error(
        "injected CUDA near-to-far execution failure");

  const auto cache_for_owner =
      [&sessions, &owners](const void *owner) -> resident_cache * {
    const auto found = std::find(owners.begin(), owners.end(), owner);
    if (found == owners.end())
      throw std::logic_error(
          "Meep CUDA near-to-far owner disappeared during preparation");
    return sessions[static_cast<std::size_t>(found - owners.begin())].cache();
  };

  std::uint64_t dft_bytes_avoided = 0;
  std::uint64_t descriptor_bytes_uploaded = 0;
  const auto add_counter_bytes = [](std::uint64_t &total,
                                    std::size_t bytes,
                                    const char *label) {
    if (bytes > std::numeric_limits<std::uint64_t>::max() - total)
      throw std::overflow_error(std::string("Meep CUDA near-to-far ") +
                                label + " counter overflow");
    total += static_cast<std::uint64_t>(bytes);
  };
  std::vector<near2far_request_snapshot> request_snapshots;
  request_snapshots.reserve(request_count);
  std::vector<meep_cuda::cartesian_point_fp64> flattened_coordinates;
  flattened_coordinates.reserve(source_point_count);
  std::vector<resident_cache *> cache_dependencies;
  cache_dependencies.reserve(owners.size());
  for (std::size_t request_index = 0; request_index < request_count;
       ++request_index) {
    const near2far_request_fp32 &request = requests[request_index];
    resident_cache *cache = cache_for_owner(request.owner);
    if (std::find(cache_dependencies.begin(), cache_dependencies.end(),
                  cache) == cache_dependencies.end())
      cache_dependencies.push_back(cache);
    const std::size_t dft_scalar_count = checked_product(
        checked_product(request.point_count, frequency_count,
                        "near-to-far DFT complex"),
        static_cast<std::size_t>(2), "near-to-far DFT scalar");
    const std::size_t dft_bytes = checked_bytes(
        dft_scalar_count, sizeof(float), "near-to-far DFT");
    const resolved_resident_range existing = resolve_resident_range(
        cache, request.dft_real_imag, dft_bytes, true,
        "Meep CUDA near-to-far DFT");
    if (existing.mirror && existing.mirror->device_dirty) {
      if (dft_bytes_avoided >
          std::numeric_limits<std::uint64_t>::max() - dft_bytes)
        throw std::overflow_error(
            "Meep CUDA near-to-far avoided DFT byte count overflow");
      dft_bytes_avoided += static_cast<std::uint64_t>(dft_bytes);
    }
    near2far_request_snapshot snapshot;
    snapshot.cache = cache;
    snapshot.device_dft = static_cast<const float *>(
        ensure_resident_mirror(cache, request.dft_real_imag, dft_bytes));
    snapshot.cache_generation = cache->allocation_generation;
    snapshot.point_count = request.point_count;
    snapshot.direction = request.direction;
    snapshot.electric = request.electric;
    snapshot.coordinates.reserve(request.point_count);
    for (std::size_t point = 0; point < request.point_count; ++point) {
      const meep_cuda::cartesian_point_fp64 coordinate =
          {request.source_points[point].x,
           request.source_points[point].y,
           request.source_points[point].z};
      snapshot.coordinates.push_back(coordinate);
      flattened_coordinates.push_back(coordinate);
    }
    request_snapshots.push_back(std::move(snapshot));
  }
  // Several DFT chunks can share one fields_chunk cache. Creating a later
  // mirror advances that cache's address generation, so snapshot the final
  // generation only after every request pointer has been resolved.
  for (near2far_request_snapshot &snapshot : request_snapshots)
    snapshot.cache_generation = snapshot.cache->allocation_generation;

  std::vector<double> runtime_frequencies(
      frequencies, frequencies + frequency_count);
  std::vector<meep_cuda::near2far_periodic_copy_fp64> runtime_copies;
  runtime_copies.reserve(periodic_copy_count);
  for (std::size_t copy = 0; copy < periodic_copy_count; ++copy)
    runtime_copies.push_back(
        {{periodic_copies[copy].displacement.x,
          periodic_copies[copy].displacement.y,
          periodic_copies[copy].displacement.z},
         {periodic_copies[copy].phase_real,
          periodic_copies[copy].phase_imaginary}});

  const std::size_t operation_bytes = checked_bytes(
      request_count,
      mixed_precision ? sizeof(meep_cuda::near2far_operation_mixed_fp32)
                      : sizeof(meep_cuda::near2far_operation_fp32),
      "near-to-far retained operation descriptors");
  const std::size_t coordinate_bytes = checked_bytes(
      flattened_coordinates.size(),
      mixed_precision ? sizeof(meep_cuda::cartesian_point_fp64)
                      : sizeof(meep_cuda::cartesian_point_fp32),
      "near-to-far retained source coordinates");
  const std::size_t frequency_bytes = checked_bytes(
      frequency_count, mixed_precision ? sizeof(double) : sizeof(float),
      "near-to-far retained frequencies");
  const std::size_t copy_bytes = checked_bytes(
      periodic_copy_count,
      mixed_precision ? sizeof(meep_cuda::near2far_periodic_copy_fp64)
                      : sizeof(meep_cuda::near2far_periodic_copy_fp32),
      "near-to-far retained periodic copies");
  const std::size_t coordinate_offset = operation_bytes;
  const std::size_t frequency_offset = checked_phase_sum(
      coordinate_offset, coordinate_bytes,
      "near-to-far retained coordinate offset");
  const std::size_t copy_offset = checked_phase_sum(
      frequency_offset, frequency_bytes,
      "near-to-far retained frequency offset");
  const std::size_t static_bytes = checked_phase_sum(
      copy_offset, copy_bytes, "near-to-far retained static plan");
  near2far_plan_registry &plan_registry = get_near2far_plan_registry();
  std::unique_lock<std::mutex> plan_lock(plan_registry.mutex);
  near2far_retained_plan &plan = plan_registry.plans[plan_owner];
  if (plan.static_storage && plan.device_ordinal != device_ordinal) {
    release_near2far_plan(plan);
    // Plan release selects the device that owns the old static and scratch
    // allocations. Restore this invocation's device before rebuilding the
    // retained plan after a defensive cross-device invalidation.
    meep_cuda::select_device(device_ordinal);
  }
  const bool plan_matches = same_near2far_static_plan(
      plan, request_snapshots, runtime_frequencies, runtime_copies,
      device_ordinal, dimension, mixed_precision);
  bool uploaded_plan = false;
  if (!plan_matches) {
    void *replacement = meep_cuda::allocate_device_bytes(static_bytes);
    try {
      if (mixed_precision) {
        std::vector<meep_cuda::near2far_operation_mixed_fp32> operations;
        operations.reserve(request_count);
        std::size_t coordinate_index = 0;
        for (const near2far_request_snapshot &snapshot : request_snapshots) {
          const auto *device_coordinates =
              reinterpret_cast<const meep_cuda::cartesian_point_fp64 *>(
                  static_cast<unsigned char *>(replacement) +
                  coordinate_offset) +
              coordinate_index;
          operations.push_back(
              {snapshot.device_dft, device_coordinates,
               snapshot.point_count, frequency_count, snapshot.direction,
               snapshot.electric});
          coordinate_index += snapshot.point_count;
        }
        meep_cuda::copy_to_device(
            replacement, operations.data(), operation_bytes);
        meep_cuda::copy_to_device(
            static_cast<unsigned char *>(replacement) + coordinate_offset,
            flattened_coordinates.data(), coordinate_bytes);
        meep_cuda::copy_to_device(
            static_cast<unsigned char *>(replacement) + frequency_offset,
            runtime_frequencies.data(), frequency_bytes);
        meep_cuda::copy_to_device(
            static_cast<unsigned char *>(replacement) + copy_offset,
            runtime_copies.data(), copy_bytes);
      }
      else {
        std::vector<meep_cuda::cartesian_point_fp32> coordinates_fp32;
        coordinates_fp32.reserve(flattened_coordinates.size());
        for (const auto &coordinate : flattened_coordinates)
          coordinates_fp32.push_back(
              {static_cast<float>(coordinate.x),
               static_cast<float>(coordinate.y),
               static_cast<float>(coordinate.z)});
        std::vector<float> frequencies_fp32;
        frequencies_fp32.reserve(runtime_frequencies.size());
        for (double frequency : runtime_frequencies)
          frequencies_fp32.push_back(static_cast<float>(frequency));
        std::vector<meep_cuda::near2far_periodic_copy_fp32> copies_fp32;
        copies_fp32.reserve(runtime_copies.size());
        for (const auto &copy : runtime_copies)
          copies_fp32.push_back(
              {{static_cast<float>(copy.displacement.x),
                static_cast<float>(copy.displacement.y),
                static_cast<float>(copy.displacement.z)},
               {static_cast<float>(copy.phase.real),
                static_cast<float>(copy.phase.imag)}});
        std::vector<meep_cuda::near2far_operation_fp32> operations;
        operations.reserve(request_count);
        std::size_t coordinate_index = 0;
        for (const near2far_request_snapshot &snapshot : request_snapshots) {
          const auto *device_coordinates =
              reinterpret_cast<const meep_cuda::cartesian_point_fp32 *>(
                  static_cast<unsigned char *>(replacement) +
                  coordinate_offset) +
              coordinate_index;
          operations.push_back(
              {snapshot.device_dft, device_coordinates,
               snapshot.point_count, frequency_count, snapshot.direction,
               snapshot.electric});
          coordinate_index += snapshot.point_count;
        }
        meep_cuda::copy_to_device(
            replacement, operations.data(), operation_bytes);
        meep_cuda::copy_to_device(
            static_cast<unsigned char *>(replacement) + coordinate_offset,
            coordinates_fp32.data(), coordinate_bytes);
        meep_cuda::copy_to_device(
            static_cast<unsigned char *>(replacement) + frequency_offset,
            frequencies_fp32.data(), frequency_bytes);
        meep_cuda::copy_to_device(
            static_cast<unsigned char *>(replacement) + copy_offset,
            copies_fp32.data(), copy_bytes);
      }
    }
    catch (...) {
      meep_cuda::free_device(replacement);
      throw;
    }
    void *previous = plan.static_storage;
    plan.static_storage = replacement;
    plan.static_capacity_bytes = static_bytes;
    plan.device_ordinal = device_ordinal;
    plan.dimension = dimension;
    plan.mixed_precision = mixed_precision;
    plan.device_operations = replacement;
    plan.device_frequencies =
        static_cast<unsigned char *>(replacement) + frequency_offset;
    plan.device_periodic_copies =
        static_cast<unsigned char *>(replacement) + copy_offset;
    plan.requests.swap(request_snapshots);
    plan.frequencies.swap(runtime_frequencies);
    plan.periodic_copies.swap(runtime_copies);
    plan.cache_dependencies = cache_dependencies;
    meep_cuda::free_device(previous);
    if (!previous) live_device_buffer_count().fetch_add(1);
    device_buffer_allocations.fetch_add(1, std::memory_order_relaxed);
    descriptor_bytes_uploaded = static_cast<std::uint64_t>(static_bytes);
    uploaded_plan = true;
  }
  else {
    device_buffer_reuses.fetch_add(1, std::memory_order_relaxed);
  }

  const std::size_t retained_workspace_bytes = checked_phase_sum(
      checked_phase_sum(plan.target_capacity_bytes,
                        plan.partial_capacity_bytes,
                        "near-to-far retained target and partial workspace"),
      plan.output_capacity_bytes,
      "near-to-far retained total workspace");
  const bool retained_buffers_are_sufficient =
      plan.target_buffer && plan.target_capacity_bytes >= target_buffer_bytes &&
      plan.partial_buffer && plan.partial_capacity_bytes >= partial_bytes &&
      plan.output_buffer &&
      plan.output_capacity_bytes >= tile_output_allocation_bytes;
  if (!retained_buffers_are_sufficient ||
      retained_workspace_bytes > workspace_ceiling) {
    // A retained allocation from a larger transform (or from a previous,
    // larger ceiling) is real device memory even when this call needs only a
    // prefix. Evict all three scratch buffers before rebuilding them so their
    // live capacity, including the reallocation transition, never exceeds the
    // configured ceiling.
    const std::uint64_t released =
        (plan.target_buffer ? 1u : 0u) +
        (plan.partial_buffer ? 1u : 0u) +
        (plan.output_buffer ? 1u : 0u);
    meep_cuda::free_device(plan.target_buffer);
    meep_cuda::free_device(plan.partial_buffer);
    meep_cuda::free_device(plan.output_buffer);
    plan.target_buffer = nullptr;
    plan.target_capacity_bytes = 0;
    plan.partial_buffer = nullptr;
    plan.partial_capacity_bytes = 0;
    plan.output_buffer = nullptr;
    plan.output_capacity_bytes = 0;
    if (released) live_device_buffer_count().fetch_sub(released);

    void *replacement_target = nullptr;
    void *replacement_partial = nullptr;
    void *replacement_output = nullptr;
    try {
      replacement_target =
          meep_cuda::allocate_device_bytes(target_buffer_bytes);
      replacement_partial = meep_cuda::allocate_device_bytes(partial_bytes);
      replacement_output =
          meep_cuda::allocate_device_bytes(tile_output_allocation_bytes);
    }
    catch (...) {
      meep_cuda::free_device(replacement_target);
      meep_cuda::free_device(replacement_partial);
      meep_cuda::free_device(replacement_output);
      throw;
    }
    plan.target_buffer = replacement_target;
    plan.target_capacity_bytes = target_buffer_bytes;
    plan.partial_buffer = replacement_partial;
    plan.partial_capacity_bytes = partial_bytes;
    plan.output_buffer = replacement_output;
    plan.output_capacity_bytes = tile_output_allocation_bytes;
    live_device_buffer_count().fetch_add(3);
    device_buffer_allocations.fetch_add(3, std::memory_order_relaxed);
  }
  else {
    device_buffer_reuses.fetch_add(3, std::memory_order_relaxed);
  }
  const std::size_t actual_retained_workspace_bytes = checked_phase_sum(
      checked_phase_sum(plan.target_capacity_bytes,
                        plan.partial_capacity_bytes,
                        "near-to-far actual target and partial workspace"),
      plan.output_capacity_bytes,
      "near-to-far actual retained workspace");
  if (actual_retained_workspace_bytes > workspace_ceiling)
    throw std::logic_error(
        "Meep CUDA near-to-far retained workspace exceeded its ceiling");

  const void *device_operation_pointer = plan.device_operations;
  void *device_target_pointer = plan.target_buffer;
  const void *device_frequency_pointer = plan.device_frequencies;
  const void *device_copy_pointer = plan.device_periodic_copies;
  void *device_partial_pointer = plan.partial_buffer;
  void *device_output_pointer = plan.output_buffer;
  const meep_cuda::near2far_cartesian_dimension runtime_dimension =
      is_2d ? meep_cuda::near2far_cartesian_dimension::two
            : (is_cylindrical
                   ? meep_cuda::near2far_cartesian_dimension::cylindrical
                   : meep_cuda::near2far_cartesian_dimension::three);
  void *device_condition_pointer =
      mixed_precision
          ? nullptr
          : static_cast<void *>(
                static_cast<unsigned char *>(plan.output_buffer) +
                tile_output_bytes);
  bool cancellation_retry_required = false;
  try {
    std::vector<double> tile_output(tile_output_scalar_capacity);
    std::vector<double> tile_accumulated(tile_output_scalar_capacity);
    std::vector<double> tile_condition(tile_work_capacity);
    std::vector<double> tile_condition_accumulated(tile_work_capacity);
    std::vector<meep_cuda::cartesian_point_fp32> runtime_targets_fp32;
    std::vector<meep_cuda::cartesian_point_fp64> runtime_targets_fp64;
    runtime_targets_fp32.reserve(target_tile_capacity);
    runtime_targets_fp64.reserve(target_tile_capacity);

    std::size_t target_begin = 0;
    std::uint64_t target_tiles = 0;
    std::uint64_t frequency_tiles = 0;
    std::uint64_t operation_tiles = 0;
    std::uint64_t execution_tiles = 0;
    const std::uint64_t descriptor_uploads = uploaded_plan ? 1u : 0u;
    std::uint64_t result_bytes = 0;
    std::uint64_t condition_bytes = 0;
    while (target_begin < target_count) {
      const std::size_t tile_targets =
          std::min(target_tile_capacity, target_count - target_begin);
      runtime_targets_fp32.clear();
      runtime_targets_fp64.clear();
      for (std::size_t local_target = 0; local_target < tile_targets;
           ++local_target) {
        const near2far_point_fp64 &target =
            targets[target_begin + local_target];
        if (mixed_precision)
          runtime_targets_fp64.push_back(
              {target.x, target.y, target.z});
        else
          runtime_targets_fp32.push_back(
              {static_cast<float>(target.x), static_cast<float>(target.y),
               static_cast<float>(target.z)});
      }
      const std::size_t tile_target_bytes = checked_bytes(
          tile_targets,
          mixed_precision ? sizeof(meep_cuda::cartesian_point_fp64)
                          : sizeof(meep_cuda::cartesian_point_fp32),
          "near-to-far target tile");
      meep_cuda::copy_to_device(
          device_target_pointer,
          mixed_precision
              ? static_cast<const void *>(runtime_targets_fp64.data())
              : static_cast<const void *>(runtime_targets_fp32.data()),
          tile_target_bytes);
      host_to_device_bytes.fetch_add(
          static_cast<std::uint64_t>(tile_target_bytes),
          std::memory_order_relaxed);
      for (std::size_t frequency_begin = 0;
           frequency_begin < frequency_count;
           frequency_begin += frequency_tile_capacity) {
        ++frequency_tiles;
        const std::size_t tile_frequencies = std::min(
            frequency_tile_capacity, frequency_count - frequency_begin);
        const std::size_t tile_scalar_count = checked_product(
            checked_product(tile_targets, tile_frequencies,
                            "near-to-far result tile work"),
            static_cast<std::size_t>(12),
            "near-to-far result tile scalar");
        const std::size_t tile_bytes = checked_bytes(
            tile_scalar_count, sizeof(double),
            "near-to-far result tile");
        std::fill(tile_accumulated.begin(),
                  tile_accumulated.begin() + tile_scalar_count, 0.0);
        if (!mixed_precision)
          std::fill(tile_condition_accumulated.begin(),
                    tile_condition_accumulated.begin() +
                        tile_targets * tile_frequencies,
                    0.0);
        for (std::size_t operation_begin = 0;
             operation_begin < request_count;
             operation_begin += operation_tile_capacity) {
          const std::size_t tile_operations = std::min(
              operation_tile_capacity, request_count - operation_begin);
          if (mixed_precision)
            meep_cuda::near2far_cartesian_mixed_fp32(
                runtime_dimension,
                static_cast<
                    const meep_cuda::near2far_operation_mixed_fp32 *>(
                    device_operation_pointer) + operation_begin,
                tile_operations,
                static_cast<const meep_cuda::cartesian_point_fp64 *>(
                    device_target_pointer),
                tile_targets,
                static_cast<const double *>(device_frequency_pointer) +
                    frequency_begin,
                tile_frequencies, frequency_begin,
                static_cast<
                    const meep_cuda::near2far_periodic_copy_fp64 *>(
                    device_copy_pointer),
                periodic_copy_count, eps, mu, partial_count,
                static_cast<double *>(device_partial_pointer),
                static_cast<double *>(device_output_pointer), threads,
                azimuthal_mode, greencyl_tolerance);
          else
            meep_cuda::near2far_cartesian_fp32(
                runtime_dimension,
                static_cast<const meep_cuda::near2far_operation_fp32 *>(
                    device_operation_pointer) + operation_begin,
                tile_operations,
                static_cast<const meep_cuda::cartesian_point_fp32 *>(
                    device_target_pointer),
                tile_targets,
                static_cast<const float *>(device_frequency_pointer) +
                    frequency_begin,
                tile_frequencies, frequency_begin,
                static_cast<const meep_cuda::near2far_periodic_copy_fp32 *>(
                    device_copy_pointer),
                periodic_copy_count, runtime_eps, runtime_mu, partial_count,
                static_cast<float *>(device_partial_pointer),
                static_cast<double *>(device_output_pointer),
                static_cast<double *>(device_condition_pointer), threads,
                static_cast<float>(phase_error_bound),
                static_cast<float>(azimuthal_mode),
                static_cast<float>(is_cylindrical
                                       ? 0.25 * greencyl_tolerance
                                       : 0.0));
          meep_cuda::copy_to_host(
              tile_output.data(), device_output_pointer, tile_bytes);
          device_to_host_bytes.fetch_add(
              static_cast<std::uint64_t>(tile_bytes),
              std::memory_order_relaxed);
          add_counter_bytes(result_bytes, tile_bytes, "result byte");
          for (std::size_t scalar = 0; scalar < tile_scalar_count; ++scalar)
            tile_accumulated[scalar] += tile_output[scalar];
          if (!mixed_precision) {
            const std::size_t evidence_bytes = checked_bytes(
                tile_targets * tile_frequencies, sizeof(double),
                "near-to-far condition evidence tile");
            meep_cuda::copy_to_host(tile_condition.data(),
                                    device_condition_pointer,
                                    evidence_bytes);
            device_to_host_bytes.fetch_add(
                static_cast<std::uint64_t>(evidence_bytes),
                std::memory_order_relaxed);
            add_counter_bytes(condition_bytes, evidence_bytes,
                              "condition evidence byte");
            for (std::size_t work = 0;
                 work < tile_targets * tile_frequencies; ++work)
              tile_condition_accumulated[work] += tile_condition[work];
          }
          ++operation_tiles;
          ++execution_tiles;
        }
        if (!mixed_precision)
          for (std::size_t local_work = 0;
               local_work < tile_targets * tile_frequencies;
               ++local_work) {
            double work_maximum_magnitude = 0.0;
            for (int channel = 0; channel < 12; ++channel)
              work_maximum_magnitude = std::max(
                  work_maximum_magnitude,
                  std::abs(tile_accumulated[12 * local_work + channel]));
            for (int channel = 0; channel < 12; ++channel) {
              const std::size_t scalar = 12 * local_work + channel;
              const double absolute_l1 =
                  tile_condition_accumulated[local_work];
              const double signed_magnitude =
                  std::abs(tile_accumulated[scalar]);
              if (std::isnan(absolute_l1) || absolute_l1 < 0.0)
                throw std::runtime_error(
                    "Meep CUDA near-to-far transform produced invalid "
                    "cancellation evidence");
              const double conservative_error_bound =
                  fast_contribution_relative_error_bound *
                  fast_true_l1_scale * absolute_l1;
              const double accepted_error =
                  fast_cancellation_relative_budget *
                      std::max(signed_magnitude, work_maximum_magnitude) +
                  fast_cancellation_absolute_budget;
              cancellation_retry_required =
                  cancellation_retry_required ||
                  (!std::isfinite(absolute_l1) ||
                   (absolute_l1 > 0.0 &&
                    conservative_error_bound > accepted_error));
            }
          }
        for (std::size_t local_target = 0; local_target < tile_targets;
             ++local_target)
          for (std::size_t local_frequency = 0;
               local_frequency < tile_frequencies; ++local_frequency) {
          const std::size_t local_work =
              local_target * tile_frequencies + local_frequency;
          const std::size_t global_work =
              (target_begin + local_target) * frequency_count +
              frequency_begin + local_frequency;
          if (!mixed_precision)
            staged_fast_error_bounds[global_work] =
                fast_contribution_relative_error_bound *
                fast_true_l1_scale *
                tile_condition_accumulated[local_work];
          for (int component = 0; component < 6; ++component) {
            const double real =
                tile_accumulated[12 * local_work + 2 * component];
            const double imaginary =
                tile_accumulated[12 * local_work + 2 * component + 1];
            if (!std::isfinite(real) || !std::isfinite(imaginary)) {
              if (mixed_precision)
                throw std::runtime_error(
                    "Meep CUDA near-to-far mixed transform produced a "
                    "non-finite result");
              // A nonfinite fast result is never publishable, but the 13th
              // evidence channel may already have identified exactly this
              // overflow/underflow failure. Discard the signed value, make
              // the work item's bound unambiguously infinite for a deferred
              // MPI decision, and retry the resident payload on mixed CUDA.
              // Throwing here would incorrectly preempt that safe retry.
              cancellation_retry_required = true;
              staged_fast_error_bounds[global_work] = fp64_infinity;
              staged_output[6 * global_work + component] =
                  std::complex<double>(0.0, 0.0);
            }
            else
              staged_output[6 * global_work + component] =
                  std::complex<double>(real, imaginary);
          }
          }
      }
      target_begin += tile_targets;
      ++target_tiles;
    }

    host_to_device_bytes.fetch_add(
        descriptor_bytes_uploaded, std::memory_order_relaxed);
    const bool immediate_retry_required =
        cancellation_retry_required && !defer_cancellation_retry;
    if (!immediate_retry_required && !collective_retry_attempt) {
      cuda_near2far_calls.fetch_add(1, std::memory_order_relaxed);
      cuda_near2far_terms.fetch_add(
          static_cast<std::uint64_t>(term_count),
          std::memory_order_relaxed);
      cuda_near2far_submitted_chunks.fetch_add(
          static_cast<std::uint64_t>(request_count),
          std::memory_order_relaxed);
      cuda_near2far_source_points.fetch_add(
          static_cast<std::uint64_t>(source_point_count),
          std::memory_order_relaxed);
      cuda_near2far_output_points.fetch_add(
          static_cast<std::uint64_t>(target_count),
          std::memory_order_relaxed);
      cuda_near2far_frequencies.fetch_add(
          static_cast<std::uint64_t>(frequency_count),
          std::memory_order_relaxed);
      cuda_near2far_periodic_copies.fetch_add(
          static_cast<std::uint64_t>(periodic_copy_count),
          std::memory_order_relaxed);
    }
    if (mixed_precision)
      cuda_near2far_mixed_precision_calls.fetch_add(
          1, std::memory_order_relaxed);
    else
      cuda_near2far_fast_precision_calls.fetch_add(
          1, std::memory_order_relaxed);
    cuda_near2far_target_tiles.fetch_add(
        target_tiles, std::memory_order_relaxed);
    cuda_near2far_frequency_tiles.fetch_add(
        frequency_tiles, std::memory_order_relaxed);
    cuda_near2far_operation_tiles.fetch_add(
        operation_tiles, std::memory_order_relaxed);
    std::uint64_t observed_workspace =
        cuda_near2far_maximum_workspace_bytes.load(
            std::memory_order_relaxed);
    while (observed_workspace < actual_retained_workspace_bytes &&
           !cuda_near2far_maximum_workspace_bytes.compare_exchange_weak(
               observed_workspace,
               static_cast<std::uint64_t>(actual_retained_workspace_bytes),
               std::memory_order_relaxed, std::memory_order_relaxed)) {}
    cuda_near2far_descriptor_uploads.fetch_add(
        descriptor_uploads, std::memory_order_relaxed);
    cuda_near2far_kernel_launches.fetch_add(
        2 * execution_tiles, std::memory_order_relaxed);
    cuda_near2far_result_device_to_host_bytes.fetch_add(
        result_bytes, std::memory_order_relaxed);
    cuda_near2far_condition_device_to_host_bytes.fetch_add(
        condition_bytes, std::memory_order_relaxed);
    if (immediate_retry_required || collective_retry_attempt) {
      cuda_near2far_cancellation_retries.fetch_add(
          1, std::memory_order_relaxed);
    }
    else {
      cuda_near2far_dft_device_to_host_bytes_avoided.fetch_add(
          dft_bytes_avoided, std::memory_order_relaxed);
    }
    if (!immediate_retry_required) {
      std::copy(staged_output.begin(), staged_output.end(), output);
      if (fast_error_bounds)
        std::copy(staged_fast_error_bounds.begin(),
                  staged_fast_error_bounds.end(), fast_error_bounds);
    }
  }
  catch (...) {
    throw;
  }
  plan_lock.unlock();
  for (resident_curl_session &session : sessions) session.finish(false);
  if (cancellation_retry_required && !defer_cancellation_retry) {
    if (force_mixed_for_cancellation)
      throw std::logic_error(
          "Meep CUDA near-to-far mixed cancellation retry recurred");
    resident_near2far_cartesian_fp32(
        dimension, plan_owner, requests, request_count, targets, target_count,
        frequencies, frequency_count, periodic_copies, periodic_copy_count,
        eps, mu, output, fast_error_bounds, false, true, false,
        used_mixed_precision, azimuthal_mode, greencyl_tolerance);
  }
#else
  (void)dimension;
  (void)plan_owner;
  (void)requests;
  (void)request_count;
  (void)targets;
  (void)target_count;
  (void)frequencies;
  (void)frequency_count;
  (void)periodic_copies;
  (void)periodic_copy_count;
  (void)eps;
  (void)mu;
  (void)output;
  (void)fast_error_bounds;
  (void)defer_cancellation_retry;
  (void)force_mixed_for_cancellation;
  (void)collective_retry_attempt;
  (void)used_mixed_precision;
  (void)azimuthal_mode;
  (void)greencyl_tolerance;
  throw std::runtime_error(
      "Meep CUDA near-to-far transform called in a CPU-only build");
#endif
}

void resident_near2far_3d_fp32(
    const void *plan_owner, const near2far_request_fp32 *requests,
    std::size_t request_count,
    const near2far_point_fp64 *targets, std::size_t target_count,
    const double *frequencies, std::size_t frequency_count,
    const near2far_periodic_copy_fp64 *periodic_copies,
    std::size_t periodic_copy_count, double eps, double mu,
    std::complex<double> *output, double *fast_error_bounds,
    bool defer_cancellation_retry, bool force_mixed_for_cancellation,
    bool collective_retry_attempt, bool *used_mixed_precision) {
  resident_near2far_cartesian_fp32(
      near2far_cartesian_dimension::three, plan_owner, requests,
      request_count, targets, target_count, frequencies, frequency_count,
      periodic_copies, periodic_copy_count, eps, mu, output,
      fast_error_bounds, defer_cancellation_retry,
      force_mixed_for_cancellation, collective_retry_attempt,
      used_mixed_precision, 0.0, 0.0);
}

void near2far_adjoint_cartesian_fp32(
    near2far_cartesian_dimension dimension,
    const void *plan_owner,
    const near2far_adjoint_request_fp64 *requests,
    std::size_t request_count,
    const near2far_point_fp64 *targets, std::size_t target_count,
    const double *frequencies, std::size_t frequency_count,
    const near2far_periodic_copy_fp64 *periodic_copies,
    std::size_t periodic_copy_count, double eps, double mu,
    const std::complex<double> *dJ, std::complex<double> *output,
    bool *used_mixed_precision, double azimuthal_mode,
    double greencyl_tolerance) {
#if MEEP_HAVE_CUDA
  if (dimension != near2far_cartesian_dimension::two &&
      dimension != near2far_cartesian_dimension::three &&
      dimension != near2far_cartesian_dimension::cylindrical)
    throw std::invalid_argument(
        "Meep CUDA adjoint near-to-far dimension must be 2D, 3D, or "
        "cylindrical");
  const bool is_2d = dimension == near2far_cartesian_dimension::two;
  const bool is_cylindrical =
      dimension == near2far_cartesian_dimension::cylindrical;
  if (!plan_owner || !requests || !targets || !frequencies ||
      !periodic_copies || !dJ || !output)
    throw std::invalid_argument(
        "Meep CUDA adjoint near-to-far inputs and output must be non-null");
  if (request_count == 0 || target_count == 0 || frequency_count == 0 ||
      periodic_copy_count == 0)
    throw std::invalid_argument(
        "Meep CUDA adjoint near-to-far counts must be nonzero");
  if (!std::isfinite(eps) || !std::isfinite(mu) || eps <= 0.0 || mu <= 0.0)
    throw std::invalid_argument(
        "Meep CUDA adjoint near-to-far epsilon and mu must be finite and "
        "positive");
  if (!std::isfinite(azimuthal_mode) ||
      (is_cylindrical
           ? (!std::isfinite(greencyl_tolerance) ||
              greencyl_tolerance <= 0.0 ||
              std::abs(azimuthal_mode) > 16380.0)
           : (azimuthal_mode != 0.0 || greencyl_tolerance != 0.0)))
    throw std::invalid_argument(
        "Meep CUDA adjoint cylindrical mode/tolerance is invalid for the "
        "selected dimension");

  std::size_t source_count = 0;
  for (std::size_t request_index = 0; request_index < request_count;
       ++request_index) {
    const near2far_adjoint_request_fp64 &request = requests[request_index];
    if (!request.source_points || !request.amplitudes ||
        request.point_count == 0 || request.direction < 0 ||
        request.direction > 2)
      throw std::invalid_argument(
          "Meep CUDA adjoint near-to-far request is incomplete");
    if (source_count > std::numeric_limits<std::size_t>::max() -
                           request.point_count)
      throw std::overflow_error(
          "Meep CUDA adjoint near-to-far source count overflow");
    source_count += request.point_count;
  }
  const std::size_t work_count = checked_product(
      source_count, frequency_count, "adjoint near-to-far work");
  const std::size_t gradient_count = checked_product(
      checked_product(target_count, frequency_count,
                      "adjoint near-to-far target-frequency"),
      static_cast<std::size_t>(6), "adjoint near-to-far dJ");
  const std::size_t term_count = checked_product(
      checked_product(work_count, target_count,
                      "adjoint near-to-far source-frequency-target term"),
      periodic_copy_count,
      "adjoint near-to-far periodic-copy term");
  (void)checked_bytes(gradient_count, sizeof(std::complex<double>),
                      "adjoint near-to-far dJ");
  (void)checked_bytes(work_count, sizeof(std::complex<double>),
                      "adjoint near-to-far output");

  const auto target_is_valid = [is_2d](
      const near2far_point_fp64 &point) {
    return std::isfinite(point.x) && std::isfinite(point.y) &&
           std::isfinite(point.z) && (!is_2d || point.z == 0.0);
  };
  const auto source_is_valid = [&target_is_valid, is_cylindrical](
      const near2far_point_fp64 &point) {
    return target_is_valid(point) &&
           (!is_cylindrical || (point.x >= 0.0 && point.y == 0.0));
  };
  for (std::size_t target = 0; target < target_count; ++target)
    if (!target_is_valid(targets[target]))
      throw std::invalid_argument(
          "Meep CUDA adjoint near-to-far target is not finite or "
          "dimensionally valid");
  for (std::size_t request_index = 0; request_index < request_count;
       ++request_index)
    for (std::size_t point = 0; point < requests[request_index].point_count;
         ++point) {
      if (!source_is_valid(requests[request_index].source_points[point]))
        throw std::invalid_argument(
            "Meep CUDA adjoint near-to-far source is not finite or "
            "dimensionally valid");
      const std::complex<double> amplitude =
          requests[request_index].amplitudes[point];
      if (!std::isfinite(real(amplitude)) ||
          !std::isfinite(imag(amplitude)))
        throw std::invalid_argument(
            "Meep CUDA adjoint near-to-far source amplitude must be "
            "finite");
    }
  for (std::size_t frequency = 0; frequency < frequency_count; ++frequency)
    if (!std::isfinite(frequencies[frequency]) ||
        (is_2d ? frequencies[frequency] <= 0.0
               : frequencies[frequency] == 0.0))
      throw std::invalid_argument(
          "Meep CUDA adjoint near-to-far frequency is invalid for the "
          "selected Green function");
  for (std::size_t copy = 0; copy < periodic_copy_count; ++copy) {
    const near2far_periodic_copy_fp64 &periodic = periodic_copies[copy];
    if (!std::isfinite(periodic.displacement.x) ||
        !std::isfinite(periodic.displacement.y) ||
        !std::isfinite(periodic.displacement.z) ||
        !std::isfinite(periodic.phase_real) ||
        !std::isfinite(periodic.phase_imaginary) ||
        (is_2d && periodic.displacement.z != 0.0) ||
        (is_cylindrical && periodic.displacement.y != 0.0))
      throw std::invalid_argument(
          "Meep CUDA adjoint near-to-far periodic copy is invalid");
  }
  for (std::size_t gradient = 0; gradient < gradient_count; ++gradient)
    if (!std::isfinite(real(dJ[gradient])) ||
        !std::isfinite(imag(dJ[gradient])))
      throw std::invalid_argument(
          "Meep CUDA adjoint near-to-far dJ must be finite");

  std::vector<meep_cuda::near2far_adjoint_source_fp64> sources_fp64;
  sources_fp64.reserve(source_count);
  for (std::size_t request_index = 0; request_index < request_count;
       ++request_index)
    for (std::size_t point = 0; point < requests[request_index].point_count;
         ++point) {
      const near2far_point_fp64 &source =
          requests[request_index].source_points[point];
      const std::complex<double> amplitude =
          requests[request_index].amplitudes[point];
      sources_fp64.push_back(
          {{source.x, source.y, source.z},
           {real(amplitude), imag(amplitude)},
           requests[request_index].direction,
           requests[request_index].electric ? 1 : 0});
    }
  std::vector<meep_cuda::cartesian_point_fp64> targets_fp64;
  targets_fp64.reserve(target_count);
  for (std::size_t target = 0; target < target_count; ++target)
    targets_fp64.push_back(
        {targets[target].x, targets[target].y, targets[target].z});
  std::vector<meep_cuda::near2far_periodic_copy_fp64> copies_fp64;
  copies_fp64.reserve(periodic_copy_count);
  for (std::size_t copy = 0; copy < periodic_copy_count; ++copy)
    copies_fp64.push_back(
        {{periodic_copies[copy].displacement.x,
          periodic_copies[copy].displacement.y,
          periodic_copies[copy].displacement.z},
         {periodic_copies[copy].phase_real,
          periodic_copies[copy].phase_imaginary}});
  std::vector<meep_cuda::complex_value_fp64> gradient_fp64;
  gradient_fp64.reserve(gradient_count);
  for (std::size_t gradient = 0; gradient < gradient_count; ++gradient)
    gradient_fp64.push_back({real(dJ[gradient]), imag(dJ[gradient])});

  constexpr double fp32_unit_roundoff =
      static_cast<double>(std::numeric_limits<float>::epsilon());
  constexpr double minimum_normal_fp32 =
      static_cast<double>(std::numeric_limits<float>::min());
  const auto fp32_normal_or_zero = [minimum_normal_fp32](double value) {
    const float converted = static_cast<float>(value);
    return std::isfinite(converted) &&
           (value == 0.0 || (std::abs(value) >= minimum_normal_fp32 &&
                            converted != 0.0f));
  };
  bool fp32_representable =
      fp32_normal_or_zero(eps) && fp32_normal_or_zero(mu) &&
      eps > 0.0 && mu > 0.0;
  double maximum_input_relative_error = 0.0;
  const auto record_fp32_conversion =
      [&fp32_representable, &maximum_input_relative_error,
       &fp32_normal_or_zero](double value) {
        fp32_representable =
            fp32_representable && fp32_normal_or_zero(value);
        if (value != 0.0) {
          const float converted = static_cast<float>(value);
          if (std::isfinite(converted) && converted != 0.0f)
            maximum_input_relative_error = std::max(
                maximum_input_relative_error,
                std::abs(static_cast<double>(converted) - value) /
                    std::abs(value));
        }
      };
  double maximum_coordinate = 1.0;
  for (const auto &source : sources_fp64) {
    const double values[] = {source.point.x, source.point.y, source.point.z,
                             source.amplitude.real,
                             source.amplitude.imag};
    for (double value : values) record_fp32_conversion(value);
    maximum_coordinate = std::max(
        maximum_coordinate,
        std::max(std::abs(source.point.x),
                 std::max(std::abs(source.point.y),
                          std::abs(source.point.z))));
  }
  for (const auto &target : targets_fp64) {
    const double values[] = {target.x, target.y, target.z};
    for (double value : values) record_fp32_conversion(value);
    maximum_coordinate = std::max(
        maximum_coordinate,
        std::max(std::abs(target.x),
                 std::max(std::abs(target.y), std::abs(target.z))));
  }
  for (const auto &copy : copies_fp64) {
    const double values[] = {copy.displacement.x, copy.displacement.y,
                             copy.displacement.z, copy.phase.real,
                             copy.phase.imag};
    for (double value : values) record_fp32_conversion(value);
    maximum_coordinate = std::max(
        maximum_coordinate,
        std::max(std::abs(copy.displacement.x),
                 std::max(std::abs(copy.displacement.y),
                          std::abs(copy.displacement.z))));
  }
  for (double frequency :
       std::vector<double>(frequencies, frequencies + frequency_count))
    record_fp32_conversion(frequency);
  for (const auto &gradient : gradient_fp64) {
    record_fp32_conversion(gradient.real);
    record_fp32_conversion(gradient.imag);
  }
  record_fp32_conversion(azimuthal_mode);
  if (is_cylindrical) record_fp32_conversion(greencyl_tolerance);

  const float eps_fp32 = static_cast<float>(eps);
  const float mu_fp32 = static_cast<float>(mu);
  const double material_references[] =
      {eps * mu, mu / eps, 1.0 / eps, 1.0 / mu,
       std::sqrt(eps * mu), std::sqrt(mu / eps)};
  const float material_candidates[] =
      {eps_fp32 * mu_fp32, mu_fp32 / eps_fp32, 1.0f / eps_fp32,
       1.0f / mu_fp32, std::sqrt(eps_fp32 * mu_fp32),
       std::sqrt(mu_fp32 / eps_fp32)};
  double maximum_material_relative_error = 0.0;
  for (std::size_t value = 0; value < 6; ++value) {
    const double reference = material_references[value];
    const float candidate = material_candidates[value];
    if (!std::isfinite(reference) || reference <= 0.0 ||
        !std::isfinite(candidate) || candidate <= 0.0f) {
      fp32_representable = false;
      continue;
    }
    const double relative_error =
        std::abs(static_cast<double>(candidate) - reference) / reference;
    maximum_material_relative_error =
        std::max(maximum_material_relative_error, relative_error);
    if (relative_error > 2.0e-5) fp32_representable = false;
  }

  double minimum_distance = std::numeric_limits<double>::infinity();
  double maximum_distance = 0.0;
  // A point-by-point host scan would duplicate the dominant
  // source*target*copy work and erase much of the GPU win. Bound each monitor
  // chunk/copy with an AABB instead. Far-field targets normally lie outside
  // these surface boxes; a target inside one conservatively selects mixed
  // CUDA without claiming that the Green tensor is singular.
  for (std::size_t request_index = 0; request_index < request_count;
       ++request_index)
    for (const auto &copy : copies_fp64) {
      double lower[3] = {
          std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::infinity()};
      double upper[3] = {
          -std::numeric_limits<double>::infinity(),
          -std::numeric_limits<double>::infinity(),
          -std::numeric_limits<double>::infinity()};
      for (std::size_t point = 0;
           point < requests[request_index].point_count; ++point) {
        const near2far_point_fp64 &source =
            requests[request_index].source_points[point];
        const double shifted[] =
            {source.x + copy.displacement.x,
             source.y + copy.displacement.y,
             source.z + copy.displacement.z};
        if (!std::isfinite(shifted[0]) || !std::isfinite(shifted[1]) ||
            !std::isfinite(shifted[2]))
          throw std::invalid_argument(
              "Meep CUDA adjoint near-to-far shifted source is invalid");
        for (int axis = 0; axis < 3; ++axis) {
          // Cylindrical greencyl rotates the signed R coordinate in the
          // Cartesian xy plane.  A negative R periodic image is therefore a
          // valid source ring with radius |R| (the CPU path accepts and
          // evaluates it).  Use the absolute radius for both AABB distance
          // bounds; signed bounds would overestimate the minimum distance
          // and make the FP32 error selector non-conservative.
          const double bounded =
              is_cylindrical && axis == 0 ? std::abs(shifted[axis])
                                           : shifted[axis];
          lower[axis] = std::min(lower[axis], bounded);
          upper[axis] = std::max(upper[axis], bounded);
          maximum_coordinate = std::max(
              maximum_coordinate, std::abs(shifted[axis]));
        }
      }
      for (const auto &target : targets_fp64) {
        const double target_values[] = {
            is_cylindrical ? std::hypot(target.x, target.y) : target.x,
            is_cylindrical ? 0.0 : target.y, target.z};
        if (!std::isfinite(target_values[0]))
          throw std::invalid_argument(
              "Meep CUDA adjoint near-to-far target radial coordinate is "
              "invalid");
        double minimum_squared = 0.0;
        double maximum_squared = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
          const double gap =
              target_values[axis] < lower[axis]
                  ? lower[axis] - target_values[axis]
                  : (target_values[axis] > upper[axis]
                         ? target_values[axis] - upper[axis]
                         : 0.0);
          const double farthest =
              is_cylindrical && axis == 0
                  ? target_values[axis] + upper[axis]
                  : std::max(
                        std::abs(target_values[axis] - lower[axis]),
                        std::abs(target_values[axis] - upper[axis]));
          minimum_squared += gap * gap;
          maximum_squared += farthest * farthest;
        }
        minimum_distance =
            std::min(minimum_distance, std::sqrt(minimum_squared));
        maximum_distance =
            std::max(maximum_distance, std::sqrt(maximum_squared));
      }
    }
  if (!std::isfinite(minimum_distance) ||
      !std::isfinite(maximum_distance) || !(maximum_distance > 0.0))
    throw std::invalid_argument(
        "Meep CUDA adjoint near-to-far distance bounds are invalid");

  constexpr double pi_fp64 =
      3.141592653589793238462643383279502884;
  const double refractive_index = std::sqrt(eps * mu);
  double minimum_k = std::numeric_limits<double>::infinity();
  double maximum_k = 0.0;
  double maximum_k_error = 0.0;
  double maximum_k_relative_error = 0.0;
  for (std::size_t frequency = 0; frequency < frequency_count; ++frequency) {
    const double k =
        2.0 * pi_fp64 * std::abs(frequencies[frequency]) *
        refractive_index;
    const float k_fp32 =
        2.0f * static_cast<float>(pi_fp64) *
        std::abs(static_cast<float>(frequencies[frequency])) *
        std::sqrt(eps_fp32 * mu_fp32);
    if (!std::isfinite(k) || !(k > 0.0) || !std::isfinite(k_fp32) ||
        !(k_fp32 > 0.0f)) {
      fp32_representable = false;
      continue;
    }
    const double error = std::abs(static_cast<double>(k_fp32) - k);
    minimum_k = std::min(minimum_k, k);
    maximum_k = std::max(maximum_k, k);
    maximum_k_error = std::max(maximum_k_error, error);
    maximum_k_relative_error =
        std::max(maximum_k_relative_error, error / k);
  }
  const double radial_rotation_coordinate_error = is_cylindrical
      ? maximum_coordinate * 32.0 * fp32_unit_roundoff * pi_fp64
      : 0.0;
  const double coordinate_error_bound =
      16.0 * fp32_unit_roundoff * maximum_coordinate +
      radial_rotation_coordinate_error;
  const double direction_error_bound =
      minimum_distance > 0.0
          ? coordinate_error_bound / minimum_distance
          : std::numeric_limits<double>::infinity();
  const double kr_relative_error_bound =
      maximum_k_relative_error + direction_error_bound +
      maximum_k_relative_error * direction_error_bound +
      8.0 * fp32_unit_roundoff;
  const double reciprocal_kr_squared_relative_error_bound =
      kr_relative_error_bound < 0.5
          ? 1.0 / ((1.0 - kr_relative_error_bound) *
                   (1.0 - kr_relative_error_bound)) -
                1.0 + 4.0 * fp32_unit_roundoff
          : std::numeric_limits<double>::infinity();
  const double maximum_green_radial_relative_error =
      3.0 * reciprocal_kr_squared_relative_error_bound;
  const double maximum_kr = maximum_k * maximum_distance;
  const double minimum_kr = minimum_k * minimum_distance;
  const double phase_error_bound =
      (maximum_k + maximum_k_error) * coordinate_error_bound +
      maximum_k_error * maximum_distance +
      8.0 * fp32_unit_roundoff * maximum_kr;
  const double azimuthal_phase_error_bound = is_cylindrical
      ? (2.0 * pi_fp64 *
             std::abs(azimuthal_mode -
                      static_cast<double>(static_cast<float>(
                          azimuthal_mode))) +
         32.0 * fp32_unit_roundoff * pi_fp64 *
             (std::abs(azimuthal_mode) + 1.0))
      : 0.0;
  bool mixed_precision =
      !fp32_representable || maximum_kr > 2048.0 ||
      direction_error_bound > 2.0e-5 || phase_error_bound > 2.5e-4 ||
      azimuthal_phase_error_bound > 2.5e-4 ||
      (is_2d && (!(minimum_kr > 0.0) ||
                 phase_error_bound >= 0.25 * minimum_kr));

  const std::size_t threads = is_cylindrical ? 128u : 256u;
  const std::size_t interaction_count = checked_product(
      target_count, periodic_copy_count,
      "adjoint near-to-far interaction");
  const std::size_t serial_interactions =
      interaction_count / threads + (interaction_count % threads != 0);
  const double rounding_steps =
      (is_2d ? 256.0 : 208.0) +
      80.0 * static_cast<double>(serial_interactions);
  const double gamma_numerator = rounding_steps * fp32_unit_roundoff;
  const double gamma = gamma_numerator < 0.5
                           ? gamma_numerator / (1.0 - gamma_numerator)
                           : std::numeric_limits<double>::infinity();
  const double fast_contribution_relative_error =
      gamma + phase_error_bound + azimuthal_phase_error_bound +
      8.0 * direction_error_bound +
      4.0 * maximum_material_relative_error +
      maximum_green_radial_relative_error +
      8.0 * maximum_input_relative_error +
      ((is_2d || is_cylindrical) ? 1.0e-5 : 0.0) +
      (is_cylindrical ? 1.5 * greencyl_tolerance : 0.0);
  if (!(fast_contribution_relative_error < 0.5)) mixed_precision = true;
  const double fast_true_l1_scale =
      fast_contribution_relative_error < 0.5
          ? 1.0 / (1.0 - fast_contribution_relative_error)
          : std::numeric_limits<double>::infinity();
  const int device_ordinal = active_cuda_device_ordinal();
  meep_cuda::select_device(device_ordinal);
  bool exact_retained_problem = false;
  if (!mixed_precision) {
    near2far_adjoint_plan_registry &registry =
        get_near2far_adjoint_plan_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.plans.find(plan_owner);
    if (found != registry.plans.end()) {
      const near2far_adjoint_retained_plan &plan = found->second;
      const bool same_frequencies =
          plan.frequencies.size() == frequency_count &&
          std::equal(plan.frequencies.begin(), plan.frequencies.end(),
                     frequencies);
      std::size_t plan_workspace = 0;
      for (std::size_t buffer = 0; buffer < 7; ++buffer)
        plan_workspace = checked_phase_sum(
            plan_workspace, plan.capacities[buffer],
            "adjoint near-to-far cached workspace");
      exact_retained_problem =
          plan.device_ordinal == device_ordinal &&
          plan_workspace <= near2far_workspace_ceiling_bytes.load(
                                std::memory_order_relaxed) &&
          plan.dimension == dimension && plan.eps == eps && plan.mu == mu &&
          plan.azimuthal_mode == azimuthal_mode &&
          plan.greencyl_tolerance == greencyl_tolerance &&
          plan.sources_valid && plan.targets_valid &&
          plan.frequencies_valid && plan.copies_valid &&
          plan.gradient_valid &&
          same_near2far_adjoint_sources(plan.sources, sources_fp64) &&
          same_near2far_points(plan.targets, targets_fp64) &&
          same_frequencies &&
          same_near2far_copies(plan.copies, copies_fp64) &&
          same_near2far_gradient(plan.gradient, gradient_fp64);
      if (exact_retained_problem && plan.mixed_precision)
        mixed_precision = true;
    }
  }
  if (used_mixed_precision) *used_mixed_precision = mixed_precision;

  std::vector<double> staged_scalars(
      checked_product(work_count, static_cast<std::size_t>(2),
                      "adjoint near-to-far staged output"));
  std::vector<double> condition(work_count, 0.0);
  std::vector<meep_cuda::near2far_adjoint_source_fp32> sources_fp32;
  std::vector<meep_cuda::cartesian_point_fp32> targets_fp32;
  std::vector<float> frequencies_fp32;
  std::vector<meep_cuda::near2far_periodic_copy_fp32> copies_fp32;
  std::vector<meep_cuda::complex_value_fp32> gradient_fp32;
  if (!mixed_precision && !exact_retained_problem) {
    sources_fp32.reserve(source_count);
    for (const auto &source : sources_fp64)
      sources_fp32.push_back(
          {{static_cast<float>(source.point.x),
            static_cast<float>(source.point.y),
            static_cast<float>(source.point.z)},
           {static_cast<float>(source.amplitude.real),
            static_cast<float>(source.amplitude.imag)},
           source.direction, source.electric});
    targets_fp32.reserve(target_count);
    for (const auto &target : targets_fp64)
      targets_fp32.push_back(
          {static_cast<float>(target.x), static_cast<float>(target.y),
           static_cast<float>(target.z)});
    frequencies_fp32.reserve(frequency_count);
    for (std::size_t frequency = 0; frequency < frequency_count;
         ++frequency)
      frequencies_fp32.push_back(static_cast<float>(frequencies[frequency]));
    copies_fp32.reserve(periodic_copy_count);
    for (const auto &copy : copies_fp64)
      copies_fp32.push_back(
          {{static_cast<float>(copy.displacement.x),
            static_cast<float>(copy.displacement.y),
            static_cast<float>(copy.displacement.z)},
           {static_cast<float>(copy.phase.real),
            static_cast<float>(copy.phase.imag)}});
    gradient_fp32.reserve(gradient_count);
    for (const auto &gradient : gradient_fp64)
      gradient_fp32.push_back(
          {static_cast<float>(gradient.real),
           static_cast<float>(gradient.imag)});
  }

  struct adjoint_tile_shape {
    std::size_t sources;
    std::size_t targets;
    std::size_t frequencies;
    std::size_t copies;
    std::size_t workspace;
  };
  const std::size_t workspace_ceiling =
      near2far_workspace_ceiling_bytes.load(std::memory_order_relaxed);
  const auto tile_shape_for = [&](bool mixed) {
    const std::size_t source_element =
        mixed ? sizeof(meep_cuda::near2far_adjoint_source_fp64)
              : sizeof(meep_cuda::near2far_adjoint_source_fp32);
    const std::size_t target_element =
        mixed ? sizeof(meep_cuda::cartesian_point_fp64)
              : sizeof(meep_cuda::cartesian_point_fp32);
    const std::size_t frequency_element =
        mixed ? sizeof(double) : sizeof(float);
    const std::size_t copy_element =
        mixed ? sizeof(meep_cuda::near2far_periodic_copy_fp64)
              : sizeof(meep_cuda::near2far_periodic_copy_fp32);
    const std::size_t gradient_element =
        mixed ? sizeof(meep_cuda::complex_value_fp64)
              : sizeof(meep_cuda::complex_value_fp32);
    const auto bytes_for = [&](const adjoint_tile_shape &shape) {
      std::size_t bytes = checked_bytes(
          shape.sources, source_element,
          "adjoint near-to-far source-tile workspace");
      bytes = checked_phase_sum(
          bytes,
          checked_bytes(shape.targets, target_element,
                        "adjoint near-to-far target-tile workspace"),
          "adjoint near-to-far descriptor workspace");
      bytes = checked_phase_sum(
          bytes,
          checked_bytes(shape.frequencies, frequency_element,
                        "adjoint near-to-far frequency-tile workspace"),
          "adjoint near-to-far descriptor workspace");
      bytes = checked_phase_sum(
          bytes,
          checked_bytes(shape.copies, copy_element,
                        "adjoint near-to-far copy-tile workspace"),
          "adjoint near-to-far descriptor workspace");
      const std::size_t gradient_tile_count = checked_product(
          checked_product(shape.targets, shape.frequencies,
                          "adjoint near-to-far gradient tile"),
          static_cast<std::size_t>(6),
          "adjoint near-to-far gradient tile components");
      bytes = checked_phase_sum(
          bytes,
          checked_bytes(gradient_tile_count, gradient_element,
                        "adjoint near-to-far gradient-tile workspace"),
          "adjoint near-to-far descriptor workspace");
      const std::size_t output_tile_count = checked_product(
          shape.sources, shape.frequencies,
          "adjoint near-to-far output tile");
      bytes = checked_phase_sum(
          bytes,
          checked_bytes(
              checked_product(output_tile_count, static_cast<std::size_t>(2),
                              "adjoint near-to-far output tile scalars"),
              sizeof(double),
              "adjoint near-to-far output-tile workspace"),
          "adjoint near-to-far dynamic workspace");
      if (!mixed)
        bytes = checked_phase_sum(
            bytes,
            checked_bytes(output_tile_count, sizeof(double),
                          "adjoint near-to-far condition-tile workspace"),
            "adjoint near-to-far dynamic workspace");
      return bytes;
    };

    adjoint_tile_shape shape = {source_count, target_count, frequency_count,
                                periodic_copy_count, 0};
    for (;;) {
      const std::size_t current = bytes_for(shape);
      if (current <= workspace_ceiling) {
        shape.workspace = current;
        return shape;
      }
      adjoint_tile_shape best = shape;
      std::size_t best_workspace = current;
      const auto consider = [&](int axis) {
        adjoint_tile_shape candidate = shape;
        std::size_t *count = axis == 0   ? &candidate.sources
                             : axis == 1 ? &candidate.targets
                             : axis == 2 ? &candidate.frequencies
                                         : &candidate.copies;
        if (*count <= 1) return;
        *count = *count / 2 + (*count % 2 != 0);
        const std::size_t candidate_workspace = bytes_for(candidate);
        if (candidate_workspace < best_workspace) {
          best = candidate;
          best_workspace = candidate_workspace;
        }
      };
      consider(0);
      consider(1);
      consider(2);
      consider(3);
      if (best_workspace >= current)
        throw std::runtime_error(
            "Meep CUDA adjoint near-to-far workspace ceiling cannot hold "
            "one source/target/frequency/copy tile");
      shape = best;
    }
  };

  std::uint64_t total_h2d_bytes = 0;
  std::uint64_t total_result_d2h_bytes = 0;
  std::uint64_t total_condition_d2h_bytes = 0;
  std::uint64_t maximum_workspace = 0;
  std::uint64_t launch_count = 0;
  std::uint64_t upload_count = 0;
  const auto add_bytes = [](std::uint64_t &counter, std::size_t bytes,
                            const char *label) {
    if (bytes > std::numeric_limits<std::uint64_t>::max() - counter)
      throw std::overflow_error(std::string("Meep CUDA adjoint near-to-far ") +
                                label + " byte counter overflow");
    counter += static_cast<std::uint64_t>(bytes);
  };
  const auto execute = [&](bool mixed) {
    const adjoint_tile_shape shape = tile_shape_for(mixed);
    const std::size_t source_bytes = checked_bytes(
        shape.sources,
        mixed ? sizeof(meep_cuda::near2far_adjoint_source_fp64)
              : sizeof(meep_cuda::near2far_adjoint_source_fp32),
        "adjoint near-to-far source tile");
    const std::size_t target_bytes = checked_bytes(
        shape.targets,
        mixed ? sizeof(meep_cuda::cartesian_point_fp64)
              : sizeof(meep_cuda::cartesian_point_fp32),
        "adjoint near-to-far target tile");
    const std::size_t frequency_bytes = checked_bytes(
        shape.frequencies, mixed ? sizeof(double) : sizeof(float),
        "adjoint near-to-far frequency tile");
    const std::size_t copy_bytes = checked_bytes(
        shape.copies,
        mixed ? sizeof(meep_cuda::near2far_periodic_copy_fp64)
              : sizeof(meep_cuda::near2far_periodic_copy_fp32),
        "adjoint near-to-far periodic-copy tile");
    const std::size_t gradient_bytes = checked_bytes(
        checked_product(
            checked_product(shape.targets, shape.frequencies,
                            "adjoint near-to-far gradient tile"),
            static_cast<std::size_t>(6),
            "adjoint near-to-far gradient tile components"),
        mixed ? sizeof(meep_cuda::complex_value_fp64)
              : sizeof(meep_cuda::complex_value_fp32),
        "adjoint near-to-far dJ tile");
    const std::size_t output_tile_count = checked_product(
        shape.sources, shape.frequencies,
        "adjoint near-to-far output tile");
    const std::size_t output_bytes = checked_bytes(
        checked_product(output_tile_count, static_cast<std::size_t>(2),
                        "adjoint near-to-far output tile scalar"),
        sizeof(double), "adjoint near-to-far output tile");
    const std::size_t condition_bytes =
        mixed ? 0 : checked_bytes(output_tile_count, sizeof(double),
                                  "adjoint near-to-far condition tile");
    const std::size_t required_capacities[7] = {
        source_bytes, target_bytes, frequency_bytes, copy_bytes,
        gradient_bytes, output_bytes, condition_bytes};
    near2far_adjoint_plan_registry &registry =
        get_near2far_adjoint_plan_registry();
    std::lock_guard<std::mutex> plan_lock(registry.mutex);
    near2far_adjoint_retained_plan &plan = registry.plans[plan_owner];
    if (plan.device_ordinal >= 0 &&
        (plan.device_ordinal != device_ordinal ||
         plan.mixed_precision != mixed)) {
      release_near2far_adjoint_plan(plan);
      // Releasing a plan selects the device that owns its old allocations.
      // The owner may have migrated to another GPU, so restore the device
      // selected for this invocation before allocating or launching.
      meep_cuda::select_device(device_ordinal);
    }
    std::size_t retained_workspace = 0;
    for (std::size_t buffer = 0; buffer < 7; ++buffer)
      retained_workspace = checked_phase_sum(
          retained_workspace,
          std::max(plan.capacities[buffer], required_capacities[buffer]),
          "adjoint near-to-far retained workspace");
    if (retained_workspace > workspace_ceiling) {
      release_near2far_adjoint_plan(plan);
      meep_cuda::select_device(device_ordinal);
      retained_workspace = shape.workspace;
    }
    plan.device_ordinal = device_ordinal;
    plan.mixed_precision = mixed;
    plan.dimension = dimension;
    plan.eps = eps;
    plan.mu = mu;
    plan.azimuthal_mode = azimuthal_mode;
    plan.greencyl_tolerance = greencyl_tolerance;
    for (std::size_t buffer = 0; buffer < 7; ++buffer) {
      const std::size_t required = required_capacities[buffer];
      if (required == 0) continue;
      if (plan.buffers[buffer] && plan.capacities[buffer] >= required) {
        device_buffer_reuses.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      void *replacement = meep_cuda::allocate_device_bytes(required);
      void *previous = plan.buffers[buffer];
      plan.buffers[buffer] = replacement;
      plan.capacities[buffer] = required;
      meep_cuda::free_device(previous);
      if (!previous) live_device_buffer_count().fetch_add(1);
      device_buffer_allocations.fetch_add(1, std::memory_order_relaxed);
      if (buffer == 0) plan.sources_valid = false;
      if (buffer == 1) plan.targets_valid = false;
      if (buffer == 2) plan.frequencies_valid = false;
      if (buffer == 3) plan.copies_valid = false;
      if (buffer == 4) plan.gradient_valid = false;
    }
    retained_workspace = 0;
    for (std::size_t buffer = 0; buffer < 7; ++buffer)
      retained_workspace = checked_phase_sum(
          retained_workspace, plan.capacities[buffer],
          "adjoint near-to-far retained workspace");
    if (retained_workspace > workspace_ceiling)
      throw std::logic_error(
          "Meep CUDA adjoint near-to-far retained workspace exceeds its "
          "ceiling");
    void *device_sources = plan.buffers[0];
    void *device_targets = plan.buffers[1];
    void *device_frequencies = plan.buffers[2];
    void *device_copies = plan.buffers[3];
    void *device_gradient = plan.buffers[4];
    void *device_output = plan.buffers[5];
    void *device_condition = mixed ? nullptr : plan.buffers[6];
    maximum_workspace = std::max(
        maximum_workspace, static_cast<std::uint64_t>(retained_workspace));

    bool uploaded_since_launch = false;
    const auto upload = [&](void *destination, const void *source,
                            std::size_t bytes) {
      meep_cuda::copy_to_device(destination, source, bytes);
      add_bytes(total_h2d_bytes, bytes, "host-to-device");
      host_to_device_bytes.fetch_add(
          static_cast<std::uint64_t>(bytes), std::memory_order_relaxed);
      uploaded_since_launch = true;
    };
    const bool full_sources = shape.sources == source_count;
    const bool full_targets = shape.targets == target_count;
    const bool full_frequencies = shape.frequencies == frequency_count;
    const bool full_copies = shape.copies == periodic_copy_count;
    const bool full_gradient = full_targets && full_frequencies;
    const bool reuse_sources =
        full_sources && plan.sources_valid &&
        same_near2far_adjoint_sources(plan.sources, sources_fp64);
    const bool reuse_targets =
        full_targets && plan.targets_valid &&
        same_near2far_points(plan.targets, targets_fp64);
    const bool reuse_frequencies =
        full_frequencies && plan.frequencies_valid &&
        plan.frequencies.size() == frequency_count &&
        std::equal(plan.frequencies.begin(), plan.frequencies.end(),
                   frequencies);
    const bool reuse_copies =
        full_copies && plan.copies_valid &&
        same_near2far_copies(plan.copies, copies_fp64);
    const bool reuse_gradient =
        full_gradient && plan.gradient_valid &&
        same_near2far_gradient(plan.gradient, gradient_fp64);
    if (!reuse_sources) plan.sources_valid = false;
    if (!reuse_targets) plan.targets_valid = false;
    if (!reuse_frequencies) plan.frequencies_valid = false;
    if (!reuse_copies) plan.copies_valid = false;
    if (!reuse_gradient) plan.gradient_valid = false;
    if (full_targets && !reuse_targets)
      upload(device_targets,
             mixed ? static_cast<const void *>(targets_fp64.data())
                   : static_cast<const void *>(targets_fp32.data()),
             target_bytes);
    if (full_frequencies && !reuse_frequencies)
      upload(device_frequencies,
             mixed ? static_cast<const void *>(frequencies)
                   : static_cast<const void *>(frequencies_fp32.data()),
             frequency_bytes);
    if (full_copies && !reuse_copies)
      upload(device_copies,
             mixed ? static_cast<const void *>(copies_fp64.data())
                   : static_cast<const void *>(copies_fp32.data()),
             copy_bytes);
    if (full_gradient && !reuse_gradient)
      upload(device_gradient,
             mixed ? static_cast<const void *>(gradient_fp64.data())
                   : static_cast<const void *>(gradient_fp32.data()),
             gradient_bytes);

    std::vector<meep_cuda::complex_value_fp64> gradient_tile_fp64;
    std::vector<meep_cuda::complex_value_fp32> gradient_tile_fp32;
    std::vector<double> output_tile(2 * output_tile_count);
    std::vector<double> condition_tile(
        mixed ? 0 : output_tile_count);
    for (std::size_t source_start = 0; source_start < source_count;
         source_start += shape.sources) {
      const std::size_t source_tile_count =
          std::min(shape.sources, source_count - source_start);
      const std::size_t source_tile_bytes = checked_bytes(
          source_tile_count,
          mixed ? sizeof(meep_cuda::near2far_adjoint_source_fp64)
                : sizeof(meep_cuda::near2far_adjoint_source_fp32),
          "adjoint near-to-far active source tile");
      if (!reuse_sources)
        upload(device_sources,
               mixed ? static_cast<const void *>(sources_fp64.data() +
                                                 source_start)
                     : static_cast<const void *>(sources_fp32.data() +
                                                 source_start),
               source_tile_bytes);
      for (std::size_t frequency_start = 0;
           frequency_start < frequency_count;
           frequency_start += shape.frequencies) {
        const std::size_t frequency_tile_count =
            std::min(shape.frequencies, frequency_count - frequency_start);
        if (!full_frequencies) {
          const std::size_t active_frequency_bytes = checked_bytes(
              frequency_tile_count, mixed ? sizeof(double) : sizeof(float),
              "adjoint near-to-far active frequency tile");
          upload(device_frequencies,
                 mixed ? static_cast<const void *>(frequencies +
                                                   frequency_start)
                       : static_cast<const void *>(frequencies_fp32.data() +
                                                   frequency_start),
                 active_frequency_bytes);
        }
        bool first_interaction_tile = true;
        for (std::size_t target_start = 0; target_start < target_count;
             target_start += shape.targets) {
          const std::size_t target_tile_count =
              std::min(shape.targets, target_count - target_start);
          if (!full_targets) {
            const std::size_t active_target_bytes = checked_bytes(
                target_tile_count,
                mixed ? sizeof(meep_cuda::cartesian_point_fp64)
                      : sizeof(meep_cuda::cartesian_point_fp32),
                "adjoint near-to-far active target tile");
            upload(device_targets,
                   mixed ? static_cast<const void *>(targets_fp64.data() +
                                                     target_start)
                         : static_cast<const void *>(targets_fp32.data() +
                                                     target_start),
                   active_target_bytes);
          }
          if (!full_gradient) {
            const std::size_t active_gradient_count = checked_product(
                checked_product(target_tile_count, frequency_tile_count,
                                "adjoint near-to-far active gradient tile"),
                static_cast<std::size_t>(6),
                "adjoint near-to-far active gradient components");
            if (mixed) {
              gradient_tile_fp64.resize(active_gradient_count);
              for (std::size_t target = 0; target < target_tile_count;
                   ++target)
                for (std::size_t frequency = 0;
                     frequency < frequency_tile_count; ++frequency)
                  for (std::size_t component = 0; component < 6;
                       ++component)
                    gradient_tile_fp64[
                        6 * (target * frequency_tile_count + frequency) +
                        component] = gradient_fp64[
                        6 * ((target_start + target) * frequency_count +
                             frequency_start + frequency) +
                        component];
              upload(device_gradient, gradient_tile_fp64.data(),
                     checked_bytes(
                         active_gradient_count,
                         sizeof(meep_cuda::complex_value_fp64),
                         "adjoint near-to-far active mixed gradient tile"));
            }
            else {
              gradient_tile_fp32.resize(active_gradient_count);
              for (std::size_t target = 0; target < target_tile_count;
                   ++target)
                for (std::size_t frequency = 0;
                     frequency < frequency_tile_count; ++frequency)
                  for (std::size_t component = 0; component < 6;
                       ++component)
                    gradient_tile_fp32[
                        6 * (target * frequency_tile_count + frequency) +
                        component] = gradient_fp32[
                        6 * ((target_start + target) * frequency_count +
                             frequency_start + frequency) +
                        component];
              upload(device_gradient, gradient_tile_fp32.data(),
                     checked_bytes(
                         active_gradient_count,
                         sizeof(meep_cuda::complex_value_fp32),
                         "adjoint near-to-far active fast gradient tile"));
            }
          }
          for (std::size_t copy_start = 0;
               copy_start < periodic_copy_count;
               copy_start += shape.copies) {
            const std::size_t copy_tile_count =
                std::min(shape.copies, periodic_copy_count - copy_start);
            if (!full_copies) {
              const std::size_t active_copy_bytes = checked_bytes(
                  copy_tile_count,
                  mixed
                      ? sizeof(meep_cuda::near2far_periodic_copy_fp64)
                      : sizeof(meep_cuda::near2far_periodic_copy_fp32),
                  "adjoint near-to-far active copy tile");
              upload(device_copies,
                     mixed
                         ? static_cast<const void *>(copies_fp64.data() +
                                                   copy_start)
                         : static_cast<const void *>(copies_fp32.data() +
                                                   copy_start),
                     active_copy_bytes);
            }
            const bool accumulate = !first_interaction_tile;
            if (mixed)
              meep_cuda::near2far_adjoint_mixed_fp32(
                  is_2d ? meep_cuda::near2far_cartesian_dimension::two
                        : (is_cylindrical
                               ? meep_cuda::near2far_cartesian_dimension::cylindrical
                               : meep_cuda::near2far_cartesian_dimension::three),
                  static_cast<
                      const meep_cuda::near2far_adjoint_source_fp64 *>(
                      device_sources),
                  source_tile_count,
                  static_cast<const meep_cuda::cartesian_point_fp64 *>(
                      device_targets),
                  target_tile_count,
                  static_cast<const double *>(device_frequencies),
                  frequency_tile_count,
                  static_cast<
                      const meep_cuda::near2far_periodic_copy_fp64 *>(
                      device_copies),
                  copy_tile_count,
                  static_cast<const meep_cuda::complex_value_fp64 *>(
                      device_gradient),
                  eps, mu, static_cast<double *>(device_output),
                  static_cast<int>(threads), azimuthal_mode,
                  greencyl_tolerance, accumulate);
            else
              meep_cuda::near2far_adjoint_fp32(
                  is_2d ? meep_cuda::near2far_cartesian_dimension::two
                        : (is_cylindrical
                               ? meep_cuda::near2far_cartesian_dimension::cylindrical
                               : meep_cuda::near2far_cartesian_dimension::three),
                  static_cast<
                      const meep_cuda::near2far_adjoint_source_fp32 *>(
                      device_sources),
                  source_tile_count,
                  static_cast<const meep_cuda::cartesian_point_fp32 *>(
                      device_targets),
                  target_tile_count,
                  static_cast<const float *>(device_frequencies),
                  frequency_tile_count,
                  static_cast<
                      const meep_cuda::near2far_periodic_copy_fp32 *>(
                      device_copies),
                  copy_tile_count,
                  static_cast<const meep_cuda::complex_value_fp32 *>(
                      device_gradient),
                  static_cast<float>(eps), static_cast<float>(mu),
                  static_cast<double *>(device_output),
                  static_cast<double *>(device_condition),
                  static_cast<int>(threads),
                  static_cast<float>(phase_error_bound),
                  static_cast<float>(azimuthal_mode),
                  static_cast<float>(is_cylindrical
                                         ? 0.25 * greencyl_tolerance
                                         : 0.0),
                  accumulate);
            first_interaction_tile = false;
            ++launch_count;
            if (uploaded_since_launch) {
              ++upload_count;
              uploaded_since_launch = false;
            }
          }
        }

        const std::size_t active_output_count = checked_product(
            source_tile_count, frequency_tile_count,
            "adjoint near-to-far active output tile");
        const std::size_t active_output_bytes = checked_bytes(
            checked_product(active_output_count, static_cast<std::size_t>(2),
                            "adjoint near-to-far active output scalars"),
            sizeof(double), "adjoint near-to-far active output tile");
        meep_cuda::copy_to_host(output_tile.data(), device_output,
                                active_output_bytes);
        add_bytes(total_result_d2h_bytes, active_output_bytes,
                  "result device-to-host");
        std::size_t active_condition_bytes = 0;
        if (!mixed) {
          active_condition_bytes = checked_bytes(
              active_output_count, sizeof(double),
              "adjoint near-to-far active condition tile");
          meep_cuda::copy_to_host(condition_tile.data(), device_condition,
                                  active_condition_bytes);
          add_bytes(total_condition_d2h_bytes, active_condition_bytes,
                    "condition device-to-host");
        }
        device_to_host_bytes.fetch_add(
            static_cast<std::uint64_t>(active_output_bytes +
                                       active_condition_bytes),
            std::memory_order_relaxed);
        for (std::size_t source = 0; source < source_tile_count; ++source)
          for (std::size_t frequency = 0;
               frequency < frequency_tile_count; ++frequency) {
            const std::size_t local_work =
                source * frequency_tile_count + frequency;
            const std::size_t global_work =
                (source_start + source) * frequency_count +
                frequency_start + frequency;
            staged_scalars[2 * global_work] = output_tile[2 * local_work];
            staged_scalars[2 * global_work + 1] =
                output_tile[2 * local_work + 1];
            if (!mixed) condition[global_work] = condition_tile[local_work];
          }
      }
    }
    if (full_sources) {
      plan.sources = sources_fp64;
      plan.sources_valid = true;
    }
    else {
      plan.sources.clear();
      plan.sources_valid = false;
    }
    if (full_targets) {
      plan.targets = targets_fp64;
      plan.targets_valid = true;
    }
    else {
      plan.targets.clear();
      plan.targets_valid = false;
    }
    if (full_frequencies) {
      plan.frequencies.assign(frequencies, frequencies + frequency_count);
      plan.frequencies_valid = true;
    }
    else {
      plan.frequencies.clear();
      plan.frequencies_valid = false;
    }
    if (full_copies) {
      plan.copies = copies_fp64;
      plan.copies_valid = true;
    }
    else {
      plan.copies.clear();
      plan.copies_valid = false;
    }
    if (full_gradient) {
      plan.gradient = gradient_fp64;
      plan.gradient_valid = true;
    }
    else {
      plan.gradient.clear();
      plan.gradient_valid = false;
    }
  };

  bool retried_for_cancellation = false;
  if (!mixed_precision) {
    execute(false);
    bool retry = false;
    for (std::size_t work = 0; work < work_count; ++work) {
      const double real_value = staged_scalars[2 * work];
      const double imaginary_value = staged_scalars[2 * work + 1];
      const double l1 = condition[work];
      if (!std::isfinite(real_value) || !std::isfinite(imaginary_value) ||
          !std::isfinite(l1) || l1 < 0.0) {
        retry = true;
        continue;
      }
      const double conservative_error =
          fast_contribution_relative_error * fast_true_l1_scale * l1;
      const double accepted_error =
          3.0e-3 * std::max(std::abs(real_value),
                            std::abs(imaginary_value)) +
          3.0e-8;
      retry = retry || conservative_error > accepted_error;
    }
    cuda_near2far_adjoint_fast_precision_calls.fetch_add(
        1, std::memory_order_relaxed);
    if (retry) {
      mixed_precision = true;
      retried_for_cancellation = true;
      if (used_mixed_precision) *used_mixed_precision = true;
    }
  }
  if (mixed_precision) {
    execute(true);
    cuda_near2far_adjoint_mixed_precision_calls.fetch_add(
        1, std::memory_order_relaxed);
  }
  for (std::size_t work = 0; work < work_count; ++work) {
    const double real_value = staged_scalars[2 * work];
    const double imaginary_value = staged_scalars[2 * work + 1];
    if (!std::isfinite(real_value) || !std::isfinite(imaginary_value))
      throw std::runtime_error(
          "Meep CUDA adjoint near-to-far produced a non-finite result");
    output[work] = std::complex<double>(real_value, imaginary_value);
  }

  cuda_near2far_adjoint_calls.fetch_add(1, std::memory_order_relaxed);
  cuda_near2far_adjoint_terms.fetch_add(
      static_cast<std::uint64_t>(term_count), std::memory_order_relaxed);
  cuda_near2far_adjoint_submitted_chunks.fetch_add(
      static_cast<std::uint64_t>(request_count), std::memory_order_relaxed);
  cuda_near2far_adjoint_source_points.fetch_add(
      static_cast<std::uint64_t>(source_count), std::memory_order_relaxed);
  cuda_near2far_adjoint_far_points.fetch_add(
      static_cast<std::uint64_t>(target_count), std::memory_order_relaxed);
  cuda_near2far_adjoint_frequencies.fetch_add(
      static_cast<std::uint64_t>(frequency_count), std::memory_order_relaxed);
  cuda_near2far_adjoint_periodic_copies.fetch_add(
      static_cast<std::uint64_t>(periodic_copy_count),
      std::memory_order_relaxed);
  if (retried_for_cancellation)
    cuda_near2far_adjoint_cancellation_retries.fetch_add(
        1, std::memory_order_relaxed);
  std::uint64_t observed_workspace =
      cuda_near2far_adjoint_maximum_workspace_bytes.load(
          std::memory_order_relaxed);
  while (observed_workspace < maximum_workspace &&
         !cuda_near2far_adjoint_maximum_workspace_bytes.compare_exchange_weak(
             observed_workspace, maximum_workspace,
             std::memory_order_relaxed, std::memory_order_relaxed)) {}
  cuda_near2far_adjoint_descriptor_uploads.fetch_add(
      upload_count, std::memory_order_relaxed);
  cuda_near2far_adjoint_kernel_launches.fetch_add(
      launch_count, std::memory_order_relaxed);
  cuda_near2far_adjoint_host_to_device_bytes.fetch_add(
      total_h2d_bytes, std::memory_order_relaxed);
  cuda_near2far_adjoint_result_device_to_host_bytes.fetch_add(
      total_result_d2h_bytes, std::memory_order_relaxed);
  cuda_near2far_adjoint_condition_device_to_host_bytes.fetch_add(
      total_condition_d2h_bytes, std::memory_order_relaxed);
#else
  (void)dimension;
  (void)plan_owner;
  (void)requests;
  (void)request_count;
  (void)targets;
  (void)target_count;
  (void)frequencies;
  (void)frequency_count;
  (void)periodic_copies;
  (void)periodic_copy_count;
  (void)eps;
  (void)mu;
  (void)dJ;
  (void)output;
  (void)used_mixed_precision;
  (void)azimuthal_mode;
  (void)greencyl_tolerance;
  throw std::runtime_error(
      "Meep CUDA adjoint near-to-far called in a CPU-only build");
#endif
}

bool resident_dft_norm2_fp32_for_owner(
    const void *owner, const float *dft_real_imag,
    std::size_t storage_point_count, std::size_t frequency_count,
    const std::ptrdiff_t *point_indices,
    std::size_t selected_point_count, double *result) {
#if MEEP_HAVE_CUDA
  if (!owner || !dft_real_imag || !result)
    throw std::invalid_argument(
        "Meep CUDA DFT norm owner, values, and result must be non-null");
  if (storage_point_count == 0 || frequency_count == 0) {
    *result = 0.0;
    return true;
  }
  if (storage_point_count >
      std::numeric_limits<std::size_t>::max() / frequency_count)
    throw std::overflow_error("Meep CUDA DFT norm storage count overflow");
  const std::size_t complex_count =
      storage_point_count * frequency_count;
  if (complex_count >
      std::numeric_limits<std::size_t>::max() / 2)
    throw std::overflow_error("Meep CUDA DFT norm scalar count overflow");
  const std::size_t output_bytes =
      checked_bytes(2 * complex_count, sizeof(float), "DFT norm values");
  const std::size_t reduction_point_count =
      point_indices ? selected_point_count : storage_point_count;
  if (point_indices) {
    if (selected_point_count == 0) {
      *result = 0.0;
      return true;
    }
    for (std::size_t point = 0; point < selected_point_count; ++point)
      if (point_indices[point] < 0 ||
          static_cast<std::size_t>(point_indices[point]) >=
              storage_point_count)
        throw std::out_of_range(
            "Meep CUDA DFT norm point index is outside the DFT allocation");
  }
  else if (selected_point_count != storage_point_count) {
    throw std::invalid_argument(
        "Meep CUDA contiguous DFT norm point count is inconsistent");
  }

  resident_cache *cache = nullptr;
  {
    resident_registry &registry = get_resident_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.caches.find(owner);
    if (found != registry.caches.end()) cache = found->second;
  }
  if (!cache || cache->device_ordinal < 0) return false;
  select_cache_device(cache);
  const resolved_resident_range resolved = resolve_resident_range(
      cache, dft_real_imag, output_bytes, true,
      "Meep CUDA DFT norm values");
  if (!resolved.mirror || !resolved.mirror->device_dirty)
    return false;

  const std::ptrdiff_t *device_indices = nullptr;
  if (point_indices) {
    const std::size_t index_bytes = checked_bytes(
        selected_point_count, sizeof(std::ptrdiff_t),
        "DFT norm point indices");
    device_indices = static_cast<const std::ptrdiff_t *>(
        ensure_resident_mirror(
            cache, reinterpret_cast<const float *>(point_indices),
            index_bytes));
  }
  if (!cache->dft_norm_result) {
    cache->dft_norm_result = static_cast<double *>(
        meep_cuda::allocate_device_bytes(sizeof(double)));
    device_buffer_allocations.fetch_add(1);
    live_device_buffer_count().fetch_add(1);
  }
  else {
    device_buffer_reuses.fetch_add(1);
  }
  const float *device_values = reinterpret_cast<const float *>(
      static_cast<const unsigned char *>(
          resolved.mirror->device_pointer) +
      resolved.byte_offset);
  meep_cuda::initialize_squared_norm_result(cache->dft_norm_result);
  meep_cuda::squared_norm_complex_fp32(
      device_values, device_indices, reduction_point_count,
      frequency_count, cache->dft_norm_result);
  meep_cuda::copy_to_host(
      result, cache->dft_norm_result, sizeof(*result));
  device_to_host_bytes.fetch_add(sizeof(*result));
  device_to_host_bytes_avoided.fetch_add(
      static_cast<std::uint64_t>(output_bytes));
  return true;
#else
  (void)owner;
  (void)dft_real_imag;
  (void)storage_point_count;
  (void)frequency_count;
  (void)point_indices;
  (void)selected_point_count;
  (void)result;
  return false;
#endif
}

} // namespace detail

} // namespace gpu
} // namespace meep
