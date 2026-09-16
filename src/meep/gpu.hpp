#ifndef MEEP_GPU_HPP
#define MEEP_GPU_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace meep {
namespace gpu {

struct device_info {
  int ordinal;
  int compute_major;
  int compute_minor;
  int multiprocessor_count;
  int max_threads_per_block;
  std::uint64_t global_memory_bytes;
  std::uint64_t memory_bandwidth_bytes_per_second;
  std::string identifier;
  std::string name;
  bool compatible;
};

struct runtime_touch_statistics {
  std::uint64_t availability_probes;
  std::uint64_t device_enumerations;
  std::uint64_t device_selections;
};

// The CPU backend remains the default. "automatic" first applies a
// workload/observed-CPU-team floor without touching CUDA, then uses selected
// device bandwidth facts for larger owners. A fields owner is entirely CPU
// or entirely CUDA for a validated time step; unsupported CUDA features do
// not create mixed per-chunk execution.
// "cuda" requires CUDA and rejects unsupported field-update configurations
// before modifying a time step.
enum class backend_mode { cpu, automatic, cuda };

struct dispatch_statistics {
  std::uint64_t cpu_curl_calls;
  std::uint64_t cpu_curl_points;
  std::uint64_t cuda_curl_calls;
  std::uint64_t cuda_curl_points;
  std::uint64_t host_to_device_bytes;
  std::uint64_t device_to_host_bytes;
};

struct resident_statistics {
  std::uint64_t host_to_device_bytes_avoided;
  std::uint64_t device_to_host_bytes_avoided;
  std::uint64_t device_buffer_allocations;
  std::uint64_t device_buffer_reuses;
};

struct field_update_statistics {
  std::uint64_t cpu_update_eh_calls;
  std::uint64_t cpu_update_eh_points;
  std::uint64_t cuda_update_eh_calls;
  std::uint64_t cuda_update_eh_points;
};

struct polarization_statistics {
  std::uint64_t cpu_update_calls;
  std::uint64_t cpu_update_points;
  std::uint64_t cuda_update_calls;
  std::uint64_t cuda_update_points;
};

struct source_statistics {
  std::uint64_t cpu_update_calls;
  std::uint64_t cpu_update_points;
  std::uint64_t cuda_update_calls;
  std::uint64_t cuda_update_points;
};

struct boundary_statistics {
  std::uint64_t cpu_update_calls;
  std::uint64_t cpu_update_points;
  std::uint64_t cuda_update_calls;
  std::uint64_t cuda_update_points;
};

struct dft_statistics {
  std::uint64_t cpu_update_calls;
  std::uint64_t cpu_update_points;
  std::uint64_t cuda_update_calls;
  std::uint64_t cuda_update_points;
};

// Logical DFT monitor updates may share one prepared phase vector and may
// execute in one physical CUDA kernel. These counters distinguish logical
// work from launch count and prove whether the resident descriptor/map plan
// was uploaded or reused.
struct dft_batch_statistics {
  std::uint64_t batch_calls;
  std::uint64_t submitted_updates;
  std::uint64_t phase_preparation_launches;
  std::uint64_t phase_reuses;
  std::uint64_t update_kernel_launches;
  std::uint64_t maximum_batch_size;
  std::uint64_t multi_monitor_automatic_checks;
  std::uint64_t multi_monitor_automatic_selected;
  std::uint64_t multi_monitor_automatic_rejected;
  std::uint64_t multi_monitor_forced_batches;
  std::uint64_t multi_monitor_batched_updates;
  std::uint64_t multi_monitor_unbatched_updates;
  std::uint64_t multi_monitor_plan_uploads;
  std::uint64_t multi_monitor_plan_reuses;
  std::uint64_t multi_monitor_metadata_host_to_device_bytes;
};

// DFT spectral-consumer evidence. CUDA calls contract resident FP32 monitor
// pairs on the device with deterministic FP64 accumulation and publish only
// O(frequency_count) results instead of complete point-frequency arrays.
struct dft_reduction_statistics {
  std::uint64_t cpu_reduction_calls;
  std::uint64_t cpu_submitted_pairs;
  std::uint64_t cpu_point_frequency_terms;
  std::uint64_t cuda_reduction_calls;
  std::uint64_t cuda_submitted_pairs;
  std::uint64_t cuda_point_frequency_terms;
  std::uint64_t cuda_descriptor_uploads;
  std::uint64_t cuda_plan_reuses;
  std::uint64_t cuda_kernel_launches;
  std::uint64_t cuda_result_device_to_host_bytes;
  std::uint64_t full_dft_device_to_host_bytes_avoided;
  std::uint64_t mpi_allreduce_calls;
  std::uint64_t mpi_allreduce_bytes;
};

// Additive evidence for DFT data consumers which materialize field arrays,
// write monitor data, or contract DFT fields with eigenmodes.  These are
// deliberately separate from step-time dft_statistics and spectral scalar
// dft_reduction_statistics: a CUDA time step followed by a host interpolation
// loop is CPU fallback, not a CUDA-qualified scientific result.
struct dft_materialization_statistics {
  std::uint64_t cpu_array_calls;
  std::uint64_t cpu_array_points;
  std::uint64_t cuda_array_calls;
  std::uint64_t cuda_array_points;
  // Dielectric/Permeability arrays are geometry queries, not resident DFT
  // materialization.  Until a device material mirror can answer them, keep
  // their host work visible without misclassifying it as a CPU DFT fallback.
  std::uint64_t host_synthetic_material_array_calls;
  std::uint64_t host_synthetic_material_array_points;
  std::uint64_t cpu_output_calls;
  std::uint64_t cpu_output_points;
  std::uint64_t cuda_output_calls;
  std::uint64_t cuda_output_points;
  // output_dft's dedicated all-component/all-frequency planar staging path.
  // Keep these distinct from cuda_output_{calls,points}, which describe the
  // public HDF5 operation rather than the resident backend work performed by
  // one or more bounded frequency tiles.
  std::uint64_t cuda_output_staging_calls;
  std::uint64_t cuda_output_staging_points;
  std::uint64_t cuda_output_staging_frequencies;
  std::uint64_t cuda_output_staging_descriptor_uploads;
  std::uint64_t cuda_output_staging_plan_reuses;
  std::uint64_t cuda_output_staging_kernel_launches;
  std::uint64_t cuda_output_staging_result_device_to_host_bytes;
  std::uint64_t cuda_output_staging_full_dft_device_to_host_bytes_avoided;
  // High-water committed active-plan bulk capacity only: device metadata +
  // device planar output + private pinned D2H scratch + published pinned-host
  // planar output. Small STL bookkeeping vectors, source mirrors, failed
  // replacement transients, and caller-held retired views are excluded.
  std::uint64_t cuda_output_staging_workspace_ceiling_bytes;
  std::uint64_t cuda_kernel_launches;
  std::uint64_t cuda_result_device_to_host_bytes;
  std::uint64_t full_dft_device_to_host_bytes_avoided;
  // All DFT-array collectives, including collapsed-output ownership
  // metadata as well as complex result payloads.
  std::uint64_t array_mpi_allreduce_calls;
  std::uint64_t array_mpi_allreduce_bytes;
};

// HDF5 checkpoint I/O necessarily crosses the host boundary, but a CUDA
// checkpoint must stage only the resident DFT payload.  Publishing every
// device-authoritative field owned by a fields_chunk is a hidden CPU fallback
// and is accounted separately as avoided full-cache traffic.
struct dft_checkpoint_statistics {
  std::uint64_t cpu_save_dataset_calls;
  std::uint64_t cpu_save_values;
  std::uint64_t cuda_save_dataset_calls;
  std::uint64_t cuda_save_values;
  std::uint64_t cuda_save_device_to_host_bytes;
  std::uint64_t cuda_save_full_cache_device_to_host_bytes_avoided;
  std::uint64_t cpu_load_dataset_calls;
  std::uint64_t cpu_load_values;
  std::uint64_t cuda_load_dataset_calls;
  std::uint64_t cuda_load_values;
  std::uint64_t cuda_load_host_to_device_bytes;
};

// DFT scale evidence is separate from checkpoint transfer evidence because
// scale_dfts is also a public operation outside load-minus-flux.  Calls and
// values are counted per dft_chunk; values are complex FP32 elements.  CUDA
// H2D bytes are nonzero only when scaling a host-authoritative chunk that did
// not already have a current resident mirror.
struct dft_scale_statistics {
  std::uint64_t cpu_scale_calls;
  std::uint64_t cpu_scale_values;
  std::uint64_t cuda_scale_calls;
  std::uint64_t cuda_scale_values;
  std::uint64_t cuda_scale_kernel_launches;
  std::uint64_t cuda_scale_host_to_device_bytes;
};

// Eigenmode profiles are currently sampled by the host MPB bridge, but their
// contraction with resident DFT fields is a distinct scientific backend.  A
// strict CUDA run may report host profile sampling while CPU overlap work must
// remain zero.
struct eigenmode_overlap_statistics {
  std::uint64_t cpu_overlap_calls;
  std::uint64_t cpu_overlap_terms;
  std::uint64_t cuda_overlap_calls;
  std::uint64_t cuda_overlap_terms;
  std::uint64_t cuda_mode_flux_calls;
  std::uint64_t cuda_mode_mode_calls;
  std::uint64_t cuda_submitted_pairs;
  std::uint64_t cuda_descriptor_uploads;
  std::uint64_t cuda_plan_reuses;
  std::uint64_t cuda_kernel_launches;
  std::uint64_t cuda_result_device_to_host_bytes;
  std::uint64_t full_dft_device_to_host_bytes_avoided;
  std::uint64_t host_mode_profile_sampling_calls;
  std::uint64_t host_mode_profile_sampling_points;
  std::uint64_t zero_rank_channels_skipped;
  std::uint64_t host_mode_profile_host_to_device_bytes;
  std::uint64_t mpi_allreduce_calls;
  std::uint64_t mpi_allreduce_bytes;
};

// Additive LDOS evidence. CUDA reductions consume resident FP32 fields and
// source profiles, accumulate in FP64, and return only four doubles per
// update instead of publishing every device-authoritative field mirror.
struct ldos_statistics {
  std::uint64_t cpu_reduction_calls;
  std::uint64_t cpu_source_points;
  std::uint64_t cuda_reduction_calls;
  std::uint64_t cuda_submitted_profiles;
  std::uint64_t cuda_source_points;
  std::uint64_t cuda_descriptor_uploads;
  std::uint64_t cuda_kernel_launches;
  std::uint64_t cuda_result_device_to_host_bytes;
  std::uint64_t full_field_device_to_host_bytes_avoided;
};

// Additive evidence for the Green-function half of near-to-far processing.
// DFT update counters alone cannot prove that far-field postprocessing ran on
// CUDA. Terms count source-point/periodic-copy/target/frequency interactions.
struct near2far_statistics {
  std::uint64_t cpu_transform_calls;
  std::uint64_t cpu_terms;
  std::uint64_t cuda_transform_calls;
  std::uint64_t cuda_terms;
  std::uint64_t cuda_submitted_chunks;
  std::uint64_t cuda_source_points;
  std::uint64_t cuda_output_points;
  std::uint64_t cuda_frequencies;
  std::uint64_t cuda_periodic_copies;
  std::uint64_t cuda_fast_precision_calls;
  std::uint64_t cuda_mixed_precision_calls;
  std::uint64_t cuda_cancellation_retries;
  std::uint64_t cuda_target_tiles;
  std::uint64_t cuda_frequency_tiles;
  std::uint64_t cuda_operation_tiles;
  std::uint64_t cuda_maximum_workspace_bytes;
  std::uint64_t cuda_descriptor_uploads;
  std::uint64_t cuda_kernel_launches;
  std::uint64_t cuda_result_device_to_host_bytes;
  std::uint64_t cuda_condition_device_to_host_bytes;
  std::uint64_t dft_device_to_host_bytes_avoided;
  std::uint64_t mpi_allreduce_calls;
  std::uint64_t mpi_allreduce_bytes;
  std::uint64_t cpu_adjoint_calls;
  std::uint64_t cpu_adjoint_terms;
  std::uint64_t cuda_adjoint_calls;
  std::uint64_t cuda_adjoint_terms;
  std::uint64_t cuda_adjoint_submitted_chunks;
  std::uint64_t cuda_adjoint_source_points;
  std::uint64_t cuda_adjoint_far_points;
  std::uint64_t cuda_adjoint_frequencies;
  std::uint64_t cuda_adjoint_periodic_copies;
  std::uint64_t cuda_adjoint_fast_precision_calls;
  std::uint64_t cuda_adjoint_mixed_precision_calls;
  std::uint64_t cuda_adjoint_cancellation_retries;
  std::uint64_t cuda_adjoint_maximum_workspace_bytes;
  std::uint64_t cuda_adjoint_descriptor_uploads;
  std::uint64_t cuda_adjoint_kernel_launches;
  std::uint64_t cuda_adjoint_host_to_device_bytes;
  std::uint64_t cuda_adjoint_result_device_to_host_bytes;
  std::uint64_t cuda_adjoint_condition_device_to_host_bytes;
};

struct multi_gpu_statistics {
  std::uint64_t mpi_messages;
  std::uint64_t mpi_scalars;
  std::uint64_t cuda_aware_bytes;
  std::uint64_t pinned_staging_bytes;
  std::uint64_t pinned_device_to_host_bytes;
  std::uint64_t pinned_host_to_device_bytes;
};

// Additive execution counters for the MPI request-completion branch used by
// distributed CUDA boundary exchanges.  This remains separate so extending
// the evidence API does not change the established multi_gpu_statistics ABI.
struct mpi_completion_statistics {
  std::uint64_t waitsome_executions;
  std::uint64_t waitall_executions;
};

// Additive API for communication/compute overlap. Keeping this separate from
// multi_gpu_statistics preserves the layout of the existing public struct.
struct boundary_eh_overlap_statistics {
  std::uint64_t checks;
  std::uint64_t eligible;
  std::uint64_t launched_h;
  std::uint64_t launched_e;
  std::uint64_t skipped_disabled;
  std::uint64_t skipped_unsupported_schedule;
  std::uint64_t skipped_no_remote;
  std::uint64_t skipped_cold_topology;
  std::uint64_t rejected;
};

// Additive statistics for overlapping the final H halo exchange with a
// partitioned D curl. Point counters describe launched work and satisfy
// interior_points + shell_points == full_points for every completed launch.
struct halo_curl_overlap_statistics {
  std::uint64_t checks;
  std::uint64_t eligible;
  std::uint64_t launches;
  std::uint64_t skipped_disabled;
  std::uint64_t skipped_unsupported_schedule;
  std::uint64_t skipped_no_remote;
  std::uint64_t skipped_cold_topology;
  std::uint64_t rejected_feature;
  std::uint64_t rejected_small;
  std::uint64_t full_points;
  std::uint64_t interior_points;
  std::uint64_t shell_points;
};

// Additive statistics for removing CPU cache-tiling descriptors from CUDA
// execution. The original tile vectors remain intact for CPU fallback;
// input_tiles counts the tile volumes represented by coalesced chunk phases.
struct tile_coalescing_statistics {
  std::uint64_t curl_chunk_phases;
  std::uint64_t curl_input_tiles;
  std::uint64_t update_eh_chunk_phases;
  std::uint64_t update_eh_input_tiles;
};

// Additive evidence for the topology-aware cross-operation phase policy.
// Automatic checks count logical phase decisions, including a validated
// replay of a previously collected phase. Singleton ordinary phases remain
// unbatched without pretending that a crossover decision was made.
struct phase_batch_policy_statistics {
  std::uint64_t curl_automatic_checks;
  std::uint64_t curl_automatic_selected;
  std::uint64_t curl_automatic_rejected;
  std::uint64_t curl_forced_batches;
  std::uint64_t curl_batched_operations;
  std::uint64_t curl_unbatched_operations;
  std::uint64_t update_eh_automatic_checks;
  std::uint64_t update_eh_automatic_selected;
  std::uint64_t update_eh_automatic_rejected;
  std::uint64_t update_eh_forced_batches;
  std::uint64_t update_eh_batched_operations;
  std::uint64_t update_eh_unbatched_operations;
};

// Additive evidence for stable complete-curl phase replay. This is a
// separate API so the previously released phase_batch_policy_statistics
// layout and its by-value getter remain ABI compatible.
struct curl_phase_replay_statistics {
  std::uint64_t checks;
  std::uint64_t hits;
  std::uint64_t unready;
  std::uint64_t generation_misses;
  std::uint64_t mirror_misses;
};

// Additive evidence for the cached MPI-boundary descriptor fast path.
// A full validation compares every gather/scatter descriptor. A fast replay
// is possible only when fields::step supplies an immutable cached-topology
// token and every resident-cache allocation generation still matches.
struct boundary_descriptor_replay_statistics {
  std::uint64_t gather_fast_replays;
  std::uint64_t gather_full_validations;
  std::uint64_t scatter_fast_replays;
  std::uint64_t scatter_full_validations;
};

// This interface exists in every Meep build. CPU-only builds return a clear
// diagnostic instead of exposing CUDA headers or failing at link time.
bool backend_compiled() noexcept;
bool runtime_available(std::string *diagnostic = nullptr);
std::vector<device_info> enumerate_devices();
// Stable physical/MIG UUID for a visible CUDA runtime ordinal. CPU-only
// builds and invalid/inaccessible ordinals return through the ordinary CUDA
// diagnostic/error path rather than fabricating an identity.
std::string device_identifier(int ordinal);
std::string compiled_architectures();
void select_device(int ordinal);

// set_backend overrides MEEP_GPU_BACKEND for this process. It is intentionally
// rank-local and contains no hidden MPI collective. Distributed programs that
// may have rank-specific configuration errors should prefer MEEP_GPU_BACKEND
// plus the fields::step collective preflight, or collectively agree on local
// set_backend success before continuing. Accepted environment values are
// "cpu", "auto", and "cuda"; the default is "cpu". MEEP_GPU_DEVICE
// optionally selects a CUDA runtime ordinal.
void set_backend(backend_mode mode);
backend_mode requested_backend();
backend_mode active_backend();
// CUDA runtime ordinal selected by this MPI rank, or -1 while the CPU
// backend is active. CUDA_VISIBLE_DEVICES is respected by the CUDA runtime,
// so ordinals are relative to each rank's visible device set.
int selected_device();
// Stable physical/MIG UUID for the active CUDA device, or an empty string
// while the CPU backend is active. Unlike the ordinal, this is invariant to
// CUDA_VISIBLE_DEVICES remapping and can bind distributed benchmark evidence.
std::string selected_device_identifier();
std::string backend_diagnostic();

dispatch_statistics get_dispatch_statistics() noexcept;
resident_statistics get_resident_statistics() noexcept;
// Current process-wide resident device buffers. This state gauge is a
// separate additive API so resident_statistics keeps its original ABI.
std::uint64_t get_live_resident_device_buffers() noexcept;
field_update_statistics get_field_update_statistics() noexcept;
polarization_statistics get_polarization_statistics() noexcept;
source_statistics get_source_statistics() noexcept;
boundary_statistics get_boundary_statistics() noexcept;
dft_statistics get_dft_statistics() noexcept;
dft_batch_statistics get_dft_batch_statistics() noexcept;
dft_reduction_statistics get_dft_reduction_statistics() noexcept;
dft_materialization_statistics get_dft_materialization_statistics() noexcept;
dft_checkpoint_statistics get_dft_checkpoint_statistics() noexcept;
dft_scale_statistics get_dft_scale_statistics() noexcept;
eigenmode_overlap_statistics get_eigenmode_overlap_statistics() noexcept;
ldos_statistics get_ldos_statistics() noexcept;
near2far_statistics get_near2far_statistics() noexcept;
multi_gpu_statistics get_multi_gpu_statistics() noexcept;
mpi_completion_statistics get_mpi_completion_statistics() noexcept;
boundary_eh_overlap_statistics
get_boundary_eh_overlap_statistics() noexcept;
halo_curl_overlap_statistics get_halo_curl_overlap_statistics() noexcept;
tile_coalescing_statistics get_tile_coalescing_statistics() noexcept;
phase_batch_policy_statistics
get_phase_batch_policy_statistics() noexcept;
curl_phase_replay_statistics
get_curl_phase_replay_statistics() noexcept;
boundary_descriptor_replay_statistics
get_boundary_descriptor_replay_statistics() noexcept;
runtime_touch_statistics get_runtime_touch_statistics() noexcept;
void reset_boundary_descriptor_replay_statistics() noexcept;
void reset_dispatch_statistics() noexcept;

} // namespace gpu
} // namespace meep

#endif
