#ifndef MEEP_GPU_BACKEND_INTERNAL_HPP
#define MEEP_GPU_BACKEND_INTERNAL_HPP

#include "meep/gpu.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <complex>
#include <memory>
#include <string>
#include <vector>

namespace meep {
class fields;
class dft_chunk;
class h5file;
namespace gpu {
namespace detail {

struct resident_cache;
struct boundary_exchange_buffer;
struct boundary_phase_graph;

class boundary_descriptor_replay_token_test_access;

// Non-forgeable-by-default capability for cached descriptor replay. Normal
// gather/scatter callers cannot construct one and therefore retain full
// validation. fields and its private topology are the sole production
// constructor; the named friend is defined only by the integrated regression.
class boundary_descriptor_replay_token {
public:
  boundary_descriptor_replay_token(
      const boundary_descriptor_replay_token &) noexcept = default;
  boundary_descriptor_replay_token &operator=(
      const boundary_descriptor_replay_token &) noexcept = default;

  const void *topology_identity() const noexcept {
    return topology_identity_;
  }
  std::uint64_t topology_generation() const noexcept {
    return topology_generation_;
  }

private:
  boundary_descriptor_replay_token(
      const void *identity, std::uint64_t generation) noexcept
      : topology_identity_(identity), topology_generation_(generation) {}

  const void *topology_identity_;
  std::uint64_t topology_generation_;

  friend class ::meep::fields;
  friend class boundary_descriptor_replay_token_test_access;
};

// CAS-based monotonic allocator shared by the production topology and a
// focused CPU-only regression. It issues UINT64_MAX once, then leaves the
// counter saturated and returns zero forever; zero is never fast-replay
// eligible.
std::uint64_t next_saturating_boundary_topology_generation(
    std::atomic<std::uint64_t> *generation) noexcept;

enum class lorentzian_state_kind { previous, increment };

// Test-only visibility into the opaque state owned by a standard
// Lorentzian/Drude susceptibility.  Keeping this in the internal header
// avoids exposing representation details through meep.hpp while allowing the
// integrated regression to prove the production dispatch/state contract.
struct lorentzian_internal_state_view {
  void *polarization;
  void *auxiliary;
  std::size_t point_count;
  lorentzian_state_kind kind;
};

lorentzian_internal_state_view
lorentzian_internal_state_for_testing(void *internal_data, int component,
                                      int complex_part);

struct backend_preflight_status {
  bool configuration_ok;
  bool cuda_active;
  bool cuda_required;
  bool automatic_requested;
  bool cpu_requested;
  bool explicit_device_requested;
  std::string diagnostic;
};

struct automatic_device_policy_facts {
  int compute_major;
  int compute_minor;
  int multiprocessor_count;
  std::uint64_t memory_bandwidth_bytes_per_second;
  std::uint64_t free_memory_bytes;
  std::uint64_t total_memory_bytes;
};

struct automatic_cuda_policy_input {
  std::uint64_t local_cells;
  std::uint64_t cpu_threads;
  automatic_device_policy_facts device;
  bool explicit_minimum;
  std::uint64_t explicit_minimum_cells;
};

struct automatic_cuda_policy_decision {
  std::uint64_t local_cells;
  std::uint64_t minimum_cells;
  std::uint64_t cpu_threads;
  automatic_device_policy_facts device;
  std::uint64_t estimated_required_memory_bytes;
  std::uint64_t memory_available_after_reserve_bytes;
  bool explicit_minimum;
  bool memory_eligible;
  bool use_cuda;
};

struct finite_check_transfer_statistics {
  std::uint64_t host_to_device_bytes;
  std::uint64_t device_to_host_bytes;
};

struct coalesced_transfer_batch {
  std::size_t begin;
  std::size_t end;
  std::size_t scalar_count;
};

struct curl_partition_validation {
  std::uint64_t full_points;
  std::uint64_t interior_points;
  std::uint64_t shell_points;
  bool interior_exists;
  bool interior_shell_disjoint;
  bool exact_cover;
};

curl_partition_validation validate_curl_partition_for_testing(
    const int lower[3], const int upper[3],
    const bool active_directions[3]);

struct boundary_phase_graph_statistics {
  std::uint64_t creations;
  std::uint64_t launches;
};

struct boundary_receive_pingpong_statistics {
  std::uint64_t secondary_allocations;
  std::uint64_t selections;
  std::uint64_t secondary_selections;
};

// Test-visible accounting for the exceptional lifetime-fence paths used by
// persistent boundary buffers. Production failure-injection counts remain
// zero; the counters make it possible to prove that a submitted CUDA transfer
// was fenced by the device-wide fallback rather than by a later incidental
// synchronous copy.
struct boundary_lifetime_fallback_statistics {
  std::uint64_t device_synchronize_fallbacks;
  std::uint64_t lifetime_uncertain_leaks;
};

boundary_phase_graph_statistics
get_boundary_phase_graph_statistics() noexcept;
boundary_receive_pingpong_statistics
get_boundary_receive_pingpong_statistics() noexcept;
boundary_lifetime_fallback_statistics
get_boundary_lifetime_fallback_statistics() noexcept;
void record_boundary_receive_pingpong_selection(bool secondary) noexcept;
void record_boundary_eh_overlap_check() noexcept;
void record_boundary_eh_overlap_eligible() noexcept;
void record_boundary_eh_overlap_launch(bool electric) noexcept;
void record_boundary_eh_overlap_disabled() noexcept;
void record_boundary_eh_overlap_unsupported_schedule() noexcept;
void record_boundary_eh_overlap_no_remote() noexcept;
void record_boundary_eh_overlap_cold_topology() noexcept;
void record_boundary_eh_overlap_rejected() noexcept;
void record_halo_curl_overlap_check() noexcept;
void record_halo_curl_overlap_eligible() noexcept;
void record_halo_curl_overlap_launch(std::uint64_t full_points,
                                     std::uint64_t interior_points,
                                     std::uint64_t shell_points) noexcept;
void record_halo_curl_overlap_disabled() noexcept;
void record_halo_curl_overlap_unsupported_schedule() noexcept;
void record_halo_curl_overlap_no_remote() noexcept;
void record_halo_curl_overlap_cold_topology() noexcept;
void record_halo_curl_overlap_rejected(bool small) noexcept;
void record_curl_tile_coalescing(std::uint64_t input_tiles) noexcept;
void record_update_eh_tile_coalescing(std::uint64_t input_tiles) noexcept;

// Partition an ordered set of indivisible logical transfers into the fewest
// contiguous MPI-count-safe batches.  Keeping batches contiguous preserves
// the tag-ordered wire layout on both peers.
std::vector<coalesced_transfer_batch> plan_coalesced_transfer_batches(
    const std::size_t *transfer_sizes, std::size_t transfer_count,
    std::size_t maximum_batch_scalars);

// Expand an exact prefix of per-operation CUDA blocks into the compact
// block-to-operation map consumed by phase-wide kernels. Zero-block
// operations are valid; every represented block must still have exactly one
// operation owner.
std::vector<std::uint32_t> plan_phase_block_operation_indices(
    const std::size_t *block_starts, std::size_t operation_count,
    std::size_t total_block_count);

finite_check_transfer_statistics
get_finite_check_transfer_statistics() noexcept;
std::size_t finite_check_descriptor_bytes(
    std::size_t descriptor_count) noexcept;

struct curl_index {
  std::ptrdiff_t field;
  int sigma;
  int sigma_u;
};

struct curl_material_fp32 {
  const float *sigma;
  const float *kappa;
  const float *sigma_inverse;
  std::size_t sigma_count;
  float *field_u;
  const float *sigma_u;
  const float *kappa_u;
  const float *sigma_u_inverse;
  std::size_t sigma_u_count;
  float dt;
  const float *conductivity;
  const float *conductivity_inverse;
  float *field_conductivity;
};

struct beta_material_fp32 {
  const float *sigma_inverse;
  std::size_t sigma_count;
  float *field_u;
  const float *sigma_u_inverse;
  std::size_t sigma_u_count;
  const float *conductivity_inverse;
  float *field_conductivity;
};

struct bfast_material_fp32 {
  const float *sigma_inverse;
  std::size_t sigma_count;
  float *field_u;
  const float *sigma_u_inverse;
  std::size_t sigma_u_count;
  const float *conductivity_inverse;
  float *field_conductivity;
};

struct update_eh_index {
  std::ptrdiff_t field;
  int sigma;
};

struct index_space_fp32 {
  std::ptrdiff_t field_start;
  std::size_t extent1;
  std::size_t extent2;
  std::size_t extent3;
  std::ptrdiff_t field_stride1;
  std::ptrdiff_t field_stride2;
  std::ptrdiff_t field_stride3;
  int coefficient_start;
  int coefficient_stride1;
  int coefficient_stride2;
  int coefficient_stride3;
  int coefficient2_start;
  int coefficient2_stride1;
  int coefficient2_stride2;
  int coefficient2_stride3;
};

struct update_eh_material_fp32 {
  const float *inverse_susceptibility;
  const float *offdiagonal1;
  const float *offdiagonal2;
  const float *chi2;
  const float *chi3;
  float *field_w;
  const float *sigma;
  const float *kappa;
  std::size_t sigma_count;
};

struct lorentzian_material_fp32 {
  const float *sigma;
  const float *offdiagonal1;
  const float *offdiagonal2;
  float gamma_inverse;
  float gamma_previous;
  float omega_dt_squared;
  float omega_dt_squared_denominator;
};

struct gyrotropic_material_fp32 {
  const float *sigma;
  int model;
  float inverse[9];
  float gyro[9];
  float diagonal;
  float gamma_previous;
  float omega_dt_squared;
  float precession_half_dt;
  float omega_dt;
  float gamma_dt;
  float alpha_half;
  float drive_dt;
};

struct multilevel_population_channel_fp32 {
  const float *field;
  const float *previous_field;
  const float *polarization;
  std::ptrdiff_t centered_offset1;
  std::ptrdiff_t centered_offset2;
};

struct multilevel_polarization_channel_fp32 {
  float *polarization;
  const float *field;
  const float *sigma;
  index_space_fp32 index_space;
  std::size_t point_count;
  std::ptrdiff_t population_offset1;
  std::ptrdiff_t population_offset2;
  int direction;
};

struct multilevel_transition_fp32 {
  int upper_level;
  int lower_level;
  float population_damping;
  float diagonal;
  float gamma_previous;
  float gamma_inverse;
  float drive_scale[5];
};

struct indexed_value_fp32 {
  std::ptrdiff_t index;
  float value;
};

struct complex_value_fp32 {
  float real;
  float imag;
};

struct cw_field_vector_segment_fp32 {
  const void *owner;
  float *field_real;
  float *field_imaginary;
  std::size_t array_count;
  index_space_fp32 index_space;
  std::size_t point_count;
  std::size_t complex_offset;
};

struct resident_cw_vector_plan;

struct boundary_operation_fp32 {
  resident_cache *source_cache;
  resident_cache *destination_cache;
  float *destination_real;
  float *destination_imag;
  const float *source_real;
  const float *source_imag;
  float phase_real;
  float phase_imag;
};

struct remote_boundary_operation_fp32 {
  resident_cache *destination_cache;
  float *destination_real;
  float *destination_imag;
  std::size_t source_offset;
  float phase_real;
  float phase_imag;
};

bool cuda_active();
bool cuda_required();
std::uint64_t backend_generation();
// Never propagates backend-configuration errors. Distributed callers first
// gather this local status so every rank takes the same success/failure path.
backend_preflight_status probe_backend_for_distributed_step();
// Automatic mode deliberately configures only the request syntax at first.
// A fields owner calls this second-stage probe only after its host-only
// workload floor says that CUDA can plausibly win.  This keeps small auto
// simulations completely outside the CUDA runtime.
backend_preflight_status prepare_cuda_candidate_for_distributed_step();
automatic_device_policy_facts selected_automatic_device_policy_facts();
std::uint64_t automatic_cuda_host_only_minimum_cells(
    std::uint64_t cpu_threads) noexcept;
automatic_cuda_policy_decision evaluate_automatic_cuda_policy(
    const automatic_cuda_policy_input &input);
// Selects the execution backend for one already-preflighted fields owner.
// Unlike set_backend(), this does not change the process request, invalidate
// another owner's resident cache, release an MPI device claim, or bump the
// configuration generation.
void activate_backend_for_owner(backend_mode execution,
                                const std::string &diagnostic);
// Collective validation for a distributed fields preflight. Generic backend
// discovery remains rank-local and never hides an MPI collective.
void validate_distributed_device_assignment();
int choose_rank_device_ordinal(const int *compatible_ordinals,
                               std::size_t compatible_count,
                               int node_rank, bool allow_oversubscription);

void record_cpu_curl(std::size_t points) noexcept;
void record_cpu_update_eh(std::size_t points) noexcept;
void record_cpu_polarization(std::size_t points) noexcept;
void record_cpu_source(std::size_t points) noexcept;
void record_cuda_source(std::size_t points) noexcept;
void record_cpu_boundary(std::size_t points) noexcept;
void record_cpu_dft(std::size_t points) noexcept;
void record_cpu_ldos(std::size_t points) noexcept;
void record_cpu_near2far(std::uint64_t terms) noexcept;
void record_cpu_near2far_adjoint(std::uint64_t terms) noexcept;

class resident_curl_session {
public:
  resident_curl_session(const void *owner, bool enabled);
  ~resident_curl_session();

  resident_curl_session(const resident_curl_session &) = delete;
  resident_curl_session &operator=(const resident_curl_session &) = delete;
  resident_curl_session(resident_curl_session &&other) noexcept;
  resident_curl_session &operator=(resident_curl_session &&) = delete;

  bool active() const noexcept { return active_; }
  resident_cache *cache() const noexcept { return cache_; }
  // synchronize_to_host=false ends only this outer resident phase and keeps
  // device-dirty mirrors authoritative for subsequent resident phases.
  void finish(bool synchronize_to_host = true);

private:
  resident_cache *cache_;
  bool active_;
  bool owns_phase_;
};

// Cross-operation curl and E/H kernels use a topology-aware automatic policy
// by default. An unset enable variable requests automatic selection, exact
// "1" forces the batched kernel for qualification, any other present value
// disables it, and any matching disable-variable presence wins. Indexed
// source batching remains experimental/forced-only because repeated evidence
// has not established a favorable crossover.
enum class phase_batch_mode { disabled, automatic, forced };
phase_batch_mode phase_batched_curl_mode() noexcept;
phase_batch_mode phase_batched_update_eh_mode() noexcept;
bool phase_batched_source_opted_in() noexcept;

// Pure operation-topology policy used by production and host-only tests.
// Logical blocks are the sum of ceil(point_count / 256) over every operation;
// the maximum argument is the largest such per-operation count so a skewed
// phase cannot hide one oversized launch behind several tiny operations.
bool automatic_phase_batch_selected(std::size_t operation_count,
                                    std::size_t logical_block_count,
                                    std::size_t maximum_operation_block_count,
                                    int multiprocessor_count) noexcept;

// Collects independent Cartesian curl launches across rank-local chunks and
// emits one phase-wide kernel. Unsupported dependent corrections flush the
// pending batch before using their existing specialized launch paths.
class resident_curl_phase_batch {
public:
  resident_curl_phase_batch(const void *owner, int phase_key,
                            phase_batch_mode mode,
                            std::uint64_t topology_fingerprint,
                            bool replay_supported);
  ~resident_curl_phase_batch();

  resident_curl_phase_batch(const resident_curl_phase_batch &) = delete;
  resident_curl_phase_batch &operator=(
      const resident_curl_phase_batch &) = delete;

  // Replays a stable device-address plan after validating every owning
  // resident-cache generation. Returns false until a phase has established
  // the same topology repeatedly or after any cache/host-write invalidation.
  bool replay_if_ready();
  void finish();
  bool active() const noexcept { return active_; }

private:
  bool active_;
};

// Test-only visibility into the replay transaction. Production callers do
// not inspect or mutate plans through this interface.
struct curl_phase_replay_plan_snapshot {
  bool exists;
  bool complete_phase;
  std::size_t stable_reuses;
};
curl_phase_replay_plan_snapshot
get_curl_phase_replay_plan_snapshot_for_testing(
    const void *owner, int phase_key) noexcept;
// Injects before the cached replay kernel is submitted. Production leaves
// this count at zero; reset_dispatch_statistics also clears it.
void set_curl_phase_replay_launch_failures_for_testing(
    std::uint64_t count) noexcept;

void destroy_resident_curl_phase_batches_for_owner(
    const void *owner) noexcept;

class resident_update_eh_phase_batch {
public:
  resident_update_eh_phase_batch(const void *owner, int phase_key,
                                 phase_batch_mode mode);
  ~resident_update_eh_phase_batch();

  resident_update_eh_phase_batch(
      const resident_update_eh_phase_batch &) = delete;
  resident_update_eh_phase_batch &operator=(
      const resident_update_eh_phase_batch &) = delete;

  void finish();
  bool active() const noexcept { return active_; }

private:
  bool active_;
};

void destroy_resident_update_eh_phase_batches_for_owner(
    const void *owner) noexcept;

class resident_source_phase_batch {
public:
  resident_source_phase_batch(const void *owner, int phase_key,
                              bool enabled);
  ~resident_source_phase_batch();

  resident_source_phase_batch(const resident_source_phase_batch &) = delete;
  resident_source_phase_batch &operator=(
      const resident_source_phase_batch &) = delete;

  void finish();
  bool active() const noexcept { return active_; }

private:
  bool active_;
};

void destroy_resident_source_phase_batches_for_owner(
    const void *owner) noexcept;

// True only while an outer resident session owns the cache. Phase-wide
// launch collection must not outlive a chunk-local session: otherwise that
// session can publish pre-launch device state to the host before the deferred
// kernel is submitted.
bool resident_phase_is_active_for_owner(const void *owner) noexcept;

// Creates a stable device descriptor plan for the exact host ordering used
// by solve_cw's packed interleaved-complex vector. Every owner must have an
// active outer resident phase for the lifetime of the plan, preventing mirror
// address invalidation between Krylov operator applications.
resident_cw_vector_plan *create_resident_cw_vector_plan(
    const cw_field_vector_segment_fp32 *segments,
    std::size_t segment_count, std::size_t complex_count);
void destroy_resident_cw_vector_plan(
    resident_cw_vector_plan *plan) noexcept;
void gather_resident_cw_vector_fp32(
    resident_cw_vector_plan *plan, float *device_packed_real_imag);
void scatter_resident_cw_vector_fp32(
    resident_cw_vector_plan *plan,
    const float *device_packed_real_imag);
void gather_resident_cw_field_operator_fp32(
    resident_cw_vector_plan *plan,
    const float *device_input_real_imag,
    float *device_output_real_imag, float dt_inverse,
    complex_value_fp32 iomega);

// Accumulates finite checks from every rank-local resident cache into one
// device result.  finish(false) leaves the cumulative result device-resident
// so a later session for the same owner can append scans before readback.
class resident_finite_check_session {
public:
  explicit resident_finite_check_session(resident_cache *result_owner);
  ~resident_finite_check_session();

  resident_finite_check_session(
      const resident_finite_check_session &) = delete;
  resident_finite_check_session &operator=(
      const resident_finite_check_session &) = delete;

  void accumulate(resident_cache *cache, float *const *host_arrays,
                  const std::size_t *array_counts,
                  std::size_t array_count);
  bool finish(bool readback = true);
  bool active() const noexcept { return active_; }

private:
  void initialize_result();

  resident_cache *result_owner_;
  bool active_;
  bool initialized_;
  std::uint32_t generation_;
};

// Test-only snapshot used to prove that rejected staging metadata does not
// create a cache or advance a resident epoch before pure validation finishes.
struct resident_cache_snapshot_for_testing {
  bool exists;
  bool phase_active;
  int device_ordinal;
  std::uint64_t epoch;
  std::size_t mirror_count;
};

resident_cache_snapshot_for_testing get_resident_cache_snapshot_for_testing(
    const void *owner) noexcept;

void destroy_resident_cache_for_owner(const void *owner) noexcept;
// Destruction-only variant: the owner and its host arrays are about to be
// deleted, so publishing device-authoritative mirrors would be wasted work.
void discard_resident_cache_for_owner(const void *owner) noexcept;
void sync_resident_cache_for_owner(const void *owner);
// Transactionally validate and invalidate every resident mirror containing a
// host pointer before an in-place host algorithm writes those arrays.  No
// mirror is invalidated when any requested mirror is still device
// authoritative or the owner has an active phase.  Missing mirrors are safe:
// their first later use uploads the host allocation normally.
void prepare_resident_host_writes_for_owner(
    const void *owner, const void *const *host_pointers,
    std::size_t pointer_count);
// Reset host-pointer topology after publishing device-authoritative values,
// but retain field-sized CUDA allocations for a same-shape Krylov iteration.
void reset_resident_cache_for_owner_reusing_allocations(const void *owner);
// Read only device-authoritative FP32 field scalars without publishing the
// resident mirror to its host array.  Bit 0/1 of the return value indicate
// that real/imaginary_value, respectively, were copied from a dirty mirror.
// A clean or absent mirror leaves the corresponding output untouched so the
// caller can use its ordinary host read.
unsigned int read_resident_field_point_fp32_for_owner(
    const void *owner, const float *real_host_base,
    const float *imaginary_host_base, std::size_t index,
    float *real_value, float *imaginary_value);
void discard_resident_mirror_for_owner(const void *owner,
                                       const void *host_pointer) noexcept;
struct resident_range_readback {
  bool resident;
  std::size_t device_to_host_bytes;
};
// Stage exactly one resident FP32 range without publishing any other mirror
// and without clearing device authority.  A clean resident range is copied
// from its already-current host allocation and reports zero D2H bytes.
resident_range_readback stage_resident_range_fp32_for_owner(
    const void *owner, const float *host_pointer, std::size_t count,
    float *destination);
struct resident_checkpoint_range_fp32 {
  const void *owner;
  const float *host_pointer;
  std::size_t count;
};
struct resident_checkpoint_traffic {
  std::size_t dirty_resident_bytes;
  std::size_t staged_dirty_union_bytes;
  std::size_t full_cache_device_to_host_bytes_avoided;
};
// Computes checkpoint-level traffic with unique-byte semantics.  For each
// unique owner, dirty_resident_bytes counts every device-authoritative mirror
// once; staged_dirty_union_bytes is the interval union of only dirty DFT
// ranges in this dataset; avoided is their difference.  No authority or
// mirror state is changed.
resident_checkpoint_traffic summarize_resident_checkpoint_ranges_fp32(
    const resident_checkpoint_range_fp32 *ranges,
    std::size_t range_count);
struct resident_range_upload {
  bool resident;
  std::size_t host_to_device_bytes;
};
// Upload an HDF5-restored range into the owner's selected-device cache,
// creating that cache when loading a monitor before its first time step.
// Zero values are a successful resident no-op.  A failed transfer removes
// the just-created mirror before propagating the exception, so callers never
// observe an undefined or partially copied mirror.
resident_range_upload upload_resident_range_fp32_for_owner(
    const void *owner, float *host_pointer, std::size_t count);
struct resident_dft_scale_result {
  std::size_t host_to_device_bytes;
  std::size_t kernel_launches;
};
// Makes a complex DFT allocation resident, applies a finite complex scale in
// place, retains device authority, and invalidates every retained scientific
// plan whose result may depend on the old contents.  count is complex values;
// zero is a successful no-op.
resident_dft_scale_result scale_resident_complex_dft_fp32_for_owner(
    const void *owner, float *host_real_imag, std::size_t count,
    double scale_real, double scale_imaginary);
// Source replacement mutates only the immutable index/amplitude inputs of an
// indexed-source kernel. Drop those mirrors and invalidate source/LDOS
// descriptor snapshots that retain them, without publishing or invalidating
// unrelated device-authoritative fields and address-bearing curl/E/H/boundary
// plans.
void discard_resident_source_profile_for_owner(
    const void *owner, const std::ptrdiff_t *indices,
    const float *amplitudes);
void destroy_boundary_exchange_for_owner(const void *owner) noexcept;

boundary_exchange_buffer *resident_boundary_exchange_buffer(
    const void *owner, const void *token, std::size_t scalar_count);
boundary_exchange_buffer *resident_boundary_exchange_buffer_slot(
    const void *owner, const void *token, std::size_t scalar_count,
    std::size_t slot);
float *boundary_exchange_host_data(boundary_exchange_buffer *buffer);
float *boundary_exchange_device_data(boundary_exchange_buffer *buffer);
std::size_t boundary_exchange_scalar_count(
    const boundary_exchange_buffer *buffer) noexcept;
void resident_gather_boundary_fp32(
    resident_cache *const *source_caches,
    boundary_exchange_buffer *buffer,
    const float *const *source_pointers, std::size_t scalar_count,
    const boundary_descriptor_replay_token *replay_token = nullptr);
void boundary_exchange_copy_to_host(boundary_exchange_buffer *buffer);
void boundary_exchange_copy_to_device(boundary_exchange_buffer *buffer);
void boundary_exchange_synchronize(boundary_exchange_buffer *buffer);
// Test-only failure injection after CUDA boundary work submission and before
// its completion event is recorded. Production code leaves this count zero.
void set_boundary_event_record_failures_for_testing(
    std::uint64_t count) noexcept;
void set_boundary_event_synchronize_failures_for_testing(
    std::uint64_t count) noexcept;
void set_boundary_device_synchronize_failures_for_testing(
    std::uint64_t count) noexcept;
// Test-only failure injection after the resident device is selected inside
// the CUDA Near2Far path. Production code leaves this count zero.
void set_near2far_execution_failures_for_testing(
    std::uint64_t count) noexcept;
boundary_phase_graph *resident_create_boundary_phase_graph(
    const void *owner, std::size_t zero_plan_token,
    std::size_t copy_plan_token,
    boundary_exchange_buffer *const *send_buffers,
    std::size_t send_buffer_count) noexcept;
bool resident_boundary_phase_graph_ready(
    const void *owner, const boundary_phase_graph *graph) noexcept;
bool resident_launch_boundary_phase_graph(
    const void *owner, boundary_phase_graph *graph);
void resident_synchronize_boundary_phase_graph_send_ready(
    boundary_phase_graph *graph);
void destroy_boundary_phase_graph(boundary_phase_graph *graph) noexcept;
void resident_scatter_boundary_fp32(
    boundary_exchange_buffer *buffer,
    const remote_boundary_operation_fp32 *operations,
    std::size_t operation_count,
    const boundary_descriptor_replay_token *replay_token = nullptr);
void record_mpi_boundary_transfer(std::size_t scalar_count,
                                  bool cuda_aware,
                                  bool receive) noexcept;
void record_mpi_boundary_messages(std::size_t message_count) noexcept;
void record_mpi_completion(bool waitall) noexcept;

void resident_step_curl_material_fp32(
    resident_cache *cache, float *field, const float *g1, const float *g2,
    std::size_t array_count, const index_space_fp32 &index_space,
    std::ptrdiff_t stride1, std::ptrdiff_t stride2, float dtdx,
    const curl_material_fp32 &material);

void resident_step_curl_material_batched_fp32(
    resident_cache *cache, float *field, const float *g1, const float *g2,
    std::size_t array_count, const index_space_fp32 *index_spaces,
    std::size_t index_space_count, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, float dtdx,
    const curl_material_fp32 &material);

void resident_step_beta_fp32(
    resident_cache *cache, float *field, const float *g,
    std::size_t array_count, const index_space_fp32 &index_space,
    float betadt, const beta_material_fp32 &material);

void resident_step_bfast_fp32(
    resident_cache *cache, float *field, const float *g1, const float *g2,
    float *bfast_field, std::size_t array_count,
    const index_space_fp32 &index_space, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, float k1, float k2,
    const bfast_material_fp32 &material);

void resident_step_bfast_batched_fp32(
    resident_cache *cache, float *field, const float *g1, const float *g2,
    float *bfast_field, std::size_t array_count,
    const index_space_fp32 *index_spaces, std::size_t index_space_count,
    std::ptrdiff_t stride1, std::ptrdiff_t stride2, float k1, float k2,
    const bfast_material_fp32 &material);

void resident_step_cylindrical_radial_curl_fp32(
    resident_cache *cache, float *field, const float *radial_operand,
    std::size_t array_count, const index_space_fp32 &index_space,
    std::ptrdiff_t radial_stride, float radial_origin_offset,
    int radial_difference_sign, float dtdx,
    const curl_material_fp32 &material);

void resident_step_cylindrical_imr_fp32(
    resident_cache *cache, float *field, const float *g,
    std::size_t array_count, const index_space_fp32 &index_space,
    int radial_coordinate_start, float coefficient,
    const beta_material_fp32 &material);

void resident_step_cylindrical_axis_fp32(
    resident_cache *cache, float *field, const float *primary,
    const float *secondary, std::size_t array_count,
    const index_space_fp32 &index_space, std::ptrdiff_t neighbor_shift,
    std::ptrdiff_t secondary_offset, float secondary_scale,
    float drive_scale,
    const curl_material_fp32 &material);

void resident_zero_span_fp32(
    resident_cache *cache, float *field, std::size_t array_count,
    std::size_t offset, std::size_t count);

void resident_update_eh_fp32(
    resident_cache *cache, float *field, const float *g, const float *g1,
    const float *g2, std::size_t array_count,
    const index_space_fp32 &index_space, std::ptrdiff_t field_stride,
    std::ptrdiff_t stride1, std::ptrdiff_t stride2,
    const update_eh_material_fp32 &material);

void resident_update_lorentzian_fp32(
    resident_cache *cache, float *polarization,
    float *polarization_auxiliary, const float *field, const float *field1,
    const float *field2, std::size_t array_count,
    const index_space_fp32 &index_space,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, const lorentzian_material_fp32 &material,
    lorentzian_state_kind state_kind);

void resident_update_gyrotropic_fp32(
    resident_cache *cache, float *polarization0, float *polarization1,
    float *polarization2, float *previous0, float *previous1,
    float *previous2, const float *field0, const float *field1,
    const float *field2, std::size_t array_count,
    const index_space_fp32 &index_space, std::ptrdiff_t field_stride,
    std::ptrdiff_t stride1, std::ptrdiff_t stride2,
    const gyrotropic_material_fp32 &material);

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
    const multilevel_transition_fp32 *transitions);
// Deterministic exception-safety hook used only by the CUDA regression suite.
// The next new multilevel descriptor plan throws after all raw device buffers
// are allocated and populated but before it is committed to the cache.
void fail_next_multilevel_plan_commit_for_testing() noexcept;

void resident_copy_fp32(resident_cache *cache, float *destination,
                        const float *source, std::size_t count);
void resident_subtract_fp32(resident_cache *cache, float *destination,
                            const float *source, std::size_t count);
void resident_indexed_subtract_fp32(
    resident_cache *cache, float *destination, std::size_t array_count,
    const indexed_value_fp32 *updates, std::size_t update_count);
// Returns true when the update was collected into an active source phase
// batch and false when it launched immediately.
bool resident_indexed_source_subtract_fp32(
    resident_cache *cache, float *destination, std::size_t array_count,
    const std::ptrdiff_t *indices, const float *amplitudes_real_imag,
    std::size_t source_count, const float *conductivity_inverse,
    complex_value_fp32 time_scale, bool imaginary_component);
void resident_ensure_mirror_fp32(resident_cache *cache, float *host_pointer,
                                 std::size_t count);
// Prepares a span for boundary access while reusing a larger resident mirror
// that already contains it. This is required by multilevel transition
// subarrays, whose physical state is mirrored as one contiguous allocation.
void resident_ensure_span_fp32(resident_cache *cache, float *host_pointer,
                               std::size_t count);
bool resident_boundary_plan_ready(const void *owner,
                                  std::size_t plan_token) noexcept;
bool resident_launch_boundary_plan(const void *owner,
                                   std::size_t plan_token);
void resident_apply_boundary_fp32(
    const void *owner, std::size_t plan_token,
    const boundary_operation_fp32 *operations, std::size_t operation_count);
bool resident_primary_fields_are_finite_fp32(
    resident_cache *cache, float *const *host_arrays,
    const std::size_t *array_counts, std::size_t array_count);
void resident_update_dft_fp32(
    resident_cache *cache, float *dft_real_imag, float *field_real,
    float *field_imag, std::size_t field_array_count,
    const std::ptrdiff_t *field_indices, const float *weights,
    std::size_t point_count, const double *angular_frequencies,
    std::size_t frequency_count, double time, double scale_real,
    double scale_imag,
    std::ptrdiff_t average_offset1, std::ptrdiff_t average_offset2);

// One logical DFT accumulation submitted to a resident-cache batch.  The
// request and every host descriptor it references need live only until
// resident_update_dft_batch_fp32 returns; the backend prepares all device
// pointers and scratch storage before launching the first operation.
struct dft_update_request_fp32 {
  resident_cache *cache;
  float *dft_real_imag;
  float *field_real;
  float *field_imag;
  std::size_t field_array_count;
  const std::ptrdiff_t *field_indices;
  const float *weights;
  std::size_t point_count;
  const double *angular_frequencies;
  std::size_t frequency_count;
  double time;
  double scale_real;
  double scale_imag;
  std::ptrdiff_t average_offset1;
  std::ptrdiff_t average_offset2;
};

using dft_batch_statistics = ::meep::gpu::dft_batch_statistics;

phase_batch_mode dft_multi_monitor_batch_mode() noexcept;
bool automatic_dft_multi_monitor_batch_selected(
    std::size_t operation_count) noexcept;

void resident_update_dft_batch_fp32(
    const dft_update_request_fp32 *requests, std::size_t request_count);
dft_batch_statistics get_dft_batch_statistics() noexcept;

// One rank-local DFT chunk contributing to a dense get_dft_array result.
// destination_indices and point_weights are call-scoped host metadata; the
// backend validates and stages them before launching. dft_real_imag remains
// owned by the fields_chunk resident cache and is never published to its host
// allocation by this operation.
struct dft_materialization_request_fp32 {
  const void *owner;
  const float *dft_real_imag;
  const std::ptrdiff_t *destination_indices;
  const float *point_weights;
  std::size_t storage_point_count;
  std::size_t point_count;
  std::size_t source_frequency_count;
  std::size_t source_frequency_start;
  std::size_t selected_frequency_count;
  float inverse_stored_weight_real;
  float inverse_stored_weight_imaginary;
  // Per-point exact CPU-semantic marker for a zero dV divisor.  Null means
  // every divisor is ordinary; non-null entries must be zero or one.
  const std::uint8_t *zero_divisor_flags;
};

// Materialize all requests directly from their resident FP32 DFT mirrors into
// one frequency-major dense host result. The operation is all-or-throw: an
// active CUDA backend never falls back to a host DFT traversal.
void resident_materialize_dft_array_fp32(
    const dft_materialization_request_fp32 *requests,
    std::size_t request_count, float *host_output_real_imag,
    std::size_t output_point_count,
    std::size_t output_frequency_count);

// Dense-array geometry for CUDA materialization followed by a deterministic
// device collapse. full_dims uses public array order; collapsed entries are
// summed away and inactive slots must be {extent=1, collapsed=0}.
struct dft_materialization_collapse_fp32 {
  std::size_t full_rank;
  std::size_t full_dims[3];
  std::uint8_t collapsed[3];
};

// Materializes the full dense rank-local contribution, collapses degenerate
// dimensions on the device, and transfers only the reduced result. The
// output remains frequency-major/interleaved complex FP32.
void resident_materialize_collapsed_dft_array_fp32(
    const dft_materialization_request_fp32 *requests,
    std::size_t request_count, float *host_output_real_imag,
    std::size_t full_output_point_count,
    std::size_t reduced_output_point_count,
    std::size_t output_frequency_count,
    const dft_materialization_collapse_fp32 &collapse);

// One disjoint packed interval for output_dft's resident planar staging
// backend. destination_indices is a permutation of [0, point_count), and the
// interval begins at output_point_offset. The complete source allocation is
// described by storage_point_count * source_frequency_count complex values.
struct dft_output_staging_request_fp32 {
  const void *owner;
  const float *dft_real_imag;
  const std::ptrdiff_t *destination_indices;
  const float *point_weights;
  const std::uint8_t *zero_divisor_flags;
  std::size_t storage_point_count;
  std::size_t point_count;
  std::size_t source_frequency_count;
  float inverse_stored_weight_real;
  float inverse_stored_weight_imaginary;
  std::size_t output_point_offset;
};

// The view aliases retained pinned storage. lifetime keeps that allocation
// alive even if any owner/dependency cache is reset, scaled, discarded, or
// destroyed. Contents remain stable until a later successful staging call
// reuses the same retained plan; callers must consume or copy the tile before
// making another staging call with the same exact topology. Layout is
// [frequency][real-or-imaginary][packed output point].
struct resident_dft_output_staging_view_fp32 {
  const float *planar;
  std::size_t output_point_count;
  std::size_t frequency_start;
  std::size_t frequency_count;
  std::size_t frequency_capacity;
  std::shared_ptr<const float> lifetime;

  resident_dft_output_staging_view_fp32(
      const float *planar_value, std::size_t point_count_value,
      std::size_t frequency_start_value, std::size_t frequency_count_value,
      std::size_t frequency_capacity_value)
      : planar(planar_value), output_point_count(point_count_value),
        frequency_start(frequency_start_value),
        frequency_count(frequency_count_value),
        frequency_capacity(frequency_capacity_value), lifetime() {}

  resident_dft_output_staging_view_fp32(
      const float *planar_value, std::size_t point_count_value,
      std::size_t frequency_start_value, std::size_t frequency_count_value,
      std::size_t frequency_capacity_value,
      const std::shared_ptr<const float> &lifetime_value)
      : planar(planar_value), output_point_count(point_count_value),
        frequency_start(frequency_start_value),
        frequency_count(frequency_count_value),
        frequency_capacity(frequency_capacity_value),
        lifetime(lifetime_value) {}
};

// Stages one bounded frequency tile from full device-resident DFT mirrors.
// The backend opens one resident phase for every unique owner and verifies
// that all of them resolve to the same selected device. There is no CPU
// fallback and no full-source D2H publication.
resident_dft_output_staging_view_fp32 resident_stage_dft_output_fp32(
    const dft_output_staging_request_fp32 *requests,
    std::size_t request_count, std::size_t output_point_count,
    std::size_t frequency_start, std::size_t selected_frequency_count,
    std::size_t frequency_capacity);

// Deterministic publication-transaction regression hook. Zero fails the next
// staging call after its D2H reaches private scratch but before the published
// pinned view is changed; -1 disables injection.
void set_dft_output_staging_d2h_failure_after_for_testing(
    std::int64_t successful_transfers) noexcept;
// Process-wide gauge for physical pinned publication allocations created by
// CUDA DFT output. It includes active plan slots and retired slots kept alive
// by caller-held views. This is deliberately test-only rather than public ABI.
std::uint64_t get_live_dft_output_pinned_buffers_for_testing() noexcept;

struct dft_pair_reduction_request_fp32 {
  const void *lhs_owner;
  const void *rhs_owner;
  const float *lhs_real_imag;
  const float *rhs_real_imag;
  // The historical host consumers reduce the lhs point count.  Centered
  // Yee-grid monitors can legitimately allocate a larger rhs chunk, so keep
  // the reduction extent separate from both complete resident allocations.
  std::size_t point_count;
  std::size_t lhs_storage_point_count;
  std::size_t rhs_storage_point_count;
  double weight_real;
  double weight_imaginary;
};

struct dft_pair_list_summary {
  std::size_t pair_count;
  std::size_t point_frequency_terms;
};

// With a non-null destination, validates two complete dft_chunk lists and
// appends CUDA request descriptors.  The historical host implementation
// consumes lockstep pairs while both lists remain and uses the lhs extent; a
// null destination preserves that behavior while only counting its work.
dft_pair_list_summary append_dft_pair_reduction_requests(
    std::vector<dft_pair_reduction_request_fp32> *destination,
    const dft_chunk *lhs, const dft_chunk *rhs,
    std::size_t frequency_count,
    std::complex<double> fixed_weight,
    bool use_lhs_extra_weight);

// Computes one interleaved-complex FP64 result per frequency from resident
// point-major/frequency-minor DFT pairs. plan_owner and plan_lane identify a
// retained descriptor/workspace plan; all lanes for an owner are destroyed
// together when the public monitor dies.
void resident_reduce_dft_pairs_fp32(
    const void *plan_owner, unsigned int plan_lane,
    const dft_pair_reduction_request_fp32 *requests,
    std::size_t request_count, std::size_t frequency_count,
    double *result_real_imag);
void destroy_resident_dft_reduction_plans_for_owner(
    const void *plan_owner) noexcept;
void record_cpu_dft_reduction(
    std::size_t pair_count, std::size_t point_frequency_terms) noexcept;
void record_dft_reduction_mpi_allreduce(std::size_t bytes) noexcept;

using dft_reduction_statistics = ::meep::gpu::dft_reduction_statistics;
dft_reduction_statistics get_dft_reduction_statistics() noexcept;

using dft_materialization_statistics =
    ::meep::gpu::dft_materialization_statistics;
dft_materialization_statistics
get_dft_materialization_statistics() noexcept;
using dft_checkpoint_statistics =
    ::meep::gpu::dft_checkpoint_statistics;
dft_checkpoint_statistics get_dft_checkpoint_statistics() noexcept;
using dft_scale_statistics = ::meep::gpu::dft_scale_statistics;
dft_scale_statistics get_dft_scale_statistics() noexcept;
using eigenmode_overlap_statistics =
    ::meep::gpu::eigenmode_overlap_statistics;
eigenmode_overlap_statistics get_eigenmode_overlap_statistics() noexcept;

struct eigenmode_overlap_request_fp32 {
  // owner selects the resident cache/device. dft_real_imag is null only for
  // mode-mode contractions, which deliberately do not materialize DFT data.
  const void *owner;
  const float *dft_real_imag;
  const std::complex<double> *weighted_conjugate_mode;
  const std::complex<double> *mode2_real_imag;
  // Present only for a fused mode-flux + mode-mode request. The first
  // profile keeps mode-flux normalization; this second profile keeps the
  // unmodified quadrature weight used by mode-mode normalization.
  const std::complex<double> *mode_mode_weighted_conjugate_mode;
  const std::uint8_t *zero_normalization_divisors;
  std::size_t point_count;
  std::size_t dft_storage_point_count;
  std::size_t dft_frequency_count;
  std::uint32_t output_index;
  double inverse_stored_weight_real;
  double inverse_stored_weight_imag;
};

enum class eigenmode_overlap_batch_kind : unsigned int {
  mode_flux = 0,
  mode_mode = 1,
  both = 2
};

// Contracts four rank-local eigenmode cross-product channels in one CUDA
// batch. plan_owner/lane retain topology and workspaces across frequencies;
// sampled MPB profiles are refreshed on every call. result_real_imag has
// sixteen doubles (eight complex outputs) in output-channel-major order.
void resident_reduce_eigenmode_overlaps_fp32(
    const void *plan_owner, eigenmode_overlap_batch_kind batch_kind,
    const eigenmode_overlap_request_fp32 *requests,
    std::size_t request_count, std::size_t selected_frequency_index,
    double result_real_imag[16]);
void destroy_resident_eigenmode_overlap_plans_for_owner(
    const void *plan_owner) noexcept;
void record_eigenmode_host_profile_sampling(
    std::size_t points) noexcept;
void record_eigenmode_zero_rank_channels_skipped(
    std::size_t channels) noexcept;
void record_eigenmode_mpi_allreduce(std::size_t bytes) noexcept;
void record_cpu_dft_array_materialization(std::size_t points) noexcept;
void record_cuda_dft_array_materialization(std::size_t points) noexcept;
void record_host_synthetic_material_array(
    std::size_t points) noexcept;
void record_cpu_dft_output_call() noexcept;
void record_cpu_dft_output_points(std::size_t points) noexcept;
void record_cuda_dft_output_call() noexcept;
void record_cuda_dft_output_points(std::size_t points) noexcept;
void record_cpu_dft_overlap_call() noexcept;
void record_cpu_dft_overlap_terms(std::size_t terms) noexcept;
void record_dft_array_mpi_allreduce(std::size_t bytes) noexcept;
void record_cpu_dft_checkpoint_save(std::size_t values) noexcept;
void record_cuda_dft_checkpoint_save(
    std::size_t values, std::size_t device_to_host_bytes,
    std::size_t full_cache_device_to_host_bytes_avoided) noexcept;
void record_cpu_dft_checkpoint_load(std::size_t values) noexcept;
void record_cuda_dft_checkpoint_load(
    std::size_t values, std::size_t host_to_device_bytes) noexcept;
void record_cpu_dft_scale(std::size_t values) noexcept;
void record_cuda_dft_scale(std::size_t values,
                           std::size_t kernel_launches,
                           std::size_t host_to_device_bytes) noexcept;

// Deterministic per-process test hooks.  A value of n injects on the transfer
// after n successful matching transfers; -1 disables the hook.  Tests choose
// the failing MPI rank explicitly, so production paths contain no timing or
// environment-variable race.
void set_dft_checkpoint_d2h_failure_after_for_testing(
    std::int64_t successful_transfers) noexcept;
void set_dft_checkpoint_h2d_failure_after_for_testing(
    std::int64_t successful_transfers) noexcept;

// One source-profile contribution to the LDOS E.J* or H.J* inner product.
// The source indices and interleaved-complex amplitudes are immutable host
// profiles mirrored by the same rank-local cache as the field allocation.
struct ldos_reduction_request_fp32 {
  resident_cache *cache;
  const float *field_real;
  const float *field_imaginary;
  std::size_t field_array_count;
  const std::ptrdiff_t *indices;
  const float *amplitudes_real_imag;
  std::size_t source_count;
  bool magnetic;
};

struct ldos_reduction_statistics {
  std::uint64_t batch_calls;
  std::uint64_t submitted_profiles;
  std::uint64_t source_points;
  std::uint64_t descriptor_uploads;
  std::uint64_t kernel_launches;
  std::uint64_t result_device_to_host_bytes;
  std::uint64_t full_field_device_to_host_bytes_avoided;
};

// Writes Re(EJ), Im(EJ), Re(HJ), Im(HJ) to result. Every request cache must
// belong to an active resident phase on the same rank-local CUDA device.
void resident_reduce_ldos_fp32(
    const ldos_reduction_request_fp32 *requests,
    std::size_t request_count, double result[4]);
ldos_reduction_statistics get_ldos_reduction_statistics() noexcept;

struct near2far_point_fp64 {
  double x;
  double y;
  double z;
};

struct near2far_periodic_copy_fp64 {
  near2far_point_fp64 displacement;
  double phase_real;
  double phase_imaginary;
};

struct near2far_request_fp32 {
  const void *owner;
  float *dft_real_imag;
  const near2far_point_fp64 *source_points;
  std::size_t point_count;
  int direction;
  bool electric;
};

struct near2far_adjoint_request_fp64 {
  const near2far_point_fp64 *source_points;
  const std::complex<double> *amplitudes;
  std::size_t point_count;
  int direction;
  bool electric;
};

enum class near2far_cartesian_dimension : unsigned int {
  two = 2,
  three = 3,
  cylindrical = 4,
};

// Computes rank-local work-major [target][frequency][component] complex
// output. The function owns resident sessions and all call-scoped descriptors
// through the final D2H copy, and never publishes device-authoritative DFT or
// field mirrors to the host.
void resident_near2far_3d_fp32(
    const void *plan_owner, const near2far_request_fp32 *requests,
    std::size_t request_count,
    const near2far_point_fp64 *targets, std::size_t target_count,
    const double *frequencies, std::size_t frequency_count,
    const near2far_periodic_copy_fp64 *periodic_copies,
    std::size_t periodic_copy_count, double eps, double mu,
    std::complex<double> *output,
    double *fast_error_bounds = nullptr,
    bool defer_cancellation_retry = false,
    bool force_mixed_for_cancellation = false,
    bool collective_retry_attempt = false,
    bool *used_mixed_precision = nullptr);
void resident_near2far_cartesian_fp32(
    near2far_cartesian_dimension dimension,
    const void *plan_owner, const near2far_request_fp32 *requests,
    std::size_t request_count,
    const near2far_point_fp64 *targets, std::size_t target_count,
    const double *frequencies, std::size_t frequency_count,
    const near2far_periodic_copy_fp64 *periodic_copies,
    std::size_t periodic_copy_count, double eps, double mu,
    std::complex<double> *output,
    double *fast_error_bounds = nullptr,
    bool defer_cancellation_retry = false,
    bool force_mixed_for_cancellation = false,
    bool collective_retry_attempt = false,
    bool *used_mixed_precision = nullptr,
    double azimuthal_mode = 0.0,
    double greencyl_tolerance = 0.0);

// Computes rank-local source-profile amplitudes for the adjoint Near2Far
// VJP. Requests are concatenated in order and each request is source-major,
// frequency-minor in `output`. The large dJ contraction stays on CUDA and a
// rejected FP32 result is retried with FP64 geometry/arithmetic on CUDA.
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
    bool *used_mixed_precision = nullptr,
    double azimuthal_mode = 0.0,
    double greencyl_tolerance = 0.0);
void destroy_resident_near2far_plan_for_owner(
    const void *plan_owner) noexcept;
// Exchanges the per-process Near2Far device-workspace ceiling and returns its
// previous value. This internal hook exists only so the integrated regression
// can force operation/frequency/target tiling without allocating hundreds of
// megabytes. Production code never calls it and uses the 64 MiB default.
std::size_t exchange_near2far_workspace_ceiling_for_testing(
    std::size_t bytes);
// Rank-local evidence that a Near2Far result entered one MPI allreduce
// batch. bytes is the logical payload contributed by this rank.
void record_near2far_mpi_allreduce(std::size_t bytes) noexcept;
// Executes the production rank-collective cancellation gate over one or more
// [12 signed scalars, conservative error bound] records. Used by the 2-rank
// adversarial regression to prove cancellation that exists only after MPI sum.
bool test_near2far_collective_requires_mixed(
    const double *local_records, double *global_records,
    std::size_t work_count);
// Deterministic exception-safety probes used only by the CUDA regression
// suite. The reentry helper owns the process-long LDOS lock before calling the
// production reduction, so its try-lock rejects before resident mutation and
// without adding a production-path probe. The plan-commit hook throws after an
// inactive descriptor upload but before active-plan publication.
void resident_reduce_ldos_reentrant_for_testing(
    const ldos_reduction_request_fp32 *requests,
    std::size_t request_count, double result[4]);
void fail_next_ldos_plan_commit_for_testing() noexcept;
std::size_t resident_ldos_plan_buffer_count_for_testing(
    const resident_cache *cache) noexcept;
// Returns true and writes result when the DFT allocation has a
// device-authoritative resident mirror. Returns false without modifying
// result when the ordinary host implementation must be used.
bool resident_dft_norm2_fp32_for_owner(
    const void *owner, const float *dft_real_imag,
    std::size_t storage_point_count, std::size_t frequency_count,
    const std::ptrdiff_t *point_indices,
    std::size_t selected_point_count, double *result);
// Deterministic MPI exception-safety hook used only by the CUDA regression
// suite. The next rank-zero fields::dft_norm local reduction throws before
// entering its collective.
void fail_next_dft_norm_on_master_for_testing() noexcept;
bool consume_dft_norm_master_failure_for_testing() noexcept;
void fail_next_cuda_dft_output_after_dataset_for_testing() noexcept;
bool consume_cuda_dft_output_dataset_failure_for_testing() noexcept;

} // namespace detail
} // namespace gpu

struct dft_hdf5_dataset {
  dft_chunk *chunks;
  const char *name;
};
// Save/load one public monitor's related DFT datasets as a transaction.  The
// load validates and reads every dataset before committing any host or
// resident state; save traffic accounting uses one owner/range union across
// the complete dataset set.
void save_dft_hdf5_many(const dft_hdf5_dataset *datasets,
                        std::size_t dataset_count, h5file *file,
                        const char *dprefix = nullptr,
                        bool single_parallel_file = true);
void load_dft_hdf5_many(const dft_hdf5_dataset *datasets,
                        std::size_t dataset_count, h5file *file,
                        const char *dprefix = nullptr,
                        bool single_parallel_file = true);
} // namespace meep

#endif
