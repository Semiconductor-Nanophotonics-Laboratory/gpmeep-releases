#ifndef MEEP_CUDA_RUNTIME_HPP
#define MEEP_CUDA_RUNTIME_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "meep_cuda/detail/index_space.hpp"

namespace meep_cuda {

struct device_info {
  int ordinal;
  int compute_major;
  int compute_minor;
  int multiprocessor_count;
  int max_threads_per_block;
  std::uint64_t global_memory_bytes;
  // Peak JEDEC transfer rate derived from cudaDeviceProp memoryClockRate
  // and memoryBusWidth.  This is a topology fact, not a measured benchmark;
  // automatic policy code uses it only to scale a conservative crossover
  // floor between devices.
  std::uint64_t memory_bandwidth_bytes_per_second;
  std::string name;
};

struct runtime_touch_statistics {
  std::uint64_t availability_probes;
  std::uint64_t device_enumerations;
  std::uint64_t device_selections;
};

struct device_memory_info {
  std::uint64_t free_bytes;
  std::uint64_t total_bytes;
};

// Returns false for a missing device, missing/incompatible driver, or another
// CUDA runtime initialization failure. The diagnostic is always populated on
// failure.
bool runtime_available(std::string *diagnostic = nullptr);

std::vector<device_info> enumerate_devices();
bool device_compatible(const device_info &device) noexcept;
std::string compiled_architectures();
// Stable physical/MIG identity independent of CUDA_VISIBLE_DEVICES ordinal
// remapping.
std::string device_uuid(int ordinal);
void select_device(int ordinal);
// Queries the currently selected CUDA context. Callers use this immediately
// after select_device so an automatic backend can reject a workload before
// allocating or mutating any device-resident field state.
device_memory_info selected_device_memory_info();
runtime_touch_statistics get_runtime_touch_statistics() noexcept;
void reset_runtime_touch_statistics() noexcept;

int runtime_version();
int driver_version();

void *allocate_device_bytes(std::size_t bytes);
void free_device(void *pointer) noexcept;
// Page-locked host buffers used by the portable MPI staging path.
void *allocate_pinned_bytes(std::size_t bytes);
void free_pinned(void *pointer) noexcept;
void copy_to_device(void *destination, const void *source, std::size_t bytes);
// Enqueue a page-locked-host to device copy on Meep's default CUDA stream.
// The caller must keep the host allocation stable until a later stream event
// has completed.
void copy_to_device_async(void *destination, const void *source,
                          std::size_t bytes);
void copy_to_host(void *destination, const void *source, std::size_t bytes);
void synchronize();

// Device-resident FP32 vector primitives used by the CW Krylov solver.  The
// elementwise operations deliberately round every vector result to FP32,
// but accept FP64 recurrence coefficients to preserve the legacy host
// solver's arithmetic (double coefficient/intermediate, one final FP32
// store). Scalar reductions accumulate into FP64 so large systems do not
// lose their convergence signal solely to reduction precision.  All
// pointers are device pointers and operations are enqueued on Meep's default
// CUDA stream.
//
// Dot products and scaled sums of squares use a deterministic two-stage
// reduction.  Callers provide the reusable block-partial workspace so Krylov
// iterations do not allocate device memory or depend on atomic completion
// order.  The result pointer must not overlap the partial workspace.
constexpr std::size_t vector_reduction_partial_capacity = 1024;
void vector_fill_fp32(float *values, std::size_t count, float value,
                      int threads_per_block = 256);
void vector_copy_fp32(float *destination, const float *source,
                      std::size_t count);
void vector_scale_fp32(float *values, std::size_t count, double scale,
                       int threads_per_block = 256);
// Scales an interleaved-complex FP32 allocation in place.  The scale is
// supplied in FP64 so every finite std::complex<double> accepted by Meep's
// public DFT API has a deterministic conversion path; each output component
// is rounded once on its final FP32 store.  count is measured in complex
// values, not scalar floats, and zero count is a successful no-op.
void complex_scale_inplace_fp32(float *values_real_imag,
                                std::size_t count, double scale_real,
                                double scale_imaginary,
                                int threads_per_block = 256);
void vector_xpay_fp32(float *x, const float *y, std::size_t count,
                      double scale, int threads_per_block = 256);
// output = left - scale * output.  This is the recurrent BiCGSTAB-L update
// and avoids a temporary vector or a second launch.
void vector_left_minus_scale_fp32(float *output, const float *left,
                                  std::size_t count, double scale,
                                  int threads_per_block = 256);

void initialize_fp64_result(double *device_result);
void initialize_fp32_result(float *device_result, float value = 0.0f);
void initialize_int_result(int *device_result, int value);
void vector_dot_fp32(const float *x, const float *y, std::size_t count,
                     double *device_partials,
                     std::size_t partial_capacity,
                     double *device_result,
                     int threads_per_block = 256);
void vector_max_abs_fp32(const float *values, std::size_t count,
                         float *device_result,
                         int *device_all_finite = nullptr,
                         int threads_per_block = 256);
void vector_scaled_sum_squares_fp32(const float *values,
                                    std::size_t count, double scale,
                                    double *device_partials,
                                    std::size_t partial_capacity,
                                    double *device_result,
                                    int threads_per_block = 256);

// One complex CW field allocation represented in a packed interleaved
// Krylov vector. index_space selects the owned Yee cells from the real and
// imaginary mirrors. complex_offset is measured in complex values, not FP32
// scalars. Segment descriptors and all nested pointers reside on the device.
struct cw_field_vector_segment_fp32 {
  float *field_real;
  float *field_imaginary;
  index_space_fp32 index_space;
  std::size_t point_count;
  std::size_t complex_offset;
};

void gather_cw_field_vector_fp32(
    const cw_field_vector_segment_fp32 *segments,
    std::size_t segment_count, std::size_t maximum_point_count,
    float *packed_real_imag, int threads_per_block = 256);
void scatter_cw_field_vector_fp32(
    const cw_field_vector_segment_fp32 *segments,
    std::size_t segment_count, std::size_t maximum_point_count,
    const float *packed_real_imag, int threads_per_block = 256);
// Fused operator publication used after a CW timestep:
//   output = (field - input) * dt_inverse + iomega * input
void gather_cw_field_operator_fp32(
    const cw_field_vector_segment_fp32 *segments,
    std::size_t segment_count, std::size_t maximum_point_count,
    const float *input_real_imag, float *output_real_imag,
    float dt_inverse, float iomega_real, float iomega_imaginary,
    int threads_per_block = 256);

// A lightweight, timing-disabled event used to wait for one buffer's last
// default-stream operation without draining unrelated work on the device.
// The handle is intentionally opaque so CUDA headers do not leak into Meep.
void *create_event();
void record_event(void *event);
void synchronize_event(void *event);
void destroy_event(void *event) noexcept;

// FP32 implementation of the most common Meep step_curl specialization:
// no PML, no conductivity, and one or two curl operands. Either g1 or g2 may
// be null, but not both. A g2-only call is normalized exactly like Meep's CPU
// implementation by swapping operands and negating dtdx.
//
// For every j in [0, count), i is indices[j] when indices is non-null and j
// otherwise:
//
//   field[i] -= dtdx * (
//       g1[i + stride1] - g1[i]
//       + (g2 ? g2[i] - g2[i + stride2] : 0))
//
// All pointers refer to device memory. The caller owns the arrays and must
// ensure every shifted index is in range. An indexed launch maps directly to
// Meep's non-contiguous Yee-grid iteration without hard-coding grid shape.
void step_curl_fp32(float *field, const float *g1, const float *g2,
                    const std::ptrdiff_t *indices, std::size_t count,
                    std::ptrdiff_t stride1, std::ptrdiff_t stride2, float dtdx,
                    int threads_per_block = 256);

struct curl_index {
  std::ptrdiff_t field;
  int sigma;
  int sigma_u;
};

struct curl_material_fp32 {
  const float *sigma;
  const float *kappa;
  const float *sigma_inverse;
  float *field_u;
  const float *sigma_u;
  const float *kappa_u;
  const float *sigma_u_inverse;
  float dt;
  const float *conductivity;
  const float *conductivity_inverse;
  float *field_conductivity;
};

// One independently-addressed Cartesian curl operation for a phase-wide
// launch. block_start is the prefix sum of 256-thread blocks for all preceding
// operations; this makes the phase grid exact instead of overlaunching every
// small PML chunk at the largest chunk's dimensions.
struct curl_phase_operation_fp32 {
  float *field;
  const float *g1;
  const float *g2;
  index_space_fp32 inline_space;
  std::size_t point_count;
  std::size_t block_start;
  std::ptrdiff_t stride1;
  std::ptrdiff_t stride2;
  float dtdx;
  curl_material_fp32 material;
};

// General FP32 Meep step_curl implementation, including all combinations of
// split-field PML and material conductivity. The optional arrays and curl
// indices refer to device memory. A negative sigma/sigma_u index is valid only
// when the corresponding PML pointer group is absent.
void step_curl_material_fp32(float *field, const float *g1, const float *g2,
                             const curl_index *indices, std::size_t count,
                             std::ptrdiff_t stride1, std::ptrdiff_t stride2,
                             float dtdx, const curl_material_fp32 &material,
                             int threads_per_block = 256);

void step_curl_material_structured_fp32(
    float *field, const float *g1, const float *g2,
    const index_space_fp32 &index_space, std::size_t count,
    std::ptrdiff_t stride1, std::ptrdiff_t stride2, float dtdx,
    const curl_material_fp32 &material, int threads_per_block = 256);

// Executes many disjoint CPU-style cache tiles in one CUDA launch. `spaces`
// points to device memory and maximum_point_count is the largest extent
// product among the descriptors.
void step_curl_material_batched_structured_fp32(
    float *field, const float *g1, const float *g2,
    const index_space_fp32 *spaces, std::size_t space_count,
    std::size_t maximum_point_count, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, float dtdx,
    const curl_material_fp32 &material, int threads_per_block = 256);

// Executes independent curl operations spanning multiple Meep chunks in one
// kernel launch. operations and block_operation_indices are device pointers;
// block_operation_indices maps each exact 256-thread phase block to an entry
// in operations. All nested pointers are device pointers and operands must
// already be normalized by the caller.
void step_curl_material_phase_batched_fp32(
    const curl_phase_operation_fp32 *operations,
    const std::uint32_t *block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count);

struct beta_material_fp32 {
  const float *sigma_inverse;
  float *field_u;
  const float *sigma_u_inverse;
  const float *conductivity_inverse;
  float *field_conductivity;
};

// Adds Meep's nonzero-beta correction for a 2D exp(i beta z) simulation,
// including split-field PML and material conductivity. All pointers refer to
// device memory. coefficient/coefficient2 in index_space select the PML-f and
// PML-u inverse arrays respectively.
void step_beta_structured_fp32(
    float *field, const float *g, const index_space_fp32 &index_space,
    std::size_t count, float betadt, const beta_material_fp32 &material,
    int threads_per_block = 256);

struct bfast_material_fp32 {
  const float *sigma_inverse;
  float *field_u;
  const float *sigma_u_inverse;
  const float *conductivity_inverse;
  float *field_conductivity;
};

// Applies Meep's fixed-angle broadband auxiliary recurrence after the
// ordinary curl update. Either operand may be null, but not both. A g2-only
// call is normalized with its stride and cross-product coefficient exactly
// like step_bfast. bfast_field is the persistent F auxiliary.
void step_bfast_structured_fp32(
    float *field, const float *g1, const float *g2, float *bfast_field,
    const index_space_fp32 &index_space, std::size_t count,
    std::ptrdiff_t stride1, std::ptrdiff_t stride2, float k1, float k2,
    const bfast_material_fp32 &material, int threads_per_block = 256);

// Batched-tile variant. spaces points to device memory and
// maximum_point_count is the largest descriptor extent product.
void step_bfast_batched_structured_fp32(
    float *field, const float *g1, const float *g2, float *bfast_field,
    const index_space_fp32 *spaces, std::size_t space_count,
    std::size_t maximum_point_count, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, float k1, float k2,
    const bfast_material_fp32 &material, int threads_per_block = 256);

// Cylindrical Z curl: computes the local finite-difference equivalent of
// (1/r)d(r*g)/dr without the CPU implementation's serial radial prefix
// scratch array, then applies the usual PML/conductivity recurrence.
void step_cylindrical_radial_curl_structured_fp32(
    float *field, const float *radial_operand,
    const index_space_fp32 &index_space, std::size_t count,
    std::ptrdiff_t radial_stride, float radial_origin_offset,
    int radial_difference_sign, float dtdx,
    const curl_material_fp32 &material, int threads_per_block = 256);

// Adds the cylindrical i*m/r correction over an owned0 structured space.
// radial_coordinate_start is the doubled radial coordinate of the first
// point; the radial coordinate is index-space axis 2 (coordinate2).
void step_cylindrical_imr_structured_fp32(
    float *field, const float *g, const index_space_fp32 &index_space,
    std::size_t count, int radial_coordinate_start, float coefficient,
    const beta_material_fp32 &material, int threads_per_block = 256);

// Applies the special r=0 cylindrical recurrence. With secondary=null, raw
// drive is drive_scale*primary[i]. Otherwise it is
// drive_scale*(primary[i]-primary[i+neighbor_shift]
//             -secondary_scale*secondary[i]).
void step_cylindrical_axis_structured_fp32(
    float *field, const float *primary, const float *secondary,
    const index_space_fp32 &index_space, std::size_t count,
    std::ptrdiff_t neighbor_shift, std::ptrdiff_t secondary_offset,
    float secondary_scale, float drive_scale,
    const curl_material_fp32 &material,
    int threads_per_block = 256);

void zero_fp32(float *destination, std::size_t count,
               int threads_per_block = 256);

struct update_eh_index {
  std::ptrdiff_t field;
  int sigma;
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
};

struct update_eh_phase_operation_fp32 {
  float *field;
  const float *g;
  const float *g1;
  const float *g2;
  index_space_fp32 index_space;
  std::size_t point_count;
  std::size_t block_start;
  std::ptrdiff_t field_stride;
  std::ptrdiff_t stride1;
  std::ptrdiff_t stride2;
  update_eh_material_fp32 material;
};

// General Cartesian FP32 implementation of Meep's step_update_EDHB,
// including diagonal/anisotropic materials, chi2/chi3 nonlinearity, and PML.
// All arrays and indices refer to device memory.
void update_eh_fp32(float *field, const float *g, const float *g1,
                    const float *g2, const update_eh_index *indices,
                    std::size_t count, std::ptrdiff_t field_stride,
                    std::ptrdiff_t stride1, std::ptrdiff_t stride2,
                    const update_eh_material_fp32 &material,
                    int threads_per_block = 256);

void update_eh_structured_fp32(
    float *field, const float *g, const float *g1, const float *g2,
    const index_space_fp32 &index_space, std::size_t count,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, const update_eh_material_fp32 &material,
    int threads_per_block = 256);

void update_eh_phase_batched_fp32(
    const update_eh_phase_operation_fp32 *operations,
    const std::uint32_t *block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count);

struct lorentzian_material_fp32 {
  const float *sigma;
  const float *offdiagonal1;
  const float *offdiagonal2;
  float gamma_inverse;
  float gamma_previous;
  float omega_dt_squared;
  float omega_dt_squared_denominator;
};

// Cartesian FP32 Lorentzian/Drude polarization update. The explicit indices
// identify the owned Yee points. Optional off-diagonal sigma rows use the
// same stable average and operand normalization as the CPU implementation.
void update_lorentzian_fp32(
    float *polarization, float *previous_polarization, const float *field,
    const float *field1, const float *field2,
    const std::ptrdiff_t *indices, std::size_t count,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, const lorentzian_material_fp32 &material,
    int threads_per_block = 256);

void update_lorentzian_structured_fp32(
    float *polarization, float *previous_polarization, const float *field,
    const float *field1, const float *field2,
    const index_space_fp32 &index_space, std::size_t count,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, const lorentzian_material_fp32 &material,
    int threads_per_block = 256);

// Numerically conditioned form of the same recurrence.  The second state
// array stores P[n] - P[n-1], rather than P[n-1].  These explicit entry points
// keep the legacy public P/Pprevious API above source- and behavior-compatible.
void update_lorentzian_increment_fp32(
    float *polarization, float *polarization_increment, const float *field,
    const float *field1, const float *field2,
    const std::ptrdiff_t *indices, std::size_t count,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, const lorentzian_material_fp32 &material,
    int threads_per_block = 256);

void update_lorentzian_increment_structured_fp32(
    float *polarization, float *polarization_increment, const float *field,
    const float *field1, const float *field2,
    const index_space_fp32 &index_space, std::size_t count,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, const lorentzian_material_fp32 &material,
    int threads_per_block = 256);

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

// Cartesian FP32 gyrotropic Lorentzian, Drude, or saturated (LLG)
// polarization update. The three polarization arrays are ordered as the
// primary Yee direction followed by its two cyclic directions. Optional
// transverse fields use Meep's four-point Yee average. The caller must
// provide readable/writable allocation spans for every index decoded from
// index_space and, for each non-null transverse field, all four neighbor
// indices implied by field_stride and the corresponding transverse stride.
// Meep's resident backend validates these spans before calling this raw API.
void update_gyrotropic_structured_fp32(
    float *polarization0, float *polarization1, float *polarization2,
    float *previous0, float *previous1, float *previous2,
    const float *field0, const float *field1, const float *field2,
    const index_space_fp32 &index_space, std::size_t count,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, const gyrotropic_material_fp32 &material,
    int threads_per_block = 256);

// A field/polarization channel participating in the centered population
// update of a multilevel atom. polarization points to transition zero; each
// transition occupies a current/previous pair of array_count-sized spans.
struct multilevel_population_channel_fp32 {
  const float *field;
  const float *previous_field;
  const float *polarization;
  std::ptrdiff_t centered_offset1;
  std::ptrdiff_t centered_offset2;
};

// One real or imaginary Yee-field channel updated for every transition.
// point_count may differ between field components. population_offset1/2 are
// scalar grid offsets; the kernel applies the level stride.
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

// Advances all populations and transition polarizations of one multilevel
// susceptibility in two launches. All pointers, including channel and
// transition descriptors, refer to device memory and must not alias writable
// outputs except for the documented in-place population/polarization state.
// Descriptor point counts, index spaces, offsets, directions, and transition
// level indices must address the storage supplied by the caller; the raw CUDA
// API cannot validate values stored in device-only descriptor arrays.
// population_scratch contains array_count*level_count floats and is
// implementation scratch rather than persistent physical state.
void update_multilevel_structured_fp32(
    float *population, float *population_scratch, const float *gamma,
    const float *gamma_inverse, const float *alpha,
    std::size_t level_count, std::size_t transition_count,
    std::size_t array_count, float half_dt,
    const index_space_fp32 &centered_space, std::size_t centered_point_count,
    const multilevel_population_channel_fp32 *population_channels,
    std::size_t population_channel_count,
    const multilevel_polarization_channel_fp32 *polarization_channels,
    std::size_t polarization_channel_count,
    const multilevel_transition_fp32 *transitions,
    std::size_t maximum_polarization_point_count,
    int threads_per_block = 256);

struct indexed_value_fp32 {
  std::ptrdiff_t index;
  float value;
};

// Generic resident-field primitives used by source and polarization phases.
// copy/subtract accept full contiguous device arrays. indexed_subtract uses
// atomic subtraction so repeated source indices remain correct.
void copy_fp32(float *destination, const float *source, std::size_t count,
               int threads_per_block = 256);
void subtract_fp32(float *destination, const float *source, std::size_t count,
                   int threads_per_block = 256);
void indexed_subtract_fp32(float *destination,
                           const indexed_value_fp32 *updates,
                           std::size_t count,
                           int threads_per_block = 256);

struct boundary_operation_fp32 {
  float *destination_real;
  float *destination_imag;
  const float *source_real;
  const float *source_imag;
  float phase_real;
  float phase_imag;
};

// Applies local chunk-boundary copies directly between resident arrays.
// A null source zeros the destination (metal boundary). Scalar copy/negate
// uses only the real pointers; complex Bloch copies use both pointer pairs.
void apply_boundary_fp32(const boundary_operation_fp32 *operations,
                         float *staging_real_imag, std::size_t count,
                         int threads_per_block = 256);

// Direct one-kernel path for a validated all-null-source metal-boundary plan.
// This avoids the staging gather required by potentially aliasing copies.
void zero_boundary_fp32(const boundary_operation_fp32 *operations,
                        std::size_t count, int threads_per_block = 256);

// One-kernel boundary copy for plans whose source and destination storage is
// known not to alias (for example, field <-> MPI exchange buffers).
void apply_nonalias_boundary_fp32(
    const boundary_operation_fp32 *operations, std::size_t count,
    int threads_per_block = 256);

// Reusable CUDA-graph form of the same ordered gather/scatter operation.
// The caller owns the operation and staging allocations and must keep both
// alive until the graph is destroyed.  Graph launch preserves the global
// gather-before-scatter dependency while reducing repeated host launch cost.
struct boundary_graph_fp32;
boundary_graph_fp32 *create_boundary_graph_fp32(
    const boundary_operation_fp32 *operations, float *staging_real_imag,
    std::size_t count, int threads_per_block = 256);
void launch_boundary_graph_fp32(boundary_graph_fp32 *graph);
void destroy_boundary_graph_fp32(boundary_graph_fp32 *graph) noexcept;

// One stage in a reusable, strictly ordered boundary phase graph. `zero`
// and `nonalias` each add one kernel; `ordered` adds the existing
// gather-then-scatter pair and therefore requires staging_real_imag.  When
// completion_event is non-null, the graph records that caller-owned event
// after the stage, allowing an MPI send buffer to expose its readiness
// without another host-side CUDA submission.
enum class boundary_phase_stage_kind {
  zero,
  nonalias,
  ordered
};

struct boundary_phase_stage_fp32 {
  boundary_phase_stage_kind kind;
  const boundary_operation_fp32 *operations;
  float *staging_real_imag;
  std::size_t count;
  void *completion_event;
};

struct boundary_phase_graph_fp32;
boundary_phase_graph_fp32 *create_boundary_phase_graph_fp32(
    const boundary_phase_stage_fp32 *stages, std::size_t stage_count,
    void *completion_event = nullptr, int threads_per_block = 256);
void launch_boundary_phase_graph_fp32(boundary_phase_graph_fp32 *graph);
void destroy_boundary_phase_graph_fp32(
    boundary_phase_graph_fp32 *graph) noexcept;

struct array_span_fp32 {
  const float *values;
  std::size_t count;
};

// Initializes device_result to 1. The initialization is ordered with
// subsequent finite scans launched through this runtime.
void initialize_finite_result(int *device_result);

// Writes 0 to device_result when any value is non-finite and otherwise leaves
// it unchanged. Initialize the result once before one or more scans to
// accumulate across disjoint span batches. device_result and spans are device
// pointers.
void all_finite_fp32(const array_span_fp32 *spans, std::size_t span_count,
                     std::size_t maximum_span_count, int *device_result,
                     int threads_per_block = 256);

// Generation-token finite scan. The device result is cleared only when it is
// allocated (or the token wraps); each scan writes its nonzero generation on
// failure. A stale generation therefore represents a successful current
// scan, eliminating the per-step result-initialization kernel.
void clear_finite_generation_result(std::uint32_t *device_result);
void all_finite_generation_fp32(
    const array_span_fp32 *spans, std::size_t span_count,
    std::size_t maximum_span_count, std::uint32_t generation,
    std::uint32_t *device_result, int threads_per_block = 256);

// Initializes a device-resident FP64 accumulator to zero. The accumulator is
// FP64 even though the input is FP32 so decay decisions do not lose accuracy
// for large DFT monitors.
void initialize_squared_norm_result(double *device_result);

// Accumulates sum(abs(values)^2) into device_result for interleaved-complex
// FP32 values. Each magnitude-squared term is evaluated with FP32 arithmetic
// to match Meep's complex<realnum> CPU semantics, then promoted into the FP64
// accumulator. When point_indices is null, selected points are contiguous;
// otherwise point_indices maps each selected point to its storage point. All
// pointers are device pointers. Call initialize_squared_norm_result before the
// first accumulation into a result.
void squared_norm_complex_fp32(
    const float *values_real_imag, const std::ptrdiff_t *point_indices,
    std::size_t point_count, std::size_t frequency_count,
    double *device_result, int threads_per_block = 256);

struct complex_value_fp32 {
  float real;
  float imag;
};

struct complex_value_fp64 {
  double real;
  double imag;
};

// One rank-local DFT spectral contraction. Values are point-major and
// frequency-minor interleaved complex FP32 arrays resident on the selected
// device. block_start is the prefix sum of ceil(point_count / 32) for all
// preceding operations. The runtime evaluates
//
//   weight * lhs[point, frequency] * conj(rhs[point, frequency])
//
// with FP32 products and deterministic FP64 reductions.
struct dft_pair_reduction_operation_fp32 {
  const float *lhs_real_imag;
  const float *rhs_real_imag;
  std::size_t point_count;
  std::size_t block_start;
  complex_value_fp32 weight;
};

constexpr std::size_t dft_pair_reduction_point_threads = 32;
constexpr std::size_t dft_pair_reduction_frequency_threads = 8;
constexpr std::size_t dft_pair_reduction_partial_capacity = 1024;

// Reduces all operations to one interleaved-complex FP64 result per
// frequency. block_operation_indices has total_spatial_block_count entries.
// device_partials contains at least
// 2*frequency_count*partial_capacity doubles; device_result contains at
// least 2*frequency_count doubles. No atomics are used and the two launches
// use fixed reduction order.
void dft_pair_reduce_fp32(
    const dft_pair_reduction_operation_fp32 *operations,
    const std::uint32_t *block_operation_indices,
    std::size_t operation_count, std::size_t total_spatial_block_count,
    std::size_t frequency_count, double *device_partials,
    std::size_t partial_capacity, double *device_result_real_imag);

// One rank-local contribution to the four cross-product integrals used by
// Meep eigenmode decomposition. weighted_conjugate_mode contains the host-MPB
// profile already multiplied by the Yee quadrature weight. Exactly one of
// dft_real_imag and mode2_real_imag is used: the former contracts a resident
// point-major/frequency-minor DFT at selected_frequency_index, while the
// latter contracts two sampled mode profiles. The resident FDTD/DFT payload
// remains FP32, while host-MPB profiles, complex products, block partials,
// and final results remain FP64 to protect near-cutoff normalization and
// cancellation-heavy overlaps.
struct eigenmode_overlap_operation_fp32 {
  const float *dft_real_imag;
  const complex_value_fp64 *weighted_conjugate_mode;
  const complex_value_fp64 *mode2_real_imag;
  // Optional per-point marker for the historical Meep normalization path.
  // A marked point divides a nonzero DFT value by zero before multiplying by
  // the already-zero quadrature profile, preserving CPU IEEE semantics.
  const std::uint8_t *zero_normalization_divisors;
  std::size_t point_count;
  std::size_t dft_frequency_count;
  std::size_t block_start;
  std::uint32_t output_index;
  complex_value_fp64 inverse_stored_weight;
};

constexpr std::size_t eigenmode_overlap_output_count = 4;
constexpr std::size_t eigenmode_overlap_max_output_count = 8;
constexpr std::size_t eigenmode_overlap_threads_per_block = 256;
constexpr std::size_t eigenmode_overlap_partial_capacity = 1024;

// Pure host validation. block_operation_indices must assign every logical
// block to the descriptor interval implied by block_start and point_count.
void validate_eigenmode_overlap_operations_fp32(
    const eigenmode_overlap_operation_fp32 *operations,
    const std::uint32_t *block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count,
    std::size_t selected_frequency_index,
    std::size_t output_count = eigenmode_overlap_output_count);

// Device-pointer execution counterpart. device_partials contains at least
// 2*output_count*partial_capacity doubles and
// device_result_real_imag contains exactly two doubles per output channel.
void eigenmode_overlap_reduce_fp32(
    const eigenmode_overlap_operation_fp32 *operations,
    const std::uint32_t *block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count,
    std::size_t selected_frequency_index, double *device_partials,
    std::size_t partial_capacity, std::size_t output_count,
    double *device_result_real_imag);

struct cartesian_point_fp32 {
  float x;
  float y;
  float z;
};

struct cartesian_point_fp64 {
  double x;
  double y;
  double z;
};

// One resident dft_chunk participating in a three-dimensional near-to-far
// transform. The DFT remains interleaved-complex FP32 and point-major; source
// coordinates are rounded to FP32 at dispatch together with frequencies,
// material scalars, and periodic phases.  The final reduction promotes FP32
// block partials to FP64 so cancellation across monitor chunks remains stable.
struct near2far_operation_fp32 {
  const float *dft_real_imag;
  const cartesian_point_fp32 *source_points;
  std::size_t point_count;
  std::size_t frequency_stride;
  int direction;
  bool electric;
};

// Precomputed nperiods image. Keeping the final coordinate displacement and
// Bloch multiplier together preserves the CPU implementation's semantics and
// avoids expanding every surface sample in device memory.
struct near2far_periodic_copy_fp32 {
  cartesian_point_fp32 displacement;
  complex_value_fp32 phase;
};

// Accuracy-preserving companion descriptors for geometries whose FP32
// coordinate or phase reduction error is not bounded. The DFT payload stays
// resident FP32; only Green geometry/phase and deterministic partials are
// promoted.
struct near2far_operation_mixed_fp32 {
  const float *dft_real_imag;
  const cartesian_point_fp64 *source_points;
  std::size_t point_count;
  std::size_t frequency_stride;
  int direction;
  bool electric;
};

struct near2far_periodic_copy_fp64 {
  cartesian_point_fp64 displacement;
  complex_value_fp64 phase;
};

// Green-function dimensionality for Cartesian Near2Far transforms.  This is
// deliberately distinct from Meep's public ndim enum so the standalone CUDA
// runtime remains independent of libmeep headers.
enum class near2far_cartesian_dimension : unsigned int {
  two = 2,
  three = 3,
  cylindrical = 4,
};

// Deterministic, source-parallel 3D Green transform. The first launch writes
// one fixed-order block partial per operation/target/frequency/partial index;
// the second reduces those partials in ascending operation/partial order.
// No atomics are used. All pointers, including nested operation pointers,
// refer to device memory. `partials` contains
// operation_count*target_count*frequency_count*partial_count*13 floats: the
// twelve signed field channels followed by the sum of each interaction's
// pre-assembly Green arithmetic envelope. This envelope includes term1,
// term2, and term3 before internal channel cancellation plus the absolute
// amplitude/material/periodic-phase products, and therefore upper-bounds the
// L1 arithmetic scale of every individual channel. If any nonzero envelope
// intermediate is subnormal or nonfinite, the evidence is +infinity so the
// host discards the fast result and retries in mixed precision. This keeps
// the normal-FP32 gamma_n proof valid even on devices which flush subnormals.
// `output` contains
// target_count*frequency_count*12 doubles in
// work-major Ex.real, Ex.imag, ..., Hz.real, Hz.imag order and `absolute_l1`
// contains one double per work item. The latter lets the host identify
// destructive FP32 cancellation and rerun the same transform on CUDA in mixed
// precision without publishing the resident DFT payload.
void near2far_3d_fp32(
    const near2far_operation_fp32 *operations,
    std::size_t operation_count, const cartesian_point_fp32 *targets,
    std::size_t target_count, const float *frequencies,
    std::size_t frequency_count, std::size_t frequency_offset,
    const near2far_periodic_copy_fp32 *periodic_copies,
    std::size_t periodic_copy_count, float eps, float mu,
    std::size_t partial_count, float *partials, double *output,
    double *absolute_l1, int threads_per_block = 256);

// Dimension-generic companion used by libmeep. `two` evaluates the outgoing
// 2D Hankel Green tensor while retaining the same deterministic partial and
// FP64-finalization contract as the 3D entry point. For 2D,
// `maximum_kr_input_error` is a conservative host bound on the FP32 kr
// argument. The kernel combines it with its own arithmetic error and
// propagates both the argument perturbation and CUDA special-function error
// through H0/H1 derivatives and the H2 recurrence. The legacy 3D entry point
// remains available as a source-compatible wrapper.
void near2far_cartesian_fp32(
    near2far_cartesian_dimension dimension,
    const near2far_operation_fp32 *operations,
    std::size_t operation_count, const cartesian_point_fp32 *targets,
    std::size_t target_count, const float *frequencies,
    std::size_t frequency_count, std::size_t frequency_offset,
    const near2far_periodic_copy_fp32 *periodic_copies,
    std::size_t periodic_copy_count, float eps, float mu,
    std::size_t partial_count, float *partials, double *output,
    double *absolute_l1, int threads_per_block = 256,
    float maximum_kr_input_error = 0.0f,
    float azimuthal_mode = 0.0f,
    float greencyl_tolerance = 0.0f);

// Mixed-precision CUDA path selected only when a host-side error bound shows
// that FP32 coordinate subtraction or phase reduction is unsafe. Geometry,
// frequency/material arithmetic, Bloch phase, block partials, and output are
// FP64 while the large resident DFT payload remains FP32.
void near2far_3d_mixed_fp32(
    const near2far_operation_mixed_fp32 *operations,
    std::size_t operation_count, const cartesian_point_fp64 *targets,
    std::size_t target_count, const double *frequencies,
    std::size_t frequency_count, std::size_t frequency_offset,
    const near2far_periodic_copy_fp64 *periodic_copies,
    std::size_t periodic_copy_count, double eps, double mu,
    std::size_t partial_count, double *partials, double *output,
    int threads_per_block = 256);

void near2far_cartesian_mixed_fp32(
    near2far_cartesian_dimension dimension,
    const near2far_operation_mixed_fp32 *operations,
    std::size_t operation_count, const cartesian_point_fp64 *targets,
    std::size_t target_count, const double *frequencies,
    std::size_t frequency_count, std::size_t frequency_offset,
    const near2far_periodic_copy_fp64 *periodic_copies,
    std::size_t periodic_copy_count, double eps, double mu,
    std::size_t partial_count, double *partials, double *output,
    int threads_per_block = 256,
    double azimuthal_mode = 0.0,
    double greencyl_tolerance = 0.0);

// One adjoint Near2Far source sample. Unlike the forward transform, the
// adjoint VJP produces an independent complex amplitude for every source
// sample/frequency pair. Folding the surface quadrature, stored monitor
// weight, symmetry multiplicity, electric sign, and cylindrical Jacobian
// correction into `amplitude` keeps the device Green evaluation linear while
// avoiding a second source-profile kernel.
struct near2far_adjoint_source_fp32 {
  cartesian_point_fp32 point;
  complex_value_fp32 amplitude;
  int direction;
  int electric;
};

struct near2far_adjoint_source_fp64 {
  cartesian_point_fp64 point;
  complex_value_fp64 amplitude;
  int direction;
  int electric;
};

// Computes the adjoint Near2Far VJP in source-major/frequency-minor order.
// Each cooperative block owns one output and deterministically reduces all
// target/periodic-copy interactions, including the six-component complex dJ
// contraction. dJ is target-major/frequency-major/component-minor. The fast
// path returns one Green-arithmetic L1 envelope per output so the host can
// reject destructive cancellation and retry on the mixed CUDA path without a
// CPU fallback.
void near2far_adjoint_fp32(
    near2far_cartesian_dimension dimension,
    const near2far_adjoint_source_fp32 *sources,
    std::size_t source_count, const cartesian_point_fp32 *targets,
    std::size_t target_count, const float *frequencies,
    std::size_t frequency_count,
    const near2far_periodic_copy_fp32 *periodic_copies,
    std::size_t periodic_copy_count, const complex_value_fp32 *dJ,
    float eps, float mu, double *output, double *absolute_l1,
    int threads_per_block = 256,
    float maximum_kr_input_error = 0.0f,
    float azimuthal_mode = 0.0f,
    float greencyl_tolerance = 0.0f,
    bool accumulate = false);

void near2far_adjoint_mixed_fp32(
    near2far_cartesian_dimension dimension,
    const near2far_adjoint_source_fp64 *sources,
    std::size_t source_count, const cartesian_point_fp64 *targets,
    std::size_t target_count, const double *frequencies,
    std::size_t frequency_count,
    const near2far_periodic_copy_fp64 *periodic_copies,
    std::size_t periodic_copy_count, const complex_value_fp64 *dJ,
    double eps, double mu, double *output,
    int threads_per_block = 256,
    double azimuthal_mode = 0.0,
    double greencyl_tolerance = 0.0,
    bool accumulate = false);

// One indexed source-profile contribution to the LDOS field/current inner
// product.  block_start is the prefix sum of 256-thread blocks for all
// preceding operations.  All nested pointers refer to device memory.
struct indexed_ldos_operation_fp32 {
  const float *field_real;
  const float *field_imaginary;
  const std::ptrdiff_t *indices;
  const complex_value_fp32 *amplitudes;
  std::size_t point_count;
  std::size_t block_start;
  bool magnetic;
};

constexpr std::size_t indexed_ldos_channel_count = 4;
constexpr std::size_t indexed_ldos_reduction_partial_capacity = 1024;

enum class indexed_ldos_result_mode : unsigned int {
  replace = 0,
  accumulate = 1,
};

// Zeroes four FP64 values holding Re(EJ), Im(EJ), Re(HJ), and Im(HJ).
void initialize_indexed_ldos_result(double *device_result);

// Deterministically reduces every indexed field * conj(source-amplitude)
// contribution in two launches. FP32 products preserve the CUDA field/source
// arithmetic and the fixed-order FP64 reduction removes block-completion-order
// dependence. block_operation_indices maps each logical block to an operation;
// block_partials provides four channels for each physical block. In accumulate
// mode device_result must first be initialized by
// initialize_indexed_ldos_result.
void indexed_ldos_reduce_fp32(
    const indexed_ldos_operation_fp32 *operations,
    const std::uint32_t *block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count,
    double *device_block_partials, std::size_t partial_capacity,
    double *device_result, indexed_ldos_result_mode result_mode);

constexpr std::size_t indexed_source_phase_max_operations = 32;

// One independent source-profile update in a cross-operation launch.
// block_start is the prefix sum of 256-thread blocks for all preceding
// operations. All nested pointers refer to device memory.
struct indexed_source_phase_operation_fp32 {
  float *destination;
  const std::ptrdiff_t *indices;
  const complex_value_fp32 *amplitudes;
  const float *conductivity_inverse;
  std::size_t point_count;
  std::size_t block_start;
  bool imaginary_component;
};

// Per-launch values deliberately live in kernel parameter space so stable
// topology descriptors do not require a synchronous H2D copy every step.
struct indexed_source_phase_time_scales_fp32 {
  complex_value_fp32 values[indexed_source_phase_max_operations];
};

// Applies a static complex source profile with a time-dependent scalar.
// indices, amplitudes, and optional conductivity_inverse are device pointers.
// imaginary_component selects Im(amplitude * time_scale); otherwise Re().
void indexed_source_subtract_fp32(
    float *destination, const std::ptrdiff_t *indices,
    const complex_value_fp32 *amplitudes,
    const float *conductivity_inverse, std::size_t count,
    complex_value_fp32 time_scale, bool imaginary_component,
    int threads_per_block = 256);

// Executes independent indexed source updates in one CUDA launch. The caller
// supplies an exact prefix sum of 256-thread blocks in block_start and a
// device-resident block-to-operation map.
void indexed_source_subtract_phase_batched_fp32(
    const indexed_source_phase_operation_fp32 *operations,
    const std::uint32_t *block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count,
    indexed_source_phase_time_scales_fp32 time_scales);

// Accumulates a DFT for one field component. Each point-frequency pair owns a
// disjoint complex output, so both dimensions execute in parallel without
// atomics. The output allocation must not overlap any input allocation; alias
// semantics are intentionally unsupported.
struct dft_update_operation_fp32 {
  float *dft_real_imag;
  const float *field_real;
  const float *field_imag;
  const std::ptrdiff_t *field_indices;
  const float *weights;
  std::size_t point_count;
  const complex_value_fp32 *phases;
  std::size_t frequency_count;
  std::ptrdiff_t average_offset1;
  std::ptrdiff_t average_offset2;
  std::size_t block_start;
  std::size_t frequency_block_count;
  std::size_t point_block_count;
  std::uint32_t frequency_threads;
  std::uint32_t point_threads;
};

constexpr int dft_batch_threads_per_block_fp32 = 256;

void update_dft_fp32(
    float *dft_real_imag, const float *field_real, const float *field_imag,
    const std::ptrdiff_t *field_indices, const float *weights,
    std::size_t point_count, const complex_value_fp32 *phases,
    std::size_t frequency_count, std::ptrdiff_t average_offset1,
    std::ptrdiff_t average_offset2, int threads_per_block = 256);

// Validates the host descriptor image before it is copied to the device.
// Every block prefix and tile count must be exact for the fixed 256-thread
// launch used below. Device-only descriptors are trusted after this check.
void validate_dft_batch_operations_fp32(
    const dft_update_operation_fp32 *host_operations,
    std::size_t operation_count, std::size_t total_block_count);

// Executes independent DFT monitor updates in one fixed-size launch.
// block_operation_indices is the exact logical-block-to-operation map
// described by each validated operation's contiguous block_start prefix.
// Every output allocation must be disjoint. Callers must validate the exact
// host descriptor image above before copying it to device_operations.
void update_dft_batch_fp32(
    const dft_update_operation_fp32 *device_operations,
    const std::uint32_t *device_block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count);

// Materializes one or more resident DFT chunks into a dense host-publication
// layout without first copying the point-major device cache to the CPU.
// dft_real_imag is point-major/frequency-minor interleaved complex FP32.
// destination_indices maps each local point to an arbitrary spatial output
// point, while point_weights supplies the already-conditioned per-point
// publication weight.  The runtime evaluates
//
//   dft[point, source_frequency_start + frequency]
//     * inverse_stored_weight * point_weights[point]
//
// and writes output[frequency, destination_indices[point]].  All nested
// pointers are host pointers during validation and device pointers during
// execution. storage_point_count describes the source allocation extent;
// point_count is the selected prefix materialized by this descriptor.
struct dft_materialization_operation_fp32 {
  const float *dft_real_imag;
  const std::ptrdiff_t *destination_indices;
  const float *point_weights;
  std::size_t storage_point_count;
  std::size_t point_count;
  std::size_t source_frequency_count;
  std::size_t source_frequency_start;
  std::size_t selected_frequency_count;
  complex_value_fp32 inverse_stored_weight;
  std::size_t block_start;
  // Optional per-point 0/1 marker.  A marked zero sample remains zero; a
  // marked nonzero sample preserves the CPU path's IEEE division by zero.
  const std::uint8_t *zero_divisor_flags;
  // Optional per-point 0/1 publication marker.  Multiple transformed chunks
  // can legitimately map to the same dense output point (closed surfaces and
  // symmetry images).  The host plan marks only the final point in legacy CPU
  // traversal order so the parallel scatter has deterministic last-writer
  // semantics without serializing the entire kernel.
  const std::uint8_t *publication_flags;
};

constexpr int dft_materialization_threads_per_block_fp32 = 256;

// Validates the exact host descriptor image before it is copied to the
// device. Every operation must cover the common output frequency extent, the
// host block-operation map must exactly match the contiguous block prefixes,
// destination indices must be in range, publication flags must select exactly
// the final occurrence of every repeated destination, and all host-side
// publication weights must be finite. Unique destinations may omit flags.
void validate_dft_materialization_operations_fp32(
    const dft_materialization_operation_fp32 *host_operations,
    const std::uint32_t *host_block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count,
    std::size_t output_point_count,
    std::size_t output_frequency_count);

// Zeroes the complete dense output, then executes the validated scatter.
// output_real_imag is selected-frequency-major/spatial-point-minor
// interleaved complex FP32. The zero operation case is supported so an MPI
// rank with no local chunks can still publish a deterministic all-zero array.
// The output allocation must not overlap any source, destination-map,
// point-weight, descriptor, or block-map allocation because zeroing output is
// ordered before the scatter kernel.
void materialize_dft_fp32(
    const dft_materialization_operation_fp32 *device_operations,
    const std::uint32_t *device_block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count,
    float *output_real_imag, std::size_t output_point_count,
    std::size_t output_frequency_count);

// Deterministic collapse layout for a dense DFT array. Dimensions are stored
// in the same slowest-to-fastest order as Meep's public arrays. A collapsed
// dimension is summed away; retained dimensions preserve their order. The
// runtime validates that the full and reduced products exactly match the
// supplied allocation extents before launching.
struct dft_collapse_layout_fp32 {
  std::size_t full_rank;
  std::size_t full_dims[3];
  std::uint8_t collapsed[3];
};

// Collapses a frequency-major/interleaved dense DFT array on the device. One
// thread owns each reduced complex value and visits collapsed coordinates in
// CPU lexicographic order, avoiding atomics and nondeterministic reduction
// order. Source and destination allocations must not overlap.
void collapse_dft_array_fp32(
    const float *full_output_real_imag, float *reduced_output_real_imag,
    std::size_t full_output_point_count,
    std::size_t reduced_output_point_count,
    std::size_t output_frequency_count,
    const dft_collapse_layout_fp32 &layout,
    int threads_per_block = 256);

// Stages resident point-major/interleaved DFT chunks into the planar layout
// used by the batched output_dft path:
//
//   output[tile_frequency][real_or_imaginary][packed_output_point]
//
// Unlike dft_materialization_operation_fp32, every operation owns one
// disjoint, gapless interval of the packed output. destination_indices is a
// chunk-local permutation of [0, point_count), and output_point_offset moves
// that permutation into the operation's interval. This makes every store
// race-free without publication flags. The logical-block plan is sized for
// point_count * frequency_capacity, so the same validated device descriptors
// and block map can be reused for every full or short frequency tile.
struct dft_output_staging_operation_fp32 {
  const float *dft_real_imag;
  const std::ptrdiff_t *destination_indices;
  const float *point_weights;
  const std::uint8_t *zero_divisor_flags;
  std::size_t storage_point_count;
  std::size_t point_count;
  std::size_t source_frequency_count;
  complex_value_fp32 inverse_stored_weight;
  std::size_t output_point_offset;
  std::size_t block_start;
};

constexpr int dft_output_staging_threads_per_block_fp32 = 256;

// Validates stable topology and allocation extents before descriptors are
// copied to the device. Operations must form an exact gapless partition of
// output_point_count, every local destination map must be a permutation, and
// host_block_operation_indices must exactly reproduce the fixed-capacity
// contiguous block prefixes. An all-empty topology is legal for planning and
// collective agreement, but it has no executable dispatch: callers must skip
// stage_dft_output_fp32 when operation_count or output_point_count is zero.
void validate_dft_output_staging_operations_fp32(
    const dft_output_staging_operation_fp32 *host_operations,
    const std::uint32_t *host_block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count,
    std::size_t output_point_count, std::size_t frequency_capacity);

// Validates per-dispatch frequency metadata against every source extent. A
// short final tile is legal, but tile_frequency_count must be in
// [1, frequency_capacity] and the half-open source range must fit every
// operation.
void validate_dft_output_staging_tile_fp32(
    const dft_output_staging_operation_fp32 *host_operations,
    std::size_t operation_count, std::size_t tile_frequency_start,
    std::size_t tile_frequency_count, std::size_t frequency_capacity);

// Zeroes the complete capacity-sized planar output before staging the active
// tile. Thus unused planes of a short final tile are deterministic zeroes and
// no poison from buffer reuse can reach HDF5. The host descriptor image and
// tile must have passed both validators above before their device image is
// dispatched here. output_planar must not overlap the source DFT storage,
// destination/weight/zero-divisor arrays, device_operations, or the block
// map: the capacity buffer is cleared before the kernel reads any of them.
void stage_dft_output_fp32(
    const dft_output_staging_operation_fp32 *device_operations,
    const std::uint32_t *device_block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count,
    float *output_planar, std::size_t output_point_count,
    std::size_t tile_frequency_start, std::size_t tile_frequency_count,
    std::size_t frequency_capacity);

// Constructs one phase vector without launching a DFT update. This allows a
// caller to prepare all distinct monitor phase groups before one batched
// update launch.
void prepare_dft_phases_fp32(
    const double *angular_frequencies, complex_value_fp32 *phases,
    std::size_t frequency_count, double time, double scale_real,
    double scale_imag, int threads_per_block = 256);

// Accumulates a DFT while constructing exp(i * omega * time) * scale on the
// device. angular_frequencies is a device-resident array of doubles. time and
// scale are currently kernel launch values: a future reusable CUDA Graph must
// update those kernel parameters or source time from device-resident timestep
// state, rather than capturing the first launch's time. The output allocation
// must not overlap any input or phase-scratch allocation.
void update_dft_from_omega_fp32(
    float *dft_real_imag, const float *field_real, const float *field_imag,
    const std::ptrdiff_t *field_indices, const float *weights,
    std::size_t point_count, const double *angular_frequencies,
    complex_value_fp32 *phase_scratch, std::size_t frequency_count,
    double time, double scale_real, double scale_imag,
    std::ptrdiff_t average_offset1, std::ptrdiff_t average_offset2,
    int threads_per_block = 256);

} // namespace meep_cuda

#endif
