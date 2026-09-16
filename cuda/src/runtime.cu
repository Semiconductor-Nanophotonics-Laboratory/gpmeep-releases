#include "meep_cuda/runtime.hpp"
#include "meep_cuda/detail/bfast.hpp"
#include "meep_cuda/detail/compiled_architectures.hpp"
#include "meep_cuda/detail/cylindrical.hpp"
#include "meep_cuda/detail/curl.hpp"
#include "meep_cuda/detail/launch.hpp"
#include "meep_cuda/detail/polarization.hpp"
#include "meep_cuda/detail/update_eh.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace meep_cuda {
namespace {

std::atomic<std::uint64_t> availability_probes(0);
std::atomic<std::uint64_t> device_enumerations(0);
std::atomic<std::uint64_t> device_selections(0);

std::runtime_error cuda_exception(cudaError_t error, const char *operation) {
  std::ostringstream message;
  message << operation << " failed: " << cudaGetErrorString(error) << " (CUDA error "
          << static_cast<int>(error) << ")";
  return std::runtime_error(message.str());
}

void check(cudaError_t error, const char *operation) {
  if (error != cudaSuccess) throw cuda_exception(error, operation);
}

struct current_device_cache {
  int ordinal = -1;
  int max_threads_per_block = 0;
  std::size_t max_grid_x = 0;
  std::size_t max_grid_y = 0;
  std::size_t max_grid_z = 0;
  bool compatible = false;
  bool initialized = false;
};

// CUDA device selection is per host thread.  Meep owns the selection for every
// thread that enters this runtime, so cache both the ordinal and immutable
// launch limits instead of querying the driver for every kernel launch.
thread_local current_device_cache launch_device;

void cache_device_properties(int ordinal, const cudaDeviceProp &property) {
  launch_device.ordinal = ordinal;
  launch_device.max_threads_per_block = property.maxThreadsPerBlock;
  launch_device.max_grid_x =
      static_cast<std::size_t>(property.maxGridSize[0]);
  launch_device.max_grid_y =
      static_cast<std::size_t>(property.maxGridSize[1]);
  launch_device.max_grid_z =
      static_cast<std::size_t>(property.maxGridSize[2]);
  launch_device.compatible =
      detail::compute_capability_is_compiled(property.major, property.minor);
  launch_device.initialized = true;
}

void initialize_current_device_cache() {
  int ordinal = 0;
  check(cudaGetDevice(&ordinal), "cudaGetDevice");
  cudaDeviceProp property;
  check(cudaGetDeviceProperties(&property, ordinal), "cudaGetDeviceProperties");
  cache_device_properties(ordinal, property);
}

__global__ void step_curl_fp32_kernel(float *field, const float *g1, const float *g2,
                                      const std::ptrdiff_t *indices, std::size_t count,
                                      std::ptrdiff_t stride1, std::ptrdiff_t stride2,
                                      float dtdx) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);
  for (std::size_t j =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       j < count; j += grid_stride) {
    const std::ptrdiff_t i = indices ? indices[j] : static_cast<std::ptrdiff_t>(j);
    field[i] -= dtdx * detail::curl_term_fp32(g1, g2, i, stride1, stride2);
  }
}

__global__ void
step_curl_material_fp32_kernel(float *field, const float *g1, const float *g2,
                               const curl_index *indices, std::size_t count,
                               std::ptrdiff_t stride1, std::ptrdiff_t stride2,
                               float dtdx, curl_material_fp32 material) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);
  for (std::size_t j =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       j < count; j += grid_stride) {
    const curl_index index =
        indices ? indices[j]
                : curl_index{static_cast<std::ptrdiff_t>(j), -1, -1};
    const bool pml_f = material.sigma != nullptr;
    const bool pml_u = material.sigma_u != nullptr;
    const bool conductivity = material.conductivity != nullptr;
    const detail::curl_coefficients_fp32 coefficients = {
        pml_f,
        pml_u,
        conductivity,
        pml_f ? material.sigma[index.sigma] : 0.0f,
        pml_f ? material.kappa[index.sigma] : 1.0f,
        pml_f ? material.sigma_inverse[index.sigma] : 1.0f,
        pml_u ? material.sigma_u[index.sigma_u] : 0.0f,
        pml_u ? material.kappa_u[index.sigma_u] : 1.0f,
        pml_u ? material.sigma_u_inverse[index.sigma_u] : 1.0f,
        conductivity ? material.conductivity[index.field] : 0.0f,
        conductivity ? material.conductivity_inverse[index.field] : 1.0f,
        material.dt};
    float *field_u =
        pml_u ? material.field_u + index.field : nullptr;
    float *field_conductivity =
        conductivity && pml_f ? material.field_conductivity + index.field : nullptr;
    const float curl =
        detail::curl_term_fp32(g1, g2, index.field, stride1, stride2);
    detail::apply_curl_update_fp32(field[index.field], field_u,
                                   field_conductivity, curl, dtdx,
                                   coefficients);
  }
}

__global__ void step_curl_material_structured_fp32_kernel(
    float *field, const float *g1, const float *g2,
    index_space_fp32 index_space, std::size_t count,
    std::ptrdiff_t stride1, std::ptrdiff_t stride2, float dtdx,
    curl_material_fp32 material) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) *
      static_cast<std::size_t>(blockDim.x);
  for (std::size_t point =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       point < count; point += grid_stride) {
    const detail::decoded_index_fp32 decoded =
        detail::decode_index_space_fp32(index_space, point);
    const bool pml_f = material.sigma != nullptr;
    const bool pml_u = material.sigma_u != nullptr;
    const bool conductivity = material.conductivity != nullptr;
    const detail::curl_coefficients_fp32 coefficients = {
        pml_f,
        pml_u,
        conductivity,
        pml_f ? material.sigma[decoded.coefficient] : 0.0f,
        pml_f ? material.kappa[decoded.coefficient] : 1.0f,
        pml_f ? material.sigma_inverse[decoded.coefficient] : 1.0f,
        pml_u ? material.sigma_u[decoded.coefficient2] : 0.0f,
        pml_u ? material.kappa_u[decoded.coefficient2] : 1.0f,
        pml_u ? material.sigma_u_inverse[decoded.coefficient2] : 1.0f,
        conductivity ? material.conductivity[decoded.field] : 0.0f,
        conductivity ? material.conductivity_inverse[decoded.field] : 1.0f,
        material.dt};
    float *field_u =
        pml_u ? material.field_u + decoded.field : nullptr;
    float *field_conductivity =
        conductivity && pml_f
            ? material.field_conductivity + decoded.field
            : nullptr;
    const float curl =
        detail::curl_term_fp32(g1, g2, decoded.field, stride1, stride2);
    detail::apply_curl_update_fp32(
        field[decoded.field], field_u, field_conductivity, curl, dtdx,
        coefficients);
  }
}

__global__ void step_curl_material_batched_structured_fp32_kernel(
    float *field, const float *g1, const float *g2,
    const index_space_fp32 *spaces, std::size_t space_count,
    std::ptrdiff_t stride1, std::ptrdiff_t stride2, float dtdx,
    curl_material_fp32 material) {
  for (std::size_t space_index = blockIdx.y;
       space_index < space_count; space_index += gridDim.y) {
    const index_space_fp32 index_space = spaces[space_index];
    const std::size_t point_count =
        index_space.extent1 * index_space.extent2 * index_space.extent3;
    const std::size_t grid_stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t point =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         point < point_count; point += grid_stride) {
      const detail::decoded_index_fp32 decoded =
          detail::decode_index_space_fp32(index_space, point);
      const bool pml_f = material.sigma != nullptr;
      const bool pml_u = material.sigma_u != nullptr;
      const bool conductivity = material.conductivity != nullptr;
      const detail::curl_coefficients_fp32 coefficients = {
          pml_f,
          pml_u,
          conductivity,
          pml_f ? material.sigma[decoded.coefficient] : 0.0f,
          pml_f ? material.kappa[decoded.coefficient] : 1.0f,
          pml_f ? material.sigma_inverse[decoded.coefficient] : 1.0f,
          pml_u ? material.sigma_u[decoded.coefficient2] : 0.0f,
          pml_u ? material.kappa_u[decoded.coefficient2] : 1.0f,
          pml_u ? material.sigma_u_inverse[decoded.coefficient2] : 1.0f,
          conductivity ? material.conductivity[decoded.field] : 0.0f,
          conductivity ? material.conductivity_inverse[decoded.field] : 1.0f,
          material.dt};
      float *field_u =
          pml_u ? material.field_u + decoded.field : nullptr;
      float *field_conductivity =
          conductivity && pml_f
              ? material.field_conductivity + decoded.field
              : nullptr;
      const float curl =
          detail::curl_term_fp32(g1, g2, decoded.field, stride1, stride2);
      detail::apply_curl_update_fp32(
          field[decoded.field], field_u, field_conductivity, curl, dtdx,
          coefficients);
    }
  }
}

__global__ void step_curl_material_phase_batched_fp32_kernel(
    const curl_phase_operation_fp32 *operations,
    const std::uint32_t *block_operation_indices,
    std::size_t total_block_count) {
  __shared__ curl_phase_operation_fp32 operation;
  for (std::size_t phase_block = blockIdx.x;
       phase_block < total_block_count;
       phase_block += gridDim.x) {
    if (threadIdx.x == 0)
      operation = operations[block_operation_indices[phase_block]];
    __syncthreads();
    const std::size_t operation_block =
        phase_block - operation.block_start;
    const std::size_t point =
        operation_block * blockDim.x + threadIdx.x;
    if (point < operation.point_count) {
      const index_space_fp32 index_space = operation.inline_space;
      const detail::decoded_index_fp32 decoded =
          detail::decode_index_space_fp32(index_space, point);
      const bool pml_f = operation.material.sigma != nullptr;
      const bool pml_u = operation.material.sigma_u != nullptr;
      const bool conductivity =
          operation.material.conductivity != nullptr;
      const detail::curl_coefficients_fp32 coefficients = {
          pml_f,
          pml_u,
          conductivity,
          pml_f ? operation.material.sigma[decoded.coefficient] : 0.0f,
          pml_f ? operation.material.kappa[decoded.coefficient] : 1.0f,
          pml_f
              ? operation.material.sigma_inverse[decoded.coefficient]
              : 1.0f,
          pml_u ? operation.material.sigma_u[decoded.coefficient2]
                : 0.0f,
          pml_u ? operation.material.kappa_u[decoded.coefficient2]
                : 1.0f,
          pml_u
              ? operation.material.sigma_u_inverse[decoded.coefficient2]
              : 1.0f,
          conductivity
              ? operation.material.conductivity[decoded.field]
              : 0.0f,
          conductivity
              ? operation.material.conductivity_inverse[decoded.field]
              : 1.0f,
          operation.material.dt};
      float *field_u =
          pml_u ? operation.material.field_u + decoded.field : nullptr;
      float *field_conductivity =
          conductivity && pml_f
              ? operation.material.field_conductivity + decoded.field
              : nullptr;
      const float curl = detail::curl_term_fp32(
          operation.g1, operation.g2, decoded.field,
          operation.stride1, operation.stride2);
      detail::apply_curl_update_fp32(
          operation.field[decoded.field], field_u,
          field_conductivity, curl, operation.dtdx, coefficients);
    }
    // A grid-stride iteration may reuse this shared descriptor. Ensure no
    // warp is still consuming it before thread zero publishes the next one.
    // The normal exact-grid case has no next iteration and pays no barrier.
    if (static_cast<std::size_t>(gridDim.x) <
        total_block_count - phase_block)
      __syncthreads();
  }
}

__global__ void step_beta_structured_fp32_kernel(
    float *field, const float *g, index_space_fp32 index_space,
    std::size_t count, float betadt, beta_material_fp32 material) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) *
      static_cast<std::size_t>(blockDim.x);
  for (std::size_t point =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       point < count; point += grid_stride) {
    const detail::decoded_index_fp32 decoded =
        detail::decode_index_space_fp32(index_space, point);
    const bool pml_f = material.sigma_inverse != nullptr;
    const bool pml_u = material.sigma_u_inverse != nullptr;
    const bool conductivity = material.conductivity_inverse != nullptr;

    float delta = betadt * g[decoded.field];
    if (conductivity) delta *= material.conductivity_inverse[decoded.field];
    if (conductivity && pml_f)
      material.field_conductivity[decoded.field] += delta;
    if (pml_f) delta *= material.sigma_inverse[decoded.coefficient];
    if (pml_u) {
      material.field_u[decoded.field] += delta;
      field[decoded.field] +=
          material.sigma_u_inverse[decoded.coefficient2] * delta;
    }
    else
      field[decoded.field] += delta;
  }
}

__device__ void apply_bfast_at(
    float *field, float *bfast_field,
    const detail::decoded_index_fp32 &decoded,
    const detail::bfast_operands_fp32 &operands,
    bfast_material_fp32 material) {
  const bool pml_f = material.sigma_inverse != nullptr;
  const bool pml_u = material.sigma_u_inverse != nullptr;
  const bool conductivity = material.conductivity_inverse != nullptr;
  const detail::bfast_coefficients_fp32 coefficients = {
      pml_f,
      pml_u,
      conductivity,
      pml_f ? material.sigma_inverse[decoded.coefficient] : 1.0f,
      pml_u ? material.sigma_u_inverse[decoded.coefficient2] : 1.0f,
      conductivity
          ? material.conductivity_inverse[decoded.field]
          : 1.0f};
  detail::apply_bfast_update_fp32(
      field[decoded.field],
      pml_u ? material.field_u + decoded.field : nullptr,
      conductivity && pml_f
          ? material.field_conductivity + decoded.field
          : nullptr,
      bfast_field[decoded.field],
      detail::bfast_drive_fp32(operands, decoded.field), coefficients,
      operands.g2 != nullptr);
}

__global__ void step_bfast_structured_fp32_kernel(
    float *field, const float *g1, const float *g2, float *bfast_field,
    index_space_fp32 index_space, std::size_t count,
    std::ptrdiff_t stride1, std::ptrdiff_t stride2, float k1, float k2,
    bfast_material_fp32 material) {
  const detail::bfast_operands_fp32 operands =
      detail::normalize_bfast_operands(
          g1, g2, stride1, stride2, k1, k2);
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) *
      static_cast<std::size_t>(blockDim.x);
  for (std::size_t point =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       point < count; point += grid_stride) {
    const detail::decoded_index_fp32 decoded =
        detail::decode_index_space_fp32(index_space, point);
    apply_bfast_at(
        field, bfast_field, decoded, operands, material);
  }
}

__global__ void step_bfast_batched_structured_fp32_kernel(
    float *field, const float *g1, const float *g2, float *bfast_field,
    const index_space_fp32 *spaces, std::size_t space_count,
    std::ptrdiff_t stride1, std::ptrdiff_t stride2, float k1, float k2,
    bfast_material_fp32 material) {
  const detail::bfast_operands_fp32 operands =
      detail::normalize_bfast_operands(
          g1, g2, stride1, stride2, k1, k2);
  for (std::size_t space_index = blockIdx.y;
       space_index < space_count; space_index += gridDim.y) {
    const index_space_fp32 index_space = spaces[space_index];
    const std::size_t point_count =
        index_space.extent1 * index_space.extent2 *
        index_space.extent3;
    const std::size_t grid_stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t point =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         point < point_count; point += grid_stride) {
      const detail::decoded_index_fp32 decoded =
          detail::decode_index_space_fp32(index_space, point);
      apply_bfast_at(
          field, bfast_field, decoded, operands, material);
    }
  }
}

__device__ detail::curl_coefficients_fp32 curl_coefficients_at(
    const curl_material_fp32 &material,
    const detail::decoded_index_fp32 &decoded) {
  const bool pml_f = material.sigma != nullptr;
  const bool pml_u = material.sigma_u != nullptr;
  const bool conductivity = material.conductivity != nullptr;
  return {
      pml_f,
      pml_u,
      conductivity,
      pml_f ? material.sigma[decoded.coefficient] : 0.0f,
      pml_f ? material.kappa[decoded.coefficient] : 1.0f,
      pml_f ? material.sigma_inverse[decoded.coefficient] : 1.0f,
      pml_u ? material.sigma_u[decoded.coefficient2] : 0.0f,
      pml_u ? material.kappa_u[decoded.coefficient2] : 1.0f,
      pml_u ? material.sigma_u_inverse[decoded.coefficient2] : 1.0f,
      conductivity ? material.conductivity[decoded.field] : 0.0f,
      conductivity ? material.conductivity_inverse[decoded.field] : 1.0f,
      material.dt};
}

__device__ void apply_additive_correction_fp32(
    float *field, const detail::decoded_index_fp32 &decoded, float delta,
    beta_material_fp32 material) {
  const bool pml_f = material.sigma_inverse != nullptr;
  const bool pml_u = material.sigma_u_inverse != nullptr;
  const bool conductivity = material.conductivity_inverse != nullptr;
  detail::apply_additive_correction_fp32(
      field[decoded.field],
      pml_u ? material.field_u + decoded.field : nullptr,
      conductivity && pml_f
          ? material.field_conductivity + decoded.field
          : nullptr,
      delta, pml_f, pml_u, conductivity,
      conductivity
          ? material.conductivity_inverse[decoded.field]
          : 1.0f,
      pml_f ? material.sigma_inverse[decoded.coefficient] : 1.0f,
      pml_u ? material.sigma_u_inverse[decoded.coefficient2] : 1.0f);
}

__global__ void step_cylindrical_radial_curl_structured_fp32_kernel(
    float *field, const float *radial_operand,
    index_space_fp32 index_space, std::size_t count,
    std::ptrdiff_t radial_stride, float radial_origin_offset,
    int radial_difference_sign, float dtdx,
    curl_material_fp32 material) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) *
      static_cast<std::size_t>(blockDim.x);
  for (std::size_t point =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       point < count; point += grid_stride) {
    const detail::decoded_index_fp32 decoded =
        detail::decode_index_space_fp32(index_space, point);
    const float radial_index =
        static_cast<float>(decoded.field / radial_stride) +
        radial_origin_offset;
    const float radial_curl =
        detail::cylindrical_radial_divergence_fp32(
            radial_operand, decoded.field, radial_stride, radial_index,
            radial_difference_sign);
    const detail::curl_coefficients_fp32 coefficients =
        curl_coefficients_at(material, decoded);
    detail::apply_curl_update_fp32(
        field[decoded.field],
        material.sigma_u ? material.field_u + decoded.field : nullptr,
        material.sigma && material.conductivity
            ? material.field_conductivity + decoded.field
            : nullptr,
        radial_curl, dtdx, coefficients);
  }
}

__global__ void step_cylindrical_imr_structured_fp32_kernel(
    float *field, const float *g, index_space_fp32 index_space,
    std::size_t count, int radial_coordinate_start, float coefficient,
    beta_material_fp32 material) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) *
      static_cast<std::size_t>(blockDim.x);
  for (std::size_t point =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       point < count; point += grid_stride) {
    std::size_t reduced = point / index_space.extent3;
    const std::size_t radial_coordinate =
        reduced % index_space.extent2;
    const int doubled_radial_coordinate =
        radial_coordinate_start +
        2 * static_cast<int>(radial_coordinate);
    const detail::decoded_index_fp32 decoded =
        detail::decode_index_space_fp32(index_space, point);
    apply_additive_correction_fp32(
        field, decoded,
        detail::cylindrical_imr_drive_fp32(
            g, decoded.field, doubled_radial_coordinate, coefficient),
        material);
  }
}

__global__ void step_cylindrical_axis_structured_fp32_kernel(
    float *field, const float *primary, const float *secondary,
    index_space_fp32 index_space, std::size_t count,
    std::ptrdiff_t neighbor_shift, std::ptrdiff_t secondary_offset,
    float secondary_scale, float drive_scale,
    curl_material_fp32 material) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) *
      static_cast<std::size_t>(blockDim.x);
  for (std::size_t point =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       point < count; point += grid_stride) {
    const detail::decoded_index_fp32 decoded =
        detail::decode_index_space_fp32(index_space, point);
    const float drive = detail::cylindrical_axis_drive_fp32(
        primary, secondary, decoded.field, neighbor_shift,
        secondary_offset, secondary_scale, drive_scale);
    const detail::curl_coefficients_fp32 coefficients =
        curl_coefficients_at(material, decoded);
    detail::apply_curl_update_fp32(
        field[decoded.field],
        material.sigma_u ? material.field_u + decoded.field : nullptr,
        material.sigma && material.conductivity
            ? material.field_conductivity + decoded.field
            : nullptr,
        drive, -1.0f, coefficients);
  }
}

__global__ void update_eh_fp32_kernel(
    float *field, const float *g, const float *g1, const float *g2,
    const update_eh_index *indices, std::size_t count,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, update_eh_material_fp32 material) {
  const detail::update_eh_operands_fp32 operands =
      detail::normalize_update_eh_operands(
          g1, g2, material.offdiagonal1, material.offdiagonal2, stride1,
          stride2);
  const bool pml = material.sigma != nullptr;
  const bool nonlinear = material.chi3 != nullptr;
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) *
      static_cast<std::size_t>(blockDim.x);
  for (std::size_t j =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       j < count; j += grid_stride) {
    const update_eh_index index =
        indices ? indices[j]
                : update_eh_index{static_cast<std::ptrdiff_t>(j), -1};
    const std::ptrdiff_t i = index.field;
    const float gs = g[i];
    const float inverse_susceptibility =
        material.inverse_susceptibility
            ? material.inverse_susceptibility[i]
            : 1.0f;
    const float offdiagonal1 =
        operands.u1
            ? detail::offdiagonal_average_fp32(
                  operands.u1, operands.g1, i, field_stride,
                  operands.stride1)
            : 0.0f;
    const float offdiagonal2 =
        operands.u2
            ? detail::offdiagonal_average_fp32(
                  operands.u2, operands.g2, i, field_stride,
                  operands.stride2)
            : 0.0f;

    float field_norm_squared = gs * gs;
    if (nonlinear && operands.g1) {
      const float sum =
          operands.g1[i] + operands.g1[i + field_stride] +
          operands.g1[i - operands.stride1] +
          operands.g1[i + field_stride - operands.stride1];
      field_norm_squared += 0.0625f * sum * sum;
    }
    if (nonlinear && operands.g2) {
      const float sum =
          operands.g2[i] + operands.g2[i + field_stride] +
          operands.g2[i - operands.stride2] +
          operands.g2[i + field_stride - operands.stride2];
      field_norm_squared += 0.0625f * sum * sum;
    }

    const detail::update_eh_coefficients_fp32 coefficients = {
        pml,
        nonlinear,
        inverse_susceptibility,
        nonlinear ? material.chi2[i] : 0.0f,
        nonlinear ? material.chi3[i] : 0.0f,
        pml ? material.sigma[index.sigma] : 0.0f,
        pml ? material.kappa[index.sigma] : 1.0f};
    detail::apply_update_eh_fp32(
        field[i], pml ? material.field_w + i : nullptr, gs, offdiagonal1,
        offdiagonal2, field_norm_squared, coefficients);
  }
}

__global__ void update_eh_structured_fp32_kernel(
    float *field, const float *g, const float *g1, const float *g2,
    index_space_fp32 index_space, std::size_t count,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, update_eh_material_fp32 material) {
  const detail::update_eh_operands_fp32 operands =
      detail::normalize_update_eh_operands(
          g1, g2, material.offdiagonal1, material.offdiagonal2, stride1,
          stride2);
  const bool pml = material.sigma != nullptr;
  const bool nonlinear = material.chi3 != nullptr;
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) *
      static_cast<std::size_t>(blockDim.x);
  for (std::size_t point =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       point < count; point += grid_stride) {
    const detail::decoded_index_fp32 decoded =
        detail::decode_index_space_fp32(index_space, point);
    const std::ptrdiff_t i = decoded.field;
    const float gs = g[i];
    const float inverse_susceptibility =
        material.inverse_susceptibility
            ? material.inverse_susceptibility[i]
            : 1.0f;
    const float offdiagonal1 =
        operands.u1
            ? detail::offdiagonal_average_fp32(
                  operands.u1, operands.g1, i, field_stride,
                  operands.stride1)
            : 0.0f;
    const float offdiagonal2 =
        operands.u2
            ? detail::offdiagonal_average_fp32(
                  operands.u2, operands.g2, i, field_stride,
                  operands.stride2)
            : 0.0f;

    float field_norm_squared = gs * gs;
    if (nonlinear && operands.g1) {
      const float sum =
          operands.g1[i] + operands.g1[i + field_stride] +
          operands.g1[i - operands.stride1] +
          operands.g1[i + field_stride - operands.stride1];
      field_norm_squared += 0.0625f * sum * sum;
    }
    if (nonlinear && operands.g2) {
      const float sum =
          operands.g2[i] + operands.g2[i + field_stride] +
          operands.g2[i - operands.stride2] +
          operands.g2[i + field_stride - operands.stride2];
      field_norm_squared += 0.0625f * sum * sum;
    }

    const detail::update_eh_coefficients_fp32 coefficients = {
        pml,
        nonlinear,
        inverse_susceptibility,
        nonlinear ? material.chi2[i] : 0.0f,
        nonlinear ? material.chi3[i] : 0.0f,
        pml ? material.sigma[decoded.coefficient] : 0.0f,
        pml ? material.kappa[decoded.coefficient] : 1.0f};
    detail::apply_update_eh_fp32(
        field[i], pml ? material.field_w + i : nullptr, gs, offdiagonal1,
        offdiagonal2, field_norm_squared, coefficients);
  }
}

__global__ void update_eh_phase_batched_fp32_kernel(
    const update_eh_phase_operation_fp32 *operations,
    const std::uint32_t *block_operation_indices,
    std::size_t total_block_count) {
  __shared__ update_eh_phase_operation_fp32 operation;
  for (std::size_t phase_block = blockIdx.x;
       phase_block < total_block_count;
       phase_block += gridDim.x) {
    if (threadIdx.x == 0)
      operation = operations[block_operation_indices[phase_block]];
    // Every thread consumes the descriptor published by lane zero, including
    // the common exact-grid launch where this loop has only one iteration.
    __syncthreads();
    const std::size_t operation_block =
        phase_block - operation.block_start;
    const std::size_t point =
        operation_block * blockDim.x + threadIdx.x;
    if (point < operation.point_count) {
      const detail::update_eh_operands_fp32 operands =
          detail::normalize_update_eh_operands(
              operation.g1, operation.g2,
              operation.material.offdiagonal1,
              operation.material.offdiagonal2, operation.stride1,
              operation.stride2);
      const bool pml = operation.material.sigma != nullptr;
      const bool nonlinear = operation.material.chi3 != nullptr;
      const detail::decoded_index_fp32 decoded =
          detail::decode_index_space_fp32(operation.index_space, point);
      const std::ptrdiff_t i = decoded.field;
      const float gs = operation.g[i];
      const float inverse_susceptibility =
          operation.material.inverse_susceptibility
              ? operation.material.inverse_susceptibility[i]
              : 1.0f;
      const float offdiagonal1 =
          operands.u1
              ? detail::offdiagonal_average_fp32(
                    operands.u1, operands.g1, i,
                    operation.field_stride, operands.stride1)
              : 0.0f;
      const float offdiagonal2 =
          operands.u2
              ? detail::offdiagonal_average_fp32(
                    operands.u2, operands.g2, i,
                    operation.field_stride, operands.stride2)
              : 0.0f;

      float field_norm_squared = gs * gs;
      if (nonlinear && operands.g1) {
        const float sum =
            operands.g1[i] +
            operands.g1[i + operation.field_stride] +
            operands.g1[i - operands.stride1] +
            operands.g1[i + operation.field_stride - operands.stride1];
        field_norm_squared += 0.0625f * sum * sum;
      }
      if (nonlinear && operands.g2) {
        const float sum =
            operands.g2[i] +
            operands.g2[i + operation.field_stride] +
            operands.g2[i - operands.stride2] +
            operands.g2[i + operation.field_stride - operands.stride2];
        field_norm_squared += 0.0625f * sum * sum;
      }

      const detail::update_eh_coefficients_fp32 coefficients = {
          pml,
          nonlinear,
          inverse_susceptibility,
          nonlinear ? operation.material.chi2[i] : 0.0f,
          nonlinear ? operation.material.chi3[i] : 0.0f,
          pml ? operation.material.sigma[decoded.coefficient] : 0.0f,
          pml ? operation.material.kappa[decoded.coefficient] : 1.0f};
      detail::apply_update_eh_fp32(
          operation.field[i],
          pml ? operation.material.field_w + i : nullptr, gs,
          offdiagonal1, offdiagonal2, field_norm_squared, coefficients);
    }
    if (static_cast<std::size_t>(gridDim.x) <
        total_block_count - phase_block)
      __syncthreads();
  }
}

__global__ void update_lorentzian_fp32_kernel(
    float *polarization, float *previous_polarization, const float *field,
    const float *field1, const float *field2,
    const std::ptrdiff_t *indices, std::size_t count,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, lorentzian_material_fp32 material) {
  const detail::lorentzian_operands_fp32 operands =
      detail::normalize_lorentzian_operands(
          field1, field2, material.offdiagonal1, material.offdiagonal2,
          stride1, stride2);
  const bool anisotropic = operands.sigma1 != nullptr;
  const detail::lorentzian_coefficients_fp32 coefficients = {
      material.gamma_inverse,
      material.gamma_previous,
      material.omega_dt_squared,
      material.omega_dt_squared_denominator};
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) *
      static_cast<std::size_t>(blockDim.x);
  for (std::size_t j =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       j < count; j += grid_stride) {
    const std::ptrdiff_t i =
        indices ? indices[j] : static_cast<std::ptrdiff_t>(j);
    const float offdiagonal1 =
        operands.sigma1
            ? detail::lorentzian_offdiagonal_average_fp32(
                  operands.sigma1, operands.w1, i, field_stride,
                  operands.stride1)
            : 0.0f;
    const float offdiagonal2 =
        operands.sigma2
            ? detail::lorentzian_offdiagonal_average_fp32(
                  operands.sigma2, operands.w2, i, field_stride,
                  operands.stride2)
            : 0.0f;
    detail::apply_lorentzian_update_fp32(
        polarization[i], previous_polarization[i], field[i],
        material.sigma[i], offdiagonal1, offdiagonal2, anisotropic,
        coefficients);
  }
}

__global__ void update_lorentzian_structured_fp32_kernel(
    float *polarization, float *previous_polarization, const float *field,
    const float *field1, const float *field2,
    index_space_fp32 index_space, std::size_t count,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, lorentzian_material_fp32 material) {
  const detail::lorentzian_operands_fp32 operands =
      detail::normalize_lorentzian_operands(
          field1, field2, material.offdiagonal1, material.offdiagonal2,
          stride1, stride2);
  const bool anisotropic = operands.sigma1 != nullptr;
  const detail::lorentzian_coefficients_fp32 coefficients = {
      material.gamma_inverse,
      material.gamma_previous,
      material.omega_dt_squared,
      material.omega_dt_squared_denominator};
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) *
      static_cast<std::size_t>(blockDim.x);
  for (std::size_t point =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       point < count; point += grid_stride) {
    const std::ptrdiff_t i =
        detail::decode_index_space_fp32(index_space, point).field;
    const float offdiagonal1 =
        operands.sigma1
            ? detail::lorentzian_offdiagonal_average_fp32(
                  operands.sigma1, operands.w1, i, field_stride,
                  operands.stride1)
            : 0.0f;
    const float offdiagonal2 =
        operands.sigma2
            ? detail::lorentzian_offdiagonal_average_fp32(
                  operands.sigma2, operands.w2, i, field_stride,
                  operands.stride2)
            : 0.0f;
    detail::apply_lorentzian_update_fp32(
        polarization[i], previous_polarization[i], field[i],
        material.sigma[i], offdiagonal1, offdiagonal2, anisotropic,
        coefficients);
  }
}

__global__ void update_lorentzian_increment_fp32_kernel(
    float *polarization, float *polarization_increment, const float *field,
    const float *field1, const float *field2,
    const std::ptrdiff_t *indices, std::size_t count,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, lorentzian_material_fp32 material) {
  const detail::lorentzian_operands_fp32 operands =
      detail::normalize_lorentzian_operands(
          field1, field2, material.offdiagonal1, material.offdiagonal2,
          stride1, stride2);
  const bool anisotropic = operands.sigma1 != nullptr;
  const detail::lorentzian_coefficients_fp32 coefficients = {
      material.gamma_inverse,
      material.gamma_previous,
      material.omega_dt_squared,
      material.omega_dt_squared_denominator};
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) *
      static_cast<std::size_t>(blockDim.x);
  for (std::size_t j =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       j < count; j += grid_stride) {
    const std::ptrdiff_t i =
        indices ? indices[j] : static_cast<std::ptrdiff_t>(j);
    const float offdiagonal1 =
        operands.sigma1
            ? detail::lorentzian_offdiagonal_average_fp32(
                  operands.sigma1, operands.w1, i, field_stride,
                  operands.stride1)
            : 0.0f;
    const float offdiagonal2 =
        operands.sigma2
            ? detail::lorentzian_offdiagonal_average_fp32(
                  operands.sigma2, operands.w2, i, field_stride,
                  operands.stride2)
            : 0.0f;
    detail::apply_lorentzian_increment_update_fp32(
        polarization[i], polarization_increment[i], field[i],
        material.sigma[i], offdiagonal1, offdiagonal2, anisotropic,
        coefficients);
  }
}

__global__ void update_lorentzian_increment_structured_fp32_kernel(
    float *polarization, float *polarization_increment, const float *field,
    const float *field1, const float *field2,
    index_space_fp32 index_space, std::size_t count,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, lorentzian_material_fp32 material) {
  const detail::lorentzian_operands_fp32 operands =
      detail::normalize_lorentzian_operands(
          field1, field2, material.offdiagonal1, material.offdiagonal2,
          stride1, stride2);
  const bool anisotropic = operands.sigma1 != nullptr;
  const detail::lorentzian_coefficients_fp32 coefficients = {
      material.gamma_inverse,
      material.gamma_previous,
      material.omega_dt_squared,
      material.omega_dt_squared_denominator};
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) *
      static_cast<std::size_t>(blockDim.x);
  for (std::size_t point =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       point < count; point += grid_stride) {
    const std::ptrdiff_t i =
        detail::decode_index_space_fp32(index_space, point).field;
    const float offdiagonal1 =
        operands.sigma1
            ? detail::lorentzian_offdiagonal_average_fp32(
                  operands.sigma1, operands.w1, i, field_stride,
                  operands.stride1)
            : 0.0f;
    const float offdiagonal2 =
        operands.sigma2
            ? detail::lorentzian_offdiagonal_average_fp32(
                  operands.sigma2, operands.w2, i, field_stride,
                  operands.stride2)
            : 0.0f;
    detail::apply_lorentzian_increment_update_fp32(
        polarization[i], polarization_increment[i], field[i],
        material.sigma[i], offdiagonal1, offdiagonal2, anisotropic,
        coefficients);
  }
}

__global__ void update_gyrotropic_structured_fp32_kernel(
    float *polarization0, float *polarization1, float *polarization2,
    float *previous0, float *previous1, float *previous2,
    const float *field0, const float *field1, const float *field2,
    index_space_fp32 index_space, std::size_t count,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, gyrotropic_material_fp32 material) {
  detail::gyrotropic_coefficients_fp32 coefficients = {
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
#pragma unroll
  for (int entry = 0; entry < 9; ++entry) {
    coefficients.inverse[entry] = material.inverse[entry];
    coefficients.gyro[entry] = material.gyro[entry];
  }

  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) *
      static_cast<std::size_t>(blockDim.x);
  for (std::size_t point =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       point < count; point += grid_stride) {
    const std::ptrdiff_t index =
        detail::decode_index_space_fp32(index_space, point).field;
    const float transverse1 =
        detail::gyrotropic_offdiagonal_average_fp32(
            field1, index, field_stride, stride1);
    const float transverse2 =
        detail::gyrotropic_offdiagonal_average_fp32(
            field2, index, field_stride, stride2);
    detail::apply_gyrotropic_update_fp32(
        polarization0[index], polarization1[index], polarization2[index],
        previous0[index], previous1[index], previous2[index],
        field0[index], transverse1, transverse2, material.sigma[index],
        coefficients);
  }
}

__global__ void update_multilevel_population_fp32_kernel(
    float *population, float *population_scratch, const float *gamma,
    const float *gamma_inverse, const float *alpha,
    std::size_t level_count, std::size_t transition_count,
    std::size_t array_count, float half_dt,
    index_space_fp32 centered_space, std::size_t centered_point_count,
    const multilevel_population_channel_fp32 *channels,
    std::size_t channel_count,
    const multilevel_transition_fp32 *transitions) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t point =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       point < centered_point_count; point += grid_stride) {
    const std::ptrdiff_t field_index =
        detail::decode_index_space_fp32(centered_space, point).field;
    const std::size_t state =
        static_cast<std::size_t>(field_index) * level_count;

    // Trapezoidal relaxation numerator:
    // scratch = (I - Gamma*dt/2) * population.
    for (std::size_t output_level = 0; output_level < level_count;
         ++output_level) {
      float value = 0.0f;
      for (std::size_t input_level = 0; input_level < level_count;
           ++input_level) {
        const float coefficient =
            (output_level == input_level ? 1.0f : 0.0f) -
            gamma[output_level * level_count + input_level] * half_dt;
        value += coefficient * population[state + input_level];
      }
      population_scratch[state + output_level] = value;
    }

    // Preserve the CPU implementation's transition -> spatial component ->
    // real/imaginary accumulation order for a tight FP32 oracle.
    for (std::size_t transition = 0; transition < transition_count;
         ++transition) {
      float field_delta_p_32 = 0.0f;
      float field_average_p_64 = 0.0f;
      for (std::size_t channel_index = 0; channel_index < channel_count;
           ++channel_index) {
        const multilevel_population_channel_fp32 channel =
            channels[channel_index];
        const std::ptrdiff_t i0 = field_index;
        const std::ptrdiff_t i1 =
            field_index + channel.centered_offset1;
        const std::ptrdiff_t i2 =
            field_index + channel.centered_offset2;
        const std::ptrdiff_t i3 =
            i1 + channel.centered_offset2;
        const float field8 =
            channel.field[i0] + channel.field[i1] +
            channel.field[i2] + channel.field[i3] +
            channel.previous_field[i0] + channel.previous_field[i1] +
            channel.previous_field[i2] + channel.previous_field[i3];
        const std::size_t transition_offset =
            2 * transition * array_count;
        const float *current =
            channel.polarization + transition_offset;
        const float *previous = current + array_count;
        const float current4 =
            current[i0] + current[i1] + current[i2] + current[i3];
        const float previous4 =
            previous[i0] + previous[i1] + previous[i2] + previous[i3];
        field_delta_p_32 += (current4 - previous4) * field8;
        field_average_p_64 += (current4 + previous4) * field8;
      }
      field_delta_p_32 *= 0.03125f;
      field_average_p_64 *= 0.015625f;
      const float interaction =
          field_delta_p_32 +
          transitions[transition].population_damping *
              field_average_p_64;
      for (std::size_t level = 0; level < level_count; ++level)
        population_scratch[state + level] +=
            alpha[level * transition_count + transition] * interaction;
    }

    // Trapezoidal relaxation denominator:
    // population = inv(I + Gamma*dt/2) * scratch.
    for (std::size_t output_level = 0; output_level < level_count;
         ++output_level) {
      float value = 0.0f;
      for (std::size_t input_level = 0; input_level < level_count;
           ++input_level)
        value +=
            gamma_inverse[output_level * level_count + input_level] *
            population_scratch[state + input_level];
      population[state + output_level] = value;
    }
  }
}

__global__ void update_multilevel_polarization_fp32_kernel(
    const float *population, std::size_t level_count,
    std::size_t transition_count, std::size_t array_count,
    const multilevel_polarization_channel_fp32 *channels,
    std::size_t channel_count,
    const multilevel_transition_fp32 *transitions) {
  const std::size_t point_grid_stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const std::size_t operation_count = channel_count * transition_count;
  for (std::size_t operation = blockIdx.y; operation < operation_count;
       operation += gridDim.y) {
    const std::size_t channel_index = operation / transition_count;
    const std::size_t transition = operation % transition_count;
    const multilevel_polarization_channel_fp32 channel =
        channels[channel_index];
    const multilevel_transition_fp32 coefficients =
        transitions[transition];
    float *current =
        channel.polarization + 2 * transition * array_count;
    float *previous = current + array_count;
    for (std::size_t point =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         point < channel.point_count; point += point_grid_stride) {
      const std::ptrdiff_t i =
          detail::decode_index_space_fp32(channel.index_space, point).field;
      const std::ptrdiff_t n0 = i;
      const std::ptrdiff_t n1 = i + channel.population_offset1;
      const std::ptrdiff_t n2 = i + channel.population_offset2;
      const std::ptrdiff_t n3 =
          n1 + channel.population_offset2;
      const std::size_t upper =
          static_cast<std::size_t>(coefficients.upper_level);
      const std::size_t lower =
          static_cast<std::size_t>(coefficients.lower_level);
      const float inversion =
          0.25f *
          (population[static_cast<std::size_t>(n0) * level_count +
                      upper] +
           population[static_cast<std::size_t>(n1) * level_count +
                      upper] +
           population[static_cast<std::size_t>(n2) * level_count +
                      upper] +
           population[static_cast<std::size_t>(n3) * level_count +
                      upper] -
           population[static_cast<std::size_t>(n0) * level_count +
                      lower] -
           population[static_cast<std::size_t>(n1) * level_count +
                      lower] -
           population[static_cast<std::size_t>(n2) * level_count +
                      lower] -
           population[static_cast<std::size_t>(n3) * level_count +
                      lower]);
      const float current_value = current[i];
      current[i] =
          coefficients.gamma_inverse *
          (current_value * coefficients.diagonal -
           coefficients.gamma_previous * previous[i] -
           coefficients.drive_scale[channel.direction] *
               channel.sigma[i] * channel.field[i] * inversion);
      previous[i] = current_value;
    }
  }
}

__global__ void copy_fp32_kernel(float *destination, const float *source,
                                 std::size_t count) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) *
      static_cast<std::size_t>(blockDim.x);
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += grid_stride)
    destination[i] = source[i];
}

__global__ void zero_fp32_kernel(float *destination, std::size_t count) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) *
      static_cast<std::size_t>(blockDim.x);
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += grid_stride)
    destination[index] = 0.0f;
}

__global__ void subtract_fp32_kernel(float *destination, const float *source,
                                     std::size_t count) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) *
      static_cast<std::size_t>(blockDim.x);
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += grid_stride)
    destination[i] -= source[i];
}

__global__ void indexed_subtract_fp32_kernel(
    float *destination, const indexed_value_fp32 *updates,
    std::size_t count) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) *
      static_cast<std::size_t>(blockDim.x);
  for (std::size_t j =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       j < count; j += grid_stride)
    atomicAdd(destination + updates[j].index, -updates[j].value);
}

__global__ void indexed_source_subtract_fp32_kernel(
    float *destination, const std::ptrdiff_t *indices,
    const complex_value_fp32 *amplitudes,
    const float *conductivity_inverse, std::size_t count,
    complex_value_fp32 time_scale, bool imaginary_component) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) *
      static_cast<std::size_t>(blockDim.x);
  for (std::size_t j =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       j < count; j += grid_stride) {
    const complex_value_fp32 amplitude = amplitudes[j];
    float value =
        imaginary_component
            ? amplitude.real * time_scale.imag +
                  amplitude.imag * time_scale.real
            : amplitude.real * time_scale.real -
                  amplitude.imag * time_scale.imag;
    if (conductivity_inverse) value *= conductivity_inverse[indices[j]];
    atomicAdd(destination + indices[j], -value);
  }
}

__global__ void indexed_source_subtract_phase_batched_fp32_kernel(
    const indexed_source_phase_operation_fp32 *operations,
    const std::uint32_t *block_operation_indices,
    std::size_t total_block_count,
    indexed_source_phase_time_scales_fp32 time_scales) {
  __shared__ indexed_source_phase_operation_fp32 operation;
  __shared__ complex_value_fp32 time_scale;
  for (std::size_t phase_block = blockIdx.x;
       phase_block < total_block_count;
       phase_block += gridDim.x) {
    if (threadIdx.x == 0) {
      const std::uint32_t operation_index =
          block_operation_indices[phase_block];
      operation = operations[operation_index];
      time_scale = time_scales.values[operation_index];
    }
    __syncthreads();
    const std::size_t operation_block =
        phase_block - operation.block_start;
    const std::size_t point =
        operation_block * blockDim.x + threadIdx.x;
    if (point < operation.point_count) {
      const complex_value_fp32 amplitude = operation.amplitudes[point];
      float value =
          operation.imaginary_component
              ? amplitude.real * time_scale.imag +
                    amplitude.imag * time_scale.real
              : amplitude.real * time_scale.real -
                    amplitude.imag * time_scale.imag;
      const std::ptrdiff_t index = operation.indices[point];
      if (operation.conductivity_inverse)
        value *= operation.conductivity_inverse[index];
      // A public source profile may contain repeated indices. Match the
      // legacy single-source kernel so concurrent updates cannot be lost.
      atomicAdd(operation.destination + index, -value);
    }
    if (static_cast<std::size_t>(gridDim.x) <
        total_block_count - phase_block)
      __syncthreads();
  }
}

__global__ void initialize_indexed_ldos_result_kernel(double *result) {
  if (threadIdx.x < 4) result[threadIdx.x] = 0.0;
}

__global__ void indexed_ldos_reduce_fp32_partials_kernel(
    const indexed_ldos_operation_fp32 *operations,
    const std::uint32_t *block_operation_indices,
    std::size_t total_block_count, double *block_partials) {
  extern __shared__ double partial_sums[];
  __shared__ indexed_ldos_operation_fp32 operation;
  double *partial_real = partial_sums;
  double *partial_imaginary = partial_sums + blockDim.x;
  double block_totals[4] = {0.0, 0.0, 0.0, 0.0};

  for (std::size_t phase_block = blockIdx.x;
       phase_block < total_block_count;) {
    if (threadIdx.x == 0)
      operation = operations[block_operation_indices[phase_block]];
    __syncthreads();
    const std::size_t operation_block =
        phase_block - operation.block_start;
    const std::size_t point =
        operation_block * blockDim.x + threadIdx.x;

    double contribution_real = 0.0;
    double contribution_imaginary = 0.0;
    if (point < operation.point_count) {
      const std::ptrdiff_t field_index = operation.indices[point];
      const float field_real = operation.field_real[field_index];
      const float field_imaginary = operation.field_imaginary
                                        ? operation.field_imaginary[field_index]
                                        : 0.0f;
      const complex_value_fp32 amplitude = operation.amplitudes[point];
      // (Fr + i Fi) * conj(Ar + i Ai). Evaluate the products and pairwise
      // sums in FP32, matching resident field/source precision, then promote
      // before the fixed-order FP64 reduction.
      contribution_real = static_cast<double>(__fadd_rn(
          __fmul_rn(field_real, amplitude.real),
          __fmul_rn(field_imaginary, amplitude.imag)));
      contribution_imaginary = static_cast<double>(__fsub_rn(
          __fmul_rn(field_imaginary, amplitude.real),
          __fmul_rn(field_real, amplitude.imag)));
    }
    partial_real[threadIdx.x] = contribution_real;
    partial_imaginary[threadIdx.x] = contribution_imaginary;
    __syncthreads();

    for (unsigned int active = blockDim.x; active > 1;) {
      const unsigned int retained = (active + 1) / 2;
      const unsigned int paired = active / 2;
      if (threadIdx.x < paired) {
        partial_real[threadIdx.x] +=
            partial_real[threadIdx.x + retained];
        partial_imaginary[threadIdx.x] +=
            partial_imaginary[threadIdx.x + retained];
      }
      active = retained;
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const std::size_t channel = operation.magnetic ? 2 : 0;
      block_totals[channel] += partial_real[0];
      block_totals[channel + 1] += partial_imaginary[0];
    }
    __syncthreads();
    const std::size_t remaining = total_block_count - phase_block;
    if (remaining <= static_cast<std::size_t>(gridDim.x)) break;
    phase_block += static_cast<std::size_t>(gridDim.x);
  }
  if (threadIdx.x == 0)
    for (std::size_t channel = 0; channel < 4; ++channel)
      block_partials[4 * static_cast<std::size_t>(blockIdx.x) + channel] =
          block_totals[channel];
}

__global__ void indexed_ldos_reduce_fp32_final_kernel(
    const double *block_partials, std::size_t partial_count,
    double *result, indexed_ldos_result_mode result_mode) {
  extern __shared__ double partial_sums[];
  for (std::size_t channel = 0; channel < 4; ++channel) {
    double thread_sum = 0.0;
    for (std::size_t partial = threadIdx.x; partial < partial_count;
         partial += blockDim.x)
      thread_sum += block_partials[4 * partial + channel];
    partial_sums[threadIdx.x] = thread_sum;
    __syncthreads();
    for (unsigned int active = blockDim.x; active > 1;) {
      const unsigned int retained = (active + 1) / 2;
      const unsigned int paired = active / 2;
      if (threadIdx.x < paired)
        partial_sums[threadIdx.x] +=
            partial_sums[threadIdx.x + retained];
      active = retained;
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      if (result_mode == indexed_ldos_result_mode::replace)
        result[channel] = partial_sums[0];
      else
        result[channel] += partial_sums[0];
    }
    __syncthreads();
  }
}

__global__ void dft_pair_reduce_fp32_partials_kernel(
    const dft_pair_reduction_operation_fp32 *operations,
    const std::uint32_t *block_operation_indices,
    std::size_t total_spatial_block_count,
    std::size_t frequency_count, double *partials) {
  constexpr std::size_t frequency_threads =
      dft_pair_reduction_frequency_threads;
  constexpr std::size_t point_threads =
      dft_pair_reduction_point_threads;
  extern __shared__ double shared_values[];
  double *shared_real = shared_values;
  double *shared_imaginary = shared_values +
                             frequency_threads * point_threads;
  __shared__ dft_pair_reduction_operation_fp32 operation;
  const std::size_t shared_index =
      static_cast<std::size_t>(threadIdx.x) * point_threads + threadIdx.y;
  const std::size_t frequency_stride =
      static_cast<std::size_t>(gridDim.x) * frequency_threads;

  for (std::size_t frequency_base =
           static_cast<std::size_t>(blockIdx.x) * frequency_threads;
       frequency_base < frequency_count;
       frequency_base += frequency_stride) {
    const std::size_t frequency = frequency_base + threadIdx.x;
    double block_total_real = 0.0;
    double block_total_imaginary = 0.0;
    for (std::size_t spatial_block = blockIdx.y;
         spatial_block < total_spatial_block_count;
         spatial_block += gridDim.y) {
      if (threadIdx.x == 0 && threadIdx.y == 0)
        operation = operations[block_operation_indices[spatial_block]];
      __syncthreads();
      const std::size_t operation_block =
          spatial_block - operation.block_start;
      const std::size_t point = operation_block * point_threads + threadIdx.y;
      double contribution_real = 0.0;
      double contribution_imaginary = 0.0;
      if (frequency < frequency_count && point < operation.point_count) {
        const std::size_t complex_index =
            point * frequency_count + frequency;
        const float lhs_real = operation.lhs_real_imag[2 * complex_index];
        const float lhs_imaginary =
            operation.lhs_real_imag[2 * complex_index + 1];
        const float rhs_real = operation.rhs_real_imag[2 * complex_index];
        const float rhs_imaginary =
            operation.rhs_real_imag[2 * complex_index + 1];
        const float product_real = __fadd_rn(
            __fmul_rn(lhs_real, rhs_real),
            __fmul_rn(lhs_imaginary, rhs_imaginary));
        const float product_imaginary = __fsub_rn(
            __fmul_rn(lhs_imaginary, rhs_real),
            __fmul_rn(lhs_real, rhs_imaginary));
        const float weighted_real = __fsub_rn(
            __fmul_rn(operation.weight.real, product_real),
            __fmul_rn(operation.weight.imag, product_imaginary));
        const float weighted_imaginary = __fadd_rn(
            __fmul_rn(operation.weight.real, product_imaginary),
            __fmul_rn(operation.weight.imag, product_real));
        contribution_real = static_cast<double>(weighted_real);
        contribution_imaginary = static_cast<double>(weighted_imaginary);
      }
      shared_real[shared_index] = contribution_real;
      shared_imaginary[shared_index] = contribution_imaginary;
      __syncthreads();
      for (unsigned int active = point_threads; active > 1;) {
        const unsigned int retained = (active + 1) / 2;
        const unsigned int paired = active / 2;
        if (threadIdx.y < paired) {
          const std::size_t paired_index =
              static_cast<std::size_t>(threadIdx.x) * point_threads +
              threadIdx.y + retained;
          shared_real[shared_index] += shared_real[paired_index];
          shared_imaginary[shared_index] +=
              shared_imaginary[paired_index];
        }
        active = retained;
        __syncthreads();
      }
      if (threadIdx.y == 0) {
        block_total_real += shared_real[shared_index];
        block_total_imaginary += shared_imaginary[shared_index];
      }
      __syncthreads();
    }
    if (threadIdx.y == 0 && frequency < frequency_count) {
      const std::size_t partial =
          frequency * static_cast<std::size_t>(gridDim.y) + blockIdx.y;
      partials[2 * partial] = block_total_real;
      partials[2 * partial + 1] = block_total_imaginary;
    }
  }
}

__global__ void dft_pair_reduce_fp32_final_kernel(
    const double *partials, std::size_t partial_count,
    std::size_t frequency_count, double *result) {
  extern __shared__ double shared_values[];
  double *shared_real = shared_values;
  double *shared_imaginary = shared_values + blockDim.x;
  for (std::size_t frequency = blockIdx.x; frequency < frequency_count;
       frequency += gridDim.x) {
    double thread_real = 0.0;
    double thread_imaginary = 0.0;
    for (std::size_t partial = threadIdx.x; partial < partial_count;
         partial += blockDim.x) {
      const std::size_t index = frequency * partial_count + partial;
      thread_real += partials[2 * index];
      thread_imaginary += partials[2 * index + 1];
    }
    shared_real[threadIdx.x] = thread_real;
    shared_imaginary[threadIdx.x] = thread_imaginary;
    __syncthreads();
    for (unsigned int active = blockDim.x; active > 1;) {
      const unsigned int retained = (active + 1) / 2;
      const unsigned int paired = active / 2;
      if (threadIdx.x < paired) {
        shared_real[threadIdx.x] += shared_real[threadIdx.x + retained];
        shared_imaginary[threadIdx.x] +=
            shared_imaginary[threadIdx.x + retained];
      }
      active = retained;
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      result[2 * frequency] = shared_real[0];
      result[2 * frequency + 1] = shared_imaginary[0];
    }
    __syncthreads();
  }
}

__global__ void eigenmode_overlap_fp32_partials_kernel(
    const eigenmode_overlap_operation_fp32 *operations,
    const std::uint32_t *block_operation_indices,
    std::size_t total_block_count,
    std::size_t selected_frequency_index, double *partials,
    std::size_t physical_block_count, std::size_t output_count) {
  extern __shared__ double shared_values[];
  double *shared_real = shared_values;
  double *shared_imaginary = shared_values + blockDim.x;
  __shared__ eigenmode_overlap_operation_fp32 operation;
  __shared__ double output_totals[
      2 * eigenmode_overlap_max_output_count];
  if (threadIdx.x < 2 * output_count)
    output_totals[threadIdx.x] = 0.0;
  __syncthreads();

  for (std::size_t logical_block = blockIdx.x;
       logical_block < total_block_count;
       logical_block += gridDim.x) {
    if (threadIdx.x == 0)
      operation = operations[block_operation_indices[logical_block]];
    __syncthreads();
    const std::size_t operation_block =
        logical_block - operation.block_start;
    const std::size_t point =
        operation_block * eigenmode_overlap_threads_per_block +
        threadIdx.x;
    double contribution_real = 0.0;
    double contribution_imaginary = 0.0;
    if (point < operation.point_count) {
      const complex_value_fp64 weighted_mode =
          operation.weighted_conjugate_mode[point];
      double rhs_real = 0.0;
      double rhs_imaginary = 0.0;
      if (operation.mode2_real_imag) {
        const complex_value_fp64 mode2 = operation.mode2_real_imag[point];
        rhs_real = mode2.real;
        rhs_imaginary = mode2.imag;
      }
      else {
        const std::size_t source_index =
            point * operation.dft_frequency_count +
            selected_frequency_index;
        const double source_real = static_cast<double>(
            operation.dft_real_imag[2 * source_index]);
        const double source_imaginary = static_cast<double>(
            operation.dft_real_imag[2 * source_index + 1]);
        rhs_real = source_real * operation.inverse_stored_weight.real -
                   source_imaginary *
                       operation.inverse_stored_weight.imag;
        rhs_imaginary =
            source_real * operation.inverse_stored_weight.imag +
            source_imaginary * operation.inverse_stored_weight.real;
        if (operation.zero_normalization_divisors &&
            operation.zero_normalization_divisors[point] &&
            (rhs_real != 0.0 || rhs_imaginary != 0.0)) {
          const double divisor = static_cast<double>(
              operation.zero_normalization_divisors[point]) - 1.0;
          rhs_real /= divisor;
          rhs_imaginary /= divisor;
        }
      }
      contribution_real = weighted_mode.real * rhs_real -
                          weighted_mode.imag * rhs_imaginary;
      contribution_imaginary = weighted_mode.real * rhs_imaginary +
                               weighted_mode.imag * rhs_real;
    }
    shared_real[threadIdx.x] = contribution_real;
    shared_imaginary[threadIdx.x] = contribution_imaginary;
    __syncthreads();
    for (unsigned int active = blockDim.x; active > 1;) {
      const unsigned int retained = (active + 1) / 2;
      const unsigned int paired = active / 2;
      if (threadIdx.x < paired) {
        shared_real[threadIdx.x] += shared_real[threadIdx.x + retained];
        shared_imaginary[threadIdx.x] +=
            shared_imaginary[threadIdx.x + retained];
      }
      active = retained;
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      output_totals[2 * operation.output_index] += shared_real[0];
      output_totals[2 * operation.output_index + 1] +=
          shared_imaginary[0];
    }
    __syncthreads();
  }

  if (threadIdx.x < 2 * output_count) {
    const std::size_t output = threadIdx.x / 2;
    const std::size_t reim = threadIdx.x % 2;
    partials[(2 * output + reim) * physical_block_count + blockIdx.x] =
        output_totals[threadIdx.x];
  }
}

__global__ void eigenmode_overlap_fp32_final_kernel(
    const double *partials, std::size_t partial_count,
    double *result) {
  extern __shared__ double shared_values[];
  const std::size_t output = blockIdx.x;
  double real_total = 0.0;
  double imaginary_total = 0.0;
  for (std::size_t partial = threadIdx.x; partial < partial_count;
       partial += blockDim.x) {
    real_total +=
        partials[(2 * output) * partial_count + partial];
    imaginary_total +=
        partials[(2 * output + 1) * partial_count + partial];
  }
  shared_values[threadIdx.x] = real_total;
  shared_values[blockDim.x + threadIdx.x] = imaginary_total;
  __syncthreads();
  for (unsigned int active = blockDim.x; active > 1;) {
    const unsigned int retained = (active + 1) / 2;
    const unsigned int paired = active / 2;
    if (threadIdx.x < paired) {
      shared_values[threadIdx.x] +=
          shared_values[threadIdx.x + retained];
      shared_values[blockDim.x + threadIdx.x] +=
          shared_values[blockDim.x + threadIdx.x + retained];
    }
    active = retained;
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    result[2 * output] = shared_values[0];
    result[2 * output + 1] = shared_values[blockDim.x];
  }
}

__global__ void gather_boundary_fp32_kernel(
    const boundary_operation_fp32 *operations, float *staging,
    std::size_t count) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) *
      static_cast<std::size_t>(blockDim.x);
  for (std::size_t j =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       j < count; j += grid_stride) {
    const boundary_operation_fp32 operation = operations[j];
    if (!operation.source_real) {
      staging[2 * j] = 0.0f;
      staging[2 * j + 1] = 0.0f;
    }
    else if (operation.source_imag) {
      const float source_real = *operation.source_real;
      const float source_imag = *operation.source_imag;
      staging[2 * j] =
          operation.phase_real * source_real -
          operation.phase_imag * source_imag;
      staging[2 * j + 1] =
          operation.phase_real * source_imag +
          operation.phase_imag * source_real;
    }
    else {
      staging[2 * j] =
          operation.phase_real * *operation.source_real;
      staging[2 * j + 1] = 0.0f;
    }
  }
}

__global__ void scatter_boundary_fp32_kernel(
    const boundary_operation_fp32 *operations, const float *staging,
    std::size_t count) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) *
      static_cast<std::size_t>(blockDim.x);
  for (std::size_t j =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       j < count; j += grid_stride) {
    const boundary_operation_fp32 operation = operations[j];
    *operation.destination_real = staging[2 * j];
    if (operation.destination_imag)
      *operation.destination_imag = staging[2 * j + 1];
  }
}

__global__ void zero_boundary_fp32_kernel(
    const boundary_operation_fp32 *operations, std::size_t count) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) *
      static_cast<std::size_t>(blockDim.x);
  for (std::size_t j =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       j < count; j += grid_stride) {
    const boundary_operation_fp32 operation = operations[j];
    *operation.destination_real = 0.0f;
    if (operation.destination_imag) *operation.destination_imag = 0.0f;
  }
}

__global__ void apply_nonalias_boundary_fp32_kernel(
    const boundary_operation_fp32 *operations, std::size_t count) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) *
      static_cast<std::size_t>(blockDim.x);
  for (std::size_t j =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       j < count; j += grid_stride) {
    const boundary_operation_fp32 operation = operations[j];
    if (!operation.source_real) {
      *operation.destination_real = 0.0f;
      if (operation.destination_imag) *operation.destination_imag = 0.0f;
    }
    else if (operation.source_imag) {
      const float source_real = *operation.source_real;
      const float source_imag = *operation.source_imag;
      *operation.destination_real =
          operation.phase_real * source_real -
          operation.phase_imag * source_imag;
      *operation.destination_imag =
          operation.phase_real * source_imag +
          operation.phase_imag * source_real;
    }
    else {
      *operation.destination_real =
          operation.phase_real * *operation.source_real;
    }
  }
}

__global__ void initialize_finite_result_kernel(int *result) {
  *result = 1;
}

__global__ void all_finite_fp32_kernel(const array_span_fp32 *spans,
                                       std::size_t span_count,
                                       int *result) {
  for (std::size_t span_index = blockIdx.y; span_index < span_count;
       span_index += gridDim.y) {
    const array_span_fp32 span = spans[span_index];
    const std::size_t grid_stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < span.count; index += grid_stride)
      if (!isfinite(span.values[index])) atomicExch(result, 0);
  }
}

__global__ void all_finite_generation_fp32_kernel(
    const array_span_fp32 *spans, std::size_t span_count,
    std::uint32_t generation, std::uint32_t *result) {
  for (std::size_t span_index = blockIdx.y; span_index < span_count;
       span_index += gridDim.y) {
    const array_span_fp32 span = spans[span_index];
    const std::size_t grid_stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < span.count; index += grid_stride)
      if (!isfinite(span.values[index])) atomicExch(result, generation);
  }
}

__global__ void initialize_squared_norm_result_kernel(double *result) {
  *result = 0.0;
}

__global__ void initialize_fp32_result_kernel(float *result, float value) {
  *result = value;
}

__global__ void initialize_int_result_kernel(int *result, int value) {
  *result = value;
}

__global__ void vector_fill_fp32_kernel(float *values, std::size_t count,
                                        float value) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += grid_stride)
    values[index] = value;
}

__global__ void vector_scale_fp32_kernel(float *values, std::size_t count,
                                         double scale) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += grid_stride)
    values[index] = static_cast<float>(
        static_cast<double>(values[index]) * scale);
}

__global__ void complex_scale_inplace_fp32_kernel(
    float *values_real_imag, std::size_t count, double scale_real,
    double scale_imaginary) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += grid_stride) {
    const double real =
        static_cast<double>(values_real_imag[2 * index]);
    const double imaginary =
        static_cast<double>(values_real_imag[2 * index + 1]);
    values_real_imag[2 * index] = static_cast<float>(
        real * scale_real - imaginary * scale_imaginary);
    values_real_imag[2 * index + 1] = static_cast<float>(
        real * scale_imaginary + imaginary * scale_real);
  }
}

__global__ void vector_xpay_fp32_kernel(float *x, const float *y,
                                        std::size_t count, double scale) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += grid_stride)
    x[index] = static_cast<float>(
        static_cast<double>(x[index]) +
        scale * static_cast<double>(y[index]));
}

__global__ void vector_left_minus_scale_fp32_kernel(
    float *output, const float *left, std::size_t count, double scale) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += grid_stride)
    output[index] = static_cast<float>(
        static_cast<double>(left[index]) -
        scale * static_cast<double>(output[index]));
}

__global__ void vector_dot_fp32_partials_kernel(
    const float *x, const float *y, std::size_t count,
    double *block_partials) {
  extern __shared__ double partial_sums[];
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  double thread_sum = 0.0;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += grid_stride)
    thread_sum += static_cast<double>(__fmul_rn(x[index], y[index]));
  partial_sums[threadIdx.x] = thread_sum;
  __syncthreads();
  for (unsigned int active = blockDim.x; active > 1;) {
    const unsigned int retained = (active + 1) / 2;
    const unsigned int paired = active / 2;
    if (threadIdx.x < paired)
      partial_sums[threadIdx.x] += partial_sums[threadIdx.x + retained];
    active = retained;
    __syncthreads();
  }
  if (threadIdx.x == 0) block_partials[blockIdx.x] = partial_sums[0];
}

__global__ void reduce_fp64_partials_kernel(
    const double *block_partials, std::size_t partial_count,
    double *result) {
  extern __shared__ double partial_sums[];
  double thread_sum = 0.0;
  for (std::size_t index = threadIdx.x; index < partial_count;
       index += blockDim.x)
    thread_sum += block_partials[index];
  partial_sums[threadIdx.x] = thread_sum;
  __syncthreads();
  for (unsigned int active = blockDim.x; active > 1;) {
    const unsigned int retained = (active + 1) / 2;
    const unsigned int paired = active / 2;
    if (threadIdx.x < paired)
      partial_sums[threadIdx.x] += partial_sums[threadIdx.x + retained];
    active = retained;
    __syncthreads();
  }
  if (threadIdx.x == 0) *result = partial_sums[0];
}

__global__ void vector_max_abs_fp32_kernel(const float *values,
                                           std::size_t count,
                                           float *result,
                                           int *all_finite) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += grid_stride) {
    const float magnitude = fabsf(values[index]);
    if (isfinite(magnitude))
      atomicMax(reinterpret_cast<unsigned int *>(result),
                __float_as_uint(magnitude));
    else if (all_finite)
      atomicExch(all_finite, 0);
  }
}

__global__ void vector_scaled_sum_squares_fp32_partials_kernel(
    const float *values, std::size_t count, double scale,
    double *block_partials) {
  extern __shared__ double partial_sums[];
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  double thread_sum = 0.0;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += grid_stride) {
    const double scaled = static_cast<double>(values[index]) * scale;
    thread_sum += scaled * scaled;
  }
  partial_sums[threadIdx.x] = thread_sum;
  __syncthreads();
  for (unsigned int active = blockDim.x; active > 1;) {
    const unsigned int retained = (active + 1) / 2;
    const unsigned int paired = active / 2;
    if (threadIdx.x < paired)
      partial_sums[threadIdx.x] += partial_sums[threadIdx.x + retained];
    active = retained;
    __syncthreads();
  }
  if (threadIdx.x == 0) block_partials[blockIdx.x] = partial_sums[0];
}

__global__ void gather_cw_field_vector_fp32_kernel(
    const cw_field_vector_segment_fp32 *segments,
    std::size_t segment_count, float *packed_real_imag) {
  for (std::size_t segment_index = blockIdx.y;
       segment_index < segment_count; segment_index += gridDim.y) {
    const cw_field_vector_segment_fp32 segment = segments[segment_index];
    const std::size_t grid_stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t point =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         point < segment.point_count; point += grid_stride) {
      const detail::decoded_index_fp32 decoded =
          detail::decode_index_space_fp32(segment.index_space, point);
      const std::size_t packed = 2 * (segment.complex_offset + point);
      packed_real_imag[packed] = segment.field_real[decoded.field];
      packed_real_imag[packed + 1] =
          segment.field_imaginary[decoded.field];
    }
  }
}

__global__ void scatter_cw_field_vector_fp32_kernel(
    const cw_field_vector_segment_fp32 *segments,
    std::size_t segment_count, const float *packed_real_imag) {
  for (std::size_t segment_index = blockIdx.y;
       segment_index < segment_count; segment_index += gridDim.y) {
    const cw_field_vector_segment_fp32 segment = segments[segment_index];
    const std::size_t grid_stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t point =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         point < segment.point_count; point += grid_stride) {
      const detail::decoded_index_fp32 decoded =
          detail::decode_index_space_fp32(segment.index_space, point);
      const std::size_t packed = 2 * (segment.complex_offset + point);
      segment.field_real[decoded.field] = packed_real_imag[packed];
      segment.field_imaginary[decoded.field] =
          packed_real_imag[packed + 1];
    }
  }
}

__global__ void gather_cw_field_operator_fp32_kernel(
    const cw_field_vector_segment_fp32 *segments,
    std::size_t segment_count, const float *input_real_imag,
    float *output_real_imag, float dt_inverse, float iomega_real,
    float iomega_imaginary) {
  for (std::size_t segment_index = blockIdx.y;
       segment_index < segment_count; segment_index += gridDim.y) {
    const cw_field_vector_segment_fp32 segment = segments[segment_index];
    const std::size_t grid_stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t point =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         point < segment.point_count; point += grid_stride) {
      const detail::decoded_index_fp32 decoded =
          detail::decode_index_space_fp32(segment.index_space, point);
      const std::size_t packed = 2 * (segment.complex_offset + point);
      const float input_real = input_real_imag[packed];
      const float input_imaginary = input_real_imag[packed + 1];
      const float delta_real = __fmul_rn(
          __fsub_rn(segment.field_real[decoded.field], input_real),
          dt_inverse);
      const float delta_imaginary = __fmul_rn(
          __fsub_rn(segment.field_imaginary[decoded.field],
                    input_imaginary),
          dt_inverse);
      const float shift_real = __fsub_rn(
          __fmul_rn(iomega_real, input_real),
          __fmul_rn(iomega_imaginary, input_imaginary));
      const float shift_imaginary = __fadd_rn(
          __fmul_rn(iomega_real, input_imaginary),
          __fmul_rn(iomega_imaginary, input_real));
      output_real_imag[packed] = __fadd_rn(delta_real, shift_real);
      output_real_imag[packed + 1] =
          __fadd_rn(delta_imaginary, shift_imaginary);
    }
  }
}

__global__ void squared_norm_complex_fp32_kernel(
    const float *values, const std::ptrdiff_t *point_indices,
    std::size_t point_count, std::size_t frequency_count, double *result) {
  extern __shared__ double partial_sums[];
  const std::size_t work_count = point_count * frequency_count;
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  double thread_sum = 0.0;
  for (std::size_t work =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       work < work_count; work += grid_stride) {
    const std::size_t selected_point = work / frequency_count;
    const std::size_t frequency = work % frequency_count;
    const std::ptrdiff_t storage_point =
        point_indices
            ? point_indices[selected_point]
            : static_cast<std::ptrdiff_t>(selected_point);
    const std::size_t complex_index =
        static_cast<std::size_t>(storage_point) * frequency_count +
        frequency;
    const float real = values[2 * complex_index];
    const float imaginary = values[2 * complex_index + 1];
    // Match the historical complex<realnum> CPU path: evaluate each
    // magnitude squared with FP32 arithmetic, then promote that term for the
    // stable FP64 reduction. Promoting the components before multiplication
    // changes decay thresholds and can make CPU/CUDA execute different work.
    const float magnitude_squared = __fadd_rn(
        __fmul_rn(real, real),
        __fmul_rn(imaginary, imaginary));
    thread_sum += static_cast<double>(magnitude_squared);
  }
  partial_sums[threadIdx.x] = thread_sum;
  __syncthreads();

  // This reduction also supports non-power-of-two block sizes.
  for (unsigned int active = blockDim.x; active > 1;) {
    const unsigned int retained = (active + 1) / 2;
    const unsigned int paired = active / 2;
    if (threadIdx.x < paired)
      partial_sums[threadIdx.x] +=
          partial_sums[threadIdx.x + retained];
    active = retained;
    __syncthreads();
  }
  // The supported architecture floor is sm_60, where native FP64 atomicAdd
  // is available.
  if (threadIdx.x == 0) atomicAdd(result, partial_sums[0]);
}

__global__ void update_dft_fp32_kernel(
    float *dft, const float *field_real, const float *field_imag,
    const std::ptrdiff_t *indices, const float *weights,
    std::size_t point_count, const complex_value_fp32 *phases,
    std::size_t frequency_count, std::ptrdiff_t average_offset1,
    std::ptrdiff_t average_offset2) {
  extern __shared__ float field_values[];
  float *real_values = field_values;
  float *imag_values = field_values + blockDim.y;
  const std::size_t point_stride =
      static_cast<std::size_t>(gridDim.y) * blockDim.y;
  const std::size_t frequency_stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;

  for (std::size_t point_base =
           static_cast<std::size_t>(blockIdx.y) * blockDim.y;
       point_base < point_count; point_base += point_stride) {
    const std::size_t point = point_base + threadIdx.y;
    if (threadIdx.x == 0 && point < point_count) {
      const std::ptrdiff_t index = indices[point];
      float real_value;
      float imag_value = 0.0f;
      if (average_offset2) {
        real_value =
            0.25f * (field_real[index] +
                     field_real[index + average_offset1] +
                     field_real[index + average_offset2] +
                     field_real[index + average_offset1 + average_offset2]);
        if (field_imag)
          imag_value =
              0.25f * (field_imag[index] +
                       field_imag[index + average_offset1] +
                       field_imag[index + average_offset2] +
                       field_imag[index + average_offset1 + average_offset2]);
      }
      else if (average_offset1) {
        real_value =
            0.5f * (field_real[index] +
                    field_real[index + average_offset1]);
        if (field_imag)
          imag_value =
              0.5f * (field_imag[index] +
                      field_imag[index + average_offset1]);
      }
      else {
        real_value = field_real[index];
        if (field_imag) imag_value = field_imag[index];
      }
      real_values[threadIdx.y] = real_value * weights[point];
      imag_values[threadIdx.y] = imag_value * weights[point];
    }
    __syncthreads();

    if (point < point_count)
      for (std::size_t frequency =
               static_cast<std::size_t>(blockIdx.x) * blockDim.x +
               threadIdx.x;
           frequency < frequency_count; frequency += frequency_stride) {
        const complex_value_fp32 phase = phases[frequency];
        const std::size_t output =
            2 * (point * frequency_count + frequency);
        const float real_value = real_values[threadIdx.y];
        const float imag_value = imag_values[threadIdx.y];
        dft[output] +=
            phase.real * real_value - phase.imag * imag_value;
        dft[output + 1] +=
            phase.real * imag_value + phase.imag * real_value;
      }
    __syncthreads();
  }
}

__global__ void update_dft_batch_fp32_kernel(
    const dft_update_operation_fp32 *operations,
    const std::uint32_t *block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count) {
  extern __shared__ float field_values[];
  float *real_values = field_values;
  float *imag_values = field_values + blockDim.x;

  for (std::size_t logical_block = blockIdx.x;
       logical_block < total_block_count;
       logical_block += gridDim.x) {
    const std::uint32_t operation_index =
        block_operation_indices[logical_block];
    if (operation_index >= operation_count) return;
    const dft_update_operation_fp32 operation =
        operations[operation_index];
    const std::size_t local_block =
        logical_block - operation.block_start;
    const std::size_t frequency_block =
        local_block % operation.frequency_block_count;
    const std::size_t point_block =
        local_block / operation.frequency_block_count;
    const std::uint32_t frequency_lane =
        threadIdx.x % operation.frequency_threads;
    const std::uint32_t point_lane =
        threadIdx.x / operation.frequency_threads;
    const bool active_thread =
        point_lane < operation.point_threads;
    const std::size_t point_stride =
        operation.point_block_count * operation.point_threads;
    const std::size_t frequency_stride =
        operation.frequency_block_count * operation.frequency_threads;

    for (std::size_t point_base =
             point_block * operation.point_threads;
         point_base < operation.point_count;
         point_base += point_stride) {
      const std::size_t point = point_base + point_lane;
      if (active_thread && frequency_lane == 0 &&
          point < operation.point_count) {
        const std::ptrdiff_t index = operation.field_indices[point];
        float real_value;
        float imag_value = 0.0f;
        if (operation.average_offset2) {
          real_value =
              0.25f *
              (operation.field_real[index] +
               operation.field_real[index + operation.average_offset1] +
               operation.field_real[index + operation.average_offset2] +
               operation.field_real[
                   index + operation.average_offset1 +
                   operation.average_offset2]);
          if (operation.field_imag)
            imag_value =
                0.25f *
                (operation.field_imag[index] +
                 operation.field_imag[index + operation.average_offset1] +
                 operation.field_imag[index + operation.average_offset2] +
                 operation.field_imag[
                     index + operation.average_offset1 +
                     operation.average_offset2]);
        }
        else if (operation.average_offset1) {
          real_value =
              0.5f *
              (operation.field_real[index] +
               operation.field_real[index + operation.average_offset1]);
          if (operation.field_imag)
            imag_value =
                0.5f *
                (operation.field_imag[index] +
                 operation.field_imag[index + operation.average_offset1]);
        }
        else {
          real_value = operation.field_real[index];
          if (operation.field_imag)
            imag_value = operation.field_imag[index];
        }
        real_values[point_lane] =
            real_value * operation.weights[point];
        imag_values[point_lane] =
            imag_value * operation.weights[point];
      }
      __syncthreads();

      if (active_thread && point < operation.point_count)
        for (std::size_t frequency =
                 frequency_block * operation.frequency_threads +
                 frequency_lane;
             frequency < operation.frequency_count;
             frequency += frequency_stride) {
          const complex_value_fp32 phase = operation.phases[frequency];
          const std::size_t output =
              2 * (point * operation.frequency_count + frequency);
          const float real_value = real_values[point_lane];
          const float imag_value = imag_values[point_lane];
          operation.dft_real_imag[output] +=
              phase.real * real_value - phase.imag * imag_value;
          operation.dft_real_imag[output + 1] +=
              phase.real * imag_value + phase.imag * real_value;
        }
      __syncthreads();
    }
  }
}

// Keep output_dft staging and the public dense materialization path on one
// explicit-rounding implementation. In particular, do not replace these
// intrinsics with ordinary expressions: --use_fast_math must not contract or
// approximate the CPU-compatible complex multiply/divide ordering.
__device__ __forceinline__ void condition_dft_sample_fp32(
    float source_real, float source_imaginary,
    complex_value_fp32 inverse_stored_weight, float point_weight,
    bool zero_divisor, float &output_real, float &output_imaginary) {
  const float weighted_real = __fsub_rn(
      __fmul_rn(source_real, inverse_stored_weight.real),
      __fmul_rn(source_imaginary, inverse_stored_weight.imag));
  const float weighted_imaginary = __fadd_rn(
      __fmul_rn(source_real, inverse_stored_weight.imag),
      __fmul_rn(source_imaginary, inverse_stored_weight.real));
  if (zero_divisor && weighted_real == 0.0f &&
      weighted_imaginary == 0.0f) {
    output_real = 0.0f;
    output_imaginary = 0.0f;
  }
  else if (zero_divisor) {
    // Match process_dft_component: divide before applying the retained
    // interpolation weight. point_weight==0 therefore preserves NaN.
    output_real =
        __fmul_rn(__fdiv_rn(weighted_real, 0.0f), point_weight);
    output_imaginary =
        __fmul_rn(__fdiv_rn(weighted_imaginary, 0.0f), point_weight);
  }
  else {
    output_real = __fmul_rn(weighted_real, point_weight);
    output_imaginary = __fmul_rn(weighted_imaginary, point_weight);
  }
}

__global__ void materialize_dft_fp32_kernel(
    const dft_materialization_operation_fp32 *operations,
    const std::uint32_t *block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count,
    float *output_real_imag, std::size_t output_point_count) {
  for (std::size_t logical_block = blockIdx.x;
       logical_block < total_block_count;
       logical_block += gridDim.x) {
    const std::uint32_t operation_index =
        block_operation_indices[logical_block];
    if (operation_index >= operation_count) continue;
    const dft_materialization_operation_fp32 operation =
        operations[operation_index];
    const std::size_t local_block =
        logical_block - operation.block_start;
    const std::size_t local_work =
        local_block * dft_materialization_threads_per_block_fp32 +
        threadIdx.x;
    const std::size_t operation_work =
        operation.point_count * operation.selected_frequency_count;
    if (local_work < operation_work) {
      const std::size_t point =
          local_work / operation.selected_frequency_count;
      if (operation.publication_flags &&
          operation.publication_flags[point] == 0)
        continue;
      const std::size_t selected_frequency =
          local_work % operation.selected_frequency_count;
      const std::size_t source_frequency =
          operation.source_frequency_start + selected_frequency;
      const std::size_t source_complex_index =
          point * operation.source_frequency_count + source_frequency;
      const std::size_t destination_point = static_cast<std::size_t>(
          operation.destination_indices[point]);
      const std::size_t output_complex_index =
          selected_frequency * output_point_count + destination_point;
      const float source_real =
          operation.dft_real_imag[2 * source_complex_index];
      const float source_imaginary =
          operation.dft_real_imag[2 * source_complex_index + 1];
      const float point_weight = operation.point_weights[point];
      const bool zero_divisor = operation.zero_divisor_flags &&
                                operation.zero_divisor_flags[point] != 0;
      condition_dft_sample_fp32(
          source_real, source_imaginary,
          operation.inverse_stored_weight, point_weight, zero_divisor,
          output_real_imag[2 * output_complex_index],
          output_real_imag[2 * output_complex_index + 1]);
    }
  }
}

__global__ void collapse_dft_array_fp32_kernel(
    const float *full_output_real_imag, float *reduced_output_real_imag,
    std::size_t full_output_point_count,
    std::size_t reduced_output_point_count,
    std::size_t output_frequency_count,
    dft_collapse_layout_fp32 layout) {
  const std::size_t work_count =
      reduced_output_point_count * output_frequency_count;
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t work =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       work < work_count; work += grid_stride) {
    const std::size_t frequency = work / reduced_output_point_count;
    const std::size_t reduced_point = work % reduced_output_point_count;

    std::size_t retained_coordinate[3] = {0, 0, 0};
    std::size_t remainder = reduced_point;
    for (int dim = static_cast<int>(layout.full_rank) - 1; dim >= 0;
         --dim) {
      if (layout.collapsed[dim] == 0) {
        retained_coordinate[dim] = remainder % layout.full_dims[dim];
        remainder /= layout.full_dims[dim];
      }
    }

    const std::size_t begin0 =
        layout.full_rank > 0 && layout.collapsed[0] == 0
            ? retained_coordinate[0]
            : 0;
    const std::size_t end0 =
        layout.full_rank > 0 && layout.collapsed[0] == 0
            ? begin0 + 1
            : (layout.full_rank > 0 ? layout.full_dims[0] : 1);
    const std::size_t begin1 =
        layout.full_rank > 1 && layout.collapsed[1] == 0
            ? retained_coordinate[1]
            : 0;
    const std::size_t end1 =
        layout.full_rank > 1 && layout.collapsed[1] == 0
            ? begin1 + 1
            : (layout.full_rank > 1 ? layout.full_dims[1] : 1);
    const std::size_t begin2 =
        layout.full_rank > 2 && layout.collapsed[2] == 0
            ? retained_coordinate[2]
            : 0;
    const std::size_t end2 =
        layout.full_rank > 2 && layout.collapsed[2] == 0
            ? begin2 + 1
            : (layout.full_rank > 2 ? layout.full_dims[2] : 1);

    const std::size_t stride0 =
        layout.full_rank > 1
            ? layout.full_dims[1] *
                  (layout.full_rank > 2 ? layout.full_dims[2] : 1)
            : 1;
    const std::size_t stride1 =
        layout.full_rank > 2 ? layout.full_dims[2] : 1;
    float sum_real = 0.0f;
    float sum_imaginary = 0.0f;
    for (std::size_t n0 = begin0; n0 < end0; ++n0)
      for (std::size_t n1 = begin1; n1 < end1; ++n1)
        for (std::size_t n2 = begin2; n2 < end2; ++n2) {
          const std::size_t full_point =
              n0 * stride0 + n1 * stride1 + n2;
          const std::size_t source =
              2 * (frequency * full_output_point_count + full_point);
          sum_real = __fadd_rn(sum_real, full_output_real_imag[source]);
          sum_imaginary =
              __fadd_rn(sum_imaginary,
                        full_output_real_imag[source + 1]);
        }
    const std::size_t destination =
        2 * (frequency * reduced_output_point_count + reduced_point);
    reduced_output_real_imag[destination] = sum_real;
    reduced_output_real_imag[destination + 1] = sum_imaginary;
  }
}

__global__ void stage_dft_output_fp32_kernel(
    const dft_output_staging_operation_fp32 *operations,
    const std::uint32_t *block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count,
    float *output_planar, std::size_t output_point_count,
    std::size_t tile_frequency_start, std::size_t tile_frequency_count,
    std::size_t frequency_capacity) {
  for (std::size_t logical_block = blockIdx.x;
       logical_block < total_block_count;
       logical_block += gridDim.x) {
    const std::uint32_t operation_index =
        block_operation_indices[logical_block];
    if (operation_index >= operation_count) continue;
    const dft_output_staging_operation_fp32 operation =
        operations[operation_index];
    const std::size_t local_block =
        logical_block - operation.block_start;
    const std::size_t local_work =
        local_block * dft_output_staging_threads_per_block_fp32 +
        threadIdx.x;
    const std::size_t operation_work =
        operation.point_count * frequency_capacity;
    if (local_work < operation_work) {
      const std::size_t point = local_work / frequency_capacity;
      const std::size_t tile_frequency = local_work % frequency_capacity;
      if (tile_frequency >= tile_frequency_count) continue;
      const std::size_t source_frequency =
          tile_frequency_start + tile_frequency;
      const std::size_t source_complex_index =
          point * operation.source_frequency_count + source_frequency;
      const std::size_t packed_point =
          operation.output_point_offset + static_cast<std::size_t>(
              operation.destination_indices[point]);
      const float source_real =
          operation.dft_real_imag[2 * source_complex_index];
      const float source_imaginary =
          operation.dft_real_imag[2 * source_complex_index + 1];
      float staged_real;
      float staged_imaginary;
      condition_dft_sample_fp32(
          source_real, source_imaginary,
          operation.inverse_stored_weight,
          operation.point_weights[point],
          operation.zero_divisor_flags &&
              operation.zero_divisor_flags[point] != 0,
          staged_real, staged_imaginary);
      const std::size_t plane_base =
          tile_frequency * 2 * output_point_count;
      output_planar[plane_base + packed_point] = staged_real;
      output_planar[plane_base + output_point_count + packed_point] =
          staged_imaginary;
    }
  }
}

__global__ void prepare_dft_phases_fp32_kernel(
    const double *angular_frequencies, complex_value_fp32 *phases,
    std::size_t frequency_count, double time, double scale_real,
    double scale_imag) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t frequency =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       frequency < frequency_count; frequency += grid_stride) {
    double sine;
    double cosine;
    sincos(angular_frequencies[frequency] * time, &sine, &cosine);
    phases[frequency].real =
        static_cast<float>(cosine * scale_real - sine * scale_imag);
    phases[frequency].imag =
        static_cast<float>(cosine * scale_imag + sine * scale_real);
  }
}

__device__ __forceinline__ complex_value_fp32 complex_add_fp32(
    complex_value_fp32 left, complex_value_fp32 right) {
  return {left.real + right.real, left.imag + right.imag};
}

__device__ __forceinline__ complex_value_fp32 complex_multiply_fp32(
    complex_value_fp32 left, complex_value_fp32 right) {
  return {left.real * right.real - left.imag * right.imag,
          left.real * right.imag + left.imag * right.real};
}

__device__ __forceinline__ complex_value_fp32 complex_scale_fp32(
    complex_value_fp32 value, float scale) {
  return {value.real * scale, value.imag * scale};
}

// The host cancellation proof uses the normal-FP32 gamma_n model. Zero is
// safe and common for Cartesian basis-vector components, but a nonzero
// subnormal can lose all relative accuracy when the device flushes it.
__device__ __forceinline__ bool
near2far_envelope_value_is_normal_or_zero(float value) {
  constexpr float minimum_normal_fp32 = 1.1754943508222875e-38f;
  return isfinite(value) &&
         (value == 0.0f || fabsf(value) >= minimum_normal_fp32);
}

__device__ __forceinline__ void near2far_validate_envelope_value(
    bool &valid, float value) {
  valid = valid && near2far_envelope_value_is_normal_or_zero(value);
}

__device__ __forceinline__ void near2far_validate_positive_envelope_value(
    bool &valid, float value) {
  constexpr float minimum_normal_fp32 = 1.1754943508222875e-38f;
  valid = valid && isfinite(value) && value >= minimum_normal_fp32;
}

__device__ __forceinline__ void near2far_validate_envelope_product(
    bool &valid, float left, float right, float product) {
  near2far_validate_envelope_value(valid, left);
  near2far_validate_envelope_value(valid, right);
  if (left == 0.0f || right == 0.0f)
    valid = valid && product == 0.0f;
  else
    near2far_validate_positive_envelope_value(valid, product);
}

__device__ __forceinline__ void near2far_validate_envelope_sum(
    bool &valid, float left, float right, float sum) {
  near2far_validate_envelope_value(valid, left);
  near2far_validate_envelope_value(valid, right);
  if (left == 0.0f && right == 0.0f)
    valid = valid && sum == 0.0f;
  else
    near2far_validate_positive_envelope_value(valid, sum);
}

__device__ __forceinline__ float near2far_invalid_envelope() {
  return __int_as_float(0x7f800000);
}

__device__ __forceinline__ void near2far_green3d_fp32(
    complex_value_fp32 fields[6], const cartesian_point_fp32 &target,
    float frequency, float eps, float mu,
    const cartesian_point_fp32 &source, int direction, bool electric,
    complex_value_fp32 amplitude, float *internal_absolute_envelope) {
  float rx = target.x - source.x;
  float ry = target.y - source.y;
  float rz = target.z - source.z;
  const float r = sqrtf(rx * rx + ry * ry + rz * rz);
  rx /= r;
  ry /= r;
  rz /= r;

  constexpr float pi_fp32 = 3.14159265358979323846f;
  const float refractive_index = sqrtf(eps * mu);
  const float k = 2.0f * pi_fp32 * frequency * refractive_index;
  const float kr = k * r;
  float phase_sine;
  float phase_cosine;
  sincosf(kr + 0.5f * pi_fp32, &phase_sine, &phase_cosine);
  complex_value_fp32 expfac = complex_multiply_fp32(
      amplitude,
      {k * refractive_index * phase_cosine / (4.0f * pi_fp32 * r),
       k * refractive_index * phase_sine / (4.0f * pi_fp32 * r)});
  const float impedance = sqrtf(mu / eps);

  const float px = direction == 0 ? 1.0f : 0.0f;
  const float py = direction == 1 ? 1.0f : 0.0f;
  const float pz = direction == 2 ? 1.0f : 0.0f;
  const float pdotr = px * rx + py * ry + pz * rz;
  const float cross_x = ry * pz - rz * py;
  const float cross_y = rz * px - rx * pz;
  const float cross_z = rx * py - ry * px;

  // 1/(i*kr) is exactly -i/kr and 1/(i*kr)^2 is -1/kr^2.
  const complex_value_fp32 term1 =
      {1.0f - 1.0f / (kr * kr), 1.0f / kr};
  const complex_value_fp32 term2 =
      {(-1.0f + 3.0f / (kr * kr)) * pdotr,
       -3.0f / kr * pdotr};
  const complex_value_fp32 term3 = {1.0f, 1.0f / kr};

  // Cancellation evidence must dominate the arithmetic before term1 and
  // term2 are combined. For p parallel to rhat at large kr their O(1) real
  // parts cancel, leaving an O(1/kr) field; measuring only that final field
  // underestimates FP32 rounding by O(kr). Use complex L1 product envelopes
  // for amplitude, polar phase, material scaling, and every Green coefficient.
  bool envelope_valid = true;
  near2far_validate_envelope_value(envelope_valid, amplitude.real);
  near2far_validate_envelope_value(envelope_valid, amplitude.imag);
  near2far_validate_positive_envelope_value(envelope_valid, r);
  near2far_validate_positive_envelope_value(
      envelope_valid, refractive_index);
  near2far_validate_envelope_value(envelope_valid, rx);
  near2far_validate_envelope_value(envelope_valid, ry);
  near2far_validate_envelope_value(envelope_valid, rz);
  near2far_validate_envelope_value(envelope_valid, pdotr);
  near2far_validate_envelope_value(envelope_valid, cross_x);
  near2far_validate_envelope_value(envelope_valid, cross_y);
  near2far_validate_envelope_value(envelope_valid, cross_z);
  near2far_validate_positive_envelope_value(envelope_valid, fabsf(k));
  near2far_validate_positive_envelope_value(envelope_valid, fabsf(kr));
  near2far_validate_envelope_value(envelope_valid, phase_cosine);
  near2far_validate_envelope_value(envelope_valid, phase_sine);
  near2far_validate_positive_envelope_value(envelope_valid, impedance);
  const float inverse_kr = fabsf(1.0f / kr);
  const float inverse_kr_squared = fabsf(1.0f / (kr * kr));
  const float term1_envelope = 1.0f + inverse_kr + inverse_kr_squared;
  const float term2_envelope =
      (1.0f + 3.0f * inverse_kr + 3.0f * inverse_kr_squared) *
      fabsf(pdotr);
  const float term3_envelope = 1.0f + inverse_kr;
  const float amplitude_envelope = fabsf(amplitude.real) + fabsf(amplitude.imag);
  const float polar_numerator = fabsf(k) * refractive_index;
  const float polar_denominator = 4.0f * pi_fp32 * r;
  const float polar_coefficient = polar_numerator / polar_denominator;
  const float polar_phase_envelope =
      fabsf(phase_cosine) + fabsf(phase_sine);
  const float polar_envelope = polar_coefficient * polar_phase_envelope;
  const float expfac_envelope = amplitude_envelope * polar_envelope;
  near2far_validate_positive_envelope_value(envelope_valid, inverse_kr);
  near2far_validate_positive_envelope_value(
      envelope_valid, inverse_kr_squared);
  near2far_validate_positive_envelope_value(envelope_valid, term1_envelope);
  near2far_validate_envelope_value(envelope_valid, term2_envelope);
  near2far_validate_positive_envelope_value(envelope_valid, term3_envelope);
  near2far_validate_envelope_sum(
      envelope_valid, fabsf(amplitude.real), fabsf(amplitude.imag),
      amplitude_envelope);
  near2far_validate_envelope_product(
      envelope_valid, fabsf(k), refractive_index, polar_numerator);
  near2far_validate_positive_envelope_value(
      envelope_valid, polar_denominator);
  if (polar_numerator == 0.0f)
    envelope_valid = false;
  else
    near2far_validate_positive_envelope_value(
        envelope_valid, polar_coefficient);
  near2far_validate_envelope_sum(
      envelope_valid, fabsf(phase_cosine), fabsf(phase_sine),
      polar_phase_envelope);
  near2far_validate_envelope_product(
      envelope_valid, polar_coefficient, polar_phase_envelope,
      polar_envelope);
  near2far_validate_envelope_product(
      envelope_valid, amplitude_envelope, polar_envelope,
      expfac_envelope);
  float maximum_envelope = 0.0f;
  const float p_values[3] = {px, py, pz};
  const float r_values[3] = {rx, ry, rz};
  const float cross_values[3] = {cross_x, cross_y, cross_z};
  if (electric) {
    const float scaled = expfac_envelope / fabsf(eps);
    near2far_validate_positive_envelope_value(envelope_valid, fabsf(eps));
    if (expfac_envelope == 0.0f)
      envelope_valid = envelope_valid && scaled == 0.0f;
    else
      near2far_validate_positive_envelope_value(envelope_valid, scaled);
    for (int axis = 0; axis < 3; ++axis) {
      const float term1_axis =
          term1_envelope * fabsf(p_values[axis]);
      const float term2_axis =
          term2_envelope * fabsf(r_values[axis]);
      const float summed_axis = term1_axis + term2_axis;
      const float electric_axis_envelope = scaled * summed_axis;
      const float magnetic_coefficient =
          term3_envelope * fabsf(cross_values[axis]);
      const float magnetic_axis_envelope =
          scaled * magnetic_coefficient / fabsf(impedance);
      near2far_validate_envelope_product(
          envelope_valid, term1_envelope, fabsf(p_values[axis]),
          term1_axis);
      near2far_validate_envelope_product(
          envelope_valid, term2_envelope, fabsf(r_values[axis]),
          term2_axis);
      near2far_validate_envelope_sum(
          envelope_valid, term1_axis, term2_axis, summed_axis);
      near2far_validate_envelope_product(
          envelope_valid, scaled, summed_axis, electric_axis_envelope);
      near2far_validate_envelope_product(
          envelope_valid, term3_envelope, fabsf(cross_values[axis]),
          magnetic_coefficient);
      if (scaled == 0.0f || magnetic_coefficient == 0.0f)
        envelope_valid = envelope_valid && magnetic_axis_envelope == 0.0f;
      else
        near2far_validate_positive_envelope_value(
            envelope_valid, magnetic_axis_envelope);
      maximum_envelope =
          fmaxf(maximum_envelope, electric_axis_envelope);
      maximum_envelope =
          fmaxf(maximum_envelope, magnetic_axis_envelope);
      near2far_validate_envelope_value(envelope_valid, maximum_envelope);
    }
  }
  else {
    const float scaled = expfac_envelope / fabsf(mu);
    near2far_validate_positive_envelope_value(envelope_valid, fabsf(mu));
    if (expfac_envelope == 0.0f)
      envelope_valid = envelope_valid && scaled == 0.0f;
    else
      near2far_validate_positive_envelope_value(envelope_valid, scaled);
    for (int axis = 0; axis < 3; ++axis) {
      const float cross_coefficient =
          term3_envelope * fabsf(cross_values[axis]);
      const float electric_intermediate = scaled * cross_coefficient;
      const float electric_axis_envelope =
          electric_intermediate * fabsf(impedance);
      const float term1_axis =
          term1_envelope * fabsf(p_values[axis]);
      const float term2_axis =
          term2_envelope * fabsf(r_values[axis]);
      const float summed_axis = term1_axis + term2_axis;
      const float magnetic_axis_envelope = scaled * summed_axis;
      near2far_validate_envelope_product(
          envelope_valid, term3_envelope, fabsf(cross_values[axis]),
          cross_coefficient);
      near2far_validate_envelope_product(
          envelope_valid, scaled, cross_coefficient,
          electric_intermediate);
      near2far_validate_envelope_product(
          envelope_valid, electric_intermediate, fabsf(impedance),
          electric_axis_envelope);
      near2far_validate_envelope_product(
          envelope_valid, term1_envelope, fabsf(p_values[axis]),
          term1_axis);
      near2far_validate_envelope_product(
          envelope_valid, term2_envelope, fabsf(r_values[axis]),
          term2_axis);
      near2far_validate_envelope_sum(
          envelope_valid, term1_axis, term2_axis, summed_axis);
      near2far_validate_envelope_product(
          envelope_valid, scaled, summed_axis, magnetic_axis_envelope);
      maximum_envelope =
          fmaxf(maximum_envelope, electric_axis_envelope);
      maximum_envelope =
          fmaxf(maximum_envelope, magnetic_axis_envelope);
      near2far_validate_envelope_value(envelope_valid, maximum_envelope);
    }
  }
  *internal_absolute_envelope =
      envelope_valid ? maximum_envelope : near2far_invalid_envelope();

  if (electric) {
    expfac = complex_scale_fp32(expfac, 1.0f / eps);
    fields[0] = complex_multiply_fp32(
        expfac, complex_add_fp32(complex_scale_fp32(term1, px),
                                 complex_scale_fp32(term2, rx)));
    fields[1] = complex_multiply_fp32(
        expfac, complex_add_fp32(complex_scale_fp32(term1, py),
                                 complex_scale_fp32(term2, ry)));
    fields[2] = complex_multiply_fp32(
        expfac, complex_add_fp32(complex_scale_fp32(term1, pz),
                                 complex_scale_fp32(term2, rz)));
    fields[3] = complex_scale_fp32(
        complex_multiply_fp32(expfac, term3), cross_x / impedance);
    fields[4] = complex_scale_fp32(
        complex_multiply_fp32(expfac, term3), cross_y / impedance);
    fields[5] = complex_scale_fp32(
        complex_multiply_fp32(expfac, term3), cross_z / impedance);
  }
  else {
    expfac = complex_scale_fp32(expfac, 1.0f / mu);
    fields[0] = complex_scale_fp32(
        complex_multiply_fp32(expfac, term3), -cross_x * impedance);
    fields[1] = complex_scale_fp32(
        complex_multiply_fp32(expfac, term3), -cross_y * impedance);
    fields[2] = complex_scale_fp32(
        complex_multiply_fp32(expfac, term3), -cross_z * impedance);
    fields[3] = complex_multiply_fp32(
        expfac, complex_add_fp32(complex_scale_fp32(term1, px),
                                 complex_scale_fp32(term2, rx)));
    fields[4] = complex_multiply_fp32(
        expfac, complex_add_fp32(complex_scale_fp32(term1, py),
                                 complex_scale_fp32(term2, ry)));
    fields[5] = complex_multiply_fp32(
        expfac, complex_add_fp32(complex_scale_fp32(term1, pz),
                                 complex_scale_fp32(term2, rz)));
  }
}

__device__ __forceinline__ void near2far_green2d_fp32(
    complex_value_fp32 fields[6], const cartesian_point_fp32 &target,
    float frequency, float eps, float mu,
    const cartesian_point_fp32 &source, int direction, bool electric,
    complex_value_fp32 amplitude, float maximum_kr_input_error,
    float *internal_absolute_envelope) {
  for (int component = 0; component < 6; ++component)
    fields[component] = {0.0f, 0.0f};

  float rx = target.x - source.x;
  float ry = target.y - source.y;
  const float r = sqrtf(rx * rx + ry * ry);
  rx /= r;
  ry /= r;

  constexpr float pi_fp32 = 3.14159265358979323846f;
  const float omega = 2.0f * pi_fp32 * frequency;
  const float k = omega * sqrtf(eps * mu);
  const float kr = k * r;
  const float impedance = sqrtf(mu / eps);
  const float j0_value = j0f(kr);
  const float y0_value = y0f(kr);
  const float j1_value = j1f(kr);
  const float y1_value = y1f(kr);
  // H2 = 2 H1 / kr - H0. Reusing the already evaluated order-zero/one
  // functions avoids two additional special-function calls per interaction.
  const float j2_value = 2.0f * j1_value / kr - j0_value;
  const float y2_value = 2.0f * y1_value / kr - y0_value;
  const complex_value_fp32 h0 = complex_multiply_fp32(
      {j0_value, y0_value}, amplitude);
  const complex_value_fp32 h1 = complex_multiply_fp32(
      {j1_value, y1_value}, amplitude);
  const complex_value_fp32 h2 = complex_multiply_fp32(
      {j2_value, y2_value}, amplitude);
  const complex_value_fp32 ik_h1 =
      {-0.25f * k * h1.imag, 0.25f * k * h1.real};

  const float px = direction == 0 ? 1.0f : 0.0f;
  const float py = direction == 1 ? 1.0f : 0.0f;
  const float pdotr = px * rx + py * ry;
  const float cross = rx * py - ry * px;
  const complex_value_fp32 h0_minus_h2 =
      {h0.real - h2.real, h0.imag - h2.imag};

  if (direction == 2) {
    if (electric) {
      fields[2] = complex_scale_fp32(h0, -0.25f * omega * mu);
      fields[3] = complex_scale_fp32(ik_h1, -ry);
      fields[4] = complex_scale_fp32(ik_h1, rx);
    }
    else {
      fields[0] = complex_scale_fp32(ik_h1, ry);
      fields[1] = complex_scale_fp32(ik_h1, -rx);
      fields[5] = complex_scale_fp32(h0, -0.25f * omega * eps);
    }
  }
  else if (electric) {
    const float longitudinal = pdotr / r * 0.25f * impedance;
    const float transverse = cross * omega * mu * 0.125f;
    fields[0] = complex_add_fp32(
        complex_scale_fp32(h1, -rx * longitudinal),
        complex_scale_fp32(h0_minus_h2, ry * transverse));
    fields[1] = complex_add_fp32(
        complex_scale_fp32(h1, -ry * longitudinal),
        complex_scale_fp32(h0_minus_h2, -rx * transverse));
    fields[5] = complex_scale_fp32(ik_h1, -cross);
  }
  else {
    const float longitudinal = pdotr / r * 0.25f / impedance;
    const float transverse = cross * omega * eps * 0.125f;
    fields[2] = complex_scale_fp32(ik_h1, cross);
    fields[3] = complex_add_fp32(
        complex_scale_fp32(h1, -rx * longitudinal),
        complex_scale_fp32(h0_minus_h2, ry * transverse));
    fields[4] = complex_add_fp32(
        complex_scale_fp32(h1, -ry * longitudinal),
        complex_scale_fp32(h0_minus_h2, -rx * transverse));
  }

  // Bound the arithmetic before H0-H2 and longitudinal/transverse terms are
  // combined.  The host adds a separate CUDA special-function error budget;
  // this channel is the per-interaction L1 scale used by that bound.
  bool envelope_valid = true;
  const float amplitude_envelope =
      fabsf(amplitude.real) + fabsf(amplitude.imag);
  const float h0_basis_envelope = fabsf(j0_value) + fabsf(y0_value);
  const float h1_basis_envelope = fabsf(j1_value) + fabsf(y1_value);
  const float h2_basis_envelope = fabsf(j2_value) + fabsf(y2_value);
  // CUDA documents 9 ULP below |x|=8 and 2.2e-6 maximum absolute
  // error otherwise for the order-zero/one functions. Propagate those bounds
  // through the H2 recurrence, then inflate the L1 scale by
  // error/special_relative_scale. The host adds that exact scale to its
  // contribution error factor, so factor*inflated_scale bounds both ordinary
  // arithmetic error and this absolute special-function error.
  constexpr float special_relative_scale = 1.0e-5f;
  constexpr float maximum_bessel_absolute_error = 2.2e-6f;
  constexpr float bessel_ulp_relative_bound =
      18.0f * 1.1920928955078125e-7f;
  const float h0_function_error =
      bessel_ulp_relative_bound * h0_basis_envelope +
      2.0f * maximum_bessel_absolute_error;
  const float h1_function_error =
      bessel_ulp_relative_bound * h1_basis_envelope +
      2.0f * maximum_bessel_absolute_error;
  // The CUDA accuracy table bounds evaluation at the FP32 argument.  The
  // integrated caller additionally supplies a conservative bound on the
  // FP64-to-FP32 input perturbation.  Account for the arithmetic which forms
  // kr in this kernel as well.  For H0'=-H1 and H1'=H0-H1/x, Gronwall gives
  // S(t)<=S(x) exp((1+1/x_min)|t-x|), S=|H0|_1+|H1|_1.  This bounds the whole
  // interval between the reference and computed arguments rather than only a
  // derivative sampled at one endpoint.  H2 is then bounded through its
  // exact recurrence, including the perturbed reciprocal argument.
  constexpr float fp32_unit_roundoff = 1.1920928955078125e-7f;
  const float kr_arithmetic_error =
      16.0f * fp32_unit_roundoff * fabsf(kr);
  const float kr_argument_error =
      maximum_kr_input_error + kr_arithmetic_error;
  const float minimum_kr = kr - kr_argument_error;
  const float h01_at_computed_argument =
      h0_basis_envelope + h0_function_error +
      h1_basis_envelope + h1_function_error;
  const float derivative_growth_rate = 1.0f + 1.0f / minimum_kr;
  const float interval_h01_envelope =
      h01_at_computed_argument *
      expf(derivative_growth_rate * kr_argument_error);
  const float h0_argument_error =
      kr_argument_error * interval_h01_envelope;
  const float h1_argument_error =
      kr_argument_error * derivative_growth_rate * interval_h01_envelope;
  const float h0_basis_error = h0_function_error + h0_argument_error;
  const float h1_basis_error = h1_function_error + h1_argument_error;
  const float h2_basis_error =
      h0_basis_error + 2.0f / minimum_kr * h1_basis_error +
      2.0f * h1_basis_envelope * kr_argument_error /
          (minimum_kr * kr);
  const float h0_proof_basis =
      h0_basis_envelope + h0_basis_error / special_relative_scale;
  const float h1_proof_basis =
      h1_basis_envelope + h1_basis_error / special_relative_scale;
  const float h2_proof_basis =
      h2_basis_envelope + h2_basis_error / special_relative_scale;
  const float h0_envelope = amplitude_envelope * h0_proof_basis;
  const float h1_envelope = amplitude_envelope * h1_proof_basis;
  const float h2_envelope = amplitude_envelope * h2_proof_basis;
  const float h0_minus_h2_envelope = h0_envelope + h2_envelope;
  const float ik_h1_envelope = 0.25f * fabsf(k) * h1_envelope;
  near2far_validate_envelope_value(envelope_valid, amplitude.real);
  near2far_validate_envelope_value(envelope_valid, amplitude.imag);
  near2far_validate_positive_envelope_value(envelope_valid, r);
  near2far_validate_envelope_value(envelope_valid, rx);
  near2far_validate_envelope_value(envelope_valid, ry);
  near2far_validate_positive_envelope_value(envelope_valid, fabsf(omega));
  near2far_validate_positive_envelope_value(envelope_valid, fabsf(k));
  near2far_validate_positive_envelope_value(envelope_valid, fabsf(kr));
  near2far_validate_envelope_value(
      envelope_valid, maximum_kr_input_error);
  envelope_valid = envelope_valid && maximum_kr_input_error >= 0.0f;
  near2far_validate_positive_envelope_value(envelope_valid, minimum_kr);
  near2far_validate_positive_envelope_value(
      envelope_valid, derivative_growth_rate);
  near2far_validate_positive_envelope_value(
      envelope_valid, interval_h01_envelope);
  near2far_validate_positive_envelope_value(envelope_valid, impedance);
  near2far_validate_envelope_sum(
      envelope_valid, fabsf(amplitude.real), fabsf(amplitude.imag),
      amplitude_envelope);
  near2far_validate_envelope_sum(
      envelope_valid, fabsf(j0_value), fabsf(y0_value), h0_basis_envelope);
  near2far_validate_envelope_sum(
      envelope_valid, fabsf(j1_value), fabsf(y1_value), h1_basis_envelope);
  near2far_validate_envelope_sum(
      envelope_valid, fabsf(j2_value), fabsf(y2_value), h2_basis_envelope);
  near2far_validate_positive_envelope_value(
      envelope_valid, h0_basis_error);
  near2far_validate_positive_envelope_value(
      envelope_valid, h1_basis_error);
  near2far_validate_positive_envelope_value(
      envelope_valid, h2_basis_error);
  near2far_validate_envelope_sum(
      envelope_valid, h0_basis_envelope,
      h0_basis_error / special_relative_scale, h0_proof_basis);
  near2far_validate_envelope_sum(
      envelope_valid, h1_basis_envelope,
      h1_basis_error / special_relative_scale, h1_proof_basis);
  near2far_validate_envelope_sum(
      envelope_valid, h2_basis_envelope,
      h2_basis_error / special_relative_scale, h2_proof_basis);
  near2far_validate_envelope_product(
      envelope_valid, amplitude_envelope, h0_proof_basis, h0_envelope);
  near2far_validate_envelope_product(
      envelope_valid, amplitude_envelope, h1_proof_basis, h1_envelope);
  near2far_validate_envelope_product(
      envelope_valid, amplitude_envelope, h2_proof_basis, h2_envelope);
  near2far_validate_envelope_sum(
      envelope_valid, h0_envelope, h2_envelope,
      h0_minus_h2_envelope);
  near2far_validate_envelope_product(
      envelope_valid, 0.25f * fabsf(k), h1_envelope,
      ik_h1_envelope);

  float maximum_envelope = 0.0f;
  const auto record_product = [&envelope_valid, &maximum_envelope](
      float left, float right) {
    const float product = left * right;
    near2far_validate_envelope_product(
        envelope_valid, left, right, product);
    maximum_envelope = fmaxf(maximum_envelope, product);
  };
  if (direction == 2) {
    record_product(0.25f * fabsf(omega) * (electric ? mu : eps),
                   h0_envelope);
    record_product(fabsf(rx), ik_h1_envelope);
    record_product(fabsf(ry), ik_h1_envelope);
  }
  else {
    const float longitudinal =
        fabsf(pdotr / r * 0.25f *
              (electric ? impedance : 1.0f / impedance));
    const float transverse =
        fabsf(cross * omega * (electric ? mu : eps) * 0.125f);
    const float longitudinal_envelope = longitudinal * h1_envelope;
    const float transverse_envelope =
        transverse * h0_minus_h2_envelope;
    near2far_validate_envelope_product(
        envelope_valid, longitudinal, h1_envelope,
        longitudinal_envelope);
    near2far_validate_envelope_product(
        envelope_valid, transverse, h0_minus_h2_envelope,
        transverse_envelope);
    const float x_envelope =
        fabsf(rx) * longitudinal_envelope +
        fabsf(ry) * transverse_envelope;
    const float y_envelope =
        fabsf(ry) * longitudinal_envelope +
        fabsf(rx) * transverse_envelope;
    near2far_validate_envelope_sum(
        envelope_valid, fabsf(rx) * longitudinal_envelope,
        fabsf(ry) * transverse_envelope, x_envelope);
    near2far_validate_envelope_sum(
        envelope_valid, fabsf(ry) * longitudinal_envelope,
        fabsf(rx) * transverse_envelope, y_envelope);
    maximum_envelope = fmaxf(maximum_envelope, x_envelope);
    maximum_envelope = fmaxf(maximum_envelope, y_envelope);
    record_product(fabsf(cross), ik_h1_envelope);
  }
  near2far_validate_envelope_value(envelope_valid, maximum_envelope);
  *internal_absolute_envelope =
      envelope_valid ? maximum_envelope : near2far_invalid_envelope();
}

__device__ __forceinline__ complex_value_fp64 complex_add_fp64(
    complex_value_fp64 left, complex_value_fp64 right) {
  return {left.real + right.real, left.imag + right.imag};
}

__device__ __forceinline__ complex_value_fp64 complex_multiply_fp64(
    complex_value_fp64 left, complex_value_fp64 right) {
  return {left.real * right.real - left.imag * right.imag,
          left.real * right.imag + left.imag * right.real};
}

__device__ __forceinline__ complex_value_fp64 complex_scale_fp64(
    complex_value_fp64 value, double scale) {
  return {value.real * scale, value.imag * scale};
}

__device__ __forceinline__ void near2far_green3d_mixed_fp32(
    complex_value_fp64 fields[6], const cartesian_point_fp64 &target,
    double frequency, double eps, double mu,
    const cartesian_point_fp64 &source, int direction, bool electric,
    complex_value_fp64 amplitude) {
  double rx = target.x - source.x;
  double ry = target.y - source.y;
  double rz = target.z - source.z;
  const double r = sqrt(rx * rx + ry * ry + rz * rz);
  rx /= r;
  ry /= r;
  rz /= r;

  constexpr double pi_fp64 = 3.141592653589793238462643383279502884;
  const double refractive_index = sqrt(eps * mu);
  const double k = 2.0 * pi_fp64 * frequency * refractive_index;
  const double kr = k * r;
  double phase_sine;
  double phase_cosine;
  sincos(kr + 0.5 * pi_fp64, &phase_sine, &phase_cosine);
  complex_value_fp64 expfac = complex_multiply_fp64(
      amplitude,
      {k * refractive_index * phase_cosine / (4.0 * pi_fp64 * r),
       k * refractive_index * phase_sine / (4.0 * pi_fp64 * r)});
  const double impedance = sqrt(mu / eps);

  const double px = direction == 0 ? 1.0 : 0.0;
  const double py = direction == 1 ? 1.0 : 0.0;
  const double pz = direction == 2 ? 1.0 : 0.0;
  const double pdotr = px * rx + py * ry + pz * rz;
  const double cross_x = ry * pz - rz * py;
  const double cross_y = rz * px - rx * pz;
  const double cross_z = rx * py - ry * px;
  const complex_value_fp64 term1 =
      {1.0 - 1.0 / (kr * kr), 1.0 / kr};
  const complex_value_fp64 term2 =
      {(-1.0 + 3.0 / (kr * kr)) * pdotr,
       -3.0 / kr * pdotr};
  const complex_value_fp64 term3 = {1.0, 1.0 / kr};

  if (electric) {
    expfac = complex_scale_fp64(expfac, 1.0 / eps);
    fields[0] = complex_multiply_fp64(
        expfac, complex_add_fp64(complex_scale_fp64(term1, px),
                                 complex_scale_fp64(term2, rx)));
    fields[1] = complex_multiply_fp64(
        expfac, complex_add_fp64(complex_scale_fp64(term1, py),
                                 complex_scale_fp64(term2, ry)));
    fields[2] = complex_multiply_fp64(
        expfac, complex_add_fp64(complex_scale_fp64(term1, pz),
                                 complex_scale_fp64(term2, rz)));
    fields[3] = complex_scale_fp64(
        complex_multiply_fp64(expfac, term3), cross_x / impedance);
    fields[4] = complex_scale_fp64(
        complex_multiply_fp64(expfac, term3), cross_y / impedance);
    fields[5] = complex_scale_fp64(
        complex_multiply_fp64(expfac, term3), cross_z / impedance);
  }
  else {
    expfac = complex_scale_fp64(expfac, 1.0 / mu);
    fields[0] = complex_scale_fp64(
        complex_multiply_fp64(expfac, term3), -cross_x * impedance);
    fields[1] = complex_scale_fp64(
        complex_multiply_fp64(expfac, term3), -cross_y * impedance);
    fields[2] = complex_scale_fp64(
        complex_multiply_fp64(expfac, term3), -cross_z * impedance);
    fields[3] = complex_multiply_fp64(
        expfac, complex_add_fp64(complex_scale_fp64(term1, px),
                                 complex_scale_fp64(term2, rx)));
    fields[4] = complex_multiply_fp64(
        expfac, complex_add_fp64(complex_scale_fp64(term1, py),
                                 complex_scale_fp64(term2, ry)));
    fields[5] = complex_multiply_fp64(
        expfac, complex_add_fp64(complex_scale_fp64(term1, pz),
                                 complex_scale_fp64(term2, rz)));
  }
}

__device__ __forceinline__ void near2far_green2d_mixed_fp32(
    complex_value_fp64 fields[6], const cartesian_point_fp64 &target,
    double frequency, double eps, double mu,
    const cartesian_point_fp64 &source, int direction, bool electric,
    complex_value_fp64 amplitude) {
  for (int component = 0; component < 6; ++component)
    fields[component] = {0.0, 0.0};

  double rx = target.x - source.x;
  double ry = target.y - source.y;
  const double r = sqrt(rx * rx + ry * ry);
  rx /= r;
  ry /= r;

  constexpr double pi_fp64 =
      3.141592653589793238462643383279502884;
  const double omega = 2.0 * pi_fp64 * frequency;
  const double k = omega * sqrt(eps * mu);
  const double kr = k * r;
  const double impedance = sqrt(mu / eps);
  const complex_value_fp64 h0 = complex_multiply_fp64(
      {j0(kr), y0(kr)}, amplitude);
  const complex_value_fp64 h1 = complex_multiply_fp64(
      {j1(kr), y1(kr)}, amplitude);
  const complex_value_fp64 h2 = complex_multiply_fp64(
      {jn(2, kr), yn(2, kr)}, amplitude);
  const complex_value_fp64 ik_h1 =
      {-0.25 * k * h1.imag, 0.25 * k * h1.real};

  const double px = direction == 0 ? 1.0 : 0.0;
  const double py = direction == 1 ? 1.0 : 0.0;
  const double pdotr = px * rx + py * ry;
  const double cross = rx * py - ry * px;
  const complex_value_fp64 h0_minus_h2 =
      {h0.real - h2.real, h0.imag - h2.imag};

  if (direction == 2) {
    if (electric) {
      fields[2] = complex_scale_fp64(h0, -0.25 * omega * mu);
      fields[3] = complex_scale_fp64(ik_h1, -ry);
      fields[4] = complex_scale_fp64(ik_h1, rx);
    }
    else {
      fields[0] = complex_scale_fp64(ik_h1, ry);
      fields[1] = complex_scale_fp64(ik_h1, -rx);
      fields[5] = complex_scale_fp64(h0, -0.25 * omega * eps);
    }
  }
  else if (electric) {
    const double longitudinal = pdotr / r * 0.25 * impedance;
    const double transverse = cross * omega * mu * 0.125;
    fields[0] = complex_add_fp64(
        complex_scale_fp64(h1, -rx * longitudinal),
        complex_scale_fp64(h0_minus_h2, ry * transverse));
    fields[1] = complex_add_fp64(
        complex_scale_fp64(h1, -ry * longitudinal),
        complex_scale_fp64(h0_minus_h2, -rx * transverse));
    fields[5] = complex_scale_fp64(ik_h1, -cross);
  }
  else {
    const double longitudinal = pdotr / r * 0.25 / impedance;
    const double transverse = cross * omega * eps * 0.125;
    fields[2] = complex_scale_fp64(ik_h1, cross);
    fields[3] = complex_add_fp64(
        complex_scale_fp64(h1, -rx * longitudinal),
        complex_scale_fp64(h0_minus_h2, ry * transverse));
    fields[4] = complex_add_fp64(
        complex_scale_fp64(h1, -ry * longitudinal),
        complex_scale_fp64(h0_minus_h2, -rx * transverse));
  }
}

// Cylindrical Green's function evaluated by the same nested trapezoidal
// quadrature as libmeep's greencyl. One CUDA thread owns one original
// source/copy/target/frequency interaction; the many independent source
// interactions provide the parallelism while each thread preserves the
// adaptive reuse order of the CPU algorithm.
__device__ __forceinline__ void near2far_greencyl_fp32(
    complex_value_fp32 fields[6],
    const cartesian_point_fp32 &target_cartesian,
    float frequency, float eps, float mu,
    const cartesian_point_fp32 &source_rz, int direction, bool electric,
    complex_value_fp32 amplitude, float azimuthal_mode, float tolerance,
    float *internal_absolute_envelope) {
  for (int component = 0; component < 6; ++component)
    fields[component] = {0.0f, 0.0f};
  float accumulated_absolute = 0.0f;
  float accumulated_envelope = 0.0f;
  constexpr int maximum_quadrature_points = 65536;
  constexpr float pi_fp32 = 3.14159265358979323846f;
  constexpr float proof_relative_scale = 1.0e-5f;
  constexpr float fp32_unit_roundoff = 1.1920928955078125e-7f;
  const int initial_points =
      16 + static_cast<int>(4.0f * fabsf(azimuthal_mode));
  float dphi = 2.0f / static_cast<float>(initial_points);
  bool converged = false;
  int final_points = initial_points;

  for (int point_count = initial_points;
       point_count <= maximum_quadrature_points;) {
    final_points = point_count;
    dphi *= 0.5f;
    const float angular_step = dphi * (2.0f * pi_fp32);
    complex_value_fp32 next[6];
    for (int component = 0; component < 6; ++component)
      next[component] = complex_scale_fp32(fields[component], 0.5f);
    float next_absolute = 0.5f * accumulated_absolute;
    float next_envelope = 0.5f * accumulated_envelope;
    const int first = point_count > initial_points ? 1 : 0;
    const int stride = point_count > initial_points ? 2 : 1;
    for (int index = first; index < point_count; index += stride) {
      const float phi = static_cast<float>(index) * angular_step;
      float sine_phi;
      float cosine_phi;
      sincosf(phi, &sine_phi, &cosine_phi);
      float sine_mode;
      float cosine_mode;
      sincosf(azimuthal_mode * phi, &sine_mode, &cosine_mode);
      complex_value_fp32 weighted_amplitude = complex_scale_fp32(
          complex_multiply_fp32(
              amplitude, {cosine_mode, sine_mode}),
          dphi);
      const cartesian_point_fp32 source = {
          source_rz.x * cosine_phi,
          source_rz.x * sine_phi,
          source_rz.z};
      const int call_count = direction == 2 ? 1 : 2;
      for (int call = 0; call < call_count; ++call) {
        int cartesian_direction = direction;
        float rotation = 1.0f;
        if (direction == 0) {
          cartesian_direction = call == 0 ? 0 : 1;
          rotation = call == 0 ? cosine_phi : sine_phi;
        }
        else if (direction == 1) {
          cartesian_direction = call == 0 ? 0 : 1;
          rotation = call == 0 ? -sine_phi : cosine_phi;
        }
        complex_value_fp32 contribution[6];
        float contribution_envelope = 0.0f;
        near2far_green3d_fp32(
            contribution, target_cartesian, frequency, eps, mu, source,
            cartesian_direction, electric,
            complex_scale_fp32(weighted_amplitude, rotation),
            &contribution_envelope);
        for (int component = 0; component < 6; ++component) {
          next[component] =
              complex_add_fp32(next[component], contribution[component]);
          next_absolute += hypotf(contribution[component].real,
                                  contribution[component].imag);
        }
        next_envelope += contribution_envelope;
      }
    }
    float difference = 0.0f;
    for (int component = 0; component < 6; ++component) {
      difference += hypotf(fields[component].real - next[component].real,
                           fields[component].imag - next[component].imag);
      fields[component] = next[component];
    }
    accumulated_absolute = next_absolute;
    accumulated_envelope = next_envelope;
    // This is the same successive-refinement criterion as CPU greencyl.  The
    // host publishes the corresponding truncation allowance against the
    // compact maximum-component envelope with an explicit factor of six for
    // this six-component L1 sum.  Nonconvergence reaches the invalid-envelope
    // path below and forces mixed CUDA rather than publishing a fast result.
    if (difference <= accumulated_absolute * tolerance) {
      converged = true;
      break;
    }
    if (point_count > maximum_quadrature_points / 2) break;
    point_count *= 2;
  }

  // Convert the quadrature's repeated FP32 additions into the same absolute
  // evidence channel used for Green arithmetic. The host includes
  // proof_relative_scale in its contribution factor; inflating the L1 scale
  // by gamma/proof_relative_scale therefore publishes the quadrature error
  // without adding another MPI reduction channel.
  const float rounding_numerator =
      (32.0f + 12.0f * static_cast<float>(final_points)) *
      fp32_unit_roundoff;
  const float rounding_gamma =
      rounding_numerator < 0.5f
          ? rounding_numerator / (1.0f - rounding_numerator)
          : near2far_invalid_envelope();
  const float proof_inflation =
      1.0f + rounding_gamma / proof_relative_scale;
  const float proven_envelope = accumulated_envelope * proof_inflation;
  bool valid = converged;
  near2far_validate_envelope_value(valid, accumulated_absolute);
  near2far_validate_positive_envelope_value(valid, tolerance);
  near2far_validate_positive_envelope_value(valid, proof_inflation);
  near2far_validate_envelope_product(
      valid, accumulated_envelope, proof_inflation, proven_envelope);
  *internal_absolute_envelope =
      valid ? proven_envelope : near2far_invalid_envelope();
}

__device__ __forceinline__ void near2far_greencyl_mixed_fp32(
    complex_value_fp64 fields[6],
    const cartesian_point_fp64 &target_cartesian,
    double frequency, double eps, double mu,
    const cartesian_point_fp64 &source_rz, int direction, bool electric,
    complex_value_fp64 amplitude, double azimuthal_mode, double tolerance) {
  for (int component = 0; component < 6; ++component)
    fields[component] = {0.0, 0.0};
  double accumulated_absolute = 0.0;
  constexpr int maximum_quadrature_points = 65536;
  constexpr double pi_fp64 =
      3.141592653589793238462643383279502884;
  const int initial_points =
      16 + static_cast<int>(4.0 * fabs(azimuthal_mode));
  double dphi = 2.0 / static_cast<double>(initial_points);
  for (int point_count = initial_points;
       point_count <= maximum_quadrature_points;) {
    dphi *= 0.5;
    const double angular_step = dphi * (2.0 * pi_fp64);
    complex_value_fp64 next[6];
    for (int component = 0; component < 6; ++component)
      next[component] = complex_scale_fp64(fields[component], 0.5);
    double next_absolute = 0.5 * accumulated_absolute;
    const int first = point_count > initial_points ? 1 : 0;
    const int stride = point_count > initial_points ? 2 : 1;
    for (int index = first; index < point_count; index += stride) {
      const double phi = static_cast<double>(index) * angular_step;
      double sine_phi;
      double cosine_phi;
      sincos(phi, &sine_phi, &cosine_phi);
      double sine_mode;
      double cosine_mode;
      sincos(azimuthal_mode * phi, &sine_mode, &cosine_mode);
      complex_value_fp64 weighted_amplitude = complex_scale_fp64(
          complex_multiply_fp64(
              amplitude, {cosine_mode, sine_mode}),
          dphi);
      const cartesian_point_fp64 source = {
          source_rz.x * cosine_phi,
          source_rz.x * sine_phi,
          source_rz.z};
      const int call_count = direction == 2 ? 1 : 2;
      for (int call = 0; call < call_count; ++call) {
        int cartesian_direction = direction;
        double rotation = 1.0;
        if (direction == 0) {
          cartesian_direction = call == 0 ? 0 : 1;
          rotation = call == 0 ? cosine_phi : sine_phi;
        }
        else if (direction == 1) {
          cartesian_direction = call == 0 ? 0 : 1;
          rotation = call == 0 ? -sine_phi : cosine_phi;
        }
        complex_value_fp64 contribution[6];
        near2far_green3d_mixed_fp32(
            contribution, target_cartesian, frequency, eps, mu, source,
            cartesian_direction, electric,
            complex_scale_fp64(weighted_amplitude, rotation));
        for (int component = 0; component < 6; ++component) {
          next[component] =
              complex_add_fp64(next[component], contribution[component]);
          next_absolute += hypot(contribution[component].real,
                                 contribution[component].imag);
        }
      }
    }
    double difference = 0.0;
    for (int component = 0; component < 6; ++component) {
      difference += hypot(fields[component].real - next[component].real,
                          fields[component].imag - next[component].imag);
      fields[component] = next[component];
    }
    accumulated_absolute = next_absolute;
    if (difference <= accumulated_absolute * tolerance) break;
    if (point_count > maximum_quadrature_points / 2) break;
    point_count *= 2;
  }
}

template <near2far_cartesian_dimension Dimension>
__global__ void near2far_3d_partials_fp32_kernel(
    const near2far_operation_fp32 *operations,
    std::size_t operation_count, const cartesian_point_fp32 *targets,
    std::size_t target_count, const float *frequencies,
    std::size_t frequency_count, std::size_t frequency_offset,
    const near2far_periodic_copy_fp32 *periodic_copies,
    std::size_t periodic_copy_count, float eps, float mu,
    std::size_t partial_count, std::size_t logical_block_count,
    float maximum_kr_input_error, float azimuthal_mode,
    float greencyl_tolerance, float *partials) {
  extern __shared__ float shared_components[];
  const std::size_t work_count = target_count * frequency_count;
  const std::size_t blocks_per_operation = work_count * partial_count;
  for (std::size_t logical_block = blockIdx.x;
       logical_block < logical_block_count; logical_block += gridDim.x) {
    const std::size_t operation_index =
        logical_block / blocks_per_operation;
    const std::size_t within_operation =
        logical_block % blocks_per_operation;
    const std::size_t work = within_operation / partial_count;
    const std::size_t partial_index = within_operation % partial_count;
    const std::size_t target_index = work / frequency_count;
    const std::size_t frequency_index = work % frequency_count;
    const near2far_operation_fp32 operation = operations[operation_index];
    const std::size_t interaction_count =
        operation.point_count * periodic_copy_count;
    float local[13] = {};

    for (std::size_t interaction =
             partial_index * blockDim.x + threadIdx.x;
         interaction < interaction_count;
         interaction += partial_count * blockDim.x) {
      const std::size_t point_index = interaction / periodic_copy_count;
      const std::size_t copy_index = interaction % periodic_copy_count;
      const cartesian_point_fp32 point = operation.source_points[point_index];
      const near2far_periodic_copy_fp32 copy = periodic_copies[copy_index];
      const cartesian_point_fp32 shifted =
          {point.x + copy.displacement.x,
           point.y + copy.displacement.y,
           point.z + copy.displacement.z};
      const std::size_t source_frequency =
          frequency_offset + frequency_index;
      const std::size_t dft_index =
          2 * (point_index * operation.frequency_stride +
               source_frequency);
      const complex_value_fp32 amplitude =
          {operation.dft_real_imag[dft_index],
           operation.dft_real_imag[dft_index + 1]};
      complex_value_fp32 fields[6];
      float internal_absolute_envelope = 0.0f;
      if (Dimension == near2far_cartesian_dimension::cylindrical)
        near2far_greencyl_fp32(
            fields, targets[target_index], frequencies[frequency_index], eps,
            mu, shifted, operation.direction, operation.electric, amplitude,
            azimuthal_mode, greencyl_tolerance,
            &internal_absolute_envelope);
      else if (Dimension == near2far_cartesian_dimension::two)
        near2far_green2d_fp32(
            fields, targets[target_index], frequencies[frequency_index], eps,
            mu, shifted, operation.direction, operation.electric, amplitude,
            maximum_kr_input_error, &internal_absolute_envelope);
      else
        near2far_green3d_fp32(
            fields, targets[target_index], frequencies[frequency_index], eps,
            mu, shifted, operation.direction, operation.electric, amplitude,
            &internal_absolute_envelope);
      for (int component = 0; component < 6; ++component) {
        const complex_value_fp32 phased =
            complex_multiply_fp32(fields[component], copy.phase);
        local[2 * component] += phased.real;
        local[2 * component + 1] += phased.imag;
      }
      const float periodic_phase_envelope =
          fabsf(copy.phase.real) + fabsf(copy.phase.imag);
      bool phase_envelope_valid = true;
      near2far_validate_envelope_value(
          phase_envelope_valid, copy.phase.real);
      near2far_validate_envelope_value(
          phase_envelope_valid, copy.phase.imag);
      near2far_validate_envelope_sum(
          phase_envelope_valid, fabsf(copy.phase.real),
          fabsf(copy.phase.imag), periodic_phase_envelope);
      const float interaction_envelope =
          internal_absolute_envelope * periodic_phase_envelope;
      near2far_validate_envelope_product(
          phase_envelope_valid, internal_absolute_envelope,
          periodic_phase_envelope, interaction_envelope);
      if (!phase_envelope_valid)
        local[12] = near2far_invalid_envelope();
      else if (!isinf(local[12])) {
        const float accumulated_envelope =
            local[12] + interaction_envelope;
        bool accumulated_envelope_valid = true;
        near2far_validate_envelope_sum(
            accumulated_envelope_valid, local[12], interaction_envelope,
            accumulated_envelope);
        local[12] = accumulated_envelope_valid
                        ? accumulated_envelope
                        : near2far_invalid_envelope();
      }
    }

    // Reduce all signed/L1 channels in registers within each warp, then use
    // one shared scalar per warp rather than one per thread. This keeps the
    // cancellation evidence from doubling shared-memory traffic and block
    // barriers on the ordinary fast path.
    constexpr unsigned int warp_width = 32;
    const unsigned int lane = threadIdx.x % warp_width;
    const unsigned int warp = threadIdx.x / warp_width;
    const unsigned int warp_count =
        (blockDim.x + warp_width - 1) / warp_width;
    const unsigned int active_mask = __activemask();
    const unsigned int active_lanes = __popc(active_mask);
    for (int channel = 0; channel < 13; ++channel) {
      float value = local[channel];
      for (unsigned int offset = warp_width / 2; offset > 0; offset >>= 1) {
        const float other = __shfl_down_sync(active_mask, value, offset);
        if (lane + offset < active_lanes) value += other;
      }
      if (lane == 0)
        shared_components[channel * warp_count + warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
      const unsigned int warp_mask = __ballot_sync(
          __activemask(), lane < warp_count);
      for (int channel = 0; channel < 13; ++channel) {
        float value = lane < warp_count
                          ? shared_components[channel * warp_count + lane]
                          : 0.0f;
        // Every caller of a *_sync warp primitive must be named in its mask.
        // In this second-stage reduction only the first warp_count lanes own
        // a shared partial, so the remaining lanes must not execute the
        // shuffle with warp_mask.
        if (lane < warp_count) {
          for (unsigned int offset = warp_width / 2; offset > 0;
               offset >>= 1) {
            const float other = __shfl_down_sync(warp_mask, value, offset);
            if (lane + offset < warp_count) value += other;
          }
        }
        if (lane == 0)
          partials[13 * logical_block + channel] = value;
      }
    }
    __syncthreads();
  }
}

__global__ void near2far_3d_finalize_fp64_kernel(
    const float *partials, std::size_t operation_count,
    std::size_t work_count, std::size_t partial_count, double *output,
    double *absolute_l1) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t work =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       work < work_count; work += grid_stride) {
    double accumulated[12] = {};
    double accumulated_l1 = 0.0;
    for (std::size_t operation = 0; operation < operation_count;
         ++operation)
      for (std::size_t partial = 0; partial < partial_count; ++partial) {
        const std::size_t logical_block =
            (operation * work_count + work) * partial_count + partial;
        for (int channel = 0; channel < 12; ++channel)
          accumulated[channel] += partials[13 * logical_block + channel];
        accumulated_l1 += partials[13 * logical_block + 12];
      }
    for (int channel = 0; channel < 12; ++channel)
      output[12 * work + channel] = accumulated[channel];
    absolute_l1[work] = accumulated_l1;
  }
}

template <near2far_cartesian_dimension Dimension>
__global__ void near2far_3d_partials_mixed_fp32_kernel(
    const near2far_operation_mixed_fp32 *operations,
    std::size_t operation_count, const cartesian_point_fp64 *targets,
    std::size_t target_count, const double *frequencies,
    std::size_t frequency_count, std::size_t frequency_offset,
    const near2far_periodic_copy_fp64 *periodic_copies,
    std::size_t periodic_copy_count, double eps, double mu,
    std::size_t partial_count, std::size_t logical_block_count,
    double azimuthal_mode, double greencyl_tolerance, double *partials) {
  extern __shared__ double shared_components_fp64[];
  const std::size_t work_count = target_count * frequency_count;
  const std::size_t blocks_per_operation = work_count * partial_count;
  for (std::size_t logical_block = blockIdx.x;
       logical_block < logical_block_count; logical_block += gridDim.x) {
    const std::size_t operation_index =
        logical_block / blocks_per_operation;
    const std::size_t within_operation =
        logical_block % blocks_per_operation;
    const std::size_t work = within_operation / partial_count;
    const std::size_t partial_index = within_operation % partial_count;
    const std::size_t target_index = work / frequency_count;
    const std::size_t frequency_index = work % frequency_count;
    const near2far_operation_mixed_fp32 operation =
        operations[operation_index];
    const std::size_t interaction_count =
        operation.point_count * periodic_copy_count;
    double local[12] = {};

    for (std::size_t interaction =
             partial_index * blockDim.x + threadIdx.x;
         interaction < interaction_count;
         interaction += partial_count * blockDim.x) {
      const std::size_t point_index = interaction / periodic_copy_count;
      const std::size_t copy_index = interaction % periodic_copy_count;
      const cartesian_point_fp64 point =
          operation.source_points[point_index];
      const near2far_periodic_copy_fp64 copy = periodic_copies[copy_index];
      const cartesian_point_fp64 shifted =
          {point.x + copy.displacement.x,
           point.y + copy.displacement.y,
           point.z + copy.displacement.z};
      const std::size_t source_frequency =
          frequency_offset + frequency_index;
      const std::size_t dft_index =
          2 * (point_index * operation.frequency_stride +
               source_frequency);
      const complex_value_fp64 amplitude =
          {static_cast<double>(operation.dft_real_imag[dft_index]),
           static_cast<double>(operation.dft_real_imag[dft_index + 1])};
      complex_value_fp64 fields[6];
      if (Dimension == near2far_cartesian_dimension::cylindrical)
        near2far_greencyl_mixed_fp32(
            fields, targets[target_index], frequencies[frequency_index], eps,
            mu, shifted, operation.direction, operation.electric, amplitude,
            azimuthal_mode, greencyl_tolerance);
      else if (Dimension == near2far_cartesian_dimension::two)
        near2far_green2d_mixed_fp32(
            fields, targets[target_index], frequencies[frequency_index], eps,
            mu, shifted, operation.direction, operation.electric, amplitude);
      else
        near2far_green3d_mixed_fp32(
            fields, targets[target_index], frequencies[frequency_index], eps,
            mu, shifted, operation.direction, operation.electric, amplitude);
      for (int component = 0; component < 6; ++component) {
        const complex_value_fp64 phased =
            complex_multiply_fp64(fields[component], copy.phase);
        local[2 * component] += phased.real;
        local[2 * component + 1] += phased.imag;
      }
    }

    for (int channel = 0; channel < 12; ++channel)
      shared_components_fp64[channel * blockDim.x + threadIdx.x] =
          local[channel];
    __syncthreads();
    for (unsigned int active = blockDim.x; active > 1; active >>= 1) {
      const unsigned int retained = active >> 1;
      if (threadIdx.x < retained)
        for (int channel = 0; channel < 12; ++channel)
          shared_components_fp64[channel * blockDim.x + threadIdx.x] +=
              shared_components_fp64[
                  channel * blockDim.x + threadIdx.x + retained];
      __syncthreads();
    }
    if (threadIdx.x == 0)
      for (int channel = 0; channel < 12; ++channel)
        partials[12 * logical_block + channel] =
            shared_components_fp64[channel * blockDim.x];
    __syncthreads();
  }
}

__global__ void near2far_3d_finalize_mixed_fp64_kernel(
    const double *partials, std::size_t operation_count,
    std::size_t work_count, std::size_t partial_count, double *output) {
  const std::size_t grid_stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t work =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       work < work_count; work += grid_stride) {
    double accumulated[12] = {};
    for (std::size_t operation = 0; operation < operation_count;
         ++operation)
      for (std::size_t partial = 0; partial < partial_count; ++partial) {
        const std::size_t logical_block =
            (operation * work_count + work) * partial_count + partial;
        for (int channel = 0; channel < 12; ++channel)
          accumulated[channel] += partials[12 * logical_block + channel];
      }
    for (int channel = 0; channel < 12; ++channel)
      output[12 * work + channel] = accumulated[channel];
  }
}

template <near2far_cartesian_dimension Dimension>
__global__ void near2far_adjoint_fp32_kernel(
    const near2far_adjoint_source_fp32 *sources,
    std::size_t source_count, const cartesian_point_fp32 *targets,
    std::size_t target_count, const float *frequencies,
    std::size_t frequency_count,
    const near2far_periodic_copy_fp32 *periodic_copies,
    std::size_t periodic_copy_count, const complex_value_fp32 *dJ,
    float eps, float mu, std::size_t logical_block_count,
    float maximum_kr_input_error, float azimuthal_mode,
    float greencyl_tolerance, bool accumulate, double *output,
    double *absolute_l1) {
  extern __shared__ float shared_adjoint_components[];
  for (std::size_t logical_block = blockIdx.x;
       logical_block < logical_block_count; logical_block += gridDim.x) {
    const std::size_t source_index = logical_block / frequency_count;
    const std::size_t frequency_index = logical_block % frequency_count;
    const near2far_adjoint_source_fp32 source = sources[source_index];
    const std::size_t interaction_count =
        target_count * periodic_copy_count;
    float local[3] = {};

    for (std::size_t interaction = threadIdx.x;
         interaction < interaction_count; interaction += blockDim.x) {
      const std::size_t target_index =
          interaction / periodic_copy_count;
      const std::size_t copy_index = interaction % periodic_copy_count;
      const near2far_periodic_copy_fp32 copy =
          periodic_copies[copy_index];
      const cartesian_point_fp32 shifted = {
          source.point.x + copy.displacement.x,
          source.point.y + copy.displacement.y,
          source.point.z + copy.displacement.z};
      complex_value_fp32 fields[6];
      float internal_absolute_envelope = 0.0f;
      if (Dimension == near2far_cartesian_dimension::cylindrical)
        near2far_greencyl_fp32(
            fields, targets[target_index], frequencies[frequency_index],
            eps, mu, shifted, source.direction, source.electric != 0,
            source.amplitude, azimuthal_mode, greencyl_tolerance,
            &internal_absolute_envelope);
      else if (Dimension == near2far_cartesian_dimension::two)
        near2far_green2d_fp32(
            fields, targets[target_index], frequencies[frequency_index],
            eps, mu, shifted, source.direction, source.electric != 0,
            source.amplitude, maximum_kr_input_error,
            &internal_absolute_envelope);
      else
        near2far_green3d_fp32(
            fields, targets[target_index], frequencies[frequency_index],
            eps, mu, shifted, source.direction, source.electric != 0,
            source.amplitude, &internal_absolute_envelope);

      bool envelope_valid = true;
      const float phase_envelope =
          fabsf(copy.phase.real) + fabsf(copy.phase.imag);
      near2far_validate_envelope_value(envelope_valid, copy.phase.real);
      near2far_validate_envelope_value(envelope_valid, copy.phase.imag);
      near2far_validate_envelope_sum(
          envelope_valid, fabsf(copy.phase.real), fabsf(copy.phase.imag),
          phase_envelope);
      float gradient_envelope = 0.0f;
      for (int component = 0; component < 6; ++component) {
        const complex_value_fp32 gradient =
            dJ[6 * (target_index * frequency_count + frequency_index) +
               component];
        const complex_value_fp32 phased =
            complex_multiply_fp32(fields[component], copy.phase);
        const complex_value_fp32 weighted =
            complex_multiply_fp32(phased, gradient);
        local[0] += weighted.real;
        local[1] += weighted.imag;
        const float component_gradient =
            fabsf(gradient.real) + fabsf(gradient.imag);
        near2far_validate_envelope_value(envelope_valid, gradient.real);
        near2far_validate_envelope_value(envelope_valid, gradient.imag);
        near2far_validate_envelope_sum(
            envelope_valid, fabsf(gradient.real), fabsf(gradient.imag),
            component_gradient);
        const float next_gradient =
            gradient_envelope + component_gradient;
        near2far_validate_envelope_sum(
            envelope_valid, gradient_envelope, component_gradient,
            next_gradient);
        gradient_envelope = next_gradient;
      }
      const float phased_green_envelope =
          internal_absolute_envelope * phase_envelope;
      const float interaction_envelope =
          phased_green_envelope * gradient_envelope;
      near2far_validate_envelope_product(
          envelope_valid, internal_absolute_envelope, phase_envelope,
          phased_green_envelope);
      near2far_validate_envelope_product(
          envelope_valid, phased_green_envelope, gradient_envelope,
          interaction_envelope);
      if (!envelope_valid)
        local[2] = near2far_invalid_envelope();
      else if (!isinf(local[2])) {
        const float next = local[2] + interaction_envelope;
        bool sum_valid = true;
        near2far_validate_envelope_sum(
            sum_valid, local[2], interaction_envelope, next);
        local[2] = sum_valid ? next : near2far_invalid_envelope();
      }
    }

    constexpr unsigned int warp_width = 32;
    const unsigned int lane = threadIdx.x % warp_width;
    const unsigned int warp = threadIdx.x / warp_width;
    const unsigned int warp_count =
        (blockDim.x + warp_width - 1) / warp_width;
    const unsigned int active_mask = __activemask();
    const unsigned int active_lanes = __popc(active_mask);
    for (int channel = 0; channel < 3; ++channel) {
      float value = local[channel];
      for (unsigned int offset = warp_width / 2; offset > 0; offset >>= 1) {
        const float other = __shfl_down_sync(active_mask, value, offset);
        if (lane + offset < active_lanes) value += other;
      }
      if (lane == 0)
        shared_adjoint_components[channel * warp_count + warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
      const unsigned int warp_mask = __ballot_sync(
          __activemask(), lane < warp_count);
      // Every caller of a *_sync warp primitive must be named in its mask.
      // Only one lane per contributing warp participates in this second
      // reduction; letting the remaining lanes call shfl with their bits
      // clear is undefined by CUDA even though their values are ignored.
      if (lane < warp_count) {
        for (int channel = 0; channel < 3; ++channel) {
          float value = shared_adjoint_components[
              channel * warp_count + lane];
          for (unsigned int offset = warp_width / 2; offset > 0;
               offset >>= 1) {
            const float other =
                __shfl_down_sync(warp_mask, value, offset);
            if (lane + offset < warp_count) value += other;
          }
          if (lane == 0) {
            if (channel < 2) {
              const std::size_t output_index =
                  2 * logical_block + static_cast<std::size_t>(channel);
              const double converted = static_cast<double>(value);
              output[output_index] =
                  accumulate ? output[output_index] + converted : converted;
            }
            else {
              const double converted = static_cast<double>(value);
              absolute_l1[logical_block] =
                  accumulate ? absolute_l1[logical_block] + converted
                             : converted;
            }
          }
        }
      }
    }
    __syncthreads();
  }
}

template <near2far_cartesian_dimension Dimension>
__global__ void near2far_adjoint_mixed_fp64_kernel(
    const near2far_adjoint_source_fp64 *sources,
    std::size_t source_count, const cartesian_point_fp64 *targets,
    std::size_t target_count, const double *frequencies,
    std::size_t frequency_count,
    const near2far_periodic_copy_fp64 *periodic_copies,
    std::size_t periodic_copy_count, const complex_value_fp64 *dJ,
    double eps, double mu, std::size_t logical_block_count,
    double azimuthal_mode, double greencyl_tolerance, bool accumulate,
    double *output) {
  extern __shared__ double shared_adjoint_components_fp64[];
  for (std::size_t logical_block = blockIdx.x;
       logical_block < logical_block_count; logical_block += gridDim.x) {
    const std::size_t source_index = logical_block / frequency_count;
    const std::size_t frequency_index = logical_block % frequency_count;
    const near2far_adjoint_source_fp64 source = sources[source_index];
    const std::size_t interaction_count =
        target_count * periodic_copy_count;
    double local_real = 0.0;
    double local_imaginary = 0.0;
    for (std::size_t interaction = threadIdx.x;
         interaction < interaction_count; interaction += blockDim.x) {
      const std::size_t target_index =
          interaction / periodic_copy_count;
      const std::size_t copy_index = interaction % periodic_copy_count;
      const near2far_periodic_copy_fp64 copy =
          periodic_copies[copy_index];
      const cartesian_point_fp64 shifted = {
          source.point.x + copy.displacement.x,
          source.point.y + copy.displacement.y,
          source.point.z + copy.displacement.z};
      complex_value_fp64 fields[6];
      if (Dimension == near2far_cartesian_dimension::cylindrical)
        near2far_greencyl_mixed_fp32(
            fields, targets[target_index], frequencies[frequency_index],
            eps, mu, shifted, source.direction, source.electric != 0,
            source.amplitude, azimuthal_mode, greencyl_tolerance);
      else if (Dimension == near2far_cartesian_dimension::two)
        near2far_green2d_mixed_fp32(
            fields, targets[target_index], frequencies[frequency_index],
            eps, mu, shifted, source.direction, source.electric != 0,
            source.amplitude);
      else
        near2far_green3d_mixed_fp32(
            fields, targets[target_index], frequencies[frequency_index],
            eps, mu, shifted, source.direction, source.electric != 0,
            source.amplitude);
      for (int component = 0; component < 6; ++component) {
        const complex_value_fp64 gradient =
            dJ[6 * (target_index * frequency_count + frequency_index) +
               component];
        const complex_value_fp64 phased =
            complex_multiply_fp64(fields[component], copy.phase);
        const complex_value_fp64 weighted =
            complex_multiply_fp64(phased, gradient);
        local_real += weighted.real;
        local_imaginary += weighted.imag;
      }
    }
    shared_adjoint_components_fp64[threadIdx.x] = local_real;
    shared_adjoint_components_fp64[blockDim.x + threadIdx.x] =
        local_imaginary;
    __syncthreads();
    for (unsigned int active = blockDim.x; active > 1; active >>= 1) {
      const unsigned int retained = active >> 1;
      if (threadIdx.x < retained) {
        shared_adjoint_components_fp64[threadIdx.x] +=
            shared_adjoint_components_fp64[threadIdx.x + retained];
        shared_adjoint_components_fp64[blockDim.x + threadIdx.x] +=
            shared_adjoint_components_fp64[
                blockDim.x + threadIdx.x + retained];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const double real_value = shared_adjoint_components_fp64[0];
      const double imaginary_value =
          shared_adjoint_components_fp64[blockDim.x];
      output[2 * logical_block] =
          accumulate ? output[2 * logical_block] + real_value : real_value;
      output[2 * logical_block + 1] =
          accumulate ? output[2 * logical_block + 1] + imaginary_value
                     : imaginary_value;
    }
    __syncthreads();
  }
}

unsigned int launch_block_count(std::size_t count, int threads_per_block) {
  if (!launch_device.initialized) initialize_current_device_cache();
  if (!launch_device.compatible)
    throw std::invalid_argument("selected CUDA device is incompatible with compiled architectures " +
                                compiled_architectures());
  if (threads_per_block <= 0 ||
      threads_per_block > launch_device.max_threads_per_block) {
    std::ostringstream message;
    message << "threads_per_block must be in [1, "
            << launch_device.max_threads_per_block
            << "] for the selected CUDA device";
    throw std::invalid_argument(message.str());
  }

  const std::size_t requested_blocks =
      1 + (count - 1) / static_cast<std::size_t>(threads_per_block);
  const std::size_t blocks =
      std::min(requested_blocks, launch_device.max_grid_x);
  return static_cast<unsigned int>(blocks);
}

// Some kernels assign one complete cooperative thread block to each logical
// work item.  Those kernels must not divide the logical count by the thread
// count: doing so silently serializes up to threads_per_block independent
// reductions inside every physical block.
unsigned int launch_cooperative_block_count(std::size_t logical_block_count,
                                            int threads_per_block) {
  (void)launch_block_count(1, threads_per_block);
  const std::size_t blocks =
      std::min(logical_block_count, launch_device.max_grid_x);
  return static_cast<unsigned int>(blocks);
}

std::size_t dft_work_count(std::size_t point_count,
                           std::size_t frequency_count) {
  if (point_count == 0 || frequency_count == 0) return 0;
  if (point_count >
      std::numeric_limits<std::size_t>::max() / frequency_count)
    throw std::overflow_error("DFT point-frequency work count overflow");
  const std::size_t work_count = point_count * frequency_count;
  if (work_count > std::numeric_limits<std::size_t>::max() / 2)
    throw std::overflow_error("DFT interleaved output count overflow");
  if (work_count >
      std::numeric_limits<std::size_t>::max() /
          (2 * sizeof(float)))
    throw std::overflow_error("DFT interleaved output byte count overflow");
  return work_count;
}

void launch_update_dft_fp32(
    float *dft, const float *field_real, const float *field_imag,
    const std::ptrdiff_t *indices, const float *weights,
    std::size_t point_count, const complex_value_fp32 *phases,
    std::size_t frequency_count, std::ptrdiff_t average_offset1,
    std::ptrdiff_t average_offset2, int threads_per_block,
    const char *operation) {
  (void)launch_block_count(1, threads_per_block);
  const unsigned int frequency_threads = static_cast<unsigned int>(
      std::min<std::size_t>(
          frequency_count,
          std::min<std::size_t>(32, threads_per_block)));
  const unsigned int point_threads = static_cast<unsigned int>(
      std::min<std::size_t>(
          point_count,
          static_cast<std::size_t>(threads_per_block) /
              frequency_threads));
  const std::size_t requested_point_blocks =
      1 + (point_count - 1) / point_threads;
  const dim3 blocks(
      launch_block_count(frequency_count, frequency_threads),
      static_cast<unsigned int>(
          std::min(requested_point_blocks, launch_device.max_grid_y)));
  const dim3 threads(frequency_threads, point_threads);
  const std::size_t shared_bytes =
      2 * static_cast<std::size_t>(point_threads) * sizeof(float);
  update_dft_fp32_kernel<<<blocks, threads, shared_bytes>>>(
      dft, field_real, field_imag, indices, weights, point_count, phases,
      frequency_count, average_offset1, average_offset2);
  check(cudaGetLastError(), operation);
}

void validate_index_space(const index_space_fp32 &space,
                          std::size_t count) {
  if (!space.extent1 || !space.extent2 || !space.extent3)
    throw std::invalid_argument(
        "structured index-space extents must be nonzero");
  if (space.extent1 >
      std::numeric_limits<std::size_t>::max() / space.extent2)
    throw std::overflow_error("structured index-space size overflow");
  const std::size_t first_two = space.extent1 * space.extent2;
  if (first_two >
      std::numeric_limits<std::size_t>::max() / space.extent3)
    throw std::overflow_error("structured index-space size overflow");
  if (first_two * space.extent3 != count)
    throw std::invalid_argument(
        "structured index-space extents do not match the launch count");
}

} // namespace

bool runtime_available(std::string *diagnostic) {
  availability_probes.fetch_add(1, std::memory_order_relaxed);
  int count = 0;
  const cudaError_t error = cudaGetDeviceCount(&count);
  if (error == cudaSuccess) {
    for (int ordinal = 0; ordinal < count; ++ordinal) {
      cudaDeviceProp property;
      const cudaError_t property_error = cudaGetDeviceProperties(&property, ordinal);
      if (property_error == cudaSuccess &&
          detail::compute_capability_is_compiled(property.major, property.minor))
        return true;
    }
  }

  if (diagnostic) {
    if (error == cudaSuccess && count == 0)
      *diagnostic = "CUDA runtime found no devices";
    else if (error == cudaSuccess)
      *diagnostic = "CUDA runtime found no device compatible with compiled architectures: " +
                    compiled_architectures();
    else
      *diagnostic = cudaGetErrorString(error);
  }
  cudaGetLastError();
  return false;
}

std::vector<device_info> enumerate_devices() {
  device_enumerations.fetch_add(1, std::memory_order_relaxed);
  int count = 0;
  check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");

  std::vector<device_info> devices;
  devices.reserve(static_cast<std::size_t>(count));
  for (int ordinal = 0; ordinal < count; ++ordinal) {
    cudaDeviceProp property;
    check(cudaGetDeviceProperties(&property, ordinal), "cudaGetDeviceProperties");
    const std::uint64_t memory_clock_hz =
        property.memoryClockRate > 0
            ? static_cast<std::uint64_t>(property.memoryClockRate) * 1000u
            : 0u;
    const std::uint64_t memory_bus_bytes =
        property.memoryBusWidth > 0
            ? static_cast<std::uint64_t>(property.memoryBusWidth) / 8u
            : 0u;
    const std::uint64_t memory_bandwidth =
        memory_clock_hz && memory_bus_bytes &&
                memory_clock_hz <=
                    std::numeric_limits<std::uint64_t>::max() /
                        memory_bus_bytes / 2u
            ? 2u * memory_clock_hz * memory_bus_bytes
            : 0u;
    devices.push_back({ordinal,
                       property.major,
                       property.minor,
                       property.multiProcessorCount,
                       property.maxThreadsPerBlock,
                       static_cast<std::uint64_t>(property.totalGlobalMem),
                       memory_bandwidth,
                       property.name});
  }
  return devices;
}

bool device_compatible(const device_info &device) noexcept {
  return detail::compute_capability_is_compiled(device.compute_major, device.compute_minor);
}

std::string compiled_architectures() {
  std::ostringstream description;
  bool first = true;
  for (std::size_t i = 1; i < detail::compiled_real_architecture_count; ++i) {
    if (!first) description << ',';
    description << "sm_" << detail::compiled_real_architectures[i];
    first = false;
  }
  for (std::size_t i = 1; i < detail::compiled_virtual_architecture_count; ++i) {
    if (!first) description << ',';
    description << "compute_" << detail::compiled_virtual_architectures[i];
    first = false;
  }
  return description.str();
}

std::string device_uuid(int ordinal) {
  cudaDeviceProp property;
  check(cudaGetDeviceProperties(&property, ordinal),
        "cudaGetDeviceProperties");
  std::ostringstream identifier;
  identifier << std::hex << std::setfill('0');
  for (unsigned char byte : property.uuid.bytes)
    identifier << std::setw(2) << static_cast<unsigned int>(byte);
  return identifier.str();
}

void select_device(int ordinal) {
  if (launch_device.initialized && launch_device.ordinal == ordinal) return;

  device_selections.fetch_add(1, std::memory_order_relaxed);

  cudaDeviceProp property;
  check(cudaGetDeviceProperties(&property, ordinal), "cudaGetDeviceProperties");
  if (!detail::compute_capability_is_compiled(property.major, property.minor)) {
    std::ostringstream message;
    message << "CUDA device " << ordinal << " has compute capability " << property.major << '.'
            << property.minor << ", incompatible with compiled architectures "
            << compiled_architectures();
    throw std::invalid_argument(message.str());
  }
  check(cudaSetDevice(ordinal), "cudaSetDevice");
  cache_device_properties(ordinal, property);
}

device_memory_info selected_device_memory_info() {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  return {static_cast<std::uint64_t>(free_bytes),
          static_cast<std::uint64_t>(total_bytes)};
}

runtime_touch_statistics get_runtime_touch_statistics() noexcept {
  return {availability_probes.load(std::memory_order_relaxed),
          device_enumerations.load(std::memory_order_relaxed),
          device_selections.load(std::memory_order_relaxed)};
}

void reset_runtime_touch_statistics() noexcept {
  availability_probes.store(0, std::memory_order_relaxed);
  device_enumerations.store(0, std::memory_order_relaxed);
  device_selections.store(0, std::memory_order_relaxed);
}

int runtime_version() {
  int version = 0;
  check(cudaRuntimeGetVersion(&version), "cudaRuntimeGetVersion");
  return version;
}

int driver_version() {
  int version = 0;
  check(cudaDriverGetVersion(&version), "cudaDriverGetVersion");
  return version;
}

void *allocate_device_bytes(std::size_t bytes) {
  void *pointer = nullptr;
  check(cudaMalloc(&pointer, bytes), "cudaMalloc");
  return pointer;
}

void free_device(void *pointer) noexcept {
  if (pointer) cudaFree(pointer);
}

void *allocate_pinned_bytes(std::size_t bytes) {
  if (bytes == 0)
    throw std::invalid_argument("pinned allocation size must be nonzero");
  void *pointer = nullptr;
  check(cudaMallocHost(&pointer, bytes), "cudaMallocHost");
  return pointer;
}

void free_pinned(void *pointer) noexcept {
  if (pointer) cudaFreeHost(pointer);
}

void copy_to_device(void *destination, const void *source, std::size_t bytes) {
  check(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy host-to-device");
}

void copy_to_device_async(void *destination, const void *source,
                          std::size_t bytes) {
  check(cudaMemcpyAsync(destination, source, bytes,
                        cudaMemcpyHostToDevice, nullptr),
        "cudaMemcpyAsync host-to-device");
}

void copy_to_host(void *destination, const void *source, std::size_t bytes) {
  check(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy device-to-host");
}

void synchronize() { check(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); }

void *create_event() {
  cudaEvent_t event = nullptr;
  check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
        "cudaEventCreateWithFlags");
  return reinterpret_cast<void *>(event);
}

void record_event(void *event) {
  if (!event) throw std::invalid_argument("CUDA event must be non-null");
  check(cudaEventRecord(reinterpret_cast<cudaEvent_t>(event), nullptr),
        "cudaEventRecord");
}

void synchronize_event(void *event) {
  if (!event) throw std::invalid_argument("CUDA event must be non-null");
  check(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(event)),
        "cudaEventSynchronize");
}

void destroy_event(void *event) noexcept {
  if (event)
    cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event));
}

void step_curl_fp32(float *field, const float *g1, const float *g2,
                    const std::ptrdiff_t *indices, std::size_t count,
                    std::ptrdiff_t stride1, std::ptrdiff_t stride2, float dtdx,
                    int threads_per_block) {
  if (!field) throw std::invalid_argument("field must be non-null");
  const detail::curl_operands_fp32 operands =
      detail::normalize_curl_operands(g1, g2, stride1, stride2, dtdx);
  if (!operands.g1) throw std::invalid_argument("at least one curl operand must be non-null");
  if (count == 0) return;
  const unsigned int blocks = launch_block_count(count, threads_per_block);
  step_curl_fp32_kernel<<<blocks, threads_per_block>>>(
      field, operands.g1, operands.g2, indices, count, operands.stride1, operands.stride2,
      operands.dtdx);
  check(cudaGetLastError(), "step_curl_fp32 kernel launch");
}

void step_curl_material_fp32(float *field, const float *g1, const float *g2,
                             const curl_index *indices, std::size_t count,
                             std::ptrdiff_t stride1, std::ptrdiff_t stride2,
                             float dtdx, const curl_material_fp32 &material,
                             int threads_per_block) {
  if (!field) throw std::invalid_argument("field must be non-null");
  const detail::curl_operands_fp32 operands =
      detail::normalize_curl_operands(g1, g2, stride1, stride2, dtdx);
  if (!operands.g1) throw std::invalid_argument("at least one curl operand must be non-null");

  const bool pml_f = material.sigma || material.kappa || material.sigma_inverse;
  if (pml_f &&
      !(material.sigma && material.kappa && material.sigma_inverse))
    throw std::invalid_argument("PML-f sigma, kappa, and inverse pointers must be all present");
  const bool pml_u =
      material.sigma_u || material.kappa_u || material.sigma_u_inverse ||
      material.field_u;
  if (pml_u &&
      !(material.sigma_u && material.kappa_u &&
        material.sigma_u_inverse && material.field_u))
    throw std::invalid_argument(
        "PML-u sigma, kappa, inverse, and auxiliary-field pointers must be all present");
  const bool conductivity =
      material.conductivity || material.conductivity_inverse;
  if (conductivity &&
      !(material.conductivity && material.conductivity_inverse))
    throw std::invalid_argument(
        "conductivity and conductivity-inverse pointers must be both present");
  if (conductivity && pml_f && !material.field_conductivity)
    throw std::invalid_argument(
        "simultaneous PML-f and conductivity requires an auxiliary field");
  if (material.field_conductivity && !(conductivity && pml_f))
    throw std::invalid_argument(
        "conductivity auxiliary field is valid only with PML-f and conductivity");
  if ((pml_f || pml_u) && !indices)
    throw std::invalid_argument("PML curl requires explicit sigma indices");
  if (count == 0) return;

  const unsigned int blocks = launch_block_count(count, threads_per_block);
  step_curl_material_fp32_kernel<<<blocks, threads_per_block>>>(
      field, operands.g1, operands.g2, indices, count, operands.stride1,
      operands.stride2, operands.dtdx, material);
  check(cudaGetLastError(), "step_curl_material_fp32 kernel launch");
}

void step_curl_material_structured_fp32(
    float *field, const float *g1, const float *g2,
    const index_space_fp32 &index_space, std::size_t count,
    std::ptrdiff_t stride1, std::ptrdiff_t stride2, float dtdx,
    const curl_material_fp32 &material, int threads_per_block) {
  if (!field) throw std::invalid_argument("field must be non-null");
  const detail::curl_operands_fp32 operands =
      detail::normalize_curl_operands(g1, g2, stride1, stride2, dtdx);
  if (!operands.g1)
    throw std::invalid_argument("at least one curl operand must be non-null");
  const bool pml_f = material.sigma || material.kappa ||
                     material.sigma_inverse;
  if (pml_f &&
      !(material.sigma && material.kappa && material.sigma_inverse))
    throw std::invalid_argument(
        "PML-f sigma, kappa, and inverse pointers must be all present");
  const bool pml_u = material.sigma_u || material.kappa_u ||
                     material.sigma_u_inverse || material.field_u;
  if (pml_u &&
      !(material.sigma_u && material.kappa_u &&
        material.sigma_u_inverse && material.field_u))
    throw std::invalid_argument(
        "PML-u sigma, kappa, inverse, and auxiliary-field pointers must be all present");
  const bool conductivity =
      material.conductivity || material.conductivity_inverse;
  if (conductivity &&
      !(material.conductivity && material.conductivity_inverse))
    throw std::invalid_argument(
        "conductivity and conductivity-inverse pointers must be both present");
  if (conductivity && pml_f && !material.field_conductivity)
    throw std::invalid_argument(
        "simultaneous PML-f and conductivity requires an auxiliary field");
  if (material.field_conductivity && !(conductivity && pml_f))
    throw std::invalid_argument(
        "conductivity auxiliary field is valid only with PML-f and conductivity");
  if ((pml_f && index_space.coefficient_start < 0) ||
      (pml_u && index_space.coefficient2_start < 0))
    throw std::invalid_argument(
        "structured PML curl requires coefficient index spaces");
  if (count == 0) return;
  validate_index_space(index_space, count);
  const unsigned int blocks =
      launch_block_count(count, threads_per_block);
  step_curl_material_structured_fp32_kernel<<<blocks, threads_per_block>>>(
      field, operands.g1, operands.g2, index_space, count, operands.stride1,
      operands.stride2, operands.dtdx, material);
  check(cudaGetLastError(),
        "step_curl_material_structured_fp32 kernel launch");
}

void step_curl_material_batched_structured_fp32(
    float *field, const float *g1, const float *g2,
    const index_space_fp32 *spaces, std::size_t space_count,
    std::size_t maximum_point_count, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, float dtdx,
    const curl_material_fp32 &material, int threads_per_block) {
  if (!field || !spaces)
    throw std::invalid_argument(
        "batched structured curl field and spaces must be non-null");
  const detail::curl_operands_fp32 operands =
      detail::normalize_curl_operands(g1, g2, stride1, stride2, dtdx);
  if (!operands.g1)
    throw std::invalid_argument(
        "batched structured curl requires at least one operand");
  const bool pml_f = material.sigma || material.kappa ||
                     material.sigma_inverse;
  if (pml_f &&
      !(material.sigma && material.kappa && material.sigma_inverse))
    throw std::invalid_argument(
        "PML-f sigma, kappa, and inverse pointers must be all present");
  const bool pml_u = material.sigma_u || material.kappa_u ||
                     material.sigma_u_inverse || material.field_u;
  if (pml_u &&
      !(material.sigma_u && material.kappa_u &&
        material.sigma_u_inverse && material.field_u))
    throw std::invalid_argument(
        "PML-u sigma, kappa, inverse, and auxiliary-field pointers must be all present");
  const bool conductivity =
      material.conductivity || material.conductivity_inverse;
  if (conductivity &&
      !(material.conductivity && material.conductivity_inverse))
    throw std::invalid_argument(
        "conductivity and conductivity-inverse pointers must be both present");
  if (conductivity && pml_f && !material.field_conductivity)
    throw std::invalid_argument(
        "simultaneous PML-f and conductivity requires an auxiliary field");
  if (material.field_conductivity && !(conductivity && pml_f))
    throw std::invalid_argument(
        "conductivity auxiliary field is valid only with PML-f and conductivity");
  if (space_count == 0 || maximum_point_count == 0) return;

  const std::size_t warp_aligned_points =
      detail::warp_aligned_point_count(maximum_point_count);
  (void)launch_block_count(1, threads_per_block);
  const int structured_threads = static_cast<int>(
      std::min<std::size_t>(
          static_cast<std::size_t>(threads_per_block),
          std::max<std::size_t>(32u, warp_aligned_points)));
  const dim3 blocks(
      launch_block_count(maximum_point_count, structured_threads),
      static_cast<unsigned int>(
          std::min(space_count, launch_device.max_grid_y)),
      1u);
  step_curl_material_batched_structured_fp32_kernel
      <<<blocks, structured_threads>>>(
          field, operands.g1, operands.g2, spaces, space_count,
          operands.stride1, operands.stride2, operands.dtdx, material);
  check(cudaGetLastError(),
        "step_curl_material_batched_structured_fp32 kernel launch");
}

void step_curl_material_phase_batched_fp32(
    const curl_phase_operation_fp32 *operations,
    const std::uint32_t *block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count) {
  if (!operations)
    throw std::invalid_argument(
        "phase-batched curl operations must be non-null");
  if (operation_count == 0) {
    if (total_block_count != 0)
      throw std::invalid_argument(
          "phase-batched curl block count requires operations");
    return;
  }
  if (total_block_count == 0) return;
  if (!block_operation_indices)
    throw std::invalid_argument(
        "phase-batched curl block-operation indices must be non-null");
  constexpr int phase_threads = 256;
  (void)launch_block_count(1, phase_threads);
  const unsigned int blocks = static_cast<unsigned int>(
      std::min(total_block_count, launch_device.max_grid_x));
  step_curl_material_phase_batched_fp32_kernel
      <<<blocks, phase_threads>>>(
          operations, block_operation_indices, total_block_count);
  check(cudaGetLastError(),
        "step_curl_material_phase_batched_fp32 kernel launch");
}

void step_beta_structured_fp32(
    float *field, const float *g, const index_space_fp32 &index_space,
    std::size_t count, float betadt, const beta_material_fp32 &material,
    int threads_per_block) {
  if (!field || !g)
    throw std::invalid_argument("beta field and operand must be non-null");
  const bool pml_f = material.sigma_inverse != nullptr;
  const bool pml_u = material.field_u || material.sigma_u_inverse;
  if (pml_u && !(material.field_u && material.sigma_u_inverse))
    throw std::invalid_argument(
        "beta PML-u inverse and auxiliary-field pointers must be both present");
  const bool conductivity = material.conductivity_inverse != nullptr;
  if (conductivity && pml_f && !material.field_conductivity)
    throw std::invalid_argument(
        "beta conductivity with PML-f requires an auxiliary field");
  if (material.field_conductivity && !(conductivity && pml_f))
    throw std::invalid_argument(
        "beta conductivity auxiliary field requires conductivity and PML-f");
  if ((pml_f && index_space.coefficient_start < 0) ||
      (pml_u && index_space.coefficient2_start < 0))
    throw std::invalid_argument(
        "structured beta PML requires coefficient index spaces");
  if ((!pml_f && index_space.coefficient_start >= 0) ||
      (!pml_u && index_space.coefficient2_start >= 0))
    throw std::invalid_argument(
        "structured beta has coefficient indices without PML arrays");
  if (count == 0) return;
  validate_index_space(index_space, count);
  const unsigned int blocks = launch_block_count(count, threads_per_block);
  step_beta_structured_fp32_kernel<<<blocks, threads_per_block>>>(
      field, g, index_space, count, betadt, material);
  check(cudaGetLastError(), "step_beta_structured_fp32 kernel launch");
}

void step_bfast_structured_fp32(
    float *field, const float *g1, const float *g2, float *bfast_field,
    const index_space_fp32 &index_space, std::size_t count,
    std::ptrdiff_t stride1, std::ptrdiff_t stride2, float k1, float k2,
    const bfast_material_fp32 &material, int threads_per_block) {
  if (!field || !bfast_field)
    throw std::invalid_argument(
        "BFAST field and auxiliary field must be non-null");
  const detail::bfast_operands_fp32 operands =
      detail::normalize_bfast_operands(
          g1, g2, stride1, stride2, k1, k2);
  if (!operands.g1)
    throw std::invalid_argument(
        "BFAST requires at least one curl operand");
  const bool pml_f = material.sigma_inverse != nullptr;
  const bool pml_u = material.field_u || material.sigma_u_inverse;
  if (pml_u && !(material.field_u && material.sigma_u_inverse))
    throw std::invalid_argument(
        "BFAST PML-u inverse and auxiliary-field pointers must be both present");
  const bool conductivity = material.conductivity_inverse != nullptr;
  if (conductivity && pml_f && !material.field_conductivity)
    throw std::invalid_argument(
        "BFAST conductivity with PML-f requires an auxiliary field");
  if (material.field_conductivity && !(conductivity && pml_f))
    throw std::invalid_argument(
        "BFAST conductivity auxiliary field requires conductivity and PML-f");
  if ((pml_f && index_space.coefficient_start < 0) ||
      (pml_u && index_space.coefficient2_start < 0))
    throw std::invalid_argument(
        "structured BFAST PML requires coefficient index spaces");
  if ((!pml_f && index_space.coefficient_start >= 0) ||
      (!pml_u && index_space.coefficient2_start >= 0))
    throw std::invalid_argument(
        "structured BFAST has coefficient indices without PML arrays");
  if (count == 0) return;
  validate_index_space(index_space, count);
  const unsigned int blocks =
      launch_block_count(count, threads_per_block);
  step_bfast_structured_fp32_kernel<<<blocks, threads_per_block>>>(
      field, operands.g1, operands.g2, bfast_field, index_space, count,
      operands.stride1, operands.stride2, operands.k1, operands.k2,
      material);
  check(cudaGetLastError(), "step_bfast_structured_fp32 kernel launch");
}

void step_bfast_batched_structured_fp32(
    float *field, const float *g1, const float *g2, float *bfast_field,
    const index_space_fp32 *spaces, std::size_t space_count,
    std::size_t maximum_point_count, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, float k1, float k2,
    const bfast_material_fp32 &material, int threads_per_block) {
  if (!field || !bfast_field || !spaces)
    throw std::invalid_argument(
        "batched BFAST field, auxiliary field, and spaces must be non-null");
  const detail::bfast_operands_fp32 operands =
      detail::normalize_bfast_operands(
          g1, g2, stride1, stride2, k1, k2);
  if (!operands.g1)
    throw std::invalid_argument(
        "batched BFAST requires at least one curl operand");
  const bool pml_f = material.sigma_inverse != nullptr;
  const bool pml_u = material.field_u || material.sigma_u_inverse;
  if (pml_u && !(material.field_u && material.sigma_u_inverse))
    throw std::invalid_argument(
        "batched BFAST PML-u pointers must be both present");
  const bool conductivity = material.conductivity_inverse != nullptr;
  if (conductivity && pml_f && !material.field_conductivity)
    throw std::invalid_argument(
        "batched BFAST conductivity with PML-f requires an auxiliary field");
  if (material.field_conductivity && !(conductivity && pml_f))
    throw std::invalid_argument(
        "batched BFAST conductivity auxiliary requires conductivity and PML-f");
  if (space_count == 0 || maximum_point_count == 0) return;

  const std::size_t warp_aligned_points =
      detail::warp_aligned_point_count(maximum_point_count);
  (void)launch_block_count(1, threads_per_block);
  const int structured_threads = static_cast<int>(
      std::min<std::size_t>(
          static_cast<std::size_t>(threads_per_block),
          std::max<std::size_t>(32u, warp_aligned_points)));
  const dim3 blocks(
      launch_block_count(maximum_point_count, structured_threads),
      static_cast<unsigned int>(
          std::min(space_count, launch_device.max_grid_y)),
      1u);
  step_bfast_batched_structured_fp32_kernel
      <<<blocks, structured_threads>>>(
          field, operands.g1, operands.g2, bfast_field, spaces,
          space_count, operands.stride1, operands.stride2, operands.k1,
          operands.k2, material);
  check(
      cudaGetLastError(),
      "step_bfast_batched_structured_fp32 kernel launch");
}

void step_cylindrical_radial_curl_structured_fp32(
    float *field, const float *radial_operand,
    const index_space_fp32 &index_space, std::size_t count,
    std::ptrdiff_t radial_stride, float radial_origin_offset,
    int radial_difference_sign, float dtdx,
    const curl_material_fp32 &material, int threads_per_block) {
  if (!field || !radial_operand)
    throw std::invalid_argument(
        "cylindrical radial field and operand must be non-null");
  if (radial_stride <= 0)
    throw std::invalid_argument(
        "cylindrical radial stride must be positive");
  if (radial_difference_sign != -1 && radial_difference_sign != 1)
    throw std::invalid_argument(
        "cylindrical radial difference sign must be -1 or 1");
  const bool pml_f =
      material.sigma || material.kappa || material.sigma_inverse;
  if (pml_f &&
      !(material.sigma && material.kappa && material.sigma_inverse))
    throw std::invalid_argument(
        "cylindrical PML-f pointers must be all present");
  const bool pml_u =
      material.sigma_u || material.kappa_u ||
      material.sigma_u_inverse || material.field_u;
  if (pml_u &&
      !(material.sigma_u && material.kappa_u &&
        material.sigma_u_inverse && material.field_u))
    throw std::invalid_argument(
        "cylindrical PML-u pointers must be all present");
  const bool conductivity =
      material.conductivity || material.conductivity_inverse;
  if (conductivity &&
      !(material.conductivity && material.conductivity_inverse))
    throw std::invalid_argument(
        "cylindrical conductivity pointers must be both present");
  if (conductivity && pml_f && !material.field_conductivity)
    throw std::invalid_argument(
        "cylindrical conductivity with PML-f requires an auxiliary field");
  if (material.field_conductivity && !(conductivity && pml_f))
    throw std::invalid_argument(
        "cylindrical conductivity auxiliary requires conductivity and PML-f");
  if ((pml_f && index_space.coefficient_start < 0) ||
      (pml_u && index_space.coefficient2_start < 0))
    throw std::invalid_argument(
        "cylindrical structured PML requires coefficient index spaces");
  if ((!pml_f && index_space.coefficient_start >= 0) ||
      (!pml_u && index_space.coefficient2_start >= 0))
    throw std::invalid_argument(
        "cylindrical coefficient indices require matching PML arrays");
  if (count == 0) return;
  validate_index_space(index_space, count);
  const unsigned int blocks = launch_block_count(count, threads_per_block);
  step_cylindrical_radial_curl_structured_fp32_kernel
      <<<blocks, threads_per_block>>>(
          field, radial_operand, index_space, count, radial_stride,
          radial_origin_offset, radial_difference_sign, dtdx, material);
  check(cudaGetLastError(),
        "step_cylindrical_radial_curl_structured_fp32 kernel launch");
}

void step_cylindrical_imr_structured_fp32(
    float *field, const float *g, const index_space_fp32 &index_space,
    std::size_t count, int radial_coordinate_start, float coefficient,
    const beta_material_fp32 &material, int threads_per_block) {
  if (radial_coordinate_start <= 0)
    throw std::invalid_argument(
        "cylindrical i*m/r radial coordinate must be positive");
  if (!field || !g)
    throw std::invalid_argument(
        "cylindrical i*m/r field and operand must be non-null");
  const bool pml_f = material.sigma_inverse != nullptr;
  const bool pml_u = material.field_u || material.sigma_u_inverse;
  if (pml_u && !(material.field_u && material.sigma_u_inverse))
    throw std::invalid_argument(
        "cylindrical i*m/r PML-u pointers must be both present");
  const bool conductivity = material.conductivity_inverse != nullptr;
  if (conductivity && pml_f && !material.field_conductivity)
    throw std::invalid_argument(
        "cylindrical i*m/r conductivity with PML-f requires an auxiliary");
  if (material.field_conductivity && !(conductivity && pml_f))
    throw std::invalid_argument(
        "cylindrical i*m/r conductivity auxiliary requires PML-f");
  if ((pml_f && index_space.coefficient_start < 0) ||
      (pml_u && index_space.coefficient2_start < 0))
    throw std::invalid_argument(
        "cylindrical i*m/r PML requires coefficient index spaces");
  if ((!pml_f && index_space.coefficient_start >= 0) ||
      (!pml_u && index_space.coefficient2_start >= 0))
    throw std::invalid_argument(
        "cylindrical i*m/r coefficient indices require PML arrays");
  if (count == 0) return;
  validate_index_space(index_space, count);
  const unsigned int blocks = launch_block_count(count, threads_per_block);
  step_cylindrical_imr_structured_fp32_kernel
      <<<blocks, threads_per_block>>>(
          field, g, index_space, count, radial_coordinate_start,
          coefficient, material);
  check(cudaGetLastError(),
        "step_cylindrical_imr_structured_fp32 kernel launch");
}

void step_cylindrical_axis_structured_fp32(
    float *field, const float *primary, const float *secondary,
    const index_space_fp32 &index_space, std::size_t count,
    std::ptrdiff_t neighbor_shift, std::ptrdiff_t secondary_offset,
    float secondary_scale, float drive_scale,
    const curl_material_fp32 &material, int threads_per_block) {
  if (!field || !primary)
    throw std::invalid_argument(
        "cylindrical axis field and primary operand must be non-null");
  if (!secondary && (neighbor_shift || secondary_offset || secondary_scale))
    throw std::invalid_argument(
        "cylindrical single-operand axis update has secondary parameters");
  const bool pml_f =
      material.sigma || material.kappa || material.sigma_inverse;
  if (pml_f &&
      !(material.sigma && material.kappa && material.sigma_inverse))
    throw std::invalid_argument(
        "cylindrical axis PML-f pointers must be all present");
  const bool pml_u =
      material.sigma_u || material.kappa_u ||
      material.sigma_u_inverse || material.field_u;
  if (pml_u &&
      !(material.sigma_u && material.kappa_u &&
        material.sigma_u_inverse && material.field_u))
    throw std::invalid_argument(
        "cylindrical axis PML-u pointers must be all present");
  const bool conductivity =
      material.conductivity || material.conductivity_inverse;
  if (conductivity &&
      !(material.conductivity && material.conductivity_inverse))
    throw std::invalid_argument(
        "cylindrical axis conductivity pointers must be both present");
  if (conductivity && pml_f && !material.field_conductivity)
    throw std::invalid_argument(
        "cylindrical axis conductivity with PML-f requires an auxiliary");
  if (material.field_conductivity && !(conductivity && pml_f))
    throw std::invalid_argument(
        "cylindrical axis conductivity auxiliary requires PML-f");
  if ((pml_f && index_space.coefficient_start < 0) ||
      (pml_u && index_space.coefficient2_start < 0))
    throw std::invalid_argument(
        "cylindrical axis PML requires coefficient index spaces");
  if ((!pml_f && index_space.coefficient_start >= 0) ||
      (!pml_u && index_space.coefficient2_start >= 0))
    throw std::invalid_argument(
        "cylindrical axis coefficient indices require PML arrays");
  if (count == 0) return;
  validate_index_space(index_space, count);
  const unsigned int blocks = launch_block_count(count, threads_per_block);
  step_cylindrical_axis_structured_fp32_kernel
      <<<blocks, threads_per_block>>>(
          field, primary, secondary, index_space, count, neighbor_shift,
          secondary_offset, secondary_scale, drive_scale, material);
  check(cudaGetLastError(),
        "step_cylindrical_axis_structured_fp32 kernel launch");
}

void update_eh_fp32(float *field, const float *g, const float *g1,
                    const float *g2, const update_eh_index *indices,
                    std::size_t count, std::ptrdiff_t field_stride,
                    std::ptrdiff_t stride1, std::ptrdiff_t stride2,
                    const update_eh_material_fp32 &material,
                    int threads_per_block) {
  if (!field) throw std::invalid_argument("field must be non-null");
  if (!g) throw std::invalid_argument("constitutive input field must be non-null");
  const detail::update_eh_operands_fp32 operands =
      detail::normalize_update_eh_operands(
          g1, g2, material.offdiagonal1, material.offdiagonal2, stride1,
          stride2);
  if (operands.u1 && !operands.g1)
    throw std::invalid_argument(
        "first off-diagonal coefficient requires its input field");
  if (operands.u2 && !operands.g2)
    throw std::invalid_argument(
        "second off-diagonal coefficient requires its input field");
  if (operands.u2 && !operands.u1)
    throw std::invalid_argument(
        "second off-diagonal coefficient requires the first coefficient");
  if (material.chi3 && !material.chi2)
    throw std::invalid_argument("chi3 requires a chi2 array");
  const bool pml =
      material.field_w || material.sigma || material.kappa;
  if (pml && !(material.field_w && material.sigma && material.kappa))
    throw std::invalid_argument(
        "PML field, sigma, and kappa pointers must be all present");
  if (pml && !indices)
    throw std::invalid_argument("PML update requires explicit sigma indices");
  if (count == 0) return;

  const unsigned int blocks = launch_block_count(count, threads_per_block);
  update_eh_fp32_kernel<<<blocks, threads_per_block>>>(
      field, g, g1, g2, indices, count, field_stride, stride1, stride2,
      material);
  check(cudaGetLastError(), "update_eh_fp32 kernel launch");
}

void update_eh_structured_fp32(
    float *field, const float *g, const float *g1, const float *g2,
    const index_space_fp32 &index_space, std::size_t count,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, const update_eh_material_fp32 &material,
    int threads_per_block) {
  if (!field) throw std::invalid_argument("field must be non-null");
  if (!g)
    throw std::invalid_argument("constitutive input field must be non-null");
  const detail::update_eh_operands_fp32 operands =
      detail::normalize_update_eh_operands(
          g1, g2, material.offdiagonal1, material.offdiagonal2, stride1,
          stride2);
  if (operands.u1 && !operands.g1)
    throw std::invalid_argument(
        "first off-diagonal coefficient requires its input field");
  if (operands.u2 && !operands.g2)
    throw std::invalid_argument(
        "second off-diagonal coefficient requires its input field");
  if (operands.u2 && !operands.u1)
    throw std::invalid_argument(
        "second off-diagonal coefficient requires the first coefficient");
  if (material.chi3 && !material.chi2)
    throw std::invalid_argument("chi3 requires a chi2 array");
  const bool pml = material.field_w || material.sigma || material.kappa;
  if (pml && !(material.field_w && material.sigma && material.kappa))
    throw std::invalid_argument(
        "PML field, sigma, and kappa pointers must be all present");
  if (pml && index_space.coefficient_start < 0)
    throw std::invalid_argument(
        "structured PML update requires a coefficient index space");
  if (count == 0) return;
  validate_index_space(index_space, count);
  const unsigned int blocks =
      launch_block_count(count, threads_per_block);
  update_eh_structured_fp32_kernel<<<blocks, threads_per_block>>>(
      field, g, g1, g2, index_space, count, field_stride, stride1, stride2,
      material);
  check(cudaGetLastError(), "update_eh_structured_fp32 kernel launch");
}

void update_eh_phase_batched_fp32(
    const update_eh_phase_operation_fp32 *operations,
    const std::uint32_t *block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count) {
  if (!operations)
    throw std::invalid_argument(
        "phase-batched E/H operations must be non-null");
  if (operation_count == 0) {
    if (total_block_count != 0)
      throw std::invalid_argument(
          "phase-batched E/H block count requires operations");
    return;
  }
  if (total_block_count == 0) return;
  if (!block_operation_indices)
    throw std::invalid_argument(
        "phase-batched E/H block-operation indices must be non-null");
  constexpr int phase_threads = 256;
  (void)launch_block_count(1, phase_threads);
  const unsigned int blocks = static_cast<unsigned int>(
      std::min(total_block_count, launch_device.max_grid_x));
  update_eh_phase_batched_fp32_kernel<<<blocks, phase_threads>>>(
      operations, block_operation_indices, total_block_count);
  check(cudaGetLastError(),
        "update_eh_phase_batched_fp32 kernel launch");
}

void update_lorentzian_fp32(
    float *polarization, float *previous_polarization, const float *field,
    const float *field1, const float *field2,
    const std::ptrdiff_t *indices, std::size_t count,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, const lorentzian_material_fp32 &material,
    int threads_per_block) {
  if (!polarization || !previous_polarization)
    throw std::invalid_argument(
        "polarization and previous-polarization pointers must be non-null");
  if (!field || !material.sigma)
    throw std::invalid_argument(
        "Lorentzian field and diagonal sigma pointers must be non-null");
  const detail::lorentzian_operands_fp32 operands =
      detail::normalize_lorentzian_operands(
          field1, field2, material.offdiagonal1, material.offdiagonal2,
          stride1, stride2);
  if (operands.sigma1 && !operands.w1)
    throw std::invalid_argument(
        "first Lorentzian off-diagonal sigma requires its field");
  if (operands.sigma2 && !operands.w2)
    throw std::invalid_argument(
        "second Lorentzian off-diagonal sigma requires its field");
  if (operands.sigma2 && !operands.sigma1)
    throw std::invalid_argument(
        "second Lorentzian off-diagonal sigma requires the first coefficient");
  if (count == 0) return;

  const unsigned int blocks = launch_block_count(count, threads_per_block);
  update_lorentzian_fp32_kernel<<<blocks, threads_per_block>>>(
      polarization, previous_polarization, field, field1, field2, indices,
      count, field_stride, stride1, stride2, material);
  check(cudaGetLastError(), "update_lorentzian_fp32 kernel launch");
}

void update_lorentzian_structured_fp32(
    float *polarization, float *previous_polarization, const float *field,
    const float *field1, const float *field2,
    const index_space_fp32 &index_space, std::size_t count,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, const lorentzian_material_fp32 &material,
    int threads_per_block) {
  if (!polarization || !previous_polarization)
    throw std::invalid_argument(
        "polarization and previous-polarization pointers must be non-null");
  if (!field || !material.sigma)
    throw std::invalid_argument(
        "Lorentzian field and diagonal sigma pointers must be non-null");
  const detail::lorentzian_operands_fp32 operands =
      detail::normalize_lorentzian_operands(
          field1, field2, material.offdiagonal1, material.offdiagonal2,
          stride1, stride2);
  if (operands.sigma1 && !operands.w1)
    throw std::invalid_argument(
        "first Lorentzian off-diagonal sigma requires its field");
  if (operands.sigma2 && !operands.w2)
    throw std::invalid_argument(
        "second Lorentzian off-diagonal sigma requires its field");
  if (operands.sigma2 && !operands.sigma1)
    throw std::invalid_argument(
        "second Lorentzian off-diagonal sigma requires the first coefficient");
  if (count == 0) return;
  validate_index_space(index_space, count);
  const unsigned int blocks =
      launch_block_count(count, threads_per_block);
  update_lorentzian_structured_fp32_kernel<<<blocks, threads_per_block>>>(
      polarization, previous_polarization, field, field1, field2,
      index_space, count, field_stride, stride1, stride2, material);
  check(cudaGetLastError(),
        "update_lorentzian_structured_fp32 kernel launch");
}

void update_lorentzian_increment_fp32(
    float *polarization, float *polarization_increment, const float *field,
    const float *field1, const float *field2,
    const std::ptrdiff_t *indices, std::size_t count,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, const lorentzian_material_fp32 &material,
    int threads_per_block) {
  if (!polarization || !polarization_increment)
    throw std::invalid_argument(
        "polarization and polarization-increment pointers must be non-null");
  if (!field || !material.sigma)
    throw std::invalid_argument(
        "Lorentzian field and diagonal sigma pointers must be non-null");
  const detail::lorentzian_operands_fp32 operands =
      detail::normalize_lorentzian_operands(
          field1, field2, material.offdiagonal1, material.offdiagonal2,
          stride1, stride2);
  if (operands.sigma1 && !operands.w1)
    throw std::invalid_argument(
        "first Lorentzian off-diagonal sigma requires its field");
  if (operands.sigma2 && !operands.w2)
    throw std::invalid_argument(
        "second Lorentzian off-diagonal sigma requires its field");
  if (operands.sigma2 && !operands.sigma1)
    throw std::invalid_argument(
        "second Lorentzian off-diagonal sigma requires the first coefficient");
  if (count == 0) return;

  const unsigned int blocks = launch_block_count(count, threads_per_block);
  update_lorentzian_increment_fp32_kernel<<<blocks, threads_per_block>>>(
      polarization, polarization_increment, field, field1, field2, indices,
      count, field_stride, stride1, stride2, material);
  check(cudaGetLastError(),
        "update_lorentzian_increment_fp32 kernel launch");
}

void update_lorentzian_increment_structured_fp32(
    float *polarization, float *polarization_increment, const float *field,
    const float *field1, const float *field2,
    const index_space_fp32 &index_space, std::size_t count,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, const lorentzian_material_fp32 &material,
    int threads_per_block) {
  if (!polarization || !polarization_increment)
    throw std::invalid_argument(
        "polarization and polarization-increment pointers must be non-null");
  if (!field || !material.sigma)
    throw std::invalid_argument(
        "Lorentzian field and diagonal sigma pointers must be non-null");
  const detail::lorentzian_operands_fp32 operands =
      detail::normalize_lorentzian_operands(
          field1, field2, material.offdiagonal1, material.offdiagonal2,
          stride1, stride2);
  if (operands.sigma1 && !operands.w1)
    throw std::invalid_argument(
        "first Lorentzian off-diagonal sigma requires its field");
  if (operands.sigma2 && !operands.w2)
    throw std::invalid_argument(
        "second Lorentzian off-diagonal sigma requires its field");
  if (operands.sigma2 && !operands.sigma1)
    throw std::invalid_argument(
        "second Lorentzian off-diagonal sigma requires the first coefficient");
  if (count == 0) return;
  validate_index_space(index_space, count);
  const unsigned int blocks =
      launch_block_count(count, threads_per_block);
  update_lorentzian_increment_structured_fp32_kernel
      <<<blocks, threads_per_block>>>(
          polarization, polarization_increment, field, field1, field2,
          index_space, count, field_stride, stride1, stride2, material);
  check(cudaGetLastError(),
        "update_lorentzian_increment_structured_fp32 kernel launch");
}

void update_gyrotropic_structured_fp32(
    float *polarization0, float *polarization1, float *polarization2,
    float *previous0, float *previous1, float *previous2,
    const float *field0, const float *field1, const float *field2,
    const index_space_fp32 &index_space, std::size_t count,
    std::ptrdiff_t field_stride, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, const gyrotropic_material_fp32 &material,
    int threads_per_block) {
  if (!polarization0 || !polarization1 || !polarization2 ||
      !previous0 || !previous1 || !previous2)
    throw std::invalid_argument(
        "gyrotropic polarization and previous-polarization pointers "
        "must be non-null");
  if (!field0 || !material.sigma)
    throw std::invalid_argument(
        "gyrotropic primary field and sigma pointers must be non-null");
  if (material.model < detail::gyrotropic_lorentzian_fp32 ||
      material.model > detail::gyrotropic_saturated_fp32)
    throw std::invalid_argument("invalid gyrotropic polarization model");
  if (count == 0) return;
  validate_index_space(index_space, count);
  const unsigned int blocks =
      launch_block_count(count, threads_per_block);
  update_gyrotropic_structured_fp32_kernel<<<blocks, threads_per_block>>>(
      polarization0, polarization1, polarization2, previous0, previous1,
      previous2, field0, field1, field2, index_space, count, field_stride,
      stride1, stride2, material);
  check(cudaGetLastError(),
        "update_gyrotropic_structured_fp32 kernel launch");
}

void update_multilevel_structured_fp32(
    float *population, float *population_scratch, const float *gamma,
    const float *gamma_inverse, const float *alpha,
    std::size_t level_count, std::size_t transition_count,
    std::size_t array_count, float half_dt,
    const index_space_fp32 &centered_space,
    std::size_t centered_point_count,
    const multilevel_population_channel_fp32 *population_channels,
    std::size_t population_channel_count,
    const multilevel_polarization_channel_fp32 *polarization_channels,
    std::size_t polarization_channel_count,
    const multilevel_transition_fp32 *transitions,
    std::size_t maximum_polarization_point_count,
    int threads_per_block) {
  if (!population || !population_scratch || !gamma || !gamma_inverse ||
      !alpha || !transitions)
    throw std::invalid_argument(
        "multilevel population, matrix, alpha, scratch, and transition "
        "pointers must be non-null");
  if (!level_count || !transition_count || !array_count)
    throw std::invalid_argument(
        "multilevel level, transition, and array counts must be nonzero");
  if (population_channel_count && !population_channels)
    throw std::invalid_argument(
        "multilevel population channels must be non-null");
  if (polarization_channel_count && !polarization_channels)
    throw std::invalid_argument(
        "multilevel polarization channels must be non-null");
  if (array_count >
      std::numeric_limits<std::size_t>::max() / level_count)
    throw std::overflow_error("multilevel population span overflow");
  if (transition_count >
      std::numeric_limits<std::size_t>::max() / 2 ||
      2 * transition_count >
          std::numeric_limits<std::size_t>::max() / array_count)
    throw std::overflow_error("multilevel polarization span overflow");
  if (level_count >
      std::numeric_limits<std::size_t>::max() / level_count)
    throw std::overflow_error("multilevel relaxation matrix span overflow");
  if (level_count >
      std::numeric_limits<std::size_t>::max() / transition_count)
    throw std::overflow_error("multilevel alpha span overflow");
  if (polarization_channel_count >
      std::numeric_limits<std::size_t>::max() / transition_count)
    throw std::overflow_error("multilevel operation count overflow");
  if (centered_point_count == 0)
    throw std::invalid_argument(
        "multilevel centered point count must be nonzero");
  validate_index_space(centered_space, centered_point_count);

  const unsigned int population_blocks =
      launch_block_count(centered_point_count, threads_per_block);
  update_multilevel_population_fp32_kernel<<<population_blocks,
                                             threads_per_block>>>(
      population, population_scratch, gamma, gamma_inverse, alpha,
      level_count, transition_count, array_count, half_dt, centered_space,
      centered_point_count, population_channels, population_channel_count,
      transitions);
  check(cudaGetLastError(),
        "update_multilevel_population_fp32 kernel launch");

  const std::size_t operation_count =
      polarization_channel_count * transition_count;
  if (!operation_count) return;
  if (!maximum_polarization_point_count)
    throw std::invalid_argument(
        "multilevel maximum polarization point count must be nonzero");
  (void)launch_block_count(1, threads_per_block);
  const dim3 blocks(
      launch_block_count(maximum_polarization_point_count,
                         threads_per_block),
      static_cast<unsigned int>(
          std::min(operation_count, launch_device.max_grid_y)));
  update_multilevel_polarization_fp32_kernel<<<blocks,
                                               threads_per_block>>>(
      population, level_count, transition_count, array_count,
      polarization_channels, polarization_channel_count, transitions);
  check(cudaGetLastError(),
        "update_multilevel_polarization_fp32 kernel launch");
}

void copy_fp32(float *destination, const float *source, std::size_t count,
               int threads_per_block) {
  if (!destination || !source)
    throw std::invalid_argument(
        "copy destination and source pointers must be non-null");
  if (count == 0) return;
  const unsigned int blocks = launch_block_count(count, threads_per_block);
  copy_fp32_kernel<<<blocks, threads_per_block>>>(destination, source, count);
  check(cudaGetLastError(), "copy_fp32 kernel launch");
}

void zero_fp32(float *destination, std::size_t count,
               int threads_per_block) {
  if (!destination)
    throw std::invalid_argument("zero destination pointer must be non-null");
  if (count == 0) return;
  const unsigned int blocks = launch_block_count(count, threads_per_block);
  zero_fp32_kernel<<<blocks, threads_per_block>>>(destination, count);
  check(cudaGetLastError(), "zero_fp32 kernel launch");
}

void subtract_fp32(float *destination, const float *source,
                   std::size_t count, int threads_per_block) {
  if (!destination || !source)
    throw std::invalid_argument(
        "subtract destination and source pointers must be non-null");
  if (count == 0) return;
  const unsigned int blocks = launch_block_count(count, threads_per_block);
  subtract_fp32_kernel<<<blocks, threads_per_block>>>(destination, source,
                                                       count);
  check(cudaGetLastError(), "subtract_fp32 kernel launch");
}

void indexed_subtract_fp32(float *destination,
                           const indexed_value_fp32 *updates,
                           std::size_t count, int threads_per_block) {
  if (!destination || !updates)
    throw std::invalid_argument(
        "indexed subtract destination and updates must be non-null");
  if (count == 0) return;
  const unsigned int blocks = launch_block_count(count, threads_per_block);
  indexed_subtract_fp32_kernel<<<blocks, threads_per_block>>>(
      destination, updates, count);
  check(cudaGetLastError(), "indexed_subtract_fp32 kernel launch");
}

void indexed_source_subtract_fp32(
    float *destination, const std::ptrdiff_t *indices,
    const complex_value_fp32 *amplitudes,
    const float *conductivity_inverse, std::size_t count,
    complex_value_fp32 time_scale, bool imaginary_component,
    int threads_per_block) {
  if (!destination || !indices || !amplitudes)
    throw std::invalid_argument(
        "indexed source destination, indices, and amplitudes must be "
        "non-null");
  if (count == 0) return;
  const unsigned int blocks = launch_block_count(count, threads_per_block);
  indexed_source_subtract_fp32_kernel<<<blocks, threads_per_block>>>(
      destination, indices, amplitudes, conductivity_inverse, count,
      time_scale, imaginary_component);
  check(cudaGetLastError(), "indexed_source_subtract_fp32 kernel launch");
}

void indexed_source_subtract_phase_batched_fp32(
    const indexed_source_phase_operation_fp32 *operations,
    const std::uint32_t *block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count,
    indexed_source_phase_time_scales_fp32 time_scales) {
  if (!operations)
    throw std::invalid_argument(
        "phase-batched source operations must be non-null");
  if (operation_count == 0) {
    if (total_block_count != 0)
      throw std::invalid_argument(
          "phase-batched source block count requires operations");
    return;
  }
  if (total_block_count == 0) return;
  if (operation_count > indexed_source_phase_max_operations)
    throw std::invalid_argument(
        "phase-batched source operation count exceeds parameter capacity");
  if (!block_operation_indices)
    throw std::invalid_argument(
        "phase-batched source block-operation indices must be non-null");
  constexpr int phase_threads = 256;
  (void)launch_block_count(1, phase_threads);
  const unsigned int blocks = static_cast<unsigned int>(
      std::min(total_block_count, launch_device.max_grid_x));
  indexed_source_subtract_phase_batched_fp32_kernel
      <<<blocks, phase_threads>>>(
          operations, block_operation_indices, total_block_count,
          time_scales);
  check(cudaGetLastError(),
        "indexed_source_subtract_phase_batched_fp32 kernel launch");
}

void initialize_indexed_ldos_result(double *device_result) {
  if (!device_result)
    throw std::invalid_argument(
        "indexed LDOS result pointer must be non-null");
  initialize_indexed_ldos_result_kernel<<<1, 4>>>(device_result);
  check(cudaGetLastError(), "initialize_indexed_ldos_result kernel launch");
}

void indexed_ldos_reduce_fp32(
    const indexed_ldos_operation_fp32 *operations,
    const std::uint32_t *block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count,
    double *device_block_partials, std::size_t partial_capacity,
    double *device_result, indexed_ldos_result_mode result_mode) {
  if (!device_result)
    throw std::invalid_argument(
        "indexed LDOS result must be non-null");
  if (operation_count == 0) {
    if (total_block_count != 0)
      throw std::invalid_argument(
          "indexed LDOS block count requires operations");
    return;
  }
  if (total_block_count == 0)
    throw std::invalid_argument(
        "indexed LDOS operation count requires logical blocks");
  if (!operations || !block_operation_indices || !device_block_partials)
    throw std::invalid_argument(
        "indexed LDOS operations, block map, and workspace must be non-null");
  if (operation_count >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
    throw std::overflow_error(
        "indexed LDOS operation count exceeds block-map capacity");
  if (operation_count > total_block_count)
    throw std::invalid_argument(
        "indexed LDOS operation count exceeds logical block count");
  if (result_mode != indexed_ldos_result_mode::replace &&
      result_mode != indexed_ldos_result_mode::accumulate)
    throw std::invalid_argument("indexed LDOS result mode is invalid");
  if (partial_capacity == 0)
    throw std::invalid_argument(
        "indexed LDOS partial workspace is too small");
  constexpr int threads = 256;
  (void)launch_block_count(1, threads);
  const std::size_t physical_block_count = std::min(
      std::min(total_block_count, launch_device.max_grid_x),
      indexed_ldos_reduction_partial_capacity);
  if (partial_capacity < physical_block_count)
    throw std::invalid_argument(
        "indexed LDOS partial workspace is too small");
  if (partial_capacity >
      std::numeric_limits<std::size_t>::max() /
          (4 * sizeof(double)))
    throw std::overflow_error(
        "indexed LDOS partial workspace byte count overflow");
  const std::uintptr_t partial_begin =
      reinterpret_cast<std::uintptr_t>(device_block_partials);
  const std::size_t partial_bytes =
      4 * partial_capacity * sizeof(double);
  if (partial_begin >
      std::numeric_limits<std::uintptr_t>::max() - partial_bytes)
    throw std::overflow_error(
        "indexed LDOS partial workspace address overflow");
  const std::uintptr_t partial_end = partial_begin + partial_bytes;
  const std::uintptr_t result_begin =
      reinterpret_cast<std::uintptr_t>(device_result);
  if (result_begin >
      std::numeric_limits<std::uintptr_t>::max() - 4 * sizeof(double))
    throw std::overflow_error("indexed LDOS result address overflow");
  const std::uintptr_t result_end = result_begin + 4 * sizeof(double);
  if (partial_begin < result_end && result_begin < partial_end)
    throw std::invalid_argument(
        "indexed LDOS partial workspace and result must not overlap");
  const unsigned int blocks =
      static_cast<unsigned int>(physical_block_count);
  indexed_ldos_reduce_fp32_partials_kernel<<<
      blocks, threads,
      2u * static_cast<std::size_t>(threads) * sizeof(double)>>>(
      operations, block_operation_indices, total_block_count,
      device_block_partials);
  check(cudaGetLastError(),
        "indexed_ldos_reduce_fp32 partial kernel launch");
  indexed_ldos_reduce_fp32_final_kernel<<<
      1, threads,
      static_cast<std::size_t>(threads) * sizeof(double)>>>(
      device_block_partials, physical_block_count, device_result,
      result_mode);
  check(cudaGetLastError(),
        "indexed_ldos_reduce_fp32 final kernel launch");
}

void dft_pair_reduce_fp32(
    const dft_pair_reduction_operation_fp32 *operations,
    const std::uint32_t *block_operation_indices,
    std::size_t operation_count, std::size_t total_spatial_block_count,
    std::size_t frequency_count, double *device_partials,
    std::size_t partial_capacity, double *device_result_real_imag) {
  if (operation_count == 0) {
    if (total_spatial_block_count != 0)
      throw std::invalid_argument(
          "DFT pair-reduction block count requires operations");
    return;
  }
  if (!operations || !block_operation_indices || !device_partials ||
      !device_result_real_imag)
    throw std::invalid_argument(
        "DFT pair-reduction operations, block map, workspace, and result "
        "must be non-null");
  if (total_spatial_block_count == 0 || frequency_count == 0)
    throw std::invalid_argument(
        "DFT pair-reduction operations require points and frequencies");
  if (operation_count > total_spatial_block_count)
    throw std::invalid_argument(
        "DFT pair-reduction operation count exceeds logical block count");
  if (operation_count >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
    throw std::overflow_error(
        "DFT pair-reduction operation count exceeds block-map capacity");
  if (partial_capacity == 0)
    throw std::invalid_argument(
        "DFT pair-reduction partial workspace is too small");
  constexpr int threads =
      static_cast<int>(dft_pair_reduction_point_threads *
                       dft_pair_reduction_frequency_threads);
  (void)launch_block_count(1, threads);
  const std::size_t physical_partial_count = std::min(
      std::min(total_spatial_block_count, launch_device.max_grid_y),
      dft_pair_reduction_partial_capacity);
  if (partial_capacity < physical_partial_count)
    throw std::invalid_argument(
        "DFT pair-reduction partial workspace is too small");
  if (partial_capacity >
      std::numeric_limits<std::size_t>::max() /
          (2 * sizeof(double)))
    throw std::overflow_error(
        "DFT pair-reduction partial workspace byte count overflow");
  const std::size_t partial_bytes_per_frequency =
      2 * partial_capacity * sizeof(double);
  if (frequency_count >
      std::numeric_limits<std::size_t>::max() /
          partial_bytes_per_frequency)
    throw std::overflow_error(
        "DFT pair-reduction partial workspace byte count overflow");
  if (frequency_count >
      std::numeric_limits<std::size_t>::max() / (2 * sizeof(double)))
    throw std::overflow_error(
        "DFT pair-reduction result byte count overflow");
  const std::size_t partial_bytes =
      frequency_count * partial_bytes_per_frequency;
  const std::size_t result_bytes =
      2 * frequency_count * sizeof(double);
  const std::uintptr_t partial_begin =
      reinterpret_cast<std::uintptr_t>(device_partials);
  const std::uintptr_t result_begin =
      reinterpret_cast<std::uintptr_t>(device_result_real_imag);
  if (partial_begin >
          std::numeric_limits<std::uintptr_t>::max() - partial_bytes ||
      result_begin >
          std::numeric_limits<std::uintptr_t>::max() - result_bytes)
    throw std::overflow_error(
        "DFT pair-reduction workspace address overflow");
  const std::uintptr_t partial_end = partial_begin + partial_bytes;
  const std::uintptr_t result_end = result_begin + result_bytes;
  if (partial_begin < result_end && result_begin < partial_end)
    throw std::invalid_argument(
        "DFT pair-reduction partial workspace and result must not overlap");

  const std::size_t requested_frequency_blocks =
      frequency_count / dft_pair_reduction_frequency_threads +
      (frequency_count % dft_pair_reduction_frequency_threads != 0);
  const std::size_t physical_frequency_blocks =
      std::min(requested_frequency_blocks, launch_device.max_grid_x);
  if (physical_frequency_blocks == 0 || physical_partial_count == 0)
    throw std::logic_error("DFT pair-reduction launch grid is empty");
  const dim3 grid(
      static_cast<unsigned int>(physical_frequency_blocks),
      static_cast<unsigned int>(physical_partial_count));
  const dim3 block(
      static_cast<unsigned int>(dft_pair_reduction_frequency_threads),
      static_cast<unsigned int>(dft_pair_reduction_point_threads));
  const std::size_t partial_shared_bytes =
      2 * dft_pair_reduction_frequency_threads *
      dft_pair_reduction_point_threads * sizeof(double);
  dft_pair_reduce_fp32_partials_kernel<<<
      grid, block, partial_shared_bytes>>>(
      operations, block_operation_indices, total_spatial_block_count,
      frequency_count, device_partials);
  check(cudaGetLastError(),
        "dft_pair_reduce_fp32 partial kernel launch");

  constexpr int final_threads = 256;
  const std::size_t final_blocks =
      std::min(frequency_count, launch_device.max_grid_x);
  dft_pair_reduce_fp32_final_kernel<<<
      static_cast<unsigned int>(final_blocks), final_threads,
      2u * static_cast<std::size_t>(final_threads) * sizeof(double)>>>(
      device_partials, physical_partial_count, frequency_count,
      device_result_real_imag);
  check(cudaGetLastError(),
        "dft_pair_reduce_fp32 final kernel launch");
}

void validate_eigenmode_overlap_operations_fp32(
    const eigenmode_overlap_operation_fp32 *operations,
    const std::uint32_t *block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count,
    std::size_t selected_frequency_index, std::size_t output_count) {
  if (output_count == 0 ||
      output_count > eigenmode_overlap_max_output_count)
    throw std::out_of_range(
        "eigenmode-overlap output count is invalid");
  if (operation_count == 0) {
    if (total_block_count != 0)
      throw std::invalid_argument(
          "eigenmode-overlap blocks require host operations");
    return;
  }
  if (!operations || !block_operation_indices)
    throw std::invalid_argument(
        "eigenmode-overlap host operations and block map must be non-null");
  if (operation_count >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
    throw std::overflow_error(
        "eigenmode-overlap operation count exceeds block-map capacity");
  std::size_t expected_block_start = 0;
  for (std::size_t operation_index = 0;
       operation_index < operation_count; ++operation_index) {
    const eigenmode_overlap_operation_fp32 &operation =
        operations[operation_index];
    const bool mode_flux = operation.dft_real_imag != nullptr;
    const bool mode_mode = operation.mode2_real_imag != nullptr;
    if (!operation.weighted_conjugate_mode ||
        operation.point_count == 0 || mode_flux == mode_mode)
      throw std::invalid_argument(
          "eigenmode-overlap operation is incomplete");
    if (operation.output_index >= output_count)
      throw std::out_of_range(
          "eigenmode-overlap output index is invalid");
    if (operation.block_start != expected_block_start)
      throw std::invalid_argument(
          "eigenmode-overlap descriptor block prefixes are inconsistent");
    if (mode_flux) {
      if (operation.dft_frequency_count == 0 ||
          selected_frequency_index >= operation.dft_frequency_count)
        throw std::out_of_range(
            "eigenmode-overlap DFT frequency is outside its source");
      if (!std::isfinite(operation.inverse_stored_weight.real) ||
          !std::isfinite(operation.inverse_stored_weight.imag))
        throw std::invalid_argument(
            "eigenmode-overlap inverse stored weight must be finite");
      if (operation.point_count >
          std::numeric_limits<std::size_t>::max() /
              operation.dft_frequency_count)
        throw std::overflow_error(
            "eigenmode-overlap DFT point-frequency count overflow");
      const std::size_t complex_source_count =
          operation.point_count * operation.dft_frequency_count;
      if (complex_source_count >
          std::numeric_limits<std::size_t>::max() / 2)
        throw std::overflow_error(
            "eigenmode-overlap interleaved DFT scalar count overflow");
    }
    else if (operation.dft_frequency_count != 0 ||
             operation.zero_normalization_divisors)
      throw std::invalid_argument(
          "mode-mode overlap must not carry DFT normalization metadata");
    const std::size_t blocks =
        operation.point_count / eigenmode_overlap_threads_per_block +
        (operation.point_count % eigenmode_overlap_threads_per_block != 0);
    if (blocks > std::numeric_limits<std::size_t>::max() -
                     expected_block_start)
      throw std::overflow_error(
          "eigenmode-overlap block count overflow");
    expected_block_start += blocks;
  }
  if (expected_block_start != total_block_count)
    throw std::invalid_argument(
        "eigenmode-overlap logical block count is inconsistent");
  for (std::size_t block = 0; block < total_block_count; ++block) {
    const std::uint32_t operation_index = block_operation_indices[block];
    if (operation_index >= operation_count)
      throw std::out_of_range(
          "eigenmode-overlap block owner is outside the descriptor array");
    const std::size_t begin = operations[operation_index].block_start;
    const std::size_t end =
        operation_index + 1 < operation_count
            ? operations[operation_index + 1].block_start
            : total_block_count;
    if (block < begin || block >= end)
      throw std::invalid_argument(
          "eigenmode-overlap block map does not match descriptor prefixes");
  }
}

void eigenmode_overlap_reduce_fp32(
    const eigenmode_overlap_operation_fp32 *operations,
    const std::uint32_t *block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count,
    std::size_t selected_frequency_index, double *device_partials,
    std::size_t partial_capacity, std::size_t output_count,
    double *device_result_real_imag) {
  if (output_count == 0 ||
      output_count > eigenmode_overlap_max_output_count)
    throw std::out_of_range(
        "eigenmode-overlap output count is invalid");
  if (operation_count == 0) {
    if (total_block_count != 0)
      throw std::invalid_argument(
          "eigenmode-overlap blocks require operations");
    return;
  }
  if (!operations || !block_operation_indices || !device_partials ||
      !device_result_real_imag)
    throw std::invalid_argument(
        "eigenmode-overlap device pointers must be non-null");
  if (total_block_count == 0 || operation_count > total_block_count)
    throw std::invalid_argument(
        "eigenmode-overlap operation/block counts are invalid");
  if (operation_count >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
    throw std::overflow_error(
        "eigenmode-overlap operation count exceeds block-map capacity");
  const std::size_t physical_block_count = std::min(
      std::min(total_block_count, launch_device.max_grid_x),
      eigenmode_overlap_partial_capacity);
  if (physical_block_count == 0 ||
      partial_capacity < physical_block_count)
    throw std::invalid_argument(
        "eigenmode-overlap partial workspace is too small");
  const std::size_t partial_scalars_per_block = 2 * output_count;
  if (partial_capacity >
      std::numeric_limits<std::size_t>::max() /
          partial_scalars_per_block)
    throw std::overflow_error(
        "eigenmode-overlap partial count overflow");
  const std::size_t partial_scalar_count =
      partial_scalars_per_block * partial_capacity;
  if (partial_scalar_count >
      std::numeric_limits<std::size_t>::max() / sizeof(double))
    throw std::overflow_error(
        "eigenmode-overlap partial byte count overflow");
  const std::size_t partial_bytes =
      partial_scalar_count * sizeof(double);
  const std::size_t result_bytes =
      2 * output_count * sizeof(double);
  const std::uintptr_t partial_begin =
      reinterpret_cast<std::uintptr_t>(device_partials);
  const std::uintptr_t result_begin =
      reinterpret_cast<std::uintptr_t>(device_result_real_imag);
  if (partial_begin >
          std::numeric_limits<std::uintptr_t>::max() - partial_bytes ||
      result_begin >
          std::numeric_limits<std::uintptr_t>::max() - result_bytes)
    throw std::overflow_error(
        "eigenmode-overlap workspace address overflow");
  const std::uintptr_t partial_end = partial_begin + partial_bytes;
  const std::uintptr_t result_end = result_begin + result_bytes;
  if (partial_begin < result_end && result_begin < partial_end)
    throw std::invalid_argument(
        "eigenmode-overlap partial workspace and result must not overlap");

  constexpr unsigned int threads =
      static_cast<unsigned int>(eigenmode_overlap_threads_per_block);
  (void)launch_block_count(1, static_cast<int>(threads));
  eigenmode_overlap_fp32_partials_kernel<<<
      static_cast<unsigned int>(physical_block_count), threads,
      2u * static_cast<std::size_t>(threads) * sizeof(double)>>>(
      operations, block_operation_indices, total_block_count,
      selected_frequency_index, device_partials,
      physical_block_count, output_count);
  check(cudaGetLastError(),
        "eigenmode_overlap_reduce_fp32 partial kernel launch");
  eigenmode_overlap_fp32_final_kernel<<<
      static_cast<unsigned int>(output_count), threads,
      2u * static_cast<std::size_t>(threads) * sizeof(double)>>>(
      device_partials, physical_block_count,
      device_result_real_imag);
  check(cudaGetLastError(),
        "eigenmode_overlap_reduce_fp32 final kernel launch");
}

void apply_boundary_fp32(const boundary_operation_fp32 *operations,
                         float *staging_real_imag, std::size_t count,
                         int threads_per_block) {
  if (!operations || !staging_real_imag)
    throw std::invalid_argument(
        "boundary operations and staging pointers must be non-null");
  if (count == 0) return;
  const unsigned int blocks = launch_block_count(count, threads_per_block);
  gather_boundary_fp32_kernel<<<blocks, threads_per_block>>>(
      operations, staging_real_imag, count);
  check(cudaGetLastError(), "gather_boundary_fp32 kernel launch");
  scatter_boundary_fp32_kernel<<<blocks, threads_per_block>>>(
      operations, staging_real_imag, count);
  check(cudaGetLastError(), "scatter_boundary_fp32 kernel launch");
}

void zero_boundary_fp32(const boundary_operation_fp32 *operations,
                        std::size_t count, int threads_per_block) {
  if (!operations)
    throw std::invalid_argument(
        "zero-boundary operations pointer must be non-null");
  if (count == 0) return;
  const unsigned int blocks = launch_block_count(count, threads_per_block);
  zero_boundary_fp32_kernel<<<blocks, threads_per_block>>>(operations, count);
  check(cudaGetLastError(), "zero_boundary_fp32 kernel launch");
}

void apply_nonalias_boundary_fp32(
    const boundary_operation_fp32 *operations, std::size_t count,
    int threads_per_block) {
  if (!operations)
    throw std::invalid_argument(
        "nonalias-boundary operations pointer must be non-null");
  if (count == 0) return;
  const unsigned int blocks = launch_block_count(count, threads_per_block);
  apply_nonalias_boundary_fp32_kernel<<<blocks, threads_per_block>>>(
      operations, count);
  check(cudaGetLastError(), "apply_nonalias_boundary_fp32 kernel launch");
}

struct boundary_graph_fp32 {
  cudaGraphExec_t executable = nullptr;
};

boundary_graph_fp32 *create_boundary_graph_fp32(
    const boundary_operation_fp32 *operations, float *staging_real_imag,
    std::size_t count, int threads_per_block) {
  if (!operations || !staging_real_imag)
    throw std::invalid_argument(
        "boundary graph operations and staging pointers must be non-null");
  if (count == 0)
    throw std::invalid_argument(
        "boundary graph operation count must be nonzero");

  const unsigned int blocks = launch_block_count(count, threads_per_block);
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  try {
    check(cudaGraphCreate(&graph, 0), "cudaGraphCreate boundary graph");

    cudaKernelNodeParams gather_parameters{};
    gather_parameters.func =
        reinterpret_cast<void *>(gather_boundary_fp32_kernel);
    gather_parameters.gridDim = dim3(blocks);
    gather_parameters.blockDim = dim3(threads_per_block);
    void *gather_arguments[] = {&operations, &staging_real_imag, &count};
    gather_parameters.kernelParams = gather_arguments;
    cudaGraphNode_t gather_node = nullptr;
    check(cudaGraphAddKernelNode(&gather_node, graph, nullptr, 0,
                                 &gather_parameters),
          "cudaGraphAddKernelNode boundary gather");

    cudaKernelNodeParams scatter_parameters{};
    scatter_parameters.func =
        reinterpret_cast<void *>(scatter_boundary_fp32_kernel);
    scatter_parameters.gridDim = dim3(blocks);
    scatter_parameters.blockDim = dim3(threads_per_block);
    const float *staging_input = staging_real_imag;
    void *scatter_arguments[] = {&operations, &staging_input, &count};
    scatter_parameters.kernelParams = scatter_arguments;
    cudaGraphNode_t scatter_node = nullptr;
    check(cudaGraphAddKernelNode(&scatter_node, graph, &gather_node, 1,
                                 &scatter_parameters),
          "cudaGraphAddKernelNode boundary scatter");

    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
          "cudaGraphInstantiate boundary graph");
    check(cudaGraphDestroy(graph), "cudaGraphDestroy boundary graph");
    graph = nullptr;

    boundary_graph_fp32 *result = new boundary_graph_fp32;
    result->executable = executable;
    return result;
  }
  catch (...) {
    if (executable) (void)cudaGraphExecDestroy(executable);
    if (graph) (void)cudaGraphDestroy(graph);
    throw;
  }
}

void launch_boundary_graph_fp32(boundary_graph_fp32 *graph) {
  if (!graph || !graph->executable)
    throw std::invalid_argument(
        "boundary graph executable must be non-null");
  check(cudaGraphLaunch(graph->executable, nullptr),
        "cudaGraphLaunch boundary graph");
}

void destroy_boundary_graph_fp32(boundary_graph_fp32 *graph) noexcept {
  if (!graph) return;
  if (graph->executable) (void)cudaGraphExecDestroy(graph->executable);
  delete graph;
}

struct boundary_phase_graph_fp32 {
  cudaGraphExec_t executable = nullptr;
};

boundary_phase_graph_fp32 *create_boundary_phase_graph_fp32(
    const boundary_phase_stage_fp32 *stages, std::size_t stage_count,
    void *completion_event, int threads_per_block) {
  if (!stages)
    throw std::invalid_argument(
        "boundary phase graph stages must be non-null");
  if (stage_count == 0)
    throw std::invalid_argument(
        "boundary phase graph stage count must be nonzero");
  // Validate the complete descriptor set before creating any CUDA object.
  // Besides making construction transactional, this preserves the runtime
  // validation suite's ability to exercise argument contracts without a GPU.
  for (std::size_t index = 0; index < stage_count; ++index) {
    const boundary_phase_stage_fp32 &stage = stages[index];
    if (!stage.operations)
      throw std::invalid_argument(
          "boundary phase graph operations must be non-null");
    if (stage.count == 0)
      throw std::invalid_argument(
          "boundary phase graph operation count must be nonzero");
    if (stage.kind != boundary_phase_stage_kind::zero &&
        stage.kind != boundary_phase_stage_kind::nonalias &&
        stage.kind != boundary_phase_stage_kind::ordered)
      throw std::invalid_argument(
          "boundary phase graph stage kind is invalid");
    if (stage.kind == boundary_phase_stage_kind::ordered &&
        !stage.staging_real_imag)
      throw std::invalid_argument(
          "ordered boundary phase graph staging must be non-null");
    (void)launch_block_count(stage.count, threads_per_block);
  }

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  try {
    check(cudaGraphCreate(&graph, 0),
          "cudaGraphCreate boundary phase graph");
    cudaGraphNode_t tail = nullptr;

    for (std::size_t index = 0; index < stage_count; ++index) {
      const boundary_phase_stage_fp32 &stage = stages[index];
      const unsigned int blocks =
          launch_block_count(stage.count, threads_per_block);
      const cudaGraphNode_t *dependencies = tail ? &tail : nullptr;
      const std::size_t dependency_count = tail ? 1u : 0u;
      cudaGraphNode_t stage_tail = nullptr;

      if (stage.kind == boundary_phase_stage_kind::zero) {
        cudaKernelNodeParams parameters{};
        parameters.func =
            reinterpret_cast<void *>(zero_boundary_fp32_kernel);
        parameters.gridDim = dim3(blocks);
        parameters.blockDim = dim3(threads_per_block);
        const boundary_operation_fp32 *operations = stage.operations;
        std::size_t count = stage.count;
        void *arguments[] = {&operations, &count};
        parameters.kernelParams = arguments;
        check(cudaGraphAddKernelNode(
                  &stage_tail, graph, dependencies, dependency_count,
                  &parameters),
              "cudaGraphAddKernelNode boundary phase zero");
      }
      else if (stage.kind == boundary_phase_stage_kind::nonalias) {
        cudaKernelNodeParams parameters{};
        parameters.func = reinterpret_cast<void *>(
            apply_nonalias_boundary_fp32_kernel);
        parameters.gridDim = dim3(blocks);
        parameters.blockDim = dim3(threads_per_block);
        const boundary_operation_fp32 *operations = stage.operations;
        std::size_t count = stage.count;
        void *arguments[] = {&operations, &count};
        parameters.kernelParams = arguments;
        check(cudaGraphAddKernelNode(
                  &stage_tail, graph, dependencies, dependency_count,
                  &parameters),
              "cudaGraphAddKernelNode boundary phase nonalias");
      }
      else if (stage.kind == boundary_phase_stage_kind::ordered) {
        cudaKernelNodeParams gather_parameters{};
        gather_parameters.func =
            reinterpret_cast<void *>(gather_boundary_fp32_kernel);
        gather_parameters.gridDim = dim3(blocks);
        gather_parameters.blockDim = dim3(threads_per_block);
        const boundary_operation_fp32 *operations = stage.operations;
        float *staging = stage.staging_real_imag;
        std::size_t count = stage.count;
        void *gather_arguments[] = {&operations, &staging, &count};
        gather_parameters.kernelParams = gather_arguments;
        cudaGraphNode_t gather_node = nullptr;
        check(cudaGraphAddKernelNode(
                  &gather_node, graph, dependencies, dependency_count,
                  &gather_parameters),
              "cudaGraphAddKernelNode boundary phase gather");

        cudaKernelNodeParams scatter_parameters{};
        scatter_parameters.func =
            reinterpret_cast<void *>(scatter_boundary_fp32_kernel);
        scatter_parameters.gridDim = dim3(blocks);
        scatter_parameters.blockDim = dim3(threads_per_block);
        const float *staging_input = stage.staging_real_imag;
        void *scatter_arguments[] = {
            &operations, &staging_input, &count};
        scatter_parameters.kernelParams = scatter_arguments;
        check(cudaGraphAddKernelNode(
                  &stage_tail, graph, &gather_node, 1,
                  &scatter_parameters),
              "cudaGraphAddKernelNode boundary phase scatter");
      }

      if (stage.completion_event) {
        cudaGraphNode_t event_node = nullptr;
        check(cudaGraphAddEventRecordNode(
                  &event_node, graph, &stage_tail, 1,
                  reinterpret_cast<cudaEvent_t>(stage.completion_event)),
              "cudaGraphAddEventRecordNode boundary phase");
        tail = event_node;
      }
      else {
        tail = stage_tail;
      }
    }

    if (completion_event) {
      cudaGraphNode_t completion_node = nullptr;
      check(cudaGraphAddEventRecordNode(
                &completion_node, graph, &tail, 1,
                reinterpret_cast<cudaEvent_t>(completion_event)),
            "cudaGraphAddEventRecordNode boundary phase completion");
      tail = completion_node;
    }

    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
          "cudaGraphInstantiate boundary phase graph");
    check(cudaGraphDestroy(graph),
          "cudaGraphDestroy boundary phase graph");
    graph = nullptr;

    boundary_phase_graph_fp32 *result = new boundary_phase_graph_fp32;
    result->executable = executable;
    return result;
  }
  catch (...) {
    if (executable) (void)cudaGraphExecDestroy(executable);
    if (graph) (void)cudaGraphDestroy(graph);
    throw;
  }
}

void launch_boundary_phase_graph_fp32(boundary_phase_graph_fp32 *graph) {
  if (!graph || !graph->executable)
    throw std::invalid_argument(
        "boundary phase graph executable must be non-null");
  check(cudaGraphLaunch(graph->executable, nullptr),
        "cudaGraphLaunch boundary phase graph");
}

void destroy_boundary_phase_graph_fp32(
    boundary_phase_graph_fp32 *graph) noexcept {
  if (!graph) return;
  if (graph->executable) (void)cudaGraphExecDestroy(graph->executable);
  delete graph;
}

void initialize_finite_result(int *device_result) {
  if (!device_result)
    throw std::invalid_argument(
        "finite-check result pointer must be non-null");
  initialize_finite_result_kernel<<<1, 1>>>(device_result);
  check(cudaGetLastError(), "initialize_finite_result kernel launch");
}

void all_finite_fp32(const array_span_fp32 *spans, std::size_t span_count,
                     std::size_t maximum_span_count, int *device_result,
                     int threads_per_block) {
  if (!spans)
    throw std::invalid_argument(
        "finite-check spans pointer must be non-null");
  if (!device_result)
    throw std::invalid_argument(
        "finite-check result pointer must be non-null");
  if (span_count == 0) return;
  if (maximum_span_count == 0)
    throw std::invalid_argument(
        "finite-check maximum span count must be nonzero");
  const unsigned int blocks =
      launch_block_count(maximum_span_count, threads_per_block);
  const unsigned int span_blocks = static_cast<unsigned int>(
      std::min<std::size_t>(span_count, 65535));
  all_finite_fp32_kernel<<<dim3(blocks, span_blocks), threads_per_block>>>(
      spans, span_count, device_result);
  check(cudaGetLastError(), "all_finite_fp32 kernel launch");
}

void clear_finite_generation_result(std::uint32_t *device_result) {
  if (!device_result)
    throw std::invalid_argument(
        "finite-generation result pointer must be non-null");
  check(cudaMemset(device_result, 0, sizeof(*device_result)),
        "cudaMemset finite-generation result");
}

void all_finite_generation_fp32(
    const array_span_fp32 *spans, std::size_t span_count,
    std::size_t maximum_span_count, std::uint32_t generation,
    std::uint32_t *device_result, int threads_per_block) {
  if (!spans)
    throw std::invalid_argument(
        "finite-generation spans pointer must be non-null");
  if (!device_result)
    throw std::invalid_argument(
        "finite-generation result pointer must be non-null");
  if (span_count == 0) return;
  if (maximum_span_count == 0)
    throw std::invalid_argument(
        "finite-generation maximum span count must be nonzero");
  if (generation == 0)
    throw std::invalid_argument(
        "finite-generation token must be nonzero");
  const unsigned int blocks =
      launch_block_count(maximum_span_count, threads_per_block);
  const unsigned int span_blocks = static_cast<unsigned int>(
      std::min<std::size_t>(span_count, 65535));
  all_finite_generation_fp32_kernel<<<dim3(blocks, span_blocks),
                                      threads_per_block>>>(
      spans, span_count, generation, device_result);
  check(cudaGetLastError(), "all_finite_generation_fp32 kernel launch");
}

void initialize_squared_norm_result(double *device_result) {
  if (!device_result)
    throw std::invalid_argument(
        "squared-norm result pointer must be non-null");
  initialize_squared_norm_result_kernel<<<1, 1>>>(device_result);
  check(cudaGetLastError(), "initialize_squared_norm_result kernel launch");
}

void vector_fill_fp32(float *values, std::size_t count, float value,
                      int threads_per_block) {
  if (!values && count)
    throw std::invalid_argument("vector-fill pointer must be non-null");
  if (count == 0) return;
  const unsigned int blocks = launch_block_count(count, threads_per_block);
  vector_fill_fp32_kernel<<<blocks, threads_per_block>>>(values, count, value);
  check(cudaGetLastError(), "vector_fill_fp32 kernel launch");
}

void vector_copy_fp32(float *destination, const float *source,
                      std::size_t count) {
  if ((!destination || !source) && count)
    throw std::invalid_argument("vector-copy pointers must be non-null");
  if (count == 0 || destination == source) return;
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(float))
    throw std::overflow_error("vector-copy byte count overflow");
  check(cudaMemcpyAsync(destination, source, count * sizeof(float),
                        cudaMemcpyDeviceToDevice),
        "cudaMemcpyAsync device-to-device vector copy");
}

void vector_scale_fp32(float *values, std::size_t count, double scale,
                       int threads_per_block) {
  if (!values && count)
    throw std::invalid_argument("vector-scale pointer must be non-null");
  if (count == 0) return;
  const unsigned int blocks = launch_block_count(count, threads_per_block);
  vector_scale_fp32_kernel<<<blocks, threads_per_block>>>(values, count,
                                                          scale);
  check(cudaGetLastError(), "vector_scale_fp32 kernel launch");
}

void complex_scale_inplace_fp32(float *values_real_imag,
                                std::size_t count, double scale_real,
                                double scale_imaginary,
                                int threads_per_block) {
  if (!std::isfinite(scale_real) || !std::isfinite(scale_imaginary))
    throw std::invalid_argument(
        "complex FP32 scale must have finite real and imaginary parts");
  if (!values_real_imag && count)
    throw std::invalid_argument(
        "complex FP32 scale pointer must be non-null");
  if (count == 0) return;
  if (count > std::numeric_limits<std::size_t>::max() / 2)
    throw std::overflow_error("complex FP32 scale scalar count overflow");
  const unsigned int blocks = launch_block_count(count, threads_per_block);
  complex_scale_inplace_fp32_kernel<<<blocks, threads_per_block>>>(
      values_real_imag, count, scale_real, scale_imaginary);
  check(cudaGetLastError(), "complex_scale_inplace_fp32 kernel launch");
}

void vector_xpay_fp32(float *x, const float *y, std::size_t count,
                      double scale, int threads_per_block) {
  if ((!x || !y) && count)
    throw std::invalid_argument("vector-xpay pointers must be non-null");
  if (count == 0) return;
  const unsigned int blocks = launch_block_count(count, threads_per_block);
  vector_xpay_fp32_kernel<<<blocks, threads_per_block>>>(x, y, count, scale);
  check(cudaGetLastError(), "vector_xpay_fp32 kernel launch");
}

void vector_left_minus_scale_fp32(float *output, const float *left,
                                  std::size_t count, double scale,
                                  int threads_per_block) {
  if ((!output || !left) && count)
    throw std::invalid_argument(
        "vector left-minus-scale pointers must be non-null");
  if (count == 0) return;
  const unsigned int blocks = launch_block_count(count, threads_per_block);
  vector_left_minus_scale_fp32_kernel<<<blocks, threads_per_block>>>(
      output, left, count, scale);
  check(cudaGetLastError(),
        "vector_left_minus_scale_fp32 kernel launch");
}

void initialize_fp64_result(double *device_result) {
  initialize_squared_norm_result(device_result);
}

void initialize_fp32_result(float *device_result, float value) {
  if (!device_result)
    throw std::invalid_argument("FP32 result pointer must be non-null");
  initialize_fp32_result_kernel<<<1, 1>>>(device_result, value);
  check(cudaGetLastError(), "initialize_fp32_result kernel launch");
}

void initialize_int_result(int *device_result, int value) {
  if (!device_result)
    throw std::invalid_argument("integer result pointer must be non-null");
  initialize_int_result_kernel<<<1, 1>>>(device_result, value);
  check(cudaGetLastError(), "initialize_int_result kernel launch");
}

void vector_dot_fp32(const float *x, const float *y, std::size_t count,
                     double *device_partials,
                     std::size_t partial_capacity,
                     double *device_result, int threads_per_block) {
  if ((!x || !y) && count)
    throw std::invalid_argument("vector-dot pointers must be non-null");
  if (!device_partials)
    throw std::invalid_argument(
        "vector-dot partial workspace must be non-null");
  if (!device_result)
    throw std::invalid_argument("vector-dot result must be non-null");
  if (count == 0) return;
  unsigned int blocks = launch_block_count(count, threads_per_block);
  blocks = std::min(
      blocks,
      static_cast<unsigned int>(vector_reduction_partial_capacity));
  if (partial_capacity < blocks)
    throw std::invalid_argument(
        "vector-dot partial workspace is too small");
  vector_dot_fp32_partials_kernel<<<
      blocks, threads_per_block,
      static_cast<std::size_t>(threads_per_block) * sizeof(double)>>>(
      x, y, count, device_partials);
  check(cudaGetLastError(), "vector_dot_fp32 partial kernel launch");
  constexpr int final_threads = 256;
  reduce_fp64_partials_kernel<<<
      1, final_threads,
      static_cast<std::size_t>(final_threads) * sizeof(double)>>>(
      device_partials, blocks, device_result);
  check(cudaGetLastError(), "vector_dot_fp32 final kernel launch");
}

void vector_max_abs_fp32(const float *values, std::size_t count,
                         float *device_result, int *device_all_finite,
                         int threads_per_block) {
  if (!values && count)
    throw std::invalid_argument("vector-max pointer must be non-null");
  if (!device_result)
    throw std::invalid_argument("vector-max result must be non-null");
  if (count == 0) return;
  unsigned int blocks = launch_block_count(count, threads_per_block);
  blocks = std::min(blocks, 1024u);
  vector_max_abs_fp32_kernel<<<blocks, threads_per_block>>>(
      values, count, device_result, device_all_finite);
  check(cudaGetLastError(), "vector_max_abs_fp32 kernel launch");
}

void vector_scaled_sum_squares_fp32(const float *values,
                                    std::size_t count, double scale,
                                    double *device_partials,
                                    std::size_t partial_capacity,
                                    double *device_result,
                                    int threads_per_block) {
  if (!values && count)
    throw std::invalid_argument(
        "vector scaled-sum-squares pointer must be non-null");
  if (!device_partials)
    throw std::invalid_argument(
        "vector scaled-sum-squares partial workspace must be non-null");
  if (!device_result)
    throw std::invalid_argument(
        "vector scaled-sum-squares result must be non-null");
  if (count == 0) return;
  unsigned int blocks = launch_block_count(count, threads_per_block);
  blocks = std::min(
      blocks,
      static_cast<unsigned int>(vector_reduction_partial_capacity));
  if (partial_capacity < blocks)
    throw std::invalid_argument(
        "vector scaled-sum-squares partial workspace is too small");
  vector_scaled_sum_squares_fp32_partials_kernel<<<
      blocks, threads_per_block,
      static_cast<std::size_t>(threads_per_block) * sizeof(double)>>>(
      values, count, scale, device_partials);
  check(cudaGetLastError(),
        "vector_scaled_sum_squares_fp32 partial kernel launch");
  constexpr int final_threads = 256;
  reduce_fp64_partials_kernel<<<
      1, final_threads,
      static_cast<std::size_t>(final_threads) * sizeof(double)>>>(
      device_partials, blocks, device_result);
  check(cudaGetLastError(),
        "vector_scaled_sum_squares_fp32 final kernel launch");
}

namespace {

dim3 cw_field_vector_grid(std::size_t segment_count,
                          std::size_t maximum_point_count,
                          int threads_per_block) {
  if (segment_count == 0 || maximum_point_count == 0)
    throw std::invalid_argument(
        "CW field-vector launch requires nonempty segments");
  if (segment_count > 65535)
    throw std::overflow_error(
        "CW field-vector segment count exceeds CUDA grid capacity");
  return dim3(launch_block_count(maximum_point_count, threads_per_block),
              static_cast<unsigned int>(segment_count));
}

} // namespace

void gather_cw_field_vector_fp32(
    const cw_field_vector_segment_fp32 *segments,
    std::size_t segment_count, std::size_t maximum_point_count,
    float *packed_real_imag, int threads_per_block) {
  if (!segments || !packed_real_imag)
    throw std::invalid_argument(
        "CW field-vector gather pointers must be non-null");
  const dim3 grid = cw_field_vector_grid(
      segment_count, maximum_point_count, threads_per_block);
  gather_cw_field_vector_fp32_kernel<<<grid, threads_per_block>>>(
      segments, segment_count, packed_real_imag);
  check(cudaGetLastError(), "gather_cw_field_vector_fp32 kernel launch");
}

void scatter_cw_field_vector_fp32(
    const cw_field_vector_segment_fp32 *segments,
    std::size_t segment_count, std::size_t maximum_point_count,
    const float *packed_real_imag, int threads_per_block) {
  if (!segments || !packed_real_imag)
    throw std::invalid_argument(
        "CW field-vector scatter pointers must be non-null");
  const dim3 grid = cw_field_vector_grid(
      segment_count, maximum_point_count, threads_per_block);
  scatter_cw_field_vector_fp32_kernel<<<grid, threads_per_block>>>(
      segments, segment_count, packed_real_imag);
  check(cudaGetLastError(), "scatter_cw_field_vector_fp32 kernel launch");
}

void gather_cw_field_operator_fp32(
    const cw_field_vector_segment_fp32 *segments,
    std::size_t segment_count, std::size_t maximum_point_count,
    const float *input_real_imag, float *output_real_imag,
    float dt_inverse, float iomega_real, float iomega_imaginary,
    int threads_per_block) {
  if (!segments || !input_real_imag || !output_real_imag)
    throw std::invalid_argument(
        "CW field-operator gather pointers must be non-null");
  const dim3 grid = cw_field_vector_grid(
      segment_count, maximum_point_count, threads_per_block);
  gather_cw_field_operator_fp32_kernel<<<grid, threads_per_block>>>(
      segments, segment_count, input_real_imag, output_real_imag,
      dt_inverse, iomega_real, iomega_imaginary);
  check(cudaGetLastError(),
        "gather_cw_field_operator_fp32 kernel launch");
}

void squared_norm_complex_fp32(
    const float *values_real_imag, const std::ptrdiff_t *point_indices,
    std::size_t point_count, std::size_t frequency_count,
    double *device_result, int threads_per_block) {
  if (!values_real_imag)
    throw std::invalid_argument(
        "squared-norm values pointer must be non-null");
  if (!device_result)
    throw std::invalid_argument(
        "squared-norm result pointer must be non-null");
  const std::size_t work_count =
      dft_work_count(point_count, frequency_count);
  if (work_count == 0) return;
  unsigned int blocks =
      launch_block_count(work_count, threads_per_block);
  // One FP64 atomic per block is enough to saturate this bandwidth-bound
  // reduction; bounding the grid avoids needless atomic contention.
  blocks = std::min(blocks, 1024u);
  squared_norm_complex_fp32_kernel<<<
      blocks, threads_per_block,
      static_cast<std::size_t>(threads_per_block) * sizeof(double)>>>(
      values_real_imag, point_indices, point_count, frequency_count,
      device_result);
  check(cudaGetLastError(), "squared_norm_complex_fp32 kernel launch");
}

void update_dft_fp32(
    float *dft_real_imag, const float *field_real, const float *field_imag,
    const std::ptrdiff_t *field_indices, const float *weights,
    std::size_t point_count, const complex_value_fp32 *phases,
    std::size_t frequency_count, std::ptrdiff_t average_offset1,
    std::ptrdiff_t average_offset2, int threads_per_block) {
  if (!dft_real_imag || !field_real || !field_indices || !weights || !phases)
    throw std::invalid_argument(
        "DFT output, field, indices, weights, and phases must be non-null");
  const std::size_t work_count =
      dft_work_count(point_count, frequency_count);
  if (work_count == 0) return;
  launch_update_dft_fp32(
      dft_real_imag, field_real, field_imag, field_indices, weights,
      point_count, phases, frequency_count, average_offset1,
      average_offset2, threads_per_block,
      "update_dft_fp32 kernel launch");
}

void validate_dft_batch_operations_fp32(
    const dft_update_operation_fp32 *host_operations,
    std::size_t operation_count, std::size_t total_block_count) {
  if (operation_count > 0 && !host_operations)
    throw std::invalid_argument(
        "DFT batch host operations must be non-null");
  if (operation_count >
      static_cast<std::size_t>(
          std::numeric_limits<std::uint32_t>::max()))
    throw std::overflow_error("DFT batch operation index overflow");
  if (operation_count == 0) {
    if (total_block_count != 0)
      throw std::invalid_argument(
          "DFT batch logical-block count requires host operations");
    return;
  }

  std::size_t expected_block_start = 0;
  for (std::size_t index = 0; index < operation_count; ++index) {
    const dft_update_operation_fp32 &operation = host_operations[index];
    if (!operation.dft_real_imag || !operation.field_real ||
        !operation.field_indices || !operation.weights || !operation.phases)
      throw std::invalid_argument(
          "DFT batch descriptor pointers must be non-null");
    if (operation.point_count == 0 || operation.frequency_count == 0)
      throw std::invalid_argument(
          "DFT batch descriptor counts must be nonzero");
    // The kernel forms a packed complex output index as
    // 2 * (point * frequency_count + frequency).  Reuse the scalar DFT
    // validator so a public low-level caller cannot make that arithmetic or
    // its byte extent wrap even when all tile metadata is self-consistent.
    (void)dft_work_count(
        operation.point_count, operation.frequency_count);
    if (operation.frequency_threads == 0 ||
        operation.frequency_threads > 32 || operation.point_threads == 0 ||
        operation.point_threads >
            static_cast<std::uint32_t>(
                dft_batch_threads_per_block_fp32) /
                operation.frequency_threads)
      throw std::invalid_argument(
          "DFT batch descriptor thread layout is invalid");
    const std::size_t expected_frequency_blocks =
        operation.frequency_count / operation.frequency_threads +
        (operation.frequency_count % operation.frequency_threads != 0);
    const std::size_t expected_point_blocks =
        operation.point_count / operation.point_threads +
        (operation.point_count % operation.point_threads != 0);
    if (operation.frequency_block_count != expected_frequency_blocks ||
        operation.point_block_count != expected_point_blocks)
      throw std::invalid_argument(
          "DFT batch descriptor tile counts are invalid");
    if (operation.block_start != expected_block_start)
      throw std::invalid_argument(
          "DFT batch descriptor block prefixes are not contiguous");
    if (operation.frequency_block_count >
        std::numeric_limits<std::size_t>::max() /
            operation.point_block_count)
      throw std::overflow_error("DFT batch descriptor block count overflow");
    const std::size_t operation_blocks =
        operation.frequency_block_count * operation.point_block_count;
    if (expected_block_start >
        std::numeric_limits<std::size_t>::max() - operation_blocks)
      throw std::overflow_error("DFT batch descriptor block prefix overflow");
    expected_block_start += operation_blocks;
  }
  if (expected_block_start != total_block_count)
    throw std::invalid_argument(
        "DFT batch descriptor blocks do not match the logical-block count");
}

void update_dft_batch_fp32(
    const dft_update_operation_fp32 *device_operations,
    const std::uint32_t *device_block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count) {
  if (!device_operations)
    throw std::invalid_argument(
        "DFT batch operations must be non-null");
  if (operation_count == 0) {
    if (total_block_count != 0)
      throw std::invalid_argument(
          "DFT batch logical-block count requires operations");
    return;
  }
  if (operation_count >
      static_cast<std::size_t>(
          std::numeric_limits<std::uint32_t>::max()))
    throw std::overflow_error("DFT batch operation index overflow");
  if (total_block_count == 0) return;
  if (!device_block_operation_indices)
    throw std::invalid_argument(
        "DFT batch block-operation indices must be non-null");
  const unsigned int blocks = launch_cooperative_block_count(
      total_block_count, dft_batch_threads_per_block_fp32);
  update_dft_batch_fp32_kernel<<<
      blocks, dft_batch_threads_per_block_fp32,
      2 * static_cast<std::size_t>(dft_batch_threads_per_block_fp32) *
          sizeof(float)>>>(
      device_operations, device_block_operation_indices,
      operation_count, total_block_count);
  check(cudaGetLastError(), "update_dft_batch_fp32 kernel launch");
}

void validate_dft_materialization_operations_fp32(
    const dft_materialization_operation_fp32 *host_operations,
    const std::uint32_t *host_block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count,
    std::size_t output_point_count,
    std::size_t output_frequency_count) {
  if (output_point_count == 0 || output_frequency_count == 0)
    throw std::invalid_argument(
        "DFT materialization output dimensions must be nonzero");
  (void)dft_work_count(output_point_count, output_frequency_count);
  if (operation_count > 0 && !host_operations)
    throw std::invalid_argument(
        "DFT materialization host operations must be non-null");
  if (operation_count >
      static_cast<std::size_t>(
          std::numeric_limits<std::uint32_t>::max()))
    throw std::overflow_error(
        "DFT materialization operation index overflow");
  if (operation_count == 0) {
    if (total_block_count != 0)
      throw std::invalid_argument(
          "DFT materialization logical-block count requires host operations");
    return;
  }

  std::size_t expected_block_start = 0;
  std::size_t destination_count = 0;
  for (std::size_t index = 0; index < operation_count; ++index) {
    const dft_materialization_operation_fp32 &operation =
        host_operations[index];
    if (!operation.dft_real_imag || !operation.destination_indices ||
        !operation.point_weights)
      throw std::invalid_argument(
          "DFT materialization descriptor pointers must be non-null");
    if (operation.storage_point_count == 0 || operation.point_count == 0 ||
        operation.source_frequency_count == 0 ||
        operation.selected_frequency_count == 0)
      throw std::invalid_argument(
          "DFT materialization descriptor counts must be nonzero");
    if (operation.point_count > operation.storage_point_count)
      throw std::invalid_argument(
          "DFT materialization point count exceeds source storage extent");
    (void)dft_work_count(operation.storage_point_count,
                         operation.source_frequency_count);
    if (operation.source_frequency_start >=
            operation.source_frequency_count ||
        operation.selected_frequency_count >
            operation.source_frequency_count -
                operation.source_frequency_start)
      throw std::invalid_argument(
          "DFT materialization source frequency range is invalid");
    if (operation.selected_frequency_count != output_frequency_count)
      throw std::invalid_argument(
          "DFT materialization selected frequencies do not match output");
    if (!std::isfinite(operation.inverse_stored_weight.real) ||
        !std::isfinite(operation.inverse_stored_weight.imag))
      throw std::invalid_argument(
          "DFT materialization inverse stored weight must be finite");
    if (operation.block_start != expected_block_start)
      throw std::invalid_argument(
          "DFT materialization descriptor block prefixes are not contiguous");

    const std::size_t operation_work = dft_work_count(
        operation.point_count, operation.selected_frequency_count);
    const std::size_t operation_blocks =
        operation_work /
            static_cast<std::size_t>(
                dft_materialization_threads_per_block_fp32) +
        (operation_work %
             static_cast<std::size_t>(
                 dft_materialization_threads_per_block_fp32) !=
         0);
    if (expected_block_start >
        std::numeric_limits<std::size_t>::max() - operation_blocks)
      throw std::overflow_error(
          "DFT materialization descriptor block prefix overflow");
    expected_block_start += operation_blocks;
    if (destination_count >
        std::numeric_limits<std::size_t>::max() - operation.point_count)
      throw std::overflow_error(
          "DFT materialization destination count overflow");
    destination_count += operation.point_count;
  }
  if (expected_block_start != total_block_count)
    throw std::invalid_argument(
        "DFT materialization descriptor blocks do not match the logical-block count");
  if (!host_block_operation_indices)
    throw std::invalid_argument(
        "DFT materialization host block-operation indices must be non-null");
  for (std::size_t operation_index = 0;
       operation_index < operation_count; ++operation_index) {
    const std::size_t operation_block_end =
        operation_index + 1 < operation_count
            ? host_operations[operation_index + 1].block_start
            : total_block_count;
    for (std::size_t logical_block =
             host_operations[operation_index].block_start;
         logical_block < operation_block_end; ++logical_block)
      if (host_block_operation_indices[logical_block] != operation_index)
        throw std::invalid_argument(
            "DFT materialization block-operation map does not match descriptor prefixes");
  }

  struct destination_publication {
    std::ptrdiff_t destination;
    std::size_t traversal_index;
    std::uint8_t publication;
  };
  std::vector<destination_publication> destinations;
  if (destination_count > destinations.max_size())
    throw std::overflow_error(
        "DFT materialization destination validation size overflow");
  destinations.reserve(destination_count);
  for (std::size_t index = 0; index < operation_count; ++index) {
    const dft_materialization_operation_fp32 &operation =
        host_operations[index];
    for (std::size_t point = 0; point < operation.point_count; ++point) {
      const std::ptrdiff_t destination =
          operation.destination_indices[point];
      if (destination < 0 ||
          static_cast<std::size_t>(destination) >= output_point_count)
        throw std::invalid_argument(
            "DFT materialization destination index is out of range");
      if (!std::isfinite(operation.point_weights[point]))
        throw std::invalid_argument(
            "DFT materialization point weight must be finite");
      if (operation.zero_divisor_flags &&
          operation.zero_divisor_flags[point] > 1)
        throw std::invalid_argument(
            "DFT materialization zero-divisor flag must be zero or one");
      if (operation.publication_flags &&
          operation.publication_flags[point] > 1)
        throw std::invalid_argument(
            "DFT materialization publication flag must be zero or one");
      destinations.push_back(
          {destination, destinations.size(),
           operation.publication_flags
               ? operation.publication_flags[point]
               : static_cast<std::uint8_t>(1)});
    }
  }
  std::sort(destinations.begin(), destinations.end(),
            [](const destination_publication &left,
               const destination_publication &right) {
              if (left.destination != right.destination)
                return left.destination < right.destination;
              return left.traversal_index < right.traversal_index;
            });
  for (std::size_t begin = 0; begin < destinations.size();) {
    std::size_t end = begin + 1;
    while (end < destinations.size() &&
           destinations[end].destination == destinations[begin].destination)
      ++end;
    for (std::size_t index = begin; index < end; ++index) {
      const std::uint8_t expected = index + 1 == end ? 1u : 0u;
      if (destinations[index].publication != expected)
        throw std::invalid_argument(
            "DFT materialization publication flags must preserve "
            "last-writer traversal order");
    }
    begin = end;
  }
}

void materialize_dft_fp32(
    const dft_materialization_operation_fp32 *device_operations,
    const std::uint32_t *device_block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count,
    float *output_real_imag, std::size_t output_point_count,
    std::size_t output_frequency_count) {
  if (!output_real_imag)
    throw std::invalid_argument(
        "DFT materialization output must be non-null");
  if (output_point_count == 0 || output_frequency_count == 0)
    throw std::invalid_argument(
        "DFT materialization output dimensions must be nonzero");
  const std::size_t output_work =
      dft_work_count(output_point_count, output_frequency_count);
  if (operation_count >
      static_cast<std::size_t>(
          std::numeric_limits<std::uint32_t>::max()))
    throw std::overflow_error(
        "DFT materialization operation index overflow");
  if (operation_count == 0) {
    if (total_block_count != 0)
      throw std::invalid_argument(
          "DFT materialization logical-block count requires operations");
  }
  else {
    if (!device_operations)
      throw std::invalid_argument(
          "DFT materialization operations must be non-null");
    if (total_block_count == 0)
      throw std::invalid_argument(
          "DFT materialization operations require logical blocks");
    if (!device_block_operation_indices)
      throw std::invalid_argument(
          "DFT materialization block-operation indices must be non-null");
  }

  check(cudaMemset(output_real_imag, 0,
                   2 * output_work * sizeof(float)),
        "cudaMemset DFT materialization output");
  if (total_block_count == 0) return;
  const unsigned int blocks = launch_cooperative_block_count(
      total_block_count, dft_materialization_threads_per_block_fp32);
  materialize_dft_fp32_kernel<<<
      blocks, dft_materialization_threads_per_block_fp32>>>(
      device_operations, device_block_operation_indices, operation_count,
      total_block_count, output_real_imag, output_point_count);
  check(cudaGetLastError(), "materialize_dft_fp32 kernel launch");
}

void collapse_dft_array_fp32(
    const float *full_output_real_imag, float *reduced_output_real_imag,
    std::size_t full_output_point_count,
    std::size_t reduced_output_point_count,
    std::size_t output_frequency_count,
    const dft_collapse_layout_fp32 &layout, int threads_per_block) {
  if (!full_output_real_imag || !reduced_output_real_imag)
    throw std::invalid_argument(
        "DFT collapse source and destination must be non-null");
  if (layout.full_rank == 0 || layout.full_rank > 3)
    throw std::invalid_argument("DFT collapse rank must be in [1, 3]");
  if (full_output_point_count == 0 || reduced_output_point_count == 0 ||
      output_frequency_count == 0)
    throw std::invalid_argument(
        "DFT collapse dimensions must be nonzero");

  std::size_t full_product = 1;
  std::size_t reduced_product = 1;
  bool has_collapsed_dimension = false;
  for (std::size_t dim = 0; dim < 3; ++dim) {
    if (layout.collapsed[dim] > 1)
      throw std::invalid_argument(
          "DFT collapse flags must be zero or one");
    if (dim >= layout.full_rank) {
      if (layout.full_dims[dim] != 1 || layout.collapsed[dim] != 0)
        throw std::invalid_argument(
            "DFT collapse inactive dimensions must be unit and retained");
      continue;
    }
    const std::size_t extent = layout.full_dims[dim];
    if (extent == 0)
      throw std::invalid_argument(
          "DFT collapse full dimensions must be nonzero");
    if (full_product > std::numeric_limits<std::size_t>::max() / extent)
      throw std::overflow_error("DFT collapse full extent overflow");
    full_product *= extent;
    if (layout.collapsed[dim] != 0)
      has_collapsed_dimension = true;
    else {
      if (reduced_product >
          std::numeric_limits<std::size_t>::max() / extent)
        throw std::overflow_error("DFT collapse reduced extent overflow");
      reduced_product *= extent;
    }
  }
  if (!has_collapsed_dimension)
    throw std::invalid_argument(
        "DFT collapse requires at least one collapsed dimension");
  if (full_product != full_output_point_count ||
      reduced_product != reduced_output_point_count)
    throw std::invalid_argument(
        "DFT collapse layout products differ from allocation extents");

  const std::size_t full_work =
      dft_work_count(full_output_point_count, output_frequency_count);
  const std::size_t reduced_work =
      dft_work_count(reduced_output_point_count, output_frequency_count);
  if (full_work > std::numeric_limits<std::size_t>::max() /
                      (2 * sizeof(float)) ||
      reduced_work > std::numeric_limits<std::size_t>::max() /
                         (2 * sizeof(float)))
    throw std::overflow_error("DFT collapse byte extent overflow");
  const std::uintptr_t source_begin =
      reinterpret_cast<std::uintptr_t>(full_output_real_imag);
  const std::uintptr_t destination_begin =
      reinterpret_cast<std::uintptr_t>(reduced_output_real_imag);
  const std::size_t source_bytes = 2 * full_work * sizeof(float);
  const std::size_t destination_bytes =
      2 * reduced_work * sizeof(float);
  if (source_begin > std::numeric_limits<std::uintptr_t>::max() -
                         source_bytes ||
      destination_begin > std::numeric_limits<std::uintptr_t>::max() -
                              destination_bytes)
    throw std::overflow_error("DFT collapse address extent overflow");
  const std::uintptr_t source_end = source_begin + source_bytes;
  const std::uintptr_t destination_end =
      destination_begin + destination_bytes;
  if (source_begin < destination_end && destination_begin < source_end)
    throw std::invalid_argument(
        "DFT collapse source and destination must not overlap");

  const unsigned int blocks =
      launch_block_count(reduced_work, threads_per_block);
  collapse_dft_array_fp32_kernel<<<blocks, threads_per_block>>>(
      full_output_real_imag, reduced_output_real_imag,
      full_output_point_count, reduced_output_point_count,
      output_frequency_count, layout);
  check(cudaGetLastError(), "collapse_dft_array_fp32 kernel launch");
}

void validate_dft_output_staging_operations_fp32(
    const dft_output_staging_operation_fp32 *host_operations,
    const std::uint32_t *host_block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count,
    std::size_t output_point_count, std::size_t frequency_capacity) {
  if (frequency_capacity == 0)
    throw std::invalid_argument(
        "DFT output staging frequency capacity must be nonzero");
  if (operation_count >
      static_cast<std::size_t>(
          std::numeric_limits<std::uint32_t>::max()))
    throw std::overflow_error(
        "DFT output staging operation index overflow");
  if (operation_count == 0) {
    if (host_operations)
      throw std::invalid_argument(
          "DFT output staging empty topology must not provide operations");
    if (total_block_count != 0)
      throw std::invalid_argument(
          "DFT output staging logical-block count requires host operations");
    if (output_point_count != 0)
      throw std::invalid_argument(
          "DFT output staging empty topology requires zero output points");
    return;
  }
  if (!host_operations)
    throw std::invalid_argument(
        "DFT output staging host operations must be non-null");
  if (output_point_count == 0)
    throw std::invalid_argument(
        "DFT output staging output point count must be nonzero");
  (void)dft_work_count(output_point_count, frequency_capacity);

  std::size_t expected_output_offset = 0;
  std::size_t expected_block_start = 0;
  for (std::size_t index = 0; index < operation_count; ++index) {
    const dft_output_staging_operation_fp32 &operation =
        host_operations[index];
    if (!operation.dft_real_imag || !operation.destination_indices ||
        !operation.point_weights)
      throw std::invalid_argument(
          "DFT output staging descriptor pointers must be non-null");
    if (operation.storage_point_count == 0 || operation.point_count == 0 ||
        operation.source_frequency_count == 0)
      throw std::invalid_argument(
          "DFT output staging descriptor counts must be nonzero");
    if (operation.point_count > operation.storage_point_count)
      throw std::invalid_argument(
          "DFT output staging point count exceeds source storage extent");
    (void)dft_work_count(operation.storage_point_count,
                         operation.source_frequency_count);
    if (!std::isfinite(operation.inverse_stored_weight.real) ||
        !std::isfinite(operation.inverse_stored_weight.imag))
      throw std::invalid_argument(
          "DFT output staging inverse stored weight must be finite");

    if (operation.output_point_offset != expected_output_offset)
      throw std::invalid_argument(
          "DFT output staging output partitions must be gapless and ordered");
    if (expected_output_offset >
        std::numeric_limits<std::size_t>::max() - operation.point_count)
      throw std::overflow_error(
          "DFT output staging output partition overflow");
    expected_output_offset += operation.point_count;
    if (expected_output_offset > output_point_count)
      throw std::invalid_argument(
          "DFT output staging output partition exceeds output extent");

    if (operation.block_start != expected_block_start)
      throw std::invalid_argument(
          "DFT output staging descriptor block prefixes are not contiguous");
    const std::size_t operation_work =
        dft_work_count(operation.point_count, frequency_capacity);
    const std::size_t operation_blocks =
        operation_work /
            static_cast<std::size_t>(
                dft_output_staging_threads_per_block_fp32) +
        (operation_work %
             static_cast<std::size_t>(
                 dft_output_staging_threads_per_block_fp32) !=
         0);
    if (expected_block_start >
        std::numeric_limits<std::size_t>::max() - operation_blocks)
      throw std::overflow_error(
          "DFT output staging descriptor block prefix overflow");
    expected_block_start += operation_blocks;

    std::vector<std::uint8_t> destinations;
    if (operation.point_count > destinations.max_size())
      throw std::overflow_error(
          "DFT output staging permutation validation size overflow");
    destinations.assign(operation.point_count, 0u);
    for (std::size_t point = 0; point < operation.point_count; ++point) {
      const std::ptrdiff_t destination =
          operation.destination_indices[point];
      if (destination < 0 ||
          static_cast<std::size_t>(destination) >= operation.point_count)
        throw std::invalid_argument(
            "DFT output staging local destination is out of range");
      const std::size_t local_destination =
          static_cast<std::size_t>(destination);
      if (destinations[local_destination] != 0)
        throw std::invalid_argument(
            "DFT output staging local destinations must be a permutation");
      destinations[local_destination] = 1u;
      if (!std::isfinite(operation.point_weights[point]))
        throw std::invalid_argument(
            "DFT output staging point weight must be finite");
      if (operation.zero_divisor_flags &&
          operation.zero_divisor_flags[point] > 1)
        throw std::invalid_argument(
            "DFT output staging zero-divisor flag must be zero or one");
    }
  }
  if (expected_output_offset != output_point_count)
    throw std::invalid_argument(
        "DFT output staging output partitions do not cover output extent");
  if (expected_block_start != total_block_count)
    throw std::invalid_argument(
        "DFT output staging descriptor blocks do not match the logical-block count");
  if (!host_block_operation_indices)
    throw std::invalid_argument(
        "DFT output staging host block-operation indices must be non-null");
  for (std::size_t operation_index = 0;
       operation_index < operation_count; ++operation_index) {
    const std::size_t operation_block_end =
        operation_index + 1 < operation_count
            ? host_operations[operation_index + 1].block_start
            : total_block_count;
    for (std::size_t logical_block =
             host_operations[operation_index].block_start;
         logical_block < operation_block_end; ++logical_block)
      if (host_block_operation_indices[logical_block] != operation_index)
        throw std::invalid_argument(
            "DFT output staging block-operation map does not match descriptor prefixes");
  }
}

void validate_dft_output_staging_tile_fp32(
    const dft_output_staging_operation_fp32 *host_operations,
    std::size_t operation_count, std::size_t tile_frequency_start,
    std::size_t tile_frequency_count, std::size_t frequency_capacity) {
  if (frequency_capacity == 0)
    throw std::invalid_argument(
        "DFT output staging frequency capacity must be nonzero");
  if (tile_frequency_count == 0 ||
      tile_frequency_count > frequency_capacity)
    throw std::invalid_argument(
        "DFT output staging tile count must be in [1, frequency capacity]");
  if (tile_frequency_start >
      std::numeric_limits<std::size_t>::max() - tile_frequency_count)
    throw std::overflow_error(
        "DFT output staging tile frequency range overflow");
  if (operation_count >
      static_cast<std::size_t>(
          std::numeric_limits<std::uint32_t>::max()))
    throw std::overflow_error(
        "DFT output staging operation index overflow");
  if (operation_count > 0 && !host_operations)
    throw std::invalid_argument(
        "DFT output staging tile host operations must be non-null");
  for (std::size_t index = 0; index < operation_count; ++index) {
    const std::size_t source_frequency_count =
        host_operations[index].source_frequency_count;
    if (source_frequency_count == 0 ||
        tile_frequency_start >= source_frequency_count ||
        tile_frequency_count >
            source_frequency_count - tile_frequency_start)
      throw std::invalid_argument(
          "DFT output staging tile source frequency range is invalid");
  }
}

void stage_dft_output_fp32(
    const dft_output_staging_operation_fp32 *device_operations,
    const std::uint32_t *device_block_operation_indices,
    std::size_t operation_count, std::size_t total_block_count,
    float *output_planar, std::size_t output_point_count,
    std::size_t tile_frequency_start, std::size_t tile_frequency_count,
    std::size_t frequency_capacity) {
  if (!output_planar)
    throw std::invalid_argument(
        "DFT output staging output must be non-null");
  if (output_point_count == 0 || frequency_capacity == 0)
    throw std::invalid_argument(
        "DFT output staging output dimensions must be nonzero");
  if (tile_frequency_count == 0 ||
      tile_frequency_count > frequency_capacity)
    throw std::invalid_argument(
        "DFT output staging tile count must be in [1, frequency capacity]");
  if (tile_frequency_start >
      std::numeric_limits<std::size_t>::max() - tile_frequency_count)
    throw std::overflow_error(
        "DFT output staging tile frequency range overflow");
  if (operation_count == 0 || !device_operations)
    throw std::invalid_argument(
        "DFT output staging operations must be non-null and nonempty");
  if (operation_count >
      static_cast<std::size_t>(
          std::numeric_limits<std::uint32_t>::max()))
    throw std::overflow_error(
        "DFT output staging operation index overflow");
  if (total_block_count == 0)
    throw std::invalid_argument(
        "DFT output staging operations require logical blocks");
  if (!device_block_operation_indices)
    throw std::invalid_argument(
        "DFT output staging block-operation indices must be non-null");
  const std::size_t output_work =
      dft_work_count(output_point_count, frequency_capacity);
  check(cudaMemset(output_planar, 0,
                   2 * output_work * sizeof(float)),
        "cudaMemset DFT output staging buffer");
  const unsigned int blocks = launch_cooperative_block_count(
      total_block_count, dft_output_staging_threads_per_block_fp32);
  stage_dft_output_fp32_kernel<<<
      blocks, dft_output_staging_threads_per_block_fp32>>>(
      device_operations, device_block_operation_indices,
      operation_count, total_block_count, output_planar,
      output_point_count, tile_frequency_start, tile_frequency_count,
      frequency_capacity);
  check(cudaGetLastError(), "stage_dft_output_fp32 kernel launch");
}

void prepare_dft_phases_fp32(
    const double *angular_frequencies, complex_value_fp32 *phases,
    std::size_t frequency_count, double time, double scale_real,
    double scale_imag, int threads_per_block) {
  if (!angular_frequencies || !phases)
    throw std::invalid_argument(
        "DFT angular frequencies and phase output must be non-null");
  if (frequency_count == 0) return;
  const unsigned int phase_blocks =
      launch_block_count(frequency_count, threads_per_block);
  prepare_dft_phases_fp32_kernel<<<phase_blocks, threads_per_block>>>(
      angular_frequencies, phases, frequency_count, time, scale_real,
      scale_imag);
  check(cudaGetLastError(), "prepare_dft_phases_fp32 kernel launch");
}

void update_dft_from_omega_fp32(
    float *dft_real_imag, const float *field_real, const float *field_imag,
    const std::ptrdiff_t *field_indices, const float *weights,
    std::size_t point_count, const double *angular_frequencies,
    complex_value_fp32 *phase_scratch, std::size_t frequency_count,
    double time, double scale_real, double scale_imag,
    std::ptrdiff_t average_offset1, std::ptrdiff_t average_offset2,
    int threads_per_block) {
  if (!dft_real_imag || !field_real || !field_indices || !weights ||
      !angular_frequencies || !phase_scratch)
    throw std::invalid_argument(
        "DFT output, field, indices, weights, angular frequencies, and "
        "phase scratch must be non-null");
  const std::size_t work_count =
      dft_work_count(point_count, frequency_count);
  if (work_count == 0) return;
  prepare_dft_phases_fp32(
      angular_frequencies, phase_scratch, frequency_count, time,
      scale_real, scale_imag, threads_per_block);
  launch_update_dft_fp32(
      dft_real_imag, field_real, field_imag, field_indices, weights,
      point_count, phase_scratch, frequency_count, average_offset1,
      average_offset2, threads_per_block,
      "update_dft_from_omega_fp32 kernel launch");
}

void near2far_cartesian_fp32(
    near2far_cartesian_dimension dimension,
    const near2far_operation_fp32 *operations,
    std::size_t operation_count, const cartesian_point_fp32 *targets,
    std::size_t target_count, const float *frequencies,
    std::size_t frequency_count, std::size_t frequency_offset,
    const near2far_periodic_copy_fp32 *periodic_copies,
    std::size_t periodic_copy_count, float eps, float mu,
    std::size_t partial_count, float *partials, double *output,
    double *absolute_l1, int threads_per_block,
    float maximum_kr_input_error, float azimuthal_mode,
    float greencyl_tolerance) {
  if (dimension != near2far_cartesian_dimension::two &&
      dimension != near2far_cartesian_dimension::three &&
      dimension != near2far_cartesian_dimension::cylindrical)
    throw std::invalid_argument(
        "near-to-far dimension must be 2D, 3D, or cylindrical");
  if (!operations || !targets || !frequencies || !periodic_copies ||
      !partials || !output || !absolute_l1)
    throw std::invalid_argument(
        "near-to-far operations, targets, frequencies, periodic copies, "
        "partials, output, and absolute L1 evidence must be non-null");
  if (operation_count == 0 || target_count == 0 || frequency_count == 0 ||
      periodic_copy_count == 0 || partial_count == 0)
    throw std::invalid_argument(
        "near-to-far operation, target, frequency, periodic-copy, and "
        "partial counts must be nonzero");
  if (threads_per_block <= 0 || threads_per_block > 256 ||
      (threads_per_block & (threads_per_block - 1)) != 0)
    throw std::invalid_argument(
        "near-to-far threads_per_block must be a power of two in [1, 256]");
  if (!std::isfinite(maximum_kr_input_error) ||
      maximum_kr_input_error < 0.0f)
    throw std::invalid_argument(
        "near-to-far maximum kr input error must be finite and nonnegative");
  const bool cylindrical =
      dimension == near2far_cartesian_dimension::cylindrical;
  if (!std::isfinite(azimuthal_mode) ||
      (cylindrical
           ? (!std::isfinite(greencyl_tolerance) ||
              greencyl_tolerance <= 0.0f ||
              fabsf(azimuthal_mode) > 16380.0f)
           : (azimuthal_mode != 0.0f || greencyl_tolerance != 0.0f)))
    throw std::invalid_argument(
        "near-to-far cylindrical mode/tolerance is invalid for the selected "
        "dimension");

  const auto checked_product = [](std::size_t left, std::size_t right,
                                  const char *label) {
    if (left && right > std::numeric_limits<std::size_t>::max() / left)
      throw std::overflow_error(std::string("near-to-far ") + label +
                                " overflow");
    return left * right;
  };
  const std::size_t work_count =
      checked_product(target_count, frequency_count, "work count");
  if (frequency_offset >
      std::numeric_limits<std::size_t>::max() - frequency_count)
    throw std::overflow_error(
        "near-to-far source frequency range overflow");
  const std::size_t output_scalar_count = checked_product(
      work_count, static_cast<std::size_t>(12), "output scalar count");
  (void)checked_product(work_count, sizeof(double),
                        "output byte count");
  (void)checked_product(output_scalar_count, sizeof(double),
                        "absolute L1 byte count");
  const std::size_t blocks_per_operation =
      checked_product(work_count, partial_count, "partial-block count");
  const std::size_t logical_block_count = checked_product(
      operation_count, blocks_per_operation, "logical-block count");
  const std::size_t partial_scalar_count = checked_product(
      logical_block_count, static_cast<std::size_t>(13),
      "partial scalar count");
  (void)checked_product(partial_scalar_count, sizeof(float),
                        "partial byte count");
  (void)checked_product(target_count, static_cast<std::size_t>(3),
                        "target coordinate count");

  const unsigned int partial_blocks =
      launch_cooperative_block_count(logical_block_count,
                                     threads_per_block);
  const std::size_t shared_bytes = checked_product(
      checked_product(
          (static_cast<std::size_t>(threads_per_block) + 31) / 32,
                      static_cast<std::size_t>(13),
                      "shared scalar count"),
      sizeof(float), "shared byte count");
  switch (dimension) {
    case near2far_cartesian_dimension::two:
      near2far_3d_partials_fp32_kernel<
          near2far_cartesian_dimension::two><<<
          partial_blocks, threads_per_block, shared_bytes>>>(
          operations, operation_count, targets, target_count, frequencies,
          frequency_count, frequency_offset, periodic_copies,
          periodic_copy_count, eps, mu, partial_count, logical_block_count,
          maximum_kr_input_error, azimuthal_mode, greencyl_tolerance,
          partials);
      break;
    case near2far_cartesian_dimension::three:
      near2far_3d_partials_fp32_kernel<
          near2far_cartesian_dimension::three><<<
          partial_blocks, threads_per_block, shared_bytes>>>(
          operations, operation_count, targets, target_count, frequencies,
          frequency_count, frequency_offset, periodic_copies,
          periodic_copy_count, eps, mu, partial_count, logical_block_count,
          maximum_kr_input_error, azimuthal_mode, greencyl_tolerance,
          partials);
      break;
    case near2far_cartesian_dimension::cylindrical:
      near2far_3d_partials_fp32_kernel<
          near2far_cartesian_dimension::cylindrical><<<
          partial_blocks, threads_per_block, shared_bytes>>>(
          operations, operation_count, targets, target_count, frequencies,
          frequency_count, frequency_offset, periodic_copies,
          periodic_copy_count, eps, mu, partial_count, logical_block_count,
          maximum_kr_input_error, azimuthal_mode, greencyl_tolerance,
          partials);
      break;
  }
  check(cudaGetLastError(), "near2far_cartesian_fp32 partial kernel launch");

  const unsigned int final_blocks =
      launch_block_count(work_count, threads_per_block);
  near2far_3d_finalize_fp64_kernel<<<final_blocks, threads_per_block>>>(
      partials, operation_count, work_count, partial_count, output,
      absolute_l1);
  check(cudaGetLastError(), "near2far_cartesian_fp32 final kernel launch");
}

void near2far_cartesian_mixed_fp32(
    near2far_cartesian_dimension dimension,
    const near2far_operation_mixed_fp32 *operations,
    std::size_t operation_count, const cartesian_point_fp64 *targets,
    std::size_t target_count, const double *frequencies,
    std::size_t frequency_count, std::size_t frequency_offset,
    const near2far_periodic_copy_fp64 *periodic_copies,
    std::size_t periodic_copy_count, double eps, double mu,
    std::size_t partial_count, double *partials, double *output,
    int threads_per_block, double azimuthal_mode,
    double greencyl_tolerance) {
  if (dimension != near2far_cartesian_dimension::two &&
      dimension != near2far_cartesian_dimension::three &&
      dimension != near2far_cartesian_dimension::cylindrical)
    throw std::invalid_argument(
        "mixed near-to-far dimension must be 2D, 3D, or cylindrical");
  if (!operations || !targets || !frequencies || !periodic_copies ||
      !partials || !output)
    throw std::invalid_argument(
        "mixed near-to-far operations, targets, frequencies, periodic "
        "copies, partials, and output must be non-null");
  if (operation_count == 0 || target_count == 0 || frequency_count == 0 ||
      periodic_copy_count == 0 || partial_count == 0)
    throw std::invalid_argument(
        "mixed near-to-far counts must be nonzero");
  if (threads_per_block <= 0 || threads_per_block > 256 ||
      (threads_per_block & (threads_per_block - 1)) != 0)
    throw std::invalid_argument(
        "mixed near-to-far threads_per_block must be a power of two in "
        "[1, 256]");
  const bool cylindrical =
      dimension == near2far_cartesian_dimension::cylindrical;
  if (!std::isfinite(azimuthal_mode) ||
      (cylindrical
           ? (!std::isfinite(greencyl_tolerance) ||
              greencyl_tolerance <= 0.0 ||
              std::abs(azimuthal_mode) > 16380.0)
           : (azimuthal_mode != 0.0 || greencyl_tolerance != 0.0)))
    throw std::invalid_argument(
        "mixed near-to-far cylindrical mode/tolerance is invalid for the "
        "selected dimension");

  const auto checked_product = [](std::size_t left, std::size_t right,
                                  const char *label) {
    if (left && right > std::numeric_limits<std::size_t>::max() / left)
      throw std::overflow_error(std::string("mixed near-to-far ") + label +
                                " overflow");
    return left * right;
  };
  const std::size_t work_count =
      checked_product(target_count, frequency_count, "work count");
  if (frequency_offset >
      std::numeric_limits<std::size_t>::max() - frequency_count)
    throw std::overflow_error(
        "mixed near-to-far source frequency range overflow");
  const std::size_t output_scalar_count = checked_product(
      work_count, static_cast<std::size_t>(12), "output scalar count");
  (void)checked_product(output_scalar_count, sizeof(double),
                        "output byte count");
  const std::size_t blocks_per_operation =
      checked_product(work_count, partial_count, "partial-block count");
  const std::size_t logical_block_count = checked_product(
      operation_count, blocks_per_operation, "logical-block count");
  const std::size_t partial_scalar_count = checked_product(
      logical_block_count, static_cast<std::size_t>(12),
      "partial scalar count");
  (void)checked_product(partial_scalar_count, sizeof(double),
                        "partial byte count");

  const unsigned int partial_blocks =
      launch_cooperative_block_count(logical_block_count,
                                     threads_per_block);
  const std::size_t shared_bytes = checked_product(
      checked_product(static_cast<std::size_t>(threads_per_block),
                      static_cast<std::size_t>(12),
                      "shared scalar count"),
      sizeof(double), "shared byte count");
  switch (dimension) {
    case near2far_cartesian_dimension::two:
      near2far_3d_partials_mixed_fp32_kernel<
          near2far_cartesian_dimension::two><<<
          partial_blocks, threads_per_block, shared_bytes>>>(
          operations, operation_count, targets, target_count, frequencies,
          frequency_count, frequency_offset, periodic_copies,
          periodic_copy_count, eps, mu, partial_count, logical_block_count,
          azimuthal_mode, greencyl_tolerance, partials);
      break;
    case near2far_cartesian_dimension::three:
      near2far_3d_partials_mixed_fp32_kernel<
          near2far_cartesian_dimension::three><<<
          partial_blocks, threads_per_block, shared_bytes>>>(
          operations, operation_count, targets, target_count, frequencies,
          frequency_count, frequency_offset, periodic_copies,
          periodic_copy_count, eps, mu, partial_count, logical_block_count,
          azimuthal_mode, greencyl_tolerance, partials);
      break;
    case near2far_cartesian_dimension::cylindrical:
      near2far_3d_partials_mixed_fp32_kernel<
          near2far_cartesian_dimension::cylindrical><<<
          partial_blocks, threads_per_block, shared_bytes>>>(
          operations, operation_count, targets, target_count, frequencies,
          frequency_count, frequency_offset, periodic_copies,
          periodic_copy_count, eps, mu, partial_count, logical_block_count,
          azimuthal_mode, greencyl_tolerance, partials);
      break;
  }
  check(cudaGetLastError(),
        "near2far_cartesian_mixed_fp32 partial kernel launch");

  const unsigned int final_blocks =
      launch_block_count(work_count, threads_per_block);
  near2far_3d_finalize_mixed_fp64_kernel<<<
      final_blocks, threads_per_block>>>(
      partials, operation_count, work_count, partial_count, output);
  check(cudaGetLastError(),
        "near2far_cartesian_mixed_fp32 final kernel launch");
}

void near2far_adjoint_fp32(
    near2far_cartesian_dimension dimension,
    const near2far_adjoint_source_fp32 *sources,
    std::size_t source_count, const cartesian_point_fp32 *targets,
    std::size_t target_count, const float *frequencies,
    std::size_t frequency_count,
    const near2far_periodic_copy_fp32 *periodic_copies,
    std::size_t periodic_copy_count, const complex_value_fp32 *dJ,
    float eps, float mu, double *output, double *absolute_l1,
    int threads_per_block, float maximum_kr_input_error,
    float azimuthal_mode, float greencyl_tolerance, bool accumulate) {
  if (dimension != near2far_cartesian_dimension::two &&
      dimension != near2far_cartesian_dimension::three &&
      dimension != near2far_cartesian_dimension::cylindrical)
    throw std::invalid_argument(
        "adjoint near-to-far dimension must be 2D, 3D, or cylindrical");
  if (!sources || !targets || !frequencies || !periodic_copies || !dJ ||
      !output || !absolute_l1)
    throw std::invalid_argument(
        "adjoint near-to-far sources, targets, frequencies, periodic "
        "copies, dJ, output, and absolute L1 evidence must be non-null");
  if (source_count == 0 || target_count == 0 || frequency_count == 0 ||
      periodic_copy_count == 0)
    throw std::invalid_argument(
        "adjoint near-to-far source, target, frequency, and periodic-copy "
        "counts must be nonzero");
  if (threads_per_block <= 0 || threads_per_block > 256 ||
      (threads_per_block & (threads_per_block - 1)) != 0)
    throw std::invalid_argument(
        "adjoint near-to-far threads_per_block must be a power of two in "
        "[1, 256]");
  if (!std::isfinite(maximum_kr_input_error) ||
      maximum_kr_input_error < 0.0f)
    throw std::invalid_argument(
        "adjoint near-to-far maximum kr input error must be finite and "
        "nonnegative");
  const bool cylindrical =
      dimension == near2far_cartesian_dimension::cylindrical;
  if (!std::isfinite(azimuthal_mode) ||
      (cylindrical
           ? (!std::isfinite(greencyl_tolerance) ||
              greencyl_tolerance <= 0.0f ||
              fabsf(azimuthal_mode) > 16380.0f)
           : (azimuthal_mode != 0.0f || greencyl_tolerance != 0.0f)))
    throw std::invalid_argument(
        "adjoint near-to-far cylindrical mode/tolerance is invalid for "
        "the selected dimension");
  const auto checked_product = [](std::size_t left, std::size_t right,
                                  const char *label) {
    if (left && right > std::numeric_limits<std::size_t>::max() / left)
      throw std::overflow_error(std::string("adjoint near-to-far ") +
                                label + " overflow");
    return left * right;
  };
  const std::size_t work_count =
      checked_product(source_count, frequency_count, "work count");
  const std::size_t interaction_count = checked_product(
      target_count, periodic_copy_count, "interaction count");
  (void)interaction_count;
  const std::size_t gradient_count = checked_product(
      checked_product(target_count, frequency_count,
                      "target-frequency count"),
      static_cast<std::size_t>(6), "dJ count");
  (void)checked_product(gradient_count, sizeof(complex_value_fp32),
                        "dJ byte count");
  (void)checked_product(
      checked_product(work_count, static_cast<std::size_t>(2),
                      "output scalar count"),
      sizeof(double), "output byte count");
  (void)checked_product(work_count, sizeof(double),
                        "absolute L1 byte count");
  const unsigned int blocks =
      launch_cooperative_block_count(work_count, threads_per_block);
  const std::size_t warp_count =
      (static_cast<std::size_t>(threads_per_block) + 31) / 32;
  const std::size_t shared_bytes = checked_product(
      checked_product(warp_count, static_cast<std::size_t>(3),
                      "shared scalar count"),
      sizeof(float), "shared byte count");
  switch (dimension) {
    case near2far_cartesian_dimension::two:
      near2far_adjoint_fp32_kernel<near2far_cartesian_dimension::two><<<
          blocks, threads_per_block, shared_bytes>>>(
          sources, source_count, targets, target_count, frequencies,
          frequency_count, periodic_copies, periodic_copy_count, dJ, eps,
          mu, work_count, maximum_kr_input_error, azimuthal_mode,
          greencyl_tolerance, accumulate, output, absolute_l1);
      break;
    case near2far_cartesian_dimension::three:
      near2far_adjoint_fp32_kernel<near2far_cartesian_dimension::three><<<
          blocks, threads_per_block, shared_bytes>>>(
          sources, source_count, targets, target_count, frequencies,
          frequency_count, periodic_copies, periodic_copy_count, dJ, eps,
          mu, work_count, maximum_kr_input_error, azimuthal_mode,
          greencyl_tolerance, accumulate, output, absolute_l1);
      break;
    case near2far_cartesian_dimension::cylindrical:
      near2far_adjoint_fp32_kernel<
          near2far_cartesian_dimension::cylindrical><<<
          blocks, threads_per_block, shared_bytes>>>(
          sources, source_count, targets, target_count, frequencies,
          frequency_count, periodic_copies, periodic_copy_count, dJ, eps,
          mu, work_count, maximum_kr_input_error, azimuthal_mode,
          greencyl_tolerance, accumulate, output, absolute_l1);
      break;
  }
  check(cudaGetLastError(), "near2far_adjoint_fp32 kernel launch");
}

void near2far_adjoint_mixed_fp32(
    near2far_cartesian_dimension dimension,
    const near2far_adjoint_source_fp64 *sources,
    std::size_t source_count, const cartesian_point_fp64 *targets,
    std::size_t target_count, const double *frequencies,
    std::size_t frequency_count,
    const near2far_periodic_copy_fp64 *periodic_copies,
    std::size_t periodic_copy_count, const complex_value_fp64 *dJ,
    double eps, double mu, double *output, int threads_per_block,
    double azimuthal_mode, double greencyl_tolerance, bool accumulate) {
  if (dimension != near2far_cartesian_dimension::two &&
      dimension != near2far_cartesian_dimension::three &&
      dimension != near2far_cartesian_dimension::cylindrical)
    throw std::invalid_argument(
        "mixed adjoint near-to-far dimension must be 2D, 3D, or "
        "cylindrical");
  if (!sources || !targets || !frequencies || !periodic_copies || !dJ ||
      !output)
    throw std::invalid_argument(
        "mixed adjoint near-to-far inputs and output must be non-null");
  if (source_count == 0 || target_count == 0 || frequency_count == 0 ||
      periodic_copy_count == 0)
    throw std::invalid_argument(
        "mixed adjoint near-to-far counts must be nonzero");
  if (threads_per_block <= 0 || threads_per_block > 256 ||
      (threads_per_block & (threads_per_block - 1)) != 0)
    throw std::invalid_argument(
        "mixed adjoint near-to-far threads_per_block must be a power of "
        "two in [1, 256]");
  const bool cylindrical =
      dimension == near2far_cartesian_dimension::cylindrical;
  if (!std::isfinite(azimuthal_mode) ||
      (cylindrical
           ? (!std::isfinite(greencyl_tolerance) ||
              greencyl_tolerance <= 0.0 ||
              std::abs(azimuthal_mode) > 16380.0)
           : (azimuthal_mode != 0.0 || greencyl_tolerance != 0.0)))
    throw std::invalid_argument(
        "mixed adjoint near-to-far cylindrical mode/tolerance is invalid "
        "for the selected dimension");
  const auto checked_product = [](std::size_t left, std::size_t right,
                                  const char *label) {
    if (left && right > std::numeric_limits<std::size_t>::max() / left)
      throw std::overflow_error(
          std::string("mixed adjoint near-to-far ") + label +
          " overflow");
    return left * right;
  };
  const std::size_t work_count =
      checked_product(source_count, frequency_count, "work count");
  (void)checked_product(target_count, periodic_copy_count,
                        "interaction count");
  const std::size_t gradient_count = checked_product(
      checked_product(target_count, frequency_count,
                      "target-frequency count"),
      static_cast<std::size_t>(6), "dJ count");
  (void)checked_product(gradient_count, sizeof(complex_value_fp64),
                        "dJ byte count");
  (void)checked_product(
      checked_product(work_count, static_cast<std::size_t>(2),
                      "output scalar count"),
      sizeof(double), "output byte count");
  const unsigned int blocks =
      launch_cooperative_block_count(work_count, threads_per_block);
  const std::size_t shared_bytes = checked_product(
      checked_product(static_cast<std::size_t>(threads_per_block),
                      static_cast<std::size_t>(2),
                      "shared scalar count"),
      sizeof(double), "shared byte count");
  switch (dimension) {
    case near2far_cartesian_dimension::two:
      near2far_adjoint_mixed_fp64_kernel<
          near2far_cartesian_dimension::two><<<
          blocks, threads_per_block, shared_bytes>>>(
          sources, source_count, targets, target_count, frequencies,
          frequency_count, periodic_copies, periodic_copy_count, dJ, eps,
          mu, work_count, azimuthal_mode, greencyl_tolerance, accumulate,
          output);
      break;
    case near2far_cartesian_dimension::three:
      near2far_adjoint_mixed_fp64_kernel<
          near2far_cartesian_dimension::three><<<
          blocks, threads_per_block, shared_bytes>>>(
          sources, source_count, targets, target_count, frequencies,
          frequency_count, periodic_copies, periodic_copy_count, dJ, eps,
          mu, work_count, azimuthal_mode, greencyl_tolerance, accumulate,
          output);
      break;
    case near2far_cartesian_dimension::cylindrical:
      near2far_adjoint_mixed_fp64_kernel<
          near2far_cartesian_dimension::cylindrical><<<
          blocks, threads_per_block, shared_bytes>>>(
          sources, source_count, targets, target_count, frequencies,
          frequency_count, periodic_copies, periodic_copy_count, dJ, eps,
          mu, work_count, azimuthal_mode, greencyl_tolerance, accumulate,
          output);
      break;
  }
  check(cudaGetLastError(),
        "near2far_adjoint_mixed_fp32 kernel launch");
}

void near2far_3d_fp32(
    const near2far_operation_fp32 *operations,
    std::size_t operation_count, const cartesian_point_fp32 *targets,
    std::size_t target_count, const float *frequencies,
    std::size_t frequency_count, std::size_t frequency_offset,
    const near2far_periodic_copy_fp32 *periodic_copies,
    std::size_t periodic_copy_count, float eps, float mu,
    std::size_t partial_count, float *partials, double *output,
    double *absolute_l1, int threads_per_block) {
  near2far_cartesian_fp32(
      near2far_cartesian_dimension::three, operations, operation_count,
      targets, target_count, frequencies, frequency_count, frequency_offset,
      periodic_copies, periodic_copy_count, eps, mu, partial_count, partials,
      output, absolute_l1, threads_per_block);
}

void near2far_3d_mixed_fp32(
    const near2far_operation_mixed_fp32 *operations,
    std::size_t operation_count, const cartesian_point_fp64 *targets,
    std::size_t target_count, const double *frequencies,
    std::size_t frequency_count, std::size_t frequency_offset,
    const near2far_periodic_copy_fp64 *periodic_copies,
    std::size_t periodic_copy_count, double eps, double mu,
    std::size_t partial_count, double *partials, double *output,
    int threads_per_block) {
  near2far_cartesian_mixed_fp32(
      near2far_cartesian_dimension::three, operations, operation_count,
      targets, target_count, frequencies, frequency_count, frequency_offset,
      periodic_copies, periodic_copy_count, eps, mu, partial_count, partials,
      output, threads_per_block);
}

} // namespace meep_cuda
