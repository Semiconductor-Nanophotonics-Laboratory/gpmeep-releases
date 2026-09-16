#ifndef MEEP_CUDA_DETAIL_CYLINDRICAL_HPP
#define MEEP_CUDA_DETAIL_CYLINDRICAL_HPP

#include <cstddef>

#ifdef __CUDACC__
#define MEEP_CUDA_CYL_HOST_DEVICE __host__ __device__
#else
#define MEEP_CUDA_CYL_HOST_DEVICE
#endif

namespace meep_cuda {
namespace detail {

MEEP_CUDA_CYL_HOST_DEVICE inline float cylindrical_radial_divergence_fp32(
    const float *radial_operand, std::ptrdiff_t field_index,
    std::ptrdiff_t radial_stride, float radial_index,
    int radial_difference_sign) {
  const std::ptrdiff_t neighbor =
      field_index + radial_difference_sign * radial_stride;
  return (
      radial_operand[neighbor] *
          (radial_index + static_cast<float>(radial_difference_sign)) -
      radial_operand[field_index] * radial_index) /
      (radial_index +
       0.5f * static_cast<float>(radial_difference_sign));
}

MEEP_CUDA_CYL_HOST_DEVICE inline float cylindrical_imr_drive_fp32(
    const float *operand, std::ptrdiff_t field_index,
    int doubled_radial_coordinate, float coefficient) {
  return coefficient * operand[field_index] /
         static_cast<float>(doubled_radial_coordinate);
}

MEEP_CUDA_CYL_HOST_DEVICE inline float cylindrical_axis_drive_fp32(
    const float *primary, const float *secondary,
    std::ptrdiff_t field_index, std::ptrdiff_t neighbor_shift,
    std::ptrdiff_t secondary_offset, float secondary_scale,
    float drive_scale) {
  float drive = primary[field_index];
  if (secondary)
    drive -= primary[field_index + neighbor_shift] +
             secondary_scale *
                 secondary[field_index + secondary_offset];
  return drive_scale * drive;
}

} // namespace detail
} // namespace meep_cuda

#undef MEEP_CUDA_CYL_HOST_DEVICE

#endif
