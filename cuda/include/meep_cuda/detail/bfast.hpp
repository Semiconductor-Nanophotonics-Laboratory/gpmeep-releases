#ifndef MEEP_CUDA_DETAIL_BFAST_HPP
#define MEEP_CUDA_DETAIL_BFAST_HPP

#include <cstddef>

#ifdef __CUDACC__
#define MEEP_CUDA_BFAST_HOST_DEVICE __host__ __device__
#else
#define MEEP_CUDA_BFAST_HOST_DEVICE
#endif

namespace meep_cuda {
namespace detail {

struct bfast_operands_fp32 {
  const float *g1;
  const float *g2;
  std::ptrdiff_t stride1;
  std::ptrdiff_t stride2;
  float k1;
  float k2;
};

// Matches step_bfast's historical operand normalization, including swapping
// the wave-vector coefficients because they participate in a cross product.
MEEP_CUDA_BFAST_HOST_DEVICE inline bfast_operands_fp32
normalize_bfast_operands(
    const float *g1, const float *g2, std::ptrdiff_t stride1,
    std::ptrdiff_t stride2, float k1, float k2) {
  if (!g1 && g2) return {g2, nullptr, stride2, 0, k2, k1};
  return {g1, g2, stride1, stride2, k1, k2};
}

MEEP_CUDA_BFAST_HOST_DEVICE inline float bfast_drive_fp32(
    const bfast_operands_fp32 &operands, std::ptrdiff_t field_index) {
  float drive =
      operands.k1 *
      (operands.g1[field_index + operands.stride1] +
       operands.g1[field_index]);
  if (operands.g2)
    drive -=
        operands.k2 *
        (operands.g2[field_index + operands.stride2] +
         operands.g2[field_index]);
  return drive;
}

struct bfast_coefficients_fp32 {
  bool pml_f;
  bool pml_u;
  bool conductivity;
  float sigma_inverse;
  float sigma_u_inverse;
  float conductivity_inverse;
};

// Applies one step_bfast recurrence after the ordinary curl update. The
// seemingly asymmetric bare single-operand case is intentional: the CPU
// implementation stores F=drive rather than F=drive-F_previous only when
// PML and conductivity are both absent. Preserve that compatibility wart
// exactly because it changes every recurrence after the first time step.
MEEP_CUDA_BFAST_HOST_DEVICE inline void apply_bfast_update_fp32(
    float &field, float *field_u, float *field_conductivity,
    float &bfast_field, float drive,
    const bfast_coefficients_fp32 &coefficients,
    bool has_second_operand) {
  const float previous = bfast_field;
  const bool alternating_recurrence =
      has_second_operand || coefficients.pml_f || coefficients.pml_u ||
      coefficients.conductivity;
  bfast_field = drive - (alternating_recurrence ? previous : 0.0f);

  float delta = bfast_field - previous;
  if (coefficients.conductivity)
    delta *= coefficients.conductivity_inverse;
  if (coefficients.conductivity && coefficients.pml_f)
    *field_conductivity += delta;
  if (coefficients.pml_f) delta *= coefficients.sigma_inverse;
  if (coefficients.pml_u) {
    *field_u += delta;
    field += coefficients.sigma_u_inverse * delta;
  }
  else
    field += delta;
}

} // namespace detail
} // namespace meep_cuda

#undef MEEP_CUDA_BFAST_HOST_DEVICE

#endif
