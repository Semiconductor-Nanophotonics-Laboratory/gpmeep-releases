#ifndef MEEP_CUDA_DETAIL_UPDATE_EH_HPP
#define MEEP_CUDA_DETAIL_UPDATE_EH_HPP

#include <cstddef>

#ifdef __CUDACC__
#define MEEP_CUDA_UPDATE_HOST_DEVICE __host__ __device__
#else
#define MEEP_CUDA_UPDATE_HOST_DEVICE
#endif

namespace meep_cuda {
namespace detail {

struct update_eh_operands_fp32 {
  const float *g1;
  const float *g2;
  const float *u1;
  const float *u2;
  std::ptrdiff_t stride1;
  std::ptrdiff_t stride2;
};

MEEP_CUDA_UPDATE_HOST_DEVICE inline update_eh_operands_fp32
normalize_update_eh_operands(const float *g1, const float *g2,
                             const float *u1, const float *u2,
                             std::ptrdiff_t stride1,
                             std::ptrdiff_t stride2) {
  if ((!g1 && g2) || (g1 && g2 && !u1 && u2))
    return {g2, g1, u2, u1, stride2, stride1};
  return {g1, g2, u1, u2, stride1, stride2};
}

MEEP_CUDA_UPDATE_HOST_DEVICE inline float
nonlinear_update_factor_fp32(float field_norm_squared, float field_component,
                             float inverse_susceptibility, float chi2,
                             float chi3) {
  const float inverse_squared =
      inverse_susceptibility * inverse_susceptibility;
  const float c2 =
      field_component * chi2 * inverse_squared;
  const float c3 = field_norm_squared * chi3 * inverse_squared *
                   inverse_susceptibility;
  return (1.0f + c2 + 2.0f * c3) /
         (1.0f + 2.0f * c2 + 3.0f * c3);
}

MEEP_CUDA_UPDATE_HOST_DEVICE inline float
offdiagonal_average_fp32(const float *coefficient, const float *field,
                         std::ptrdiff_t index, std::ptrdiff_t field_stride,
                         std::ptrdiff_t neighbor_stride) {
  return 0.25f *
         ((field[index] + field[index - neighbor_stride]) *
              coefficient[index] +
          (field[index + field_stride] +
           field[index + field_stride - neighbor_stride]) *
              coefficient[index + field_stride]);
}

struct update_eh_coefficients_fp32 {
  bool pml;
  bool nonlinear;
  float inverse_susceptibility;
  float chi2;
  float chi3;
  float sigma;
  float kappa;
};

// Applies one point of Meep's step_update_EDHB after the caller has evaluated
// the optional off-diagonal averages and neighboring-field norm terms.
MEEP_CUDA_UPDATE_HOST_DEVICE inline void apply_update_eh_fp32(
    float &field, float *field_w, float g, float offdiagonal1,
    float offdiagonal2, float field_norm_squared,
    const update_eh_coefficients_fp32 &coefficients) {
  float updated =
      g * coefficients.inverse_susceptibility + offdiagonal1 + offdiagonal2;
  if (coefficients.nonlinear)
    updated *= nonlinear_update_factor_fp32(
        field_norm_squared, g, coefficients.inverse_susceptibility,
        coefficients.chi2, coefficients.chi3);

  if (!coefficients.pml) {
    field = updated;
    return;
  }

  const float previous_w = *field_w;
  *field_w = updated;
  field += (coefficients.kappa + coefficients.sigma) * updated -
           (coefficients.kappa - coefficients.sigma) * previous_w;
}

} // namespace detail
} // namespace meep_cuda

#undef MEEP_CUDA_UPDATE_HOST_DEVICE

#endif
