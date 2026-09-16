#ifndef MEEP_CUDA_DETAIL_CURL_HPP
#define MEEP_CUDA_DETAIL_CURL_HPP

#include <cstddef>

#ifdef __CUDACC__
#define MEEP_CUDA_HOST_DEVICE __host__ __device__
#else
#define MEEP_CUDA_HOST_DEVICE
#endif

namespace meep_cuda {
namespace detail {

struct curl_operands_fp32 {
  const float *g1;
  const float *g2;
  std::ptrdiff_t stride1;
  std::ptrdiff_t stride2;
  float dtdx;
};

MEEP_CUDA_HOST_DEVICE inline curl_operands_fp32
normalize_curl_operands(const float *g1, const float *g2, std::ptrdiff_t stride1,
                        std::ptrdiff_t stride2, float dtdx) {
  if (!g1 && g2) return {g2, nullptr, stride2, 0, -dtdx};
  return {g1, g2, stride1, stride2, dtdx};
}

MEEP_CUDA_HOST_DEVICE inline float curl_term_fp32(const float *g1, const float *g2,
                                                  std::ptrdiff_t i,
                                                  std::ptrdiff_t stride1,
                                                  std::ptrdiff_t stride2) {
  float curl = g1[i + stride1] - g1[i];
  if (g2) curl += g2[i] - g2[i + stride2];
  return curl;
}

struct curl_coefficients_fp32 {
  bool pml_f;
  bool pml_u;
  bool conductivity;
  float sigma;
  float kappa;
  float sigma_inverse;
  float sigma_u;
  float kappa_u;
  float sigma_u_inverse;
  float conductivity_value;
  float conductivity_inverse;
  float dt;
};

// Implements every PML/conductivity specialization in Meep's step_curl.
// field_u is required when pml_u is true. field_conductivity is required only
// for the simultaneous pml_f + conductivity case.
MEEP_CUDA_HOST_DEVICE inline void
apply_curl_update_fp32(float &field, float *field_u, float *field_conductivity,
                       float curl, float dtdx,
                       const curl_coefficients_fp32 &coefficients) {
  if (!coefficients.pml_f) {
    if (!coefficients.pml_u) {
      if (coefficients.conductivity) {
        const float dt2 = coefficients.dt * 0.5f;
        field = ((1.0f - dt2 * coefficients.conductivity_value) * field -
                 dtdx * curl) *
                coefficients.conductivity_inverse;
      }
      else {
        field -= dtdx * curl;
      }
      return;
    }

    const float field_u_previous = *field_u;
    if (coefficients.conductivity) {
      const float dt2 = coefficients.dt * 0.5f;
      *field_u =
          ((1.0f - dt2 * coefficients.conductivity_value) * field_u_previous -
           dtdx * curl) *
          coefficients.conductivity_inverse;
    }
    else {
      *field_u -= dtdx * curl;
    }
    field = coefficients.sigma_u_inverse *
            ((coefficients.kappa_u - coefficients.sigma_u) * field +
             *field_u - field_u_previous);
    return;
  }

  if (!coefficients.pml_u) {
    if (coefficients.conductivity) {
      const float field_conductivity_previous = *field_conductivity;
      const float dt2 = coefficients.dt * 0.5f;
      *field_conductivity =
          ((1.0f - dt2 * coefficients.conductivity_value) *
               field_conductivity_previous -
           dtdx * curl) *
          coefficients.conductivity_inverse;
      field = ((coefficients.kappa - coefficients.sigma) * field +
               *field_conductivity - field_conductivity_previous) *
              coefficients.sigma_inverse;
    }
    else {
      field = ((coefficients.kappa - coefficients.sigma) * field -
               dtdx * curl) *
              coefficients.sigma_inverse;
    }
    return;
  }

  const float field_u_previous = *field_u;
  if (coefficients.conductivity) {
    const float field_conductivity_previous = *field_conductivity;
    const float dt2 = coefficients.dt * 0.5f;
    *field_conductivity =
        ((1.0f - dt2 * coefficients.conductivity_value) *
             field_conductivity_previous -
         dtdx * curl) *
        coefficients.conductivity_inverse;
    *field_u = ((coefficients.kappa - coefficients.sigma) * field_u_previous +
                *field_conductivity - field_conductivity_previous) *
               coefficients.sigma_inverse;
  }
  else {
    *field_u = ((coefficients.kappa - coefficients.sigma) * field_u_previous -
                dtdx * curl) *
               coefficients.sigma_inverse;
  }
  field = coefficients.sigma_u_inverse *
          ((coefficients.kappa_u - coefficients.sigma_u) * field +
           *field_u - field_u_previous);
}

// Applies an already time-integrated additive curl correction such as the
// 2D beta term or cylindrical i*m/r term through the same inverse PML and
// conductivity factors as Meep's CPU specializations.
MEEP_CUDA_HOST_DEVICE inline void apply_additive_correction_fp32(
    float &field, float *field_u, float *field_conductivity, float correction,
    bool pml_f, bool pml_u, bool conductivity,
    float conductivity_inverse, float sigma_inverse,
    float sigma_u_inverse) {
  if (conductivity) correction *= conductivity_inverse;
  if (conductivity && pml_f) *field_conductivity += correction;
  if (pml_f) correction *= sigma_inverse;
  if (pml_u) {
    *field_u += correction;
    field += sigma_u_inverse * correction;
  }
  else
    field += correction;
}

} // namespace detail
} // namespace meep_cuda

#undef MEEP_CUDA_HOST_DEVICE

#endif
