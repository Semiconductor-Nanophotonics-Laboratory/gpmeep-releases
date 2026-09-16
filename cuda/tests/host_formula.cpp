#include "meep_cuda/detail/bfast.hpp"
#include "meep_cuda/detail/cylindrical.hpp"
#include "meep_cuda/detail/curl.hpp"
#include "meep_cuda/detail/index_space.hpp"
#include "meep_cuda/detail/launch.hpp"
#include "meep_cuda/detail/polarization.hpp"
#include "meep_cuda/detail/update_eh.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

void require_close(float actual, float expected, const char *message) {
  const float tolerance =
      2.0e-6f * std::max(1.0f, std::max(std::abs(actual), std::abs(expected)));
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL: " << message << " actual=" << actual
              << " expected=" << expected << " tolerance=" << tolerance << '\n';
    std::exit(1);
  }
}

void check_material_specialization(bool pml_f, bool pml_u, bool conductivity) {
  constexpr float initial_field = 0.73f;
  constexpr float initial_field_u = -0.41f;
  constexpr float initial_field_conductivity = 0.29f;
  constexpr float curl = -0.37f;
  constexpr float dtdx = 0.21f;

  const meep_cuda::detail::curl_coefficients_fp32 coefficients = {
      pml_f, pml_u, conductivity,
      0.07f, 1.13f, 0.83f,
      0.05f, 1.09f, 0.87f,
      0.19f, 0.91f, 0.04f};

  float field = initial_field;
  float field_u = initial_field_u;
  float field_conductivity = initial_field_conductivity;
  meep_cuda::detail::apply_curl_update_fp32(
      field, pml_u ? &field_u : nullptr,
      pml_f && conductivity ? &field_conductivity : nullptr, curl, dtdx,
      coefficients);

  float expected_field = initial_field;
  float expected_field_u = initial_field_u;
  float expected_field_conductivity = initial_field_conductivity;
  const float dt2 = 0.5f * coefficients.dt;
  if (!pml_f && !pml_u) {
    expected_field =
        conductivity
            ? ((1.0f - dt2 * coefficients.conductivity_value) * expected_field -
               dtdx * curl) *
                  coefficients.conductivity_inverse
            : expected_field - dtdx * curl;
  }
  else if (!pml_f) {
    const float previous = expected_field_u;
    expected_field_u =
        conductivity
            ? ((1.0f - dt2 * coefficients.conductivity_value) * previous -
               dtdx * curl) *
                  coefficients.conductivity_inverse
            : previous - dtdx * curl;
    expected_field =
        coefficients.sigma_u_inverse *
        ((coefficients.kappa_u - coefficients.sigma_u) * expected_field +
         expected_field_u - previous);
  }
  else if (!pml_u) {
    if (conductivity) {
      const float previous = expected_field_conductivity;
      expected_field_conductivity =
          ((1.0f - dt2 * coefficients.conductivity_value) * previous -
           dtdx * curl) *
          coefficients.conductivity_inverse;
      expected_field =
          ((coefficients.kappa - coefficients.sigma) * expected_field +
           expected_field_conductivity - previous) *
          coefficients.sigma_inverse;
    }
    else {
      expected_field =
          ((coefficients.kappa - coefficients.sigma) * expected_field -
           dtdx * curl) *
          coefficients.sigma_inverse;
    }
  }
  else {
    const float previous_u = expected_field_u;
    if (conductivity) {
      const float previous_conductivity = expected_field_conductivity;
      expected_field_conductivity =
          ((1.0f - dt2 * coefficients.conductivity_value) *
               previous_conductivity -
           dtdx * curl) *
          coefficients.conductivity_inverse;
      expected_field_u =
          ((coefficients.kappa - coefficients.sigma) * previous_u +
           expected_field_conductivity - previous_conductivity) *
          coefficients.sigma_inverse;
    }
    else {
      expected_field_u =
          ((coefficients.kappa - coefficients.sigma) * previous_u -
           dtdx * curl) *
          coefficients.sigma_inverse;
    }
    expected_field =
        coefficients.sigma_u_inverse *
        ((coefficients.kappa_u - coefficients.sigma_u) * expected_field +
         expected_field_u - previous_u);
  }

  require_close(field, expected_field, "material curl field specialization");
  require_close(field_u, expected_field_u, "material curl PML-u specialization");
  require_close(field_conductivity, expected_field_conductivity,
                "material curl conductivity specialization");
}

void check_additive_specialization(bool pml_f, bool pml_u,
                                   bool conductivity) {
  constexpr float initial_field = 0.47f;
  constexpr float initial_field_u = -0.18f;
  constexpr float initial_field_conductivity = 0.23f;
  constexpr float raw_correction = -0.31f;
  constexpr float conductivity_inverse = 0.91f;
  constexpr float sigma_inverse = 0.83f;
  constexpr float sigma_u_inverse = 0.87f;
  float field = initial_field;
  float field_u = initial_field_u;
  float field_conductivity = initial_field_conductivity;
  meep_cuda::detail::apply_additive_correction_fp32(
      field, pml_u ? &field_u : nullptr,
      pml_f && conductivity ? &field_conductivity : nullptr,
      raw_correction, pml_f, pml_u, conductivity,
      conductivity_inverse, sigma_inverse, sigma_u_inverse);

  float expected_correction =
      raw_correction * (conductivity ? conductivity_inverse : 1.0f);
  const float expected_field_conductivity =
      initial_field_conductivity +
      (pml_f && conductivity ? expected_correction : 0.0f);
  if (pml_f) expected_correction *= sigma_inverse;
  const float expected_field_u =
      initial_field_u + (pml_u ? expected_correction : 0.0f);
  const float expected_field =
      initial_field +
      (pml_u ? sigma_u_inverse * expected_correction
             : expected_correction);
  require_close(field, expected_field, "additive correction field");
  require_close(field_u, expected_field_u, "additive correction PML-u");
  require_close(
      field_conductivity, expected_field_conductivity,
      "additive correction conductivity auxiliary");
}

void check_bfast_specialization(bool pml_f, bool pml_u,
                                bool conductivity,
                                bool has_second_operand) {
  constexpr float initial_field = 0.39f;
  constexpr float initial_field_u = -0.17f;
  constexpr float initial_field_conductivity = 0.21f;
  constexpr float initial_bfast_field = 0.14f;
  const float drives[] = {-0.31f, 0.27f};
  const meep_cuda::detail::bfast_coefficients_fp32 coefficients = {
      pml_f, pml_u, conductivity, 0.83f, 0.89f, 0.91f};

  float field = initial_field;
  float field_u = initial_field_u;
  float field_conductivity = initial_field_conductivity;
  float bfast_field = initial_bfast_field;
  float expected_field = initial_field;
  float expected_field_u = initial_field_u;
  float expected_field_conductivity = initial_field_conductivity;
  float expected_bfast_field = initial_bfast_field;

  for (const float drive : drives) {
    const float previous = expected_bfast_field;
    const bool alternating =
        has_second_operand || pml_f || pml_u || conductivity;
    expected_bfast_field = drive - (alternating ? previous : 0.0f);
    float delta = expected_bfast_field - previous;
    if (conductivity) delta *= coefficients.conductivity_inverse;
    if (conductivity && pml_f)
      expected_field_conductivity += delta;
    if (pml_f) delta *= coefficients.sigma_inverse;
    if (pml_u) {
      expected_field_u += delta;
      expected_field += coefficients.sigma_u_inverse * delta;
    }
    else
      expected_field += delta;

    meep_cuda::detail::apply_bfast_update_fp32(
        field, pml_u ? &field_u : nullptr,
        conductivity && pml_f ? &field_conductivity : nullptr,
        bfast_field, drive, coefficients, has_second_operand);
  }

  require_close(field, expected_field, "BFAST field recurrence");
  require_close(field_u, expected_field_u, "BFAST PML-u recurrence");
  require_close(
      field_conductivity, expected_field_conductivity,
      "BFAST conductivity recurrence");
  require_close(
      bfast_field, expected_bfast_field, "BFAST auxiliary recurrence");

  if (!pml_f && !pml_u && !conductivity && !has_second_operand)
    require_close(
        bfast_field, drives[1],
        "bare single-operand BFAST compatibility recurrence");
}

void check_cylindrical_formulas(float radial_origin_offset) {
  constexpr std::ptrdiff_t radial_stride = 1;
  const float radial_operand[] = {0.21f, -0.33f, 0.57f, 0.18f, -0.44f};
  float prefix[5] = {};
  for (int radial_index = 1; radial_index < 5; ++radial_index) {
    const float coordinate = radial_index + radial_origin_offset;
    prefix[radial_index] =
        prefix[radial_index - 1] +
        (radial_operand[radial_index] * coordinate -
         radial_operand[radial_index - 1] * (coordinate - 1.0f)) /
            (coordinate - 0.5f);
  }
  for (int radial_index = 0; radial_index < 4; ++radial_index) {
    const float actual =
        meep_cuda::detail::cylindrical_radial_divergence_fp32(
            radial_operand, radial_index, radial_stride,
            radial_index + radial_origin_offset, +1);
    require_close(
        actual, prefix[radial_index + 1] - prefix[radial_index],
        "positive cylindrical radial divergence");
  }
  for (int radial_index = 1; radial_index < 5; ++radial_index) {
    const float actual =
        meep_cuda::detail::cylindrical_radial_divergence_fp32(
            radial_operand, radial_index, radial_stride,
            radial_index + radial_origin_offset, -1);
    require_close(
        actual, prefix[radial_index - 1] - prefix[radial_index],
        "negative cylindrical radial divergence");
  }

  require_close(
      meep_cuda::detail::cylindrical_imr_drive_fp32(
          radial_operand, 2, 7, -1.4f),
      -1.4f * radial_operand[2] / 7.0f,
      "cylindrical i*m/r drive");
  require_close(
      meep_cuda::detail::cylindrical_axis_drive_fp32(
          radial_operand, radial_operand, 2, -1, 1, 0.6f, -0.4f),
      -0.4f *
          (radial_operand[2] - radial_operand[1] -
           0.6f * radial_operand[3]),
      "cylindrical multi-operand axis drive");
  require_close(
      meep_cuda::detail::cylindrical_axis_drive_fp32(
          radial_operand, nullptr, 2, 0, 0, 0.0f, 1.7f),
      1.7f * radial_operand[2],
      "cylindrical single-operand axis drive");
}

void check_update_eh_specialization(bool pml, int offdiagonal_count,
                                    bool nonlinear,
                                    bool inverse_present) {
  constexpr float initial_field = 0.61f;
  constexpr float initial_field_w = -0.27f;
  constexpr float g = 0.43f;
  constexpr float offdiagonal1 = -0.08f;
  constexpr float offdiagonal2 = 0.05f;
  constexpr float field_norm_squared = 0.71f;

  const meep_cuda::detail::update_eh_coefficients_fp32 coefficients = {
      pml, nonlinear, inverse_present ? 0.82f : 1.0f,
      nonlinear ? 0.04f : 0.0f, nonlinear ? 0.03f : 0.0f, 0.07f,
      1.11f};
  float field = initial_field;
  float field_w = initial_field_w;
  meep_cuda::detail::apply_update_eh_fp32(
      field, pml ? &field_w : nullptr, g,
      offdiagonal_count >= 1 ? offdiagonal1 : 0.0f,
      offdiagonal_count >= 2 ? offdiagonal2 : 0.0f,
      field_norm_squared, coefficients);

  float expected =
      g * coefficients.inverse_susceptibility +
      (offdiagonal_count >= 1 ? offdiagonal1 : 0.0f) +
      (offdiagonal_count >= 2 ? offdiagonal2 : 0.0f);
  if (nonlinear) {
    const float inverse_squared =
        coefficients.inverse_susceptibility *
        coefficients.inverse_susceptibility;
    const float c2 =
        g * coefficients.chi2 * inverse_squared;
    const float c3 =
        field_norm_squared * coefficients.chi3 * inverse_squared *
        coefficients.inverse_susceptibility;
    expected *=
        (1.0f + c2 + 2.0f * c3) / (1.0f + 2.0f * c2 + 3.0f * c3);
  }

  if (pml) {
    const float expected_w = expected;
    expected = initial_field +
               (coefficients.kappa + coefficients.sigma) * expected_w -
               (coefficients.kappa - coefficients.sigma) * initial_field_w;
    require_close(field_w, expected_w, "E/H PML auxiliary specialization");
  }
  require_close(field, expected, "E/H material specialization");
}

void check_lorentzian_specialization(int offdiagonal_count, bool drude,
                                     bool zero_diagonal) {
  constexpr float initial_p = 0.37f;
  constexpr float initial_previous = -0.16f;
  constexpr float field = 0.42f;
  constexpr float offdiagonal1 = 0.07f;
  constexpr float offdiagonal2 = -0.03f;
  const float sigma = zero_diagonal ? 0.0f : 0.61f;
  const meep_cuda::detail::lorentzian_coefficients_fp32 coefficients = {
      0.91f, 0.87f, 0.12f, drude ? 0.0f : 0.12f};

  float polarization = initial_p;
  float previous = initial_previous;
  meep_cuda::detail::apply_lorentzian_update_fp32(
      polarization, previous, field, sigma,
      offdiagonal_count >= 1 ? offdiagonal1 : 0.0f,
      offdiagonal_count >= 2 ? offdiagonal2 : 0.0f,
      offdiagonal_count > 0, coefficients);

  if (zero_diagonal && offdiagonal_count > 0) {
    require_close(polarization, initial_p,
                  "anisotropic zero-sigma polarization guard");
    require_close(previous, initial_previous,
                  "anisotropic zero-sigma previous guard");
    return;
  }

  const float expected =
      coefficients.gamma_inverse *
      (initial_p *
           (2.0f - coefficients.omega_dt_squared_denominator) -
       coefficients.gamma_previous * initial_previous +
       coefficients.omega_dt_squared *
           (sigma * field +
            (offdiagonal_count >= 1 ? offdiagonal1 : 0.0f) +
            (offdiagonal_count >= 2 ? offdiagonal2 : 0.0f)));
  require_close(polarization, expected, "Lorentzian polarization update");
  require_close(previous, initial_p, "Lorentzian previous polarization");
}

void check_lorentzian_increment_specialization(int offdiagonal_count,
                                                bool drude,
                                                bool zero_diagonal) {
  constexpr float initial_p = 0.37f;
  constexpr float initial_increment = -0.16f;
  constexpr float field = 0.42f;
  constexpr float offdiagonal1 = 0.07f;
  constexpr float offdiagonal2 = -0.03f;
  const float sigma = zero_diagonal ? 0.0f : 0.61f;
  const meep_cuda::detail::lorentzian_coefficients_fp32 coefficients = {
      0.91f, 0.87f, 0.12f, drude ? 0.0f : 0.12f};

  float polarization = initial_p;
  float increment = initial_increment;
  meep_cuda::detail::apply_lorentzian_increment_update_fp32(
      polarization, increment, field, sigma,
      offdiagonal_count >= 1 ? offdiagonal1 : 0.0f,
      offdiagonal_count >= 2 ? offdiagonal2 : 0.0f,
      offdiagonal_count > 0, coefficients);

  if (zero_diagonal && offdiagonal_count > 0) {
    require_close(polarization, initial_p,
                  "anisotropic zero-sigma increment-state polarization guard");
    require_close(increment, initial_increment,
                  "anisotropic zero-sigma increment guard");
    return;
  }

  const float expected_increment =
      coefficients.gamma_inverse *
      (coefficients.gamma_previous * initial_increment -
       coefficients.omega_dt_squared_denominator * initial_p +
       coefficients.omega_dt_squared *
           (sigma * field +
            (offdiagonal_count >= 1 ? offdiagonal1 : 0.0f) +
            (offdiagonal_count >= 2 ? offdiagonal2 : 0.0f)));
  require_close(increment, expected_increment,
                "Lorentzian increment-state increment update");
  require_close(polarization, initial_p + expected_increment,
                "Lorentzian increment-state polarization update");
}

void check_lorentzian_increment_long_horizon() {
  // A zero-drive Drude pole with zero increment must preserve its neutral
  // mode bit-for-bit, including the sign bit of the increment.
  const meep_cuda::detail::lorentzian_coefficients_fp32 drude = {
      0.97f, 0.93f, 0.0f, 0.0f};
  float polarization = 0.75f;
  float increment = 0.0f;
  const float initial_polarization = polarization;
  const float initial_increment = increment;
  for (int step = 0; step < 1000000; ++step)
    meep_cuda::detail::apply_lorentzian_increment_update_fp32(
        polarization, increment, 0.0f, 1.0f, 0.0f, 0.0f, false,
        drude);
  require(std::memcmp(&polarization, &initial_polarization, sizeof(float)) ==
              0,
          "Drude neutral polarization is bitwise invariant");
  require(std::memcmp(&increment, &initial_increment, sizeof(float)) == 0,
          "Drude zero increment is bitwise invariant");

  // With damping and no drive, a nonzero Drude increment must decay.
  const meep_cuda::detail::lorentzian_coefficients_fp32 damped_drude = {
      1.0f / 1.04f, 0.96f, 0.0f, 0.0f};
  polarization = -0.25f;
  increment = 0.125f;
  for (int step = 0; step < 512; ++step)
    meep_cuda::detail::apply_lorentzian_increment_update_fp32(
        polarization, increment, 0.0f, 1.0f, 0.0f, 0.0f, false,
        damped_drude);
  require(std::isfinite(polarization) && std::isfinite(increment),
          "damped Drude increment remains finite");
  require(std::abs(increment) < 1.0e-12f,
          "damped Drude increment decays");

  // Compare a long driven Lorentz recurrence against a long-double oracle.
  const meep_cuda::detail::lorentzian_coefficients_fp32 lorentz = {
      1.0f / 1.02f, 0.98f, 0.015625f, 0.015625f};
  constexpr float sigma = 0.71f;
  polarization = 0.13f;
  increment = -0.021f;
  long double reference_polarization = polarization;
  long double reference_increment = increment;
  for (int step = 0; step < 200000; ++step) {
    const float field =
        static_cast<float>(0.23L * std::sin(0.011L * step));
    meep_cuda::detail::apply_lorentzian_increment_update_fp32(
        polarization, increment, field, sigma, 0.0f, 0.0f, false,
        lorentz);
    const long double next_increment =
        static_cast<long double>(lorentz.gamma_inverse) *
        (static_cast<long double>(lorentz.gamma_previous) *
             reference_increment -
         static_cast<long double>(lorentz.omega_dt_squared_denominator) *
             reference_polarization +
         static_cast<long double>(lorentz.omega_dt_squared) *
             static_cast<long double>(sigma) *
             static_cast<long double>(field));
    reference_polarization += next_increment;
    reference_increment = next_increment;
  }
  require(std::isfinite(polarization) && std::isfinite(increment),
          "long Lorentz increment recurrence remains finite");
  const long double scale =
      std::max(1.0L, std::abs(reference_polarization));
  require(std::abs(static_cast<long double>(polarization) -
                   reference_polarization) <=
              5.0e-4L * scale,
          "long Lorentz increment recurrence agrees with long-double oracle");
}

void check_gyrotropic_specialization(int model) {
  const float initial_p[3] = {0.31f, -0.27f, 0.19f};
  const float initial_previous[3] = {-0.11f, 0.23f, -0.17f};
  const float field[3] = {0.43f, -0.29f, 0.37f};
  constexpr float sigma = 0.61f;
  meep_cuda::detail::gyrotropic_coefficients_fp32 coefficients = {
      model,
      {0.91f, 0.02f, -0.03f,
       -0.01f, 0.88f, 0.04f,
       0.05f, -0.02f, 0.93f},
      {0.0f, 0.12f, -0.07f,
       -0.12f, 0.0f, 0.09f,
       0.07f, -0.09f, 0.0f},
      1.83f,
      0.94f,
      0.17f,
      0.026f,
      0.14f,
      0.031f,
      0.008f,
      0.052f};

  float r[3];
  if (model != meep_cuda::detail::gyrotropic_saturated_fp32) {
    r[0] = coefficients.diagonal * initial_p[0] -
           coefficients.gamma_previous * initial_previous[0] +
           coefficients.omega_dt_squared * sigma * field[0] -
           coefficients.precession_half_dt * coefficients.gyro[1] *
               initial_previous[1] -
           coefficients.precession_half_dt * coefficients.gyro[2] *
               initial_previous[2];
    r[1] = coefficients.diagonal * initial_p[1] -
           coefficients.gamma_previous * initial_previous[1] +
           coefficients.omega_dt_squared * sigma * field[1] -
           coefficients.precession_half_dt * coefficients.gyro[3] *
               initial_previous[0] -
           coefficients.precession_half_dt * coefficients.gyro[5] *
               initial_previous[2];
    r[2] = coefficients.diagonal * initial_p[2] -
           coefficients.gamma_previous * initial_previous[2] +
           coefficients.omega_dt_squared * sigma * field[2] -
           coefficients.precession_half_dt * coefficients.gyro[7] *
               initial_previous[1] -
           coefficients.precession_half_dt * coefficients.gyro[6] *
               initial_previous[0];
  }
  else {
    float q[3];
    for (int direction = 0; direction < 3; ++direction)
      q[direction] =
          -coefficients.omega_dt * initial_p[direction] +
          coefficients.alpha_half * initial_previous[direction] +
          coefficients.drive_dt * sigma * field[direction];
    r[0] = 0.5f * initial_previous[0] -
           coefficients.gamma_dt * initial_p[0] +
           coefficients.gyro[1] * q[1] + coefficients.gyro[2] * q[2];
    r[1] = 0.5f * initial_previous[1] -
           coefficients.gamma_dt * initial_p[1] +
           coefficients.gyro[5] * q[2] + coefficients.gyro[3] * q[0];
    r[2] = 0.5f * initial_previous[2] -
           coefficients.gamma_dt * initial_p[2] +
           coefficients.gyro[6] * q[0] + coefficients.gyro[7] * q[1];
  }

  float expected[3];
  for (int row = 0; row < 3; ++row)
    expected[row] =
        coefficients.inverse[3 * row] * r[0] +
        coefficients.inverse[3 * row + 1] * r[1] +
        coefficients.inverse[3 * row + 2] * r[2];

  float polarization[3] = {
      initial_p[0], initial_p[1], initial_p[2]};
  float previous[3] = {
      initial_previous[0], initial_previous[1], initial_previous[2]};
  meep_cuda::detail::apply_gyrotropic_update_fp32(
      polarization[0], polarization[1], polarization[2], previous[0],
      previous[1], previous[2], field[0], field[1], field[2], sigma,
      coefficients);
  for (int direction = 0; direction < 3; ++direction) {
    require_close(
        polarization[direction], expected[direction],
        "gyrotropic polarization update");
    require_close(
        previous[direction], initial_p[direction],
        "gyrotropic previous-polarization update");
  }
}

} // namespace

int main() {
  const std::size_t size_max = std::numeric_limits<std::size_t>::max();
  require(
      meep_cuda::detail::warp_aligned_point_count(31u) == 32u,
      "warp alignment below one warp");
  require(
      meep_cuda::detail::warp_aligned_point_count(32u) == 32u,
      "warp alignment at one warp");
  require(
      meep_cuda::detail::warp_aligned_point_count(33u) == 64u,
      "warp alignment above one warp");
  require(
      meep_cuda::detail::warp_aligned_point_count(size_max) == size_max,
      "warp alignment SIZE_MAX saturation");
  require(
      meep_cuda::detail::warp_aligned_point_count(size_max - 15u) ==
          size_max,
      "warp alignment SIZE_MAX-15 saturation");
  require(
      meep_cuda::detail::warp_aligned_point_count(size_max - 31u) ==
          size_max - 31u,
      "warp alignment SIZE_MAX-31 exact boundary");

  const meep_cuda::index_space_fp32 index_space = {
      7, 2, 3, 4, 100, 10, 1, 5, 0, 2, 0, 9, 2, 0, 0};
  const meep_cuda::detail::decoded_index_fp32 decoded =
      meep_cuda::detail::decode_index_space_fp32(index_space, 18);
  require(decoded.field == 7 + 100 + 10 + 2,
          "structured field index decoding");
  require(decoded.coefficient == 5 + 2,
          "structured first coefficient decoding");
  require(decoded.coefficient2 == 9 + 2,
          "structured second coefficient decoding");

  const float g1[] = {1.0f, 2.0f, 4.0f, 8.0f, 16.0f};
  const float g2[] = {3.0f, 5.0f, 9.0f, 17.0f, 33.0f};

  const float two_operand = meep_cuda::detail::curl_term_fp32(g1, g2, 1, 2, 1);
  require(two_operand == (8.0f - 2.0f + 5.0f - 9.0f), "two-operand curl formula");

  const float negative_stride = meep_cuda::detail::curl_term_fp32(g1, nullptr, 3, -2, 0);
  require(negative_stride == 2.0f - 8.0f, "negative-stride curl formula");

  const auto plus_only =
      meep_cuda::detail::normalize_curl_operands(g1, nullptr, 2, 0, 0.5f);
  require(plus_only.g1 == g1 && plus_only.g2 == nullptr && plus_only.stride1 == 2 &&
              plus_only.dtdx == 0.5f,
          "plus-only normalization");

  const auto minus_only =
      meep_cuda::detail::normalize_curl_operands(nullptr, g2, 0, -1, 0.25f);
  require(minus_only.g1 == g2 && minus_only.g2 == nullptr && minus_only.stride1 == -1 &&
              minus_only.dtdx == -0.25f,
          "minus-only normalization");

  const auto bfast_plus_only =
      meep_cuda::detail::normalize_bfast_operands(
          g1, nullptr, -1, 0, 0.3f, -0.4f);
  require(
      bfast_plus_only.g1 == g1 && bfast_plus_only.g2 == nullptr &&
          bfast_plus_only.stride1 == -1 &&
          bfast_plus_only.k1 == 0.3f,
      "BFAST plus-only negative-stride normalization");
  const auto bfast_minus_only =
      meep_cuda::detail::normalize_bfast_operands(
          nullptr, g2, 0, 2, 0.3f, -0.4f);
  require(
      bfast_minus_only.g1 == g2 && bfast_minus_only.g2 == nullptr &&
          bfast_minus_only.stride1 == 2 &&
          bfast_minus_only.k1 == -0.4f &&
          bfast_minus_only.k2 == 0.3f,
      "BFAST minus-only coefficient normalization");
  require_close(
      meep_cuda::detail::bfast_drive_fp32(
          bfast_plus_only, 3),
      0.3f * (g1[2] + g1[3]),
      "BFAST negative-stride drive");
  const auto bfast_two =
      meep_cuda::detail::normalize_bfast_operands(
          g1, g2, 2, -1, 0.3f, -0.4f);
  require_close(
      meep_cuda::detail::bfast_drive_fp32(bfast_two, 2),
      0.3f * (g1[4] + g1[2]) -
          (-0.4f) * (g2[1] + g2[2]),
      "BFAST two-operand mixed-stride drive");

  for (int pml_f = 0; pml_f < 2; ++pml_f)
    for (int pml_u = 0; pml_u < 2; ++pml_u)
      for (int conductivity = 0; conductivity < 2; ++conductivity)
        check_material_specialization(pml_f != 0, pml_u != 0,
                                      conductivity != 0);
  for (int pml_f = 0; pml_f < 2; ++pml_f)
    for (int pml_u = 0; pml_u < 2; ++pml_u)
      for (int conductivity = 0; conductivity < 2; ++conductivity)
        check_additive_specialization(
            pml_f != 0, pml_u != 0, conductivity != 0);
  for (int pml_f = 0; pml_f < 2; ++pml_f)
    for (int pml_u = 0; pml_u < 2; ++pml_u)
      for (int conductivity = 0; conductivity < 2; ++conductivity)
        for (int second_operand = 0; second_operand < 2;
             ++second_operand)
          check_bfast_specialization(
              pml_f != 0, pml_u != 0, conductivity != 0,
              second_operand != 0);
  check_cylindrical_formulas(0.0f);
  check_cylindrical_formulas(1.35f);

  const float offdiagonal_coefficient[] = {0.2f, 0.3f, 0.5f, 0.7f};
  const float offdiagonal_field[] = {1.0f, 2.0f, 4.0f, 8.0f};
  require_close(meep_cuda::detail::offdiagonal_average_fp32(
                    offdiagonal_coefficient, offdiagonal_field, 1, 1, 1),
                0.25f * ((2.0f + 1.0f) * 0.3f +
                         (4.0f + 2.0f) * 0.5f),
                "off-diagonal stable average");
  const auto swapped_update =
      meep_cuda::detail::normalize_update_eh_operands(
          nullptr, g2, nullptr, offdiagonal_coefficient, 0, -1);
  require(swapped_update.g1 == g2 && swapped_update.g2 == nullptr &&
              swapped_update.u1 == offdiagonal_coefficient &&
              swapped_update.u2 == nullptr &&
              swapped_update.stride1 == -1,
          "E/H off-diagonal normalization");

  const auto swapped_lorentzian =
      meep_cuda::detail::normalize_lorentzian_operands(
          nullptr, g2, nullptr, offdiagonal_coefficient, 0, -1);
  require(swapped_lorentzian.w1 == g2 &&
              swapped_lorentzian.w2 == nullptr &&
              swapped_lorentzian.sigma1 == offdiagonal_coefficient &&
              swapped_lorentzian.sigma2 == nullptr &&
              swapped_lorentzian.stride1 == -1,
          "Lorentzian off-diagonal normalization");

  for (int pml = 0; pml < 2; ++pml)
    for (int offdiagonal_count = 0; offdiagonal_count < 3;
         ++offdiagonal_count)
      for (int nonlinear = 0; nonlinear < 2; ++nonlinear)
        for (int inverse_present = 0; inverse_present < 2;
             ++inverse_present)
          check_update_eh_specialization(
              pml != 0, offdiagonal_count, nonlinear != 0,
              inverse_present != 0);

  for (int offdiagonal_count = 0; offdiagonal_count < 3;
       ++offdiagonal_count)
    for (int drude = 0; drude < 2; ++drude)
      for (int zero_diagonal = 0; zero_diagonal < 2; ++zero_diagonal)
        check_lorentzian_specialization(
            offdiagonal_count, drude != 0, zero_diagonal != 0);
  for (int offdiagonal_count = 0; offdiagonal_count < 3;
       ++offdiagonal_count)
    for (int drude = 0; drude < 2; ++drude)
      for (int zero_diagonal = 0; zero_diagonal < 2; ++zero_diagonal)
        check_lorentzian_increment_specialization(
            offdiagonal_count, drude != 0, zero_diagonal != 0);
  check_lorentzian_increment_long_horizon();
  for (int model = meep_cuda::detail::gyrotropic_lorentzian_fp32;
       model <= meep_cuda::detail::gyrotropic_saturated_fp32; ++model)
    check_gyrotropic_specialization(model);

  std::cout << "PASS: host structured indexing, Cartesian/cylindrical curl, "
               "additive/BFAST correction, E/H, Lorentzian, conditioned "
               "Lorentzian, and gyrotropic "
               "formulas, normalization, and material specializations\n";
  return 0;
}
