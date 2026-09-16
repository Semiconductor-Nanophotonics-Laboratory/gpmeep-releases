#include "meep_cuda/runtime.hpp"
#include "meep_cuda/detail/polarization.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

template <typename Callable>
void expect_invalid(const char *name, const char *message_fragment,
                    Callable callable) {
  try {
    callable();
  } catch (const std::invalid_argument &error) {
    if (std::string(error.what()).find(message_fragment) != std::string::npos)
      return;
    throw std::runtime_error(std::string(name) + ": unexpected diagnostic: " +
                             error.what());
  }
  throw std::runtime_error(std::string(name) +
                           ": expected std::invalid_argument");
}

template <typename Callable>
void expect_overflow(const char *name, const char *message_fragment,
                     Callable callable) {
  try {
    callable();
  } catch (const std::overflow_error &error) {
    if (std::string(error.what()).find(message_fragment) != std::string::npos)
      return;
    throw std::runtime_error(std::string(name) + ": unexpected diagnostic: " +
                             error.what());
  }
  throw std::runtime_error(std::string(name) +
                           ": expected std::overflow_error");
}

} // namespace

int main() {
  float field = 0.0f;
  float operand = 0.0f;
  float sigma = 0.0f;
  float kappa = 1.0f;
  float inverse = 1.0f;
  float auxiliary = 0.0f;
  meep_cuda::curl_index index = {0, 0, 0};

  expect_invalid("null CUDA event record", "event must be non-null", [&] {
    meep_cuda::record_event(nullptr);
  });
  expect_invalid("null CUDA event synchronize", "event must be non-null",
                 [&] { meep_cuda::synchronize_event(nullptr); });
  meep_cuda::destroy_event(nullptr);

  const auto invoke = [&](const meep_cuda::curl_material_fp32 &material,
                          const meep_cuda::curl_index *indices) {
    meep_cuda::step_curl_material_fp32(
        &field, &operand, nullptr, indices, 0, 0, 0, 0.25f, material);
  };

  const meep_cuda::curl_material_fp32 empty = {};
  expect_invalid("null field", "field must be non-null", [&] {
    meep_cuda::step_curl_material_fp32(
        nullptr, &operand, nullptr, nullptr, 0, 0, 0, 0.25f, empty);
  });
  expect_invalid("null curl operands", "curl operand", [&] {
    meep_cuda::step_curl_material_fp32(
        &field, nullptr, nullptr, nullptr, 0, 0, 0, 0.25f, empty);
  });

  meep_cuda::curl_material_fp32 material = {};
  material.sigma = &sigma;
  expect_invalid("partial PML-f group", "PML-f",
                 [&] { invoke(material, &index); });

  material = {};
  material.sigma_u = &sigma;
  expect_invalid("partial PML-u group", "PML-u",
                 [&] { invoke(material, &index); });

  material.kappa_u = &kappa;
  material.sigma_u_inverse = &inverse;
  expect_invalid("missing PML-u auxiliary", "PML-u",
                 [&] { invoke(material, &index); });

  material = {};
  material.conductivity = &sigma;
  expect_invalid("partial conductivity group", "conductivity-inverse",
                 [&] { invoke(material, &index); });

  material = {};
  material.sigma = &sigma;
  material.kappa = &kappa;
  material.sigma_inverse = &inverse;
  material.conductivity = &sigma;
  material.conductivity_inverse = &inverse;
  expect_invalid("missing conductivity auxiliary", "auxiliary field",
                 [&] { invoke(material, &index); });

  material = {};
  material.field_conductivity = &auxiliary;
  expect_invalid("orphan conductivity auxiliary", "valid only",
                 [&] { invoke(material, &index); });

  material = {};
  material.sigma = &sigma;
  material.kappa = &kappa;
  material.sigma_inverse = &inverse;
  expect_invalid("PML-f without indices", "explicit sigma indices",
                 [&] { invoke(material, nullptr); });

  material = {};
  material.field_u = &auxiliary;
  material.sigma_u = &sigma;
  material.kappa_u = &kappa;
  material.sigma_u_inverse = &inverse;
  expect_invalid("PML-u without indices", "explicit sigma indices",
                 [&] { invoke(material, nullptr); });

  // Zero work must not query a CUDA device after all invariants are satisfied.
  invoke(empty, nullptr);
  material = {};
  material.sigma = &sigma;
  material.kappa = &kappa;
  material.sigma_inverse = &inverse;
  material.field_u = &auxiliary;
  material.sigma_u = &sigma;
  material.kappa_u = &kappa;
  material.sigma_u_inverse = &inverse;
  material.conductivity = &sigma;
  material.conductivity_inverse = &inverse;
  material.field_conductivity = &auxiliary;
  invoke(material, &index);

  meep_cuda::index_space_fp32 structured_space = {};
  expect_invalid("empty structured curl space", "extents", [&] {
    meep_cuda::step_curl_material_structured_fp32(
        &field, &operand, nullptr, structured_space, 1, 0, 0, 0.25f,
        empty);
  });
  structured_space.extent1 = 1;
  structured_space.extent2 = 1;
  structured_space.extent3 = 1;
  structured_space.coefficient_start = -1;
  structured_space.coefficient2_start = -1;
  expect_invalid("structured curl count mismatch", "do not match", [&] {
    meep_cuda::step_curl_material_structured_fp32(
        &field, &operand, nullptr, structured_space, 2, 0, 0, 0.25f,
        empty);
  });
  meep_cuda::step_curl_material_structured_fp32(
      &field, &operand, nullptr, structured_space, 0, 0, 0, 0.25f,
      empty);
  meep_cuda::curl_phase_operation_fp32 curl_phase_operation = {};
  expect_invalid("null phase-batched curl operations", "operations", [&] {
    meep_cuda::step_curl_material_phase_batched_fp32(
        nullptr, nullptr, 0, 0);
  });
  // A non-null, zero-work phase descriptor must not query a CUDA device.
  meep_cuda::step_curl_material_phase_batched_fp32(
      &curl_phase_operation, nullptr, 0, 0);
  expect_invalid("ownerless phase-batched curl blocks", "requires operations", [&] {
    meep_cuda::step_curl_material_phase_batched_fp32(
        &curl_phase_operation, nullptr, 0, 1);
  });
  expect_invalid("null phase-batched curl block map", "indices", [&] {
    meep_cuda::step_curl_material_phase_batched_fp32(
        &curl_phase_operation, nullptr, 1, 1);
  });

  const meep_cuda::beta_material_fp32 empty_beta = {};
  expect_invalid("null beta field", "field and operand", [&] {
    meep_cuda::step_beta_structured_fp32(
        nullptr, &operand, structured_space, 0, 0.1f, empty_beta);
  });
  expect_invalid("null beta operand", "field and operand", [&] {
    meep_cuda::step_beta_structured_fp32(
        &field, nullptr, structured_space, 0, 0.1f, empty_beta);
  });
  meep_cuda::beta_material_fp32 beta_material = {};
  beta_material.field_u = &auxiliary;
  expect_invalid("partial beta PML-u group", "PML-u", [&] {
    meep_cuda::step_beta_structured_fp32(
        &field, &operand, structured_space, 0, 0.1f, beta_material);
  });
  beta_material = {};
  beta_material.sigma_inverse = &inverse;
  expect_invalid("beta PML-f without coefficient index", "coefficient", [&] {
    meep_cuda::step_beta_structured_fp32(
        &field, &operand, structured_space, 0, 0.1f, beta_material);
  });
  beta_material = {};
  beta_material.field_conductivity = &auxiliary;
  expect_invalid("orphan beta conductivity auxiliary", "requires", [&] {
    meep_cuda::step_beta_structured_fp32(
        &field, &operand, structured_space, 0, 0.1f, beta_material);
  });
  meep_cuda::step_beta_structured_fp32(
      &field, &operand, structured_space, 0, 0.1f, empty_beta);
  structured_space.coefficient_start = 0;
  structured_space.coefficient2_start = 0;
  beta_material = {};
  beta_material.sigma_inverse = &inverse;
  beta_material.field_u = &auxiliary;
  beta_material.sigma_u_inverse = &inverse;
  beta_material.conductivity_inverse = &inverse;
  beta_material.field_conductivity = &auxiliary;
  meep_cuda::step_beta_structured_fp32(
      &field, &operand, structured_space, 0, 0.1f, beta_material);
  structured_space.coefficient_start = -1;
  structured_space.coefficient2_start = -1;

  const meep_cuda::bfast_material_fp32 empty_bfast = {};
  expect_invalid("null BFAST field", "field and auxiliary", [&] {
    meep_cuda::step_bfast_structured_fp32(
        nullptr, &operand, nullptr, &auxiliary, structured_space, 0,
        1, 0, 0.2f, 0.0f, empty_bfast);
  });
  expect_invalid("null BFAST auxiliary", "field and auxiliary", [&] {
    meep_cuda::step_bfast_structured_fp32(
        &field, &operand, nullptr, nullptr, structured_space, 0,
        1, 0, 0.2f, 0.0f, empty_bfast);
  });
  expect_invalid("null BFAST operands", "curl operand", [&] {
    meep_cuda::step_bfast_structured_fp32(
        &field, nullptr, nullptr, &auxiliary, structured_space, 0,
        1, 0, 0.2f, 0.0f, empty_bfast);
  });
  meep_cuda::bfast_material_fp32 bfast_material = {};
  bfast_material.field_u = &auxiliary;
  expect_invalid("partial BFAST PML-u group", "PML-u", [&] {
    meep_cuda::step_bfast_structured_fp32(
        &field, &operand, nullptr, &auxiliary, structured_space, 0,
        1, 0, 0.2f, 0.0f, bfast_material);
  });
  bfast_material = {};
  bfast_material.sigma_inverse = &inverse;
  expect_invalid(
      "BFAST PML-f without coefficient index", "coefficient", [&] {
        meep_cuda::step_bfast_structured_fp32(
            &field, &operand, nullptr, &auxiliary, structured_space, 0,
            1, 0, 0.2f, 0.0f, bfast_material);
      });
  bfast_material = {};
  bfast_material.field_conductivity = &auxiliary;
  expect_invalid(
      "orphan BFAST conductivity auxiliary", "requires", [&] {
        meep_cuda::step_bfast_structured_fp32(
            &field, &operand, nullptr, &auxiliary, structured_space, 0,
            1, 0, 0.2f, 0.0f, bfast_material);
      });
  meep_cuda::step_bfast_structured_fp32(
      &field, nullptr, &operand, &auxiliary, structured_space, 0,
      0, -1, 0.0f, -0.2f, empty_bfast);
  expect_invalid("null batched BFAST spaces", "spaces", [&] {
    meep_cuda::step_bfast_batched_structured_fp32(
        &field, &operand, nullptr, &auxiliary, nullptr, 0, 0,
        1, 0, 0.2f, 0.0f, empty_bfast);
  });
  expect_invalid("null batched BFAST operands", "curl operand", [&] {
    meep_cuda::step_bfast_batched_structured_fp32(
        &field, nullptr, nullptr, &auxiliary, &structured_space, 0, 0,
        1, 0, 0.2f, 0.0f, empty_bfast);
  });
  meep_cuda::step_bfast_batched_structured_fp32(
      &field, &operand, nullptr, &auxiliary, &structured_space, 0, 0,
      1, 0, 0.2f, 0.0f, empty_bfast);
  structured_space.coefficient_start = 0;
  structured_space.coefficient2_start = 0;
  bfast_material = {};
  bfast_material.sigma_inverse = &inverse;
  bfast_material.field_u = &auxiliary;
  bfast_material.sigma_u_inverse = &inverse;
  bfast_material.conductivity_inverse = &inverse;
  bfast_material.field_conductivity = &auxiliary;
  meep_cuda::step_bfast_structured_fp32(
      &field, &operand, nullptr, &auxiliary, structured_space, 0,
      1, 0, 0.2f, 0.0f, bfast_material);
  structured_space.coefficient_start = -1;
  structured_space.coefficient2_start = -1;

  meep_cuda::update_eh_index update_index = {0, 0};
  const meep_cuda::update_eh_material_fp32 empty_update = {};
  const auto invoke_update =
      [&](const meep_cuda::update_eh_material_fp32 &update_material,
          const meep_cuda::update_eh_index *indices) {
        meep_cuda::update_eh_fp32(
            &field, &operand, nullptr, nullptr, indices, 0, 0, 0, 0,
            update_material);
      };
  expect_invalid("null E/H field", "field must be non-null", [&] {
    meep_cuda::update_eh_fp32(
        nullptr, &operand, nullptr, nullptr, nullptr, 0, 0, 0, 0,
        empty_update);
  });
  expect_invalid("null E/H input", "input field must be non-null", [&] {
    meep_cuda::update_eh_fp32(
        &field, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0,
        empty_update);
  });

  meep_cuda::update_eh_material_fp32 update_material = {};
  update_material.offdiagonal1 = &inverse;
  expect_invalid("off-diagonal without input", "requires its input", [&] {
    invoke_update(update_material, &update_index);
  });

  update_material = {};
  update_material.chi3 = &sigma;
  expect_invalid("chi3 without chi2", "chi3 requires", [&] {
    invoke_update(update_material, &update_index);
  });

  update_material = {};
  update_material.field_w = &auxiliary;
  expect_invalid("partial E/H PML group", "PML field", [&] {
    invoke_update(update_material, &update_index);
  });

  update_material.sigma = &sigma;
  update_material.kappa = &kappa;
  expect_invalid("E/H PML without indices", "explicit sigma indices", [&] {
    invoke_update(update_material, nullptr);
  });

  // Valid zero-work E/H updates must also avoid querying a CUDA device.
  invoke_update(empty_update, nullptr);
  invoke_update(update_material, &update_index);
  meep_cuda::update_eh_phase_operation_fp32 update_phase_operation = {};
  expect_invalid("null phase-batched E/H operations", "operations", [&] {
    meep_cuda::update_eh_phase_batched_fp32(nullptr, nullptr, 0, 0);
  });
  // A non-null, zero-work phase descriptor must not query a CUDA device.
  meep_cuda::update_eh_phase_batched_fp32(
      &update_phase_operation, nullptr, 0, 0);
  expect_invalid("ownerless phase-batched E/H blocks", "requires operations", [&] {
    meep_cuda::update_eh_phase_batched_fp32(
        &update_phase_operation, nullptr, 0, 1);
  });
  expect_invalid("null phase-batched E/H block map", "indices", [&] {
    meep_cuda::update_eh_phase_batched_fp32(
        &update_phase_operation, nullptr, 1, 1);
  });

  const meep_cuda::lorentzian_material_fp32 empty_lorentzian = {
      &sigma, nullptr, nullptr, 1.0f, 1.0f, 0.1f, 0.1f};
  const auto invoke_lorentzian =
      [&](const meep_cuda::lorentzian_material_fp32 &lorentzian) {
        meep_cuda::update_lorentzian_fp32(
            &field, &auxiliary, &operand, nullptr, nullptr, nullptr, 0, 0,
            0, 0, lorentzian);
      };
  expect_invalid("null Lorentzian polarization", "polarization", [&] {
    meep_cuda::update_lorentzian_fp32(
        nullptr, &auxiliary, &operand, nullptr, nullptr, nullptr, 0, 0, 0,
        0, empty_lorentzian);
  });
  meep_cuda::lorentzian_material_fp32 lorentzian = empty_lorentzian;
  lorentzian.sigma = nullptr;
  expect_invalid("null Lorentzian sigma", "diagonal sigma", [&] {
    invoke_lorentzian(lorentzian);
  });
  lorentzian = empty_lorentzian;
  lorentzian.offdiagonal1 = &inverse;
  expect_invalid("Lorentzian off-diagonal without field", "requires its field",
                 [&] { invoke_lorentzian(lorentzian); });

  // A valid zero-work Lorentzian update must not query a CUDA device.
  invoke_lorentzian(empty_lorentzian);

  const auto invoke_lorentzian_increment =
      [&](const meep_cuda::lorentzian_material_fp32 &increment_material) {
        meep_cuda::update_lorentzian_increment_fp32(
            &field, &auxiliary, &operand, nullptr, nullptr, nullptr, 0, 0,
            0, 0, increment_material);
      };
  expect_invalid("null Lorentzian increment state", "increment", [&] {
    meep_cuda::update_lorentzian_increment_fp32(
        &field, nullptr, &operand, nullptr, nullptr, nullptr, 0, 0, 0, 0,
        empty_lorentzian);
  });
  lorentzian = empty_lorentzian;
  lorentzian.sigma = nullptr;
  expect_invalid("null increment-state Lorentzian sigma", "diagonal sigma",
                 [&] { invoke_lorentzian_increment(lorentzian); });
  // Both increment-state entry points must accept valid zero-work calls
  // without querying a CUDA device.
  invoke_lorentzian_increment(empty_lorentzian);
  meep_cuda::update_lorentzian_increment_structured_fp32(
      &field, &auxiliary, &operand, nullptr, nullptr, structured_space, 0, 0,
      0, 0, empty_lorentzian);

  meep_cuda::gyrotropic_material_fp32 gyrotropic = {};
  gyrotropic.sigma = &sigma;
  gyrotropic.model =
      meep_cuda::detail::gyrotropic_lorentzian_fp32;
  gyrotropic.inverse[0] = 1.0f;
  gyrotropic.inverse[4] = 1.0f;
  gyrotropic.inverse[8] = 1.0f;
  const auto invoke_gyrotropic =
      [&](const meep_cuda::gyrotropic_material_fp32 &gyro,
          const meep_cuda::index_space_fp32 &space,
          std::size_t count) {
        meep_cuda::update_gyrotropic_structured_fp32(
            &field, &field, &field, &auxiliary, &auxiliary, &auxiliary,
            &operand, nullptr, nullptr, space, count, 0, 0, 0, gyro);
      };
  expect_invalid("null gyrotropic polarization", "polarization", [&] {
    meep_cuda::update_gyrotropic_structured_fp32(
        nullptr, &field, &field, &auxiliary, &auxiliary, &auxiliary,
        &operand, nullptr, nullptr, structured_space, 0, 0, 0, 0,
        gyrotropic);
  });
  meep_cuda::gyrotropic_material_fp32 invalid_gyrotropic = gyrotropic;
  invalid_gyrotropic.sigma = nullptr;
  expect_invalid("null gyrotropic sigma", "sigma", [&] {
    invoke_gyrotropic(invalid_gyrotropic, structured_space, 0);
  });
  invalid_gyrotropic = gyrotropic;
  invalid_gyrotropic.model = 3;
  expect_invalid("invalid gyrotropic model", "invalid gyrotropic", [&] {
    invoke_gyrotropic(invalid_gyrotropic, structured_space, 0);
  });
  meep_cuda::index_space_fp32 empty_gyrotropic_space = {};
  expect_invalid("empty gyrotropic index space", "extents", [&] {
    invoke_gyrotropic(gyrotropic, empty_gyrotropic_space, 1);
  });
  // A valid zero-work gyrotropic update must not query a CUDA device.
  invoke_gyrotropic(gyrotropic, structured_space, 0);

  meep_cuda::multilevel_transition_fp32 multilevel_transition = {
      1, 0, 0.01f, 1.9f, 0.95f, 0.98f,
      {0.01f, 0.01f, 0.01f, 0.01f, 0.01f}};
  meep_cuda::multilevel_population_channel_fp32
      multilevel_population_channel = {
          &field, &operand, &auxiliary, 0, 0};
  meep_cuda::multilevel_polarization_channel_fp32
      multilevel_polarization_channel = {
          &auxiliary, &field, &sigma, structured_space, 1, 0, 0, 0};
  const auto invoke_multilevel =
      [&](float *population, std::size_t levels,
          std::size_t transitions, std::size_t array_count,
          std::size_t polarization_channels,
          const meep_cuda::index_space_fp32 &space) {
        meep_cuda::update_multilevel_structured_fp32(
            population, &auxiliary, &sigma, &inverse, &operand, levels,
            transitions, array_count, 0.01f, space, 1,
            &multilevel_population_channel, 1,
            &multilevel_polarization_channel, polarization_channels,
            &multilevel_transition, 1);
      };
  expect_invalid("null multilevel population", "multilevel population", [&] {
    invoke_multilevel(nullptr, 2, 1, 1, 1, structured_space);
  });
  expect_invalid("zero multilevel levels", "counts must be nonzero", [&] {
    invoke_multilevel(&field, 0, 1, 1, 1, structured_space);
  });
  expect_invalid(
      "null multilevel population channels", "population channels", [&] {
        meep_cuda::update_multilevel_structured_fp32(
            &field, &auxiliary, &sigma, &inverse, &operand, 2, 1, 1,
            0.01f, structured_space, 1, nullptr, 1,
            &multilevel_polarization_channel, 1, &multilevel_transition, 1);
      });
  expect_overflow("multilevel population overflow", "population span", [&] {
    invoke_multilevel(
        &field, 2, 1, std::numeric_limits<std::size_t>::max(), 1,
        structured_space);
  });
  expect_overflow(
      "multilevel polarization overflow", "polarization span", [&] {
        invoke_multilevel(
            &field, 1,
            std::numeric_limits<std::size_t>::max(), 1, 1,
            structured_space);
      });
  expect_overflow("multilevel operation overflow", "operation count", [&] {
    invoke_multilevel(
        &field, 1, 2, 1,
        std::numeric_limits<std::size_t>::max(), structured_space);
  });
  meep_cuda::index_space_fp32 empty_multilevel_space = {};
  expect_invalid("empty multilevel index space", "extents", [&] {
    invoke_multilevel(&field, 2, 1, 1, 1, empty_multilevel_space);
  });

  expect_invalid("null copy destination", "copy destination", [&] {
    meep_cuda::copy_fp32(nullptr, &field, 0);
  });
  expect_invalid("null subtract source", "subtract destination", [&] {
    meep_cuda::subtract_fp32(&field, nullptr, 0);
  });
  meep_cuda::indexed_value_fp32 indexed_update = {0, 0.0f};
  expect_invalid("null indexed updates", "indexed subtract", [&] {
    meep_cuda::indexed_subtract_fp32(&field, nullptr, 0);
  });
  meep_cuda::copy_fp32(&field, &operand, 0);
  meep_cuda::subtract_fp32(&field, &operand, 0);
  meep_cuda::indexed_subtract_fp32(&field, &indexed_update, 0);
  std::ptrdiff_t source_index = 0;
  meep_cuda::complex_value_fp32 source_amplitude = {1.0f, 0.0f};
  meep_cuda::complex_value_fp32 source_scale = {1.0f, 0.0f};
  expect_invalid("null source indices", "indexed source", [&] {
    meep_cuda::indexed_source_subtract_fp32(
        &field, nullptr, &source_amplitude, nullptr, 0, source_scale,
        false);
  });
  expect_invalid("null source amplitudes", "indexed source", [&] {
    meep_cuda::indexed_source_subtract_fp32(
        &field, &source_index, nullptr, nullptr, 0, source_scale, false);
  });
  meep_cuda::indexed_source_subtract_fp32(
      &field, &source_index, &source_amplitude, nullptr, 0, source_scale,
      false);
  meep_cuda::indexed_source_phase_operation_fp32 source_operation = {
      &field, &source_index, &source_amplitude, nullptr, 0, 0, false};
  meep_cuda::indexed_source_phase_time_scales_fp32 source_time_scales = {};
  source_time_scales.values[0] = source_scale;
  expect_invalid("null phase-batched source operations",
                 "phase-batched source operations", [&] {
                   meep_cuda::indexed_source_subtract_phase_batched_fp32(
                       nullptr, nullptr, 0, 0, source_time_scales);
                 });
  meep_cuda::indexed_source_subtract_phase_batched_fp32(
      &source_operation, nullptr, 0, 0, source_time_scales);
  expect_invalid("ownerless phase-batched source blocks", "requires operations", [&] {
    meep_cuda::indexed_source_subtract_phase_batched_fp32(
        &source_operation, nullptr, 0, 1, source_time_scales);
  });
  expect_invalid("null phase-batched source block map", "indices", [&] {
    meep_cuda::indexed_source_subtract_phase_batched_fp32(
        &source_operation, nullptr, 1, 1, source_time_scales);
  });
  expect_invalid("oversized phase-batched source operation count",
                 "parameter capacity", [&] {
                   meep_cuda::indexed_source_subtract_phase_batched_fp32(
                       &source_operation, nullptr,
                       meep_cuda::indexed_source_phase_max_operations + 1,
                       1, source_time_scales);
                 });

  double ldos_result = 0.0;
  double ldos_partial = 0.0;
  const std::uint32_t ldos_block_operation = 0;
  meep_cuda::indexed_ldos_operation_fp32 ldos_operation = {
      &field, nullptr, &source_index, &source_amplitude, 0, 0, false};
  expect_invalid("null indexed LDOS initialization result",
                 "result pointer", [&] {
                   meep_cuda::initialize_indexed_ldos_result(nullptr);
                 });
  expect_invalid("null indexed LDOS operations", "operations", [&] {
                   meep_cuda::indexed_ldos_reduce_fp32(
                       nullptr, &ldos_block_operation, 1, 1,
                       &ldos_partial, 1, &ldos_result,
                       meep_cuda::indexed_ldos_result_mode::replace);
                 });
  expect_invalid("null indexed LDOS block map", "block map", [&] {
    meep_cuda::indexed_ldos_reduce_fp32(
        &ldos_operation, nullptr, 1, 1, &ldos_partial, 1,
        &ldos_result, meep_cuda::indexed_ldos_result_mode::replace);
  });
  expect_invalid("null indexed LDOS workspace", "workspace", [&] {
    meep_cuda::indexed_ldos_reduce_fp32(
        &ldos_operation, &ldos_block_operation, 1, 1, nullptr, 1,
        &ldos_result, meep_cuda::indexed_ldos_result_mode::replace);
  });
  expect_invalid("null indexed LDOS result", "result", [&] {
    meep_cuda::indexed_ldos_reduce_fp32(
        &ldos_operation, &ldos_block_operation, 0, 0, &ldos_partial, 1,
        nullptr, meep_cuda::indexed_ldos_result_mode::replace);
  });
  // Empty work is a host-only no-op.  The descriptor deliberately contains
  // host pointers so this also detects an accidental zero-work launch.
  meep_cuda::indexed_ldos_reduce_fp32(
      nullptr, nullptr, 0, 0, nullptr, 0, &ldos_result,
      meep_cuda::indexed_ldos_result_mode::replace);
  expect_invalid("ownerless indexed LDOS blocks", "requires operations", [&] {
    meep_cuda::indexed_ldos_reduce_fp32(
        &ldos_operation, &ldos_block_operation, 0, 1, &ldos_partial, 1,
        &ldos_result, meep_cuda::indexed_ldos_result_mode::replace);
  });
  expect_invalid(
      "blockless indexed LDOS operations", "requires logical blocks", [&] {
        meep_cuda::indexed_ldos_reduce_fp32(
            &ldos_operation, nullptr, 1, 0, nullptr, 0, &ldos_result,
            meep_cuda::indexed_ldos_result_mode::replace);
      });
  expect_invalid("indexed LDOS operations exceed blocks", "exceeds logical", [&] {
    meep_cuda::indexed_ldos_reduce_fp32(
        &ldos_operation, &ldos_block_operation, 2, 1, &ldos_partial, 1,
        &ldos_result, meep_cuda::indexed_ldos_result_mode::replace);
  });
  expect_invalid("invalid indexed LDOS result mode", "mode", [&] {
    meep_cuda::indexed_ldos_reduce_fp32(
        &ldos_operation, &ldos_block_operation, 1, 1, &ldos_partial, 1,
        &ldos_result,
        static_cast<meep_cuda::indexed_ldos_result_mode>(99));
  });
  expect_invalid("empty indexed LDOS workspace", "too small", [&] {
    meep_cuda::indexed_ldos_reduce_fp32(
        &ldos_operation, &ldos_block_operation, 1, 1, &ldos_partial, 0,
        &ldos_result, meep_cuda::indexed_ldos_result_mode::replace);
  });
  expect_invalid("undersized indexed LDOS workspace", "too small", [&] {
    meep_cuda::indexed_ldos_reduce_fp32(
        &ldos_operation, &ldos_block_operation, 1, 2, &ldos_partial, 1,
        &ldos_result, meep_cuda::indexed_ldos_result_mode::replace);
  });
  const std::size_t overflowing_ldos_partial_capacity =
      std::numeric_limits<std::size_t>::max() /
          (4 * sizeof(double)) +
      1;
  expect_overflow("indexed LDOS workspace byte overflow", "byte count", [&] {
    meep_cuda::indexed_ldos_reduce_fp32(
        &ldos_operation, &ldos_block_operation, 1, 1, &ldos_partial,
        overflowing_ldos_partial_capacity, &ldos_result,
        meep_cuda::indexed_ldos_result_mode::replace);
  });
  const std::uintptr_t aligned_address_limit =
      std::numeric_limits<std::uintptr_t>::max() &
      ~static_cast<std::uintptr_t>(alignof(double) - 1);
  double *const overflowing_ldos_address =
      reinterpret_cast<double *>(aligned_address_limit);
  expect_overflow(
      "indexed LDOS workspace address overflow", "workspace address", [&] {
        meep_cuda::indexed_ldos_reduce_fp32(
            &ldos_operation, &ldos_block_operation, 1, 1,
            overflowing_ldos_address, 1, &ldos_result,
            meep_cuda::indexed_ldos_result_mode::replace);
      });
  expect_overflow("indexed LDOS result address overflow", "result address", [&] {
    meep_cuda::indexed_ldos_reduce_fp32(
        &ldos_operation, &ldos_block_operation, 1, 1, &ldos_partial, 1,
        overflowing_ldos_address,
        meep_cuda::indexed_ldos_result_mode::replace);
  });
  double overlapping_ldos_storage[8] = {};
  expect_invalid(
      "identical indexed LDOS workspace and result", "overlap", [&] {
        meep_cuda::indexed_ldos_reduce_fp32(
            &ldos_operation, &ldos_block_operation, 1, 1,
            overlapping_ldos_storage, 1, overlapping_ldos_storage,
            meep_cuda::indexed_ldos_result_mode::replace);
      });
  expect_invalid(
      "overlapping indexed LDOS workspace and result", "overlap", [&] {
        meep_cuda::indexed_ldos_reduce_fp32(
            &ldos_operation, &ldos_block_operation, 1, 1,
            overlapping_ldos_storage, 1, overlapping_ldos_storage + 3,
            meep_cuda::indexed_ldos_result_mode::replace);
      });

  meep_cuda::boundary_operation_fp32 boundary = {
      &field, nullptr, &operand, nullptr, 1.0f, 0.0f};
  float boundary_staging[2] = {0.0f, 0.0f};
  expect_invalid("null boundary staging", "staging", [&] {
    meep_cuda::apply_boundary_fp32(&boundary, nullptr, 0);
  });
  meep_cuda::apply_boundary_fp32(&boundary, boundary_staging, 0);
  expect_invalid("null zero-boundary operations", "operations", [&] {
    meep_cuda::zero_boundary_fp32(nullptr, 0);
  });
  meep_cuda::zero_boundary_fp32(&boundary, 0);
  expect_invalid("null nonalias-boundary operations", "operations", [&] {
    meep_cuda::apply_nonalias_boundary_fp32(nullptr, 0);
  });
  meep_cuda::apply_nonalias_boundary_fp32(&boundary, 0);
  expect_invalid("null boundary graph operations", "operations", [&] {
    (void)meep_cuda::create_boundary_graph_fp32(nullptr, boundary_staging, 1);
  });
  expect_invalid("null boundary graph staging", "staging", [&] {
    (void)meep_cuda::create_boundary_graph_fp32(&boundary, nullptr, 1);
  });
  expect_invalid("zero boundary graph operations", "nonzero", [&] {
    (void)meep_cuda::create_boundary_graph_fp32(&boundary, boundary_staging, 0);
  });
  expect_invalid("null boundary graph launch", "non-null", [&] {
    meep_cuda::launch_boundary_graph_fp32(nullptr);
  });
  meep_cuda::destroy_boundary_graph_fp32(nullptr);
  expect_invalid("null boundary phase graph stages", "stages", [&] {
    (void)meep_cuda::create_boundary_phase_graph_fp32(nullptr, 1);
  });
  meep_cuda::boundary_phase_stage_fp32 phase_stage = {
      meep_cuda::boundary_phase_stage_kind::zero, &boundary, nullptr, 1,
      nullptr};
  expect_invalid("zero boundary phase graph stages", "nonzero", [&] {
    (void)meep_cuda::create_boundary_phase_graph_fp32(&phase_stage, 0);
  });
  phase_stage.operations = nullptr;
  expect_invalid("null boundary phase graph operations", "operations", [&] {
    (void)meep_cuda::create_boundary_phase_graph_fp32(&phase_stage, 1);
  });
  phase_stage.operations = &boundary;
  phase_stage.count = 0;
  expect_invalid("zero boundary phase graph operations", "operation count", [&] {
    (void)meep_cuda::create_boundary_phase_graph_fp32(&phase_stage, 1);
  });
  phase_stage.count = 1;
  phase_stage.kind = meep_cuda::boundary_phase_stage_kind::ordered;
  expect_invalid("null ordered boundary phase staging", "staging", [&] {
    (void)meep_cuda::create_boundary_phase_graph_fp32(&phase_stage, 1);
  });
  phase_stage.kind =
      static_cast<meep_cuda::boundary_phase_stage_kind>(99);
  expect_invalid("invalid boundary phase kind", "kind", [&] {
    (void)meep_cuda::create_boundary_phase_graph_fp32(&phase_stage, 1);
  });
  expect_invalid("null boundary phase graph launch", "non-null", [&] {
    meep_cuda::launch_boundary_phase_graph_fp32(nullptr);
  });
  meep_cuda::destroy_boundary_phase_graph_fp32(nullptr);
  meep_cuda::array_span_fp32 span = {&field, 1};
  int finite_result = 1;
  expect_invalid("null finite initialization result", "result", [&] {
    meep_cuda::initialize_finite_result(nullptr);
  });
  expect_invalid("null finite spans", "finite-check", [&] {
    meep_cuda::all_finite_fp32(nullptr, 0, 0, &finite_result);
  });
  expect_invalid("null finite result", "result", [&] {
    meep_cuda::all_finite_fp32(&span, 0, 0, nullptr);
  });
  meep_cuda::all_finite_fp32(&span, 0, 0, &finite_result);
  std::uint32_t finite_generation = 0;
  expect_invalid("null finite-generation clear", "result", [&] {
    meep_cuda::clear_finite_generation_result(nullptr);
  });
  expect_invalid("null finite-generation spans", "spans", [&] {
    meep_cuda::all_finite_generation_fp32(
        nullptr, 0, 0, 1, &finite_generation);
  });
  expect_invalid("null finite-generation result", "result", [&] {
    meep_cuda::all_finite_generation_fp32(&span, 0, 0, 1, nullptr);
  });
  meep_cuda::all_finite_generation_fp32(
      &span, 0, 0, 1, &finite_generation);

  double squared_norm_result = 0.0;
  expect_invalid("null squared-norm initialization result", "result", [&] {
    meep_cuda::initialize_squared_norm_result(nullptr);
  });
  expect_invalid("null squared-norm values", "values", [&] {
    meep_cuda::squared_norm_complex_fp32(
        nullptr, nullptr, 0, 0, &squared_norm_result);
  });
  expect_invalid("null squared-norm result", "result", [&] {
    meep_cuda::squared_norm_complex_fp32(
        &field, nullptr, 0, 0, nullptr);
  });
  meep_cuda::squared_norm_complex_fp32(
      &field, nullptr, 0, 0, &squared_norm_result);
  expect_overflow("squared-norm point-frequency overflow", "work count", [&] {
    meep_cuda::squared_norm_complex_fp32(
        &field, nullptr, std::numeric_limits<std::size_t>::max(), 2,
        &squared_norm_result);
  });

  std::ptrdiff_t dft_index = 0;
  float dft_weight = 1.0f;
  meep_cuda::complex_value_fp32 dft_phase = {1.0f, 0.0f};
  expect_invalid("null DFT output", "DFT output", [&] {
    meep_cuda::update_dft_fp32(
        nullptr, &field, nullptr, &dft_index, &dft_weight, 0, &dft_phase,
        0, 0, 0);
  });
  meep_cuda::update_dft_fp32(
      &field, &operand, nullptr, &dft_index, &dft_weight, 0, &dft_phase, 0,
      0, 0);
  meep_cuda::update_dft_fp32(
      &field, &operand, nullptr, &dft_index, &dft_weight, 0, &dft_phase,
      std::numeric_limits<std::size_t>::max(), 0, 0);
  meep_cuda::update_dft_fp32(
      &field, &operand, nullptr, &dft_index, &dft_weight,
      std::numeric_limits<std::size_t>::max(), &dft_phase, 0, 0, 0);
  expect_overflow("DFT point-frequency overflow", "work count", [&] {
    meep_cuda::update_dft_fp32(
        &field, &operand, nullptr, &dft_index, &dft_weight,
        std::numeric_limits<std::size_t>::max(), &dft_phase, 2, 0, 0);
  });
  expect_overflow("DFT interleaved output overflow", "output count", [&] {
    meep_cuda::update_dft_fp32(
        &field, &operand, nullptr, &dft_index, &dft_weight,
        std::numeric_limits<std::size_t>::max() / 2 + 1, &dft_phase, 1, 0,
        0);
  });
  expect_overflow("DFT output byte overflow", "byte count", [&] {
    meep_cuda::update_dft_fp32(
        &field, &operand, nullptr, &dft_index, &dft_weight,
        std::numeric_limits<std::size_t>::max() /
                (2 * sizeof(float)) +
            1,
        &dft_phase, 1, 0, 0);
  });
  meep_cuda::dft_update_operation_fp32 dft_batch_operation = {};
  std::uint32_t dft_batch_block_operation = 0;
  meep_cuda::validate_dft_batch_operations_fp32(nullptr, 0, 0);
  expect_invalid("null host DFT batch operations", "host operations", [&] {
    meep_cuda::validate_dft_batch_operations_fp32(nullptr, 1, 1);
  });
  expect_invalid(
      "ownerless host DFT batch blocks", "requires host operations", [&] {
        meep_cuda::validate_dft_batch_operations_fp32(nullptr, 0, 1);
      });
  meep_cuda::dft_update_operation_fp32 valid_dft_batch_operation = {
      &field, &operand, nullptr, &dft_index, &dft_weight, 1, &dft_phase, 1,
      0, 0, 0, 1, 1, 1, 1};
  meep_cuda::validate_dft_batch_operations_fp32(
      &valid_dft_batch_operation, 1, 1);
  {
    auto invalid = valid_dft_batch_operation;
    invalid.phases = nullptr;
    expect_invalid("null host DFT batch phase", "pointers", [&] {
      meep_cuda::validate_dft_batch_operations_fp32(&invalid, 1, 1);
    });
  }
  {
    auto invalid = valid_dft_batch_operation;
    invalid.point_count = 0;
    expect_invalid("zero host DFT batch count", "counts", [&] {
      meep_cuda::validate_dft_batch_operations_fp32(&invalid, 1, 1);
    });
  }
  {
    auto invalid = valid_dft_batch_operation;
    invalid.frequency_threads = 0;
    expect_invalid("zero host DFT batch threads", "thread layout", [&] {
      meep_cuda::validate_dft_batch_operations_fp32(&invalid, 1, 1);
    });
    invalid.frequency_threads = 1;
    invalid.point_threads =
        meep_cuda::dft_batch_threads_per_block_fp32 + 1;
    expect_invalid("oversize host DFT batch tile", "thread layout", [&] {
      meep_cuda::validate_dft_batch_operations_fp32(&invalid, 1, 1);
    });
  }
  {
    auto invalid = valid_dft_batch_operation;
    invalid.frequency_block_count = 2;
    expect_invalid("wrong host DFT batch tile count", "tile counts", [&] {
      meep_cuda::validate_dft_batch_operations_fp32(&invalid, 1, 1);
    });
  }
  {
    auto invalid = valid_dft_batch_operation;
    invalid.block_start = 1;
    expect_invalid("noncontiguous host DFT batch prefix", "prefixes", [&] {
      meep_cuda::validate_dft_batch_operations_fp32(&invalid, 1, 1);
    });
  }
  expect_invalid(
      "mismatched host DFT batch total", "do not match", [&] {
        meep_cuda::validate_dft_batch_operations_fp32(
            &valid_dft_batch_operation, 1, 2);
      });
  {
    auto invalid = valid_dft_batch_operation;
    invalid.point_count = std::numeric_limits<std::size_t>::max();
    invalid.frequency_count = 2;
    invalid.frequency_block_count = 2;
    invalid.point_block_count =
        1 + (invalid.point_count - 1) / invalid.point_threads;
    expect_overflow(
        "host DFT batch point-frequency overflow", "work count", [&] {
          meep_cuda::validate_dft_batch_operations_fp32(&invalid, 1, 1);
        });
  }
  expect_invalid("null DFT batch operations", "operations", [&] {
    meep_cuda::update_dft_batch_fp32(nullptr, nullptr, 0, 0);
  });
  meep_cuda::update_dft_batch_fp32(
      &dft_batch_operation, nullptr, 0, 0);
  expect_invalid(
      "ownerless DFT batch blocks", "requires operations", [&] {
        meep_cuda::update_dft_batch_fp32(
            &dft_batch_operation, &dft_batch_block_operation, 0, 1);
      });
  meep_cuda::update_dft_batch_fp32(
      &dft_batch_operation, nullptr, 1, 0);
  expect_invalid("null DFT batch block map", "indices", [&] {
    meep_cuda::update_dft_batch_fp32(
        &dft_batch_operation, nullptr, 1, 1);
  });
  if (std::numeric_limits<std::size_t>::max() >
      static_cast<std::size_t>(
          std::numeric_limits<std::uint32_t>::max()))
    expect_overflow("DFT batch operation index overflow", "index", [&] {
      meep_cuda::update_dft_batch_fp32(
          &dft_batch_operation, &dft_batch_block_operation,
          static_cast<std::size_t>(
              std::numeric_limits<std::uint32_t>::max()) +
              1u,
          1);
    });
  double dft_omega = 0.25;
  expect_invalid("null DFT phase frequencies", "frequencies", [&] {
    meep_cuda::prepare_dft_phases_fp32(
        nullptr, &dft_phase, 0, 0.5, 1.0, 0.0);
  });
  expect_invalid("null DFT phase output", "phase output", [&] {
    meep_cuda::prepare_dft_phases_fp32(
        &dft_omega, nullptr, 0, 0.5, 1.0, 0.0);
  });
  meep_cuda::prepare_dft_phases_fp32(
      &dft_omega, &dft_phase, 0, 0.5, 1.0, 0.0);
  expect_invalid("null DFT phase scratch", "phase scratch", [&] {
    meep_cuda::update_dft_from_omega_fp32(
        &field, &operand, nullptr, &dft_index, &dft_weight, 0, &dft_omega,
        nullptr, 0, 0.5, 1.0, 0.0, 0, 0);
  });
  meep_cuda::update_dft_from_omega_fp32(
      &field, &operand, nullptr, &dft_index, &dft_weight, 0, &dft_omega,
      &dft_phase, 0, 0.5, 1.0, 0.0, 0, 0);
  meep_cuda::update_dft_from_omega_fp32(
      &field, &operand, nullptr, &dft_index, &dft_weight, 0, &dft_omega,
      &dft_phase, std::numeric_limits<std::size_t>::max(), 0.5, 1.0, 0.0,
      0, 0);
  meep_cuda::update_dft_from_omega_fp32(
      &field, &operand, nullptr, &dft_index, &dft_weight,
      std::numeric_limits<std::size_t>::max(), &dft_omega, &dft_phase, 0,
      0.5, 1.0, 0.0, 0, 0);
  expect_overflow("omega DFT point-frequency overflow", "work count", [&] {
    meep_cuda::update_dft_from_omega_fp32(
        &field, &operand, nullptr, &dft_index, &dft_weight,
        std::numeric_limits<std::size_t>::max(), &dft_omega, &dft_phase, 2,
        0.5, 1.0, 0.0, 0, 0);
  });
  expect_overflow("omega DFT interleaved output overflow", "output count",
                  [&] {
    meep_cuda::update_dft_from_omega_fp32(
        &field, &operand, nullptr, &dft_index, &dft_weight,
        std::numeric_limits<std::size_t>::max() / 2 + 1, &dft_omega,
        &dft_phase, 1, 0.5, 1.0, 0.0, 0, 0);
  });
  expect_overflow("omega DFT output byte overflow", "byte count", [&] {
    meep_cuda::update_dft_from_omega_fp32(
        &field, &operand, nullptr, &dft_index, &dft_weight,
        std::numeric_limits<std::size_t>::max() /
                (2 * sizeof(float)) +
            1,
        &dft_omega, &dft_phase, 1, 0.5, 1.0, 0.0, 0, 0);
  });

  std::cout << "PASS: CUDA curl, E/H, Lorentzian, gyrotropic, multilevel, "
               "source, LDOS, DFT, and resident primitive runtime validation "
               "is device-independent\n";
  return 0;
}
