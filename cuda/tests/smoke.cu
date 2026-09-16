#include "meep_cuda/runtime.hpp"
#include "meep_cuda/detail/bfast.hpp"
#include "meep_cuda/detail/curl.hpp"
#include "meep_cuda/detail/polarization.hpp"
#include "meep_cuda/detail/update_eh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct device_deleter {
  void operator()(void *pointer) const { meep_cuda::free_device(pointer); }
};

using device_pointer = std::unique_ptr<void, device_deleter>;

template <typename T>
device_pointer copy_vector_to_device(const std::vector<T> &values) {
  if (values.empty()) return device_pointer(nullptr);
  const std::size_t bytes = values.size() * sizeof(T);
  device_pointer pointer(meep_cuda::allocate_device_bytes(bytes));
  meep_cuda::copy_to_device(pointer.get(), values.data(), bytes);
  return pointer;
}

bool run_event_lifecycle_case() {
  const std::vector<float> source = {0.25f, -0.5f, 0.75f, 1.0f};
  std::vector<float> observed(source.size(), 0.0f);
  auto device_source = copy_vector_to_device(source);
  auto device_destination = copy_vector_to_device(observed);
  void *event = meep_cuda::create_event();
  try {
    meep_cuda::copy_fp32(
        static_cast<float *>(device_destination.get()),
        static_cast<const float *>(device_source.get()), source.size());
    meep_cuda::record_event(event);
    meep_cuda::synchronize_event(event);
    // A completion event is deliberately reusable across boundary phases.
    // Record and synchronize it a second time after another real kernel.
    meep_cuda::copy_fp32(
        static_cast<float *>(device_destination.get()),
        static_cast<const float *>(device_source.get()), source.size());
    meep_cuda::record_event(event);
    meep_cuda::synchronize_event(event);
  }
  catch (...) {
    meep_cuda::destroy_event(event);
    throw;
  }
  meep_cuda::destroy_event(event);
  meep_cuda::copy_to_host(
      observed.data(), device_destination.get(),
      observed.size() * sizeof(float));
  return observed == source;
}

struct test_case {
  const char *name;
  bool indexed;
  bool have_g1;
  bool have_g2;
  std::ptrdiff_t stride1;
  std::ptrdiff_t stride2;
  std::size_t count;
  int threads_per_block;
};

int requested_device_ordinal() {
  const char *value = std::getenv("MEEP_GPU_DEVICE");
  if (!value || !value[0]) return -1;

  char *end = nullptr;
  const long ordinal = std::strtol(value, &end, 10);
  if (!end || *end != '\0' || ordinal < 0 ||
      ordinal > std::numeric_limits<int>::max())
    throw std::invalid_argument("MEEP_GPU_DEVICE must be a nonnegative integer");
  return static_cast<int>(ordinal);
}

bool run_case(const test_case &test) {
  constexpr std::size_t array_size = 32768;
  constexpr float dtdx = 0.37f;

  const std::ptrdiff_t margin =
      std::max(std::abs(test.stride1), std::abs(test.stride2));
  const std::size_t safe_count =
      array_size - 2 * static_cast<std::size_t>(margin);
  if (test.count > safe_count)
    throw std::invalid_argument(std::string(test.name) + ": count exceeds safe array range");

  std::mt19937 generator(
      static_cast<unsigned int>(20260730 + test.count + test.threads_per_block));
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

  std::vector<float> field(array_size);
  std::vector<float> g1(array_size);
  std::vector<float> g2(array_size);
  for (std::size_t i = 0; i < array_size; ++i) {
    field[i] = distribution(generator);
    g1[i] = distribution(generator);
    g2[i] = distribution(generator);
  }

  std::vector<std::ptrdiff_t> indices;
  std::size_t base_offset = static_cast<std::size_t>(margin);
  if (test.indexed) {
    indices.reserve(test.count);
    for (std::size_t j = 0; j < test.count; ++j)
      indices.push_back(margin + static_cast<std::ptrdiff_t>(j));
    base_offset = 0;
  }

  std::vector<float> reference = field;
  for (std::size_t j = 0; j < test.count; ++j) {
    const std::ptrdiff_t i =
        test.indexed ? indices[j] : static_cast<std::ptrdiff_t>(base_offset + j);
    float curl = 0.0f;
    if (test.have_g1) curl += g1[i + test.stride1] - g1[i];
    if (test.have_g2) curl += g2[i] - g2[i + test.stride2];
    reference[i] -= dtdx * curl;
  }

  auto device_field = copy_vector_to_device(field);
  auto device_g1 = copy_vector_to_device(g1);
  auto device_g2 = copy_vector_to_device(g2);
  auto device_indices = copy_vector_to_device(indices);

  auto *field_pointer = static_cast<float *>(device_field.get()) + base_offset;
  const auto *g1_pointer =
      test.have_g1 ? static_cast<const float *>(device_g1.get()) + base_offset : nullptr;
  const auto *g2_pointer =
      test.have_g2 ? static_cast<const float *>(device_g2.get()) + base_offset : nullptr;
  const auto *index_pointer = test.indexed
                                  ? static_cast<const std::ptrdiff_t *>(device_indices.get())
                                  : nullptr;

  meep_cuda::step_curl_fp32(field_pointer, g1_pointer, g2_pointer, index_pointer, test.count,
                            test.stride1, test.stride2, dtdx, test.threads_per_block);
  meep_cuda::synchronize();
  meep_cuda::copy_to_host(field.data(), device_field.get(), field.size() * sizeof(float));

  float max_absolute_error = 0.0f;
  float max_reference = 0.0f;
  for (std::size_t i = 0; i < field.size(); ++i) {
    max_absolute_error = std::max(max_absolute_error, std::abs(field[i] - reference[i]));
    max_reference = std::max(max_reference, std::abs(reference[i]));
  }
  const float relative_error = max_absolute_error / std::max(max_reference, 1.0e-20f);
  std::cout << "case=" << test.name << " count=" << test.count
            << " max_abs_error=" << max_absolute_error << " relative_error=" << relative_error
            << '\n';

  return max_absolute_error <= 5.0e-6f && relative_error <= 5.0e-6f;
}

bool run_material_case(bool pml_f, bool pml_u, bool conductivity) {
  constexpr std::size_t array_size = 4096;
  constexpr std::size_t count = 2048;
  constexpr std::ptrdiff_t stride1 = 1;
  constexpr std::ptrdiff_t stride2 = 3;
  constexpr float dtdx = 0.23f;
  constexpr float dt = 0.04f;
  constexpr std::size_t sigma_count = 13;
  constexpr std::size_t sigma_u_count = 17;

  std::mt19937 generator(
      static_cast<unsigned int>(20260801 + 100 * pml_f + 10 * pml_u + conductivity));
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  std::vector<float> field(array_size);
  std::vector<float> field_u(array_size);
  std::vector<float> field_conductivity(array_size);
  std::vector<float> g1(array_size);
  std::vector<float> g2(array_size);
  std::vector<float> conductivity_values(array_size);
  std::vector<float> conductivity_inverse(array_size);
  for (std::size_t i = 0; i < array_size; ++i) {
    field[i] = distribution(generator);
    field_u[i] = distribution(generator);
    field_conductivity[i] = distribution(generator);
    g1[i] = distribution(generator);
    g2[i] = distribution(generator);
    conductivity_values[i] = 0.03f + 0.00001f * static_cast<float>(i);
    conductivity_inverse[i] =
        1.0f / (1.0f + 0.5f * dt * conductivity_values[i]);
  }

  std::vector<float> sigma(sigma_count);
  std::vector<float> kappa(sigma_count);
  std::vector<float> sigma_inverse(sigma_count);
  for (std::size_t i = 0; i < sigma_count; ++i) {
    sigma[i] = 0.002f * static_cast<float>(i + 1);
    kappa[i] = 1.0f + 0.01f * static_cast<float>(i);
    sigma_inverse[i] = 1.0f / (kappa[i] + sigma[i]);
  }
  std::vector<float> sigma_u(sigma_u_count);
  std::vector<float> kappa_u(sigma_u_count);
  std::vector<float> sigma_u_inverse(sigma_u_count);
  for (std::size_t i = 0; i < sigma_u_count; ++i) {
    sigma_u[i] = 0.0015f * static_cast<float>(i + 1);
    kappa_u[i] = 1.0f + 0.008f * static_cast<float>(i);
    sigma_u_inverse[i] = 1.0f / (kappa_u[i] + sigma_u[i]);
  }

  std::vector<meep_cuda::curl_index> indices;
  indices.reserve(count);
  for (std::size_t j = 0; j < count; ++j)
    indices.push_back(
        {static_cast<std::ptrdiff_t>(4 + j),
         pml_f ? static_cast<int>((3 * j + 1) % sigma_count) : -1,
         pml_u ? static_cast<int>((5 * j + 2) % sigma_u_count) : -1});

  std::vector<float> reference_field = field;
  std::vector<float> reference_field_u = field_u;
  std::vector<float> reference_field_conductivity = field_conductivity;
  for (const meep_cuda::curl_index &index : indices) {
    const meep_cuda::detail::curl_coefficients_fp32 coefficients = {
        pml_f,
        pml_u,
        conductivity,
        pml_f ? sigma[index.sigma] : 0.0f,
        pml_f ? kappa[index.sigma] : 1.0f,
        pml_f ? sigma_inverse[index.sigma] : 1.0f,
        pml_u ? sigma_u[index.sigma_u] : 0.0f,
        pml_u ? kappa_u[index.sigma_u] : 1.0f,
        pml_u ? sigma_u_inverse[index.sigma_u] : 1.0f,
        conductivity ? conductivity_values[index.field] : 0.0f,
        conductivity ? conductivity_inverse[index.field] : 1.0f,
        dt};
    const float curl = meep_cuda::detail::curl_term_fp32(
        g1.data(), g2.data(), index.field, stride1, stride2);
    meep_cuda::detail::apply_curl_update_fp32(
        reference_field[index.field],
        pml_u ? &reference_field_u[index.field] : nullptr,
        pml_f && conductivity
            ? &reference_field_conductivity[index.field]
            : nullptr,
        curl, dtdx, coefficients);
  }

  auto device_field = copy_vector_to_device(field);
  auto device_field_u = copy_vector_to_device(field_u);
  auto device_field_conductivity = copy_vector_to_device(field_conductivity);
  auto device_g1 = copy_vector_to_device(g1);
  auto device_g2 = copy_vector_to_device(g2);
  auto device_conductivity = copy_vector_to_device(conductivity_values);
  auto device_conductivity_inverse = copy_vector_to_device(conductivity_inverse);
  auto device_sigma = copy_vector_to_device(sigma);
  auto device_kappa = copy_vector_to_device(kappa);
  auto device_sigma_inverse = copy_vector_to_device(sigma_inverse);
  auto device_sigma_u = copy_vector_to_device(sigma_u);
  auto device_kappa_u = copy_vector_to_device(kappa_u);
  auto device_sigma_u_inverse = copy_vector_to_device(sigma_u_inverse);
  auto device_indices = copy_vector_to_device(indices);

  const meep_cuda::curl_material_fp32 material = {
      pml_f ? static_cast<const float *>(device_sigma.get()) : nullptr,
      pml_f ? static_cast<const float *>(device_kappa.get()) : nullptr,
      pml_f ? static_cast<const float *>(device_sigma_inverse.get()) : nullptr,
      pml_u ? static_cast<float *>(device_field_u.get()) : nullptr,
      pml_u ? static_cast<const float *>(device_sigma_u.get()) : nullptr,
      pml_u ? static_cast<const float *>(device_kappa_u.get()) : nullptr,
      pml_u ? static_cast<const float *>(device_sigma_u_inverse.get()) : nullptr,
      dt,
      conductivity ? static_cast<const float *>(device_conductivity.get()) : nullptr,
      conductivity
          ? static_cast<const float *>(device_conductivity_inverse.get())
          : nullptr,
      pml_f && conductivity
          ? static_cast<float *>(device_field_conductivity.get())
          : nullptr};
  meep_cuda::step_curl_material_fp32(
      static_cast<float *>(device_field.get()),
      static_cast<const float *>(device_g1.get()),
      static_cast<const float *>(device_g2.get()),
      static_cast<const meep_cuda::curl_index *>(device_indices.get()), count,
      stride1, stride2, dtdx, material);
  meep_cuda::synchronize();
  meep_cuda::copy_to_host(field.data(), device_field.get(),
                            field.size() * sizeof(float));
  meep_cuda::copy_to_host(field_u.data(), device_field_u.get(),
                            field_u.size() * sizeof(float));
  meep_cuda::copy_to_host(field_conductivity.data(),
                            device_field_conductivity.get(),
                            field_conductivity.size() * sizeof(float));

  float max_absolute_error = 0.0f;
  for (std::size_t i = 0; i < array_size; ++i) {
    max_absolute_error =
        std::max(max_absolute_error, std::abs(field[i] - reference_field[i]));
    max_absolute_error = std::max(
        max_absolute_error, std::abs(field_u[i] - reference_field_u[i]));
    max_absolute_error =
        std::max(max_absolute_error,
                 std::abs(field_conductivity[i] -
                          reference_field_conductivity[i]));
  }
  std::cout << "material_case pml_f=" << pml_f << " pml_u=" << pml_u
            << " conductivity=" << conductivity
            << " max_abs_error=" << max_absolute_error << '\n';
  return max_absolute_error <= 5.0e-6f;
}

struct bfast_test_case {
  const char *name;
  bool have_g1;
  bool have_g2;
  std::ptrdiff_t stride1;
  std::ptrdiff_t stride2;
  bool pml_f;
  bool pml_u;
  bool conductivity;
};

std::size_t point_count(const meep_cuda::index_space_fp32 &space) {
  return space.extent1 * space.extent2 * space.extent3;
}

float maximum_error(const std::vector<float> &actual,
                    const std::vector<float> &reference) {
  float result = 0.0f;
  for (std::size_t index = 0; index < actual.size(); ++index)
    result =
        std::max(result, std::abs(actual[index] - reference[index]));
  return result;
}

float maximum_change(const std::vector<float> &actual,
                     const std::vector<float> &initial) {
  return maximum_error(actual, initial);
}

bool run_bfast_case(const bfast_test_case &test, bool batched) {
  constexpr std::size_t array_size = 512;
  constexpr std::size_t sigma_count = 128;
  constexpr std::size_t recurrence_count = 3;
  constexpr float k1 = 0.29f;
  constexpr float k2 = -0.17f;

  std::mt19937 generator(static_cast<unsigned int>(
      20260901 + 1000 * test.have_g1 + 100 * test.have_g2 +
      10 * test.pml_f + 7 * test.pml_u + 3 * test.conductivity +
      (test.stride1 < 0 ? 17 : 0) + (test.stride2 < 0 ? 31 : 0)));
  std::uniform_real_distribution<float> distribution(-0.45f, 0.45f);
  std::vector<float> initial_field(array_size);
  std::vector<float> initial_bfast_field(array_size);
  std::vector<float> initial_field_u(array_size);
  std::vector<float> initial_field_conductivity(array_size);
  std::vector<float> g1(array_size);
  std::vector<float> g2(array_size);
  std::vector<float> conductivity_inverse(array_size);
  for (std::size_t index = 0; index < array_size; ++index) {
    initial_field[index] = distribution(generator);
    initial_bfast_field[index] = distribution(generator);
    initial_field_u[index] = distribution(generator);
    initial_field_conductivity[index] = distribution(generator);
    g1[index] = distribution(generator);
    g2[index] = distribution(generator);
    conductivity_inverse[index] =
        0.72f + 0.0003f * static_cast<float>(index);
  }
  std::vector<float> sigma_inverse(sigma_count);
  std::vector<float> sigma_u_inverse(sigma_count);
  for (std::size_t index = 0; index < sigma_count; ++index) {
    sigma_inverse[index] =
        0.81f + 0.0005f * static_cast<float>(index);
    sigma_u_inverse[index] =
        0.86f + 0.0004f * static_cast<float>(index);
  }

  const int no_coefficient = -1;
  const std::vector<meep_cuda::index_space_fp32> spaces = {
      {64, 31, 1, 1, 1, 0, 0,
       test.pml_f ? 3 : no_coefficient, test.pml_f ? 1 : 0, 0, 0,
       test.pml_u ? 7 : no_coefficient, test.pml_u ? 1 : 0, 0, 0},
      {256, 33, 1, 1, 1, 0, 0,
       test.pml_f ? 47 : no_coefficient, test.pml_f ? 1 : 0, 0, 0,
       test.pml_u ? 53 : no_coefficient, test.pml_u ? 1 : 0, 0, 0}};

  std::vector<float> reference_field = initial_field;
  std::vector<float> reference_bfast_field = initial_bfast_field;
  std::vector<float> reference_field_u = initial_field_u;
  std::vector<float> reference_field_conductivity =
      initial_field_conductivity;
  const float *reference_g1 = test.have_g1 ? g1.data() : nullptr;
  const float *reference_g2 = test.have_g2 ? g2.data() : nullptr;
  std::ptrdiff_t reference_stride1 = test.stride1;
  std::ptrdiff_t reference_stride2 = test.stride2;
  float reference_k1 = k1;
  float reference_k2 = k2;
  if (!reference_g1) {
    reference_g1 = reference_g2;
    reference_g2 = nullptr;
    reference_stride1 = reference_stride2;
    reference_stride2 = 0;
    reference_k1 = reference_k2;
    reference_k2 = k1;
  }
  for (std::size_t recurrence = 0; recurrence < recurrence_count;
       ++recurrence)
    for (const meep_cuda::index_space_fp32 &space : spaces)
      for (std::size_t point = 0; point < point_count(space); ++point) {
        std::size_t linear_index = point;
        const std::size_t coordinate3 =
            linear_index % space.extent3;
        linear_index /= space.extent3;
        const std::size_t coordinate2 =
            linear_index % space.extent2;
        const std::size_t coordinate1 =
            linear_index / space.extent2;
        const std::ptrdiff_t field_index =
            space.field_start +
            static_cast<std::ptrdiff_t>(coordinate1) *
                space.field_stride1 +
            static_cast<std::ptrdiff_t>(coordinate2) *
                space.field_stride2 +
            static_cast<std::ptrdiff_t>(coordinate3) *
                space.field_stride3;
        const int coefficient =
            space.coefficient_start +
            static_cast<int>(coordinate1) *
                space.coefficient_stride1 +
            static_cast<int>(coordinate2) *
                space.coefficient_stride2 +
            static_cast<int>(coordinate3) *
                space.coefficient_stride3;
        const int coefficient2 =
            space.coefficient2_start +
            static_cast<int>(coordinate1) *
                space.coefficient2_stride1 +
            static_cast<int>(coordinate2) *
                space.coefficient2_stride2 +
            static_cast<int>(coordinate3) *
                space.coefficient2_stride3;

        float drive =
            reference_k1 *
            (reference_g1[field_index + reference_stride1] +
             reference_g1[field_index]);
        if (reference_g2)
          drive -=
              reference_k2 *
              (reference_g2[field_index + reference_stride2] +
               reference_g2[field_index]);
        const float previous = reference_bfast_field[field_index];
        const bool alternating_recurrence =
            reference_g2 || test.pml_f || test.pml_u ||
            test.conductivity;
        reference_bfast_field[field_index] =
            drive - (alternating_recurrence ? previous : 0.0f);
        float delta =
            reference_bfast_field[field_index] - previous;
        if (test.conductivity)
          delta *= conductivity_inverse[field_index];
        if (test.conductivity && test.pml_f)
          reference_field_conductivity[field_index] += delta;
        if (test.pml_f) delta *= sigma_inverse[coefficient];
        if (test.pml_u) {
          reference_field_u[field_index] += delta;
          reference_field[field_index] +=
              sigma_u_inverse[coefficient2] * delta;
        }
        else
          reference_field[field_index] += delta;
      }

  std::vector<float> field = initial_field;
  std::vector<float> bfast_field = initial_bfast_field;
  std::vector<float> field_u = initial_field_u;
  std::vector<float> field_conductivity = initial_field_conductivity;
  auto device_field = copy_vector_to_device(field);
  auto device_bfast_field = copy_vector_to_device(bfast_field);
  auto device_field_u = copy_vector_to_device(field_u);
  auto device_field_conductivity =
      copy_vector_to_device(field_conductivity);
  auto device_g1 = copy_vector_to_device(g1);
  auto device_g2 = copy_vector_to_device(g2);
  auto device_sigma_inverse = copy_vector_to_device(sigma_inverse);
  auto device_sigma_u_inverse = copy_vector_to_device(sigma_u_inverse);
  auto device_conductivity_inverse =
      copy_vector_to_device(conductivity_inverse);
  auto device_spaces = copy_vector_to_device(spaces);

  const meep_cuda::bfast_material_fp32 material = {
      test.pml_f
          ? static_cast<const float *>(device_sigma_inverse.get())
          : nullptr,
      test.pml_u ? static_cast<float *>(device_field_u.get()) : nullptr,
      test.pml_u
          ? static_cast<const float *>(device_sigma_u_inverse.get())
          : nullptr,
      test.conductivity
          ? static_cast<const float *>(device_conductivity_inverse.get())
          : nullptr,
      test.conductivity && test.pml_f
          ? static_cast<float *>(device_field_conductivity.get())
          : nullptr};
  for (std::size_t recurrence = 0; recurrence < recurrence_count;
       ++recurrence) {
    if (batched)
      meep_cuda::step_bfast_batched_structured_fp32(
          static_cast<float *>(device_field.get()),
          test.have_g1 ? static_cast<const float *>(device_g1.get())
                       : nullptr,
          test.have_g2 ? static_cast<const float *>(device_g2.get())
                       : nullptr,
          static_cast<float *>(device_bfast_field.get()),
          static_cast<const meep_cuda::index_space_fp32 *>(
              device_spaces.get()),
          spaces.size(), 33u, test.stride1, test.stride2, k1, k2,
          material, 128);
    else
      for (const meep_cuda::index_space_fp32 &space : spaces)
        meep_cuda::step_bfast_structured_fp32(
            static_cast<float *>(device_field.get()),
            test.have_g1 ? static_cast<const float *>(device_g1.get())
                         : nullptr,
            test.have_g2 ? static_cast<const float *>(device_g2.get())
                         : nullptr,
            static_cast<float *>(device_bfast_field.get()), space,
            point_count(space), test.stride1, test.stride2, k1, k2,
            material, 128);
  }
  meep_cuda::synchronize();
  meep_cuda::copy_to_host(
      field.data(), device_field.get(), field.size() * sizeof(float));
  meep_cuda::copy_to_host(
      bfast_field.data(), device_bfast_field.get(),
      bfast_field.size() * sizeof(float));
  meep_cuda::copy_to_host(
      field_u.data(), device_field_u.get(),
      field_u.size() * sizeof(float));
  meep_cuda::copy_to_host(
      field_conductivity.data(), device_field_conductivity.get(),
      field_conductivity.size() * sizeof(float));

  const float field_error = maximum_error(field, reference_field);
  const float bfast_error =
      maximum_error(bfast_field, reference_bfast_field);
  const float field_u_error =
      maximum_error(field_u, reference_field_u);
  const float field_conductivity_error =
      maximum_error(
          field_conductivity, reference_field_conductivity);
  std::cout << "bfast_case=" << test.name
            << " launcher=" << (batched ? "batched" : "single")
            << " recurrence=" << recurrence_count
            << " field_error=" << field_error
            << " F_error=" << bfast_error
            << " field_u_error=" << field_u_error
            << " field_conductivity_error="
            << field_conductivity_error << '\n';

  const bool required_outputs_changed =
      maximum_change(field, initial_field) > 1.0e-7f &&
      maximum_change(bfast_field, initial_bfast_field) > 1.0e-7f &&
      (!test.pml_u ||
       maximum_change(field_u, initial_field_u) > 1.0e-7f) &&
      (!(test.pml_f && test.conductivity) ||
       maximum_change(
           field_conductivity, initial_field_conductivity) > 1.0e-7f);
  return field_error <= 4.0e-6f && bfast_error <= 4.0e-6f &&
         field_u_error <= 4.0e-6f &&
         field_conductivity_error <= 4.0e-6f &&
         required_outputs_changed;
}

bool run_update_eh_case(bool pml, int offdiagonal_count, bool nonlinear,
                        bool inverse_present, bool negative_strides) {
  constexpr std::size_t array_size = 4096;
  constexpr std::size_t count = 2048;
  const std::ptrdiff_t field_stride = negative_strides ? -1 : 1;
  const std::ptrdiff_t stride1 = negative_strides ? -2 : 2;
  const std::ptrdiff_t stride2 = negative_strides ? -3 : 3;
  constexpr std::size_t sigma_count = 19;

  std::mt19937 generator(static_cast<unsigned int>(
      20260802 + 1000 * pml + 100 * offdiagonal_count +
      10 * nonlinear + inverse_present + 10000 * negative_strides));
  std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
  std::vector<float> field(array_size);
  std::vector<float> g(array_size);
  std::vector<float> g1(array_size);
  std::vector<float> g2(array_size);
  std::vector<float> inverse(array_size);
  std::vector<float> u1(array_size);
  std::vector<float> u2(array_size);
  std::vector<float> chi2(array_size);
  std::vector<float> chi3(array_size);
  std::vector<float> field_w(array_size);
  for (std::size_t i = 0; i < array_size; ++i) {
    field[i] = distribution(generator);
    g[i] = distribution(generator);
    g1[i] = distribution(generator);
    g2[i] = distribution(generator);
    inverse[i] = 0.8f + 0.00002f * static_cast<float>(i);
    u1[i] = 0.04f + 0.000003f * static_cast<float>(i);
    u2[i] = -0.03f + 0.000002f * static_cast<float>(i);
    chi2[i] = 0.015f + 0.000001f * static_cast<float>(i);
    chi3[i] = 0.009f + 0.000001f * static_cast<float>(i);
    field_w[i] = distribution(generator);
  }
  std::vector<float> sigma(sigma_count);
  std::vector<float> kappa(sigma_count);
  for (std::size_t i = 0; i < sigma_count; ++i) {
    sigma[i] = 0.002f * static_cast<float>(i + 1);
    kappa[i] = 1.0f + 0.007f * static_cast<float>(i);
  }

  std::vector<meep_cuda::update_eh_index> indices;
  indices.reserve(count);
  for (std::size_t j = 0; j < count; ++j)
    indices.push_back(
        {static_cast<std::ptrdiff_t>(4 + j),
         pml ? static_cast<int>((7 * j + 3) % sigma_count) : -1});

  const bool have_g1 = offdiagonal_count >= 1 || nonlinear;
  const bool have_g2 = offdiagonal_count >= 2 || nonlinear;
  std::vector<float> reference_field = field;
  std::vector<float> reference_field_w = field_w;
  for (const meep_cuda::update_eh_index &index : indices) {
    const std::ptrdiff_t i = index.field;
    const float off1 =
        offdiagonal_count >= 1
            ? meep_cuda::detail::offdiagonal_average_fp32(
                  u1.data(), g1.data(), i, field_stride, stride1)
            : 0.0f;
    const float off2 =
        offdiagonal_count >= 2
            ? meep_cuda::detail::offdiagonal_average_fp32(
                  u2.data(), g2.data(), i, field_stride, stride2)
            : 0.0f;
    float norm = g[i] * g[i];
    if (nonlinear && have_g1) {
      const float sum =
          g1[i] + g1[i + field_stride] + g1[i - stride1] +
          g1[i + field_stride - stride1];
      norm += 0.0625f * sum * sum;
    }
    if (nonlinear && have_g2) {
      const float sum =
          g2[i] + g2[i + field_stride] + g2[i - stride2] +
          g2[i + field_stride - stride2];
      norm += 0.0625f * sum * sum;
    }
    const meep_cuda::detail::update_eh_coefficients_fp32 coefficients = {
        pml,
        nonlinear,
        inverse_present ? inverse[i] : 1.0f,
        nonlinear ? chi2[i] : 0.0f,
        nonlinear ? chi3[i] : 0.0f,
        pml ? sigma[index.sigma] : 0.0f,
        pml ? kappa[index.sigma] : 1.0f};
    meep_cuda::detail::apply_update_eh_fp32(
        reference_field[i], pml ? &reference_field_w[i] : nullptr, g[i],
        off1, off2, norm, coefficients);
  }

  auto device_field = copy_vector_to_device(field);
  auto device_g = copy_vector_to_device(g);
  auto device_g1 = copy_vector_to_device(g1);
  auto device_g2 = copy_vector_to_device(g2);
  auto device_inverse = copy_vector_to_device(inverse);
  auto device_u1 = copy_vector_to_device(u1);
  auto device_u2 = copy_vector_to_device(u2);
  auto device_chi2 = copy_vector_to_device(chi2);
  auto device_chi3 = copy_vector_to_device(chi3);
  auto device_field_w = copy_vector_to_device(field_w);
  auto device_sigma = copy_vector_to_device(sigma);
  auto device_kappa = copy_vector_to_device(kappa);
  auto device_indices = copy_vector_to_device(indices);

  const meep_cuda::update_eh_material_fp32 material = {
      inverse_present ? static_cast<const float *>(device_inverse.get())
                      : nullptr,
      offdiagonal_count >= 1
          ? static_cast<const float *>(device_u1.get())
          : nullptr,
      offdiagonal_count >= 2
          ? static_cast<const float *>(device_u2.get())
          : nullptr,
      nonlinear ? static_cast<const float *>(device_chi2.get()) : nullptr,
      nonlinear ? static_cast<const float *>(device_chi3.get()) : nullptr,
      pml ? static_cast<float *>(device_field_w.get()) : nullptr,
      pml ? static_cast<const float *>(device_sigma.get()) : nullptr,
      pml ? static_cast<const float *>(device_kappa.get()) : nullptr};
  meep_cuda::update_eh_fp32(
      static_cast<float *>(device_field.get()),
      static_cast<const float *>(device_g.get()),
      have_g1 ? static_cast<const float *>(device_g1.get()) : nullptr,
      have_g2 ? static_cast<const float *>(device_g2.get()) : nullptr,
      static_cast<const meep_cuda::update_eh_index *>(device_indices.get()),
      count, field_stride, stride1, stride2, material);
  meep_cuda::synchronize();
  meep_cuda::copy_to_host(field.data(), device_field.get(),
                          field.size() * sizeof(float));
  meep_cuda::copy_to_host(field_w.data(), device_field_w.get(),
                          field_w.size() * sizeof(float));

  float max_absolute_error = 0.0f;
  for (std::size_t i = 0; i < array_size; ++i) {
    max_absolute_error =
        std::max(max_absolute_error,
                 std::abs(field[i] - reference_field[i]));
    max_absolute_error =
        std::max(max_absolute_error,
                 std::abs(field_w[i] - reference_field_w[i]));
  }
  std::cout << "update_eh_case pml=" << pml
            << " offdiagonal=" << offdiagonal_count
            << " nonlinear=" << nonlinear
            << " inverse=" << inverse_present
            << " negative_strides=" << negative_strides
            << " max_abs_error=" << max_absolute_error << '\n';
  return max_absolute_error <= 8.0e-6f;
}

bool run_phase_batched_curl_case() {
  constexpr std::size_t array_size = 2048;
  constexpr std::size_t coefficient_count = 1024;
  constexpr std::ptrdiff_t stride1 = 3;
  constexpr std::ptrdiff_t stride2 = -5;
  constexpr float dtdx = 0.21f;
  constexpr float dt = 0.035f;

  std::mt19937 generator(20260804u);
  std::uniform_real_distribution<float> distribution(-0.4f, 0.4f);
  std::vector<float> initial_a(array_size);
  std::vector<float> initial_b(array_size);
  std::vector<float> g1_a(array_size);
  std::vector<float> g2_a(array_size);
  std::vector<float> g1_b(array_size);
  std::vector<float> g2_b(array_size);
  std::vector<float> initial_u(array_size);
  std::vector<float> initial_conductivity_field(array_size);
  std::vector<float> conductivity(array_size);
  std::vector<float> conductivity_inverse(array_size);
  for (std::size_t i = 0; i < array_size; ++i) {
    initial_a[i] = distribution(generator);
    initial_b[i] = distribution(generator);
    g1_a[i] = distribution(generator);
    g2_a[i] = distribution(generator);
    g1_b[i] = distribution(generator);
    g2_b[i] = distribution(generator);
    initial_u[i] = distribution(generator);
    initial_conductivity_field[i] = distribution(generator);
    conductivity[i] = 0.02f + 0.00001f * static_cast<float>(i);
    conductivity_inverse[i] =
        1.0f / (1.0f + 0.5f * dt * conductivity[i]);
  }
  std::vector<float> sigma(coefficient_count);
  std::vector<float> kappa(coefficient_count);
  std::vector<float> sigma_inverse(coefficient_count);
  std::vector<float> sigma_u(coefficient_count);
  std::vector<float> kappa_u(coefficient_count);
  std::vector<float> sigma_u_inverse(coefficient_count);
  for (std::size_t i = 0; i < coefficient_count; ++i) {
    sigma[i] = 0.001f * static_cast<float>(i % 17 + 1);
    kappa[i] = 1.0f + 0.002f * static_cast<float>(i % 23);
    sigma_inverse[i] = 1.0f / (kappa[i] + sigma[i]);
    sigma_u[i] = 0.0008f * static_cast<float>(i % 19 + 1);
    kappa_u[i] = 1.0f + 0.0015f * static_cast<float>(i % 29);
    sigma_u_inverse[i] = 1.0f / (kappa_u[i] + sigma_u[i]);
  }

  std::vector<float> reference_a = initial_a;
  std::vector<float> reference_b = initial_b;
  std::vector<float> candidate_a = initial_a;
  std::vector<float> candidate_b = initial_b;
  std::vector<float> reference_u = initial_u;
  std::vector<float> candidate_u = initial_u;
  std::vector<float> reference_conductivity_field =
      initial_conductivity_field;
  std::vector<float> candidate_conductivity_field =
      initial_conductivity_field;

  auto device_reference_a = copy_vector_to_device(reference_a);
  auto device_reference_b = copy_vector_to_device(reference_b);
  auto device_candidate_a = copy_vector_to_device(candidate_a);
  auto device_candidate_b = copy_vector_to_device(candidate_b);
  auto device_g1_a = copy_vector_to_device(g1_a);
  auto device_g2_a = copy_vector_to_device(g2_a);
  auto device_g1_b = copy_vector_to_device(g1_b);
  auto device_g2_b = copy_vector_to_device(g2_b);
  auto device_reference_u = copy_vector_to_device(reference_u);
  auto device_candidate_u = copy_vector_to_device(candidate_u);
  auto device_reference_conductivity_field =
      copy_vector_to_device(reference_conductivity_field);
  auto device_candidate_conductivity_field =
      copy_vector_to_device(candidate_conductivity_field);
  auto device_conductivity = copy_vector_to_device(conductivity);
  auto device_conductivity_inverse =
      copy_vector_to_device(conductivity_inverse);
  auto device_sigma = copy_vector_to_device(sigma);
  auto device_kappa = copy_vector_to_device(kappa);
  auto device_sigma_inverse = copy_vector_to_device(sigma_inverse);
  auto device_sigma_u = copy_vector_to_device(sigma_u);
  auto device_kappa_u = copy_vector_to_device(kappa_u);
  auto device_sigma_u_inverse = copy_vector_to_device(sigma_u_inverse);

  const meep_cuda::index_space_fp32 bare_space = {
      32, 257, 1, 1, 1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0};
  const meep_cuda::index_space_fp32 full_space = {
      96, 17, 19, 2, 1, 17, 323,
      11, 1, 17, 323, 23, 1, 17, 323};
  const meep_cuda::curl_material_fp32 bare_material = {};
  const auto make_full_material = [&](float *field_u,
                                      float *field_conductivity) {
    return meep_cuda::curl_material_fp32{
        static_cast<const float *>(device_sigma.get()),
        static_cast<const float *>(device_kappa.get()),
        static_cast<const float *>(device_sigma_inverse.get()), field_u,
        static_cast<const float *>(device_sigma_u.get()),
        static_cast<const float *>(device_kappa_u.get()),
        static_cast<const float *>(device_sigma_u_inverse.get()), dt,
        static_cast<const float *>(device_conductivity.get()),
        static_cast<const float *>(device_conductivity_inverse.get()),
        field_conductivity};
  };
  const meep_cuda::curl_material_fp32 reference_full_material =
      make_full_material(
          static_cast<float *>(device_reference_u.get()),
          static_cast<float *>(device_reference_conductivity_field.get()));
  const meep_cuda::curl_material_fp32 candidate_full_material =
      make_full_material(
          static_cast<float *>(device_candidate_u.get()),
          static_cast<float *>(device_candidate_conductivity_field.get()));

  meep_cuda::step_curl_material_structured_fp32(
      static_cast<float *>(device_reference_a.get()),
      static_cast<const float *>(device_g1_a.get()),
      static_cast<const float *>(device_g2_a.get()), bare_space,
      point_count(bare_space), stride1, stride2, dtdx, bare_material);
  meep_cuda::step_curl_material_structured_fp32(
      static_cast<float *>(device_reference_b.get()),
      static_cast<const float *>(device_g1_b.get()),
      static_cast<const float *>(device_g2_b.get()), full_space,
      point_count(full_space), stride1, stride2, dtdx,
      reference_full_material);

  constexpr std::size_t threads = 256;
  const std::size_t first_blocks =
      (point_count(bare_space) + threads - 1) / threads;
  const std::size_t second_blocks =
      (point_count(full_space) + threads - 1) / threads;
  const std::vector<meep_cuda::curl_phase_operation_fp32> operations = {
      {static_cast<float *>(device_candidate_a.get()),
       static_cast<const float *>(device_g1_a.get()),
       static_cast<const float *>(device_g2_a.get()), bare_space,
       point_count(bare_space), 0, stride1, stride2, dtdx, bare_material},
      {static_cast<float *>(device_candidate_b.get()),
       static_cast<const float *>(device_g1_b.get()),
       static_cast<const float *>(device_g2_b.get()), full_space,
       point_count(full_space), first_blocks, stride1, stride2, dtdx,
       candidate_full_material}};
  std::vector<std::uint32_t> block_operation_indices(
      first_blocks, static_cast<std::uint32_t>(0));
  block_operation_indices.insert(
      block_operation_indices.end(), second_blocks,
      static_cast<std::uint32_t>(1));
  auto device_operations = copy_vector_to_device(operations);
  auto device_block_operation_indices =
      copy_vector_to_device(block_operation_indices);
  meep_cuda::step_curl_material_phase_batched_fp32(
      static_cast<const meep_cuda::curl_phase_operation_fp32 *>(
          device_operations.get()),
      static_cast<const std::uint32_t *>(
          device_block_operation_indices.get()),
      operations.size(), first_blocks + second_blocks);
  meep_cuda::synchronize();

  meep_cuda::copy_to_host(reference_a.data(), device_reference_a.get(),
                          reference_a.size() * sizeof(float));
  meep_cuda::copy_to_host(reference_b.data(), device_reference_b.get(),
                          reference_b.size() * sizeof(float));
  meep_cuda::copy_to_host(candidate_a.data(), device_candidate_a.get(),
                          candidate_a.size() * sizeof(float));
  meep_cuda::copy_to_host(candidate_b.data(), device_candidate_b.get(),
                          candidate_b.size() * sizeof(float));
  meep_cuda::copy_to_host(reference_u.data(), device_reference_u.get(),
                          reference_u.size() * sizeof(float));
  meep_cuda::copy_to_host(candidate_u.data(), device_candidate_u.get(),
                          candidate_u.size() * sizeof(float));
  meep_cuda::copy_to_host(
      reference_conductivity_field.data(),
      device_reference_conductivity_field.get(),
      reference_conductivity_field.size() * sizeof(float));
  meep_cuda::copy_to_host(
      candidate_conductivity_field.data(),
      device_candidate_conductivity_field.get(),
      candidate_conductivity_field.size() * sizeof(float));

  float error = maximum_error(candidate_a, reference_a);
  error = std::max(error, maximum_error(candidate_b, reference_b));
  error = std::max(error, maximum_error(candidate_u, reference_u));
  error = std::max(
      error, maximum_error(candidate_conductivity_field,
                           reference_conductivity_field));
  const bool changed = maximum_change(candidate_a, initial_a) > 1.0e-7f &&
                       maximum_change(candidate_b, initial_b) > 1.0e-7f &&
                       maximum_change(candidate_u, initial_u) > 1.0e-7f &&
                       maximum_change(candidate_conductivity_field,
                                      initial_conductivity_field) > 1.0e-7f;
  std::cout << "phase_batched_curl operations=" << operations.size()
            << " blocks=" << first_blocks + second_blocks
            << " max_abs_error=" << error << '\n';
  return error <= 5.0e-6f && changed;
}

bool run_phase_batched_update_eh_case() {
  constexpr std::size_t array_size = 2048;
  constexpr std::size_t coefficient_count = 1024;
  constexpr std::ptrdiff_t field_stride = -1;
  constexpr std::ptrdiff_t stride1 = -2;
  constexpr std::ptrdiff_t stride2 = 3;

  std::mt19937 generator(20260805u);
  std::uniform_real_distribution<float> distribution(-0.35f, 0.35f);
  std::vector<float> initial_a(array_size);
  std::vector<float> initial_b(array_size);
  std::vector<float> g_a(array_size);
  std::vector<float> g_b(array_size);
  std::vector<float> g1_b(array_size);
  std::vector<float> g2_b(array_size);
  std::vector<float> inverse(array_size);
  std::vector<float> offdiagonal1(array_size);
  std::vector<float> offdiagonal2(array_size);
  std::vector<float> chi2(array_size);
  std::vector<float> chi3(array_size);
  std::vector<float> initial_w(array_size);
  for (std::size_t i = 0; i < array_size; ++i) {
    initial_a[i] = distribution(generator);
    initial_b[i] = distribution(generator);
    g_a[i] = distribution(generator);
    g_b[i] = distribution(generator);
    g1_b[i] = distribution(generator);
    g2_b[i] = distribution(generator);
    inverse[i] = 0.75f + 0.00003f * static_cast<float>(i);
    offdiagonal1[i] = 0.025f + 0.000002f * static_cast<float>(i);
    offdiagonal2[i] = -0.017f + 0.000001f * static_cast<float>(i);
    chi2[i] = 0.011f + 0.000001f * static_cast<float>(i);
    chi3[i] = 0.006f + 0.000001f * static_cast<float>(i);
    initial_w[i] = distribution(generator);
  }
  std::vector<float> sigma(coefficient_count);
  std::vector<float> kappa(coefficient_count);
  for (std::size_t i = 0; i < coefficient_count; ++i) {
    sigma[i] = 0.0012f * static_cast<float>(i % 13 + 1);
    kappa[i] = 1.0f + 0.002f * static_cast<float>(i % 31);
  }

  std::vector<float> reference_a = initial_a;
  std::vector<float> reference_b = initial_b;
  std::vector<float> candidate_a = initial_a;
  std::vector<float> candidate_b = initial_b;
  std::vector<float> reference_w = initial_w;
  std::vector<float> candidate_w = initial_w;
  auto device_reference_a = copy_vector_to_device(reference_a);
  auto device_reference_b = copy_vector_to_device(reference_b);
  auto device_candidate_a = copy_vector_to_device(candidate_a);
  auto device_candidate_b = copy_vector_to_device(candidate_b);
  auto device_g_a = copy_vector_to_device(g_a);
  auto device_g_b = copy_vector_to_device(g_b);
  auto device_g1_b = copy_vector_to_device(g1_b);
  auto device_g2_b = copy_vector_to_device(g2_b);
  auto device_inverse = copy_vector_to_device(inverse);
  auto device_offdiagonal1 = copy_vector_to_device(offdiagonal1);
  auto device_offdiagonal2 = copy_vector_to_device(offdiagonal2);
  auto device_chi2 = copy_vector_to_device(chi2);
  auto device_chi3 = copy_vector_to_device(chi3);
  auto device_reference_w = copy_vector_to_device(reference_w);
  auto device_candidate_w = copy_vector_to_device(candidate_w);
  auto device_sigma = copy_vector_to_device(sigma);
  auto device_kappa = copy_vector_to_device(kappa);

  const meep_cuda::index_space_fp32 bare_space = {
      40, 257, 1, 1, 1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0};
  const meep_cuda::index_space_fp32 full_space = {
      112, 13, 17, 2, 1, 13, 221,
      19, 1, 13, 221, -1, 0, 0, 0};
  const meep_cuda::update_eh_material_fp32 bare_material = {};
  const auto make_full_material = [&](float *field_w) {
    return meep_cuda::update_eh_material_fp32{
        static_cast<const float *>(device_inverse.get()),
        static_cast<const float *>(device_offdiagonal1.get()),
        static_cast<const float *>(device_offdiagonal2.get()),
        static_cast<const float *>(device_chi2.get()),
        static_cast<const float *>(device_chi3.get()), field_w,
        static_cast<const float *>(device_sigma.get()),
        static_cast<const float *>(device_kappa.get())};
  };
  const meep_cuda::update_eh_material_fp32 reference_full_material =
      make_full_material(static_cast<float *>(device_reference_w.get()));
  const meep_cuda::update_eh_material_fp32 candidate_full_material =
      make_full_material(static_cast<float *>(device_candidate_w.get()));

  meep_cuda::update_eh_structured_fp32(
      static_cast<float *>(device_reference_a.get()),
      static_cast<const float *>(device_g_a.get()), nullptr, nullptr,
      bare_space, point_count(bare_space), field_stride, stride1, stride2,
      bare_material);
  meep_cuda::update_eh_structured_fp32(
      static_cast<float *>(device_reference_b.get()),
      static_cast<const float *>(device_g_b.get()),
      static_cast<const float *>(device_g1_b.get()),
      static_cast<const float *>(device_g2_b.get()), full_space,
      point_count(full_space), field_stride, stride1, stride2,
      reference_full_material);

  constexpr std::size_t threads = 256;
  const std::size_t first_blocks =
      (point_count(bare_space) + threads - 1) / threads;
  const std::size_t second_blocks =
      (point_count(full_space) + threads - 1) / threads;
  const std::vector<meep_cuda::update_eh_phase_operation_fp32> operations = {
      {static_cast<float *>(device_candidate_a.get()),
       static_cast<const float *>(device_g_a.get()), nullptr, nullptr,
       bare_space, point_count(bare_space), 0, field_stride, stride1,
       stride2, bare_material},
      {static_cast<float *>(device_candidate_b.get()),
       static_cast<const float *>(device_g_b.get()),
       static_cast<const float *>(device_g1_b.get()),
       static_cast<const float *>(device_g2_b.get()), full_space,
       point_count(full_space), first_blocks, field_stride, stride1,
       stride2, candidate_full_material}};
  std::vector<std::uint32_t> block_operation_indices(
      first_blocks, static_cast<std::uint32_t>(0));
  block_operation_indices.insert(
      block_operation_indices.end(), second_blocks,
      static_cast<std::uint32_t>(1));
  auto device_operations = copy_vector_to_device(operations);
  auto device_block_operation_indices =
      copy_vector_to_device(block_operation_indices);
  meep_cuda::update_eh_phase_batched_fp32(
      static_cast<const meep_cuda::update_eh_phase_operation_fp32 *>(
          device_operations.get()),
      static_cast<const std::uint32_t *>(
          device_block_operation_indices.get()),
      operations.size(), first_blocks + second_blocks);
  meep_cuda::synchronize();

  meep_cuda::copy_to_host(reference_a.data(), device_reference_a.get(),
                          reference_a.size() * sizeof(float));
  meep_cuda::copy_to_host(reference_b.data(), device_reference_b.get(),
                          reference_b.size() * sizeof(float));
  meep_cuda::copy_to_host(candidate_a.data(), device_candidate_a.get(),
                          candidate_a.size() * sizeof(float));
  meep_cuda::copy_to_host(candidate_b.data(), device_candidate_b.get(),
                          candidate_b.size() * sizeof(float));
  meep_cuda::copy_to_host(reference_w.data(), device_reference_w.get(),
                          reference_w.size() * sizeof(float));
  meep_cuda::copy_to_host(candidate_w.data(), device_candidate_w.get(),
                          candidate_w.size() * sizeof(float));

  float error = maximum_error(candidate_a, reference_a);
  error = std::max(error, maximum_error(candidate_b, reference_b));
  error = std::max(error, maximum_error(candidate_w, reference_w));
  const bool changed = maximum_change(candidate_a, initial_a) > 1.0e-7f &&
                       maximum_change(candidate_b, initial_b) > 1.0e-7f &&
                       maximum_change(candidate_w, initial_w) > 1.0e-7f;
  std::cout << "phase_batched_update_eh operations=" << operations.size()
            << " blocks=" << first_blocks + second_blocks
            << " max_abs_error=" << error << '\n';
  return error <= 8.0e-6f && changed;
}

bool run_lorentzian_case(int offdiagonal_count, bool drude,
                         bool negative_strides, bool increment_state,
                         bool structured) {
  constexpr std::size_t array_size = 4096;
  constexpr std::size_t count = 2048;
  const std::ptrdiff_t field_stride = negative_strides ? -1 : 1;
  const std::ptrdiff_t stride1 = negative_strides ? -2 : 2;
  const std::ptrdiff_t stride2 = negative_strides ? -3 : 3;

  std::mt19937 generator(static_cast<unsigned int>(
      20260803 + 100 * offdiagonal_count + 10 * drude +
      negative_strides + 1000 * increment_state + 2000 * structured));
  std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
  std::vector<float> polarization(array_size);
  std::vector<float> previous(array_size);
  std::vector<float> field(array_size);
  std::vector<float> field1(array_size);
  std::vector<float> field2(array_size);
  std::vector<float> sigma(array_size);
  std::vector<float> sigma1(array_size);
  std::vector<float> sigma2(array_size);
  for (std::size_t i = 0; i < array_size; ++i) {
    polarization[i] = distribution(generator);
    previous[i] = distribution(generator);
    field[i] = distribution(generator);
    field1[i] = distribution(generator);
    field2[i] = distribution(generator);
    sigma[i] =
        offdiagonal_count > 0 && i % 37 == 0
            ? 0.0f
            : 0.2f + 0.00003f * static_cast<float>(i);
    sigma1[i] = 0.03f + 0.000002f * static_cast<float>(i);
    sigma2[i] = -0.02f + 0.000001f * static_cast<float>(i);
  }
  std::vector<std::ptrdiff_t> indices;
  indices.reserve(count);
  for (std::size_t j = 0; j < count; ++j)
    indices.push_back(static_cast<std::ptrdiff_t>(4 + j));
  meep_cuda::index_space_fp32 index_space = {};
  index_space.field_start = 4;
  index_space.extent1 = count;
  index_space.extent2 = 1;
  index_space.extent3 = 1;
  index_space.field_stride1 = 1;
  index_space.coefficient_start = -1;
  index_space.coefficient2_start = -1;

  const meep_cuda::detail::lorentzian_coefficients_fp32 coefficients = {
      0.93f, 0.89f, 0.08f, drude ? 0.0f : 0.08f};
  std::vector<float> reference_polarization = polarization;
  std::vector<float> reference_previous = previous;
  for (const std::ptrdiff_t i : indices) {
    const float offdiagonal1 =
        offdiagonal_count >= 1
            ? meep_cuda::detail::lorentzian_offdiagonal_average_fp32(
                  sigma1.data(), field1.data(), i, field_stride, stride1)
            : 0.0f;
    const float offdiagonal2 =
        offdiagonal_count >= 2
            ? meep_cuda::detail::lorentzian_offdiagonal_average_fp32(
                  sigma2.data(), field2.data(), i, field_stride, stride2)
            : 0.0f;
    if (increment_state)
      meep_cuda::detail::apply_lorentzian_increment_update_fp32(
          reference_polarization[i], reference_previous[i], field[i],
          sigma[i], offdiagonal1, offdiagonal2, offdiagonal_count > 0,
          coefficients);
    else
      meep_cuda::detail::apply_lorentzian_update_fp32(
          reference_polarization[i], reference_previous[i], field[i],
          sigma[i], offdiagonal1, offdiagonal2, offdiagonal_count > 0,
          coefficients);
  }

  auto device_polarization = copy_vector_to_device(polarization);
  auto device_previous = copy_vector_to_device(previous);
  auto device_field = copy_vector_to_device(field);
  auto device_field1 = copy_vector_to_device(field1);
  auto device_field2 = copy_vector_to_device(field2);
  auto device_sigma = copy_vector_to_device(sigma);
  auto device_sigma1 = copy_vector_to_device(sigma1);
  auto device_sigma2 = copy_vector_to_device(sigma2);
  auto device_indices = copy_vector_to_device(indices);
  const meep_cuda::lorentzian_material_fp32 material = {
      static_cast<const float *>(device_sigma.get()),
      offdiagonal_count >= 1
          ? static_cast<const float *>(device_sigma1.get())
          : nullptr,
      offdiagonal_count >= 2
          ? static_cast<const float *>(device_sigma2.get())
          : nullptr,
      coefficients.gamma_inverse,
      coefficients.gamma_previous,
      coefficients.omega_dt_squared,
      coefficients.omega_dt_squared_denominator};
  float *device_p = static_cast<float *>(device_polarization.get());
  float *device_state = static_cast<float *>(device_previous.get());
  const float *device_w = static_cast<const float *>(device_field.get());
  const float *device_w1 =
      offdiagonal_count >= 1
          ? static_cast<const float *>(device_field1.get())
          : nullptr;
  const float *device_w2 =
      offdiagonal_count >= 2
          ? static_cast<const float *>(device_field2.get())
          : nullptr;
  if (increment_state && structured)
    meep_cuda::update_lorentzian_increment_structured_fp32(
        device_p, device_state, device_w, device_w1, device_w2, index_space,
        count, field_stride, stride1, stride2, material);
  else if (increment_state)
    meep_cuda::update_lorentzian_increment_fp32(
        device_p, device_state, device_w, device_w1, device_w2,
        static_cast<const std::ptrdiff_t *>(device_indices.get()), count,
        field_stride, stride1, stride2, material);
  else if (structured)
    meep_cuda::update_lorentzian_structured_fp32(
        device_p, device_state, device_w, device_w1, device_w2, index_space,
        count, field_stride, stride1, stride2, material);
  else
    meep_cuda::update_lorentzian_fp32(
        device_p, device_state, device_w, device_w1, device_w2,
        static_cast<const std::ptrdiff_t *>(device_indices.get()), count,
        field_stride, stride1, stride2, material);
  meep_cuda::synchronize();
  meep_cuda::copy_to_host(
      polarization.data(), device_polarization.get(),
      polarization.size() * sizeof(float));
  meep_cuda::copy_to_host(previous.data(), device_previous.get(),
                          previous.size() * sizeof(float));

  float max_absolute_error = 0.0f;
  for (std::size_t i = 0; i < array_size; ++i) {
    max_absolute_error =
        std::max(max_absolute_error,
                 std::abs(polarization[i] - reference_polarization[i]));
    max_absolute_error =
        std::max(max_absolute_error,
                 std::abs(previous[i] - reference_previous[i]));
  }
  std::cout << "lorentzian_case offdiagonal=" << offdiagonal_count
            << " drude=" << drude
            << " negative_strides=" << negative_strides
            << " increment_state=" << increment_state
            << " structured=" << structured
            << " max_abs_error=" << max_absolute_error << '\n';
  return max_absolute_error <= 8.0e-6f;
}

bool run_gyrotropic_case(int model, int transverse_count,
                         bool negative_strides) {
  constexpr std::size_t array_size = 4096;
  constexpr std::size_t count = 2048;
  const std::ptrdiff_t field_stride = negative_strides ? -1 : 1;
  const std::ptrdiff_t stride1 = negative_strides ? -2 : 2;
  const std::ptrdiff_t stride2 = negative_strides ? -3 : 3;

  std::mt19937 generator(static_cast<unsigned int>(
      20260817 + 100 * model + 10 * transverse_count +
      negative_strides));
  std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
  std::vector<float> polarization0(array_size);
  std::vector<float> polarization1(array_size);
  std::vector<float> polarization2(array_size);
  std::vector<float> previous0(array_size);
  std::vector<float> previous1(array_size);
  std::vector<float> previous2(array_size);
  std::vector<float> field0(array_size);
  std::vector<float> field1(array_size);
  std::vector<float> field2(array_size);
  std::vector<float> sigma(array_size);
  for (std::size_t index = 0; index < array_size; ++index) {
    polarization0[index] = distribution(generator);
    polarization1[index] = distribution(generator);
    polarization2[index] = distribution(generator);
    previous0[index] = distribution(generator);
    previous1[index] = distribution(generator);
    previous2[index] = distribution(generator);
    field0[index] = distribution(generator);
    field1[index] = distribution(generator);
    field2[index] = distribution(generator);
    sigma[index] = 0.2f + 0.00003f * static_cast<float>(index);
  }

  meep_cuda::detail::gyrotropic_coefficients_fp32 coefficients = {
      model,
      {0.91f, 0.02f, -0.03f,
       -0.01f, 0.88f, 0.04f,
       0.05f, -0.02f, 0.93f},
      {0.0f, 0.12f, -0.07f,
       -0.12f, 0.0f, 0.09f,
       0.07f, -0.09f, 0.0f},
      model == meep_cuda::detail::gyrotropic_drude_fp32 ? 2.0f : 1.83f,
      0.94f,
      0.17f,
      0.026f,
      0.14f,
      0.031f,
      0.008f,
      0.052f};
  std::vector<float> reference_polarization0 = polarization0;
  std::vector<float> reference_polarization1 = polarization1;
  std::vector<float> reference_polarization2 = polarization2;
  std::vector<float> reference_previous0 = previous0;
  std::vector<float> reference_previous1 = previous1;
  std::vector<float> reference_previous2 = previous2;

  meep_cuda::index_space_fp32 index_space = {};
  index_space.field_start = 4;
  index_space.extent1 = count;
  index_space.extent2 = 1;
  index_space.extent3 = 1;
  index_space.field_stride1 = 1;
  index_space.coefficient_start = -1;
  index_space.coefficient2_start = -1;
  for (std::size_t point = 0; point < count; ++point) {
    const std::ptrdiff_t index =
        index_space.field_start + static_cast<std::ptrdiff_t>(point);
    const float transverse1 =
        transverse_count >= 1
            ? meep_cuda::detail::gyrotropic_offdiagonal_average_fp32(
                  field1.data(), index, field_stride, stride1)
            : 0.0f;
    const float transverse2 =
        transverse_count >= 2
            ? meep_cuda::detail::gyrotropic_offdiagonal_average_fp32(
                  field2.data(), index, field_stride, stride2)
            : 0.0f;
    meep_cuda::detail::apply_gyrotropic_update_fp32(
        reference_polarization0[index],
        reference_polarization1[index],
        reference_polarization2[index], reference_previous0[index],
        reference_previous1[index], reference_previous2[index],
        field0[index], transverse1, transverse2, sigma[index],
        coefficients);
  }

  auto device_polarization0 = copy_vector_to_device(polarization0);
  auto device_polarization1 = copy_vector_to_device(polarization1);
  auto device_polarization2 = copy_vector_to_device(polarization2);
  auto device_previous0 = copy_vector_to_device(previous0);
  auto device_previous1 = copy_vector_to_device(previous1);
  auto device_previous2 = copy_vector_to_device(previous2);
  auto device_field0 = copy_vector_to_device(field0);
  auto device_field1 = copy_vector_to_device(field1);
  auto device_field2 = copy_vector_to_device(field2);
  auto device_sigma = copy_vector_to_device(sigma);
  meep_cuda::gyrotropic_material_fp32 material = {
      static_cast<const float *>(device_sigma.get()),
      coefficients.model,
      {},
      {},
      coefficients.diagonal,
      coefficients.gamma_previous,
      coefficients.omega_dt_squared,
      coefficients.precession_half_dt,
      coefficients.omega_dt,
      coefficients.gamma_dt,
      coefficients.alpha_half,
      coefficients.drive_dt};
  for (int entry = 0; entry < 9; ++entry) {
    material.inverse[entry] = coefficients.inverse[entry];
    material.gyro[entry] = coefficients.gyro[entry];
  }
  meep_cuda::update_gyrotropic_structured_fp32(
      static_cast<float *>(device_polarization0.get()),
      static_cast<float *>(device_polarization1.get()),
      static_cast<float *>(device_polarization2.get()),
      static_cast<float *>(device_previous0.get()),
      static_cast<float *>(device_previous1.get()),
      static_cast<float *>(device_previous2.get()),
      static_cast<const float *>(device_field0.get()),
      transverse_count >= 1
          ? static_cast<const float *>(device_field1.get())
          : nullptr,
      transverse_count >= 2
          ? static_cast<const float *>(device_field2.get())
          : nullptr,
      index_space, count, field_stride, stride1, stride2, material);
  meep_cuda::synchronize();
  meep_cuda::copy_to_host(
      polarization0.data(), device_polarization0.get(),
      polarization0.size() * sizeof(float));
  meep_cuda::copy_to_host(
      polarization1.data(), device_polarization1.get(),
      polarization1.size() * sizeof(float));
  meep_cuda::copy_to_host(
      polarization2.data(), device_polarization2.get(),
      polarization2.size() * sizeof(float));
  meep_cuda::copy_to_host(
      previous0.data(), device_previous0.get(),
      previous0.size() * sizeof(float));
  meep_cuda::copy_to_host(
      previous1.data(), device_previous1.get(),
      previous1.size() * sizeof(float));
  meep_cuda::copy_to_host(
      previous2.data(), device_previous2.get(),
      previous2.size() * sizeof(float));

  float max_absolute_error = 0.0f;
  for (std::size_t index = 0; index < array_size; ++index) {
    max_absolute_error = std::max(
        max_absolute_error,
        std::abs(polarization0[index] -
                 reference_polarization0[index]));
    max_absolute_error = std::max(
        max_absolute_error,
        std::abs(polarization1[index] -
                 reference_polarization1[index]));
    max_absolute_error = std::max(
        max_absolute_error,
        std::abs(polarization2[index] -
                 reference_polarization2[index]));
    max_absolute_error = std::max(
        max_absolute_error,
        std::abs(previous0[index] - reference_previous0[index]));
    max_absolute_error = std::max(
        max_absolute_error,
        std::abs(previous1[index] - reference_previous1[index]));
    max_absolute_error = std::max(
        max_absolute_error,
        std::abs(previous2[index] - reference_previous2[index]));
  }
  std::cout << "gyrotropic_case model=" << model
            << " transverse=" << transverse_count
            << " negative_strides=" << negative_strides
            << " max_abs_error=" << max_absolute_error << '\n';
  return max_absolute_error <= 8.0e-6f;
}

bool run_multilevel_case() {
  constexpr std::size_t array_size = 1024;
  constexpr std::size_t level_count = 3;
  constexpr std::size_t transition_count = 2;
  constexpr float half_dt = 0.013f;

  std::mt19937 generator(20260829);
  std::uniform_real_distribution<float> distribution(-0.2f, 0.2f);
  std::vector<float> population(array_size * level_count);
  std::vector<float> population_scratch(array_size * level_count, -17.0f);
  std::vector<float> gamma = {
      0.07f, -0.02f, 0.01f,
      -0.03f, 0.06f, -0.01f,
      -0.04f, -0.04f, 0.05f};
  std::vector<float> gamma_inverse = {
      0.997f, 0.001f, -0.0004f,
      0.0015f, 0.996f, 0.0005f,
      0.002f, 0.0017f, 0.9975f};
  std::vector<float> alpha = {
      -0.11f, 0.0f,
      0.11f, -0.08f,
      0.0f, 0.08f};
  for (float &value : population) value = distribution(generator);

  std::vector<float> field0(array_size);
  std::vector<float> previous_field0(array_size);
  std::vector<float> field1(array_size);
  std::vector<float> previous_field1(array_size);
  std::vector<float> sigma0(array_size);
  std::vector<float> sigma1(array_size);
  for (std::size_t index = 0; index < array_size; ++index) {
    field0[index] = distribution(generator);
    previous_field0[index] = distribution(generator);
    field1[index] = distribution(generator);
    previous_field1[index] = distribution(generator);
    sigma0[index] = 0.2f + 0.0001f * static_cast<float>(index);
    sigma1[index] = 0.15f + 0.00007f * static_cast<float>(index);
  }

  const std::size_t polarization_size =
      2 * transition_count * array_size;
  std::vector<float> polarization0(polarization_size);
  std::vector<float> polarization1(polarization_size);
  for (float &value : polarization0) value = distribution(generator);
  for (float &value : polarization1) value = distribution(generator);

  meep_cuda::index_space_fp32 centered_space = {};
  centered_space.field_start = 8;
  centered_space.extent1 = 600;
  centered_space.extent2 = 1;
  centered_space.extent3 = 1;
  centered_space.field_stride1 = 1;
  centered_space.coefficient_start = -1;
  centered_space.coefficient2_start = -1;

  meep_cuda::index_space_fp32 polarization_space0 = {};
  polarization_space0.field_start = 20;
  polarization_space0.extent1 = 480;
  polarization_space0.extent2 = 1;
  polarization_space0.extent3 = 1;
  polarization_space0.field_stride1 = 1;
  polarization_space0.coefficient_start = -1;
  polarization_space0.coefficient2_start = -1;

  meep_cuda::index_space_fp32 polarization_space1 = {};
  polarization_space1.field_start = 30;
  polarization_space1.extent1 = 20;
  polarization_space1.extent2 = 17;
  polarization_space1.extent3 = 1;
  polarization_space1.field_stride1 = 25;
  polarization_space1.field_stride2 = 1;
  polarization_space1.coefficient_start = -1;
  polarization_space1.coefficient2_start = -1;
  constexpr std::size_t polarization_point_count1 = 20 * 17;

  std::vector<meep_cuda::multilevel_transition_fp32> transitions(2);
  transitions[0] =
      {1, 0, 0.018f, 1.91f, 0.96f, 0.98f,
       {0.013f, 0.017f, 0.021f, 0.025f, 0.029f}};
  transitions[1] =
      {2, 1, 0.023f, 1.87f, 0.94f, 0.97f,
       {0.011f, 0.015f, 0.019f, 0.023f, 0.027f}};

  std::vector<float> reference_population = population;
  std::vector<float> reference_scratch = population_scratch;
  std::vector<float> reference_polarization0 = polarization0;
  std::vector<float> reference_polarization1 = polarization1;

  struct host_population_channel {
    const std::vector<float> *field;
    const std::vector<float> *previous_field;
    const std::vector<float> *polarization;
    std::ptrdiff_t offset1;
    std::ptrdiff_t offset2;
  };
  const std::vector<host_population_channel> population_channels = {
      {&field0, &previous_field0, &reference_polarization0, 1, 4},
      {&field1, &previous_field1, &reference_polarization1, -2, 5}};

  for (std::size_t point = 0; point < centered_space.extent1; ++point) {
    const std::ptrdiff_t i =
        centered_space.field_start + static_cast<std::ptrdiff_t>(point);
    const std::size_t state =
        static_cast<std::size_t>(i) * level_count;
    for (std::size_t output_level = 0; output_level < level_count;
         ++output_level) {
      float value = 0.0f;
      for (std::size_t input_level = 0; input_level < level_count;
           ++input_level)
        value +=
            ((output_level == input_level ? 1.0f : 0.0f) -
             gamma[output_level * level_count + input_level] * half_dt) *
            reference_population[state + input_level];
      reference_scratch[state + output_level] = value;
    }
    for (std::size_t transition = 0; transition < transition_count;
         ++transition) {
      float field_delta_p_32 = 0.0f;
      float field_average_p_64 = 0.0f;
      for (const host_population_channel &channel : population_channels) {
        const std::ptrdiff_t indices[] = {
            i, i + channel.offset1, i + channel.offset2,
            i + channel.offset1 + channel.offset2};
        float field8 = 0.0f;
        float current4 = 0.0f;
        float previous4 = 0.0f;
        const std::size_t transition_offset =
            2 * transition * array_size;
        for (const std::ptrdiff_t index : indices) {
          field8 += (*channel.field)[index] +
                    (*channel.previous_field)[index];
          current4 +=
              (*channel.polarization)[transition_offset + index];
          previous4 +=
              (*channel.polarization)[transition_offset + array_size +
                                      index];
        }
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
        reference_scratch[state + level] +=
            alpha[level * transition_count + transition] * interaction;
    }
    for (std::size_t output_level = 0; output_level < level_count;
         ++output_level) {
      float value = 0.0f;
      for (std::size_t input_level = 0; input_level < level_count;
           ++input_level)
        value +=
            gamma_inverse[output_level * level_count + input_level] *
            reference_scratch[state + input_level];
      reference_population[state + output_level] = value;
    }
  }

  struct host_polarization_channel {
    std::vector<float> *polarization;
    const std::vector<float> *field;
    const std::vector<float> *sigma;
    meep_cuda::index_space_fp32 space;
    std::size_t point_count;
    std::ptrdiff_t offset1;
    std::ptrdiff_t offset2;
    int direction;
  };
  const std::vector<host_polarization_channel> polarization_channels = {
      {&reference_polarization0, &field0, &sigma0,
       polarization_space0, polarization_space0.extent1, -1, 3, 0},
      {&reference_polarization1, &field1, &sigma1,
       polarization_space1, polarization_point_count1, -3, 2, 2}};
  for (const host_polarization_channel &channel :
       polarization_channels)
    for (std::size_t transition = 0; transition < transition_count;
         ++transition)
      for (std::size_t point = 0; point < channel.point_count; ++point) {
        const std::ptrdiff_t i =
            meep_cuda::detail::decode_index_space_fp32(
                channel.space, point)
                .field;
        const std::ptrdiff_t indices[] = {
            i, i + channel.offset1, i + channel.offset2,
            i + channel.offset1 + channel.offset2};
        float inversion = 0.0f;
        for (const std::ptrdiff_t index : indices) {
          const std::size_t state =
              static_cast<std::size_t>(index) * level_count;
          inversion +=
              reference_population[
                  state + transitions[transition].upper_level] -
              reference_population[
                  state + transitions[transition].lower_level];
        }
        inversion *= 0.25f;
        const std::size_t current_index =
            2 * transition * array_size +
            static_cast<std::size_t>(i);
        const std::size_t previous_index =
            current_index + array_size;
        const float current = (*channel.polarization)[current_index];
        (*channel.polarization)[current_index] =
            transitions[transition].gamma_inverse *
            (current * transitions[transition].diagonal -
             transitions[transition].gamma_previous *
                 (*channel.polarization)[previous_index] -
             transitions[transition]
                     .drive_scale[channel.direction] *
                 (*channel.sigma)[i] * (*channel.field)[i] *
                 inversion);
        (*channel.polarization)[previous_index] = current;
      }

  auto device_population = copy_vector_to_device(population);
  auto device_population_scratch =
      copy_vector_to_device(population_scratch);
  auto device_gamma = copy_vector_to_device(gamma);
  auto device_gamma_inverse = copy_vector_to_device(gamma_inverse);
  auto device_alpha = copy_vector_to_device(alpha);
  auto device_field0 = copy_vector_to_device(field0);
  auto device_previous_field0 = copy_vector_to_device(previous_field0);
  auto device_field1 = copy_vector_to_device(field1);
  auto device_previous_field1 = copy_vector_to_device(previous_field1);
  auto device_sigma0 = copy_vector_to_device(sigma0);
  auto device_sigma1 = copy_vector_to_device(sigma1);
  auto device_polarization0 = copy_vector_to_device(polarization0);
  auto device_polarization1 = copy_vector_to_device(polarization1);
  std::vector<meep_cuda::multilevel_population_channel_fp32>
      device_population_channel_values = {
          {static_cast<const float *>(device_field0.get()),
           static_cast<const float *>(device_previous_field0.get()),
           static_cast<const float *>(device_polarization0.get()), 1, 4},
          {static_cast<const float *>(device_field1.get()),
           static_cast<const float *>(device_previous_field1.get()),
           static_cast<const float *>(device_polarization1.get()), -2, 5}};
  std::vector<meep_cuda::multilevel_polarization_channel_fp32>
      device_polarization_channel_values = {
          {static_cast<float *>(device_polarization0.get()),
           static_cast<const float *>(device_field0.get()),
           static_cast<const float *>(device_sigma0.get()),
           polarization_space0, polarization_space0.extent1, -1, 3, 0},
          {static_cast<float *>(device_polarization1.get()),
           static_cast<const float *>(device_field1.get()),
           static_cast<const float *>(device_sigma1.get()),
           polarization_space1, polarization_point_count1, -3, 2, 2}};
  auto device_population_channels =
      copy_vector_to_device(device_population_channel_values);
  auto device_polarization_channels =
      copy_vector_to_device(device_polarization_channel_values);
  auto device_transitions = copy_vector_to_device(transitions);

  meep_cuda::update_multilevel_structured_fp32(
      static_cast<float *>(device_population.get()),
      static_cast<float *>(device_population_scratch.get()),
      static_cast<const float *>(device_gamma.get()),
      static_cast<const float *>(device_gamma_inverse.get()),
      static_cast<const float *>(device_alpha.get()), level_count,
      transition_count, array_size, half_dt, centered_space,
      centered_space.extent1,
      static_cast<
          const meep_cuda::multilevel_population_channel_fp32 *>(
          device_population_channels.get()),
      device_population_channel_values.size(),
      static_cast<
          const meep_cuda::multilevel_polarization_channel_fp32 *>(
          device_polarization_channels.get()),
      device_polarization_channel_values.size(),
      static_cast<const meep_cuda::multilevel_transition_fp32 *>(
          device_transitions.get()),
      polarization_space0.extent1, 127);
  meep_cuda::synchronize();
  meep_cuda::copy_to_host(
      population.data(), device_population.get(),
      population.size() * sizeof(float));
  meep_cuda::copy_to_host(
      population_scratch.data(), device_population_scratch.get(),
      population_scratch.size() * sizeof(float));
  meep_cuda::copy_to_host(
      polarization0.data(), device_polarization0.get(),
      polarization0.size() * sizeof(float));
  meep_cuda::copy_to_host(
      polarization1.data(), device_polarization1.get(),
      polarization1.size() * sizeof(float));

  float maximum_error = 0.0f;
  for (std::size_t index = 0; index < population.size(); ++index) {
    maximum_error =
        std::max(maximum_error,
                 std::abs(population[index] -
                          reference_population[index]));
    maximum_error =
        std::max(maximum_error,
                 std::abs(population_scratch[index] -
                          reference_scratch[index]));
  }
  for (std::size_t index = 0; index < polarization_size; ++index) {
    maximum_error =
        std::max(maximum_error,
                 std::abs(polarization0[index] -
                          reference_polarization0[index]));
    maximum_error =
        std::max(maximum_error,
                 std::abs(polarization1[index] -
                          reference_polarization1[index]));
  }
  std::cout << "multilevel_case channels=2 transitions=2 levels=3"
            << " max_abs_error=" << maximum_error << '\n';
  return maximum_error <= 2.0e-5f;
}

bool run_boundary_case() {
  std::vector<float> real = {1.25f, -2.0f, 3.0f, 4.0f, -0.75f, 9.0f};
  std::vector<float> imag = {-0.5f, 0.25f, -1.0f, 2.0f, 1.5f, -7.0f};
  const std::vector<float> original_real = real;
  const std::vector<float> original_imag = imag;
  auto device_real = copy_vector_to_device(real);
  auto device_imag = copy_vector_to_device(imag);
  float *device_real_values = static_cast<float *>(device_real.get());
  float *device_imag_values = static_cast<float *>(device_imag.get());

  // The second operation reads real[1], which the first operation writes.
  // Correct gather/scatter semantics must retain the original real[1].
  const std::vector<meep_cuda::boundary_operation_fp32> operations = {
      {device_real_values + 1, nullptr, device_real_values + 0, nullptr,
       1.0f, 0.0f},
      {device_real_values + 2, nullptr, device_real_values + 1, nullptr,
       -1.0f, 0.0f},
      {device_real_values + 3, device_imag_values + 3,
       device_real_values + 4, device_imag_values + 4, 0.6f, 0.8f},
      {device_real_values + 5, device_imag_values + 5, nullptr, nullptr,
       0.0f, 0.0f}};
  auto device_operations = copy_vector_to_device(operations);
  std::vector<float> staging(2 * operations.size(), 0.0f);
  auto device_staging = copy_vector_to_device(staging);

  meep_cuda::apply_boundary_fp32(
      static_cast<const meep_cuda::boundary_operation_fp32 *>(
          device_operations.get()),
      static_cast<float *>(device_staging.get()), operations.size());
  meep_cuda::synchronize();
  meep_cuda::copy_to_host(real.data(), device_real.get(),
                          real.size() * sizeof(float));
  meep_cuda::copy_to_host(imag.data(), device_imag.get(),
                          imag.size() * sizeof(float));

  const float expected_complex_real =
      0.6f * original_real[4] - 0.8f * original_imag[4];
  const float expected_complex_imag =
      0.6f * original_imag[4] + 0.8f * original_real[4];
  const float tolerance = 2.0e-6f;
  return std::abs(real[1] - original_real[0]) <= tolerance &&
         std::abs(real[2] + original_real[1]) <= tolerance &&
         std::abs(real[3] - expected_complex_real) <= tolerance &&
         std::abs(imag[3] - expected_complex_imag) <= tolerance &&
         real[5] == 0.0f && imag[5] == 0.0f;
}

bool run_boundary_graph_case() {
  std::vector<float> real = {1.25f, -2.0f, 3.0f, 4.0f, -0.75f, 9.0f};
  std::vector<float> imag = {-0.5f, 0.25f, -1.0f, 2.0f, 1.5f, -7.0f};
  const std::vector<float> original_real = real;
  const std::vector<float> original_imag = imag;
  auto device_real = copy_vector_to_device(real);
  auto device_imag = copy_vector_to_device(imag);
  float *device_real_values = static_cast<float *>(device_real.get());
  float *device_imag_values = static_cast<float *>(device_imag.get());
  const std::vector<meep_cuda::boundary_operation_fp32> operations = {
      {device_real_values + 1, nullptr, device_real_values + 0, nullptr,
       1.0f, 0.0f},
      {device_real_values + 2, nullptr, device_real_values + 1, nullptr,
       -1.0f, 0.0f},
      {device_real_values + 3, device_imag_values + 3,
       device_real_values + 4, device_imag_values + 4, 0.6f, 0.8f},
      {device_real_values + 5, device_imag_values + 5, nullptr, nullptr,
       0.0f, 0.0f}};
  auto device_operations = copy_vector_to_device(operations);
  std::vector<float> staging(2 * operations.size(), 0.0f);
  auto device_staging = copy_vector_to_device(staging);
  meep_cuda::boundary_graph_fp32 *graph =
      meep_cuda::create_boundary_graph_fp32(
          static_cast<const meep_cuda::boundary_operation_fp32 *>(
              device_operations.get()),
          static_cast<float *>(device_staging.get()), operations.size());
  try {
    meep_cuda::launch_boundary_graph_fp32(graph);
    meep_cuda::synchronize();
  }
  catch (...) {
    meep_cuda::destroy_boundary_graph_fp32(graph);
    throw;
  }
  meep_cuda::destroy_boundary_graph_fp32(graph);
  meep_cuda::copy_to_host(real.data(), device_real.get(),
                          real.size() * sizeof(float));
  meep_cuda::copy_to_host(imag.data(), device_imag.get(),
                          imag.size() * sizeof(float));
  const float expected_complex_real =
      0.6f * original_real[4] - 0.8f * original_imag[4];
  const float expected_complex_imag =
      0.6f * original_imag[4] + 0.8f * original_real[4];
  const float tolerance = 2.0e-6f;
  return std::abs(real[1] - original_real[0]) <= tolerance &&
         std::abs(real[2] + original_real[1]) <= tolerance &&
         std::abs(real[3] - expected_complex_real) <= tolerance &&
         std::abs(imag[3] - expected_complex_imag) <= tolerance &&
         real[5] == 0.0f && imag[5] == 0.0f;
}

bool run_boundary_phase_graph_case() {
  std::vector<float> zero_values = {7.0f, -8.0f};
  std::vector<float> remote_source = {1.25f, -0.5f};
  std::vector<float> remote_destination(2, 9.0f);
  std::vector<float> remote_complex_real = {0.75f, -1.5f};
  std::vector<float> remote_complex_imag = {-0.25f, 2.0f};
  std::vector<float> remote_complex_destination_real(2, 9.0f);
  std::vector<float> remote_complex_destination_imag(2, -9.0f);
  std::vector<float> alias_values = {1.0f, 2.0f, 3.0f};
  std::vector<float> alias_complex_real = {1.0f, 2.0f, 3.0f};
  std::vector<float> alias_complex_imag = {-1.0f, 0.5f, 1.5f};
  auto device_zero = copy_vector_to_device(zero_values);
  auto device_remote_source = copy_vector_to_device(remote_source);
  auto device_remote_destination = copy_vector_to_device(remote_destination);
  auto device_remote_complex_real =
      copy_vector_to_device(remote_complex_real);
  auto device_remote_complex_imag =
      copy_vector_to_device(remote_complex_imag);
  auto device_remote_complex_destination_real =
      copy_vector_to_device(remote_complex_destination_real);
  auto device_remote_complex_destination_imag =
      copy_vector_to_device(remote_complex_destination_imag);
  auto device_alias = copy_vector_to_device(alias_values);
  auto device_alias_complex_real =
      copy_vector_to_device(alias_complex_real);
  auto device_alias_complex_imag =
      copy_vector_to_device(alias_complex_imag);
  float *zero = static_cast<float *>(device_zero.get());
  const float *remote_input =
      static_cast<const float *>(device_remote_source.get());
  float *remote_output =
      static_cast<float *>(device_remote_destination.get());
  const float *remote_complex_input_real =
      static_cast<const float *>(device_remote_complex_real.get());
  const float *remote_complex_input_imag =
      static_cast<const float *>(device_remote_complex_imag.get());
  float *remote_complex_output_real = static_cast<float *>(
      device_remote_complex_destination_real.get());
  float *remote_complex_output_imag = static_cast<float *>(
      device_remote_complex_destination_imag.get());
  float *alias = static_cast<float *>(device_alias.get());
  float *alias_complex_real_values =
      static_cast<float *>(device_alias_complex_real.get());
  float *alias_complex_imag_values =
      static_cast<float *>(device_alias_complex_imag.get());

  const std::vector<meep_cuda::boundary_operation_fp32> zero_operations = {
      {zero, nullptr, nullptr, nullptr, 0.0f, 0.0f},
      {zero + 1, nullptr, nullptr, nullptr, 0.0f, 0.0f}};
  const std::vector<meep_cuda::boundary_operation_fp32> gather_operations = {
      {remote_output, nullptr, remote_input, nullptr, 1.0f, 0.0f},
      {remote_output + 1, nullptr, remote_input + 1, nullptr, 1.0f, 0.0f},
      {remote_complex_output_real, remote_complex_output_imag,
       remote_complex_input_real, remote_complex_input_imag, 0.6f, 0.8f},
      {remote_complex_output_real + 1, remote_complex_output_imag + 1,
       remote_complex_input_real + 1, remote_complex_input_imag + 1,
       -0.8f, 0.6f}};
  // These operations intentionally alias: the second source is overwritten
  // by the first destination unless the graph takes a complete snapshot.
  const std::vector<meep_cuda::boundary_operation_fp32> ordered_operations = {
      {alias + 1, nullptr, alias, nullptr, 1.0f, 0.0f},
      {alias + 2, nullptr, alias + 1, nullptr, 1.0f, 0.0f},
      {alias_complex_real_values + 1, alias_complex_imag_values + 1,
       alias_complex_real_values, alias_complex_imag_values, 0.6f, 0.8f},
      {alias_complex_real_values + 2, alias_complex_imag_values + 2,
       alias_complex_real_values + 1, alias_complex_imag_values + 1,
       -0.8f, 0.6f}};
  auto device_zero_operations = copy_vector_to_device(zero_operations);
  auto device_gather_operations = copy_vector_to_device(gather_operations);
  auto device_ordered_operations = copy_vector_to_device(ordered_operations);
  std::vector<float> staging(2 * ordered_operations.size(), 0.0f);
  auto device_staging = copy_vector_to_device(staging);
  void *gather_complete = meep_cuda::create_event();
  void *phase_complete = meep_cuda::create_event();
  meep_cuda::boundary_phase_graph_fp32 *graph = nullptr;

  try {
    const meep_cuda::boundary_phase_stage_fp32 stages[] = {
        {meep_cuda::boundary_phase_stage_kind::zero,
         static_cast<const meep_cuda::boundary_operation_fp32 *>(
             device_zero_operations.get()),
         nullptr, zero_operations.size(), nullptr},
        {meep_cuda::boundary_phase_stage_kind::nonalias,
         static_cast<const meep_cuda::boundary_operation_fp32 *>(
             device_gather_operations.get()),
         nullptr, gather_operations.size(), gather_complete},
        {meep_cuda::boundary_phase_stage_kind::ordered,
         static_cast<const meep_cuda::boundary_operation_fp32 *>(
             device_ordered_operations.get()),
         static_cast<float *>(device_staging.get()),
         ordered_operations.size(), nullptr}};
    graph = meep_cuda::create_boundary_phase_graph_fp32(
        stages, sizeof(stages) / sizeof(stages[0]), phase_complete);

    // Launch twice so the graph-owned event record node is proven reusable.
    for (int recurrence = 0; recurrence < 2; ++recurrence) {
      if (recurrence) {
        zero_values = {11.0f, -12.0f};
        remote_source = {2.5f, 3.5f};
        remote_destination = {-4.0f, -5.0f};
        alias_values = {4.0f, 5.0f, 6.0f};
        remote_complex_real = {1.25f, -0.75f};
        remote_complex_imag = {0.5f, 1.5f};
        remote_complex_destination_real = {-4.0f, -5.0f};
        remote_complex_destination_imag = {4.0f, 5.0f};
        alias_complex_real = {4.0f, 5.0f, 6.0f};
        alias_complex_imag = {-2.0f, 3.0f, 1.0f};
        meep_cuda::copy_to_device(
            device_zero.get(), zero_values.data(),
            zero_values.size() * sizeof(float));
        meep_cuda::copy_to_device(
            device_remote_source.get(), remote_source.data(),
            remote_source.size() * sizeof(float));
        meep_cuda::copy_to_device(
            device_remote_destination.get(), remote_destination.data(),
            remote_destination.size() * sizeof(float));
        meep_cuda::copy_to_device(
            device_remote_complex_real.get(), remote_complex_real.data(),
            remote_complex_real.size() * sizeof(float));
        meep_cuda::copy_to_device(
            device_remote_complex_imag.get(), remote_complex_imag.data(),
            remote_complex_imag.size() * sizeof(float));
        meep_cuda::copy_to_device(
            device_remote_complex_destination_real.get(),
            remote_complex_destination_real.data(),
            remote_complex_destination_real.size() * sizeof(float));
        meep_cuda::copy_to_device(
            device_remote_complex_destination_imag.get(),
            remote_complex_destination_imag.data(),
            remote_complex_destination_imag.size() * sizeof(float));
        meep_cuda::copy_to_device(
            device_alias.get(), alias_values.data(),
            alias_values.size() * sizeof(float));
        meep_cuda::copy_to_device(
            device_alias_complex_real.get(), alias_complex_real.data(),
            alias_complex_real.size() * sizeof(float));
        meep_cuda::copy_to_device(
            device_alias_complex_imag.get(), alias_complex_imag.data(),
            alias_complex_imag.size() * sizeof(float));
      }
      meep_cuda::launch_boundary_phase_graph_fp32(graph);
      meep_cuda::synchronize_event(gather_complete);
      meep_cuda::synchronize_event(phase_complete);
    }
  }
  catch (...) {
    meep_cuda::destroy_boundary_phase_graph_fp32(graph);
    meep_cuda::destroy_event(gather_complete);
    meep_cuda::destroy_event(phase_complete);
    throw;
  }
  meep_cuda::destroy_boundary_phase_graph_fp32(graph);
  meep_cuda::destroy_event(gather_complete);
  meep_cuda::destroy_event(phase_complete);

  meep_cuda::copy_to_host(
      zero_values.data(), device_zero.get(),
      zero_values.size() * sizeof(float));
  meep_cuda::copy_to_host(
      remote_destination.data(), device_remote_destination.get(),
      remote_destination.size() * sizeof(float));
  meep_cuda::copy_to_host(
      remote_complex_destination_real.data(),
      device_remote_complex_destination_real.get(),
      remote_complex_destination_real.size() * sizeof(float));
  meep_cuda::copy_to_host(
      remote_complex_destination_imag.data(),
      device_remote_complex_destination_imag.get(),
      remote_complex_destination_imag.size() * sizeof(float));
  meep_cuda::copy_to_host(
      alias_values.data(), device_alias.get(),
      alias_values.size() * sizeof(float));
  meep_cuda::copy_to_host(
      alias_complex_real.data(), device_alias_complex_real.get(),
      alias_complex_real.size() * sizeof(float));
  meep_cuda::copy_to_host(
      alias_complex_imag.data(), device_alias_complex_imag.get(),
      alias_complex_imag.size() * sizeof(float));
  const float tolerance = 2.0e-6f;
  const auto close = [tolerance](float actual, float expected) {
    return std::abs(actual - expected) <= tolerance;
  };
  return zero_values[0] == 0.0f && zero_values[1] == 0.0f &&
         remote_destination == remote_source &&
         alias_values[0] == 4.0f && alias_values[1] == 4.0f &&
         alias_values[2] == 5.0f &&
         close(remote_complex_destination_real[0], 0.35f) &&
         close(remote_complex_destination_imag[0], 1.3f) &&
         close(remote_complex_destination_real[1], -0.3f) &&
         close(remote_complex_destination_imag[1], -1.65f) &&
         close(alias_complex_real[1], 4.0f) &&
         close(alias_complex_imag[1], 2.0f) &&
         close(alias_complex_real[2], -5.8f) &&
         close(alias_complex_imag[2], 0.6f);
}

bool run_zero_boundary_case() {
  std::vector<float> real = {1.0f, -2.0f, 3.0f, -4.0f};
  std::vector<float> imag = {0.5f, -0.25f, 0.75f, -1.25f};
  auto device_real = copy_vector_to_device(real);
  auto device_imag = copy_vector_to_device(imag);
  float *device_real_values = static_cast<float *>(device_real.get());
  float *device_imag_values = static_cast<float *>(device_imag.get());
  const std::vector<meep_cuda::boundary_operation_fp32> operations = {
      {device_real_values + 1, nullptr, nullptr, nullptr, 0.0f, 0.0f},
      {device_real_values + 3, device_imag_values + 3, nullptr, nullptr,
       0.0f, 0.0f}};
  auto device_operations = copy_vector_to_device(operations);
  meep_cuda::zero_boundary_fp32(
      static_cast<const meep_cuda::boundary_operation_fp32 *>(
          device_operations.get()),
      operations.size());
  meep_cuda::synchronize();
  meep_cuda::copy_to_host(real.data(), device_real.get(),
                          real.size() * sizeof(float));
  meep_cuda::copy_to_host(imag.data(), device_imag.get(),
                          imag.size() * sizeof(float));
  return real[0] == 1.0f && real[1] == 0.0f && real[2] == 3.0f &&
         real[3] == 0.0f && imag[0] == 0.5f && imag[1] == -0.25f &&
         imag[2] == 0.75f && imag[3] == 0.0f;
}

bool run_nonalias_boundary_case() {
  std::vector<float> source_real = {1.25f, -2.0f, 0.75f};
  std::vector<float> source_imag = {-0.5f, 0.25f, 1.5f};
  std::vector<float> destination_real(3, 9.0f);
  std::vector<float> destination_imag(3, -7.0f);
  auto device_source_real = copy_vector_to_device(source_real);
  auto device_source_imag = copy_vector_to_device(source_imag);
  auto device_destination_real = copy_vector_to_device(destination_real);
  auto device_destination_imag = copy_vector_to_device(destination_imag);
  const auto *sr = static_cast<const float *>(device_source_real.get());
  const auto *si = static_cast<const float *>(device_source_imag.get());
  auto *dr = static_cast<float *>(device_destination_real.get());
  auto *di = static_cast<float *>(device_destination_imag.get());
  const std::vector<meep_cuda::boundary_operation_fp32> operations = {
      {dr, nullptr, sr, nullptr, -1.0f, 0.0f},
      {dr + 1, di + 1, sr + 1, si + 1, 0.6f, 0.8f},
      {dr + 2, di + 2, nullptr, nullptr, 0.0f, 0.0f}};
  auto device_operations = copy_vector_to_device(operations);
  meep_cuda::apply_nonalias_boundary_fp32(
      static_cast<const meep_cuda::boundary_operation_fp32 *>(
          device_operations.get()),
      operations.size());
  meep_cuda::synchronize();
  meep_cuda::copy_to_host(destination_real.data(),
                          device_destination_real.get(),
                          destination_real.size() * sizeof(float));
  meep_cuda::copy_to_host(destination_imag.data(),
                          device_destination_imag.get(),
                          destination_imag.size() * sizeof(float));
  const float tolerance = 2.0e-6f;
  const float expected_real =
      0.6f * source_real[1] - 0.8f * source_imag[1];
  const float expected_imag =
      0.6f * source_imag[1] + 0.8f * source_real[1];
  return std::abs(destination_real[0] + source_real[0]) <= tolerance &&
         std::abs(destination_real[1] - expected_real) <= tolerance &&
         std::abs(destination_imag[1] - expected_imag) <= tolerance &&
         destination_real[2] == 0.0f && destination_imag[2] == 0.0f;
}

bool run_source_case() {
  std::vector<float> field = {
      0.2f, -0.1f, 0.7f, 0.4f, -0.3f, 0.9f, 0.1f, -0.2f};
  const std::vector<std::ptrdiff_t> indices = {1, 3, 6};
  const std::vector<meep_cuda::complex_value_fp32> amplitudes = {
      {0.5f, -0.25f}, {-0.3f, 0.8f}, {0.1f, 0.4f}};
  std::vector<float> conductivity_inverse(field.size());
  for (std::size_t index = 0; index < conductivity_inverse.size(); ++index)
    conductivity_inverse[index] = 0.7f + 0.03f * static_cast<float>(index);
  const meep_cuda::complex_value_fp32 time_scale = {0.4f, -0.7f};

  std::vector<float> reference = field;
  for (std::size_t point = 0; point < indices.size(); ++point) {
    const meep_cuda::complex_value_fp32 amplitude = amplitudes[point];
    const float real_value =
        amplitude.real * time_scale.real -
        amplitude.imag * time_scale.imag;
    reference[indices[point]] -=
        real_value * conductivity_inverse[indices[point]];
  }

  auto device_field = copy_vector_to_device(field);
  auto device_indices = copy_vector_to_device(indices);
  auto device_amplitudes = copy_vector_to_device(amplitudes);
  auto device_conductivity = copy_vector_to_device(conductivity_inverse);
  meep_cuda::indexed_source_subtract_fp32(
      static_cast<float *>(device_field.get()),
      static_cast<const std::ptrdiff_t *>(device_indices.get()),
      static_cast<const meep_cuda::complex_value_fp32 *>(
          device_amplitudes.get()),
      static_cast<const float *>(device_conductivity.get()), indices.size(),
      time_scale, false);
  meep_cuda::synchronize();
  meep_cuda::copy_to_host(field.data(), device_field.get(),
                          field.size() * sizeof(float));
  for (std::size_t index = 0; index < field.size(); ++index)
    if (std::abs(field[index] - reference[index]) > 2.0e-6f) return false;

  for (std::size_t point = 0; point < indices.size(); ++point) {
    const meep_cuda::complex_value_fp32 amplitude = amplitudes[point];
    const float imaginary_value =
        amplitude.real * time_scale.imag +
        amplitude.imag * time_scale.real;
    reference[indices[point]] -= imaginary_value;
  }
  meep_cuda::indexed_source_subtract_fp32(
      static_cast<float *>(device_field.get()),
      static_cast<const std::ptrdiff_t *>(device_indices.get()),
      static_cast<const meep_cuda::complex_value_fp32 *>(
          device_amplitudes.get()),
      nullptr, indices.size(), time_scale, true);
  meep_cuda::synchronize();
  meep_cuda::copy_to_host(field.data(), device_field.get(),
                          field.size() * sizeof(float));
  for (std::size_t index = 0; index < field.size(); ++index)
    if (std::abs(field[index] - reference[index]) > 2.0e-6f) return false;
  return true;
}

bool run_phase_batched_source_case() {
  constexpr std::size_t threads = 256;
  std::vector<float> first(600);
  std::vector<float> second(37);
  std::vector<float> third(800);
  std::vector<float> duplicate(4, 1.0f);
  for (std::size_t index = 0; index < first.size(); ++index)
    first[index] = 0.001f * static_cast<float>(index) - 0.2f;
  for (std::size_t index = 0; index < second.size(); ++index)
    second[index] = 0.03f * static_cast<float>(index) + 0.1f;
  for (std::size_t index = 0; index < third.size(); ++index)
    third[index] = -0.002f * static_cast<float>(index) + 0.7f;

  std::vector<std::ptrdiff_t> first_indices(300);
  std::vector<std::ptrdiff_t> second_indices(17);
  std::vector<std::ptrdiff_t> third_indices(513);
  std::vector<std::ptrdiff_t> duplicate_indices(513, 2);
  std::vector<meep_cuda::complex_value_fp32> first_amplitudes(300);
  std::vector<meep_cuda::complex_value_fp32> second_amplitudes(17);
  std::vector<meep_cuda::complex_value_fp32> third_amplitudes(513);
  std::vector<meep_cuda::complex_value_fp32> duplicate_amplitudes(
      duplicate_indices.size(), {1.0f / 1024.0f, 0.0f});
  for (std::size_t point = 0; point < first_indices.size(); ++point) {
    first_indices[point] = static_cast<std::ptrdiff_t>(2 * point);
    first_amplitudes[point] = {
        0.01f * static_cast<float>(point % 11) - 0.04f,
        0.015f * static_cast<float>(point % 7) - 0.03f};
  }
  for (std::size_t point = 0; point < second_indices.size(); ++point) {
    second_indices[point] = static_cast<std::ptrdiff_t>(2 * point + 1);
    second_amplitudes[point] = {
        0.02f * static_cast<float>(point) - 0.08f,
        -0.01f * static_cast<float>(point) + 0.05f};
  }
  for (std::size_t point = 0; point < third_indices.size(); ++point) {
    third_indices[point] = static_cast<std::ptrdiff_t>(point + 101);
    third_amplitudes[point] = {
        0.004f * static_cast<float>(point % 19) - 0.03f,
        0.006f * static_cast<float>(point % 13) - 0.025f};
  }
  std::vector<float> third_conductivity(third.size());
  for (std::size_t index = 0; index < third_conductivity.size(); ++index)
    third_conductivity[index] = 0.6f + 0.0003f * static_cast<float>(index);

  const meep_cuda::complex_value_fp32 first_scale = {0.4f, -0.3f};
  const meep_cuda::complex_value_fp32 second_scale = {-0.2f, 0.7f};
  const meep_cuda::complex_value_fp32 third_scale = {0.55f, 0.15f};
  const meep_cuda::complex_value_fp32 duplicate_scale = {1.0f, 0.0f};
  std::vector<float> first_reference = first;
  std::vector<float> second_reference = second;
  std::vector<float> third_reference = third;
  std::vector<float> duplicate_reference = duplicate;
  const auto apply_reference = [](
      std::vector<float> &field,
      const std::vector<std::ptrdiff_t> &indices,
      const std::vector<meep_cuda::complex_value_fp32> &amplitudes,
      const std::vector<float> *conductivity,
      meep_cuda::complex_value_fp32 scale, bool imaginary) {
    for (std::size_t point = 0; point < indices.size(); ++point) {
      const auto amplitude = amplitudes[point];
      float value = imaginary
                        ? amplitude.real * scale.imag +
                              amplitude.imag * scale.real
                        : amplitude.real * scale.real -
                              amplitude.imag * scale.imag;
      if (conductivity) value *= (*conductivity)[indices[point]];
      field[indices[point]] -= value;
    }
  };
  apply_reference(first_reference, first_indices, first_amplitudes, nullptr,
                  first_scale, false);
  apply_reference(second_reference, second_indices, second_amplitudes, nullptr,
                  second_scale, true);
  apply_reference(third_reference, third_indices, third_amplitudes,
                  &third_conductivity, third_scale, false);
  apply_reference(duplicate_reference, duplicate_indices,
                  duplicate_amplitudes, nullptr, duplicate_scale, false);

  auto device_first = copy_vector_to_device(first);
  auto device_second = copy_vector_to_device(second);
  auto device_third = copy_vector_to_device(third);
  auto device_duplicate = copy_vector_to_device(duplicate);
  auto device_first_indices = copy_vector_to_device(first_indices);
  auto device_second_indices = copy_vector_to_device(second_indices);
  auto device_third_indices = copy_vector_to_device(third_indices);
  auto device_duplicate_indices = copy_vector_to_device(duplicate_indices);
  auto device_first_amplitudes = copy_vector_to_device(first_amplitudes);
  auto device_second_amplitudes = copy_vector_to_device(second_amplitudes);
  auto device_third_amplitudes = copy_vector_to_device(third_amplitudes);
  auto device_duplicate_amplitudes =
      copy_vector_to_device(duplicate_amplitudes);
  auto device_third_conductivity = copy_vector_to_device(third_conductivity);
  const std::size_t first_blocks =
      (first_indices.size() + threads - 1) / threads;
  const std::size_t second_blocks =
      (second_indices.size() + threads - 1) / threads;
  const std::size_t third_blocks =
      (third_indices.size() + threads - 1) / threads;
  const std::size_t duplicate_blocks =
      (duplicate_indices.size() + threads - 1) / threads;
  const std::vector<meep_cuda::indexed_source_phase_operation_fp32>
      operations = {
          {static_cast<float *>(device_first.get()),
           static_cast<const std::ptrdiff_t *>(device_first_indices.get()),
           static_cast<const meep_cuda::complex_value_fp32 *>(
               device_first_amplitudes.get()),
           nullptr, first_indices.size(), 0, false},
          {static_cast<float *>(device_second.get()),
           static_cast<const std::ptrdiff_t *>(device_second_indices.get()),
           static_cast<const meep_cuda::complex_value_fp32 *>(
               device_second_amplitudes.get()),
           nullptr, second_indices.size(), first_blocks, true},
          {static_cast<float *>(device_third.get()),
           static_cast<const std::ptrdiff_t *>(device_third_indices.get()),
           static_cast<const meep_cuda::complex_value_fp32 *>(
               device_third_amplitudes.get()),
           static_cast<const float *>(device_third_conductivity.get()),
           third_indices.size(), first_blocks + second_blocks, false},
          {static_cast<float *>(device_duplicate.get()),
           static_cast<const std::ptrdiff_t *>(
               device_duplicate_indices.get()),
           static_cast<const meep_cuda::complex_value_fp32 *>(
               device_duplicate_amplitudes.get()),
           nullptr, duplicate_indices.size(),
           first_blocks + second_blocks + third_blocks, false}};
  std::vector<std::uint32_t> block_operation_indices(
      first_blocks, static_cast<std::uint32_t>(0));
  block_operation_indices.insert(
      block_operation_indices.end(), second_blocks,
      static_cast<std::uint32_t>(1));
  block_operation_indices.insert(
      block_operation_indices.end(), third_blocks,
      static_cast<std::uint32_t>(2));
  block_operation_indices.insert(
      block_operation_indices.end(), duplicate_blocks,
      static_cast<std::uint32_t>(3));
  auto device_operations = copy_vector_to_device(operations);
  auto device_block_operation_indices =
      copy_vector_to_device(block_operation_indices);
  meep_cuda::indexed_source_phase_time_scales_fp32 time_scales = {};
  time_scales.values[0] = first_scale;
  time_scales.values[1] = second_scale;
  time_scales.values[2] = third_scale;
  time_scales.values[3] = duplicate_scale;
  meep_cuda::indexed_source_subtract_phase_batched_fp32(
      static_cast<const meep_cuda::indexed_source_phase_operation_fp32 *>(
          device_operations.get()),
      static_cast<const std::uint32_t *>(
          device_block_operation_indices.get()),
      operations.size(),
      first_blocks + second_blocks + third_blocks + duplicate_blocks,
      time_scales);
  meep_cuda::synchronize();
  meep_cuda::copy_to_host(first.data(), device_first.get(),
                          first.size() * sizeof(float));
  meep_cuda::copy_to_host(second.data(), device_second.get(),
                          second.size() * sizeof(float));
  meep_cuda::copy_to_host(third.data(), device_third.get(),
                          third.size() * sizeof(float));
  meep_cuda::copy_to_host(duplicate.data(), device_duplicate.get(),
                          duplicate.size() * sizeof(float));
  const auto matches = [](const std::vector<float> &observed,
                          const std::vector<float> &expected) {
    for (std::size_t index = 0; index < observed.size(); ++index)
      if (std::abs(observed[index] - expected[index]) > 2.0e-6f)
        return false;
    return true;
  };
  return matches(first, first_reference) &&
         matches(second, second_reference) &&
         matches(third, third_reference) &&
         duplicate == duplicate_reference;
}

bool run_finite_case() {
  std::vector<float> first = {0.25f, -1.5f, 3.0f};
  std::vector<float> second = {-0.75f, 2.5f};
  auto device_first = copy_vector_to_device(first);
  auto device_second = copy_vector_to_device(second);
  const std::vector<meep_cuda::array_span_fp32> spans = {
      {static_cast<const float *>(device_first.get()), first.size()},
      {static_cast<const float *>(device_second.get()), second.size()}};
  auto device_spans = copy_vector_to_device(spans);
  device_pointer device_result(
      meep_cuda::allocate_device_bytes(sizeof(int)));
  auto *result = static_cast<int *>(device_result.get());
  const auto *device_span_pointer =
      static_cast<const meep_cuda::array_span_fp32 *>(device_spans.get());
  const std::size_t maximum_span_count =
      std::max(first.size(), second.size());

  const auto read_result = [&] {
    meep_cuda::synchronize();
    int host_result = -1;
    meep_cuda::copy_to_host(
        &host_result, device_result.get(), sizeof(host_result));
    return host_result;
  };

  meep_cuda::initialize_finite_result(result);
  meep_cuda::all_finite_fp32(
      device_span_pointer, spans.size(), maximum_span_count, result);
  if (read_result() != 1) return false;

  first[1] = std::numeric_limits<float>::quiet_NaN();
  meep_cuda::copy_to_device(
      device_first.get(), first.data(), first.size() * sizeof(float));
  meep_cuda::all_finite_fp32(
      device_span_pointer, 1, first.size(), result);
  if (read_result() != 0) return false;

  first[1] = -1.5f;
  second[0] = std::numeric_limits<float>::infinity();
  meep_cuda::copy_to_device(
      device_first.get(), first.data(), first.size() * sizeof(float));
  meep_cuda::copy_to_device(
      device_second.get(), second.data(), second.size() * sizeof(float));
  meep_cuda::all_finite_fp32(
      device_span_pointer + 1, 1, second.size(), result);
  if (read_result() != 0) return false;

  second[0] = -0.75f;
  meep_cuda::copy_to_device(
      device_second.get(), second.data(), second.size() * sizeof(float));
  meep_cuda::all_finite_fp32(
      device_span_pointer, spans.size(), maximum_span_count, result);
  if (read_result() != 0) return false;

  meep_cuda::initialize_finite_result(result);
  meep_cuda::all_finite_fp32(
      device_span_pointer, spans.size(), maximum_span_count, result);
  if (read_result() != 1) return false;

  auto *generation_result =
      static_cast<std::uint32_t *>(device_result.get());
  meep_cuda::clear_finite_generation_result(generation_result);
  meep_cuda::all_finite_generation_fp32(
      device_span_pointer, spans.size(), maximum_span_count, 1,
      generation_result);
  if (read_result() != 0) return false;
  first[1] = std::numeric_limits<float>::quiet_NaN();
  meep_cuda::copy_to_device(
      device_first.get(), first.data(), first.size() * sizeof(float));
  meep_cuda::all_finite_generation_fp32(
      device_span_pointer, spans.size(), maximum_span_count, 2,
      generation_result);
  if (read_result() != 2) return false;
  first[1] = -1.5f;
  meep_cuda::copy_to_device(
      device_first.get(), first.data(), first.size() * sizeof(float));
  meep_cuda::all_finite_generation_fp32(
      device_span_pointer, spans.size(), maximum_span_count, 3,
      generation_result);
  return read_result() == 2;
}

bool run_squared_norm_case() {
  const std::vector<float> values = {
      1.0f, -2.0f, 0.5f, 0.25f,
      -3.0f, 4.0f, 2.0f, -1.0f,
      0.125f, -0.75f, -1.5f, 0.625f};
  const std::vector<std::ptrdiff_t> indices = {2, 0};
  auto device_values = copy_vector_to_device(values);
  auto device_indices = copy_vector_to_device(indices);
  device_pointer device_result(
      meep_cuda::allocate_device_bytes(sizeof(double)));
  auto *result = static_cast<double *>(device_result.get());

  const auto read_result = [&] {
    meep_cuda::synchronize();
    double host_result = -1.0;
    meep_cuda::copy_to_host(
        &host_result, device_result.get(), sizeof(host_result));
    return host_result;
  };
  const auto reference = [&](const std::vector<std::ptrdiff_t> *selection) {
    double sum = 0.0;
    const std::size_t points = selection ? selection->size() : 3;
    for (std::size_t selected = 0; selected < points; ++selected) {
      const std::size_t point = selection
                                    ? static_cast<std::size_t>(
                                          (*selection)[selected])
                                    : selected;
      for (std::size_t frequency = 0; frequency < 2; ++frequency) {
        const std::size_t complex_index = point * 2 + frequency;
        const float real = values[2 * complex_index];
        const float imaginary = values[2 * complex_index + 1];
        const float magnitude_squared =
            real * real + imaginary * imaginary;
        sum += static_cast<double>(magnitude_squared);
      }
    }
    return sum;
  };

  meep_cuda::initialize_squared_norm_result(result);
  meep_cuda::squared_norm_complex_fp32(
      static_cast<const float *>(device_values.get()), nullptr, 3, 2,
      result, 37);
  if (std::abs(read_result() - reference(nullptr)) > 1e-12)
    return false;

  meep_cuda::initialize_squared_norm_result(result);
  meep_cuda::squared_norm_complex_fp32(
      static_cast<const float *>(device_values.get()),
      static_cast<const std::ptrdiff_t *>(device_indices.get()),
      indices.size(), 2, result, 31);
  if (std::abs(read_result() - reference(&indices)) > 1e-12)
    return false;

  // Exceed the implementation's 1024-block launch cap so every thread must
  // traverse the grid-stride loop.  Odd storage points contain deliberately
  // dominant padding values; the indexed reduction selects only even points,
  // making an ignored or misapplied persistent-point map unmistakable.
  constexpr std::size_t large_point_count = 300001;
  constexpr std::size_t large_frequency_count = 2;
  std::vector<float> large_values(
      2 * large_point_count * large_frequency_count);
  std::vector<std::ptrdiff_t> large_indices;
  large_indices.reserve((large_point_count + 1) / 2);
  for (std::size_t point = 0; point < large_point_count; ++point) {
    if ((point & 1u) == 0) large_indices.push_back(
        static_cast<std::ptrdiff_t>(point));
    for (std::size_t frequency = 0;
         frequency < large_frequency_count; ++frequency) {
      const std::size_t complex_index =
          point * large_frequency_count + frequency;
      if ((point & 1u) == 0) {
        large_values[2 * complex_index] =
            static_cast<float>(
                static_cast<int>((complex_index * 17) % 251) - 125) *
            0.001f;
        large_values[2 * complex_index + 1] =
            static_cast<float>(
                static_cast<int>((complex_index * 29) % 239) - 119) *
            0.001f;
      }
      else {
        large_values[2 * complex_index] = 1000.0f;
        large_values[2 * complex_index + 1] = -750.0f;
      }
    }
  }
  double large_reference = 0.0;
  for (std::ptrdiff_t raw_point : large_indices) {
    const std::size_t point = static_cast<std::size_t>(raw_point);
    for (std::size_t frequency = 0;
         frequency < large_frequency_count; ++frequency) {
      const std::size_t complex_index =
          point * large_frequency_count + frequency;
      const float real = large_values[2 * complex_index];
      const float imaginary = large_values[2 * complex_index + 1];
      const float magnitude_squared =
          real * real + imaginary * imaginary;
      large_reference += static_cast<double>(magnitude_squared);
    }
  }
  auto device_large_values = copy_vector_to_device(large_values);
  auto device_large_indices = copy_vector_to_device(large_indices);
  meep_cuda::initialize_squared_norm_result(result);
  meep_cuda::squared_norm_complex_fp32(
      static_cast<const float *>(device_large_values.get()),
      static_cast<const std::ptrdiff_t *>(device_large_indices.get()),
      large_indices.size(), large_frequency_count, result, 53);
  const double large_observed = read_result();
  return std::abs(large_observed - large_reference) <=
         1e-12 * std::max(1.0, std::abs(large_reference));
}

bool run_indexed_ldos_case() {
  constexpr std::size_t threads = 256;
  std::vector<float> first_real(701);
  std::vector<float> first_imaginary(701);
  std::vector<float> second_real(41);
  std::vector<float> third_real(811);
  std::vector<float> third_imaginary(811);
  for (std::size_t index = 0; index < first_real.size(); ++index) {
    first_real[index] =
        static_cast<float>(static_cast<int>((index * 17) % 211) - 105) *
        0.003f;
    first_imaginary[index] =
        static_cast<float>(static_cast<int>((index * 29) % 197) - 98) *
        0.002f;
  }
  for (std::size_t index = 0; index < second_real.size(); ++index)
    second_real[index] =
        static_cast<float>(static_cast<int>((index * 13) % 37) - 18) *
        0.017f;
  for (std::size_t index = 0; index < third_real.size(); ++index) {
    third_real[index] =
        static_cast<float>(static_cast<int>((index * 31) % 257) - 128) *
        0.0015f;
    third_imaginary[index] =
        static_cast<float>(static_cast<int>((index * 43) % 241) - 120) *
        0.00125f;
  }

  std::vector<std::ptrdiff_t> first_indices(513);
  std::vector<std::ptrdiff_t> second_indices(17);
  // More than 1,024 logical blocks exercises the capped physical-grid stride
  // in the deterministic reduction workspace.
  std::vector<std::ptrdiff_t> third_indices(256 * 1024 + 1);
  std::vector<meep_cuda::complex_value_fp32> first_amplitudes(513);
  std::vector<meep_cuda::complex_value_fp32> second_amplitudes(17);
  std::vector<meep_cuda::complex_value_fp32> third_amplitudes(
      third_indices.size());
  for (std::size_t point = 0; point < first_indices.size(); ++point) {
    first_indices[point] = static_cast<std::ptrdiff_t>((point * 37) % 701);
    first_amplitudes[point] = {
        static_cast<float>(static_cast<int>((point * 7) % 67) - 33) *
            0.004f,
        static_cast<float>(static_cast<int>((point * 11) % 71) - 35) *
            0.003f};
  }
  for (std::size_t point = 0; point < second_indices.size(); ++point) {
    second_indices[point] = static_cast<std::ptrdiff_t>(2 * point + 1);
    second_amplitudes[point] = {
        static_cast<float>(static_cast<int>(point) - 8) * 0.021f,
        static_cast<float>(static_cast<int>((point * 5) % 19) - 9) *
            0.013f};
  }
  for (std::size_t point = 0; point < third_indices.size(); ++point) {
    third_indices[point] = static_cast<std::ptrdiff_t>((point * 53) % 811);
    third_amplitudes[point] = {
        static_cast<float>(static_cast<int>((point * 13) % 83) - 41) *
            0.0025f,
        static_cast<float>(static_cast<int>((point * 17) % 79) - 39) *
            0.002f};
  }

  std::array<double, 4> reference = {};
  const auto accumulate_reference = [&reference](
      const std::vector<float> &field_real,
      const std::vector<float> *field_imaginary,
      const std::vector<std::ptrdiff_t> &indices,
      const std::vector<meep_cuda::complex_value_fp32> &amplitudes,
      bool magnetic) {
    const std::size_t offset = magnetic ? 2 : 0;
    for (std::size_t point = 0; point < indices.size(); ++point) {
      const std::size_t index = static_cast<std::size_t>(indices[point]);
      const volatile float product_rr =
          field_real[index] * amplitudes[point].real;
      const volatile float product_ii =
          (field_imaginary ? (*field_imaginary)[index] : 0.0f) *
          amplitudes[point].imag;
      const volatile float product_ir =
          (field_imaginary ? (*field_imaginary)[index] : 0.0f) *
          amplitudes[point].real;
      const volatile float product_ri =
          field_real[index] * amplitudes[point].imag;
      const volatile float contribution_real = product_rr + product_ii;
      const volatile float contribution_imaginary = product_ir - product_ri;
      reference[offset] += static_cast<double>(contribution_real);
      reference[offset + 1] +=
          static_cast<double>(contribution_imaginary);
    }
  };
  accumulate_reference(
      first_real, &first_imaginary, first_indices, first_amplitudes, false);
  accumulate_reference(
      second_real, nullptr, second_indices, second_amplitudes, true);
  accumulate_reference(
      third_real, &third_imaginary, third_indices, third_amplitudes, false);

  auto device_first_real = copy_vector_to_device(first_real);
  auto device_first_imaginary = copy_vector_to_device(first_imaginary);
  auto device_second_real = copy_vector_to_device(second_real);
  auto device_third_real = copy_vector_to_device(third_real);
  auto device_third_imaginary = copy_vector_to_device(third_imaginary);
  auto device_first_indices = copy_vector_to_device(first_indices);
  auto device_second_indices = copy_vector_to_device(second_indices);
  auto device_third_indices = copy_vector_to_device(third_indices);
  auto device_first_amplitudes = copy_vector_to_device(first_amplitudes);
  auto device_second_amplitudes = copy_vector_to_device(second_amplitudes);
  auto device_third_amplitudes = copy_vector_to_device(third_amplitudes);

  const std::size_t first_blocks =
      (first_indices.size() + threads - 1) / threads;
  const std::size_t second_blocks =
      (second_indices.size() + threads - 1) / threads;
  const std::size_t third_blocks =
      (third_indices.size() + threads - 1) / threads;
  const std::vector<meep_cuda::indexed_ldos_operation_fp32> operations = {
      {static_cast<const float *>(device_first_real.get()),
       static_cast<const float *>(device_first_imaginary.get()),
       static_cast<const std::ptrdiff_t *>(device_first_indices.get()),
       static_cast<const meep_cuda::complex_value_fp32 *>(
           device_first_amplitudes.get()),
       first_indices.size(), 0, false},
      {static_cast<const float *>(device_second_real.get()), nullptr,
       static_cast<const std::ptrdiff_t *>(device_second_indices.get()),
       static_cast<const meep_cuda::complex_value_fp32 *>(
           device_second_amplitudes.get()),
       second_indices.size(), first_blocks, true},
      {static_cast<const float *>(device_third_real.get()),
       static_cast<const float *>(device_third_imaginary.get()),
       static_cast<const std::ptrdiff_t *>(device_third_indices.get()),
       static_cast<const meep_cuda::complex_value_fp32 *>(
           device_third_amplitudes.get()),
       third_indices.size(), first_blocks + second_blocks, false}};
  auto device_operations = copy_vector_to_device(operations);
  const std::size_t total_blocks =
      first_blocks + second_blocks + third_blocks;
  std::vector<std::uint32_t> block_operation_indices(total_blocks);
  for (std::size_t block = 0; block < total_blocks; ++block)
    block_operation_indices[block] =
        block < first_blocks
            ? 0u
            : block < first_blocks + second_blocks ? 1u : 2u;
  auto device_block_operation_indices =
      copy_vector_to_device(block_operation_indices);
  const std::array<double, 4> poison = {1.0, -2.0, 3.0, -4.0};
  std::array<double, 4> observed = poison;
  device_pointer device_partials(meep_cuda::allocate_device_bytes(
      4 * meep_cuda::indexed_ldos_reduction_partial_capacity *
      sizeof(double)));
  device_pointer device_result(
      meep_cuda::allocate_device_bytes(observed.size() * sizeof(double)));
  meep_cuda::copy_to_device(
      device_result.get(), poison.data(), poison.size() * sizeof(double));
  meep_cuda::initialize_indexed_ldos_result(
      static_cast<double *>(device_result.get()));
  meep_cuda::synchronize();
  meep_cuda::copy_to_host(
      observed.data(), device_result.get(), observed.size() * sizeof(double));
  if (observed != std::array<double, 4>{}) return false;

  const auto *device_operation_pointer =
      static_cast<const meep_cuda::indexed_ldos_operation_fp32 *>(
          device_operations.get());
  const auto *device_block_map_pointer =
      static_cast<const std::uint32_t *>(
          device_block_operation_indices.get());
  std::array<double, 4> deterministic_reference = {};
  for (std::size_t repetition = 0; repetition < 32; ++repetition) {
    meep_cuda::indexed_ldos_reduce_fp32(
        device_operation_pointer, device_block_map_pointer,
        operations.size(), total_blocks,
        static_cast<double *>(device_partials.get()),
        meep_cuda::indexed_ldos_reduction_partial_capacity,
        static_cast<double *>(device_result.get()),
        meep_cuda::indexed_ldos_result_mode::replace);
    meep_cuda::synchronize();
    meep_cuda::copy_to_host(
        observed.data(), device_result.get(),
        observed.size() * sizeof(double));
    for (std::size_t index = 0; index < observed.size(); ++index)
      if (std::abs(observed[index] - reference[index]) >
          2e-12 * std::max(1.0, std::abs(reference[index])))
        return false;
    if (repetition == 0)
      deterministic_reference = observed;
    else if (std::memcmp(
                 observed.data(), deterministic_reference.data(),
                 observed.size() * sizeof(double)) != 0)
      return false;
  }

  // The reduction contract is additive until explicitly initialized.
  meep_cuda::initialize_indexed_ldos_result(
      static_cast<double *>(device_result.get()));
  meep_cuda::indexed_ldos_reduce_fp32(
      device_operation_pointer, device_block_map_pointer,
      operations.size(), total_blocks,
      static_cast<double *>(device_partials.get()),
      meep_cuda::indexed_ldos_reduction_partial_capacity,
      static_cast<double *>(device_result.get()),
      meep_cuda::indexed_ldos_result_mode::accumulate);
  meep_cuda::indexed_ldos_reduce_fp32(
      device_operation_pointer, device_block_map_pointer,
      operations.size(), total_blocks,
      static_cast<double *>(device_partials.get()),
      meep_cuda::indexed_ldos_reduction_partial_capacity,
      static_cast<double *>(device_result.get()),
      meep_cuda::indexed_ldos_result_mode::accumulate);
  meep_cuda::synchronize();
  meep_cuda::copy_to_host(
      observed.data(), device_result.get(), observed.size() * sizeof(double));
  for (std::size_t index = 0; index < observed.size(); ++index)
    if (std::abs(observed[index] - 2.0 * reference[index]) >
        4e-12 * std::max(1.0, std::abs(reference[index])))
      return false;
  return true;
}

bool run_dft_case() {
  std::vector<float> field_real = {
      0.2f, -0.1f, 0.7f, 0.4f, -0.3f, 0.9f, 0.1f, -0.2f};
  std::vector<float> field_imag = {
      -0.5f, 0.3f, 0.2f, -0.4f, 0.8f, -0.6f, 0.1f, 0.7f};
  const std::vector<std::ptrdiff_t> indices = {0, 2, 4};
  const std::vector<float> weights = {0.5f, 1.25f, -0.75f};
  const std::vector<meep_cuda::complex_value_fp32> phases = {
      {0.8f, 0.6f}, {-0.3f, 0.4f}};
  constexpr std::ptrdiff_t average1 = 1;
  constexpr std::ptrdiff_t average2 = 2;
  std::vector<float> output(2 * indices.size() * phases.size(), 0.0f);
  std::vector<float> reference = output;
  for (std::size_t point = 0; point < indices.size(); ++point) {
    const std::ptrdiff_t index = indices[point];
    const float real_value =
        weights[point] * 0.25f *
        (field_real[index] + field_real[index + average1] +
         field_real[index + average2] +
         field_real[index + average1 + average2]);
    const float imag_value =
        weights[point] * 0.25f *
        (field_imag[index] + field_imag[index + average1] +
         field_imag[index + average2] +
         field_imag[index + average1 + average2]);
    for (std::size_t frequency = 0; frequency < phases.size(); ++frequency) {
      const std::size_t destination =
          2 * (point * phases.size() + frequency);
      reference[destination] +=
          phases[frequency].real * real_value -
          phases[frequency].imag * imag_value;
      reference[destination + 1] +=
          phases[frequency].real * imag_value +
          phases[frequency].imag * real_value;
    }
  }

  auto device_output = copy_vector_to_device(output);
  auto device_real = copy_vector_to_device(field_real);
  auto device_imag = copy_vector_to_device(field_imag);
  auto device_indices = copy_vector_to_device(indices);
  auto device_weights = copy_vector_to_device(weights);
  auto device_phases = copy_vector_to_device(phases);
  meep_cuda::update_dft_fp32(
      static_cast<float *>(device_output.get()),
      static_cast<const float *>(device_real.get()),
      static_cast<const float *>(device_imag.get()),
      static_cast<const std::ptrdiff_t *>(device_indices.get()),
      static_cast<const float *>(device_weights.get()), indices.size(),
      static_cast<const meep_cuda::complex_value_fp32 *>(
          device_phases.get()),
      phases.size(), average1, average2);
  meep_cuda::synchronize();
  meep_cuda::copy_to_host(output.data(), device_output.get(),
                          output.size() * sizeof(float));
  float max_error = 0.0f;
  for (std::size_t index = 0; index < output.size(); ++index)
    max_error =
        std::max(max_error, std::abs(output[index] - reference[index]));
  return max_error <= 3.0e-6f;
}

bool run_dft_wide_frequency_case() {
  const std::vector<float> field_real = {
      0.2f, -0.1f, 0.7f, 0.4f, -0.3f, 0.9f, 0.1f, -0.2f};
  const std::vector<float> field_imag = {
      -0.5f, 0.3f, 0.2f, -0.4f, 0.8f, -0.6f, 0.1f, 0.7f};
  const std::vector<std::ptrdiff_t> indices = {2};
  const std::vector<float> weights = {-0.625f};
  std::vector<meep_cuda::complex_value_fp32> phases(513);
  for (std::size_t frequency = 0; frequency < phases.size(); ++frequency) {
    const double angle = -1.7 + 0.019 * static_cast<double>(frequency);
    phases[frequency] = {
        static_cast<float>(0.73 * std::cos(angle)),
        static_cast<float>(0.73 * std::sin(angle))};
  }

  constexpr std::ptrdiff_t average1 = 1;
  constexpr std::ptrdiff_t average2 = 2;
  constexpr std::size_t guard_count = 4;
  const std::size_t scalar_count = 2 * phases.size();
  constexpr float canary = 12345.5f;
  std::vector<float> storage(
      guard_count + scalar_count + guard_count, canary);
  std::vector<float> reference(scalar_count);
  for (std::size_t scalar = 0; scalar < scalar_count; ++scalar) {
    const float initial =
        0.01f * static_cast<float>(static_cast<int>(scalar % 17) - 8);
    storage[guard_count + scalar] = initial;
    reference[scalar] = initial;
  }

  const std::ptrdiff_t index = indices[0];
  const float real_value =
      weights[0] * 0.25f *
      (field_real[index] + field_real[index + average1] +
       field_real[index + average2] +
       field_real[index + average1 + average2]);
  const float imag_value =
      weights[0] * 0.25f *
      (field_imag[index] + field_imag[index + average1] +
       field_imag[index + average2] +
       field_imag[index + average1 + average2]);
  for (int accumulation = 0; accumulation < 2; ++accumulation)
    for (std::size_t frequency = 0; frequency < phases.size(); ++frequency) {
      const std::size_t output = 2 * frequency;
      reference[output] += phases[frequency].real * real_value -
                           phases[frequency].imag * imag_value;
      reference[output + 1] += phases[frequency].real * imag_value +
                               phases[frequency].imag * real_value;
    }

  auto device_storage = copy_vector_to_device(storage);
  auto device_real = copy_vector_to_device(field_real);
  auto device_imag = copy_vector_to_device(field_imag);
  auto device_indices = copy_vector_to_device(indices);
  auto device_weights = copy_vector_to_device(weights);
  auto device_phases = copy_vector_to_device(phases);
  for (int accumulation = 0; accumulation < 2; ++accumulation)
    meep_cuda::update_dft_fp32(
        static_cast<float *>(device_storage.get()) + guard_count,
        static_cast<const float *>(device_real.get()),
        static_cast<const float *>(device_imag.get()),
        static_cast<const std::ptrdiff_t *>(device_indices.get()),
        static_cast<const float *>(device_weights.get()), indices.size(),
        static_cast<const meep_cuda::complex_value_fp32 *>(
            device_phases.get()),
        phases.size(), average1, average2, 64);

  meep_cuda::synchronize();
  meep_cuda::copy_to_host(storage.data(), device_storage.get(),
                          storage.size() * sizeof(float));
  for (std::size_t guard = 0; guard < guard_count; ++guard)
    if (storage[guard] != canary ||
        storage[guard_count + scalar_count + guard] != canary)
      return false;
  for (std::size_t scalar = 0; scalar < scalar_count; ++scalar)
    if (std::abs(storage[guard_count + scalar] - reference[scalar]) >
        4.0e-6f)
      return false;
  return true;
}

bool run_dft_many_point_case() {
  constexpr std::size_t point_count = 513;
  constexpr std::size_t field_count = 1024;
  constexpr std::ptrdiff_t average1 = -1;
  constexpr std::ptrdiff_t average2 = -2;
  constexpr std::size_t guard_count = 4;
  constexpr float canary = -23456.25f;
  const meep_cuda::complex_value_fp32 phase = {-0.37f, 0.81f};

  std::vector<float> field_real(field_count);
  std::vector<float> field_imag(field_count);
  for (std::size_t index = 0; index < field_count; ++index) {
    field_real[index] =
        static_cast<float>(std::sin(0.017 * static_cast<double>(index)));
    field_imag[index] =
        static_cast<float>(std::cos(0.023 * static_cast<double>(index)));
  }
  std::vector<std::ptrdiff_t> indices(point_count);
  std::vector<float> weights(point_count);
  for (std::size_t point = 0; point < point_count; ++point) {
    indices[point] =
        static_cast<std::ptrdiff_t>(3 + (7 * point) % (field_count - 3));
    if (point > 0 && point % 31 == 0)
      indices[point] = indices[point - 1];
    weights[point] =
        point % 11 == 0
            ? 0.0f
            : 0.01f *
                  static_cast<float>(static_cast<int>(point % 37) - 18);
  }

  const std::size_t scalar_count = 2 * point_count;
  std::vector<float> storage(
      guard_count + scalar_count + guard_count, canary);
  std::vector<float> reference(scalar_count);
  for (std::size_t point = 0; point < point_count; ++point) {
    const float initial_real =
        0.001f * static_cast<float>(static_cast<int>(point % 13) - 6);
    const float initial_imag =
        0.001f * static_cast<float>(static_cast<int>(point % 9) - 4);
    storage[guard_count + 2 * point] = initial_real;
    storage[guard_count + 2 * point + 1] = initial_imag;
    const std::ptrdiff_t index = indices[point];
    const float real_value =
        weights[point] * 0.25f *
        (field_real[index] + field_real[index + average1] +
         field_real[index + average2] +
         field_real[index + average1 + average2]);
    const float imag_value =
        weights[point] * 0.25f *
        (field_imag[index] + field_imag[index + average1] +
         field_imag[index + average2] +
         field_imag[index + average1 + average2]);
    reference[2 * point] =
        initial_real + phase.real * real_value - phase.imag * imag_value;
    reference[2 * point + 1] =
        initial_imag + phase.real * imag_value + phase.imag * real_value;
  }

  const std::vector<meep_cuda::complex_value_fp32> phases = {phase};
  auto device_storage = copy_vector_to_device(storage);
  auto device_real = copy_vector_to_device(field_real);
  auto device_imag = copy_vector_to_device(field_imag);
  auto device_indices = copy_vector_to_device(indices);
  auto device_weights = copy_vector_to_device(weights);
  auto device_phases = copy_vector_to_device(phases);
  meep_cuda::update_dft_fp32(
      static_cast<float *>(device_storage.get()) + guard_count,
      static_cast<const float *>(device_real.get()),
      static_cast<const float *>(device_imag.get()),
      static_cast<const std::ptrdiff_t *>(device_indices.get()),
      static_cast<const float *>(device_weights.get()), indices.size(),
      static_cast<const meep_cuda::complex_value_fp32 *>(
          device_phases.get()),
      phases.size(), average1, average2, 64);

  meep_cuda::synchronize();
  meep_cuda::copy_to_host(storage.data(), device_storage.get(),
                          storage.size() * sizeof(float));
  for (std::size_t guard = 0; guard < guard_count; ++guard)
    if (storage[guard] != canary ||
        storage[guard_count + scalar_count + guard] != canary)
      return false;
  for (std::size_t scalar = 0; scalar < scalar_count; ++scalar)
    if (std::abs(storage[guard_count + scalar] - reference[scalar]) >
        4.0e-6f)
      return false;
  return true;
}

bool run_dft_omega_case() {
  const std::vector<float> field_real = {
      0.2f, -0.1f, 0.7f, 0.4f, -0.3f, 0.9f, 0.1f, -0.2f};
  const std::vector<float> field_imag = {
      -0.5f, 0.3f, 0.2f, -0.4f, 0.8f, -0.6f, 0.1f, 0.7f};
  const std::vector<std::ptrdiff_t> indices = {0, 2, 4};
  const std::vector<float> weights = {0.5f, 1.25f, -0.75f};
  std::vector<double> angular_frequencies(513);
  for (std::size_t frequency = 0;
       frequency < angular_frequencies.size(); ++frequency)
    angular_frequencies[frequency] =
        -2.75 + 0.013 * static_cast<double>(frequency);
  constexpr double scale_real = -0.42;
  constexpr double scale_imag = 0.31;

  struct omega_case {
    bool complex_field;
    std::ptrdiff_t average1;
    std::ptrdiff_t average2;
  };
  const omega_case cases[] = {
      {false, 0, 0}, {true, 1, 0}, {true, 1, 2}};
  const double times[] = {-0.37, 0.125, 1234.5};

  auto device_real = copy_vector_to_device(field_real);
  auto device_imag = copy_vector_to_device(field_imag);
  auto device_indices = copy_vector_to_device(indices);
  auto device_weights = copy_vector_to_device(weights);
  auto device_frequencies = copy_vector_to_device(angular_frequencies);

  for (const omega_case &test : cases) {
    std::vector<float> output(
        2 * indices.size() * angular_frequencies.size(), 0.0f);
    std::vector<float> reference = output;
    auto device_output = copy_vector_to_device(output);
    std::vector<meep_cuda::complex_value_fp32> phase_scratch(
        angular_frequencies.size(), {0.0f, 0.0f});
    auto device_phase_scratch = copy_vector_to_device(phase_scratch);

    for (double time : times) {
      for (std::size_t point = 0; point < indices.size(); ++point) {
        const std::ptrdiff_t index = indices[point];
        float real_value;
        float imag_value = 0.0f;
        if (test.average2) {
          real_value =
              0.25f * (field_real[index] +
                       field_real[index + test.average1] +
                       field_real[index + test.average2] +
                       field_real[index + test.average1 + test.average2]);
          if (test.complex_field)
            imag_value =
                0.25f * (field_imag[index] +
                         field_imag[index + test.average1] +
                         field_imag[index + test.average2] +
                         field_imag[index + test.average1 + test.average2]);
        }
        else if (test.average1) {
          real_value =
              0.5f * (field_real[index] +
                      field_real[index + test.average1]);
          if (test.complex_field)
            imag_value =
                0.5f * (field_imag[index] +
                        field_imag[index + test.average1]);
        }
        else {
          real_value = field_real[index];
          if (test.complex_field) imag_value = field_imag[index];
        }
        real_value *= weights[point];
        imag_value *= weights[point];

        for (std::size_t frequency = 0;
             frequency < angular_frequencies.size(); ++frequency) {
          const double angle = angular_frequencies[frequency] * time;
          const double cosine = std::cos(angle);
          const double sine = std::sin(angle);
          const float phase_real = static_cast<float>(
              cosine * scale_real - sine * scale_imag);
          const float phase_imag = static_cast<float>(
              cosine * scale_imag + sine * scale_real);
          const std::size_t destination =
              2 * (point * angular_frequencies.size() + frequency);
          reference[destination] +=
              phase_real * real_value - phase_imag * imag_value;
          reference[destination + 1] +=
              phase_real * imag_value + phase_imag * real_value;
        }
      }

      meep_cuda::update_dft_from_omega_fp32(
          static_cast<float *>(device_output.get()),
          static_cast<const float *>(device_real.get()),
          test.complex_field
              ? static_cast<const float *>(device_imag.get())
              : nullptr,
          static_cast<const std::ptrdiff_t *>(device_indices.get()),
          static_cast<const float *>(device_weights.get()), indices.size(),
          static_cast<const double *>(device_frequencies.get()),
          static_cast<meep_cuda::complex_value_fp32 *>(
              device_phase_scratch.get()),
          angular_frequencies.size(), time, scale_real, scale_imag,
          test.average1, test.average2);
    }

    meep_cuda::synchronize();
    meep_cuda::copy_to_host(output.data(), device_output.get(),
                            output.size() * sizeof(float));
    for (std::size_t index = 0; index < output.size(); ++index)
      if (std::abs(output[index] - reference[index]) > 4.0e-6f)
        return false;
  }
  return true;
}

bool run_dft_multi_monitor_batch_case() {
  constexpr std::size_t field_count = 2048;
  constexpr std::size_t guard_count = 4;
  constexpr float canary = 31415.25f;
  constexpr std::size_t threads_per_block =
      meep_cuda::dft_batch_threads_per_block_fp32;
  const std::size_t point_counts[] = {1, 3, 17, 257, 33};
  const std::size_t frequency_counts[] = {513, 2, 37, 1, 33};
  const std::ptrdiff_t average1[] = {0, 1, 1, -1, 3};
  const std::ptrdiff_t average2[] = {0, 0, 2, -2, 0};
  const bool complex_fields[] = {false, true, true, true, false};
  constexpr std::size_t operation_count =
      sizeof(point_counts) / sizeof(point_counts[0]);

  std::vector<float> field_real(field_count);
  std::vector<float> field_imag(field_count);
  for (std::size_t index = 0; index < field_count; ++index) {
    field_real[index] = static_cast<float>(
        std::sin(0.013 * static_cast<double>(index)));
    field_imag[index] = static_cast<float>(
        std::cos(0.019 * static_cast<double>(index)));
  }
  auto device_real = copy_vector_to_device(field_real);
  auto device_imag = copy_vector_to_device(field_imag);

  std::vector<std::vector<std::ptrdiff_t> > indices(operation_count);
  std::vector<std::vector<float> > weights(operation_count);
  std::vector<std::vector<meep_cuda::complex_value_fp32> > phases(
      operation_count);
  std::vector<std::vector<float> > storage(operation_count);
  std::vector<std::vector<float> > reference(operation_count);
  std::vector<device_pointer> device_indices;
  std::vector<device_pointer> device_weights;
  std::vector<device_pointer> device_phases;
  std::vector<device_pointer> device_storage;
  device_indices.reserve(operation_count);
  device_weights.reserve(operation_count);
  device_phases.reserve(operation_count);
  device_storage.reserve(operation_count);

  std::vector<meep_cuda::dft_update_operation_fp32> operations;
  std::vector<std::uint32_t> block_operation_indices;
  operations.reserve(operation_count);
  std::size_t total_block_count = 0;
  for (std::size_t operation = 0; operation < operation_count;
       ++operation) {
    const std::size_t point_count = point_counts[operation];
    const std::size_t frequency_count = frequency_counts[operation];
    indices[operation].resize(point_count);
    weights[operation].resize(point_count);
    for (std::size_t point = 0; point < point_count; ++point) {
      indices[operation][point] = static_cast<std::ptrdiff_t>(
          4 + (17 * point + 29 * operation) % (field_count - 8));
      weights[operation][point] =
          0.0125f * static_cast<float>(
                        static_cast<int>((point + 3 * operation) % 31) -
                        15);
    }
    phases[operation].resize(frequency_count);
    for (std::size_t frequency = 0; frequency < frequency_count;
         ++frequency) {
      const double angle =
          -0.73 + 0.017 * static_cast<double>(frequency) +
          0.11 * static_cast<double>(operation);
      phases[operation][frequency] = {
          static_cast<float>((0.5 + 0.07 * operation) * std::cos(angle)),
          static_cast<float>((0.5 + 0.07 * operation) * std::sin(angle))};
    }
    const std::size_t output_scalar_count =
        2 * point_count * frequency_count;
    storage[operation].assign(
        guard_count + output_scalar_count + guard_count, canary);
    reference[operation].resize(output_scalar_count);
    for (std::size_t scalar = 0; scalar < output_scalar_count; ++scalar) {
      const float initial =
          0.0005f * static_cast<float>(
                        static_cast<int>((scalar + operation) % 19) - 9);
      storage[operation][guard_count + scalar] = initial;
      reference[operation][scalar] = initial;
    }

    for (int accumulation = 0; accumulation < 2; ++accumulation)
      for (std::size_t point = 0; point < point_count; ++point) {
        const std::ptrdiff_t index = indices[operation][point];
        float real_value;
        float imag_value = 0.0f;
        if (average2[operation]) {
          real_value =
              0.25f *
              (field_real[index] +
               field_real[index + average1[operation]] +
               field_real[index + average2[operation]] +
               field_real[index + average1[operation] +
                                  average2[operation]]);
          if (complex_fields[operation])
            imag_value =
                0.25f *
                (field_imag[index] +
                 field_imag[index + average1[operation]] +
                 field_imag[index + average2[operation]] +
                 field_imag[index + average1[operation] +
                                    average2[operation]]);
        }
        else if (average1[operation]) {
          real_value =
              0.5f *
              (field_real[index] +
               field_real[index + average1[operation]]);
          if (complex_fields[operation])
            imag_value =
                0.5f *
                (field_imag[index] +
                 field_imag[index + average1[operation]]);
        }
        else {
          real_value = field_real[index];
          if (complex_fields[operation]) imag_value = field_imag[index];
        }
        real_value *= weights[operation][point];
        imag_value *= weights[operation][point];
        for (std::size_t frequency = 0; frequency < frequency_count;
             ++frequency) {
          const meep_cuda::complex_value_fp32 phase =
              phases[operation][frequency];
          const std::size_t output =
              2 * (point * frequency_count + frequency);
          reference[operation][output] +=
              phase.real * real_value - phase.imag * imag_value;
          reference[operation][output + 1] +=
              phase.real * imag_value + phase.imag * real_value;
        }
      }

    device_indices.push_back(copy_vector_to_device(indices[operation]));
    device_weights.push_back(copy_vector_to_device(weights[operation]));
    device_phases.push_back(copy_vector_to_device(phases[operation]));
    device_storage.push_back(copy_vector_to_device(storage[operation]));
    const std::size_t frequency_threads =
        std::min<std::size_t>(frequency_count, 32);
    const std::size_t point_threads = std::min<std::size_t>(
        point_count, threads_per_block / frequency_threads);
    const std::size_t frequency_blocks =
        frequency_count / frequency_threads +
        (frequency_count % frequency_threads != 0);
    const std::size_t point_blocks = point_count / point_threads +
                                     (point_count % point_threads != 0);
    const std::size_t operation_blocks =
        frequency_blocks * point_blocks;
    const std::size_t block_start = total_block_count;
    total_block_count += operation_blocks;
    block_operation_indices.insert(
        block_operation_indices.end(), operation_blocks,
        static_cast<std::uint32_t>(operation));
    operations.push_back(
        {static_cast<float *>(device_storage.back().get()) + guard_count,
         static_cast<const float *>(device_real.get()),
         complex_fields[operation]
             ? static_cast<const float *>(device_imag.get())
             : nullptr,
         static_cast<const std::ptrdiff_t *>(device_indices.back().get()),
         static_cast<const float *>(device_weights.back().get()),
         point_count,
         static_cast<const meep_cuda::complex_value_fp32 *>(
             device_phases.back().get()),
         frequency_count, average1[operation], average2[operation],
         block_start, frequency_blocks, point_blocks,
         static_cast<std::uint32_t>(frequency_threads),
         static_cast<std::uint32_t>(point_threads)});
  }

  meep_cuda::validate_dft_batch_operations_fp32(
      operations.data(), operations.size(), total_block_count);
  auto device_operations = copy_vector_to_device(operations);
  auto device_block_map = copy_vector_to_device(block_operation_indices);
  for (int accumulation = 0; accumulation < 2; ++accumulation)
    meep_cuda::update_dft_batch_fp32(
        static_cast<const meep_cuda::dft_update_operation_fp32 *>(
            device_operations.get()),
        static_cast<const std::uint32_t *>(device_block_map.get()),
        operations.size(), total_block_count);
  meep_cuda::synchronize();

  for (std::size_t operation = 0; operation < operation_count;
       ++operation) {
    meep_cuda::copy_to_host(
        storage[operation].data(), device_storage[operation].get(),
        storage[operation].size() * sizeof(float));
    const std::size_t output_scalar_count = reference[operation].size();
    for (std::size_t guard = 0; guard < guard_count; ++guard)
      if (storage[operation][guard] != canary ||
          storage[operation][guard_count + output_scalar_count + guard] !=
              canary)
        return false;
    for (std::size_t scalar = 0; scalar < output_scalar_count; ++scalar)
      if (std::abs(
              storage[operation][guard_count + scalar] -
              reference[operation][scalar]) > 5.0e-6f)
        return false;
  }
  return true;
}

template <typename Point>
std::array<std::complex<double>, 6> near2far_green3d_reference(
    const Point &target, double frequency,
    double eps, double mu,
    const Point &source, int direction,
    bool electric, std::complex<double> amplitude) {
  double rx = target.x - source.x;
  double ry = target.y - source.y;
  double rz = target.z - source.z;
  const double r = std::sqrt(rx * rx + ry * ry + rz * rz);
  rx /= r;
  ry /= r;
  rz /= r;
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double refractive_index = std::sqrt(eps * mu);
  const double k = 2.0 * pi * frequency * refractive_index;
  const std::complex<double> ikr(0.0, k * r);
  const double ikr2 = -(k * r) * (k * r);
  std::complex<double> expfac =
      amplitude * std::polar(k * refractive_index / (4.0 * pi * r),
                             k * r + 0.5 * pi);
  const double impedance = std::sqrt(mu / eps);
  const double p[3] = {direction == 0 ? 1.0 : 0.0,
                       direction == 1 ? 1.0 : 0.0,
                       direction == 2 ? 1.0 : 0.0};
  const double rhat[3] = {rx, ry, rz};
  const double pdotr = p[0] * rx + p[1] * ry + p[2] * rz;
  const double cross[3] = {ry * p[2] - rz * p[1],
                           rz * p[0] - rx * p[2],
                           rx * p[1] - ry * p[0]};
  const std::complex<double> term1 = 1.0 - 1.0 / ikr + 1.0 / ikr2;
  const std::complex<double> term2 =
      (-1.0 + 3.0 / ikr - 3.0 / ikr2) * pdotr;
  const std::complex<double> term3 = 1.0 - 1.0 / ikr;
  std::array<std::complex<double>, 6> fields;
  if (electric) {
    expfac /= eps;
    for (int axis = 0; axis < 3; ++axis)
      fields[axis] =
          expfac * (term1 * p[axis] + term2 * rhat[axis]);
    for (int axis = 0; axis < 3; ++axis)
      fields[3 + axis] = expfac * term3 * cross[axis] / impedance;
  }
  else {
    expfac /= mu;
    for (int axis = 0; axis < 3; ++axis)
      fields[axis] = -expfac * term3 * cross[axis] * impedance;
    for (int axis = 0; axis < 3; ++axis)
      fields[3 + axis] =
          expfac * (term1 * p[axis] + term2 * rhat[axis]);
  }
  return fields;
}

template <typename Point>
std::array<std::complex<double>, 6> near2far_green2d_reference(
    const Point &target, double frequency, double eps, double mu,
    const Point &source, int direction, bool electric,
    std::complex<double> amplitude) {
  double rx = target.x - source.x;
  double ry = target.y - source.y;
  const double r = std::sqrt(rx * rx + ry * ry);
  rx /= r;
  ry /= r;
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double omega = 2.0 * pi * frequency;
  const double k = omega * std::sqrt(eps * mu);
  const double kr = k * r;
  const double impedance = std::sqrt(mu / eps);
  const std::complex<double> h0 =
      std::complex<double>(::j0(kr), ::y0(kr)) * amplitude;
  const std::complex<double> h1 =
      std::complex<double>(::j1(kr), ::y1(kr)) * amplitude;
  const std::complex<double> h2 =
      std::complex<double>(::jn(2, kr), ::yn(2, kr)) * amplitude;
  const std::complex<double> ik_h1(0.0, 0.25 * k);
  const std::complex<double> rotated_h1 = ik_h1 * h1;
  std::array<std::complex<double>, 6> fields = {};
  const double px = direction == 0 ? 1.0 : 0.0;
  const double py = direction == 1 ? 1.0 : 0.0;
  const double pdotr = px * rx + py * ry;
  const double cross = rx * py - ry * px;
  if (direction == 2) {
    if (electric) {
      fields[2] = -0.25 * omega * mu * h0;
      fields[3] = -ry * rotated_h1;
      fields[4] = rx * rotated_h1;
    }
    else {
      fields[0] = ry * rotated_h1;
      fields[1] = -rx * rotated_h1;
      fields[5] = -0.25 * omega * eps * h0;
    }
  }
  else if (electric) {
    const double longitudinal = pdotr / r * 0.25 * impedance;
    const double transverse = cross * omega * mu * 0.125;
    fields[0] = -rx * longitudinal * h1 +
                ry * transverse * (h0 - h2);
    fields[1] = -ry * longitudinal * h1 -
                rx * transverse * (h0 - h2);
    fields[5] = -cross * rotated_h1;
  }
  else {
    const double longitudinal = pdotr / r * 0.25 / impedance;
    const double transverse = cross * omega * eps * 0.125;
    fields[2] = cross * rotated_h1;
    fields[3] = -rx * longitudinal * h1 +
                ry * transverse * (h0 - h2);
    fields[4] = -ry * longitudinal * h1 -
                rx * transverse * (h0 - h2);
  }
  return fields;
}

template <typename Point>
std::array<std::complex<double>, 6> near2far_greencyl_reference(
    const Point &target_cartesian, double frequency, double eps, double mu,
    const Point &source_rz, int direction, bool electric,
    std::complex<double> amplitude, double azimuthal_mode,
    double tolerance) {
  std::array<std::complex<double>, 6> fields = {};
  double sum_absolute = 0.0;
  constexpr double pi = 3.141592653589793238462643383279502884;
  const int initial_points =
      16 + static_cast<int>(4.0 * std::abs(azimuthal_mode));
  double dphi = 2.0 / initial_points;
  for (int point_count = initial_points; point_count <= 65536;) {
    dphi *= 0.5;
    const double angular_step = dphi * 2.0 * pi;
    std::array<std::complex<double>, 6> next;
    for (int component = 0; component < 6; ++component)
      next[component] = 0.5 * fields[component];
    double next_absolute = 0.5 * sum_absolute;
    const int first = point_count > initial_points ? 1 : 0;
    const int stride = point_count > initial_points ? 2 : 1;
    for (int index = first; index < point_count; index += stride) {
      const double phi = index * angular_step;
      const double cosine = std::cos(phi);
      const double sine = std::sin(phi);
      const Point source = {
          source_rz.x * cosine, source_rz.x * sine, source_rz.z};
      const std::complex<double> weighted =
          amplitude * std::polar(1.0, azimuthal_mode * phi) * dphi;
      const int call_count = direction == 2 ? 1 : 2;
      for (int call = 0; call < call_count; ++call) {
        int cartesian_direction = direction;
        double rotation = 1.0;
        if (direction == 0) {
          cartesian_direction = call == 0 ? 0 : 1;
          rotation = call == 0 ? cosine : sine;
        }
        else if (direction == 1) {
          cartesian_direction = call == 0 ? 0 : 1;
          rotation = call == 0 ? -sine : cosine;
        }
        const auto contribution = near2far_green3d_reference(
            target_cartesian, frequency, eps, mu, source,
            cartesian_direction,
            electric, weighted * rotation);
        for (int component = 0; component < 6; ++component) {
          next[component] += contribution[component];
          next_absolute += std::abs(contribution[component]);
        }
      }
    }
    double difference = 0.0;
    for (int component = 0; component < 6; ++component)
      difference += std::abs(fields[component] - next[component]);
    fields = next;
    sum_absolute = next_absolute;
    if (difference <= sum_absolute * tolerance) break;
    if (point_count > 32768) break;
    point_count *= 2;
  }
  return fields;
}

bool run_near2far_3d_case() {
  constexpr std::size_t operation_count = 6;
  constexpr std::size_t target_count = 5;
  constexpr std::size_t frequency_count = 3;
  constexpr std::size_t copy_count = 3;
  constexpr std::size_t partial_count = 4;
  constexpr double eps = 2.25;
  constexpr double mu = 1.44;
  const std::array<std::size_t, operation_count> operation_point_counts =
      {1, 255, 256, 257, 1025, 1031};
  const std::vector<float> frequencies = {0.17f, 0.43f, 0.79f};
  const std::vector<meep_cuda::cartesian_point_fp32> targets = {
      {2.3, -0.7, 1.1}, {-1.8, 2.1, 0.9}, {0.8, 1.7, -2.4},
      {11.0, -7.0, 4.0}, {170.0, 91.0, -53.0}};
  const std::vector<meep_cuda::near2far_periodic_copy_fp32> copies = {
      {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
      {{0.75f, -0.25f, 0.0f},
       {static_cast<float>(std::cos(0.37)),
        static_cast<float>(std::sin(0.37))}},
      {{-0.75f, 0.25f, 0.0f},
       {static_cast<float>(std::cos(-0.37)),
        static_cast<float>(std::sin(-0.37))}}};

  std::vector<std::vector<meep_cuda::cartesian_point_fp32> > points(
      operation_count);
  std::vector<std::vector<float> > dfts(operation_count);
  std::vector<std::vector<std::complex<float> > > complex_dfts(
      operation_count);
  std::vector<device_pointer> device_points;
  std::vector<device_pointer> device_dfts;
  std::vector<meep_cuda::near2far_operation_fp32> operations;
  device_points.reserve(operation_count);
  device_dfts.reserve(operation_count);
  operations.reserve(operation_count);
  for (std::size_t operation = 0; operation < operation_count; ++operation) {
    for (std::size_t point = 0;
         point < operation_point_counts[operation]; ++point) {
      const double p = static_cast<double>(point);
      points[operation].push_back(
          {static_cast<float>(-0.8 + 0.071 * p + 0.013 * operation),
           static_cast<float>(0.4 - 0.047 * p + 0.019 * operation),
           static_cast<float>(-0.3 + 0.031 * p - 0.011 * operation)});
      for (std::size_t frequency = 0; frequency < frequency_count;
           ++frequency) {
        const double angle = 0.23 * p + 0.41 * operation +
                             0.17 * frequency;
        const float real = static_cast<float>(
            (0.04 + 0.003 * point) * std::cos(angle));
        const float imaginary = static_cast<float>(
            (0.04 + 0.003 * point) * std::sin(angle));
        complex_dfts[operation].push_back({real, imaginary});
        dfts[operation].push_back(real);
        dfts[operation].push_back(imaginary);
      }
    }
    device_points.push_back(copy_vector_to_device(points[operation]));
    device_dfts.push_back(copy_vector_to_device(dfts[operation]));
    operations.push_back(
        {static_cast<const float *>(device_dfts.back().get()),
         static_cast<const meep_cuda::cartesian_point_fp32 *>(
             device_points.back().get()),
         operation_point_counts[operation], frequency_count,
         static_cast<int>(operation % 3), operation < 3});
  }

  std::vector<double> reference(target_count * frequency_count * 12, 0.0);
  for (std::size_t operation = 0; operation < operation_count; ++operation)
    for (std::size_t target = 0; target < target_count; ++target)
      for (std::size_t frequency = 0; frequency < frequency_count;
           ++frequency)
        for (std::size_t point = 0; point < points[operation].size(); ++point)
          for (const auto &copy : copies) {
            const meep_cuda::cartesian_point_fp32 shifted =
                {points[operation][point].x + copy.displacement.x,
                 points[operation][point].y + copy.displacement.y,
                 points[operation][point].z + copy.displacement.z};
            const auto fields = near2far_green3d_reference(
                targets[target], frequencies[frequency], eps, mu, shifted,
                static_cast<int>(operation % 3), operation < 3,
                static_cast<std::complex<double> >(
                    complex_dfts[operation]
                                [point * frequency_count + frequency]));
            const std::complex<double> phase(copy.phase.real,
                                             copy.phase.imag);
            const std::size_t work = target * frequency_count + frequency;
            for (int component = 0; component < 6; ++component) {
              const std::complex<double> value = fields[component] * phase;
              reference[12 * work + 2 * component] += value.real();
              reference[12 * work + 2 * component + 1] += value.imag();
            }
          }

  auto device_operations = copy_vector_to_device(operations);
  auto device_targets = copy_vector_to_device(targets);
  auto device_frequencies = copy_vector_to_device(frequencies);
  auto device_copies = copy_vector_to_device(copies);
  std::vector<float> partials(operation_count * target_count *
                                  frequency_count * partial_count * 13,
                              0.0f);
  std::vector<double> observed(reference.size(), 0.0);
  std::vector<double> absolute_l1(target_count * frequency_count, 0.0);
  auto device_partials = copy_vector_to_device(partials);
  auto device_output = copy_vector_to_device(observed);
  auto device_absolute_l1 = copy_vector_to_device(absolute_l1);
  const auto launch = [&]() {
    meep_cuda::near2far_3d_fp32(
        static_cast<const meep_cuda::near2far_operation_fp32 *>(
            device_operations.get()),
        operation_count,
        static_cast<const meep_cuda::cartesian_point_fp32 *>(
            device_targets.get()),
        target_count, static_cast<const float *>(device_frequencies.get()),
        frequency_count, 0,
        static_cast<const meep_cuda::near2far_periodic_copy_fp32 *>(
            device_copies.get()),
        copy_count, static_cast<float>(eps), static_cast<float>(mu),
        partial_count, static_cast<float *>(device_partials.get()),
        static_cast<double *>(device_output.get()),
        static_cast<double *>(device_absolute_l1.get()));
    meep_cuda::synchronize();
  };
  launch();
  meep_cuda::copy_to_host(observed.data(), device_output.get(),
                          observed.size() * sizeof(double));
  meep_cuda::copy_to_host(absolute_l1.data(), device_absolute_l1.get(),
                          absolute_l1.size() * sizeof(double));
  const std::vector<double> first = observed;
  launch();
  meep_cuda::copy_to_host(observed.data(), device_output.get(),
                          observed.size() * sizeof(double));
  if (std::memcmp(first.data(), observed.data(),
                  observed.size() * sizeof(double)) != 0)
    return false;

  double maximum_error = 0.0;
  double maximum_reference = 0.0;
  for (std::size_t index = 0; index < observed.size(); ++index) {
    maximum_error =
        std::max(maximum_error, std::abs(observed[index] - reference[index]));
    maximum_reference =
        std::max(maximum_reference, std::abs(reference[index]));
  }
  const double relative_error =
      maximum_error / std::max(maximum_reference, 1.0e-300);
  std::cout << "near2far_3d max_abs_error=" << maximum_error
            << " relative_error=" << relative_error << '\n';
  std::vector<float> observed_partials(partials.size());
  meep_cuda::copy_to_host(observed_partials.data(), device_partials.get(),
                          observed_partials.size() * sizeof(float));
  for (std::size_t partial = 0; partial < partial_count; ++partial) {
    bool did_work = false;
    const std::size_t operation = operation_count - 1;
    for (std::size_t work = 0; work < target_count * frequency_count;
         ++work)
      for (std::size_t channel = 0; channel < 12; ++channel) {
        const std::size_t index =
            13 * ((operation * target_count * frequency_count + work) *
                      partial_count +
                  partial) +
            channel;
        did_work = did_work || observed_partials[index] != 0.0f;
      }
    if (!did_work) return false;
  }
  for (std::size_t work = 0; work < absolute_l1.size(); ++work) {
    for (std::size_t channel = 0; channel < 12; ++channel)
      if (!std::isfinite(absolute_l1[work]) ||
          absolute_l1[work] +
                  1.0e-6 * std::max(absolute_l1[work], 1.0) <
              std::abs(observed[12 * work + channel]))
        return false;
  }
  return maximum_error <= 2.0e-4 * std::max(maximum_reference, 1.0) &&
         relative_error <= 2.0e-4;
}

bool run_near2far_2d_case() {
  constexpr std::size_t operation_count = 6;
  constexpr std::size_t target_count = 4;
  constexpr std::size_t frequency_count = 3;
  constexpr std::size_t copy_count = 3;
  constexpr std::size_t partial_count = 3;
  constexpr double eps = 2.25;
  constexpr double mu = 1.44;
  const std::array<std::size_t, operation_count> operation_point_counts =
      {1, 17, 33, 65, 129, 257};
  const std::vector<double> frequencies = {0.17, 0.43, 0.79};
  const std::vector<meep_cuda::cartesian_point_fp64> targets = {
      {2.3, -0.7, 0.0}, {-1.8, 2.1, 0.0},
      {11.0, -7.0, 0.0}, {170.0, 91.0, 0.0}};
  const std::vector<meep_cuda::near2far_periodic_copy_fp64> copies = {
      {{0.0, 0.0, 0.0}, {1.0, 0.0}},
      {{0.75, -0.25, 0.0}, {std::cos(0.37), std::sin(0.37)}},
      {{-0.75, 0.25, 0.0}, {std::cos(-0.37), std::sin(-0.37)}}};

  std::vector<std::vector<meep_cuda::cartesian_point_fp64> > points(
      operation_count);
  std::vector<std::vector<float> > dfts(operation_count);
  std::vector<std::vector<std::complex<float> > > complex_dfts(
      operation_count);
  for (std::size_t operation = 0; operation < operation_count; ++operation)
    for (std::size_t point = 0;
         point < operation_point_counts[operation]; ++point) {
      const double p = static_cast<double>(point);
      points[operation].push_back(
          {-0.8 + 0.019 * p + 0.013 * operation,
           0.4 - 0.011 * p + 0.017 * operation, 0.0});
      for (std::size_t frequency = 0; frequency < frequency_count;
           ++frequency) {
        const double angle = 0.23 * p + 0.41 * operation +
                             0.17 * frequency;
        const float real = static_cast<float>(
            (0.04 + 0.001 * point) * std::cos(angle));
        const float imaginary = static_cast<float>(
            (0.04 + 0.001 * point) * std::sin(angle));
        complex_dfts[operation].push_back({real, imaginary});
        dfts[operation].push_back(real);
        dfts[operation].push_back(imaginary);
      }
    }

  std::vector<double> reference(target_count * frequency_count * 12, 0.0);
  for (std::size_t operation = 0; operation < operation_count; ++operation)
    for (std::size_t target = 0; target < target_count; ++target)
      for (std::size_t frequency = 0; frequency < frequency_count;
           ++frequency)
        for (std::size_t point = 0; point < points[operation].size(); ++point)
          for (const auto &copy : copies) {
            const meep_cuda::cartesian_point_fp64 shifted =
                {points[operation][point].x + copy.displacement.x,
                 points[operation][point].y + copy.displacement.y, 0.0};
            const auto fields = near2far_green2d_reference(
                targets[target], frequencies[frequency], eps, mu, shifted,
                static_cast<int>(operation % 3), operation < 3,
                static_cast<std::complex<double> >(
                    complex_dfts[operation]
                                [point * frequency_count + frequency]));
            const std::complex<double> phase(copy.phase.real,
                                             copy.phase.imag);
            const std::size_t work = target * frequency_count + frequency;
            for (int component = 0; component < 6; ++component) {
              const std::complex<double> value = fields[component] * phase;
              reference[12 * work + 2 * component] += value.real();
              reference[12 * work + 2 * component + 1] += value.imag();
            }
          }

  std::vector<meep_cuda::cartesian_point_fp32> targets_fp32;
  std::vector<float> frequencies_fp32;
  std::vector<meep_cuda::near2far_periodic_copy_fp32> copies_fp32;
  for (const auto &target : targets)
    targets_fp32.push_back({static_cast<float>(target.x),
                            static_cast<float>(target.y), 0.0f});
  for (double frequency : frequencies)
    frequencies_fp32.push_back(static_cast<float>(frequency));
  for (const auto &copy : copies)
    copies_fp32.push_back(
        {{static_cast<float>(copy.displacement.x),
          static_cast<float>(copy.displacement.y), 0.0f},
         {static_cast<float>(copy.phase.real),
          static_cast<float>(copy.phase.imag)}});

  std::vector<device_pointer> device_dfts;
  std::vector<device_pointer> device_points_fp32;
  std::vector<device_pointer> device_points_fp64;
  std::vector<meep_cuda::near2far_operation_fp32> operations_fp32;
  std::vector<meep_cuda::near2far_operation_mixed_fp32> operations_fp64;
  for (std::size_t operation = 0; operation < operation_count; ++operation) {
    std::vector<meep_cuda::cartesian_point_fp32> points_fp32;
    for (const auto &point : points[operation])
      points_fp32.push_back({static_cast<float>(point.x),
                             static_cast<float>(point.y), 0.0f});
    device_dfts.push_back(copy_vector_to_device(dfts[operation]));
    device_points_fp32.push_back(copy_vector_to_device(points_fp32));
    device_points_fp64.push_back(copy_vector_to_device(points[operation]));
    operations_fp32.push_back(
        {static_cast<const float *>(device_dfts.back().get()),
         static_cast<const meep_cuda::cartesian_point_fp32 *>(
             device_points_fp32.back().get()),
         points[operation].size(), frequency_count,
         static_cast<int>(operation % 3), operation < 3});
    operations_fp64.push_back(
        {static_cast<const float *>(device_dfts.back().get()),
         static_cast<const meep_cuda::cartesian_point_fp64 *>(
             device_points_fp64.back().get()),
         points[operation].size(), frequency_count,
         static_cast<int>(operation % 3), operation < 3});
  }

  auto device_operations_fp32 = copy_vector_to_device(operations_fp32);
  auto device_operations_fp64 = copy_vector_to_device(operations_fp64);
  auto device_targets_fp32 = copy_vector_to_device(targets_fp32);
  auto device_targets_fp64 = copy_vector_to_device(targets);
  auto device_frequencies_fp32 = copy_vector_to_device(frequencies_fp32);
  auto device_frequencies_fp64 = copy_vector_to_device(frequencies);
  auto device_copies_fp32 = copy_vector_to_device(copies_fp32);
  auto device_copies_fp64 = copy_vector_to_device(copies);
  std::vector<float> partials_fp32(operation_count * target_count *
                                       frequency_count * partial_count * 13,
                                   0.0f);
  std::vector<double> partials_fp64(operation_count * target_count *
                                        frequency_count * partial_count * 12,
                                    0.0);
  std::vector<double> observed_fp32(reference.size(), 0.0);
  std::vector<double> observed_fp64(reference.size(), 0.0);
  std::vector<double> absolute_l1(target_count * frequency_count, 0.0);
  std::vector<double> partial_warp_observed(reference.size(), 0.0);
  std::vector<double> partial_warp_l1(
      target_count * frequency_count, 0.0);
  std::vector<double> two_warp_observed(reference.size(), 0.0);
  std::vector<double> two_warp_l1(
      target_count * frequency_count, 0.0);
  auto device_partials_fp32 = copy_vector_to_device(partials_fp32);
  auto device_partials_fp64 = copy_vector_to_device(partials_fp64);
  auto device_output_fp32 = copy_vector_to_device(observed_fp32);
  auto device_output_fp64 = copy_vector_to_device(observed_fp64);
  auto device_absolute_l1 = copy_vector_to_device(absolute_l1);
  auto device_partial_warp_output =
      copy_vector_to_device(partial_warp_observed);
  auto device_partial_warp_l1 = copy_vector_to_device(partial_warp_l1);
  auto device_two_warp_output =
      copy_vector_to_device(two_warp_observed);
  auto device_two_warp_l1 = copy_vector_to_device(two_warp_l1);

  meep_cuda::near2far_cartesian_fp32(
      meep_cuda::near2far_cartesian_dimension::two,
      static_cast<const meep_cuda::near2far_operation_fp32 *>(
          device_operations_fp32.get()),
      operation_count,
      static_cast<const meep_cuda::cartesian_point_fp32 *>(
          device_targets_fp32.get()),
      target_count, static_cast<const float *>(device_frequencies_fp32.get()),
      frequency_count, 0,
      static_cast<const meep_cuda::near2far_periodic_copy_fp32 *>(
          device_copies_fp32.get()),
      copy_count, static_cast<float>(eps), static_cast<float>(mu),
      partial_count, static_cast<float *>(device_partials_fp32.get()),
      static_cast<double *>(device_output_fp32.get()),
      static_cast<double *>(device_absolute_l1.get()));
  // Exercise both a partial first warp and a two-warp second-stage
  // reduction. The independent FP64 oracle below catches participation-mask
  // mistakes that a production-only 256-thread launch can conceal.
  meep_cuda::near2far_cartesian_fp32(
      meep_cuda::near2far_cartesian_dimension::two,
      static_cast<const meep_cuda::near2far_operation_fp32 *>(
          device_operations_fp32.get()),
      operation_count,
      static_cast<const meep_cuda::cartesian_point_fp32 *>(
          device_targets_fp32.get()),
      target_count, static_cast<const float *>(device_frequencies_fp32.get()),
      frequency_count, 0,
      static_cast<const meep_cuda::near2far_periodic_copy_fp32 *>(
          device_copies_fp32.get()),
      copy_count, static_cast<float>(eps), static_cast<float>(mu),
      partial_count, static_cast<float *>(device_partials_fp32.get()),
      static_cast<double *>(device_partial_warp_output.get()),
      static_cast<double *>(device_partial_warp_l1.get()), 16);
  meep_cuda::near2far_cartesian_fp32(
      meep_cuda::near2far_cartesian_dimension::two,
      static_cast<const meep_cuda::near2far_operation_fp32 *>(
          device_operations_fp32.get()),
      operation_count,
      static_cast<const meep_cuda::cartesian_point_fp32 *>(
          device_targets_fp32.get()),
      target_count, static_cast<const float *>(device_frequencies_fp32.get()),
      frequency_count, 0,
      static_cast<const meep_cuda::near2far_periodic_copy_fp32 *>(
          device_copies_fp32.get()),
      copy_count, static_cast<float>(eps), static_cast<float>(mu),
      partial_count, static_cast<float *>(device_partials_fp32.get()),
      static_cast<double *>(device_two_warp_output.get()),
      static_cast<double *>(device_two_warp_l1.get()), 64);
  meep_cuda::near2far_cartesian_mixed_fp32(
      meep_cuda::near2far_cartesian_dimension::two,
      static_cast<const meep_cuda::near2far_operation_mixed_fp32 *>(
          device_operations_fp64.get()),
      operation_count,
      static_cast<const meep_cuda::cartesian_point_fp64 *>(
          device_targets_fp64.get()),
      target_count,
      static_cast<const double *>(device_frequencies_fp64.get()),
      frequency_count, 0,
      static_cast<const meep_cuda::near2far_periodic_copy_fp64 *>(
          device_copies_fp64.get()),
      copy_count, eps, mu, partial_count,
      static_cast<double *>(device_partials_fp64.get()),
      static_cast<double *>(device_output_fp64.get()));
  meep_cuda::synchronize();
  meep_cuda::copy_to_host(observed_fp32.data(), device_output_fp32.get(),
                          observed_fp32.size() * sizeof(double));
  meep_cuda::copy_to_host(observed_fp64.data(), device_output_fp64.get(),
                          observed_fp64.size() * sizeof(double));
  meep_cuda::copy_to_host(absolute_l1.data(), device_absolute_l1.get(),
                          absolute_l1.size() * sizeof(double));
  meep_cuda::copy_to_host(
      partial_warp_observed.data(), device_partial_warp_output.get(),
      partial_warp_observed.size() * sizeof(double));
  meep_cuda::copy_to_host(
      partial_warp_l1.data(), device_partial_warp_l1.get(),
      partial_warp_l1.size() * sizeof(double));
  meep_cuda::copy_to_host(
      two_warp_observed.data(), device_two_warp_output.get(),
      two_warp_observed.size() * sizeof(double));
  meep_cuda::copy_to_host(
      two_warp_l1.data(), device_two_warp_l1.get(),
      two_warp_l1.size() * sizeof(double));

  double maximum_reference = 0.0;
  double fast_error = 0.0;
  double mixed_error = 0.0;
  for (std::size_t index = 0; index < reference.size(); ++index) {
    maximum_reference = std::max(maximum_reference, std::abs(reference[index]));
    fast_error =
        std::max(fast_error, std::abs(observed_fp32[index] - reference[index]));
    fast_error = std::max(
        fast_error,
        std::abs(partial_warp_observed[index] - reference[index]));
    fast_error = std::max(
        fast_error,
        std::abs(two_warp_observed[index] - reference[index]));
    mixed_error =
        std::max(mixed_error, std::abs(observed_fp64[index] - reference[index]));
  }
  for (std::size_t work = 0; work < absolute_l1.size(); ++work)
    for (std::size_t channel = 0; channel < 12; ++channel)
      if (!std::isfinite(absolute_l1[work]) ||
          absolute_l1[work] +
                  1.0e-6 * std::max(absolute_l1[work], 1.0) <
              std::abs(observed_fp32[12 * work + channel]) ||
          !std::isfinite(partial_warp_l1[work]) ||
          partial_warp_l1[work] +
                  1.0e-6 * std::max(partial_warp_l1[work], 1.0) <
              std::abs(partial_warp_observed[12 * work + channel]) ||
          !std::isfinite(two_warp_l1[work]) ||
          two_warp_l1[work] +
                  1.0e-6 * std::max(two_warp_l1[work], 1.0) <
              std::abs(two_warp_observed[12 * work + channel]))
        return false;
  const double scale = std::max(maximum_reference, 1.0);
  std::cout << "near2far_2d fast_relative_error=" << fast_error / scale
            << " mixed_relative_error=" << mixed_error / scale << '\n';
  return fast_error <= 5.0e-4 * scale &&
         mixed_error <= 5.0e-8 * scale;
}

bool run_near2far_cylindrical_case() {
  constexpr std::size_t operation_count = 6;
  constexpr std::size_t target_count = 2;
  constexpr std::size_t frequency_count = 1;
  constexpr std::size_t copy_count = 2;
  constexpr std::size_t partial_count = 2;
  constexpr double eps = 2.25;
  constexpr double mu = 1.44;
  constexpr double reference_tolerance = 1.0e-9;
  // Cylindrical greencyl is built from the 3D Green tensor and supports both
  // signs of nonzero frequency.  Use the negative branch here; public
  // integration coverage exercises positive and negative frequencies in one
  // monitor.
  const std::vector<double> frequencies = {-0.37};
  const std::vector<meep_cuda::cartesian_point_fp64> targets = {
      {2.4, -1.1, 1.7}, {-2.2, 1.7, -1.3}};
  const std::vector<meep_cuda::near2far_periodic_copy_fp64> copies = {
      {{0.0, 0.0, 0.0}, {1.0, 0.0}},
      {{0.08, 0.0, -0.13}, {std::cos(0.29), std::sin(0.29)}}};
  const double modes[] = {-2.0, 0.0, 3.0};

  for (double mode : modes) {
    std::vector<std::vector<meep_cuda::cartesian_point_fp64> > points(
        operation_count);
    std::vector<std::vector<float> > dfts(operation_count);
    std::vector<std::complex<float> > amplitudes(operation_count);
    std::vector<double> reference(target_count * frequency_count * 12, 0.0);
    for (std::size_t operation = 0; operation < operation_count; ++operation) {
      points[operation].push_back(
          {0.41 + 0.037 * operation, 0.0, -0.32 + 0.09 * operation});
      amplitudes[operation] = std::complex<float>(
          static_cast<float>(0.17 + 0.021 * operation),
          static_cast<float>(-0.09 + 0.013 * operation));
      dfts[operation] = {amplitudes[operation].real(),
                         amplitudes[operation].imag()};
      for (std::size_t target = 0; target < target_count; ++target)
        for (const auto &copy : copies) {
          const meep_cuda::cartesian_point_fp64 shifted = {
              points[operation][0].x + copy.displacement.x, 0.0,
              points[operation][0].z + copy.displacement.z};
          const auto fields = near2far_greencyl_reference(
              targets[target], frequencies[0], eps, mu, shifted,
              static_cast<int>(operation % 3), operation < 3,
              static_cast<std::complex<double> >(amplitudes[operation]),
              mode, reference_tolerance);
          const std::complex<double> phase(copy.phase.real,
                                           copy.phase.imag);
          for (int component = 0; component < 6; ++component) {
            const std::complex<double> value = fields[component] * phase;
            reference[12 * target + 2 * component] += value.real();
            reference[12 * target + 2 * component + 1] += value.imag();
          }
        }
    }

    std::vector<meep_cuda::cartesian_point_fp32> targets_fp32;
    std::vector<float> frequencies_fp32;
    std::vector<meep_cuda::near2far_periodic_copy_fp32> copies_fp32;
    for (const auto &target : targets)
      targets_fp32.push_back({static_cast<float>(target.x),
                              static_cast<float>(target.y),
                              static_cast<float>(target.z)});
    frequencies_fp32.push_back(static_cast<float>(frequencies[0]));
    for (const auto &copy : copies)
      copies_fp32.push_back(
          {{static_cast<float>(copy.displacement.x), 0.0f,
            static_cast<float>(copy.displacement.z)},
           {static_cast<float>(copy.phase.real),
            static_cast<float>(copy.phase.imag)}});

    std::vector<device_pointer> device_dfts;
    std::vector<device_pointer> device_points_fp32;
    std::vector<device_pointer> device_points_fp64;
    std::vector<meep_cuda::near2far_operation_fp32> operations_fp32;
    std::vector<meep_cuda::near2far_operation_mixed_fp32> operations_fp64;
    for (std::size_t operation = 0; operation < operation_count; ++operation) {
      const std::vector<meep_cuda::cartesian_point_fp32> point_fp32 = {
          {static_cast<float>(points[operation][0].x), 0.0f,
           static_cast<float>(points[operation][0].z)}};
      device_dfts.push_back(copy_vector_to_device(dfts[operation]));
      device_points_fp32.push_back(copy_vector_to_device(point_fp32));
      device_points_fp64.push_back(copy_vector_to_device(points[operation]));
      operations_fp32.push_back(
          {static_cast<const float *>(device_dfts.back().get()),
           static_cast<const meep_cuda::cartesian_point_fp32 *>(
               device_points_fp32.back().get()),
           1, 1, static_cast<int>(operation % 3), operation < 3});
      operations_fp64.push_back(
          {static_cast<const float *>(device_dfts.back().get()),
           static_cast<const meep_cuda::cartesian_point_fp64 *>(
               device_points_fp64.back().get()),
           1, 1, static_cast<int>(operation % 3), operation < 3});
    }
    auto device_operations_fp32 = copy_vector_to_device(operations_fp32);
    auto device_operations_fp64 = copy_vector_to_device(operations_fp64);
    auto device_targets_fp32 = copy_vector_to_device(targets_fp32);
    auto device_targets_fp64 = copy_vector_to_device(targets);
    auto device_frequencies_fp32 = copy_vector_to_device(frequencies_fp32);
    auto device_frequencies_fp64 = copy_vector_to_device(frequencies);
    auto device_copies_fp32 = copy_vector_to_device(copies_fp32);
    auto device_copies_fp64 = copy_vector_to_device(copies);
    std::vector<float> partials_fp32(
        operation_count * target_count * partial_count * 13, 0.0f);
    std::vector<double> partials_fp64(
        operation_count * target_count * partial_count * 12, 0.0);
    std::vector<double> observed_fp32(reference.size(), 0.0);
    std::vector<double> observed_fp64(reference.size(), 0.0);
    std::vector<double> absolute_l1(target_count, 0.0);
    auto device_partials_fp32 = copy_vector_to_device(partials_fp32);
    auto device_partials_fp64 = copy_vector_to_device(partials_fp64);
    auto device_output_fp32 = copy_vector_to_device(observed_fp32);
    auto device_output_fp64 = copy_vector_to_device(observed_fp64);
    auto device_l1 = copy_vector_to_device(absolute_l1);
    meep_cuda::near2far_cartesian_fp32(
        meep_cuda::near2far_cartesian_dimension::cylindrical,
        static_cast<const meep_cuda::near2far_operation_fp32 *>(
            device_operations_fp32.get()),
        operation_count,
        static_cast<const meep_cuda::cartesian_point_fp32 *>(
            device_targets_fp32.get()),
        target_count,
        static_cast<const float *>(device_frequencies_fp32.get()), 1, 0,
        static_cast<const meep_cuda::near2far_periodic_copy_fp32 *>(
            device_copies_fp32.get()),
        copy_count, static_cast<float>(eps), static_cast<float>(mu),
        partial_count, static_cast<float *>(device_partials_fp32.get()),
        static_cast<double *>(device_output_fp32.get()),
        static_cast<double *>(device_l1.get()), 128, 0.0f,
        static_cast<float>(mode), 2.5e-5f);
    meep_cuda::near2far_cartesian_mixed_fp32(
        meep_cuda::near2far_cartesian_dimension::cylindrical,
        static_cast<const meep_cuda::near2far_operation_mixed_fp32 *>(
            device_operations_fp64.get()),
        operation_count,
        static_cast<const meep_cuda::cartesian_point_fp64 *>(
            device_targets_fp64.get()),
        target_count,
        static_cast<const double *>(device_frequencies_fp64.get()), 1, 0,
        static_cast<const meep_cuda::near2far_periodic_copy_fp64 *>(
            device_copies_fp64.get()),
        copy_count, eps, mu, partial_count,
        static_cast<double *>(device_partials_fp64.get()),
        static_cast<double *>(device_output_fp64.get()), 128, mode, 1.0e-8);
    meep_cuda::synchronize();
    meep_cuda::copy_to_host(observed_fp32.data(), device_output_fp32.get(),
                            observed_fp32.size() * sizeof(double));
    meep_cuda::copy_to_host(observed_fp64.data(), device_output_fp64.get(),
                            observed_fp64.size() * sizeof(double));
    meep_cuda::copy_to_host(absolute_l1.data(), device_l1.get(),
                            absolute_l1.size() * sizeof(double));
    double scale = 1.0e-300;
    double fast_error = 0.0;
    double mixed_error = 0.0;
    for (std::size_t index = 0; index < reference.size(); ++index) {
      scale = std::max(scale, std::abs(reference[index]));
      fast_error = std::max(
          fast_error, std::abs(observed_fp32[index] - reference[index]));
      mixed_error = std::max(
          mixed_error, std::abs(observed_fp64[index] - reference[index]));
    }
    for (std::size_t work = 0; work < target_count; ++work)
      for (int channel = 0; channel < 12; ++channel)
        if (!std::isfinite(absolute_l1[work]) ||
            absolute_l1[work] + 1.0e-6 * std::max(absolute_l1[work], 1.0) <
                std::abs(observed_fp32[12 * work + channel]))
          return false;
    std::cout << "near2far_cylindrical m=" << mode
              << " fast_relative_error=" << fast_error / scale
              << " mixed_relative_error=" << mixed_error / scale << '\n';
    if (fast_error > 8.0e-4 * std::max(scale, 1.0) ||
        mixed_error > 5.0e-8 * std::max(scale, 1.0))
      return false;
  }
  return true;
}

bool run_near2far_adjoint_cartesian_case(
    meep_cuda::near2far_cartesian_dimension dimension) {
  const bool two_dimensional =
      dimension == meep_cuda::near2far_cartesian_dimension::two;
  constexpr std::size_t source_count = 6;
  constexpr std::size_t target_count = 3;
  constexpr std::size_t frequency_count = 2;
  constexpr std::size_t copy_count = 2;
  constexpr double eps = 2.1;
  constexpr double mu = 1.3;
  std::vector<meep_cuda::near2far_adjoint_source_fp64> sources_fp64;
  for (std::size_t source = 0; source < source_count; ++source)
    sources_fp64.push_back(
        {{-0.37 + 0.12 * source, 0.19 - 0.041 * source,
          two_dimensional ? 0.0 : -0.22 + 0.073 * source},
         {0.31 - 0.027 * source, -0.17 + 0.019 * source},
         static_cast<int>(source % 3), source < 3 ? 1 : 0});
  const std::vector<meep_cuda::cartesian_point_fp64> targets_fp64 = {
      {2.3, -1.4, two_dimensional ? 0.0 : 1.1},
      {-1.8, 2.7, two_dimensional ? 0.0 : -0.9},
      {3.2, 1.3, two_dimensional ? 0.0 : 2.1}};
  const std::vector<double> frequencies_fp64 = {0.23, 0.41};
  const std::vector<meep_cuda::near2far_periodic_copy_fp64> copies_fp64 = {
      {{0.0, 0.0, 0.0}, {1.0, 0.0}},
      {{0.14, -0.09, 0.0}, {std::cos(0.37), std::sin(0.37)}}};
  std::vector<meep_cuda::complex_value_fp64> gradient_fp64;
  for (std::size_t target = 0; target < target_count; ++target)
    for (std::size_t frequency = 0; frequency < frequency_count;
         ++frequency)
      for (int component = 0; component < 6; ++component)
        gradient_fp64.push_back(
            {0.13 + 0.017 * target - 0.011 * frequency +
                 0.007 * component,
             -0.09 + 0.013 * target + 0.019 * frequency -
                 0.005 * component});

  std::vector<std::complex<double> > reference(
      source_count * frequency_count, std::complex<double>(0.0, 0.0));
  for (std::size_t source = 0; source < source_count; ++source)
    for (std::size_t frequency = 0; frequency < frequency_count;
         ++frequency)
      for (std::size_t target = 0; target < target_count; ++target)
        for (const auto &copy : copies_fp64) {
          const meep_cuda::cartesian_point_fp64 shifted = {
              sources_fp64[source].point.x + copy.displacement.x,
              sources_fp64[source].point.y + copy.displacement.y,
              sources_fp64[source].point.z + copy.displacement.z};
          const std::complex<double> amplitude(
              sources_fp64[source].amplitude.real,
              sources_fp64[source].amplitude.imag);
          const auto fields = two_dimensional
              ? near2far_green2d_reference(
                    targets_fp64[target], frequencies_fp64[frequency], eps,
                    mu, shifted, sources_fp64[source].direction,
                    sources_fp64[source].electric != 0, amplitude)
              : near2far_green3d_reference(
                    targets_fp64[target], frequencies_fp64[frequency], eps,
                    mu, shifted, sources_fp64[source].direction,
                    sources_fp64[source].electric != 0, amplitude);
          const std::complex<double> phase(copy.phase.real,
                                           copy.phase.imag);
          for (int component = 0; component < 6; ++component) {
            const auto &gradient =
                gradient_fp64[6 * (target * frequency_count + frequency) +
                              component];
            reference[source * frequency_count + frequency] +=
                fields[component] * phase *
                std::complex<double>(gradient.real, gradient.imag);
          }
        }

  std::vector<meep_cuda::near2far_adjoint_source_fp32> sources_fp32;
  for (const auto &source : sources_fp64)
    sources_fp32.push_back(
        {{static_cast<float>(source.point.x),
          static_cast<float>(source.point.y),
          static_cast<float>(source.point.z)},
         {static_cast<float>(source.amplitude.real),
          static_cast<float>(source.amplitude.imag)},
         source.direction, source.electric});
  std::vector<meep_cuda::cartesian_point_fp32> targets_fp32;
  for (const auto &target : targets_fp64)
    targets_fp32.push_back(
        {static_cast<float>(target.x), static_cast<float>(target.y),
         static_cast<float>(target.z)});
  std::vector<float> frequencies_fp32;
  for (double frequency : frequencies_fp64)
    frequencies_fp32.push_back(static_cast<float>(frequency));
  std::vector<meep_cuda::near2far_periodic_copy_fp32> copies_fp32;
  for (const auto &copy : copies_fp64)
    copies_fp32.push_back(
        {{static_cast<float>(copy.displacement.x),
          static_cast<float>(copy.displacement.y),
          static_cast<float>(copy.displacement.z)},
         {static_cast<float>(copy.phase.real),
          static_cast<float>(copy.phase.imag)}});
  std::vector<meep_cuda::complex_value_fp32> gradient_fp32;
  for (const auto &gradient : gradient_fp64)
    gradient_fp32.push_back(
        {static_cast<float>(gradient.real),
         static_cast<float>(gradient.imag)});

  auto device_sources_fp32 = copy_vector_to_device(sources_fp32);
  auto device_sources_fp64 = copy_vector_to_device(sources_fp64);
  auto device_targets_fp32 = copy_vector_to_device(targets_fp32);
  auto device_targets_fp64 = copy_vector_to_device(targets_fp64);
  auto device_frequencies_fp32 = copy_vector_to_device(frequencies_fp32);
  auto device_frequencies_fp64 = copy_vector_to_device(frequencies_fp64);
  auto device_copies_fp32 = copy_vector_to_device(copies_fp32);
  auto device_copies_fp64 = copy_vector_to_device(copies_fp64);
  auto device_gradient_fp32 = copy_vector_to_device(gradient_fp32);
  auto device_gradient_fp64 = copy_vector_to_device(gradient_fp64);
  std::vector<double> fast_output(2 * reference.size(), 0.0);
  std::vector<double> mixed_output(2 * reference.size(), 0.0);
  std::vector<double> absolute_l1(reference.size(), 0.0);
  auto device_fast_output = copy_vector_to_device(fast_output);
  auto device_mixed_output = copy_vector_to_device(mixed_output);
  auto device_l1 = copy_vector_to_device(absolute_l1);
  std::vector<double> tiled_fast_output(2 * reference.size(), 0.0);
  std::vector<double> tiled_mixed_output(2 * reference.size(), 0.0);
  std::vector<double> tiled_absolute_l1(reference.size(), 0.0);
  auto device_tiled_fast_output = copy_vector_to_device(tiled_fast_output);
  auto device_tiled_mixed_output = copy_vector_to_device(tiled_mixed_output);
  auto device_tiled_l1 = copy_vector_to_device(tiled_absolute_l1);
  std::vector<double> narrow_fast_output(2 * reference.size(), 0.0);
  std::vector<double> narrow_absolute_l1(reference.size(), 0.0);
  auto device_narrow_fast_output =
      copy_vector_to_device(narrow_fast_output);
  auto device_narrow_l1 = copy_vector_to_device(narrow_absolute_l1);
  meep_cuda::near2far_adjoint_fp32(
      dimension,
      static_cast<const meep_cuda::near2far_adjoint_source_fp32 *>(
          device_sources_fp32.get()),
      source_count,
      static_cast<const meep_cuda::cartesian_point_fp32 *>(
          device_targets_fp32.get()),
      target_count, static_cast<const float *>(device_frequencies_fp32.get()),
      frequency_count,
      static_cast<const meep_cuda::near2far_periodic_copy_fp32 *>(
          device_copies_fp32.get()),
      copy_count,
      static_cast<const meep_cuda::complex_value_fp32 *>(
          device_gradient_fp32.get()),
      static_cast<float>(eps), static_cast<float>(mu),
      static_cast<double *>(device_fast_output.get()),
      static_cast<double *>(device_l1.get()), 256, 0.0f);
  meep_cuda::near2far_adjoint_mixed_fp32(
      dimension,
      static_cast<const meep_cuda::near2far_adjoint_source_fp64 *>(
          device_sources_fp64.get()),
      source_count,
      static_cast<const meep_cuda::cartesian_point_fp64 *>(
          device_targets_fp64.get()),
      target_count,
      static_cast<const double *>(device_frequencies_fp64.get()),
      frequency_count,
      static_cast<const meep_cuda::near2far_periodic_copy_fp64 *>(
          device_copies_fp64.get()),
      copy_count,
      static_cast<const meep_cuda::complex_value_fp64 *>(
          device_gradient_fp64.get()),
      eps, mu, static_cast<double *>(device_mixed_output.get()), 256);
  // Exercise both a partial first warp and a multi-warp second-stage
  // reduction. This catches participation-mask mistakes in the cooperative
  // VJP reduction independently of the production 256-thread launch.
  const int narrow_threads = two_dimensional ? 16 : 64;
  meep_cuda::near2far_adjoint_fp32(
      dimension,
      static_cast<const meep_cuda::near2far_adjoint_source_fp32 *>(
          device_sources_fp32.get()),
      source_count,
      static_cast<const meep_cuda::cartesian_point_fp32 *>(
          device_targets_fp32.get()),
      target_count, static_cast<const float *>(device_frequencies_fp32.get()),
      frequency_count,
      static_cast<const meep_cuda::near2far_periodic_copy_fp32 *>(
          device_copies_fp32.get()),
      copy_count,
      static_cast<const meep_cuda::complex_value_fp32 *>(
          device_gradient_fp32.get()),
      static_cast<float>(eps), static_cast<float>(mu),
      static_cast<double *>(device_narrow_fast_output.get()),
      static_cast<double *>(device_narrow_l1.get()), narrow_threads, 0.0f);
  for (std::size_t target_start = 0; target_start < target_count;
       ++target_start) {
    const auto *const tiled_targets_fp32 =
        static_cast<const meep_cuda::cartesian_point_fp32 *>(
            device_targets_fp32.get()) + target_start;
    const auto *const tiled_targets_fp64 =
        static_cast<const meep_cuda::cartesian_point_fp64 *>(
            device_targets_fp64.get()) + target_start;
    const auto *const tiled_gradient_fp32 =
        static_cast<const meep_cuda::complex_value_fp32 *>(
            device_gradient_fp32.get()) +
        target_start * frequency_count * 6;
    const auto *const tiled_gradient_fp64 =
        static_cast<const meep_cuda::complex_value_fp64 *>(
            device_gradient_fp64.get()) +
        target_start * frequency_count * 6;
    const bool accumulate = target_start != 0;
    meep_cuda::near2far_adjoint_fp32(
        dimension,
        static_cast<const meep_cuda::near2far_adjoint_source_fp32 *>(
            device_sources_fp32.get()),
        source_count, tiled_targets_fp32, 1,
        static_cast<const float *>(device_frequencies_fp32.get()),
        frequency_count,
        static_cast<const meep_cuda::near2far_periodic_copy_fp32 *>(
            device_copies_fp32.get()),
        copy_count, tiled_gradient_fp32, static_cast<float>(eps),
        static_cast<float>(mu),
        static_cast<double *>(device_tiled_fast_output.get()),
        static_cast<double *>(device_tiled_l1.get()), 256, 0.0f, 0.0f,
        0.0f, accumulate);
    meep_cuda::near2far_adjoint_mixed_fp32(
        dimension,
        static_cast<const meep_cuda::near2far_adjoint_source_fp64 *>(
            device_sources_fp64.get()),
        source_count, tiled_targets_fp64, 1,
        static_cast<const double *>(device_frequencies_fp64.get()),
        frequency_count,
        static_cast<const meep_cuda::near2far_periodic_copy_fp64 *>(
            device_copies_fp64.get()),
        copy_count, tiled_gradient_fp64, eps, mu,
        static_cast<double *>(device_tiled_mixed_output.get()), 256, 0.0,
        0.0, accumulate);
  }
  meep_cuda::synchronize();
  meep_cuda::copy_to_host(fast_output.data(), device_fast_output.get(),
                          fast_output.size() * sizeof(double));
  meep_cuda::copy_to_host(mixed_output.data(), device_mixed_output.get(),
                          mixed_output.size() * sizeof(double));
  meep_cuda::copy_to_host(absolute_l1.data(), device_l1.get(),
                          absolute_l1.size() * sizeof(double));
  meep_cuda::copy_to_host(
      tiled_fast_output.data(), device_tiled_fast_output.get(),
      tiled_fast_output.size() * sizeof(double));
  meep_cuda::copy_to_host(
      tiled_mixed_output.data(), device_tiled_mixed_output.get(),
      tiled_mixed_output.size() * sizeof(double));
  meep_cuda::copy_to_host(
      tiled_absolute_l1.data(), device_tiled_l1.get(),
      tiled_absolute_l1.size() * sizeof(double));
  meep_cuda::copy_to_host(
      narrow_fast_output.data(), device_narrow_fast_output.get(),
      narrow_fast_output.size() * sizeof(double));
  meep_cuda::copy_to_host(
      narrow_absolute_l1.data(), device_narrow_l1.get(),
      narrow_absolute_l1.size() * sizeof(double));
  double scale = 1.0e-300;
  double fast_error = 0.0;
  double mixed_error = 0.0;
  for (std::size_t work = 0; work < reference.size(); ++work) {
    const std::complex<double> fast(fast_output[2 * work],
                                    fast_output[2 * work + 1]);
    const std::complex<double> mixed(mixed_output[2 * work],
                                     mixed_output[2 * work + 1]);
    const std::complex<double> tiled_fast(tiled_fast_output[2 * work],
                                          tiled_fast_output[2 * work + 1]);
    const std::complex<double> tiled_mixed(
        tiled_mixed_output[2 * work], tiled_mixed_output[2 * work + 1]);
    const std::complex<double> narrow_fast(
        narrow_fast_output[2 * work], narrow_fast_output[2 * work + 1]);
    scale = std::max(scale, std::abs(reference[work]));
    fast_error = std::max(fast_error, std::abs(fast - reference[work]));
    mixed_error =
        std::max(mixed_error, std::abs(mixed - reference[work]));
    fast_error =
        std::max(fast_error, std::abs(tiled_fast - reference[work]));
    fast_error =
        std::max(fast_error, std::abs(narrow_fast - reference[work]));
    mixed_error =
        std::max(mixed_error, std::abs(tiled_mixed - reference[work]));
    if (!std::isfinite(absolute_l1[work]) ||
        absolute_l1[work] + 1.0e-6 * std::max(absolute_l1[work], 1.0) <
            std::abs(fast) ||
        !std::isfinite(tiled_absolute_l1[work]) ||
        tiled_absolute_l1[work] +
                1.0e-6 * std::max(tiled_absolute_l1[work], 1.0) <
            std::abs(tiled_fast) ||
        !std::isfinite(narrow_absolute_l1[work]) ||
        narrow_absolute_l1[work] +
                1.0e-6 * std::max(narrow_absolute_l1[work], 1.0) <
            std::abs(narrow_fast))
      return false;
  }
  std::cout << "near2far_adjoint_" << (two_dimensional ? "2d" : "3d")
            << " fast_relative_error=" << fast_error / scale
            << " mixed_relative_error=" << mixed_error / scale << '\n';
  return fast_error <= (two_dimensional ? 8.0e-4 : 3.0e-4) * scale &&
         mixed_error <= 5.0e-8 * scale;
}

bool run_near2far_adjoint_cylindrical_case() {
  constexpr std::size_t source_count = 6;
  constexpr std::size_t target_count = 2;
  constexpr std::size_t frequency_count = 1;
  constexpr std::size_t copy_count = 2;
  constexpr double eps = 2.25;
  constexpr double mu = 1.44;
  constexpr double mode = -2.0;
  const std::vector<double> frequencies_fp64 = {0.37};
  const std::vector<meep_cuda::cartesian_point_fp64> targets_fp64 = {
      {2.4, -1.1, 1.7}, {-2.2, 1.7, -1.3}};
  const std::vector<meep_cuda::near2far_periodic_copy_fp64> copies_fp64 = {
      {{0.0, 0.0, 0.0}, {1.0, 0.0}},
      {{0.08, 0.0, -0.13}, {std::cos(0.29), std::sin(0.29)}}};
  std::vector<meep_cuda::near2far_adjoint_source_fp64> sources_fp64;
  for (std::size_t source = 0; source < source_count; ++source)
    sources_fp64.push_back(
        {{0.41 + 0.037 * source, 0.0, -0.32 + 0.09 * source},
         {0.17 + 0.021 * source, -0.09 + 0.013 * source},
         static_cast<int>(source % 3), source < 3 ? 1 : 0});
  std::vector<meep_cuda::complex_value_fp64> gradient_fp64;
  for (std::size_t target = 0; target < target_count; ++target)
    for (int component = 0; component < 6; ++component)
      gradient_fp64.push_back(
          {0.12 + 0.019 * target + 0.007 * component,
           -0.08 + 0.013 * target - 0.005 * component});
  std::vector<std::complex<double> > reference(source_count);
  for (std::size_t source = 0; source < source_count; ++source)
    for (std::size_t target = 0; target < target_count; ++target)
      for (const auto &copy : copies_fp64) {
        const meep_cuda::cartesian_point_fp64 shifted = {
            sources_fp64[source].point.x + copy.displacement.x, 0.0,
            sources_fp64[source].point.z + copy.displacement.z};
        const auto fields = near2far_greencyl_reference(
            targets_fp64[target], frequencies_fp64[0], eps, mu, shifted,
            sources_fp64[source].direction,
            sources_fp64[source].electric != 0,
            std::complex<double>(sources_fp64[source].amplitude.real,
                                 sources_fp64[source].amplitude.imag),
            mode, 1.0e-9);
        const std::complex<double> phase(copy.phase.real, copy.phase.imag);
        for (int component = 0; component < 6; ++component) {
          const auto &gradient = gradient_fp64[6 * target + component];
          reference[source] +=
              fields[component] * phase *
              std::complex<double>(gradient.real, gradient.imag);
        }
      }

  std::vector<meep_cuda::near2far_adjoint_source_fp32> sources_fp32;
  for (const auto &source : sources_fp64)
    sources_fp32.push_back(
        {{static_cast<float>(source.point.x), 0.0f,
          static_cast<float>(source.point.z)},
         {static_cast<float>(source.amplitude.real),
          static_cast<float>(source.amplitude.imag)},
         source.direction, source.electric});
  std::vector<meep_cuda::cartesian_point_fp32> targets_fp32;
  for (const auto &target : targets_fp64)
    targets_fp32.push_back(
        {static_cast<float>(target.x), static_cast<float>(target.y),
         static_cast<float>(target.z)});
  const std::vector<float> frequencies_fp32 = {
      static_cast<float>(frequencies_fp64[0])};
  std::vector<meep_cuda::near2far_periodic_copy_fp32> copies_fp32;
  for (const auto &copy : copies_fp64)
    copies_fp32.push_back(
        {{static_cast<float>(copy.displacement.x), 0.0f,
          static_cast<float>(copy.displacement.z)},
         {static_cast<float>(copy.phase.real),
          static_cast<float>(copy.phase.imag)}});
  std::vector<meep_cuda::complex_value_fp32> gradient_fp32;
  for (const auto &gradient : gradient_fp64)
    gradient_fp32.push_back(
        {static_cast<float>(gradient.real),
         static_cast<float>(gradient.imag)});
  auto device_sources_fp32 = copy_vector_to_device(sources_fp32);
  auto device_sources_fp64 = copy_vector_to_device(sources_fp64);
  auto device_targets_fp32 = copy_vector_to_device(targets_fp32);
  auto device_targets_fp64 = copy_vector_to_device(targets_fp64);
  auto device_frequencies_fp32 = copy_vector_to_device(frequencies_fp32);
  auto device_frequencies_fp64 = copy_vector_to_device(frequencies_fp64);
  auto device_copies_fp32 = copy_vector_to_device(copies_fp32);
  auto device_copies_fp64 = copy_vector_to_device(copies_fp64);
  auto device_gradient_fp32 = copy_vector_to_device(gradient_fp32);
  auto device_gradient_fp64 = copy_vector_to_device(gradient_fp64);
  std::vector<double> fast_output(2 * source_count, 0.0);
  std::vector<double> mixed_output(2 * source_count, 0.0);
  std::vector<double> absolute_l1(source_count, 0.0);
  auto device_fast_output = copy_vector_to_device(fast_output);
  auto device_mixed_output = copy_vector_to_device(mixed_output);
  auto device_l1 = copy_vector_to_device(absolute_l1);
  std::vector<double> tiled_fast_output(2 * source_count, 0.0);
  std::vector<double> tiled_mixed_output(2 * source_count, 0.0);
  std::vector<double> tiled_absolute_l1(source_count, 0.0);
  auto device_tiled_fast_output =
      copy_vector_to_device(tiled_fast_output);
  auto device_tiled_mixed_output =
      copy_vector_to_device(tiled_mixed_output);
  auto device_tiled_l1 = copy_vector_to_device(tiled_absolute_l1);
  meep_cuda::near2far_adjoint_fp32(
      meep_cuda::near2far_cartesian_dimension::cylindrical,
      static_cast<const meep_cuda::near2far_adjoint_source_fp32 *>(
          device_sources_fp32.get()),
      source_count,
      static_cast<const meep_cuda::cartesian_point_fp32 *>(
          device_targets_fp32.get()),
      target_count, static_cast<const float *>(device_frequencies_fp32.get()),
      frequency_count,
      static_cast<const meep_cuda::near2far_periodic_copy_fp32 *>(
          device_copies_fp32.get()),
      copy_count,
      static_cast<const meep_cuda::complex_value_fp32 *>(
          device_gradient_fp32.get()),
      static_cast<float>(eps), static_cast<float>(mu),
      static_cast<double *>(device_fast_output.get()),
      static_cast<double *>(device_l1.get()), 128, 0.0f,
      static_cast<float>(mode), 2.5e-5f);
  meep_cuda::near2far_adjoint_mixed_fp32(
      meep_cuda::near2far_cartesian_dimension::cylindrical,
      static_cast<const meep_cuda::near2far_adjoint_source_fp64 *>(
          device_sources_fp64.get()),
      source_count,
      static_cast<const meep_cuda::cartesian_point_fp64 *>(
          device_targets_fp64.get()),
      target_count,
      static_cast<const double *>(device_frequencies_fp64.get()),
      frequency_count,
      static_cast<const meep_cuda::near2far_periodic_copy_fp64 *>(
          device_copies_fp64.get()),
      copy_count,
      static_cast<const meep_cuda::complex_value_fp64 *>(
          device_gradient_fp64.get()),
      eps, mu, static_cast<double *>(device_mixed_output.get()), 128, mode,
      1.0e-8);
  bool first_tile = true;
  for (std::size_t target = 0; target < target_count; ++target)
    for (std::size_t copy = 0; copy < copy_count; ++copy) {
      const bool accumulate = !first_tile;
      meep_cuda::near2far_adjoint_fp32(
          meep_cuda::near2far_cartesian_dimension::cylindrical,
          static_cast<const meep_cuda::near2far_adjoint_source_fp32 *>(
              device_sources_fp32.get()),
          source_count,
          static_cast<const meep_cuda::cartesian_point_fp32 *>(
              device_targets_fp32.get()) + target,
          1, static_cast<const float *>(device_frequencies_fp32.get()),
          frequency_count,
          static_cast<const meep_cuda::near2far_periodic_copy_fp32 *>(
              device_copies_fp32.get()) + copy,
          1,
          static_cast<const meep_cuda::complex_value_fp32 *>(
              device_gradient_fp32.get()) + target * 6,
          static_cast<float>(eps), static_cast<float>(mu),
          static_cast<double *>(device_tiled_fast_output.get()),
          static_cast<double *>(device_tiled_l1.get()), 128, 0.0f,
          static_cast<float>(mode), 2.5e-5f, accumulate);
      meep_cuda::near2far_adjoint_mixed_fp32(
          meep_cuda::near2far_cartesian_dimension::cylindrical,
          static_cast<const meep_cuda::near2far_adjoint_source_fp64 *>(
              device_sources_fp64.get()),
          source_count,
          static_cast<const meep_cuda::cartesian_point_fp64 *>(
              device_targets_fp64.get()) + target,
          1, static_cast<const double *>(device_frequencies_fp64.get()),
          frequency_count,
          static_cast<const meep_cuda::near2far_periodic_copy_fp64 *>(
              device_copies_fp64.get()) + copy,
          1,
          static_cast<const meep_cuda::complex_value_fp64 *>(
              device_gradient_fp64.get()) + target * 6,
          eps, mu,
          static_cast<double *>(device_tiled_mixed_output.get()), 128,
          mode, 1.0e-8, accumulate);
      first_tile = false;
    }
  meep_cuda::synchronize();
  meep_cuda::copy_to_host(fast_output.data(), device_fast_output.get(),
                          fast_output.size() * sizeof(double));
  meep_cuda::copy_to_host(mixed_output.data(), device_mixed_output.get(),
                          mixed_output.size() * sizeof(double));
  meep_cuda::copy_to_host(absolute_l1.data(), device_l1.get(),
                          absolute_l1.size() * sizeof(double));
  meep_cuda::copy_to_host(
      tiled_fast_output.data(), device_tiled_fast_output.get(),
      tiled_fast_output.size() * sizeof(double));
  meep_cuda::copy_to_host(
      tiled_mixed_output.data(), device_tiled_mixed_output.get(),
      tiled_mixed_output.size() * sizeof(double));
  meep_cuda::copy_to_host(
      tiled_absolute_l1.data(), device_tiled_l1.get(),
      tiled_absolute_l1.size() * sizeof(double));
  double scale = 1.0e-300;
  double fast_error = 0.0;
  double mixed_error = 0.0;
  for (std::size_t work = 0; work < source_count; ++work) {
    const std::complex<double> fast(fast_output[2 * work],
                                    fast_output[2 * work + 1]);
    const std::complex<double> mixed(mixed_output[2 * work],
                                     mixed_output[2 * work + 1]);
    const std::complex<double> tiled_fast(
        tiled_fast_output[2 * work], tiled_fast_output[2 * work + 1]);
    const std::complex<double> tiled_mixed(
        tiled_mixed_output[2 * work], tiled_mixed_output[2 * work + 1]);
    scale = std::max(scale, std::abs(reference[work]));
    fast_error = std::max(fast_error, std::abs(fast - reference[work]));
    mixed_error =
        std::max(mixed_error, std::abs(mixed - reference[work]));
    fast_error = std::max(
        fast_error, std::abs(tiled_fast - reference[work]));
    mixed_error = std::max(
        mixed_error, std::abs(tiled_mixed - reference[work]));
    if (!std::isfinite(absolute_l1[work]) ||
        absolute_l1[work] + 1.0e-6 * std::max(absolute_l1[work], 1.0) <
            std::abs(fast) ||
        !std::isfinite(tiled_absolute_l1[work]) ||
        tiled_absolute_l1[work] +
                1.0e-6 * std::max(tiled_absolute_l1[work], 1.0) <
            std::abs(tiled_fast))
      return false;
  }
  std::cout << "near2far_adjoint_cylindrical fast_relative_error="
            << fast_error / scale
            << " mixed_relative_error=" << mixed_error / scale << '\n';
  return fast_error <= 1.0e-3 * std::max(scale, 1.0) &&
         mixed_error <= 5.0e-8 * std::max(scale, 1.0);
}

bool run_near2far_mixed_far_distance_case() {
  const std::vector<meep_cuda::cartesian_point_fp64> sources = {
      {0.25, 0.0, 0.0}, {-0.25, 0.0, 0.0}};
  const std::vector<float> dft = {1.0f, 0.0f, -1.0f, 0.0f};
  const std::vector<meep_cuda::cartesian_point_fp64> targets = {
      {1.0e8, 0.0, 0.0}};
  const std::vector<double> frequencies = {0.3};
  const std::vector<meep_cuda::near2far_periodic_copy_fp64> copies = {
      {{0.0, 0.0, 0.0}, {1.0, 0.0}}};
  auto device_sources = copy_vector_to_device(sources);
  auto device_dft = copy_vector_to_device(dft);
  auto device_targets = copy_vector_to_device(targets);
  auto device_frequencies = copy_vector_to_device(frequencies);
  auto device_copies = copy_vector_to_device(copies);
  const std::vector<meep_cuda::near2far_operation_mixed_fp32> operations = {
      {static_cast<const float *>(device_dft.get()),
       static_cast<const meep_cuda::cartesian_point_fp64 *>(
           device_sources.get()),
       sources.size(), frequencies.size(), 1, true}};
  auto device_operations = copy_vector_to_device(operations);
  std::vector<double> partials(12, 0.0);
  std::vector<double> observed(12, 0.0);
  auto device_partials = copy_vector_to_device(partials);
  auto device_output = copy_vector_to_device(observed);
  meep_cuda::near2far_3d_mixed_fp32(
      static_cast<const meep_cuda::near2far_operation_mixed_fp32 *>(
          device_operations.get()),
      operations.size(),
      static_cast<const meep_cuda::cartesian_point_fp64 *>(
          device_targets.get()),
      targets.size(), static_cast<const double *>(device_frequencies.get()),
      frequencies.size(), 0,
      static_cast<const meep_cuda::near2far_periodic_copy_fp64 *>(
          device_copies.get()),
      copies.size(), 1.0, 1.0, 1,
      static_cast<double *>(device_partials.get()),
      static_cast<double *>(device_output.get()));
  meep_cuda::synchronize();
  meep_cuda::copy_to_host(observed.data(), device_output.get(),
                          observed.size() * sizeof(double));

  std::array<std::complex<double>, 6> reference = {};
  for (std::size_t point = 0; point < sources.size(); ++point) {
    const auto fields = near2far_green3d_reference(
        targets[0], frequencies[0], 1.0, 1.0, sources[point], 1, true,
        std::complex<double>(dft[2 * point], dft[2 * point + 1]));
    for (int component = 0; component < 6; ++component)
      reference[component] += fields[component];
  }
  double reference_norm = 0.0;
  double error_norm = 0.0;
  for (int component = 0; component < 6; ++component) {
    const std::complex<double> value(observed[2 * component],
                                     observed[2 * component + 1]);
    reference_norm += std::norm(reference[component]);
    error_norm += std::norm(value - reference[component]);
  }
  const double relative_error =
      std::sqrt(error_norm / std::max(reference_norm, 1.0e-300));
  std::cout << "near2far_3d_mixed_far_distance reference_norm="
            << std::sqrt(reference_norm)
            << " relative_error=" << relative_error << '\n';
  return reference_norm > 0.0 && relative_error <= 2.0e-6;
}

bool run_near2far_subnormal_envelope_case() {
  constexpr double frequency = 8.5136750370755e-33;
  constexpr double distance = 2.4302246106229e12;
  const std::vector<float> dft = {1.0f, 0.0f};
  auto device_dft = copy_vector_to_device(dft);

  const std::vector<meep_cuda::cartesian_point_fp32> sources_fp32 = {
      {0.0f, 0.0f, 0.0f}};
  const std::vector<meep_cuda::cartesian_point_fp32> targets_fp32 = {
      {static_cast<float>(distance), 0.0f, 0.0f}};
  const std::vector<float> frequencies_fp32 = {
      static_cast<float>(frequency)};
  const std::vector<meep_cuda::near2far_periodic_copy_fp32> copies_fp32 = {
      {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f}}};
  auto device_sources_fp32 = copy_vector_to_device(sources_fp32);
  auto device_targets_fp32 = copy_vector_to_device(targets_fp32);
  auto device_frequencies_fp32 = copy_vector_to_device(frequencies_fp32);
  auto device_copies_fp32 = copy_vector_to_device(copies_fp32);
  const std::vector<meep_cuda::near2far_operation_fp32> operations_fp32 = {
      {static_cast<const float *>(device_dft.get()),
       static_cast<const meep_cuda::cartesian_point_fp32 *>(
           device_sources_fp32.get()),
       1, 1, 0, true}};
  auto device_operations_fp32 = copy_vector_to_device(operations_fp32);
  std::vector<float> partials_fp32(13, 0.0f);
  std::vector<double> output_fp32(12, 0.0);
  std::vector<double> evidence(1, 0.0);
  auto device_partials_fp32 = copy_vector_to_device(partials_fp32);
  auto device_output_fp32 = copy_vector_to_device(output_fp32);
  auto device_evidence = copy_vector_to_device(evidence);
  meep_cuda::near2far_3d_fp32(
      static_cast<const meep_cuda::near2far_operation_fp32 *>(
          device_operations_fp32.get()),
      1,
      static_cast<const meep_cuda::cartesian_point_fp32 *>(
          device_targets_fp32.get()),
      1, static_cast<const float *>(device_frequencies_fp32.get()), 1, 0,
      static_cast<const meep_cuda::near2far_periodic_copy_fp32 *>(
          device_copies_fp32.get()),
      1, 1.0f, 1.0f, 1,
      static_cast<float *>(device_partials_fp32.get()),
      static_cast<double *>(device_output_fp32.get()),
      static_cast<double *>(device_evidence.get()));
  meep_cuda::synchronize();
  meep_cuda::copy_to_host(
      evidence.data(), device_evidence.get(), sizeof(double));
  meep_cuda::copy_to_host(
      output_fp32.data(), device_output_fp32.get(),
      output_fp32.size() * sizeof(double));
  if (!std::isinf(evidence[0]) || evidence[0] < 0.0) return false;

  const std::vector<meep_cuda::cartesian_point_fp64> sources_fp64 = {
      {0.0, 0.0, 0.0}};
  const std::vector<meep_cuda::cartesian_point_fp64> targets_fp64 = {
      {distance, 0.0, 0.0}};
  const std::vector<double> frequencies_fp64 = {frequency};
  const std::vector<meep_cuda::near2far_periodic_copy_fp64> copies_fp64 = {
      {{0.0, 0.0, 0.0}, {1.0, 0.0}}};
  auto device_sources_fp64 = copy_vector_to_device(sources_fp64);
  auto device_targets_fp64 = copy_vector_to_device(targets_fp64);
  auto device_frequencies_fp64 = copy_vector_to_device(frequencies_fp64);
  auto device_copies_fp64 = copy_vector_to_device(copies_fp64);
  const std::vector<meep_cuda::near2far_operation_mixed_fp32>
      operations_mixed = {
          {static_cast<const float *>(device_dft.get()),
           static_cast<const meep_cuda::cartesian_point_fp64 *>(
               device_sources_fp64.get()),
           1, 1, 0, true}};
  auto device_operations_mixed = copy_vector_to_device(operations_mixed);
  std::vector<double> partials_mixed(12, 0.0);
  std::vector<double> output_mixed(12, 0.0);
  auto device_partials_mixed = copy_vector_to_device(partials_mixed);
  auto device_output_mixed = copy_vector_to_device(output_mixed);
  meep_cuda::near2far_3d_mixed_fp32(
      static_cast<const meep_cuda::near2far_operation_mixed_fp32 *>(
          device_operations_mixed.get()),
      1,
      static_cast<const meep_cuda::cartesian_point_fp64 *>(
          device_targets_fp64.get()),
      1, static_cast<const double *>(device_frequencies_fp64.get()), 1, 0,
      static_cast<const meep_cuda::near2far_periodic_copy_fp64 *>(
          device_copies_fp64.get()),
      1, 1.0, 1.0, 1,
      static_cast<double *>(device_partials_mixed.get()),
      static_cast<double *>(device_output_mixed.get()));
  meep_cuda::synchronize();
  meep_cuda::copy_to_host(
      output_mixed.data(), device_output_mixed.get(),
      output_mixed.size() * sizeof(double));

  const auto reference = near2far_green3d_reference(
      targets_fp64[0], frequency, 1.0, 1.0, sources_fp64[0], 0, true,
      std::complex<double>(1.0, 0.0));
  double reference_maximum = 0.0;
  double fast_error_maximum = 0.0;
  double mixed_error_maximum = 0.0;
  for (int component = 0; component < 6; ++component) {
    const std::complex<double> fast(
        output_fp32[2 * component], output_fp32[2 * component + 1]);
    const std::complex<double> mixed(
        output_mixed[2 * component], output_mixed[2 * component + 1]);
    reference_maximum =
        std::max(reference_maximum, std::abs(reference[component]));
    fast_error_maximum = std::max(
        fast_error_maximum, std::abs(fast - reference[component]));
    mixed_error_maximum = std::max(
        mixed_error_maximum, std::abs(mixed - reference[component]));
  }
  std::cout << "near2far_3d_subnormal_envelope evidence=" << evidence[0]
            << " fast_abs_error=" << fast_error_maximum
            << " mixed_abs_error=" << mixed_error_maximum << '\n';
  return std::abs(reference[0].imag()) > 2.0e-7 &&
         std::abs(reference[0].imag()) < 2.2e-7 &&
         fast_error_maximum > 4.0e-8 &&
         mixed_error_maximum <=
             5.0e-7 * reference_maximum + 1.0e-18;
}

} // namespace

int main() {
  std::string diagnostic;
  if (!meep_cuda::runtime_available(&diagnostic)) {
    std::cout << "SKIP: CUDA kernel compiled, but runtime is unavailable: " << diagnostic << '\n';
    return 77;
  }

  const auto devices = meep_cuda::enumerate_devices();
  const int requested_ordinal = requested_device_ordinal();
  const meep_cuda::device_info *selected = nullptr;
  for (const auto &device : devices) {
    if ((requested_ordinal < 0 || device.ordinal == requested_ordinal) &&
        meep_cuda::device_compatible(device)) {
      selected = &device;
      break;
    }
  }
  if (!selected) {
    std::cerr << "FAIL: requested CUDA device is absent or incompatible with "
              << meep_cuda::compiled_architectures() << '\n';
    return 1;
  }

  meep_cuda::select_device(selected->ordinal);
  std::cout << "device=" << selected->name << " ordinal=" << selected->ordinal
            << " compute=" << selected->compute_major << '.' << selected->compute_minor
            << " sms=" << selected->multiprocessor_count
            << " memory=" << selected->global_memory_bytes << '\n';
  std::cout << "driver_version=" << meep_cuda::driver_version()
            << " runtime_version=" << meep_cuda::runtime_version() << '\n';

  if (!run_event_lifecycle_case()) {
    std::cerr << "FAIL: CUDA event create/record/synchronize/re-record "
                 "lifecycle is incorrect\n";
    return 1;
  }

  const std::vector<test_case> cases = {
      {"empty", false, true, true, 3, 11, 0, 64},
      {"one-element", false, true, true, 3, 11, 1, 64},
      {"contiguous-255", false, true, true, 3, 11, 255, 128},
      {"contiguous-256", false, true, true, 3, 11, 256, 256},
      {"contiguous-257", false, true, true, 3, 11, 257, 512},
      {"indexed-two-operands", true, true, true, 3, 11, 4096, 256},
      {"indexed-plus-only", true, true, false, -3, 0, 2048, 128},
      {"indexed-minus-only", true, false, true, 0, -11, 2048, 64},
  };

  for (const auto &test : cases) {
    if (!run_case(test)) {
      std::cerr << "FAIL: CUDA FP32 curl update differs from the CPU FP32 reference in "
                << test.name << '\n';
      return 1;
    }
  }

  for (int pml_f = 0; pml_f < 2; ++pml_f)
    for (int pml_u = 0; pml_u < 2; ++pml_u)
      for (int conductivity = 0; conductivity < 2; ++conductivity)
        if (!run_material_case(pml_f != 0, pml_u != 0,
                               conductivity != 0)) {
          std::cerr << "FAIL: CUDA FP32 material curl differs from the CPU "
                       "reference\n";
          return 1;
        }

  if (!run_phase_batched_curl_case()) {
    std::cerr << "FAIL: phase-batched CUDA FP32 curl differs from the "
                 "structured reference\n";
    return 1;
  }

  const std::vector<bfast_test_case> bfast_cases = {
      {"g1-only-bare-positive", true, false, 2, 0, false, false, false},
      {"g1-only-bare-negative", true, false, -2, 0, false, false, false},
      {"g2-only-bare-positive", false, true, 0, 3, false, false, false},
      {"g2-only-bare-negative", false, true, 0, -3, false, false, false},
      {"g1-g2-bare-mixed", true, true, 2, -3, false, false, false},
      {"g1-g2-bare-reverse", true, true, -2, 3, false, false, false},
      {"pml-f", true, true, 2, -3, true, false, false},
      {"pml-u", true, true, -2, 3, false, true, false},
      {"pml-f-u", true, true, 2, -3, true, true, false},
      {"conductivity-only", true, true, -2, 3, false, false, true},
      {"pml-f-conductivity", true, true, 2, -3, true, false, true}};
  for (const bfast_test_case &test : bfast_cases)
    for (int batched = 0; batched < 2; ++batched)
      if (!run_bfast_case(test, batched != 0)) {
        std::cerr << "FAIL: CUDA FP32 BFAST output differs from the CPU "
                     "reference in "
                  << test.name << " using "
                  << (batched ? "batched" : "single") << " launch\n";
        return 1;
      }

  for (int pml = 0; pml < 2; ++pml)
    for (int offdiagonal_count = 0; offdiagonal_count < 3;
         ++offdiagonal_count)
      for (int nonlinear = 0; nonlinear < 2; ++nonlinear)
        for (int inverse_present = 0; inverse_present < 2;
             ++inverse_present)
          for (int negative_strides = 0; negative_strides < 2;
               ++negative_strides)
            if (!run_update_eh_case(
                    pml != 0, offdiagonal_count, nonlinear != 0,
                    inverse_present != 0, negative_strides != 0)) {
              std::cerr << "FAIL: CUDA FP32 E/H update differs from the CPU "
                           "reference\n";
              return 1;
            }

  if (!run_phase_batched_update_eh_case()) {
    std::cerr << "FAIL: phase-batched CUDA FP32 E/H update differs from the "
                 "structured reference\n";
    return 1;
  }

  for (int offdiagonal_count = 0; offdiagonal_count < 3;
       ++offdiagonal_count)
    for (int drude = 0; drude < 2; ++drude)
      for (int negative_strides = 0; negative_strides < 2;
           ++negative_strides)
        for (int increment_state = 0; increment_state < 2;
             ++increment_state)
          for (int structured = 0; structured < 2; ++structured)
            if (!run_lorentzian_case(
                    offdiagonal_count, drude != 0,
                    negative_strides != 0, increment_state != 0,
                    structured != 0)) {
              std::cerr
                  << "FAIL: CUDA FP32 Lorentzian update differs from the "
                     "CPU reference\n";
              return 1;
            }

  for (int model = meep_cuda::detail::gyrotropic_lorentzian_fp32;
       model <= meep_cuda::detail::gyrotropic_saturated_fp32; ++model)
    for (int transverse_count = 0; transverse_count < 3;
         ++transverse_count)
      for (int negative_strides = 0; negative_strides < 2;
           ++negative_strides)
        if (!run_gyrotropic_case(
                model, transverse_count, negative_strides != 0)) {
          std::cerr << "FAIL: CUDA FP32 gyrotropic update differs from the "
                       "CPU reference\n";
          return 1;
        }

  if (!run_multilevel_case()) {
    std::cerr << "FAIL: CUDA FP32 multilevel population/polarization update "
                 "differs from the CPU reference\n";
    return 1;
  }

  if (!run_boundary_case()) {
    std::cerr << "FAIL: CUDA FP32 boundary gather/scatter differs from the "
                 "CPU snapshot reference\n";
    return 1;
  }
  if (!run_boundary_graph_case()) {
    std::cerr << "FAIL: CUDA-graph FP32 boundary gather/scatter differs from "
                 "the CPU snapshot reference\n";
    return 1;
  }
  if (!run_boundary_phase_graph_case()) {
    std::cerr << "FAIL: ordered CUDA boundary phase graph or event readiness "
                 "is incorrect\n";
    return 1;
  }
  if (!run_zero_boundary_case()) {
    std::cerr << "FAIL: direct CUDA FP32 zero-boundary path is incorrect\n";
    return 1;
  }
  if (!run_nonalias_boundary_case()) {
    std::cerr << "FAIL: direct nonalias CUDA FP32 boundary path is incorrect\n";
    return 1;
  }
  if (!run_source_case()) {
    std::cerr << "FAIL: CUDA FP32 static source differs from the CPU "
                 "reference\n";
    return 1;
  }
  if (!run_phase_batched_source_case()) {
    std::cerr << "FAIL: phase-batched CUDA FP32 source differs from the CPU "
                 "reference\n";
    return 1;
  }
  if (!run_finite_case()) {
    std::cerr << "FAIL: CUDA FP32 finite-result initialization or cumulative "
                 "scan is incorrect\n";
    return 1;
  }
  if (!run_squared_norm_case()) {
    std::cerr << "FAIL: CUDA FP32 contiguous/indexed complex squared norm "
                 "reduction mismatch\n";
    return 1;
  }
  if (!run_indexed_ldos_case()) {
    std::cerr << "FAIL: CUDA FP32 indexed LDOS reduction differs from the "
                 "FP32-product/FP64-sum reference\n";
    return 1;
  }
  if (!run_dft_case()) {
    std::cerr << "FAIL: CUDA FP32 DFT differs from the CPU reference\n";
    return 1;
  }
  if (!run_dft_wide_frequency_case()) {
    std::cerr << "FAIL: CUDA FP32 1-point x 513-frequency DFT differs from "
                 "the CPU reference or wrote outside its output\n";
    return 1;
  }
  if (!run_dft_many_point_case()) {
    std::cerr << "FAIL: CUDA FP32 513-point x 1-frequency DFT differs from "
                 "the CPU reference or wrote outside its output\n";
    return 1;
  }
  if (!run_dft_omega_case()) {
    std::cerr << "FAIL: CUDA FP32 device-phase DFT differs from the CPU "
                 "reference\n";
    return 1;
  }
  if (!run_dft_multi_monitor_batch_case()) {
    std::cerr << "FAIL: heterogeneous multi-monitor CUDA FP32 DFT batch "
                 "differs from independent references or wrote outside an "
                 "output\n";
    return 1;
  }
  if (!run_near2far_3d_case()) {
    std::cerr << "FAIL: deterministic CUDA 3D near-to-far transform differs "
                 "from the independent FP64 Green reference\n";
    return 1;
  }
  if (!run_near2far_2d_case()) {
    std::cerr << "FAIL: deterministic CUDA 2D near-to-far transform differs "
                 "from the independent FP64 Hankel reference\n";
    return 1;
  }
  if (!run_near2far_cylindrical_case()) {
    std::cerr << "FAIL: deterministic CUDA cylindrical near-to-far "
                 "quadrature differs from the independent FP64 azimuthal "
                 "reference\n";
    return 1;
  }
  if (!run_near2far_adjoint_cartesian_case(
          meep_cuda::near2far_cartesian_dimension::two) ||
      !run_near2far_adjoint_cartesian_case(
          meep_cuda::near2far_cartesian_dimension::three)) {
    std::cerr << "FAIL: CUDA adjoint Near2Far VJP differs from the "
                 "independent FP64 Green/dJ contraction reference\n";
    return 1;
  }
  if (!run_near2far_adjoint_cylindrical_case()) {
    std::cerr << "FAIL: CUDA cylindrical adjoint Near2Far VJP differs from "
                 "the independent FP64 azimuthal Green/dJ contraction "
                 "reference\n";
    return 1;
  }
  if (!run_near2far_mixed_far_distance_case()) {
    std::cerr << "FAIL: mixed-precision CUDA 3D near-to-far lost "
                 "far-distance phase information\n";
    return 1;
  }
  if (!run_near2far_subnormal_envelope_case()) {
    std::cerr << "FAIL: CUDA Near2Far finite-normal envelope evidence\n";
    return 1;
  }

  std::cout << "PASS: CUDA FP32 curl, BFAST, E/H, Lorentzian, gyrotropic, "
               "multilevel, boundary, source, finite scan, LDOS reduction, "
               "DFT updates, and Near2Far transforms passed\n";
  return 0;
}
