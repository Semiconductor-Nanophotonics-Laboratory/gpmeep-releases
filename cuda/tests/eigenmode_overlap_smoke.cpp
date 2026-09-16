#include "meep_cuda/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
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

int requested_device_ordinal() {
  const char *value = std::getenv("MEEP_GPU_DEVICE");
  if (!value || !value[0]) return -1;
  char *end = nullptr;
  const long ordinal = std::strtol(value, &end, 10);
  if (!end || *end != '\0' || ordinal < 0 ||
      ordinal > std::numeric_limits<int>::max())
    throw std::invalid_argument(
        "MEEP_GPU_DEVICE must be a nonnegative integer");
  return static_cast<int>(ordinal);
}

struct input {
  std::uint32_t output;
  std::size_t points;
  std::size_t frequencies;
  bool mode_flux;
  meep_cuda::complex_value_fp64 inverse_weight;
  std::vector<float> dft;
  std::vector<meep_cuda::complex_value_fp64> weighted_mode1;
  std::vector<meep_cuda::complex_value_fp64> mode2;
  std::vector<std::uint8_t> zero_normalization_divisors;
};

input make_input(std::uint32_t output, std::size_t points,
                 std::size_t frequencies, bool mode_flux,
                 std::size_t seed) {
  input value;
  value.output = output;
  value.points = points;
  value.frequencies = mode_flux ? frequencies : 0;
  value.mode_flux = mode_flux;
  value.inverse_weight = {0.75 + 0.03125 * seed,
                          -0.25 + 0.015625 * seed};
  value.weighted_mode1.resize(points);
  if (mode_flux) value.dft.resize(2 * points * frequencies);
  else value.mode2.resize(points);
  for (std::size_t point = 0; point < points; ++point) {
    const double x = static_cast<double>(point + 1 + 3 * seed);
    value.weighted_mode1[point] = {
        std::sin(0.031 * x) * (1.0 + 0.001 * point),
        std::cos(0.047 * x) * (0.5 - 0.0003 * point)};
    if (mode_flux)
      for (std::size_t frequency = 0; frequency < frequencies;
           ++frequency) {
        const std::size_t index = point * frequencies + frequency;
        value.dft[2 * index] = static_cast<float>(
            std::sin(0.019 * x * (frequency + 1)));
        value.dft[2 * index + 1] = static_cast<float>(
            std::cos(0.023 * x * (frequency + 2)));
      }
    else
      value.mode2[point] = {
          std::cos(0.017 * x) * (0.75 + 0.0002 * point),
          std::sin(0.029 * x) * (-0.25 + 0.0001 * point)};
  }
  return value;
}

std::vector<double> cpu_oracle(const std::vector<input> &inputs,
                               std::size_t frequency,
                               std::size_t output_count) {
  std::vector<double> output(2 * output_count, 0.0);
  for (const auto &value : inputs)
    for (std::size_t point = 0; point < value.points; ++point) {
      double rhs_real = 0.0;
      double rhs_imag = 0.0;
      if (value.mode_flux) {
        const std::size_t index = point * value.frequencies + frequency;
        const double dft_real = value.dft[2 * index];
        const double dft_imag = value.dft[2 * index + 1];
        rhs_real = dft_real * value.inverse_weight.real -
                   dft_imag * value.inverse_weight.imag;
        rhs_imag = dft_real * value.inverse_weight.imag +
                   dft_imag * value.inverse_weight.real;
        if (!value.zero_normalization_divisors.empty() &&
            value.zero_normalization_divisors[point] &&
            (rhs_real != 0.0 || rhs_imag != 0.0)) {
          const double zero = 0.0;
          rhs_real /= zero;
          rhs_imag /= zero;
        }
      }
      else {
        rhs_real = value.mode2[point].real;
        rhs_imag = value.mode2[point].imag;
      }
      const auto weighted = value.weighted_mode1[point];
      output[2 * value.output] +=
          weighted.real * rhs_real - weighted.imag * rhs_imag;
      output[2 * value.output + 1] +=
          weighted.real * rhs_imag + weighted.imag * rhs_real;
    }
  return output;
}

bool close_enough(const std::vector<double> &observed,
                  const std::vector<double> &expected,
                  const char *case_name) {
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (std::isnan(expected[index]) && std::isnan(observed[index]))
      continue;
    if (std::isinf(expected[index]) &&
        observed[index] == expected[index])
      continue;
    const double tolerance =
        2.0e-12 * std::max(1.0, std::abs(expected[index]));
    if (!std::isfinite(observed[index]) ||
        std::abs(observed[index] - expected[index]) > tolerance) {
      std::cerr << case_name << ": scalar " << index
                << " observed=" << observed[index]
                << " expected=" << expected[index]
                << " tolerance=" << tolerance << '\n';
      return false;
    }
  }
  return true;
}

bool run_case(const char *case_name, const std::vector<input> &inputs,
              std::size_t selected_frequency,
              std::size_t output_count =
                  meep_cuda::eigenmode_overlap_output_count) {
  std::vector<device_pointer> dft_devices;
  std::vector<device_pointer> mode1_devices;
  std::vector<device_pointer> mode2_devices;
  std::vector<device_pointer> zero_flag_devices;
  std::vector<meep_cuda::eigenmode_overlap_operation_fp32> operations;
  std::vector<std::uint32_t> block_map;
  dft_devices.reserve(inputs.size());
  mode1_devices.reserve(inputs.size());
  mode2_devices.reserve(inputs.size());
  zero_flag_devices.reserve(inputs.size());
  operations.reserve(inputs.size());
  std::size_t block_start = 0;
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    const input &value = inputs[index];
    dft_devices.push_back(copy_vector_to_device(value.dft));
    mode1_devices.push_back(copy_vector_to_device(value.weighted_mode1));
    mode2_devices.push_back(copy_vector_to_device(value.mode2));
    zero_flag_devices.push_back(
        copy_vector_to_device(value.zero_normalization_divisors));
    meep_cuda::eigenmode_overlap_operation_fp32 operation = {
        static_cast<const float *>(dft_devices.back().get()),
        static_cast<const meep_cuda::complex_value_fp64 *>(
            mode1_devices.back().get()),
        static_cast<const meep_cuda::complex_value_fp64 *>(
            mode2_devices.back().get()),
        static_cast<const std::uint8_t *>(
            zero_flag_devices.back().get()),
        value.points, value.frequencies, block_start, value.output,
        value.inverse_weight};
    operations.push_back(operation);
    const std::size_t blocks =
        value.points / meep_cuda::eigenmode_overlap_threads_per_block +
        (value.points %
             meep_cuda::eigenmode_overlap_threads_per_block !=
         0);
    for (std::size_t block = 0; block < blocks; ++block)
      block_map.push_back(static_cast<std::uint32_t>(index));
    block_start += blocks;
  }
  meep_cuda::validate_eigenmode_overlap_operations_fp32(
      operations.data(), block_map.data(), operations.size(),
      block_map.size(), selected_frequency, output_count);
  device_pointer device_operations = copy_vector_to_device(operations);
  device_pointer device_block_map = copy_vector_to_device(block_map);
  const std::size_t partial_capacity = std::min(
      block_map.size(), meep_cuda::eigenmode_overlap_partial_capacity);
  device_pointer device_partials(meep_cuda::allocate_device_bytes(
      2 * output_count * partial_capacity * sizeof(double)));
  device_pointer device_result(meep_cuda::allocate_device_bytes(
      2 * output_count * sizeof(double)));

  const std::vector<double> expected =
      cpu_oracle(inputs, selected_frequency, output_count);
  std::vector<double> observed(expected.size());
  std::vector<double> repeated(expected.size());
  meep_cuda::eigenmode_overlap_reduce_fp32(
      static_cast<const meep_cuda::eigenmode_overlap_operation_fp32 *>(
          device_operations.get()),
      static_cast<const std::uint32_t *>(device_block_map.get()),
      operations.size(), block_map.size(), selected_frequency,
      static_cast<double *>(device_partials.get()), partial_capacity,
      output_count,
      static_cast<double *>(device_result.get()));
  meep_cuda::copy_to_host(observed.data(), device_result.get(),
                          observed.size() * sizeof(double));
  meep_cuda::eigenmode_overlap_reduce_fp32(
      static_cast<const meep_cuda::eigenmode_overlap_operation_fp32 *>(
          device_operations.get()),
      static_cast<const std::uint32_t *>(device_block_map.get()),
      operations.size(), block_map.size(), selected_frequency,
      static_cast<double *>(device_partials.get()), partial_capacity,
      output_count,
      static_cast<double *>(device_result.get()));
  meep_cuda::copy_to_host(repeated.data(), device_result.get(),
                          repeated.size() * sizeof(double));
  meep_cuda::synchronize();
  if (std::memcmp(observed.data(), repeated.data(),
                  observed.size() * sizeof(double)) != 0) {
    std::cerr << case_name << ": repeated result was not bitwise stable\n";
    return false;
  }
  return close_enough(observed, expected, case_name);
}

} // namespace

int main() {
  std::string diagnostic;
  if (!meep_cuda::runtime_available(&diagnostic)) {
    std::cout << "SKIP: CUDA eigenmode overlap runtime unavailable: "
              << diagnostic << '\n';
    return 77;
  }
  const auto devices = meep_cuda::enumerate_devices();
  const int requested = requested_device_ordinal();
  const meep_cuda::device_info *selected = nullptr;
  for (const auto &device : devices)
    if ((requested < 0 || device.ordinal == requested) &&
        meep_cuda::device_compatible(device)) {
      selected = &device;
      break;
    }
  if (!selected) {
    std::cerr << "FAIL: requested CUDA device is absent or incompatible\n";
    return 1;
  }
  meep_cuda::select_device(selected->ordinal);

  const std::vector<input> mode_flux = {
      make_input(0, 523, 5, true, 1),
      make_input(1, 17, 5, true, 2),
      make_input(2, 781, 5, true, 3),
      make_input(3, 257, 5, true, 4),
      make_input(0, 91, 5, true, 5)};
  if (!run_case("mode-flux", mode_flux, 3)) return 1;

  const std::vector<input> mode_mode = {
      make_input(3, 601, 0, false, 7),
      make_input(0, 9, 0, false, 8),
      make_input(2, 1031, 0, false, 9),
      make_input(1, 255, 0, false, 10),
      make_input(3, 13, 0, false, 11)};
  if (!run_case("mode-mode-without-dft", mode_mode, 0)) return 1;

  std::vector<input> fused = mode_flux;
  for (std::uint32_t output = 0; output < 4; ++output)
    fused.push_back(make_input(output + 4, 129 + 37 * output, 0,
                               false, 21 + output));
  if (!run_case("fused-mode-flux-and-mode-mode", fused, 3,
                meep_cuda::eigenmode_overlap_max_output_count))
    return 1;

  input zero_weight = make_input(0, 3, 2, true, 12);
  zero_weight.zero_normalization_divisors = {1u, 1u, 0u};
  zero_weight.weighted_mode1[0] = {0.0, 0.0};
  zero_weight.weighted_mode1[1] = {0.0, 0.0};
  zero_weight.dft[2] = 0.0f;
  zero_weight.dft[3] = 0.0f;
  if (!run_case("zero-normalization-divisor", {zero_weight}, 1))
    return 1;

  const std::vector<input> capped_grid_stride = {
      make_input(2,
                 meep_cuda::eigenmode_overlap_partial_capacity *
                         meep_cuda::eigenmode_overlap_threads_per_block +
                     37,
                 0, false, 13)};
  if (!run_case("capped-grid-stride", capped_grid_stride, 0)) return 1;

  std::cout << "PASS: CUDA eigenmode overlap matches an FP64 CPU oracle "
               "for separate four-channel and fused eight-channel batches, "
               "complex inverse weights, multi-"
               "block tails, repeated-output accumulation, and the mode-"
               "mode path without reading DFT storage, zero-divisor IEEE "
               "semantics, and logical grids beyond the physical partial "
               "cap; repeated launches are bitwise stable\n";
  return 0;
}
