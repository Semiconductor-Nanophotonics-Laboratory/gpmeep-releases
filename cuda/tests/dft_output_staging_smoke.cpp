#include "meep_cuda/runtime.hpp"

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

float rounded_multiply(float left, float right) {
  volatile float result = left * right;
  return result;
}

float rounded_add(float left, float right) {
  volatile float result = left + right;
  return result;
}

float rounded_subtract(float left, float right) {
  volatile float result = left - right;
  return result;
}

float rounded_divide_by_zero(float value) {
  volatile float zero = 0.0f;
  volatile float result = value / zero;
  return result;
}

void condition_cpu(
    float source_real, float source_imaginary,
    meep_cuda::complex_value_fp32 inverse_stored_weight,
    float point_weight, bool zero_divisor,
    float &output_real, float &output_imaginary) {
  const float weighted_real = rounded_subtract(
      rounded_multiply(source_real, inverse_stored_weight.real),
      rounded_multiply(source_imaginary, inverse_stored_weight.imag));
  const float weighted_imaginary = rounded_add(
      rounded_multiply(source_real, inverse_stored_weight.imag),
      rounded_multiply(source_imaginary, inverse_stored_weight.real));
  if (zero_divisor && weighted_real == 0.0f &&
      weighted_imaginary == 0.0f) {
    output_real = 0.0f;
    output_imaginary = 0.0f;
  }
  else if (zero_divisor) {
    output_real = rounded_multiply(
        rounded_divide_by_zero(weighted_real), point_weight);
    output_imaginary = rounded_multiply(
        rounded_divide_by_zero(weighted_imaginary), point_weight);
  }
  else {
    output_real = rounded_multiply(weighted_real, point_weight);
    output_imaginary = rounded_multiply(weighted_imaginary, point_weight);
  }
}

struct staging_input {
  std::vector<float> source_real_imag;
  std::vector<std::ptrdiff_t> destinations;
  std::vector<float> point_weights;
  std::vector<std::uint8_t> zero_divisor_flags;
  std::size_t storage_point_count;
  std::size_t source_frequency_count;
  meep_cuda::complex_value_fp32 inverse_stored_weight;
};

staging_input make_input(
    std::size_t storage_point_count, std::size_t point_count,
    std::size_t source_frequency_count,
    std::vector<std::ptrdiff_t> destinations,
    std::vector<float> point_weights,
    std::vector<std::uint8_t> zero_divisor_flags,
    meep_cuda::complex_value_fp32 inverse_stored_weight,
    float seed) {
  std::vector<float> source(
      2 * storage_point_count * source_frequency_count,
      std::numeric_limits<float>::quiet_NaN());
  for (std::size_t point = 0; point < point_count; ++point)
    for (std::size_t frequency = 0; frequency < source_frequency_count;
         ++frequency)
      for (std::size_t reim = 0; reim < 2; ++reim) {
        const std::size_t scalar =
            2 * (point * source_frequency_count + frequency) + reim;
        const float magnitude =
            seed + static_cast<float>(scalar + 1) * 0.03125f;
        source[scalar] = (scalar % 3 == 0) ? -magnitude : magnitude;
      }
  return {source, destinations, point_weights, zero_divisor_flags,
          storage_point_count, source_frequency_count,
          inverse_stored_weight};
}

std::vector<float> cpu_oracle(
    const std::vector<meep_cuda::dft_output_staging_operation_fp32>
        &operations,
    std::size_t output_point_count, std::size_t tile_frequency_start,
    std::size_t tile_frequency_count, std::size_t frequency_capacity) {
  std::vector<float> output(
      2 * output_point_count * frequency_capacity, 0.0f);
  for (const auto &operation : operations)
    for (std::size_t point = 0; point < operation.point_count; ++point)
      for (std::size_t tile_frequency = 0;
           tile_frequency < tile_frequency_count; ++tile_frequency) {
        const std::size_t source_frequency =
            tile_frequency_start + tile_frequency;
        const std::size_t source_index =
            point * operation.source_frequency_count + source_frequency;
        float real;
        float imaginary;
        condition_cpu(
            operation.dft_real_imag[2 * source_index],
            operation.dft_real_imag[2 * source_index + 1],
            operation.inverse_stored_weight,
            operation.point_weights[point],
            operation.zero_divisor_flags &&
                operation.zero_divisor_flags[point] != 0,
            real, imaginary);
        const std::size_t packed_point =
            operation.output_point_offset + static_cast<std::size_t>(
                operation.destination_indices[point]);
        const std::size_t plane =
            tile_frequency * 2 * output_point_count;
        output[plane + packed_point] = real;
        output[plane + output_point_count + packed_point] = imaginary;
      }
  return output;
}

bool cpu_ieee_equal(const std::vector<float> &observed,
                    const std::vector<float> &expected,
                    const char *case_name) {
  if (observed.size() != expected.size()) return false;
  for (std::size_t index = 0; index < observed.size(); ++index) {
    if (std::isnan(observed[index]) && std::isnan(expected[index]))
      continue;
    if (std::memcmp(&observed[index], &expected[index], sizeof(float)) != 0) {
      std::cerr << case_name << ": CPU IEEE mismatch at scalar " << index
                << " observed=" << observed[index]
                << " expected=" << expected[index] << '\n';
      return false;
    }
  }
  return true;
}

bool bitwise_equal(const std::vector<float> &left,
                   const std::vector<float> &right,
                   const char *case_name) {
  if (left.size() != right.size()) return false;
  for (std::size_t index = 0; index < left.size(); ++index)
    if (std::memcmp(&left[index], &right[index], sizeof(float)) != 0) {
      std::cerr << case_name << ": repeat mismatch at scalar " << index
                << " first=" << left[index]
                << " second=" << right[index] << '\n';
      return false;
    }
  return true;
}

struct coverage {
  bool negative_zero = false;
  bool infinity = false;
  bool nan = false;
};

bool run_tile(
    const char *case_name, const std::vector<staging_input> &inputs,
    std::size_t tile_frequency_start, std::size_t tile_frequency_count,
    std::size_t frequency_capacity, coverage &observed_coverage) {
  std::vector<meep_cuda::dft_output_staging_operation_fp32>
      host_operations;
  std::vector<std::uint32_t> block_operation_indices;
  std::size_t output_point_count = 0;
  std::size_t block_start = 0;
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    const staging_input &input = inputs[index];
    const std::size_t point_count = input.destinations.size();
    host_operations.push_back(
        {input.source_real_imag.data(), input.destinations.data(),
         input.point_weights.data(), input.zero_divisor_flags.data(),
         input.storage_point_count, point_count,
         input.source_frequency_count, input.inverse_stored_weight,
         output_point_count, block_start});
    const std::size_t work = point_count * frequency_capacity;
    const std::size_t blocks =
        work /
            static_cast<std::size_t>(
                meep_cuda::dft_output_staging_threads_per_block_fp32) +
        (work %
             static_cast<std::size_t>(
                 meep_cuda::dft_output_staging_threads_per_block_fp32) !=
         0);
    for (std::size_t block = 0; block < blocks; ++block)
      block_operation_indices.push_back(static_cast<std::uint32_t>(index));
    output_point_count += point_count;
    block_start += blocks;
  }
  meep_cuda::validate_dft_output_staging_operations_fp32(
      host_operations.data(), block_operation_indices.data(),
      host_operations.size(), block_operation_indices.size(),
      output_point_count, frequency_capacity);
  meep_cuda::validate_dft_output_staging_tile_fp32(
      host_operations.data(), host_operations.size(),
      tile_frequency_start, tile_frequency_count, frequency_capacity);
  const std::vector<float> expected = cpu_oracle(
      host_operations, output_point_count, tile_frequency_start,
      tile_frequency_count, frequency_capacity);

  std::vector<device_pointer> device_sources;
  std::vector<device_pointer> device_destinations;
  std::vector<device_pointer> device_weights;
  std::vector<device_pointer> device_zero_flags;
  for (const auto &input : inputs) {
    device_sources.push_back(copy_vector_to_device(input.source_real_imag));
    device_destinations.push_back(copy_vector_to_device(input.destinations));
    device_weights.push_back(copy_vector_to_device(input.point_weights));
    device_zero_flags.push_back(
        copy_vector_to_device(input.zero_divisor_flags));
  }
  auto device_operation_image = host_operations;
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    device_operation_image[index].dft_real_imag =
        static_cast<const float *>(device_sources[index].get());
    device_operation_image[index].destination_indices =
        static_cast<const std::ptrdiff_t *>(
            device_destinations[index].get());
    device_operation_image[index].point_weights =
        static_cast<const float *>(device_weights[index].get());
    device_operation_image[index].zero_divisor_flags =
        static_cast<const std::uint8_t *>(device_zero_flags[index].get());
  }
  auto device_operations = copy_vector_to_device(device_operation_image);
  auto device_blocks = copy_vector_to_device(block_operation_indices);
  const std::vector<float> poison(expected.size(), 12345.25f);
  auto device_output = copy_vector_to_device(poison);

  meep_cuda::stage_dft_output_fp32(
      static_cast<const meep_cuda::dft_output_staging_operation_fp32 *>(
          device_operations.get()),
      static_cast<const std::uint32_t *>(device_blocks.get()),
      host_operations.size(), block_operation_indices.size(),
      static_cast<float *>(device_output.get()), output_point_count,
      tile_frequency_start, tile_frequency_count, frequency_capacity);
  meep_cuda::synchronize();
  std::vector<float> first(expected.size());
  meep_cuda::copy_to_host(
      first.data(), device_output.get(), first.size() * sizeof(float));
  if (!cpu_ieee_equal(first, expected, case_name)) return false;

  meep_cuda::copy_to_device(
      device_output.get(), poison.data(), poison.size() * sizeof(float));
  meep_cuda::stage_dft_output_fp32(
      static_cast<const meep_cuda::dft_output_staging_operation_fp32 *>(
          device_operations.get()),
      static_cast<const std::uint32_t *>(device_blocks.get()),
      host_operations.size(), block_operation_indices.size(),
      static_cast<float *>(device_output.get()), output_point_count,
      tile_frequency_start, tile_frequency_count, frequency_capacity);
  meep_cuda::synchronize();
  std::vector<float> second(expected.size());
  meep_cuda::copy_to_host(
      second.data(), device_output.get(), second.size() * sizeof(float));
  if (!cpu_ieee_equal(second, expected, case_name) ||
      !bitwise_equal(first, second, case_name))
    return false;

  for (float value : first) {
    observed_coverage.negative_zero =
        observed_coverage.negative_zero ||
        (value == 0.0f && std::signbit(value));
    observed_coverage.infinity =
        observed_coverage.infinity || std::isinf(value);
    observed_coverage.nan = observed_coverage.nan || std::isnan(value);
  }
  std::cout << "case=" << case_name
            << " operations=" << inputs.size()
            << " points=" << output_point_count
            << " tile_start=" << tile_frequency_start
            << " tile_count=" << tile_frequency_count
            << " capacity=" << frequency_capacity
            << " poison_overwritten=yes bitwise_repeatable=yes\n";
  return true;
}

} // namespace

int main() {
  std::string diagnostic;
  if (!meep_cuda::runtime_available(&diagnostic)) {
    std::cout << "SKIP: CUDA DFT output staging runtime unavailable: "
              << diagnostic << '\n';
    return 77;
  }
  const auto devices = meep_cuda::enumerate_devices();
  const int requested_ordinal = requested_device_ordinal();
  const meep_cuda::device_info *selected = nullptr;
  for (const auto &device : devices)
    if ((requested_ordinal < 0 || device.ordinal == requested_ordinal) &&
        meep_cuda::device_compatible(device)) {
      selected = &device;
      break;
    }
  if (!selected) {
    std::cerr << "FAIL: requested CUDA device is absent or incompatible with "
              << meep_cuda::compiled_architectures() << '\n';
    return 1;
  }
  meep_cuda::select_device(selected->ordinal);

  staging_input first = make_input(
      6, 4, 7, {3, 2, 1, 0},
      {1.0f, -0.0f, 0.5f, -2.0f}, {0u, 0u, 1u, 1u},
      {0.75f, -0.25f}, 0.125f);
  // A zero-divisor-marked exactly-zero complex sample must publish +0, not
  // poison, NaN, or a signed zero inherited from its interpolation weight.
  for (std::size_t frequency = 0;
       frequency < first.source_frequency_count; ++frequency) {
    const std::size_t scalar =
        2 * (3 * first.source_frequency_count + frequency);
    first.source_real_imag[scalar] = 0.0f;
    first.source_real_imag[scalar + 1] = 0.0f;
  }

  staging_input second = make_input(
      5, 3, 6, {2, 1, 0}, {1.25f, -0.75f, 2.0f},
      {0u, 0u, 0u}, {-0.5f, 0.375f}, -0.25f);
  const std::size_t infinity_scalar =
      2 * (0 * second.source_frequency_count + 2);
  second.source_real_imag[infinity_scalar] =
      std::numeric_limits<float>::infinity();
  second.source_real_imag[infinity_scalar + 1] = 1.0f;
  const std::size_t nan_scalar =
      2 * (1 * second.source_frequency_count + 3);
  second.source_real_imag[nan_scalar] =
      std::numeric_limits<float>::quiet_NaN();
  second.source_real_imag[nan_scalar + 1] = -1.0f;

  const std::vector<staging_input> inputs = {first, second};
  coverage observed;
  if (!run_tile("full-tile-planar-reverse-maps", inputs, 1, 4, 4,
                observed)) {
    std::cerr << "FAIL: full DFT output staging tile differs from the "
                 "explicit FP32 CPU oracle\n";
    return 1;
  }
  if (!run_tile("short-final-tile-zero-tail", inputs, 5, 1, 4,
                observed)) {
    std::cerr << "FAIL: short DFT output staging tile differs from the "
                 "explicit FP32 CPU oracle or retained poison\n";
    return 1;
  }

  // Force one operation to span multiple logical CUDA blocks.  The smaller
  // cases above intentionally stress unusual IEEE values, but their work
  // extents fit in one block and cannot catch a broken local_block offset.
  std::vector<std::ptrdiff_t> multi_block_destinations(130);
  std::vector<float> multi_block_weights(130);
  std::vector<std::uint8_t> multi_block_zero_flags(130, 0u);
  for (std::size_t point = 0; point < 130; ++point) {
    multi_block_destinations[point] =
        static_cast<std::ptrdiff_t>(129 - point);
    multi_block_weights[point] =
        point % 2 == 0 ? 1.0f : -0.5f;
  }
  const std::vector<staging_input> multi_block_inputs = {
      make_input(140, 130, 5, multi_block_destinations,
                 multi_block_weights, multi_block_zero_flags,
                 {0.625f, -0.125f}, 0.0625f)};
  if (!run_tile("multi-block-local-offset", multi_block_inputs, 1, 3, 3,
                observed)) {
    std::cerr << "FAIL: multi-block DFT output staging differs from the "
                 "explicit FP32 CPU oracle\n";
    return 1;
  }
  if (!observed.negative_zero || !observed.infinity || !observed.nan) {
    std::cerr << "FAIL: smoke vectors did not exercise negative zero, "
                 "infinity, and NaN outputs\n";
    return 1;
  }

  std::cout << "PASS: CUDA planar DFT output staging matches explicit-RN "
               "CPU semantics for full/short tiles, two disjoint packed "
               "operations, reverse maps, complex inverse weights, signed-"
               "zero weights, padded storage, divide-by-zero, Inf/NaN, "
               "multi-block local offsets, complete poison overwrite, and "
               "bitwise-repeatable launches\n";
  return 0;
}
