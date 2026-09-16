#include "meep_cuda/runtime.hpp"

#include <algorithm>
#include <chrono>
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

struct materialization_input {
  std::vector<float> source_real_imag;
  std::vector<std::ptrdiff_t> destinations;
  std::vector<float> point_weights;
  std::size_t storage_point_count;
  std::size_t source_frequency_count;
  std::size_t source_frequency_start;
  std::size_t selected_frequency_count;
  meep_cuda::complex_value_fp32 inverse_stored_weight;
  std::vector<std::uint8_t> zero_divisor_flags;
};

std::vector<float> cpu_oracle(
    const std::vector<meep_cuda::dft_materialization_operation_fp32>
        &operations,
    std::size_t output_point_count,
    std::size_t output_frequency_count) {
  std::vector<float> output(
      2 * output_point_count * output_frequency_count, 0.0f);
  for (const auto &operation : operations)
    for (std::size_t point = 0; point < operation.point_count; ++point)
      for (std::size_t selected_frequency = 0;
           selected_frequency < operation.selected_frequency_count;
           ++selected_frequency) {
        const std::size_t source_frequency =
            operation.source_frequency_start + selected_frequency;
        const std::size_t source_index =
            point * operation.source_frequency_count + source_frequency;
        const float source_real = operation.dft_real_imag[2 * source_index];
        const float source_imaginary =
            operation.dft_real_imag[2 * source_index + 1];
        const float weighted_real = rounded_subtract(
            rounded_multiply(
                source_real, operation.inverse_stored_weight.real),
            rounded_multiply(
                source_imaginary, operation.inverse_stored_weight.imag));
        const float weighted_imaginary = rounded_add(
            rounded_multiply(
                source_real, operation.inverse_stored_weight.imag),
            rounded_multiply(
                source_imaginary, operation.inverse_stored_weight.real));
        const std::size_t destination = static_cast<std::size_t>(
            operation.destination_indices[point]);
        const std::size_t output_index =
            selected_frequency * output_point_count + destination;
        const bool zero_divisor = operation.zero_divisor_flags &&
                                  operation.zero_divisor_flags[point] != 0;
        if (zero_divisor && weighted_real == 0.0f &&
            weighted_imaginary == 0.0f) {
          output[2 * output_index] = 0.0f;
          output[2 * output_index + 1] = 0.0f;
        }
        else if (zero_divisor) {
          output[2 * output_index] = rounded_multiply(
              weighted_real / 0.0f, operation.point_weights[point]);
          output[2 * output_index + 1] = rounded_multiply(
              weighted_imaginary / 0.0f,
              operation.point_weights[point]);
        }
        else {
          output[2 * output_index] = rounded_multiply(
              weighted_real, operation.point_weights[point]);
          output[2 * output_index + 1] = rounded_multiply(
              weighted_imaginary, operation.point_weights[point]);
        }
      }
  return output;
}

bool bitwise_equal(const std::vector<float> &left,
                   const std::vector<float> &right,
                   const char *case_name) {
  if (left.size() != right.size()) return false;
  for (std::size_t index = 0; index < left.size(); ++index)
    if (std::memcmp(&left[index], &right[index], sizeof(float)) != 0) {
      std::cerr << case_name << ": mismatch at scalar " << index
                << " observed=" << left[index]
                << " expected=" << right[index] << '\n';
      return false;
    }
  return true;
}

bool cpu_ieee_equal(const std::vector<float> &observed,
                    const std::vector<float> &expected,
                    const char *case_name) {
  if (observed.size() != expected.size()) return false;
  for (std::size_t index = 0; index < observed.size(); ++index) {
    // IEEE does not prescribe a NaN sign/payload, and CUDA and the host CPU
    // legitimately canonicalize invalid 0*Inf differently. Classification
    // is the observable Meep contract; finite values and signed infinities
    // remain bitwise checked.
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

bool run_materialization_case(
    const char *case_name,
    const std::vector<materialization_input> &inputs,
    std::size_t output_point_count,
    std::size_t output_frequency_count) {
  std::vector<meep_cuda::dft_materialization_operation_fp32>
      host_operations;
  std::vector<std::vector<std::uint8_t> > zero_divisor_flags;
  std::vector<std::vector<std::uint8_t> > publication_flags(inputs.size());
  zero_divisor_flags.reserve(inputs.size());
  std::vector<std::ptrdiff_t> published_destinations;
  for (std::size_t reverse_input = inputs.size(); reverse_input > 0;
       --reverse_input) {
    const std::size_t input_index = reverse_input - 1;
    const auto &destinations = inputs[input_index].destinations;
    publication_flags[input_index].assign(destinations.size(), 1u);
    for (std::size_t reverse_point = destinations.size();
         reverse_point > 0; --reverse_point) {
      const std::size_t point = reverse_point - 1;
      if (std::find(published_destinations.begin(),
                    published_destinations.end(),
                    destinations[point]) != published_destinations.end())
        publication_flags[input_index][point] = 0u;
      else
        published_destinations.push_back(destinations[point]);
    }
  }
  std::vector<std::uint32_t> block_operation_indices;
  std::size_t block_start = 0;
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    const materialization_input &input = inputs[index];
    if (input.source_real_imag.size() !=
        2 * input.storage_point_count * input.source_frequency_count)
      throw std::invalid_argument(
          std::string(case_name) + ": inconsistent source extent");
    if (input.destinations.size() != input.point_weights.size())
      throw std::invalid_argument(
          std::string(case_name) + ": inconsistent point metadata");
    zero_divisor_flags.push_back(input.zero_divisor_flags);
    if (zero_divisor_flags.back().empty())
      zero_divisor_flags.back().assign(input.destinations.size(), 0u);
    if (zero_divisor_flags.back().size() != input.destinations.size())
      throw std::invalid_argument(
          std::string(case_name) + ": inconsistent zero-divisor metadata");
    host_operations.push_back({
        input.source_real_imag.data(), input.destinations.data(),
        input.point_weights.data(), input.storage_point_count,
        input.destinations.size(), input.source_frequency_count,
        input.source_frequency_start, input.selected_frequency_count,
        input.inverse_stored_weight, block_start,
        zero_divisor_flags.back().data(),
        publication_flags[index].data()});
    const std::size_t work =
        input.destinations.size() * input.selected_frequency_count;
    const std::size_t blocks =
        work /
            static_cast<std::size_t>(
                meep_cuda::dft_materialization_threads_per_block_fp32) +
        (work %
             static_cast<std::size_t>(
                 meep_cuda::dft_materialization_threads_per_block_fp32) !=
         0);
    for (std::size_t block = 0; block < blocks; ++block)
      block_operation_indices.push_back(static_cast<std::uint32_t>(index));
    block_start += blocks;
  }

  meep_cuda::validate_dft_materialization_operations_fp32(
      host_operations.empty() ? nullptr : host_operations.data(),
      block_operation_indices.empty()
          ? nullptr
          : block_operation_indices.data(),
      host_operations.size(), block_operation_indices.size(),
      output_point_count, output_frequency_count);
  const std::vector<float> expected = cpu_oracle(
      host_operations, output_point_count, output_frequency_count);

  std::vector<device_pointer> device_sources;
  std::vector<device_pointer> device_destinations;
  std::vector<device_pointer> device_weights;
  std::vector<device_pointer> device_zero_divisor_flags;
  std::vector<device_pointer> device_publication_flags;
  device_sources.reserve(inputs.size());
  device_destinations.reserve(inputs.size());
  device_weights.reserve(inputs.size());
  device_zero_divisor_flags.reserve(inputs.size());
  device_publication_flags.reserve(inputs.size());
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    const materialization_input &input = inputs[index];
    device_sources.push_back(
        copy_vector_to_device(input.source_real_imag));
    device_destinations.push_back(
        copy_vector_to_device(input.destinations));
    device_weights.push_back(
        copy_vector_to_device(input.point_weights));
    device_zero_divisor_flags.push_back(
        copy_vector_to_device(zero_divisor_flags[index]));
    device_publication_flags.push_back(
        copy_vector_to_device(publication_flags[index]));
  }

  auto device_operations_image = host_operations;
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    device_operations_image[index].dft_real_imag =
        static_cast<const float *>(device_sources[index].get());
    device_operations_image[index].destination_indices =
        static_cast<const std::ptrdiff_t *>(
            device_destinations[index].get());
    device_operations_image[index].point_weights =
        static_cast<const float *>(device_weights[index].get());
    device_operations_image[index].zero_divisor_flags =
        static_cast<const std::uint8_t *>(
            device_zero_divisor_flags[index].get());
    device_operations_image[index].publication_flags =
        static_cast<const std::uint8_t *>(
            device_publication_flags[index].get());
  }
  auto device_operations =
      copy_vector_to_device(device_operations_image);
  auto device_block_operation_indices =
      copy_vector_to_device(block_operation_indices);

  std::vector<float> poison(expected.size(), 12345.25f);
  auto device_output = copy_vector_to_device(poison);
  meep_cuda::materialize_dft_fp32(
      static_cast<const meep_cuda::dft_materialization_operation_fp32 *>(
          device_operations.get()),
      static_cast<const std::uint32_t *>(
          device_block_operation_indices.get()),
      host_operations.size(), block_operation_indices.size(),
      static_cast<float *>(device_output.get()), output_point_count,
      output_frequency_count);
  meep_cuda::synchronize();
  std::vector<float> first(expected.size());
  meep_cuda::copy_to_host(first.data(), device_output.get(),
                          first.size() * sizeof(float));
  if (!cpu_ieee_equal(first, expected, case_name)) return false;

  meep_cuda::copy_to_device(device_output.get(), poison.data(),
                            poison.size() * sizeof(float));
  meep_cuda::materialize_dft_fp32(
      static_cast<const meep_cuda::dft_materialization_operation_fp32 *>(
          device_operations.get()),
      static_cast<const std::uint32_t *>(
          device_block_operation_indices.get()),
      host_operations.size(), block_operation_indices.size(),
      static_cast<float *>(device_output.get()), output_point_count,
      output_frequency_count);
  meep_cuda::synchronize();
  std::vector<float> second(expected.size());
  meep_cuda::copy_to_host(second.data(), device_output.get(),
                          second.size() * sizeof(float));
  if (!cpu_ieee_equal(second, expected, case_name) ||
      !bitwise_equal(second, first, case_name))
    return false;

  std::cout << "case=" << case_name << " operations=" << inputs.size()
            << " output_points=" << output_point_count
            << " frequencies=" << output_frequency_count
            << " bitwise_repeatable=yes\n";
  return true;
}

materialization_input make_input(
    std::size_t storage_point_count,
    std::size_t source_frequency_count,
    std::vector<std::ptrdiff_t> destinations,
    std::vector<float> point_weights,
    std::size_t source_frequency_start,
    std::size_t selected_frequency_count,
    meep_cuda::complex_value_fp32 inverse_stored_weight,
    float offset) {
  std::vector<float> source(
      2 * storage_point_count * source_frequency_count);
  for (std::size_t index = 0; index < source.size(); ++index) {
    const float magnitude =
        offset + static_cast<float>(index + 1) * 0.0625f;
    source[index] = (index % 3 == 0) ? -magnitude : magnitude;
  }
  return {source, destinations, point_weights, storage_point_count,
          source_frequency_count, source_frequency_start,
          selected_frequency_count, inverse_stored_weight, {}};
}

bool run_collapse_case(
    const char *case_name,
    const meep_cuda::dft_collapse_layout_fp32 &layout,
    std::size_t output_frequency_count) {
  std::size_t full_point_count = 1;
  std::size_t reduced_point_count = 1;
  for (std::size_t dim = 0; dim < layout.full_rank; ++dim) {
    full_point_count *= layout.full_dims[dim];
    if (layout.collapsed[dim] == 0)
      reduced_point_count *= layout.full_dims[dim];
  }
  std::vector<float> source(
      2 * full_point_count * output_frequency_count);
  for (std::size_t index = 0; index < source.size(); ++index) {
    const float magnitude =
        static_cast<float>((index % 29) + 1) * 0.03125f;
    source[index] = index % 3 == 0 ? -magnitude : magnitude;
  }

  std::vector<float> expected(
      2 * reduced_point_count * output_frequency_count, 0.0f);
  const std::size_t stride0 =
      layout.full_rank > 1
          ? layout.full_dims[1] *
                (layout.full_rank > 2 ? layout.full_dims[2] : 1)
          : 1;
  const std::size_t stride1 =
      layout.full_rank > 2 ? layout.full_dims[2] : 1;
  for (std::size_t frequency = 0;
       frequency < output_frequency_count; ++frequency)
    for (std::size_t reduced_point = 0;
         reduced_point < reduced_point_count; ++reduced_point) {
      std::size_t retained_coordinate[3] = {0, 0, 0};
      std::size_t remainder = reduced_point;
      for (int dim = static_cast<int>(layout.full_rank) - 1;
           dim >= 0; --dim)
        if (layout.collapsed[dim] == 0) {
          retained_coordinate[dim] =
              remainder % layout.full_dims[dim];
          remainder /= layout.full_dims[dim];
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
      float sum_real = 0.0f;
      float sum_imaginary = 0.0f;
      for (std::size_t n0 = begin0; n0 < end0; ++n0)
        for (std::size_t n1 = begin1; n1 < end1; ++n1)
          for (std::size_t n2 = begin2; n2 < end2; ++n2) {
            const std::size_t full_point =
                n0 * stride0 + n1 * stride1 + n2;
            const std::size_t source_index =
                2 * (frequency * full_point_count + full_point);
            sum_real = rounded_add(sum_real, source[source_index]);
            sum_imaginary =
                rounded_add(sum_imaginary, source[source_index + 1]);
          }
      const std::size_t destination =
          2 * (frequency * reduced_point_count + reduced_point);
      expected[destination] = sum_real;
      expected[destination + 1] = sum_imaginary;
    }

  auto device_source = copy_vector_to_device(source);
  std::vector<float> poison(expected.size(), -12345.5f);
  auto device_output = copy_vector_to_device(poison);
  meep_cuda::collapse_dft_array_fp32(
      static_cast<const float *>(device_source.get()),
      static_cast<float *>(device_output.get()), full_point_count,
      reduced_point_count, output_frequency_count, layout);
  meep_cuda::synchronize();
  std::vector<float> first(expected.size());
  meep_cuda::copy_to_host(first.data(), device_output.get(),
                          first.size() * sizeof(float));
  if (!bitwise_equal(first, expected, case_name)) return false;

  meep_cuda::copy_to_device(device_output.get(), poison.data(),
                            poison.size() * sizeof(float));
  meep_cuda::collapse_dft_array_fp32(
      static_cast<const float *>(device_source.get()),
      static_cast<float *>(device_output.get()), full_point_count,
      reduced_point_count, output_frequency_count, layout);
  meep_cuda::synchronize();
  std::vector<float> second(expected.size());
  meep_cuda::copy_to_host(second.data(), device_output.get(),
                          second.size() * sizeof(float));
  if (!bitwise_equal(second, expected, case_name) ||
      !bitwise_equal(second, first, case_name))
    return false;

  std::cout << "collapse-case=" << case_name
            << " rank=" << layout.full_rank
            << " full_points=" << full_point_count
            << " reduced_points=" << reduced_point_count
            << " frequencies=" << output_frequency_count
            << " bitwise_repeatable=yes\n";
  return true;
}

bool run_collapse_subnormal_contract_case() {
  const float denormal = std::numeric_limits<float>::denorm_min();
  const std::vector<float> source = {
      denormal, -0.0f, 2.0f * denormal, 0.0f,
      -denormal, -denormal, 4.0f * denormal, denormal};
  float expected_real = 0.0f;
  float expected_imaginary = 0.0f;
  for (std::size_t point = 0; point < 4; ++point) {
    expected_real = rounded_add(expected_real, source[2 * point]);
    expected_imaginary =
        rounded_add(expected_imaginary, source[2 * point + 1]);
  }
  const meep_cuda::dft_collapse_layout_fp32 layout = {
      1, {4, 1, 1}, {1u, 0u, 0u}};
  device_pointer device_source = copy_vector_to_device(source);
  std::vector<float> poison(2, 123.0f);
  device_pointer device_output = copy_vector_to_device(poison);
  const auto launch = [&]() {
    meep_cuda::collapse_dft_array_fp32(
        static_cast<const float *>(device_source.get()),
        static_cast<float *>(device_output.get()), 4, 1, 1, layout);
    meep_cuda::synchronize();
    std::vector<float> result(2);
    meep_cuda::copy_to_host(result.data(), device_output.get(),
                            result.size() * sizeof(float));
    return result;
  };
  const std::vector<float> first = launch();
  meep_cuda::copy_to_device(device_output.get(), poison.data(),
                            poison.size() * sizeof(float));
  const std::vector<float> second = launch();
  if (!bitwise_equal(first, second, "collapse-subnormal-repeat"))
    return false;
  const float expected[2] = {expected_real, expected_imaginary};
  bool observed_ftz = false;
  for (std::size_t index = 0; index < 2; ++index) {
    if (std::memcmp(&first[index], &expected[index], sizeof(float)) == 0)
      continue;
    // --use_fast_math is documented to enable FTZ.  Subnormal collapse
    // results may therefore become either signed zero, but must remain
    // finite and exactly zero; normal finite values stay bitwise checked in
    // all other collapse cases.
    if (std::fpclassify(expected[index]) == FP_SUBNORMAL &&
        first[index] == 0.0f && std::isfinite(first[index])) {
      observed_ftz = true;
      continue;
    }
    if (expected[index] == 0.0f && first[index] == 0.0f &&
        std::isfinite(first[index]))
      continue;
    std::cerr << "collapse-subnormal-contract: unexpected scalar "
              << index << " observed=" << first[index]
              << " expected=" << expected[index] << '\n';
    return false;
  }
  std::cout << "collapse-case=subnormal-contract bitwise_repeatable=yes "
            << "behavior=" << (observed_ftz ? "ftz" : "preserved")
            << '\n';
  return true;
}

double median_milliseconds(std::vector<double> samples) {
  if (samples.empty())
    throw std::invalid_argument("benchmark samples must be nonempty");
  std::sort(samples.begin(), samples.end());
  const std::size_t middle = samples.size() / 2;
  return samples.size() % 2
             ? samples[middle]
             : 0.5 * (samples[middle - 1] + samples[middle]);
}

void collapse_benchmark_cpu(
    const std::vector<float> &full, std::vector<float> *reduced,
    std::size_t retained_points, std::size_t frequencies) {
  if (!reduced ||
      full.size() != 2 * retained_points * 4 * frequencies ||
      reduced->size() != 2 * retained_points * frequencies)
    throw std::invalid_argument("collapse benchmark extent mismatch");
  for (std::size_t frequency = 0; frequency < frequencies; ++frequency)
    for (std::size_t retained = 0; retained < retained_points;
         ++retained) {
      float real_sum = 0.0f;
      float imaginary_sum = 0.0f;
      for (std::size_t first = 0; first < 2; ++first)
        for (std::size_t second = 0; second < 2; ++second) {
          const std::size_t full_point =
              retained * 4 + first * 2 + second;
          const std::size_t source =
              2 * (frequency * retained_points * 4 + full_point);
          // Match the production host fallback in dft.cpp: ordinary FP32
          // accumulation in dense-point order, with no benchmark-only
          // volatile barrier that would disadvantage the CPU baseline.
          real_sum += full[source];
          imaginary_sum += full[source + 1];
        }
      const std::size_t destination =
          2 * (frequency * retained_points + retained);
      (*reduced)[destination] = real_sum;
      (*reduced)[destination + 1] = imaginary_sum;
    }
}

void run_collapse_transfer_benchmark(const meep_cuda::device_info &device) {
  constexpr std::size_t retained_points = 8192;
  constexpr std::size_t collapsed_points_per_fiber = 4;
  constexpr std::size_t frequencies = 101;
  constexpr int warmups = 5;
  constexpr int repetitions = 30;
  const std::size_t full_points =
      retained_points * collapsed_points_per_fiber;
  std::vector<float> source(2 * full_points * frequencies);
  for (std::size_t index = 0; index < source.size(); ++index)
    source[index] =
        static_cast<float>((static_cast<int>(index % 257) - 128) *
                           0.0009765625);
  std::vector<float> host_full(source.size());
  std::vector<float> cpu_reduced(2 * retained_points * frequencies);
  std::vector<float> gpu_reduced(cpu_reduced.size());
  device_pointer device_full = copy_vector_to_device(source);
  device_pointer device_reduced(meep_cuda::allocate_device_bytes(
      gpu_reduced.size() * sizeof(float)));
  const meep_cuda::dft_collapse_layout_fp32 layout = {
      3, {retained_points, 2, 2}, {0u, 1u, 1u}};

  const auto run_baseline = [&]() {
    meep_cuda::copy_to_host(host_full.data(), device_full.get(),
                            host_full.size() * sizeof(float));
    collapse_benchmark_cpu(host_full, &cpu_reduced, retained_points,
                           frequencies);
  };
  const auto run_gpu = [&]() {
    meep_cuda::collapse_dft_array_fp32(
        static_cast<const float *>(device_full.get()),
        static_cast<float *>(device_reduced.get()), full_points,
        retained_points, frequencies, layout);
    meep_cuda::copy_to_host(gpu_reduced.data(), device_reduced.get(),
                            gpu_reduced.size() * sizeof(float));
  };
  for (int iteration = 0; iteration < warmups; ++iteration)
    if (iteration % 2 == 0) {
      run_baseline();
      run_gpu();
    }
    else {
      run_gpu();
      run_baseline();
    }
  if (!bitwise_equal(gpu_reduced, cpu_reduced,
                     "collapse-transfer-benchmark"))
    throw std::runtime_error(
        "collapse transfer benchmark differs from its CPU oracle");

  std::vector<double> baseline_samples;
  std::vector<double> gpu_samples;
  baseline_samples.reserve(repetitions);
  gpu_samples.reserve(repetitions);
  for (int iteration = 0; iteration < repetitions; ++iteration) {
    const auto measure = [](const auto &operation) {
      const auto start = std::chrono::steady_clock::now();
      operation();
      const auto finish = std::chrono::steady_clock::now();
      return std::chrono::duration<double, std::milli>(finish - start)
          .count();
    };
    if (iteration % 2 == 0) {
      baseline_samples.push_back(measure(run_baseline));
      gpu_samples.push_back(measure(run_gpu));
    }
    else {
      gpu_samples.push_back(measure(run_gpu));
      baseline_samples.push_back(measure(run_baseline));
    }
  }
  if (!bitwise_equal(gpu_reduced, cpu_reduced,
                     "collapse-transfer-benchmark-final"))
    throw std::runtime_error(
        "collapse transfer benchmark became nondeterministic");
  const double baseline_median = median_milliseconds(baseline_samples);
  const double gpu_median = median_milliseconds(gpu_samples);
  const double speedup = baseline_median / gpu_median;
  std::cout << "COLLAPSE_TRANSFER_BENCHMARK device='" << device.name
            << "' retained_points=" << retained_points
            << " full_points=" << full_points
            << " frequencies=" << frequencies
            << " full_d2h_bytes=" << source.size() * sizeof(float)
            << " reduced_d2h_bytes="
            << gpu_reduced.size() * sizeof(float)
            << " baseline_d2h_host_collapse_median_ms="
            << baseline_median
            << " gpu_collapse_reduced_d2h_median_ms=" << gpu_median
            << " speedup=" << speedup
            << " repetitions=" << repetitions
            << " scope=collapse-plus-d2h-microbenchmark"
            << " order=alternating\n";
  std::cout << "COLLAPSE_TRANSFER_BENCHMARK_RAW baseline_ms=";
  for (std::size_t index = 0; index < baseline_samples.size(); ++index)
    std::cout << (index == 0 ? "[" : ",") << baseline_samples[index];
  std::cout << "] gpu_ms=";
  for (std::size_t index = 0; index < gpu_samples.size(); ++index)
    std::cout << (index == 0 ? "[" : ",") << gpu_samples[index];
  std::cout << "]\n";
  if (!(speedup >= 1.20))
    throw std::runtime_error(
        "production-scale CUDA collapse did not beat dense D2H plus host "
        "collapse by at least 20%");
}

} // namespace

int main() {
  std::string diagnostic;
  if (!meep_cuda::runtime_available(&diagnostic)) {
    std::cout << "SKIP: CUDA DFT materialization runtime unavailable: "
              << diagnostic << '\n';
    return 77;
  }

  const auto devices = meep_cuda::enumerate_devices();
  const int requested_ordinal = requested_device_ordinal();
  const meep_cuda::device_info *selected_device = nullptr;
  for (const auto &device : devices)
    if ((requested_ordinal < 0 || device.ordinal == requested_ordinal) &&
        meep_cuda::device_compatible(device)) {
      selected_device = &device;
      break;
    }
  if (!selected_device) {
    std::cerr << "FAIL: requested CUDA device is absent or incompatible with "
              << meep_cuda::compiled_architectures() << '\n';
    return 1;
  }
  meep_cuda::select_device(selected_device->ordinal);

  const std::vector<materialization_input> selected_inputs = {
      make_input(4, 5, {6, 1, 8}, {1.0f, -0.5f, 0.0f},
                 1, 3, {0.75f, -0.25f}, 0.125f),
      make_input(3, 5, {0, 4}, {2.0f, -1.25f},
                 1, 3, {-0.5f, 0.375f}, -0.25f)};
  if (!run_materialization_case(
          "selected-frequency-arbitrary-map", selected_inputs, 9, 3)) {
    std::cerr << "FAIL: selected-frequency DFT materialization differs from "
                 "the explicit FP32 CPU oracle\n";
    return 1;
  }

  std::vector<std::ptrdiff_t> multi_block_destinations;
  std::vector<float> multi_block_weights;
  for (std::size_t point = 0; point < 130; ++point) {
    multi_block_destinations.push_back(
        static_cast<std::ptrdiff_t>((37 * point) % 130));
    multi_block_weights.push_back(
        point % 11 == 0
            ? 0.0f
            : (point % 2 == 0 ? 0.375f : -0.625f));
  }
  const std::vector<materialization_input> multi_block_inputs = {
      make_input(140, 5, multi_block_destinations, multi_block_weights,
                 2, 3, {-0.875f, 0.3125f}, -0.125f),
      make_input(2, 5, {136, 132}, {1.25f, -0.75f},
                 2, 3, {0.625f, -0.1875f}, 0.375f)};
  if (!run_materialization_case(
          "multi-logical-block-tail", multi_block_inputs, 137, 3)) {
    std::cerr << "FAIL: multi-logical-block DFT materialization differs "
                 "from the explicit FP32 CPU oracle\n";
    return 1;
  }

  const std::vector<materialization_input> all_frequency_inputs = {
      make_input(3, 4, {3, 0}, {0.25f, -2.0f},
                 0, 4, {1.125f, 0.5f}, 0.5f)};
  if (!run_materialization_case(
          "all-frequency-arbitrary-map", all_frequency_inputs, 5, 4)) {
    std::cerr << "FAIL: all-frequency DFT materialization differs from the "
                 "explicit FP32 CPU oracle\n";
    return 1;
  }

  const std::vector<materialization_input> zero_divisor_inputs = {
      {{0.0f, 0.0f,
        1.0f, 0.0f,
        -2.0f, 1.0f,
        0.0f, 3.0f},
       {0, 1, 2, 3}, {1.0f, 0.0f, 0.5f, -2.0f},
       4, 1, 0, 1, {1.0f, 0.0f}, {1u, 1u, 1u, 1u}}};
  if (!run_materialization_case(
          "cylindrical-axis-zero-divisor", zero_divisor_inputs, 4, 1)) {
    std::cerr << "FAIL: cylindrical-axis zero/nonzero exceptional samples "
                 "differ from CPU divide-then-interpolate semantics\n";
    return 1;
  }

  const std::vector<materialization_input> duplicate_destination_inputs = {
      make_input(2, 3, {0, 1}, {1.0f, 0.5f},
                 0, 3, {1.0f, 0.0f}, 0.125f),
      make_input(2, 3, {1, 2}, {-0.75f, 2.0f},
                 0, 3, {0.5f, -0.25f}, -0.375f)};
  if (!run_materialization_case(
          "last-writer-overlapping-destinations",
          duplicate_destination_inputs, 3, 3)) {
    std::cerr << "FAIL: overlapping transformed chunks did not preserve "
                 "CPU last-writer traversal semantics\n";
    return 1;
  }

  if (!run_materialization_case(
          "empty-rank-zero-fill", {}, 6, 2)) {
    std::cerr << "FAIL: empty-rank DFT materialization did not zero the "
                 "complete output\n";
    return 1;
  }

  const meep_cuda::dft_collapse_layout_fp32 collapse_cases[] = {
      {1, {5, 1, 1}, {1u, 0u, 0u}},
      {2, {3, 4, 1}, {1u, 0u, 0u}},
      {2, {3, 4, 1}, {0u, 1u, 0u}},
      {3, {2, 3, 4}, {0u, 1u, 0u}},
      {3, {2, 3, 4}, {1u, 0u, 1u}}};
  const char *collapse_labels[] = {
      "1d-point", "2d-first-axis", "2d-last-axis",
      "3d-middle-axis", "3d-multiple-axes"};
  for (std::size_t index = 0;
       index < sizeof(collapse_cases) / sizeof(collapse_cases[0]);
       ++index)
    if (!run_collapse_case(collapse_labels[index], collapse_cases[index],
                           3)) {
      std::cerr << "FAIL: CUDA DFT collapse differs from its ordered FP32 "
                   "CPU oracle for "
                << collapse_labels[index] << '\n';
      return 1;
    }
  if (!run_collapse_subnormal_contract_case()) {
    std::cerr << "FAIL: CUDA DFT collapse violated the documented "
                 "subnormal preserve-or-FTZ contract\n";
    return 1;
  }

  if (std::getenv("MEEP_CUDA_DFT_COLLAPSE_BENCHMARK"))
    run_collapse_transfer_benchmark(*selected_device);

  std::cout << "PASS: CUDA DFT materialization matches the explicit FP32 "
               "CPU oracle bit-for-bit for defined values and IEEE-"
               "equivalently for NaNs across selected/all frequencies, "
               "arbitrary maps, complex inverse weights, per-point weights, "
               "overlapping last-writer maps, holes, repeated launches, and "
               "deterministic 1D/2D/3D collapse layouts\n";
  return 0;
}
