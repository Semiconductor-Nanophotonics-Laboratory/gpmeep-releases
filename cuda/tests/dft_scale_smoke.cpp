#include "meep_cuda/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
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

int requested_device_ordinal() {
  const char *value = std::getenv("MEEP_GPU_DEVICE");
  if (!value || !*value) return 0;
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (!end || *end || parsed < 0 ||
      parsed > std::numeric_limits<int>::max())
    throw std::invalid_argument(
        "MEEP_GPU_DEVICE must be a nonnegative integer");
  return static_cast<int>(parsed);
}

std::vector<float> oracle(const std::vector<float> &input,
                          double scale_real,
                          double scale_imaginary) {
  std::vector<float> output(input.size());
  for (std::size_t index = 0; index < input.size() / 2; ++index) {
    const double real = input[2 * index];
    const double imaginary = input[2 * index + 1];
    output[2 * index] = static_cast<float>(
        real * scale_real - imaginary * scale_imaginary);
    output[2 * index + 1] = static_cast<float>(
        real * scale_imaginary + imaginary * scale_real);
  }
  return output;
}

void require_close(const std::vector<float> &expected,
                   const std::vector<float> &actual) {
  if (expected.size() != actual.size())
    throw std::runtime_error("DFT scale result size differs");
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const float tolerance =
        4.0f * std::numeric_limits<float>::epsilon() *
            std::max(1.0f, std::abs(expected[index]));
    if (!(std::abs(expected[index] - actual[index]) <= tolerance))
      throw std::runtime_error(
          "resident complex DFT scale disagrees with FP32 CPU oracle");
  }
}

} // namespace

int main() {
  std::string diagnostic;
  if (!meep_cuda::runtime_available(&diagnostic)) {
    std::cout << "SKIP: " << diagnostic << '\n';
    return 77;
  }
  try {
    meep_cuda::select_device(requested_device_ordinal());
    const std::vector<float> initial = {
        1.25f, -0.75f, -3.5f, 2.25f, 0.0f, -0.0f,
        1.0e-20f, -2.0e-20f, 8192.5f, -4096.25f};
    const std::size_t bytes = initial.size() * sizeof(float);
    std::unique_ptr<void, device_deleter> device(
        meep_cuda::allocate_device_bytes(bytes));
    meep_cuda::copy_to_device(device.get(), initial.data(), bytes);

    const double scale_real = -0.375;
    const double scale_imaginary = 0.125;
    meep_cuda::complex_scale_inplace_fp32(
        static_cast<float *>(device.get()), initial.size() / 2,
        scale_real, scale_imaginary);
    std::vector<float> actual(initial.size());
    meep_cuda::copy_to_host(actual.data(), device.get(), bytes);
    require_close(oracle(initial, scale_real, scale_imaginary), actual);

    bool null_rejected = false;
    try {
      meep_cuda::complex_scale_inplace_fp32(nullptr, 1, 1.0, 0.0);
    }
    catch (const std::invalid_argument &) { null_rejected = true; }
    if (!null_rejected)
      throw std::runtime_error("nonempty null DFT scale was accepted");
    meep_cuda::complex_scale_inplace_fp32(nullptr, 0, 1.0, 0.0);

    bool nonfinite_rejected = false;
    try {
      meep_cuda::complex_scale_inplace_fp32(
          static_cast<float *>(device.get()), initial.size() / 2,
          std::numeric_limits<double>::infinity(), 0.0);
    }
    catch (const std::invalid_argument &) { nonfinite_rejected = true; }
    if (!nonfinite_rejected)
      throw std::runtime_error("nonfinite DFT scale was accepted");
    std::cout << "PASS: resident complex FP32 DFT scale matches the CPU "
                 "oracle and validates zero/null/nonfinite inputs\n";
    return 0;
  }
  catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
