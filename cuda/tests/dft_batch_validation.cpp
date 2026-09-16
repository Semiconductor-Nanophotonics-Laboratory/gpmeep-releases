#include "meep_cuda/runtime.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

template <typename Exception, typename Callable>
void expect_failure(const char *name, const char *fragment,
                    Callable callable) {
  try {
    callable();
  }
  catch (const Exception &error) {
    if (std::string(error.what()).find(fragment) != std::string::npos)
      return;
    throw std::runtime_error(
        std::string(name) + ": unexpected diagnostic: " + error.what());
  }
  throw std::runtime_error(
      std::string(name) + ": expected validation failure");
}

} // namespace

int main() {
  float output = 0.0f;
  float field = 0.0f;
  float weight = 1.0f;
  std::ptrdiff_t index = 0;
  meep_cuda::complex_value_fp32 phase = {1.0f, 0.0f};

  meep_cuda::validate_dft_batch_operations_fp32(nullptr, 0, 0);
  expect_failure<std::invalid_argument>(
      "null operations", "host operations", [&] {
        meep_cuda::validate_dft_batch_operations_fp32(nullptr, 1, 1);
      });
  expect_failure<std::invalid_argument>(
      "ownerless blocks", "requires host operations", [&] {
        meep_cuda::validate_dft_batch_operations_fp32(nullptr, 0, 1);
      });

  meep_cuda::dft_update_operation_fp32 valid = {
      &output, &field, nullptr, &index, &weight, 1, &phase, 1, 0, 0,
      0, 1, 1, 1, 1};
  meep_cuda::validate_dft_batch_operations_fp32(&valid, 1, 1);

  {
    auto invalid = valid;
    invalid.phases = nullptr;
    expect_failure<std::invalid_argument>("null phase", "pointers", [&] {
      meep_cuda::validate_dft_batch_operations_fp32(&invalid, 1, 1);
    });
  }
  {
    auto invalid = valid;
    invalid.point_count = 0;
    expect_failure<std::invalid_argument>("zero count", "counts", [&] {
      meep_cuda::validate_dft_batch_operations_fp32(&invalid, 1, 1);
    });
  }
  {
    auto invalid = valid;
    invalid.frequency_threads = 0;
    expect_failure<std::invalid_argument>(
        "zero thread tile", "thread layout", [&] {
          meep_cuda::validate_dft_batch_operations_fp32(&invalid, 1, 1);
        });
    invalid.frequency_threads = 1;
    invalid.point_threads =
        meep_cuda::dft_batch_threads_per_block_fp32 + 1;
    expect_failure<std::invalid_argument>(
        "oversize thread tile", "thread layout", [&] {
          meep_cuda::validate_dft_batch_operations_fp32(&invalid, 1, 1);
        });
  }
  {
    auto invalid = valid;
    invalid.frequency_block_count = 2;
    expect_failure<std::invalid_argument>(
        "wrong tile count", "tile counts", [&] {
          meep_cuda::validate_dft_batch_operations_fp32(&invalid, 1, 1);
        });
  }
  {
    auto invalid = valid;
    invalid.block_start = 1;
    expect_failure<std::invalid_argument>(
        "noncontiguous prefix", "prefixes", [&] {
          meep_cuda::validate_dft_batch_operations_fp32(&invalid, 1, 1);
        });
  }
  expect_failure<std::invalid_argument>(
      "wrong total", "do not match", [&] {
        meep_cuda::validate_dft_batch_operations_fp32(&valid, 1, 2);
      });

  {
    auto invalid = valid;
    invalid.point_count = std::numeric_limits<std::size_t>::max();
    invalid.frequency_count = 2;
    expect_failure<std::overflow_error>(
        "point-frequency overflow", "work count", [&] {
          meep_cuda::validate_dft_batch_operations_fp32(&invalid, 1, 1);
        });
  }
  {
    auto invalid = valid;
    invalid.point_count = std::numeric_limits<std::size_t>::max() / 2 + 1;
    expect_failure<std::overflow_error>(
        "interleaved output overflow", "output count", [&] {
          meep_cuda::validate_dft_batch_operations_fp32(&invalid, 1, 1);
        });
  }
  {
    auto invalid = valid;
    invalid.point_count =
        std::numeric_limits<std::size_t>::max() /
            (2 * sizeof(float)) +
        1;
    expect_failure<std::overflow_error>(
        "interleaved byte overflow", "byte count", [&] {
          meep_cuda::validate_dft_batch_operations_fp32(&invalid, 1, 1);
        });
  }
  if (std::numeric_limits<std::size_t>::max() >
      static_cast<std::size_t>(
          std::numeric_limits<std::uint32_t>::max()))
    expect_failure<std::overflow_error>(
        "operation index overflow", "index", [&] {
          meep_cuda::validate_dft_batch_operations_fp32(
              &valid,
              static_cast<std::size_t>(
                  std::numeric_limits<std::uint32_t>::max()) +
                  1u,
              1);
        });

  std::cout << "PASS: fused DFT batch host descriptor validation is "
               "device-independent and overflow-safe\n";
  return 0;
}
