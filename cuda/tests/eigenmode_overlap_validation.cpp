#include "meep_cuda/runtime.hpp"

#include <cmath>
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
  throw std::runtime_error(std::string(name) +
                           ": expected validation failure");
}

} // namespace

int main() {
  float dft[6] = {};
  meep_cuda::complex_value_fp64 mode1[1] = {{1.0, 0.0}};
  meep_cuda::complex_value_fp64 mode2[1] = {{0.0, 1.0}};
  std::uint32_t block_map[2] = {0u, 1u};

  meep_cuda::validate_eigenmode_overlap_operations_fp32(
      nullptr, nullptr, 0, 0, 0);
  expect_failure<std::out_of_range>(
      "zero outputs", "output count", [&] {
        meep_cuda::validate_eigenmode_overlap_operations_fp32(
            nullptr, nullptr, 0, 0, 0, 0);
      });
  expect_failure<std::out_of_range>(
      "too many outputs", "output count", [&] {
        meep_cuda::validate_eigenmode_overlap_operations_fp32(
            nullptr, nullptr, 0, 0, 0,
            meep_cuda::eigenmode_overlap_max_output_count + 1);
      });
  expect_failure<std::invalid_argument>(
      "orphan blocks", "require host operations", [&] {
        meep_cuda::validate_eigenmode_overlap_operations_fp32(
            nullptr, nullptr, 0, 1, 0);
      });

  meep_cuda::eigenmode_overlap_operation_fp32 operations[2] = {
      {dft, mode1, nullptr, nullptr, 1, 3, 0, 0, {0.75, -0.25}},
      {nullptr, mode1, mode2, nullptr, 1, 0, 1, 3, {0.0, 0.0}}};
  meep_cuda::validate_eigenmode_overlap_operations_fp32(
      operations, block_map, 2, 2, 2);
  {
    auto high_channel = operations[0];
    high_channel.output_index = 7;
    const std::uint32_t one_block_map[1] = {0u};
    meep_cuda::validate_eigenmode_overlap_operations_fp32(
        &high_channel, one_block_map, 1, 1, 2,
        meep_cuda::eigenmode_overlap_max_output_count);
  }

  expect_failure<std::invalid_argument>(
      "null operations", "non-null", [&] {
        meep_cuda::validate_eigenmode_overlap_operations_fp32(
            nullptr, block_map, 2, 2, 0);
      });
  expect_failure<std::invalid_argument>(
      "null map", "non-null", [&] {
        meep_cuda::validate_eigenmode_overlap_operations_fp32(
            operations, nullptr, 2, 2, 0);
      });
  {
    auto invalid = operations[0];
    invalid.weighted_conjugate_mode = nullptr;
    expect_failure<std::invalid_argument>(
        "missing mode1", "incomplete", [&] {
          meep_cuda::validate_eigenmode_overlap_operations_fp32(
              &invalid, block_map, 1, 1, 0);
        });
  }
  {
    auto invalid = operations[0];
    invalid.mode2_real_imag = mode2;
    expect_failure<std::invalid_argument>(
        "ambiguous rhs", "incomplete", [&] {
          meep_cuda::validate_eigenmode_overlap_operations_fp32(
              &invalid, block_map, 1, 1, 0);
        });
  }
  {
    auto invalid = operations[0];
    invalid.point_count = 0;
    expect_failure<std::invalid_argument>(
        "empty operation", "incomplete", [&] {
          meep_cuda::validate_eigenmode_overlap_operations_fp32(
              &invalid, block_map, 1, 1, 0);
        });
  }
  {
    auto invalid = operations[0];
    invalid.output_index =
        static_cast<std::uint32_t>(
            meep_cuda::eigenmode_overlap_output_count);
    expect_failure<std::out_of_range>(
        "bad output", "output index", [&] {
          meep_cuda::validate_eigenmode_overlap_operations_fp32(
              &invalid, block_map, 1, 1, 0);
        });
  }
  {
    auto invalid = operations[0];
    invalid.block_start = 1;
    expect_failure<std::invalid_argument>(
        "bad prefix", "prefixes", [&] {
          meep_cuda::validate_eigenmode_overlap_operations_fp32(
              &invalid, block_map, 1, 1, 0);
        });
  }
  expect_failure<std::invalid_argument>(
      "bad total", "logical block count", [&] {
        meep_cuda::validate_eigenmode_overlap_operations_fp32(
            operations, block_map, 2, 1, 0);
      });
  {
    std::uint32_t invalid_map[2] = {0u, 2u};
    expect_failure<std::out_of_range>(
        "bad map owner", "block owner", [&] {
          meep_cuda::validate_eigenmode_overlap_operations_fp32(
              operations, invalid_map, 2, 2, 0);
        });
  }
  {
    std::uint32_t invalid_map[2] = {1u, 1u};
    expect_failure<std::invalid_argument>(
        "bad map interval", "block map", [&] {
          meep_cuda::validate_eigenmode_overlap_operations_fp32(
              operations, invalid_map, 2, 2, 0);
        });
  }
  expect_failure<std::out_of_range>(
      "frequency extent", "frequency", [&] {
        meep_cuda::validate_eigenmode_overlap_operations_fp32(
            operations, block_map, 2, 2, 3);
      });
  {
    auto invalid = operations[0];
    invalid.dft_frequency_count = 0;
    expect_failure<std::out_of_range>(
        "zero frequencies", "frequency", [&] {
          meep_cuda::validate_eigenmode_overlap_operations_fp32(
              &invalid, block_map, 1, 1, 0);
        });
  }
  {
    auto invalid = operations[1];
    invalid.block_start = 0;
    invalid.dft_frequency_count = 1;
    expect_failure<std::invalid_argument>(
        "mode-mode frequency", "normalization metadata", [&] {
          meep_cuda::validate_eigenmode_overlap_operations_fp32(
              &invalid, block_map, 1, 1, 0);
        });
  }
  {
    auto invalid = operations[1];
    invalid.block_start = 0;
    const std::uint8_t flag = 0;
    invalid.zero_normalization_divisors = &flag;
    expect_failure<std::invalid_argument>(
        "mode-mode zero flag", "normalization metadata", [&] {
          meep_cuda::validate_eigenmode_overlap_operations_fp32(
              &invalid, block_map, 1, 1, 0);
        });
  }
  {
    auto invalid = operations[0];
    invalid.point_count =
        std::numeric_limits<std::size_t>::max() / 2 + 1;
    invalid.dft_frequency_count = 2;
    expect_failure<std::overflow_error>(
        "DFT point-frequency overflow", "point-frequency", [&] {
          meep_cuda::validate_eigenmode_overlap_operations_fp32(
              &invalid, block_map, 1, 1, 0);
        });
  }
  {
    auto invalid = operations[0];
    invalid.point_count =
        std::numeric_limits<std::size_t>::max() / 2 + 1;
    invalid.dft_frequency_count = 1;
    expect_failure<std::overflow_error>(
        "interleaved DFT scalar overflow", "interleaved", [&] {
          meep_cuda::validate_eigenmode_overlap_operations_fp32(
              &invalid, block_map, 1, 1, 0);
        });
  }
  {
    auto invalid = operations[0];
    invalid.inverse_stored_weight.real =
        std::numeric_limits<double>::infinity();
    expect_failure<std::invalid_argument>(
        "nonfinite inverse", "finite", [&] {
          meep_cuda::validate_eigenmode_overlap_operations_fp32(
              &invalid, block_map, 1, 1, 0);
        });
  }
  std::cout << "PASS: eigenmode-overlap descriptor validation rejects "
               "ambiguous, malformed, and nonfinite plans "
               "without touching a CUDA device\n";
  return 0;
}
