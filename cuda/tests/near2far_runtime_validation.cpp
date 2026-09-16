#include "meep_cuda/runtime.hpp"

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
  float dft[2] = {};
  meep_cuda::cartesian_point_fp32 point = {0.0f, 0.0f, 0.0f};
  meep_cuda::near2far_periodic_copy_fp32 copy = {
      {0.0, 0.0, 0.0}, {1.0, 0.0}};
  meep_cuda::near2far_operation_fp32 operation = {
      dft, &point, 1, 1, 0, true};
  float frequency = 0.25f;
  float partial = 0.0f;
  double output = 0.0;
  double absolute_l1 = 0.0;
  const auto invoke =
      [&](const meep_cuda::near2far_operation_fp32 *operations,
          std::size_t operation_count, std::size_t target_count,
          std::size_t frequency_count, std::size_t partial_count,
          int threads) {
        meep_cuda::near2far_3d_fp32(
            operations, operation_count, &point, target_count, &frequency,
            frequency_count, 0, &copy, 1, 1.0f, 1.0f, partial_count,
            &partial, &output, &absolute_l1, threads);
      };

  expect_failure<std::invalid_argument>(
      "invalid dimension", "dimension must be 2D, 3D, or cylindrical", [&] {
        meep_cuda::near2far_cartesian_fp32(
            static_cast<meep_cuda::near2far_cartesian_dimension>(7),
            &operation, 1, &point, 1, &frequency, 1, 0, &copy, 1, 1.0f,
            1.0f, 1, &partial, &output, &absolute_l1);
      });
  expect_failure<std::invalid_argument>(
      "missing cylindrical tolerance", "mode/tolerance", [&] {
        meep_cuda::near2far_cartesian_fp32(
            meep_cuda::near2far_cartesian_dimension::cylindrical,
            &operation, 1, &point, 1, &frequency, 1, 0, &copy, 1, 1.0f,
            1.0f, 1, &partial, &output, &absolute_l1, 256, 0.0f, 1.0f,
            0.0f);
      });
  expect_failure<std::invalid_argument>(
      "Cartesian cylindrical arguments", "mode/tolerance", [&] {
        meep_cuda::near2far_cartesian_fp32(
            meep_cuda::near2far_cartesian_dimension::three,
            &operation, 1, &point, 1, &frequency, 1, 0, &copy, 1, 1.0f,
            1.0f, 1, &partial, &output, &absolute_l1, 256, 0.0f, 1.0f,
            1.0e-3f);
      });

  expect_failure<std::invalid_argument>(
      "null operations", "must be non-null",
      [&] { invoke(nullptr, 1, 1, 1, 1, 256); });
  expect_failure<std::invalid_argument>(
      "zero operations", "must be nonzero",
      [&] { invoke(&operation, 0, 1, 1, 1, 256); });
  expect_failure<std::invalid_argument>(
      "non-power block", "power of two",
      [&] { invoke(&operation, 1, 1, 1, 1, 192); });
  expect_failure<std::invalid_argument>(
      "resource-heavy block", "[1, 256]",
      [&] { invoke(&operation, 1, 1, 1, 1, 512); });
  expect_failure<std::overflow_error>(
      "work count", "work count", [&] {
        invoke(&operation, 1, std::numeric_limits<std::size_t>::max(), 2,
               1, 256);
      });
  expect_failure<std::overflow_error>(
      "output scalars", "output scalar", [&] {
        invoke(&operation, 1,
               std::numeric_limits<std::size_t>::max() / 12 + 1, 1, 1,
               256);
      });
  expect_failure<std::overflow_error>(
      "partial blocks", "partial-block", [&] {
        invoke(&operation, 1, 2, 1,
               std::numeric_limits<std::size_t>::max(), 256);
      });

  meep_cuda::near2far_adjoint_source_fp32 adjoint_source = {
      point, {1.0f, 0.0f}, 0, 1};
  meep_cuda::complex_value_fp32 gradient = {1.0f, 0.0f};
  const auto invoke_adjoint =
      [&](const meep_cuda::near2far_adjoint_source_fp32 *sources,
          std::size_t source_count, std::size_t target_count,
          std::size_t frequency_count, int threads) {
        meep_cuda::near2far_adjoint_fp32(
            meep_cuda::near2far_cartesian_dimension::three, sources,
            source_count, &point, target_count, &frequency,
            frequency_count, &copy, 1, &gradient, 1.0f, 1.0f, &output,
            &absolute_l1, threads);
      };
  expect_failure<std::invalid_argument>(
      "adjoint invalid dimension", "dimension must be 2D, 3D, or cylindrical",
      [&] {
        meep_cuda::near2far_adjoint_fp32(
            static_cast<meep_cuda::near2far_cartesian_dimension>(9),
            &adjoint_source, 1, &point, 1, &frequency, 1, &copy, 1,
            &gradient, 1.0f, 1.0f, &output, &absolute_l1);
      });
  expect_failure<std::invalid_argument>(
      "adjoint null sources", "must be non-null",
      [&] { invoke_adjoint(nullptr, 1, 1, 1, 256); });
  expect_failure<std::invalid_argument>(
      "adjoint zero sources", "must be nonzero",
      [&] { invoke_adjoint(&adjoint_source, 0, 1, 1, 256); });
  expect_failure<std::invalid_argument>(
      "adjoint non-power block", "power of two",
      [&] { invoke_adjoint(&adjoint_source, 1, 1, 1, 192); });
  expect_failure<std::overflow_error>(
      "adjoint work count", "work count", [&] {
        invoke_adjoint(&adjoint_source,
                       std::numeric_limits<std::size_t>::max(), 1, 2,
                       256);
      });
  expect_failure<std::overflow_error>(
      "adjoint dJ count", "target-frequency count", [&] {
        invoke_adjoint(&adjoint_source, 1,
                       std::numeric_limits<std::size_t>::max(), 2, 256);
      });

  std::cout << "PASS: CUDA near-to-far validation rejects malformed and "
               "overflowing plans without touching a device\n";
  return 0;
}
