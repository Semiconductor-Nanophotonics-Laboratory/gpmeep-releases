#include "meep_cuda/runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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
  float source = 0.0f;
  const std::size_t first_point_count = 65;
  std::vector<std::ptrdiff_t> first_destinations(first_point_count);
  std::vector<float> first_weights(first_point_count, 1.0f);
  std::vector<std::uint8_t> first_zero_flags(first_point_count, 0u);
  for (std::size_t point = 0; point < first_point_count; ++point)
    first_destinations[point] = static_cast<std::ptrdiff_t>(
        first_point_count - point - 1);
  const std::ptrdiff_t second_destinations[] = {1, 0};
  const float second_weights[] = {-0.0f, 0.5f};
  const std::uint8_t second_zero_flags[] = {1u, 0u};

  meep_cuda::dft_output_staging_operation_fp32 operations[] = {
      {&source, first_destinations.data(), first_weights.data(),
       first_zero_flags.data(), 70, first_point_count, 7,
       {0.75f, -0.25f}, 0, 0},
      {&source, second_destinations, second_weights, second_zero_flags,
       4, 2, 6, {-0.5f, 0.375f}, first_point_count, 2}};
  const std::uint32_t blocks[] = {0u, 0u, 1u};
  constexpr std::size_t operation_count = 2;
  constexpr std::size_t total_block_count = 3;
  constexpr std::size_t output_point_count = first_point_count + 2;
  constexpr std::size_t frequency_capacity = 4;

  meep_cuda::validate_dft_output_staging_operations_fp32(
      operations, blocks, operation_count, total_block_count,
      output_point_count, frequency_capacity);
  meep_cuda::validate_dft_output_staging_tile_fp32(
      operations, operation_count, 2, 4, frequency_capacity);
  meep_cuda::validate_dft_output_staging_tile_fp32(
      operations, operation_count, 5, 1, frequency_capacity);
  {
    const auto saved = operations[1];
    operations[1].zero_divisor_flags = nullptr;
    meep_cuda::validate_dft_output_staging_operations_fp32(
        operations, blocks, operation_count, total_block_count,
        output_point_count, frequency_capacity);
    operations[1] = saved;
  }
  meep_cuda::validate_dft_output_staging_operations_fp32(
      nullptr, nullptr, 0, 0, 0, frequency_capacity);

  expect_failure<std::invalid_argument>(
      "zero capacity", "capacity", [&] {
        meep_cuda::validate_dft_output_staging_operations_fp32(
            operations, blocks, operation_count, total_block_count,
            output_point_count, 0);
      });
  expect_failure<std::invalid_argument>(
      "null operations", "host operations", [&] {
        meep_cuda::validate_dft_output_staging_operations_fp32(
            nullptr, blocks, operation_count, total_block_count,
            output_point_count, frequency_capacity);
      });
  expect_failure<std::invalid_argument>(
      "empty topology output", "zero output points", [&] {
        meep_cuda::validate_dft_output_staging_operations_fp32(
            nullptr, nullptr, 0, 0, 1, frequency_capacity);
      });
  expect_failure<std::invalid_argument>(
      "empty topology blocks", "requires host operations", [&] {
        meep_cuda::validate_dft_output_staging_operations_fp32(
            nullptr, blocks, 0, 1, 0, frequency_capacity);
      });
  expect_failure<std::invalid_argument>(
      "empty topology descriptor", "must not provide operations", [&] {
        meep_cuda::validate_dft_output_staging_operations_fp32(
            operations, nullptr, 0, 0, 0, frequency_capacity);
      });
  expect_failure<std::invalid_argument>(
      "nonempty topology zero output", "output point count", [&] {
        meep_cuda::validate_dft_output_staging_operations_fp32(
            operations, blocks, operation_count, total_block_count,
            0, frequency_capacity);
      });

  {
    auto saved = operations[0];
    operations[0].dft_real_imag = nullptr;
    expect_failure<std::invalid_argument>("null source", "pointers", [&] {
      meep_cuda::validate_dft_output_staging_operations_fp32(
          operations, blocks, operation_count, total_block_count,
          output_point_count, frequency_capacity);
    });
    operations[0] = saved;
    operations[0].destination_indices = nullptr;
    expect_failure<std::invalid_argument>("null permutation", "pointers", [&] {
      meep_cuda::validate_dft_output_staging_operations_fp32(
          operations, blocks, operation_count, total_block_count,
          output_point_count, frequency_capacity);
    });
    operations[0] = saved;
    operations[0].point_weights = nullptr;
    expect_failure<std::invalid_argument>("null weights", "pointers", [&] {
      meep_cuda::validate_dft_output_staging_operations_fp32(
          operations, blocks, operation_count, total_block_count,
          output_point_count, frequency_capacity);
    });
    operations[0] = saved;
  }

  {
    auto saved = operations[0];
    operations[0].storage_point_count = 0;
    expect_failure<std::invalid_argument>("zero storage", "counts", [&] {
      meep_cuda::validate_dft_output_staging_operations_fp32(
          operations, blocks, operation_count, total_block_count,
          output_point_count, frequency_capacity);
    });
    operations[0] = saved;
    operations[0].point_count = 0;
    expect_failure<std::invalid_argument>("zero points", "counts", [&] {
      meep_cuda::validate_dft_output_staging_operations_fp32(
          operations, blocks, operation_count, total_block_count,
          output_point_count, frequency_capacity);
    });
    operations[0] = saved;
    operations[0].source_frequency_count = 0;
    expect_failure<std::invalid_argument>("zero source frequencies", "counts", [&] {
      meep_cuda::validate_dft_output_staging_operations_fp32(
          operations, blocks, operation_count, total_block_count,
          output_point_count, frequency_capacity);
    });
    operations[0] = saved;
    operations[0].storage_point_count = operations[0].point_count - 1;
    expect_failure<std::invalid_argument>("storage extent", "storage extent", [&] {
      meep_cuda::validate_dft_output_staging_operations_fp32(
          operations, blocks, operation_count, total_block_count,
          output_point_count, frequency_capacity);
    });
    operations[0] = saved;
  }

  {
    auto saved = operations[0];
    operations[0].inverse_stored_weight.real =
        std::numeric_limits<float>::infinity();
    expect_failure<std::invalid_argument>("nonfinite inverse", "finite", [&] {
      meep_cuda::validate_dft_output_staging_operations_fp32(
          operations, blocks, operation_count, total_block_count,
          output_point_count, frequency_capacity);
    });
    operations[0] = saved;
    const float original = first_weights[3];
    first_weights[3] = std::numeric_limits<float>::quiet_NaN();
    expect_failure<std::invalid_argument>("nonfinite weight", "finite", [&] {
      meep_cuda::validate_dft_output_staging_operations_fp32(
          operations, blocks, operation_count, total_block_count,
          output_point_count, frequency_capacity);
    });
    first_weights[3] = original;
    first_zero_flags[4] = 2u;
    expect_failure<std::invalid_argument>("invalid zero flag", "zero or one", [&] {
      meep_cuda::validate_dft_output_staging_operations_fp32(
          operations, blocks, operation_count, total_block_count,
          output_point_count, frequency_capacity);
    });
    first_zero_flags[4] = 0u;
  }

  {
    const std::ptrdiff_t original = first_destinations[0];
    first_destinations[0] = -1;
    expect_failure<std::invalid_argument>("negative destination", "out of range", [&] {
      meep_cuda::validate_dft_output_staging_operations_fp32(
          operations, blocks, operation_count, total_block_count,
          output_point_count, frequency_capacity);
    });
    first_destinations[0] = static_cast<std::ptrdiff_t>(first_point_count);
    expect_failure<std::invalid_argument>("large destination", "out of range", [&] {
      meep_cuda::validate_dft_output_staging_operations_fp32(
          operations, blocks, operation_count, total_block_count,
          output_point_count, frequency_capacity);
    });
    first_destinations[0] = first_destinations[1];
    expect_failure<std::invalid_argument>("duplicate destination", "permutation", [&] {
      meep_cuda::validate_dft_output_staging_operations_fp32(
          operations, blocks, operation_count, total_block_count,
          output_point_count, frequency_capacity);
    });
    first_destinations[0] = original;
  }

  {
    const auto saved = operations[1];
    operations[1].output_point_offset = first_point_count - 1;
    expect_failure<std::invalid_argument>("overlapping partition", "gapless", [&] {
      meep_cuda::validate_dft_output_staging_operations_fp32(
          operations, blocks, operation_count, total_block_count,
          output_point_count, frequency_capacity);
    });
    operations[1] = saved;
    operations[1].output_point_offset = first_point_count + 1;
    expect_failure<std::invalid_argument>("partition gap", "gapless", [&] {
      meep_cuda::validate_dft_output_staging_operations_fp32(
          operations, blocks, operation_count, total_block_count,
          output_point_count, frequency_capacity);
    });
    operations[1] = saved;
    expect_failure<std::invalid_argument>("incomplete partition", "cover", [&] {
      meep_cuda::validate_dft_output_staging_operations_fp32(
          operations, blocks, operation_count, total_block_count,
          output_point_count + 1, frequency_capacity);
    });
  }

  {
    const auto saved = operations[1];
    operations[1].block_start = 1;
    expect_failure<std::invalid_argument>("wrong block prefix", "prefixes", [&] {
      meep_cuda::validate_dft_output_staging_operations_fp32(
          operations, blocks, operation_count, total_block_count,
          output_point_count, frequency_capacity);
    });
    operations[1] = saved;
    expect_failure<std::invalid_argument>("wrong block total", "logical-block count", [&] {
      meep_cuda::validate_dft_output_staging_operations_fp32(
          operations, blocks, operation_count, total_block_count + 1,
          output_point_count, frequency_capacity);
    });
    expect_failure<std::invalid_argument>("null block map", "block-operation", [&] {
      meep_cuda::validate_dft_output_staging_operations_fp32(
          operations, nullptr, operation_count, total_block_count,
          output_point_count, frequency_capacity);
    });
    const std::uint32_t wrong_blocks[] = {0u, 1u, 1u};
    expect_failure<std::invalid_argument>("wrong block owner", "does not match", [&] {
      meep_cuda::validate_dft_output_staging_operations_fp32(
          operations, wrong_blocks, operation_count, total_block_count,
          output_point_count, frequency_capacity);
    });
  }

  {
    const auto saved = operations[0];
    operations[0].storage_point_count =
        std::numeric_limits<std::size_t>::max();
    operations[0].point_count = 1;
    operations[0].source_frequency_count = 2;
    expect_failure<std::overflow_error>("source work overflow", "work count", [&] {
      meep_cuda::validate_dft_output_staging_operations_fp32(
          operations, blocks, operation_count, total_block_count,
          output_point_count, frequency_capacity);
    });
    operations[0] = saved;
    operations[0].storage_point_count =
        std::numeric_limits<std::size_t>::max() / 2 + 1;
    operations[0].point_count = 1;
    operations[0].source_frequency_count = 1;
    expect_failure<std::overflow_error>("source interleaved overflow", "output count", [&] {
      meep_cuda::validate_dft_output_staging_operations_fp32(
          operations, blocks, operation_count, total_block_count,
          output_point_count, frequency_capacity);
    });
    operations[0] = saved;
    operations[0].storage_point_count =
        std::numeric_limits<std::size_t>::max() /
            (2 * sizeof(float)) +
        1;
    operations[0].point_count = 1;
    operations[0].source_frequency_count = 1;
    expect_failure<std::overflow_error>("source byte overflow", "byte count", [&] {
      meep_cuda::validate_dft_output_staging_operations_fp32(
          operations, blocks, operation_count, total_block_count,
          output_point_count, frequency_capacity);
    });
    operations[0] = saved;
  }
  expect_failure<std::overflow_error>("output work overflow", "work count", [&] {
    meep_cuda::validate_dft_output_staging_operations_fp32(
        operations, blocks, operation_count, total_block_count,
        std::numeric_limits<std::size_t>::max(), 2);
  });
  expect_failure<std::overflow_error>("output interleaved overflow", "output count", [&] {
    meep_cuda::validate_dft_output_staging_operations_fp32(
        operations, blocks, operation_count, total_block_count,
        std::numeric_limits<std::size_t>::max() / 2 + 1, 1);
  });
  expect_failure<std::overflow_error>("output byte overflow", "byte count", [&] {
    meep_cuda::validate_dft_output_staging_operations_fp32(
        operations, blocks, operation_count, total_block_count,
        std::numeric_limits<std::size_t>::max() /
                (2 * sizeof(float)) +
            1,
        1);
  });
  if (std::numeric_limits<std::size_t>::max() >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
    expect_failure<std::overflow_error>("operation index overflow", "index", [&] {
      meep_cuda::validate_dft_output_staging_operations_fp32(
          operations, blocks,
          static_cast<std::size_t>(
              std::numeric_limits<std::uint32_t>::max()) +
              1u,
          total_block_count, output_point_count, frequency_capacity);
    });

  expect_failure<std::invalid_argument>("tile zero capacity", "capacity", [&] {
    meep_cuda::validate_dft_output_staging_tile_fp32(
        operations, operation_count, 0, 1, 0);
  });
  expect_failure<std::invalid_argument>("tile zero count", "[1", [&] {
    meep_cuda::validate_dft_output_staging_tile_fp32(
        operations, operation_count, 0, 0, frequency_capacity);
  });
  expect_failure<std::invalid_argument>("tile over capacity", "[1", [&] {
    meep_cuda::validate_dft_output_staging_tile_fp32(
        operations, operation_count, 0, frequency_capacity + 1,
        frequency_capacity);
  });
  expect_failure<std::overflow_error>("tile range overflow", "overflow", [&] {
    meep_cuda::validate_dft_output_staging_tile_fp32(
        operations, operation_count,
        std::numeric_limits<std::size_t>::max(), 1,
        frequency_capacity);
  });
  expect_failure<std::invalid_argument>("tile null operations", "host operations", [&] {
    meep_cuda::validate_dft_output_staging_tile_fp32(
        nullptr, operation_count, 0, 1, frequency_capacity);
  });
  expect_failure<std::invalid_argument>("tile source end", "source frequency range", [&] {
    meep_cuda::validate_dft_output_staging_tile_fp32(
        operations, operation_count, 4, 3, frequency_capacity);
  });
  if (std::numeric_limits<std::size_t>::max() >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
    expect_failure<std::overflow_error>("tile operation index overflow", "index", [&] {
      meep_cuda::validate_dft_output_staging_tile_fp32(
          operations,
          static_cast<std::size_t>(
              std::numeric_limits<std::uint32_t>::max()) +
              1u,
          0, 1, frequency_capacity);
    });
  {
    const auto saved = operations[1];
    operations[1].source_frequency_count = 0;
    expect_failure<std::invalid_argument>("tile zero source", "source frequency range", [&] {
      meep_cuda::validate_dft_output_staging_tile_fp32(
          operations, operation_count, 0, 1, frequency_capacity);
    });
    operations[1] = saved;
  }

  std::cout << "PASS: DFT output staging host validation enforces finite "
               "metadata, 0/1 flags, local permutations, exact gapless "
               "packed partitions, fixed-capacity block maps, tile source "
               "ranges, and checked allocation arithmetic without a CUDA "
               "driver\n";
  return 0;
}
