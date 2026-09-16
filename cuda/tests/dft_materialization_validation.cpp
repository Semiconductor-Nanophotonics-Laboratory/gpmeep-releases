#include "meep_cuda/runtime.hpp"

#include <cstddef>
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
  float source[24] = {};
  const std::ptrdiff_t destinations[] = {4, 1, 6};
  const float weights[] = {1.0f, -0.5f, 0.0f};
  meep_cuda::dft_materialization_operation_fp32 selected = {
      source, destinations, weights,
      3, 3, 4, 1, 2, {0.75f, -0.25f}, 0};
  const std::uint32_t selected_blocks[] = {0};

  meep_cuda::validate_dft_materialization_operations_fp32(
      &selected, selected_blocks, 1, 1, 7, 2);
  meep_cuda::validate_dft_materialization_operations_fp32(
      nullptr, nullptr, 0, 0, 7, 2);

  const std::ptrdiff_t all_destinations[] = {2, 0};
  const float all_weights[] = {0.25f, 2.0f};
  meep_cuda::dft_materialization_operation_fp32 all_frequencies = {
      source, all_destinations, all_weights,
      2, 2, 3, 0, 3, {1.0f, 0.0f}, 0};
  meep_cuda::validate_dft_materialization_operations_fp32(
      &all_frequencies, selected_blocks, 1, 1, 3, 3);

  {
    const std::uint8_t invalid_zero_divisor_flags[] = {0u, 2u, 0u};
    auto invalid = selected;
    invalid.zero_divisor_flags = invalid_zero_divisor_flags;
    expect_failure<std::invalid_argument>(
        "invalid zero-divisor flag", "zero-divisor flag", [&] {
          meep_cuda::validate_dft_materialization_operations_fp32(
              &invalid, selected_blocks, 1, 1, 7, 2);
        });
  }
  {
    const std::uint8_t invalid_publication_flags[] = {1u, 2u, 1u};
    auto invalid = selected;
    invalid.publication_flags = invalid_publication_flags;
    expect_failure<std::invalid_argument>(
        "invalid publication flag", "publication flag", [&] {
          meep_cuda::validate_dft_materialization_operations_fp32(
              &invalid, selected_blocks, 1, 1, 7, 2);
        });
  }

  expect_failure<std::invalid_argument>(
      "zero output points", "dimensions", [&] {
        meep_cuda::validate_dft_materialization_operations_fp32(
            nullptr, nullptr, 0, 0, 0, 2);
      });
  expect_failure<std::invalid_argument>(
      "zero output frequencies", "dimensions", [&] {
        meep_cuda::validate_dft_materialization_operations_fp32(
            nullptr, nullptr, 0, 0, 7, 0);
      });
  expect_failure<std::invalid_argument>(
      "null operations", "host operations", [&] {
        meep_cuda::validate_dft_materialization_operations_fp32(
            nullptr, selected_blocks, 1, 1, 7, 2);
      });
  expect_failure<std::invalid_argument>(
      "ownerless blocks", "requires host operations", [&] {
        meep_cuda::validate_dft_materialization_operations_fp32(
            nullptr, selected_blocks, 0, 1, 7, 2);
      });
  expect_failure<std::invalid_argument>(
      "null block map", "host block-operation indices", [&] {
        meep_cuda::validate_dft_materialization_operations_fp32(
            &selected, nullptr, 1, 1, 7, 2);
      });
  {
    const std::uint32_t wrong_owner[] = {1};
    expect_failure<std::invalid_argument>(
        "wrong block owner", "does not match descriptor prefixes", [&] {
          meep_cuda::validate_dft_materialization_operations_fp32(
              &selected, wrong_owner, 1, 1, 7, 2);
        });
  }

  {
    auto invalid = selected;
    invalid.destination_indices = nullptr;
    expect_failure<std::invalid_argument>(
        "null map", "pointers", [&] {
          meep_cuda::validate_dft_materialization_operations_fp32(
              &invalid, selected_blocks, 1, 1, 7, 2);
        });
  }
  {
    auto invalid = selected;
    invalid.point_count = 0;
    expect_failure<std::invalid_argument>(
        "zero count", "counts", [&] {
          meep_cuda::validate_dft_materialization_operations_fp32(
              &invalid, selected_blocks, 1, 1, 7, 2);
        });
  }
  {
    auto invalid = selected;
    invalid.point_count = invalid.storage_point_count + 1;
    expect_failure<std::invalid_argument>(
        "source extent", "storage extent", [&] {
          meep_cuda::validate_dft_materialization_operations_fp32(
              &invalid, selected_blocks, 1, 1, 7, 2);
        });
  }
  {
    auto invalid = selected;
    invalid.source_frequency_start = invalid.source_frequency_count;
    expect_failure<std::invalid_argument>(
        "source start", "frequency range", [&] {
          meep_cuda::validate_dft_materialization_operations_fp32(
              &invalid, selected_blocks, 1, 1, 7, 2);
        });
  }
  {
    auto invalid = selected;
    invalid.selected_frequency_count = 4;
    expect_failure<std::invalid_argument>(
        "source end", "frequency range", [&] {
          meep_cuda::validate_dft_materialization_operations_fp32(
              &invalid, selected_blocks, 1, 1, 7, 4);
        });
  }
  {
    auto invalid = selected;
    invalid.selected_frequency_count = 1;
    expect_failure<std::invalid_argument>(
        "output frequency mismatch", "do not match output", [&] {
          meep_cuda::validate_dft_materialization_operations_fp32(
              &invalid, selected_blocks, 1, 1, 7, 2);
        });
  }
  {
    auto invalid = selected;
    invalid.inverse_stored_weight.real =
        std::numeric_limits<float>::infinity();
    expect_failure<std::invalid_argument>(
        "nonfinite inverse", "must be finite", [&] {
          meep_cuda::validate_dft_materialization_operations_fp32(
              &invalid, selected_blocks, 1, 1, 7, 2);
        });
  }
  {
    const float invalid_weights[] = {
        1.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f};
    auto invalid = selected;
    invalid.point_weights = invalid_weights;
    expect_failure<std::invalid_argument>(
        "nonfinite point weight", "must be finite", [&] {
          meep_cuda::validate_dft_materialization_operations_fp32(
              &invalid, selected_blocks, 1, 1, 7, 2);
        });
  }
  {
    const std::ptrdiff_t invalid_destinations[] = {4, -1, 6};
    auto invalid = selected;
    invalid.destination_indices = invalid_destinations;
    expect_failure<std::invalid_argument>(
        "negative destination", "out of range", [&] {
          meep_cuda::validate_dft_materialization_operations_fp32(
              &invalid, selected_blocks, 1, 1, 7, 2);
        });
  }
  {
    const std::ptrdiff_t invalid_destinations[] = {4, 7, 6};
    auto invalid = selected;
    invalid.destination_indices = invalid_destinations;
    expect_failure<std::invalid_argument>(
        "large destination", "out of range", [&] {
          meep_cuda::validate_dft_materialization_operations_fp32(
              &invalid, selected_blocks, 1, 1, 7, 2);
        });
  }
  {
    const std::ptrdiff_t duplicate_destinations[] = {4, 1, 4};
    auto invalid = selected;
    invalid.destination_indices = duplicate_destinations;
    expect_failure<std::invalid_argument>(
        "duplicate without publication ordering", "publication flags", [&] {
          meep_cuda::validate_dft_materialization_operations_fp32(
              &invalid, selected_blocks, 1, 1, 7, 2);
        });
    const std::uint8_t publication_flags[] = {0u, 1u, 1u};
    invalid.publication_flags = publication_flags;
    meep_cuda::validate_dft_materialization_operations_fp32(
        &invalid, selected_blocks, 1, 1, 7, 2);
  }
  {
    const std::ptrdiff_t second_destinations[] = {1};
    const float second_weights[] = {1.0f};
    const std::uint8_t selected_publication_flags[] = {1u, 0u, 1u};
    const std::uint8_t second_publication_flags[] = {1u};
    auto selected_with_publication = selected;
    selected_with_publication.publication_flags =
        selected_publication_flags;
    meep_cuda::dft_materialization_operation_fp32 operations[] = {
        selected_with_publication,
        {source, second_destinations, second_weights,
         1, 1, 4, 1, 2, {1.0f, 0.0f}, 1, nullptr,
         second_publication_flags}};
    const std::uint32_t operation_blocks[] = {0, 1};
    meep_cuda::validate_dft_materialization_operations_fp32(
        operations, operation_blocks, 2, 2, 7, 2);
    const std::uint8_t wrong_selected_publication_flags[] = {1u, 1u, 1u};
    operations[0].publication_flags = wrong_selected_publication_flags;
    expect_failure<std::invalid_argument>(
        "incorrect duplicate last-writer", "publication flags", [&] {
          meep_cuda::validate_dft_materialization_operations_fp32(
              operations, operation_blocks, 2, 2, 7, 2);
        });
  }
  {
    auto invalid = selected;
    invalid.block_start = 1;
    expect_failure<std::invalid_argument>(
        "noncontiguous prefix", "prefixes", [&] {
          meep_cuda::validate_dft_materialization_operations_fp32(
              &invalid, selected_blocks, 1, 1, 7, 2);
        });
  }
  expect_failure<std::invalid_argument>(
      "wrong total", "do not match", [&] {
        meep_cuda::validate_dft_materialization_operations_fp32(
            &selected, selected_blocks, 1, 2, 7, 2);
      });

  {
    auto invalid = selected;
    invalid.storage_point_count =
        std::numeric_limits<std::size_t>::max();
    invalid.point_count = 1;
    invalid.source_frequency_count = 2;
    invalid.source_frequency_start = 0;
    invalid.selected_frequency_count = 2;
    expect_failure<std::overflow_error>(
        "source point-frequency overflow", "work count", [&] {
          meep_cuda::validate_dft_materialization_operations_fp32(
              &invalid, selected_blocks, 1, 1, 7, 2);
        });
  }
  {
    auto invalid = selected;
    invalid.storage_point_count =
        std::numeric_limits<std::size_t>::max() / 2 + 1;
    invalid.point_count = 1;
    invalid.source_frequency_count = 1;
    invalid.source_frequency_start = 0;
    invalid.selected_frequency_count = 1;
    expect_failure<std::overflow_error>(
        "source interleaved overflow", "output count", [&] {
          meep_cuda::validate_dft_materialization_operations_fp32(
              &invalid, selected_blocks, 1, 1, 7, 1);
        });
  }
  {
    auto invalid = selected;
    invalid.storage_point_count =
        std::numeric_limits<std::size_t>::max() /
            (2 * sizeof(float)) +
        1;
    invalid.point_count = 1;
    invalid.source_frequency_count = 1;
    invalid.source_frequency_start = 0;
    invalid.selected_frequency_count = 1;
    expect_failure<std::overflow_error>(
        "source byte overflow", "byte count", [&] {
          meep_cuda::validate_dft_materialization_operations_fp32(
              &invalid, selected_blocks, 1, 1, 7, 1);
        });
  }
  expect_failure<std::overflow_error>(
      "output point-frequency overflow", "work count", [&] {
        meep_cuda::validate_dft_materialization_operations_fp32(
            nullptr, nullptr, 0, 0,
            std::numeric_limits<std::size_t>::max(), 2);
      });
  expect_failure<std::overflow_error>(
      "output interleaved overflow", "output count", [&] {
        meep_cuda::validate_dft_materialization_operations_fp32(
            nullptr, nullptr, 0, 0,
            std::numeric_limits<std::size_t>::max() / 2 + 1, 1);
      });
  expect_failure<std::overflow_error>(
      "output byte overflow", "byte count", [&] {
        meep_cuda::validate_dft_materialization_operations_fp32(
            nullptr, nullptr, 0, 0,
            std::numeric_limits<std::size_t>::max() /
                    (2 * sizeof(float)) +
                1,
            1);
      });

  if (std::numeric_limits<std::size_t>::max() >
      static_cast<std::size_t>(
          std::numeric_limits<std::uint32_t>::max()))
    expect_failure<std::overflow_error>(
        "operation index overflow", "index", [&] {
          meep_cuda::validate_dft_materialization_operations_fp32(
              &selected, selected_blocks,
              static_cast<std::size_t>(
                  std::numeric_limits<std::uint32_t>::max()) +
                  1u,
              1, 7, 2);
        });

  const float *const collapse_source =
      reinterpret_cast<const float *>(static_cast<std::uintptr_t>(0x10000));
  float *const collapse_destination =
      reinterpret_cast<float *>(static_cast<std::uintptr_t>(0x20000));
  const meep_cuda::dft_collapse_layout_fp32 valid_collapse = {
      2, {3, 4, 1}, {1u, 0u, 0u}};
  expect_failure<std::invalid_argument>(
      "collapse null source", "non-null", [&] {
        meep_cuda::collapse_dft_array_fp32(
            nullptr, collapse_destination, 12, 4, 2, valid_collapse);
      });
  expect_failure<std::invalid_argument>(
      "collapse null destination", "non-null", [&] {
        meep_cuda::collapse_dft_array_fp32(
            collapse_source, nullptr, 12, 4, 2, valid_collapse);
      });
  expect_failure<std::invalid_argument>(
      "collapse zero full count", "nonzero", [&] {
        meep_cuda::collapse_dft_array_fp32(
            collapse_source, collapse_destination, 0, 4, 2,
            valid_collapse);
      });
  expect_failure<std::invalid_argument>(
      "collapse zero reduced count", "nonzero", [&] {
        meep_cuda::collapse_dft_array_fp32(
            collapse_source, collapse_destination, 12, 0, 2,
            valid_collapse);
      });
  expect_failure<std::invalid_argument>(
      "collapse zero frequency count", "nonzero", [&] {
        meep_cuda::collapse_dft_array_fp32(
            collapse_source, collapse_destination, 12, 4, 0,
            valid_collapse);
      });
  {
    auto invalid = valid_collapse;
    invalid.full_rank = 0;
    expect_failure<std::invalid_argument>(
        "collapse zero rank", "rank", [&] {
          meep_cuda::collapse_dft_array_fp32(
              collapse_source, collapse_destination, 12, 4, 2, invalid);
        });
  }
  {
    auto invalid = valid_collapse;
    invalid.full_rank = 4;
    expect_failure<std::invalid_argument>(
        "collapse excessive rank", "rank", [&] {
          meep_cuda::collapse_dft_array_fp32(
              collapse_source, collapse_destination, 12, 4, 2, invalid);
        });
  }
  {
    auto invalid = valid_collapse;
    invalid.full_dims[0] = 0;
    expect_failure<std::invalid_argument>(
        "collapse zero active extent", "nonzero", [&] {
          meep_cuda::collapse_dft_array_fp32(
              collapse_source, collapse_destination, 12, 4, 2, invalid);
        });
  }
  {
    auto invalid = valid_collapse;
    invalid.collapsed[0] = 2u;
    expect_failure<std::invalid_argument>(
        "collapse invalid flag", "flags", [&] {
          meep_cuda::collapse_dft_array_fp32(
              collapse_source, collapse_destination, 12, 4, 2, invalid);
        });
  }
  {
    auto invalid = valid_collapse;
    invalid.full_dims[2] = 2;
    expect_failure<std::invalid_argument>(
        "collapse inactive extent", "inactive dimensions", [&] {
          meep_cuda::collapse_dft_array_fp32(
              collapse_source, collapse_destination, 12, 4, 2, invalid);
        });
  }
  {
    auto invalid = valid_collapse;
    invalid.collapsed[0] = 0u;
    expect_failure<std::invalid_argument>(
        "collapse missing axis", "collapsed dimension", [&] {
          meep_cuda::collapse_dft_array_fp32(
              collapse_source, collapse_destination, 12, 12, 2, invalid);
        });
  }
  expect_failure<std::invalid_argument>(
      "collapse full product mismatch", "allocation extents", [&] {
        meep_cuda::collapse_dft_array_fp32(
            collapse_source, collapse_destination, 11, 4, 2,
            valid_collapse);
      });
  expect_failure<std::invalid_argument>(
      "collapse reduced product mismatch", "allocation extents", [&] {
        meep_cuda::collapse_dft_array_fp32(
            collapse_source, collapse_destination, 12, 3, 2,
            valid_collapse);
      });
  {
    const meep_cuda::dft_collapse_layout_fp32 overflow = {
        2, {std::numeric_limits<std::size_t>::max(), 2, 1},
        {1u, 0u, 0u}};
    expect_failure<std::overflow_error>(
        "collapse full product overflow", "full extent", [&] {
          meep_cuda::collapse_dft_array_fp32(
              collapse_source, collapse_destination, 1, 2, 1, overflow);
        });
  }
  expect_failure<std::invalid_argument>(
      "collapse overlapping allocations", "must not overlap", [&] {
        meep_cuda::collapse_dft_array_fp32(
            collapse_source,
            reinterpret_cast<float *>(
                static_cast<std::uintptr_t>(0x10004)),
            12, 4, 2, valid_collapse);
      });
  expect_failure<std::invalid_argument>(
      "collapse reverse overlapping allocations", "must not overlap", [&] {
        meep_cuda::collapse_dft_array_fp32(
            reinterpret_cast<const float *>(
                static_cast<std::uintptr_t>(0x20004)),
            reinterpret_cast<float *>(
                static_cast<std::uintptr_t>(0x20000)),
            12, 4, 2, valid_collapse);
      });
  expect_failure<std::overflow_error>(
      "collapse source address overflow", "address extent", [&] {
        meep_cuda::collapse_dft_array_fp32(
            reinterpret_cast<const float *>(
                std::numeric_limits<std::uintptr_t>::max() - 8),
            collapse_destination, 12, 4, 2, valid_collapse);
      });
  expect_failure<std::overflow_error>(
      "collapse destination address overflow", "address extent", [&] {
        meep_cuda::collapse_dft_array_fp32(
            collapse_source,
            reinterpret_cast<float *>(
                std::numeric_limits<std::uintptr_t>::max() - 8),
            12, 4, 2, valid_collapse);
      });

  std::cout << "PASS: DFT materialization host validation accepts arbitrary "
               "maps, ordered last-writer overlaps, and selected/all-frequency "
               "extents while rejecting malformed publication metadata, "
               "nonfinite values, bounds errors, overflow, and invalid DFT "
               "collapse layouts or overlapping allocations\n";
  return 0;
}
