#ifndef MEEP_GPU_GRID_INDEX_HPP
#define MEEP_GPU_GRID_INDEX_HPP

#include "meep.hpp"
#include "gpu_backend_internal.hpp"

#include <limits>
#include <stdexcept>

namespace meep {
namespace gpu {
namespace detail {

inline int structured_coefficient_start(const grid_volume &gv,
                                        const ivec &is,
                                        direction coefficient_direction) {
  if (coefficient_direction == NO_DIRECTION) return -1;
  const std::ptrdiff_t index =
      is.in_direction(coefficient_direction) -
      gv.little_corner().in_direction(coefficient_direction);
  if (index < 0 || index > std::numeric_limits<int>::max())
    throw std::out_of_range(
        "Meep CUDA coefficient index is outside the supported integer range");
  return static_cast<int>(index);
}

inline index_space_fp32 make_index_space_fp32(
    const grid_volume &gv, const ivec &is, const ivec &ie,
    direction coefficient_direction = NO_DIRECTION,
    direction coefficient2_direction = NO_DIRECTION) {
  const ivec offset = is - gv.little_corner();
  std::size_t extents[3];
  std::ptrdiff_t strides[3];
  for (int axis = 0; axis < 3; ++axis) {
    const std::ptrdiff_t extent =
        (ie.yucky_val(axis) - is.yucky_val(axis)) / 2 + 1;
    if (extent <= 0)
      throw std::invalid_argument(
          "Meep CUDA structured index extent must be positive");
    extents[axis] = static_cast<std::size_t>(extent);
    strides[axis] = gv.stride(gv.yucky_direction(axis));
  }
  const auto coefficient_stride =
      [&gv](int axis, direction coefficient) {
        return coefficient != NO_DIRECTION &&
                       gv.yucky_direction(axis) == coefficient
                   ? 2
                   : 0;
      };
  return {
      offset.yucky_val(0) / 2 * strides[0] +
          offset.yucky_val(1) / 2 * strides[1] +
          offset.yucky_val(2) / 2 * strides[2],
      extents[0],
      extents[1],
      extents[2],
      strides[0],
      strides[1],
      strides[2],
      structured_coefficient_start(gv, is, coefficient_direction),
      coefficient_stride(0, coefficient_direction),
      coefficient_stride(1, coefficient_direction),
      coefficient_stride(2, coefficient_direction),
      structured_coefficient_start(gv, is, coefficient2_direction),
      coefficient_stride(0, coefficient2_direction),
      coefficient_stride(1, coefficient2_direction),
      coefficient_stride(2, coefficient2_direction)};
}

} // namespace detail
} // namespace gpu
} // namespace meep

#endif
