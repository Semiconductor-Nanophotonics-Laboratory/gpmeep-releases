#ifndef MEEP_CUDA_DETAIL_LAUNCH_HPP
#define MEEP_CUDA_DETAIL_LAUNCH_HPP

#include <cstddef>
#include <limits>

namespace meep_cuda {
namespace detail {

// Rounds a point count up to a CUDA warp without overflowing size_t.  The
// saturated result is sufficient for launch selection because callers cap it
// to the device's threads-per-block limit before converting to int.
inline std::size_t warp_aligned_point_count(std::size_t point_count) {
  constexpr std::size_t warp_size = 32u;
  const std::size_t remainder = point_count % warp_size;
  if (remainder == 0u) return point_count;
  const std::size_t increment = warp_size - remainder;
  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  return point_count > maximum - increment
             ? maximum
             : point_count + increment;
}

} // namespace detail
} // namespace meep_cuda

#endif
