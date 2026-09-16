#ifndef MEEP_CUDA_DETAIL_INDEX_SPACE_HPP
#define MEEP_CUDA_DETAIL_INDEX_SPACE_HPP

#include <cstddef>

#ifdef __CUDACC__
#define MEEP_CUDA_INDEX_HOST_DEVICE __host__ __device__
#else
#define MEEP_CUDA_INDEX_HOST_DEVICE
#endif

namespace meep_cuda {

// Compact description of Meep's three nested LOOP_OVER_IVECS loops.
// A coefficient start of -1 denotes an unused PML coefficient direction.
struct index_space_fp32 {
  std::ptrdiff_t field_start;
  std::size_t extent1;
  std::size_t extent2;
  std::size_t extent3;
  std::ptrdiff_t field_stride1;
  std::ptrdiff_t field_stride2;
  std::ptrdiff_t field_stride3;
  int coefficient_start;
  int coefficient_stride1;
  int coefficient_stride2;
  int coefficient_stride3;
  int coefficient2_start;
  int coefficient2_stride1;
  int coefficient2_stride2;
  int coefficient2_stride3;
};

namespace detail {

struct decoded_index_fp32 {
  std::ptrdiff_t field;
  int coefficient;
  int coefficient2;
};

MEEP_CUDA_INDEX_HOST_DEVICE inline std::ptrdiff_t field_index_contribution(
    std::size_t coordinate, std::ptrdiff_t stride) {
  return stride ? static_cast<std::ptrdiff_t>(coordinate) * stride : 0;
}

MEEP_CUDA_INDEX_HOST_DEVICE inline int coefficient_index_contribution(
    std::size_t coordinate, int stride) {
  return stride ? static_cast<int>(coordinate) * stride : 0;
}

MEEP_CUDA_INDEX_HOST_DEVICE inline decoded_index_fp32
decode_index_coordinates_fp32(const index_space_fp32 &space,
                              std::size_t coordinate1,
                              std::size_t coordinate2,
                              std::size_t coordinate3) {
  return {
      space.field_start +
          field_index_contribution(coordinate1, space.field_stride1) +
          field_index_contribution(coordinate2, space.field_stride2) +
          field_index_contribution(coordinate3, space.field_stride3),
      space.coefficient_start < 0
          ? -1
          : space.coefficient_start +
                coefficient_index_contribution(
                    coordinate1, space.coefficient_stride1) +
                coefficient_index_contribution(
                    coordinate2, space.coefficient_stride2) +
                coefficient_index_contribution(
                    coordinate3, space.coefficient_stride3),
      space.coefficient2_start < 0
          ? -1
          : space.coefficient2_start +
                coefficient_index_contribution(
                    coordinate1, space.coefficient2_stride1) +
                coefficient_index_contribution(
                    coordinate2, space.coefficient2_stride2) +
                coefficient_index_contribution(
                    coordinate3, space.coefficient2_stride3)};
}

MEEP_CUDA_INDEX_HOST_DEVICE inline decoded_index_fp32
decode_index_space_fp32(const index_space_fp32 &space,
                        std::size_t linear_index) {
  const std::size_t coordinate3 = linear_index % space.extent3;
  linear_index /= space.extent3;
  const std::size_t coordinate2 = linear_index % space.extent2;
  const std::size_t coordinate1 = linear_index / space.extent2;
  return decode_index_coordinates_fp32(
      space, coordinate1, coordinate2, coordinate3);
}

} // namespace detail
} // namespace meep_cuda

#undef MEEP_CUDA_INDEX_HOST_DEVICE

#endif
