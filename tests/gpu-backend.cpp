#include "meep/gpu.hpp"
#include "meep/meep-config.h"
#include "gpu_grid_index.hpp"
#include "meep_cuda/detail/index_space.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static_assert(sizeof(meep::gpu::resident_statistics) ==
                  4 * sizeof(std::uint64_t),
              "resident_statistics ABI must remain four uint64_t fields");
static_assert(sizeof(meep::gpu::phase_batch_policy_statistics) ==
                  12 * sizeof(std::uint64_t),
              "phase_batch_policy_statistics ABI must remain twelve "
              "uint64_t fields");

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

void check_structured_grid_indices() {
  const meep::grid_volume gv = meep::vol3d(2.0, 3.0, 4.0, 2.0);
  const meep::component component = meep::Ex;
  const meep::ivec is = gv.little_owned_corner0(component);
  const meep::ivec ie = gv.big_corner();
  const meep::gpu::detail::index_space_fp32 host_space =
      meep::gpu::detail::make_index_space_fp32(
          gv, is, ie, meep::X, meep::Y);
  const meep_cuda::index_space_fp32 device_space = {
      host_space.field_start,
      host_space.extent1,
      host_space.extent2,
      host_space.extent3,
      host_space.field_stride1,
      host_space.field_stride2,
      host_space.field_stride3,
      host_space.coefficient_start,
      host_space.coefficient_stride1,
      host_space.coefficient_stride2,
      host_space.coefficient_stride3,
      host_space.coefficient2_start,
      host_space.coefficient2_stride1,
      host_space.coefficient2_stride2,
      host_space.coefficient2_stride3};

  std::size_t linear = 0;
  LOOP_OVER_IVECS(gv, is, ie, field_index) {
    const meep_cuda::detail::decoded_index_fp32 decoded =
        meep_cuda::detail::decode_index_space_fp32(device_space, linear);
    require(decoded.field == field_index,
            "structured field index disagrees with LOOP_OVER_IVECS");
    const int expected_x =
        is.in_direction(meep::X) -
        gv.little_corner().in_direction(meep::X) +
        (gv.yucky_direction(0) == meep::X ? 2 * loop_i1 : 0) +
        (gv.yucky_direction(1) == meep::X ? 2 * loop_i2 : 0) +
        (gv.yucky_direction(2) == meep::X ? 2 * loop_i3 : 0);
    const int expected_y =
        is.in_direction(meep::Y) -
        gv.little_corner().in_direction(meep::Y) +
        (gv.yucky_direction(0) == meep::Y ? 2 * loop_i1 : 0) +
        (gv.yucky_direction(1) == meep::Y ? 2 * loop_i2 : 0) +
        (gv.yucky_direction(2) == meep::Y ? 2 * loop_i3 : 0);
    require(decoded.coefficient == expected_x,
            "structured first PML index disagrees with LOOP_OVER_IVECS");
    require(decoded.coefficient2 == expected_y,
            "structured second PML index disagrees with LOOP_OVER_IVECS");
    ++linear;
  }
  require(linear ==
              host_space.extent1 * host_space.extent2 *
                  host_space.extent3,
          "structured index point count disagrees with LOOP_OVER_IVECS");
}

void check_rank_device_mapping() {
  const int devices[] = {2, 4, 7, 9};
  require(meep::gpu::detail::choose_rank_device_ordinal(devices, 4, 0, false) == 2,
          "node rank zero must select the first compatible GPU");
  require(meep::gpu::detail::choose_rank_device_ordinal(devices, 4, 3, false) == 9,
          "node rank must select the matching compatible GPU");
  require(meep::gpu::detail::choose_rank_device_ordinal(devices, 4, 5, true) == 4,
          "explicit oversubscription must wrap node ranks");
  const int isolated[] = {0};
  require(meep::gpu::detail::choose_rank_device_ordinal(isolated, 1, 19, false) == 0,
          "one-device-per-rank visibility must always select ordinal zero");
  require(meep::gpu::detail::choose_rank_device_ordinal(nullptr, 0, 0, false) == -1,
          "an empty compatible-device set must return no device");

  bool rejected_oversubscription = false;
  try {
    (void)meep::gpu::detail::choose_rank_device_ordinal(devices, 4, 4, false);
  }
  catch (const std::runtime_error &) {
    rejected_oversubscription = true;
  }
  require(rejected_oversubscription,
          "implicit multi-rank GPU oversubscription must be rejected");
}

void expect_tile_rejection(const meep::grid_volume &root,
                           const std::vector<meep::grid_volume> &tiles, std::size_t base,
                           const char *message) {
  bool rejected = false;
  try {
    meep::check_tiles(root, tiles, base);
  }
  catch (const std::runtime_error &) {
    rejected = true;
  }
  require(rejected, message);
}

void check_linear_tile_validation() {
  int legacy_split_point = -1;
  meep::direction legacy_split_direction = meep::X;
  meep::vol1d(2.0, 1.0).tile_split(legacy_split_point, legacy_split_direction);
  require(legacy_split_point == 0 && legacy_split_direction == meep::NO_DIRECTION,
          "public tiny-volume tile_split behavior must remain backward compatible");
  meep::volcyl(4.0, 2.0, 1.0).tile_split(legacy_split_point, legacy_split_direction);
  require(legacy_split_point == 2 && legacy_split_direction == meep::X,
          "public cylindrical tile_split priority must remain backward compatible");

  const std::vector<meep::grid_volume> roots = {
      meep::vol1d(17.0, 1.0), meep::vol2d(15.0, 13.0, 1.0),
      meep::vol3d(11.0, 9.0, 7.0, 1.0), meep::volcyl(13.0, 11.0, 1.0)};
  const std::size_t bases[] = {2, 7, 32, 128};
  for (auto root : roots) {
    root.shift_origin(meep::start_at_direction(root.dim), 18);
    for (const std::size_t base : bases) {
      std::vector<meep::grid_volume> tiles;
      meep::split_into_tiles(root, &tiles, base);
      meep::check_tiles(root, tiles, base);
      meep::check_tiles(root, tiles); // independent order-agnostic small-volume oracle
    }
  }

  const meep::grid_volume root = meep::vol3d(11.0, 9.0, 7.0, 1.0);
  std::vector<meep::grid_volume> tiles;
  meep::split_into_tiles(root, &tiles, 32);

  std::vector<meep::grid_volume> missing = tiles;
  missing.pop_back();
  expect_tile_rejection(root, missing, 32, "linear tile validation must reject a missing leaf");

  std::vector<meep::grid_volume> trailing = tiles;
  trailing.push_back(tiles.back());
  expect_tile_rejection(root, trailing, 32,
                        "linear tile validation must reject a trailing duplicate leaf");

  std::vector<meep::grid_volume> reordered = tiles;
  std::swap(reordered[0], reordered[1]);
  expect_tile_rejection(root, reordered, 32,
                        "generated-partition validation must reject reordered leaves");

  std::vector<meep::grid_volume> shifted = tiles;
  shifted[0].shift_origin(meep::X, 2);
  expect_tile_rejection(root, shifted, 32,
                        "linear tile validation must reject shifted leaf geometry");

  std::vector<meep::grid_volume> resized = tiles;
  resized[0].set_num_direction(meep::X, resized[0].nx() + 1);
  expect_tile_rejection(root, resized, 32,
                        "linear tile validation must reject resized leaf geometry");

  std::vector<meep::grid_volume> wrong_resolution = tiles;
  wrong_resolution[0].a *= 2.0;
  expect_tile_rejection(root, wrong_resolution, 32,
                        "linear tile validation must reject inconsistent resolution");

  std::vector<meep::grid_volume> wrong_dimension = tiles;
  wrong_dimension[0].dim = static_cast<meep::ndim>(99);
  expect_tile_rejection(root, wrong_dimension, 32,
                        "linear tile validation must reject invalid dimensions safely");
  expect_tile_rejection(root, tiles, 128,
                        "linear tile validation must reject a partition generated for another base");

  bool base_one_rejected = false;
  try {
    std::vector<meep::grid_volume> invalid_base_tiles;
    meep::split_into_tiles(root, &invalid_base_tiles, 1);
  }
  catch (const std::runtime_error &) {
    base_one_rejected = true;
  }
  require(base_one_rejected, "tile base one must be rejected instead of selecting NO_DIRECTION");

  meep::grid_volume overflow_root = meep::vol3d(1.0, 1.0, 1.0, 1.0);
  overflow_root.set_num_direction(meep::X, std::numeric_limits<int>::max());
  overflow_root.set_num_direction(meep::Y, std::numeric_limits<int>::max());
  overflow_root.set_num_direction(meep::Z, std::numeric_limits<int>::max());
  expect_tile_rejection(overflow_root, {overflow_root}, 2,
                        "tile validation must reject a grid-point count overflow safely");

  bool unsplit_end_overflow_rejected = false;
  try {
    meep::grid_volume unsplit_end_overflow = meep::vol1d(1.0, 1.0);
    unsplit_end_overflow.set_origin(meep::Z, std::numeric_limits<int>::max() - 1);
    std::vector<meep::grid_volume> unsplit_end_overflow_tiles;
    meep::split_into_tiles(unsplit_end_overflow, &unsplit_end_overflow_tiles, 2);
  }
  catch (const std::runtime_error &) {
    unsplit_end_overflow_rejected = true;
  }
  require(unsplit_end_overflow_rejected,
          "an unsplit tile with an overflowing end corner must be rejected safely");

  bool origin_overflow_rejected = false;
  try {
    meep::grid_volume origin_overflow = meep::vol1d(4.0, 1.0);
    origin_overflow.set_origin(meep::Z, std::numeric_limits<int>::max() - 2);
    std::vector<meep::grid_volume> origin_overflow_tiles;
    meep::split_into_tiles(origin_overflow, &origin_overflow_tiles, 2);
  }
  catch (const std::runtime_error &) {
    origin_overflow_rejected = true;
  }
  require(origin_overflow_rejected, "tile splitting must reject an integer origin overflow safely");

  meep::grid_volume stale_origin = meep::vol1d(1.0, 1.0);
  stale_origin.set_origin(meep::Z, 2);
  stale_origin.a = 2.0;
  stale_origin.inva = 0.5;
  expect_tile_rejection(stale_origin, {stale_origin}, 2,
                        "tile validation must reject a stale cached physical origin");

  const meep::grid_volume large_root = meep::vol3d(384.0, 384.0, 192.0, 1.0);
  std::vector<meep::grid_volume> large_tiles;
  meep::split_into_tiles(large_root, &large_tiles, 128);
  require(large_tiles.size() == 294912,
          "384x384x192/base128 partition has an unexpected leaf count");
  meep::check_tiles(large_root, large_tiles, 128);
}

void check_fields_tile_base_lifecycle() {
  const meep::grid_volume gv = meep::vol2d(1.0, 1.0, 4.0);
  meep::structure_chunk structure_chunk(gv, gv.surroundings(), 0.5, 0);
  const int initial_refcount = structure_chunk.refcount;
  for (const int invalid_base : {-1, 1}) {
    bool rejected = false;
    try {
      meep::fields_chunk invalid(&structure_chunk, "", 0.0, 0.0, true, 0, invalid_base,
                                 {0.0, 0.0, 0.0});
    }
    catch (const std::runtime_error &) {
      rejected = true;
    }
    require(rejected, "direct fields_chunk construction must reject an invalid tile base");
    require(structure_chunk.refcount == initial_refcount,
            "rejected fields_chunk construction must not leak a structure refcount");
  }
}

} // namespace

int main() {
  check_structured_grid_indices();
  check_rank_device_mapping();
  check_linear_tile_validation();
  check_fields_tile_base_lifecycle();
  require(meep::gpu::backend_compiled() == (MEEP_HAVE_CUDA != 0),
          "public compile-time macro and runtime backend state disagree");

  std::string diagnostic;
  const bool available = meep::gpu::runtime_available(&diagnostic);

  const char *original_backend = std::getenv("MEEP_GPU_BACKEND");
  const bool had_original_backend = original_backend != nullptr;
  const std::string saved_backend = original_backend ? original_backend : "";
  setenv("MEEP_GPU_BACKEND", "invalid-test-value", 1);
  bool invalid_environment_threw = false;
  try {
    (void)meep::gpu::requested_backend();
  }
  catch (const std::invalid_argument &) {
    invalid_environment_threw = true;
  }
  if (had_original_backend)
    setenv("MEEP_GPU_BACKEND", saved_backend.c_str(), 1);
  else
    unsetenv("MEEP_GPU_BACKEND");
  require(invalid_environment_threw, "invalid MEEP_GPU_BACKEND must fail clearly");

  if (!available) {
    setenv("MEEP_GPU_BACKEND", "cuda", 1);
    for (int attempt = 0; attempt < 2; ++attempt) {
      bool unavailable_environment_threw = false;
      try {
        (void)meep::gpu::requested_backend();
      }
      catch (const std::runtime_error &) {
        unavailable_environment_threw = true;
      }
      require(unavailable_environment_threw,
              "failed environment CUDA initialization must remain failed transactionally");
    }
    if (had_original_backend)
      setenv("MEEP_GPU_BACKEND", saved_backend.c_str(), 1);
    else
      unsetenv("MEEP_GPU_BACKEND");
  }

#if MEEP_HAVE_CUDA
  require(meep::gpu::compiled_architectures() != "none",
          "CUDA build must expose compiled architectures");
  if (available) {
    const auto devices = meep::gpu::enumerate_devices();
    bool found_compatible = false;
    for (const auto &device : devices)
      found_compatible = found_compatible || device.compatible;
    require(found_compatible, "available CUDA runtime must enumerate a compatible device");
  }
  else {
    require(!diagnostic.empty(), "unavailable CUDA runtime must provide a diagnostic");
  }
#else
  require(!available, "CPU-only build cannot report an available CUDA backend");
  require(!diagnostic.empty(), "CPU-only stub must provide a diagnostic");
  require(meep::gpu::enumerate_devices().empty(), "CPU-only stub must enumerate no devices");
  require(meep::gpu::compiled_architectures() == "none",
          "CPU-only stub must report no compiled architectures");
  bool threw = false;
  try {
    meep::gpu::select_device(0);
  }
  catch (const std::runtime_error &) {
    threw = true;
  }
  require(threw, "CPU-only select_device must fail clearly");
#endif

  meep::gpu::set_backend(meep::gpu::backend_mode::cpu);
  require(meep::gpu::requested_backend() == meep::gpu::backend_mode::cpu,
          "set_backend(cpu) must update the requested backend");
  require(meep::gpu::active_backend() == meep::gpu::backend_mode::cpu,
          "set_backend(cpu) must keep CPU active");

#if MEEP_HAVE_CUDA
  const auto require_rejected_automatic_configuration =
      [](const char *name, const char *value, const char *message) {
        const char *original = std::getenv(name);
        const bool had_original = original != nullptr;
        const std::string saved = original ? original : "";
        setenv(name, value, 1);
        bool rejected = false;
        try {
          meep::gpu::set_backend(meep::gpu::backend_mode::automatic);
        }
        catch (const std::exception &) {
          rejected = true;
        }
        if (had_original)
          setenv(name, saved.c_str(), 1);
        else
          unsetenv(name);
        require(rejected, message);
        require(
            meep::gpu::requested_backend() ==
                    meep::gpu::backend_mode::cpu &&
                meep::gpu::active_backend() ==
                    meep::gpu::backend_mode::cpu,
            "failed automatic configuration must preserve the CPU transaction");
      };
  require_rejected_automatic_configuration(
      "MEEP_GPU_DEVICE", "invalid",
      "automatic mode must reject a malformed explicit CUDA device");
  require_rejected_automatic_configuration(
      "MEEP_GPU_DEVICE", "2147483647",
      "automatic mode must reject an unavailable explicit CUDA device");
  require_rejected_automatic_configuration(
      "MEEP_GPU_ALLOW_OVERSUBSCRIBE", "invalid",
      "automatic mode must reject malformed oversubscription policy");
#endif

  meep::gpu::reset_dispatch_statistics();
  const auto empty_statistics = meep::gpu::get_dispatch_statistics();
  require(empty_statistics.cpu_curl_calls == 0 && empty_statistics.cuda_curl_calls == 0,
          "reset dispatch statistics must clear curl counters");
  const auto empty_resident_statistics = meep::gpu::get_resident_statistics();
  require(empty_resident_statistics.host_to_device_bytes_avoided == 0 &&
              empty_resident_statistics.device_to_host_bytes_avoided == 0 &&
              empty_resident_statistics.device_buffer_allocations == 0 &&
              empty_resident_statistics.device_buffer_reuses == 0,
          "reset dispatch statistics must clear resident-cache counters");
  require(meep::gpu::get_live_resident_device_buffers() == 0,
          "backend API test unexpectedly retained a live resident buffer");
  const auto empty_field_update_statistics =
      meep::gpu::get_field_update_statistics();
  require(empty_field_update_statistics.cpu_update_eh_calls == 0 &&
              empty_field_update_statistics.cpu_update_eh_points == 0 &&
              empty_field_update_statistics.cuda_update_eh_calls == 0 &&
              empty_field_update_statistics.cuda_update_eh_points == 0,
          "reset dispatch statistics must clear E/H update counters");
  const auto empty_polarization_statistics =
      meep::gpu::get_polarization_statistics();
  require(empty_polarization_statistics.cpu_update_calls == 0 &&
              empty_polarization_statistics.cpu_update_points == 0 &&
              empty_polarization_statistics.cuda_update_calls == 0 &&
              empty_polarization_statistics.cuda_update_points == 0,
          "reset dispatch statistics must clear polarization counters");
  const auto empty_source_statistics = meep::gpu::get_source_statistics();
  require(empty_source_statistics.cpu_update_calls == 0 &&
              empty_source_statistics.cpu_update_points == 0 &&
              empty_source_statistics.cuda_update_calls == 0 &&
              empty_source_statistics.cuda_update_points == 0,
          "reset dispatch statistics must clear source counters");
  const auto empty_boundary_statistics =
      meep::gpu::get_boundary_statistics();
  require(empty_boundary_statistics.cpu_update_calls == 0 &&
              empty_boundary_statistics.cpu_update_points == 0 &&
              empty_boundary_statistics.cuda_update_calls == 0 &&
              empty_boundary_statistics.cuda_update_points == 0,
          "reset dispatch statistics must clear boundary counters");
  const auto empty_dft_statistics = meep::gpu::get_dft_statistics();
  require(empty_dft_statistics.cpu_update_calls == 0 &&
              empty_dft_statistics.cpu_update_points == 0 &&
              empty_dft_statistics.cuda_update_calls == 0 &&
              empty_dft_statistics.cuda_update_points == 0,
          "reset dispatch statistics must clear DFT counters");
  const auto empty_runtime_statistics =
      meep::gpu::get_runtime_touch_statistics();
  require(empty_runtime_statistics.availability_probes == 0 &&
              empty_runtime_statistics.device_enumerations == 0 &&
              empty_runtime_statistics.device_selections == 0,
          "reset dispatch statistics must clear CUDA runtime-touch counters");
  const auto empty_mpi_completion_statistics =
      meep::gpu::get_mpi_completion_statistics();
  require(empty_mpi_completion_statistics.waitsome_executions == 0 &&
              empty_mpi_completion_statistics.waitall_executions == 0,
          "reset dispatch statistics must clear MPI completion counters");

  meep::gpu::set_backend(meep::gpu::backend_mode::automatic);
  require(meep::gpu::requested_backend() == meep::gpu::backend_mode::automatic,
          "set_backend(auto) must update the requested backend");
  require(meep::gpu::active_backend() == meep::gpu::backend_mode::cpu,
          "automatic backend touched CUDA before fields workload preflight");
  const auto lazy_runtime_statistics =
      meep::gpu::get_runtime_touch_statistics();
  require(lazy_runtime_statistics.availability_probes == 0 &&
              lazy_runtime_statistics.device_enumerations == 0 &&
              lazy_runtime_statistics.device_selections == 0,
          "automatic backend request eagerly touched the CUDA runtime");
  require(!meep::gpu::backend_diagnostic().empty(),
          "resolved backend must expose a diagnostic");

  if (available) {
    meep::gpu::set_backend(meep::gpu::backend_mode::cuda);
    require(meep::gpu::active_backend() == meep::gpu::backend_mode::cuda,
            "required CUDA backend must become active");
  }
  else {
    bool threw = false;
    try {
      meep::gpu::set_backend(meep::gpu::backend_mode::cuda);
    }
    catch (const std::runtime_error &) {
      threw = true;
    }
    require(threw, "required CUDA backend must reject an unavailable runtime");
    require(meep::gpu::requested_backend() == meep::gpu::backend_mode::automatic,
            "failed backend selection must preserve the previous configuration");
  }
  meep::gpu::set_backend(meep::gpu::backend_mode::cpu);

  std::cout << "PASS: Meep GPU backend API compiled=" << meep::gpu::backend_compiled()
            << " runtime_available=" << available
            << " architectures=" << meep::gpu::compiled_architectures() << '\n';
  return 0;
}
