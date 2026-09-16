/* Fixed-workload multi-GPU correctness and strong-scaling measurement.
   This is built by `make check` but run explicitly by
   scripts/benchmark-multi-gpu.sh on physical GPUs. */

#include <meep.hpp>
#include <meep/gpu.hpp>
#include "gpu_backend_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using namespace meep;

namespace {

constexpr double workload_length = 6.0;
constexpr const char *record_prefix =
    "gpmeep-multi-benchmark-v4:";
constexpr const char *device_record_prefix =
    "gpmeep-multi-device-v1:";
constexpr const char *process_record_prefix =
    "gpmeep-multi-process-v1:";
constexpr const char *initial_condition_record_prefix =
    "gpmeep-initial-condition-v1:";
constexpr const char *source_profile_record_prefix =
    "gpmeep-source-profile-v1:";
constexpr const char *initialization_timing_record_prefix =
    "gpmeep-initialize-field-timing-v1:";
constexpr const char *dft_workload_record_prefix =
    "gpmeep-dft-workload-v1:";
constexpr const char *dft_warmup_plan_record_prefix =
    "gpmeep-dft-warmup-plan-v1:";

double vacuum(const vec &) { return 1.0; }
double overlap_permittivity(const vec &) { return 2.25; }
double overlap_permeability(const vec &) { return 1.44; }

std::complex<double> deterministic_initial_ez(const vec &point) {
  // Populate every prospective MPI cut before the untimed warmup.  The old
  // point/DFT probes were outside the source light cone for the fixed 80-step
  // workload and were therefore exactly zero on both one and two ranks.
  const double two_pi = 6.283185307179586476925286766559;
  const double value =
      0.40 + 0.06 * std::sin(two_pi * point.x() / workload_length) +
      0.05 * std::cos(two_pi * point.y() / workload_length) +
      0.04 * std::sin(two_pi * point.z() / workload_length + 0.37);
  return std::complex<double>(value, 0.0);
}

std::complex<double> affine_initial_ez(const vec &point) {
  // A cheap, strictly positive, decomposition-independent development
  // diagnostic. Binary-fraction coefficients keep this profile simple while
  // making every prospective MPI cut nonzero.
  const double value =
      0.375 + 0.0078125 * point.x() + 0.00390625 * point.y() +
      0.001953125 * point.z();
  return std::complex<double>(value, 0.0);
}

std::string parse_initial_condition(const char *value) {
  if (!value) return "trigonometric-v1";
  if (std::string(value) == "trigonometric-v1") return "trigonometric-v1";
  if (std::string(value) == "affine-v1") return "affine-v1";
  throw std::invalid_argument(
      std::string("invalid MEEP_GPU_MULTI_INITIAL_CONDITION='") + value +
      "' (expected trigonometric-v1 or affine-v1)");
}

std::string parse_source_profile(const char *value) {
  if (!value) return "single-ez-v1";
  if (std::string(value) == "single-ez-v1") return "single-ez-v1";
  if (std::string(value) == "dual-electric-v1")
    return "dual-electric-v1";
  throw std::invalid_argument(
      std::string("invalid MEEP_GPU_MULTI_SOURCE_PROFILE='") + value +
      "' (expected single-ez-v1 or dual-electric-v1)");
}

void require_source_profile_parser_contract() {
  bool empty_rejected = false;
  bool unknown_rejected = false;
  try {
    (void)parse_source_profile("");
  }
  catch (const std::invalid_argument &) { empty_rejected = true; }
  try {
    (void)parse_source_profile("not-a-profile");
  }
  catch (const std::invalid_argument &) { unknown_rejected = true; }
  if (parse_source_profile(nullptr) != "single-ez-v1" ||
      parse_source_profile("single-ez-v1") != "single-ez-v1" ||
      parse_source_profile("dual-electric-v1") != "dual-electric-v1" ||
      !empty_rejected || !unknown_rejected)
    throw std::logic_error("source-profile parser contract failed");
}

void require_initial_condition_parser_contract() {
  bool empty_rejected = false;
  bool unknown_rejected = false;
  try {
    (void)parse_initial_condition("");
  }
  catch (const std::invalid_argument &) { empty_rejected = true; }
  try {
    (void)parse_initial_condition("not-a-profile");
  }
  catch (const std::invalid_argument &) { unknown_rejected = true; }
  if (parse_initial_condition(nullptr) != "trigonometric-v1" ||
      parse_initial_condition("trigonometric-v1") != "trigonometric-v1" ||
      parse_initial_condition("affine-v1") != "affine-v1" ||
      !empty_rejected || !unknown_rejected)
    throw std::logic_error("initial-condition parser contract failed");
}

void require_initial_condition_formula_contract() {
  // These are frozen results from the pre-profile trigonometric initializer,
  // computed independently of the implementation above.  Keep them as
  // literals so this guard is not a tautological restatement of the formula.
  struct fixed_sample {
    vec point;
    double trigonometric_expected;
    double affine_expected;
  };
  const fixed_sample samples[] = {
      {vec(0.0, 0.0, 0.0), 0x1.db9c9cd571651p-2,
       0x1.8000000000000p-2},
      {vec(1.25, 2.5, 4.75), 0x1.878d7a8282effp-2,
       0x1.9d80000000000p-2},
  };
  constexpr double trigonometric_tolerance = 2.0e-15;
  for (const fixed_sample &sample : samples) {
    const std::complex<double> trigonometric =
        deterministic_initial_ez(sample.point);
    const std::complex<double> affine = affine_initial_ez(sample.point);
    if (!std::isfinite(trigonometric.real()) ||
        !std::isfinite(trigonometric.imag()) ||
        std::abs(trigonometric.real() - sample.trigonometric_expected) >
            trigonometric_tolerance ||
        trigonometric.imag() != 0.0)
      throw std::logic_error(
          "trigonometric-v1 initial-condition formula contract failed");
    // The points, coefficients, and expected values are all exact binary
    // fractions, so any difference here is a semantic change.
    if (!std::isfinite(affine.real()) || !std::isfinite(affine.imag()) ||
        affine.real() != sample.affine_expected || affine.imag() != 0.0)
      throw std::logic_error(
          "affine-v1 initial-condition formula contract failed");
  }
}

std::complex<double> squared_field_integrand(
    const std::complex<realnum> *values, const vec &, void *) {
  return std::complex<double>(std::norm(values[0]), 0.0);
}

std::complex<double> weighted_squared_field_integrand(
    const std::complex<realnum> *values, const vec &point, void *) {
  // A positive spatial moment is a decomposition-independent checksum which
  // cannot pass merely because a signed field integral cancels to zero.
  const double weight =
      1.0 + point.x() / 24.0 + point.y() / 48.0 + point.z() / 96.0;
  return std::complex<double>(weight * std::norm(values[0]), 0.0);
}

std::vector<vec> cut_probe_points(double resolution) {
  // Equal-cost binary partitions of this cubic workload cut one of these
  // quarter/half planes for two- and four-rank layouts.  Probe both sides by
  // more than one Yee pixel, while using asymmetric transverse coordinates
  // to avoid a symmetry node.
  const double offset = 1.25 / resolution;
  const double cuts[] = {
      0.25 * workload_length,
      0.50 * workload_length,
      0.75 * workload_length};
  std::vector<vec> points;
  points.reserve(18);
  for (double cut : cuts) {
    points.emplace_back(cut - offset, 2.17, 3.83);
    points.emplace_back(cut + offset, 2.17, 3.83);
    points.emplace_back(1.91, cut - offset, 4.09);
    points.emplace_back(1.91, cut + offset, 4.09);
    points.emplace_back(2.29, 3.71, cut - offset);
    points.emplace_back(2.29, 3.71, cut + offset);
  }
  return points;
}

std::string requested_transport() {
  const char *value = std::getenv("MEEP_GPU_MPI_TRANSPORT");
  if (!value || !*value || std::string(value) == "auto") return "auto";
  if (std::string(value) == "pinned" || std::string(value) == "host")
    return "pinned";
  if (std::string(value) == "cuda-aware" ||
      std::string(value) == "device")
    return "cuda-aware";
  throw std::invalid_argument(
      std::string("invalid MEEP_GPU_MPI_TRANSPORT='") + value + "'");
}

std::string requested_completion_policy() {
  const char *value = std::getenv("MEEP_GPU_MPI_COMPLETION");
  if (!value || !*value || std::string(value) == "waitsome")
    return "waitsome";
  if (std::string(value) == "waitall") return "waitall";
  throw std::invalid_argument(
      std::string("invalid MEEP_GPU_MPI_COMPLETION='") + value + "'");
}

int environment_int(const char *name, int fallback) {
  const char *value = std::getenv(name);
  if (!value || !*value) return fallback;
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (!end || *end != '\0' || parsed <= 0 ||
      parsed > std::numeric_limits<int>::max())
    throw std::invalid_argument(std::string(name) +
                                " must be a positive integer");
  return static_cast<int>(parsed);
}

int environment_nonnegative_int(const char *name, int fallback) {
  const char *value = std::getenv(name);
  if (!value || !*value) return fallback;
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (!end || *end != '\0' || parsed < 0 ||
      parsed > std::numeric_limits<int>::max())
    throw std::invalid_argument(std::string(name) +
                                " must be a non-negative integer");
  return static_cast<int>(parsed);
}

bool environment_flag(const char *name) {
  const char *value = std::getenv(name);
  if (!value || !*value || std::string(value) == "0") return false;
  if (std::string(value) == "1") return true;
  throw std::invalid_argument(std::string(name) + " must be 0 or 1");
}

struct fixed_workload_options {
  int pixels = -1;
  int warmup_steps = -1;
  int measured_steps = -1;
  int dft_monitor_count = -1;
  int dft_frequency_count = -1;
  int loop_tile_base_db = -1;
  bool bfast = false;
  bool source_enabled = false;
  bool overlap_material = false;
  bool boundary_graph_lifecycle = false;
  bool initial_condition_preflight_only = false;
  bool dft_workload_preflight_only = false;
  std::string initial_condition;
  std::string source_profile;
  std::string local_error;
};

fixed_workload_options parse_local_fixed_workload_options() {
  fixed_workload_options options;
  try {
    const bool quick = environment_flag("MEEP_GPU_MULTI_QUICK");
    options.pixels =
        environment_int("MEEP_GPU_MULTI_PIXELS", quick ? 96 : 192);
    options.warmup_steps =
        environment_int("MEEP_GPU_MULTI_WARMUP_STEPS", quick ? 6 : 12);
    options.measured_steps =
        environment_int("MEEP_GPU_MULTI_STEPS", quick ? 20 : 80);
    options.dft_monitor_count =
        environment_int("MEEP_GPU_MULTI_DFT_MONITORS", 1);
    options.dft_frequency_count =
        environment_int("MEEP_GPU_MULTI_DFT_FREQUENCIES", 2);
    if (options.dft_monitor_count > 256)
      throw std::invalid_argument(
          "MEEP_GPU_MULTI_DFT_MONITORS must not exceed 256");
    if (options.dft_frequency_count > 1024)
      throw std::invalid_argument(
          "MEEP_GPU_MULTI_DFT_FREQUENCIES must not exceed 1024");
    options.loop_tile_base_db = environment_nonnegative_int(
        "MEEP_GPU_MULTI_LOOP_TILE_BASE_DB", 128);
    options.bfast = environment_flag("MEEP_GPU_MULTI_BFAST");
    options.source_enabled =
        !environment_flag("MEEP_GPU_MULTI_DISABLE_SOURCE");
    options.overlap_material =
        environment_flag("MEEP_GPU_MULTI_OVERLAP_MATERIAL");
    options.boundary_graph_lifecycle =
        environment_flag("MEEP_GPU_TEST_BOUNDARY_GRAPH_LIFECYCLE");
    options.initial_condition_preflight_only = environment_flag(
        "MEEP_GPU_TEST_INITIAL_CONDITION_PREFLIGHT_ONLY");
    options.dft_workload_preflight_only = environment_flag(
        "MEEP_GPU_TEST_DFT_WORKLOAD_PREFLIGHT_ONLY");
    if (options.initial_condition_preflight_only &&
        options.dft_workload_preflight_only)
      throw std::invalid_argument(
          "initial-condition and DFT-workload preflight gates are mutually exclusive");

    const bool initial_condition_mismatch = environment_flag(
        "MEEP_GPU_TEST_INITIAL_CONDITION_RANK_MISMATCH");
    const bool initial_condition_invalid = environment_flag(
        "MEEP_GPU_TEST_INITIAL_CONDITION_RANK_INVALID");
    const bool source_mismatch = environment_flag(
        "MEEP_GPU_TEST_SOURCE_ENABLED_RANK_MISMATCH");
    const bool source_profile_mismatch = environment_flag(
        "MEEP_GPU_TEST_SOURCE_PROFILE_RANK_MISMATCH");
    const bool source_profile_invalid = environment_flag(
        "MEEP_GPU_TEST_SOURCE_PROFILE_RANK_INVALID");
    const bool lifecycle_mismatch = environment_flag(
        "MEEP_GPU_TEST_BOUNDARY_GRAPH_LIFECYCLE_RANK_MISMATCH");
    const bool initial_condition_preflight_mismatch = environment_flag(
        "MEEP_GPU_TEST_INITIAL_CONDITION_PREFLIGHT_RANK_MISMATCH");
    const bool dft_workload_preflight_mismatch = environment_flag(
        "MEEP_GPU_TEST_DFT_WORKLOAD_PREFLIGHT_RANK_MISMATCH");
    if ((initial_condition_mismatch || initial_condition_invalid ||
         source_mismatch || source_profile_mismatch ||
         source_profile_invalid || lifecycle_mismatch ||
         initial_condition_preflight_mismatch ||
         dft_workload_preflight_mismatch) &&
        count_processors() < 2)
      throw std::invalid_argument(
          "rank-mismatch workload preflight requires at least two MPI ranks");
    if (initial_condition_mismatch && initial_condition_invalid)
      throw std::invalid_argument(
          "initial-condition mismatch and invalid-rank gates are mutually exclusive");
    if (source_profile_mismatch && source_profile_invalid)
      throw std::invalid_argument(
          "source-profile mismatch and invalid-rank gates are mutually exclusive");

    if (source_mismatch)
      options.source_enabled = my_global_rank() % 2 == 0;
    if (lifecycle_mismatch)
      options.boundary_graph_lifecycle = my_global_rank() % 2 != 0;
    if (initial_condition_preflight_mismatch)
      options.initial_condition_preflight_only =
          my_global_rank() % 2 == 0;
    if (dft_workload_preflight_mismatch)
      options.dft_workload_preflight_only =
          my_global_rank() % 2 == 0;

    require_initial_condition_parser_contract();
    require_initial_condition_formula_contract();
    require_source_profile_parser_contract();
    std::string injected_initial_condition;
    const char *initial_condition_value =
        std::getenv("MEEP_GPU_MULTI_INITIAL_CONDITION");
    if (initial_condition_mismatch) {
      injected_initial_condition =
          my_global_rank() % 2 == 0 ? "trigonometric-v1" : "affine-v1";
      initial_condition_value = injected_initial_condition.c_str();
    }
    else if (initial_condition_invalid && my_global_rank() % 2 != 0) {
      injected_initial_condition = "invalid-rank-profile";
      initial_condition_value = injected_initial_condition.c_str();
    }
    options.initial_condition =
        parse_initial_condition(initial_condition_value);
    std::string injected_source_profile;
    const char *source_profile_value =
        std::getenv("MEEP_GPU_MULTI_SOURCE_PROFILE");
    if (source_profile_mismatch) {
      injected_source_profile =
          my_global_rank() % 2 == 0 ? "single-ez-v1" : "dual-electric-v1";
      source_profile_value = injected_source_profile.c_str();
    }
    else if (source_profile_invalid && my_global_rank() % 2 != 0) {
      injected_source_profile = "invalid-rank-profile";
      source_profile_value = injected_source_profile.c_str();
    }
    options.source_profile = parse_source_profile(source_profile_value);
  }
  catch (const std::exception &error) { options.local_error = error.what(); }
  return options;
}

bool distributed_value_is_unanimous(int value) {
  const int minimum = min_to_all(value);
  const int maximum = max_to_all(value);
  return minimum == maximum;
}

void require_distributed_fixed_workload(
    const fixed_workload_options &options) {
  // Every rank participates in this exact collective sequence even when its
  // local parser rejected a value. No rank can advance to allocation while a
  // peer exits early or selected a different physical workload.
  const bool all_valid = and_to_all(options.local_error.empty());
  const int initial_condition_code =
      options.initial_condition == "trigonometric-v1"
          ? 1
          : (options.initial_condition == "affine-v1" ? 2 : -1);
  const int source_profile_code =
      options.source_profile == "single-ez-v1"
          ? 1
          : (options.source_profile == "dual-electric-v1" ? 2 : -1);
  const bool same_pixels =
      distributed_value_is_unanimous(options.pixels);
  const bool same_warmup_steps =
      distributed_value_is_unanimous(options.warmup_steps);
  const bool same_measured_steps =
      distributed_value_is_unanimous(options.measured_steps);
  const bool same_dft_monitor_count =
      distributed_value_is_unanimous(options.dft_monitor_count);
  const bool same_dft_frequency_count =
      distributed_value_is_unanimous(options.dft_frequency_count);
  const bool same_loop_tile_base_db =
      distributed_value_is_unanimous(options.loop_tile_base_db);
  const bool same_bfast =
      distributed_value_is_unanimous(options.bfast ? 1 : 0);
  const bool same_source =
      distributed_value_is_unanimous(options.source_enabled ? 1 : 0);
  const bool same_overlap_material =
      distributed_value_is_unanimous(options.overlap_material ? 1 : 0);
  const bool same_lifecycle = distributed_value_is_unanimous(
      options.boundary_graph_lifecycle ? 1 : 0);
  const bool same_initial_condition_preflight =
      distributed_value_is_unanimous(
          options.initial_condition_preflight_only ? 1 : 0);
  const bool same_dft_workload_preflight =
      distributed_value_is_unanimous(
          options.dft_workload_preflight_only ? 1 : 0);
  const bool same_initial_condition =
      distributed_value_is_unanimous(initial_condition_code);
  const bool same_source_profile =
      distributed_value_is_unanimous(source_profile_code);
  if (!all_valid) {
    if (!options.local_error.empty())
      throw std::runtime_error(options.local_error);
    throw std::runtime_error(
        "another MPI rank rejected the fixed workload before allocation");
  }
  if (!(same_pixels && same_warmup_steps && same_measured_steps &&
        same_dft_monitor_count && same_dft_frequency_count &&
        same_loop_tile_base_db && same_bfast && same_source &&
        same_overlap_material && same_lifecycle &&
        same_initial_condition_preflight && same_dft_workload_preflight &&
        same_initial_condition && same_source_profile))
    throw std::runtime_error(
        "fixed workload profile differs across MPI ranks before allocation");
}

std::string json_string(const std::string &value) {
  std::ostringstream escaped;
  escaped << '"';
  for (unsigned char character : value) {
    switch (character) {
      case '"': escaped << "\\\""; break;
      case '\\': escaped << "\\\\"; break;
      case '\b': escaped << "\\b"; break;
      case '\f': escaped << "\\f"; break;
      case '\n': escaped << "\\n"; break;
      case '\r': escaped << "\\r"; break;
      case '\t': escaped << "\\t"; break;
      default:
        if (character < 0x20) {
          escaped << "\\u00" << std::hex << std::setw(2)
                  << std::setfill('0') << static_cast<int>(character)
                  << std::dec << std::setfill(' ');
        }
        else
          escaped << static_cast<char>(character);
    }
  }
  escaped << '"';
  return escaped.str();
}

std::uint64_t global_sum(std::uint64_t value) {
  if (value > static_cast<std::uint64_t>(
                  std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error("GPU statistic does not fit size_t");
  return static_cast<std::uint64_t>(
      sum_to_all(static_cast<std::size_t>(value)));
}

std::uint64_t global_max(std::uint64_t value) {
  if (value > static_cast<std::uint64_t>(
                  std::numeric_limits<int>::max()))
    throw std::overflow_error("GPU statistic does not fit int");
  return static_cast<std::uint64_t>(
      max_to_all(static_cast<int>(value)));
}

struct throwing_copy_callback {
  bool *throw_on_copy;
  int *invocation_count;

  throwing_copy_callback(bool *should_throw, int *count)
      : throw_on_copy(should_throw), invocation_count(count) {}
  throwing_copy_callback(const throwing_copy_callback &other)
      : throw_on_copy(other.throw_on_copy),
        invocation_count(other.invocation_count) {
    if (*throw_on_copy)
      throw std::runtime_error("injected callback copy failure");
  }
  void operator()() const { ++*invocation_count; }
};

void require_boundary_receive_slot_registry() {
  int owner_token = 0;
  int exchange_token = 0;
  gpu::detail::boundary_exchange_buffer *primary =
      gpu::detail::resident_boundary_exchange_buffer_slot(
          &owner_token, &exchange_token, 8, 0);
  gpu::detail::boundary_exchange_buffer *primary_reuse =
      gpu::detail::resident_boundary_exchange_buffer(
          &owner_token, &exchange_token, 8);
  gpu::detail::boundary_exchange_buffer *secondary =
      gpu::detail::resident_boundary_exchange_buffer_slot(
          &owner_token, &exchange_token, 8, 1);
  gpu::detail::boundary_exchange_buffer *secondary_reuse =
      gpu::detail::resident_boundary_exchange_buffer_slot(
          &owner_token, &exchange_token, 8, 1);
  if (!primary || !secondary || primary != primary_reuse ||
      secondary != secondary_reuse || primary == secondary ||
      gpu::detail::boundary_exchange_device_data(primary) ==
          gpu::detail::boundary_exchange_device_data(secondary))
    throw std::runtime_error(
        "boundary receive slot registry did not provide two stable distinct buffers");
  gpu::detail::destroy_boundary_exchange_for_owner(&owner_token);
}

[[noreturn]] void trigger_comms_manager_destructor_abort() {
  if (count_processors() != 2)
    throw std::runtime_error(
        "communications-manager destructor regression requires two ranks");
  const int rank = my_global_rank();
  const int peer = 1 - rank;
  std::unique_ptr<comms_manager> manager = create_comms_manager();
  realnum receive_value = 0;
  const realnum send_value = static_cast<realnum>(80 + rank);
  manager->receive_real_async(
      &receive_value, 1, peer, 29998, []() {});
  manager->send_real_async(&send_value, 1, peer, 29998);
  if (comms_supports_cuda_device_buffers(manager.get()))
    comms_start_cuda_device_receives(manager.get());
  throw std::runtime_error(
      "injected unwind with pending communications");
}

void require_comms_manager_exception_recovery() {
  if (count_processors() != 2)
    throw std::runtime_error(
        "communications-manager recovery regression requires two ranks");
  const int rank = my_global_rank();
  const int peer = 1 - rank;
  std::unique_ptr<comms_manager> manager = create_comms_manager();
  const bool cuda_aware =
      comms_supports_cuda_device_buffers(manager.get());
  const char *completion = std::getenv("MEEP_GPU_MPI_COMPLETION");
  const bool expect_waitall = completion &&
                              std::string(completion) == "waitall";
  if (comms_uses_waitall_completion(manager.get()) != expect_waitall)
    throw std::runtime_error(
        "communications manager selected the wrong completion policy");

  realnum enqueue_copy_receive = 0;
  bool throw_on_enqueue_copy = false;
  int enqueue_copy_callbacks = 0;
  comms_manager::receive_callback enqueue_copy_callback =
      throwing_copy_callback(
          &throw_on_enqueue_copy, &enqueue_copy_callbacks);
  throw_on_enqueue_copy = true;
  bool enqueue_copy_threw = false;
  try {
    manager->receive_real_async(
        &enqueue_copy_receive, 1, peer, 29999,
        enqueue_copy_callback);
  }
  catch (const std::runtime_error &error) {
    enqueue_copy_threw =
        std::string(error.what()) == "injected callback copy failure";
  }
  if (!and_to_all(enqueue_copy_threw) || enqueue_copy_callbacks != 0 ||
      comms_physical_message_count(manager.get()) != 0)
    throw std::runtime_error(
        "communications manager retained state after enqueue callback-copy failure");

  // A std::function target may throw while finish() copies the logical
  // receive callbacks into its physical request plan.  No MPI operation may
  // be posted and no partial callback/request state may remain in that case.
  realnum receive_plan_recovery = 0;
  const realnum send_plan_recovery = static_cast<realnum>(90 + rank);
  bool throw_on_callback_copy = false;
  int plan_recovery_callbacks = 0;
  comms_manager::receive_callback planning_callback = throwing_copy_callback(
      &throw_on_callback_copy, &plan_recovery_callbacks);
  manager->receive_real_async(
      &receive_plan_recovery, 1, peer, 30000, planning_callback);
  manager->send_real_async(&send_plan_recovery, 1, peer, 30000);
  if (comms_physical_message_count(manager.get()) != 2)
    throw std::runtime_error(
        "communications manager miscounted planning-failure messages");
  throw_on_callback_copy = true;
  bool planning_copy_threw = false;
  try {
    if (cuda_aware)
      comms_start_cuda_device_receives(manager.get());
    else
      comms_finish(manager.get());
  }
  catch (const std::runtime_error &error) {
    planning_copy_threw =
        std::string(error.what()) == "injected callback copy failure";
  }
  if (!and_to_all(planning_copy_threw) || plan_recovery_callbacks != 0 ||
      comms_physical_message_count(manager.get()) != 2)
    throw std::runtime_error(
        "communications manager poisoned state after planning failure");
  throw_on_callback_copy = false;
  if (cuda_aware) {
    comms_start_cuda_device_receives(manager.get());
    comms_start_cuda_device_sends(manager.get());
  }
  comms_finish(manager.get());
  if (plan_recovery_callbacks != 1 ||
      receive_plan_recovery != static_cast<realnum>(90 + peer) ||
      comms_physical_message_count(manager.get()) != 0)
    throw std::runtime_error(
        "communications manager did not recover from planning failure");

  realnum receive_first[2] = {0, 0};
  realnum receive_second[3] = {0, 0, 0};
  const realnum send_first[2] = {
      static_cast<realnum>(100 + rank),
      static_cast<realnum>(110 + rank)};
  const realnum send_second[3] = {
      static_cast<realnum>(120 + rank),
      static_cast<realnum>(130 + rank),
      static_cast<realnum>(140 + rank)};
  const realnum *const send_first_pointer = send_first;
  int callbacks_completed = 0;
  comms_manager *const manager_pointer = manager.get();
  manager->receive_real_async(
      receive_first, 2, peer, 30001,
      [manager_pointer, send_first_pointer, peer, &callbacks_completed]() {
        ++callbacks_completed;
        // Enqueueing from a completion callback used to reallocate the
        // pinned pending-transfer vector (invalidating aggregate callback
        // pointers) or append an uncounted CUDA-aware request.
        manager_pointer->send_real_async(
            send_first_pointer, 1, peer, 30004);
      });
  manager->receive_real_async(
      receive_second, 3, peer, 30002,
      [&callbacks_completed]() { ++callbacks_completed; });
  manager->send_real_async(send_first, 2, peer, 30001);
  manager->send_real_async(send_second, 3, peer, 30002);
  const std::size_t first_expected_messages = cuda_aware ? 4u : 2u;
  if (comms_physical_message_count(manager.get()) !=
      first_expected_messages)
    throw std::runtime_error(
        "communications-manager physical message count is incorrect");

  bool callback_threw = false;
  try {
    comms_finish(manager.get());
  }
  catch (const std::logic_error &error) {
    callback_threw =
        std::string(error.what()).find(
            "cannot enqueue while communications manager is finishing") !=
        std::string::npos;
  }
  if (!and_to_all(callback_threw) || callbacks_completed != 2)
    throw std::runtime_error(
        "communications manager did not drain every callback after failure");
  if (comms_physical_message_count(manager.get()) != 0)
    throw std::runtime_error(
        "communications manager retained messages after callback failure");
  // The failed finish must have released requests/datatypes and be
  // idempotent before the same allocation is used for another exchange.
  comms_finish(manager.get());

  realnum receive_reentrant = 0;
  const realnum send_reentrant = static_cast<realnum>(150 + rank);
  bool reentrant_callback = false;
  manager->receive_real_async(
      &receive_reentrant, 1, peer, 30005,
      [manager_pointer, &reentrant_callback]() {
        reentrant_callback = true;
        // Reentrant finish would otherwise mutate the request vectors being
        // traversed by the outer MPI_Waitsome loop.
        comms_finish(manager_pointer);
      });
  manager->send_real_async(&send_reentrant, 1, peer, 30005);
  if (comms_physical_message_count(manager.get()) != 2)
    throw std::runtime_error(
        "communications manager miscounted reentrant-test messages");
  bool reentrant_finish_threw = false;
  try {
    comms_finish(manager.get());
  }
  catch (const std::logic_error &error) {
    reentrant_finish_threw =
        std::string(error.what()).find("finish is not reentrant") !=
        std::string::npos;
  }
  if (!and_to_all(reentrant_finish_threw) || !reentrant_callback ||
      receive_reentrant != static_cast<realnum>(150 + peer))
    throw std::runtime_error(
        "communications manager accepted reentrant finish");
  if (comms_physical_message_count(manager.get()) != 0)
    throw std::runtime_error(
        "communications manager retained messages after reentrant finish");
  comms_finish(manager.get());

  realnum receive_throwing_body = 0;
  realnum receive_after_throw = 0;
  const realnum send_throwing_body = static_cast<realnum>(170 + rank);
  const realnum send_after_throw = static_cast<realnum>(180 + rank);
  int callback_bodies_completed = 0;
  manager->receive_real_async(
      &receive_throwing_body, 1, peer, 30008,
      [&callback_bodies_completed]() {
        ++callback_bodies_completed;
        throw std::runtime_error("injected callback body failure");
      });
  manager->receive_real_async(
      &receive_after_throw, 1, peer, 30009,
      [&callback_bodies_completed]() { ++callback_bodies_completed; });
  manager->send_real_async(&send_throwing_body, 1, peer, 30008);
  manager->send_real_async(&send_after_throw, 1, peer, 30009);
  bool callback_body_threw = false;
  try {
    comms_finish(manager.get());
  }
  catch (const std::runtime_error &error) {
    callback_body_threw =
        std::string(error.what()) == "injected callback body failure";
  }
  if (!and_to_all(callback_body_threw) ||
      callback_bodies_completed != 2 ||
      receive_throwing_body != static_cast<realnum>(170 + peer) ||
      receive_after_throw != static_cast<realnum>(180 + peer) ||
      comms_physical_message_count(manager.get()) != 0)
    throw std::runtime_error(
        "communications manager did not drain after a callback body failure");
  comms_finish(manager.get());

  realnum receive_reuse[2] = {0, 0};
  const realnum send_reuse[2] = {
      static_cast<realnum>(200 + rank),
      static_cast<realnum>(210 + rank)};
  bool reuse_callback = false;
  manager->receive_real_async(
      receive_reuse, 2, peer, 30003,
      [&reuse_callback]() { reuse_callback = true; });
  manager->send_real_async(send_reuse, 2, peer, 30003);
  if (comms_physical_message_count(manager.get()) != 2)
    throw std::runtime_error(
        "communications manager miscounted reuse-test messages");
  comms_finish(manager.get());
  comms_finish(manager.get());
  if (!reuse_callback ||
      receive_reuse[0] != static_cast<realnum>(200 + peer) ||
      receive_reuse[1] != static_cast<realnum>(210 + peer))
    throw std::runtime_error(
        "communications manager was not reusable after callback failure");
  if (comms_physical_message_count(manager.get()) != 0)
    throw std::runtime_error(
        "communications manager retained messages after successful reuse");

  if (cuda_aware) {
    realnum eager_receive = 0;
    const realnum eager_send = static_cast<realnum>(220 + rank);
    bool eager_callback = false;
    manager->receive_real_async(
        &eager_receive, 1, peer, 30006,
        [&eager_callback]() { eager_callback = true; });
    manager->send_real_async(&eager_send, 1, peer, 30006);

    bool send_before_receive_rejected = false;
    try {
      comms_start_cuda_device_sends(manager.get());
    }
    catch (const std::logic_error &error) {
      send_before_receive_rejected =
          std::string(error.what()).find(
              "receives must be posted before sends") !=
          std::string::npos;
    }
    if (!and_to_all(send_before_receive_rejected))
      throw std::runtime_error(
          "communications manager accepted an eager send before receive posting");

    comms_start_cuda_device_receives(manager.get());
    if (eager_callback)
      throw std::runtime_error(
          "eager receive invoked its callback before comms_finish");

    bool enqueue_after_post_rejected = false;
    try {
      manager->send_real_async(&eager_send, 1, peer, 30007);
    }
    catch (const std::logic_error &error) {
      enqueue_after_post_rejected =
          std::string(error.what()).find(
              "cannot enqueue after CUDA-aware request posting") !=
          std::string::npos;
    }
    bool duplicate_receive_rejected = false;
    try {
      comms_start_cuda_device_receives(manager.get());
    }
    catch (const std::logic_error &error) {
      duplicate_receive_rejected =
          std::string(error.what()).find("already posted") !=
          std::string::npos;
    }
    if (!and_to_all(enqueue_after_post_rejected &&
                    duplicate_receive_rejected))
      throw std::runtime_error(
          "communications manager accepted mutation of an active eager plan");

    comms_start_cuda_device_sends(manager.get());
    bool duplicate_send_rejected = false;
    try {
      comms_start_cuda_device_sends(manager.get());
    }
    catch (const std::logic_error &error) {
      duplicate_send_rejected =
          std::string(error.what()).find("already posted") !=
          std::string::npos;
    }
    if (!and_to_all(duplicate_send_rejected))
      throw std::runtime_error(
          "communications manager accepted duplicate eager send posting");

    comms_finish(manager.get());
    comms_finish(manager.get());
    if (!eager_callback ||
        eager_receive != static_cast<realnum>(220 + peer) ||
        comms_physical_message_count(manager.get()) != 0)
      throw std::runtime_error(
          "eager communications manager did not complete and reset cleanly");
  }
  all_wait();
}

} // namespace

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  try {
    const bool completion_policy_mismatch = environment_flag(
        "MEEP_GPU_TEST_COMPLETION_POLICY_MISMATCH");
    if (completion_policy_mismatch)
      setenv("MEEP_GPU_MPI_COMPLETION",
             my_global_rank() % 2 ? "waitall" : "waitsome", 1);
    if (completion_policy_mismatch ||
        environment_flag(
            "MEEP_GPU_TEST_COMPLETION_POLICY_PREFLIGHT_ONLY")) {
      validate_distributed_mpi_transport();
      throw std::runtime_error(
          "completion-policy preflight accepted an invalid request");
    }
    if (std::getenv("MEEP_GPU_MULTI_QUERY_DEVICE_COUNT")) {
      const std::vector<gpu::device_info> devices =
          gpu::enumerate_devices();
      const std::size_t compatible_devices =
          static_cast<std::size_t>(std::count_if(
              devices.begin(), devices.end(),
              [](const gpu::device_info &device) {
                return device.compatible;
              }));
      if (am_master())
        std::cout << "compatible_cuda_devices="
                  << compatible_devices << '\n';
      return compatible_devices ? 0 : 77;
    }

    // Parse and collectively agree on every core workload-identity value
    // before allocating a grid, structure, fields, or profile-dependent
    // vectors. A rank-local typo or mismatch must fail as one distributed
    // preflight instead of stranding peers in later collectives.
    const fixed_workload_options workload =
        parse_local_fixed_workload_options();
    require_distributed_fixed_workload(workload);
    const int pixels = workload.pixels;
    const int warmup_steps = workload.warmup_steps;
    const int measured_steps = workload.measured_steps;
    const int dft_monitor_count = workload.dft_monitor_count;
    const int dft_frequency_count = workload.dft_frequency_count;
    const int loop_tile_base_db = workload.loop_tile_base_db;
    const bool bfast = workload.bfast;
    const bool source_enabled = workload.source_enabled;
    const bool overlap_material = workload.overlap_material;
    const bool boundary_graph_lifecycle =
        workload.boundary_graph_lifecycle;
    const std::string initial_condition = workload.initial_condition;
    const std::string source_profile = workload.source_profile;
    std::complex<double> (*const initial_ez)(const vec &) =
        initial_condition == "affine-v1" ? affine_initial_ez
                                          : deterministic_initial_ez;
    if (workload.dft_workload_preflight_only) {
      // Allocation-free contract lane used when the CUDA driver is absent.
      // It proves that every rank parsed and agreed on the exact DFT workload
      // before a production run can allocate fields or enter CUDA.
      if (am_master())
        std::cout << dft_workload_record_prefix
                  << "{\"components\":2,\"frequencies\":"
                  << dft_frequency_count << ",\"monitors\":"
                  << dft_monitor_count << "}\n";
      return 0;
    }
    if (workload.initial_condition_preflight_only) {
      // This test-only, allocation-free lane emits the selected profile but
      // intentionally has no benchmark or initialize-field timing record.
      if (am_master())
        std::cout << source_profile_record_prefix
                  << "{\"profile\":" << json_string(source_profile)
                  << "}\n"
                  << initial_condition_record_prefix
                  << "{\"profile\":" << json_string(initial_condition)
                  << "}\n";
      return 0;
    }
    // Leave CUDA configuration lazy so fields::step can gather rank-local
    // configuration failures before any rank enters a boundary collective.
    setenv("MEEP_GPU_BACKEND", "cuda", 1);

    const double length = workload_length;
    const double resolution = static_cast<double>(pixels) / length;
    const grid_volume gv =
        vol3d(length, length, length, resolution);
    structure s(gv, vacuum, pml(0.6));
    if (overlap_material) {
      // Force nontrivial diagonal constitutive work for both E and H on
      // every rank. The default vacuum workload intentionally remains
      // unchanged for historical strong-scaling comparisons.
      s.set_epsilon(overlap_permittivity, false);
      s.set_mu(overlap_permeability, false);
    }
    const std::vector<double> bfast_scaled_k =
        bfast ? std::vector<double>{0.13, -0.07, 0.05}
              : std::vector<double>{0.0, 0.0, 0.0};
    fields f(&s, 0.0, 0.0, true, loop_tile_base_db, 128,
             bfast_scaled_k);
    f.use_real_fields();
    int initialization_applications = 0;
    double initialization_seconds = 0.0;
    const auto initialize_ez = [&]() {
      const auto initialization_start = std::chrono::steady_clock::now();
      f.initialize_field(Ez, initial_ez);
      const auto initialization_stop = std::chrono::steady_clock::now();
      initialization_seconds +=
          std::chrono::duration<double>(initialization_stop -
                                        initialization_start)
              .count();
      ++initialization_applications;
    };

    continuous_src_time source(0.24);
    const volume source_volume(
        vec(1.1, 0.8, 0.8),
        vec(1.1, length - 0.8, length - 0.8));
    if (source_enabled) {
      f.add_volume_source(
          Ez, source, source_volume, 1.0);
      if (source_profile == "dual-electric-v1")
        f.add_volume_source(Ey, source, source_volume, 1.0);
    }
    else
      // DFT registration requires field allocation. initialize_field
      // allocates the same coupled Yee components without introducing an
      // integrated source that would intentionally reject E/H overlap.
      initialize_ez();
    component components[] = {Ez, Hy};
    std::vector<dft_fields> monitors;
    monitors.reserve(static_cast<std::size_t>(dft_monitor_count));
    const volume monitor_volume(
        vec(length - 1.1, 0.8, 0.8),
        vec(length - 1.1, length - 0.8, length - 0.8));
    for (int monitor_index = 0; monitor_index < dft_monitor_count;
         ++monitor_index)
      monitors.push_back(f.add_dft_fields(
          components, 2, monitor_volume, 0.20, 0.28,
          dft_frequency_count));
    if (environment_flag("MEEP_GPU_EMIT_DFT_WORKLOAD") && am_master())
      std::cout << dft_workload_record_prefix
                << "{\"components\":2,\"frequencies\":"
                << dft_frequency_count << ",\"monitors\":"
                << dft_monitor_count << "}\n";

    // Initialization is deliberately outside the timed region.  It makes
    // every partition boundary, cut-adjacent point, and DFT plane carry a
    // deterministic nonzero signal without changing the fixed step count.
    if (source_enabled)
      initialize_ez();

    for (int step = 0; step < warmup_steps; ++step)
      f.step();
    all_wait();
    if (boundary_graph_lifecycle) {
      // A warm phase graph used to retain each exchange buffer's CUDA event.
      // zero_fields invalidates the chunk caches and exchange buffers before
      // the fields-owned topology shell is replaced, so this sequence is a
      // regression for graph/event lifetime and topology rebuilding.
      f.zero_fields();
      initialize_ez();
      for (int step = 0; step < warmup_steps; ++step)
        f.step();
      all_wait();
    }
    const int expected_initialization_applications =
        boundary_graph_lifecycle ? 2 : 1;
    if (!and_to_all(initialization_applications ==
                    expected_initialization_applications))
      throw std::runtime_error(
          "initial-condition application count differs from the fixed workload");
    const double max_initialization_seconds =
        max_to_all(initialization_seconds);
    if (!(max_initialization_seconds > 0.0) ||
        !std::isfinite(max_initialization_seconds))
      throw std::runtime_error(
          "initial-condition timing is not finite and positive");
    if (std::getenv("MEEP_GPU_TEST_COMMS_MANAGER_DESTRUCTOR_ABORT_ONLY"))
      trigger_comms_manager_destructor_abort();
    if (std::getenv("MEEP_GPU_TEST_COMMS_MANAGER_FAILURE_ONLY")) {
      require_boundary_receive_slot_registry();
      require_comms_manager_exception_recovery();
      if (am_master())
        std::cout
            << "PASS: MPI communications manager drains callback failures, "
               "double-finish is idempotent, and reuse succeeds\n";
      gpu::set_backend(gpu::backend_mode::cpu);
      return 0;
    }
    const gpu::detail::boundary_receive_pingpong_statistics
        warmup_receive_pingpong =
            gpu::detail::get_boundary_receive_pingpong_statistics();
    const gpu::detail::dft_batch_statistics warmup_dft_batches =
        gpu::detail::get_dft_batch_statistics();
    const gpu::boundary_eh_overlap_statistics warmup_boundary_eh_overlap =
        gpu::get_boundary_eh_overlap_statistics();
    const gpu::halo_curl_overlap_statistics warmup_halo_curl_overlap =
        gpu::get_halo_curl_overlap_statistics();
    const std::uint64_t warmup_boundary_eh_overlap_cold_topology =
        global_sum(warmup_boundary_eh_overlap.skipped_cold_topology);
    const std::uint64_t warmup_boundary_eh_overlap_launched_h =
        global_sum(warmup_boundary_eh_overlap.launched_h);
    const std::uint64_t warmup_boundary_eh_overlap_launched_e =
        global_sum(warmup_boundary_eh_overlap.launched_e);
    const std::uint64_t warmup_halo_curl_overlap_cold_topology =
        global_sum(warmup_halo_curl_overlap.skipped_cold_topology);
    const std::uint64_t warmup_halo_curl_overlap_launches =
        global_sum(warmup_halo_curl_overlap.launches);
    if (environment_flag("MEEP_GPU_EXPECT_RECEIVE_PINGPONG") &&
        !and_to_all(warmup_receive_pingpong.secondary_allocations > 0))
      throw std::runtime_error(
          "not every MPI rank lazily allocated a secondary CUDA receive buffer");
    if (environment_flag(
            "MEEP_GPU_EXPECT_NO_RECEIVE_SECONDARY_ALLOCATION") &&
        !and_to_all(warmup_receive_pingpong.secondary_allocations == 0))
      throw std::runtime_error(
          "a fallback MPI transport allocated an unused secondary CUDA receive buffer");
    if (environment_flag("MEEP_GPU_EXPECT_BOUNDARY_EH_OVERLAP_COLD_WARMUP") &&
        warmup_boundary_eh_overlap_cold_topology == 0) {
      std::ostringstream error;
      error << "warmup did not prove cold-topology fallback"
            << " (cold=" << warmup_boundary_eh_overlap_cold_topology
            << ", H=" << warmup_boundary_eh_overlap_launched_h
            << ", E=" << warmup_boundary_eh_overlap_launched_e << ')';
      throw std::runtime_error(error.str());
    }
    gpu::reset_dispatch_statistics();
    reset_comms_overlap_statistics();
    all_wait();
    // Keep detailed field timers scoped to the same measured interval as the
    // authoritative steady-clock result.  Without this reset, print_times()
    // mixed setup/topology construction and warmup into an 80-step report.
    f.reset_timers();

    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < measured_steps; ++step)
      f.step();
    all_wait();
    const auto stop = std::chrono::steady_clock::now();
    const double local_seconds =
        std::chrono::duration<double>(stop - start).count();
    const double seconds = max_to_all(local_seconds);
    if (std::getenv("MEEP_GPU_MULTI_PRINT_TIMES"))
      f.print_times();

    const gpu::dispatch_statistics local_dispatch =
        gpu::get_dispatch_statistics();
    const gpu::field_update_statistics local_fields =
        gpu::get_field_update_statistics();
    const gpu::polarization_statistics local_polarizations =
        gpu::get_polarization_statistics();
    const gpu::source_statistics local_sources =
        gpu::get_source_statistics();
    const gpu::boundary_statistics local_boundaries =
        gpu::get_boundary_statistics();
    const gpu::dft_statistics local_dfts =
        gpu::get_dft_statistics();
    const gpu::detail::dft_batch_statistics local_dft_batches =
        gpu::detail::get_dft_batch_statistics();
    const gpu::multi_gpu_statistics local_multi =
        gpu::get_multi_gpu_statistics();
    const gpu::detail::boundary_phase_graph_statistics local_phase_graph =
        gpu::detail::get_boundary_phase_graph_statistics();
    const gpu::detail::boundary_receive_pingpong_statistics
        local_receive_pingpong =
            gpu::detail::get_boundary_receive_pingpong_statistics();
    const gpu::boundary_eh_overlap_statistics
        local_boundary_eh_overlap =
            gpu::get_boundary_eh_overlap_statistics();
    const gpu::halo_curl_overlap_statistics local_halo_curl_overlap =
        gpu::get_halo_curl_overlap_statistics();
    const gpu::tile_coalescing_statistics local_tile_coalescing =
        gpu::get_tile_coalescing_statistics();
    const gpu::phase_batch_policy_statistics local_phase_batch_policy =
        gpu::get_phase_batch_policy_statistics();
    const gpu::curl_phase_replay_statistics local_curl_phase_replay =
        gpu::get_curl_phase_replay_statistics();
    const gpu::boundary_descriptor_replay_statistics
        local_boundary_descriptor_replay =
            gpu::get_boundary_descriptor_replay_statistics();
    const bool boundary_descriptor_fast_replay_disabled =
        std::getenv(
            "MEEP_GPU_DISABLE_BOUNDARY_DESCRIPTOR_FAST_REPLAY") != nullptr;
    const std::uint64_t boundary_descriptor_fast_replay_disabled_ranks =
        global_sum(
            boundary_descriptor_fast_replay_disabled ? 1u : 0u);
    const bool boundary_descriptor_fast_replay_disabled_everywhere =
        boundary_descriptor_fast_replay_disabled_ranks ==
        static_cast<std::uint64_t>(count_processors());
    const bool boundary_descriptor_fast_replay_enabled_everywhere =
        boundary_descriptor_fast_replay_disabled_ranks == 0;
    if (!boundary_descriptor_fast_replay_disabled_everywhere &&
        !boundary_descriptor_fast_replay_enabled_everywhere)
      throw std::runtime_error(
          "boundary descriptor replay opt-out differs across MPI ranks");
    const bool expect_boundary_descriptor_fast_replay =
        environment_flag(
            "MEEP_GPU_EXPECT_BOUNDARY_DESCRIPTOR_FAST_REPLAY");
    const bool expect_boundary_gather_descriptor_fast_replay =
        environment_flag(
            "MEEP_GPU_EXPECT_BOUNDARY_GATHER_DESCRIPTOR_FAST_REPLAY");
    const bool expect_no_boundary_descriptor_fast_replay =
        environment_flag(
            "MEEP_GPU_EXPECT_NO_BOUNDARY_DESCRIPTOR_FAST_REPLAY");
    const std::uint64_t expect_boundary_descriptor_fast_replay_ranks =
        global_sum(expect_boundary_descriptor_fast_replay ? 1u : 0u);
    const std::uint64_t
        expect_boundary_gather_descriptor_fast_replay_ranks =
            global_sum(
                expect_boundary_gather_descriptor_fast_replay ? 1u : 0u);
    const std::uint64_t expect_no_boundary_descriptor_fast_replay_ranks =
        global_sum(expect_no_boundary_descriptor_fast_replay ? 1u : 0u);
    const std::uint64_t world_size =
        static_cast<std::uint64_t>(count_processors());
    const auto expectation_is_consistent =
        [world_size](std::uint64_t enabled_ranks) {
          return enabled_ranks == 0 || enabled_ranks == world_size;
        };
    if (!expectation_is_consistent(
            expect_boundary_descriptor_fast_replay_ranks) ||
        !expectation_is_consistent(
            expect_boundary_gather_descriptor_fast_replay_ranks) ||
        !expectation_is_consistent(
            expect_no_boundary_descriptor_fast_replay_ranks))
      throw std::runtime_error(
          "boundary descriptor replay expectation differs across MPI ranks");
    const bool expect_boundary_descriptor_fast_replay_everywhere =
        expect_boundary_descriptor_fast_replay_ranks == world_size;
    const bool expect_boundary_gather_descriptor_fast_replay_everywhere =
        expect_boundary_gather_descriptor_fast_replay_ranks == world_size;
    const bool expect_no_boundary_descriptor_fast_replay_everywhere =
        expect_no_boundary_descriptor_fast_replay_ranks == world_size;
    if (expect_no_boundary_descriptor_fast_replay_everywhere &&
        (expect_boundary_descriptor_fast_replay_everywhere ||
         expect_boundary_gather_descriptor_fast_replay_everywhere))
      throw std::runtime_error(
          "conflicting boundary descriptor replay expectations");
    const comms_overlap_statistics local_overlap =
        get_comms_overlap_statistics();
    if (!and_to_all(local_dispatch.cuda_curl_calls > 0 &&
                    local_dispatch.cpu_curl_calls == 0))
      throw std::runtime_error(
          "not every MPI rank used an exclusive CUDA curl path");
    if (count_processors() > 1 &&
        !and_to_all(local_multi.mpi_messages > 0 &&
                    local_multi.mpi_scalars > 0 &&
                    local_multi.cuda_aware_bytes +
                            local_multi.pinned_staging_bytes >
                        0))
      throw std::runtime_error(
          "not every MPI rank exchanged resident CUDA boundaries");
    if (environment_flag("MEEP_GPU_EXPECT_BOUNDARY_PHASE_GRAPH") &&
        !and_to_all(local_phase_graph.launches > 0))
      throw std::runtime_error(
          "not every MPI rank launched the required CUDA boundary phase graph");
    if (expect_boundary_descriptor_fast_replay_everywhere &&
        (!boundary_descriptor_fast_replay_enabled_everywhere ||
         !and_to_all(
             local_boundary_descriptor_replay.scatter_fast_replays > 0 &&
             local_boundary_descriptor_replay.scatter_full_validations ==
                 0)))
      throw std::runtime_error(
          "not every MPI rank exclusively used cached scatter-descriptor fast replay");
    if (expect_boundary_gather_descriptor_fast_replay_everywhere &&
        (!boundary_descriptor_fast_replay_enabled_everywhere ||
         !and_to_all(
             local_boundary_descriptor_replay.gather_fast_replays > 0 &&
             local_boundary_descriptor_replay.gather_full_validations ==
                 0)))
      throw std::runtime_error(
          "not every MPI rank exclusively used cached gather-descriptor fast replay");
    if (expect_no_boundary_descriptor_fast_replay_everywhere &&
        (!boundary_descriptor_fast_replay_disabled_everywhere ||
         !and_to_all(
             local_boundary_descriptor_replay.gather_fast_replays == 0 &&
             local_boundary_descriptor_replay.scatter_fast_replays == 0 &&
             local_boundary_descriptor_replay.scatter_full_validations >
                 0)))
      throw std::runtime_error(
          "descriptor replay opt-out did not force full scatter validation on every rank");
    if (environment_flag("MEEP_GPU_EXPECT_EAGER_MPI") &&
        !and_to_all(local_overlap.eager_receive_start_calls > 0 &&
                    local_overlap.eager_send_start_calls > 0 &&
                    local_overlap.eager_receive_requests > 0 &&
                    local_overlap.eager_send_requests > 0))
      throw std::runtime_error(
          "not every MPI rank used eager CUDA-aware MPI posting");
    if (!and_to_all(
            local_overlap.eager_receive_start_calls <=
                local_overlap.eager_receive_requests &&
            local_overlap.eager_send_start_calls <=
                local_overlap.eager_send_requests))
      throw std::runtime_error(
          "eager MPI start statistics include a zero-request boundary phase");
    if (environment_flag("MEEP_GPU_EXPECT_NO_EAGER_MPI") &&
        !and_to_all(local_overlap.eager_receive_start_calls == 0 &&
                    local_overlap.eager_send_start_calls == 0 &&
                    local_overlap.eager_receive_requests == 0 &&
                    local_overlap.eager_send_requests == 0))
      throw std::runtime_error(
          "a fallback MPI transport unexpectedly used eager device posting");
    const bool expect_boundary_eh_overlap =
        environment_flag("MEEP_GPU_EXPECT_BOUNDARY_EH_OVERLAP");
    const bool expect_no_boundary_eh_overlap =
        environment_flag("MEEP_GPU_EXPECT_NO_BOUNDARY_EH_OVERLAP");
    const bool expect_boundary_eh_overlap_unsupported_schedule =
        environment_flag(
            "MEEP_GPU_EXPECT_BOUNDARY_EH_OVERLAP_UNSUPPORTED_SCHEDULE");
    const bool expect_boundary_eh_overlap_mixed =
        environment_flag("MEEP_GPU_EXPECT_BOUNDARY_EH_OVERLAP_MIXED");
    if (static_cast<int>(expect_boundary_eh_overlap) +
            static_cast<int>(expect_no_boundary_eh_overlap) +
            static_cast<int>(
                expect_boundary_eh_overlap_unsupported_schedule) +
            static_cast<int>(expect_boundary_eh_overlap_mixed) >
        1)
      throw std::runtime_error(
          "conflicting boundary E/H overlap expectations");
    if (expect_boundary_eh_overlap &&
        !and_to_all(local_boundary_eh_overlap.checks > 0 &&
                    local_boundary_eh_overlap.eligible > 0 &&
                    local_boundary_eh_overlap.launched_h > 0 &&
                    local_boundary_eh_overlap.launched_e > 0 &&
                    local_boundary_eh_overlap.skipped_disabled == 0))
      throw std::runtime_error(
          "not every MPI rank launched both boundary-overlapped E/H phases");
    if (expect_no_boundary_eh_overlap &&
        !and_to_all(local_boundary_eh_overlap.checks > 0 &&
                    local_boundary_eh_overlap.eligible == 0 &&
                    local_boundary_eh_overlap.launched_h == 0 &&
                    local_boundary_eh_overlap.launched_e == 0 &&
                    local_boundary_eh_overlap.skipped_disabled ==
                        local_boundary_eh_overlap.checks))
      throw std::runtime_error(
          "disabled boundary E/H overlap unexpectedly launched work");
    if (expect_boundary_eh_overlap_unsupported_schedule &&
        !and_to_all(local_boundary_eh_overlap.checks > 0 &&
                    local_boundary_eh_overlap.eligible == 0 &&
                    local_boundary_eh_overlap.launched_h == 0 &&
                    local_boundary_eh_overlap.launched_e == 0 &&
                    local_boundary_eh_overlap
                            .skipped_unsupported_schedule ==
                        local_boundary_eh_overlap.checks))
      throw std::runtime_error(
          "unsupported MPI schedule unexpectedly reordered E/H work");
    if (environment_flag("MEEP_GPU_EXPECT_RECEIVE_PINGPONG") &&
        !and_to_all(local_receive_pingpong.selections > 0 &&
                    local_receive_pingpong.secondary_selections > 0))
      throw std::runtime_error(
          "not every MPI rank used double-buffered CUDA-aware receives");
    if (!and_to_all(
            local_dft_batches.submitted_updates ==
                local_dfts.cuda_update_calls &&
            local_dft_batches.phase_preparation_launches +
                    local_dft_batches.phase_reuses ==
                local_dft_batches.submitted_updates &&
            local_dft_batches.multi_monitor_automatic_selected +
                    local_dft_batches.multi_monitor_automatic_rejected ==
                local_dft_batches.multi_monitor_automatic_checks &&
            local_dft_batches.multi_monitor_batched_updates +
                    local_dft_batches.multi_monitor_unbatched_updates ==
                local_dft_batches.submitted_updates &&
            local_dft_batches.update_kernel_launches ==
                local_dft_batches.multi_monitor_automatic_selected +
                    local_dft_batches.multi_monitor_forced_batches +
                    local_dft_batches.multi_monitor_unbatched_updates &&
            local_dft_batches.multi_monitor_plan_uploads +
                    local_dft_batches.multi_monitor_plan_reuses ==
                local_dft_batches.multi_monitor_automatic_selected +
                    local_dft_batches.multi_monitor_forced_batches &&
            (local_dft_batches.multi_monitor_plan_uploads == 0) ==
                (local_dft_batches
                     .multi_monitor_metadata_host_to_device_bytes == 0)))
      throw std::runtime_error(
          "DFT phase-sharing/multi-monitor statistics are internally "
          "inconsistent");
    if (environment_flag("MEEP_GPU_EXPECT_NO_DFT_PHASE_SHARING") &&
        !and_to_all(local_dft_batches.phase_reuses == 0 &&
                    local_dft_batches.phase_preparation_launches ==
                        local_dft_batches.submitted_updates))
      throw std::runtime_error(
          "disabled DFT phase sharing unexpectedly reused a phase");

    const std::uint64_t curl_cpu_calls =
        global_sum(local_dispatch.cpu_curl_calls);
    const std::uint64_t curl_cuda_calls =
        global_sum(local_dispatch.cuda_curl_calls);
    const std::uint64_t update_eh_cpu_calls =
        global_sum(local_fields.cpu_update_eh_calls);
    const std::uint64_t update_eh_cuda_calls =
        global_sum(local_fields.cuda_update_eh_calls);
    const std::uint64_t polarization_cpu_calls =
        global_sum(local_polarizations.cpu_update_calls);
    const std::uint64_t polarization_cuda_calls =
        global_sum(local_polarizations.cuda_update_calls);
    const std::uint64_t source_cpu_calls =
        global_sum(local_sources.cpu_update_calls);
    const std::uint64_t source_cuda_calls =
        global_sum(local_sources.cuda_update_calls);
    const std::uint64_t boundary_cpu_calls =
        global_sum(local_boundaries.cpu_update_calls);
    const std::uint64_t boundary_cuda_calls =
        global_sum(local_boundaries.cuda_update_calls);
    const std::uint64_t dft_cpu_calls =
        global_sum(local_dfts.cpu_update_calls);
    const std::uint64_t dft_cuda_calls =
        global_sum(local_dfts.cuda_update_calls);
    const std::uint64_t phase_graph_creations =
        global_sum(local_phase_graph.creations);
    const std::uint64_t phase_graph_launches =
        global_sum(local_phase_graph.launches);
    const std::uint64_t eager_receive_start_calls =
        global_sum(local_overlap.eager_receive_start_calls);
    const std::uint64_t eager_send_start_calls =
        global_sum(local_overlap.eager_send_start_calls);
    const std::uint64_t eager_receive_requests =
        global_sum(local_overlap.eager_receive_requests);
    const std::uint64_t eager_send_requests =
        global_sum(local_overlap.eager_send_requests);
    const std::uint64_t receive_pingpong_selections =
        global_sum(local_receive_pingpong.selections);
    const std::uint64_t receive_secondary_selections =
        global_sum(local_receive_pingpong.secondary_selections);
    const std::uint64_t boundary_eh_overlap_checks =
        global_sum(local_boundary_eh_overlap.checks);
    const std::uint64_t boundary_eh_overlap_eligible =
        global_sum(local_boundary_eh_overlap.eligible);
    const std::uint64_t boundary_eh_overlap_launched_h =
        global_sum(local_boundary_eh_overlap.launched_h);
    const std::uint64_t boundary_eh_overlap_launched_e =
        global_sum(local_boundary_eh_overlap.launched_e);
    const std::uint64_t boundary_eh_overlap_skipped_disabled =
        global_sum(local_boundary_eh_overlap.skipped_disabled);
    const std::uint64_t
        boundary_eh_overlap_skipped_unsupported_schedule =
            global_sum(
                local_boundary_eh_overlap.skipped_unsupported_schedule);
    const std::uint64_t boundary_eh_overlap_skipped_no_remote =
        global_sum(local_boundary_eh_overlap.skipped_no_remote);
    const std::uint64_t boundary_eh_overlap_skipped_cold_topology =
        global_sum(local_boundary_eh_overlap.skipped_cold_topology);
    const std::uint64_t boundary_eh_overlap_rejected =
        global_sum(local_boundary_eh_overlap.rejected);
    const std::uint64_t boundary_eh_overlap_eligible_ranks =
        global_sum(local_boundary_eh_overlap.eligible > 0 ? 1u : 0u);
    const std::uint64_t halo_curl_overlap_checks =
        global_sum(local_halo_curl_overlap.checks);
    const std::uint64_t halo_curl_overlap_eligible =
        global_sum(local_halo_curl_overlap.eligible);
    const std::uint64_t halo_curl_overlap_launches =
        global_sum(local_halo_curl_overlap.launches);
    const std::uint64_t halo_curl_overlap_skipped_disabled =
        global_sum(local_halo_curl_overlap.skipped_disabled);
    const std::uint64_t halo_curl_overlap_skipped_unsupported_schedule =
        global_sum(local_halo_curl_overlap.skipped_unsupported_schedule);
    const std::uint64_t halo_curl_overlap_skipped_no_remote =
        global_sum(local_halo_curl_overlap.skipped_no_remote);
    const std::uint64_t halo_curl_overlap_skipped_cold_topology =
        global_sum(local_halo_curl_overlap.skipped_cold_topology);
    const std::uint64_t halo_curl_overlap_rejected_feature =
        global_sum(local_halo_curl_overlap.rejected_feature);
    const std::uint64_t halo_curl_overlap_rejected_small =
        global_sum(local_halo_curl_overlap.rejected_small);
    const std::uint64_t halo_curl_overlap_full_points =
        global_sum(local_halo_curl_overlap.full_points);
    const std::uint64_t halo_curl_overlap_interior_points =
        global_sum(local_halo_curl_overlap.interior_points);
    const std::uint64_t halo_curl_overlap_shell_points =
        global_sum(local_halo_curl_overlap.shell_points);
    const std::uint64_t tile_coalesced_curl_chunk_phases =
        global_sum(local_tile_coalescing.curl_chunk_phases);
    const std::uint64_t tile_coalesced_curl_input_tiles =
        global_sum(local_tile_coalescing.curl_input_tiles);
    const std::uint64_t tile_coalesced_update_eh_chunk_phases =
        global_sum(local_tile_coalescing.update_eh_chunk_phases);
    const std::uint64_t tile_coalesced_update_eh_input_tiles =
        global_sum(local_tile_coalescing.update_eh_input_tiles);
    const std::uint64_t phase_curl_automatic_checks =
        global_sum(local_phase_batch_policy.curl_automatic_checks);
    const std::uint64_t phase_curl_automatic_selected =
        global_sum(local_phase_batch_policy.curl_automatic_selected);
    const std::uint64_t phase_curl_automatic_rejected =
        global_sum(local_phase_batch_policy.curl_automatic_rejected);
    const std::uint64_t phase_curl_forced_batches =
        global_sum(local_phase_batch_policy.curl_forced_batches);
    const std::uint64_t phase_curl_batched_operations =
        global_sum(local_phase_batch_policy.curl_batched_operations);
    const std::uint64_t phase_curl_unbatched_operations =
        global_sum(local_phase_batch_policy.curl_unbatched_operations);
    const std::uint64_t phase_curl_replay_checks =
        global_sum(local_curl_phase_replay.checks);
    const std::uint64_t phase_curl_replay_hits =
        global_sum(local_curl_phase_replay.hits);
    const std::uint64_t phase_curl_replay_unready =
        global_sum(local_curl_phase_replay.unready);
    const std::uint64_t phase_curl_replay_generation_misses =
        global_sum(local_curl_phase_replay.generation_misses);
    const std::uint64_t phase_curl_replay_mirror_misses =
        global_sum(local_curl_phase_replay.mirror_misses);
    const std::uint64_t phase_curl_replay_disabled_ranks =
        global_sum(
            std::getenv("MEEP_GPU_DISABLE_CURL_PHASE_REPLAY") ? 1u : 0u);
    const std::uint64_t phase_update_eh_automatic_checks =
        global_sum(local_phase_batch_policy.update_eh_automatic_checks);
    const std::uint64_t phase_update_eh_automatic_selected =
        global_sum(local_phase_batch_policy.update_eh_automatic_selected);
    const std::uint64_t phase_update_eh_automatic_rejected =
        global_sum(local_phase_batch_policy.update_eh_automatic_rejected);
    const std::uint64_t phase_update_eh_forced_batches =
        global_sum(local_phase_batch_policy.update_eh_forced_batches);
    const std::uint64_t phase_update_eh_batched_operations =
        global_sum(local_phase_batch_policy.update_eh_batched_operations);
    const std::uint64_t phase_update_eh_unbatched_operations =
        global_sum(local_phase_batch_policy.update_eh_unbatched_operations);
    const std::uint64_t boundary_gather_fast_replays =
        global_sum(
            local_boundary_descriptor_replay.gather_fast_replays);
    const std::uint64_t boundary_gather_full_validations =
        global_sum(
            local_boundary_descriptor_replay.gather_full_validations);
    const std::uint64_t boundary_scatter_fast_replays =
        global_sum(
            local_boundary_descriptor_replay.scatter_fast_replays);
    const std::uint64_t boundary_scatter_full_validations =
        global_sum(
            local_boundary_descriptor_replay.scatter_full_validations);
    if (halo_curl_overlap_interior_points +
            halo_curl_overlap_shell_points !=
        halo_curl_overlap_full_points)
      throw std::runtime_error(
          "halo/curl overlap point accounting does not partition full curl work");
    if (environment_flag("MEEP_GPU_EXPECT_HALO_CURL_OVERLAP") &&
        !and_to_all(local_halo_curl_overlap.checks > 0 &&
                    local_halo_curl_overlap.eligible > 0 &&
                    local_halo_curl_overlap.launches > 0 &&
                    local_halo_curl_overlap.skipped_disabled == 0 &&
                    local_halo_curl_overlap.full_points > 0 &&
                    local_halo_curl_overlap.interior_points > 0 &&
                    local_halo_curl_overlap.shell_points > 0))
      throw std::runtime_error(
          "not every MPI rank launched the required halo/curl overlap path");
    if (environment_flag("MEEP_GPU_EXPECT_NO_HALO_CURL_OVERLAP") &&
        !and_to_all(local_halo_curl_overlap.checks > 0 &&
                    local_halo_curl_overlap.eligible == 0 &&
                    local_halo_curl_overlap.launches == 0 &&
                    local_halo_curl_overlap.skipped_disabled ==
                        local_halo_curl_overlap.checks))
      throw std::runtime_error(
          "disabled halo/curl overlap path unexpectedly launched");
    if (environment_flag(
            "MEEP_GPU_EXPECT_HALO_CURL_OVERLAP_UNSUPPORTED_SCHEDULE") &&
        !and_to_all(
            local_halo_curl_overlap.checks > 0 &&
            local_halo_curl_overlap.eligible == 0 &&
            local_halo_curl_overlap.launches == 0 &&
            local_halo_curl_overlap.skipped_unsupported_schedule ==
                local_halo_curl_overlap.checks))
      throw std::runtime_error(
          "halo/curl overlap did not prove unsupported-schedule fallback");
    if (environment_flag(
            "MEEP_GPU_EXPECT_HALO_CURL_OVERLAP_REJECTED_FEATURE") &&
        !and_to_all(local_halo_curl_overlap.checks > 0 &&
                    local_halo_curl_overlap.eligible == 0 &&
                    local_halo_curl_overlap.launches == 0 &&
                    local_halo_curl_overlap.rejected_feature ==
                        local_halo_curl_overlap.checks))
      throw std::runtime_error(
          "halo/curl overlap did not prove feature-rejection fallback");
    if (environment_flag(
            "MEEP_GPU_EXPECT_HALO_CURL_OVERLAP_REJECTED_SMALL") &&
        !and_to_all(local_halo_curl_overlap.checks > 0 &&
                    local_halo_curl_overlap.eligible == 0 &&
                    local_halo_curl_overlap.launches == 0 &&
                    local_halo_curl_overlap.rejected_small ==
                        local_halo_curl_overlap.checks))
      throw std::runtime_error(
          "halo/curl overlap did not prove small-workload fallback");
    if (environment_flag("MEEP_GPU_EXPECT_TILE_COALESCING") &&
        !and_to_all(local_tile_coalescing.curl_chunk_phases > 0 &&
                    local_tile_coalescing.curl_input_tiles >
                        local_tile_coalescing.curl_chunk_phases))
      throw std::runtime_error(
          "CUDA loop-tiled curl did not prove tile coalescing");
    if (environment_flag("MEEP_GPU_EXPECT_NO_TILE_COALESCING") &&
        !and_to_all(local_tile_coalescing.curl_chunk_phases == 0 &&
                    local_tile_coalescing.curl_input_tiles == 0 &&
                    local_tile_coalescing.update_eh_chunk_phases == 0 &&
                    local_tile_coalescing.update_eh_input_tiles == 0))
      throw std::runtime_error(
          "disabled CUDA tile coalescing reported optimized execution");
    if (expect_boundary_eh_overlap_mixed &&
        !(boundary_eh_overlap_eligible > 0 &&
          boundary_eh_overlap_rejected > 0 &&
          boundary_eh_overlap_eligible_ranks > 0 &&
          boundary_eh_overlap_eligible_ranks <
              static_cast<std::uint64_t>(count_processors()) &&
          boundary_eh_overlap_launched_h +
                  boundary_eh_overlap_launched_e ==
              boundary_eh_overlap_eligible))
      throw std::runtime_error(
          "mixed-rank E/H overlap workload did not prove safe asymmetric fallback");
    const std::uint64_t warmup_receive_secondary_allocations =
        global_sum(warmup_receive_pingpong.secondary_allocations);
    const std::uint64_t dft_batch_calls =
        global_sum(local_dft_batches.batch_calls);
    const std::uint64_t warmup_dft_multi_plan_uploads =
        global_sum(warmup_dft_batches.multi_monitor_plan_uploads);
    const std::uint64_t warmup_dft_multi_plan_reuses =
        global_sum(warmup_dft_batches.multi_monitor_plan_reuses);
    const std::uint64_t warmup_dft_multi_metadata_h2d_bytes = global_sum(
        warmup_dft_batches.multi_monitor_metadata_host_to_device_bytes);
    const std::uint64_t dft_submitted_updates =
        global_sum(local_dft_batches.submitted_updates);
    const std::uint64_t dft_phase_preparation_launches =
        global_sum(local_dft_batches.phase_preparation_launches);
    const std::uint64_t dft_phase_reuses =
        global_sum(local_dft_batches.phase_reuses);
    const std::uint64_t dft_update_kernel_launches =
        global_sum(local_dft_batches.update_kernel_launches);
    const std::uint64_t dft_maximum_batch_size =
        global_max(local_dft_batches.maximum_batch_size);
    const std::uint64_t dft_multi_automatic_checks =
        global_sum(local_dft_batches.multi_monitor_automatic_checks);
    const std::uint64_t dft_multi_automatic_selected =
        global_sum(local_dft_batches.multi_monitor_automatic_selected);
    const std::uint64_t dft_multi_automatic_rejected =
        global_sum(local_dft_batches.multi_monitor_automatic_rejected);
    const std::uint64_t dft_multi_forced_batches =
        global_sum(local_dft_batches.multi_monitor_forced_batches);
    const std::uint64_t dft_multi_batched_updates =
        global_sum(local_dft_batches.multi_monitor_batched_updates);
    const std::uint64_t dft_multi_unbatched_updates =
        global_sum(local_dft_batches.multi_monitor_unbatched_updates);
    const std::uint64_t dft_multi_plan_uploads =
        global_sum(local_dft_batches.multi_monitor_plan_uploads);
    const std::uint64_t dft_multi_plan_reuses =
        global_sum(local_dft_batches.multi_monitor_plan_reuses);
    const std::uint64_t dft_multi_metadata_h2d_bytes = global_sum(
        local_dft_batches.multi_monitor_metadata_host_to_device_bytes);
    const bool expect_dft_phase_sharing =
        environment_flag("MEEP_GPU_EXPECT_DFT_PHASE_SHARING");
    const bool expect_no_dft_phase_sharing =
        environment_flag("MEEP_GPU_EXPECT_NO_DFT_PHASE_SHARING");
    if (expect_dft_phase_sharing && expect_no_dft_phase_sharing)
      throw std::runtime_error(
          "conflicting DFT phase-sharing benchmark expectations");
    if (dft_batch_calls == 0 || dft_submitted_updates == 0 ||
        dft_phase_preparation_launches + dft_phase_reuses !=
            dft_submitted_updates ||
        dft_maximum_batch_size == 0 ||
        dft_multi_automatic_selected + dft_multi_automatic_rejected !=
            dft_multi_automatic_checks ||
        dft_multi_batched_updates + dft_multi_unbatched_updates !=
            dft_submitted_updates ||
        dft_update_kernel_launches !=
            dft_multi_automatic_selected + dft_multi_forced_batches +
                dft_multi_unbatched_updates ||
        dft_multi_plan_uploads + dft_multi_plan_reuses !=
            dft_multi_automatic_selected + dft_multi_forced_batches ||
        (dft_multi_plan_uploads == 0) !=
            (dft_multi_metadata_h2d_bytes == 0))
      throw std::runtime_error(
          "distributed DFT batch accounting is inconsistent");
    if (expect_dft_phase_sharing &&
        (dft_phase_reuses == 0 ||
         dft_phase_preparation_launches >= dft_submitted_updates))
      throw std::runtime_error(
          "the distributed workload reused no resident DFT phases");
    if (expect_no_dft_phase_sharing &&
        (dft_phase_reuses != 0 ||
         dft_phase_preparation_launches != dft_submitted_updates))
      throw std::runtime_error(
          "disabled distributed DFT phase sharing reused a phase");
    const bool expect_multi_monitor_dft =
        environment_flag("MEEP_GPU_EXPECT_MULTI_MONITOR_DFT_BATCH");
    const bool expect_no_multi_monitor_dft =
        environment_flag("MEEP_GPU_EXPECT_NO_MULTI_MONITOR_DFT_BATCH");
    if (expect_multi_monitor_dft && expect_no_multi_monitor_dft)
      throw std::runtime_error(
          "conflicting multi-monitor DFT benchmark expectations");
    if (expect_multi_monitor_dft &&
        (dft_multi_batched_updates == 0 ||
         dft_update_kernel_launches >= dft_submitted_updates ||
         dft_multi_plan_reuses == 0))
      throw std::runtime_error(
          "the distributed workload did not reuse a fused multi-monitor DFT "
          "plan");
    if (expect_no_multi_monitor_dft &&
        (dft_multi_batched_updates != 0 ||
         dft_multi_unbatched_updates != dft_submitted_updates ||
         dft_update_kernel_launches != dft_submitted_updates ||
         dft_multi_plan_uploads != 0 || dft_multi_plan_reuses != 0 ||
         dft_multi_metadata_h2d_bytes != 0))
      throw std::runtime_error(
          "disabled multi-monitor DFT batching reported fused execution");

    const std::uint64_t cpu_calls =
        curl_cpu_calls + update_eh_cpu_calls + polarization_cpu_calls +
        source_cpu_calls + boundary_cpu_calls + dft_cpu_calls;
    const std::uint64_t cuda_calls =
        curl_cuda_calls + update_eh_cuda_calls + polarization_cuda_calls +
        source_cuda_calls + boundary_cuda_calls + dft_cuda_calls;
    if (curl_cpu_calls != 0 || curl_cuda_calls == 0 ||
        update_eh_cpu_calls != 0 || update_eh_cuda_calls == 0 ||
        boundary_cpu_calls != 0 || boundary_cuda_calls == 0 ||
        dft_cpu_calls != 0 || dft_cuda_calls == 0)
      throw std::runtime_error(
          "multi-GPU workload did not use every required CUDA phase exclusively");
    if ((source_enabled &&
         (source_cpu_calls != 0 || source_cuda_calls == 0)) ||
        (!source_enabled &&
         (source_cpu_calls != 0 || source_cuda_calls != 0)))
      throw std::runtime_error(
          "multi-GPU source dispatch differs from the fixed workload");
    if (polarization_cpu_calls != 0 || polarization_cuda_calls != 0)
      throw std::runtime_error(
          "vacuum multi-GPU workload unexpectedly updated polarization");
    if (cpu_calls != 0 || cuda_calls == 0)
      throw std::logic_error(
          "multi-GPU phase statistics aggregate is inconsistent");

    const std::uint64_t mpi_messages =
        global_sum(local_multi.mpi_messages);
    const std::uint64_t mpi_scalars =
        global_sum(local_multi.mpi_scalars);
    const std::uint64_t cuda_aware_bytes =
        global_sum(local_multi.cuda_aware_bytes);
    const std::uint64_t pinned_bytes =
        global_sum(local_multi.pinned_staging_bytes);
    if (count_processors() > 1 &&
        (mpi_messages == 0 || mpi_scalars == 0 ||
         cuda_aware_bytes + pinned_bytes == 0))
      throw std::runtime_error(
          "multi-GPU performance workload exchanged no device boundary data");
    if (count_processors() == 1 &&
        (mpi_messages != 0 || mpi_scalars != 0))
      throw std::runtime_error(
          "single-GPU performance workload unexpectedly used MPI boundaries");

    const std::vector<vec> probe_points = cut_probe_points(resolution);
    std::vector<std::complex<double> > probes;
    probes.reserve(probe_points.size());
    double cut_probe_squared = 0.0;
    for (const vec &point : probe_points) {
      const std::complex<double> value = f.get_field(Ez, point);
      probes.push_back(value);
      cut_probe_squared += std::norm(value);
    }
    const double cut_probe_l2 = std::sqrt(cut_probe_squared);
    const double energy = f.field_energy();
    const double dft = f.dft_norm();
    const component ez_component[] = {Ez};
    const double ez_l2_squared = std::real(f.integrate(
        1, ez_component, squared_field_integrand, nullptr, f.v));
    const double ez_l2 =
        ez_l2_squared > 0.0 ? std::sqrt(ez_l2_squared) : 0.0;
    const double ez_weighted_checksum = std::real(f.integrate(
        1, ez_component, weighted_squared_field_integrand, nullptr, f.v));

    // connections_in points directly at the rank-local halo destinations.
    // Synchronize their owning mirrors before dereferencing them, then reduce
    // only connections whose source chunk lives on another rank.  This proves
    // that actual remote cut data, rather than merely MPI byte counters, is
    // nonzero in the measured state.
    for (int chunk_index = 0; chunk_index < f.num_chunks; ++chunk_index)
      if (f.chunks[chunk_index]->is_mine())
        gpu::detail::sync_resident_cache_for_owner(f.chunks[chunk_index]);
    std::uint64_t local_remote_cut_scalars = 0;
    long double local_remote_cut_squared = 0.0;
    for (int chunk_index = 0; chunk_index < f.num_chunks; ++chunk_index) {
      fields_chunk *destination = f.chunks[chunk_index];
      if (!destination->is_mine()) continue;
      for (const auto &entry : destination->connections_in) {
        const comms_key &key = entry.first;
        if (key.pair.first < 0 || key.pair.first >= f.num_chunks ||
            key.pair.second != chunk_index)
          throw std::logic_error(
              "multi-GPU benchmark encountered malformed boundary topology");
        if (f.chunks[key.pair.first]->is_mine()) continue;
        for (const realnum *value : entry.second) {
          if (!value)
            throw std::logic_error(
                "multi-GPU benchmark encountered a null remote-cut value");
          const long double promoted = static_cast<long double>(*value);
          local_remote_cut_squared += promoted * promoted;
          ++local_remote_cut_scalars;
        }
      }
    }
    const std::uint64_t remote_cut_scalars =
        global_sum(local_remote_cut_scalars);
    const double remote_cut_squared =
        sum_to_all(static_cast<double>(local_remote_cut_squared));
    const double remote_cut_l2 =
        remote_cut_squared > 0.0 ? std::sqrt(remote_cut_squared) : 0.0;

    bool probes_finite = true;
    for (const std::complex<double> &probe : probes)
      probes_finite = probes_finite && std::isfinite(probe.real()) &&
                      std::isfinite(probe.imag());
    if (!probes_finite || !std::isfinite(cut_probe_l2) ||
        !std::isfinite(energy) || !std::isfinite(dft) ||
        !std::isfinite(ez_l2) || !std::isfinite(ez_weighted_checksum) ||
        !std::isfinite(remote_cut_l2))
      throw std::runtime_error(
          "multi-GPU performance workload produced a non-finite result");
    if (!(cut_probe_l2 > 0.0 && energy > 0.0 && dft > 0.0 &&
          ez_l2 > 0.0 && ez_weighted_checksum > 0.0))
      throw std::runtime_error(
          "multi-GPU performance workload produced a vacuous zero observable");
    if (count_processors() > 1 &&
        !(remote_cut_scalars > 0 && remote_cut_l2 > 0.0))
      throw std::runtime_error(
          "multi-GPU performance workload produced no nonzero remote-cut data");
    if (count_processors() == 1 &&
        (remote_cut_scalars != 0 || remote_cut_l2 != 0.0))
      throw std::runtime_error(
          "single-GPU workload unexpectedly reported remote-cut data");

    const std::uint64_t cells =
        static_cast<std::uint64_t>(pixels) * pixels * pixels;
    const double mcells_per_second =
        static_cast<double>(cells) * measured_steps / seconds / 1e6;
    const std::uint64_t h2d_bytes =
        global_sum(local_dispatch.host_to_device_bytes);
    const std::uint64_t d2h_bytes =
        global_sum(local_dispatch.device_to_host_bytes);

    const std::string transport_request = requested_transport();
    const std::string completion_policy = requested_completion_policy();
    std::string selected_transport = "none";
    if (count_processors() > 1) {
      if ((cuda_aware_bytes > 0) == (pinned_bytes > 0))
        throw std::runtime_error(
            "multi-GPU workload did not use exactly one MPI transport");
      selected_transport =
          cuda_aware_bytes > 0 ? "cuda-aware" : "pinned";
      if (transport_request != "auto" &&
          transport_request != selected_transport)
        throw std::runtime_error(
            "selected MPI transport differs from the forced request");
      if (mpi_scalars >
          std::numeric_limits<std::uint64_t>::max() / sizeof(realnum))
        throw std::overflow_error("MPI scalar byte count overflow");
      if (cuda_aware_bytes + pinned_bytes !=
          mpi_scalars * sizeof(realnum))
        throw std::runtime_error(
            "MPI transport bytes differ from the FP32 scalar count");
      if (selected_transport == "pinned" &&
          (h2d_bytes == 0 || d2h_bytes == 0))
        throw std::runtime_error(
            "pinned MPI transport recorded no H2D or D2H activity");
    }

    const int selected_ordinal = gpu::selected_device();
    const std::string selected_identifier =
        gpu::selected_device_identifier();
    const std::vector<gpu::device_info> devices = gpu::enumerate_devices();
    const auto selected_info = std::find_if(
        devices.begin(), devices.end(),
        [selected_ordinal](const gpu::device_info &device) {
          return device.ordinal == selected_ordinal;
        });
    if (selected_ordinal < 0 || selected_identifier.empty() ||
        selected_info == devices.end() || !selected_info->compatible)
      throw std::runtime_error(
          "rank cannot prove its selected compatible CUDA device identity");
    std::ostringstream device_record;
    device_record << device_record_prefix
                  << "{\"rank\":" << my_global_rank()
                  << ",\"ordinal\":" << selected_ordinal
                  << ",\"uuid\":" << json_string(selected_identifier)
                  << ",\"name\":" << json_string(selected_info->name)
                  << ",\"compute_major\":"
                  << selected_info->compute_major
                  << ",\"compute_minor\":"
                  << selected_info->compute_minor
                  << ",\"compatible\":true}";
    std::cout << device_record.str() << '\n' << std::flush;
    if (environment_flag("MEEP_GPU_EMIT_PROCESS_ID"))
      std::cout << process_record_prefix
                << "{\"rank\":" << my_global_rank()
                << ",\"pid\":" << static_cast<long long>(getpid())
                << "}\n" << std::flush;
    all_wait();

    if (am_master()) {
      std::cout << "gpmeep-boundary-phase-graph-v1:{\"creations\":"
                << phase_graph_creations << ",\"launches\":"
                << phase_graph_launches << "}\n";
      std::cout << "gpmeep-eager-mpi-v1:{\"receive_start_calls\":"
                << eager_receive_start_calls
                << ",\"send_start_calls\":" << eager_send_start_calls
                << ",\"receive_requests\":" << eager_receive_requests
                << ",\"send_requests\":" << eager_send_requests
                << "}\n";
      std::cout << "gpmeep-receive-pingpong-v1:{\"warmup_secondary_allocations\":"
                << warmup_receive_secondary_allocations
                << ",\"selections\":" << receive_pingpong_selections
                << ",\"secondary_selections\":"
                << receive_secondary_selections << "}\n";
      std::cout << "gpmeep-boundary-eh-overlap-v1:{\"checks\":"
                << boundary_eh_overlap_checks
                << ",\"warmup_skipped_cold_topology\":"
                << warmup_boundary_eh_overlap_cold_topology
                << ",\"warmup_launched_h\":"
                << warmup_boundary_eh_overlap_launched_h
                << ",\"warmup_launched_e\":"
                << warmup_boundary_eh_overlap_launched_e
                << ",\"eligible_ranks\":"
                << boundary_eh_overlap_eligible_ranks
                << ",\"eligible\":" << boundary_eh_overlap_eligible
                << ",\"launched_h\":"
                << boundary_eh_overlap_launched_h
                << ",\"launched_e\":"
                << boundary_eh_overlap_launched_e
                << ",\"skipped_disabled\":"
                << boundary_eh_overlap_skipped_disabled
                << ",\"skipped_unsupported_schedule\":"
                << boundary_eh_overlap_skipped_unsupported_schedule
                << ",\"skipped_no_remote\":"
                << boundary_eh_overlap_skipped_no_remote
                << ",\"skipped_cold_topology\":"
                << boundary_eh_overlap_skipped_cold_topology
                << ",\"rejected\":" << boundary_eh_overlap_rejected
                << "}\n";
      std::cout << "gpmeep-halo-curl-overlap-v1:{\"checks\":"
                << halo_curl_overlap_checks
                << ",\"warmup_skipped_cold_topology\":"
                << warmup_halo_curl_overlap_cold_topology
                << ",\"warmup_launches\":"
                << warmup_halo_curl_overlap_launches
                << ",\"eligible\":" << halo_curl_overlap_eligible
                << ",\"launches\":" << halo_curl_overlap_launches
                << ",\"skipped_disabled\":"
                << halo_curl_overlap_skipped_disabled
                << ",\"skipped_unsupported_schedule\":"
                << halo_curl_overlap_skipped_unsupported_schedule
                << ",\"skipped_no_remote\":"
                << halo_curl_overlap_skipped_no_remote
                << ",\"skipped_cold_topology\":"
                << halo_curl_overlap_skipped_cold_topology
                << ",\"rejected_feature\":"
                << halo_curl_overlap_rejected_feature
                << ",\"rejected_small\":"
                << halo_curl_overlap_rejected_small
                << ",\"full_points\":"
                << halo_curl_overlap_full_points
                << ",\"interior_points\":"
                << halo_curl_overlap_interior_points
                << ",\"shell_points\":"
                << halo_curl_overlap_shell_points << "}\n";
      std::cout << "gpmeep-tile-coalescing-v1:{\"curl_chunk_phases\":"
                << tile_coalesced_curl_chunk_phases
                << ",\"curl_input_tiles\":"
                << tile_coalesced_curl_input_tiles
                << ",\"update_eh_chunk_phases\":"
                << tile_coalesced_update_eh_chunk_phases
                << ",\"update_eh_input_tiles\":"
                << tile_coalesced_update_eh_input_tiles << "}\n";
      std::cout
          << "gpmeep-phase-batch-policy-v1:{\"curl_automatic_checks\":"
          << phase_curl_automatic_checks
          << ",\"curl_automatic_selected\":"
          << phase_curl_automatic_selected
          << ",\"curl_automatic_rejected\":"
          << phase_curl_automatic_rejected
          << ",\"curl_forced_batches\":"
          << phase_curl_forced_batches
          << ",\"curl_batched_operations\":"
          << phase_curl_batched_operations
          << ",\"curl_unbatched_operations\":"
          << phase_curl_unbatched_operations
          << ",\"update_eh_automatic_checks\":"
          << phase_update_eh_automatic_checks
          << ",\"update_eh_automatic_selected\":"
          << phase_update_eh_automatic_selected
          << ",\"update_eh_automatic_rejected\":"
          << phase_update_eh_automatic_rejected
          << ",\"update_eh_forced_batches\":"
          << phase_update_eh_forced_batches
          << ",\"update_eh_batched_operations\":"
          << phase_update_eh_batched_operations
          << ",\"update_eh_unbatched_operations\":"
          << phase_update_eh_unbatched_operations << "}\n";
      std::cout
          << "gpmeep-curl-phase-replay-v1:{\"disabled_ranks\":"
          << phase_curl_replay_disabled_ranks
          << ",\"checks\":" << phase_curl_replay_checks
          << ",\"hits\":" << phase_curl_replay_hits
          << ",\"unready\":" << phase_curl_replay_unready
          << ",\"generation_misses\":"
          << phase_curl_replay_generation_misses
          << ",\"mirror_misses\":"
          << phase_curl_replay_mirror_misses << "}\n";
      std::cout
          << "gpmeep-boundary-descriptor-replay-v1:{\"disabled\":"
          << (boundary_descriptor_fast_replay_disabled_everywhere
                  ? "true" : "false")
          << ",\"disabled_ranks\":"
          << boundary_descriptor_fast_replay_disabled_ranks
          << ",\"gather_fast_replays\":"
          << boundary_gather_fast_replays
          << ",\"gather_full_validations\":"
          << boundary_gather_full_validations
          << ",\"scatter_fast_replays\":"
          << boundary_scatter_fast_replays
          << ",\"scatter_full_validations\":"
          << boundary_scatter_full_validations << "}\n";
      std::cout << dft_warmup_plan_record_prefix
                << "{\"metadata_host_to_device_bytes\":"
                << warmup_dft_multi_metadata_h2d_bytes
                << ",\"plan_reuses\":" << warmup_dft_multi_plan_reuses
                << ",\"plan_uploads\":" << warmup_dft_multi_plan_uploads
                << "}\n";
      std::cout << "gpmeep-dft-phase-sharing-v1:{\"batch_calls\":"
                << dft_batch_calls
                << ",\"submitted_updates\":" << dft_submitted_updates
                << ",\"phase_preparation_launches\":"
                << dft_phase_preparation_launches
                << ",\"phase_reuses\":" << dft_phase_reuses
                << ",\"update_kernel_launches\":"
                << dft_update_kernel_launches
                << ",\"maximum_batch_size\":"
                << dft_maximum_batch_size << "}\n";
      std::cout
          << "gpmeep-dft-multi-monitor-batch-v1:{\"automatic_checks\":"
          << dft_multi_automatic_checks
          << ",\"automatic_selected\":"
          << dft_multi_automatic_selected
          << ",\"automatic_rejected\":"
          << dft_multi_automatic_rejected
          << ",\"forced_batches\":" << dft_multi_forced_batches
          << ",\"batched_updates\":" << dft_multi_batched_updates
          << ",\"unbatched_updates\":"
          << dft_multi_unbatched_updates
          << ",\"plan_uploads\":" << dft_multi_plan_uploads
          << ",\"plan_reuses\":" << dft_multi_plan_reuses
          << ",\"metadata_host_to_device_bytes\":"
          << dft_multi_metadata_h2d_bytes << "}\n";
      std::cout << initial_condition_record_prefix
                << "{\"profile\":" << json_string(initial_condition)
                << "}\n";
      std::cout << source_profile_record_prefix
                << "{\"profile\":" << json_string(source_profile)
                << "}\n";
      std::cout << initialization_timing_record_prefix
                << std::setprecision(17)
                << "{\"applications_per_rank\":"
                << initialization_applications
                << ",\"max_seconds\":" << max_initialization_seconds
                << "}\n";
      std::cout << record_prefix << std::setprecision(17)
                << "{\"schema_version\":4"
                << ",\"mpi_ranks\":" << count_processors()
                << ",\"pixels\":" << pixels
                << ",\"cells\":" << cells
                << ",\"warmup_steps\":" << warmup_steps
                << ",\"steps\":" << measured_steps
                << ",\"loop_tile_base_db\":" << loop_tile_base_db
                << ",\"bfast\":" << (bfast ? "true" : "false")
                << ",\"source_enabled\":"
                << (source_enabled ? "true" : "false")
                << ",\"overlap_material\":"
                << (overlap_material ? "true" : "false")
                << ",\"seconds\":" << seconds
                << ",\"mcells_per_second\":" << mcells_per_second
                << ",\"requested_transport\":\"" << transport_request
                << "\",\"selected_transport\":\"" << selected_transport
                << "\",\"completion_policy\":\"" << completion_policy
                << "\",\"initial_condition\":"
                << json_string(initial_condition)
                << ",\"cut_probe_values\":[";
      for (std::size_t index = 0; index < probes.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << '[' << probes[index].real() << ','
                  << probes[index].imag() << ']';
      }
      std::cout << "]"
                << ",\"cut_probe_l2\":" << cut_probe_l2
                << ",\"ez_l2\":" << ez_l2
                << ",\"ez_weighted_checksum\":"
                << ez_weighted_checksum
                << ",\"energy\":" << energy
                << ",\"dft_norm\":" << dft
                << ",\"remote_cut_scalars\":" << remote_cut_scalars
                << ",\"remote_cut_l2\":" << remote_cut_l2
                << ",\"cpu_calls\":" << cpu_calls
                << ",\"cuda_calls\":" << cuda_calls
                << ",\"phase_calls\":{"
                << "\"curl\":{\"cpu_calls\":" << curl_cpu_calls
                << ",\"cuda_calls\":" << curl_cuda_calls
                << ",\"expectation\":\"cuda_required\"},"
                << "\"update_eh\":{\"cpu_calls\":" << update_eh_cpu_calls
                << ",\"cuda_calls\":" << update_eh_cuda_calls
                << ",\"expectation\":\"cuda_required\"},"
                << "\"source\":{\"cpu_calls\":" << source_cpu_calls
                << ",\"cuda_calls\":" << source_cuda_calls
                << ",\"expectation\":\""
                << (source_enabled ? "cuda_required" : "not_applicable")
                << "\"},"
                << "\"boundary\":{\"cpu_calls\":" << boundary_cpu_calls
                << ",\"cuda_calls\":" << boundary_cuda_calls
                << ",\"expectation\":\"cuda_required\"},"
                << "\"dft\":{\"cpu_calls\":" << dft_cpu_calls
                << ",\"cuda_calls\":" << dft_cuda_calls
                << ",\"expectation\":\"cuda_required\"},"
                << "\"polarization\":{\"cpu_calls\":"
                << polarization_cpu_calls << ",\"cuda_calls\":"
                << polarization_cuda_calls
                << ",\"expectation\":\"not_applicable\"}}"
                << ",\"h2d_bytes\":" << h2d_bytes
                << ",\"d2h_bytes\":" << d2h_bytes
                << ",\"mpi_messages\":" << mpi_messages
                << ",\"mpi_scalars\":" << mpi_scalars
                << ",\"cuda_aware_bytes\":" << cuda_aware_bytes
                << ",\"pinned_bytes\":" << pinned_bytes
                << "}\n";
    }
    gpu::set_backend(gpu::backend_mode::cpu);
    return 0;
  }
  catch (const std::exception &error) {
    std::cerr << "FAIL: rank " << my_global_rank() << ": " << error.what()
              << '\n';
    return 1;
  }
}
