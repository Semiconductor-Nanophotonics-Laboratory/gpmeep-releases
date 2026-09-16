/* Production-scale CUDA performance gate. This is built by `make check` but
   run explicitly by scripts/benchmark-gpu-core.sh on a physical GPU. */

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
#include <vector>

using namespace meep;

namespace {

double vacuum(const vec &) { return 1.0; }

struct measurement {
  double seconds;
  std::complex<double> field_sample;
  double dft_norm;
  gpu::dispatch_statistics dispatch;
  gpu::detail::finite_check_transfer_statistics finite_check;
  gpu::field_update_statistics fields;
  gpu::polarization_statistics polarizations;
  gpu::source_statistics sources;
  gpu::boundary_statistics boundaries;
  gpu::dft_statistics dfts;
};

double environment_double(const char *name, double fallback) {
  const char *value = std::getenv(name);
  if (!value || !value[0]) return fallback;
  char *end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (!end || *end != '\0' || !std::isfinite(parsed) || parsed <= 0)
    throw std::invalid_argument(std::string(name) +
                                " must be a positive finite number");
  return parsed;
}

int environment_int(const char *name, int fallback) {
  const char *value = std::getenv(name);
  if (!value || !value[0]) return fallback;
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (!end || *end != '\0' || parsed <= 0 ||
      parsed > std::numeric_limits<int>::max())
    throw std::invalid_argument(std::string(name) +
                                " must be a positive integer");
  return static_cast<int>(parsed);
}

measurement run_2d(gpu::backend_mode mode, int pixels, int measured_steps,
                   bool dispersive) {
  gpu::set_backend(mode);
  const double length = 8.0;
  const double resolution = pixels / length;
  const grid_volume gv = vol2d(length, length, resolution);
  structure s(gv, vacuum, pml(0.8));
  if (dispersive)
    s.add_susceptibility(
        vacuum, E_stuff, lorentzian_susceptibility(0.32, 0.06));
  fields f(&s, 0.0, 0.0, true, 128, 128);
  f.use_real_fields();
  // A continuous source keeps source and DFT paths active at every grid
  // resolution. A fixed number of Gaussian-source steps covers a different
  // physical time as resolution changes and can leave larger cases before
  // the monitor's active interval, producing a false coverage failure.
  continuous_src_time source(0.24);
  // Use a production-like line source so the benchmark detects accidental
  // O(source-points) host/device profile uploads on every time step.
  f.add_volume_source(
      Ez, source, volume(vec(0.5 * length, 1.0),
                         vec(0.5 * length, length - 1.0)), 1.0);
  component components[] = {Ez, Hx, Hy};
  dft_fields monitor =
      f.add_dft_fields(components, 3, f.v, 0.18, 0.34, 4);
  (void)monitor;

  for (int step = 0; step < 12; ++step)
    f.step();
  gpu::reset_dispatch_statistics();
  const auto start = std::chrono::steady_clock::now();
  for (int step = 0; step < measured_steps; ++step)
    f.step();
  const auto stop = std::chrono::steady_clock::now();

  measurement result;
  result.seconds = std::chrono::duration<double>(stop - start).count();
  result.dispatch = gpu::get_dispatch_statistics();
  result.finite_check =
      gpu::detail::get_finite_check_transfer_statistics();
  result.fields = gpu::get_field_update_statistics();
  result.polarizations = gpu::get_polarization_statistics();
  result.sources = gpu::get_source_statistics();
  result.boundaries = gpu::get_boundary_statistics();
  result.dfts = gpu::get_dft_statistics();
  result.field_sample = f.get_field(Ez, vec(4.37, 3.61));
  result.dft_norm = f.dft_norm();
  return result;
}

measurement run_2d_gyrotropic(gpu::backend_mode mode, int pixels,
                              int measured_steps) {
  gpu::set_backend(mode);
  const double length = 8.0;
  const double resolution = pixels / length;
  const grid_volume gv = vol2d(length, length, resolution);
  structure s(gv, vacuum, pml(0.8));
  s.add_susceptibility(
      [](const vec &) { return 0.08; }, E_stuff,
      gyrotropic_susceptibility(
          vec(0.0, 0.0, 1.0), 0.74, 0.025, 2e-4,
          GYROTROPIC_SATURATED));
  fields f(&s, 0.0, 0.0, true, 128, 128);
  f.use_real_fields();
  continuous_src_time source(0.24);
  f.add_volume_source(
      Ex, source, volume(vec(0.5 * length, 1.0),
                         vec(0.5 * length, length - 1.0)), 1.0);
  component components[] = {Ex, Ey, Hz};
  dft_fields monitor =
      f.add_dft_fields(components, 3, f.v, 0.18, 0.34, 4);
  (void)monitor;

  for (int step = 0; step < 12; ++step)
    f.step();
  gpu::reset_dispatch_statistics();
  const auto start = std::chrono::steady_clock::now();
  for (int step = 0; step < measured_steps; ++step)
    f.step();
  const auto stop = std::chrono::steady_clock::now();

  measurement result;
  result.seconds =
      std::chrono::duration<double>(stop - start).count();
  result.dispatch = gpu::get_dispatch_statistics();
  result.finite_check =
      gpu::detail::get_finite_check_transfer_statistics();
  result.fields = gpu::get_field_update_statistics();
  result.polarizations = gpu::get_polarization_statistics();
  result.sources = gpu::get_source_statistics();
  result.boundaries = gpu::get_boundary_statistics();
  result.dfts = gpu::get_dft_statistics();
  result.field_sample = f.get_field(Ex, vec(4.37, 3.61));
  result.dft_norm = f.dft_norm();
  return result;
}

measurement run_2d_multilevel(gpu::backend_mode mode, int pixels,
                              int measured_steps) {
  gpu::set_backend(mode);
  const double length = 8.0;
  const double resolution = pixels / length;
  const grid_volume gv = vol2d(length, length, resolution);
  structure s(gv, vacuum, pml(0.8));
  const realnum gamma_matrix[] = {
      0.030f, 0.000f, 0.000f,
      -0.030f, 0.020f, 0.000f,
      0.000f, -0.020f, 0.000f};
  const realnum initial_populations[] = {1.0f, 0.2f, 0.05f};
  const realnum alpha[] = {
      -0.37894034f, 0.0f,
      0.37894034f, -0.28420526f,
      0.0f, 0.28420526f};
  const realnum frequencies[] = {0.42f, 0.56f};
  const realnum linewidths[] = {0.04f, 0.05f};
  const realnum transition_sigma[] = {
      0.18f, 0.16f, 0.14f, 0.12f, 0.10f,
      0.15f, 0.13f, 0.11f, 0.09f, 0.07f};
  s.add_susceptibility(
      [](const vec &) { return 0.045; }, E_stuff,
      multilevel_susceptibility(
          3, 2, gamma_matrix, initial_populations, alpha, frequencies,
          linewidths, transition_sigma));
  fields f(&s, 0.0, 0.0, true, 128, 128);
  f.use_real_fields();
  continuous_src_time source(0.24);
  f.add_volume_source(
      Ex, source, volume(vec(0.5 * length, 1.0),
                         vec(0.5 * length, length - 1.0)), 1.0);
  f.add_volume_source(
      Ey, source, volume(vec(1.0, 0.5 * length),
                         vec(length - 1.0, 0.5 * length)), -0.31);
  component components[] = {Ex, Ey, Hz};
  dft_fields monitor =
      f.add_dft_fields(components, 3, f.v, 0.18, 0.34, 4);
  (void)monitor;

  for (int step = 0; step < 12; ++step)
    f.step();
  gpu::reset_dispatch_statistics();
  const auto start = std::chrono::steady_clock::now();
  for (int step = 0; step < measured_steps; ++step)
    f.step();
  const auto stop = std::chrono::steady_clock::now();

  measurement result;
  result.seconds =
      std::chrono::duration<double>(stop - start).count();
  result.dispatch = gpu::get_dispatch_statistics();
  result.finite_check =
      gpu::detail::get_finite_check_transfer_statistics();
  result.fields = gpu::get_field_update_statistics();
  result.polarizations = gpu::get_polarization_statistics();
  result.sources = gpu::get_source_statistics();
  result.boundaries = gpu::get_boundary_statistics();
  result.dfts = gpu::get_dft_statistics();
  result.field_sample = f.get_field(Ex, vec(4.37, 3.61));
  result.dft_norm = f.dft_norm();
  return result;
}

measurement run_3d(gpu::backend_mode mode, int pixels, int measured_steps) {
  gpu::set_backend(mode);
  const double length = 4.0;
  const double resolution = pixels / length;
  const grid_volume gv = vol3d(length, length, length, resolution);
  structure s(gv, vacuum, pml(0.5));
  fields f(&s, 0.0, 0.0, true, 128, 128);
  f.use_real_fields();
  continuous_src_time source(0.29);
  f.add_point_source(Ex, source, gv.center(), 1.0);
  component components[] = {Ex};
  dft_fields monitor =
      f.add_dft_fields(components, 1, f.v, 0.24, 0.24, 1);
  (void)monitor;

  for (int step = 0; step < 8; ++step)
    f.step();
  gpu::reset_dispatch_statistics();
  const auto start = std::chrono::steady_clock::now();
  for (int step = 0; step < measured_steps; ++step)
    f.step();
  const auto stop = std::chrono::steady_clock::now();

  measurement result;
  result.seconds = std::chrono::duration<double>(stop - start).count();
  result.dispatch = gpu::get_dispatch_statistics();
  result.finite_check =
      gpu::detail::get_finite_check_transfer_statistics();
  result.fields = gpu::get_field_update_statistics();
  result.polarizations = gpu::get_polarization_statistics();
  result.sources = gpu::get_source_statistics();
  result.boundaries = gpu::get_boundary_statistics();
  result.dfts = gpu::get_dft_statistics();
  result.field_sample = f.get_field(Ex, vec(2.19, 1.83, 2.31));
  result.dft_norm = f.dft_norm();
  return result;
}

void require_close(const measurement &cpu, const measurement &cuda,
                   const std::string &name) {
  const double field_error =
      std::abs(cpu.field_sample - cuda.field_sample);
  const double field_scale =
      std::max(std::abs(cpu.field_sample), std::abs(cuda.field_sample));
  if (field_error > 5e-6 + 5e-4 * field_scale)
    throw std::runtime_error(name + ": CPU/CUDA field mismatch");
  const double dft_error = std::abs(cpu.dft_norm - cuda.dft_norm);
  const double dft_scale = std::max(std::abs(cpu.dft_norm),
                                    std::abs(cuda.dft_norm));
  if (dft_error > 2e-5 + 8e-4 * dft_scale)
    throw std::runtime_error(name + ": CPU/CUDA DFT mismatch");
}

std::uint64_t global_transfer_sum(std::uint64_t value);

void require_cuda_coverage(const measurement &result, bool dispersive,
                           const std::string &name) {
  const std::uint64_t cuda_curl =
      global_transfer_sum(result.dispatch.cuda_curl_calls);
  const std::uint64_t cpu_curl =
      global_transfer_sum(result.dispatch.cpu_curl_calls);
  const std::uint64_t cuda_eh =
      global_transfer_sum(result.fields.cuda_update_eh_calls);
  const std::uint64_t cpu_eh =
      global_transfer_sum(result.fields.cpu_update_eh_calls);
  const std::uint64_t cuda_source =
      global_transfer_sum(result.sources.cuda_update_calls);
  const std::uint64_t cpu_source =
      global_transfer_sum(result.sources.cpu_update_calls);
  const std::uint64_t cuda_boundary =
      global_transfer_sum(result.boundaries.cuda_update_calls);
  const std::uint64_t cpu_boundary =
      global_transfer_sum(result.boundaries.cpu_update_calls);
  const std::uint64_t cuda_dft =
      global_transfer_sum(result.dfts.cuda_update_calls);
  const std::uint64_t cpu_dft =
      global_transfer_sum(result.dfts.cpu_update_calls);
  if (cuda_curl == 0 || cpu_curl != 0 ||
      cuda_eh == 0 || cpu_eh != 0 ||
      cuda_source == 0 || cpu_source != 0 ||
      cuda_boundary == 0 || cpu_boundary != 0 ||
      cuda_dft == 0 || cpu_dft != 0) {
    std::ostringstream message;
    message << name << ": incomplete CUDA dispatch coverage"
            << " curl(cpu=" << cpu_curl
            << ",cuda=" << cuda_curl << ')'
            << " eh(cpu=" << cpu_eh
            << ",cuda=" << cuda_eh << ')'
            << " source(cpu=" << cpu_source
            << ",cuda=" << cuda_source << ')'
            << " boundary(cpu=" << cpu_boundary
            << ",cuda=" << cuda_boundary << ')'
            << " dft(cpu=" << cpu_dft
            << ",cuda=" << cuda_dft << ')';
    throw std::runtime_error(message.str());
  }
  if (dispersive) {
    const std::uint64_t cuda_polarization =
        global_transfer_sum(result.polarizations.cuda_update_calls);
    const std::uint64_t cpu_polarization =
        global_transfer_sum(result.polarizations.cpu_update_calls);
    if (cuda_polarization == 0 || cpu_polarization != 0)
      throw std::runtime_error(
          name + ": incomplete CUDA polarization coverage");
  }
}

std::uint64_t global_transfer_sum(std::uint64_t value) {
  if (value >
      static_cast<std::uint64_t>(
          std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error(
        "GPU transfer statistic does not fit MPI size_t");
  return static_cast<std::uint64_t>(
      sum_to_all(static_cast<std::size_t>(value)));
}

void require_steady_resident_transfers(
    const measurement &result, int measured_steps,
    const std::string &name) {
  const std::uint64_t global_h2d =
      global_transfer_sum(result.dispatch.host_to_device_bytes);
  const std::uint64_t global_d2h =
      global_transfer_sum(result.dispatch.device_to_host_bytes);
  const std::uint64_t global_finite_h2d =
      global_transfer_sum(
          result.finite_check.host_to_device_bytes);
  const std::uint64_t global_finite_d2h =
      global_transfer_sum(
          result.finite_check.device_to_host_bytes);
  const std::uint64_t expected_d2h =
      static_cast<std::uint64_t>(sizeof(int)) *
      static_cast<std::uint64_t>(measured_steps) *
      static_cast<std::uint64_t>(count_processors());
  const std::uint64_t expected_local_d2h =
      static_cast<std::uint64_t>(sizeof(int)) *
      static_cast<std::uint64_t>(measured_steps);
  const bool every_rank_has_finite_work =
      and_to_all(
          result.finite_check.host_to_device_bytes == 0 &&
          result.finite_check.device_to_host_bytes ==
              expected_local_d2h);
  const bool finite_exact =
      every_rank_has_finite_work &&
      global_finite_h2d == 0 &&
      global_finite_d2h == expected_d2h;
  const bool single_rank_total_exact =
      count_processors() != 1 ||
      (global_h2d == 0 && global_d2h == expected_d2h);
  if (!finite_exact || !single_rank_total_exact) {
    std::ostringstream message;
    message << name
            << ": steady resident transfer mismatch after warmup"
            << " (global H2D=" << global_h2d
            << ", global D2H=" << global_d2h
            << ", finite H2D=" << global_finite_h2d
            << ", finite D2H=" << global_finite_d2h
            << ", expected finite H2D=0, expected finite D2H="
            << expected_d2h << ')';
    throw std::runtime_error(message.str());
  }
}

void print_row(const std::string &name, int linear_pixels,
               std::uint64_t cells, int steps, const measurement &cpu,
               const measurement &cuda, bool production) {
  const double speedup = cpu.seconds / cuda.seconds;
  const double transfer_bytes_per_cell_step =
      (static_cast<double>(cuda.dispatch.host_to_device_bytes) +
       static_cast<double>(cuda.dispatch.device_to_host_bytes)) /
      (static_cast<double>(cells) * steps);
  std::cout << name << ',' << linear_pixels << ',' << cells << ',' << steps
            << ',' << std::setprecision(10) << cpu.seconds << ','
            << cuda.seconds << ',' << speedup << ','
            << cuda.dispatch.host_to_device_bytes << ','
            << cuda.dispatch.device_to_host_bytes << ','
            << transfer_bytes_per_cell_step << ','
            << cuda.dispatch.cuda_curl_points << ','
            << cuda.fields.cuda_update_eh_points << ','
            << cuda.polarizations.cuda_update_points << ','
            << cuda.sources.cuda_update_points << ','
            << cuda.boundaries.cuda_update_points << ','
            << cuda.dfts.cuda_update_points << ','
            << (production ? 1 : 0) << '\n';
}

bool passes_production_gate(const measurement &cpu,
                            const measurement &cuda,
                            std::uint64_t cells, int steps,
                            double minimum_speedup,
                            double maximum_transfer_bytes_per_cell_step) {
  const double speedup = cpu.seconds / cuda.seconds;
  const double transfer_bytes_per_cell_step =
      (static_cast<double>(cuda.dispatch.host_to_device_bytes) +
       static_cast<double>(cuda.dispatch.device_to_host_bytes)) /
      (static_cast<double>(cells) * steps);
  return speedup >= minimum_speedup &&
         transfer_bytes_per_cell_step <=
             maximum_transfer_bytes_per_cell_step;
}

} // namespace

int main(int argc, char **argv) {
  initialize mpi(argc, argv);
  try {
    std::string diagnostic;
    if (!gpu::runtime_available(&diagnostic)) {
      std::cout << "SKIP: " << diagnostic << '\n';
      return 77;
    }

    const bool quick = std::getenv("MEEP_GPU_BENCH_QUICK") != nullptr;
    const int steps_2d =
        environment_int("MEEP_GPU_BENCH_2D_STEPS", quick ? 30 : 120);
    const int steps_3d =
        environment_int("MEEP_GPU_BENCH_3D_STEPS", quick ? 20 : 60);
    const double minimum_speedup =
        environment_double("MEEP_GPU_MIN_SPEEDUP", 2.0);
    const double maximum_transfer_bytes_per_cell_step =
        environment_double("MEEP_GPU_MAX_TRANSFER_BYTES_PER_CELL_STEP",
                           64.0);
    const std::vector<int> sizes_2d =
        quick ? std::vector<int>{64, 128}
              : std::vector<int>{64, 128, 256, 512};
    const std::vector<int> sizes_3d =
        quick ? std::vector<int>{24, 32}
              : std::vector<int>{24, 32, 48, 64};

    std::cout
        << "case,linear_pixels,cells,steps,cpu_seconds,gpu_seconds,speedup,"
           "h2d_bytes,d2h_bytes,transfer_bytes_per_cell_step,"
           "cuda_curl_points,cuda_eh_points,"
           "cuda_polarization_points,cuda_source_points,"
           "cuda_boundary_points,cuda_dft_points,production_gate\n";

    bool gate_passed = true;
    for (int pixels : sizes_2d) {
      const bool production = pixels == sizes_2d.back();
      const measurement cpu =
          run_2d(gpu::backend_mode::cpu, pixels, steps_2d, false);
      const measurement cuda =
          run_2d(gpu::backend_mode::cuda, pixels, steps_2d, false);
      require_close(cpu, cuda, "2d-vacuum-dft");
      require_cuda_coverage(cuda, false, "2d-vacuum-dft");
      require_steady_resident_transfers(
          cuda, steps_2d, "2d-vacuum-dft");
      print_row("2d-vacuum-dft", pixels,
                static_cast<std::uint64_t>(pixels) * pixels, steps_2d, cpu,
                cuda, production);
      if (production &&
          !passes_production_gate(
              cpu, cuda,
              static_cast<std::uint64_t>(pixels) * pixels, steps_2d,
              minimum_speedup,
              maximum_transfer_bytes_per_cell_step))
        gate_passed = false;
    }
    for (int pixels : sizes_2d) {
      const bool production = pixels == sizes_2d.back();
      const measurement cpu =
          run_2d(gpu::backend_mode::cpu, pixels, steps_2d, true);
      const measurement cuda =
          run_2d(gpu::backend_mode::cuda, pixels, steps_2d, true);
      require_close(cpu, cuda, "2d-lorentz-dft");
      require_cuda_coverage(cuda, true, "2d-lorentz-dft");
      require_steady_resident_transfers(
          cuda, steps_2d, "2d-lorentz-dft");
      print_row("2d-lorentz-dft", pixels,
                static_cast<std::uint64_t>(pixels) * pixels, steps_2d, cpu,
                cuda, production);
      if (production &&
          !passes_production_gate(
              cpu, cuda,
              static_cast<std::uint64_t>(pixels) * pixels, steps_2d,
              minimum_speedup,
              maximum_transfer_bytes_per_cell_step))
        gate_passed = false;
    }
    for (int pixels : sizes_2d) {
      const bool production = pixels == sizes_2d.back();
      const measurement cpu = run_2d_gyrotropic(
          gpu::backend_mode::cpu, pixels, steps_2d);
      const measurement cuda = run_2d_gyrotropic(
          gpu::backend_mode::cuda, pixels, steps_2d);
      require_close(cpu, cuda, "2d-gyrotropic-llg-dft");
      require_cuda_coverage(cuda, true, "2d-gyrotropic-llg-dft");
      require_steady_resident_transfers(
          cuda, steps_2d, "2d-gyrotropic-llg-dft");
      print_row(
          "2d-gyrotropic-llg-dft", pixels,
          static_cast<std::uint64_t>(pixels) * pixels, steps_2d, cpu,
          cuda, production);
      if (production &&
          !passes_production_gate(
              cpu, cuda,
              static_cast<std::uint64_t>(pixels) * pixels, steps_2d,
              minimum_speedup,
              maximum_transfer_bytes_per_cell_step))
        gate_passed = false;
    }
    for (int pixels : sizes_2d) {
      const bool production = pixels == sizes_2d.back();
      const measurement cpu = run_2d_multilevel(
          gpu::backend_mode::cpu, pixels, steps_2d);
      const measurement cuda = run_2d_multilevel(
          gpu::backend_mode::cuda, pixels, steps_2d);
      require_close(cpu, cuda, "2d-multilevel-dft");
      require_cuda_coverage(cuda, true, "2d-multilevel-dft");
      require_steady_resident_transfers(
          cuda, steps_2d, "2d-multilevel-dft");
      print_row(
          "2d-multilevel-dft", pixels,
          static_cast<std::uint64_t>(pixels) * pixels, steps_2d, cpu,
          cuda, production);
      if (production &&
          !passes_production_gate(
              cpu, cuda,
              static_cast<std::uint64_t>(pixels) * pixels, steps_2d,
              minimum_speedup,
              maximum_transfer_bytes_per_cell_step))
        gate_passed = false;
    }
    for (int pixels : sizes_3d) {
      const bool production = pixels == sizes_3d.back();
      const measurement cpu =
          run_3d(gpu::backend_mode::cpu, pixels, steps_3d);
      const measurement cuda =
          run_3d(gpu::backend_mode::cuda, pixels, steps_3d);
      require_close(cpu, cuda, "3d-vacuum-dft");
      require_cuda_coverage(cuda, false, "3d-vacuum-dft");
      require_steady_resident_transfers(
          cuda, steps_3d, "3d-vacuum-dft");
      print_row("3d-vacuum-dft", pixels,
                static_cast<std::uint64_t>(pixels) * pixels * pixels,
                steps_3d, cpu, cuda, production);
      if (production &&
          !passes_production_gate(
              cpu, cuda,
              static_cast<std::uint64_t>(pixels) * pixels * pixels,
              steps_3d, minimum_speedup,
              maximum_transfer_bytes_per_cell_step))
        gate_passed = false;
    }

    if (!gate_passed) {
      std::cerr << "FAIL: a production-scale case is below "
                << minimum_speedup
                << "x speedup or exceeds "
                << maximum_transfer_bytes_per_cell_step
                << " transfer bytes per cell-step\n";
      return 1;
    }
    return 0;
  }
  catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
