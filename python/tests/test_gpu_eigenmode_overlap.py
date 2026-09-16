import os
import unittest

import numpy as np

import meep as mp


class TestGpuEigenmodeOverlap(unittest.TestCase):
    def require_cuda(self):
        reason = None
        if not mp.gpu.compiled or not mp.gpu.runtime_available:
            reason = mp.gpu.runtime_diagnostic or "CUDA is unavailable"
        else:
            compatible = [
                device for device in mp.gpu.devices() if device["compatible"]
            ]
            if not compatible:
                reason = "no compatible CUDA device"
        if reason and os.environ.get("GPMEEP_REQUIRE_CUDA_TEST") == "1":
            self.fail(reason)
        if reason:
            self.skipTest(reason)

    def test_multiband_repeat_uses_resident_cuda_overlap(self):
        requested_backend = mp.gpu.requested_backend
        if requested_backend != "cpu":
            self.require_cuda()
        frequency = 2.0
        index = 1.5
        eig_volume = mp.Volume(
            center=mp.Vector3(), size=mp.Vector3(0, 1e-7)
        )

        def make_simulation():
            simulation = mp.Simulation(
                resolution=10,
                cell_size=mp.Vector3(2, 1),
                k_point=mp.Vector3(index * frequency),
                default_material=mp.Medium(index=index),
                sources=[
                    mp.Source(
                        mp.GaussianSource(
                            frequency, fwidth=0.1 * frequency
                        ),
                        component=mp.Ez,
                        center=mp.Vector3(),
                    )
                ],
            )
            monitors = [
                simulation.add_mode_monitor(
                    frequency,
                    0,
                    1,
                    mp.FluxRegion(
                        center=eig_volume.center, size=eig_volume.size
                    ),
                )
                for _ in range(2)
            ]
            return simulation, monitors

        def solve(simulation, monitor):
            return simulation.get_eigenmode_coefficients(
                monitor,
                bands=[1, 2],
                kpoint_func=lambda *unused: mp.Vector3(
                    index * frequency
                ),
                direction=mp.X,
                eig_vol=eig_volume,
                eig_tolerance=1e-12,
            )

        if requested_backend == "cpu":
            simulation, monitors = make_simulation()
            simulation.run(until=1)
            mp.gpu.reset_statistics()
            first = solve(simulation, monitors[0])
            second_monitor = solve(simulation, monitors[1])
            second = solve(simulation, monitors[0])
            np.testing.assert_allclose(
                first.alpha, second.alpha, rtol=0, atol=1e-7
            )
            np.testing.assert_allclose(
                first.alpha, second_monitor.alpha, rtol=0, atol=1e-7
            )
            statistics = mp.gpu.statistics()["eigenmode_overlaps"]
            self.assertEqual(statistics["cpu_eigenmode_overlap_calls"], 12)
            self.assertGreater(statistics["cpu_eigenmode_overlap_terms"], 0)
            self.assertEqual(statistics["cuda_eigenmode_overlap_calls"], 0)
            self.assertGreater(
                statistics["eigenmode_zero_rank_channels_skipped"], 0
            )
            return

        if mp.count_processors() == 1:
            compatible = [
                device for device in mp.gpu.devices() if device["compatible"]
            ]
            mp.gpu.select_device(compatible[0]["ordinal"])
        mp.gpu.set_backend("cuda")
        baseline_live_buffers = mp.gpu.statistics()["resident"][
            "live_resident_device_buffers"
        ]
        simulation, monitors = make_simulation()
        simulation.run(until=1)
        mp.gpu.reset_statistics()
        first = solve(simulation, monitors[0])
        second_monitor = solve(simulation, monitors[1])
        second = solve(simulation, monitors[0])
        np.testing.assert_allclose(first.alpha, second.alpha, rtol=0, atol=1e-7)
        np.testing.assert_allclose(
            first.alpha, second_monitor.alpha, rtol=0, atol=1e-7
        )
        np.testing.assert_allclose(first.vgrp, second.vgrp, rtol=0, atol=1e-7)
        np.testing.assert_allclose(first.cscale, second.cscale, rtol=0, atol=1e-7)

        statistics = mp.gpu.statistics()["eigenmode_overlaps"]
        self.assertEqual(statistics["cpu_eigenmode_overlap_calls"], 0)
        self.assertEqual(statistics["cpu_eigenmode_overlap_terms"], 0)
        self.assertEqual(statistics["cuda_eigenmode_overlap_calls"], 12)
        self.assertEqual(statistics["cuda_eigenmode_mode_flux_calls"], 6)
        self.assertEqual(statistics["cuda_eigenmode_mode_mode_calls"], 6)
        self.assertEqual(statistics["cuda_eigenmode_kernel_launches"], 12)
        self.assertEqual(
            statistics["cuda_eigenmode_result_device_to_host_bytes"],
            12 * 4 * np.dtype(np.complex128).itemsize,
        )
        # Each fused solve uses one batched backend/extent consensus and one
        # eight-complex result reduction, independent of component count.
        self.assertEqual(statistics["eigenmode_mpi_allreduce_calls"], 12)
        self.assertEqual(
            statistics["eigenmode_mpi_allreduce_bytes"],
            6
            * (
                43 * np.dtype(np.intc).itemsize
                + 8 * np.dtype(np.complex128).itemsize
            ),
        )
        self.assertEqual(statistics["cuda_eigenmode_descriptor_uploads"], 2)
        self.assertEqual(statistics["cuda_eigenmode_plan_reuses"], 4)
        self.assertGreater(statistics["cuda_eigenmode_submitted_pairs"], 0)
        self.assertGreater(statistics["cuda_eigenmode_overlap_terms"], 0)
        self.assertEqual(statistics["host_mode_profile_sampling_calls"], 6)
        self.assertGreater(
            statistics["host_mode_profile_sampling_points"], 0
        )
        self.assertGreater(
            statistics["eigenmode_zero_rank_channels_skipped"], 0
        )
        self.assertGreater(
            statistics["host_mode_profile_host_to_device_bytes"], 0
        )
        self.assertGreater(
            statistics["eigenmode_full_dft_device_to_host_bytes_avoided"],
            0,
        )
        self.assertGreater(
            mp.gpu.statistics()["resident"]["live_resident_device_buffers"],
            baseline_live_buffers,
        )
        simulation.reset_meep()
        self.assertEqual(
            mp.gpu.statistics()["resident"]["live_resident_device_buffers"],
            baseline_live_buffers,
        )

    def test_injected_failures_retry_without_leaking_resident_plans(self):
        if mp.gpu.requested_backend == "cpu":
            self.skipTest("CUDA failure-atomicity test")
        self.require_cuda()
        if mp.count_processors() == 1:
            compatible = [
                device for device in mp.gpu.devices() if device["compatible"]
            ]
            mp.gpu.select_device(compatible[0]["ordinal"])
        mp.gpu.set_backend("cuda")
        baseline_live_buffers = mp.gpu.statistics()["resident"][
            "live_resident_device_buffers"
        ]
        frequency = 2.0
        index = 1.5
        eig_volume = mp.Volume(
            center=mp.Vector3(), size=mp.Vector3(0, 1e-7)
        )
        simulation = mp.Simulation(
            resolution=10,
            cell_size=mp.Vector3(2, 1),
            k_point=mp.Vector3(index * frequency),
            default_material=mp.Medium(index=index),
            sources=[
                mp.Source(
                    mp.GaussianSource(frequency, fwidth=0.1 * frequency),
                    component=mp.Ez,
                    center=mp.Vector3(),
                )
            ],
        )
        monitors = [
            simulation.add_mode_monitor(
                frequency,
                0,
                1,
                mp.FluxRegion(
                    center=eig_volume.center, size=eig_volume.size
                ),
            )
            for _ in range(4)
        ]
        simulation.run(until=1)
        mp.gpu.reset_statistics()

        def solve(monitor):
            return simulation.get_eigenmode_coefficients(
                monitor,
                bands=[1],
                kpoint_func=lambda *unused: mp.Vector3(index * frequency),
                direction=mp.X,
                eig_vol=eig_volume,
                eig_tolerance=1e-12,
            )

        reference = solve(monitors[0])
        reference_statistics = mp.gpu.statistics()["eigenmode_overlaps"]
        self.assertEqual(
            reference_statistics["cuda_eigenmode_descriptor_uploads"], 1
        )
        self.assertEqual(
            reference_statistics["cuda_eigenmode_plan_reuses"], 0
        )
        self.assertEqual(
            reference_statistics["cuda_eigenmode_kernel_launches"], 2
        )
        self.assertEqual(
            reference_statistics[
                "cuda_eigenmode_result_device_to_host_bytes"
            ],
            8 * np.dtype(np.complex128).itemsize,
        )
        self.assertEqual(
            reference_statistics["host_mode_profile_sampling_calls"], 1
        )
        reference_sampling_points = reference_statistics[
            "host_mode_profile_sampling_points"
        ]
        reference_profile_bytes = reference_statistics[
            "host_mode_profile_host_to_device_bytes"
        ]
        reference_avoided_bytes = reference_statistics[
            "eigenmode_full_dft_device_to_host_bytes_avoided"
        ]
        self.assertGreater(reference_sampling_points, 0)
        self.assertGreater(reference_profile_bytes, 0)
        self.assertGreater(reference_avoided_bytes, 0)
        try:
            for stage, monitor in zip(
                ("profile_h2d", "kernel", "d2h"), monitors[1:]
            ):
                os.environ["MEEP_GPU_TEST_EIGENMODE_FAIL_STAGE"] = stage
                with self.assertRaisesRegex(
                    RuntimeError, "injected CUDA eigenmode"
                ):
                    solve(monitor)
                os.environ.pop("MEEP_GPU_TEST_EIGENMODE_FAIL_STAGE", None)
                recovered = solve(monitor)
                np.testing.assert_allclose(
                    reference.alpha, recovered.alpha, rtol=0, atol=1e-7
                )
        finally:
            os.environ.pop("MEEP_GPU_TEST_EIGENMODE_FAIL_STAGE", None)
            simulation.reset_meep()
        statistics = mp.gpu.statistics()["eigenmode_overlaps"]
        # Four solves publish scientific results.  The three failed attempts
        # still contribute every physical upload, launch, and result copy
        # completed before their injected failure boundary.
        self.assertEqual(statistics["cuda_eigenmode_overlap_calls"], 8)
        self.assertEqual(statistics["cuda_eigenmode_mode_flux_calls"], 4)
        self.assertEqual(statistics["cuda_eigenmode_mode_mode_calls"], 4)
        self.assertEqual(
            statistics["cuda_eigenmode_descriptor_uploads"], 5
        )
        self.assertEqual(statistics["cuda_eigenmode_plan_reuses"], 2)
        self.assertEqual(statistics["cuda_eigenmode_kernel_launches"], 10)
        self.assertEqual(
            statistics["cuda_eigenmode_result_device_to_host_bytes"],
            5 * 8 * np.dtype(np.complex128).itemsize,
        )
        self.assertEqual(statistics["host_mode_profile_sampling_calls"], 7)
        self.assertEqual(
            statistics["host_mode_profile_sampling_points"],
            7 * reference_sampling_points,
        )
        self.assertEqual(
            statistics["host_mode_profile_host_to_device_bytes"],
            7 * reference_profile_bytes,
        )
        self.assertEqual(
            statistics[
                "eigenmode_full_dft_device_to_host_bytes_avoided"
            ],
            7 * reference_avoided_bytes,
        )
        self.assertEqual(
            mp.gpu.statistics()["resident"]["live_resident_device_buffers"],
            baseline_live_buffers,
        )


if __name__ == "__main__":
    unittest.main()
