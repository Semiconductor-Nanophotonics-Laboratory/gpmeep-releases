import contextlib
import os
import unittest

import meep as mp
import meep._meep as _meep
import numpy as np


@contextlib.contextmanager
def environment_override(**changes):
    saved = {name: os.environ.get(name) for name in changes}
    try:
        for name, value in changes.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class TestGpuBackend(unittest.TestCase):
    def tearDown(self):
        mp.gpu.set_backend("cpu")

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

    def test_backend_control_and_inventory(self):
        mp.gpu.set_backend("cpu")
        self.assertEqual(mp.gpu.requested_backend, "cpu")
        self.assertEqual(mp.gpu.active_backend, "cpu")
        self.assertIsInstance(mp.gpu.compiled_architectures, str)
        self.assertIsInstance(mp.gpu.runtime_diagnostic, str)

        with self.assertRaises(ValueError):
            mp.gpu.set_backend("not-a-backend")

        devices = mp.gpu.devices()
        self.assertIsInstance(devices, list)
        if not mp.gpu.runtime_available:
            self.assertEqual(devices, [])
        for device in devices:
            self.assertEqual(
                {
                    "ordinal",
                    "compute_capability",
                    "multiprocessor_count",
                    "max_threads_per_block",
                    "compatible",
                    "global_memory_bytes",
                    "memory_bandwidth_bytes_per_second",
                    "identifier",
                    "name",
                },
                set(device),
            )

    def test_statistics_have_stable_groups(self):
        mp.gpu.reset_statistics()
        statistics = mp.gpu.statistics()
        self.assertEqual(
            {
                "runtime",
                "dispatch",
                "resident",
                "field_updates",
                "polarizations",
                "sources",
                "boundaries",
                "dfts",
                "dft_batches",
                "dft_reductions",
                "dft_materializations",
                "dft_checkpoints",
                "dft_scales",
                "eigenmode_overlaps",
                "ldos",
                "near2far",
                "multi_gpu",
                "mpi_completion",
                "boundary_eh_overlap",
                "halo_curl_overlap",
                "tile_coalescing",
                "phase_batch_policy",
            },
            set(statistics),
        )
        self.assertEqual(
            {
                "runtime_availability_probes",
                "runtime_device_enumerations",
                "runtime_device_selections",
            },
            set(statistics["runtime"]),
        )
        self.assertEqual(
            {
                "dft_batch_calls",
                "dft_submitted_updates",
                "dft_phase_preparation_launches",
                "dft_phase_reuses",
                "dft_update_kernel_launches",
                "dft_maximum_batch_size",
                "dft_multi_monitor_automatic_checks",
                "dft_multi_monitor_automatic_selected",
                "dft_multi_monitor_automatic_rejected",
                "dft_multi_monitor_forced_batches",
                "dft_multi_monitor_batched_updates",
                "dft_multi_monitor_unbatched_updates",
                "dft_multi_monitor_plan_uploads",
                "dft_multi_monitor_plan_reuses",
                "dft_multi_monitor_metadata_host_to_device_bytes",
            },
            set(statistics["dft_batches"]),
        )
        self.assertEqual(
            {
                "cpu_dft_reduction_calls",
                "cpu_dft_reduction_pairs",
                "cpu_dft_reduction_terms",
                "cuda_dft_reduction_calls",
                "cuda_dft_reduction_pairs",
                "cuda_dft_reduction_terms",
                "cuda_dft_reduction_descriptor_uploads",
                "cuda_dft_reduction_plan_reuses",
                "cuda_dft_reduction_kernel_launches",
                "cuda_dft_reduction_result_device_to_host_bytes",
                "dft_reduction_full_dft_device_to_host_bytes_avoided",
                "dft_reduction_mpi_allreduce_calls",
                "dft_reduction_mpi_allreduce_bytes",
            },
            set(statistics["dft_reductions"]),
        )
        self.assertEqual(
            {
                "cpu_dft_array_materialization_calls",
                "cpu_dft_array_materialization_points",
                "cuda_dft_array_materialization_calls",
                "cuda_dft_array_materialization_points",
                "host_synthetic_material_array_calls",
                "host_synthetic_material_array_points",
                "cpu_dft_output_calls",
                "cpu_dft_output_points",
                "cuda_dft_output_calls",
                "cuda_dft_output_points",
                "cuda_dft_output_staging_calls",
                "cuda_dft_output_staging_points",
                "cuda_dft_output_staging_frequencies",
                "cuda_dft_output_staging_descriptor_uploads",
                "cuda_dft_output_staging_plan_reuses",
                "cuda_dft_output_staging_kernel_launches",
                "cuda_dft_output_staging_result_device_to_host_bytes",
                (
                    "cuda_dft_output_staging_full_dft_"
                    "device_to_host_bytes_avoided"
                ),
                "cuda_dft_output_staging_workspace_ceiling_bytes",
                "cuda_dft_materialization_kernel_launches",
                "cuda_dft_materialization_result_device_to_host_bytes",
                "dft_materialization_full_dft_device_to_host_bytes_avoided",
                "dft_array_mpi_allreduce_calls",
                "dft_array_mpi_allreduce_bytes",
            },
            set(statistics["dft_materializations"]),
        )
        self.assertEqual(
            {
                "cpu_dft_scale_calls",
                "cpu_dft_scale_values",
                "cuda_dft_scale_calls",
                "cuda_dft_scale_values",
                "cuda_dft_scale_kernel_launches",
                "cuda_dft_scale_host_to_device_bytes",
            },
            set(statistics["dft_scales"]),
        )
        self.assertEqual(
            {
                "cpu_eigenmode_overlap_calls",
                "cpu_eigenmode_overlap_terms",
                "cuda_eigenmode_overlap_calls",
                "cuda_eigenmode_overlap_terms",
                "cuda_eigenmode_mode_flux_calls",
                "cuda_eigenmode_mode_mode_calls",
                "cuda_eigenmode_submitted_pairs",
                "cuda_eigenmode_descriptor_uploads",
                "cuda_eigenmode_plan_reuses",
                "cuda_eigenmode_kernel_launches",
                "cuda_eigenmode_result_device_to_host_bytes",
                "eigenmode_full_dft_device_to_host_bytes_avoided",
                "host_mode_profile_sampling_calls",
                "host_mode_profile_sampling_points",
                "eigenmode_zero_rank_channels_skipped",
                "host_mode_profile_host_to_device_bytes",
                "eigenmode_mpi_allreduce_calls",
                "eigenmode_mpi_allreduce_bytes",
            },
            set(statistics["eigenmode_overlaps"]),
        )
        self.assertEqual(
            0,
            statistics["dft_materializations"][
                "host_synthetic_material_array_calls"
            ],
        )
        self.assertEqual(
            0,
            statistics["dft_materializations"][
                "host_synthetic_material_array_points"
            ],
        )
        self.assertEqual(
            {"mpi_waitsome_executions", "mpi_waitall_executions"},
            set(statistics["mpi_completion"]),
        )
        self.assertEqual(
            {
                "cpu_ldos_reduction_calls",
                "cpu_ldos_source_points",
                "cuda_ldos_reduction_calls",
                "cuda_ldos_submitted_profiles",
                "cuda_ldos_source_points",
                "cuda_ldos_descriptor_uploads",
                "cuda_ldos_kernel_launches",
                "cuda_ldos_result_device_to_host_bytes",
                "ldos_full_field_device_to_host_bytes_avoided",
            },
            set(statistics["ldos"]),
        )
        self.assertEqual(
            {
                "cpu_near2far_transform_calls",
                "cpu_near2far_terms",
                "cuda_near2far_transform_calls",
                "cuda_near2far_terms",
                "cuda_near2far_submitted_chunks",
                "cuda_near2far_source_points",
                "cuda_near2far_output_points",
                "cuda_near2far_frequencies",
                "cuda_near2far_periodic_copies",
                "cuda_near2far_fast_precision_calls",
                "cuda_near2far_mixed_precision_calls",
                "cuda_near2far_cancellation_retries",
                "cuda_near2far_target_tiles",
                "cuda_near2far_frequency_tiles",
                "cuda_near2far_operation_tiles",
                "cuda_near2far_maximum_workspace_bytes",
                "cuda_near2far_descriptor_uploads",
                "cuda_near2far_kernel_launches",
                "cuda_near2far_result_device_to_host_bytes",
                "cuda_near2far_condition_device_to_host_bytes",
                "near2far_dft_device_to_host_bytes_avoided",
                "near2far_mpi_allreduce_calls",
                "near2far_mpi_allreduce_bytes",
                "cpu_near2far_adjoint_calls",
                "cpu_near2far_adjoint_terms",
                "cuda_near2far_adjoint_calls",
                "cuda_near2far_adjoint_terms",
                "cuda_near2far_adjoint_submitted_chunks",
                "cuda_near2far_adjoint_source_points",
                "cuda_near2far_adjoint_far_points",
                "cuda_near2far_adjoint_frequencies",
                "cuda_near2far_adjoint_periodic_copies",
                "cuda_near2far_adjoint_fast_precision_calls",
                "cuda_near2far_adjoint_mixed_precision_calls",
                "cuda_near2far_adjoint_cancellation_retries",
                "cuda_near2far_adjoint_maximum_workspace_bytes",
                "cuda_near2far_adjoint_descriptor_uploads",
                "cuda_near2far_adjoint_kernel_launches",
                "cuda_near2far_adjoint_host_to_device_bytes",
                "cuda_near2far_adjoint_result_device_to_host_bytes",
                "cuda_near2far_adjoint_condition_device_to_host_bytes",
            },
            set(statistics["near2far"]),
        )
        self.assertEqual(
            {
                "boundary_eh_overlap_checks",
                "boundary_eh_overlap_eligible",
                "boundary_eh_overlap_launched_h",
                "boundary_eh_overlap_launched_e",
                "boundary_eh_overlap_skipped_disabled",
                "boundary_eh_overlap_skipped_unsupported_schedule",
                "boundary_eh_overlap_skipped_no_remote",
                "boundary_eh_overlap_skipped_cold_topology",
                "boundary_eh_overlap_rejected",
            },
            set(statistics["boundary_eh_overlap"]),
        )
        self.assertEqual(
            {
                "halo_curl_overlap_checks",
                "halo_curl_overlap_eligible",
                "halo_curl_overlap_launches",
                "halo_curl_overlap_skipped_disabled",
                "halo_curl_overlap_skipped_unsupported_schedule",
                "halo_curl_overlap_skipped_no_remote",
                "halo_curl_overlap_skipped_cold_topology",
                "halo_curl_overlap_rejected_feature",
                "halo_curl_overlap_rejected_small",
                "halo_curl_overlap_full_points",
                "halo_curl_overlap_interior_points",
                "halo_curl_overlap_shell_points",
            },
            set(statistics["halo_curl_overlap"]),
        )
        self.assertEqual(
            {
                "tile_coalesced_curl_chunk_phases",
                "tile_coalesced_curl_input_tiles",
                "tile_coalesced_update_eh_chunk_phases",
                "tile_coalesced_update_eh_input_tiles",
            },
            set(statistics["tile_coalescing"]),
        )
        self.assertEqual(
            {
                "phase_curl_automatic_checks",
                "phase_curl_automatic_selected",
                "phase_curl_automatic_rejected",
                "phase_curl_forced_batches",
                "phase_curl_batched_operations",
                "phase_curl_unbatched_operations",
                "phase_curl_replay_checks",
                "phase_curl_replay_hits",
                "phase_curl_replay_unready",
                "phase_curl_replay_generation_misses",
                "phase_curl_replay_mirror_misses",
                "phase_update_eh_automatic_checks",
                "phase_update_eh_automatic_selected",
                "phase_update_eh_automatic_rejected",
                "phase_update_eh_forced_batches",
                "phase_update_eh_batched_operations",
                "phase_update_eh_unbatched_operations",
            },
            set(statistics["phase_batch_policy"]),
        )
        self.assertTrue(
            all(
                isinstance(value, int)
                for group in statistics.values()
                for value in group.values()
            )
        )

    def test_internal_mpi_overlap_state_is_not_python_api(self):
        internal_names = (
            "comms_overlap_statistics",
            "comms_supports_cuda_device_buffers",
            "comms_start_cuda_device_receives",
            "comms_start_cuda_device_sends",
            "reset_comms_overlap_statistics",
            "get_comms_overlap_statistics",
            "comms_physical_message_count",
            "comms_finish",
        )
        for name in internal_names:
            self.assertFalse(hasattr(_meep, name), name)

    def test_automatic_00_small_owner_does_not_touch_cuda_runtime(self):
        with environment_override(
            MEEP_GPU_DEVICE=None,
            MEEP_GPU_ALLOW_OVERSUBSCRIBE=None,
            MEEP_GPU_AUTO_MIN_CELLS=None,
        ):
            mp.gpu.reset_statistics()
            mp.gpu.set_backend("auto")
            simulation = mp.Simulation(
                cell_size=mp.Vector3(1.0, 0.8),
                resolution=8,
                sources=[
                    mp.Source(
                        mp.GaussianSource(0.3, fwidth=0.1),
                        component=mp.Ez,
                        center=mp.Vector3(),
                    )
                ],
            )
            simulation.run(until=0.2)

            statistics = mp.gpu.statistics()
            self.assertEqual(mp.gpu.requested_backend, "auto")
            self.assertEqual(mp.gpu.active_backend, "cpu")
            self.assertEqual(
                {
                    "runtime_availability_probes": 0,
                    "runtime_device_enumerations": 0,
                    "runtime_device_selections": 0,
                },
                statistics["runtime"],
            )
            self.assertGreater(statistics["dispatch"]["cpu_curl_calls"], 0)
            self.assertEqual(statistics["dispatch"]["cuda_curl_calls"], 0)
            self.assertFalse(simulation.fields.gpu_cuda_execution_selected())
            self.assertEqual(
                mp.gpu.backend_diagnostic,
                simulation.fields.gpu_execution_diagnostic(),
            )
            self.assertIn(
                "selected CPU", simulation.fields.gpu_execution_diagnostic()
            )

    def test_automatic_10_owner_selects_cuda(self):
        self.require_cuda()
        with environment_override(
            MEEP_GPU_DEVICE=None,
            MEEP_GPU_ALLOW_OVERSUBSCRIBE=None,
            MEEP_GPU_AUTO_MIN_CELLS="0",
        ):
            mp.gpu.reset_statistics()
            mp.gpu.set_backend("auto")
            simulation = mp.Simulation(
                cell_size=mp.Vector3(1.6, 1.4),
                resolution=10,
                boundary_layers=[mp.PML(0.2)],
                sources=[
                    mp.Source(
                        mp.GaussianSource(0.3, fwidth=0.1),
                        component=mp.Ez,
                        center=mp.Vector3(),
                    )
                ],
            )
            simulation.run(until=1)
            statistics = mp.gpu.statistics()
            self.assertEqual(mp.gpu.requested_backend, "auto")
            self.assertEqual(mp.gpu.active_backend, "cuda")
            self.assertGreater(
                statistics["runtime"]["runtime_availability_probes"], 0
            )
            self.assertGreater(
                statistics["runtime"]["runtime_device_enumerations"], 0
            )
            self.assertTrue(
                statistics["runtime"]["runtime_device_selections"] > 0
                or mp.gpu.selected_device >= 0,
                "automatic CUDA activation neither selected nor reused a "
                "valid device",
            )
            self.assertGreater(statistics["dispatch"]["cuda_curl_calls"], 0)
            self.assertEqual(statistics["dispatch"]["cpu_curl_calls"], 0)
            self.assertTrue(simulation.fields.gpu_cuda_execution_selected())
            self.assertEqual(
                mp.gpu.backend_diagnostic,
                simulation.fields.gpu_execution_diagnostic(),
            )

    def test_required_cuda_python_step(self):
        self.require_cuda()
        compatible = [device for device in mp.gpu.devices() if device["compatible"]]

        # In an MPI build the runtime's node-local-rank policy must choose a
        # distinct device for each rank.  Explicitly selecting ordinal zero on
        # every rank defeats that policy and correctly trips the physical-GPU
        # claim table.  The singleton test still covers the public selector;
        # dedicated MPI tests cover default and explicit per-rank mappings.
        if mp.count_processors() == 1:
            mp.gpu.select_device(compatible[0]["ordinal"])
        mp.gpu.set_backend("cuda")
        mp.gpu.reset_statistics()
        simulation = mp.Simulation(
            cell_size=mp.Vector3(1.6, 1.4),
            resolution=10,
            boundary_layers=[mp.PML(0.2)],
            sources=[
                mp.Source(
                    mp.ContinuousSource(0.3),
                    component=mp.Ez,
                    center=mp.Vector3(),
                )
            ],
        )
        simulation.run(until=2)

        statistics = mp.gpu.statistics()
        self.assertEqual(mp.gpu.active_backend, "cuda")
        self.assertGreater(statistics["dispatch"]["cuda_curl_calls"], 0)
        self.assertEqual(statistics["dispatch"]["cpu_curl_calls"], 0)
        self.assertGreater(statistics["field_updates"]["cuda_update_eh_calls"], 0)
        self.assertEqual(statistics["field_updates"]["cpu_update_eh_calls"], 0)
        # A point source is owned by one MPI rank, while every rank owns curl
        # work.  The distributed C++ suite aggregates and proves nonzero
        # source dispatch; this public-binding test must not demand the same
        # rank-local source count from a peer that owns no source point.
        if mp.count_processors() == 1:
            self.assertGreater(statistics["sources"]["cuda_source_calls"], 0)
        self.assertEqual(statistics["sources"]["cpu_source_calls"], 0)
        self.assertTrue(simulation.fields.gpu_cuda_execution_selected())
        self.assertIn("CUDA", simulation.fields.gpu_execution_diagnostic())

    def test_required_cuda_resident_dft_scale_statistics(self):
        self.require_cuda()
        compatible = [device for device in mp.gpu.devices() if device["compatible"]]
        if mp.count_processors() == 1:
            mp.gpu.select_device(compatible[0]["ordinal"])
        mp.gpu.set_backend("cuda")

        simulation = mp.Simulation(
            cell_size=mp.Vector3(1.6, 1.4),
            resolution=10,
            sources=[
                mp.Source(
                    mp.ContinuousSource(0.3),
                    component=mp.Ez,
                    center=mp.Vector3(),
                )
            ],
        )
        flux = simulation.add_flux(
            0.3,
            0.0,
            1,
            mp.FluxRegion(center=mp.Vector3(0.2), size=mp.Vector3(0, 0.8)),
            decimation_factor=1,
        )
        simulation.run(until=1)

        mp.gpu.reset_statistics()
        flux.scale_dfts(complex(-0.25, 0.125))
        statistics = mp.gpu.statistics()["dft_scales"]
        from mpi4py import MPI

        # The monitor plane can be owned entirely by one MPI rank.  Statistics
        # are deliberately process-local, so require zero fallback locally but
        # evaluate positive CUDA work over the distributed simulation instead
        # of manufacturing a no-op kernel on a rank with no DFT allocation.
        rank_statistics = MPI.COMM_WORLD.allgather(statistics)
        for rank_statistic in rank_statistics:
            self.assertEqual(rank_statistic["cpu_dft_scale_calls"], 0)
            self.assertEqual(rank_statistic["cpu_dft_scale_values"], 0)
            self.assertEqual(
                rank_statistic["cuda_dft_scale_host_to_device_bytes"], 0
            )
            if rank_statistic["cuda_dft_scale_calls"] == 0:
                self.assertEqual(rank_statistic["cuda_dft_scale_values"], 0)
                self.assertEqual(
                    rank_statistic["cuda_dft_scale_kernel_launches"], 0
                )
        global_statistics = {
            key: sum(rank_statistic[key] for rank_statistic in rank_statistics)
            for key in statistics
        }
        self.assertGreater(global_statistics["cuda_dft_scale_calls"], 0)
        self.assertGreater(global_statistics["cuda_dft_scale_values"], 0)
        self.assertGreater(global_statistics["cuda_dft_scale_kernel_launches"], 0)

    def test_required_cuda_centered_grid_unequal_dft_reduction(self):
        """Centered Yee monitors may store more H than E points."""
        self.require_cuda()
        from mpi4py import MPI

        def make_simulation():
            simulation = mp.Simulation(
                cell_size=mp.Vector3(2.0, 2.0),
                resolution=16,
                sources=[
                    mp.Source(
                        mp.ContinuousSource(0.3),
                        component=mp.Ez,
                        center=mp.Vector3(),
                    )
                ],
            )
            monitor = simulation.add_mode_monitor(
                [0.27, 0.33],
                mp.ModeRegion(
                    center=mp.Vector3(0.5, 0.0),
                    size=mp.Vector3(0.0, 1.2),
                ),
                yee_grid=True,
                decimation_factor=1,
            )
            return simulation, monitor

        mp.gpu.set_backend("cuda")
        mp.gpu.reset_statistics()
        simulation, monitor = make_simulation()
        simulation.run(until=2.0)

        electric = monitor.E
        magnetic = monitor.H
        local_unequal_pairs = 0
        local_pairs = 0
        while electric and magnetic:
            self.assertLessEqual(electric.N, magnetic.N)
            local_unequal_pairs += int(electric.N != magnetic.N)
            local_pairs += 1
            electric = electric.next_in_dft
            magnetic = magnetic.next_in_dft
        self.assertFalse(bool(electric) or bool(magnetic))
        self.assertGreater(
            MPI.COMM_WORLD.allreduce(local_pairs, op=MPI.SUM), 0
        )
        self.assertGreater(
            MPI.COMM_WORLD.allreduce(local_unequal_pairs, op=MPI.SUM), 0
        )

        values = np.asarray(mp.get_fluxes(monitor))
        statistics = mp.gpu.statistics()
        reductions = statistics["dft_reductions"]

        def global_sum(value):
            return MPI.COMM_WORLD.allreduce(value, op=MPI.SUM)

        self.assertEqual(values.shape, (2,))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertEqual(mp.gpu.active_backend, "cuda")
        self.assertTrue(simulation.fields.gpu_cuda_execution_selected())
        cuda_reduction_calls = global_sum(
            reductions["cuda_dft_reduction_calls"]
        )
        self.assertGreater(cuda_reduction_calls, 0)
        self.assertGreater(
            global_sum(reductions["cuda_dft_reduction_pairs"]), 0
        )
        self.assertGreater(
            global_sum(reductions["cuda_dft_reduction_terms"]), 0
        )
        self.assertEqual(
            global_sum(reductions["cuda_dft_reduction_kernel_launches"]),
            2 * cuda_reduction_calls,
        )
        self.assertGreater(
            global_sum(
                reductions[
                    "cuda_dft_reduction_result_device_to_host_bytes"
                ]
            ),
            0,
        )
        self.assertEqual(
            global_sum(reductions["cpu_dft_reduction_calls"]), 0
        )
        self.assertGreater(
            global_sum(statistics["dfts"]["cuda_dft_calls"]), 0
        )
        self.assertEqual(
            global_sum(statistics["dfts"]["cpu_dft_calls"]), 0
        )
        simulation.reset_meep()

        mp.gpu.set_backend("cpu")
        mp.gpu.reset_statistics()
        cpu_simulation, cpu_monitor = make_simulation()
        cpu_simulation.run(until=2.0)
        cpu_values = np.asarray(mp.get_fluxes(cpu_monitor))
        self.assertGreater(np.max(np.abs(cpu_values)), 1.0e-3)
        np.testing.assert_allclose(
            values, cpu_values, rtol=3.0e-4, atol=1.0e-8
        )
        cpu_reductions = mp.gpu.statistics()["dft_reductions"]
        self.assertGreater(
            global_sum(cpu_reductions["cpu_dft_reduction_calls"]), 0
        )
        self.assertEqual(
            global_sum(cpu_reductions["cuda_dft_reduction_calls"]), 0
        )
        cpu_simulation.reset_meep()

    def test_required_cuda_cylindrical_cartesian_near2far(self):
        self.require_cuda()
        import meep.adjoint as mpa

        mp.gpu.set_backend("cuda")
        simulation = mp.Simulation(
            cell_size=mp.Vector3(1.2, 0.0, 1.2),
            resolution=9,
            dimensions=mp.CYLINDRICAL,
            m=2,
            sources=[
                mp.Source(
                    mp.GaussianSource(0.31, fwidth=0.12),
                    component=mp.Ep,
                    center=mp.Vector3(0.35, 0.0, -0.12),
                )
            ],
        )
        regions = [
            mp.Near2FarRegion(
                center=mp.Vector3(0.38, 0.0, 0.32),
                size=mp.Vector3(0.76, 0.0, 0.0),
                weight=1.0,
            )
        ]
        observation_points = [
            mp.Vector3(-2.2, 1.7, -0.13),
            mp.Vector3(2.4, -1.1, 0.29),
        ]
        far_fields = mpa.Near2FarFields(
            simulation,
            regions,
            observation_points,
            greencyl_tol=1.0e-5,
            cartesian=True,
        )
        near2far = far_fields.register_monitors(np.array([0.31]))
        simulation.run(until=12.0)

        mp.gpu.reset_statistics()
        cuda_batch = simulation.get_farfields_points(
            near2far,
            observation_points,
            greencyl_tol=1.0e-5,
            cartesian=True,
        )
        cuda_scalars = np.asarray(
            [
                simulation.get_farfield(
                    near2far,
                    point,
                    greencyl_tol=1.0e-5,
                    cartesian=True,
                )
                for point in observation_points
            ]
        ).reshape(cuda_batch.shape)
        cuda_grid = simulation.get_farfields(
            near2far,
            resolution=5.0,
            center=mp.Vector3(-2.2, 1.5, 0.0),
            size=mp.Vector3(0.5, 0.5, 0.5),
            greencyl_tol=1.0e-5,
            cartesian=True,
        )
        objective_cuda = far_fields()
        cuda_statistics = mp.gpu.statistics()["near2far"]

        np.testing.assert_allclose(
            cuda_scalars, cuda_batch, rtol=5.0e-6, atol=5.0e-8
        )
        np.testing.assert_allclose(
            objective_cuda, cuda_batch, rtol=5.0e-6, atol=5.0e-8
        )
        self.assertEqual(cuda_batch.shape, (2, 1, 6))
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            self.assertEqual(np.asarray(cuda_grid[component]).size, 8)
        self.assertGreater(
            cuda_statistics["cuda_near2far_transform_calls"]
            + cuda_statistics["near2far_mpi_allreduce_calls"],
            0,
        )
        self.assertEqual(cuda_statistics["cpu_near2far_transform_calls"], 0)

        mp.gpu.set_backend("cpu")
        mp.gpu.reset_statistics()
        cpu_batch = simulation.get_farfields_points(
            near2far,
            observation_points,
            greencyl_tol=1.0e-5,
            cartesian=True,
        )
        cpu_grid = simulation.get_farfields(
            near2far,
            resolution=5.0,
            center=mp.Vector3(-2.2, 1.5, 0.0),
            size=mp.Vector3(0.5, 0.5, 0.5),
            greencyl_tol=1.0e-5,
            cartesian=True,
        )
        legacy = np.asarray(
            simulation.get_farfield(
                near2far,
                observation_points[1],
                greencyl_tol=1.0e-5,
            )
        )
        cpu_statistics = mp.gpu.statistics()["near2far"]
        np.testing.assert_allclose(
            cuda_batch, cpu_batch, rtol=4.0e-3, atol=3.0e-8
        )
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            np.testing.assert_allclose(
                cuda_grid[component],
                cpu_grid[component],
                rtol=4.0e-3,
                atol=3.0e-8,
            )
        self.assertGreater(
            np.max(np.abs(legacy - cpu_batch[1, 0])),
            1.0e-7 * max(1.0, np.max(np.abs(cpu_batch[1, 0]))),
        )
        self.assertGreater(cpu_statistics["cpu_near2far_transform_calls"], 0)
        self.assertEqual(cpu_statistics["cuda_near2far_transform_calls"], 0)

        mp.gpu.set_backend("cuda")
        mp.gpu.reset_statistics()
        gradient = np.empty((2, 1, 6), dtype=np.complex128)
        for index in range(gradient.size):
            gradient.flat[index] = 0.11 + 0.003j * index
        adjoint_sources = far_fields.place_adjoint_source(gradient)
        adjoint_statistics = mp.gpu.statistics()["near2far"]
        if adjoint_sources:
            self.assertEqual(
                adjoint_statistics["cuda_near2far_adjoint_calls"], 1
            )
            self.assertGreater(
                adjoint_statistics["cuda_near2far_adjoint_kernel_launches"], 0
            )
        else:
            self.assertEqual(
                adjoint_statistics["cuda_near2far_adjoint_calls"], 0
            )
        self.assertEqual(adjoint_statistics["cpu_near2far_adjoint_calls"], 0)
        simulation.reset_meep()


if __name__ == "__main__":
    unittest.main()
