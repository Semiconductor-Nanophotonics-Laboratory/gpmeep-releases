import os
import unittest

import h5py
import numpy as np
from utils import ApproxComparisonTestCase

import meep as mp


class TestDFTFields(ApproxComparisonTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = mp.make_output_directory()

    @classmethod
    def tearDownClass(cls):
        mp.delete_directory(cls.temp_dir)

    def init(self):
        resolution = 10
        n = 3.4
        w = 1.0
        r = 1.0
        pad = 4
        self.dpml = 2
        self.sxy = 2.0 * (r + w + pad + self.dpml)
        cell = mp.Vector3(self.sxy, self.sxy)
        pml_layers = [mp.PML(self.dpml)]

        geometry = [
            mp.Cylinder(r + w, material=mp.Medium(epsilon=n**2)),
            mp.Cylinder(r, material=mp.vacuum),
        ]

        self.fcen = 0.118
        self.df = 0.1

        src = mp.GaussianSource(self.fcen, fwidth=self.df)
        sources = [mp.Source(src=src, component=mp.Ez, center=mp.Vector3(r + 0.1))]

        return mp.Simulation(
            cell_size=cell,
            resolution=resolution,
            geometry=geometry,
            sources=sources,
            boundary_layers=pml_layers,
        )

    def assert_dft_hdf5_contract(self, sim, monitor, filename, nfreq, ndim):
        expected = [
            np.asarray(sim.get_dft_array(monitor, mp.Ez, freq_idx))
            for freq_idx in range(nfreq)
        ]
        for array in expected:
            self.assertEqual(array.ndim, ndim)

        statistics_before = mp.gpu.statistics()["dft_materializations"]
        sim.output_dft(monitor, filename)
        statistics_after = mp.gpu.statistics()["dft_materializations"]
        statistics = {
            key: statistics_after[key] - statistics_before[key]
            for key in statistics_after
        }
        cuda_output = mp.gpu.active_backend == "cuda"
        expected_output_points = sum(array.size for array in expected)
        if cuda_output:
            self.assertEqual(statistics["cpu_dft_output_calls"], 0)
            self.assertEqual(statistics["cpu_dft_output_points"], 0)
            self.assertEqual(statistics["cuda_dft_output_calls"], 1)
            self.assertEqual(
                statistics["cuda_dft_output_points"], expected_output_points
            )
            if ndim == 2:
                self.assertGreater(statistics["cuda_dft_output_staging_calls"], 0)
                self.assertGreater(
                    statistics["cuda_dft_output_staging_points"], 0
                )
                self.assertGreater(
                    statistics["cuda_dft_output_staging_frequencies"], 0
                )
                self.assertGreater(
                    statistics["cuda_dft_output_staging_kernel_launches"], 0
                )
                self.assertGreater(
                    statistics[
                        "cuda_dft_output_staging_result_device_to_host_bytes"
                    ],
                    0,
                )
                self.assertGreater(
                    statistics[
                        "cuda_dft_output_staging_full_dft_device_to_host_bytes_avoided"
                    ],
                    0,
                )
                self.assertEqual(
                    statistics["cuda_dft_materialization_kernel_launches"], 0
                )
                self.assertEqual(statistics["dft_array_mpi_allreduce_calls"], 0)
                self.assertEqual(statistics["dft_array_mpi_allreduce_bytes"], 0)
            else:
                expected_result_bytes = expected_output_points * np.dtype(
                    np.complex64
                ).itemsize
                self.assertEqual(statistics["cuda_dft_output_staging_calls"], 0)
                self.assertEqual(
                    statistics["cuda_dft_output_staging_kernel_launches"], 0
                )
                # get_dft_array above warms the dense all-frequency cache.
                # A geometry whose internal full rank is already reduced can
                # reuse it with zero launches; a genuinely collapsed layout
                # launches one materialization and one collapse kernel.
                self.assertIn(
                    statistics["cuda_dft_materialization_kernel_launches"],
                    (0, 2),
                )
                self.assertEqual(
                    statistics[
                        "cuda_dft_materialization_result_device_to_host_bytes"
                    ],
                    expected_result_bytes,
                )
                self.assertEqual(
                    statistics["dft_array_mpi_allreduce_calls"], nfreq
                )
                self.assertEqual(
                    statistics["dft_array_mpi_allreduce_bytes"],
                    expected_result_bytes,
                )
        else:
            self.assertEqual(statistics["cpu_dft_output_calls"], 1)
            # Collapsed point/line output is gathered and written by the
            # master rank, so only that rank records the emitted points. A
            # nondegenerate parallel HDF5 output is processed on every rank.
            expected_cpu_output_points = (
                expected_output_points if ndim == 2 or mp.am_master() else 0
            )
            self.assertEqual(
                statistics["cpu_dft_output_points"], expected_cpu_output_points
            )
            self.assertEqual(statistics["cuda_dft_output_calls"], 0)
            self.assertEqual(statistics["cuda_dft_output_staging_calls"], 0)

        expected_datasets = {
            f"ez_{freq_idx}.{part}"
            for freq_idx in range(nfreq)
            for part in ("r", "i")
        }
        with h5py.File(f"{filename}.h5", "r") as output:
            self.assertEqual(set(output.keys()), expected_datasets)
            for freq_idx, array in enumerate(expected):
                real = output[f"ez_{freq_idx}.r"][()]
                imag = output[f"ez_{freq_idx}.i"][()]

                # Meep stores a rank-0 result as a one-element HDF5 dataset.
                expected_shape = array.shape if array.ndim else (1,)
                self.assertEqual(real.shape, expected_shape)
                self.assertEqual(imag.shape, expected_shape)
                # Preserve Meep's historical DFT output ABI: an ordinary
                # nondegenerate FP32 monitor is stored as float32, while the
                # master-only collapsed path uses its FP64 real_array.
                expected_itemsize = (
                    4 if ndim == 2 and mp.is_single_precision() else 8
                )
                self.assertEqual(real.dtype.kind, "f")
                self.assertEqual(imag.dtype.kind, "f")
                self.assertEqual(real.dtype.itemsize, expected_itemsize)
                self.assertEqual(imag.dtype.itemsize, expected_itemsize)
                np.testing.assert_allclose(
                    mp.complexarray(real, imag).reshape(-1),
                    array.reshape(-1),
                    rtol=0,
                    atol=1e-6,
                )

    def test_use_centered_grid(self):
        sim = self.init()
        sim.init_sim()
        dft_fields = sim.add_dft_fields([mp.Ez], self.fcen, 0, 1, yee_grid=True)
        sim.run(until=100)

    def test_get_dft_array(self):
        sim = self.init()
        sim.init_sim()
        dft_fields = sim.add_dft_fields([mp.Ez], self.fcen, 0, 1)
        fr = mp.FluxRegion(
            mp.Vector3(), size=mp.Vector3(self.sxy, self.sxy), direction=mp.X
        )
        dft_flux = sim.add_flux(self.fcen, 0, 1, fr)

        # volumes with zero thickness in x and y directions to test collapsing
        # of empty dimensions in DFT array and HDF5 output routines
        thin_x_volume = mp.Volume(
            center=mp.Vector3(0.35 * self.sxy), size=mp.Vector3(y=0.8 * self.sxy)
        )
        thin_x_flux = sim.add_dft_fields([mp.Ez], self.fcen, 0, 1, where=thin_x_volume)
        thin_y_volume = mp.Volume(
            center=mp.Vector3(y=0.25 * self.sxy), size=mp.Vector3(x=self.sxy)
        )
        thin_y_flux = sim.add_flux(self.fcen, 0, 1, mp.FluxRegion(volume=thin_y_volume))

        sim.run(until_after_sources=100)

        # test proper collapsing of degenerate dimensions in HDF5 files and arrays
        thin_x_array = sim.get_dft_array(thin_x_flux, mp.Ez, 0)
        thin_y_array = sim.get_dft_array(thin_y_flux, mp.Ez, 0)
        np.testing.assert_equal(thin_x_array.ndim, 1)
        np.testing.assert_equal(thin_y_array.ndim, 1)

        sim.output_dft(thin_x_flux, os.path.join(self.temp_dir, "thin-x-flux"))
        sim.output_dft(thin_y_flux, os.path.join(self.temp_dir, "thin-y-flux"))

        with h5py.File(os.path.join(self.temp_dir, "thin-x-flux.h5"), "r") as thin_x:
            thin_x_h5 = mp.complexarray(thin_x["ez_0.r"][()], thin_x["ez_0.i"][()])

        with h5py.File(os.path.join(self.temp_dir, "thin-y-flux.h5"), "r") as thin_y:
            thin_y_h5 = mp.complexarray(thin_y["ez_0.r"][()], thin_y["ez_0.i"][()])

        tol = 1e-6
        self.assertClose(thin_x_array, thin_x_h5, epsilon=tol)
        self.assertClose(thin_y_array, thin_y_h5, epsilon=tol)

        # compare array data to HDF5 file content for fields and flux
        fields_arr = sim.get_dft_array(dft_fields, mp.Ez, 0)
        flux_arr = sim.get_dft_array(dft_flux, mp.Ez, 0)

        sim.output_dft(dft_fields, os.path.join(self.temp_dir, "dft-fields"))
        sim.output_dft(dft_flux, os.path.join(self.temp_dir, "dft-flux"))

        with h5py.File(
            os.path.join(self.temp_dir, "dft-fields.h5"), "r"
        ) as fields, h5py.File(os.path.join(self.temp_dir, "dft-flux.h5"), "r") as flux:
            exp_fields = mp.complexarray(fields["ez_0.r"][()], fields["ez_0.i"][()])
            exp_flux = mp.complexarray(flux["ez_0.r"][()], flux["ez_0.i"][()])

        tol = 1e-6
        self.assertClose(exp_fields, fields_arr, epsilon=tol)
        self.assertClose(exp_flux, flux_arr, epsilon=tol)

    def test_output_dft_geometry_and_frequency_contract(self):
        sim = self.init()
        sim.init_sim()

        monitor_volumes = {
            "point": (mp.Volume(center=mp.Vector3(0.2, -0.1)), 0),
            "thin-x": (
                mp.Volume(center=mp.Vector3(x=0.3), size=mp.Vector3(y=2.0)),
                1,
            ),
            "thin-y": (
                mp.Volume(center=mp.Vector3(y=-0.25), size=mp.Vector3(x=2.0)),
                1,
            ),
            "full-2d": (mp.Volume(size=mp.Vector3(2.0, 2.0)), 2),
        }
        monitors = {}
        for monitor_name, (volume, ndim) in monitor_volumes.items():
            for nfreq in (1, 3):
                monitors[(monitor_name, nfreq)] = (
                    sim.add_dft_fields(
                        [mp.Ez],
                        self.fcen,
                        0 if nfreq == 1 else self.df,
                        nfreq,
                        where=volume,
                    ),
                    ndim,
                )

        sim.run(until=5)

        for (monitor_name, nfreq), (monitor, ndim) in monitors.items():
            with self.subTest(monitor=monitor_name, nfreq=nfreq):
                self.assert_dft_hdf5_contract(
                    sim,
                    monitor,
                    os.path.join(
                        self.temp_dir, f"dft-contract-{monitor_name}-{nfreq}freq"
                    ),
                    nfreq,
                    ndim,
                )

    def test_output_dft_does_not_duplicate_h5_suffix(self):
        sim = self.init()
        monitor = sim.add_dft_fields(
            [mp.Ez],
            self.fcen,
            0,
            1,
            where=mp.Volume(size=mp.Vector3(y=2.0)),
        )
        sim.run(until=1)

        filename = os.path.join(self.temp_dir, "dft-already-suffixed.h5")
        sim.output_dft(monitor, filename)
        self.assertTrue(os.path.isfile(filename))
        self.assertFalse(os.path.exists(f"{filename}.h5"))

    def test_decimated_dft_fields_are_almost_equal_to_undecimated_fields(self):
        sim = self.init()
        sim.init_sim()
        undecimated_field = sim.add_dft_fields(
            [mp.Ez], self.fcen, 0, 1, decimation_factor=1
        )
        decimated_field = sim.add_dft_fields(
            [mp.Ez], self.fcen, 0, 1, decimation_factor=4
        )

        sim.run(until_after_sources=100)

        expected_dft = sim.get_dft_array(undecimated_field, mp.Ez, 0)
        actual_dft = sim.get_dft_array(decimated_field, mp.Ez, 0)
        self.assertClose(expected_dft, actual_dft, epsilon=1e-3)


if __name__ == "__main__":
    unittest.main()
