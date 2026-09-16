import cmath
import math
import unittest
from unittest import mock

import numpy as np
import parameterized

import meep as mp


class TestEigCoeffs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolution = 30  # pixels/μm

        cls.dpml = 1.0  # PML thickness
        cls.dsub = 1.0  # substrate thickness
        cls.dpad = 1.0  # padding thickness between grating and PML
        cls.gp = 6.0  # grating period
        cls.gh = 0.5  # grating height
        cls.gdc = 0.5  # grating duty cycle

        cls.sx = cls.dpml + cls.dsub + cls.gh + cls.dpad + cls.dpml
        cls.sy = cls.gp

        cls.cell_size = mp.Vector3(cls.sx, cls.sy, 0)

        cls.boundary_layers = [mp.PML(thickness=cls.dpml, direction=mp.X)]

        wvl = 0.5  # center wavelength
        cls.fcen = 1 / wvl  # center frequency
        cls.df = 0.05 * cls.fcen  # frequency width

        cls.ng = 1.5
        cls.glass = mp.Medium(index=cls.ng)

        cls.geometry = [
            mp.Block(
                material=cls.glass,
                size=mp.Vector3(cls.dpml + cls.dsub, mp.inf, mp.inf),
                center=mp.Vector3(
                    -0.5 * cls.sx + 0.5 * (cls.dpml + cls.dsub),
                    0,
                    0,
                ),
            ),
            mp.Block(
                material=cls.glass,
                size=mp.Vector3(cls.gh, cls.gdc * cls.gp, mp.inf),
                center=mp.Vector3(
                    -0.5 * cls.sx + cls.dpml + cls.dsub + 0.5 * cls.gh,
                    0,
                    0,
                ),
            ),
        ]

    def test_binary_grating_fp32_newton_precision(self):
        """FP32 Newton matching stays live without hiding resolvable corrections."""

        src_pt = mp.Vector3(-0.5 * self.sx + self.dpml, 0, 0)
        source = mp.Source(
            mp.GaussianSource(self.fcen, fwidth=self.df),
            component=mp.Ez,
            center=src_pt,
            size=mp.Vector3(0, self.sy, 0),
        )
        sim = mp.Simulation(
            resolution=self.resolution,
            cell_size=self.cell_size,
            boundary_layers=self.boundary_layers,
            k_point=mp.Vector3(),
            default_material=self.glass,
            sources=[source],
            symmetries=[mp.Mirror(mp.Y)],
        )

        refl_pt = mp.Vector3(
            -0.5 * self.sx + self.dpml + 0.5 * self.dsub,
            0,
            0,
        )
        refl_flux = sim.add_mode_monitor(
            self.fcen,
            0,
            1,
            mp.FluxRegion(center=refl_pt, size=mp.Vector3(0, self.sy, 0)),
        )
        sim.init_sim()

        result = sim.get_eigenmode_coefficients(
            refl_flux,
            bands=[1],
            kpoint_func=lambda *not_used: mp.Vector3(self.fcen * self.ng, 0, 0),
            eig_parity=mp.ODD_Z + mp.EVEN_Y,
            direction=mp.NO_DIRECTION,
            eig_vol=mp.Volume(center=refl_pt, size=mp.Vector3(0, 1e-7, 0)),
            eig_tolerance=1e-12,
        )

        self.assertAlmostEqual(result.vgrp[0], 1 / self.ng, places=6)
        self.assertTrue(np.isfinite(result.vgrp[0]))
        self.assertTrue(
            np.all(
                np.isfinite(
                    [
                        result.kpoints[0].x,
                        result.kpoints[0].y,
                        result.kpoints[0].z,
                    ]
                )
            )
        )
        if mp.is_single_precision():
            self.assertAlmostEqual(result.kpoints[0].x, self.fcen * self.ng, places=7)
        else:
            self.assertAlmostEqual(result.kpoints[0].x, self.fcen * self.ng, places=10)

        # Generic (non-DiffractedPlanewave) special-kz matching must keep beta fixed while Newton
        # updates the transverse wavevector.  This also exercises the affine R*k conversion for a
        # zero-width monitor axis; the legacy implementation normalized kdir by the full 3D k and
        # dropped beta from non-diagonal reciprocal-basis components.
        beta = 1.0
        transverse_guess = 2.7
        special_kz_source = mp.Source(
            mp.GaussianSource(self.fcen, fwidth=self.df),
            component=mp.Ez,
            center=mp.Vector3(),
        )
        special_kz_sim = mp.Simulation(
            resolution=10,
            cell_size=mp.Vector3(2, 1, 0),
            k_point=mp.Vector3(transverse_guess, 0, beta),
            default_material=self.glass,
            sources=[special_kz_source],
            kz_2d="complex",
        )
        special_kz_volume = mp.Volume(
            center=mp.Vector3(),
            size=mp.Vector3(0, 1, 0),
        )
        special_kz_flux = special_kz_sim.add_mode_monitor(
            self.fcen,
            0,
            1,
            mp.FluxRegion(
                center=special_kz_volume.center,
                size=special_kz_volume.size,
            ),
        )
        special_kz_sim.init_sim()
        special_kz_result = special_kz_sim.get_eigenmode_coefficients(
            special_kz_flux,
            bands=[1],
            kpoint_func=lambda *not_used: mp.Vector3(transverse_guess, 0, beta),
            eig_parity=0,
            direction=mp.NO_DIRECTION,
            eig_vol=special_kz_volume,
            eig_tolerance=1e-12,
        )

        transverse_expected = math.sqrt((self.ng * self.fcen) ** 2 - beta**2)
        special_kz = special_kz_result.kpoints[0]
        self.assertAlmostEqual(special_kz.x, transverse_expected, places=5)
        self.assertAlmostEqual(special_kz.y, 0, places=8)
        self.assertAlmostEqual(special_kz.z, beta, places=8)
        self.assertAlmostEqual(
            special_kz_result.vgrp[0],
            transverse_expected / (self.ng**2 * self.fcen),
            places=6,
        )
        self.assertAlmostEqual(
            math.sqrt(special_kz.x**2 + special_kz.z**2) / self.ng,
            self.fcen,
            places=6,
        )

        # Exercise the non-diagonal reciprocal basis produced by an oblique transverse k on an
        # X-normal line monitor, including the fixed-beta affine round trip.
        transverse_angle = math.radians(20)
        oblique_guess = mp.Vector3(
            transverse_guess * math.cos(transverse_angle),
            transverse_guess * math.sin(transverse_angle),
            beta,
        )
        oblique_special_kz_sim = mp.Simulation(
            resolution=10,
            cell_size=mp.Vector3(2, 1, 0),
            k_point=oblique_guess,
            default_material=self.glass,
            sources=[
                mp.Source(
                    mp.GaussianSource(self.fcen, fwidth=self.df),
                    component=mp.Ez,
                    center=mp.Vector3(),
                )
            ],
            kz_2d="complex",
        )
        oblique_special_kz_flux = oblique_special_kz_sim.add_mode_monitor(
            self.fcen,
            0,
            1,
            mp.FluxRegion(
                center=special_kz_volume.center,
                size=special_kz_volume.size,
            ),
        )
        oblique_special_kz_sim.init_sim()
        oblique_special_kz_result = (
            oblique_special_kz_sim.get_eigenmode_coefficients(
                oblique_special_kz_flux,
                bands=[1],
                kpoint_func=lambda *not_used: oblique_guess,
                eig_parity=0,
                direction=mp.NO_DIRECTION,
                eig_vol=special_kz_volume,
                eig_tolerance=1e-12,
            )
        )
        oblique_special_kz = oblique_special_kz_result.kpoints[0]
        self.assertAlmostEqual(
            oblique_special_kz.x,
            transverse_expected * math.cos(transverse_angle),
            places=5,
        )
        self.assertAlmostEqual(
            oblique_special_kz.y,
            transverse_expected * math.sin(transverse_angle),
            places=5,
        )
        self.assertAlmostEqual(oblique_special_kz.z, beta, places=8)
        self.assertAlmostEqual(
            math.sqrt(
                oblique_special_kz.x**2
                + oblique_special_kz.y**2
                + oblique_special_kz.z**2
            )
            / self.ng,
            self.fcen,
            places=6,
        )

        # A beta-only initial guess previously constructed two parallel rows in R for a line
        # monitor.  The generic solver must choose the monitor normal as a deterministic
        # transverse direction and converge without creating a singular reciprocal basis.
        zero_transverse_sim = mp.Simulation(
            resolution=10,
            cell_size=mp.Vector3(2, 1, 0),
            k_point=mp.Vector3(0, 0, beta),
            default_material=self.glass,
            sources=[
                mp.Source(
                    mp.GaussianSource(self.fcen, fwidth=self.df),
                    component=mp.Ez,
                    center=mp.Vector3(),
                )
            ],
            kz_2d="complex",
        )
        zero_transverse_flux = zero_transverse_sim.add_mode_monitor(
            self.fcen,
            0,
            1,
            mp.FluxRegion(
                center=special_kz_volume.center,
                size=special_kz_volume.size,
            ),
        )
        zero_transverse_sim.init_sim()
        zero_transverse_result = zero_transverse_sim.get_eigenmode_coefficients(
            zero_transverse_flux,
            bands=[1],
            kpoint_func=lambda *not_used: mp.Vector3(0, 0, beta),
            eig_parity=0,
            direction=mp.NO_DIRECTION,
            eig_vol=special_kz_volume,
            eig_tolerance=1e-12,
        )
        zero_transverse_k = zero_transverse_result.kpoints[0]
        self.assertAlmostEqual(zero_transverse_k.x, transverse_expected, places=5)
        self.assertAlmostEqual(zero_transverse_k.y, 0, places=8)
        self.assertAlmostEqual(zero_transverse_k.z, beta, places=8)
        self.assertAlmostEqual(
            math.sqrt(zero_transverse_k.x**2 + zero_transverse_k.z**2)
            / self.ng,
            self.fcen,
            places=6,
        )

        # Preserve the zero-width axis before R construction mutates its internal size array: a
        # Y-normal monitor must choose +Y, not silently fall back to +X.
        zero_transverse_y_volume = mp.Volume(
            center=mp.Vector3(),
            size=mp.Vector3(1, 0, 0),
        )
        zero_transverse_y_sim = mp.Simulation(
            resolution=10,
            cell_size=mp.Vector3(1, 2, 0),
            k_point=mp.Vector3(0, 0, beta),
            default_material=self.glass,
            sources=[
                mp.Source(
                    mp.GaussianSource(self.fcen, fwidth=self.df),
                    component=mp.Ez,
                    center=mp.Vector3(),
                )
            ],
            kz_2d="complex",
        )
        zero_transverse_y_flux = zero_transverse_y_sim.add_mode_monitor(
            self.fcen,
            0,
            1,
            mp.FluxRegion(
                center=zero_transverse_y_volume.center,
                size=zero_transverse_y_volume.size,
            ),
        )
        zero_transverse_y_sim.init_sim()
        zero_transverse_y_result = (
            zero_transverse_y_sim.get_eigenmode_coefficients(
                zero_transverse_y_flux,
                bands=[1],
                kpoint_func=lambda *not_used: mp.Vector3(0, 0, beta),
                eig_parity=0,
                direction=mp.NO_DIRECTION,
                eig_vol=zero_transverse_y_volume,
                eig_tolerance=1e-12,
            )
        )
        zero_transverse_y_k = zero_transverse_y_result.kpoints[0]
        self.assertAlmostEqual(zero_transverse_y_k.x, 0, places=8)
        self.assertAlmostEqual(zero_transverse_y_k.y, transverse_expected, places=5)
        self.assertAlmostEqual(zero_transverse_y_k.z, beta, places=8)
        self.assertAlmostEqual(
            math.sqrt(zero_transverse_y_k.y**2 + zero_transverse_y_k.z**2)
            / self.ng,
            self.fcen,
            places=6,
        )

        if not mp.is_single_precision():
            return

        # Near cutoff, a tiny frequency residual corresponds to a significant kx correction.  It
        # must remain eligible for Newton refinement rather than being hidden by a frequency floor.
        ky = self.fcen * self.ng - 1e-6
        kx_guess = 3e-3
        cutoff_source = mp.Source(
            mp.GaussianSource(self.fcen, fwidth=self.df),
            component=mp.Ez,
            center=mp.Vector3(),
        )
        cutoff_sim = mp.Simulation(
            resolution=self.resolution,
            cell_size=self.cell_size,
            k_point=mp.Vector3(0, ky, 0),
            default_material=self.glass,
            sources=[cutoff_source],
        )
        cutoff_volume = mp.Volume(
            center=mp.Vector3(),
            size=mp.Vector3(0, 1e-7, 0),
        )
        cutoff_flux = cutoff_sim.add_mode_monitor(
            self.fcen,
            0,
            1,
            mp.FluxRegion(center=cutoff_volume.center, size=cutoff_volume.size),
        )
        cutoff_sim.init_sim()
        cutoff_result = cutoff_sim.get_eigenmode_coefficients(
            cutoff_flux,
            bands=[1],
            kpoint_func=lambda *not_used: mp.Vector3(kx_guess, ky, 0),
            eig_parity=mp.ODD_Z,
            direction=mp.X,
            eig_vol=cutoff_volume,
            eig_tolerance=1e-12,
        )

        kx = cutoff_result.kpoints[0].x
        ky_found = cutoff_result.kpoints[0].y
        self.assertNotEqual(kx, kx_guess)
        analytic_kx = math.sqrt((self.ng * self.fcen) ** 2 - ky_found**2)
        self.assertTrue(
            math.isclose(kx, analytic_kx, rel_tol=0.01),
            msg=f"near-cutoff kx mismatch: MPB={kx}, analytic={analytic_kx}",
        )
        self.assertAlmostEqual(
            math.sqrt(kx**2 + ky_found**2) / self.ng,
            self.fcen,
            places=7,
        )
        self.assertAlmostEqual(
            cutoff_result.vgrp[0],
            kx / (self.ng**2 * self.fcen),
            places=10,
        )

    @unittest.skipUnless(mp.is_single_precision(), "FP32 one-cell MPB liveness regression")
    def test_fp32_one_cell_multiband_determinism(self):
        """A degenerate two-band plane-wave solve terminates and is repeatable."""

        # Start at the analytic root so the first FP32 residual produces only a tiny update; this
        # is the warm-start pattern which previously exposed MPB's one-cell stall.
        kx_guess = self.fcen * self.ng
        sim = mp.Simulation(
            resolution=10,
            cell_size=mp.Vector3(2, 1, 0),
            k_point=mp.Vector3(kx_guess, 0, 0),
            default_material=self.glass,
            sources=[
                mp.Source(
                    mp.GaussianSource(self.fcen, fwidth=self.df),
                    component=mp.Ez,
                    center=mp.Vector3(),
                )
            ],
        )
        eig_vol = mp.Volume(center=mp.Vector3(), size=mp.Vector3(0, 1e-7, 0))
        monitor = sim.add_mode_monitor(
            self.fcen,
            0,
            1,
            mp.FluxRegion(center=eig_vol.center, size=eig_vol.size),
        )
        sim.run(until=1)

        def solve():
            return sim.get_eigenmode_coefficients(
                monitor,
                bands=[1, 2],
                kpoint_func=lambda *not_used: mp.Vector3(kx_guess, 0, 0),
                eig_parity=0,
                direction=mp.X,
                eig_vol=eig_vol,
                eig_tolerance=1e-12,
            )

        first = solve()
        second = solve()
        first_k = np.array([[k.x, k.y, k.z] for k in first.kpoints])
        second_k = np.array([[k.x, k.y, k.z] for k in second.kpoints])
        self.assertTrue(np.all(np.isfinite(first_k)))
        self.assertTrue(np.all(np.isfinite(first.vgrp)))
        self.assertTrue(np.all(np.isfinite(first.cscale)))
        self.assertTrue(np.all(np.linalg.norm(first_k, axis=1) > 0))
        self.assertTrue(np.all(np.abs(first.vgrp) > 0))
        self.assertTrue(np.all(np.abs(first.cscale) > 0))
        band_alpha_norms = np.linalg.norm(
            first.alpha.reshape(first.alpha.shape[0], -1), axis=1
        )
        self.assertTrue(np.all(band_alpha_norms > 1e-12))
        np.testing.assert_allclose(first_k, second_k, rtol=0, atol=1e-7)
        np.testing.assert_allclose(first.vgrp, second.vgrp, rtol=0, atol=1e-7)
        np.testing.assert_allclose(first.cscale, second.cscale, rtol=0, atol=1e-7)
        np.testing.assert_allclose(first.alpha, second.alpha, rtol=0, atol=1e-7)

    def test_diffracted_planewave_cutoff_is_rejected(self):
        """An exact-cutoff order has no finite unit-power normalization."""

        period = 1 / (self.ng * self.fcen)
        sim = mp.Simulation(
            resolution=30,
            cell_size=mp.Vector3(1, period, 0),
            default_material=self.glass,
            sources=[
                mp.Source(
                    mp.GaussianSource(self.fcen, fwidth=self.df),
                    component=mp.Ez,
                    center=mp.Vector3(),
                )
            ],
        )
        monitor = sim.add_mode_monitor(
            self.fcen,
            0,
            1,
            mp.FluxRegion(center=mp.Vector3(), size=mp.Vector3(0, period, 0)),
        )
        sim.init_sim()
        result = sim.get_eigenmode_coefficients(
            monitor,
            mp.DiffractedPlanewave(
                (0, 1, 0),
                mp.Vector3(0, 1, 0),
                1,
                0,
            ),
        )
        self.assertEqual(result.vgrp[0], 0)
        self.assertEqual(result.cscale[0], 0)
        self.assertTrue(np.all(result.alpha == 0))
        self.assertTrue(result.kpoints[0].close(mp.Vector3()))

        # The optional MPB eig_vol may have finite thickness normal to the actual monitor plane.
        # DiffractedPlanewave propagation must still use the monitor normal and must not count the
        # fixed normal k as a parallel diffraction component.
        thick_sim = mp.Simulation(
            resolution=30,
            cell_size=mp.Vector3(1, period, 0),
            default_material=self.glass,
            sources=[
                mp.Source(
                    mp.GaussianSource(self.fcen, fwidth=self.df),
                    component=mp.Ez,
                    center=mp.Vector3(),
                )
            ],
        )
        thick_monitor = thick_sim.add_mode_monitor(
            self.fcen,
            0,
            1,
            mp.FluxRegion(center=mp.Vector3(), size=mp.Vector3(0, period, 0)),
        )
        thick_sim.init_sim()
        propagating = thick_sim.get_eigenmode_coefficients(
            thick_monitor,
            mp.DiffractedPlanewave(
                (0, 0, 0),
                mp.Vector3(0, 1, 0),
                1,
                0,
            ),
            eig_vol=mp.Volume(
                center=mp.Vector3(),
                size=mp.Vector3(1e-3, period, 0),
            ),
        )
        self.assertAlmostEqual(propagating.kpoints[0].x, self.ng * self.fcen, places=6)
        self.assertTrue(np.isfinite(propagating.vgrp[0]))
        self.assertGreater(propagating.vgrp[0], 0)
        self.assertTrue(np.isfinite(propagating.cscale[0]))
        self.assertGreater(propagating.cscale[0], 0)

        # MPB treats an axis parallel to k+G as a fatal CHECK.  Meep must reject it as an
        # ordinary non-normalizable mode, preserve the monitor-owned cache, and recover on the
        # next valid request instead of terminating or later double-freeing the cache.
        parallel_axis = thick_sim.get_eigenmode_coefficients(
            thick_monitor,
            mp.DiffractedPlanewave(
                (0, 0, 0),
                mp.Vector3(1, 0, 0),
                1,
                0,
            ),
            eig_vol=mp.Volume(
                center=mp.Vector3(),
                size=mp.Vector3(1e-3, period, 0),
            ),
        )
        self.assertEqual(parallel_axis.vgrp[0], 0)
        self.assertEqual(parallel_axis.cscale[0], 0)
        self.assertTrue(np.all(parallel_axis.alpha == 0))

        recovered = thick_sim.get_eigenmode_coefficients(
            thick_monitor,
            mp.DiffractedPlanewave(
                (0, 0, 0),
                mp.Vector3(0, 1, 0),
                1,
                0,
            ),
            eig_vol=mp.Volume(
                center=mp.Vector3(),
                size=mp.Vector3(1e-3, period, 0),
            ),
        )
        np.testing.assert_allclose(recovered.kpoints, propagating.kpoints, rtol=0, atol=1e-7)
        np.testing.assert_allclose(recovered.vgrp, propagating.vgrp, rtol=0, atol=1e-7)
        np.testing.assert_allclose(recovered.cscale, propagating.cscale, rtol=0, atol=1e-7)

    def test_eigenmode_api_guards_and_fixed_k(self):
        """Fixed-k frequency scaling and direct-API failure guards remain safe."""

        sim = mp.Simulation(
            resolution=10,
            cell_size=mp.Vector3(2, 1, 0),
            k_point=mp.Vector3(self.ng * self.fcen, 0, 0),
            default_material=self.glass,
            sources=[
                mp.Source(
                    mp.GaussianSource(self.fcen, fwidth=self.df),
                    component=mp.Ez,
                    center=mp.Vector3(),
                )
            ],
        )
        guard_monitor = sim.add_mode_monitor(
            self.fcen,
            0,
            1,
            mp.FluxRegion(center=mp.Vector3(), size=mp.Vector3(0, 1, 0)),
        )
        eig_vol = mp.Volume(center=mp.Vector3(), size=mp.Vector3(0, 1e-7, 0))
        sim.init_sim()

        fixed_k = sim.get_eigenmode(
            0,
            mp.X,
            eig_vol,
            1,
            mp.Vector3(self.ng * self.fcen, 0, 0),
            match_frequency=False,
            parity=mp.ODD_Z,
            resolution=10,
            eigensolver_tol=1e-12,
        )
        self.assertAlmostEqual(fixed_k.freq, self.fcen, places=6)
        self.assertAlmostEqual(fixed_k.k.x, self.ng * self.fcen, places=6)

        # The public API documents DiffractedPlanewave as a valid band_num.  In fixed-k mode the
        # complete |k+G| must be converted to frequency using epsilon*mu, including both the normal
        # and periodic components.  A non-vacuum medium makes a missing material factor observable.
        dp_vol = mp.Volume(center=mp.Vector3(), size=mp.Vector3(0, 1, 0))
        dp_band = mp.DiffractedPlanewave(
            (0, 1, 0),
            mp.Vector3(0, 1, 0),
            1,
            0,
        )
        fixed_dp = sim.get_eigenmode(
            0,
            mp.X,
            dp_vol,
            dp_band,
            mp.Vector3(1, 0, 0),
            eig_vol=dp_vol,
            match_frequency=False,
            resolution=10,
        )
        self.assertIs(fixed_dp.band_num, dp_band)
        self.assertAlmostEqual(fixed_dp.freq, math.sqrt(2) / self.ng, places=6)
        self.assertAlmostEqual(fixed_dp.k.x, 1, places=6)
        # `k` is the Bloch wavevector; `kdom` includes the selected reciprocal order.
        self.assertAlmostEqual(fixed_dp.k.y, 0, places=6)
        self.assertAlmostEqual(fixed_dp.k.z, 0, places=6)
        self.assertAlmostEqual(fixed_dp.kdom.x, 1, places=6)
        self.assertAlmostEqual(fixed_dp.kdom.y, 1, places=6)
        self.assertAlmostEqual(fixed_dp.kdom.z, 0, places=6)
        self.assertTrue(math.isfinite(fixed_dp.group_velocity))
        self.assertGreater(fixed_dp.group_velocity, 0)

        # NO_DIRECTION asks Meep to infer the source-plane normal.  A zero Bloch vector is valid
        # for a normally incident DiffractedPlanewave and must not be rejected by the generic
        # zero-k direction guard.
        normal_dp = sim.get_eigenmode(
            self.fcen,
            mp.NO_DIRECTION,
            dp_vol,
            mp.DiffractedPlanewave(
                (0, 0, 0),
                mp.Vector3(0, 1, 0),
                1,
                0,
            ),
            mp.Vector3(),
            eig_vol=dp_vol,
            resolution=10,
        )
        self.assertAlmostEqual(normal_dp.freq, self.fcen, places=6)
        self.assertAlmostEqual(normal_dp.k.x, self.ng * self.fcen, places=6)
        self.assertAlmostEqual(normal_dp.k.y, 0, places=6)
        self.assertAlmostEqual(normal_dp.group_velocity, 1 / self.ng, places=6)

        # With an oblique in-plane Bloch component, the inferred zero-width X lattice row must
        # still be +X.  Reusing the generic NO_DIRECTION k-aligned row makes R singular with the
        # full-width Y row.
        oblique_ky = 0.25
        oblique_dp_sim = mp.Simulation(
            resolution=10,
            cell_size=mp.Vector3(2, 1, 0),
            k_point=mp.Vector3(0, oblique_ky, 0),
            default_material=self.glass,
            sources=[
                mp.Source(
                    mp.GaussianSource(self.fcen, fwidth=self.df),
                    component=mp.Ez,
                    center=mp.Vector3(),
                )
            ],
        )
        oblique_dp_sim.init_sim()
        oblique_dp = oblique_dp_sim.get_eigenmode(
            self.fcen,
            mp.NO_DIRECTION,
            dp_vol,
            mp.DiffractedPlanewave(
                (0, 0, 0),
                mp.Vector3(0, 1, 0),
                1,
                0,
            ),
            mp.Vector3(0, oblique_ky, 0),
            eig_vol=dp_vol,
            resolution=10,
        )
        expected_oblique_kx = math.sqrt((self.ng * self.fcen) ** 2 - oblique_ky**2)
        self.assertAlmostEqual(oblique_dp.k.x, expected_oblique_kx, places=6)
        self.assertAlmostEqual(oblique_dp.k.y, oblique_ky, places=6)
        self.assertAlmostEqual(oblique_dp.freq, self.fcen, places=6)
        self.assertGreater(oblique_dp.group_velocity, 0)

        # Invalid planewave inputs must never reach MPB's process-terminating CHECKs.
        with self.assertRaisesRegex(RuntimeError, "normalizable mode"):
            sim.get_eigenmode(
                self.fcen,
                mp.X,
                dp_vol,
                mp.DiffractedPlanewave(
                    (0, 0, 0),
                    mp.Vector3(1, 0, 0),
                    1,
                    0,
                ),
                mp.Vector3(),
                eig_vol=dp_vol,
                resolution=10,
            )
        with self.assertRaisesRegex(RuntimeError, "normalizable mode"):
            sim.get_eigenmode(
                self.fcen,
                mp.X,
                dp_vol,
                mp.DiffractedPlanewave(
                    (0, 1000, 0),
                    mp.Vector3(0, 1, 0),
                    1,
                    0,
                ),
                mp.Vector3(),
                eig_vol=dp_vol,
                resolution=10,
            )

        with self.assertRaisesRegex(RuntimeError, "parallel to its source plane"):
            sim.get_eigenmode(
                self.fcen,
                mp.X,
                dp_vol,
                mp.DiffractedPlanewave(
                    (1, 0, 0),
                    mp.Vector3(0, 1, 0),
                    1,
                    0,
                ),
                mp.Vector3(1, 0, 0),
                eig_vol=dp_vol,
            )

        invalid_normal_order = mp.DiffractedPlanewave(
            (1, 0, 0),
            mp.Vector3(0, 1, 0),
            1,
            0,
        )
        with mock.patch.object(
            mp, "get_eigenmode_coefficients_and_kpoints"
        ) as lower_coefficients:
            with self.assertRaisesRegex(
                RuntimeError, "parallel to its source plane"
            ):
                sim.get_eigenmode_coefficients(
                    guard_monitor,
                    invalid_normal_order,
                    direction=mp.X,
                )
            lower_coefficients.assert_not_called()

        invalid_source = mp.EigenModeSource(
            src=mp.GaussianSource(self.fcen, fwidth=self.df),
            center=mp.Vector3(),
            size=mp.Vector3(0, 1, 0),
            direction=mp.X,
            eig_band=invalid_normal_order,
        )
        fake_fields = mock.Mock()
        fake_simulation = mock.Mock(
            dimensions=2,
            is_cylindrical=False,
            fields=fake_fields,
        )
        with self.assertRaisesRegex(RuntimeError, "parallel to its source plane"):
            invalid_source.add_source(fake_simulation)
        fake_fields.add_eigenmode_source.assert_not_called()

        # A reciprocal order at zero Bloch k is not a generic zero-frequency MPB band.  With no
        # normal k it is grazing and therefore rejected as non-normalizable, but it must not enter
        # the generic k=0 constant-band reduction.
        with self.assertRaisesRegex(RuntimeError, "normalizable mode"):
            sim.get_eigenmode(
                0,
                mp.X,
                dp_vol,
                dp_band,
                mp.Vector3(),
                eig_vol=dp_vol,
                match_frequency=False,
                resolution=10,
            )

        with self.assertRaisesRegex(RuntimeError, "band index must be positive"):
            sim.get_eigenmode(
                self.fcen,
                mp.X,
                eig_vol,
                0,
                mp.Vector3(self.ng * self.fcen, 0, 0),
            )
        with self.assertRaisesRegex(RuntimeError, "resolution"):
            sim.get_eigenmode(
                self.fcen,
                mp.X,
                eig_vol,
                1,
                mp.Vector3(self.ng * self.fcen, 0, 0),
                resolution=math.inf,
            )
        with mock.patch.object(mp, "_get_eigenmode", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "normalizable mode"):
                sim.get_eigenmode(
                    self.fcen,
                    mp.X,
                    eig_vol,
                    1,
                    mp.Vector3(self.ng * self.fcen, 0, 0),
                )
        with mock.patch.object(mp, "_get_eigenmode_dp", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "normalizable mode"):
                sim.get_eigenmode(
                    self.fcen,
                    mp.X,
                    dp_vol,
                    dp_band,
                    mp.Vector3(1, 0, 0),
                    eig_vol=dp_vol,
                )

    @parameterized.parameterized.expand([(0.0,), (10.7,)])
    def test_binary_grating_oblique(self, theta):
        """Verifies energy conservation."""

        if theta == 0:
            symmetries = [mp.Mirror(mp.Y)]
            eig_parity = mp.ODD_Z + mp.EVEN_Y
            k = mp.Vector3()
        else:
            symmetries = []
            eig_parity = mp.ODD_Z
            # Wavevector of incident planewave in source medium.
            # Plane of incidence is XY. Rotation angle is counterclockwise
            # about Z axis with 0° along +X axis.
            k = mp.Vector3(self.fcen * self.ng).rotate(
                mp.Vector3(0, 0, 1),
                math.radians(theta),
            )

        def pw_amp(k, x0):
            def _pw_amp(x):
                return cmath.exp(1j * 2 * math.pi * k.dot(x + x0))

            return _pw_amp

        src_cmpt = mp.Ez  # S polarization
        src_pt = mp.Vector3(-0.5 * self.sx + self.dpml, 0, 0)
        sources = [
            mp.Source(
                mp.GaussianSource(self.fcen, fwidth=self.df),
                component=src_cmpt,
                center=src_pt,
                size=mp.Vector3(0, self.sy, 0),
                amp_func=pw_amp(k, src_pt),
            )
        ]

        sim = mp.Simulation(
            resolution=self.resolution,
            cell_size=self.cell_size,
            boundary_layers=self.boundary_layers,
            k_point=k,
            default_material=self.glass,
            sources=sources,
            symmetries=symmetries,
        )

        refl_pt = mp.Vector3(
            -0.5 * self.sx + self.dpml + 0.5 * self.dsub,
            0,
            0,
        )
        refl_flux = sim.add_mode_monitor(
            self.fcen,
            0,
            1,
            mp.FluxRegion(center=refl_pt, size=mp.Vector3(0, self.sy, 0)),
        )

        stop_cond = mp.stop_when_fields_decayed(50.0, src_cmpt, refl_pt, 1e-8)
        sim.run(until_after_sources=stop_cond)

        input_flux = mp.get_fluxes(refl_flux)
        input_flux_data = sim.get_flux_data(refl_flux)

        sim.reset_meep()

        sim = mp.Simulation(
            resolution=self.resolution,
            cell_size=self.cell_size,
            boundary_layers=self.boundary_layers,
            geometry=self.geometry,
            k_point=k,
            sources=sources,
            symmetries=symmetries,
        )

        refl_flux = sim.add_mode_monitor(
            self.fcen,
            0,
            1,
            mp.FluxRegion(center=refl_pt, size=mp.Vector3(0, self.sy, 0)),
        )

        sim.load_minus_flux_data(refl_flux, input_flux_data)

        tran_pt = mp.Vector3(
            0.5 * self.sx - self.dpml - 0.5 * self.dpad,
            0,
            0,
        )
        tran_flux = sim.add_mode_monitor(
            self.fcen,
            0,
            1,
            mp.FluxRegion(center=tran_pt, size=mp.Vector3(0, self.sy, 0)),
        )

        sim.run(until_after_sources=stop_cond)

        # number of reflected orders
        m_plus = int(np.floor((self.fcen * self.ng - k.y) * self.gp))
        m_minus = int(np.ceil((-self.fcen * self.ng - k.y) * self.gp))

        if theta == 0:
            orders = range(m_plus + 1)
        else:
            orders = range(m_minus, m_plus + 1)

        Rsum = 0
        for nm in orders:
            ky = k.y + nm / self.cell_size.y
            kx2 = (self.fcen * self.ng) ** 2 - ky**2
            if kx2 > 0:
                res = sim.get_eigenmode_coefficients(
                    refl_flux,
                    bands=[1],
                    kpoint_func=lambda *not_used: mp.Vector3(np.sqrt(kx2), ky, 0),
                    eig_parity=eig_parity,
                    direction=mp.NO_DIRECTION,
                    # We must specify the length of the line monitor to be ~0
                    # in the periodic direction in order for MPB to interpret
                    # its Bloch wavevector as a planewave wavevector.
                    eig_vol=mp.Volume(center=refl_pt, size=mp.Vector3(0, 1e-7, 0)),
                )
                R = abs(res.alpha[0, 0, 1]) ** 2 / input_flux[0]
                print(f"refl-order:, {nm:+d}, {R:.6f}")
                Rsum += 2 * R if (theta == 0 and nm != 0) else R

        # number of transmitted orders
        m_plus = int(np.floor((self.fcen - k.y) * self.gp))
        m_minus = int(np.ceil((-self.fcen - k.y) * self.gp))

        if theta == 0:
            orders = range(m_plus + 1)
        else:
            orders = range(m_minus, m_plus + 1)

        Tsum = 0
        for nm in orders:
            ky = k.y + nm / self.cell_size.y
            kx2 = self.fcen**2 - ky**2
            if kx2 > 0:
                res = sim.get_eigenmode_coefficients(
                    tran_flux,
                    bands=[1],
                    kpoint_func=lambda *not_used: mp.Vector3(np.sqrt(kx2), ky, 0),
                    eig_parity=eig_parity,
                    direction=mp.NO_DIRECTION,
                    # We must specify the length of the line monitor to be ~0
                    # in the periodic direction in order for MPB to interpret
                    # its Bloch wavevector as a planewave wavevector.
                    eig_vol=mp.Volume(center=tran_pt, size=mp.Vector3(0, 1e-7, 0)),
                )
                T = abs(res.alpha[0, 0, 0]) ** 2 / input_flux[0]
                print(f"tran-order:, {nm:+d}, {T:.6f}")
                Tsum += 2 * T if (theta == 0 and nm != 0) else T

        r_flux = mp.get_fluxes(refl_flux)
        t_flux = mp.get_fluxes(tran_flux)
        Rflux = -r_flux[0] / input_flux[0]
        Tflux = t_flux[0] / input_flux[0]

        print(f"refl:, {Rsum:.6f}, {Rflux:.6f}")
        print(f"tran:, {Tsum:.6f}, {Tflux:.6f}")
        print(f"sum:,  {Rsum + Tsum:.6f}, {Rflux + Tflux:.6f}")

        self.assertAlmostEqual(Rsum, Rflux, places=2)
        self.assertAlmostEqual(Tsum, Tflux, places=2)
        self.assertAlmostEqual(Rsum + Tsum, 1.00, places=2)

    @parameterized.parameterized.expand(
        [(13.2, "real/imag"), (17.7, "complex"), (21.2, "3d")]
    )
    def test_binary_grating_special_kz(self, theta, kz_2d):
        # rotation angle of incident planewave
        # counterclockwise (CCW) about Y axis, 0 degrees along +X axis
        theta_in = math.radians(theta)

        # k (in source medium) with correct length (plane of incidence: XZ)
        k = mp.Vector3(self.fcen * self.ng).rotate(mp.Vector3(0, 1, 0), theta_in)

        symmetries = [mp.Mirror(mp.Y)]

        def pw_amp(k, x0):
            def _pw_amp(x):
                return cmath.exp(1j * 2 * math.pi * k.dot(x + x0))

            return _pw_amp

        src_pt = mp.Vector3(-0.5 * self.sx + self.dpml, 0, 0)
        sources = [
            mp.Source(
                mp.GaussianSource(self.fcen, fwidth=self.df),
                component=mp.Ez,
                center=src_pt,
                size=mp.Vector3(0, self.sy, 0),
                amp_func=pw_amp(k, src_pt),
            )
        ]

        sim = mp.Simulation(
            resolution=self.resolution,
            cell_size=self.cell_size,
            boundary_layers=self.boundary_layers,
            k_point=k,
            default_material=self.glass,
            sources=sources,
            symmetries=symmetries,
            kz_2d=kz_2d,
        )

        refl_pt = mp.Vector3(-0.5 * self.sx + self.dpml + 0.5 * self.dsub, 0, 0)
        refl_flux = sim.add_mode_monitor(
            self.fcen,
            0,
            1,
            mp.FluxRegion(center=refl_pt, size=mp.Vector3(0, self.sy, 0)),
        )

        sim.run(until_after_sources=mp.stop_when_dft_decayed())

        input_flux = mp.get_fluxes(refl_flux)
        input_flux_data = sim.get_flux_data(refl_flux)

        sim.reset_meep()

        sim = mp.Simulation(
            resolution=self.resolution,
            cell_size=self.cell_size,
            boundary_layers=self.boundary_layers,
            geometry=self.geometry,
            k_point=k,
            sources=sources,
            symmetries=symmetries,
            kz_2d=kz_2d,
        )

        refl_flux = sim.add_mode_monitor(
            self.fcen,
            0,
            1,
            mp.FluxRegion(center=refl_pt, size=mp.Vector3(0, self.sy, 0)),
        )

        sim.load_minus_flux_data(refl_flux, input_flux_data)

        tran_pt = mp.Vector3(0.5 * self.sx - self.dpml - 0.5 * self.dpad, 0, 0)
        tran_flux = sim.add_mode_monitor(
            self.fcen,
            0,
            1,
            mp.FluxRegion(center=tran_pt, size=mp.Vector3(0, self.sy, 0)),
        )

        sim.run(until_after_sources=mp.stop_when_dft_decayed())

        # number of reflected orders
        nm_r = np.ceil(
            (np.sqrt((self.fcen * self.ng) ** 2 - k.z**2) - k.y) * self.gp
        ) - np.floor((-np.sqrt((self.fcen * self.ng) ** 2 - k.z**2) - k.y) * self.gp)
        nm_r = int(nm_r / 2)

        Rsum = 0
        for nm in range(nm_r):
            for S_pol in [False, True]:
                res = sim.get_eigenmode_coefficients(
                    refl_flux,
                    mp.DiffractedPlanewave(
                        [0, nm, 0],
                        mp.Vector3(1, 0, 0),
                        1 if S_pol else 0,
                        0 if S_pol else 1,
                    ),
                )
                r_coeffs = res.alpha
                Rmode = abs(r_coeffs[0, 0, 1]) ** 2 / input_flux[0]
                print(
                    "refl-order:, {}, {}, {}".format("s" if S_pol else "p", nm, Rmode)
                )
                Rsum += Rmode if nm == 0 else 2 * Rmode

        # number of transmitted orders
        nm_t = np.ceil((np.sqrt(self.fcen**2 - k.z**2) - k.y) * self.gp) - np.floor(
            (-np.sqrt(self.fcen**2 - k.z**2) - k.y) * self.gp
        )
        nm_t = int(nm_t / 2)

        Tsum = 0
        for nm in range(nm_t):
            for S_pol in [False, True]:
                res = sim.get_eigenmode_coefficients(
                    tran_flux,
                    mp.DiffractedPlanewave(
                        [0, nm, 0],
                        mp.Vector3(1, 0, 0),
                        1 if S_pol else 0,
                        0 if S_pol else 1,
                    ),
                )
                t_coeffs = res.alpha
                Tmode = abs(t_coeffs[0, 0, 0]) ** 2 / input_flux[0]
                print(
                    "tran-order:, {}, {}, {}".format("s" if S_pol else "p", nm, Tmode)
                )
                Tsum += Tmode if nm == 0 else 2 * Tmode

        r_flux = mp.get_fluxes(refl_flux)
        t_flux = mp.get_fluxes(tran_flux)
        Rflux = -r_flux[0] / input_flux[0]
        Tflux = t_flux[0] / input_flux[0]

        print(f"refl:, {Rsum}, {Rflux}")
        print(f"tran:, {Tsum}, {Tflux}")
        print(f"sum:,  {Rsum + Tsum}, {Rflux + Tflux}")

        self.assertAlmostEqual(Rsum, Rflux, places=2)
        self.assertAlmostEqual(Tsum, Tflux, places=2)
        self.assertAlmostEqual(Rsum + Tsum, 1.00, places=2)

        # Verify eigenmode_cache produces identical results on repeated calls.
        # The first call creates the cache; the second call reuses it.
        for nm in range(nm_t):
            for S_pol in [False, True]:
                dp = mp.DiffractedPlanewave(
                    [0, nm, 0],
                    mp.Vector3(1, 0, 0),
                    1 if S_pol else 0,
                    0 if S_pol else 1,
                )
                res1 = sim.get_eigenmode_coefficients(tran_flux, dp)
                res2 = sim.get_eigenmode_coefficients(tran_flux, dp)
                power1 = abs(res1.alpha[0, 0, 0]) ** 2
                power2 = abs(res2.alpha[0, 0, 0]) ** 2
                if max(power1, power2) > 1e-20:
                    self.assertAlmostEqual(
                        power2 / power1,
                        1.0,
                        places=5,
                        msg=f"eigenmode_cache mismatch for order {nm}, "
                        f"{'S' if S_pol else 'P'}-pol",
                    )


if __name__ == "__main__":
    unittest.main()
