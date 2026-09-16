#!/usr/bin/env python3
"""Fail-closed synthetic tests for the bounded GDS example physics gates."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest

import numpy as np


REPO = pathlib.Path(__file__).resolve().parents[3]


def load_example(name: str):
    meep = types.ModuleType("meep")
    meep.Medium = lambda **kwargs: ("medium", kwargs)
    gdstk = types.ModuleType("gdstk")
    saved = {key: sys.modules.get(key) for key in ("meep", "gdstk")}
    sys.modules["meep"] = meep
    sys.modules["gdstk"] = gdstk
    try:
        path = REPO / "python" / "examples" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"{name}_physics_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


class Polygon:
    def __init__(self, bounds, area: float, vertices: int = 4):
        xmin, ymin, xmax, ymax = bounds
        corners = np.asarray(
            [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
            dtype=float,
        )
        self.points = (
            np.asarray([[xmin, ymin], [xmax, ymax]], dtype=float)
            if vertices == 2
            else np.resize(corners, (vertices, 2))
        )
        self._area = area

    def area(self):
        return self._area


class Cell:
    def __init__(self, layers):
        self.layers = layers

    def get_polygons(self, *, layer, datatype):
        if datatype != 0:
            return []
        return self.layers.get(layer, [])


class GdsExamplePhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coupler = load_example("coupler")
        cls.ring = load_example("ring_gds")

    def test_coupler_rejects_reflective_missing_and_wrong_port_vectors(self):
        self.assertAlmostEqual(
            self.coupler.validate_coupler_powers(
                [2.765180015948767e-7, 0.9731013433790336, 0.00884006620774288]
            ),
            0.9819416861047782,
        )
        invalid = (
            [100.0, 0.0, 0.5],
            [0.0, 0.99, 0.0],
            [0.0, 0.1, 0.9],
            [0.0, 1.2, 0.01],
        )
        for powers in invalid:
            with self.subTest(powers=powers), self.assertRaises(RuntimeError):
                self.coupler.validate_coupler_powers(powers)

    def test_coupler_gds_topology_is_exact(self):
        layers = {}
        for layer, count, vertex_counts, bounds, area in zip(
            self.coupler.COUPLER_GDS_LAYERS,
            self.coupler.COUPLER_GDS_POLYGON_COUNTS,
            self.coupler.COUPLER_GDS_VERTEX_COUNTS,
            self.coupler.COUPLER_GDS_BOUNDS,
            self.coupler.COUPLER_GDS_AREAS,
        ):
            layers[layer] = [Polygon(bounds, area, vertex_counts[0])] + [
                Polygon(bounds, 0.0, vertex_counts[index])
                for index in range(1, count)
            ]
        cell = Cell(layers)
        original_digest = self.coupler.COUPLER_GDS_COORDINATE_SHA256
        self.coupler.COUPLER_GDS_COORDINATE_SHA256 = (
            self.coupler.gds_coordinate_sha256(
                cell, self.coupler.COUPLER_GDS_LAYERS
            )
        )
        try:
            self.coupler.validate_coupler_gds(Cell(layers))
            layers[31][1].points = np.roll(
                layers[31][1].points[::-1], 7, axis=0
            )
            self.coupler.validate_coupler_gds(Cell(layers))
            # Preserve the summary bounds, count, vertex count, and synthetic
            # area while changing one non-unique interior coordinate.
            layers[31][1].points[5, 0] += 0.01
            with self.assertRaises(RuntimeError):
                self.coupler.validate_coupler_gds(Cell(layers))
        finally:
            self.coupler.COUPLER_GDS_COORDINATE_SHA256 = original_digest

    def test_ring_rejects_broad_but_unphysical_modes(self):
        reference_wavelengths = [
            1.6045986857896173,
            1.5399322552400123,
            1.4799263109188725,
        ]
        reference_q = [87798.0954556765, 15909.50631002931, 6312.777797030292]
        self.ring.validate_ring_modes(reference_wavelengths, reference_q)
        invalid = (
            ([1.60, 1.55, 1.50], [1001, 1001, 1001]),
            ([1.6046, 1.5399, 1.4799], [50000, 20000, 18000]),
            ([1.6046, 1.5399, 1.46], reference_q),
        )
        for wavelengths, qs in invalid:
            with self.subTest(wavelengths=wavelengths, qs=qs), self.assertRaises(
                RuntimeError
            ):
                self.ring.validate_ring_modes(wavelengths, qs)

    def test_ring_gds_topology_closes_to_analytic_annulus(self):
        radius = 1.5
        width = 0.4
        outer = radius + width / 2
        expected_bounds = (
            [-1.69, -1.698, 1.7, 1.698],
            [radius - width, 0, radius + width, 0],
            [-radius - width, 0, -radius + width, 0],
            [radius - width / 2, 0, radius + width / 2, 0],
            [-outer - 2, -outer - 2, outer + 2, outer + 2],
        )
        layers = {
            layer: [
                Polygon(
                    bounds,
                    3.754473 if layer == self.ring.RING_LAYER else 0.0,
                    56 if layer == self.ring.RING_LAYER else 4,
                )
            ]
            for layer, bounds in zip(
                (
                    self.ring.RING_LAYER,
                    self.ring.SOURCE0_LAYER,
                    self.ring.SOURCE1_LAYER,
                    self.ring.MONITOR_LAYER,
                    self.ring.SIMULATION_LAYER,
                ),
                expected_bounds,
            )
        }
        layers[self.ring.SIMULATION_LAYER][0]._area = (2 * (outer + 2)) ** 2
        cell = Cell(layers)
        original_digest = self.ring.RING_GDS_COORDINATE_SHA256
        self.ring.RING_GDS_COORDINATE_SHA256 = self.ring.gds_coordinate_sha256(
            cell,
            (
                self.ring.RING_LAYER,
                self.ring.SOURCE0_LAYER,
                self.ring.SOURCE1_LAYER,
                self.ring.MONITOR_LAYER,
                self.ring.SIMULATION_LAYER,
            ),
        )
        try:
            self.ring.validate_ring_gds(cell, radius, width)
            ring_points = layers[self.ring.RING_LAYER][0].points
            layers[self.ring.RING_LAYER][0].points = np.roll(
                ring_points[::-1], 9, axis=0
            )
            self.ring.validate_ring_gds(cell, radius, width)
            # Summary geometry is unchanged; only the full coordinate digest
            # can observe this non-unique vertex perturbation.
            layers[self.ring.RING_LAYER][0].points[5, 0] += 0.01
            with self.assertRaises(RuntimeError):
                self.ring.validate_ring_gds(cell, radius, width)
        finally:
            self.ring.RING_GDS_COORDINATE_SHA256 = original_digest

    def test_ring_transient_gds_is_removed_after_success(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = pathlib.Path(temporary_directory) / "generated.gds"
            path.write_bytes(b"fixture")
            with self.ring.transient_generated_gds(path) as filename:
                self.assertEqual(filename, path)
                self.assertTrue(path.is_file())
            self.assertFalse(path.exists())

    def test_ring_transient_gds_is_removed_after_failure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = pathlib.Path(temporary_directory) / "generated.gds"
            path.write_bytes(b"fixture")
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                with self.ring.transient_generated_gds(path):
                    self.assertTrue(path.is_file())
                    raise RuntimeError("injected failure")
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
