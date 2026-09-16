import argparse
import contextlib
import hashlib
import json
import pathlib

import gdstk
from matplotlib import pyplot as plt
import meep as mp
import numpy as np

# core and cladding materials
Si = mp.Medium(index=3.4)
SiO2 = mp.Medium(index=1.4)

# layer numbers for GDS file
RING_LAYER = 0
SOURCE0_LAYER = 1
SOURCE1_LAYER = 2
MONITOR_LAYER = 3
SIMULATION_LAYER = 4

resolution = 50  # pixels/μm
dpml = 1  # thickness of PML
zmin = 0  # minimum z value of simulation domain (0 for 2D)
zmax = 0  # maximum z value of simulation domain (0 for 2D)

# Frozen from an independent resolution-40, 315-time-unit CPU refinement of
# the same analytic ring.  The bounded validation run below uses resolution 20.
RING_REFINED_WAVELENGTH_REFERENCE = np.asarray(
    [1.5919030793463915, 1.5262772982784945]
)
RING_GDS_COORDINATE_SHA256 = bytes(
    [
        254, 114, 236, 152, 245, 118, 49, 204,
        102, 104, 151, 191, 215, 45, 224, 23,
        70, 56, 196, 99, 185, 224, 206, 242,
        77, 238, 95, 85, 174, 163, 207, 144,
    ]
)


def gds_coordinate_sha256(cell, layers):
    """Hash every vertex modulo polygon order, cycle start, and direction."""
    def canonical_polygon(points):
        coordinates = tuple((float(x), float(y)) for x, y in points)
        candidates = []
        for sequence in (coordinates, tuple(reversed(coordinates))):
            candidates.extend(
                sequence[index:] + sequence[:index]
                for index in range(len(sequence))
            )
        return min(candidates)

    records = []
    for layer in layers:
        polygons = [
            canonical_polygon(polygon.points)
            for polygon in cell.get_polygons(layer=layer, datatype=0)
        ]
        polygons.sort()
        records.append([layer, polygons])
    payload = json.dumps(
        records, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def validate_ring_gds(cell, radius, width):
    """Validate every generated GDS layer and the annulus discretization."""
    layers = (RING_LAYER, SOURCE0_LAYER, SOURCE1_LAYER, MONITOR_LAYER, SIMULATION_LAYER)
    polygons_by_layer = [cell.get_polygons(layer=layer, datatype=0) for layer in layers]
    counts = np.asarray([len(polygons) for polygons in polygons_by_layer], dtype=int)
    if not np.array_equal(counts, np.ones(len(layers), dtype=int)):
        raise RuntimeError("ring GDS layer polygon counts differ from the fixture")
    bounds = []
    areas = []
    vertices = []
    for polygons in polygons_by_layer:
        polygon = polygons[0]
        points = np.asarray(polygon.points)
        bounds.append(
            [
                float(np.min(points[:, 0])),
                float(np.min(points[:, 1])),
                float(np.max(points[:, 0])),
                float(np.max(points[:, 1])),
            ]
        )
        areas.append(float(polygon.area()))
        vertices.append(len(points))
    bounds = np.asarray(bounds, dtype=float)
    areas = np.asarray(areas, dtype=float)
    vertices = np.asarray(vertices, dtype=int)
    outer = radius + width / 2
    expected_area = np.pi * (outer**2 - (radius - width / 2) ** 2)
    expected_bounds = np.asarray(
        [
            [-outer, -outer, outer, outer],
            [radius - width, 0.0, radius + width, 0.0],
            [-radius - width, 0.0, -radius + width, 0.0],
            [radius - width / 2, 0.0, radius + width / 2, 0.0],
            [-outer - 2, -outer - 2, outer + 2, outer + 2],
        ]
    )
    if not np.allclose(bounds[1:], expected_bounds[1:], rtol=0, atol=1e-9):
        raise RuntimeError("ring GDS source/monitor/cell bounds differ from the fixture")
    if not np.allclose(bounds[0], expected_bounds[0], rtol=0, atol=0.02):
        raise RuntimeError("ring GDS annulus bounds differ from the analytic ring")
    area_relative_error = float(abs(areas[0] - expected_area) / expected_area)
    expected_other_areas = np.asarray([0.0, 0.0, 0.0, (2 * (outer + 2)) ** 2])
    if not np.allclose(areas[1:], expected_other_areas, rtol=0, atol=1e-9):
        raise RuntimeError("ring GDS non-annulus layer areas differ from the fixture")
    if area_relative_error > 0.01 or not np.array_equal(
        vertices, [56, 4, 4, 4, 4]
    ):
        raise RuntimeError("ring GDS annulus tessellation is not faithful")
    coordinate_digest = gds_coordinate_sha256(cell, layers)
    if coordinate_digest != RING_GDS_COORDINATE_SHA256:
        raise RuntimeError("ring GDS polygon coordinates differ from the fixture")
    return counts, bounds, areas, vertices, area_relative_error, coordinate_digest


def validate_ring_modes(wavelengths, quality_factors):
    """Apply refinement, free-spectral-range, and Q-hierarchy oracles."""
    wvls = np.asarray(wavelengths, dtype=float)
    qs = np.asarray(quality_factors, dtype=float)
    if (
        wvls.shape != (3,)
        or qs.shape != (3,)
        or not np.all(np.isfinite(wvls))
        or not np.all(np.isfinite(qs))
        or not np.all(np.diff(wvls) < 0)
        or np.any(qs <= 0)
    ):
        raise RuntimeError("ring modes have invalid shape, order, or values")
    refinement_relative_errors = np.abs(
        wvls[:2] - RING_REFINED_WAVELENGTH_REFERENCE
    ) / RING_REFINED_WAVELENGTH_REFERENCE
    frequency_spacings = np.diff(1 / wvls)
    spacing_ratio = float(frequency_spacings[1] / frequency_spacings[0])
    q_ratios = qs[:-1] / qs[1:]
    if np.max(refinement_relative_errors) > 0.012:
        raise RuntimeError("ring wavelengths do not close to the refined reference")
    if not (0.95 <= spacing_ratio <= 1.05):
        raise RuntimeError("ring resonances do not have a consistent free spectral range")
    if not (4e4 <= qs[0] <= 2e5 and 8e3 <= qs[1] <= 4e4 and 3e3 <= qs[2] <= 1.5e4):
        raise RuntimeError("ring quality factors differ from the bounded fixture")
    if q_ratios[0] <= 3 or q_ratios[1] <= 1.5:
        raise RuntimeError("ring quality-factor hierarchy is not resolved")
    return refinement_relative_errors, frequency_spacings, spacing_ratio, q_ratios


def get_gds_cell(fname):
    """Returns the (single) top-level cell of the GDS file `fname`."""
    return gdstk.read_gds(fname).top_level()[0]


def get_gds_prisms(material, cell, layer, datatype=0, zmin=0.0, zmax=0.0):
    """Returns a list of `mp.Prism`s, one for each polygon on (`layer`, `datatype`)."""
    prisms = []
    for poly in cell.get_polygons(layer=layer, datatype=datatype):
        vertices = [mp.Vector3(x, y, zmin) for x, y in poly.points]
        prisms.append(
            mp.Prism(
                vertices,
                height=zmax - zmin,
                axis=mp.Vector3(0, 0, 1),
                material=material,
            )
        )
    return prisms


def get_gds_vol(cell, layer, datatype=0, zmin=0.0, zmax=0.0):
    """Returns an `mp.Volume` spanning the bounding box of (`layer`, `datatype`)."""
    polygons = cell.get_polygons(layer=layer, datatype=datatype)
    xs = [x for poly in polygons for x, y in poly.points]
    ys = [y for poly in polygons for x, y in poly.points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    center = mp.Vector3(0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax))
    size = mp.Vector3(xmax - xmin, ymax - ymin, zmax - zmin)
    dims = 2 if (zmin == 0 and zmax == 0) else 3
    return mp.Volume(center=center, size=size, dims=dims)


def create_ring_gds(radius, width):
    lib = gdstk.Library()
    ring_cell = lib.new_cell(f"ring_resonator_r{radius}_w{width}")

    # Draw the ring
    ring_cell.add(
        gdstk.ellipse(
            (0, 0),
            radius + width / 2,
            inner_radius=radius - width / 2,
            layer=RING_LAYER,
        )
    )

    # Draw the first source
    ring_cell.add(
        gdstk.rectangle((radius - width, 0), (radius + width, 0), layer=SOURCE0_LAYER)
    )

    # Draw the second source
    ring_cell.add(
        gdstk.rectangle((-radius - width, 0), (-radius + width, 0), layer=SOURCE1_LAYER)
    )

    # Draw the monitor location
    ring_cell.add(
        gdstk.rectangle(
            (radius - width / 2, 0), (radius + width / 2, 0), layer=MONITOR_LAYER
        )
    )

    # Draw the simulation domain
    pad = 2  # padding between waveguide and edge of PML
    ring_cell.add(
        gdstk.rectangle(
            (-radius - width / 2 - pad, -radius - width / 2 - pad),
            (radius + width / 2 + pad, radius + width / 2 + pad),
            layer=SIMULATION_LAYER,
        )
    )

    filename = f"ring_r{radius}_w{width}.gds"
    lib.write_gds(filename)

    return filename


@contextlib.contextmanager
def transient_generated_gds(filename):
    """Remove one generated GDS file after either success or failure."""
    path = pathlib.Path(filename)
    try:
        yield filename
    finally:
        path.unlink(missing_ok=True)


def find_modes(
    filename,
    wvl=1.55,
    bw=0.05,
    *,
    resolution_override=None,
    until_after_sources=100,
    plot_fields=True,
):
    # Read in the ring structure using gdstk
    gds_cell = get_gds_cell(filename)

    geometry = get_gds_prisms(Si, gds_cell, RING_LAYER, zmin=-100, zmax=100)

    cell = get_gds_vol(gds_cell, SIMULATION_LAYER, zmin=zmin, zmax=zmax)

    src_vol0 = get_gds_vol(gds_cell, SOURCE0_LAYER, zmin=zmin, zmax=zmax)
    src_vol1 = get_gds_vol(gds_cell, SOURCE1_LAYER, zmin=zmin, zmax=zmax)

    mon_vol = get_gds_vol(gds_cell, MONITOR_LAYER, zmin=zmin, zmax=zmax)

    fcen = 1 / wvl
    df = bw * fcen

    src = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=mp.Hz,
            volume=src_vol0,
        ),
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=mp.Hz,
            volume=src_vol1,
            amplitude=-1,
        ),
    ]

    sim = mp.Simulation(
        cell_size=cell.size,
        geometry=geometry,
        sources=src,
        resolution=resolution if resolution_override is None else resolution_override,
        boundary_layers=[mp.PML(dpml)],
        default_material=SiO2,
    )

    h = mp.Harminv(mp.Hz, mon_vol.center, fcen, df)

    sim.run(mp.after_sources(h), until_after_sources=until_after_sources)

    if plot_fields:
        fig, ax = plt.subplots()
        sim.plot2D(ax=ax, fields=mp.Hz, eps_parameters={"contour": True})
        fig.savefig("ring_fields.png", bbox_inches="tight", dpi=150)

    wvl = np.array([1 / m.freq for m in h.modes])
    Q = np.array([m.Q for m in h.modes])

    sim.reset_meep()

    return wvl, Q


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation",
        action="store_true",
        help="run a bounded deterministic GDS/FDTD/Harminv oracle",
    )
    args = parser.parse_args()

    if args.validation:
        mp.verbosity(0)
        radius = 1.5
        width = 0.4
        validation_resolution = 20
        center_wavelength = 1.55
        bandwidth = 0.1
        with transient_generated_gds(create_ring_gds(radius, width)) as filename:
            gds_cell = get_gds_cell(filename)
            (
                gds_counts,
                gds_bounds,
                gds_areas,
                gds_vertices,
                gds_area_relative_error,
                gds_coordinate_digest,
            ) = validate_ring_gds(gds_cell, radius, width)
            wvls, Qs = find_modes(
                filename,
                center_wavelength,
                bandwidth,
                resolution_override=validation_resolution,
                until_after_sources=100,
                plot_fields=False,
            )
            (
                refinement_relative_errors,
                frequency_spacings,
                frequency_spacing_ratio,
                quality_factor_ratios,
            ) = validate_ring_modes(wvls, Qs)
            metrics = {
                "schema_version": 1,
                "resolution": validation_resolution,
                "radius": radius,
                "width": width,
                "wavelengths": wvls.tolist(),
                "quality_factors": Qs.tolist(),
                "mode_count": int(len(wvls)),
                "refined_wavelength_reference": RING_REFINED_WAVELENGTH_REFERENCE.tolist(),
                "refinement_relative_errors": refinement_relative_errors.tolist(),
                "frequency_spacings": frequency_spacings.tolist(),
                "frequency_spacing_ratio": frequency_spacing_ratio,
                "quality_factor_ratios": quality_factor_ratios.tolist(),
                "gds_polygon_counts": gds_counts.tolist(),
                "gds_layer_bounds": gds_bounds.tolist(),
                "gds_layer_areas": gds_areas.tolist(),
                "gds_vertex_counts": gds_vertices.tolist(),
                "gds_annulus_area_relative_error": gds_area_relative_error,
                "gds_coordinate_sha256_bytes": list(gds_coordinate_digest),
            }
            print(
                "gpmeep-ring-gds-metrics:"
                + json.dumps(metrics, sort_keys=True, allow_nan=False)
            )
    else:
        filename = create_ring_gds(2.0, 0.5)
        wvls, Qs = find_modes(filename, 1.55, 0.05)
    for w, Q in zip(wvls, Qs):
        print(f"mode: {w}, {Q}")
