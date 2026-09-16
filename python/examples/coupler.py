import argparse
import hashlib
import json
from pathlib import Path

import gdstk
import meep as mp
import numpy as np


gds_file = "coupler.gds"
CELL_LAYER = 0
PORT1_LAYER = 1
PORT2_LAYER = 2
PORT3_LAYER = 3
PORT4_LAYER = 4
SOURCE_LAYER = 5
UPPER_BRANCH_LAYER = 31
LOWER_BRANCH_LAYER = 32
default_d = 0.3

t_oxide = 1.0
t_Si = 0.22
t_air = 0.78

dpml = 1
cell_thickness = dpml + t_oxide + t_Si + t_air + dpml

oxide = mp.Medium(epsilon=2.25)
silicon = mp.Medium(epsilon=12)

fcen = 1 / 1.55
df = 0.2 * fcen

COUPLER_GDS_LAYERS = (0, 1, 2, 3, 4, 5, 31, 32)
COUPLER_GDS_POLYGON_COUNTS = (1, 1, 1, 1, 1, 1, 3, 3)
COUPLER_GDS_VERTEX_COUNTS = (
    (4,),
    (2,),
    (2,),
    (2,),
    (2,),
    (2,),
    (4, 174, 174),
    (4, 174, 174),
)
COUPLER_GDS_BOUNDS = np.asarray(
    [
        [-17.2, -4.0, 17.2, 4.0],
        [-14.5, 1.4, -14.5, 3.4],
        [-14.5, -3.4, -14.5, -1.4],
        [14.5, 1.4, 14.5, 3.4],
        [14.5, -3.4, 14.5, -1.4],
        [-14.7, 1.4, -14.7, 3.4],
        [-17.2, 0.15, 17.2, 2.65],
        [-17.2, -2.65, 17.2, -0.15],
    ]
)
COUPLER_GDS_AREAS = np.asarray(
    [275.2, 0.0, 0.0, 0.0, 0.0, 0.0, 17.44032, 17.44032]
)
COUPLER_GDS_COORDINATE_SHA256 = bytes(
    [
        176, 167, 220, 115, 255, 28, 238, 4,
        0, 84, 16, 208, 30, 113, 93, 168,
        224, 140, 72, 16, 71, 247, 145, 53,
        184, 127, 123, 179, 195, 46, 210, 58,
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


def validate_coupler_gds(cell):
    """Validate the checked-in coupler's complete layer topology."""
    counts = []
    areas = []
    bounds = []
    vertex_counts = []
    for layer in COUPLER_GDS_LAYERS:
        polygons = cell.get_polygons(layer=layer, datatype=0)
        counts.append(len(polygons))
        vertex_counts.append(tuple(sorted(len(polygon.points) for polygon in polygons)))
        if not polygons:
            raise RuntimeError(f"coupler GDS layer {layer} is empty")
        points = np.concatenate([np.asarray(polygon.points) for polygon in polygons])
        bounds.append(
            [
                float(np.min(points[:, 0])),
                float(np.min(points[:, 1])),
                float(np.max(points[:, 0])),
                float(np.max(points[:, 1])),
            ]
        )
        areas.append(float(sum(polygon.area() for polygon in polygons)))
    counts_array = np.asarray(counts, dtype=int)
    areas_array = np.asarray(areas, dtype=float)
    bounds_array = np.asarray(bounds, dtype=float)
    if not np.array_equal(counts_array, COUPLER_GDS_POLYGON_COUNTS):
        raise RuntimeError("coupler GDS layer polygon counts differ from the fixture")
    if tuple(vertex_counts) != COUPLER_GDS_VERTEX_COUNTS:
        raise RuntimeError("coupler GDS layer vertex counts differ from the fixture")
    if not np.allclose(bounds_array, COUPLER_GDS_BOUNDS, rtol=0, atol=1e-9):
        raise RuntimeError("coupler GDS layer bounds differ from the fixture")
    if not np.allclose(areas_array, COUPLER_GDS_AREAS, rtol=0, atol=1e-8):
        raise RuntimeError("coupler GDS layer areas differ from the fixture")
    coordinate_digest = gds_coordinate_sha256(cell, COUPLER_GDS_LAYERS)
    if coordinate_digest != COUPLER_GDS_COORDINATE_SHA256:
        raise RuntimeError("coupler GDS polygon coordinates differ from the fixture")
    return counts_array, bounds_array, areas_array, vertex_counts, coordinate_digest


def validate_coupler_powers(normalized_powers):
    """Apply topology-specific passive four-port gates to (p2,p3,p4)."""
    powers = np.asarray(normalized_powers, dtype=float)
    if powers.shape != (3,) or not np.all(np.isfinite(powers)):
        raise RuntimeError("coupler normalized powers are invalid")
    reflected, through, coupled = powers
    port_power_sum = float(np.sum(powers))
    if np.any(powers < 0) or not (0.90 <= port_power_sum <= 1.05):
        raise RuntimeError("coupler violates its passive port-power closure")
    if reflected > 1e-3:
        raise RuntimeError("coupler reflection is inconsistent with the fixture")
    if not (0.85 <= through <= 1.02):
        raise RuntimeError("coupler through-port power is inconsistent with the fixture")
    if not (0.002 <= coupled <= 0.03):
        raise RuntimeError("coupler cross-port power is inconsistent with the fixture")
    return port_power_sum


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


def main(args):
    if args.validation:
        mp.verbosity(0)
    cell_zmax = 0.5 * cell_thickness if args.three_d else 0
    cell_zmin = -0.5 * cell_thickness if args.three_d else 0
    si_zmax = 0.5 * t_Si if args.three_d else 10
    si_zmin = -0.5 * t_Si if args.three_d else -10

    # read cell size, volumes for source region and flux monitors,
    # and coupler geometry from the GDS file using gdstk
    gds_cell = get_gds_cell(args.gds_file)
    if args.validation:
        (
            gds_counts,
            gds_bounds,
            gds_areas,
            gds_vertices,
            gds_coordinate_digest,
        ) = validate_coupler_gds(gds_cell)

    upper_branch = get_gds_prisms(
        silicon, gds_cell, UPPER_BRANCH_LAYER, zmin=si_zmin, zmax=si_zmax
    )
    lower_branch = get_gds_prisms(
        silicon, gds_cell, LOWER_BRANCH_LAYER, zmin=si_zmin, zmax=si_zmax
    )

    cell = get_gds_vol(gds_cell, CELL_LAYER, zmin=cell_zmin, zmax=cell_zmax)
    p1 = get_gds_vol(gds_cell, PORT1_LAYER, zmin=si_zmin, zmax=si_zmax)
    p2 = get_gds_vol(gds_cell, PORT2_LAYER, zmin=si_zmin, zmax=si_zmax)
    p3 = get_gds_vol(gds_cell, PORT3_LAYER, zmin=si_zmin, zmax=si_zmax)
    p4 = get_gds_vol(gds_cell, PORT4_LAYER, zmin=si_zmin, zmax=si_zmax)
    src_vol = get_gds_vol(gds_cell, SOURCE_LAYER, zmin=si_zmin, zmax=si_zmax)

    # displace upper and lower branches of coupler (as well as source and flux regions)
    if args.d != default_d:
        delta_y = 0.5 * (args.d - default_d)
        delta = mp.Vector3(y=delta_y)
        p1.center += delta
        p2.center -= delta
        p3.center += delta
        p4.center -= delta
        src_vol.center += delta
        cell.size += 2 * delta
        for polygon_index in range(len(lower_branch)):
            lower_branch[polygon_index].center -= delta
            for vertex_index in range(len(lower_branch[polygon_index].vertices)):
                lower_branch[polygon_index].vertices[vertex_index] -= delta
        for polygon_index in range(len(upper_branch)):
            upper_branch[polygon_index].center += delta
            for vertex_index in range(len(upper_branch[polygon_index].vertices)):
                upper_branch[polygon_index].vertices[vertex_index] += delta

    geometry = upper_branch + lower_branch

    if args.three_d:
        oxide_center = mp.Vector3(z=-0.5 * t_oxide)
        oxide_size = mp.Vector3(cell.size.x, cell.size.y, t_oxide)
        oxide_layer = [mp.Block(material=oxide, center=oxide_center, size=oxide_size)]
        geometry = geometry + oxide_layer

    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fcen, fwidth=df),
            volume=src_vol,
            eig_parity=mp.NO_PARITY if args.three_d else mp.EVEN_Y + mp.ODD_Z,
        )
    ]

    sim = mp.Simulation(
        resolution=args.res,
        cell_size=cell.size,
        boundary_layers=[mp.PML(dpml)],
        sources=sources,
        geometry=geometry,
    )

    mode1 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=p1))
    mode2 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=p2))
    mode3 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=p3))
    mode4 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=p4))

    sim.run(until_after_sources=args.until_after_sources)

    # S parameters
    p1_coeff = sim.get_eigenmode_coefficients(
        mode1, [1], eig_parity=mp.NO_PARITY if args.three_d else mp.EVEN_Y + mp.ODD_Z
    ).alpha[0, 0, 0]
    p2_coeff = sim.get_eigenmode_coefficients(
        mode2, [1], eig_parity=mp.NO_PARITY if args.three_d else mp.EVEN_Y + mp.ODD_Z
    ).alpha[0, 0, 1]
    p3_coeff = sim.get_eigenmode_coefficients(
        mode3, [1], eig_parity=mp.NO_PARITY if args.three_d else mp.EVEN_Y + mp.ODD_Z
    ).alpha[0, 0, 0]
    p4_coeff = sim.get_eigenmode_coefficients(
        mode4, [1], eig_parity=mp.NO_PARITY if args.three_d else mp.EVEN_Y + mp.ODD_Z
    ).alpha[0, 0, 0]

    # transmittance
    p2_trans = abs(p2_coeff) ** 2 / abs(p1_coeff) ** 2
    p3_trans = abs(p3_coeff) ** 2 / abs(p1_coeff) ** 2
    p4_trans = abs(p4_coeff) ** 2 / abs(p1_coeff) ** 2

    print(f"trans:, {args.d:.2f}, {p2_trans:.6f}, {p3_trans:.6f}, {p4_trans:.6f}")
    if args.validation:
        values = np.asarray(
            [
                abs(p1_coeff),
                abs(p2_coeff),
                abs(p3_coeff),
                abs(p4_coeff),
                p2_trans,
                p3_trans,
                p4_trans,
            ],
            dtype=float,
        )
        total_output = float(p3_trans + p4_trans)
        port_power_sum = validate_coupler_powers(values[4:])
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise RuntimeError("GDS coupler validation produced invalid S parameters")
        if abs(p1_coeff) <= 1e-10:
            raise RuntimeError("GDS coupler validation has no incident mode")
        metrics = {
            "schema_version": 1,
            "resolution": int(args.res),
            "final_timestep": int(sim.timestep()),
            "coefficient_magnitudes": values[:4].tolist(),
            "normalized_powers": values[4:].tolist(),
            "total_output_power": total_output,
            "reflected_power": float(p2_trans),
            "port_power_sum": port_power_sum,
            "output_imbalance": float(
                abs(p3_trans - p4_trans) / max(total_output, np.finfo(float).tiny)
            ),
            "gds_polygon_counts": gds_counts.tolist(),
            "gds_layer_bounds": gds_bounds.tolist(),
            "gds_layer_areas": gds_areas.tolist(),
            "gds_vertex_counts": gds_vertices,
            "gds_coordinate_sha256_bytes": list(gds_coordinate_digest),
        }
        print(
            "gpmeep-coupler-gds-metrics:"
            + json.dumps(metrics, sort_keys=True, allow_nan=False)
        )
    sim.reset_meep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-res", type=int, default=50, help="resolution (default: 50 pixels/um)"
    )
    parser.add_argument(
        "-d", type=float, default=0.1, help="branch separation (default: 0.1 um)"
    )
    parser.add_argument(
        "--three_d",
        action="store_true",
        default=False,
        help="d calculation? (default: False)",
    )
    parser.add_argument(
        "--gds-file",
        type=Path,
        default=Path(gds_file),
        help="input GDS file (default: coupler.gds in the working directory)",
    )
    parser.add_argument(
        "--until-after-sources",
        type=float,
        default=100,
        help="post-source simulation time (default: 100)",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="emit deterministic bounded S-parameter metrics",
    )
    args = parser.parse_args()
    main(args)
