"""Tutorial example for dipole current sources in cylindrical coordinates.

tutorial reference:
https://meep.readthedocs.io/en/latest/Python_Tutorials/Cylindrical_Coordinates/#nonaxisymmetric-dipole-sources
"""

import argparse
from collections.abc import Sequence
from typing import Any, Tuple

import meep as mp
import numpy as np


RESOLUTION_UM = 50
WAVELENGTH_UM = 1.0
N_SLAB = 2.4
SLAB_THICKNESS_UM = 0.7 * WAVELENGTH_UM / N_SLAB

# The published example below intentionally uses a large radial cell and an
# adaptive DFT-decay stop.  The validation profile is opt-in and uses a smaller
# physical problem plus fixed post-source checkpoints so CPU/CUDA comparisons
# have deterministic timesteps.  None of these values affect the default run.
VALIDATION_RESOLUTION_UM = 16
VALIDATION_PML_UM = 0.5
VALIDATION_PADDING_UM = 0.5
VALIDATION_RADIUS_UM = 3.0
VALIDATION_SOURCE_FWIDTH_FRACTION = 0.2
VALIDATION_DIPOLE_ZPOS = 0.5
VALIDATION_DIPOLE_RPOS_UM = 0.75
VALIDATION_M_VALUES = (-1, 0, 1)
VALIDATION_AFTER_SOURCES = (10.0, 30.0, 50.0)
VALIDATION_PM_SYMMETRY_LIMIT = 2e-3
VALIDATION_TAIL_RELATIVE_LIMIT = 0.08
VALIDATION_TAIL_CORRECTION_ABSOLUTE_LIMIT = 2e-6
VALIDATION_TAIL_CONTRACTION_LIMIT = 0.01
VALIDATION_MINIMUM_MODE_RATIO = 1e-3
VALIDATION_MINIMUM_M0_SIGNAL_CONTRAST = 0.02
VALIDATION_MAXIMUM_EXTRACTION_EFFICIENCY = 1.05


def _finite_vector(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """Returns a finite validation vector with an exact shape."""
    result = np.asarray(value)
    if result.shape != shape:
        raise RuntimeError(f"{name} has shape {result.shape}, expected {shape}")
    if not np.all(np.isfinite(result)):
        raise RuntimeError(f"{name} contains nonfinite values")
    return result


def point_dipole_validation_metrics(
    radiated_fluxes: np.ndarray,
    source_fluxes: np.ndarray,
    ldos_values: np.ndarray,
) -> dict[str, np.ndarray]:
    """Computes deterministic symmetry, tail, and modal-signal diagnostics.

    Rows are ordered as m=(-1,0,+1), and columns are the three increasing
    post-source checkpoints.  This helper has no simulation side effects so its
    acceptance logic can be tested with synthetic arrays.
    """
    expected_shape = (len(VALIDATION_M_VALUES), len(VALIDATION_AFTER_SOURCES))
    radiated = _finite_vector(
        radiated_fluxes, "validation radiated fluxes", expected_shape
    ).astype(float, copy=False)
    source = _finite_vector(
        source_fluxes, "validation source fluxes", expected_shape
    ).astype(float, copy=False)
    ldos = _finite_vector(ldos_values, "validation LDOS", expected_shape).astype(
        float, copy=False
    )
    signals = np.stack((radiated, source, ldos), axis=0)
    if np.min(signals) <= 0:
        raise RuntimeError("validation radiated flux, source flux, and LDOS must be positive")

    efficiencies = radiated / source
    if not np.all(np.isfinite(efficiencies)) or np.min(efficiencies) <= 0:
        raise RuntimeError("validation extraction efficiencies must be finite and positive")

    all_observables = np.concatenate((signals, efficiencies[np.newaxis, ...]), axis=0)
    tiny = np.finfo(float).tiny

    # Relative mismatch of m=-1 and m=+1 at every checkpoint.  The geometry is
    # real and axisymmetric, hence the two signs must carry identical power.
    pm_denominator = np.maximum(
        np.maximum(np.abs(all_observables[:, 0, :]), np.abs(all_observables[:, 2, :])),
        tiny,
    )
    pm_symmetry = (
        np.abs(all_observables[:, 0, :] - all_observables[:, 2, :])
        / pm_denominator
    )

    # The final correction is normalized to the final signal.  A separate
    # contraction ratio demonstrates whether the late correction is smaller
    # than the preceding correction instead of merely being absolutely small.
    early_to_middle = all_observables[:, :, 1] - all_observables[:, :, 0]
    middle_to_final = all_observables[:, :, 2] - all_observables[:, :, 1]
    tail_relative_changes = np.abs(middle_to_final) / np.maximum(
        np.abs(all_observables[:, :, 2]), tiny
    )
    tail_contraction_ratios = np.abs(middle_to_final) / np.maximum(
        np.abs(early_to_middle), tiny
    )

    final_signals = all_observables[:, :, -1]
    mode_ratios = final_signals / np.max(final_signals, axis=1, keepdims=True)
    pm_average = 0.5 * (final_signals[:, 0] + final_signals[:, 2])
    m0_contrast = np.abs(final_signals[:, 1] - pm_average) / np.maximum(
        np.maximum(np.abs(final_signals[:, 1]), np.abs(pm_average)), tiny
    )

    return {
        "efficiencies": efficiencies,
        "pm_symmetry_relative_errors": pm_symmetry,
        "tail_signed_corrections": middle_to_final,
        "tail_relative_changes": tail_relative_changes,
        "tail_contraction_ratios": tail_contraction_ratios,
        "final_mode_ratios": mode_ratios,
        "m0_contrast": m0_contrast,
    }


def validate_point_dipole_metrics(metrics: dict[str, np.ndarray]) -> None:
    """Applies broad FP32-friendly physical gates to synthetic-safe metrics."""
    symmetry = _finite_vector(
        metrics.get("pm_symmetry_relative_errors"), "m-sign symmetry", (4, 3)
    )
    tail_relative = _finite_vector(
        metrics.get("tail_relative_changes"), "tail relative changes", (4, 3)
    )
    tail_correction = _finite_vector(
        metrics.get("tail_signed_corrections"), "tail signed corrections", (4, 3)
    )
    tail_contraction = _finite_vector(
        metrics.get("tail_contraction_ratios"), "tail contraction ratios", (4, 3)
    )
    mode_ratios = _finite_vector(
        metrics.get("final_mode_ratios"), "final mode ratios", (4, 3)
    )
    m0_contrast = _finite_vector(
        metrics.get("m0_contrast"), "m=0 contrast", (4,)
    )
    efficiencies = _finite_vector(
        metrics.get("efficiencies"), "extraction efficiencies", (3, 3)
    )
    if np.max(symmetry) > VALIDATION_PM_SYMMETRY_LIMIT:
        raise RuntimeError("m=-1 and m=+1 power observables violate symmetry")
    if np.max(tail_relative) > VALIDATION_TAIL_RELATIVE_LIMIT:
        raise RuntimeError("point-dipole DFT/LDOS tail has not converged")
    if (
        np.max(np.abs(tail_correction))
        > VALIDATION_TAIL_CORRECTION_ABSOLUTE_LIMIT
    ):
        raise RuntimeError("point-dipole absolute late correction is too large")
    if np.max(tail_contraction) > VALIDATION_TAIL_CONTRACTION_LIMIT:
        raise RuntimeError("point-dipole late correction did not contract")
    if np.min(mode_ratios) < VALIDATION_MINIMUM_MODE_RATIO:
        raise RuntimeError("one cylindrical mode has no meaningful final signal")
    if np.min(m0_contrast[:3]) < VALIDATION_MINIMUM_M0_SIGNAL_CONTRAST:
        raise RuntimeError("m=0 radiation or LDOS is not distinct from |m|=1")
    if np.max(efficiencies) > VALIDATION_MAXIMUM_EXTRACTION_EFFICIENCY:
        raise RuntimeError("radiated flux exceeds the LDOS-derived source power")


def _dipole_in_slab_profile(
    zpos: float,
    rpos_um: float,
    m: int,
    *,
    resolution_um: int,
    pml_um: float,
    padding_um: float,
    radius_um: float,
    source_fwidth_fraction: float,
    checkpoint_after_sources: Sequence[float] | None,
    flux_decay_threshold: float,
    emit_flux: bool,
) -> dict[str, np.ndarray]:
    """Runs one cylindrical Fourier component and returns every checkpoint."""
    if m not in (-1, 0, 1) and checkpoint_after_sources is not None:
        raise ValueError("the bounded validation profile only supports m=-1,0,+1")
    if resolution_um <= 0:
        raise ValueError("resolution must be positive")
    if min(pml_um, padding_um, radius_um, source_fwidth_fraction) <= 0:
        raise ValueError("validation profile lengths and source bandwidth must be positive")
    if not (0 < rpos_um < radius_um):
        raise ValueError("the radial source position must be inside the non-PML cell")
    if not (0 <= zpos <= 1):
        raise ValueError("the normalized source height must be inside the slab")
    if checkpoint_after_sources is not None:
        checkpoints = tuple(float(value) for value in checkpoint_after_sources)
        if not checkpoints or any(
            not np.isfinite(value) or value <= 0 for value in checkpoints
        ):
            raise ValueError("post-source checkpoints must be finite and positive")
        if any(right <= left for left, right in zip(checkpoints, checkpoints[1:])):
            raise ValueError("post-source checkpoints must be strictly increasing")

    frequency = 1 / WAVELENGTH_UM
    size_r = radius_um + pml_um
    size_z = SLAB_THICKNESS_UM + padding_um + pml_um
    cell_size = mp.Vector3(size_r, 0, size_z)
    boundary_layers = [
        mp.PML(pml_um, direction=mp.R),
        mp.PML(pml_um, direction=mp.Z, side=mp.High),
    ]
    src_pt = mp.Vector3(rpos_um, 0, -0.5 * size_z + zpos * SLAB_THICKNESS_UM)
    sources = [
        mp.Source(
            src=mp.GaussianSource(
                frequency, fwidth=source_fwidth_fraction * frequency
            ),
            component=mp.Er,
            center=src_pt,
        )
    ]
    geometry = [
        mp.Block(
            material=mp.Medium(index=N_SLAB),
            center=mp.Vector3(0, 0, -0.5 * size_z + 0.5 * SLAB_THICKNESS_UM),
            size=mp.Vector3(mp.inf, mp.inf, SLAB_THICKNESS_UM),
        )
    ]
    sim = mp.Simulation(
        resolution=resolution_um,
        cell_size=cell_size,
        dimensions=mp.CYLINDRICAL,
        m=m,
        boundary_layers=boundary_layers,
        sources=sources,
        geometry=geometry,
        force_complex_fields=True,
    )
    flux_mon = sim.add_flux(
        frequency,
        0,
        1,
        mp.FluxRegion(
            center=mp.Vector3(0.5 * radius_um, 0, 0.5 * size_z - pml_um),
            size=mp.Vector3(radius_um, 0, 0),
        ),
        mp.FluxRegion(
            center=mp.Vector3(
                radius_um, 0, 0.5 * size_z - pml_um - 0.5 * padding_um
            ),
            size=mp.Vector3(0, 0, padding_um),
        ),
    )
    ldos_step = mp.dft_ldos(frequency, 0, 1)
    delta_vol = 2 * np.pi * rpos_um / (resolution_um**2)

    radiated_fluxes: list[float] = []
    source_fluxes: list[float] = []
    ldos_values: list[float] = []
    ldos_fdata: list[complex] = []
    ldos_jdata: list[complex] = []
    dft_norms: list[float] = []
    timesteps: list[int] = []
    meep_times: list[float] = []

    def capture() -> None:
        radiated = float(mp.get_fluxes(flux_mon)[0])
        fdata = complex(sim.ldos_Fdata[0])
        jdata = complex(sim.ldos_Jdata[0])
        source = float(-np.real(fdata * np.conj(jdata)) * delta_vol)
        ldos_value = float(sim.ldos_data[0])
        dft_norm = float(sim.fields.dft_norm())
        values = (radiated, source, ldos_value, dft_norm)
        if not all(np.isfinite(value) for value in values) or not (
            np.isfinite(fdata.real)
            and np.isfinite(fdata.imag)
            and np.isfinite(jdata.real)
            and np.isfinite(jdata.imag)
        ):
            raise RuntimeError(f"m={m} produced nonfinite validation observables")
        radiated_fluxes.append(radiated)
        source_fluxes.append(source)
        ldos_values.append(ldos_value)
        ldos_fdata.append(fdata)
        ldos_jdata.append(jdata)
        dft_norms.append(dft_norm)
        timesteps.append(int(sim.timestep()))
        meep_times.append(float(sim.meep_time()))

    if checkpoint_after_sources is None:
        sim.run(
            ldos_step,
            until_after_sources=mp.stop_when_dft_decayed(tol=flux_decay_threshold),
        )
        capture()
    else:
        for after_sources in checkpoints:
            sim.run(ldos_step, until_after_sources=after_sources)
            capture()

    result = {
        "radiated_fluxes": np.asarray(radiated_fluxes, dtype=float),
        "source_fluxes": np.asarray(source_fluxes, dtype=float),
        "ldos_values": np.asarray(ldos_values, dtype=float),
        "ldos_fdata": np.asarray(ldos_fdata, dtype=np.complex128),
        "ldos_jdata": np.asarray(ldos_jdata, dtype=np.complex128),
        "dft_norms": np.asarray(dft_norms, dtype=float),
        "timesteps": np.asarray(timesteps, dtype=float),
        "meep_times": np.asarray(meep_times, dtype=float),
    }
    if emit_flux and mp.am_master():
        print(
            f"flux-cyl:, {rpos_um:.2f}, {m:3d}, "
            f"{source_fluxes[-1]:.6f}, {radiated_fluxes[-1]:.6f}"
        )
    return result


def dipole_in_slab(zpos: float, rpos_um: float, m: int) -> Tuple[float, float]:
    """Computes the flux from a dipole in a slab.

    Args:
      zpos: position of dipole as a fraction of layer thickness.
      rpos_um: position of source in radial direction.
      m: angular φ dependence of the fields exp(imφ).

    Returns:
      A 2-tuple of the radiated and total flux.
    """
    result = _dipole_in_slab_profile(
        zpos,
        rpos_um,
        m,
        resolution_um=RESOLUTION_UM,
        pml_um=1.0,
        padding_um=1.0,
        radius_um=20.0,
        source_fwidth_fraction=0.05,
        checkpoint_after_sources=None,
        flux_decay_threshold=1e-4,
        emit_flux=True,
    )
    return float(result["radiated_fluxes"][-1]), float(result["source_fluxes"][-1])


def run_validation() -> dict[str, np.ndarray]:
    """Runs the bounded deterministic m=-1,0,+1 CUDA-validation profile."""
    run_results = [
        _dipole_in_slab_profile(
            VALIDATION_DIPOLE_ZPOS,
            VALIDATION_DIPOLE_RPOS_UM,
            m,
            resolution_um=VALIDATION_RESOLUTION_UM,
            pml_um=VALIDATION_PML_UM,
            padding_um=VALIDATION_PADDING_UM,
            radius_um=VALIDATION_RADIUS_UM,
            source_fwidth_fraction=VALIDATION_SOURCE_FWIDTH_FRACTION,
            checkpoint_after_sources=VALIDATION_AFTER_SOURCES,
            flux_decay_threshold=1e-4,
            emit_flux=False,
        )
        for m in VALIDATION_M_VALUES
    ]
    radiated = np.stack([result["radiated_fluxes"] for result in run_results])
    source = np.stack([result["source_fluxes"] for result in run_results])
    ldos = np.stack([result["ldos_values"] for result in run_results])
    metrics = point_dipole_validation_metrics(radiated, source, ldos)
    # These broad physical gates reject zero-signal, asymmetric, unconverged,
    # or non-passive fixtures. Exact CPU/GPU agreement is a separate manifest
    # comparison using the full result vectors below.
    validate_point_dipole_metrics(metrics)

    checkpoint_count = len(VALIDATION_AFTER_SOURCES)
    parameters = np.asarray(
        [
            VALIDATION_RESOLUTION_UM,
            VALIDATION_PML_UM,
            VALIDATION_PADDING_UM,
            VALIDATION_RADIUS_UM,
            VALIDATION_SOURCE_FWIDTH_FRACTION,
            VALIDATION_DIPOLE_ZPOS,
            VALIDATION_DIPOLE_RPOS_UM,
        ],
        dtype=float,
    )
    result_vectors = {
        "point_dipole_m_values": np.asarray(VALIDATION_M_VALUES, dtype=float),
        "point_dipole_parameters": parameters,
        "point_dipole_checkpoint_after_sources": np.asarray(
            VALIDATION_AFTER_SOURCES, dtype=float
        ),
        "point_dipole_checkpoint_timesteps": np.stack(
            [result["timesteps"] for result in run_results]
        ),
        "point_dipole_checkpoint_meep_times": np.stack(
            [result["meep_times"] for result in run_results]
        ),
        "point_dipole_radiated_fluxes": radiated,
        "point_dipole_source_fluxes": source,
        "point_dipole_ldos_values": ldos,
        "point_dipole_ldos_fdata": np.stack(
            [result["ldos_fdata"] for result in run_results]
        ),
        "point_dipole_ldos_jdata": np.stack(
            [result["ldos_jdata"] for result in run_results]
        ),
        "point_dipole_dft_norms": np.stack(
            [result["dft_norms"] for result in run_results]
        ),
        "point_dipole_extraction_efficiencies": metrics["efficiencies"],
        "point_dipole_pm_symmetry_relative_errors": metrics[
            "pm_symmetry_relative_errors"
        ],
        "point_dipole_tail_signed_corrections": metrics[
            "tail_signed_corrections"
        ],
        "point_dipole_tail_relative_changes": metrics["tail_relative_changes"],
        "point_dipole_tail_contraction_ratios": metrics[
            "tail_contraction_ratios"
        ],
        "point_dipole_final_mode_ratios": metrics["final_mode_ratios"],
        "point_dipole_m0_contrast": metrics["m0_contrast"],
    }
    expected_shapes = {
        "point_dipole_m_values": (3,),
        "point_dipole_parameters": (7,),
        "point_dipole_checkpoint_after_sources": (checkpoint_count,),
        "point_dipole_checkpoint_timesteps": (3, checkpoint_count),
        "point_dipole_checkpoint_meep_times": (3, checkpoint_count),
        "point_dipole_radiated_fluxes": (3, checkpoint_count),
        "point_dipole_source_fluxes": (3, checkpoint_count),
        "point_dipole_ldos_values": (3, checkpoint_count),
        "point_dipole_ldos_fdata": (3, checkpoint_count),
        "point_dipole_ldos_jdata": (3, checkpoint_count),
        "point_dipole_dft_norms": (3, checkpoint_count),
        "point_dipole_extraction_efficiencies": (3, checkpoint_count),
        "point_dipole_pm_symmetry_relative_errors": (4, checkpoint_count),
        "point_dipole_tail_signed_corrections": (4, 3),
        "point_dipole_tail_relative_changes": (4, 3),
        "point_dipole_tail_contraction_ratios": (4, 3),
        "point_dipole_final_mode_ratios": (4, 3),
        "point_dipole_m0_contrast": (4,),
    }
    for name, shape in expected_shapes.items():
        _finite_vector(result_vectors[name], name, shape)
    return result_vectors


def _run_published_example() -> None:
    """Executes the unchanged published sweep."""
    dipole_height = 0.5

    # An Er source at r = 0 needs to be slightly offset.
    # https://github.com/NanoComp/meep/issues/2704
    dipole_rpos_um = 1.5 / RESOLUTION_UM

    # Er source at r = 0 requires a single simulation with m = ±1.
    m = 1
    radiated_flux, source_flux = dipole_in_slab(
        dipole_height,
        dipole_rpos_um,
        m,
    )
    extraction_efficiency = radiated_flux / source_flux
    print(f"exteff:, {dipole_rpos_um}, {extraction_efficiency:.6f}")

    # Er source at r > 0 requires Fourier-series expansion of φ.
    flux_decay_threshold = 1e-2
    dipole_rpos_um = [3.5, 6.7, 9.5]
    for rpos_um in dipole_rpos_um:
        source_flux_total = 0
        radiated_flux_total = 0
        radiated_flux_max = 0
        m = 0
        while True:
            radiated_flux, source_flux = dipole_in_slab(
                dipole_height,
                rpos_um,
                m,
            )
            radiated_flux_total += radiated_flux * (1 if m == 0 else 2)
            source_flux_total += source_flux * (1 if m == 0 else 2)
            if radiated_flux > radiated_flux_max:
                radiated_flux_max = radiated_flux
            if m > 0 and (radiated_flux / radiated_flux_max) < flux_decay_threshold:
                break
            m += 1

        extraction_efficiency = radiated_flux_total / source_flux_total
        print(f"exteff:, {rpos_um}, {extraction_efficiency:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation",
        action="store_true",
        help="run the bounded deterministic CPU/CUDA validation profile",
    )
    args = parser.parse_args()
    if args.validation:
        globals().update(run_validation())
    else:
        _run_published_example()
