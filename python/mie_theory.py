"""Small, deterministic Lorenz-Mie reference used by the Python examples.

The implementation intentionally depends only on NumPy and SciPy, which are
already required by the Meep Python package.  It avoids making the FDTD
examples depend on a second optional Mie package while keeping the analytic
reference independent of Meep's time-domain solver.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import spherical_jn, spherical_yn


def _finite_complex(value: complex, name: str) -> complex:
    result = complex(value)
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError(f"{name} must be finite")
    return result


def _mie_coefficients(relative_index: complex, value: float):
    """Return orders and electric/magnetic Lorenz-Mie coefficients."""
    order_max = max(1, int(math.ceil(value + 4 * value ** (1 / 3) + 2)))
    order = np.arange(1, order_max + 1)
    internal = relative_index * value

    j_external = spherical_jn(order, value)
    j_external_prime = spherical_jn(order, value, derivative=True)
    y_external = spherical_yn(order, value)
    y_external_prime = spherical_yn(order, value, derivative=True)
    j_internal = spherical_jn(order, internal)
    j_internal_prime = spherical_jn(order, internal, derivative=True)

    psi_external = value * j_external
    psi_external_prime = j_external + value * j_external_prime
    psi_internal = internal * j_internal
    psi_internal_prime = j_internal + internal * j_internal_prime
    xi_external = value * (j_external + 1j * y_external)
    xi_external_prime = (
        j_external
        + 1j * y_external
        + value * (j_external_prime + 1j * y_external_prime)
    )

    numerator_a = (
        relative_index * psi_internal * psi_external_prime
        - psi_external * psi_internal_prime
    )
    denominator_a = (
        relative_index * psi_internal * xi_external_prime
        - xi_external * psi_internal_prime
    )
    numerator_b = (
        psi_internal * psi_external_prime
        - relative_index * psi_external * psi_internal_prime
    )
    denominator_b = (
        psi_internal * xi_external_prime
        - relative_index * xi_external * psi_internal_prime
    )
    if np.any(denominator_a == 0) or np.any(denominator_b == 0):
        raise FloatingPointError("singular Lorenz-Mie coefficient denominator")
    return order, numerator_a / denominator_a, numerator_b / denominator_b


def mie_scattering_efficiency(
    refractive_index: complex, size_parameter
):
    """Return the nonmagnetic-sphere scattering efficiency ``Qsca``.

    ``size_parameter`` is ``2*pi*r/wavelength`` in the surrounding medium.
    A scalar input returns ``float`` and an array-like input returns an array
    with the same shape.
    """

    relative_index = _finite_complex(refractive_index, "refractive_index")
    if relative_index == 0:
        raise ValueError("refractive_index must be nonzero")
    raw = np.asarray(size_parameter, dtype=float)
    if np.any(~np.isfinite(raw)) or np.any(raw <= 0):
        raise ValueError("size_parameter must contain finite positive values")

    def one(value: float) -> float:
        order, coefficient_a, coefficient_b = _mie_coefficients(
            relative_index, value
        )
        result = float(
            2
            / value**2
            * np.sum(
                (2 * order + 1)
                * (np.abs(coefficient_a) ** 2 + np.abs(coefficient_b) ** 2)
            )
        )
        if not math.isfinite(result) or result < 0:
            raise FloatingPointError("Lorenz-Mie series produced an invalid Qsca")
        return result

    output = np.asarray([one(float(value)) for value in raw.reshape(-1)]).reshape(
        raw.shape
    )
    return float(output) if raw.ndim == 0 else output


def mie_differential_cross_section(
    refractive_index: complex,
    size_parameter: float,
    radius: float,
    angles,
):
    """Return ``d sigma/d Omega`` for circular/unpolarized illumination.

    ``angles`` are polar scattering angles in radians relative to the incident
    wavevector. For a sphere, circular polarization has the same azimuthally
    independent intensity as the average of the two linear polarizations.
    """
    relative_index = _finite_complex(refractive_index, "refractive_index")
    if relative_index == 0:
        raise ValueError("refractive_index must be nonzero")
    value = float(size_parameter)
    radius = float(radius)
    theta = np.asarray(angles, dtype=float)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("size_parameter must be finite and positive")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if (
        np.any(~np.isfinite(theta))
        or np.any(theta < 0)
        or np.any(theta > math.pi)
    ):
        raise ValueError("angles must be finite values in [0, pi]")

    orders, coefficient_a, coefficient_b = _mie_coefficients(
        relative_index, value
    )
    mu = np.cos(theta)
    s1 = np.zeros(theta.shape, dtype=np.complex128)
    s2 = np.zeros(theta.shape, dtype=np.complex128)
    pi_previous = np.zeros(theta.shape, dtype=float)
    pi_current = np.ones(theta.shape, dtype=float)
    for index, order_value in enumerate(orders):
        n = int(order_value)
        tau = n * mu * pi_current - (n + 1) * pi_previous
        factor = (2 * n + 1) / (n * (n + 1))
        s1 += factor * (
            coefficient_a[index] * pi_current + coefficient_b[index] * tau
        )
        s2 += factor * (
            coefficient_a[index] * tau + coefficient_b[index] * pi_current
        )
        pi_next = (
            (2 * n + 1) / n * mu * pi_current
            - (n + 1) / n * pi_previous
        )
        pi_previous, pi_current = pi_current, pi_next

    result = radius**2 / (2 * value**2) * (np.abs(s1) ** 2 + np.abs(s2) ** 2)
    if np.any(~np.isfinite(result)) or np.any(result < 0):
        raise FloatingPointError("Lorenz-Mie angular series produced invalid values")
    return float(result) if theta.ndim == 0 else result


def mie_scattering_cross_section(
    refractive_index: complex, size_parameter, radius: float
):
    """Return the scattering cross section in the squared units of ``radius``."""

    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    return mie_scattering_efficiency(
        refractive_index, size_parameter
    ) * (math.pi * radius**2)
