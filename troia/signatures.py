"""Photometric signatures used in compact-object companion searches."""

import numpy as np

import chalcedon
import nicomedia


def compute_photometric_signatures(
    period_days,
    companion_mass_solar,
    stellar_radius_solar=1.0,
    stellar_mass_solar=1.0,
    stellar_density_cgs=1.41,
) -> dict[str, np.ndarray]:
    """Return beaming, ellipsoidal, and self-lensing amplitudes in ppt."""

    period_days = np.asarray(period_days, dtype=float)
    companion_mass_solar = np.asarray(companion_mass_solar, dtype=float)
    if np.any(period_days <= 0.0):
        raise ValueError("Orbital periods must be positive.")
    if np.any(companion_mass_solar <= 0.0):
        raise ValueError("Companion masses must be positive.")
    if stellar_radius_solar <= 0.0 or stellar_mass_solar <= 0.0:
        raise ValueError("Stellar radius and mass must be positive.")
    if stellar_density_cgs <= 0.0:
        raise ValueError("Stellar density must be positive.")

    return {
        "beaming": np.asarray(
            nicomedia.retr_deptbeam(
                period_days,
                stellar_mass_solar,
                companion_mass_solar,
            )
        ),
        "ellipsoidal": np.asarray(
            nicomedia.retr_deptelli(
                period_days,
                stellar_density_cgs,
                stellar_mass_solar,
                companion_mass_solar,
            )
        ),
        "self_lensing": np.asarray(
            chalcedon.retr_amplslen(
                period_days,
                stellar_radius_solar,
                companion_mass_solar,
                stellar_mass_solar,
            )
        ),
    }