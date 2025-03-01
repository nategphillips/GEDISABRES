# module convolve
"""Contains functions used for convolution."""

import numpy as np
from scipy.special import wofz

from line import Line


def broadening_fn(
    wavenumbers: np.ndarray, line: Line, fwhm_selections: dict[str, bool], inst_broadening_wl: float
) -> np.ndarray:
    """Return the contribution of a single rotational line to the total spectra.

    Uses a Voigt probability density function.
    """
    # Instrument broadening in [1/cm] is added to thermal broadening to get the full Gaussian FWHM.
    gaussian: float = line.fwhm_instrument(
        fwhm_selections["instrument"], inst_broadening_wl
    ) + line.fwhm_doppler(fwhm_selections["doppler"])

    # NOTE: 10/25/14 - Since predissociating repulsive states have no interfering absorption, the
    #       broadened absorption lines will be Lorentzian in shape. See Julienne, 1975.

    # Add the effects of natural, collisional, and predissociation broadening to get the full
    # Lorentzian FWHM.
    lorentzian: float = (
        line.fwhm_natural(fwhm_selections["natural"])
        + line.fwhm_collisional(fwhm_selections["collisional"])
        + line.fwhm_predissociation(fwhm_selections["predissociation"])
    )

    # If only Gaussian FWHM parameters are present, then return a Gaussian profile.
    if (gaussian > 0.0) and (lorentzian == 0.0):
        return np.exp(-((wavenumbers - line.wavenumber) ** 2) / (2 * gaussian**2)) / (
            gaussian * np.sqrt(2 * np.pi)
        )

    # Similarly, if only Lorentzian FWHM parameters exist, then return a Lorentzian profile.
    if (gaussian == 0.0) and (lorentzian > 0.0):
        return lorentzian / (np.pi * ((wavenumbers - line.wavenumber) ** 2 + lorentzian**2))

    # TODO: 25/02/14 - Should check if both Gaussian and Lorentzian FWHM params are zero here and
    #       return an error if so.

    # Otherwise, compute the argument of the complex Faddeeva function and return a Voigt profile.
    z: np.ndarray = ((wavenumbers - line.wavenumber) + 1j * lorentzian) / (gaussian * np.sqrt(2))

    # The probability density function for the Voigt profile.
    return np.real(wofz(z)) / (gaussian * np.sqrt(2 * np.pi))


def convolve_brod(
    lines: list[Line],
    wavenumbers_conv: np.ndarray,
    fwhm_selections: dict[str, bool],
    inst_broadening_wl: float,
) -> np.ndarray:
    """Convolve a discrete number of spectral lines into a continuous spectra."""
    intensities_conv: np.ndarray = np.zeros_like(wavenumbers_conv)

    # TODO: 25/02/12 - See if switching to scipy's convolve method improves the speed of this,
    #       especially with a large number of bands or points.

    # Add the effects of each line to the continuous spectra by computing its broadening function
    # multiplied by its intensity and adding it to the total intensity.
    for line in lines:
        intensities_conv += line.intensity * broadening_fn(
            wavenumbers_conv, line, fwhm_selections, inst_broadening_wl
        )

    return intensities_conv
