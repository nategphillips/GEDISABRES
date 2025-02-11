# module convolve
"""
Contains functions used for convolution.
"""

import numpy as np
from scipy.special import wofz  # pylint: disable=no-name-in-module

from line import Line


def broadening_fn(wavenumbers: np.ndarray, line: Line) -> np.ndarray:
    """
    Returns the contribution of a single rotational line to the total spectra using a Voigt
    probability density function.
    """

    # Each line has its own broadening parameters.
    gaussian, lorentzian = line.fwhm_params()

    # Compute the argument of the complex Faddeeva function.
    z: np.ndarray = ((wavenumbers - line.wavenumber) + 1j * lorentzian) / (gaussian * np.sqrt(2))

    # The probability density function for the Voigt profile.
    return np.real(wofz(z)) / (gaussian * np.sqrt(2 * np.pi))


def convolve_brod(lines: list[Line], wavenumbers_conv: np.ndarray) -> np.ndarray:
    """
    Convolves a discrete number of spectral lines into a continuous spectra by applying a broadening
    function.
    """

    intensities_conv: np.ndarray = np.zeros_like(wavenumbers_conv)

    # Add the effects of each line to the continuous spectra by computing its broadening function
    # multiplied by its intensity and adding it to the total intensity.
    for line in lines:
        intensities_conv += line.intensity * broadening_fn(wavenumbers_conv, line)

    return intensities_conv


def instrument_fn(
    convolved_wavenumbers: np.ndarray, wavenumber_peak: float, broadening: float
) -> np.ndarray:
    """
    Simulates the effects of instrument broadening using a Gaussian probability density function.
    """

    return np.exp(-0.5 * (convolved_wavenumbers - wavenumber_peak) ** 2 / broadening**2) / (
        broadening * np.sqrt(2 * np.pi)
    )


def convolve_inst(
    wavenumbers_conv: np.ndarray, intensities_conv: np.ndarray, broadening: float
) -> np.ndarray:
    """
    Convolves a discrete number of spectral lines into a continuous spectra by applying an
    instrument function.
    """

    intensities_inst: np.ndarray = np.zeros_like(wavenumbers_conv)

    for wave, intn in zip(wavenumbers_conv, intensities_conv):
        intensities_inst += intn * instrument_fn(wavenumbers_conv, wave, broadening)

    return intensities_inst
