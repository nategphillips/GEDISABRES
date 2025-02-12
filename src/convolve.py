# module convolve
"""
Contains functions used for convolution.
"""

import numpy as np
from scipy.special import wofz  # pylint: disable=no-name-in-module

from line import Line


def broadening_fn(wavenumbers: np.ndarray, line: Line, inst_broadening: float) -> np.ndarray:
    """
    Returns the contribution of a single rotational line to the total spectra using a Voigt
    probability density function.
    """

    # Each line has its own broadening parameters.
    gaussian, lorentzian = line.fwhm_params(inst_broadening)

    # Compute the argument of the complex Faddeeva function.
    z: np.ndarray = ((wavenumbers - line.wavenumber) + 1j * lorentzian) / (gaussian * np.sqrt(2))

    # The probability density function for the Voigt profile.
    return np.real(wofz(z)) / (gaussian * np.sqrt(2 * np.pi))


def convolve_brod(
    lines: list[Line], wavenumbers_conv: np.ndarray, inst_broadening: float
) -> np.ndarray:
    """
    Convolves a discrete number of spectral lines into a continuous spectra by applying a broadening
    function.
    """

    intensities_conv: np.ndarray = np.zeros_like(wavenumbers_conv)

    # TODO: 25/02/12 - See if switching to scipy's convolve method improves the speed of this,
    #       especially with a large number of bands or points.

    # Add the effects of each line to the continuous spectra by computing its broadening function
    # multiplied by its intensity and adding it to the total intensity.
    for line in lines:
        intensities_conv += line.intensity * broadening_fn(wavenumbers_conv, line, inst_broadening)

    return intensities_conv
