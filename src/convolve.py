# module convolve.py
"""Contains functions used for convolution."""

# Copyright (C) 2023-2025 Nathan G. Phillips

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from numpy.typing import NDArray
from scipy.special import voigt_profile

from line import Line


def broadening_fn(
    wavenumbers_conv: NDArray[np.float64],
    line: Line,
    fwhm_selections: dict[str, bool],
    inst_broadening_wl: float,
) -> NDArray[np.float64]:
    """Return the contribution of a single rotational line to the total spectra.

    Uses a Voigt probability density function.

    Args:
        wavenumbers_conv (NDArray[np.float64]): A continuous array of wavenumbers.
        line (Line): A rotational `Line` object.
        fwhm_selections (dict[str, bool]): The types of broadening to be simulated.
        inst_broadening_wl (float): Instrument broadening FWHM in [nm].

    Returns:
        NDArray[np.float64]: The Voigt probability density function for a single rotational line.
    """
    # Instrument broadening in [1/cm] is added to thermal broadening to get the full Gaussian FWHM.
    # Note that Gaussian FWHMs must be summed in quadrature: see "Hypersonic Nonequilibrium Flows:
    # Fundamentals and Recent Advances" p. 361.
    fwhm_gaussian: float = np.sqrt(
        line.fwhm_instrument(fwhm_selections["instrument"], inst_broadening_wl) ** 2
        + line.fwhm_doppler(fwhm_selections["doppler"]) ** 2
    )

    # NOTE: 24/10/25 - Since predissociating repulsive states have no interfering absorption, the
    #       broadened absorption lines will be Lorentzian in shape. See Julienne, 1975.

    # Add the effects of natural, collisional, and predissociation broadening to get the full
    # Lorentzian FWHM. Lorentzian FHWMs are summed linearly: see "Hypersonic Nonequilibrium Flows:
    # Fundamentals and Recent Advances" p. 361.
    fwhm_lorentzian: float = (
        line.fwhm_natural(fwhm_selections["natural"])
        + line.fwhm_collisional(fwhm_selections["collisional"])
        + line.fwhm_predissociation(fwhm_selections["predissociation"])
    )

    # NOTE: 25/04/25 - The forms of the Gaussian and Lorentzian PDFs used here are written in terms
    #       of the FWHM. More commonly, the Gaussian PDF is written in terms of σ, the standard
    #       deviation, and the Lorentzian is written in terms of γ, the half-width at half-maximum.

    # If only Gaussian FWHM parameters are present, then return a Gaussian profile.
    if (fwhm_gaussian > 0.0) and (fwhm_lorentzian == 0.0):
        return (
            (2 / fwhm_gaussian)
            * np.sqrt(np.log(2) / np.pi)
            * np.exp(-4 * np.log(2) * ((wavenumbers_conv - line.wavenumber) / fwhm_gaussian) ** 2)
        )

    # Similarly, if only Lorentzian FWHM parameters exist, then return a Lorentzian profile.
    if (fwhm_gaussian == 0.0) and (fwhm_lorentzian > 0.0):
        return np.divide(
            fwhm_lorentzian,
            (2 * np.pi * ((wavenumbers_conv - line.wavenumber) ** 2 + (fwhm_lorentzian / 2) ** 2)),
        )

    # TODO: 25/02/14 - Should check if both Gaussian and Lorentzian FWHM params are zero here and
    #       return an error if so.

    # The FWHM of the Gaussian PDF is 2 * sigma * sqrt(2 * ln(2)), where sigma is the standard
    # deviation.
    gaussian_stddev: float = fwhm_gaussian / (2 * np.sqrt(2 * np.log(2)))

    # The FWHM of the Lorentzian PDF is 2 * gamma, where gamma is the half-width at half-maximum.
    lorentzian_hwhm: float = fwhm_lorentzian / 2

    # The probability density function for the Voigt profile.
    # real(w(z)) / (sigma * sqrt(2 * pi)), where z = (x + i * gamma) / (sqrt(2) * sigma)
    return voigt_profile((wavenumbers_conv - line.wavenumber), gaussian_stddev, lorentzian_hwhm)


def convolve(
    lines: list[Line],
    wavenumbers_conv: NDArray[np.float64],
    fwhm_selections: dict[str, bool],
    inst_broadening_wl: float,
) -> NDArray[np.float64]:
    """Convolve a discrete number of spectral lines into a continuous spectra.

    Args:
        lines (list[Line]): A list of rotational `Line` objects.
        wavenumbers_conv (NDArray[np.float64]): A continuous array of wavenumbers.
        fwhm_selections (dict[str, bool]): The types of broadening to be simulated.
        inst_broadening_wl (float): Instrument broadening FWHM in [nm].

    Returns:
        NDArray[np.float64]: The total intensity spectrum with contributions from all lines.
    """
    intensities_conv: NDArray[np.float64] = np.zeros_like(wavenumbers_conv)

    # TODO: 25/02/12 - See if switching to scipy's convolve method improves the speed of this,
    #       especially with a large number of bands or points.

    # Add the effects of each line to the continuous spectra by computing its broadening function
    # multiplied by its intensity and adding it to the total intensity.
    for line in lines:
        intensities_conv += line.intensity * broadening_fn(
            wavenumbers_conv, line, fwhm_selections, inst_broadening_wl
        )

    return intensities_conv
