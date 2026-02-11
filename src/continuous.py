# module continuous.py
"""Contains functions used for computing the continuous spectrum."""

# Copyright (C) 2023-2026 Nathan G. Phillips

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

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import voigt_profile

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from line import Line


def fwhm_gaussian(line: Line, inst_gauss: float) -> float:
    """Returns the Gaussian full width at half maximum (FWHM) for a line.

    Gaussian FWHMs are summed in quadrature: see "Hypersonic Nonequilibrium Flows: Fundamentals and
    Recent Advances" p. 361.

    Args:
        line: A rotational `Line` object.
        inst_gauss: Instrument Gaussian broadening.

    Returns:
        The Gaussian FWHM for a line.
    """
    return math.sqrt(inst_gauss**2 + line.fwhm_doppler() ** 2 + line.fwhm_transit() ** 2)


def fwhm_lorentzian(line: Line, inst_loren: float) -> float:
    """Returns the Lorentzian full width at half maximum (FWHM) for a line.

    Lorentzian FHWMs are summed linearly: see "Hypersonic Nonequilibrium Flows: Fundamentals and
    Recent Advances" p. 361.

    Args:
        line: A rotational `Line` object.
        inst_loren: Instrument Lorentzian broadening.

    Returns:
        The Lorentzian FWHM for a line.
    """
    # NOTE: 24/10/25 - Since predissociating repulsive states have no interfering absorption, the
    #       broadened absorption lines will be Lorentzian in shape. See Julienne, 1975.
    return (
        inst_loren
        + line.fwhm_natural()
        + line.fwhm_collisional()
        + line.fwhm_predissociation()
        + line.fwhm_power()
    )


def lineshape_profile(wavenumbers_cont: NDArray[np.float64], line: Line) -> NDArray[np.float64]:
    """Return the lineshape profile of a single rotational line.

    If only Gaussian broadening parameters are present, then a Gaussian profile is returned. Ditto
    for a Lorentzian lineshape. If both Gaussian and Lorentzian broadening parameters are supplied,
    a Voigt profile is returned.

    Args:
        wavenumbers_cont: A continuous array of wavenumbers.
        line: A rotational `Line` object.

    Returns:
        The relevant probability density function.

    Raises:
        ValueError: If the line has zero total width.
    """
    inst_gauss, inst_loren = line.fwhm_instrument()

    fwhm_gauss = fwhm_gaussian(line, inst_gauss)
    fwhm_loren = fwhm_lorentzian(line, inst_loren)

    x = wavenumbers_cont - line.wavenumber

    # The FWHM of the Gaussian PDF is 2 * sigma * sqrt(2 * ln(2)), where sigma is the standard
    # deviation. See https://mathworld.wolfram.com/FullWidthatHalfMaximum.html.
    gaussian_stddev = fwhm_gauss / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    # The FWHM of the Lorentzian PDF is 2 * gamma, where gamma is the half-width at half-maximum.
    lorentzian_hwhm = 0.5 * fwhm_loren

    iszero_gauss = math.isclose(fwhm_gauss, 0.0)
    iszero_loren = math.isclose(fwhm_loren, 0.0)

    # If only Gaussian FWHM parameters are present, then return a Gaussian profile.
    if not iszero_gauss and iszero_loren:
        return np.exp(-(x**2) / (2.0 * gaussian_stddev**2)) / (
            math.sqrt(2.0 * np.pi) * gaussian_stddev
        )

    # Similarly, if only Lorentzian FWHM parameters exist, then return a Lorentzian profile.
    if not iszero_loren and iszero_gauss:
        return lorentzian_hwhm / (np.pi * (x**2 + lorentzian_hwhm**2))

    if iszero_gauss and iszero_loren:
        raise ValueError(f"Line at {line.wavenumber} has zero total width.")

    # The probability density function for the Voigt profile.
    # real(w(z)) / (sigma * sqrt(2 * pi)), where z = (x + i * gamma) / (sqrt(2) * sigma)
    return voigt_profile(x, gaussian_stddev, lorentzian_hwhm)


def sum_lines(
    lines: list[Line],
    wavenumbers_cont: NDArray[np.float64],
    windowed_eval: bool = False,
    window_fwhm_factor: float = 50.0,
) -> NDArray[np.float64]:
    """Sum spectral lines into a continuous spectrum using windowed evaluation.

    Args:
        lines: A rotational `Line` object.
        wavenumbers_cont: A continuous array of wavenumbers.
        windowed_eval: True if a windowed evaluation of the spectrum should be performed, cuts down
            on the computation time but is less accurate. Defaults to False.
        window_fwhm_factor: The "range of influence" of the line, i.e., how many FWHM steps away
            from the line center the intensity profile should be computed. Defaults to 25.0.

    Returns:
        The total intensity spectrum with contributions from all lines.
    """
    intensities_cont = np.zeros_like(wavenumbers_cont)

    if windowed_eval:
        for line in lines:
            inst_gauss, inst_loren = line.fwhm_instrument()
            fwhm_gauss = fwhm_gaussian(line, inst_gauss)
            fwhm_loren = fwhm_lorentzian(line, inst_loren)

            # The two FWHMs are summed to ensure that we don't undercount a contribution from
            # either.
            half_window = window_fwhm_factor * (fwhm_gauss + fwhm_loren)

            # The left side of the window is located at (wavenumber - half_window), so we find the
            # index at which this value *would* exist if it were placed into the wavenumber array
            # from the left side.
            lo_idx = np.searchsorted(wavenumbers_cont, line.wavenumber - half_window, side="left")
            # Same thing for the right side of the window, just reverse the direction we search
            # from.
            hi_idx = np.searchsorted(wavenumbers_cont, line.wavenumber + half_window, side="right")

            # The lineshape profile and sum are only computed for the part of the wavenumber array
            # which the line itself influences, i.e., the slice that falls within the full window.
            prof = lineshape_profile(wavenumbers_cont[lo_idx:hi_idx], line)
            intensities_cont[lo_idx:hi_idx] += line.intensity * prof

        return intensities_cont

    for line in lines:
        intensities_cont += line.intensity * lineshape_profile(wavenumbers_cont, line)

    return intensities_cont
