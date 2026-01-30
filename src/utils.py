# module utils.py
"""Contains useful utility functions."""

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

from typing import TYPE_CHECKING, overload

import constants

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@overload
def wavenum_to_wavelen(wavenumber: float) -> float: ...


@overload
def wavenum_to_wavelen(wavenumber: NDArray[np.float64]) -> NDArray[np.float64]: ...


def wavenum_to_wavelen(wavenumber: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    """Convert wavenumbers to wavelengths.

    Not valid for bandwidth conversions since there is an inverse relationship.

    Args:
        wavenumber (float | NDArray[np.float64]): Wavenumber(s) in [1/cm].

    Returns:
        float | NDArray[np.float64]: The corresponding wavelength(s) in [nm].
    """
    return 1.0 / wavenumber * 1e7


def freq_to_wavenum(freq: float) -> float:
    """Convert frequency to wavenumber.

    Valid for bandwidth conversions since there is no inverse relationship.

    Args:
        freq (float): Frequency in [1/s].

    Returns:
        float: The corresponding wavenumber in [1/cm].
    """
    return freq / constants.LIGHT


def bandwidth_wavelen_to_wavenum(center_wl: float, fwhm_wl: float) -> float:
    """Convert a FWHM bandwidth from [nm] to [1/cm] given a center wavelength.

    Note that this is not a linear approximation, so it is accurate for large FWHM parameters. See
    https://www.lasercalculator.com/spectral-bandwidth-converter/ for details.

    Args:
        center_wl (float): Center wavelength in [nm] around which the bandwidth is defined.
        fwhm_wl (float): FWHM bandwidth in [nm].

    Returns:
        float: The FWHM bandwidth in [1/cm].
    """
    wl_min: float = center_wl - 0.5 * fwhm_wl
    wl_max: float = center_wl + 0.5 * fwhm_wl
    wn_min: float = 1.0 / wl_max
    wn_max: float = 1.0 / wl_min

    # Convert from [1/nm] to [1/cm].
    return 1e7 * (wn_max - wn_min)
