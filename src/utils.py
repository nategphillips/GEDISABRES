# module utils.py
"""Contains useful utility functions."""

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

import sys
from pathlib import Path
from typing import overload

import numpy as np
from numpy.typing import NDArray


@overload
def wavenum_to_wavelen(wavenumber: float) -> float: ...


@overload
def wavenum_to_wavelen(wavenumber: NDArray[np.float64]) -> NDArray[np.float64]: ...


def wavenum_to_wavelen(wavenumber: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    """Convert wavenumbers to wavelengths.

    Args:
        wavenumber (float | NDArray[np.float64]): Wavenumber(s) in [1/cm].

    Returns:
        float | NDArray[np.float64]: The corresponding wavelength(s) in [nm].
    """
    return 1.0 / wavenumber * 1e7


def bandwidth_wavelen_to_wavenum(center_wl: float, fwhm_wl: float) -> float:
    """Convert a FWHM bandwidth from [nm] to [1/cm] given a center wavelength.

    Note that this is not a linear approximation, so it is accurate for large FWHM parameters. See
    https://toolbox.lightcon.com/tools/bandwidthconverter for details.

    Args:
        center_wl (float): Center wavelength in [nm] around which the bandwidth is defined.
        fwhm_wl (float): FWHM bandwidth in [nm].

    Returns:
        float: The FWHM bandwidth in [1/cm].
    """
    return 1e7 * fwhm_wl / (center_wl**2 - fwhm_wl**2 / 4)


def get_data_path(*relative_path_parts) -> Path:
    """Get the correct data path, accounting for PyInstaller executable.

    Returns:
        Path: A relative path if developing, the absolute path to the bundle folder if Pyinstaller.
    """
    if getattr(sys, "frozen", False):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).resolve().parent.parent

    return base_path.joinpath(*relative_path_parts)
