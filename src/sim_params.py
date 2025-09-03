# module sim_params.py
"""Contains parameters used for the Sim class."""

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

from dataclasses import dataclass


@dataclass
class TemperatureParams:
    """Temperature parameters."""

    translational: float = 300.0
    electronic: float = 300.0
    vibrational: float = 300.0
    rotational: float = 300.0


@dataclass
class ShiftBools:
    """Switches for line shifting mechanisms."""

    collisional: bool = False
    doppler: bool = False


@dataclass
class ShiftParams:
    """Line shift parameters."""

    collisional_a: float = 0.0
    collisional_b: float = 0.0


@dataclass
class LaserParams:
    """Laser parameters."""

    power_w: float = 0.0
    beam_diameter_mm: float = 1.0
    molecule_velocity_ms: float = 0.0


@dataclass
class InstrumentParams:
    """Instrument broadening parameters."""

    gauss_fwhm_wl: float = 0.0
    loren_fwhm_wl: float = 0.0


@dataclass
class BroadeningBools:
    """Switches for broadening mechanisms."""

    collisional: bool = False
    doppler: bool = False
    instrument: bool = False
    natural: bool = False
    power: bool = False
    predissociation: bool = False
    transit: bool = False


@dataclass
class PlotBools:
    """Switches for plot parameters."""

    limits: bool = False


@dataclass
class PlotParams:
    """Minimum and maximum wavelength limits for plotting."""

    limit_min: float = 0.0
    limit_max: float = 10000.0
