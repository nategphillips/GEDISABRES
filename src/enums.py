# module enums.py
"""Contains enums for defining simulation properties."""

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

from enum import Enum


class SimType(Enum):
    """Defines the type of simulation to be performed."""

    ABSORPTION = 1
    EMISSION = 2

class InversionSymmetry(Enum):
    """Inversion parity through a centre of symmetry (g/u)."""

    NONE = 1
    GERADE = 2
    UNGERADE = 3


class ReflectionSymmetry(Enum):
    """Reflection symmetry along an arbitrary plane containing the internuclear axis (+/-)."""

    NONE = 1
    PLUS = 2
    MINUS = 3


class TermSymbol(Enum):
    """The term symbol of a given electronic state."""

    SIGMA = 1
    PI = 2
    DELTA = 3

