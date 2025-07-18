# module atom.py
"""Contains the implementation of the Atom class."""

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

from typing import TYPE_CHECKING

import constants
from enums import NuclearStatistics

if TYPE_CHECKING:
    from fractions import Fraction


class Atom:
    """Represents an atom with a name and mass."""

    def __init__(self, name: str) -> None:
        """Initialize class variables.

        Args:
            name (str): Molecule name.
        """
        self.name: str = name
        self.nuclear_spin: Fraction = constants.NUCLEAR_SPIN[self.name]

    @property
    def mass(self) -> float:
        """Return the atomic mass in [kg].

        Args:
            name (str): Name of the atom.

        Raises:
            ValueError: If the selected atom is not supported.

        Returns:
            float: The atomic mass in [kg].
        """
        if self.name not in constants.ATOMIC_MASSES:
            raise ValueError(f"Atom `{self.name}` not supported.")

        # Convert from [g/mol] to [kg].
        return constants.ATOMIC_MASSES[self.name] / constants.AVOGD / 1e3

    @property
    def nuclear_statistics(self) -> NuclearStatistics:
        """Determine the nuclear spin statistics of the nuclei.

        Raises:
            ValueError: If the spin is not an integer or half-integer.

        Returns:
            NuclearStatistics: The nuclear spin statistics, Bose or Fermi.
        """
        # Bosons have integer values of spin.
        if self.nuclear_spin.is_integer():
            return NuclearStatistics.BOSE
        # Fermions have half-integer values of spin.
        if self.nuclear_spin.denominator == 2:
            return NuclearStatistics.FERMI

        raise ValueError(
            f"Bad nuclear statistics: {self.nuclear_spin} is not integer or half-integer."
        )
