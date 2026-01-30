# module atom.py
"""Contains the implementation of the Atom class."""

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

from fractions import Fraction

import constants
from enums import NuclearStatistics


class Atom:
    """Represents an atom with a name and mass."""

    def __init__(self, atomic_mass_number: int, chemical_symbol: str) -> None:
        """Initialize class variables.

        Args:
            atomic_mass_number: The total number of protons and neutrons, A.
            chemical_symbol: The chemical symbol, e.g., C for carbon.
        """
        self.atomic_mass_number: int = atomic_mass_number
        self.chemical_symbol: str = chemical_symbol
        # A concatenation of the atomic mass number and chemical symbol.
        self.ael: str = str(atomic_mass_number) + chemical_symbol
        self.nuclear_spin: Fraction = Fraction(constants.NUCLEAR_SPIN[self.ael])

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
        if self.ael not in constants.ATOMIC_MASSES:
            raise ValueError(f"Atom `{self.ael}` not supported.")

        # Convert from [g/mol] to [kg].
        return constants.ATOMIC_MASSES[self.ael] / constants.AVOGD / 1e3

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
