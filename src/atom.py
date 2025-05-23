# module atom.py
"""Contains the implementation of the Atom class."""

# Copyright (C) 2025 Nathan G. Phillips

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

import constants


class Atom:
    """Represents an atom with a name and mass."""

    def __init__(self, name: str) -> None:
        """Initialize class variables.

        Args:
            name (str): Molecule name.
        """
        self.name: str = name
        self.mass: float = self.get_mass(name) / constants.AVOGD / 1e3

    @staticmethod
    def get_mass(name: str) -> float:
        """Return the atomic mass in [g/mol].

        Args:
            name (str): Name of the atom.

        Raises:
            ValueError: If the selected atom is not supported.

        Returns:
            float: The atomic mass in [g/mol].
        """
        if name not in constants.ATOMIC_MASSES:
            raise ValueError(f"Atom `{name}` not supported.")

        return constants.ATOMIC_MASSES[name]
