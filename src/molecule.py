# module molecule.py
"""Contains the implementation of the Molecule class."""

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

from atom import Atom


class Molecule:
    """Represents a diatomic molecule consisting of two atoms."""

    def __init__(self, name: str, atom_1: Atom, atom_2: Atom) -> None:
        """Initialize class variables.

        Args:
            name (str): Name of the molecule.
            atom_1 (Atom): First constituent atom.
            atom_2 (Atom): Second constituent atom.
        """
        self.name: str = name
        self.atom_1: Atom = atom_1
        self.atom_2: Atom = atom_2
        self.mass: float = self.atom_1.mass + self.atom_2.mass
        self.symmetry_param: int = self.get_symmetry_param(atom_1, atom_2)

    @staticmethod
    def get_symmetry_param(atom_1: Atom, atom_2: Atom) -> int:
        """Return the symmetry parameter of the molecule.

        Args:
            atom_1 (Atom): First constituent atom.
            atom_2 (Atom): Second constituent atom.

        Returns:
            int: The symmetry parameter of the molecule: 2 for homonuclear, 1 for heteronuclear.
        """
        # For homonuclear diatomic molecules like O2, the symmetry parameter is 2.
        if atom_1.name == atom_2.name:
            return 2

        # For heteronuclear diatomics, it's 1.
        return 1
