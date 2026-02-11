# module molecule.py
"""Contains the implementation of the Molecule class."""

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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atom import Atom


class Molecule:
    """Represents a diatomic molecule consisting of two atoms."""

    def __init__(self, atom_1: Atom, atom_2: Atom) -> None:
        """Initialize class variables.

        Args:
            atom_1: First constituent atom.
            atom_2: Second constituent atom.
        """
        self.atom_1 = atom_1
        self.atom_2 = atom_2
        self.name = atom_1.ael + atom_2.ael
        self.mass = self.atom_1.mass + self.atom_2.mass
        self.is_homonuclear = self.atom_1.ael == self.atom_2.ael

    @property
    def symmetry_param(self) -> int:
        """Return the symmetry parameter of the molecule.

        Returns:
            The symmetry parameter of the molecule: 2 for homonuclear, 1 for heteronuclear.
        """
        if self.is_homonuclear:
            return 2

        return 1
