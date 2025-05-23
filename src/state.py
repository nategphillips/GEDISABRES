# module state.py
"""Contains the implementation of the State class."""

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

import polars as pl

import utils
from molecule import Molecule


class State:
    """Represents an electronic state of a particular molecule."""

    def __init__(self, name: str, spin_multiplicity: int, molecule: Molecule) -> None:
        """Initialize class variables.

        Args:
            name (str): Name of the electronic state.
            spin_multiplicity (int): Spin multiplicity.
            molecule (Molecule): Parent molecule.
        """
        self.name: str = name
        self.spin_multiplicity: int = spin_multiplicity
        self.molecule: Molecule = molecule
        self.constants: dict[str, list[float]] = self.get_constants(molecule.name, name)

    @staticmethod
    def get_constants(molecule: str, state: str) -> dict[str, list[float]]:
        """Return the molecular constants for the specified electronic state in [1/cm].

        Args:
            molecule (str): Parent molecule.
            state (str): Name of the electronic state.

        Returns:
            dict[str, list[float]]: A `dict` of molecular constants for the electronic state.
        """
        return pl.read_csv(utils.get_data_path("data", molecule, "states", f"{state}.csv")).to_dict(
            as_series=False
        )

    def is_allowed(self, n_qn: int) -> bool:
        """Return whether or not the selected rotational level is allowed.

        Args:
            n_qn (int): Rotational quantum number N.

        Raises:
            ValueError: If the electronic state does not exist.

        Returns:
            bool: True if the selected rotational level is allowed.
        """
        if self.name == "X3Sg-":
            # For X3Σg-, only the rotational levels with odd N can be populated.
            return bool(n_qn % 2 == 1)
        if self.name == "B3Su-":
            # For B3Σu-, only the rotational levels with even N can be populated.
            return bool(n_qn % 2 == 0)

        raise ValueError(f"State {self.name} not supported.")
