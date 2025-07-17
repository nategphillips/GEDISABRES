# module state.py
"""Contains the implementation of the State class."""

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

from functools import cached_property

import polars as pl

import utils
from constants import INVERSION_SYMMETRY_MAP, REFLECTION_SYMMETRY_MAP, TERM_SYMBOL_MAP
from enums import InversionSymmetry, ReflectionSymmetry, TermSymbol
from molecule import Molecule


class State:
    """Represents an electronic state of a particular molecule."""

    def __init__(
        self,
        molecule: Molecule,
        letter: str,
        spin_multiplicity: int,
        term_symbol: TermSymbol,
        inversion_symmetry: InversionSymmetry = InversionSymmetry.NONE,
        reflection_symmetry: ReflectionSymmetry = ReflectionSymmetry.NONE,
    ) -> None:
        """Initialize class variables.

        Args:
            molecule (Molecule): Parent molecule.
            letter (str): Letter corresponding to the electronic state, e.g., A, B, X, etc.
            spin_multiplicity (int): Spin multiplicity.
            term_symbol (TermSymbol): Term symbol associated with the electronic state.
            inversion_symmetry (InversionSymmetry, optional): Symmetry w.r.t. inversion (g/u).
                Defaults to None.
            reflection_symmetry (ReflectionSymmetry, optional): Symmetry w.r.t. reflection (+/-).
                Defaults to None.
        """
        self.molecule: Molecule = molecule
        self.letter: str = letter
        self.spin_multiplicity: int = spin_multiplicity
        self.term_symbol: TermSymbol = term_symbol
        self.inversion_symmetry: InversionSymmetry = inversion_symmetry
        self.reflection_symmetry: ReflectionSymmetry = reflection_symmetry

    @cached_property
    def name(self) -> str:
        """Maps the properties of an electronic state to a short string representation.

        Example: X 3 SIGMA GERADE MINUS -> X3Sg-.

        Returns:
            str: A string representation of the full molecular term symbol, e.g., X3Sg-.
        """
        return (
            self.letter
            + str(self.spin_multiplicity)
            + TERM_SYMBOL_MAP[self.term_symbol]
            + INVERSION_SYMMETRY_MAP[self.inversion_symmetry]
            + REFLECTION_SYMMETRY_MAP[self.reflection_symmetry]
        )

    @cached_property
    def constants(self) -> dict[str, list[float]]:
        """Return the molecular constants for the specified electronic state in [1/cm].

        Args:
            molecule (str): Parent molecule.
            term_symbol (TermSymbol): Name of the electronic state.

        Returns:
            dict[str, list[float]]: A `dict` of molecular constants for the electronic state.
        """
        return pl.read_csv(
            utils.get_data_path("data", self.molecule.name, "states", f"{self.name}.csv")
        ).to_dict(as_series=False)

    def is_allowed(self, n_qn: int) -> bool:
        """Return whether or not the selected rotational level is allowed.

        Args:
            n_qn (int): Rotational quantum number N.

        Raises:
            ValueError: If the electronic state does not exist.

        Returns:
            bool: True if the selected rotational level is allowed.
        """
        if (
            self.molecule.atom_1.name == self.molecule.atom_2.name
        ) and self.term_symbol == TermSymbol.SIGMA:
            if self.name == "X3Sg-":
                # For X3Σg-, only the rotational levels with odd N can be populated.
                return bool(n_qn % 2 == 1)
            if self.name == "B3Su-":
                # For B3Σu-, only the rotational levels with even N can be populated.
                return bool(n_qn % 2 == 0)

        raise ValueError(f"State {self.term_symbol} not supported.")
