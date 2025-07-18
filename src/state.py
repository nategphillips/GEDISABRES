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

from enum import Enum
from fractions import Fraction
from functools import cached_property

import polars as pl

import utils
from constants import INVERSION_SYMMETRY_MAP, REFLECTION_SYMMETRY_MAP, TERM_SYMBOL_MAP
from enums import InversionSymmetry, NuclearStatistics, ReflectionSymmetry, TermSymbol
from molecule import Molecule


class Sign(Enum):
    """Denotes a plus or minus sign in the expression for homonuclear degeneracy."""

    PLUS = 1
    MINUS = 2


def homonuclear_degeneracy(nuclear_spin: Fraction, sign: Sign) -> Fraction:
    """Computes the homonuclear degeneracy 0.5[(2I + 1)^2 ± (2I + 1)].

    Equation (5.16) in "Spectroscopy and Optical Diagnostics for Gases" by Hanson, et al.

    Args:
        nuclear_spin (Fraction): Nuclear spin of either atom.
        sign (Sign): Which sign to use in the computation.

    Returns:
        Fraction: Homonuclear degeneracy.
    """
    if sign == Sign.PLUS:
        return Fraction(1, 2) * ((2 * nuclear_spin + 1) ** 2 + (2 * nuclear_spin + 1))

    return Fraction(1, 2) * ((2 * nuclear_spin + 1) ** 2 - (2 * nuclear_spin + 1))


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

    def nuclear_partition_fn(self) -> Fraction:
        """Computes the nuclear partition function.

        Equation (5.9) in "Spectroscopy and Optical Diagnostics for Gases" by Hanson, et al.

        Returns:
            Fraction: Nuclear partition function.
        """
        return (2 * self.molecule.atom_1.nuclear_spin + 1) * (
            2 * self.molecule.atom_2.nuclear_spin + 1
        )

    @cached_property
    def nuclear_degeneracy(self) -> tuple[Fraction, Fraction]:
        """Nuclear degeneracy of the electronic state.

        Based on Table 5.2 in "Spectroscopy and Optical Diagnostics for Gases" by Hanson, et al.

        Returns:
            tuple[int, int]: Degeneracy scaling factors corresponding to even N and odd N.
        """
        atom_1 = self.molecule.atom_1
        atom_2 = self.molecule.atom_2

        # Only homonuclear molecules have different degeneracies for states with different
        # wavefunction symmetries. Additionally, states other than Σ usually have negligible nuclear
        # spin effects.
        if self.molecule.is_homonuclear and self.term_symbol == TermSymbol.SIGMA:
            # If both constituent nuclei are fermions, Fermi statistics apply.
            if (atom_1.nuclear_statistics == NuclearStatistics.FERMI) and (
                atom_2.nuclear_statistics == NuclearStatistics.FERMI
            ):
                # Σg+ or Σu-
                if (
                    self.inversion_symmetry == InversionSymmetry.GERADE
                    and self.reflection_symmetry == ReflectionSymmetry.PLUS
                ) or (
                    self.inversion_symmetry == InversionSymmetry.UNGERADE
                    and self.reflection_symmetry == ReflectionSymmetry.MINUS
                ):
                    # Even N (-), odd N (+)
                    return homonuclear_degeneracy(
                        atom_1.nuclear_spin, Sign.MINUS
                    ), homonuclear_degeneracy(atom_1.nuclear_spin, Sign.PLUS)
                # Σu+ or Σg-
                if (
                    self.inversion_symmetry == InversionSymmetry.UNGERADE
                    and self.reflection_symmetry == ReflectionSymmetry.PLUS
                ) or (
                    self.inversion_symmetry == InversionSymmetry.GERADE
                    and self.reflection_symmetry == ReflectionSymmetry.MINUS
                ):
                    # Even N (+), odd N (-)
                    return homonuclear_degeneracy(
                        atom_1.nuclear_spin, Sign.PLUS
                    ), homonuclear_degeneracy(atom_1.nuclear_spin, Sign.MINUS)
            # If there are no fermions, or if there is only one fermion, Bose statistics apply.
            else:
                # Σg+ or Σu-
                if (
                    self.inversion_symmetry == InversionSymmetry.GERADE
                    and self.reflection_symmetry == ReflectionSymmetry.PLUS
                ) or (
                    self.inversion_symmetry == InversionSymmetry.UNGERADE
                    and self.reflection_symmetry == ReflectionSymmetry.MINUS
                ):
                    # Even N (+), odd N (-)
                    return homonuclear_degeneracy(
                        atom_1.nuclear_spin, Sign.PLUS
                    ), homonuclear_degeneracy(atom_1.nuclear_spin, Sign.MINUS)
                # Σu+ or Σg-
                if (
                    self.inversion_symmetry == InversionSymmetry.UNGERADE
                    and self.reflection_symmetry == ReflectionSymmetry.PLUS
                ) or (
                    self.inversion_symmetry == InversionSymmetry.GERADE
                    and self.reflection_symmetry == ReflectionSymmetry.MINUS
                ):
                    # Even N (-), odd N (+)
                    return homonuclear_degeneracy(
                        atom_1.nuclear_spin, Sign.MINUS
                    ), homonuclear_degeneracy(atom_1.nuclear_spin, Sign.PLUS)

        heteronuclear_degeneracy: Fraction = (2 * atom_1.nuclear_spin + 1) * (
            2 * atom_2.nuclear_spin + 1
        )

        # Nuclear degeneracies for heteronuclear diatomics are not dependent on even or odd N.
        return heteronuclear_degeneracy, heteronuclear_degeneracy
