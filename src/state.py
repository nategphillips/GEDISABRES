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

from fractions import Fraction
from functools import cached_property

import numpy as np
import polars as pl
from numpy.typing import NDArray

import utils
from constants import INVERSION_SYMMETRY_MAP, REFLECTION_SYMMETRY_MAP, TERM_SYMBOL_MAP
from enums import (
    ConstantsType,
    InversionSymmetry,
    NuclearStatistics,
    ReflectionSymmetry,
    Sign,
    TermSymbol,
)
from molecule import Molecule


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
        constants_type: ConstantsType = ConstantsType.DUNHAM,
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
            constants_type (ConstantsType, optional): Whether to use the Dunham expansion for
                molecular parameters or to specify them per vibrational level. Defaults to Dunham.
        """
        self.molecule: Molecule = molecule
        self.letter: str = letter
        self.spin_multiplicity: int = spin_multiplicity
        self.term_symbol: TermSymbol = term_symbol
        self.inversion_symmetry: InversionSymmetry = inversion_symmetry
        self.reflection_symmetry: ReflectionSymmetry = reflection_symmetry
        self.constants_type: ConstantsType = constants_type

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
    def all_constants(self) -> pl.DataFrame:
        """Return all constants (either Dunham or per-level) for the given state.

        Returns:
            pl.DataFrame: DataFrame of all constants.
        """
        match self.constants_type:
            case ConstantsType.PERLEVEL:
                pathname = "states"
            case ConstantsType.DUNHAM:
                pathname = "dunham"

        return pl.read_csv(
            utils.get_data_path("data", self.molecule.name, pathname, f"{self.name}.csv")
        )

    def constants_vqn(self, v_qn: int) -> dict[str, float]:
        """Return all available constants for the selected rotational level.

        Designed to work the same if per-level constants or Dunham parameters are used. If per-level
        constants are used, just return the row of constants corresponding to the desired v. If
        Dunham parameters are used, use the equilibrium parameters to solve for the parameters as
        functions of v, then return the row of constants.

        Args:
            v_qn (int): Vibrational quantum number v.

        Returns:
            dict[str, float]: All available constants for the desired vibrational level.
        """
        if self.constants_type == ConstantsType.PERLEVEL:
            return self.all_constants.row(v_qn, named=True)

        # NOTE: 25/07/21 - Calculations for the vibrational term value G start with the power of
        #       (v + 0.5) equal to one, while all other constants start with zero:
        #
        #       G = ω_e(v + 0.5) − ω_ex_e(v + 0.5)^2 + ω_ey_e(v + 0.5)^3 + ...
        #       B = B_e - α_e(v + 0.5) + γ_e(v + 0.5)^2 + ...
        #       D = D_e − β_e(v + 0.5) + ...
        #
        #       The signs of all the constants are set through the data files themselves such that
        #       they can all be summed here without worry.

        base: float = v_qn + 0.5
        coeffs: NDArray[np.float64] = self.all_constants.to_numpy()

        # The maximum number of rows for the entire bundle of constants. Some rows will have more or
        # less constants than others, which means that terms like (0.0)(v + 0.5)^3 will get computed
        # for columns with 0s in place of actual constants. This is fine other than being slightly
        # inefficient.
        n_terms: int = coeffs.shape[0]
        row_vals: dict[str, float] = {}

        for j, column_name in enumerate(self.all_constants.columns):
            start_power = 1 if column_name == "G" else 0
            powers = base ** np.arange(start_power, start_power + n_terms)
            row_vals[column_name] = np.dot(coeffs[:, j], powers)

        return row_vals

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
