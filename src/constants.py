# module constants.py
"""Provides physical and molecular constants."""

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

from enums import InversionSymmetry, ReflectionSymmetry, TermSymbol

# Data from NIST CODATA - <https://physics.nist.gov/cuu/Constants/index.html>
# Avodagro constant [1/mol]
AVOGD: float = 6.02214076e23
# Boltzmann constant [J/K]
BOLTZ: float = 1.380649e-23
# Speed of light [cm/s]
LIGHT: float = 2.99792458e10
# Planck constant [J*s]
PLANC: float = 6.62607015e-34

# Atomic masses [g/mol]
# Data from IUPAC: Atomic Weights of The Elements 2023 - <https://iupac.qmul.ac.uk/AtWt/>
ATOMIC_MASSES: dict[str, float] = {"H": 1.008, "N": 14.007, "O": 15.999}

# Mapping ΔQN = QN' - QN'' to a branch name. As far as I know, the names O, P, Q, R, and S are all
# standard, while T and N are used in PGOPHER to denote +/- 3 transitions.
BRANCH_NAME_MAP: dict[int, str] = {-3: "N", -2: "O", -1: "P", 0: "Q", +1: "R", +2: "S", +3: "T"}

# TODO: 25/07/17 - Different isotopes of the same nuclei have different nuclear spins, so this table
#       should also contain the atomic mass number.

# Nuclear spin [-]
# Data from NUBASE 2020 - <https://doi.org/10.1088/1674-1137/abddae>
NUCLEAR_SPIN: dict[str, Fraction] = {"H": Fraction(1, 2), "N": Fraction(1), "O": Fraction(0)}

# Mappings from enums to strings for use with the dictionaries below.
TERM_SYMBOL_MAP: dict[TermSymbol, str] = {
    TermSymbol.SIGMA: "S",
    TermSymbol.PI: "P",
    TermSymbol.DELTA: "D",
}
INVERSION_SYMMETRY_MAP: dict[InversionSymmetry, str] = {
    InversionSymmetry.NONE: "",
    InversionSymmetry.GERADE: "g",
    InversionSymmetry.UNGERADE: "u",
}
REFLECTION_SYMMETRY_MAP: dict[ReflectionSymmetry, str] = {
    ReflectionSymmetry.NONE: "",
    ReflectionSymmetry.PLUS: "+",
    ReflectionSymmetry.MINUS: "-",
}

# Internuclear distance [m]
# Data from NIST Chemistry WebBook - <https://webbook.nist.gov/chemistry/>
INTERNUCLEAR_DISTANCE: dict[str, dict[str, float]] = {
    "O2": {"X3Sg-": 1.20752e-10, "B3Su-": 1.6042e-10},
    "NO": {"X2P": 1.15077e-10, "A2S+": 1.06434e-10},
    "OH": {"X2P": 0.96966e-10, "A2S+": 1.0121e-10},
}

# Electronic energies [1/cm]
# Data from NIST Chemistry WebBook - <https://webbook.nist.gov/chemistry/>
ELECTRONIC_ENERGIES: dict[str, dict[str, float]] = {
    "O2": {
        "X3Sg-": 0.0,
        "a1Pg": 7918.1,
        "b1Sg+": 13195.1,
        "c1Su-": 33057.0,
        "A3Pu": 34690.0,
        "A3Su+": 35397.8,
        "B3Su-": 49793.28,
    },
    "NO": {
        "X2P": 0.0,
        "a4P": 38440.0,
        "A2S+": 43965.7,
        "B2P": 45942.6,
        "b4S-": 48680.0,
        "C2P": 52126.0,
        "D2S+": 53084.7,
    },
    "OH": {
        "X2P": 0.0,
        "A2S+": 32684.1,
        "B2S+": 69774.0,
        "D2S-": 82130.0,
        "C2S+": 89459.1,
    },
}

# Electronic degeneracies [-]
# Data from Table 1.4 of "Nonequilibrium Hypersonic Aerodynamics" by Chul Park - <https://ntrs.nasa.gov/citations/19910029860>
# (Degeneracies are also directly calculable from the term symbols themselves.)
ELECTRONIC_DEGENERACIES: dict[str, dict[str, int]] = {
    "O2": {"X3Sg-": 3, "a1Pg": 2, "b1Sg+": 1, "c1Su-": 1, "A3Pu": 6, "A3Su+": 3, "B3Su-": 3},
    "NO": {"X2P": 4, "a4P": 8, "A2S+": 2, "B2P": 4, "b4S-": 4, "C2P": 4, "D2S+": 2},
    "OH": {"X2P": 4, "A2S+": 2, "B2S+": 2, "D2S-": 2, "C2S+": 2},
}

# A somewhat arbitrary cutoff value for the Hönl-London factors. If the HLF of a line is lower than
# this value, the transition is considered "forbidden" and the line is not simulated.
HONL_LONDON_CUTOFF: float = 1e-6
# The maximum v' and v'' levels to be used when evaluating the vibrational partition function for
# a simulation using Dunham coefficients.
V_QN_MAX_DUNHAM: int = 20
