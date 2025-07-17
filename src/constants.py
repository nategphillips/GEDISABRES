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

from enums import InversionSymmetry, ReflectionSymmetry, TermSymbol

# Avodagro constant [1/mol]
AVOGD: float = 6.02214076e23
# Boltzmann constant [J/K]
BOLTZ: float = 1.380649e-23
# Speed of light [cm/s]
LIGHT: float = 2.99792458e10
# Planck constant [J*s]
PLANC: float = 6.62607015e-34

# Atomic masses [g/mol]
ATOMIC_MASSES: dict[str, float] = {"O": 15.999}

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
# Data from NIST Chemistry WebBook
INTERNUCLEAR_DISTANCE: dict[str, dict[str, float]] = {
    "O2": {"X3Sg-": 1.20752e-10, "B3Su-": 1.6042e-10}
}

# Electronic energies [1/cm]
# Data from NIST Chemistry WebBook
ELECTRONIC_ENERGIES: dict[str, dict[str, float]] = {
    "O2": {
        "X3Sg-": 0.0,
        "a1Pg": 7918.1,
        "b1Sg+": 13195.1,
        "c1Su-": 33057.0,
        "A3Pu": 34690.0,
        "A3Su+": 35397.8,
        "B3Su-": 49793.28,
    }
}

# Electronic degeneracies [-]
# Data from Park, 1990
ELECTRONIC_DEGENERACIES: dict[str, dict[str, int]] = {
    "O2": {"X3Sg-": 3, "a1Pg": 2, "b1Sg+": 1, "c1Su-": 1, "A3Pu": 6, "A3Su+": 3, "B3Su-": 3}
}

# A somewhat arbitrary cutoff value for the Hönl-London factors. If the HLF of a line is lower than
# this value, the transition is considered "forbidden" and the line is not simulated.
HONL_LONDON_CUTOFF: float = 1e-6
