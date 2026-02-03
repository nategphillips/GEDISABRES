# module constants.py
"""Provides physical and molecular constants."""

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

import lookup
from sim_props import InversionSymmetry, ReflectionSymmetry, TermSymbol

# Data from NIST CODATA - <https://physics.nist.gov/cuu/Constants/index.html>.
# Avodagro constant [1/mol]
AVOGD = 6.02214076e23
# Boltzmann constant [J/K]
BOLTZ = 1.380649e-23
# Speed of light [cm/s]
LIGHT = 2.99792458e10
# Planck constant [J*s]
PLANC = 6.62607015e-34
# Vacuum electric permittivity [F/m]
EPERM = 8.8541878188e-12

# Atomic masses [g/mol] (same as [amu])
# Data from Atomic Mass Evaluation 2020 - <https://www.anl.gov/phy/reference/ame-2020-mass1mas20>.
ATOMIC_MASSES = lookup.mass_lookup()

# Nuclear spin [-]
# Data from NUBASE 2020 - <https://www.anl.gov/phy/reference/nubase-2020-nubase4mas20>.
NUCLEAR_SPIN = lookup.spin_lookup()

# Mapping ΔQN = QN' - QN'' to a branch name. As far as I know, the names O, P, Q, R, and S are all
# standard, while T and N are used in PGOPHER to denote +/- 3 transitions.
BRANCH_NAME_MAP = {-3: "N", -2: "O", -1: "P", 0: "Q", +1: "R", +2: "S", +3: "T"}

# Electric dipole moment [C*m]
# Data from NIST Diatomic Spectral Database Holdings - <https://physics.nist.gov/cgi-bin/MolSpec/diperiodic.pl>.
# If available, the ground state electric dipole moment is used. See also
# <https://cccbdb.nist.gov/diplistx.asp>. Note that homonuclear diatomics have no permanent dipole
# moment.
DIPOLE_MOMENT = {"16O16O": 0.0, "14N16O": 0.52943e-30, "16O1H": 5.56245e-30}

# Mappings from enums to strings for use with the dictionaries below.
TERM_SYMBOL_MAP = {
    TermSymbol.SIGMA: "S",
    TermSymbol.PI: "P",
    TermSymbol.DELTA: "D",
}
INVERSION_SYMMETRY_MAP = {
    InversionSymmetry.NONE: "",
    InversionSymmetry.GERADE: "g",
    InversionSymmetry.UNGERADE: "u",
}
REFLECTION_SYMMETRY_MAP = {
    ReflectionSymmetry.NONE: "",
    ReflectionSymmetry.PLUS: "+",
    ReflectionSymmetry.MINUS: "-",
}

# Internuclear distance [m]
# Data from NIST Chemistry WebBook - <https://webbook.nist.gov/chemistry/>.
INTERNUCLEAR_DISTANCE = {
    "16O16O": {"X3Sg-": 1.20752e-10, "B3Su-": 1.6042e-10},
    "14N16O": {"X2P": 1.15077e-10, "A2S+": 1.06434e-10},
    "16O1H": {"X2P": 0.96966e-10, "A2S+": 1.0121e-10},
}

# Electronic energies [1/cm]
# Data from NIST Chemistry WebBook - <https://webbook.nist.gov/chemistry/>.
ELECTRONIC_ENERGIES = {
    "16O16O": {
        "X3Sg-": 0.0,
        "a1Pg": 7918.1,
        "b1Sg+": 13195.1,
        "c1Su-": 33057.0,
        "A3Pu": 34690.0,
        "A3Su+": 35397.8,
        "B3Su-": 49793.28,
    },
    "14N16O": {
        "X2P": 0.0,
        "a4P": 38440.0,
        "A2S+": 43965.7,
        "B2P": 45942.6,
        "b4S-": 48680.0,
        "C2P": 52126.0,
        "D2S+": 53084.7,
    },
    "16O1H": {
        "X2P": 0.0,
        "A2S+": 32684.1,
        "B2S+": 69774.0,
        "D2S-": 82130.0,
        "C2S+": 89459.1,
    },
}

# TODO: 26/01/30 - Listing these is useless since degeneracy is directly calculable from the term
#       symbols themselves.

# Electronic degeneracies [-]
# Data from Table 1.4 of "Nonequilibrium Hypersonic Aerodynamics" by Chul Park - <https://ntrs.nasa.gov/citations/19910029860>.
ELECTRONIC_DEGENERACIES = {
    "16O16O": {"X3Sg-": 3, "a1Pg": 2, "b1Sg+": 1, "c1Su-": 1, "A3Pu": 6, "A3Su+": 3, "B3Su-": 3},
    "14N16O": {"X2P": 4, "a4P": 8, "A2S+": 2, "B2P": 4, "b4S-": 4, "C2P": 4, "D2S+": 2},
    "16O1H": {"X2P": 4, "A2S+": 2, "B2S+": 2, "D2S-": 2, "C2S+": 2},
}

# A somewhat arbitrary cutoff value for the Hönl-London factors. If the HLF of a line is lower than
# this value, the transition is considered "forbidden" and the line is not simulated.
HONL_LONDON_CUTOFF = 1e-6
# The maximum v' and v'' levels to be used when evaluating the vibrational partition function for
# a simulation using Dunham coefficients.
V_QN_MAX_DUNHAM = 20
