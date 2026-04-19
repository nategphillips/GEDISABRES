# Constants.jl
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

# Data from NIST CODATA - <https://physics.nist.gov/cuu/Constants/index.html>.
# Avodagro constant [1/mol].
const AVOGD = 6.02214076e23
# Boltzmann constant [J/K].
const BOLTZ = 1.380649e-23
# Speed of light [cm/s].
const LIGHT = 2.99792458e10
# Planck constant [J*s].
const PLANC = 6.62607015e-34
# Vacuum electric permittivity [F/m].
const EPERM = 8.8541878188e-12

# TODO: 26/04/18 - Implement these once the AME and NUBASE parsers are ready.

# # Atomic masses [amu].
# # Data from Atomic Mass Evaluation 2020 - <https://www.anl.gov/phy/reference/ame-2020-mass1mas20>.
# ATOMIC_MASSES = lookup.mass_lookup()

# # Nuclear spin [-].
# # Data from NUBASE 2020 - <https://www.anl.gov/phy/reference/nubase-2020-nubase4mas20>.
# NUCLEAR_SPIN = lookup.spin_lookup()

# Mapping ΔQN = QN' - QN'' to a branch name.
# As far as I know, the names O, P, Q, R, and S are all standard, while T and N are used in PGOPHER
# to denote +/- 3 transitions.
const BRANCH_NAME_MAP = Dict(
    "N" => -3, "O" => -2, "P" => -1, "Q" => 0, "R" => +1, "S" => +2, "T" => +3
)

# Electric dipole moment [C*m].
# Data from NIST Diatomic Spectral Database Holdings - <https://physics.nist.gov/cgi-bin/MolSpec/diperiodic.pl>.
# If available, the ground state electric dipole moment is used.
# See also <https://cccbdb.nist.gov/diplistx.asp>.
# Note that homonuclear diatomics have no permanent dipole moment.
const DIPOLE_MOMENT = Dict("16O16O" => 0.0, "14N16O" => 0.52943e-30, "16O1H" => 5.56245e-30)

# Mappings from enums to strings for use with the dictionaries below.
const TERM_SYMBOL_MAP = Dict(
    TermSymbol.term_sigma => "S", TermSymbol.term_pi => "P", TermSymbol.term_delta => "D"
)
const INVERSION_SYMMETRY_MAP = Dict(
    InversionSymmetry.inv_none => "",
    InversionSymmetry.inv_gerade => "g",
    InversionSymmetry.inv_ungerade => "u",
)
const REFLECTION_SYMMETRY_MAP = Dict(
    ReflectionSymmetry.ref_none => "",
    ReflectionSymmetry.ref_plus => "+",
    ReflectionSymmetry.ref_minus => "-",
)

# Internuclear distance [m].
# Data from NIST Chemistry WebBook - <https://webbook.nist.gov/chemistry/>.
const INTERNUCLEAR_DISTANCE = Dict(
    "16O16O" => Dict("X3Sg-" => 1.20752e-10, "B3Su-" => 1.6042e-10),
    "14N16O" => Dict("X2P" => 1.15077e-10, "A2S+" => 1.06434e-10),
    "16O1H" => Dict("X2P" => 0.96966e-10, "A2S+" => 1.0121e-10),
)

# Electronic energies [1/cm].
# Data from NIST Chemistry WebBook - <https://webbook.nist.gov/chemistry/>.
const ELECTRONIC_ENERGIES = Dict(
    "16O16O" => Dict(
        "X3Sg-" => 0.0,
        "a1Pg" => 7918.1,
        "b1Sg+" => 13195.1,
        "c1Su-" => 33057.0,
        "A3Pu" => 34690.0,
        "A3Su+" => 35397.8,
        "B3Su-" => 49793.28,
    ),
    "14N16O" => Dict(
        "X2P" => 0.0,
        "a4P" => 38440.0,
        "A2S+" => 43965.7,
        "B2P" => 45942.6,
        "b4S-" => 48680.0,
        "C2P" => 52126.0,
        "D2S+" => 53084.7,
    ),
    "16O1H" => Dict(
        "X2P" => 0.0, "A2S+" => 32684.1, "B2S+" => 69774.0, "D2S-" => 82130.0, "C2S+" => 89459.1
    ),
)

# TODO: 26/01/30 - Listing these is useless since degeneracy is directly calculable from the term
#       symbols themselves.

# Electronic degeneracies [-].
# Data from Table 1.4 of "Nonequilibrium Hypersonic Aerodynamics" by Chul Park - <https://ntrs.nasa.gov/citations/19910029860>.
const ELECTRONIC_DEGENERACIES = Dict(
    "16O16O" => Dict(
        "X3Sg-" => 3,
        "a1Pg" => 2,
        "b1Sg+" => 1,
        "c1Su-" => 1,
        "A3Pu" => 6,
        "A3Su+" => 3,
        "B3Su-" => 3,
    ),
    "14N16O" =>
        Dict("X2P" => 4, "a4P" => 8, "A2S+" => 2, "B2P" => 4, "b4S-" => 4, "C2P" => 4, "D2S+" => 2),
    "16O1H" => Dict("X2P" => 4, "A2S+" => 2, "B2S+" => 2, "D2S-" => 2, "C2S+" => 2),
)

# A somewhat arbitrary cutoff value for the Hönl-London factors.
# If the HLF of a line is lower than this value, the transition is considered "forbidden" and the
# line is not simulated.
const HONL_LONDON_CUTOFF = 1e-6

# The maximum v' and v'' levels to be used when evaluating the vibrational partition function for
# a simulation using Dunham coefficients.
const V_QN_MAX_DUNHAM = 20
