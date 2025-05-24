# module constants
'''
Physical and diatomic constants for the X3Σg- to B3Σu- transition of oxygen.
'''

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

import pandas as pd
import numpy as np

import input as inp

# physical constants
BOLTZ = 1.380649e-23   # Boltzmann constant [J/K]
PLANC = 6.62607015e-34 # Planck constant    [J*s]
LIGHT = 2.99792458e10  # speed of light     [cm/s]
AVOGD = 6.02214076e23  # Avodagro constant  [1/mol]

# choose the first column as the index so that the states can be isolated
MOLECULAR_CONSTS = pd.read_csv(f'../data/molecular_constants/{inp.MOLECULE}.csv', index_col=0)

# access the dataframe according to the desired state, converting to a dict since the extra
# functionality of the pandas series isn't needed
CONSTS_UP = MOLECULAR_CONSTS.loc[inp.STATE_UP].to_dict()
CONSTS_LO = MOLECULAR_CONSTS.loc[inp.STATE_LO].to_dict()

# atomic masses for diatoms [amu]
MASS_DATA = pd.read_csv('../data/molecular_masses.csv', index_col=0).loc[inp.MOLECULE].to_dict()

# mass of atoms [kg]
MASS_ATOM_1 = MASS_DATA['mass_1'] / AVOGD / 1e3
MASS_ATOM_2 = MASS_DATA['mass_2'] / AVOGD / 1e3

# molecular mass [kg]
MASS_MOLECULE = MASS_ATOM_1 + MASS_ATOM_2

# reduced mass [kg]
MASS_REDUCED = (MASS_ATOM_1 * MASS_ATOM_2) / MASS_MOLECULE

# ground state cross-sectional area
CROSS_SEC = np.pi * (CONSTS_LO['rad'] + CONSTS_LO['rad'])**2
