# module constants
'''
Physical and diatomic constants for the X3Σg- to B3Σu- transition of oxygen.
'''

import pandas as pd

import input as inp

# global constants
## physical constants
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
