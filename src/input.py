# module input
'''
User-defined control inputs for quantum numbers and plotting, among others.
'''

import numpy as np

# Global temperature and pressure
# By default, all vibrational transitions are considered at the same temperature and pressure
TEMP: float = 300.0
PRES: float = 101325.0

# Rotational levels
# Cosby predissociation data only goes up to N = 36
ROT_LVLS: np.ndarray = np.arange(0, 37, 1)

# List of vibrational transitions considered in (v', v'') format
VIB_BANDS: list[tuple] = [(2, 0), (1, 0)]

# Line data
LINE_DATA: bool = True

# Convolved data
CONV_DATA: bool = True
# Granulatity of the convolved data
CONV_GRAN: int  = 10000

# Sample data
SAMP_DATA: bool = True
SAMP_FILE: list[str] = ['harvard', 'pgopher']
SAMP_COLS: list[str] = ['purple', 'skyblue']
SAMP_LABL: list[str] = ['Harvard Data', 'PGOPHER Data']

# General plotting
PLOT_SAVE:  bool  = False
PLOT_PATH:  str   = '../img/example.webp'
DPI:        int   = 96
SCREEN_RES: tuple = (1920, 1080)
