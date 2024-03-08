# module test

import matplotlib.pyplot as plt
import numpy as np

from simulation import Simulation
from molecule import Molecule
import plot

def main():
    default_temp = 300    # [K]
    default_pres = 101325 # [Pa]

    mol_o2  = Molecule('o2', 'o', 'o')
    mol_o2p = Molecule('o2+', 'o', 'o')

    o2_sim  = Simulation(mol_o2, default_temp, default_pres,
                         np.arange(0, 36, 1), 'b3su', 'x3sg', [(2, 0)])
    o2p_sim = Simulation(mol_o2p, default_temp, default_pres,
                         np.arange(0.5, 35.5, 1), 'a2pu', 'x2pg', [(0, 0)])

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors     = prop_cycle.by_key()['color']

    plot.plot_conv(o2_sim, colors[0])
    plot.plot_samp('harvard/harvard20', colors[1], 'plot')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
