# module test

import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt

import plot
from molecule import Molecule
from simulation import Simulation, SimType

def main():
    # NOTE: 05/06/24 - only O2 is working right now, selection rules, etc. for other molecules are
    #       currently not implemented

    temp: float = 300.0    # [K]
    pres: float = 101325.0 # [Pa]

    rot_qn_up: int = 22
    rot_qn_lo: int = 21

    mol_o2 = Molecule('o2', 'o', 'o')

    bands_sim = [(2, 0)]
    bands_lif = [(7, v) for v in range(12, -1, -1)]

    o2_sim = Simulation(mol_o2, temp, pres, np.arange(0, 36), 'b3su', 'x3sg', bands_sim,
                        SimType.ABSORPTION)

    o2_lif = Simulation(mol_o2, temp, pres, np.arange(0, 36), 'b3su', 'x3sg', bands_lif,
                        SimType.LIF)

    palette_lif = plt.cycler('color', plt.cm.tab20c.colors).by_key()['color']
    colors_lif  = [matplotlib.colors.to_hex(color) for color in palette_lif]
    colors_sim  = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # FIXME: 05/06/24 - each time a plot is called (plot_line, plot_info, etc.), the vibrational
    #        bands are iterated through, meaning the wavelength and intensity info for each band is
    #        potentially being re-calculated several times

    # TODO: 06/04/24 - remove SimType and just have the option to plot absorption or emission
    #       through the plots

    plot.plot_lif(o2_lif, rot_qn_up, rot_qn_lo, colors_lif)
    plot.plot_lif_info(o2_lif, rot_qn_up, rot_qn_lo)
    plot.plot_show()

    # # plot.plot_line(o2_sim, colors_lif)
    # plot.plot_conv(o2_sim, colors_lif)
    # # plot.plot_samp('pgopher', 'red', 'stem')
    # plot.plot_samp('harvard/harvard20', 'red', 'plot')
    # plot.plot_residual(o2_sim, 'green', 'harvard/harvard20')

    # # testing how the PGOPHER data compares when convolved (estimating predissociation rates)
    # from test import cwls, cins

    # plt.plot(cwls, cins, 'black', label='pgopher')

    # plot.plot_show()

if __name__ == '__main__':
    main()
