# module test

import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

from simulation import Simulation, SimType
from molecule import Molecule
import plot


def main():
    # NOTE: 05/06/24 - only O2 is working right now, selection rules, etc. for other molecules are
    #       currently not implemented

    temp: float = 300.0    # [K]
    pres: float = 101325.0 # [Pa]

    rot_qn_up: int = 22
    rot_qn_lo: int = 21

    mol_o2 = Molecule('o2', 'o', 'o')

    bands_sim = [(2, 0)]
    bands_lif = [(7, v) for v in range(1, -1, -1)]

    o2_sim = Simulation(mol_o2, temp, pres, np.arange(0, 36), 'b3su', 'x3sg', bands_sim,
                        SimType.ABSORPTION)

    # FIXME: 05/06/24 - LIF currently not working since all lines are simulated in each band in
    #        order for the the rotational partition function to work properly; need to only plot the
    #        selected lines with the correct quantum numbers

    # o2_lif = Simulation(mol_o2, temp, pres, np.arange(0, 36), 'b3su', 'x3sg', bands_lif, rot_qn_up,
    #                     rot_qn_lo)

    palette_lif = plt.cycler('color', plt.cm.tab20c.colors).by_key()['color']
    colors_lif  = [matplotlib.colors.to_hex(color) for color in palette_lif]
    colors_sim  = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # FIXME: 05/06/24 - each time a plot is called (plot_line, plot_info, etc.), the vibrational
    #        bands are iterated through, meaning the wavelength and intensity info for each band is
    #        potentially being re-calculated several times

    # plot.plot_line(o2_lif, colors_lif)
    # plot.plot_info(o2_lif)
    # plot.plot_show()

    plot.plot_line(o2_sim, colors_lif)
    plot.plot_conv(o2_sim, colors_lif)
    plot.plot_samp('pgopher', 'red', 'stem')
    plot.plot_samp('harvard/harvard20', 'green', 'plot')
    plot.plot_show()


if __name__ == '__main__':
    main()
