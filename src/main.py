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

    temp: float = 3000.0
    pres: float = 101325.0

    bands: list[tuple[int, int]] = [(x, v) for x in range(0, 7) for v in range(18, -1, -1)]

    o2_mol: Molecule = Molecule('o2', 'o', 'o')

    o2_sim: Simulation = Simulation(o2_mol, temp, pres, np.arange(0, 36), 'b3su', 'x3sg', bands,
                                    SimType.ABSORPTION)

    palette: list[tuple] = plt.cycler('color', plt.cm.tab20c.colors).by_key()['color']
    colors:  list[str]   = [matplotlib.colors.to_hex(color) for color in palette]

    # colors for lots of bands
    cmap      = plt.get_cmap('rainbow')
    num_bands = 19 * 7
    colors    = [matplotlib.colors.to_hex(cmap(i / (num_bands - 1))) for i in range(num_bands)]

    # FIXME: 05/06/24 - each time a plot is called (plot_line, plot_info, etc.), the vibrational
    #        bands are iterated through, meaning the wavelength and intensity info for each band is
    #        potentially being re-calculated several times

    # FIXME: 05/06/24 - the intensities are no longer normalized after adding the vibrational
    #        partition function; normalization would have to be performed on all vibrational bands
    #        at once

    plot.plot_line(o2_sim, colors)
    # plot.plot_samp('harvard/harvard20', 'red', 'plot')
    # plot.plot_residual(o2_sim, 'green', 'harvard/harvard20')

    # testing how the PGOPHER data compares when convolved (estimating predissociation rates)
    # from test import cwls, cins

    # plt.plot(cwls, cins, 'black', label='pgopher')

    plot.plot_show()

if __name__ == '__main__':
    main()
