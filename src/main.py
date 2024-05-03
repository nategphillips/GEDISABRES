# module test

import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

from simulation import Spectra, LIF
from molecule import Molecule
import plot


def main():
    default_temp = 300    # [K]
    default_pres = 101325  # [Pa]

    rot_qn_up = 22
    rot_qn_lo = 21

    vib_qn_up = 2
    vib_qn_lo_max = 12

    mol_o2 = Molecule('o2', 'o', 'o')

    bands = [(0, v) for v in range(18, -1, -1)]

    o2_sim = Spectra(mol_o2, default_temp, default_pres,
                     np.arange(0, 36, 1), 'b3su', 'x3sg', bands)
    o2_lif = LIF(mol_o2, default_temp, default_pres, rot_qn_up, rot_qn_lo, vib_qn_up, vib_qn_lo_max,
                 'b3su', 'x3sg')

    palette_lif = plt.cycler('color', plt.cm.tab20c.colors).by_key()['color']
    colors_lif = [matplotlib.colors.to_hex(color) for color in palette_lif]
    colors_sim = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # plot.plot_line(o2_lif, colors_lif)
    # plot.plot_conv_all(o2_lif, colors_lif[0])
    # plot.plot_info(o2_lif)

    plot.plot_conv(o2_sim, colors_lif)
    # plot.plot_inst(o2_sim, colors_sim, 2)
    # plot.plot_samp('harvard/harvard20', 'red', 'plot')

    plot.plot_show()


if __name__ == '__main__':
    main()
