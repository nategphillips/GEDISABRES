# module plot

import scienceplots # pylint: disable = unused-import
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from simulation import Simulation
import convolve

# plt.style.use(['science', 'grid'])

def plot_show():
    ax = plt.gca()

    def inverse(x):
        x = np.array(x, float)
        near_zero = np.isclose(x, 0)
        x[near_zero] = np.inf
        x[~near_zero] = 1 / x[~near_zero]
        return x * 1e7

    secax = ax.secondary_xaxis('top', functions=(inverse, inverse))
    secax.set_xlabel('Wavelength $\\lambda$, [nm]')

    plt.xlabel('Wavenumber $\\nu$, [cm$^{-1}$]')
    plt.ylabel('Normalized Intensity')

    plt.legend()
    plt.show()

def plot_samp(samp_file: str, color: str, plot_as: str = 'stem') -> None:
    sample_data = pd.read_csv(f'../data/samples/{samp_file}.csv')

    wavenumbers = sample_data['wavenumbers'].to_numpy()
    intensities = sample_data['intensities'].to_numpy()
    intensities /= np.max(intensities)

    match plot_as:
        case 'stem':
            plt.stem(wavenumbers, intensities, color, markerfmt='', label=samp_file)
        case 'plot':
            plt.plot(wavenumbers, intensities, color, label=samp_file)
        case _:
            raise ValueError(f'Invalid value for plot_as: {plot_as}.')

def plot_info(sim: Simulation) -> None:
    for vib_band in sim.vib_bands:
        wavenumbers_line = vib_band.wavenumbers_line()
        intensities_line = vib_band.intensities_line()
        lines            = vib_band.lines

        for idx, line in enumerate(lines):
            plt.text(wavenumbers_line[idx], intensities_line[idx], f'{line.branch}')

def plot_line(sim: Simulation, colors: list) -> None:
    for idx, vib_band in enumerate(sim.vib_bands):
        plt.stem(vib_band.wavenumbers_line(), vib_band.intensities_line(), colors[idx],
                 markerfmt='', label=f'{sim.molecule.name} {vib_band.name} line')

def plot_conv(sim: Simulation, colors: list) -> None:
    for idx, vib_band in enumerate(sim.vib_bands):
        plt.plot(vib_band.wavenumbers_conv(), vib_band.intensities_conv(), colors[idx],
                 label=f'{sim.molecule.name} {vib_band.name} conv')

def plot_conv_all(sim: Simulation, color: str) -> None:
    wavenumbers_conv, intensities_conv = sim.all_convolved_data()

    intensities_conv /= np.max(intensities_conv)

    plt.plot(wavenumbers_conv, intensities_conv, color, label=f'{sim.molecule.name} conv all')

def plot_inst(sim: Simulation, colors: list, broadening: float) -> None:
    for idx, vib_band in enumerate(sim.vib_bands):
        plt.plot(vib_band.wavenumbers_conv(), vib_band.intensities_inst(broadening), colors[idx],
                 label=f'{sim.molecule.name} {vib_band.name} inst')

def plot_inst_all(sim: Simulation, color: str, broadening: float) -> None:
    wavenumbers_conv, intensities_conv = sim.all_convolved_data()
    intensities_inst = convolve.convolve_inst(wavenumbers_conv, intensities_conv, broadening)

    intensities_inst /= np.max(intensities_inst)

    plt.plot(wavenumbers_conv, intensities_inst, color, label=f'{sim.molecule.name} inst all')
