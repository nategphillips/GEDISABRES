# module plot

import scienceplots # pylint: disable = unused-import
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from simulation import Simulation
import convolve

# plt.style.use(['science', 'grid'])

def plot_samp(samp_file: str, color: str, plot_as: str = 'stem') -> None:
    sample_data = pd.read_csv(f'../data/samples/{samp_file}.csv')

    wavenumbers = sample_data['wavenumbers'].to_numpy()
    intensities = sample_data['intensities'].to_numpy()
    intensities /= np.max(intensities)

    if plot_as == 'stem':
        plt.stem(wavenumbers, intensities, color, markerfmt='', label=samp_file)
    else:
        plt.plot(wavenumbers, intensities, color, label=samp_file)

def plot_info(sim: Simulation) -> None:
    for vib_band in sim.vib_bands:
        wavenumbers_line = vib_band.wavenumbers_line()
        intensities_line = vib_band.intensities_line()
        lines            = vib_band.lines

        for idx, line in enumerate(lines):
            plt.text(wavenumbers_line[idx], intensities_line[idx], f'{line.branch}')

def plot_line(sim: Simulation, color: str) -> None:
    for vib_band in sim.vib_bands:
        plt.stem(vib_band.wavenumbers_line(), vib_band.intensities_line(), color,
                 markerfmt='', label=f'{sim.molecule.name} {vib_band.name} line')

def plot_conv(sim: Simulation, color: str) -> None:
    for vib_band in sim.vib_bands:
        plt.plot(vib_band.wavenumbers_conv(), vib_band.intensities_conv(), color,
                 label=f'{sim.molecule.name} {vib_band.name} conv')

def plot_conv_all(sim: Simulation, color: str) -> None:
    wavenumbers_conv, intensities_conv = sim.all_convolved_data()

    intensities_conv /= np.max(intensities_conv)

    plt.plot(wavenumbers_conv, intensities_conv, color, label=f'{sim.molecule.name} conv all')

def plot_inst(sim: Simulation, color: str, broadening: float) -> None:
    for vib_band in sim.vib_bands:
        plt.plot(vib_band.wavenumbers_conv(), vib_band.intensities_inst(broadening), color,
                 label=f'{sim.molecule.name} {vib_band.name} inst')

def plot_inst_all(sim: Simulation, color: str, broadening: float) -> None:
    wavenumbers_conv, intensities_conv = sim.all_convolved_data()
    intensities_inst = convolve.convolve_inst(wavenumbers_conv, intensities_conv, broadening)

    intensities_inst /= np.max(intensities_inst)

    plt.plot(wavenumbers_conv, intensities_inst, color, label=f'{sim.molecule.name} inst all')
