# module plot

import scienceplots  # pylint: disable = unused-import
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from simulation import Simulation
import convolve

# plt.style.use(['science', 'grid'])


def wavenum_to_wavelen(x):
    x             = np.array(x, float)
    near_zero     = np.isclose(x, 0)
    x[near_zero]  = np.inf
    x[~near_zero] = 1 / x[~near_zero]

    return x * 1e7


def plot_show():
    ax = plt.gca()

    secax = ax.secondary_xaxis('top', functions=(wavenum_to_wavelen, wavenum_to_wavelen))
    secax.set_xlabel('Wavenumber $\\nu$, [cm$^{-1}$]')

    plt.xlabel('Wavelength $\\lambda$, [nm]')
    plt.ylabel('Normalized Intensity')

    plt.legend()
    plt.show()


def plot_samp(samp_file: str, color: str, plot_as: str = 'stem') -> None:
    sample_data = pd.read_csv(f'../data/samples/{samp_file}.csv')

    wavenumbers = sample_data['wavenumbers'].to_numpy()
    wavelengths = wavenum_to_wavelen(wavenumbers)
    intensities = sample_data['intensities'].to_numpy()
    intensities /= intensities.max()

    match plot_as:
        case 'stem':
            plt.stem(wavelengths, intensities, color, markerfmt='', label=samp_file)
        case 'plot':
            plt.plot(wavelengths, intensities, color, label=samp_file)
        case _:
            raise ValueError(f'Invalid value for plot_as: {plot_as}.')


def plot_info(sim: Simulation) -> None:
    for vib_band in sim.vib_bands:
        wavenumbers_line = vib_band.wavenumbers_line()
        wavelengths_line = wavenum_to_wavelen(wavenumbers_line)
        intensities_line = vib_band.intensities_line()
        lines = vib_band.lines

        for idx, line in enumerate(lines):
            plt.text(wavelengths_line[idx], intensities_line[idx], f'{line.branch_name}')


def plot_line(sim: Simulation, colors: list) -> None:
    for idx, vib_band in enumerate(sim.vib_bands):
        wavelengths_line = wavenum_to_wavelen(vib_band.wavenumbers_line())

        plt.stem(wavelengths_line, vib_band.intensities_line(), colors[idx], markerfmt='',
                 label=f'{sim.molecule.name} {vib_band.vib_qn_up, vib_band.vib_qn_lo} line')


def plot_conv(sim: Simulation, colors: list) -> None:
    for idx, vib_band in enumerate(sim.vib_bands):
        wavelengths_conv = wavenum_to_wavelen(vib_band.wavenumbers_conv())

        plt.plot(wavelengths_conv, vib_band.intensities_conv(), colors[idx],
                 label=f'{sim.molecule.name} {vib_band.vib_qn_up, vib_band.vib_qn_lo} conv')


def plot_conv_all(sim: Simulation, color: str) -> None:
    wavenumbers_conv, intensities_conv = sim.all_convolved_data()
    wavelengths_conv = wavenum_to_wavelen(wavenumbers_conv)

    intensities_conv /= intensities_conv.max()

    plt.plot(wavelengths_conv, intensities_conv, color, label=f'{sim.molecule.name} conv all')


def plot_inst(sim: Simulation, colors: list, broadening: float) -> None:
    for idx, vib_band in enumerate(sim.vib_bands):
        wavelengths_conv = wavenum_to_wavelen(vib_band.wavenumbers_conv())

        plt.plot(wavelengths_conv, vib_band.intensities_inst(broadening), colors[idx],
                 label=f'{sim.molecule.name} {vib_band.vib_qn_up, vib_band.vib_qn_lo} inst')


def plot_inst_all(sim: Simulation, color: str, broadening: float) -> None:
    wavenumbers_conv, intensities_conv = sim.all_convolved_data()
    wavelengths_conv = wavenum_to_wavelen(wavenumbers_conv)

    intensities_inst = convolve.convolve_inst(wavenumbers_conv, intensities_conv, broadening)
    intensities_inst /= intensities_inst.max()

    plt.plot(wavelengths_conv, intensities_inst, color, label=f'{sim.molecule.name} inst all')
