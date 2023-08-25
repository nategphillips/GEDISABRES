# module output
'''
Handles the plotting of line, convolved, and sample data. Also reads and configures sample data.
'''

import matplotlib.pyplot as plt
import scienceplots # pylint: disable=unused-import
import pandas as pd
import numpy as np

import input as inp

def plot_style() -> None:
    '''
    Sets a consistent plot style and output resolution based on the input screen parameters.
    '''

    plt.style.use(['science', 'grid'])
    plt.figure(figsize=(inp.SCREEN_RES[0]/inp.DPI, inp.SCREEN_RES[1]/inp.DPI), dpi=inp.DPI)

def show_plot():
    '''
    Sets global plot labels and saves the figure if necessary.
    '''

    if inp.SET_LIMS[0]:
        plt.xlim(inp.SET_LIMS[1][0], inp.SET_LIMS[1][1])

    plt.xlabel('Wavenumber $\\nu$, [cm$^{-1}$]')
    plt.ylabel('Normalized Intensity')
    plt.legend()

    if inp.PLOT_SAVE:
        plt.savefig(inp.PLOT_PATH, dpi=inp.DPI * 2)
    else:
        plt.show()

def configure_samples(samp_file: str) -> tuple[np.ndarray, np.ndarray]:
    '''
    Reads the selected sample file into a pandas dataframe, returns the wavenumbers and intensities
    of the data.

    Args:
        samp_file (str): name of the sample file

    Returns:
        tuple[np.ndarray, np.ndarray]: wavenumber and intensity data
    '''

    sample_data = pd.read_csv(f'../data/{samp_file}.csv', delimiter=' ')

    if samp_file == 'cosby09':
        sample_data['wavenumbers'] = sample_data['wavenumbers'].add(36185)

    wns = sample_data['wavenumbers'].to_numpy()
    ins = sample_data['intensities'].to_numpy()
    ins /= max(ins)

    return wns, ins

def plot_line(data: list[tuple], colors: list[str], labels: list[str]) -> None:
    '''
    Plots wavenumber vs. intensity for line data.

    Args:
        data (list[tuple]): (wavenumbers, intensities)
        colors (list[str]): desired colors
        labels (list[str]): band labels
    '''

    for i, (wave, intn) in enumerate(data):
        plt.stem(wave, intn, colors[i], markerfmt='', label=f'{labels[i]}')

def plot_conv(data: list[tuple], colors: list[str], labels: list[str]) -> None:
    '''
    Plots wavenumber vs. intensity for convolved data.

    Args:
        data (list[tuple]): (wavenumbers, intensities)
        colors (list[str]): desired colors
        labels (list[str]): band labels
    '''

    for i, (wave, intn) in enumerate(data):
        plt.plot(wave, intn, colors[i], label=f'{labels[i]}')

def plot_samp(data: list[tuple], colors: list[str], labels: list[str]) -> None:
    '''
    Plots wavenumber vs. intensity for sample data.

    Args:
        data (list[tuple]): (wavenumbers, intensities)
        colors (list[str]): desired colors
        labels (list[str]): band labels
    '''

    for i, (wave, intn) in enumerate(data):
        plt.plot(wave, intn, colors[i], label=f'{labels[i]}')
