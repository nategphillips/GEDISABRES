# module output
'''
Handles the plotting of line, convolved, and sample data. Also reads and configures sample data.
'''

import matplotlib.pyplot as plt
import scienceplots # pylint: disable=unused-import
import pandas as pd
import numpy as np

import input as inp

def plot_style():
    # plt.style.use(['science', 'grid'])
    # plt.figure(figsize=(inp.SCREEN_RES[0]/inp.DPI, inp.SCREEN_RES[1]/inp.DPI), dpi=inp.DPI)

    # if inp.FONT_SIZE[0]:
        # plt.rcParams.update({'font.size': inp.FONT_SIZE[1]})

    pass

def show_plot():
    if inp.SET_LIMS[0]:
        plt.xlim(inp.SET_LIMS[1][0], inp.SET_LIMS[1][1])

    # plt.title(f'{inp.PRES} Pa, {inp.TEMP} K')
    # plt.xlabel('Wavenumber $\\nu$, [cm$^{-1}$]')
    # plt.ylabel('Normalized Intensity')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # FIXME: 11/19/23 this conversion results in a divide by zero error obviously. there must be a
    #        way to constrain the minimum and maximum wavenumebers considered since we're never
    #        dealing with wavenumbers of 0

    # convert from wavenumber to wavelength
    def wn2wl(wns):
        return (1 / wns) * 1e7

    ax = plt.gca()

    # add a secondary axis for wavelength
    secax = ax.secondary_xaxis('top', functions=(wn2wl, wn2wl))
    secax.set_xlabel('Wavelength $\\lambda$, [nm]')

    if inp.PLOT_SAVE:
        plt.savefig(inp.PLOT_PATH, dpi=inp.DPI * 2)
    else:
        plt.show()

def configure_samples(samp_file: str) -> tuple[np.ndarray, np.ndarray]:
    sample_data = pd.read_csv(f'../data/{samp_file}.csv', delimiter=' ')

    if samp_file == 'cosby09':
        sample_data['wavenumbers'] = sample_data['wavenumbers'].add(36185)

    wns = sample_data['wavenumbers'].to_numpy()
    ins = sample_data['intensities'].to_numpy()
    ins /= max(ins)

    return wns, ins

def plot_line(wavenumbers, intensities, colors: list[str], labels: list[str]) -> None:
    for idx, (wave, intn) in enumerate(zip(wavenumbers, intensities)):
        plt.stem(wave, intn, colors[idx], markerfmt='', label=f'{labels[idx]}')

def print_info(df):
    for _, row in df.iterrows():
        plt.text(row['wavenumber'], row['intensity'], f'({row['branch']},{row['triplet']})', ha='center')

def plot_sep_conv(wavenumbers, intensites, colors: list[str], labels: list[str]) -> None:
    for idx, (wave, intn) in enumerate(zip(wavenumbers, intensites)):
        plt.plot(wave, intn, colors[idx], label=f'{labels[idx]}')

def plot_all_conv(wavenumbers, intensities) -> None:
    plt.plot(wavenumbers, intensities, 'r', label=f'Convolved Data: {inp.VIB_BANDS}')

def plot_samp(data: list[tuple], colors: list[str], labels: list[str]) -> None:
    for i, (wave, intn) in enumerate(data):
        plt.plot(wave, intn, colors[i], label=f'{labels[i]}')
