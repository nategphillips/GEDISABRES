# module output
'''
Handles plotting.
'''

import matplotlib.pyplot as plt
import scienceplots # pylint: disable=unused-import
import pandas as pd

import input as inp

def plot_style():
    plt.figure(figsize=(inp.SCREEN_RES[0]/inp.DPI, inp.SCREEN_RES[1]/inp.DPI), dpi=inp.DPI)
    plt.style.use(['science', 'grid'])

def show_plot():
    plt.xlabel('Wavenumber $\\nu$, [cm$^{-1}$]')
    plt.ylabel('Normalized Intensity')
    plt.legend()

    if inp.PLOT_SAVE:
        plt.savefig(inp.PLOT_PATH, dpi=inp.DPI * 2)
    else:
        plt.show()

def configure_samples():
    wns = []
    ins = []

    for sample in inp.COMPARED_DATA:
        df = pd.read_csv(f'../data/{sample}.csv', delimiter=' ')

        if sample == 'cosby09':
            df['wavenumber'] = df['wavenumber'].add(36185)

        wns.append(df['wavenumber'].to_numpy())
        temp = df['intensity'].to_numpy()
        ins.append(temp / max(temp))

    return wns, ins

def plot_lines(wavenumbers, intensities, colors, labels):
    for i, (wave, intn) in enumerate(zip(wavenumbers, intensities)):
        plt.stem(wave, intn, f'{colors[i]}', markerfmt='', label=f'{labels[i]}')

def plot_convolved(wavenumbers, intensities, labels):
    for i, (wave, intn) in enumerate(zip(wavenumbers, intensities)):
        plt.plot(wave, intn, label=f'{labels[i]}')

def plot_samples(wavenumbers, intensities, colors, labels):
    for i, (wave, intn) in enumerate(zip(wavenumbers, intensities)):
        plt.plot(wave, intn, f'{colors[i]}', label=f'{labels[i]}')
