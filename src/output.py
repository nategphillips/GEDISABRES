# module output
'''
Handles plotting.
'''

import matplotlib.pyplot as plt
import scienceplots # pylint: disable=unused-import
import pandas as pd

import input as inp

def sample_plotter() -> tuple:
    sample_data = []
    sample_data.append(pd.read_csv('../data/harvrd.csv', delimiter=' '))
    sample_data.append(pd.read_csv('../data/hitran.csv', delimiter=' '))
    sample_data.append(pd.read_csv('../data/pgopher.csv', delimiter=' '))
    sample_data.append(pd.read_csv('../data/webplot_09_band.csv', delimiter=' '))

    # TODO: this is cosby data stuff, need to fix
    df = sample_data[3]
    # add band origin
    df['wavenumber'] = df['wavenumber'].add(36185)

    samp_wn = []
    samp_in = []
    for _, val in enumerate(sample_data):
        samp_wn.append(val['wavenumber'])
        samp_in.append(val['intensity'])

    for i, val in enumerate(samp_in):
        samp_in[i] = val / val.max()

    return samp_wn, samp_in

def plotter(line_data: list, convolved_data: tuple, sample_data: tuple):
    mydpi = 96

    plt.figure(figsize=(1920/mydpi, 1080/mydpi), dpi=mydpi)

    plt.style.use(['science', 'grid'])

    if inp.LINE_DATA:
        plt.stem(line_data[0][0], line_data[1][0], 'black', markerfmt='', label='R Branch')
        plt.stem(line_data[0][1], line_data[1][1], 'red', markerfmt='', label='P Branch')
        plt.stem(line_data[0][2], line_data[1][2], 'gray', markerfmt='', label='QR Branch')
        plt.stem(line_data[0][3], line_data[1][3], 'orange', markerfmt='', label='QP Branch')

    if inp.CONVOLVED_DATA:
        plt.plot(convolved_data[0], convolved_data[1], label='Convolved Data')

    if 'harvard' in inp.COMPARED_DATA:
        plt.plot(sample_data[0][0], sample_data[1][0], label='Harvard Data')
    if 'hitran' in inp.COMPARED_DATA:
        plt.stem(sample_data[0][1], sample_data[1][1], 'yellow', markerfmt='', label='HITRAN Data')
    if 'pgopher' in inp.COMPARED_DATA:
        plt.stem(sample_data[0][2], sample_data[1][2], 'blue', markerfmt='', label='PGOPHER Data')
    if 'cosby' in inp.COMPARED_DATA:
        plt.plot(sample_data[0][3], sample_data[1][3], 'green', label='Cosby 1993')
        plt.xlim([min(sample_data[0][3]), max(sample_data[0][3])])

    plt.xlabel('Wavenumber $\\nu$, [cm$^{-1}$]')
    plt.ylabel('Normalized Intensity')
    plt.legend()

    if inp.PLOT_SAVE:
        plt.savefig(inp.PLOT_PATH, dpi=mydpi * 2)
    else:
        plt.show()
