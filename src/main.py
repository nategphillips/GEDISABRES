# module main
'''
Computes spectral lines for triplet oxygen. See the README for details on implementation along with
available features.
'''

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import initialize as init
import convolution as conv
import output as out
import input as inp
import bands

def main():
    '''
    Runs the program.
    '''

    # Create a vibrational band line plot for each of the user-selected bands
    vibrational_bands = []
    for band in inp.VIB_BANDS:
        vibrational_bands.append(bands.VibrationalBand(inp.TEMP, inp.PRES, inp.ROT_LVLS,
                                                       band[0], band[1]))

    # Find the maximum Franck-Condon factor of all the bands, this is used to normalize the
    # intensities of each band with respect to the largest band
    max_fc = max((band.fc_data for band in vibrational_bands))

    # Set the plotting style
    out.plot_style()

    # Since each set of lines is plotted separately, matplotlib doesn't know to cycle colors after
    # each one is plotted

    # This generates a list of hex values that are fed to the plot generators
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Automatic coloring and labeling is done for both line and convolved plots
    if inp.LINE_DATA:
        # Wavenumber and intensity data for each line contained within a tuple for each vibrational
        # transition
        wns_line    = [band.wavenumbers_line() for band in vibrational_bands]
        ins_line    = [band.intensities_line(max_fc) for band in vibrational_bands]
        colors_line = color_list[0:len(inp.VIB_BANDS)]
        lables_line = [str(band) + ' Band' for band in inp.VIB_BANDS]

        out.plot_line(wns_line, ins_line, colors_line, lables_line)

        if inp.PRINT_INFO:
            # TODO: 11/19/23 still not happy with now this is being performed. converting the list
            #                comprehensions directly to numpy arrays is somewhat inefficient and I
            #                feel like there's a better way to do this

            # One thing to look at would be making wavelenth and intensity direct attributes of each
            # SpectralLine, but this looks more complicated than it's worth

            # flatten the data so we can directly access attributes for each line
            lines = np.array([band.lines for band in vibrational_bands]).ravel()
            wns   = np.array(wns_line).ravel()
            ins   = np.array(ins_line).ravel()

            df = pd.DataFrame({
                'wavenumber': wns,
                'intensity': ins,
                'branch': [line.branch for line in lines],
                'triplet': [line.ext_triplet_idx for line in lines],
                'rot_qn': [line.ext_rot_qn for line in lines],
                'predissociation': [line.predissociation() for line in lines]
            })

            df = df[(df['wavenumber'].between(inp.INFO_LIMS[0], inp.INFO_LIMS[1])) & 
                    (df['branch'].isin(['p', 'r']))].sort_values(by=['wavenumber'])

            df.to_csv('../data/test.csv', index=False)

            out.print_info(df)

    if inp.CONV_SEP:
        wns_conv    = [band.wavenumbers_conv() for band in vibrational_bands]
        ins_conv    = [band.intensities_conv(max_fc) for band in vibrational_bands]
        colors_conv = color_list[0:len(inp.VIB_BANDS)]
        labels_conv = ['Convolved ' + str(band) + ' Band' for band in inp.VIB_BANDS]

        if inp.INST_SEP:
            instr = [band.instrument_conv(max_fc, 5) for band in vibrational_bands]
            out.plot_sep_conv(wns_conv, instr, colors_conv, labels_conv)

        out.plot_sep_conv(wns_conv, ins_conv, colors_conv, labels_conv)

    # FIXME: 11/19/23 don't use this if possible, it's horribly inefficient and needs to be fixed.
    #                 broken with the refactor of bands.py
    if inp.CONV_ALL:
        line_wns = [band.wavenumbers_line() for band in vibrational_bands]
        line_ins = [band.intensities_line(max_fc) for band in vibrational_bands]

        all_wavenumbers = []
        all_intensities = []

        for wavenumbers, intensities in zip(line_wns, line_ins):
            all_wavenumbers.extend(wavenumbers)
            all_intensities.extend(intensities)

        lines = init.selection_rules(inp.ROT_LVLS)

        # NOTE: every vibrational band needs a full lines list, meaning that just using a single one
        #       doesn't work because it's half as long as it needs to be
        total_lines = np.array([])
        for _ in inp.VIB_BANDS:
            total_lines = np.append(total_lines, lines)

        wns_all_conv, ins_all_conv = conv.convolve(np.array(all_wavenumbers),
                                                   np.array(all_intensities), inp.TEMP,
                                                   inp.PRES, total_lines)

        # FIXME: 11/19/23 been working on this for 5 hours please for the love of code fix this at
        #                 some point

        print(1)
        if inp.INST_ALL:
            print(2)
            instr = conv.placeholder(wns_all_conv, ins_all_conv, 5)
            instr /= instr.max()
            print(3)
            out.plot_all_conv(wns_all_conv, instr)

        out.plot_all_conv(wns_all_conv, ins_all_conv)

    # Colors and labels for sample data are set in the input.py file
    if inp.SAMP_DATA:
        samp_data   = []
        for file in inp.SAMP_FILE:
            samp_data.append(out.configure_samples(file))
        out.plot_samp(samp_data, inp.SAMP_COLS, inp.SAMP_LABL)

    # Display all data on one plot
    out.show_plot()

if __name__ == '__main__':
    main()
