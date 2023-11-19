# module main
'''
Computes spectral lines for triplet oxygen. See the README for details on implementation along with
available features.
'''


# TODO: 11/19/23 add the ability to convolve already convolved data with an instrument function, it
#                should be implemented for both individual bands and an overall convolution

# TODO: 11/19/23 for major bands within a selected range, optionally add text on the final plot to
#                show various information (branch, index, etc.)

# TODO: 11/19/23 make that same information available to the user via an output .csv

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
    max_fc = max((band.get_fc() for band in vibrational_bands))

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

        line_data   = [band.return_line_data(max_fc) for band in vibrational_bands]
        line_colors = color_list[0:len(line_data)]
        line_labels = [str(band) + ' Band' for band in inp.VIB_BANDS]
        print(line_data)

        # TODO: 11/19/23 add back ability to output line data within a range

        # lines       = [band.get_lines() for band in band_list]
        # wavenumbers = [item[0] for item in line_data]

        # valid_data = [(wave, line.branch, line.ext_branch_idx, line.ext_rot_qn)
        #               for (line, wave) in zip(lines, wavenumbers)
        #               if (30910 <= wave.any() <= 30920) and (line.branch in ('p', 'r'))]

        # # Create a DataFrame
        # df = pd.DataFrame(valid_data, columns=['wavenumber', 'branch', 'triplet', 'rot_qn']).sort_values(by=['wavenumber'])

        # # Save the DataFrame to a CSV file
        # df.to_csv('../data/test.csv', index=False)

        out.plot_line(line_data, line_colors, line_labels)

    if inp.CONV_SEP:
        conv_data   = [band.return_conv_data(max_fc) for band in vibrational_bands]
        conv_colors = color_list[0:len(inp.VIB_BANDS)]
        conv_labels = ['Convolved ' + str(band) + ' Band' for band in inp.VIB_BANDS]

        out.plot_sep_conv(conv_data, conv_colors, conv_labels)

    # FIXME: 11/19/23 don't use this if possible, it's horribly inefficient and needs to be fixed
    if inp.CONV_ALL:
        line_data = [band.return_line_data(max_fc) for band in vibrational_bands]

        all_wavenumbers = []
        all_intensities = []

        for wavenumbers, intensities in line_data:
            all_wavenumbers.extend(wavenumbers)
            all_intensities.extend(intensities)

        lines = init.selection_rules(inp.ROT_LVLS)

        # NOTE: every vibrational band needs a full lines list, meaning that just using a single one
        #       doesn't work because it's half as long as it needs to be
        total_lines = np.array([])
        for _ in inp.VIB_BANDS:
            total_lines = np.append(total_lines, lines)

        all_conv = conv.convolved_data(np.array(all_wavenumbers), np.array(all_intensities),
                                       inp.TEMP, inp.PRES, total_lines)

        out.plot_all_conv(all_conv)

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
