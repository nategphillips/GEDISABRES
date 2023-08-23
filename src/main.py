# module main
'''
Computes spectral lines for triplet oxygen.
'''

import numpy as np

import convolution as conv
import initialize as init
import constants as cn
import output as out
import input as inp
import energy
import state

# TODO: make an object for a vibrational band so that franck-condon factors will be easier
# TODO: only process the sample data if the user specified which comparisons they want
# TODO: make separate plotting functions for lines, convolutions, and samples

def normalization(intensity_data: list) -> np.ndarray:
    max_val = np.array(intensity_data).max()

    return intensity_data / max_val

class LinePlot:
    def __init__(self, temp, pres, rot_qn_list, states):
        self.temp        = temp
        self.pres        = pres
        self.rot_qn_list = rot_qn_list
        self.states      = states

    def get_data(self):
        # Initialize ground and excited states
        grnd_state = state.State(cn.X_CONSTS, self.states[0])
        exct_state = state.State(cn.B_CONSTS, self.states[1])

        # Calculate the band origin energy
        band_origin = energy.get_band_origin(grnd_state, exct_state)

        # Initialize the list of valid spectral lines
        lines = init.selection_rules(self.rot_qn_list)

        # Get the wavenumbers and intensities
        wns = [line.wavenumber(band_origin, grnd_state, exct_state) for line in lines]
        ins = [line.intensity(band_origin, grnd_state, exct_state, self.temp) for line in lines]
        ins = normalization(ins)

        return wns, ins

    def get_conv(self):
        wns, ins = self.get_data()
        return conv.convolved_data(wns, ins, self.temp, self.pres)

def main():
    # Temperature used in Cosby is 300 K
    # Pressure used in Cosby is 20 Torr (2666.45 Pa)
    # pres = 2666.45
    # v_00 = 36185

    # Range of desired rotational quantum numbers
    rot_qn_list = np.arange(0, 37, 1)

    band_02 = LinePlot(300, 101325, rot_qn_list, [0, 2])
    band_02_wns, band_02_ins = band_02.get_data()
    cwns2, cins2 = band_02.get_conv()

    band_03 = LinePlot(300, 101325, rot_qn_list, [1, 4])
    band_03_wns, band_03_ins = band_03.get_data()
    cwns3, cins3 = band_03.get_conv()

    # Fetch sample data for plotting
    swns, sins = out.configure_samples()

    colors = ['black', 'red']
    labels = ['(0, 2)', '(1, 4)']

    out.plot_style()
    out.plot_lines([band_02_wns, band_03_wns], [band_02_ins, band_03_ins], colors, labels)
    out.plot_convolved([cwns2, cwns3], [cins2, cins3], labels)
    out.plot_samples([swns[0], swns[1]], [sins[0], sins[1]], colors, labels)
    out.show_plot()

if __name__ == '__main__':
    main()
