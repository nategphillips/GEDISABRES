# module bands
'''
Holds the LinePlot class, which is different for each vibrational transition considered. Here, the
Franck-Condon factor is computed for each spectral plot and influences the final intensity data.
'''

from dataclasses import dataclass

import numpy as np

import convolution as conv
import initialize as init
import input as inp
import energy

@dataclass
class VibrationalBand:
    temp:        float
    pres:        float
    rot_qn_list: np.ndarray
    vib_qn_up:   int
    vib_qn_lo:   int
    consts_lo:   dict
    consts_up:   dict

    def __post_init__(self):
        self.lines    = init.selection_rules(self.rot_qn_list)
        self.fc_data  = inp.FC_DATA[self.vib_qn_up][self.vib_qn_lo]
        self.state_up = energy.State(self.consts_up, self.vib_qn_up)
        self.state_lo = energy.State(self.consts_lo, self.vib_qn_lo)

        if inp.BAND_ORIG[0]:
            self.band_origin = inp.BAND_ORIG[1]
        else:
            self.band_origin = energy.get_band_origin(self.state_lo, self.state_up)

    def wavenumbers_line(self):
        return np.array([line.wavenumber(self.band_origin, self.state_lo, self.state_up)
                        for line in self.lines])

    def wavenumbers_conv(self):
        wns = self.wavenumbers_line()

        return np.linspace(wns.min(), wns.max(), inp.CONV_GRAN)

    def intensities_line(self, max_fc):
        ins = np.array([line.intensity(self.band_origin, self.state_lo,
                                       self.state_up, self.temp) for line in self.lines])

        # normalize the plot with respect to itself
        ins /= ins.max()

        # find the ratio between the largest Franck-Condon factor and the current plot
        norm_fc = self.fc_data / max_fc

        # normalization of the plot with respect to others
        ins *= norm_fc

        return ins

    # FIXME: 11/20/23 move convolution back to a separate module, the function needs to be
    #                 consolidated, separate implementation of instrument vs. normal broadening is
    #                 not ideal
    def intensities_conv(self, max_fc):
        wns_line = self.wavenumbers_line()
        wns_conv = self.wavenumbers_conv()

        ins_line = self.intensities_line(max_fc)
        ins_conv = np.zeros_like(wns_conv)

        for idx, (wavenumber_peak, intensity_peak) in enumerate(zip(wns_line, ins_line)):
            ins_conv += intensity_peak * conv.broadening_fn(wns_conv, wavenumber_peak, self.temp,
                                                       self.pres, idx, self.lines)

        ins_conv /= ins_conv.max()

        ins_conv *= self.fc_data / max_fc

        return ins_conv

    def instrument_conv(self, max_fc, broadening):
        # start with the already convolved data and do the instrument function on top of that
        wns_conv = self.wavenumbers_conv()

        ins_conv = self.intensities_conv(max_fc)
        ins_inst = np.zeros_like(wns_conv)

        for wavenumber_peak, intensity_peak in zip(wns_conv, ins_conv):
            ins_inst += intensity_peak * conv.instrument_fn(wns_conv, wavenumber_peak, broadening)

        ins_inst /= ins_inst.max()

        ins_inst *= self.fc_data / max_fc

        return ins_inst
