# module bands
'''
Holds the LinePlot class, which is different for each vibrational transition considered. Here, the
Franck-Condon factor is computed for each spectral plot and influences the final intensity data.
'''

import numpy as np

import convolution as conv
import initialize as init
import constants as cn
import input as inp
import energy

class LinePlot:
    '''
    Each LinePlot is a separate vibrational band.
    '''

    def __init__(self, temp: float, pres: float, rot_qn_list: np.ndarray, ext_vib_qn: int,
                 gnd_vib_qn: int) -> None:
        self.temp        = temp
        self.pres        = pres
        self.rot_qn_list = rot_qn_list
        self.ext_vib_qn  = ext_vib_qn
        self.gnd_vib_qn  = gnd_vib_qn

    def get_fc(self) -> float:
        '''
        From the global Franck-Condon data array, grabs the correct FC factor for the current
        vibrational transition.

        Returns:
            float: Franck-Condon factor for the current vibrational transition
        '''

        return inp.FC_DATA[self.ext_vib_qn][self.gnd_vib_qn]

    def get_line(self, max_fc: float) -> tuple[np.ndarray, np.ndarray]:
        '''
        Finds the wavenumbers and intensities for each line in the plot.

        Args:
            max_fc (float): maximum Franck-Condon factor from all vibrational transitions considered

        Returns:
            tuple[list, list]: (wavenumbers, intensities)
        '''

        # Initialize ground and excited states
        exct_state = energy.State(cn.B_CONSTS, self.ext_vib_qn)
        grnd_state = energy.State(cn.X_CONSTS, self.gnd_vib_qn)

        # Calculate the band origin energy
        if inp.BAND_ORIG[0]:
            band_origin = inp.BAND_ORIG[1]
        else:
            band_origin = energy.get_band_origin(grnd_state, exct_state)

        # Initialize the list of valid spectral lines
        lines = init.selection_rules(self.rot_qn_list)

        # Get the wavenumbers and intensities
        wns = np.array([line.wavenumber(band_origin, grnd_state, exct_state)
                        for line in lines])
        ins = np.array([line.intensity(band_origin, grnd_state, exct_state, self.temp)
                        for line in lines])

        # Normalize the intensity data w.r.t. the largest value
        # This is normalization of the plot with respect to itself
        ins /= ins.max()

        # Find the ratio between the largest Franck-Condon factor and the current plot
        norm_fc = self.get_fc() / max_fc

        # This is normalization of the plot with respect to others
        ins *= norm_fc

        return wns, ins

    def get_conv(self, max_fc: float) -> tuple[np.ndarray, np.ndarray]:
        '''
        Finds the wavenumbers and intensities for the convolved data.

        Args:
            max_fc (float): maximum Franck-Condon factor from all vibrational transitions considered

        Returns:
            tuple[np.ndarray, np.ndarray]: (wavenumbers, intensities)
        '''

        lines = init.selection_rules(self.rot_qn_list)

        wns, ins = self.get_line(max_fc)

        conv_wns, conv_ins = conv.convolved_data(wns, ins, self.temp, self.pres, lines)

        conv_ins *= self.get_fc() / max_fc

        return conv_wns, conv_ins
