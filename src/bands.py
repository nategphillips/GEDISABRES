# module bands
'''
Holds the LinePlot class, which is different for each vibrational transition considered. Here, the
Franck-Condon factor is computed for each spectral plot and influences the final intensity data.
'''

from dataclasses import dataclass

import numpy as np

import convolution as conv
import initialize as init
import constants as cn
import input as inp
import energy

@dataclass
class VibrationalBand:
    '''
    Each is a separate vibrational band.
    '''

    temp:        float
    pres:        float
    rot_qn_list: np.ndarray
    ext_vib_qn:  int 
    gnd_vib_qn:  int

    def get_lines(self) -> np.ndarray:
        '''
        Returns the spectral lines that fall within the transition.

        Returns:
            np.ndarray: valid lines
        '''

        return init.selection_rules(self.rot_qn_list)

    def get_fc(self) -> float:
        '''
        From the global Franck-Condon data array, grabs the correct FC factor for the current
        vibrational transition.

        Returns:
            float: Franck-Condon factor for the current vibrational transition
        '''

        return inp.FC_DATA[self.ext_vib_qn][self.gnd_vib_qn]

    def return_line_data(self, max_fc: float) -> tuple[np.ndarray, np.ndarray]:
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
        lines = self.get_lines()

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

    def return_conv_data(self, max_fc: float) -> tuple[np.ndarray, np.ndarray]:
        '''
        Finds the wavenumbers and intensities for the convolved data.

        Args:
            max_fc (float): maximum Franck-Condon factor from all vibrational transitions considered

        Returns:
            tuple[np.ndarray, np.ndarray]: (wavenumbers, intensities)
        '''

        # FIXME: calling get_lines() a second time here, even though it was already called within
        #        return_line_data(), which is also called here - need to optimize this
        lines = self.get_lines()

        wns, ins = self.return_line_data(max_fc)

        conv_wns, conv_ins = conv.convolved_data(wns, ins, self.temp, self.pres, lines)

        conv_ins *= self.get_fc() / max_fc

        return conv_wns, conv_ins
