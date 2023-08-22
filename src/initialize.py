# module initialize
'''
Initializes the spectral lines by applying selection rules.
'''

import numpy as np

from state import State
import constants as cn
import energy

def selection_rules(rot_qn_list: np.ndarray) -> np.ndarray:
    # Empty list to contain all valid spectral lines
    lines = []

    for grnd_rot_qn in rot_qn_list:
        for exct_rot_qn in rot_qn_list:
            # Remove every even R and P line since the nuclear spin of oxygen is zero
            if grnd_rot_qn % 2 == 1:

                # Selection rules for the R branch
                if exct_rot_qn - grnd_rot_qn == 1:
                    for grnd_branch_idx in range(1, 4):
                        for exct_branch_idx in range(1, 4):
                            if grnd_branch_idx == exct_branch_idx:
                                lines.append(SpectralLine(grnd_rot_qn, exct_rot_qn, 'r',
                                                          grnd_branch_idx, exct_branch_idx))
                            if grnd_branch_idx > exct_branch_idx:
                                lines.append(SpectralLine(grnd_rot_qn, exct_rot_qn, 'rq',
                                                          grnd_branch_idx, exct_branch_idx))

                # Selection rules for the P branch
                elif exct_rot_qn - grnd_rot_qn == -1:
                    for grnd_branch_idx in range(1, 4):
                        for exct_branch_idx in range(1, 4):
                            if grnd_branch_idx == exct_branch_idx:
                                lines.append(SpectralLine(grnd_rot_qn, exct_rot_qn, 'p',
                                                          grnd_branch_idx, exct_branch_idx))
                            if grnd_branch_idx < exct_branch_idx:
                                lines.append(SpectralLine(grnd_rot_qn, exct_rot_qn, 'pq',
                                                          grnd_branch_idx, exct_branch_idx))

    return np.array(lines)

class SpectralLine:
    def __init__(self, grnd_rot_qn: int, exct_rot_qn: int, branch: str, grnd_branch_idx: int,
                 exct_branch_idx: int) -> None:
        self.grnd_rot_qn     = grnd_rot_qn
        self.exct_rot_qn     = exct_rot_qn
        self.branch          = branch
        self.grnd_branch_idx = grnd_branch_idx
        self.exct_branch_idx = exct_branch_idx

    def wavenumber(self, band_origin: float, grnd_state: 'State', exct_state: 'State') -> float:
        return band_origin + \
               energy.rotational_term(self.exct_rot_qn, exct_state, self.exct_branch_idx) - \
               energy.rotational_term(self.grnd_rot_qn, grnd_state, self.grnd_branch_idx)

    def intensity(self, band_origin: float, grnd_state: 'State', exct_state: 'State', temp: float) -> float:
        part = (cn.BOLTZ * temp) / (cn.PLANC * cn.LIGHT * cn.X_BE)

        base = (self.wavenumber(band_origin, grnd_state, exct_state) / part) * \
               np.exp(- (energy.rotational_term(self.grnd_rot_qn, grnd_state, \
               self.grnd_branch_idx) * cn.PLANC * cn.LIGHT) / (cn.BOLTZ * temp))

        if self.branch == 'r':
            linestr = ((self.grnd_rot_qn + 1)**2 - 0.25) / (self.grnd_rot_qn + 1)
            intn =  base * linestr
        elif self.branch == 'p':
            linestr  = ((self.grnd_rot_qn)**2 - 0.25) / (self.grnd_rot_qn)
            intn =  base * linestr
        else:
            linestr = (2 * self.grnd_rot_qn + 1) / (4 * self.grnd_rot_qn * (self.exct_rot_qn + 1))
            intn = base * linestr

        # Naive approach of applying 1:2:1 line intensity ratio to each band, this way the two peaks
        # on either side of the main peak have 1/2 the intensity

        # NOTE: this *seems* to be what PGOPHER is doing from what I can tell, also haven't been
        #       able to find anything in Herzberg about it yet
        if self.grnd_branch_idx == 1 or self.grnd_branch_idx == 3:
            return intn / 2

        return intn
