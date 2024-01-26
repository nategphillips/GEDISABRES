# module initialize
'''
Initializes each spectral line by applying valid selection rules.
'''

from dataclasses import dataclass
import itertools

import numpy as np

from energy import State
import constants as cn
import input as inp
import energy

@dataclass
class SpectralLine:
    rot_qn_up:      int
    rot_qn_lo:      int
    triplet_idx_up: int
    triplet_idx_lo: int
    branch:         str

    def predissociation(self) -> float:
        return inp.PD_DATA[f'f{self.triplet_idx_lo}'] \
                          [inp.PD_DATA['rot_qn'] == self.rot_qn_up].iloc[0]

    def wavenumber(self, band_origin: float, state_lo: 'State', state_up: 'State') -> float:
        return band_origin + \
               energy.rotational_term(self.rot_qn_up, state_up, self.triplet_idx_up) - \
               energy.rotational_term(self.rot_qn_lo, state_lo, self.triplet_idx_lo)

    def intensity(self, band_origin: float, state_lo: 'State', state_up: 'State',
                  temp: float) -> float:
        # TODO: 11/19/23 implement electronic, vibrational, rotational, etc. temperatures instead of
        #                just a single temperature. i.e. add separate partition functions for each

        # NOTE: 01/22/24 the partition function calculations are based on Herzberg pp. 126-127, but
        #                the equations in the book are for IR and Raman spectra, so I'm not sure if
        #                if they're applicable to what I'm doing

        # calculate Q_r, the rotational temperature-dependent partition function for the ground
        # state
        part = (cn.BOLTZ * temp) / (cn.PLANC * cn.LIGHT * cn.CONSTS_LO['b_e'])

        # the basic intensity function if no branches are considered
        base = (self.wavenumber(band_origin, state_lo, state_up) / part) * \
               np.exp(- (energy.rotational_term(self.rot_qn_lo, state_lo, \
               self.triplet_idx_lo) * cn.PLANC * cn.LIGHT) / (cn.BOLTZ * temp))

        # intensity is dependent upon branch, with satellite branches having a much lower intensity
        # (notice that r and p scale with N**2, while rq and rp scale with 1/N**2)
        match self.branch:
            case 'r':
                linestr = ((self.rot_qn_lo + 1)**2 - 0.25) / (self.rot_qn_lo + 1)
                intn = base * linestr
            case 'p':
                linestr = ((self.rot_qn_lo)**2 - 0.25) / (self.rot_qn_lo)
                intn = base * linestr
            case _:
                linestr = (2 * self.rot_qn_lo + 1) / (4 * self.rot_qn_lo * (self.rot_qn_lo + 1))
                intn = base * linestr

        # naive approach of applying 1:2:1 line intensity ratio to each band, this way the two peaks
        # on either side of the main peak have 1/2 the intensity

        # NOTE: this *seems* to be what PGOPHER is doing from what I can tell, also haven't been
        #       able to find anything in Herzberg about it yet
        if self.triplet_idx_lo in (1, 3):
            return intn / 2

        return intn

def selection_rules(rot_qn_list: np.ndarray) -> np.ndarray:
    # empty list to contain all valid spectral lines
    lines = []

    for gnd_rot_qn, ext_rot_qn in itertools.product(rot_qn_list, repeat=2):
        d_rot_qn = ext_rot_qn - gnd_rot_qn

        # for molecular oxygen, all transitions with even values of J'' are forbidden
        if gnd_rot_qn % 2 == 1:

            # selection rules for the R branch
            if d_rot_qn == 1:
                for gnd_triplet_idx, ext_triplet_idx in itertools.product(range(1, 4), repeat=2):
                    if gnd_triplet_idx == ext_triplet_idx:
                        lines.append(SpectralLine(ext_rot_qn, gnd_rot_qn,
                                                  ext_triplet_idx, gnd_triplet_idx, 'r'))
                    if gnd_triplet_idx > ext_triplet_idx:
                        lines.append(SpectralLine(ext_rot_qn, gnd_rot_qn,
                                                  ext_triplet_idx, gnd_triplet_idx, 'rq'))

            # selection rules for the P branch
            elif d_rot_qn == -1:
                for gnd_triplet_idx, ext_triplet_idx in itertools.product(range(1, 4), repeat=2):
                    if gnd_triplet_idx == ext_triplet_idx:
                        lines.append(SpectralLine(ext_rot_qn, gnd_rot_qn,
                                                  ext_triplet_idx, gnd_triplet_idx, 'p'))
                    if gnd_triplet_idx < ext_triplet_idx:
                        lines.append(SpectralLine(ext_rot_qn, gnd_rot_qn,
                                                  ext_triplet_idx, gnd_triplet_idx, 'pq'))

    return np.array(lines)
