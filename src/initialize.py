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
    '''
    Holds the necessary data for a single spectral line.
    '''

    ext_rot_qn:      int
    gnd_rot_qn:      int
    ext_triplet_idx: int
    gnd_triplet_idx: int
    branch:          str

    def predissociation(self) -> float:
        '''
        Gets the predissociation broadening coefficient in cm^-1 for each line.

        Returns:
            float: predissociation coefficient
        '''

        return inp.PD_DATA[f'f{self.gnd_triplet_idx}'] \
                          [inp.PD_DATA['rot_qn'] == self.ext_rot_qn].iloc[0]

    def wavenumber(self, band_origin: float, gnd_state: 'State', ext_state: 'State') -> float:
        '''
        Given the electronic, vibrational, and rotational term values, caluclates the wavenumnber
        (energy) of the resulting emission/absorption.

        Args:
            band_origin (float): electronic + vibrational term values
            gnd_state (State): ground state
            ext_state (State): excited state

        Returns:
            float: emitted/absorbed wavenumber
        '''

        return band_origin + \
               energy.rotational_term(self.ext_rot_qn, ext_state, self.ext_triplet_idx) - \
               energy.rotational_term(self.gnd_rot_qn, gnd_state, self.gnd_triplet_idx)

    def intensity(self, band_origin: float, gnd_state: 'State', ext_state: 'State',
                  temp: float) -> float:
        '''
        Uses the Gaussian distribution function to calculate the population density (and therefore
        intensity) of each spectral line.

        Args:
            band_origin (float): electronic + vibrational term values
            gnd_state (State): ground state
            ext_state (State): excited state
            temp (float): temperature

        Returns:
            float: intensity
        '''

        # TODO: 11/19/23 implement electronic, vibrational, rotational, etc. temperatures instead of
        #                just a single temperature. i.e. add separate partition functions for each

        # NOTE: 01/22/24 the partition function calculations are based on Herzberg pp. 126-127, but
        #                the equations in the book are for IR and Raman spectra, so I'm not sure if
        #                if they're applicable to what I'm doing

        # calculate Q_r, the rotational temperature-dependent partition function for the ground
        # state
        part = (cn.BOLTZ * temp) / (cn.PLANC * cn.LIGHT * cn.X_BE)

        # the basic intensity function if no branches are considered
        base = (self.wavenumber(band_origin, gnd_state, ext_state) / part) * \
               np.exp(- (energy.rotational_term(self.gnd_rot_qn, gnd_state, \
               self.gnd_triplet_idx) * cn.PLANC * cn.LIGHT) / (cn.BOLTZ * temp))

        # intensity is dependent upon branch, with satellite branches having a much lower intensity
        # (notice that r and p scale with N**2, while rq and rp scale with 1/N**2)
        match self.branch:
            case 'r':
                linestr = ((self.gnd_rot_qn + 1)**2 - 0.25) / (self.gnd_rot_qn + 1)
                intn = base * linestr
            case 'p':
                linestr = ((self.gnd_rot_qn)**2 - 0.25) / (self.gnd_rot_qn)
                intn = base * linestr
            case _:
                linestr = (2 * self.gnd_rot_qn + 1) / (4 * self.gnd_rot_qn * (self.gnd_rot_qn + 1))
                intn = base * linestr

        # naive approach of applying 1:2:1 line intensity ratio to each band, this way the two peaks
        # on either side of the main peak have 1/2 the intensity

        # NOTE: this *seems* to be what PGOPHER is doing from what I can tell, also haven't been
        #       able to find anything in Herzberg about it yet
        if self.gnd_triplet_idx in (1, 3):
            return intn / 2

        return intn

def selection_rules(rot_qn_list: np.ndarray) -> np.ndarray:
    '''
    Initializes spectral lines with ground and excited state rotational quantum numbers, along with
    their respective triplet index given the valid selection rules for triplet oxygen, i.e. ΔN = ±1.

    Args:
        rot_qn_list (np.ndarray): a list of rotational quantum numbers that are to be considered

    Returns:
        np.ndarray: array of SpectralLine objects
    '''

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
