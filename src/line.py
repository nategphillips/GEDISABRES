# module line

from dataclasses import dataclass

import numpy as np

from molecule import Molecule
from state import State
import constants as cn
import terms


@dataclass
class Line:
    rot_qn_up:     int
    rot_qn_lo:     int
    branch_idx_up: int
    branch_idx_lo: int
    branch:        str
    molecule:      Molecule

    def predissociation(self) -> float:
        if self.molecule.name == 'o2':
            return (self.molecule.prediss[f'f{self.branch_idx_lo}']
                    [self.molecule.prediss['rot_qn'] == self.rot_qn_up].iloc[0])

        return 0

    def wavenumber(self, band_origin: float, vib_qn_up: int, vib_qn_lo: int, state_up: State,
                   state_lo: State) -> float:
        return (band_origin +
                terms.rotational_term(state_up, vib_qn_up, self.rot_qn_up, self.branch_idx_up) -
                terms.rotational_term(state_lo, vib_qn_lo, self.rot_qn_lo, self.branch_idx_lo))

    def intensity(self, band_origin: float, vib_qn_up: int, vib_qn_lo: int, state_up: State,
                  state_lo: State, temp: float) -> float:
        # NOTE: 05/02/24 this comes from an approximation of q_r from Herzberg pp. 125 - needs to be
        #                calculated more accurately in the future
        part = cn.BOLTZ * temp / (cn.PLANC * cn.LIGHT * state_lo.consts['b_e'])

        base = (self.wavenumber(band_origin, vib_qn_up, vib_qn_lo, state_up, state_lo) / part *
                np.exp(-terms.rotational_term(state_lo, vib_qn_lo, self.rot_qn_lo,
                                              self.branch_idx_lo) *
                       cn.PLANC * cn.LIGHT / (cn.BOLTZ * temp)))

        if state_up.name == 'b3su':
            match self.branch:
                case 'r':
                    linestr = ((self.rot_qn_lo + 1)**2 - 0.25) / (self.rot_qn_lo + 1)
                    intn = base * linestr
                case 'p':
                    linestr = ((self.rot_qn_lo)**2 - 0.25) / (self.rot_qn_lo)
                    intn = base * linestr
                case 'pq' | 'rq':
                    linestr = (2 * self.rot_qn_lo + 1) / (4 * self.rot_qn_lo * (self.rot_qn_lo + 1))
                    intn = base * linestr

            if self.branch_idx_lo in (1, 3):
                return intn / 2

        else:
            lambda_lo = 1

            match self.branch:
                case 'r':
                    linestr = (((self.rot_qn_lo + 1 + lambda_lo) *
                                (self.rot_qn_lo + 1 - lambda_lo)) / (self.rot_qn_lo + 1))
                    intn = base * linestr
                case 'p':
                    linestr = (((self.rot_qn_lo + lambda_lo) *
                                (self.rot_qn_lo - lambda_lo)) / self.rot_qn_lo)
                    intn = base * linestr
                case 'q':
                    linestr = (((2 * self.rot_qn_lo + 1) * lambda_lo**2) /
                               (self.rot_qn_lo * (self.rot_qn_lo + 1)))
                    intn = base * linestr

        return intn
