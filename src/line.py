# module line

from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

import numpy as np

from molecule import Molecule
from state import State
import constants as cn
import terms

if TYPE_CHECKING:
    from band import Band

@dataclass
class Line:
    rot_qn_up:     int
    rot_qn_lo:     int
    branch_idx_up: int
    branch_idx_lo: int
    branch_name:   str
    band:          Band
    molecule:      Molecule

    def predissociation(self) -> float:
        return (self.molecule.prediss[f'f{self.branch_idx_lo}']
                [self.molecule.prediss['rot_qn'] == self.rot_qn_up].iloc[0])

    def wavenumber(self, state_up: State, state_lo: State, vib_qn_up: int, vib_qn_lo: int,
                   band_origin: float) -> float:
        # calculates the wavenumber
        # Herzberg p. 168, eq. (IV, 24)

        return (band_origin +
                terms.rotational_term(state_up, vib_qn_up, self.rot_qn_up, self.branch_idx_up) -
                terms.rotational_term(state_lo, vib_qn_lo, self.rot_qn_lo, self.branch_idx_lo))

    def boltzmann_factor(self, state_lo: State, vib_qn_lo: int, temp: float) -> float:
        # calculates the Boltzmann factor
        # Herzberg p. 125, eq. (III, 164)

        return np.exp(-terms.rotational_term(state_lo, vib_qn_lo, self.rot_qn_lo,
                                             self.branch_idx_lo) *
                       cn.PLANC * cn.LIGHT / (cn.BOLTZ * temp))

    def honl_london_factor(self) -> float:
        # calculates the Hönl-London factors (line strengths)
        # Herzberg p. 250, eq. (V, 57)

        match self.branch_name:
            case 'r':
                line_strength = ((self.rot_qn_lo + 1)**2 - 0.25) / (self.rot_qn_lo + 1)
            case 'p':
                line_strength = (self.rot_qn_lo**2 - 0.25) / (self.rot_qn_lo)
            case 'pq' | 'rq':
                line_strength = (2 * self.rot_qn_lo + 1) / (4 * self.rot_qn_lo *
                                                            (self.rot_qn_lo + 1))

        return line_strength

    def intensity(self, state_up: State, state_lo: State, vib_qn_up: int, vib_qn_lo: int,
                  band_origin: float, temp: float) -> float:
        # calculates the intensity in absorption
        # Herzberg p. 126, eq. (III, 2)

        intensity = (self.wavenumber(state_up, state_lo, vib_qn_up, vib_qn_lo, band_origin) *
                     self.honl_london_factor() *
                     self.boltzmann_factor(state_lo, vib_qn_lo, temp) /
                     self.band.rotational_partition())

        if self.branch_idx_lo in (1, 3):
            return intensity / 2

        return intensity
