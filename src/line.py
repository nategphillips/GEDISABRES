# module line

from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

import numpy as np

import terms
import constants as cn
from simtype import SimType
from molecule import Molecule

if TYPE_CHECKING:
    from band import Band
    from simulation import Simulation

@dataclass
class Line:
    rot_qn_up:     int
    rot_qn_lo:     int
    branch_idx_up: int
    branch_idx_lo: int
    branch_name:   str
    sim:           Simulation
    band:          Band
    molecule:      Molecule

    def predissociation(self) -> float:
        return (self.molecule.prediss[f'f{self.branch_idx_lo}']
                [self.molecule.prediss['rot_qn'] == self.rot_qn_up].iloc[0])

    def wavenumber(self) -> float:
        # calculates the wavenumber
        # Herzberg p. 168, eq. (IV, 24)

        return (self.band.band_origin +
                terms.rotational_term(self.sim.state_up, self.band.vib_qn_up, self.rot_qn_up,
                                      self.branch_idx_up) -
                terms.rotational_term(self.sim.state_lo, self.band.vib_qn_lo, self.rot_qn_lo,
                                      self.branch_idx_lo))

    def rot_boltzmann_factor(self) -> float:
        # calculates the rotational Boltzmann factor
        # Herzberg p. 125, eq. (III, 164)

        match self.sim.sim_type:
            case SimType.ABSORPTION:
                state      = self.sim.state_lo
                vib_qn     = self.band.vib_qn_lo
                rot_qn     = self.rot_qn_lo
                branch_idx = self.branch_idx_lo
            case SimType.EMISSION | SimType.LIF:
                state      = self.sim.state_up
                vib_qn     = self.band.vib_qn_up
                rot_qn     = self.rot_qn_up
                branch_idx = self.branch_idx_up
            case _:
                raise ValueError('Invalid SimType.')

        return np.exp(-terms.rotational_term(state, vib_qn, rot_qn, branch_idx) *
                       cn.PLANC * cn.LIGHT / (cn.BOLTZ * self.sim.temp))

    def honl_london_factor(self) -> float:
        # calculates the Hönl-London factors (line strengths)
        # Herzberg p. 250, eq. (V, 57)

        # FIXME: 05/07/24 - the Boltzmann factor changes based on emission or absorption, which
        #        presumably means these need to change as well

        match self.branch_name:
            case 'r':
                line_strength = ((self.rot_qn_lo + 1)**2 - 0.25) / (self.rot_qn_lo + 1)
            case 'p':
                line_strength = (self.rot_qn_lo**2 - 0.25) / (self.rot_qn_lo)
            case 'pq' | 'rq':
                line_strength = (2 * self.rot_qn_lo + 1) / (4 * self.rot_qn_lo *
                                                            (self.rot_qn_lo + 1))

        return line_strength

    def intensity(self) -> float:
        # calculates the intensity
        # Herzberg p. 126, eqs. (III, 169-170)

        match self.sim.sim_type:
            case SimType.ABSORPTION:
                wavenumber_factor = self.wavenumber()
            case SimType.EMISSION | SimType.LIF:
                wavenumber_factor = self.wavenumber()**4
            case _:
                raise ValueError('Invalid SimType.')

        # Honl-London contribution
        intensity = wavenumber_factor * self.honl_london_factor()
        # rotational contribution
        intensity *= self.rot_boltzmann_factor() / self.band.rot_part

        if self.branch_idx_lo in (1, 3):
            return intensity / 2

        return intensity
