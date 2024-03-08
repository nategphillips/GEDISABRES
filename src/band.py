# module band

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from molecule import Molecule
from state import State
import input as inp
import convolve
import terms

if TYPE_CHECKING:
    from simulation import Simulation

class Band:
    def __init__(self, name: tuple[int, int], lines: np.ndarray, state_up: State, state_lo: State,
                 temp: float, sim: Simulation, molecule: Molecule) -> None:
        self.name:          tuple      = name
        self.vib_qn_up:     int        = name[0]
        self.vib_qn_lo:     int        = name[1]
        self.lines:         np.ndarray = lines
        self.state_up:      State      = state_up
        self.state_lo:      State      = state_lo
        self.temp:          float      = temp
        self.sim:           Simulation = sim
        self.molecule:      Molecule   = molecule
        self.band_origin:   float      = self.get_band_origin()
        self.franck_condon: float      = self.molecule.fc_data[self.vib_qn_up][self.vib_qn_lo]

    def get_band_origin(self) -> float:
        elc_energy = self.state_up.consts['t_e'] - self.state_lo.consts['t_e']

        vib_energy = terms.vibrational_term(self.state_up, self.vib_qn_up) - \
                     terms.vibrational_term(self.state_lo, self.vib_qn_lo)

        return elc_energy + vib_energy

    def wavenumbers_line(self) -> np.ndarray:
        return np.array([line.wavenumber(self.band_origin, self.vib_qn_up, self.vib_qn_lo,
                                         self.state_up, self.state_lo) for line in self.lines])

    def intensities_line(self) -> np.ndarray:
        intensities_line = np.array([line.intensity(self.band_origin, self.vib_qn_up,
                                                    self.vib_qn_lo, self.state_up, self.state_lo,
                                                    self.temp)
                                    for line in self.lines])

        intensities_line /= intensities_line.max()
        intensities_line *= self.franck_condon / self.sim.max_fc

        return intensities_line

    def wavenumbers_conv(self) -> np.ndarray:
        wavenumbers_line = self.wavenumbers_line()

        return np.linspace(wavenumbers_line.min(), wavenumbers_line.max(), inp.GRANULARITY)

    def intensities_conv(self) -> np.ndarray:
        intensities_conv = convolve.convolve_brod(self.sim, self.lines, self.wavenumbers_line(),
                                                  self.intensities_line(), self.wavenumbers_conv())

        intensities_conv /= intensities_conv.max()
        intensities_conv *= self.franck_condon / self.sim.max_fc

        return intensities_conv

    def intensities_inst(self, broadening: float) -> np.ndarray:
        intensities_inst = convolve.convolve_inst(self.wavenumbers_conv(), self.intensities_conv(),
                                                  broadening)

        intensities_inst /= intensities_inst.max()
        intensities_inst *= self.franck_condon / self.sim.max_fc

        return intensities_inst
