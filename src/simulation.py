# module simulation

import numpy as np

from molecule import Molecule
from state import State
from band import Band
import input as inp
import convolve


class Simulation:
    def __init__(self, molecule: Molecule, temp: float, pres: float, rot_lvls: np.ndarray,
                 state_up: str, state_lo: str, band_list: list[tuple[int, int]]) -> None:
        self.molecule:  Molecule   = molecule
        self.temp:      float      = temp
        self.pres:      float      = pres
        self.rot_lvls:  np.ndarray = rot_lvls
        self.state_up:  State      = State(state_up, self.molecule.consts)
        self.state_lo:  State      = State(state_lo, self.molecule.consts)
        self.vib_bands: list[Band] = [Band(vib_band[0], vib_band[1], self)
                                      for vib_band in band_list]
        self.max_fc:    float      = max(vib_band.franck_condon for vib_band in self.vib_bands)

    def all_convolved_data(self) -> tuple[np.ndarray, np.ndarray]:
        wavenumbers_line = np.ndarray([0])
        intensities_line = np.ndarray([0])
        lines = np.ndarray([0])

        for vib_band in self.vib_bands:
            wavenumbers_line = np.concatenate((wavenumbers_line, vib_band.wavenumbers_line()))
            intensities_line = np.concatenate((intensities_line, vib_band.intensities_line()))
            lines = np.concatenate((lines, vib_band.lines))

        wavenumbers_conv = np.linspace(np.min(wavenumbers_line), np.max(wavenumbers_line),
                                       inp.GRANULARITY)
        intensities_conv = convolve.convolve_brod(self, lines, wavenumbers_line, intensities_line,
                                                  wavenumbers_conv)

        return wavenumbers_conv, intensities_conv


class LIF(Simulation):
    def __init__(self, molecule: Molecule, temp: float, pres: float, rot_lvls: np.ndarray,
                 state_up: str, state_lo: str, band_list: list[tuple[int, int]], rot_qn_up: int,
                 rot_qn_lo: int) -> None:
        super().__init__(molecule, temp, pres, rot_lvls, state_up, state_lo, band_list)

        self.rot_qn_up: int = rot_qn_up
        self.rot_qn_lo: int = rot_qn_lo


class Spectra(Simulation):
    def __init__(self, molecule: Molecule, temp: float, pres: float, rot_lvls: np.ndarray,
                 state_up: str, state_lo: str, band_list: list[tuple[int, int]]) -> None:
        super().__init__(molecule, temp, pres, rot_lvls, state_up, state_lo, band_list)
