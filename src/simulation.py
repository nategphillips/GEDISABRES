# module simulation

import numpy as np

from molecule import Molecule
from state import State
from band import Band
from line import Line
import input as inp
import convolve

class Simulation:
    def __init__(self, molecule: Molecule, temp: float, pres: float, rot_lvls: np.ndarray,
                 state_up: str, state_lo: str, vib_bands: list[tuple[int, int]]) -> None:
        self.molecule:      Molecule   = molecule
        self.temp:          float      = temp
        self.pres:          float      = pres
        self.rot_lvls:      np.ndarray = rot_lvls
        self.state_up:      State      = State(state_up, self.molecule.consts)
        self.state_lo:      State      = State(state_lo, self.molecule.consts)
        self.allowed_lines: np.ndarray = self.get_allowed_lines()
        self.vib_bands:     list[Band] = [Band(vib_band, self.allowed_lines, self.state_up,
                                               self.state_lo, self.temp, self, self.molecule)
                                         for vib_band in vib_bands]
        self.max_fc:        float      = max(vib_band.franck_condon for vib_band in self.vib_bands)

    def get_allowed_lines(self) -> np.ndarray:
        lines = []

        for rot_qn_lo in self.rot_lvls:
            for rot_qn_up in self.rot_lvls:
                d_rot_qn = rot_qn_up - rot_qn_lo

                if self.state_up.name == 'b3su':
                    rule = rot_qn_lo
                else:
                    rule = rot_qn_lo + 0.5

                if rule % 2:
                    lines.extend(self.get_allowed_branches(rot_qn_up, rot_qn_lo, d_rot_qn))

        return np.array(lines)

    def get_allowed_branches(self, rot_qn_up: int, rot_qn_lo: int, d_rot_qn: int) -> list[float]:
        lines = []

        if self.state_up.name == 'b3su':
            branch_range = range(1, 4)

            # P branch
            if d_rot_qn == -1:
                lines.extend(self.get_branch_idx(rot_qn_up, rot_qn_lo, branch_range, 'p', 'pq'))

            # R branch
            elif d_rot_qn == 1:
                lines.extend(self.get_branch_idx(rot_qn_up, rot_qn_lo, branch_range, 'r', 'rq'))

        else:
            branch_range = range(1, 3)

            # P branch
            if d_rot_qn == -1:
                lines.extend(self.get_branch_idx(rot_qn_up, rot_qn_lo, branch_range, 'p', 'pq'))

            # Q branch
            elif d_rot_qn == 0:
                lines.extend(self.get_branch_idx(rot_qn_up, rot_qn_lo, branch_range, 'q', 'qq'))

            # R branch
            elif d_rot_qn == 1:
                lines.extend(self.get_branch_idx(rot_qn_up, rot_qn_lo, branch_range, 'r', 'rq'))


        return lines

    def get_branch_idx(self, rot_qn_up: int, rot_qn_lo: int, branch_range: range, branch_main: str,
                       branch_secondary: str) -> list[float]:
        lines = []

        for branch_idx_lo in branch_range:
            for branch_idx_up in branch_range:
                if branch_idx_lo == branch_idx_up:
                    lines.append(Line(rot_qn_up, rot_qn_lo, branch_idx_up,
                                      branch_idx_lo, branch_main, self.molecule))
                if self.molecule.name == 'o2':
                    # NOTE: 03/04/24 don't have Honl-London factors for satellite bands, therefore
                    #                can't do anything other than O2 for now
                    if branch_main == 'p':
                        if branch_idx_lo < branch_idx_up:
                            lines.append(Line(rot_qn_up, rot_qn_lo, branch_idx_up,
                                              branch_idx_lo, branch_secondary, self.molecule))
                    elif branch_main == 'r':
                        if branch_idx_lo > branch_idx_up:
                            lines.append(Line(rot_qn_up, rot_qn_lo, branch_idx_up,
                                              branch_idx_lo, branch_secondary, self.molecule))

        return lines

    def all_convolved_data(self) -> tuple[np.ndarray, np.ndarray]:
        wavenumbers_line = np.ndarray([0])
        intensities_line = np.ndarray([0])
        lines            = np.ndarray([0])

        for vib_band in self.vib_bands:
            wavenumbers_line = np.concatenate((wavenumbers_line, vib_band.wavenumbers_line()))
            intensities_line = np.concatenate((intensities_line, vib_band.intensities_line()))
            lines            = np.concatenate((lines, vib_band.lines))

        wavenumbers_conv = np.linspace(wavenumbers_line.min(), wavenumbers_line.max(),
                                       inp.GRANULARITY)
        intensities_conv = convolve.convolve_brod(self, lines, wavenumbers_line, intensities_line,
                                                  wavenumbers_conv)

        return wavenumbers_conv, intensities_conv
