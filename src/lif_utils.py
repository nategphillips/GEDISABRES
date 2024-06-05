# module lif_utils

from dataclasses import dataclass

import numpy as np

import terms
import constants as cn
from state import State
from molecule import Molecule

class LifSimulation:
    def __init__(self, molecule: Molecule, temp: float, pres: float, rot_lvls: np.ndarray,
                 state_up: str, state_lo: str, band_list: list[tuple[int, int]]) -> None:
        self.molecule:  Molecule      = molecule
        self.temp:      float         = temp
        self.pres:      float         = pres
        self.rot_lvls:  np.ndarray    = rot_lvls
        self.state_up:  State         = State(state_up, self.molecule.consts)
        self.state_lo:  State         = State(state_lo, self.molecule.consts)
        self.vib_bands: list[LifBand] = [LifBand(vib_band[0], vib_band[1], self)
                                         for vib_band in band_list]
        self.max_fc:    float         = max(vib_band.franck_condon for vib_band in self.vib_bands)

class LifBand:
    def __init__(self, vib_qn_up: int, vib_qn_lo: int, sim: LifSimulation) -> None:
        self.vib_qn_up:     int        = vib_qn_up
        self.vib_qn_lo:     int        = vib_qn_lo
        self.sim:           LifSimulation = sim
        self.band_origin:   float      = self.get_band_origin()
        self.lines:         np.ndarray = self.get_allowed_lines()
        self.franck_condon: float      = self.sim.molecule.fc_data[self.vib_qn_up][self.vib_qn_lo]

    def get_lif_lines(self, rot_qn_up: int, rot_qn_lo: int) -> np.ndarray:
        lines = []

        if rot_qn_lo % 2:
            lines.extend(self.get_allowed_branches(rot_qn_up, rot_qn_lo))

        return np.array(lines)

    def get_allowed_lines(self) -> np.ndarray:
        lines = []

        for rot_qn_up in self.sim.rot_lvls:
            for rot_qn_lo in self.sim.rot_lvls:
                if rot_qn_lo % 2:
                    lines.extend(self.get_allowed_branches(rot_qn_up, rot_qn_lo))

        return np.array(lines)

    def get_allowed_branches(self, rot_qn_up: int, rot_qn_lo: int) -> list[float]:
        lines = []

        branch_range = range(1, 4)

        delta_rot_qn = rot_qn_up - rot_qn_lo

        if delta_rot_qn == 1:
            lines.extend(self.get_branch_idx(rot_qn_up, rot_qn_lo, branch_range, 'r'))

        elif delta_rot_qn == -1:
            lines.extend(self.get_branch_idx(rot_qn_up, rot_qn_lo, branch_range, 'p'))

        return lines

    def get_branch_idx(self, rot_qn_up: int, rot_qn_lo: int, branch_range: range,
                       branch_main: str) -> list[float]:
        lines = []

        for branch_idx_up in branch_range:
            for branch_idx_lo in branch_range:
                # NOTE: 06/05/24 - for LIF, only the non-satellite branches and main F2 triplet are
                #       calculated to reduce the total number of lines
                if (branch_idx_up == branch_idx_lo) & (branch_idx_up == 2):
                    lines.append(LifLine(rot_qn_up, rot_qn_lo, branch_idx_up, branch_idx_lo,
                                         branch_main, self.sim, self, self.sim.molecule))

        return lines

    def rotational_partition(self) -> float:
        q_r = 0

        for line in self.lines:
            honl_london = line.honl_london_factor()
            boltzmann   = line.boltzmann_factor()

            q_r += honl_london * boltzmann

        return q_r

    def get_band_origin(self) -> float:
        elc_energy = self.sim.state_up.consts['t_e'] - self.sim.state_lo.consts['t_e']

        vib_energy = (terms.vibrational_term(self.sim.state_up, self.vib_qn_up) -
                      terms.vibrational_term(self.sim.state_lo, self.vib_qn_lo))

        return elc_energy + vib_energy

    def wavenumbers_lif(self, rot_qn_up: int, rot_qn_lo: int) -> np.ndarray:
        return np.array([line.wavenumber() for line in self.get_lif_lines(rot_qn_up, rot_qn_lo)])

    def intensities_lif(self, rot_qn_up: int, rot_qn_lo: int) -> np.ndarray:
        intensities_lif = np.array([line.intensity() for line in
                                    self.get_lif_lines(rot_qn_up, rot_qn_lo)])

        intensities_lif /= intensities_lif.max()
        intensities_lif *= self.franck_condon / self.sim.max_fc

        return intensities_lif

    def wavenumbers_line(self) -> np.ndarray:
        return np.array([line.wavenumber() for line in self.lines])

    def intensities_line(self) -> np.ndarray:
        intensities_line = np.array([line.intensity() for line in self.lines])

        intensities_line /= intensities_line.max()
        intensities_line *= self.franck_condon / self.sim.max_fc

        return intensities_line

@dataclass
class LifLine:
    rot_qn_up:     int
    rot_qn_lo:     int
    branch_idx_up: int
    branch_idx_lo: int
    branch_name:   str
    sim:           LifSimulation
    band:          LifBand
    molecule:      Molecule

    def predissociation(self) -> float:
        return (self.molecule.prediss[f'f{self.branch_idx_lo}']
                [self.molecule.prediss['rot_qn'] == self.rot_qn_up].iloc[0])

    def wavenumber(self) -> float:
        return (self.band.band_origin +
                terms.rotational_term(self.sim.state_up, self.band.vib_qn_up, self.rot_qn_up,
                                      self.branch_idx_up) -
                terms.rotational_term(self.sim.state_lo, self.band.vib_qn_lo, self.rot_qn_lo,
                                      self.branch_idx_lo))

    def boltzmann_factor(self) -> float:
        # NOTE: 06/05/24 - assuming emission here for LIF
        state      = self.sim.state_up
        vib_qn     = self.band.vib_qn_up
        rot_qn     = self.rot_qn_up
        branch_idx = self.branch_idx_up

        return np.exp(-terms.rotational_term(state, vib_qn, rot_qn, branch_idx) *
                       cn.PLANC * cn.LIGHT / (cn.BOLTZ * self.sim.temp))

    def honl_london_factor(self) -> float:
        match self.branch_name:
            case 'r':
                line_strength = ((self.rot_qn_lo + 1)**2 - 0.25) / (self.rot_qn_lo + 1)
            case 'p':
                line_strength = (self.rot_qn_lo**2 - 0.25) / (self.rot_qn_lo)
            # NOTE: 06/05/24 - removed 'rq' and 'pq' for LIF

        return line_strength

    def intensity(self) -> float:
        # NOTE: 06/05/24 - assuming emission here for LIF
        wavenumber_part = self.wavenumber()**4

        intensity = (wavenumber_part * self.honl_london_factor() * self.boltzmann_factor() /
                     self.band.rotational_partition())

        if self.branch_idx_lo in (1, 3):
            return intensity / 2

        return intensity
