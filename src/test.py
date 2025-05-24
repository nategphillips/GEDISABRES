# module test

# Copyright (C) 2023-2025 Nathan G. Phillips

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from dataclasses import dataclass

from scipy.special import wofz # pylint: disable=no-name-in-module
import scienceplots # pylint: disable = unused-import
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# plt.style.use(['science', 'grid'])

BOLTZ = 1.380649e-23   # Boltzmann constant [J/K]
PLANC = 6.62607015e-34 # Planck constant    [J*s]
LIGHT = 2.99792458e10  # speed of light     [cm/s]
AVOGD = 6.02214076e23  # Avodagro constant  [1/mol]

GRANULARITY = 10000

class Atom:
    def __init__(self, name: str) -> None:
        self.name: str   = name
        self.mass: float = (pd.read_csv('../data/atomic_masses.csv', index_col=0)
                            .loc[self.name]
                            .iloc[0] / AVOGD / 1e3)

class Molecule:
    def __init__(self, name: str, atom_1: str, atom_2: str) -> None:
        self.name:    str          = name
        self.atom_1:  Atom         = Atom(atom_1)
        self.atom_2:  Atom         = Atom(atom_2)
        self.consts:  pd.DataFrame = pd.read_csv(f'../data/molecular_constants/{self.name}.csv',
                                                 index_col=0)
        self.prediss: pd.DataFrame = pd.read_csv(f'../data/predissociation/{self.name}.csv')
        self.fc_data: np.ndarray   = np.loadtxt(f'../data/franck-condon/{self.name}.csv',
                                                delimiter=',')

        self.molecular_mass: float = self.atom_1.mass + self.atom_2.mass
        self.reduced_mass:   float = self.atom_1.mass * self.atom_2.mass / self.molecular_mass

class ElectronicState:
    def __init__(self, name: str, consts: pd.DataFrame) -> None:
        self.name:          str              = name
        self.consts:        dict[str, float] = consts.loc[self.name].to_dict()
        self.cross_section: float            = np.pi * (2 * self.consts['rad'])**2

class Simulation:
    def __init__(self, molecule: Molecule, temp: float, pres: float, rot_lvls: np.ndarray,
                 state_up: str, state_lo: str, vib_bands: list[tuple[int, int]]) -> None:
        self.molecule:      Molecule              = molecule
        self.temp:          float                 = temp
        self.pres:          float                 = pres
        self.rot_lvls:      np.ndarray            = rot_lvls
        self.state_up:      ElectronicState       = ElectronicState(state_up, self.molecule.consts)
        self.state_lo:      ElectronicState       = ElectronicState(state_lo, self.molecule.consts)
        self.allowed_lines: np.ndarray            = self.get_allowed_lines()
        self.vib_bands:     list[VibrationalBand] = [VibrationalBand(vib_band, self.allowed_lines,
                                                                     self.state_up, self.state_lo,
                                                                     self.temp, self, self.molecule)
                                                    for vib_band in vib_bands]
        self.max_fc:        float                 = max(vib_band.franck_condon
                                                        for vib_band in self.vib_bands)

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
                    lines.append(SpectralLine(rot_qn_up, rot_qn_lo, branch_idx_up,
                                              branch_idx_lo, branch_main, self.molecule))
                if self.molecule.name == 'o2':
                    # NOTE: 03/04/24 don't have Honl-London factors for satellite bands, therefore
                    #                can't do anything other than O2 for now
                    if branch_main == 'p':
                        if branch_idx_lo < branch_idx_up:
                            lines.append(SpectralLine(rot_qn_up, rot_qn_lo, branch_idx_up,
                                                    branch_idx_lo, branch_secondary, self.molecule))
                    elif branch_main == 'r':
                        if branch_idx_lo > branch_idx_up:
                            lines.append(SpectralLine(rot_qn_up, rot_qn_lo, branch_idx_up,
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

        wavenumbers_conv = np.linspace(wavenumbers_line.min(), wavenumbers_line.max(), GRANULARITY)
        intensities_conv = convolve_brod(self, lines, wavenumbers_line, intensities_line,
                                         wavenumbers_conv)

        return wavenumbers_conv, intensities_conv

class VibrationalBand:
    def __init__(self, name: tuple[int, int], lines: np.ndarray, state_up: ElectronicState,
                 state_lo: ElectronicState, temp: float, sim: Simulation,
                 molecule: Molecule) -> None:
        self.name:          tuple           = name
        self.vib_qn_up:     int             = name[0]
        self.vib_qn_lo:     int             = name[1]
        self.lines:         np.ndarray      = lines
        self.state_up:      ElectronicState = state_up
        self.state_lo:      ElectronicState = state_lo
        self.temp:          float           = temp
        self.sim:           Simulation      = sim
        self.molecule:      Molecule        = molecule
        self.band_origin:   float           = self.get_band_origin()
        self.franck_condon: float           = self.molecule.fc_data[self.vib_qn_up][self.vib_qn_lo]

    def get_band_origin(self) -> float:
        elc_energy = self.state_up.consts['t_e'] - self.state_lo.consts['t_e']

        vib_energy = vibrational_term(self.state_up, self.vib_qn_up) - \
                     vibrational_term(self.state_lo, self.vib_qn_lo)

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

        return np.linspace(wavenumbers_line.min(), wavenumbers_line.max(), GRANULARITY)

    def intensities_conv(self) -> np.ndarray:
        intensities_conv = convolve_brod(self.sim, self.lines, self.wavenumbers_line(),
                                         self.intensities_line(), self.wavenumbers_conv())

        intensities_conv /= intensities_conv.max()
        intensities_conv *= self.franck_condon / self.sim.max_fc

        return intensities_conv

    def intensities_inst(self, broadening: float) -> np.ndarray:
        intensities_inst = convolve_inst(self.wavenumbers_conv(), self.intensities_conv(),
                                         broadening)

        intensities_inst /= intensities_inst.max()
        intensities_inst *= self.franck_condon / self.sim.max_fc

        return intensities_inst

@dataclass
class SpectralLine:
    rot_qn_up:     int
    rot_qn_lo:     int
    branch_idx_up: int
    branch_idx_lo: int
    branch:        str
    molecule:      Molecule

    def predissociation(self) -> float:
        if self.molecule.name == 'o2':
            return self.molecule.prediss[f'f{self.branch_idx_lo}'] \
                                        [self.molecule.prediss['rot_qn'] == self.rot_qn_up].iloc[0]

        return 0

    def wavenumber(self, band_origin: float, vib_qn_up: int, vib_qn_lo: int,
                   state_up: ElectronicState, state_lo: ElectronicState) -> float:
        return band_origin + \
               rotational_term(self.rot_qn_up, vib_qn_up, state_up, self.branch_idx_up) - \
               rotational_term(self.rot_qn_lo, vib_qn_lo, state_lo, self.branch_idx_lo)

    def intensity(self, band_origin: float, vib_qn_up: int, vib_qn_lo: int,
                  state_up: ElectronicState, state_lo: ElectronicState, temp: float) -> float:
        part = BOLTZ * temp / (PLANC * LIGHT * state_lo.consts['b_e'])

        base = self.wavenumber(band_origin, vib_qn_up, vib_qn_lo, state_up, state_lo) / part * \
               np.exp(- rotational_term(self.rot_qn_lo, vib_qn_lo, state_lo,
                                        self.branch_idx_lo) * PLANC * LIGHT / (BOLTZ * temp))

        if state_up.name == 'b3su':
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

            if self.branch_idx_lo in (1, 3):
                return intn / 2

        else:
            lambda_lo = 1

            match self.branch:
                case 'r':
                    linestr = ((self.rot_qn_lo + 1 + lambda_lo) * \
                               (self.rot_qn_lo + 1 - lambda_lo)) / (self.rot_qn_lo + 1)
                    intn = base * linestr
                case 'p':
                    linestr = ((self.rot_qn_lo + lambda_lo) * \
                               (self.rot_qn_lo - lambda_lo)) / self.rot_qn_lo
                    intn = base * linestr
                case 'q':
                    linestr = ((2 * self.rot_qn_lo + 1) * lambda_lo**2) / \
                              (self.rot_qn_lo * (self.rot_qn_lo + 1))
                    intn = base * linestr

        return intn

def vibrational_term(state: ElectronicState, vib_qn: int) -> float:
    return state.consts['w_e']   * (vib_qn + 0.5)    - \
           state.consts['we_xe'] * (vib_qn + 0.5)**2 + \
           state.consts['we_ye'] * (vib_qn + 0.5)**3 + \
           state.consts['we_ze'] * (vib_qn + 0.5)**4

def rotational_constants(state: ElectronicState, vib_qn: int) -> list[float]:
    b_v = state.consts['b_e']                        - \
          state.consts['alph_e'] * (vib_qn + 0.5)    + \
          state.consts['gamm_e'] * (vib_qn + 0.5)**2 + \
          state.consts['delt_e'] * (vib_qn + 0.5)**3

    d_v = state.consts['d_e'] - state.consts['beta_e'] * (vib_qn + 0.5)

    h_v = state.consts['h_e']

    return [b_v, d_v, h_v]

def rotational_term(rot_qn: int, vib_qn: int, state: ElectronicState, branch_idx: int) -> float:
    b, d, h = rotational_constants(state, vib_qn)

    if state.name in ('b3su', 'x3sg'):
        sqrt_sign  = 1
        first_term = b * rot_qn * (rot_qn + 1)       - \
                     d * rot_qn**2 * (rot_qn + 1)**2 + \
                     h * rot_qn**3 * (rot_qn + 1)**3

        match branch_idx:
            case 1:
                return first_term + (2 * rot_qn + 3) * b - \
                       state.consts['lamd'] - sqrt_sign * np.sqrt((2 * rot_qn + 3)**2 * \
                       b**2 + state.consts['lamd']**2 - 2 * \
                       state.consts['lamd'] * b) + \
                       state.consts['gamm'] * (rot_qn + 1)

            case 3:
                return first_term - (2 * rot_qn - 1) * b - \
                       state.consts['lamd'] + sqrt_sign * np.sqrt((2 * rot_qn - 1)**2 * \
                       b**2 + state.consts['lamd']**2 - 2 * \
                       state.consts['lamd'] * b) - \
                       state.consts['gamm'] * rot_qn

            case _:
                return first_term
    else:
        lamb = 1
        a = state.consts['coupling']
        y = a / b

        match branch_idx:
            case 1:
                return b * ((rot_qn + 0.5)**2 - lamb**2 - \
                       0.5 * np.sqrt(4 * (rot_qn + 0.5)**2 + y * (y - 4) * lamb**2)) - \
                       d * rot_qn**4
            case 2:
                return b * ((rot_qn + 0.5)**2 - lamb**2 + \
                       0.5 * np.sqrt(4 * (rot_qn + 0.5)**2 + y * (y - 4) * lamb**2)) - \
                       d * (rot_qn + 1)**4
            case _:
                raise ValueError('error')

def plot_samp(samp_file: str, color: str, type: str = 'stem') -> None:
    sample_data = pd.read_csv(f'../data/{samp_file}.csv')

    wavenumbers = sample_data['wavenumbers'].to_numpy()
    intensities = sample_data['intensities'].to_numpy()
    intensities /= max(intensities)

    if type == 'stem':
        plt.stem(wavenumbers, intensities, color, markerfmt='', label=samp_file)
    else:
        plt.plot(wavenumbers, intensities, color, label=samp_file)

def plot_info(sim: Simulation) -> None:
    for vib_band in sim.vib_bands:
        wavenumbers_line = vib_band.wavenumbers_line()
        intensities_line = vib_band.intensities_line()
        lines            = vib_band.lines

        for idx, line in enumerate(lines):
            plt.text(wavenumbers_line[idx], intensities_line[idx], f'{line.branch}')

def plot_line(sim: Simulation, color: str) -> None:
    for vib_band in sim.vib_bands:
        plt.stem(vib_band.wavenumbers_line(), vib_band.intensities_line(), color,
                 markerfmt='', label=f'{sim.molecule.name} {vib_band.name} line')

def plot_conv(sim: Simulation, color: str) -> None:
    for vib_band in sim.vib_bands:
        plt.plot(vib_band.wavenumbers_conv(), vib_band.intensities_conv(), color,
                 label=f'{sim.molecule.name} {vib_band.name} conv')

def plot_conv_all(sim: Simulation, color: str) -> None:
    wavenumbers_conv, intensities_conv = sim.all_convolved_data()

    intensities_conv /= intensities_conv.max()

    plt.plot(wavenumbers_conv, intensities_conv, color, label=f'{sim.molecule.name} conv all')

def plot_inst(sim: Simulation, color: str, broadening: float) -> None:
    for vib_band in sim.vib_bands:
        plt.plot(vib_band.wavenumbers_conv(), vib_band.intensities_inst(broadening), color,
                 label=f'{sim.molecule.name} {vib_band.name} inst')

def plot_inst_all(sim: Simulation, color: str, broadening: float) -> None:
    wavenumbers_conv, intensities_conv = sim.all_convolved_data()
    intensities_inst = convolve_inst(wavenumbers_conv, intensities_conv, broadening)

    intensities_inst /= intensities_inst.max()

    plt.plot(wavenumbers_conv, intensities_inst, color, label=f'{sim.molecule.name} inst all')

def convolve_inst(wavenumbers_conv: np.ndarray, intensities_conv: np.ndarray,
                  broadening: float) -> np.ndarray:
    intensities_inst = np.zeros_like(wavenumbers_conv)

    for wave, intn in zip(wavenumbers_conv, intensities_conv):
        intensities_inst += intn * instrument_fn(wavenumbers_conv, wave, broadening)

    return intensities_inst

def convolve_brod(sim: Simulation, lines: np.ndarray, wavenumbers_line: np.ndarray,
                  intensities_line: np.ndarray, wavenumbers_conv: np.ndarray) -> np.ndarray:
    intensities_conv = np.zeros_like(wavenumbers_conv)

    for idx, (wave, intn) in enumerate(zip(wavenumbers_line, intensities_line)):
        intensities_conv += intn * broadening_fn(sim, lines, wavenumbers_conv, wave, idx)

    return intensities_conv

def instrument_fn(convolved_wavenumbers: np.ndarray, wavenumber_peak: float,
                  broadening: float) -> float:
    return np.exp(- 0.5 * (convolved_wavenumbers - wavenumber_peak)**2 / broadening**2) / \
           (broadening * np.sqrt(2 * np.pi))

def broadening_fn(sim: Simulation, lines: np.ndarray, convolved_wavenumbers: np.ndarray,
                  wavenumber_peak: float, line_idx: int) -> float:
    # TODO: 11/19/23 this function needs to be reworked since I also want to include the ability to
    #                convolve with an instrument function - ideally it takes in a convolution type
    #                and broadening parameters

    # natural (Lorentzian)
    natural = sim.state_lo.cross_section**2 * \
              np.sqrt(8 / (np.pi * sim.molecule.reduced_mass * BOLTZ * sim.temp)) / 4

    # doppler (Gaussian)
    doppler = wavenumber_peak * \
              np.sqrt(BOLTZ * sim.temp / (sim.molecule.molecular_mass * (LIGHT / 1e2)**2))

    # collisional (Lorentzian)
    # convert pressure in [N/m^2] to pressure in [dyne/cm^2]
    collide = (sim.pres * 10) * sim.state_lo.cross_section**2 * \
              np.sqrt(8 / (np.pi * sim.molecule.reduced_mass * BOLTZ * sim.temp)) / 2

    # predissociation (Lorentzian)
    prediss = lines[line_idx].predissociation()

    # TODO: this might be wrong, not sure if the parameters just add together or what
    gauss = doppler
    loren = natural + collide + prediss

    # Faddeeva function
    fadd = ((convolved_wavenumbers - wavenumber_peak) + 1j * loren) / (gauss * np.sqrt(2))

    return np.real(wofz(fadd)) / (gauss * np.sqrt(2 * np.pi))

def main():
    default_temp = 300    # [K]
    default_pres = 101325 # [Pa]

    mol_o2  = Molecule('o2', 'o', 'o')
    mol_o2p = Molecule('o2+', 'o', 'o')

    o2_sim  = Simulation(mol_o2, default_temp, default_pres,
                         np.arange(0, 36, 1), 'b3su', 'x3sg', [(2, 0)])
    o2p_sim = Simulation(mol_o2p, default_temp, default_pres,
                         np.arange(0.5, 35.5, 1), 'a2pu', 'x2pg', [(0, 0)])

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors     = prop_cycle.by_key()['color']

    plot_conv(o2_sim, colors[0])
    plot_samp('harvard/harvard20', colors[1], 'plot')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
