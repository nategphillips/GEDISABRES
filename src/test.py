# module test

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

class Atom:
    def __init__(self, name: str) -> None:
        self.name: str   = name
        self.mass: float = (pd.read_csv('../data/atomic_masses.csv', index_col=0)
                            .loc[self.name]
                            .iloc[0] / AVOGD / 1e3)

class Molecule:
    def __init__(self, name: str, atom_1: str, atom_2: str) -> None:
        self.name:   str          = name
        self.atom_1: Atom         = Atom(atom_1)
        self.atom_2: Atom         = Atom(atom_2)
        self.consts: pd.DataFrame = pd.read_csv(f'../data/molecular_constants/{self.name}.csv',
                                                index_col=0)

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
                                                                     self.temp)
                                                    for vib_band in vib_bands]

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

    def get_branch_idx(self, rot_qn_up: int, rot_qn_lo: int, branch_indices: range,
                       branch_main: str, branch_secondary: str) -> list[float]:
        lines = []

        for branch_idx_lo in branch_indices:
            for branch_idx_up in branch_indices:
                if branch_idx_lo == branch_idx_up:
                    lines.append(SpectralLine(rot_qn_up, rot_qn_lo, branch_idx_up,
                                              branch_idx_lo, branch_main))
                elif branch_idx_lo > branch_idx_up:
                    if self.state_up.name == 'b3su':
                        lines.append(SpectralLine(rot_qn_up, rot_qn_lo, branch_idx_up,
                                                  branch_idx_lo, branch_secondary))

        return lines

class VibrationalBand:
    def __init__(self, name: tuple[int, int], lines: np.ndarray, state_up: ElectronicState,
                 state_lo: ElectronicState, temp: float) -> None:
        self.vib_qn_up:   int             = name[0]
        self.vib_qn_lo:   int             = name[1]
        self.lines:       np.ndarray      = lines
        self.state_up:    ElectronicState = state_up
        self.state_lo:    ElectronicState = state_lo
        self.temp:        float           = temp
        self.band_origin: float           = self.get_band_origin()

    def get_band_origin(self) -> float:
        elc_energy = self.state_up.consts['t_e'] - self.state_lo.consts['t_e']

        vib_energy = vibrational_term(self.state_up, self.vib_qn_up) - \
                     vibrational_term(self.state_lo, self.vib_qn_lo)

        return elc_energy + vib_energy

    def wavenumbers(self) -> np.ndarray:
        return np.array([line.wavenumber(self.band_origin, self.vib_qn_up, self.vib_qn_lo,
                                         self.state_up, self.state_lo) for line in self.lines])

    def intensities(self) -> np.ndarray:
        return np.array([line.intensity(self.band_origin, self.vib_qn_up, self.vib_qn_lo,
                                        self.state_up, self.state_lo, self.temp)
                                        for line in self.lines])

@dataclass
class SpectralLine:
    rot_qn_up:     int
    rot_qn_lo:     int
    branch_idx_up: int
    branch_idx_lo: int
    branch:        str

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
        x = rot_qn * (rot_qn + 1)

        l = state.consts['lamd']
        g = state.consts['gamm']

        single = x * b - x**2 * d + x**3 * h

        e0 = x * b - (x**2 + 4 * x) * d
        e1 = (x + 2) * b - (x**2 + 8 * x + 4) * d - 2 * l - g
        odiag = - 2 * np.sqrt(x) * (b - 2 * (x + 1) * d - g / 2)

        energy1 = 0.5 * ((e0 + e1) + np.sqrt((e0 + e1)**2 - 4 * (e0 * e1 - odiag**2)))
        energy2 = 0.5 * ((e0 + e1) - np.sqrt((e0 + e1)**2 - 4 * (e0 * e1 - odiag**2)))

        match branch_idx:
            case 1:
                return energy1
            case 3:
                return energy2
            case _:
                return single
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

def plot_lines(sim: Simulation, color: str, max_intn: float) -> None:
    for vib_band in sim.vib_bands:
        plt.stem(vib_band.wavenumbers(), vib_band.intensities() / max_intn, color, markerfmt='',
                 label=sim.molecule.name)

def broadening_fn(sim: Simulation, convolved_wavenumbers: np.ndarray,
                  wavenumber_peak: float) -> float:
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
    prediss = 0 # lines[idx].predissociation()

    # TODO: this might be wrong, not sure if the parameters just add together or what
    gauss = doppler
    loren = natural + collide + prediss

    # Faddeeva function
    fadd = ((convolved_wavenumbers - wavenumber_peak) + 1j * loren) / (gauss * np.sqrt(2))

    return np.real(wofz(fadd)) / (gauss * np.sqrt(2 * np.pi))

def convolve_lines(sim: Simulation, band: VibrationalBand,
                   convolved_wavenumbers: np.ndarray) -> np.ndarray:
    convolved_intensities = np.zeros_like(convolved_wavenumbers)

    for _, line in enumerate(sim.allowed_lines):
        convolved_intensities += line.intensity(band.band_origin, band.vib_qn_up, band.vib_qn_lo,
                                                sim.state_up, sim.state_lo, sim.temp) * \
                                                broadening_fn(sim, convolved_wavenumbers,
                                                line.wavenumber(band.band_origin, band.vib_qn_up,
                                                                band.vib_qn_lo, sim.state_up,
                                                                sim.state_lo))

    return convolved_intensities / convolved_intensities.max()

def main():
    default_temp = 300    # [K]
    default_pres = 101325 # [Pa]

    mol_o2  = Molecule('o2', 'o', 'o')
    mol_o2p = Molecule('o2+', 'o', 'o')

    o2_sim  = Simulation(mol_o2, default_temp, default_pres,
                         np.arange(0, 36, 1), 'b3su', 'x3sg', [(0, 6)])
    o2p_sim = Simulation(mol_o2p, default_temp, default_pres,
                         np.arange(0.5, 25.5, 1), 'a2pu', 'x2pg', [(0, 0)])

    max_o2   = np.array([vib_band.intensities() for vib_band in o2_sim.vib_bands]).flatten().max()
    max_o2p  = np.array([vib_band.intensities() for vib_band in o2p_sim.vib_bands]).flatten().max()
    max_intn = max(max_o2, max_o2p)

    plot_lines(o2_sim, 'r', max_intn)
    plot_lines(o2p_sim, 'b', max_intn)

    conv_wns = np.linspace(39600, 40400, 5000)
    conv_ins = convolve_lines(o2_sim, o2_sim.vib_bands[0], conv_wns)

    plt.plot(conv_wns, conv_ins)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
