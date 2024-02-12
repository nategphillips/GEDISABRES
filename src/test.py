from dataclasses import dataclass
import itertools

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

AVOGD = 6.02214076e23

class Atom:
    def __init__(self, name):
        self.name = name
        self.mass = pd.read_csv('../data/atomic_masses.csv', index_col=0).loc[self.name].iloc[0] \
                    / AVOGD / 1e3

class Molecule:
    def __init__(self, name, atom_1, atom_2):
        self.name   = name
        self.atom_1 = Atom(atom_1)
        self.atom_2 = Atom(atom_2)
        self.consts = pd.read_csv(f'../data/molecular_constants/{self.name}.csv', index_col=0)

        self.molecular_mass = self.atom_1.mass + self.atom_2.mass
        self.reduced_mass   = (self.atom_1.mass * self.atom_2.mass) / self.molecular_mass

class Simulation:
    def __init__(self, molecule, rot_levels, upper_state, lower_state, vib_bands):
        self.molecule      = molecule
        self.rot_levels    = rot_levels
        self.upper_state   = ElectronicState(upper_state, self.molecule.consts)
        self.lower_state   = ElectronicState(lower_state, self.molecule.consts)
        self.allowed_lines = self.get_allowed_lines()
        self.vib_bands     = [VibrationalBand(vib_band, self.allowed_lines,
                             self.upper_state, self.lower_state) for vib_band in vib_bands]

    def get_allowed_lines(self):
        lines = []

        for gnd_rot_qn, ext_rot_qn in itertools.product(self.rot_levels, repeat=2):
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

    def plot_lines(self):
        for vib_band in self.vib_bands:
            plt.stem(vib_band.wavenumbers(), vib_band.intensities())

        plt.show()

class ElectronicState:
    def __init__(self, name, consts):
        self.name   = name
        self.consts = consts.loc[self.name].to_dict()

    def electronic_term(self):
        return self.consts['t_e']

class VibrationalBand:
    def __init__(self, name, lines, upper_state, lower_state):
        self.vib_qn_up   = name[0]
        self.vib_qn_lo   = name[1]
        self.lines       = lines
        self.upper_state = upper_state
        self.lower_state = lower_state

    def vibrational_term(self, state, vib_qn):
        return state.consts['w_e']   * (vib_qn + 0.5)    - \
               state.consts['we_xe'] * (vib_qn + 0.5)**2 + \
               state.consts['we_ye'] * (vib_qn + 0.5)**3 + \
               state.consts['we_ze'] * (vib_qn + 0.5)**4

    def rotational_constants(self, state, vib_qn):
        b_v = state.consts['b_e']                        - \
              state.consts['alph_e'] * (vib_qn + 0.5)    + \
              state.consts['gamm_e'] * (vib_qn + 0.5)**2 + \
              state.consts['delt_e'] * (vib_qn + 0.5)**3

        d_v = state.consts['d_e'] - state.consts['beta_e'] * (vib_qn + 0.5)

        h_v = state.consts['h_e']

        return [b_v, d_v, h_v]

    def band_origin(self):
        # called once per band
        elc_energy = self.upper_state.electronic_term() - \
                     self.lower_state.electronic_term()

        vib_energy = self.vibrational_term(self.upper_state, self.vib_qn_up) - \
                     self.vibrational_term(self.lower_state, self.vib_qn_lo)

        return elc_energy + vib_energy

    def wavenumbers(self):
        return np.array([line.wavenumber(self.band_origin(), self.upper_state, self.lower_state, self.vib_qn_up, self.vib_qn_lo) for line in self.lines])

    def intensities(self):
        return np.array([line.intensity() for line in self.lines])

@dataclass
class SpectralLine:
    rot_qn_up:      int
    rot_qn_lo:      int
    triplet_idx_up: int
    triplet_idx_lo: int
    branch:         str

    def rotational_term(self, rot_qn, vib_qn, state, branch_idx):
        # called for each line
        x = rot_qn * (rot_qn + 1)
        b = state.rotational_constants(vib_qn)[0]
        d = state.rotational_constants(vib_qn)[1]
        h = state.rotational_constants(vib_qn)[2]

        # accessed
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

    def wavenumber(self, band_origin, state_up, state_lo, vib_qn_up, vib_qn_lo):
        return band_origin + self.rotational_term(self.rot_qn_up, vib_qn_up, state_up, self.triplet_idx_up) - \
                             self.rotational_term(self.rot_qn_lo, vib_qn_lo, state_lo, self.triplet_idx_lo)

    def intensity(self):
        return 1

def main():
    o2_mol = Molecule('o2', 'o', 'o')
    o2_sim = Simulation(o2_mol, np.arange(0, 36), 'b3su', 'x3sg', [(1, 0), (2, 0)])

    o2_sim.plot_lines()

if __name__ == '__main__':
    main()
