# module energy
'''
Used for calculating the energy term values for each state.
'''

import numpy as np

class State:
    def __init__(self, constants: dict, vib_qn: int):
        self.consts = constants
        self.vib_qn = vib_qn

    def rotational_constants(self) -> list[float]:
        b_v = self.consts['b_e']                             - \
              self.consts['alph_e'] * (self.vib_qn + 0.5)    + \
              self.consts['gamm_e'] * (self.vib_qn + 0.5)**2 + \
              self.consts['delt_e'] * (self.vib_qn + 0.5)**3

        d_v = self.consts['d_e'] - self.consts['beta_e'] * (self.vib_qn + 0.5)

        h_v = self.consts['h_e']

        return [b_v, d_v, h_v]

    def electronic_term(self) -> float:
        return self.consts['t_e']

    def vibrational_term(self) -> float:
        return self.consts['w_e']   * (self.vib_qn + 0.5)    - \
               self.consts['we_xe'] * (self.vib_qn + 0.5)**2 + \
               self.consts['we_ye'] * (self.vib_qn + 0.5)**3 + \
               self.consts['we_ze'] * (self.vib_qn + 0.5)**4

def get_band_origin(gnd_state: 'State', ext_state: 'State') -> float:
    elc_energy = ext_state.electronic_term() - gnd_state.electronic_term()
    vib_energy = ext_state.vibrational_term() - gnd_state.vibrational_term()

    return elc_energy + vib_energy

def rotational_term(rot_qn: int, state: 'State', branch_idx: int) -> float:
    # TODO: 9/24/23 this was causing the weird lines appearing where they shouldn't be, removing it
    #               fixed the issue of random lines at high wavenumbers. Matches data much better &
    #               not sure why it's in the book if it's wrong, I probably misunderstood it

    # NOTE: see footnote 2 on pg. 223 of Herzberg
    #       for N = 1, the sign in front of the square root must be inverted
    # if rot_qn == 1:
    #     sqrt_sign = -1
    # else:
    #     sqrt_sign = 1

    # sqrt_sign = 1

    # first_term = state.rotational_constants()[0] * rot_qn * (rot_qn + 1) - \
    #              state.rotational_constants()[1] * rot_qn**2 * (rot_qn + 1)**2 + \
    #              state.rotational_constants()[2] * rot_qn**3 * (rot_qn + 1)**3

    # # FIXME: 9/19/23 the sign in front of the state.spn_const[0] should 100% be a negative. I've
    # #                checked multiple sources at this point. I need to find out why there seems to
    # #                be issues with how the triplet branches are spaced. leaving it positive for now
    # #                since it seems to give better results

    # match branch_idx:
    #     case 1:
    #         return first_term + (2 * rot_qn + 3) * state.rotational_constants()[0] - \
    #                state.spn_consts[0] - sqrt_sign * np.sqrt((2 * rot_qn + 3)**2 * \
    #                state.rotational_constants()[0]**2 + state.spn_consts[0]**2 - 2 * \
    #                state.spn_consts[0] * state.rotational_constants()[0]) + \
    #                state.spn_consts[1] * (rot_qn + 1)

    #     case 3:
    #         return first_term - (2 * rot_qn - 1) * state.rotational_constants()[0] - \
    #                state.spn_consts[0] + sqrt_sign * np.sqrt((2 * rot_qn - 1)**2 * \
    #                state.rotational_constants()[0]**2 + state.spn_consts[0]**2 - 2 * \
    #                state.spn_consts[0] * state.rotational_constants()[0]) - \
    #                state.spn_consts[1] * rot_qn

    #     case _:
    #         return first_term

    # NOTE: 01/24/24 this formulation (from Bergeman) of the triplets uses the negative
    #                spin-rotation constant in the upper term and works well. I still need to figure
    #                out the Hamiltonian matrix elements given since they give a second term for the
    #                two triplets (and I'm not sure how to implement it correctly)

    # FIXME: 01/24/24 work in progress (see note), also I had to subtract a factor of 2 * lambda / 3
    #                 from the elements given in Bergeman; I think this is due to different authors
    #                 using different definitions of the Hamiltonian amtrix elements like Bergeman
    #                 mentions in his paper

    x = rot_qn * (rot_qn + 1)
    b = state.rotational_constants()[0]
    d = state.rotational_constants()[1]
    h = state.rotational_constants()[2]
    l = state.consts['lamd']
    g = state.consts['gamm']

    e0 = x * b - x**2 * d + x**3 * h
    e1 = x * b - (x**2 + 4 * x) * d
    e2 = (x + 2) * b - (x**2 + 8 * x + 4) * d - 2 * l - g
    thing = - 2 * np.sqrt(x) * (b - 2 * (x + 1) * d - g / 2)

    match branch_idx:
        case 1:
            return e1
        case 3:
            return e2
        case _:
            return e0
