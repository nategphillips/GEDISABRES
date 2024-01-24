# module energy
'''
Used for calculating the energy term values for each state.
'''

import numpy as np

class State:
    '''
    Stores the necessary constants for either the ground state or the excited state of the current
    electronic transition.
    '''

    def __init__(self, constants: list, vib_qn: int) -> None:
        self.elc_consts = constants[0]
        self.vib_consts = constants[1:5]
        self.rot_consts = constants[5:12]
        self.spn_consts = constants[12:14]
        self.vib_qn     = vib_qn

    def rotational_constants(self) -> list[float]:
        '''
        Calculates the rotational constants B_v, D_v, and H_v.

        Returns:
            list[float]: [B_v, D_v, H_v]
        '''

        b_v = self.rot_consts[0]                          - \
              self.rot_consts[1] * (self.vib_qn + 0.5)    + \
              self.rot_consts[2] * (self.vib_qn + 0.5)**2 + \
              self.rot_consts[3] * (self.vib_qn + 0.5)**3

        d_v = self.rot_consts[4] - self.rot_consts[5] * (self.vib_qn + 0.5)

        h_v = self.rot_consts[6]

        return [b_v, d_v, h_v]

    def electronic_term(self) -> float:
        '''
        Calculates the electronic term T_e.

        Returns:
            float: electronic term T_e
        '''

        return self.elc_consts

    def vibrational_term(self) -> float:
        '''
        Calculates the vibrational term G(v).

        Returns:
            float: vibrational term G(v)
        '''
        return self.vib_consts[0] * (self.vib_qn + 0.5)    - \
               self.vib_consts[1] * (self.vib_qn + 0.5)**2 + \
               self.vib_consts[2] * (self.vib_qn + 0.5)**3 + \
               self.vib_consts[3] * (self.vib_qn + 0.5)**4

def get_band_origin(gnd_state: 'State', ext_state: 'State') -> float:
    '''
    Computes the electronic + vibrational term values.

    Args:
        gnd_state (State): ground state
        ext_state (State): excited state

    Returns:
        float: electronic + vibrational energy
    '''

    elc_energy = ext_state.electronic_term() - gnd_state.electronic_term()
    vib_energy = ext_state.vibrational_term() - gnd_state.vibrational_term()

    return elc_energy + vib_energy

def rotational_term(rot_qn: int, state: 'State', branch_idx: int) -> float:
    '''
    Calculates the rotational term value F(N).

    Args:
        rot_qn (int): current rotational quantum number
        state (State): current state
        branch_idx (int): branch index: r, p, or a satellite variation

    Returns:
        float: rotational term F(N)
    '''

    first_term = state.rotational_constants()[0] * rot_qn * (rot_qn + 1) - \
                 state.rotational_constants()[1] * rot_qn**2 * (rot_qn + 1)**2 + \
                 state.rotational_constants()[2] * rot_qn**3 * (rot_qn + 1)**3

    # TODO: 9/24/23 this was causing the weird lines appearing where they shouldn't be, removing it
    #               fixed the issue of random lines at high wavenumbers. Matches data much better &
    #               not sure why it's in the book if it's wrong, I probably misunderstood it

    # NOTE: see footnote 2 on pg. 223 of Herzberg
    #       for N = 1, the sign in front of the square root must be inverted
    # if rot_qn == 1:
    #     sqrt_sign = -1
    # else:
    #     sqrt_sign = 1

    sqrt_sign = 1

    # FIXME: 9/19/23 the sign in front of the state.spn_const[0] should 100% be a negative. I've
    #                checked multiple sources at this point. I need to find out why there seems to
    #                be issues with how the triplet branches are spaced. leaving it positive for now
    #                since it seems to give better results
    match branch_idx:
        case 1:
            return first_term + (2 * rot_qn + 3) * state.rotational_constants()[0] + \
                   state.spn_consts[0] - sqrt_sign * np.sqrt((2 * rot_qn + 3)**2 * \
                   state.rotational_constants()[0]**2 + state.spn_consts[0]**2 - 2 * \
                   state.spn_consts[0] * state.rotational_constants()[0]) + \
                   state.spn_consts[1] * (rot_qn + 1)

        case 3:
            return first_term - (2 * rot_qn - 1) * state.rotational_constants()[0] - \
                   state.spn_consts[0] + sqrt_sign * np.sqrt((2 * rot_qn - 1)**2 * \
                   state.rotational_constants()[0]**2 + state.spn_consts[0]**2 - 2 * \
                   state.spn_consts[0] * state.rotational_constants()[0]) - \
                   state.spn_consts[1] * rot_qn

        case _:
            return first_term
