# module term

import numpy as np

from state import State

def vibrational_term(state: State, vib_qn: int) -> float:
    return state.consts['w_e']   * (vib_qn + 0.5)    - \
           state.consts['we_xe'] * (vib_qn + 0.5)**2 + \
           state.consts['we_ye'] * (vib_qn + 0.5)**3 + \
           state.consts['we_ze'] * (vib_qn + 0.5)**4

def rotational_constants(state: State, vib_qn: int) -> list[float]:
    b_v = state.consts['b_e']                        - \
          state.consts['alph_e'] * (vib_qn + 0.5)    + \
          state.consts['gamm_e'] * (vib_qn + 0.5)**2 + \
          state.consts['delt_e'] * (vib_qn + 0.5)**3

    d_v = state.consts['d_e'] - state.consts['beta_e'] * (vib_qn + 0.5)

    h_v = state.consts['h_e']

    return [b_v, d_v, h_v]

def rotational_term(state: State, vib_qn: int, rot_qn: int, branch_idx: int) -> float:
    b, d, h = rotational_constants(state, vib_qn)

    if state.name in ('b3su', 'x3sg'):
        x = rot_qn * (rot_qn + 1)
        l = state.consts['lamd']
        g = state.consts['gamm']

        match branch_idx:
            case 1:
                return x * b - (x**2 + 4 * x) * d

            case 2:
                return x * b - x**2 * d + x**3 * h

            case 3:
                return (x + 2) * b - (x**2 + 8 * x + 4) * d - 2 * l - g

            case _:
                raise ValueError('Invalid branch index.')

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
                raise ValueError('Invalid branch index.')
