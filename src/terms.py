# module term

from state import State


def vibrational_term(state: State, vib_qn: int) -> float:
    # calculates the vibrational term for the vibrating rotator
    # Herzberg p. 149, eq. (IV, 10)

    return (state.consts['w_e'] * (vib_qn + 0.5)      -
            state.consts['we_xe'] * (vib_qn + 0.5)**2 +
            state.consts['we_ye'] * (vib_qn + 0.5)**3 +
            state.consts['we_ze'] * (vib_qn + 0.5)**4)


def rotational_constants(state: State, vib_qn: int) -> list[float]:
    # calculates the rotational constants for the vibrating rotator
    # Herzberg pp. 107-109, eqs. (III, 117-127)

    b_v = (state.consts['b_e']                        -
           state.consts['alph_e'] * (vib_qn + 0.5)    +
           state.consts['gamm_e'] * (vib_qn + 0.5)**2 +
           state.consts['delt_e'] * (vib_qn + 0.5)**3)

    d_v = state.consts['d_e'] - state.consts['beta_e'] * (vib_qn + 0.5)

    h_v = state.consts['h_e']

    return [b_v, d_v, h_v]


def rotational_term(state: State, vib_qn: int, rot_qn: int, branch_idx: int) -> float:
    # calculates the rotational term value
    # Bergeman, 1972 - The Fine Structure of O2

    b, d, h = rotational_constants(state, vib_qn)

    x    = rot_qn * (rot_qn + 1)
    lamd = state.consts['lamd']
    gamm = state.consts['gamm']

    match branch_idx:
        # F1
        case 1:
            return x * b - (x**2 + 4 * x) * d
        # F2
        case 2:
            return x * b - x**2 * d + x**3 * h
        # F3
        case 3:
            return (x + 2) * b - (x**2 + 8 * x + 4) * d - 2 * lamd - gamm
        case _:
            raise ValueError('Invalid branch index.')
