# module utils
"""
Contains useful utility functions.
"""


def n2j_qn(n_qn: int, branch_idx: int) -> int:
    """
    Converts from N to J.
    """

    # For Hund's case (b), spin multiplicity 3.
    match branch_idx:
        case 1:
            # F1: J = N + 1
            return n_qn + 1
        case 2:
            # F2: J = N
            return n_qn
        case 3:
            # F3: J = N - 1
            return n_qn - 1
        case _:
            raise ValueError(f"Unknown branch index: {branch_idx}.")
