# module hlf_test
"""Testing direct numerical computation of Hönl-London factors.

The algorithm used is described in "Diatomic Hönl-London factor computer program" by
James O. Hornkohl, Christian G. Parigger, and László Nemes.
"""

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

from fractions import Fraction

import numpy as np
from hamilterm import numerics as num
from numpy.typing import NDArray
from py3nj import clebsch_gordan


class Line:
    """A bare rotational line class for testing."""

    def __init__(
        self,
        branch_idx_up: int,
        branch_idx_lo: int,
        j_qn_up: int,
        j_qn_lo: int,
        branch: str,
        wavenumber: float,
        honl_london_factor: float,
    ) -> None:
        self.branch_idx_up = branch_idx_up
        self.branch_idx_lo = branch_idx_lo
        self.j_qn_up = j_qn_up
        self.j_qn_lo = j_qn_lo
        self.branch = branch
        self.wavenumber = wavenumber
        self.honl_london_factor = honl_london_factor


def honl_london_factor(
    branch_idx_up: int,
    branch_idx_lo: int,
    unitary_up: NDArray[np.float64],
    unitary_lo: NDArray[np.float64],
    j_qn_up: int,
    j_qn_lo: int,
    omega_basis_up: list[Fraction],
    omega_basis_lo: list[Fraction],
    transition_order: int = 1,
) -> float:
    """Computes the Hönl-London factor of a rotational line.

    Algorithm based on Equation 22 in Hornkohl, et al.

    Args:
        branch_idx_up (int): Upper state branch index
        branch_idx_lo (int): Lower state branch index
        unitary_up (NDArray[np.float64]): Upper state unitary matrix
        unitary_lo (NDArray[np.float64]): Lower state unitary matrix
        j_qn_up (int): Upper state rotational quantum number J'
        j_qn_lo (int): Lower state rotational quantum number J''
        omega_basis_up (list[Fraction]): Upper state Ω quantum numbers
        omega_basis_lo (list[Fraction]): Lower state Ω quantum numbers
        transition_order (int): Transition order, 1 for dipole transitions

    Returns:
        float: The Hönl-London factor
    """
    total: float = 0.0

    for n in range(unitary_up.shape[0]):
        for m in range(unitary_lo.shape[0]):
            delta_omega: Fraction = omega_basis_up[n] - omega_basis_lo[m]

            # NOTE: 25/07/09 - This Clebsch-Gordan method comes from the py3nj package
            #       (https://github.com/fujiisoup/py3nj), which requires a Fortran compiler and the
            #       Ninja build system to be installed. On Windows, Quickstart Fortran
            #       (https://github.com/LKedward/quickstart-fortran) installs a MinGW backend along
            #       with GFortran and the Ninja build system. Word of caution: if you have multiple
            #       MinGW or GFortran versions installed, make sure to move the Quickstart Fortran
            #       versions to the top of your PATH, or the build might fail! Linux is more
            #       straightforward, just ensure that GFortran and Ninja are installed via the
            #       appropriate package manager and you're good to go.

            # NOTE: 25/07/09 - Since the values for Λ are always integers, while the values for Σ
            #       can be half-integers, Ω = Λ + Σ is generally a half-integer. The arguments
            #       passed to the CG method are doubled so that half-integer values are properly
            #       handled, see https://py3nj.readthedocs.io/en/master/examples.html for details.
            #       Hornkohl, et al. list the CG coefficient as ⟨J'', Ω''; q, Ω' - Ω''|J', Ω'⟩.
            cg: np.float64 | NDArray[np.float64] = clebsch_gordan(
                int(2 * j_qn_lo),
                int(2 * transition_order),
                int(2 * j_qn_up),
                int(2 * omega_basis_lo[m]),
                int(2 * delta_omega),
                int(2 * omega_basis_up[n]),
                ignore_invalid=True,
            )

            total += unitary_up[n, branch_idx_up] * cg * unitary_lo[m, branch_idx_lo]

    return abs(total) ** 2 * (2 * j_qn_lo + 1)


term_symbol_up: str = "3Sigma"
term_symbol_lo: str = "3Sigma"

consts_up: num.Constants = num.Constants(
    rotational=num.RotationalConsts(B=0.8132, D=4.50e-06),
    spin_spin=num.SpinSpinConsts(lamda=1.69),
    spin_rotation=num.SpinRotationConsts(gamma=-0.028),
)
consts_lo: num.Constants = num.Constants(
    rotational=num.RotationalConsts(B=1.43767603572, D=4.84057450e-6),
    spin_spin=num.SpinSpinConsts(lamda=1.98475142368),
    spin_rotation=num.SpinRotationConsts(gamma=-8.425375759e-3),
)

s_qn_up, lambda_qn_up = num.parse_term_symbol(term_symbol_up)
basis_fns_up: list[tuple[int, Fraction, Fraction]] = num.generate_basis_fns(s_qn_up, lambda_qn_up)
s_qn_lo, lambda_qn_lo = num.parse_term_symbol(term_symbol_lo)
basis_fns_lo: list[tuple[int, Fraction, Fraction]] = num.generate_basis_fns(s_qn_lo, lambda_qn_lo)

omega_basis_up: list[Fraction] = [omega for (_, _, omega) in basis_fns_up]
omega_basis_lo: list[Fraction] = [omega for (_, _, omega) in basis_fns_lo]

j_qn_up_max: int = 10

lines: list[Line] = []

# This loop will occur within a vibrational band, i.e., a class with a defined v' and v''.

for j_qn_up in range(0, j_qn_up_max + 1):
    hamiltonian_up: NDArray[np.float64] = num.build_hamiltonian(
        basis_fns_up, s_qn_up, j_qn_up, consts_up
    )
    eigenvals_up, unitary_up = np.linalg.eigh(hamiltonian_up)

    # R Branch: J'' = J' - 1
    # Q Branch: J'' = J'
    # P Branch: J'' = J' + 1
    j_qn_lo_list: list[int] = [j_qn_up - 1, j_qn_up, j_qn_up + 1]
    branch_labels: list[str] = ["R", "Q", "P"]

    hamiltonian_lo_list: list[NDArray[np.float64]] = []
    unitary_lo_list: list[NDArray[np.float64]] = []
    eigenvals_lo_list: list[NDArray[np.float64]] = []

    for j_qn_lo in j_qn_lo_list:
        hamiltonian_lo: NDArray[np.float64] = num.build_hamiltonian(
            basis_fns_lo, s_qn_lo, j_qn_lo, consts_lo
        )
        eigenvals_lo, unitary_lo = np.linalg.eigh(hamiltonian_lo)
        hamiltonian_lo_list.append(hamiltonian_lo)
        unitary_lo_list.append(unitary_lo)
        eigenvals_lo_list.append(eigenvals_lo)

    for branch_idx_up in range(unitary_up.shape[1]):
        # Only needs to be computed once for each upper branch.
        term_value_up: float = eigenvals_up[branch_idx_up]

        for branch_idx_lo in range(unitary_lo_list[0].shape[1]):
            for j_qn_lo, hamiltonian_lo, unitary_lo, eigenvals_lo, branch_label in zip(
                j_qn_lo_list, hamiltonian_lo_list, unitary_lo_list, eigenvals_lo_list, branch_labels
            ):
                hlf: float = honl_london_factor(
                    branch_idx_up,
                    branch_idx_lo,
                    unitary_up,
                    unitary_lo,
                    j_qn_up,
                    j_qn_lo,
                    omega_basis_up,
                    omega_basis_lo,
                )
                if hlf > 1e-6:
                    term_value_lo: float = eigenvals_lo[branch_idx_lo]
                    wavenumber: float = term_value_up - term_value_lo
                    lines.append(
                        Line(
                            branch_idx_up,
                            branch_idx_lo,
                            j_qn_up,
                            j_qn_lo,
                            branch_label,
                            wavenumber,
                            hlf,
                        )
                    )

print(len(lines))
