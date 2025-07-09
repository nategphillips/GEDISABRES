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
from typing import TYPE_CHECKING

import numpy as np
from hamilterm import numerics as num
from numpy.typing import NDArray
from sympy.physics.quantum.cg import CG

if TYPE_CHECKING:
    import sympy as sp


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

            # FIXME: 25/07/09 - Using sympy to numerically compute the Clebsch-Gordan coefficients
            #        is certainly not the most efficient way to do things. Change in the future.
            cg: sp.Expr = CG(
                j_qn_lo,
                omega_basis_lo[m],
                transition_order,
                delta_omega,
                j_qn_up,
                omega_basis_up[n],
            ).doit()

            total += unitary_up[n, branch_idx_up] * cg * unitary_lo[m, branch_idx_lo]

    return abs(total) ** 2 * (2 * j_qn_lo + 1)


def term_value(
    branch_idx: int, unitary: NDArray[np.float64], hamiltonian: NDArray[np.float64]
) -> float:
    """Computes the rotational term value of a rotational line.

    Algorithm based on Equation 21 in Hornkohl, et al.

    Args:
        branch_idx (int): Branch index
        unitary (NDArray[np.float64]): Unitary matrix
        hamiltonian (NDArray[np.float64]): Hamiltonian matrix

    Returns:
        float: The rotational term value
    """
    # FIXME: 25/07/09 - This function is unnecessary and redundant since the eigenvalues for each
    #        Hamiltonian are already computed when np.eigh() is called. Need to figure out how the
    #        eigenvalues obtained using np.eigh() map to the rotational lines.
    total: float = 0.0

    for n in range(unitary.shape[0]):
        for m in range(unitary.shape[0]):
            total += unitary[n, branch_idx] * hamiltonian[n, m] * unitary[m, branch_idx]

    return total


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
    _, unitary_up = np.linalg.eigh(hamiltonian_up)

    # R Branch: J'' = J' - 1
    # Q Branch: J'' = J'
    # P Branch: J'' = J' + 1
    j_qn_lo_list: list[int] = [j_qn_up - 1, j_qn_up, j_qn_up + 1]
    branch_labels: list[str] = ["R", "Q", "P"]

    hamiltonian_lo_list: list[NDArray[np.float64]] = []
    unitary_lo_list: list[NDArray[np.float64]] = []

    for j_qn_lo in j_qn_lo_list:
        hamiltonian_lo: NDArray[np.float64] = num.build_hamiltonian(
            basis_fns_lo, s_qn_lo, j_qn_lo, consts_lo
        )
        _, unitary_lo = np.linalg.eigh(hamiltonian_lo)
        hamiltonian_lo_list.append(hamiltonian_lo)
        unitary_lo_list.append(unitary_lo)

    for branch_idx_up in range(unitary_up.shape[1]):
        # Only needs to be computed once for each upper branch.
        term_value_up: float = term_value(branch_idx_up, unitary_up, hamiltonian_up)

        for branch_idx_lo in range(unitary_lo_list[0].shape[1]):
            for j_qn_lo, hamiltonian_lo, unitary_lo, branch_label in zip(
                j_qn_lo_list, hamiltonian_lo_list, unitary_lo_list, branch_labels
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
                    term_value_lo: float = term_value(branch_idx_lo, unitary_lo, hamiltonian_lo)
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
