# module band.py
"""Contains the implementation of the Band class."""

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

from __future__ import annotations

import time
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from hamilterm import numerics
from py3nj import clebsch_gordan

import constants
import convolve
import terms
import utils
from line import Line
from simtype import SimType

if TYPE_CHECKING:
    from fractions import Fraction

    from numpy.typing import NDArray

    from sim import Sim


def honl_london_factor(
    i: int,
    j: int,
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
        i (int): Column index of the upper state unitary matrix.
        j (int): Column index of the lower state unitary matrix.
        unitary_up (NDArray[np.float64]): Upper state unitary matrix.
        unitary_lo (NDArray[np.float64]): Lower state unitary matrix.
        j_qn_up (int): Upper state rotational quantum number J'.
        j_qn_lo (int): Lower state rotational quantum number J''.
        omega_basis_up (list[Fraction]): Upper state Ω quantum numbers.
        omega_basis_lo (list[Fraction]): Lower state Ω quantum numbers.
        transition_order (int, optional): Transition order. Defaults to 1.

    Returns:
        float: The Hönl-London factor.
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
            # FIXME: 25/07/10 - As written, this expression might only be valid for emission.
            #        Need to compare line strengths with the algebraic HLFs used in previous
            #        versions of the code.
            cg: np.float64 | NDArray[np.float64] = clebsch_gordan(
                two_j1=int(2 * j_qn_lo),
                two_j2=int(2 * transition_order),
                two_j3=int(2 * j_qn_up),
                two_m1=int(2 * omega_basis_lo[m]),
                two_m2=int(2 * delta_omega),
                two_m3=int(2 * omega_basis_up[n]),
                ignore_invalid=True,
            )

            total += unitary_up[n, i] * cg * unitary_lo[m, j]

    return abs(total) ** 2 * (2 * j_qn_lo + 1)


class Band:
    """Represents a vibrational band of a particular molecule."""

    def __init__(self, sim: Sim, v_qn_up: int, v_qn_lo: int) -> None:
        """Initialize class variables.

        Args:
            sim (Sim): Parent simulation.
            v_qn_up (int): Upper vibrational quantum number v'.
            v_qn_lo (int): Lower vibrational quantum number v''.
        """
        self.sim: Sim = sim
        self.v_qn_up: int = v_qn_up
        self.v_qn_lo: int = v_qn_lo

    def wavenumbers_line(self) -> NDArray[np.float64]:
        """Return an array of wavenumbers, one for each line.

        Returns:
            NDArray[np.float64]: All discrete rotational line wavenumbers belonging to the
                vibrational band.
        """
        return np.array([line.wavenumber for line in self.lines])

    def intensities_line(self) -> NDArray[np.float64]:
        """Return an array of intensities, one for each line.

        Returns:
            NDArray[np.float64]: All rotational line intensities belonging to the vibrational band.
        """
        return np.array([line.intensity for line in self.lines])

    def wavenumbers_conv(self, inst_broadening_wl: float, granularity: int) -> NDArray[np.float64]:
        """Return an array of convolved wavenumbers.

        Args:
            inst_broadening_wl (float): Instrument broadening FWHM in [nm].
            granularity (int): Number of points on the wavenumber axis.

        Returns:
            NDArray[np.float64]: A continuous range of wavenumbers.
        """
        # A qualitative amount of padding added to either side of the x-axis limits. Ensures that
        # spectral features at either extreme are not clipped when the FWHM parameters are large.
        # The first line's instrument FWHM is chosen as an arbitrary reference to keep things
        # simple. The minimum Gaussian FWHM allowed is 2 to ensure that no clipping is encountered.
        padding: float = 10.0 * max(self.lines[0].fwhm_instrument(True, inst_broadening_wl), 2)

        # The individual line wavenumbers are only used to find the minimum and maximum bounds of
        # the spectrum since the spectrum itself is no longer quantized.
        wns_line: NDArray[np.float64] = self.wavenumbers_line()

        # Generate a fine-grained x-axis using existing wavenumber data.
        return np.linspace(wns_line.min() - padding, wns_line.max() + padding, granularity)

    def intensities_conv(
        self,
        fwhm_selections: dict[str, bool],
        inst_broadening_wl: float,
        wavenumbers_conv: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return an array of convolved intensities.

        Args:
            fwhm_selections (dict[str, bool]): The types of broadening to be simulated.
            inst_broadening_wl (float): Instrument broadening FWHM in [nm].
            wavenumbers_conv (NDArray[np.float64]): The convolved wavelengths to use.

        Returns:
            NDArray[np.float64]: A continuous range of intensities.
        """
        return convolve.convolve(
            self.lines,
            wavenumbers_conv,
            fwhm_selections,
            inst_broadening_wl,
        )

    @cached_property
    def vib_boltz_frac(self) -> float:
        """Return the vibrational Boltzmann fraction N_v / N.

        Returns:
            float: The vibrational Boltzmann fraction, N_v / N.
        """
        match self.sim.sim_type:
            case SimType.EMISSION:
                state = self.sim.state_up
                v_qn = self.v_qn_up
            case SimType.ABSORPTION:
                state = self.sim.state_lo
                v_qn = self.v_qn_lo

        # NOTE: 24/10/25 - Calculates the vibrational Boltzmann fraction with respect to the
        #       zero-point vibrational energy to match the vibrational partition function.
        return (
            np.exp(
                -(terms.vibrational_term(state, v_qn) - terms.vibrational_term(state, 0))
                * constants.PLANC
                * constants.LIGHT
                / (constants.BOLTZ * self.sim.temp_vib)
            )
            / self.sim.vib_partition_fn
        )

    @cached_property
    def band_origin(self) -> float:
        """Return the band origin in [1/cm].

        Returns:
            float: The band origin in [1/cm].
        """
        # Herzberg p. 168, eq. (IV, 24)

        upper_state: dict[str, list[float]] = self.sim.state_up.constants
        lower_state: dict[str, list[float]] = self.sim.state_lo.constants

        # NOTE: 24/11/05 - In the Cheung paper, the electronic energy is defined differently than in
        #       Herzberg's book. The conversion specified by Cheung on p. 5 is
        #       nu_0 = T + 2 / 3 * lamda - gamma.
        energy_offset: float = (
            2 / 3 * upper_state["lamda"][self.v_qn_up] - upper_state["gamma"][self.v_qn_up]
        )

        # NOTE: 24/11/05 - The band origin as defined by Herzberg is nu_0 = nu_e + nu_v, and is
        #       different for each vibrational transition. The T values in Cheung include the
        #       vibrational term for each level, i.e. T = T_e + G. The ground state has no
        #       electronic energy, so it is not subtracted. In Cheung's data, the term values
        #       provided are measured above the zeroth vibrational level of the ground state. This
        #       means that the lower state zero-point vibrational energy must be used.
        return (
            upper_state["T"][self.v_qn_up]
            + energy_offset
            - (lower_state["G"][self.v_qn_lo] - lower_state["G"][0])
        )

    @cached_property
    def rot_partition_fn(self) -> float:
        """Return the rotational partition function, Q_r.

        The rotational partition function is computed using the high-temperature approximation,
        given by Equation 2.21 in the 2016 book "Spectroscopy and Optical Diagnostics for Gases" by
        Ronald K. Hanson et al.

        Returns:
            float: The rotational partition function, Q_r.
        """
        # TODO: 24/10/25 - Add nuclear effects to make this the effective rotational partition
        #       function.

        match self.sim.sim_type:
            case SimType.EMISSION:
                state = self.sim.state_up
                v_qn = self.v_qn_up
            case SimType.ABSORPTION:
                state = self.sim.state_lo
                v_qn = self.v_qn_lo

        # TODO: 25/07/10 - For now, use the high-temperature approximation instead of directly
        #       computing the sum. Now that rotational term values are directly associated with
        #       rotational lines instead of being computed separately, computing the sum would
        #       require a bit more logic, and I'm not sure it's worth it.
        q_r = (
            constants.BOLTZ
            * self.sim.temp_rot
            / (constants.PLANC * constants.LIGHT * state.constants["B"][v_qn])
        )

        # The state sum must be divided by the symmetry parameter to account for identical
        # rotational orientations in space.
        return q_r / self.sim.molecule.symmetry_param

    @cached_property
    def lines(self) -> list[Line]:
        """Return a list of all allowed rotational lines.

        Returns:
            list[Line]: A list of all allowed `Line` objects for the given selection rules.
        """
        # FIXME: 25/07/11 - The lines being generated are not correct when comparing the rotational
        #        quantum numbers with previous versions of the code. Not sure if this is because
        #        we're now iterating over J instead of N. Try implementing previous hard-coded
        #        selection rules verbatim to recreate the correct lines, then check Honl-London
        #        factors to narrow down the problem

        start_time: float = time.time()

        # FIXME: 25/07/10 - Use the State class to automatically pull the correct term symbols for
        #        the molecule in question.
        term_symbol_up: str = "3Sigma"
        term_symbol_lo: str = "3Sigma"

        # TODO: 25/07/10 - There's certainly a better way to get the constants from the State
        #       class into the form used in Hamilterm.
        table_up: dict[str, list[float]] = self.sim.state_up.constants
        table_lo: dict[str, list[float]] = self.sim.state_lo.constants

        b_up: float = table_up["B"][self.v_qn_up]
        d_up: float = table_up["D"][self.v_qn_up]
        l_up: float = table_up["lamda"][self.v_qn_up]
        g_up: float = table_up["gamma"][self.v_qn_up]
        ld_up: float = table_up["lamda_D"][self.v_qn_up]
        gd_up: float = table_up["gamma_D"][self.v_qn_up]

        b_lo: float = table_lo["B"][self.v_qn_lo]
        d_lo: float = table_lo["D"][self.v_qn_lo]
        l_lo: float = table_lo["lamda"][self.v_qn_lo]
        g_lo: float = table_lo["gamma"][self.v_qn_lo]
        ld_lo: float = table_lo["lamda_D"][self.v_qn_lo]
        gd_lo: float = table_lo["gamma_D"][self.v_qn_lo]

        # NOTE: 24/11/05 - The Hamiltonians in Cheung and Yu are defined slightly differently, which
        #       leads to some constants having different values. Since the Cheung Hamiltonian matrix
        #       elements are used to solve for the energy eigenvalues, the constants from Yu are
        #       changed to fit the convention used by Cheung. See the table below for details.
        #
        #       Cheung  | Yu
        #       --------|------------
        #       D       | -D
        #       lamda_D | 2 * lamda_D
        #       gamma_D | 2 * gamma_D
        # TODO: 25/07/10 - At some point, notation used by pyGEONOSIS should be standardized (it
        #       already mostly is) such that the constants supplied must follow the correct form.
        #       This case in particular makes the issues obvious (i.e. hardcoding a workaround).

        if self.sim.state_lo.name == "X3Sg-":
            d_lo *= -1
            ld_lo *= 2
            gd_lo *= 2

        consts_up: numerics.Constants = numerics.Constants(
            rotational=numerics.RotationalConsts(B=b_up, D=d_up),
            spin_spin=numerics.SpinSpinConsts(lamda=l_up, lambda_D=ld_up),
            spin_rotation=numerics.SpinRotationConsts(gamma=g_up, gamma_D=gd_up),
        )
        consts_lo: numerics.Constants = numerics.Constants(
            rotational=numerics.RotationalConsts(B=b_lo, D=d_lo),
            spin_spin=numerics.SpinSpinConsts(lamda=l_lo, lambda_D=ld_lo),
            spin_rotation=numerics.SpinRotationConsts(gamma=g_lo, gamma_D=gd_lo),
        )

        s_qn_up, lambda_qn_up = numerics.parse_term_symbol(term_symbol_up)
        basis_fns_up: list[tuple[int, Fraction, Fraction]] = numerics.generate_basis_fns(
            s_qn_up, lambda_qn_up
        )
        s_qn_lo, lambda_qn_lo = numerics.parse_term_symbol(term_symbol_lo)
        basis_fns_lo: list[tuple[int, Fraction, Fraction]] = numerics.generate_basis_fns(
            s_qn_lo, lambda_qn_lo
        )

        omega_basis_up: list[Fraction] = [omega for (_, _, omega) in basis_fns_up]
        omega_basis_lo: list[Fraction] = [omega for (_, _, omega) in basis_fns_lo]

        # FIXME: 25/07/10 - Make a simulation take in J' max instead of "rotational levels".
        j_qn_up_max: int = self.sim.rot_lvls.max()

        lines: list[Line] = []

        for j_qn_up in range(0, j_qn_up_max + 1):
            hamiltonian_up: NDArray[np.float64] = numerics.build_hamiltonian(
                basis_fns_up, s_qn_up, j_qn_up, consts_up
            )
            eigenvals_up, unitary_up = np.linalg.eigh(hamiltonian_up)

            # NOTE: 25/07/10 - From Herzberg p. 169, if Λ = 0 for both electronic states, the Q
            #       branch transition is forbidden.
            # TODO: 25/07/10 - See also Herzberg p. 243 stating that if Ω = 0 for both electronic
            #       states, the Q branch transition is forbidden.
            # FIXME: 25/07/10 - Implement parity calculations and see if these rules are enforced
            #        automatically
            # R Branch: J'' = J' - 1
            # Q Branch: J'' = J'
            # P Branch: J'' = J' + 1
            if lambda_qn_up == lambda_qn_lo:
                j_qn_lo_list: list[int] = [j_qn_up - 1, j_qn_up + 1]
                branch_names: list[str] = ["R", "P"]
            else:
                j_qn_lo_list = [j_qn_up - 1, j_qn_up, j_qn_up + 1]
                branch_names = ["R", "Q", "P"]

            hamiltonian_lo_list: list[NDArray[np.float64]] = []
            unitary_lo_list: list[NDArray[np.float64]] = []
            eigenvals_lo_list: list[NDArray[np.float64]] = []

            for j_qn_lo in j_qn_lo_list:
                hamiltonian_lo: NDArray[np.float64] = numerics.build_hamiltonian(
                    basis_fns_lo, s_qn_lo, j_qn_lo, consts_lo
                )
                eigenvals_lo, unitary_lo = np.linalg.eigh(hamiltonian_lo)
                hamiltonian_lo_list.append(hamiltonian_lo)
                unitary_lo_list.append(unitary_lo)
                eigenvals_lo_list.append(eigenvals_lo)

            for i in range(unitary_up.shape[1]):
                # Only needs to be computed once for each upper branch.
                rot_term_value_up: float = eigenvals_up[i]

                # All the unitary matrices within a given electronic state will have the same
                # dimensions since the Hamiltonian is of a fixed dimension for said state.
                for j in range(unitary_lo_list[0].shape[1]):
                    # NOTE: 25/07/10 - The dimensions of the unitary matrices determine how many
                    #       branches exist for the given transition, but they are zero-indexed.
                    #       Standard spectroscopic notation gives branch indices as one-indexed
                    #       values, so we make that change here.
                    branch_idx_up: int = i + 1
                    branch_idx_lo: int = j + 1

                    n_qn_up = utils.j_to_n(j_qn_up, branch_idx_up)
                    n_qn_lo = utils.j_to_n(j_qn_lo, branch_idx_lo)

                    # Ensure the rotational selection rules corresponding to each electronic state
                    # are properly followed. In this case, the oxygen nucleus has zero nuclear spin
                    # angular momentum, meaning symmetry considerations demand that N may only have
                    # odd values.
                    # FIXME: 25/07/10 - Implement parity calculations and see if this rule is
                    #        enforced automatically.
                    if self.sim.state_up.is_allowed(n_qn_up) & self.sim.state_lo.is_allowed(
                        n_qn_lo
                    ):
                        for j_qn_lo, hamiltonian_lo, unitary_lo, eigenvals_lo, branch_name in zip(
                            j_qn_lo_list,
                            hamiltonian_lo_list,
                            unitary_lo_list,
                            eigenvals_lo_list,
                            branch_names,
                        ):
                            hlf: float = honl_london_factor(
                                i=i,
                                j=j,
                                unitary_up=unitary_up,
                                unitary_lo=unitary_lo,
                                j_qn_up=j_qn_up,
                                j_qn_lo=j_qn_lo,
                                omega_basis_up=omega_basis_up,
                                omega_basis_lo=omega_basis_lo,
                            )
                            if hlf > constants.HONL_LONDON_CUTOFF:
                                rot_term_value_lo: float = eigenvals_lo[j]

                                # Denote satellite branches for use in plotting.
                                is_satellite = False
                                if branch_idx_up != branch_idx_lo:
                                    is_satellite = True

                                lines.append(
                                    Line(
                                        sim=self.sim,
                                        band=self,
                                        j_qn_up=j_qn_up,
                                        j_qn_lo=j_qn_lo,
                                        n_qn_up=n_qn_up,
                                        n_qn_lo=n_qn_lo,
                                        branch_idx_up=branch_idx_up,
                                        branch_idx_lo=branch_idx_lo,
                                        branch_name=branch_name,
                                        is_satellite=is_satellite,
                                        honl_london_factor=hlf,
                                        rot_term_value_up=rot_term_value_up,
                                        rot_term_value_lo=rot_term_value_lo,
                                    )
                                )

        print(f"Time to compute lines: {time.time() - start_time} s")
        return lines
