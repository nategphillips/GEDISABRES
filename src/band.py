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

from fractions import Fraction
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from hamilterm import constants as hconsts
from hamilterm import numerics
from hamilterm import utils as hutils
from py3nj import clebsch_gordan

import constants
import convolve
import terms
from enums import SimType
from line import Line

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from sim import Sim


def n_values_for_j(j_qn: Fraction, s_qn: Fraction) -> list[Fraction]:
    """Computes the possible values of N for a given J, from J - S to J + S.

    Args:
        j_qn (Fraction): Quantum number J.
        s_qn (Fraction): Quantum number S.

    Returns:
        list[Fraction]: Valid N for the given J.
    """
    lo: Fraction = j_qn - s_qn
    hi: Fraction = j_qn + s_qn

    step_size: Fraction = Fraction(1)

    n_values: list[Fraction] = []
    current: Fraction = lo
    while current <= hi:
        n_values.append(current)
        current += step_size

    return n_values


def nuclear_parity_mask(
    n_qn_vals: list[int], degeneracy_even: Fraction, degeneracy_odd: Fraction
) -> NDArray[np.bool]:
    """Create a mask of allowed rotational quantum numbers using nuclear degeneracies.

    Args:
        n_qn_vals (list[int]): Quantum numbers N.
        degeneracy_even (float): Nuclear degeneracy for even values of N.
        degeneracy_odd (float): Nuclear degeneracy for odd values of N.

    Raises:
        ValueError: If both even and odd nuclear degeneracies are zero.

    Returns:
        NDArray[np.bool]: Allowed rotational quantum numbers N.
    """
    n_qn_arr: NDArray[np.int64] = np.array(n_qn_vals)

    # If even N values are forbidden, return true only for odd values.
    if (degeneracy_even == 0) and (degeneracy_odd != 0):
        return n_qn_arr % 2 == 1
    # If odd N values are forbidden, return true only for even values.
    if (degeneracy_odd == 0) and (degeneracy_even != 0):
        return n_qn_arr % 2 == 0
    # Both nuclear degeneracies being zero within a given electronic state should never happen.
    if (degeneracy_even == 0) and (degeneracy_odd == 0):
        raise ValueError("Nuclear degeneracy is zero for even and odd values of N!")

    return np.ones_like(n_qn_arr, dtype=np.bool)


def honl_london_matrix(
    j_qn_up: int,
    j_qn_lo: int,
    unitary_up: NDArray[np.float64],
    unitary_lo: NDArray[np.float64],
    omega_basis_up: NDArray[np.float64],
    omega_basis_lo: NDArray[np.float64],
    transition_order: int = 1,
) -> NDArray[np.float64]:
    """Computes the Hönl-London factor of a rotational line.

    Algorithm based on Equation 22 in Hornkohl, et al.

    Args:
        i (int): Column index of the upper state unitary matrix.
        j (int): Column index of the lower state unitary matrix.
        unitary_up (NDArray[np.float64]): Upper state unitary matrix.
        unitary_lo (NDArray[np.float64]): Lower state unitary matrix.
        j_qn_up (int): Upper state rotational quantum number J'.
        j_qn_lo (int): Lower state rotational quantum number J''.
        omega_basis_up (NDArray[np.float64]): Upper state Ω' quantum numbers.
        omega_basis_lo (NDArray[np.float64]): Lower state Ω'' quantum numbers.
        transition_order (int, optional): Transition order. Defaults to 1.

    Returns:
        NDArray[np.float64]: The (num_branches_up, num_branches_low) dimensional Hönl-London factor
            matrix corresponding to the given (J', J'') pair.
    """
    # NOTE: 25/07/09 - This Clebsch-Gordan method comes from the py3nj package
    #       (https://github.com/fujiisoup/py3nj), which requires a Fortran compiler and the Ninja
    #       build system to be installed. On Windows, Quickstart Fortran
    #       (https://github.com/LKedward/quickstart-fortran) installs a MinGW backend along with
    #       GFortran and the Ninja build system. Word of caution: if you have multiple MinGW or
    #       GFortran versions installed, make sure to move the Quickstart Fortran versions to the
    #       top of your PATH, or the build might fail! Linux is more straightforward, just ensure
    #       that GFortran and Ninja are installed via the appropriate package manager and you're
    #       good to go.

    two_j1: int = int(2 * j_qn_lo)
    two_j2: int = int(2 * transition_order)
    two_j3: int = int(2 * j_qn_up)

    # (m, 1) matrix containing all lower state Ω'' values
    two_m1: NDArray[np.int64] = (2 * omega_basis_lo).astype(int)[:, None]
    # (1, n) matrix containing all upper state Ω' values
    two_m3: NDArray[np.int64] = (2 * omega_basis_up).astype(int)[None, :]
    # (m, n) matrix containing all (Ω'', Ω') pairs
    two_m2: NDArray[np.int64] = two_m1 - two_m3

    # NOTE: 25/07/09 - Since the values for Λ are always integers, while the values for Σ can be
    #       half-integers, Ω = Λ + Σ is generally a half-integer. The arguments passed to the CG
    #       method are doubled so that half-integer values are properly handled, see
    #       https://py3nj.readthedocs.io/en/master/examples.html for details.

    # Clebsch-Gordan coefficients for all (Ω'', Ω') pairs, has dimension (m, n)
    cg: NDArray = clebsch_gordan(
        two_j1=two_j1,
        two_j2=two_j2,
        two_j3=two_j3,
        two_m1=two_m1,
        two_m2=two_m2,
        two_m3=two_m3,
        ignore_invalid=True,
    )

    # FIXME: 25/07/15 - Hornkohl, et al. list the Clebsch-Gordan coefficient as
    #        ⟨J'', Ω''; q, Ω' - Ω''|J', Ω'⟩, but using this formula does not align with either
    #        experimental data or previous versions of the code. Using either
    #        ⟨J'', Ω''; q, Ω'' - Ω'|J', Ω'⟩ or ⟨J', Ω'; q, Ω' - Ω''|J'', Ω''⟩ (they output exactly
    #        the same values) aligns nearly perfectly with the old code and compares to the
    #        experimental spectra much better, at least in absorption. Not sure if this is a typo in
    #        the paper, or a case of different definitions appearing in different sources.

    # Compute the matrix of all possible transitions for a given (J', J'') pair. This results in an
    # (m, n) dimensional matrix, with one HLF for each (i, j) branch index pair.
    transition_amplitude: NDArray[np.float64] = unitary_up.T @ cg.T @ unitary_lo

    # FIXME: 25/07/15 - In "Spectroscopy of Low Temperature Plasma" by Ochkin (Appendix E), the
    #        Wigner 3j coefficients in the HLFs are defined slightly differently and are multiplied
    #        by (2J' + 1) * (2J'' + 1) for Hund's case (a) transitions. The paper "A comment on
    #        Hönl-London factors" by Hansson has yet another form of the Wigner 3j coefficients that
    #        are used. I need to sort this out eventually.
    return (2 * j_qn_lo + 1) * np.abs(transition_amplitude) ** 2


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

        # This is the effective rotational partition function, i.e., it includes the nuclear
        # partition function.
        theta_r: float = (
            constants.PLANC * constants.LIGHT * state.constants["B"][v_qn] / constants.BOLTZ
        )
        q_r: float = (
            self.sim.temp_rot
            * state.nuclear_partition_fn()
            / (theta_r * self.sim.molecule.symmetry_param)
        )

        return q_r

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

        consts_up: hconsts.NumericConstants = hconsts.NumericConstants(
            rotational=hconsts.RotationalConsts.numeric(B=b_up, D=d_up),
            spin_spin=hconsts.SpinSpinConsts.numeric(lamda=l_up, lambda_D=ld_up),
            spin_rotation=hconsts.SpinRotationConsts.numeric(gamma=g_up, gamma_D=gd_up),
        )
        consts_lo: hconsts.NumericConstants = hconsts.NumericConstants(
            rotational=hconsts.RotationalConsts.numeric(B=b_lo, D=d_lo),
            spin_spin=hconsts.SpinSpinConsts.numeric(lamda=l_lo, lambda_D=ld_lo),
            spin_rotation=hconsts.SpinRotationConsts.numeric(gamma=g_lo, gamma_D=gd_lo),
        )

        s_qn_up, lambda_qn_up = hutils.parse_term_symbol(term_symbol_up)
        basis_fns_up: list[tuple[int, Fraction, Fraction]] = hutils.generate_basis_fns(
            s_qn_up, lambda_qn_up
        )
        s_qn_lo, lambda_qn_lo = hutils.parse_term_symbol(term_symbol_lo)
        basis_fns_lo: list[tuple[int, Fraction, Fraction]] = hutils.generate_basis_fns(
            s_qn_lo, lambda_qn_lo
        )

        omega_basis_up = np.array([omega for (_, _, omega) in basis_fns_up])
        omega_basis_lo = np.array([omega for (_, _, omega) in basis_fns_lo])

        # FIXME: 25/07/10 - Make a simulation take in J' max instead of "rotational levels".
        j_qn_up_max: int = self.sim.rot_lvls.max()
        j_qn_up_min: int = 0

        # NOTE: 25/07/15 - Precomputing the upper state eigenvalues/vectors is somewhat (~14 ms)
        #       faster than computing them inside the main loop. This is somewhat surprising to me
        #       considering this method actually requires somewhat more memory overhead. Some very
        #       rough benchmarks are listed below:
        #
        #                    20x specific bands   | 5x band ranges
        #                    ---------------------|-------------------
        #       inside loop: 0.7735961437225342 s | 0.7343460819937966 s
        #       precomputed: 0.7578609466552735 s | 0.7206905841827392 s
        eigenvals_up_cache: dict[int, NDArray[np.float64]] = {}
        unitary_up_cache: dict[int, NDArray[np.float64]] = {}

        for j_qn_up in range(j_qn_up_min, j_qn_up_max + 1):
            comp_up = numerics.NumericComputation(
                term_symbol_up, consts_up, j_qn_up, max_n_power=4, max_acomm_power=2
            )
            eigenvals_up_cache[j_qn_up] = comp_up.eigenvalues
            unitary_up_cache[j_qn_up] = comp_up.eigenvectors

        # Precompute lower state eigenvalues and eigenvectors since adjacent J' values share 2 out
        # of 3 J'' values. For example, if J' = 1, then J'' = 0, 1, 2; if J' = 2, then J'' = 1, 2, 3
        # and so on.
        eigenvals_lo_cache: dict[int, NDArray[np.float64]] = {}
        unitary_lo_cache: dict[int, NDArray[np.float64]] = {}

        for j_qn_lo in range(j_qn_up_min - 1, j_qn_up_max + 2):
            comp_lo = numerics.NumericComputation(
                term_symbol_lo, consts_lo, j_qn_lo, max_n_power=4, max_acomm_power=2
            )
            eigenvals_lo_cache[j_qn_lo] = comp_lo.eigenvalues
            unitary_lo_cache[j_qn_lo] = comp_lo.eigenvectors

        # In the case of a 3x3 Hamiltonian, the number of branches will be 3. Since the Hamiltonian
        # (and therefore unitary matrices) are always square, either dimension can be used.
        num_branches_up: int = unitary_up_cache[0].shape[0]
        num_branches_lo: int = unitary_lo_cache[0].shape[0]

        # Branch indices should range from 1 to the dimension of the Hamiltonian.
        branch_up_range: range = range(1, num_branches_up + 1)
        branch_lo_range: range = range(1, num_branches_lo + 1)

        # Each (J', J'') pair will have an associated Hönl-London factor matrix with the dimensions
        # (num_branches_up, branch_branches_lo). For example, each (J', J'') pair for a 3x3
        # Hamiltonian will have an associated 3x3 HLF matrix. Each entry within the matrix
        # corresponds to a possible (i, j) combination, where i and j are the upper and lower branch
        # index labels, respectively.
        hlf_matrix_cache: dict[tuple[int, int], NDArray[np.float64]] = {}
        # The allowed values of N'' are cached since each J'' value can be encountered multiple
        # times. For example, J' = 1, J' = 2, and J' = 3 all share J'' = 1.
        n_qn_lo_cache: dict[int, list[int]] = {}
        allowed_n_nq_lo_cache: dict[int, NDArray[np.bool]] = {}

        # R Branch: ΔJ = J' - J'' = +1
        # Q Branch: ΔJ = J' - J'' = 0
        # P Branch: ΔJ = J' - J'' = -1
        branch_names: list[str] = ["R", "Q", "P"]

        degeneracy_up_even, degeneracy_up_odd = self.sim.state_up.nuclear_degeneracy
        degeneracy_lo_even, degeneracy_lo_odd = self.sim.state_lo.nuclear_degeneracy

        lines: list[Line] = []

        # TODO: 25/07/17 - In general, J can be half-integer, meaning iterating over a range like
        #       isn't valid. Can create lists of J' and J'' beforehand and iterate over those
        #       instead.
        for j_qn_up in range(j_qn_up_min, j_qn_up_max + 1):
            # Get all possible N' values for each J'.
            n_qn_up_vals: list[Fraction] = n_values_for_j(Fraction(j_qn_up), s_qn_up)

            # Check if the generated N' values have any zero-valued degeneracies and mask them off
            # if so.
            allowed_n_qn_up: NDArray[np.bool] = nuclear_parity_mask(
                n_qn_up_vals, degeneracy_up_even, degeneracy_up_odd
            )

            # Upper state eigenvalues, dimension (1, num_branches_up).
            eigenvals_up: NDArray[np.float64] = eigenvals_up_cache[j_qn_up]
            # Upper state unitary matrix, dimension (num_branches_up, num_branches_up).
            unitary_up: NDArray[np.float64] = unitary_up_cache[j_qn_up]

            # NOTE: 25/07/10 - From Herzberg p. 169, if Λ = 0 for both electronic states, the Q
            #       branch transition is forbidden. See also Herzberg p. 243 stating that if Ω = 0
            #       for both electronic states, the Q branch transition is forbidden. The
            #       Hönl-London factors should enforce these automatically.

            # Allowed ΔJ = J' - J'' values for dipole transitions are +1, 0, and -1.
            j_qn_lo_list: list[int] = [j_qn_up - 1, j_qn_up, j_qn_up + 1]

            # TODO: 25/07/15 - The branches are labeled with respect to J' and J'', while the old
            #       version of the code labeled the branches with respect to N' and N''. The
            #       differences can be seen on the J' = 1, J'' = 1, N' = 0, N'' = 1 line:
            #
            #            J' | J'' | N' | N'' | Branch |
            #            ---|-----|----|-----|--------|
            #       old: 1  | 1   | 0  | 1   | P_12   | - w.r.t N
            #       new: 1  | 1   | 0  | 1   | Q_12   | - w.r.t J
            #
            #       Consider adding two branch labels: one with respect to J and the other with
            #       respect to N. PGOPHER does this as well.

            for j_qn_lo, branch_name in zip(j_qn_lo_list, branch_names):
                # If J'' has not yet been encountered, compute its allowed N'' values.
                if j_qn_lo not in n_qn_lo_cache:
                    n_qn_lo_cache[j_qn_lo] = n_values_for_j(Fraction(j_qn_lo), s_qn_lo)
                    allowed_n_nq_lo_cache[j_qn_lo] = nuclear_parity_mask(
                        n_qn_lo_cache[j_qn_lo], degeneracy_lo_even, degeneracy_lo_odd
                    )

                # Get all allowed N'' values for each J''.
                n_qn_lo_vals: list[int] = n_qn_lo_cache[j_qn_lo]
                allowed_n_qn_lo: NDArray[np.bool] = allowed_n_nq_lo_cache[j_qn_lo]

                key: tuple[int, int] = (j_qn_up, j_qn_lo)
                # Check if the HLFs for the given (J', J'') pair have already been computed.
                hlf_mat: NDArray[np.float64] | None = hlf_matrix_cache.get(key)

                if hlf_mat is None:
                    # Lower state unitary matrix, dimension (num_branches_lo, num_branches_lo).
                    unitary_lo: NDArray[np.float64] = unitary_lo_cache[j_qn_lo]

                    # Hönl-London factor matrix for the given (J', J'') combination, has dimensions
                    # (num_branches_up, branch_branches_lo).
                    hlf_mat = honl_london_matrix(
                        j_qn_up=j_qn_up,
                        j_qn_lo=j_qn_lo,
                        unitary_up=unitary_up,
                        unitary_lo=unitary_lo,
                        omega_basis_up=omega_basis_up,
                        omega_basis_lo=omega_basis_lo,
                        transition_order=1,
                    )
                    hlf_matrix_cache[key] = hlf_mat

                # Enforce the Hönl-London cutoff and allowed rotational quantum number conditions
                # for each branch index pair (i, j), has dimensions
                # (num_branches_up, branch_branches_lo).
                mask: NDArray[np.bool] = (
                    (hlf_mat > constants.HONL_LONDON_CUTOFF)
                    & allowed_n_qn_up[:, None]
                    & allowed_n_qn_lo[None, :]
                )
                # Get the row and column (i, j) indices where the mask is true.
                i_range, j_range = np.nonzero(mask)

                # Upper state eigenvalues, dimension (1, num_branches_lo).
                eigenvals_lo: NDArray[np.float64] = eigenvals_lo_cache[j_qn_lo]

                for i, j in zip(i_range.tolist(), j_range.tolist()):
                    branch_idx_up: int = branch_up_range[i]
                    branch_idx_lo: int = branch_lo_range[j]
                    n_qn_up: int = n_qn_up_vals[i]
                    n_qn_lo: int = n_qn_lo_vals[j]
                    hlf: float = float(hlf_mat[i, j])
                    eigenval_up: float = float(eigenvals_up[i])
                    eigenval_lo: float = float(eigenvals_lo[j])

                    # TODO: 25/07/15 - This is a placeholder for what should be more strict rules.
                    #       In "PGOPHER: A program for simulating rotational, vibrational and
                    #       electronic spectra" by Colin M. Western, the low quantum number rules
                    #       are stated as: J ≥ |Ω|, N ≥ |Λ|, and J ≥ |N - S|.

                    # Removing this for now so bad quantum numbers are obvious in the table.
                    # if n_qn_up < 0 or n_qn_lo < 0:
                    #     continue

                    is_satellite: bool = branch_idx_up != branch_idx_lo

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
                            rot_term_value_up=eigenval_up,
                            rot_term_value_lo=eigenval_lo,
                        )
                    )

        return lines
