# module band
"""
Contains the implementation of the Band class.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

import constants
import convolve
from line import Line
import params
from simtype import SimType
import terms
import utils

if TYPE_CHECKING:
    from sim import Sim


class Band:
    """
    Represents a vibrational band of a particular molecule.
    """

    def __init__(self, sim: Sim, v_qn_up: int, v_qn_lo: int) -> None:
        self.sim: Sim = sim
        self.v_qn_up: int = v_qn_up
        self.v_qn_lo: int = v_qn_lo
        self.band_origin: float = self.get_band_origin()
        self.rot_part: float = self.get_rot_partition_fn()
        self.vib_boltz_frac: float = self.get_vib_boltz_frac()
        self.lines: list[Line] = self.get_lines()

    def wavenumbers_line(self) -> np.ndarray:
        """
        Returns an array of wavenumbers, one for each line.
        """

        return np.array([line.wavenumber for line in self.lines])

    def intensities_line(self) -> np.ndarray:
        """
        Returns an array of intensities, one for each line.
        """

        return np.array([line.intensity for line in self.lines])

    def wavenumbers_conv(self) -> np.ndarray:
        """
        Returns an array of convolved wavenumbers.
        """

        # The individual line wavenumbers are only used to find the minimum and maximum bounds of
        # the spectrum since the spectrum itself is no longer quantized.
        wns_line: np.ndarray = self.wavenumbers_line()

        # Generate a fine-grained x-axis using existing wavenumber data.
        return np.linspace(wns_line.min(), wns_line.max(), params.GRANULARITY)

    def intensities_conv(self) -> np.ndarray:
        """
        Returns an array of convolved intensities.
        """

        return convolve.convolve_brod(self.lines, self.wavenumbers_conv())

    def get_vib_boltz_frac(self) -> float:
        """
        Returns the vibrational Boltzmann fraction N_v / N.
        """

        match self.sim.sim_type:
            case SimType.EMISSION:
                state = self.sim.state_up
                v_qn = self.v_qn_up
            case SimType.ABSORPTION:
                state = self.sim.state_lo
                v_qn = self.v_qn_lo

        # NOTE: 10/25/25 - Calculates the vibrational Boltzmann fraction with respect to the
        #       zero-point vibrational energy to match the vibrational partition function.
        return (
            np.exp(
                -(terms.vibrational_term(state, v_qn) - terms.vibrational_term(state, 0))
                * constants.PLANC
                * constants.LIGHT
                / (constants.BOLTZ * self.sim.temp_vib)
            )
            / self.sim.vib_part
        )

    def get_band_origin(self) -> float:
        """
        Returns the band origin in [1/cm].
        """

        # Herzberg p. 168, eq. (IV, 24)

        upper_state: dict[str, dict[int, float]] = self.sim.state_up.constants
        lower_state: dict[str, dict[int, float]] = self.sim.state_lo.constants

        # NOTE: 11/05/24 - In the Cheung paper, the electronic energy is defined differently than in
        #       Herzberg's book. The conversion specified by Cheung on p. 5 is
        #       nu_0 = T + 2 / 3 * lamda - gamma.
        energy_offset: float = (
            2 / 3 * upper_state["lamda"][self.v_qn_up] - upper_state["gamma"][self.v_qn_up]
        )

        # NOTE: 11/05/24 - The band origin as defined by Herzberg is nu_0 = nu_e + nu_v, and is
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

    def get_rot_partition_fn(self) -> float:
        """
        Returns the rotational partition function.
        """

        # TODO: 10/25/24 - Add nuclear effects to make this the effective rotational partition
        #       function.

        match self.sim.sim_type:
            case SimType.EMISSION:
                state = self.sim.state_up
                v_qn = self.v_qn_up
            case SimType.ABSORPTION:
                state = self.sim.state_lo
                v_qn = self.v_qn_lo

        q_r: float = 0.0

        # NOTE: 10/22/24 - The rotational partition function is always computed using the same
        #       number of lines. At reasonable temperatures (~300 K), only around 50 rotational
        #       lines contribute to the state sum. However, at high temperatures (~3000 K), at least
        #       100 lines need to be considered to obtain an accurate estimate of the state sum.
        #       This approach is used to ensure the sum is calculated correctly regardless of the
        #       number of rotational lines simulated by the user.
        for j_qn in range(201):
            # TODO: 10/22/24 - Not sure which branch index should be used here. The triplet energies
            #       are all close together, so it shouldn't matter too much. Averaging could work,
            #       but I'm not sure if this is necessary.
            q_r += (2 * j_qn + 1) * np.exp(
                -terms.rotational_term(state, v_qn, j_qn, 2)
                * constants.PLANC
                * constants.LIGHT
                / (constants.BOLTZ * self.sim.temp_rot)
            )

        # NOTE: 10/22/24 - Alternatively, the high-temperature approximation can be used instead of
        #       the direct sum approach. This also works well.
        # q_r = constants.BOLTZ * self.sim.temperature / (constants.PLANC * constants.LIGHT * state.constants["B"][v_qn])

        # The state sum must be divided by the symmetry parameter to account for identical
        # rotational orientations in space.
        return q_r / self.sim.molecule.symmetry_param

    def get_lines(self):
        """
        Returns a list of all allowed rotational lines.
        """

        lines = []

        for n_qn_up in self.sim.rot_lvls:
            for n_qn_lo in self.sim.rot_lvls:
                # Ensure the rotational selection rules corresponding to each electronic state are
                # properly followed.
                if self.sim.state_up.is_allowed(n_qn_up) & self.sim.state_lo.is_allowed(n_qn_lo):
                    lines.extend(self.allowed_branches(n_qn_up, n_qn_lo))

        return lines

    def allowed_branches(self, n_qn_up: int, n_qn_lo: int):
        """
        Determines the selection rules for Hund's case (b).
        """

        # For Σ-Σ transitions, the rotational selection rules are ∆N = ±1, ∆N ≠ 0.
        # Herzberg p. 244, eq. (V, 44)

        lines = []

        # Determine how many lines should be present in the fine structure of the molecule due to
        # the effects of spin multiplicity.
        if self.sim.state_up.spin_multiplicity == self.sim.state_lo.spin_multiplicity:
            branch_range = range(1, self.sim.state_up.spin_multiplicity + 1)
        else:
            raise ValueError("Spin multiplicity of the two electronic states do not match.")

        delta_n_qn = n_qn_up - n_qn_lo

        # R branch
        if delta_n_qn == 1:
            lines.extend(self.branch_index(n_qn_up, n_qn_lo, branch_range, "R"))
        # Q branch
        if delta_n_qn == 0:
            # Note that the Q branch doesn't exist for the Schumann-Runge bands of O2.
            lines.extend(self.branch_index(n_qn_up, n_qn_lo, branch_range, "Q"))
        # P branch
        elif delta_n_qn == -1:
            lines.extend(self.branch_index(n_qn_up, n_qn_lo, branch_range, "P"))

        return lines

    def branch_index(self, n_qn_up: int, n_qn_lo: int, branch_range: range, branch_name: str):
        """
        Returns the rotational lines within a given branch.
        """

        def add_line(branch_idx_up: int, branch_idx_lo: int, is_satellite: bool):
            """
            Helper to create and append a rotational line.
            """

            lines.append(
                Line(
                    sim=self.sim,
                    band=self,
                    n_qn_up=n_qn_up,
                    n_qn_lo=n_qn_lo,
                    j_qn_up=utils.n_to_j(n_qn_up, branch_idx_up),
                    j_qn_lo=utils.n_to_j(n_qn_lo, branch_idx_lo),
                    branch_idx_up=branch_idx_up,
                    branch_idx_lo=branch_idx_lo,
                    branch_name=branch_name,
                    is_satellite=is_satellite,
                )
            )

        # Herzberg pp. 249-251, eqs. (V, 48-53)

        # NOTE: 10/16/24 - Every transition has 6 total lines (3 main + 3 satellite) except for the
        #       N' = 0 to N'' = 1 transition, which has 3 total lines (1 main + 2 satellite).

        lines = []

        # Handle the special case where N' = 0 (only the P1, PQ12, and PQ13 lines exist).
        if n_qn_up == 0:
            if branch_name == "P":
                add_line(1, 1, False)
            for branch_idx_lo in (2, 3):
                add_line(1, branch_idx_lo, True)

            return lines

        # Handle regular cases for other N'.
        for branch_idx_up in branch_range:
            for branch_idx_lo in branch_range:
                # Main branches: R1, R2, R3, P1, P2, P3
                if branch_idx_up == branch_idx_lo:
                    add_line(branch_idx_up, branch_idx_lo, False)
                # Satellite branches: RQ31, RQ32, RQ21
                elif (branch_name == "R") and (branch_idx_up > branch_idx_lo):
                    add_line(branch_idx_up, branch_idx_lo, True)
                # Satellite branches: PQ13, PQ23, PQ12
                elif (branch_name == "P") and (branch_idx_up < branch_idx_lo):
                    add_line(branch_idx_up, branch_idx_lo, True)

        return lines
