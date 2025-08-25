# module sim.py
"""Contains the implementation of the Sim class."""

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

from functools import cached_property

import numpy as np
import polars as pl
from numpy.typing import NDArray

import constants
import utils
from band import Band
from enums import ConstantsType, SimType
from molecule import Molecule
from state import State


class Sim:
    """Simulate the spectra of a particular molecule."""

    def __init__(
        self,
        sim_type: SimType,
        molecule: Molecule,
        state_up: State,
        state_lo: State,
        j_qn_up_max: int,
        temp_trn: float,
        temp_elc: float,
        temp_vib: float,
        temp_rot: float,
        pressure: float,
        bands_input: list[tuple[int, int]],
        inst_broad_wl_gauss: float = 0.0,
        inst_broad_wl_loren: float = 0.0,
        laser_power_w: float = 0.0,
        beam_diameter_mm: float = 1.0,
        molecule_velocity_ms: float = 0.0,
        coll_shift_a: float = 0.0,
        coll_shift_b: float = 0.0,
        coll_shift: bool = False,
        dopp_shift: bool = False,
    ) -> None:
        """Initialize class variables.

        Args:
            sim_type (SimType): The type of simulation to perform.
            molecule (Molecule): Which molecule to simulate.
            state_up (State): Upper electronic state.
            state_lo (State): Lower electronic state.
            j_qn_up_max (int): Maximum J' quantum number.
            temp_trn (float): Translational temperature.
            temp_elc (float): Electronic temperature.
            temp_vib (float): Vibrational temperature.
            temp_rot (float): Rotational temperature.
            pressure (float): Pressure.
            bands_input (list[tuple[int, int]]): Which vibrational bands to simulate.
        """
        self.sim_type: SimType = sim_type
        self.molecule: Molecule = molecule
        self.state_up: State = state_up
        self.state_lo: State = state_lo
        self.j_qn_up_max: int = j_qn_up_max
        self.temp_trn: float = temp_trn
        self.temp_elc: float = temp_elc
        self.temp_vib: float = temp_vib
        self.temp_rot: float = temp_rot
        self.pressure: float = pressure
        self.bands_input: list[tuple[int, int]] = bands_input
        self.inst_broad_wl_gauss: float = inst_broad_wl_gauss
        self.inst_broad_wl_loren: float = inst_broad_wl_loren
        self.laser_power_w: float = laser_power_w
        self.beam_diameter_mm: float = beam_diameter_mm
        self.molecule_velocity_ms: float = molecule_velocity_ms
        self.coll_shift_a: float = coll_shift_a
        self.coll_shift_b: float = coll_shift_b
        self.coll_shift: bool = coll_shift
        self.dopp_shift: bool = dopp_shift

    def all_line_data(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Combine the line data for all vibrational bands."""
        wavenumbers_line: NDArray[np.float64] = np.array([])
        intensities_line: NDArray[np.float64] = np.array([])

        for band in self.bands:
            wavenumbers_line = np.concatenate((wavenumbers_line, band.wavenumbers_line()))
            intensities_line = np.concatenate((intensities_line, band.intensities_line()))

        return wavenumbers_line, intensities_line

    def all_conv_data(
        self, fwhm_selections: dict[str, bool], granularity: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Create common axes for superimposing the convolved data of all vibrational bands."""
        # NOTE: 25/02/12 - In the case of overlapping lines, the overall absorption coefficient is
        # expressed as a sum over the individual line absorption coefficients. See "Analysis of
        # Collision-Broadened and Overlapped Spectral Lines to Obtain Individual Line Parameters" by
        # BelBruno (1981).

        # The total span of wavenumbers from all bands.
        wavenumbers_line: NDArray[np.float64] = np.concatenate(
            [band.wavenumbers_line() for band in self.bands]
        )

        # A qualitative amount of padding added to either side of the x-axis limits. Ensures that
        # spectral features at either extreme are not clipped when the FWHM parameters are large.
        # The first line's Doppler FWHM is chosen as an arbitrary reference to keep things simple.
        # The minimum Gaussian FWHM allowed is 2 to ensure that no clipping is encountered.
        inst_broadening: float = max(self.bands[0].lines[0].fwhm_instrument(True))
        padding: float = 10.0 * max(inst_broadening, 2)

        grid_min: float = wavenumbers_line.min() - padding
        grid_max: float = wavenumbers_line.max() + padding

        # Create common wavenumber and intensity grids to hold all of the vibrational band data.
        wavenumbers_conv: NDArray[np.float64] = np.linspace(
            grid_min, grid_max, granularity, dtype=np.float64
        )
        intensities_conv: NDArray[np.float64] = np.zeros_like(wavenumbers_conv)

        # The wavelength axis is common to all vibrational bands so that their contributions to the
        # spectra can be summed.
        for band in self.bands:
            intensities_conv += band.intensities_conv(fwhm_selections, wavenumbers_conv)

        return wavenumbers_conv, intensities_conv

    @cached_property
    def predissociation(self) -> dict[str, list[float]]:
        """Return polynomial coefficients for computing predissociation linewidths."""
        return pl.read_csv(
            utils.get_data_path("data", self.molecule.name, "predissociation", "lewis_coeffs.csv")
        ).to_dict(as_series=False)

    @cached_property
    def einstein(self) -> NDArray[np.float64]:
        """Return a table of Einstein coefficients for spontaneous emission: A_{v'v''}.

        Rows correspond to the upper state vibrational quantum number (v'), while columns correspond
        to the lower state vibrational quantum number (v'').
        """
        return np.loadtxt(
            utils.get_data_path(
                "data",
                self.molecule.name,
                "einstein",
                f"{self.state_up.name}_to_{self.state_lo.name}.csv",
            ),
            delimiter=",",
        )

    @cached_property
    def franck_condon(self) -> NDArray[np.float64]:
        """Return a table of Franck-Condon factors for the associated electronic transition.

        Rows correspond to the upper state vibrational quantum number (v'), while columns correspond
        to the lower state vibrational quantum number (v'').
        """
        return np.loadtxt(
            utils.get_data_path(
                "data",
                self.molecule.name,
                "franck-condon",
                f"{self.state_up.name}_to_{self.state_lo.name}.csv",
            ),
            delimiter=",",
        )

    @cached_property
    def bands(self) -> list[Band]:
        """Return the selected vibrational bands within the simulation."""
        return [Band(sim=self, v_qn_up=band[0], v_qn_lo=band[1]) for band in self.bands_input]

    @cached_property
    def vib_partition_fn(self) -> float:
        """Return the vibrational partition function."""
        # NOTE: 25/07/21 - If per-vibrational level data is being used, the maximum vibrational
        #       quantum number is set by the data available. Otherwise, an arbitrary maximum of 20
        #       is set when the Dunham expansion is used.
        match self.sim_type:
            case SimType.EMISSION:
                state = self.state_up
                if state.constants_type == ConstantsType.PERLEVEL:
                    v_qn_max = self.state_up.all_constants.height
                else:
                    v_qn_max = constants.V_QN_MAX_DUNHAM
            case SimType.ABSORPTION:
                state = self.state_lo
                if state.constants_type == ConstantsType.PERLEVEL:
                    v_qn_max = self.state_lo.all_constants.height
                else:
                    v_qn_max = constants.V_QN_MAX_DUNHAM

        q_v: float = 0.0

        # NOTE: 24/10/22 - The vibrational partition function is always computed using a set number
        #       of vibrational bands to ensure an accurate estimate of the state sum is obtained.
        #       This approach is used to ensure the sum is calculated correctly regardless of the
        #       number of vibrational bands simulated by the user.
        for v_qn in range(0, v_qn_max):
            # NOTE: 24/10/25 - The zero-point vibrational energy is used as a reference to which all
            #       other vibrational energies are measured. This ensures the state sum begins at a
            #       value of 1 when v = 0.
            q_v += np.exp(
                -(state.constants_vqn(v_qn)["G"] - state.constants_vqn(0)["G"])
                * constants.PLANC
                * constants.LIGHT
                / (constants.BOLTZ * self.temp_vib)
            )

        return q_v

    @cached_property
    def elc_partition_fn(self) -> float:
        """Return the electronic partition function."""
        energies: list[float] = list(constants.ELECTRONIC_ENERGIES[self.molecule.name].values())
        degeneracies: list[int] = list(
            constants.ELECTRONIC_DEGENERACIES[self.molecule.name].values()
        )

        q_e: float = 0.0

        # NOTE: 24/10/25 - This sum is basically unnecessary since the energies of electronic states
        #       above the ground state are so high. This means that any contribution to the
        #       electronic partition function from anything other than the ground state is
        #       negligible.
        for energy, degeneracy in zip(energies, degeneracies):
            q_e += degeneracy * np.exp(
                -energy * constants.PLANC * constants.LIGHT / (constants.BOLTZ * self.temp_elc)
            )

        return q_e

    @cached_property
    def elc_boltz_frac(self) -> float:
        """Return the electronic Boltzmann fraction N_e / N."""
        match self.sim_type:
            case SimType.EMISSION:
                state = self.state_up.name
            case SimType.ABSORPTION:
                state = self.state_lo.name

        energy: float = constants.ELECTRONIC_ENERGIES[self.molecule.name][state]
        degeneracy: int = constants.ELECTRONIC_DEGENERACIES[self.molecule.name][state]

        return (
            degeneracy
            * np.exp(
                -energy * constants.PLANC * constants.LIGHT / (constants.BOLTZ * self.temp_elc)
            )
            / self.elc_partition_fn
        )
