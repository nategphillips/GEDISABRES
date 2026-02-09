# module sim.py
"""Contains the implementation of the Sim class."""

# Copyright (C) 2023-2026 Nathan G. Phillips

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

import math
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

import constants
import data_path
import utils
from band import Band
from sim_params import (
    BroadeningBools,
    InstrumentParams,
    LaserParams,
    PlotBools,
    PlotParams,
    ShiftBools,
    ShiftParams,
    TemperatureParams,
)
from sim_props import ConstantsType, SimType

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from line import Line
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
        pressure: float,
        bands_input: list[tuple[int, int]],
        temp_params: TemperatureParams = TemperatureParams(),
        laser_params: LaserParams = LaserParams(),
        inst_params: InstrumentParams = InstrumentParams(),
        shift_params: ShiftParams = ShiftParams(),
        shift_bools: ShiftBools = ShiftBools(),
        broad_bools: BroadeningBools = BroadeningBools(),
        plot_bools: PlotBools = PlotBools(),
        plot_params: PlotParams = PlotParams(),
        pumped_line: Line | None = None,
    ) -> None:
        """Initialize class variables.

        Args:
            sim_type: The type of simulation to perform.
            molecule: Which molecule to simulate.
            state_up: Upper electronic state.
            state_lo: Lower electronic state.
            j_qn_up_max: Maximum J' quantum number.
            pressure: Pressure.
            bands_input: Which vibrational bands to simulate.
            temp_params: Temperature parameters.
            laser_params: Laser parameters.
            inst_params: Instrument broadening parameters.
            shift_params: Line shift parameters.
            shift_bools: Line shift switches.
            broad_bools: Broadening switches.
            plot_bools: Plot switches.
            plot_params: Plot parameters.
            pumped_line: If running a LIF simulation, the absorption line pumped by the laser.
        """
        self.sim_type = sim_type
        self.molecule = molecule
        self.state_up = state_up
        self.state_lo = state_lo
        self.j_qn_up_max = j_qn_up_max
        self.pressure = pressure
        self.bands_input = bands_input
        self.temp_params = temp_params
        self.laser_params = laser_params
        self.inst_params = inst_params
        self.shift_params = shift_params
        self.shift_bools = shift_bools
        self.broad_bools = broad_bools
        self.plot_bools = plot_bools
        self.plot_params = plot_params
        self.pumped_line = pumped_line

    def all_line_data(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Combine the line data for all vibrational bands.

        Returns:
            Rotational lines for all vibrational bands simulated.
        """
        wavenumbers_line = np.array([], dtype=np.float64)
        intensities_line = np.array([], dtype=np.float64)

        for band in self.bands:
            wavenumbers_line = np.concatenate((wavenumbers_line, band.wavenumbers_line()))
            intensities_line = np.concatenate((intensities_line, band.intensities_line()))

        return wavenumbers_line, intensities_line

    def all_cont_data(self, granularity: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Create common axes for superimposing the continuous data of all vibrational bands.

        Args:
            granularity: How many points to place on the x-axis.

        Returns:
            Wavenumbers and intensities for all vibrational bands.
        """
        # NOTE: 25/02/12 - In the case of overlapping lines, the overall absorption coefficient is
        # expressed as a sum over the individual line absorption coefficients. See "Analysis of
        # Collision-Broadened and Overlapped Spectral Lines to Obtain Individual Line Parameters" by
        # BelBruno (1981).

        if self.plot_bools.limits:
            grid_min = utils.wavenum_to_wavelen(self.plot_params.limit_min)
            grid_max = utils.wavenum_to_wavelen(self.plot_params.limit_max)
        else:
            wavenumbers_line = np.concatenate([band.wavenumbers_line() for band in self.bands])
            # A qualitative amount of padding added to either side of the x-axis limits. Ensures
            # that spectral features at either extreme are not clipped when the FWHM parameters are
            # large. The first line's Doppler FWHM is chosen as an arbitrary reference to keep
            # things simple. The minimum Gaussian FWHM allowed is 2 to ensure that no clipping is
            # encountered.
            inst_broadening = max(self.bands[0].lines[0].fwhm_instrument())
            padding = 10.0 * max(inst_broadening, 2.0)

            grid_min = wavenumbers_line.min() - padding
            grid_max = wavenumbers_line.max() + padding

        # Create common wavenumber and intensity grids to hold all of the vibrational band data.
        wavenumbers_cont = np.linspace(grid_min, grid_max, granularity, dtype=np.float64)
        intensities_cont = np.zeros_like(wavenumbers_cont)

        # The wavelength axis is common to all vibrational bands so that their contributions to the
        # spectra can be summed.
        for band in self.bands:
            intensities_cont += band.intensities_cont(wavenumbers_cont)

        return wavenumbers_cont, intensities_cont

    @cached_property
    def predissociation(self) -> dict[str, list[float]]:
        """Return polynomial coefficients for computing predissociation linewidths.

        Returns:
            A dictionary containing polynomial coefficients for predissociation.
        """
        return pl.read_csv(
            data_path.get_data_path(
                "data", self.molecule.name, "predissociation", "lewis_coeffs.csv"
            )
        ).to_dict(as_series=False)

    @cached_property
    def einstein(self) -> NDArray[np.float64]:
        """Return an array of Einstein coefficients for spontaneous emission: A_{v'v''}.

        Rows correspond to the upper state vibrational quantum number (v'), while columns correspond
        to the lower state vibrational quantum number (v'').

        Returns:
            An array of Einstein coefficients for spontaneous emission.
        """
        return np.loadtxt(
            data_path.get_data_path(
                "data",
                self.molecule.name,
                "einstein",
                f"{self.state_up.name}_to_{self.state_lo.name}.csv",
            ),
            delimiter=",",
        )

    @cached_property
    def franck_condon(self) -> NDArray[np.float64]:
        """Return an array of Franck-Condon factors for the associated electronic transition.

        Rows correspond to the upper state vibrational quantum number (v'), while columns correspond
        to the lower state vibrational quantum number (v'').

        Returns:
            An array of Franck-Condon factors.
        """
        return np.loadtxt(
            data_path.get_data_path(
                "data",
                self.molecule.name,
                "franck-condon",
                f"{self.state_up.name}_to_{self.state_lo.name}.csv",
            ),
            delimiter=",",
        )

    @cached_property
    def bands(self) -> list[Band]:
        """Return the selected vibrational bands within the simulation.

        Returns:
            A list of vibrational `Band` objects.
        """
        return [Band(sim=self, v_qn_up=band[0], v_qn_lo=band[1]) for band in self.bands_input]

    @cached_property
    def vib_partition_fn(self) -> float:
        """Return the vibrational partition function Q_v.

        Returns:
            The vibrational partition function Q_v.
        """
        # NOTE: 25/07/21 - If per-vibrational-level data is being used, the maximum vibrational
        #       quantum number is set by the data available. Otherwise, an arbitrary maximum of 20
        #       is set when the Dunham expansion is used.
        match self.sim_type:
            case SimType.EMISSION | SimType.LIF:
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

        q_v = 0.0

        # NOTE: 24/10/22 - The vibrational partition function is always computed using a set number
        #       of vibrational bands to ensure an accurate estimate of the state sum is obtained.
        #       This approach is used to ensure the sum is calculated correctly regardless of the
        #       number of vibrational bands simulated by the user.
        for v_qn in range(0, v_qn_max):
            # NOTE: 24/10/25 - The zero-point vibrational energy is used as a reference to which all
            #       other vibrational energies are measured. This ensures the state sum begins at a
            #       value of 1 when v = 0.
            q_v += math.exp(
                -(state.constants_vqn(v_qn)["G"] - state.constants_vqn(0)["G"])
                * constants.PLANC
                * constants.LIGHT
                / (constants.BOLTZ * self.temp_params.vibrational)
            )

        return q_v

    @cached_property
    def elc_partition_fn(self) -> float:
        """Return the electronic partition function Q_e.

        Returns:
            The electronic partition function Q_e.
        """
        energies = list(constants.ELECTRONIC_ENERGIES[self.molecule.name].values())
        degeneracies = list(constants.ELECTRONIC_DEGENERACIES[self.molecule.name].values())

        q_e = 0.0

        # NOTE: 24/10/25 - This sum is basically unnecessary since the energies of electronic states
        #       above the ground state are so high. This means that any contribution to the
        #       electronic partition function from anything other than the ground state is
        #       negligible.
        for energy, degeneracy in zip(energies, degeneracies):
            q_e += degeneracy * math.exp(
                -energy
                * constants.PLANC
                * constants.LIGHT
                / (constants.BOLTZ * self.temp_params.electronic)
            )

        return q_e

    @cached_property
    def elc_boltz_frac(self) -> tuple[float, float]:
        """Return the electronic Boltzmann fraction N_e / N.

        Returns:
            The electronic Boltzmann fraction N_e / N.
        """
        energies = constants.ELECTRONIC_ENERGIES[self.molecule.name]
        degeneracies = constants.ELECTRONIC_DEGENERACIES[self.molecule.name]

        temperature_factor = (
            constants.PLANC * constants.LIGHT / (constants.BOLTZ * self.temp_params.electronic)
        )

        def boltzmann_fraction(state_name: str) -> float:
            return (
                degeneracies[state_name]
                * math.exp(-energies[state_name] * temperature_factor)
                / self.elc_partition_fn
            )

        electronic_fraction_up = boltzmann_fraction(self.state_up.name)
        electronic_fraction_lo = boltzmann_fraction(self.state_lo.name)

        return electronic_fraction_up, electronic_fraction_lo
