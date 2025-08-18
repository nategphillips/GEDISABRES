# module line.py
"""Contains the implementation of the Line class."""

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

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

import constants
import utils
from enums import SimType

if TYPE_CHECKING:
    from band import Band
    from sim import Sim


class Line:
    """Represents a rotational line within a vibrational band."""

    def __init__(
        self,
        sim: Sim,
        band: Band,
        j_qn_up: float,
        j_qn_lo: float,
        n_qn_up: float,
        n_qn_lo: float,
        branch_idx_up: int,
        branch_idx_lo: int,
        branch_name_j: str,
        branch_name_n: str,
        is_satellite: bool,
        honl_london_factor: float,
        rot_term_value_up: float,
        rot_term_value_lo: float,
    ) -> None:
        """Initialize class variables.

        Args:
            sim (Sim): Parent simulation.
            band (Band): Vibrational band.
            j_qn_up (float): Upper state rotational quantum number J'.
            j_qn_lo (float): Lower state rotational quantum number J''.
            n_qn_up (float): Upper state rotational quantum number N'.
            n_qn_lo (float): Lower state rotational quantum number N''.
            branch_idx_up (int): Upper branch index.
            branch_idx_lo (int): Lower branch index.
            branch_name_j (str): Branch name with respect to ΔJ.
            branch_name_n (str): Branch name with respect to ΔN.
            is_satellite (bool): Whether or not the line is a satellite line.
            honl_london_factor (float): Hönl-London rotational line strength.
            rot_term_value_up (float): Upper state rotational term value.
            rot_term_value_lo (float): Lower state rotational term value.
        """
        self.sim: Sim = sim
        self.band: Band = band
        # NOTE: 25/07/18 - J (and therefore N) can both be half-integer valued, see "The Spectra and
        # Dynamics of Diatomics Molecules" by Brion, p. 3.
        self.j_qn_up: float = j_qn_up
        self.j_qn_lo: float = j_qn_lo
        self.n_qn_up: float = n_qn_up
        self.n_qn_lo: float = n_qn_lo
        self.branch_idx_up: int = branch_idx_up
        self.branch_idx_lo: int = branch_idx_lo
        self.branch_name_j: str = branch_name_j
        self.branch_name_n: str = branch_name_n
        self.is_satellite: bool = is_satellite
        self.honl_london_factor: float = honl_london_factor
        self.rot_term_value_up: float = rot_term_value_up
        self.rot_term_value_lo: float = rot_term_value_lo

    def fwhm_predissociation(self, is_selected: bool) -> float:
        """Return the predissociation broadening FWHM in [1/cm].

        The predissociation FWHM linewidths are computed using a polynomial fit given in the 1985
        paper "Rotational Variation of Predissociation Linewidths in the Schumann-Runge Bands of O2"
        by B. R. Lewis et al. Homogeneous (Lorentzian).

        Args:
            is_selected (bool): True if predissociation broadening should be simulated.

        Returns:
            float: The predissociation broadening FWHM in [1/cm].
        """
        if is_selected:
            # TODO: 24/10/25 - Using the polynomial fit and coefficients described by Lewis, 1986
            #       for the predissociation of all bands for now. The goal is to use experimental
            #       values when available, and use this fit otherwise. The fit is good up to J = 40
            #       and v = 21. Check this to make sure v' and J' should be used even in absorption.

            # FIXME: 25/02/12 - This will break the simulation for some vibrational bands if J
            #        exceeds 40 by too large of a margin.

            a1: float = self.sim.predissociation["a1"][self.band.v_qn_up]
            a2: float = self.sim.predissociation["a2"][self.band.v_qn_up]
            a3: float = self.sim.predissociation["a3"][self.band.v_qn_up]
            a4: float = self.sim.predissociation["a4"][self.band.v_qn_up]
            a5: float = self.sim.predissociation["a5"][self.band.v_qn_up]
            x: float = self.j_qn_up * (self.j_qn_up + 1)

            # Predissociation broadening in [1/cm].
            return a1 + a2 * x + a3 * x**2 + a4 * x**3 + a5 * x**4

        return 0.0

    def fwhm_natural(self, is_selected: bool) -> float:
        """Return the natural broadening FWHM in [1/cm].

        The natural FWHM linewidths are computed using Equation 8.11 in the 2016 book "Spectroscopy
        and Optical Diagnostics for Gases" by Ronald K. Hanson et al. Homogeneous (Lorentzian).

        Args:
            is_selected (bool): True if natural broadening should be simulated.

        Returns:
            float: The natural broadening FWHM in [1/cm].
        """
        if is_selected:
            # TODO: 24/10/21 - Look over this, seems weird still.

            # The sum of the Einstein A coefficients for all downward transitions from the two
            # levels of the transitions i and j.
            i: int = self.band.v_qn_up
            a_ik: float = 0.0
            for k in range(0, i):
                a_ik += self.sim.einstein[i][k]

            j: int = self.band.v_qn_lo
            a_jk: float = 0.0
            for k in range(0, j):
                a_jk += self.sim.einstein[j][k]

            # Natural broadening in [1/cm].
            return ((a_ik + a_jk) / (2 * np.pi)) / constants.LIGHT

        return 0.0

    def fwhm_collisional(self, is_selected: bool) -> float:
        """Return the collisional broadening FWHM in [1/cm].

        The collisional FWHM linewidths are computed using Equation 8.18 in the 2016 book
        "Spectroscopy and Optical Diagnostics for Gases" by Ronald K. Hanson et al. Homogeneous
        (Lorentzian).

        Args:
            is_selected (bool): True if collisional broadening should be simulated.

        Returns:
            float: The collisional broadening FWHM in [1/cm].
        """
        if is_selected:
            # NOTE: 24/11/05 - In most cases, the amount of electronically excited molecules in the
            #       gas is essentially zero, meaning that most molecules are in the ground state.
            #       Therefore, the ground state radius is used to compute the cross-section. An even
            #       more accurate approach would be to multiply the radius in each state by its
            #       Boltzmann fraction and add them together.
            cross_section: float = (
                np.pi
                * (
                    2
                    * constants.INTERNUCLEAR_DISTANCE[self.sim.molecule.name][
                        self.sim.state_lo.name
                    ]
                )
                ** 2
            )

            # NOTE: 24/10/22 - Both the cross-section and reduced mass refer to the interactions
            #       between two molecules, not the two atoms that compose a molecule. For now, only
            #       homogeneous gases are considered, so the diameter and masses of the two
            #       molecules are identical. The internuclear distance is being used as the
            #       effective radius of the molecule. For homogeneous gases, the reduced mass is
            #       just half the molecular mass (remember, this is for molecule-molecule
            #       interactions).
            reduced_mass: float = self.sim.molecule.mass / 2

            # NOTE: 24/11/05 - The translational tempearature is used for collisional and Doppler
            #       broadening since both effects are direct consequences of the thermal velocity of
            #       molecules.

            # Collisional (pressure) broadening in [1/cm].
            return (
                self.sim.pressure
                * cross_section
                * np.sqrt(8 / (np.pi * reduced_mass * constants.BOLTZ * self.sim.temp_trn))
                / np.pi
            ) / constants.LIGHT

        return 0.0

    def fwhm_doppler(self, is_selected: bool) -> float:
        """Return the Doppler broadening FWHM in [1/cm].

        The doppler FWHM linewidths are computed using Equation 8.24 in the 2016 book "Spectroscopy
        and Optical Diagnostics for Gases" by Ronald K. Hanson et al. Inhomogeneous (Gaussian).

        Args:
            is_selected (bool): True if Doppler broadening should be simulated.

        Returns:
            float: The Doppler broadening FWHM in [1/cm].
        """
        if is_selected:
            # Doppler (thermal) broadening in [1/cm]. Note that the speed of light is converted from
            # [cm/s] to [m/s] to ensure that the units work out correctly.
            return self.wavenumber * np.sqrt(
                8
                * constants.BOLTZ
                * self.sim.temp_trn
                * np.log(2)
                / (self.sim.molecule.mass * (constants.LIGHT / 1e2) ** 2)
            )

        return 0.0

    def fwhm_instrument(self, is_selected: bool, inst_broadening_wl: float) -> float:
        """Return the instrument broadening FWHM in [1/cm].

        The instrument FWHM linewidths are given as inputs from the user in units of [nm], which are
        then converted to units of [1/cm]. Inhomogeneous (Gaussian).

        Args:
            is_selected (bool): True if instrument broadening should be simulated.
            inst_broadening_wl (float): Instrument broadening FWHM in [nm].

        Returns:
            float: The instrument broadening FWHM in [1/cm].
        """
        # TODO: 25/08/18 - Allow for Gaussian and Lorentzian FWHMs to be returned.
        if is_selected:
            # NOTE: 25/02/12 - Instrument broadening is passed into this function with units [nm],
            #       so we must convert it to [1/cm]. Note that the FWHM is a bandwidth, so we cannot
            #       simply convert [nm] to [1/cm] in the normal sense - there must be a central
            #       wavelength to expand about.
            return utils.bandwidth_wavelen_to_wavenum(
                utils.wavenum_to_wavelen(self.wavenumber), inst_broadening_wl
            )

        return 0.0

    def fwhm_power(self, is_selected: bool, laser_power_w: float, beam_diameter_mm: float) -> float:
        """Return the power broadening FWHM in [1/cm].

        The power FWHM linewidths are computed using Equation 1.99 in the 2016 book "Spectra of
        Atoms and Molecules, 3rd ed." by Bernath. Homogeneous (Lorentzian).

        Args:
            is_selected (bool): True if power broadening should be simulated.
            laser_power_w (float): Laser power in [W].
            beam_diameter_mm (float): Beam diameter in [mm].

        Returns:
            float: The power broadening FWHM in [1/cm].
        """
        if is_selected:
            # Intensity in [W/m^2]. Convert beam diameter from [mm] to [m].
            intensity: float = laser_power_w / (np.pi * ((beam_diameter_mm / 1e3) / 2.0) ** 2)
            # Intensity of a plane wave in air: I = 0.5 * ε * c * E^2. Convert the speed of light
            # from [cm/s] to [m/s] to ensure units work out.
            electric_field: float = np.sqrt(
                2.0 * intensity / (constants.EPERM * (constants.LIGHT / 1e2))
            )
            # Convert from [1/s] to [1/cm].
            return utils.freq_to_wavenum(
                constants.DIPOLE_MOMENT[self.sim.molecule.name]
                * electric_field
                / (2.0 * np.pi * constants.PLANC)
            )

        return 0.0

    @cached_property
    def wavenumber(self) -> float:
        """Return the wavenumber in [1/cm].

        Returns:
            float: The wavenumber of the rotational line in [1/cm].
        """
        # NOTE: 24/10/18 - Make sure to understand transition structure: Herzberg pp. 149-152, and
        #       pp. 168-169.

        # Herzberg p. 168, eq. (IV, 24)
        return self.band.band_origin + (self.rot_term_value_up - self.rot_term_value_lo)

    @cached_property
    def intensity(self) -> float:
        """Return the intensity.

        Returns:
            float: The intensity of the rotational line.
        """
        # NOTE: 24/10/18 - Before going any further make sure to read Herzberg pp. 20-21,
        #       pp. 126-127, pp. 200-201, and pp. 382-383.

        match self.sim.sim_type:
            case SimType.EMISSION:
                j_qn = self.j_qn_up
                wavenumber_factor = self.wavenumber**4
            case SimType.ABSORPTION:
                j_qn = self.j_qn_lo
                wavenumber_factor = self.wavenumber

        return (
            wavenumber_factor
            * self.rot_boltz_frac
            * self.band.vib_boltz_frac
            * self.sim.elc_boltz_frac
            * self.honl_london_factor
            / (2 * j_qn + 1)
            * self.sim.franck_condon[self.band.v_qn_up][self.band.v_qn_lo]
        )

    @cached_property
    def rot_boltz_frac(self) -> float:
        """Return the rotational Boltzmann fraction, N_J / N.

        Returns:
            float: The vibrational Boltzmann fraction, N_J / N.
        """
        match self.sim.sim_type:
            case SimType.EMISSION:
                state = self.sim.state_up
                j_qn = self.j_qn_up
                n_qn = self.n_qn_up
                rot_term_value = self.rot_term_value_up
            case SimType.ABSORPTION:
                state = self.sim.state_lo
                j_qn = self.j_qn_lo
                n_qn = self.n_qn_lo
                rot_term_value = self.rot_term_value_lo

        # Degeneracies for homonuclear diatomics can be different depending on the evenness of N.
        even_degeneracy, odd_degeneracy = state.nuclear_degeneracy
        is_n_even: bool = n_qn % 2 == 0

        match is_n_even:
            case True:
                nuclear_degen = even_degeneracy
            case False:
                nuclear_degen = odd_degeneracy

        # Effective rotational degeneracy, including nuclear effects.
        degeneracy: float = (2 * j_qn + 1) * nuclear_degen

        return (
            degeneracy
            * np.exp(
                -rot_term_value
                * constants.PLANC
                * constants.LIGHT
                / (constants.BOLTZ * self.sim.temp_rot)
            )
            / self.band.rot_partition_fn
        )
