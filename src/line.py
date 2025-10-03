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
from state import State

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
        # Dynamics of Diatomic Molecules" by Brion, p. 3.
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

    def fwhm_predissociation(self) -> float:
        """Return the predissociation broadening FWHM in [1/cm].

        The predissociation FWHM linewidths are computed using a polynomial fit given in the 1985
        paper "Rotational Variation of Predissociation Linewidths in the Schumann-Runge Bands of O2"
        by B. R. Lewis et al. Homogeneous (Lorentzian).

        Returns:
            float: The predissociation broadening FWHM in [1/cm].
        """
        if self.sim.broad_bools.predissociation:
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

    def fwhm_natural(self) -> float:
        """Return the natural broadening FWHM in [1/cm].

        Homogeneous (Lorentzian).

        Returns:
            float: The natural broadening FWHM in [1/cm].
        """
        if self.sim.broad_bools.natural:
            # The Einstein A coefficient for each rotational line is the product of the vibronic
            # component A^{e'v'}_{e''v''} and the rotational component A^{r'}_{r''}. See SPARK
            # documentation, pg. 19.
            # A_evr = A_ev * A_r = A^{e'v'}_{e''v''} * S^{J'}_{J''} / (2J' + 1)

            # The sum of the Einstein A coefficients for all downward transitions from the upper
            # state.
            sum_spontaneous_emission: float = (
                self.sim.einstein[self.band.v_qn_up].sum()
                * self.honl_london_factor
                / (2.0 * self.j_qn_up + 1.0)
            )

            # Natural broadening in [1/cm].
            return sum_spontaneous_emission / (2.0 * np.pi * constants.LIGHT)

        return 0.0

    def fwhm_collisional(self) -> float:
        """Return the collisional broadening FWHM in [1/cm].

        The collisional FWHM linewidths are computed using Equation 8.18 in the 2016 book
        "Spectroscopy and Optical Diagnostics for Gases" by Ronald K. Hanson et al. Homogeneous
        (Lorentzian).

        Returns:
            float: The collisional broadening FWHM in [1/cm].
        """
        if self.sim.broad_bools.collisional:
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
                * np.sqrt(
                    8
                    / (np.pi * reduced_mass * constants.BOLTZ * self.sim.temp_params.translational)
                )
                / np.pi
            ) / constants.LIGHT

        return 0.0

    def fwhm_doppler(self) -> float:
        """Return the Doppler broadening FWHM in [1/cm].

        The doppler FWHM linewidths are computed using Equation 8.24 in the 2016 book "Spectroscopy
        and Optical Diagnostics for Gases" by Ronald K. Hanson et al. Inhomogeneous (Gaussian).

        Returns:
            float: The Doppler broadening FWHM in [1/cm].
        """
        if self.sim.broad_bools.doppler:
            # Doppler (thermal) broadening in [1/cm]. Note that the speed of light is converted from
            # [cm/s] to [m/s] to ensure that the units work out correctly.
            return self.wavenumber * np.sqrt(
                8
                * constants.BOLTZ
                * self.sim.temp_params.translational
                * np.log(2)
                / (self.sim.molecule.mass * (constants.LIGHT / 1e2) ** 2)
            )

        return 0.0

    def fwhm_instrument(self) -> tuple[float, float]:
        """Return the instrument broadening FWHM in [1/cm].

        The instrument FWHM linewidths are given as inputs from the user in units of [nm], which are
        then converted to units of [1/cm]. Inhomogeneous (Gaussian) & homogeneous (Lorentzian).

        Returns:
            float: The instrument broadening FWHM in [1/cm].
        """
        if self.sim.broad_bools.instrument:
            # NOTE: 25/02/12 - Instrument broadening is passed into this function with units [nm],
            #       so we must convert it to [1/cm]. Note that the FWHM is a bandwidth, so we cannot
            #       simply convert [nm] to [1/cm] in the normal sense - there must be a central
            #       wavelength to expand about.
            wn_gauss: float = utils.bandwidth_wavelen_to_wavenum(
                utils.wavenum_to_wavelen(self.wavenumber), self.sim.inst_params.gauss_fwhm_wl
            )
            wn_loren: float = utils.bandwidth_wavelen_to_wavenum(
                utils.wavenum_to_wavelen(self.wavenumber), self.sim.inst_params.loren_fwhm_wl
            )

            return wn_gauss, wn_loren

        return 0.0, 0.0

    def fwhm_power(self) -> float:
        """Return the power broadening FWHM in [1/cm].

        The power FWHM linewidths are computed using Equation 1.99 in the 2016 book "Spectra of
        Atoms and Molecules, 3rd ed." by Bernath. Homogeneous (Lorentzian).

        Returns:
            float: The power broadening FWHM in [1/cm].
        """
        if self.sim.broad_bools.power:
            # Intensity in [W/m^2]. Convert beam diameter from [mm] to [m].
            intensity: float = self.sim.laser_params.power_w / (
                np.pi * (1e-3 * 0.5 * self.sim.laser_params.beam_diameter_mm) ** 2
            )
            # Intensity of a plane wave in air: I = 0.5 * ε_0 * c * E^2. Convert the speed of light
            # from [cm/s] to [m/s] to ensure units work out.
            electric_field: float = np.sqrt(
                2.0 * intensity / (constants.EPERM * (1e-2 * constants.LIGHT))
            )
            # Convert from [1/s] to [1/cm].
            return utils.freq_to_wavenum(
                constants.DIPOLE_MOMENT[self.sim.molecule.name]
                * electric_field
                / (2.0 * np.pi * constants.PLANC)
            )

        return 0.0

    def fwhm_transit(self) -> float:
        """Return the transit-time broadening FWHM in [1/cm].

        The transit-time FWHM linewidths are computed using Equation 3.63 in the 2008 book "Laser
        Spectroscopy: Volume 1, 4th ed." by Demtröder. Inhomogeneous (Gaussian).

        Returns:
            float: The transit-time broadening FWHM in [1/cm].
        """
        if self.sim.broad_bools.transit:
            # This approximation is only valid for a 90° interaction angle between the molecular
            # beam and the laser. Furthermore, the laser pulse must be Gaussian in order to produce
            # a Gaussian line profile, as used here. A flat-top beam will produce a sinc² line
            # intensity profile instead. See §3.4 in Demtröder for more information.

            # Assume the diameter of the beam is exactly twice the beam waist. Also convert from
            # [mm] to [m].
            beam_waist_m: float = 1e-3 * 0.5 * self.sim.laser_params.beam_diameter_mm

            # Convert from [1/s] to [1/cm].
            return utils.freq_to_wavenum(
                2.0
                * (self.sim.laser_params.molecule_velocity_ms / beam_waist_m)
                * np.sqrt(2.0 * np.log(2))
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
        unshifted_wavenumber: float = self.band.band_origin + (
            self.rot_term_value_up - self.rot_term_value_lo
        )

        if self.sim.shift_bools.collisional and not self.sim.shift_bools.doppler:
            return unshifted_wavenumber + self.shift_collisional()

        if self.sim.shift_bools.doppler and not self.sim.shift_bools.collisional:
            return unshifted_wavenumber + self.shift_doppler(unshifted_wavenumber)

        if self.sim.shift_bools.collisional and self.sim.shift_bools.doppler:
            return (
                unshifted_wavenumber
                + self.shift_collisional()
                + self.shift_doppler(unshifted_wavenumber)
            )

        return unshifted_wavenumber

    def shift_doppler(self, wavenumber: float) -> float:
        """Return the Doppler shift in [1/cm].

        Computed using Eq. 8.40 in the 2016 book "Spectroscopy and Optical Diagnostics for Gases" by
        Ronald K. Hanson et al.

        Returns:
            float: The Doppler shift in [1/cm].
        """
        # Convert the speed of light from [cm/s] to [m/s].
        return wavenumber * self.sim.laser_params.molecule_velocity_ms / (1e-2 * constants.LIGHT)

    def shift_collisional(self) -> float:
        """Return the collisional shift in [1/cm].

        Computed using Eq. 8.39 in the 2016 book "Spectroscopy and Optical Diagnostics for Gases" by
        Ronald K. Hanson et al.

        Returns:
            float: The collisional shift in [1/cm].
        """
        return (
            self.sim.shift_params.collisional_a
            * (self.sim.pressure / 101325.0)
            * (300.0 / self.sim.temp_params.translational) ** self.sim.shift_params.collisional_b
        )

    @cached_property
    def intensity(self) -> float:
        """Return the intensity.

        Returns:
            float: The intensity of the rotational line.
        """
        # The Einstein A coefficient for each rotational line is the product of the vibronic
        # component A^{e'v'}_{e''v''} and the rotational component A^{r'}_{r''}. See SPARK
        # documentation, pg. 19.
        # A_evr = A_ev * A_r = A^{e'v'}_{e''v''} * S^{J'}_{J''} / (2J' + 1)
        spontaneous_emission: float = (
            self.sim.einstein[self.band.v_qn_up][self.band.v_qn_lo]
            * self.honl_london_factor
            / (2.0 * self.j_qn_up + 1.0)
        )

        # Given an input pressure, use the ideal gas law to obtain a total number density.
        # p = N * k * T
        total_number_density: float = self.sim.pressure / (
            constants.BOLTZ * self.sim.temp_params.translational
        )

        # The state number density is given as the product of the individual Boltzmann fractions
        # times the total number density. See SPARK documentation, pg. 58.
        number_density_up: float = (
            total_number_density
            * self.sim.elc_boltz_frac[0]
            * self.band.vib_boltz_frac[0]
            * self.rot_boltz_frac[0]
        )
        number_density_lo: float = (
            total_number_density
            * self.sim.elc_boltz_frac[1]
            * self.band.vib_boltz_frac[1]
            * self.rot_boltz_frac[1]
        )

        # TODO: 25/10/02 - For individual spectral lines, the lineshape function ϕ(ν) is a Dirac
        #       delta function. Since the delta function is normalized as ∫ ϕ(ν) dν = 1, it
        #       technically fits the criteria needed in the emission and absorption coefficients.
        #       Because ϕ(ν_0) = 1 only at the center wavelength of the line and is zero everywhere
        #       else, these formulae *should* be correct. I'm not sure if adding broadening effects
        #       after computing the intensity is correct, however, since the lineshape functions
        #       must be normalized.

        if self.sim.sim_type == SimType.EMISSION:
            # The emission coefficient as given in eq. 1.73 of "Radiative Processes in
            # Astrophysics" by Rybicki and Lightman. It has units [W m^-3 sr^-1 1/cm^-1].
            # j_v = h * c * ν_0 * N_u * A_ul * ϕ(ν) / 4π
            emission_coefficient: float = (
                constants.PLANC
                * constants.LIGHT
                * self.wavenumber
                * number_density_up
                * spontaneous_emission
                / (4.0 * np.pi)
            )

            return emission_coefficient

        stimulated_emission: float = spontaneous_emission / (
            8.0 * np.pi * constants.PLANC * constants.LIGHT * self.wavenumber**3
        )

        # Upper and lower state degeneracies g are the product of the electronic, vibrational, and
        # rotational degeneracies. Since g_v is simply one, it does not contribute to the product.
        # g_evr = g_e * g_v * g_r = (2 - δ_{0,Λ})(2S + 1) * 1 * (2J + 1)
        degeneracy_up: float = constants.ELECTRONIC_DEGENERACIES[self.sim.molecule.name][
            self.sim.state_up.name
        ] * (2.0 * self.j_qn_up + 1.0)
        degeneracy_lo: float = constants.ELECTRONIC_DEGENERACIES[self.sim.molecule.name][
            self.sim.state_lo.name
        ] * (2.0 * self.j_qn_lo + 1.0)

        photon_absorption: float = stimulated_emission * degeneracy_up / degeneracy_lo

        # The absorption coefficient as given in eq. 1.75 of "Radiative Processes in Astrophysics"
        # by Rybicki and Lightman. It has units [m^-1].
        # α_v = h * c * ν_0 * (N_l * B_lu - N_u * B_ul) * ϕ(ν) / 4π
        absorption_coefficient: float = (
            constants.PLANC
            * constants.LIGHT
            * self.wavenumber
            * (number_density_lo * photon_absorption - number_density_up * stimulated_emission)
            / (4.0 * np.pi)
        )

        return absorption_coefficient

    @cached_property
    def rot_boltz_frac(self) -> tuple[float, float]:
        """Return the rotational Boltzmann fraction, N_J / N."""
        temperature_factor: float = (
            constants.PLANC * constants.LIGHT / (constants.BOLTZ * self.sim.temp_params.rotational)
        )

        def degeneracy(state: State, j_qn: float, n_qn: float) -> float:
            # Degeneracies for homonuclear diatomics can be different depending on the evenness of
            # N.
            even_degeneracy, odd_degeneracy = state.nuclear_degeneracy
            is_n_even: bool = n_qn % 2 == 0

            match is_n_even:
                case True:
                    nuclear_degen = even_degeneracy
                case False:
                    nuclear_degen = odd_degeneracy

            # Effective rotational degeneracy, including nuclear effects.
            return (2.0 * j_qn + 1.0) * nuclear_degen

        def boltzmann_fraction(degeneracy: float, rot_term_value: float) -> float:
            return (
                degeneracy
                * np.exp(-rot_term_value * temperature_factor)
                / self.band.rot_partition_fn
            )

        degeneracy_up: float = degeneracy(self.sim.state_up, self.j_qn_up, self.n_qn_up)
        degeneracy_lo: float = degeneracy(self.sim.state_lo, self.j_qn_lo, self.n_qn_lo)

        rotational_fraction_up: float = boltzmann_fraction(degeneracy_up, self.rot_term_value_up)
        rotational_fraction_lo: float = boltzmann_fraction(degeneracy_lo, self.rot_term_value_lo)

        return rotational_fraction_up, rotational_fraction_lo
