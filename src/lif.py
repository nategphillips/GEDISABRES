# module lif.py
"""A general three-level LIF model."""

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
from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

import matplotlib.pyplot as plt
import numpy as np
import scipy as sy

import constants
import utils
from atom import Atom
from molecule import Molecule
from sim import Sim
from sim_params import BroadeningBools, InstrumentParams, TemperatureParams
from sim_props import ConstantsType, InversionSymmetry, ReflectionSymmetry, SimType, TermSymbol
from state import State

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.integrate._ivp.ivp import OdeResult

    from line import Line

N_FLUENCE = 100


@dataclass
class RateParams:
    """Holds parameters related to the rate equations.

    Attributes:
        a_21: Einstein coefficient for spontaneous emission in [1/s].
        b_12: Einstein coefficient for photon absorption in [cm^2/(J*s)].
        b_21: Einstein coefficient for stimulated emission in [cm^2/(J*s)].
        w_c: Collisional repopulation rate of the lower state in [1/s].
        w_d: Predissociation rate in [1/s].
        w_f: Fluorescent radiative decay rate in [1/s].
        w_q: Quenching rate of the upper state in [1/s].
    """

    a_21: float
    b_12: float
    b_21: float
    w_c: float
    w_d: float
    w_f: float
    w_q: float


@dataclass
class LIFLaserParams:
    """Holds parameters related to the laser.

    Attributes:
        pulse_center: Center of the laser pulse in [s].
        pulse_width: FWHM of the laser pulse in [s].
        fluence: Laser energy per unit area in [J/cm^2].
    """

    pulse_center: float
    pulse_width: float
    fluence: float


@overload
def laser_intensity(t: float, laser_params: LIFLaserParams) -> float: ...


@overload
def laser_intensity(
    t: NDArray[np.float64], laser_params: LIFLaserParams
) -> NDArray[np.float64]: ...


def laser_intensity(
    t: float | NDArray[np.float64], laser_params: LIFLaserParams
) -> float | NDArray[np.float64]:
    """Return the laser intensity for a given time point or for an array of time points.

    Calculates the laser intensity based on provided laser parameters. When given a scalar time
    input, it returns a scalar intensity value. When given an array of time points, it returns an
    array of intensity values. Computed using Eq. (7) in "A 3-Level Model for O2 LIF" by Diskin -
    <https://doi.org/10.2514/6.1996-1991>.

    Args:
        t: Time point(s) at which to calculate the intensity in [s].
        laser_params: Parameters defining the laser beam properties.

    Returns:
        Laser intensity at the specified time point(s) in [J/(cm^2 * s)].
    """
    return (
        laser_params.fluence
        / laser_params.pulse_width
        * math.sqrt(4.0 * math.log(2.0) / np.pi)
        * np.exp(
            -4.0 * math.log(2.0) * ((t - laser_params.pulse_center) / laser_params.pulse_width) ** 2
        )
    )


def time_independent_rates(emission_sim: Sim, pumped_sim: Sim, pumped_line: Line) -> RateParams:
    """Return the time-independent rate parameters.

    Args:
        emission_sim: The simulation containing the LIF emission lines.
        pumped_sim: The parent simulation.
        pumped_line: The rotational absorption line pumped by the laser.

    Returns:
        Parameters related to the rate equations for the selected rotational line.
    """
    g_u = constants.ELECTRONIC_DEGENERACIES[pumped_sim.molecule.name][pumped_sim.state_up.name]
    g_l = constants.ELECTRONIC_DEGENERACIES[pumped_sim.molecule.name][pumped_sim.state_lo.name]

    # Only a single vibrational band will be simulated at a time.
    v_qn_up = pumped_sim.bands[0].v_qn_up
    v_qn_lo = pumped_sim.bands[0].v_qn_lo

    j_qn_up = pumped_line.j_qn_up
    s_j = pumped_line.honl_london_factor

    # Einstein coefficient for spontaneous emission in [1/s].
    a_21: float = pumped_sim.einstein[v_qn_up, v_qn_lo] * s_j / (2.0 * j_qn_up + 1.0)

    # Fluorescent radiative decay rate in [1/s].
    w_f = compute_wf_from_lines(emission_sim, pumped_line.band.v_qn_lo)

    # Einstein coefficient for photon absorption in [cm^2/(J*s)], Herzberg Eq. (I, 56).
    b_12 = (
        a_21
        / (8.0 * np.pi * constants.PLANC * constants.LIGHT * pumped_line.wavenumber**3)
        * g_u
        / g_l
    )

    # Einstein coefficient for stimulated emission in [cm^2/(J*s)].
    b_21 = b_12 * g_l / g_u

    # Predissociation rate in [1/s].
    w_d = 2.0 * np.pi * constants.LIGHT * pumped_line.fwhm_predissociation()

    # NOTE: 26/01/30 - This is a really naïve way of calculating both the upper state quenching and
    #       collisional bath terms, but it's good enough for a first estimation. Further refinement
    #       will be dependent on the molecule used, so this is a general, if simple, approach.

    # Collisional repopulation of the lower state in [1/s].
    w_c = 2.0 * np.pi * constants.LIGHT * pumped_line.fwhm_collisional()
    # Quenching rate of the upper state in [1/s].
    w_q = w_c

    return RateParams(a_21, b_12, b_21, w_c, w_d, w_f, w_q)


def compute_wf_from_lines(emission_sim: Sim, exclude_resonant_vlo: int | None = None) -> float:
    """Computes the fluorescent radiative decay using the LIF emission lines.

    Args:
        emission_sim: The simulation containing the LIF emission lines.
        exclude_resonant_vlo: Which lower state v'' to exclude from the total fluorescence, if any.

    Returns:
        The fluorescent radiative decay rate in [1/s]/
    """
    w_f = 0.0

    # The sum of all downward radiative transitions A_{ul} where u = v', minus the resonant
    # contribution where l = v'.
    for band in emission_sim.bands:
        if exclude_resonant_vlo is not None and band.v_qn_lo == exclude_resonant_vlo:
            continue

        for line in band.lines:
            spontaneous_emission = (
                emission_sim.einstein[band.v_qn_up, band.v_qn_lo]
                * line.honl_london_factor
                / (2.0 * line.j_qn_up + 1.0)
            )
            w_f += spontaneous_emission

    return w_f


def rate_equations(
    t: float,
    n_hat: NDArray[np.float64],
    rate_params: RateParams,
    laser_params: LIFLaserParams,
    pumped_line: Line,
) -> list[float]:
    """Return the rate equations governing the LIF system.

    Args:
        t: Current time in [s].
        n_hat: Nondimensional population densities of N1, N2, and N3 at a point in time.
        rate_params: Rate parameters and Einstein coefficients for the system.
        laser_params: Laser parameters.
        pumped_line: The rotational absorption line pumped by the laser.

    Returns:
        Differential equations dN1/dt, dN2/dt, and dN3/dt.
    """
    n1_hat, n2_hat, n3_hat = n_hat

    # TODO: 24/10/29 - Implement the overlap integral between the transition and laser lineshapes.

    # Laser and rotational absorption feature overlap integral in [cm].
    overlap_integral = 0.8

    # Laser intensity in [J/(cm^2 * s)].
    i_l = laser_intensity(t, laser_params)

    # Laser-stimulated absorption in [1/s].
    w_la = i_l * rate_params.b_12 * overlap_integral / constants.LIGHT
    # Laser-stimulated emission in [1/s].
    w_le = i_l * rate_params.b_21 * overlap_integral / constants.LIGHT

    # Lower state rotational Boltzmann fraction.
    f_b = pumped_line.rot_boltz_frac[1]

    # Normalized rate equations from "A 3-Level Model for O2 LIF" by Diskin; see the `simulate`
    # function for the normalization used.
    dn1_dt: float = (
        -(w_la + rate_params.w_c) * n1_hat
        + (w_le + rate_params.a_21) * n2_hat
        + rate_params.w_c * n3_hat
    )
    dn2_dt: float = (
        w_la * n1_hat
        - (w_le + rate_params.w_d + rate_params.a_21 + rate_params.w_f + rate_params.w_q) * n2_hat
    )
    dn3_dt: float = (f_b / (1.0 - f_b)) * rate_params.w_c * (n1_hat - n3_hat)

    return [dn1_dt, dn2_dt, dn3_dt]


def simulate(
    rate_params: RateParams,
    laser_params: LIFLaserParams,
    pumped_line: Line,
    t_eval: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the normalized population densities of the three states as functions of time.

    Args:
        rate_params: Rate parameters.
        laser_params: Laser parameters.
        pumped_line: The rotational absorption line pumped by the laser.
        t_eval: The time domain to simulate over in [s].

    Returns:
        The normalized population densities.

    Raises:
        RuntimeError: If the solution fails to converge.
    """
    # All state populations used in this module are normalized according to the convention given in
    # "A 3-Level Model for O2 LIF" by Diskin:
    # \hat{N}_1 = N_1 / N_{1,0}
    # \hat{N}_2 = N_2 / N_{1,0}
    # \hat{N}_3 = (N_3 / N_{1,0}) * (f_b / (1 - f_b))

    # The initial state populations are normalized, as are the rate equations.
    n0_hat = [1.0, 0.0, 1.0]

    # By default, the solver chooses its own time array, but we choose to pass one for more control.
    t_span = (t_eval[0], t_eval[-1])

    solution: OdeResult = sy.integrate.solve_ivp(
        fun=rate_equations,
        t_span=t_span,
        y0=n0_hat,
        t_eval=t_eval,
        args=(rate_params, laser_params, pumped_line),
        method="Radau",
        # Default tolerance (rtol=1e-3, atol=1e-6) seems to work well enough.
        # rtol=1e-6,
        # atol=1e-9,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return solution.y


def get_signal(
    t_eval: NDArray[np.float64], n2_hat: NDArray[np.float64], rate_params: RateParams
) -> NDArray[np.float64]:
    """Return the LIF signal as a function of time.

    Args:
        t_eval: The time domain to simulate over in [s].
        n2_hat: Normalized number density of state 2.
        rate_params: Rate parameters and Einstein coefficients for the system.

    Returns:
        The total integrated LIF signal from state 2 as a function of time.
    """
    return rate_params.w_f * sy.integrate.cumulative_simpson(n2_hat, x=t_eval, initial=0)


def create_pumped_sim(
    molecule: Molecule,
    state_up: State,
    state_lo: State,
    temp: float,
    pres: float,
    v_qn_up: int,
    v_qn_lo: int,
) -> Sim:
    """Create a simulation for the pumped absorption line.

    Args:
        molecule: Molecule of interest.
        state_up: Upper electronic state.
        state_lo: Lower electronic state.
        temp: Equilibrium temperature.
        pres: Pressure.
        v_qn_up: Upper state vibrational quantum number v'.
        v_qn_lo: Lower state vibrational quantum number v''.

    Returns:
        A `Sim` object for the pumped absorption line.
    """
    return Sim(
        sim_type=SimType.ABSORPTION,
        molecule=molecule,
        state_up=state_up,
        state_lo=state_lo,
        j_qn_up_max=40,
        pressure=pres,
        bands_input=[(v_qn_up, v_qn_lo)],
        temp_params=TemperatureParams(temp, temp, temp, temp),
        broad_bools=BroadeningBools(
            collisional=True, doppler=True, natural=True, predissociation=True
        ),
    )


def create_emission_sim(pumped_sim: Sim, pumped_line: Line, v_qn_lo_max: int) -> Sim:
    """Create a simulation for LIF emission.

    Args:
        pumped_sim: The parent simulation for the pumped rotational line.
        pumped_line: The pumped rotational line.
        v_qn_lo_max: Maximum lower state vibrational quantum number v''.

    Returns:
        A `Sim` object for LIF emission.
    """
    band_range = [(pumped_sim.bands[0].v_qn_up, v_qn_lo) for v_qn_lo in range(v_qn_lo_max + 1)]

    return Sim(
        sim_type=SimType.LIF,
        molecule=pumped_sim.molecule,
        state_up=pumped_sim.state_up,
        state_lo=pumped_sim.state_lo,
        j_qn_up_max=40,
        pressure=pumped_sim.pressure,
        bands_input=band_range,
        inst_params=InstrumentParams(1, 1),
        temp_params=pumped_sim.temp_params,
        broad_bools=BroadeningBools(
            instrument=True, collisional=True, doppler=True, natural=True, predissociation=True
        ),
        pumped_line=pumped_line,
    )


def find_line(pumped_sim: Sim, branch_name_j: str, branch_idx_lo: int, n_qn_lo: int) -> Line:
    """Return a rotational line with the desired parameters.

    Args:
        pumped_sim: The parent simulation for the pumped rotational line.
        branch_name_j: Branch name with respect to ΔJ.
        branch_idx_lo : Lower state branch index.
        n_qn_lo: Lower state rotational quantum number N''.

    Returns:
        The rotational line matching the input parameters.

    Raises:
        ValueError: If the requested rotational line does not exist within the simulation.
    """
    for line in pumped_sim.bands[0].lines:
        if (
            line.branch_name_j == branch_name_j
            and line.branch_idx_lo == branch_idx_lo
            and line.n_qn_lo == n_qn_lo
            and not line.is_satellite
        ):
            return line

    raise ValueError("No matching rotational line found.")


def gated_n2_integral(
    t_eval: NDArray[np.float64], n2: NDArray[np.float64], gate_start: float, gate_stop: float
) -> float:
    """Integrates the upper state number density over the selected time interval.

    Args:
        t_eval: The time domain to simulate over in [s].
        n2: The upper state number density.
        gate_start: Gate start time in [s].
        gate_stop: Gate stop time in [s].

    Returns:
        The integrated upper state number density.

    Raises:
        ValueError: If t_stop <= t_start.
        ValueError: If the selected gate is too narrow for the current time resolution.
    """
    if gate_stop <= gate_start:
        raise ValueError("The value of gate_stop must be > gate_start.")

    mask = (t_eval >= gate_start) & (t_eval <= gate_stop)

    # Here, `.sum()` finds the total number of True elements in the mask.
    if mask.sum() < 2:
        raise ValueError("Gate is too narrow for the current time resolution.")

    return sy.integrate.simpson(n2[mask], t_eval[mask])


def populations_vs_time(
    emission_sim: Sim,
    pumped_sim: Sim,
    pumped_line: Line,
    laser_params: LIFLaserParams,
    t_eval: NDArray[np.float64],
) -> None:
    """Plot the population densities, signal, and laser intensity as functions of time.

    Args:
        emission_sim: The simulation containing the LIF emission lines.
        pumped_sim: The parent simulation for the pumped rotational line.
        pumped_line: The pumped rotational line.
        laser_params: Laser parameters.
        t_eval: Time steps to integrate over.
    """
    rate_params = time_independent_rates(emission_sim, pumped_sim, pumped_line)

    n1_hat, n2_hat, n3_hat = simulate(rate_params, laser_params, pumped_line, t_eval)

    # Normalize the signal with respect to N2.
    sf = get_signal(t_eval, n2_hat, rate_params)
    sf /= n2_hat.max()

    # Normalize the laser with respect to itself.
    il = laser_intensity(t_eval, laser_params)
    il /= il.max()

    _, ax1 = plt.subplots()
    t_eval = t_eval * 1e9
    ax1.set_xlabel("Time, $t$ [ns]")
    ax1.set_ylabel("$N_1$, $N_3$, $I_l$ (Normalized)")
    ax1.plot(t_eval, n1_hat, label="$N_1$")
    ax1.plot(t_eval, n3_hat, label="$N_3$")
    ax1.plot(t_eval, il, label="$I_l$")
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.set_ylabel("$N_2$, $S_f$ (Normalized)")
    ax2.plot(t_eval, n2_hat, label="$N_2$", linestyle="-.")
    ax2.plot(t_eval, sf, label="$S_f$", linestyle="-.")
    ax2.legend()

    plt.show()


def max_signal_vs_fluence(
    emission_sim: Sim,
    pumped_sim: Sim,
    pumped_line: Line,
    laser_params: LIFLaserParams,
    t_eval: NDArray[np.float64],
) -> None:
    """Plot the maximum fluorescence signal as a function of laser fluence.

    Args:
        emission_sim: The simulation containing the LIF emission lines.
        pumped_sim: The parent simulation for the pumped rotational line.
        pumped_line: The pumped rotational line.
        laser_params: Laser parameters.
        t_eval: Time steps to integrate over.
    """
    rate_params = time_independent_rates(emission_sim, pumped_sim, pumped_line)

    max_fluence = laser_params.fluence

    fluences = np.linspace(0.0, max_fluence, N_FLUENCE, dtype=np.float64)
    max_signals = np.zeros_like(fluences)

    for idx, fluence in enumerate(fluences):
        iterate_laser_params = LIFLaserParams(
            laser_params.pulse_center, laser_params.pulse_width, fluence
        )

        _, n2_hat, _ = simulate(rate_params, iterate_laser_params, pumped_line, t_eval)

        signal = get_signal(t_eval, n2_hat, rate_params)
        max_signals[idx] = signal.max()

    plt.plot(fluences, max_signals / max_signals.max())

    plt.xlabel("Laser Fluence, $\\Phi$ [J/cm$^2$]")
    plt.ylabel("Maximum Signal, $S_f$ [a.u.]")
    plt.show()


def n2_vs_time_and_fluence(
    emission_sim: Sim,
    pumped_sim: Sim,
    pumped_line: Line,
    laser_params: LIFLaserParams,
    t_eval: NDArray[np.float64],
) -> None:
    """Plot upper state population as a function of laser fluence and time.

    Args:
        emission_sim: The simulation containing the LIF emission lines.
        pumped_sim: The parent simulation for the pumped rotational line.
        pumped_line: The pumped rotational line.
        laser_params: Laser parameters.
        t_eval: Time steps to integrate over.
    """
    rate_params = time_independent_rates(emission_sim, pumped_sim, pumped_line)

    fluences = np.linspace(0.0, laser_params.fluence, N_FLUENCE)
    n2_populations = np.zeros((len(fluences), len(t_eval)))

    for idx, fluence in enumerate(fluences):
        iterate_laser_params: LIFLaserParams = LIFLaserParams(
            laser_params.pulse_center, laser_params.pulse_width, fluence
        )

        _, n2_hat, _ = simulate(rate_params, iterate_laser_params, pumped_line, t_eval)

        n2_populations[idx, :] = n2_hat

    t, f = np.meshgrid(t_eval, fluences)

    contour = plt.contourf(t * 1e9, f, n2_populations, levels=50, cmap="magma")

    cbar = plt.colorbar(contour)
    cbar.set_label("$N_2$")

    plt.xlabel("Time, $t$ [ns]")
    plt.ylabel("Laser Fluence, $\\Phi$ [J/cm$^2$]")
    plt.show()


def lif_spectra_vs_time(
    emission_sim: Sim,
    pumped_sim: Sim,
    pumped_line: Line,
    laser_params: LIFLaserParams,
    t_eval: NDArray[np.float64],
) -> None:
    """Plots the LIF spectra.

    Args:
        emission_sim: The simulation containing the LIF emission lines.
        pumped_sim: The parent simulation for the pumped rotational line.
        pumped_line: The pumped rotational line.
        laser_params: Laser parameters.
        t_eval: Time steps to integrate over.
        emission_sim: The simulation containing the LIF emission lines.
    """
    granularity = 10000

    total_number_density = pumped_sim.pressure / (
        constants.BOLTZ * pumped_sim.temp_params.translational
    )

    # N_{1, 0}, the lower state number density of the pumped line.
    number_density_lo = (
        total_number_density
        * pumped_sim.elc_boltz_frac[1]
        * pumped_line.band.vib_boltz_frac[1]
        * pumped_line.rot_boltz_frac[1]
    )

    points_per_ns = t_eval.size / (t_eval[-1] - t_eval[0])

    rate_params = time_independent_rates(emission_sim, pumped_sim, pumped_line)
    _, n2_hat, _ = simulate(rate_params, laser_params, pumped_line, t_eval)

    n2 = n2_hat * number_density_lo

    selected_time = 30e-9
    idx_at_time = int(points_per_ns * selected_time)

    gate_start = 10e-9
    gate_stop = 20e-9

    n2_gate = gated_n2_integral(t_eval, n2, gate_start, gate_stop)

    wavenumbers_line = np.concatenate([band.wavenumbers_line() for band in emission_sim.bands])
    inst_broadening = max(emission_sim.bands[0].lines[0].fwhm_instrument())
    padding = 10.0 * max(inst_broadening, 2.0)

    grid_min = wavenumbers_line.min() - padding
    grid_max = wavenumbers_line.max() + padding

    wavenumbers_cont = np.linspace(grid_min, grid_max, granularity, dtype=np.float64)
    wavelengths_cont = utils.wavenum_to_wavelen(wavenumbers_cont)

    # Intensity per upper number density.
    intensities_cont = np.zeros_like(wavenumbers_cont)

    for band in emission_sim.bands:
        intensities_cont += band.intensities_cont(wavenumbers_cont)

    # Intensity vs. time and wavelength.
    i_t_wl = n2[None, :] * intensities_cont[:, None]

    w, t = np.meshgrid(wavelengths_cont, t_eval, indexing="ij")

    contour = plt.contourf(w, t * 1e9, i_t_wl, levels=50, cmap="magma")
    cbar = plt.colorbar(contour)
    cbar.set_label("Intensity, $I$ [a.u.]")

    plt.xlabel(r"Wavelength, $\lambda$ [nm]")
    plt.ylabel("Time, $t$ [ns]")
    plt.show()

    plt.plot(
        wavelengths_cont,
        intensities_cont * n2_gate / (gate_stop - gate_start),
        label=f"Gated average spectrum from {gate_start * 1e9} ns to {gate_stop * 1e9} ns",
    )
    plt.plot(
        wavelengths_cont,
        intensities_cont * n2[idx_at_time],
        label=f"Instantaneous spectrum at t={selected_time * 1e9:.1f} ns",
    )
    plt.xlabel(r"Wavelength, $\lambda$ [nm]")
    plt.ylabel("Intensity, $I$ [a.u.]")
    plt.legend()
    plt.show()


def main() -> None:
    """Entry point."""
    molecule = Molecule(atom_1=Atom(16, "O"), atom_2=Atom(16, "O"))
    state_up = State(
        molecule=molecule,
        letter="B",
        spin_multiplicity=3,
        term_symbol=TermSymbol.SIGMA,
        inversion_symmetry=InversionSymmetry.UNGERADE,
        reflection_symmetry=ReflectionSymmetry.MINUS,
        constants_type=ConstantsType.PERLEVEL,
    )
    state_lo = State(
        molecule=molecule,
        letter="X",
        spin_multiplicity=3,
        term_symbol=TermSymbol.SIGMA,
        inversion_symmetry=InversionSymmetry.GERADE,
        reflection_symmetry=ReflectionSymmetry.MINUS,
        constants_type=ConstantsType.PERLEVEL,
    )

    pres = 101325.0
    temp = 1800.0
    v_qn_up = 2
    v_qn_lo = 7

    pumped_sim = create_pumped_sim(molecule, state_up, state_lo, temp, pres, v_qn_up, v_qn_lo)

    branch_name_j = "P"
    branch_idx_lo = 1
    n_qn_lo = 9

    pumped_line = find_line(pumped_sim, branch_name_j, branch_idx_lo, n_qn_lo)

    # For a given v', get the maximum value of v'' (account for 0-indexing).
    v_qn_lo_max = pumped_sim.einstein[v_qn_up].size - 1
    emission_sim = create_emission_sim(pumped_sim, pumped_line, v_qn_lo_max)

    pulse_center = 30e-9
    pulse_width = 20e-9
    fluence = 42.5e-3

    laser_params = LIFLaserParams(pulse_center, pulse_width, fluence)

    n_points = 1000
    min_time = 0.0
    max_time = 60.0

    # Convert times from [ns] to [s].
    t_eval = np.linspace(min_time * 1e-9, max_time * 1e-9, n_points)

    lif_spectra_vs_time(emission_sim, pumped_sim, pumped_line, laser_params, t_eval)
    populations_vs_time(emission_sim, pumped_sim, pumped_line, laser_params, t_eval)

    # Experimental data for the Schumann-Runge bands of O2 at 1800 K, 1 atm, extracted from
    # Figure 5 of <https://doi.org/10.1364/AO.34.005501>.
    jay_27_p9x = np.array([0.0, 1.8, 3.6, 6.0, 12.0, 24.0, 42.5]) / 1e3
    jay_27_p9y = np.array([0.0, 0.08, 0.15, 0.27, 0.47, 0.7, 1.0])
    plt.scatter(jay_27_p9x, jay_27_p9y)
    max_signal_vs_fluence(emission_sim, pumped_sim, pumped_line, laser_params, t_eval)

    n2_vs_time_and_fluence(emission_sim, pumped_sim, pumped_line, laser_params, t_eval)


if __name__ == "__main__":
    main()
