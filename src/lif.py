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
from atom import Atom
from molecule import Molecule
from sim import Sim
from sim_params import BroadeningBools, TemperatureParams
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
        b_12: Einstein coefficient for photon absorption in [cm^2/(J.s)].
        b_21: Einstein coefficient for stimulated emission in [cm^2/(J.s)].
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


def time_independent_rates(sim: Sim, line: Line) -> RateParams:
    """Return the time-independent rate parameters.

    Args:
        sim: The parent simulation.
        line: The desired rotational line.

    Returns:
        Parameters related to the rate equations for the selected rotational line.
    """
    g_u = constants.ELECTRONIC_DEGENERACIES[sim.molecule.name][sim.state_up.name]
    g_l = constants.ELECTRONIC_DEGENERACIES[sim.molecule.name][sim.state_lo.name]

    # Only a single vibrational band will be simulated at a time.
    v_qn_up = sim.bands[0].v_qn_up
    v_qn_lo = sim.bands[0].v_qn_lo

    j_qn_lo = line.j_qn_lo
    s_j = line.honl_london_factor

    # Einstein coefficient for spontaneous emission in [1/s].
    a_21: float = sim.einstein[v_qn_up, v_qn_lo] * s_j / (2 * j_qn_lo + 1)

    # Fluorescent radiative decay rate in [1/s].
    # The sum of all downward radiative transitions A_{ul} where u = v', minus the resonant
    # contribution where l = v'.
    w_f: float = (
        (sim.einstein[v_qn_up].sum() - sim.einstein[v_qn_up, v_qn_lo]) * s_j / (2 * j_qn_lo + 1)
    )

    # Einstein coefficient for photon absorption in [cm^2/(J.s)], Herzberg Eq. (I, 56).
    b_12 = a_21 / (8 * np.pi * constants.PLANC * constants.LIGHT * line.wavenumber**3) * g_u / g_l

    # Einstein coefficient for stimulated emission in [cm^2/(J.s)].
    b_21 = b_12 * g_l / g_u

    # Predissociation rate in [1/s].
    w_d = 2.0 * np.pi * constants.LIGHT * line.fwhm_predissociation()

    # NOTE: 26/01/30 - This is a really naïve way of calculating both the upper state quenching and
    #       collisional bath terms, but it's good enough for a first estimation. Further refinement
    #       will be dependent on the molecule used, so this is a general, if simple, approach.

    # Collisional repopulation of the lower state in [1/s].
    w_c = 2.0 * np.pi * constants.LIGHT * line.fwhm_collisional()
    # Quenching rate of the upper state in [1/s].
    w_q = w_c

    return RateParams(a_21, b_12, b_21, w_c, w_d, w_f, w_q)


def rate_equations(
    t: float,
    n: NDArray[np.float64],
    rate_params: RateParams,
    laser_params: LIFLaserParams,
    line: Line,
) -> list[float]:
    """Return the rate equations governing the LIF system.

    Args:
        t: Current time in [s].
        n: Nondimensional population densities of N1, N2, and N3 at a point in time.
        rate_params: Rate parameters and Einstein coefficients for the system.
        laser_params: Laser parameters.
        line: The rotational line of interest.

    Returns:
        Differential equations dN1/dt, dN2/dt, and dN3/dt.
    """
    n1, n2, n3 = n

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
    f_b = line.rot_boltz_frac[1]

    # Normalized rate equations from "A 3-Level Model for O2 LIF" by Diskin; see the `simulate`
    # function for the normalization used.
    dn1_dt: float = (
        -(w_la + rate_params.w_c) * n1 + (w_le + rate_params.a_21) * n2 + rate_params.w_c * n3
    )
    dn2_dt: float = (
        w_la * n1
        - (w_le + rate_params.w_d + rate_params.a_21 + rate_params.w_f + rate_params.w_q) * n2
    )
    dn3_dt: float = -(f_b / (1.0 - f_b)) * rate_params.w_c * (n3 - n1)

    return [dn1_dt, dn2_dt, dn3_dt]


def simulate(
    rate_params: RateParams,
    laser_params: LIFLaserParams,
    line: Line,
    t_eval: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the time and population densities of the three states as functions of time.

    Args:
        rate_params: Rate parameters.
        laser_params: Laser parameters.
        line: The rotational line of interest.
        t_eval: Time steps to integrate over.

    Returns:
        The solution array.

    Raises:
        RuntimeError: If the solution fails to converge.
    """
    # All state populations used in this module are normalized according to the convention given in
    # "A 3-Level Model for O2 LIF" by Diskin:
    # \hat{N}_1 = N_1 / N_{1,0}
    # \hat{N}_2 = N_2 / N_{1,0}
    # \hat{N}_3 = (N_3 / N_{1,0}) * (f_b / (1 - f_b))

    # The initial state populations are normalized, as are the rate equations.
    n0 = [1.0, 0.0, 1.0]

    # By default, the solver chooses its own time array, but we choose to pass one for more control.
    t_span = (t_eval[0], t_eval[-1])

    solution: OdeResult = sy.integrate.solve_ivp(
        fun=rate_equations,
        t_span=t_span,
        y0=n0,
        t_eval=t_eval,
        args=(rate_params, laser_params, line),
        method="Radau",
        # Default tolerance (rtol=1e-3, atol=1e-6) seems to work well enough.
        # rtol=1e-6,
        # atol=1e-9,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return solution.y


def get_signal(
    t: NDArray[np.float64], n2: NDArray[np.float64], rate_params: RateParams
) -> NDArray[np.float64]:
    """Return the LIF signal as a function of time.

    Args:
        t: The time domain to simulate over in [s].
        n2: Normalized population density of state 2.
        rate_params: Rate parameters and Einstein coefficients for the system.

    Returns:
        The total integrated LIF signal from state 2 as a function of time.
    """
    return rate_params.w_f * sy.integrate.cumulative_trapezoid(n2, t, initial=0)


def create_sim(
    molecule: Molecule,
    state_up: State,
    state_lo: State,
    temp: float,
    pres: float,
    v_qn_up: int,
    v_qn_lo: int,
) -> Sim:
    """Return a simulation object with the desired parameters.

    Args:
        molecule: Molecule of interest.
        state_up: Upper electronic state.
        state_lo: Lower electronic state.
        temp: Equilibrium temperature.
        pres: Pressure.
        v_qn_up: Upper state vibrational quantum number v'.
        v_qn_lo: Lower state vibrational quantum number v''.

    Returns:
        A `Sim` object with the desired parameters.
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


def find_line(sim: Sim, branch_name_j: str, branch_idx_lo: int, n_qn_lo: int) -> Line:
    """Return a rotational line with the desired parameters.

    Args:
        sim: The parent simulation.
        branch_name_j: Branch name with respect to ΔJ.
        branch_idx_lo : Lower state branch index.
        n_qn_lo: Lower state rotational quantum number N''.

    Returns:
        The rotational line matching the input parameters.

    Raises:
        ValueError: If the requested rotational line does not exist within the simulation.
    """
    for line in sim.bands[0].lines:
        if (
            line.branch_name_j == branch_name_j
            and line.branch_idx_lo == branch_idx_lo
            and line.n_qn_lo == n_qn_lo
            and not line.is_satellite
        ):
            return line

    raise ValueError("No matching rotational line found.")


def populations_vs_time(
    sim: Sim, line: Line, laser_params: LIFLaserParams, t_eval: NDArray[np.float64]
) -> None:
    """Plot the population densities, signal, and laser intensity as functions of time.

    Args:
        sim: Simulation.
        line: Line.
        laser_params: Laser parameters.
        t_eval: Time steps to integrate over.
    """
    rate_params = time_independent_rates(sim, line)

    n1, n2, n3 = simulate(rate_params, laser_params, line, t_eval)

    # Normalize the signal with respect to N2.
    sf = get_signal(t_eval, n2, rate_params)
    sf /= n2.max()

    # Normalize the laser with respect to itself.
    il = laser_intensity(t_eval, laser_params)
    il /= il.max()

    _, ax1 = plt.subplots()
    ax1.set_xlabel("Time, $t$ [s]")
    ax1.set_ylabel("$N_1$, $N_3$, $I_l$ (Normalized)")
    ax1.plot(t_eval, n1, label="$N_1$")
    ax1.plot(t_eval, n3, label="$N_3$")
    ax1.plot(t_eval, il, label="$I_l$")
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.set_ylabel("$N_2$, $S_f$ (Normalized)")
    ax2.plot(t_eval, n2, label="$N_2$", linestyle="-.")
    ax2.plot(t_eval, sf, label="$S_f$", linestyle="-.")
    ax2.legend()

    plt.show()


def max_signal_vs_fluence(
    sim: Sim, line: Line, laser_params: LIFLaserParams, t_eval: NDArray[np.float64]
) -> None:
    """Plot the maximum fluorescence signal as a function of laser fluence.

    Args:
        sim: Simulation.
        line: Line.
        laser_params: Laser parameters.
        t_eval: Time steps to integrate over.
    """
    rate_params = time_independent_rates(sim, line)

    max_fluence = laser_params.fluence

    fluences = np.linspace(0.0, max_fluence, N_FLUENCE, dtype=np.float64)
    max_signals = np.zeros_like(fluences)

    for idx, fluence in enumerate(fluences):
        iterate_laser_params = LIFLaserParams(
            laser_params.pulse_center, laser_params.pulse_width, fluence
        )

        _, n2, _ = simulate(rate_params, iterate_laser_params, line, t_eval)

        signal = get_signal(t_eval, n2, rate_params)
        max_signals[idx] = signal.max()

    plt.plot(fluences, max_signals / max_signals.max())

    plt.xlabel("Laser Fluence, $\\Phi$ [J/cm$^2$]")
    plt.ylabel("Maximum Signal, $S_f$ [a.u.]")
    plt.show()


def n2_vs_time_and_fluence(
    sim: Sim, line: Line, laser_params: LIFLaserParams, t_eval: NDArray[np.float64]
) -> None:
    """Plot upper state population as a function of laser fluence and time.

    Args:
        sim: Simulation.
        line: Line.
        laser_params: Laser parameters.
        t_eval: Time steps to integrate over.
    """
    rate_params = time_independent_rates(sim, line)

    fluences = np.linspace(0.0, laser_params.fluence, N_FLUENCE)
    n2_populations = np.zeros((len(fluences), len(t_eval)))

    for idx, fluence in enumerate(fluences):
        iterate_laser_params: LIFLaserParams = LIFLaserParams(
            laser_params.pulse_center, laser_params.pulse_width, fluence
        )

        _, n2, _ = simulate(rate_params, iterate_laser_params, line, t_eval)

        n2_populations[idx, :] = n2

    t, f = np.meshgrid(t_eval, fluences)

    contour = plt.contourf(t, f, n2_populations, levels=50, cmap="magma")

    cbar = plt.colorbar(contour)
    cbar.set_label("$N_2$")

    plt.xlabel("Time, $t$ [s]")
    plt.ylabel("Laser Fluence, $\\Phi$ [J/cm$^2$]")
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

    sim = create_sim(molecule, state_up, state_lo, temp, pres, v_qn_up, v_qn_lo)

    branch_name_j = "P"
    branch_idx_lo = 1
    n_qn_lo = 9

    line = find_line(sim, branch_name_j, branch_idx_lo, n_qn_lo)

    pulse_center = 30e-9
    pulse_width = 20e-9
    fluence = 42.5e-3

    laser_params = LIFLaserParams(pulse_center, pulse_width, fluence)

    t = np.linspace(0.0, 60e-9, 1000)

    populations_vs_time(sim, line, laser_params, t)

    # Experimental data for the Schumann-Runge bands of O2 at 1800 K, 1 atm, extracted from
    # Figure 5 of <https://doi.org/10.1364/AO.34.005501>.
    jay_27_p9x = np.array([0.0, 1.8, 3.6, 6.0, 12.0, 24.0, 42.5]) / 1e3
    jay_27_p9y = np.array([0.0, 0.08, 0.15, 0.27, 0.47, 0.7, 1.0])
    plt.scatter(jay_27_p9x, jay_27_p9y)
    max_signal_vs_fluence(sim, line, laser_params, t)

    n2_vs_time_and_fluence(sim, line, laser_params, t)


if __name__ == "__main__":
    main()
