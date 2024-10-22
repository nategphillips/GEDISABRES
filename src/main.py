# module main
"""
A simulation of the Schumann-Runge bands of molecular oxygen written in Python.
"""

from enum import Enum

from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import wofz  # pylint: disable=no-name-in-module

import constants as cn


plot_colors: dict[str, str] = {"background": "#1e1e2e", "text": "#cdd6f4"}
line_colors: list[str] = [
    "#cba6f7",
    "#f38ba8",
    "#fab387",
    "#f9e2af",
    "#a6e3a1",
    "#89dceb",
    "#89b4fa",
]

plt.rcParams.update(
    {
        "figure.facecolor": plot_colors["background"],
        "figure.edgecolor": plot_colors["background"],
        "axes.facecolor": plot_colors["background"],
        "axes.edgecolor": plot_colors["text"],
        "axes.labelcolor": plot_colors["text"],
        "grid.color": plot_colors["text"],
        "text.color": plot_colors["text"],
        "xtick.color": plot_colors["text"],
        "ytick.color": plot_colors["text"],
        "font.size": 12,
    }
)

plt.rcParams["axes.prop_cycle"] = cycler(color=line_colors)


class Atom:
    """
    Represents an atom with a name and mass.
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.mass: float = self.get_mass(name) / cn.AVOGD / 1e3

    @staticmethod
    def get_mass(name: str) -> float:
        """
        Returns the atomic mass in [g/mol].
        """

        if name not in cn.ATOMIC_MASSES:
            raise ValueError(f"Atom `{name}` not supported.")

        return cn.ATOMIC_MASSES[name]


class Molecule:
    """
    Represents a diatomic molecule consisting of two atoms.
    """

    def __init__(self, name: str, atom_1: Atom, atom_2: Atom) -> None:
        self.name: str = name
        self.atom_1: Atom = atom_1
        self.atom_2: Atom = atom_2
        self.mass: float = self.atom_1.mass + self.atom_2.mass
        # FIXME: 10/21/24 - This is wrong: in the Hanson book, the reduced mass uses the masses of
        #        the two molecules interacting with one another, not the two atoms that form the
        #        molecule.
        self.reduced_mass: float = self.atom_1.mass * self.atom_2.mass / self.mass
        self.symmetry_param: int = self.get_symmetry_param(atom_1, atom_2)

    @staticmethod
    def get_symmetry_param(atom_1: Atom, atom_2: Atom) -> int:
        """
        Returns the symmetry parameter of the molecule.
        """

        # For homonuclear diatomic molecules like O2, the symmetry parameter is 2.
        if atom_1.name == atom_2.name:
            return 2

        # For heteronuclear diatomics, it's 1.
        return 1


class SimulationType(Enum):
    """
    Defines the type of simulation to be performed.
    """

    ABSORPTION = 1
    EMISSION = 2


class ElectronicState:
    """
    Represents an electronic state of a particular molecule.
    """

    def __init__(self, name: str, spin_multiplicity: int, molecule: Molecule) -> None:
        self.name: str = name
        self.spin_multiplicity: int = spin_multiplicity
        self.molecule: Molecule = molecule
        self.constants: dict[str, dict[int, float]] = self.get_constants(molecule.name, name)
        self.cross_section: float = self.get_cross_section(molecule.atom_1, molecule.atom_2)

    @staticmethod
    def get_cross_section(atom_1: Atom, atom_2: Atom) -> float:
        """
        Returns the cross-section of the molecule in [m^2].
        """

        # TODO: 10/21/24 - Placeholder, need to add the radius of the molecule in each electronic
        #       state to the constants.
        return np.pi * (2 * 1.2e-10) ** 2

    @staticmethod
    def get_constants(molecule: str, state: str) -> dict[str, dict[int, float]]:
        """
        Returns a dictionary of molecular constants for the specified electronic state in [1/cm].
        """

        # TODO: 10/18/24 - Add errors here if the molecule or state is not supported.

        return pd.read_csv(f"../data/{molecule}/states/{state}.csv").to_dict()

    def is_allowed(self, n_qn: int) -> bool:
        """
        Returns a boolean value corresponding to whether or not the selected rotational level is
        allowed.
        """

        if self.name == "X3Sg-":
            # For X3Σg-, only the rotational levels with odd N can be populated.
            return bool(n_qn % 2 == 1)
        if self.name == "B3Su-":
            # For B3Σu-, only the rotational levels with even N can be populated.
            return bool(n_qn % 2 == 0)

        raise ValueError(f"State {self.name} not supported.")


class Simulation:
    """
    Simulates the spectra of a particular molecule.
    """

    def __init__(
        self,
        sim_type: SimulationType,
        molecule: Molecule,
        state_up: ElectronicState,
        state_lo: ElectronicState,
        rot_lvls: np.ndarray,
        temperature: float,
        pressure: float,
        vib_bands: list[tuple[int, int]],
    ) -> None:
        self.sim_type: SimulationType = sim_type
        self.molecule: Molecule = molecule
        self.state_up: ElectronicState = state_up
        self.state_lo: ElectronicState = state_lo
        self.rot_lvls: np.ndarray = rot_lvls
        self.temperature: float = temperature
        self.pressure: float = pressure
        self.vib_part: float = self.get_vib_partition_fn()
        self.franck_condon: np.ndarray = self.get_franck_condon()
        self.einstein: np.ndarray = self.get_einstein()
        self.vib_bands: list[VibrationalBand] = self.get_vib_bands(vib_bands)

    def get_einstein(self) -> np.ndarray:
        """
        Returns a table of Einstein coefficients for spontaneous emission: A_{v'v''}. Rows
        correspond to the upper state vibrational quantum number (v'), while columns correspond to
        the lower state vibrational quantum number (v'').
        """

        return np.loadtxt(
            f"../data/{self.molecule.name}/einstein/{self.state_up.name}_to_{self.state_lo.name}_laux.csv",
            delimiter=",",
        )

    def get_franck_condon(self) -> np.ndarray:
        """
        Returns a table of Franck-Condon factors for the associated electronic transition. Rows
        correspond to the upper state vibrational quantum number (v'), while columns correspond to
        the lower state vibrational quantum number (v'').
        """

        # TODO: 10/21/24 - Add error checking here.

        return np.loadtxt(
            f"../data/{self.molecule.name}/franck-condon/{self.state_up.name}_to_{self.state_lo.name}_cheung.csv",
            delimiter=",",
        )

    def get_vib_bands(self, vib_bands: list[tuple[int, int]]):
        """
        Returns the selected vibrational bands within the simulation.
        """

        return [
            VibrationalBand(sim=self, v_qn_up=vib_band[0], v_qn_lo=vib_band[1])
            for vib_band in vib_bands
        ]

    def get_vib_partition_fn(self) -> float:
        """
        Returns the vibrational partition function.
        """

        # NOTE: 10/22/24 - The maximum vibrational quantum number is dictated by the tabulated data
        #       available.
        match self.sim_type:
            case SimulationType.EMISSION:
                state = self.state_up
                v_qn_max = len(self.state_up.constants["G"])
            case SimulationType.ABSORPTION:
                state = self.state_lo
                v_qn_max = len(self.state_lo.constants["G"])
        print(v_qn_max)

        q_v: float = 0.0

        # NOTE: 10/22/24 - The vibrational partition function is always computed using a set number
        #       of vibrational bands to ensure an accurate estimate of the state sum is obtained.
        #       This approach is used to ensure the sum is calculated correctly regardless of the
        #       number of vibrational bands simulated by the user.
        for v_qn in range(0, v_qn_max):
            q_v += np.exp(
                -vibrational_term(state, v_qn) * cn.PLANC * cn.LIGHT / (cn.BOLTZ * self.temperature)
            )

        return q_v


class VibrationalBand:
    """
    Represents a vibrational band of a particular molecule.
    """

    def __init__(self, sim: Simulation, v_qn_up: int, v_qn_lo: int) -> None:
        self.sim: Simulation = sim
        self.v_qn_up: int = v_qn_up
        self.v_qn_lo: int = v_qn_lo
        self.band_origin: float = self.get_band_origin()
        self.rot_part: float = self.get_rot_partition_fn()
        self.vib_boltz_frac: float = self.get_vib_boltz_frac()
        self.rot_lines: list[RotationalLine] = self.get_rot_lines()

    def wavenumbers_line(self) -> np.ndarray:
        return np.array([line.wavenumber for line in self.rot_lines])

    def intensities_line(self) -> np.ndarray:
        return np.array([line.intensity for line in self.rot_lines])

    def wavenumbers_conv(self) -> np.ndarray:
        wns_line: np.ndarray = self.wavenumbers_line()

        # TODO: 10/21/24 - Allow the granularity to be selected by the user, think about using
        #       linear interpolation to reduce computational time.

        # Generate a fine-grained x-axis using existing wavenumber data.
        return np.linspace(wns_line.min(), wns_line.max(), 100000)

    def intensities_conv(self) -> np.ndarray:
        return convolve_brod(self.rot_lines, self.wavenumbers_conv())

    def get_vib_boltz_frac(self) -> float:
        """
        Returns the vibrational Boltzmann fraction N_v / N.
        """

        match self.sim.sim_type:
            case SimulationType.EMISSION:
                state = self.sim.state_up
                v_qn = self.v_qn_up
            case SimulationType.ABSORPTION:
                state = self.sim.state_lo
                v_qn = self.v_qn_lo

        return (
            np.exp(
                -vibrational_term(state, v_qn)
                * cn.PLANC
                * cn.LIGHT
                / (cn.BOLTZ * self.sim.temperature)
            )
            / self.sim.vib_part
        )

    def get_band_origin(self) -> float:
        """
        Returns the band origin in [1/cm].
        """

        # Herzberg p. 168, eq. (IV, 24)

        upper_state: dict[str, dict[int, float]] = self.sim.state_up.constants

        # Convert Cheung's definition of the band origin (T) to Herzberg's definition (nu_0).
        energy_offset: float = (
            2 / 3 * upper_state["lamda"][self.v_qn_up] - upper_state["gamma"][self.v_qn_up]
        )

        return upper_state["T"][self.v_qn_up] + energy_offset

    def get_rot_partition_fn(self) -> float:
        """
        Returns the rotational partition function.
        """

        match self.sim.sim_type:
            case SimulationType.EMISSION:
                state = self.sim.state_up
                v_qn = self.v_qn_up
            case SimulationType.ABSORPTION:
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
            #       are all close together, so it shouldn't matter too much.
            q_r += (2 * j_qn + 1) * np.exp(
                -rotational_term(state, v_qn, j_qn, 2)
                * cn.PLANC
                * cn.LIGHT
                / (cn.BOLTZ * self.sim.temperature)
            )

        # NOTE: 10/22/24 - Alternatively, the high-temperature approximation can be used instead of
        #       the direct sum approach. This also works well.
        # q_r = cn.BOLTZ * self.sim.temperature / (cn.PLANC * cn.LIGHT * state.constants["B"][v_qn])

        # The state sum must be divided by the symmetry parameter to account for identical
        # rotational orientations in space.
        return q_r / self.sim.molecule.symmetry_param

    def get_rot_lines(self):
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
                RotationalLine(
                    sim=self.sim,
                    band=self,
                    n_qn_up=n_qn_up,
                    n_qn_lo=n_qn_lo,
                    j_qn_up=n2j_qn(n_qn_up, branch_idx_up),
                    j_qn_lo=n2j_qn(n_qn_lo, branch_idx_lo),
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


class RotationalLine:
    """
    Represents a rotational line within a vibrational band.
    """

    def __init__(
        self,
        sim: Simulation,
        band: VibrationalBand,
        n_qn_up: int,
        n_qn_lo: int,
        j_qn_up: int,
        j_qn_lo: int,
        branch_idx_up: int,
        branch_idx_lo: int,
        branch_name: str,
        is_satellite: bool,
    ) -> None:
        self.sim: Simulation = sim
        self.band: VibrationalBand = band
        self.n_qn_up: int = n_qn_up
        self.n_qn_lo: int = n_qn_lo
        self.j_qn_up: int = j_qn_up
        self.j_qn_lo: int = j_qn_lo
        self.branch_idx_up: int = branch_idx_up
        self.branch_idx_lo: int = branch_idx_lo
        self.branch_name: str = branch_name
        self.is_satellite: bool = is_satellite
        self.wavenumber: float = self.get_wavenumber()
        self.honl_london_factor: float = self.get_honl_london_factor()
        self.rot_boltz_frac: float = self.get_rot_boltz_frac()
        self.intensity: float = self.get_intensity()

    def fwhm_params(self) -> tuple[float, float]:
        """
        Returns the Gaussian and Lorentzian full width at half maximum parameters in [1/cm].
        """

        # TODO: 10/21/24 - Look over this, seems weird still.

        # The sum of the Einstein A coefficients for all downward transitions from the two levels of
        # the transitions i and j.
        i: int = self.band.v_qn_up
        a_ik: float = 0.0
        for k in range(0, i):
            a_ik += self.sim.einstein[i][k]

        j: int = self.band.v_qn_lo
        a_jk: float = 0.0
        for k in range(0, j):
            a_jk += self.sim.einstein[j][k]

        # Natural broadening in [1/s].
        natural: float = (a_ik + a_jk) / (2 * np.pi)

        # TODO: 10/21/24 - Not sure which electronic state should be used for the cross-section.

        # Collisional (pressure) broadening in [1/s].
        collisional: float = (
            self.sim.pressure
            * self.sim.state_lo.cross_section
            * np.sqrt(
                8 / (np.pi * self.sim.molecule.reduced_mass * cn.BOLTZ * self.sim.temperature)
            )
            / np.pi
        )

        # Doppler (thermal) broadening in [1/cm]. Note that the speed of light is converted from
        # [cm/s] to [m/s] to ensure that the units work out correctly.
        doppler: float = self.wavenumber * np.sqrt(
            8
            * cn.BOLTZ
            * self.sim.temperature
            * np.log(2)
            / (self.sim.molecule.mass * (cn.LIGHT / 1e2) ** 2)
        )

        # FIXME: 10/21/24 - Temporarily adding a 0.3 here to simulate the effects of predissociation

        # Convert the Lorentzian broadening parameters from [1/s] to [1/cm].
        lorentzian: float = (natural + collisional) / cn.LIGHT + 0.3

        return doppler, lorentzian

    def get_wavenumber(self) -> float:
        """
        Returns the wavenumber.
        """

        # NOTE: 10/18/24 - Make sure to understand transition structure: Herzberg pp. 149-152, and
        #       pp. 168-169.

        # Herzberg p. 168, eq. (IV, 24)
        return (
            self.band.band_origin
            + rotational_term(
                self.sim.state_up, self.band.v_qn_up, self.j_qn_up, self.branch_idx_up
            )
            - rotational_term(
                self.sim.state_lo, self.band.v_qn_lo, self.j_qn_lo, self.branch_idx_lo
            )
        )

    def get_intensity(self) -> float:
        """
        Returns the intensity.
        """

        # NOTE: 10/18/24 - Before going any further make sure to read Herzberg pp. 20-21,
        #       pp. 126-127, pp. 200-201, and pp. 382-383.

        match self.sim.sim_type:
            case SimulationType.EMISSION:
                j_qn = self.j_qn_up
                wavenumber_factor = self.wavenumber**4
            case SimulationType.ABSORPTION:
                j_qn = self.j_qn_lo
                wavenumber_factor = self.wavenumber

        return (
            wavenumber_factor
            * self.rot_boltz_frac
            * self.band.vib_boltz_frac
            * self.honl_london_factor
            / (2 * j_qn + 1)
            * self.sim.franck_condon[self.band.v_qn_up][self.band.v_qn_lo]
        )

    def get_rot_boltz_frac(self) -> float:
        """
        Returns the rotational Boltzmann fraction N_J / N.
        """

        match self.sim.sim_type:
            case SimulationType.EMISSION:
                state = self.sim.state_up
                v_qn = self.band.v_qn_up
                j_qn = self.j_qn_up
                branch_idx = self.branch_idx_up
            case SimulationType.ABSORPTION:
                state = self.sim.state_lo
                v_qn = self.band.v_qn_lo
                j_qn = self.j_qn_lo
                branch_idx = self.branch_idx_lo

        return (
            (2 * j_qn + 1)
            * np.exp(
                -rotational_term(state, v_qn, j_qn, branch_idx)
                * cn.PLANC
                * cn.LIGHT
                / (cn.BOLTZ * self.sim.temperature)
            )
            / self.band.rot_part
        )

    def get_honl_london_factor(self) -> float:
        """
        Returns the Hönl-London factor (line strength).
        """

        # For emission, the relevant rotational quantum number is N'; for absorption, it's N''.
        match self.sim.sim_type:
            case SimulationType.EMISSION:
                n_qn = self.n_qn_up
            case SimulationType.ABSORPTION:
                n_qn = self.n_qn_lo

        # Convert the properties of the current rotational line into a useful key.
        if self.is_satellite:
            key = f"{self.branch_name}Q{self.branch_idx_up}{self.branch_idx_lo}"
        else:
            # For main branches, the upper and lower branches indicies are the same, so it doesn't
            # matter which one is used here.
            key = f"{self.branch_name}{self.branch_idx_up}"

        # These factors are from Tatum - 1966: Hönl-London Factors for 3Σ±-3Σ± Transitions.
        factors: dict[SimulationType, dict[str, float]] = {
            SimulationType.EMISSION: {
                "P1": ((n_qn + 1) * (2 * n_qn + 5)) / (2 * n_qn + 3),
                "R1": (n_qn * (2 * n_qn + 3)) / (2 * n_qn + 1),
                "P2": (n_qn * (n_qn + 2)) / (n_qn + 1),
                "R2": ((n_qn - 1) * (n_qn + 1)) / n_qn,
                "P3": ((n_qn + 1) * (2 * n_qn - 1)) / (2 * n_qn + 1),
                "R3": (n_qn * (2 * n_qn - 3)) / (2 * n_qn - 1),
                "PQ12": 1 / (n_qn + 1),
                "RQ21": 1 / n_qn,
                "PQ13": 1 / ((n_qn + 1) * (2 * n_qn + 1) * (2 * n_qn + 3)),
                "RQ31": 1 / (n_qn * (2 * n_qn - 1) * (2 * n_qn + 1)),
                "PQ23": 1 / (n_qn + 1),
                "RQ32": 1 / n_qn,
            },
            SimulationType.ABSORPTION: {
                "P1": (n_qn * (2 * n_qn + 3)) / (2 * n_qn + 1),
                "R1": ((n_qn + 1) * (2 * n_qn + 5)) / (2 * n_qn + 3),
                "P2": ((n_qn - 1) * (n_qn + 1)) / n_qn,
                "R2": (n_qn * (n_qn + 2)) / (n_qn + 1),
                "P3": (n_qn * (2 * n_qn - 3)) / (2 * n_qn - 1),
                "R3": ((n_qn + 1) * (2 * n_qn - 1)) / (2 * n_qn + 1),
                "PQ12": 1 / n_qn,
                "RQ21": 1 / (n_qn + 1),
                "PQ13": 1 / (n_qn * (2 * n_qn - 1) * (2 * n_qn + 1)),
                "RQ31": 1 / ((n_qn + 1) * (2 * n_qn + 1) * (2 * n_qn + 3)),
                "PQ23": 1 / n_qn,
                "RQ32": 1 / (n_qn + 1),
            },
        }

        return factors[self.sim.sim_type][key]


def vibrational_term(state: ElectronicState, v_qn: int) -> float:
    """
    Returns the vibrational term value.
    """

    return state.constants["G"][v_qn]


def rotational_term(state: ElectronicState, v_qn: int, j_qn: int, branch_idx: int) -> float:
    """
    Returns the rotational term value.
    """

    # TODO: 10/17/24 - Change molecular constant data for the ground state to reflect the
    #       conventions in Cheung.
    #       Yu      | Cheung
    #       --------|------------
    #       D       | -D
    #       lamda_D | lamda_D / 2
    #       gamma_D | gamma_D / 2

    lookup: dict = state.constants

    b: float = lookup["B"][v_qn]
    d: float = lookup["D"][v_qn]
    l: float = lookup["lamda"][v_qn]
    g: float = lookup["gamma"][v_qn]
    ld: float = lookup["lamda_D"][v_qn]
    gd: float = lookup["gamma_D"][v_qn]

    # The Hamiltonian from Cheung is written in Hund's case (a) representation, so J is used instead
    # of N.
    x: int = j_qn * (j_qn + 1)

    # The four Hamiltonian matrix elements given in Cheung.
    h11: float = (
        b * (x + 2)
        - d * (x**2 + 8 * x + 4)
        - 4 / 3 * l
        - 2 * g
        - 4 / 3 * ld * (x + 2)
        - 4 * gd * (x + 1)
    )
    h12: float = -2 * np.sqrt(x) * (b - 2 * d * (x + 1) - g / 2 - 2 / 3 * ld - gd / 2 * (x + 4))
    h21: float = h12
    h22: float = b * x - d * (x**2 + 4 * x) + 2 / 3 * l - g + 2 / 3 * x * ld - 3 * x * gd

    hamiltonian: np.ndarray = np.array([[h11, h12], [h21, h22]])
    f1, f3 = np.linalg.eigvals(hamiltonian)

    match branch_idx:
        case 1:
            return f1
        case 2:
            return b * x - d * x**2 + 2 / 3 * l - g + 2 / 3 * x * ld - x * gd
        case 3:
            return f3
        case _:
            raise ValueError(f"Invalid branch index: {branch_idx}")


def n2j_qn(n_qn: int, branch_idx: int) -> int:
    """
    Converts from N to J.
    """

    # For Hund's case (b), spin multiplicity 3.
    match branch_idx:
        case 1:
            # F1: J = N + 1
            return n_qn + 1
        case 2:
            # F2: J = N
            return n_qn
        case 3:
            # F3: J = N - 1
            return n_qn - 1
        case _:
            raise ValueError(f"Unknown branch index: {branch_idx}.")


def broadening_fn(wavenumbers: np.ndarray, line: RotationalLine):
    """
    Returns the contribution of a single rotational line to the total spectra using a Voigt
    probability density function.
    """

    gaussian, lorentzian = line.fwhm_params()

    faddeeva: np.ndarray = ((wavenumbers - line.wavenumber) + 1j * lorentzian) / (
        gaussian * np.sqrt(2)
    )

    return np.real(wofz(faddeeva)) / (gaussian * np.sqrt(2 * np.pi))


def convolve_brod(lines: list[RotationalLine], wavenumbers_conv: np.ndarray) -> np.ndarray:
    """
    Convolves a discrete number of spectral lines into a continuous spectra by applying a broadening
    function.
    """

    # TODO: 10/21/24 - Go over this and all the broadening functions to make sure they're being
    #       computed efficiently, not sure if this is the best way to do it.

    intensities_conv: np.ndarray = np.zeros_like(wavenumbers_conv)

    for line in lines:
        intensities_conv += line.intensity * broadening_fn(wavenumbers_conv, line)

    return intensities_conv


def main() -> None:
    """
    Entry point.
    """

    molecule: Molecule = Molecule(name="O2", atom_1=Atom("O"), atom_2=Atom("O"))

    state_up: ElectronicState = ElectronicState(
        name="B3Su-", spin_multiplicity=3, molecule=molecule
    )
    state_lo: ElectronicState = ElectronicState(
        name="X3Sg-", spin_multiplicity=3, molecule=molecule
    )

    vib_bands: list[tuple[int, int]] = [(2, 0)]

    sim: Simulation = Simulation(
        sim_type=SimulationType.ABSORPTION,
        molecule=molecule,
        state_up=state_up,
        state_lo=state_lo,
        rot_lvls=np.arange(0, 36),
        temperature=300.0,
        pressure=101325.0,
        vib_bands=vib_bands,
    )

    wns_line = sim.vib_bands[0].wavenumbers_line()
    ins_line = sim.vib_bands[0].intensities_line()
    wns_conv = sim.vib_bands[0].wavenumbers_conv()
    ins_conv = sim.vib_bands[0].intensities_conv()

    ins_line /= ins_line.max()
    ins_conv /= ins_conv.max()

    sample: np.ndarray = np.genfromtxt(
        "../data/samples/harvard_20.csv", delimiter=",", skip_header=1
    )

    plt.plot(sample[:, 0], sample[:, 1] / sample[:, 1].max(), color="orange")
    plt.stem(wns_line, ins_line, markerfmt="")
    plt.plot(wns_conv, ins_conv)
    plt.show()


if __name__ == "__main__":
    main()
