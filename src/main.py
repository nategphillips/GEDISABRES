# module main
"""
A simulation of the Schumann-Runge bands of molecular oxygen written in Python.
"""

from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

import constants as cn


class Atom:
    """
    Represents an atom with a name and mass.
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.mass: float = self.lookup_mass(name) / cn.AVOGD / 1e3

    @staticmethod
    def lookup_mass(name: str) -> float:
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
        self.reduced_mass: float = self.atom_1.mass * self.atom_2.mass / self.mass
        self.symmetry_parameter: int = self.get_symmetry_parameter(atom_1, atom_2)

    @staticmethod
    def get_symmetry_parameter(atom_1: Atom, atom_2: Atom) -> int:
        """
        Returns the symmetry parameter of the molecule.
        """

        # Hanson p. 17
        # For homonuclear diatomic molecules like O2, the symmetry parameter is 2
        if atom_1.name == atom_2.name:
            return 2

        # For heteronuclear diatomics, it's 1
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
        self.constants: dict[str, float] = self.lookup_constants(molecule.name, name)

    @staticmethod
    def lookup_constants(molecule: str, state: str) -> dict[str, float]:
        """
        Returns a dictionary of molecular constants for the specified electronic state in [1/cm].
        """

        if molecule not in cn.MOLECULAR_CONSTANTS:
            raise ValueError(f"Molecule `{molecule}` not supported.")
        if state not in cn.MOLECULAR_CONSTANTS[molecule]:
            raise ValueError(f"State `{state}` in molecule `{molecule}` not supported.")

        return cn.MOLECULAR_CONSTANTS[molecule][state]

    def is_allowed(self, n_qn: int) -> bool:
        """
        Returns a boolean value corresponding to whether or not the selected rotational level is
        allowed.
        """

        if self.name == "X3Sg-":
            # For X3Sg-, only the rotational levels with odd N can be populated
            return bool(n_qn % 2 == 1)
        if self.name == "B3Su-":
            # For B3Su-, only the rotational levels with even N can be populated
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
        vib_bands: list[tuple[int, int]],
        rot_lvls: np.ndarray,
        temperature: float,
        pressure: float,
    ) -> None:
        self.sim_type: SimulationType = sim_type
        self.molecule: Molecule = molecule
        self.state_up: ElectronicState = state_up
        self.state_lo: ElectronicState = state_lo
        self.vib_bands: list[VibrationalBand] = self.get_vib_bands(vib_bands)
        self.rot_lvls: np.ndarray = rot_lvls
        self.temperature: float = temperature
        self.pressure: float = pressure

    def get_vib_bands(self, vib_bands: list[tuple[int, int]]):
        """
        Returns the selected vibrational bands within the simulation.
        """

        return [
            VibrationalBand(sim=self, v_qn_up=vib_band[0], v_qn_lo=vib_band[1])
            for vib_band in vib_bands
        ]


class VibrationalBand:
    """
    Represents a vibrational band of a particular molecule.
    """

    def __init__(self, sim: Simulation, v_qn_up: int, v_qn_lo: int) -> None:
        self.sim: Simulation = sim
        self.v_qn_up: int = v_qn_up
        self.v_qn_lo: int = v_qn_lo

    def rotational_partition_function(self) -> float:
        """
        Returns the rotational partition function.
        """

        q_r: float = 0.0

        for line in self.get_rotational_lines():
            q_r += line.rotational_boltzmann_factor()

        return q_r / self.sim.molecule.symmetry_parameter

    def band_origin(self) -> float:
        """
        Returns the band origin in [1/cm].
        """

        # Herzberg p. 151, eq. (IV, 12)

        electronic_energy: float = (
            self.sim.state_up.constants["T_e"] - self.sim.state_lo.constants["T_e"]
        )

        vibrational_energy: float = self.vibrational_term(
            self.sim.state_up, self.v_qn_up
        ) - self.vibrational_term(self.sim.state_lo, self.v_qn_lo)

        return electronic_energy + vibrational_energy

    def vibrational_term(self, state: ElectronicState, v_qn: int) -> float:
        """
        Returns the vibrational term value in [1/cm].
        """

        # Herzberg p. 149, eq. (IV, 10)

        return (
            state.constants["w_e"] * (v_qn + 0.5)
            - state.constants["we_xe"] * (v_qn + 0.5) ** 2
            + state.constants["we_ye"] * (v_qn + 0.5) ** 3
            + state.constants["we_ze"] * (v_qn + 0.5) ** 4
        )

    def get_rotational_lines(self):
        """
        Returns a list of all allowed rotational lines.
        """

        lines = []

        for n_qn_up in self.sim.rot_lvls:
            for n_qn_lo in self.sim.rot_lvls:
                # Ensure the rotational selection rules corresponding to each electronic state are
                # properly followed
                if self.sim.state_up.is_allowed(n_qn_up) & self.sim.state_lo.is_allowed(n_qn_lo):
                    lines.extend(self.get_allowed_branches(n_qn_up, n_qn_lo))

        return lines

    def get_allowed_branches(self, n_qn_up: int, n_qn_lo: int):
        """
        Determines the selection rules for Hund's case (b).
        """

        # ∆N = ±1, ∆N = 0 is forbidden for Σ-Σ transitions
        # Herzberg p. 244, eq. (V, 44)

        lines = []

        # Determine how many lines should be present in the fine structure of the molecule due to
        # the effects of spin multiplicity
        if self.sim.state_up.spin_multiplicity == self.sim.state_lo.spin_multiplicity:
            branch_range = range(1, self.sim.state_up.spin_multiplicity + 1)
        else:
            raise ValueError("Spin multiplicity of the two electronic states do not match.")

        delta_n_qn = n_qn_up - n_qn_lo

        # R branch
        if delta_n_qn == 1:
            lines.extend(self.get_branch_idx(n_qn_up, n_qn_lo, branch_range, "R"))
        # Q branch (doesn't exist for the Schumann-Runge bands of O2 due to the allowed transitions
        # for the electronic states)
        if delta_n_qn == 0:
            lines.extend(self.get_branch_idx(n_qn_up, n_qn_lo, branch_range, "Q"))
        # P branch
        elif delta_n_qn == -1:
            lines.extend(self.get_branch_idx(n_qn_up, n_qn_lo, branch_range, "P"))

        return lines

    def get_branch_idx(self, n_qn_up: int, n_qn_lo: int, branch_range: range, branch_name: str):
        """
        Returns the rotational lines within a given branch.
        """

        def add_line(branch_idx_up: int, branch_idx_lo: int, is_satellite: bool):
            """
            Helper to create and append a rotational line.
            """

            # print(f"{n_qn_up} {n_qn_lo}: {branch_name} {branch_idx_up} {branch_idx_lo}")

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
        #       N' = 0 to N'' = 1 transition, which has 3 total lines (1 main + 2 satellite)

        lines = []

        # Handle the special case where N' = 0 (only the P1, PQ12, and PQ13 lines exist)
        if n_qn_up == 0:
            if branch_name == "P":
                add_line(1, 1, False)
            for branch_idx_lo in (2, 3):
                add_line(1, branch_idx_lo, True)

            return lines

        # Handle regular cases for other N'
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


def n2j_qn(n_qn: int, branch_idx: int) -> int:
    """
    Converts from N to J.
    """

    # For Hund's case (b), spin multiplicity 3
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


@dataclass
class RotationalLine:
    """
    Represents a rotational line within a vibrational band.
    """

    sim: Simulation
    band: VibrationalBand
    n_qn_up: int
    n_qn_lo: int
    j_qn_up: int
    j_qn_lo: int
    branch_idx_up: int
    branch_idx_lo: int
    branch_name: str
    is_satellite: bool

    def rotational_boltzmann_factor(self) -> float:
        """
        Returns the rotational Boltzmann factor, that is: (2J + 1) * exp(-F(J) * h * c / (k * T)).
        This is different than the rotational Boltzmann fraction, which is the Boltzmann factor
        divided by the total rotational partition function.
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

        return (2 * j_qn + 1) * np.exp(
            -self.rotational_term(state, v_qn, j_qn, branch_idx)
            * cn.PLANC
            * cn.LIGHT
            / (cn.BOLTZ * self.sim.temperature)
        )

    def intensity(self) -> float:
        """
        Returns the intensity.
        """

        # NOTE: 10/18/24 - Before going any further make sure to read Herzberg pp. 20-21,
        #       pp. 126-127, pp. 200-201, and pp. 382-383

        match self.sim.sim_type:
            case SimulationType.EMISSION:
                j_qn = self.j_qn_up
                wavenumber_factor = self.wavenumber() ** 4
            case SimulationType.ABSORPTION:
                j_qn = self.j_qn_lo
                wavenumber_factor = self.wavenumber()

        boltzmann_fraction: float = (
            self.rotational_boltzmann_factor() / self.band.rotational_partition_function()
        )

        return wavenumber_factor * self.honl_london_factor() / (2 * j_qn + 1) * boltzmann_fraction

    def honl_london_factor(self) -> float:
        """
        Returns the Hönl-London factor (line strength).
        """

        # For emission, the relevant rotational quantum number is N'; for absorption, it's N''
        match self.sim.sim_type:
            case SimulationType.EMISSION:
                n_qn = self.n_qn_up
            case SimulationType.ABSORPTION:
                n_qn = self.n_qn_lo

        # Convert the properties of the current rotational line into a useful key
        if self.is_satellite:
            key = f"{self.branch_name}Q{self.branch_idx_up}{self.branch_idx_lo}"
        else:
            # For main branches, the upper and lower branches indicies are the same, so it doesn't
            # matter which one is used here
            key = f"{self.branch_name}{self.branch_idx_up}"

        # Factors are from Tatum - 1966: Hönl-London Factors for 3Σ±-3Σ± Transitions
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

    def rotational_term(
        self, state: ElectronicState, v_qn: int, j_qn: int, branch_idx: int
    ) -> float:
        """
        Testing new Hamiltonian from Cheung.
        """

        # TODO: 10/17/24 - Change molecular constant data for the ground state to reflect the
        #       conventions in Cheung
        #       Yu      | Cheung
        #       --------|------------
        #       D       | -D
        #       lamda_D | lamda_D / 2
        #       gamma_D | gamma_D / 2

        match state.name:
            case "B3Su-":
                lookup = cn.CONSTS_UP
            case "X3Sg-":
                lookup = cn.CONSTS_LO

        b: float = lookup["B"][v_qn]
        d: float = lookup["D"][v_qn]
        l: float = lookup["lamda"][v_qn]
        g: float = lookup["gamma"][v_qn]
        ld: float = lookup["lamda_D"][v_qn]
        gd: float = lookup["gamma_D"][v_qn]

        x: int = j_qn * (j_qn + 1)

        # Hamiltonian matrix elements
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

    def wavenumber(self) -> float:
        """
        Testing new Hamiltonian from Cheung.
        """

        # NOTE: 10/18/24 - Make sure to understand transition structure: Herzberg pp. 149-152, and
        #       pp. 168-169

        # Converts Cheung's definition of the band origin (T) to Herzberg's definition (nu_0)
        energy_offset: float = (
            2 / 3 * cn.CONSTS_UP["lamda"][self.band.v_qn_up]
            - cn.CONSTS_UP["gamma"][self.band.v_qn_up]
        )

        # Herzberg p. 168, eq. (IV, 24)
        return (
            cn.CONSTS_UP["T"][self.band.v_qn_up]
            + energy_offset
            + self.rotational_term(
                self.sim.state_up, self.band.v_qn_up, self.j_qn_up, self.branch_idx_up
            )
            - self.rotational_term(
                self.sim.state_lo, self.band.v_qn_lo, self.j_qn_lo, self.branch_idx_lo
            )
        )

    # def wavenumber(self) -> float:
    #     """
    #     Returns the wavenumber in [1/cm].
    #     """

    #     # Herzberg p. 168, eq. (IV, 24)

    #     return (
    #         self.band.band_origin()
    #         + self.rotational_term(
    #             self.sim.state_up, self.j_qn_up, self.band.v_qn_up, self.branch_idx_up
    #         )
    #         - self.rotational_term(
    #             self.sim.state_lo, self.j_qn_lo, self.band.v_qn_lo, self.branch_idx_lo
    #         )
    #     )

    # def rotational_term(self, state: ElectronicState, j_qn: int, v_qn: int, branch_idx: int) -> float:
    #     """
    #     Returns the rotational term value in [1/cm].
    #     """

    #     # Rotational constants
    #     # Herzberg pp. 107-109, eqs. (III, 117-127)

    #     b_v: float = (
    #         state.constants["B_e"]
    #         - state.constants["alpha_e"] * (v_qn + 0.5)
    #         + state.constants["gamma_e"] * (v_qn + 0.5) ** 2
    #         + state.constants["delta_e"] * (v_qn + 0.5) ** 3
    #     )
    #     lamda: float = state.constants["lamda"]
    #     gamma: float = state.constants["gamma"]

    #     # Shorthand notation for the rotational quantum number
    #     x = j_qn * (j_qn + 1)

    #     # Schlapp, 1936 - Fine Structure in the 3Σ Ground State of the Oxygen Molecule
    #     # From matrix elements - "precise" values
    #     f1: float = (
    #         b_v * x + b_v - lamda - np.sqrt((b_v - lamda) ** 2 + (b_v - gamma / 2) ** 2 * 4 * x)
    #     )
    #     f2: float = b_v * x
    #     f3: float = (
    #         b_v * x + b_v - lamda + np.sqrt((b_v - lamda) ** 2 + (b_v - gamma / 2) ** 2 * 4 * x)
    #     )

    #     # NOTE: 10/15/24 - For J = 0, the energy is -2 * lamd + b * rot_qn * (rot_qn + 1) + 2 * b
    #     #       (Hougen: The Calculation of Rotational Energy Levels in Diatomic Molecules, p. 15)

    #     if j_qn == 0:
    #         f3 = -2 * lamda + b_v * x + 2 * b_v

    #     match branch_idx:
    #         case 1:
    #             return f1
    #         case 2:
    #             return f2
    #         case 3:
    #             return f3
    #         case _:
    #             raise ValueError(f"Invalid branch index: {branch_idx}")


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
        vib_bands=vib_bands,
        rot_lvls=np.arange(0, 50),
        temperature=300.0,
        pressure=101325.0,
    )

    wavenumbers: list[float] = []
    intensities: list[float] = []
    for line in sim.vib_bands[0].get_rotational_lines():
        wavenumbers.append(line.wavenumber())
        intensities.append(line.intensity())

    intn = np.array(intensities)
    intn /= intn.max()

    sample: np.ndarray = np.genfromtxt(
        "../data/samples/harvard_20.csv", delimiter=",", skip_header=1
    )

    plt.stem(wavenumbers, intn, markerfmt="")
    plt.plot(sample[:, 0], sample[:, 1] / sample[:, 1].max(), color="orange")
    plt.show()


if __name__ == "__main__":
    main()
