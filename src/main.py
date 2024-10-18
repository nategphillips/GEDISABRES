# module main
"""
A simulation of the Schumann-Runge bands of molecular oxygen written in Python.
"""

from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import constants as cn


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
        self.constants: dict[str, dict[int, float]] = self.get_constants(molecule.name, name)

    @staticmethod
    def get_constants(molecule: str, state: str) -> dict[str, dict[int, float]]:
        """
        Returns a dictionary of molecular constants for the specified electronic state in [1/cm].
        """

        # TODO: 10/18/24 - Add errors here if the molecule or state is not supported

        return pd.read_csv(f"../data/constants/{molecule}/{state}.csv").to_dict()

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
        self.vib_bands: list[VibrationalBand] = self.get_vib_bands(vib_bands)

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
        self.band_origin: float = self.get_band_origin()
        self.lines: list[RotationalLine] = self.get_rotational_lines()
        self.rot_part: float = self.get_rotational_partition_function()

    def get_band_origin(self) -> float:
        """
        Returns the band origin in [1/cm].
        """

        # Herzberg p. 168, eq. (IV, 24)

        upper_state: dict[str, dict[int, float]] = self.sim.state_up.constants

        # Converts Cheung's definition of the band origin (T) to Herzberg's definition (nu_0)
        energy_offset: float = (
            2 / 3 * upper_state["lamda"][self.v_qn_up] - upper_state["gamma"][self.v_qn_up]
        )

        return upper_state["T"][self.v_qn_up] + energy_offset

    def get_rotational_partition_function(self) -> float:
        """
        Returns the rotational partition function.
        """

        return sum(line.rotational_boltzmann_factor for line in self.lines)

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
                    lines.extend(self.allowed_branches(n_qn_up, n_qn_lo))

        return lines

    def allowed_branches(self, n_qn_up: int, n_qn_lo: int):
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
            lines.extend(self.branch_index(n_qn_up, n_qn_lo, branch_range, "R"))
        # Q branch
        if delta_n_qn == 0:
            # Note that the Q branch doesn't exist for the Schumann-Runge bands of O2
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
        self.rotational_boltzmann_factor: float = self.get_rotational_boltzmann_factor()
        # TODO: 10/18/24 - This doesn't work because of a circular dependency: the line needs the
        #       rotational partition function from the band, which in turn needs the Boltzmann
        #       factor for each line
        # self.intensity: float = self.get_intensity()

    def get_rotational_boltzmann_factor(self) -> float:
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

    def get_intensity(self) -> float:
        """
        Returns the intensity.
        """

        # NOTE: 10/18/24 - Before going any further make sure to read Herzberg pp. 20-21,
        #       pp. 126-127, pp. 200-201, and pp. 382-383

        match self.sim.sim_type:
            case SimulationType.EMISSION:
                j_qn = self.j_qn_up
                wavenumber_factor = self.wavenumber**4
            case SimulationType.ABSORPTION:
                j_qn = self.j_qn_lo
                wavenumber_factor = self.wavenumber

        boltzmann_fraction: float = self.rotational_boltzmann_factor / self.band.rot_part

        return wavenumber_factor * self.honl_london_factor / (2 * j_qn + 1) * boltzmann_fraction

    def get_honl_london_factor(self) -> float:
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

        lookup: dict = state.constants

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

    def get_wavenumber(self) -> float:
        """
        Testing new Hamiltonian from Cheung.
        """

        # NOTE: 10/18/24 - Make sure to understand transition structure: Herzberg pp. 149-152, and
        #       pp. 168-169

        # Herzberg p. 168, eq. (IV, 24)
        return (
            self.band.band_origin
            + self.rotational_term(
                self.sim.state_up, self.band.v_qn_up, self.j_qn_up, self.branch_idx_up
            )
            - self.rotational_term(
                self.sim.state_lo, self.band.v_qn_lo, self.j_qn_lo, self.branch_idx_lo
            )
        )


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
        rot_lvls=np.arange(0, 50),
        temperature=300.0,
        pressure=101325.0,
        vib_bands=vib_bands,
    )

    wavenumbers: list[float] = []
    intensities: list[float] = []
    for line in sim.vib_bands[0].lines:
        wavenumbers.append(line.wavenumber)
        intensities.append(line.get_intensity())

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
