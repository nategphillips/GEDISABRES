# module sim
"""
Contains the implementation of the Sim class.
"""

import numpy as np
import pandas as pd

from band import Band
import constants
import convolve
from line import Line
from molecule import Molecule
import params
from simtype import SimType
from state import State
import terms


class Sim:
    """
    Simulates the spectra of a particular molecule.
    """

    def __init__(
        self,
        sim_type: SimType,
        molecule: Molecule,
        state_up: State,
        state_lo: State,
        rot_lvls: np.ndarray,
        temp_trn: float,
        temp_elc: float,
        temp_vib: float,
        temp_rot: float,
        pressure: float,
        bands: list[tuple[int, int]],
    ) -> None:
        self.sim_type: SimType = sim_type
        self.molecule: Molecule = molecule
        self.state_up: State = state_up
        self.state_lo: State = state_lo
        self.rot_lvls: np.ndarray = rot_lvls
        self.temp_trn: float = temp_trn
        self.temp_elc: float = temp_elc
        self.temp_vib: float = temp_vib
        self.temp_rot: float = temp_rot
        self.pressure: float = pressure
        self.elc_part: float = self.get_elc_partition_fn()
        self.vib_part: float = self.get_vib_partition_fn()
        self.elc_boltz_frac: float = self.get_elc_boltz_frac()
        self.franck_condon: np.ndarray = self.get_franck_condon()
        self.einstein: np.ndarray = self.get_einstein()
        self.predissociation: dict[str, dict[int, float]] = self.get_predissociation()
        self.bands: list[Band] = self.get_bands(bands)

    def all_line_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Combines the line data for all vibrational bands.
        """

        wavenumbers_line: np.ndarray = np.array([])
        intensities_line: np.ndarray = np.array([])

        for band in self.bands:
            wavenumbers_line = np.concatenate((wavenumbers_line, band.wavenumbers_line()))
            intensities_line = np.concatenate((intensities_line, band.intensities_line()))

        return wavenumbers_line, intensities_line

    def all_conv_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Creates common axes for plotting the convolved data of all vibrational bands at once.
        """

        wavenumbers_line: np.ndarray = np.array([])
        intensities_line: np.ndarray = np.array([])
        lines: list[Line] = []

        for band in self.bands:
            wavenumbers_line = np.concatenate((wavenumbers_line, band.wavenumbers_line()))
            intensities_line = np.concatenate((intensities_line, band.intensities_line()))
            lines.extend(band.lines)

        wavenumbers_conv = np.linspace(
            wavenumbers_line.min(), wavenumbers_line.max(), params.GRANULARITY
        )
        intensities_conv = convolve.convolve_brod(lines, wavenumbers_conv)

        return wavenumbers_conv, intensities_conv

    def get_predissociation(self) -> dict[str, dict[int, float]]:
        """
        Returns polynomial coefficients for computing predissociation linewidths.
        """

        return pd.read_csv(
            f"../data/{self.molecule.name}/predissociation/lewis_coeffs.csv"
        ).to_dict()

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

        return np.loadtxt(
            f"../data/{self.molecule.name}/franck-condon/{self.state_up.name}_to_{self.state_lo.name}_cheung.csv",
            delimiter=",",
        )

    def get_bands(self, bands: list[tuple[int, int]]):
        """
        Returns the selected vibrational bands within the simulation.
        """

        return [Band(sim=self, v_qn_up=band[0], v_qn_lo=band[1]) for band in bands]

    def get_vib_partition_fn(self) -> float:
        """
        Returns the vibrational partition function.
        """

        # NOTE: 10/22/24 - The maximum vibrational quantum number is dictated by the tabulated data
        #       available.
        match self.sim_type:
            case SimType.EMISSION:
                state = self.state_up
                v_qn_max = len(self.state_up.constants["G"])
            case SimType.ABSORPTION:
                state = self.state_lo
                v_qn_max = len(self.state_lo.constants["G"])

        q_v: float = 0.0

        # NOTE: 10/22/24 - The vibrational partition function is always computed using a set number
        #       of vibrational bands to ensure an accurate estimate of the state sum is obtained.
        #       This approach is used to ensure the sum is calculated correctly regardless of the
        #       number of vibrational bands simulated by the user.
        for v_qn in range(0, v_qn_max):
            # NOTE: 10/25/24 - The zero-point vibrational energy is used as a reference to which all
            #       other vibrational energies are measured. This ensures the state sum begins at a
            #       value of 1 when v = 0.
            q_v += np.exp(
                -(terms.vibrational_term(state, v_qn) - terms.vibrational_term(state, 0))
                * constants.PLANC
                * constants.LIGHT
                / (constants.BOLTZ * self.temp_vib)
            )

        return q_v

    def get_elc_partition_fn(self) -> float:
        """
        Returns the electronic partition function.
        """

        energies: list[float] = list(constants.ELECTRONIC_ENERGIES[self.molecule.name].values())
        degeneracies: list[int] = list(
            constants.ELECTRONIC_DEGENERACIES[self.molecule.name].values()
        )

        q_e: float = 0.0

        # NOTE: 10/25/24 - This sum is basically unnecessary since the energies of electronic states
        #       above the ground state are so high. This means that any contribution to the
        #       electronic partition function from anything other than the ground state is
        #       negligible.
        for e, g in zip(energies, degeneracies):
            q_e += g * np.exp(
                -e * constants.PLANC * constants.LIGHT / (constants.BOLTZ * self.temp_elc)
            )

        return q_e

    def get_elc_boltz_frac(self) -> float:
        """
        Returns the electronic Boltzmann fraction N_e / N.
        """

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
            / self.elc_part
        )
