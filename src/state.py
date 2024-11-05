# module state
"""
Contains the implementation of the State class.
"""

import pandas as pd

from molecule import Molecule


class State:
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
