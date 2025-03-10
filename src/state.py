# module state
"""Contains the implementation of the State class."""

from pathlib import Path

import polars as pl

from molecule import Molecule


class State:
    """Represents an electronic state of a particular molecule."""

    def __init__(self, name: str, spin_multiplicity: int, molecule: Molecule) -> None:
        """Initialize class variables.

        Args:
            name (str): Name of the electronic state.
            spin_multiplicity (int): Spin multiplicity.
            molecule (Molecule): Parent molecule.
        """
        self.name: str = name
        self.spin_multiplicity: int = spin_multiplicity
        self.molecule: Molecule = molecule
        self.constants: dict[str, list[float]] = self.get_constants(molecule.name, name)

    @staticmethod
    def get_constants(molecule: str, state: str) -> dict[str, list[float]]:
        """Return the molecular constants for the specified electronic state in [1/cm]."""
        return pl.read_csv(Path("..", "data", molecule, "states", f"{state}.csv")).to_dict(
            as_series=False
        )

    def is_allowed(self, n_qn: int) -> bool:
        """Return whether or not the selected rotational level is allowed."""
        if self.name == "X3Sg-":
            # For X3Σg-, only the rotational levels with odd N can be populated.
            return bool(n_qn % 2 == 1)
        if self.name == "B3Su-":
            # For B3Σu-, only the rotational levels with even N can be populated.
            return bool(n_qn % 2 == 0)

        raise ValueError(f"State {self.name} not supported.")
