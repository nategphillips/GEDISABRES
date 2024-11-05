# module atom
"""
Contains the implementation of the Atom class.
"""

import constants


class Atom:
    """
    Represents an atom with a name and mass.
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.mass: float = self.get_mass(name) / constants.AVOGD / 1e3

    @staticmethod
    def get_mass(name: str) -> float:
        """
        Returns the atomic mass in [g/mol].
        """

        if name not in constants.ATOMIC_MASSES:
            raise ValueError(f"Atom `{name}` not supported.")

        return constants.ATOMIC_MASSES[name]
