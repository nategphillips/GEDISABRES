# module molecule
"""
Contains the implementation of the Molecule class.
"""

from atom import Atom


class Molecule:
    """
    Represents a diatomic molecule consisting of two atoms.
    """

    def __init__(self, name: str, atom_1: Atom, atom_2: Atom) -> None:
        self.name: str = name
        self.atom_1: Atom = atom_1
        self.atom_2: Atom = atom_2
        self.mass: float = self.atom_1.mass + self.atom_2.mass
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
