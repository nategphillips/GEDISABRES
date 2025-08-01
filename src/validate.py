"""Compares the output of pyGEONOSIS to that of PGOPHER for selected molecules."""

# Copyright (C) 2023-2025 Nathan G. Phillips

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

import matplotlib.pyplot as plt
import polars as pl

from atom import Atom
from enums import ConstantsType, InversionSymmetry, ReflectionSymmetry, SimType, TermSymbol
from molecule import Molecule
from sim import Sim
from state import State


def o2(sample_name: str):
    molecule: Molecule = Molecule(name="O2", atom_1=Atom("O"), atom_2=Atom("O"))
    state_up: State = State(
        molecule=molecule,
        letter="B",
        spin_multiplicity=3,
        term_symbol=TermSymbol.SIGMA,
        inversion_symmetry=InversionSymmetry.UNGERADE,
        reflection_symmetry=ReflectionSymmetry.MINUS,
        constants_type=ConstantsType.PERLEVEL,
    )
    state_lo: State = State(
        molecule=molecule,
        letter="X",
        spin_multiplicity=3,
        term_symbol=TermSymbol.SIGMA,
        inversion_symmetry=InversionSymmetry.GERADE,
        reflection_symmetry=ReflectionSymmetry.MINUS,
        constants_type=ConstantsType.PERLEVEL,
    )

    temp: float = 300.0

    sim: Sim = Sim(
        sim_type=SimType.ABSORPTION,
        molecule=molecule,
        state_up=state_up,
        state_lo=state_lo,
        j_qn_up_max=48,
        temp_trn=temp,
        temp_elc=temp,
        temp_vib=temp,
        temp_rot=temp,
        pressure=101325,
        bands_input=[(2, 0)],
    )

    plot(sim, sample_name)


def no_ax(sample_name: str):
    molecule: Molecule = Molecule(name="NO", atom_1=Atom("N"), atom_2=Atom("O"))
    state_up: State = State(
        molecule=molecule,
        letter="A",
        spin_multiplicity=2,
        term_symbol=TermSymbol.SIGMA,
        inversion_symmetry=InversionSymmetry.NONE,
        reflection_symmetry=ReflectionSymmetry.PLUS,
        constants_type=ConstantsType.DUNHAM,
    )
    state_lo: State = State(
        molecule=molecule,
        letter="X",
        spin_multiplicity=2,
        term_symbol=TermSymbol.PI,
        inversion_symmetry=InversionSymmetry.NONE,
        reflection_symmetry=ReflectionSymmetry.NONE,
        constants_type=ConstantsType.DUNHAM,
    )

    temp: float = 300.0

    sim: Sim = Sim(
        sim_type=SimType.ABSORPTION,
        molecule=molecule,
        state_up=state_up,
        state_lo=state_lo,
        j_qn_up_max=48,
        temp_trn=temp,
        temp_elc=temp,
        temp_vib=temp,
        temp_rot=temp,
        pressure=101325,
        bands_input=[(0, 0)],
    )

    plot(sim, sample_name)


def no_bx(sample_name: str):
    molecule: Molecule = Molecule(name="NO", atom_1=Atom("N"), atom_2=Atom("O"))
    state_up: State = State(
        molecule=molecule,
        letter="B",
        spin_multiplicity=2,
        term_symbol=TermSymbol.PI,
        inversion_symmetry=InversionSymmetry.NONE,
        reflection_symmetry=ReflectionSymmetry.NONE,
        constants_type=ConstantsType.DUNHAM,
    )
    state_lo: State = State(
        molecule=molecule,
        letter="X",
        spin_multiplicity=2,
        term_symbol=TermSymbol.PI,
        inversion_symmetry=InversionSymmetry.NONE,
        reflection_symmetry=ReflectionSymmetry.NONE,
        constants_type=ConstantsType.DUNHAM,
    )

    temp: float = 300.0

    sim: Sim = Sim(
        sim_type=SimType.ABSORPTION,
        molecule=molecule,
        state_up=state_up,
        state_lo=state_lo,
        j_qn_up_max=48,
        temp_trn=temp,
        temp_elc=temp,
        temp_vib=temp,
        temp_rot=temp,
        pressure=101325,
        bands_input=[(0, 0)],
    )

    plot(sim, sample_name)


def oh(sample_name: str):
    molecule: Molecule = Molecule(name="OH", atom_1=Atom("O"), atom_2=Atom("H"))
    state_up: State = State(
        molecule=molecule,
        letter="A",
        spin_multiplicity=2,
        term_symbol=TermSymbol.SIGMA,
        inversion_symmetry=InversionSymmetry.NONE,
        reflection_symmetry=ReflectionSymmetry.PLUS,
        constants_type=ConstantsType.DUNHAM,
    )
    state_lo: State = State(
        molecule=molecule,
        letter="X",
        spin_multiplicity=2,
        term_symbol=TermSymbol.PI,
        inversion_symmetry=InversionSymmetry.NONE,
        reflection_symmetry=ReflectionSymmetry.NONE,
        constants_type=ConstantsType.DUNHAM,
    )

    temp: float = 300.0

    sim: Sim = Sim(
        sim_type=SimType.ABSORPTION,
        molecule=molecule,
        state_up=state_up,
        state_lo=state_lo,
        j_qn_up_max=48,
        temp_trn=temp,
        temp_elc=temp,
        temp_vib=temp,
        temp_rot=temp,
        pressure=101325,
        bands_input=[(0, 0)],
    )

    plot(sim, sample_name)


def plot(sim: Sim, sample_name: str):
    wns_sim, ins_sim = sim.all_line_data()
    ins_sim /= ins_sim.max()

    plt.stem(wns_sim, ins_sim, "r", label="pyGEONOSIS", markerfmt="")

    sample = (
        pl.read_csv(f"../data/samples/{sample_name}.csv").select("Position", "Intensity").to_numpy()
    )
    wns_smp = sample[:, 0]
    ins_smp = sample[:, 1] / sample[:, 1].max()

    plt.stem(wns_smp, -ins_smp, "b", label="PGOPHER", markerfmt="")
    plt.title(
        f"{sim.molecule.name}: {sim.state_up.name} v' = {sim.bands_input[0][0]} to {sim.state_lo.name} v'' = {sim.bands_input[0][1]} Transition"
    )
    plt.legend()
    plt.show()


def main() -> None:
    """Entry point."""
    o2("pgopher-o2-bx-20")
    no_ax("pgopher-no-ax-00")
    no_bx("pgopher-no-bx-00")
    oh("pgopher-oh-ax-00")


if __name__ == "__main__":
    main()
