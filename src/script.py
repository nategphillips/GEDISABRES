# module script.py
"""This module demonstrates how to use the capabilities of pyGEONOSIS without using the GUI."""

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

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

import utils
from atom import Atom
from enums import ConstantsType, InversionSymmetry, ReflectionSymmetry, SimType, TermSymbol
from molecule import Molecule
from sim import Sim
from sim_params import BroadeningBools, InstrumentParams
from state import State

if TYPE_CHECKING:
    from numpy.typing import NDArray


def main() -> None:
    """Entry point."""
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

    bands: list[tuple[int, int]] = [(2, 0), (4, 1)]

    broad_bools = BroadeningBools(
        instrument=True, doppler=True, natural=True, collisional=True, predissociation=True
    )

    inst_params = InstrumentParams(gauss_fwhm_wl=0.001, loren_fwhm_wl=0.001)

    sim: Sim = Sim(
        sim_type=SimType.ABSORPTION,
        molecule=molecule,
        state_up=state_up,
        state_lo=state_lo,
        j_qn_up_max=40,
        pressure=101325.0,
        bands_input=bands,
        inst_params=inst_params,
        broad_bools=broad_bools,
    )

    sample: NDArray[np.float64] = np.genfromtxt(
        fname=utils.get_data_path("data", "samples", "harvard-o2-bx-20.csv"),
        delimiter=",",
        skip_header=1,
    )
    wns_samp: NDArray[np.float64] = sample[:, 0]
    ins_samp: NDArray[np.float64] = sample[:, 1] / sample[:, 1].max()

    plt.plot(wns_samp, ins_samp, label="sample")

    granularity: int = int(1e4)

    # Find the max intensity in all the bands.
    max_intensity: float = max(
        band.intensities_conv(band.wavenumbers_conv(granularity)).max() for band in sim.bands
    )

    # Plot all bands normalized to one while conserving the relative intensities between bands.
    for band in sim.bands:
        plt.plot(
            band.wavenumbers_conv(granularity),
            band.intensities_conv(band.wavenumbers_conv(granularity)) / max_intensity,
            label=f"band: {band.v_qn_up, band.v_qn_lo}",
        )

    # Convolve all bands together and normalize to one.
    wns, ins = sim.all_conv_data(granularity)
    ins /= ins.max()

    plt.plot(wns, ins, label="all convolved")

    # Interpolate simulated data to have the same number of points as the experimental data and
    # compute the residual.
    ins_inrp: NDArray[np.float64] = np.interp(sample[:, 0], wns, ins)
    residual: NDArray[np.float64] = np.abs(ins_samp - ins_inrp)

    # Show residual below the main data for clarity.
    plt.plot(wns_samp, -residual, label="residual")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
