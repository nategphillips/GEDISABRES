# module main
"""
A simulation of the Schumann-Runge bands of molecular oxygen written in Python.
"""

from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np

from atom import Atom
from molecule import Molecule
from sim import Sim
from simtype import SimType
from state import State

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


def main() -> None:
    """
    Entry point.
    """

    molecule: Molecule = Molecule(name="O2", atom_1=Atom("O"), atom_2=Atom("O"))

    state_up: State = State(name="B3Su-", spin_multiplicity=3, molecule=molecule)
    state_lo: State = State(name="X3Sg-", spin_multiplicity=3, molecule=molecule)

    vib_bands: list[tuple[int, int]] = [(2, 0), (4, 1)]

    # TODO: 10/25/24 - Implement an option for switching between equilibrium and nonequilibrium
    #       simulations.

    temp: float = 300.0

    sim: Sim = Sim(
        sim_type=SimType.ABSORPTION,
        molecule=molecule,
        state_up=state_up,
        state_lo=state_lo,
        rot_lvls=np.arange(0, 40),
        temp_trn=temp,
        temp_elc=temp,
        temp_vib=temp,
        temp_rot=temp,
        pressure=101325.0,
        vib_bands=vib_bands,
    )

    sample: np.ndarray = np.genfromtxt(
        "../data/samples/harvard_20.csv", delimiter=",", skip_header=1
    )
    wns_samp = sample[:, 0]
    ins_samp = sample[:, 1] / sample[:, 1].max()

    plt.plot(wns_samp, ins_samp, color="white")

    max_total: float = 0.0

    for band in sim.vib_bands:
        # Get the maximum intensity for each band in the simulation.
        max_band: float = band.intensities_conv().max()

        # Find the maximum intensity across all bands.
        if band.intensities_conv().max() > max_total:
            max_total = max_band

    # Plot all bands normalized to one while conserving the relative intensities between bands.
    for band in sim.vib_bands:
        plt.plot(band.wavenumbers_conv(), band.intensities_conv() / max_total)

    # Convolve all bands together and normalize to one.
    wns, ins = sim.all_conv_data()
    ins /= ins.max()

    plt.plot(wns, ins)

    # Interpolate simulated data to have the same number of points as the experimental data and
    # compute the residual.
    ins_inrp: np.ndarray = np.interp(sample[:, 0], wns, ins)
    residual: np.ndarray = np.abs(ins_samp - ins_inrp)

    # Show residual below the main data for clarity.
    plt.plot(wns_samp, -residual)
    plt.show()


if __name__ == "__main__":
    main()
