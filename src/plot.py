# module plot
"""
Contains functions used for plotting.
"""

import numpy as np
from matplotlib.axes import Axes

from line import Line
from sim import Sim
import utils


def plot_line(axs: Axes, sim: Sim, colors: list[str]) -> None:
    """
    Plots each rotational line.
    """

    for idx, band in enumerate(sim.bands):
        wavelengths_line: np.ndarray = utils.wavenum_to_wavelen(band.wavenumbers_line())
        intensities_line: np.ndarray = band.intensities_line()

        axs.stem(
            wavelengths_line,
            intensities_line,
            colors[idx],
            markerfmt="",
            label=f"{sim.molecule.name} {band.v_qn_up, band.v_qn_lo} line",
        )


def plot_line_info(axs: Axes, sim: Sim, colors: list[str]) -> None:
    """
    Plots information about each rotational line.
    """

    # In order to show text, a plot must first exist.
    plot_line(axs, sim, colors)

    for band in sim.bands:
        # Only select non-satellite lines to reduce the amount of data on screen.
        lines: list[Line] = [line for line in band.lines if not line.is_satellite]

        for line in lines:
            axs.text(
                utils.wavenum_to_wavelen(line.wavenumber),
                line.intensity,
                f"${line.branch_name}_{{{line.branch_idx_up}{line.branch_idx_lo}}}$",
            )


def plot_conv_sep(axs: Axes, sim: Sim, colors: list[str]) -> None:
    """
    Plots convolved data for each vibrational band separately.
    """

    for idx, band in enumerate(sim.bands):
        wavelengths_conv: np.ndarray = utils.wavenum_to_wavelen(band.wavenumbers_conv())
        intensities_conv: np.ndarray = band.intensities_conv()

        axs.plot(
            wavelengths_conv,
            intensities_conv,
            colors[idx],
            label=f"{sim.molecule.name} {band.v_qn_up, band.v_qn_lo} conv",
        )


def plot_conv_all(axs: Axes, sim: Sim, colors: list[str]) -> None:
    """
    Plots convolved data for all vibrational bands simultaneously.
    """

    wavenumbers_conv, intensities_conv = sim.all_conv_data()
    wavelengths_conv: np.ndarray = utils.wavenum_to_wavelen(wavenumbers_conv)

    axs.plot(wavelengths_conv, intensities_conv, colors[0], label=f"{sim.molecule.name} conv all")
