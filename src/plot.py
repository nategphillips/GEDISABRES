# module plot.py
"""Contains functions used for plotting."""

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

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray

import utils
from sim import Sim

if TYPE_CHECKING:
    from line import Line

PEN_WIDTH: int = 1


def plot_sample(
    plot_widget: pg.PlotWidget,
    x_values: NDArray[np.float64],
    intensities: NDArray[np.float64],
    display_name: str,
    value_type: str,
) -> None:
    """Plot sample data.

    Args:
        plot_widget (pg.PlotWidget): A `GraphicsView` widget with a single `PlotItem` inside.
        x_values (NDArray[np.float64]): Sample wavenumbers.
        intensities (NDArray[np.float64]): Sample intensities.
        display_name (str): The name of the file without directory information.
        value_type (str): Either wavenumbers or wavelengths.
    """
    if value_type == "wavenumber":
        x_values = utils.wavenum_to_wavelen(x_values)

    plot_widget.plot(
        x_values,
        intensities / intensities.max(),
        pen=pg.mkPen("w", width=PEN_WIDTH),
        name=display_name,
    )


def plot_line(
    plot_widget: pg.PlotWidget,
    sim: Sim,
    colors: list[str],
    max_intensity: float | None = None,
    color_index: int | None = None,
) -> None:
    """Plot each rotational line.

    Args:
        plot_widget (pg.PlotWidget): A `GraphicsView` widget with a single `PlotItem` inside.
        sim (Sim): The parent simulation.
        colors (list[str]): A list of colors for plotting.
        max_intensity (float | None, optional): Provided only if multiple simulations are being run
            together. Defaults to None.
        color_index (int, optional): Provided only if multiple simulations are being run together.
            Defaults to None.
    """
    if max_intensity is None:
        max_intensity = sim.all_line_data()[1].max()

    assert max_intensity is not None

    for idx, band in enumerate(sim.bands):
        color_idx = color_index if color_index is not None else idx

        wavelengths_line: NDArray[np.float64] = utils.wavenum_to_wavelen(band.wavenumbers_line())
        intensities_line: NDArray[np.float64] = band.intensities_line()

        # Create a scatter plot with points at zero and peak intensity.
        scatter_data: NDArray[np.float64] = np.column_stack(
            [
                np.repeat(wavelengths_line, 2),
                np.column_stack(
                    [np.zeros_like(wavelengths_line), intensities_line / max_intensity]
                ).flatten(),
            ],
        ).astype(np.float64)

        plot_widget.plot(
            scatter_data[:, 0],
            scatter_data[:, 1],
            pen=pg.mkPen(colors[color_idx], width=PEN_WIDTH),
            connect="pairs",
            name=f"{sim.molecule.name} {band.v_qn_up, band.v_qn_lo} line",
        )


def plot_line_info(
    plot_widget: pg.PlotWidget,
    sim: Sim,
    colors: list[str],
    max_intensity: float | None = None,
    color_index: int | None = None,
) -> None:
    """Plot information about each rotational line.

    Args:
        plot_widget (pg.PlotWidget): A `GraphicsView` widget with a single `PlotItem` inside.
        sim (Sim): The parent simulation.
        colors (list[str]): A list of colors for plotting.
        max_intensity (float | None, optional): Provided only if multiple simulations are being run
            together. Defaults to None.
        color_index (int, optional): Provided only if multiple simulations are being run together.
            Defaults to None.
    """
    if max_intensity is None:
        max_intensity = sim.all_line_data()[1].max()

    assert max_intensity is not None

    # In order to show text, a plot must first exist.
    plot_line(plot_widget, sim, colors, max_intensity, color_index)

    for band in sim.bands:
        # Only select non-satellite lines to reduce the amount of data on screen.
        lines: list[Line] = [line for line in band.lines if not line.is_satellite]

        for line in lines:
            wavelength: float = utils.wavenum_to_wavelen(line.wavenumber)
            intensity: float = line.intensity / max_intensity
            text: pg.TextItem = pg.TextItem(
                f"ΔJ: {line.branch_name_j}_{line.branch_idx_up}{line.branch_idx_lo}",
                color="w",
                anchor=(0.5, 1.2),
            )
            plot_widget.addItem(text)
            text.setPos(wavelength, intensity)


def plot_conv_sep(
    plot_widget: pg.PlotWidget,
    sim: Sim,
    colors: list[str],
    fwhm_selections: dict[str, bool],
    granularity: int,
    max_intensity: float | None = None,
    color_index: int | None = None,
) -> None:
    """Plot convolved data for each vibrational band separately.

    Args:
        plot_widget (pg.PlotWidget): A `GraphicsView` widget with a single `PlotItem` inside.
        sim (Sim): The parent simulation.
        colors (list[str]): A list of colors for plotting.
        fwhm_selections (dict[str, bool]): The types of broadening to be simulated.
        granularity (int): Number of points on the wavenumber axis.
        max_intensity (float | None, optional): Provided only if multiple simulations are being run
            together. Defaults to None.
    """
    convolved_data: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    calculated_max_intensity: float = 0.0

    # Need to convolve all bands separately, get their maximum intensities, store the largest, and
    # then divide all bands by that maximum. If the max intensity was found for all bands convolved
    # together, it would be inaccurate because of vibrational band overlap.
    for band in sim.bands:
        wavenumbers_conv = band.wavenumbers_conv(granularity)
        intensities_conv = band.intensities_conv(fwhm_selections, wavenumbers_conv)
        wavelengths_conv = utils.wavenum_to_wavelen(wavenumbers_conv)
        convolved_data.append((wavelengths_conv, intensities_conv))

        if max_intensity is None:
            calculated_max_intensity = max(calculated_max_intensity, intensities_conv.max())

    normalization_factor = max_intensity if max_intensity is not None else calculated_max_intensity

    for idx, (wavelengths_conv, intensities_conv) in enumerate(convolved_data):
        color_idx = color_index if color_index is not None else idx
        plot_widget.plot(
            wavelengths_conv,
            intensities_conv / normalization_factor,
            pen=pg.mkPen(colors[color_idx], width=PEN_WIDTH),
            name=f"{sim.molecule.name} {sim.bands[idx].v_qn_up, sim.bands[idx].v_qn_lo} conv",
        )


def plot_conv_all(
    plot_widget: pg.PlotWidget,
    sim: Sim,
    colors: list[str],
    fwhm_selections: dict[str, bool],
    granularity: int,
    max_intensity: float | None = None,
    color_idx: int = 0,
) -> None:
    """Plot convolved data for all vibrational bands simultaneously.

    Args:
        plot_widget (pg.PlotWidget): A `GraphicsView` widget with a single `PlotItem` inside.
        sim (Sim): The parent simulation.
        colors (list[str]): A list of colors for plotting.
        fwhm_selections (dict[str, bool]): The types of broadening to be simulated.
        granularity (int): Number of points on the wavenumber axis.
        max_intensity (float | None, optional): Provided only if multiple simulations are being run
            together. Defaults to None.
        color_idx (int, optional): Provided only if multiple simulations are being run together.
            Defaults to 0.
    """
    wavenumbers_conv, intensities_conv = sim.all_conv_data(fwhm_selections, granularity)
    wavelengths_conv: NDArray[np.float64] = utils.wavenum_to_wavelen(wavenumbers_conv)

    if max_intensity is None:
        max_intensity = intensities_conv.max()

    assert max_intensity is not None

    plot_widget.plot(
        wavelengths_conv,
        intensities_conv / max_intensity,
        pen=pg.mkPen(colors[color_idx], width=PEN_WIDTH),
        name=f"{sim.molecule.name} conv all",
    )
