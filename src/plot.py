# module plot.py
"""Contains functions used for plotting."""

# Copyright (C) 2023-2026 Nathan G. Phillips

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

import utils

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from sim import Sim

PEN_WIDTH = 1


def plot_sample(
    plot_widget: pg.PlotWidget,
    x_values: NDArray[np.float64],
    intensities: NDArray[np.float64],
    display_name: str,
    value_type: str,
) -> None:
    """Plot sample data.

    Args:
        plot_widget: A `GraphicsView` widget with a single `PlotItem` inside.
        x_values: Sample wavenumbers.
        intensities: Sample intensities.
        display_name: The name of the file without directory information.
        value_type: Either wavenumbers or wavelengths.
    """
    if value_type == "wavelength":
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
        plot_widget: A `GraphicsView` widget with a single `PlotItem` inside.
        sim: The parent simulation.
        colors: A list of colors for plotting.
        max_intensity: Provided only if multiple simulations are being run together. Defaults to
            None.
        color_index: Provided only if multiple simulations are being run together. Defaults to None.
    """
    if max_intensity is None:
        max_intensity = sim.all_line_data()[1].max()

    assert max_intensity is not None

    for idx, band in enumerate(sim.bands):
        color_idx = color_index if color_index is not None else idx

        wavenumbers_line = band.wavenumbers_line()
        intensities_line = band.intensities_line()

        # Create a scatter plot with points at zero and peak intensity.
        scatter_data = np.column_stack(
            [
                np.repeat(wavenumbers_line, 2),
                np.column_stack(
                    [np.zeros_like(wavenumbers_line), intensities_line / max_intensity]
                ).flatten(),
            ],
        )

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
        plot_widget: A `GraphicsView` widget with a single `PlotItem` inside.
        sim: The parent simulation.
        colors: A list of colors for plotting.
        max_intensity: Provided only if multiple simulations are being run together. Defaults to
            None.
        color_index: Provided only if multiple simulations are being run together. Defaults to None.
    """
    if max_intensity is None:
        max_intensity = sim.all_line_data()[1].max()

    assert max_intensity is not None

    # In order to show text, a plot must first exist.
    plot_line(plot_widget, sim, colors, max_intensity, color_index)

    for band in sim.bands:
        for line in band.lines:
            text = pg.TextItem(
                f"ΔJ: {line.branch_name_j}{line.branch_idx_up}{line.branch_idx_lo}(J'={line.j_qn_up}, J''={line.j_qn_lo})\nΔN: {line.branch_name_n}{line.branch_idx_up}{line.branch_idx_lo}(N'={line.n_qn_up}, N''={line.n_qn_lo})",
                color="w",
                anchor=(0.5, 1.2),
            )
            plot_widget.addItem(text)
            text.setPos(line.wavenumber, line.intensity / max_intensity)


def plot_cont_sep(
    plot_widget: pg.PlotWidget,
    sim: Sim,
    colors: list[str],
    granularity: int,
    max_intensity: float | None = None,
    color_index: int | None = None,
) -> None:
    """Plot continuous data for each vibrational band separately.

    Args:
        plot_widget: A `GraphicsView` widget with a single `PlotItem` inside.
        sim: The parent simulation.
        colors: A list of colors for plotting.
        granularity: Number of points on the wavenumber axis.
        max_intensity: Provided only if multiple simulations are being run together. Defaults to
            None.
        color_index: Provided only if multiple simulations are being run together. Defaults to 0.
    """
    continuous_data: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    calculated_max_intensity = 0.0

    # Need to sum all bands separately, get their maximum intensities, store the largest, and
    # then divide all bands by that maximum.
    for band in sim.bands:
        wavenumbers_cont = band.wavenumbers_cont(granularity)
        intensities_cont = band.intensities_cont(wavenumbers_cont)
        continuous_data.append((wavenumbers_cont, intensities_cont))

        if max_intensity is None:
            calculated_max_intensity = max(calculated_max_intensity, intensities_cont.max())

    normalization_factor = max_intensity if max_intensity is not None else calculated_max_intensity

    for idx, (wavenumbers_cont, intensities_cont) in enumerate(continuous_data):
        color_idx = color_index if color_index is not None else idx
        plot_widget.plot(
            wavenumbers_cont,
            intensities_cont / normalization_factor,
            pen=pg.mkPen(colors[color_idx], width=PEN_WIDTH),
            name=f"{sim.molecule.name} {sim.bands[idx].v_qn_up, sim.bands[idx].v_qn_lo} cont",
        )


def plot_cont_all(
    plot_widget: pg.PlotWidget,
    sim: Sim,
    colors: list[str],
    granularity: int,
    max_intensity: float | None = None,
    color_index: int = 0,
) -> None:
    """Plot continuous data for all vibrational bands simultaneously.

    Args:
        plot_widget: A `GraphicsView` widget with a single `PlotItem` inside.
        sim: The parent simulation.
        colors: A list of colors for plotting.
        granularity: Number of points on the wavenumber axis.
        max_intensity: Provided only if multiple simulations are being run together. Defaults to
            None.
        color_index: Provided only if multiple simulations are being run together. Defaults to 0.
    """
    wavenumbers_cont, intensities_cont = sim.all_cont_data(granularity)

    if max_intensity is None:
        max_intensity = intensities_cont.max()

    assert max_intensity is not None

    plot_widget.plot(
        wavenumbers_cont,
        intensities_cont / max_intensity,
        pen=pg.mkPen(colors[color_index], width=PEN_WIDTH),
        name=f"{sim.molecule.name} cont all",
    )
