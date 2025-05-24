# module colors.py
"""Provides a list of colors based on the number of bands to be plotted."""

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
from matplotlib import color_sequences
from matplotlib.colors import Colormap, to_hex

if TYPE_CHECKING:
    from matplotlib.typing import ColorType

# NOTE: 25/03/27 - The use of Matplotlib is retained here since PyQtGraph's default color maps
#       aren't as useful as those from Matplotlib. Furthermore, PyQtGraph can use color maps from
#       Matplotlib, but it still requires Matplotlib to be installed in the local environment, so
#       the maps from Matplotlib are just used directly instead.


def get_colors(bands: list[tuple[int, int]]) -> list[str]:
    """Return a list of colors.

    Args:
        bands (list[tuple[int, int]]): A list of vibrational bands, e.g. [(0, 1), (0, 2)].

    Returns:
        list[str]: A list of colors in hex format.
    """
    num_bands: int = len(bands)

    colors_small: list[str] = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if num_bands <= len(colors_small):
        return colors_small[:num_bands]

    palette: list[ColorType] = color_sequences["tab20c"]
    colors_medium: list[str] = [to_hex(color) for color in palette]

    if num_bands <= len(colors_medium):
        return colors_medium[:num_bands]

    cmap: Colormap = plt.get_cmap("rainbow")
    colors_large: list[str] = [to_hex(cmap(i / num_bands)) for i in range(num_bands)]

    return colors_large
