# module colors
"""Provides a list of colors based on the number of bands to be plotted."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib import color_sequences
from matplotlib.colors import Colormap, to_hex

if TYPE_CHECKING:
    from matplotlib.typing import ColorType


def get_colors(bands: list[tuple[int, int]]) -> list[str]:
    """Return a list of colors.

    Args:
        bands (list[tuple[int, int]]): A list of vibrational bands, e.g. [(0, 1), (0, 2)].

    Returns:
        list[str]: A list of colors in hex format.
    """
    num_bands: int = len(bands)

    colors_small: list[str] = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if num_bands < len(colors_small):
        return colors_small

    palette: list[ColorType] = color_sequences["tab20c"]
    colors_medium: list[str] = [to_hex(color) for color in palette]

    if num_bands < len(colors_medium):
        return colors_medium

    cmap: Colormap = plt.get_cmap("rainbow")
    colors_large: list[str] = [to_hex(cmap(i / num_bands)) for i in range(num_bands)]

    return colors_large
