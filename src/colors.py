# module colors
"""Provides a list of colors based on the number of bands to be plotted."""

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import Colormap, to_hex


def get_colors(bands: list[tuple[int, int]]) -> list[str]:
    """Return a list of colors."""
    num_bands: int = len(bands)

    colors_small: list[str] = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if num_bands < len(colors_small):
        return colors_small

    # NOTE: 25/03/10 - The `ListedColormap` class has a `colors` attribute, but for some reason, the
    #       `get_cmap` method only ever returns a `Colormap`, even when the map is discrete. All
    #       that to say, ignore this Pyright error. Also, see the link below for more info.
    #
    # https://matplotlib.org/stable/gallery/color/individual_colors_from_cmap.html#extracting-colors-from-a-discrete-colormap
    palette: list[tuple] = cycler("color", plt.get_cmap("tab20c").colors).by_key()["color"]
    colors_medium: list[str] = [to_hex(color) for color in palette]

    if num_bands < len(colors_medium):
        return colors_medium

    cmap: Colormap = plt.get_cmap("rainbow")
    colors_large: list[str] = [to_hex(cmap(i / num_bands)) for i in range(num_bands)]

    return colors_large
