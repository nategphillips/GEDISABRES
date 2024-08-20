# module gui
"""
Testing GUI functionality.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import plot
from simtype import SimType
from colors import get_colors
from molecule import Molecule
from simulation import Simulation

DEFAULT_TEMPERATURE: float = 300.0    # [K]
DEFAULT_PRESSURE:    float = 101325.0 # [Pa]
DEFAULT_SIMTYPE:     str   = "Absorption"
DEFAULT_PLOTTYPE:    str   = "Line"

def set_axis_labels(ax: Axes) -> None:
    secax = ax.secondary_xaxis("top", functions=(plot.wavenum_to_wavelen, plot.wavenum_to_wavelen))
    secax.set_xlabel("Wavenumber, $\\nu$ [cm$^{-1}$]")

    ax.set_xlabel("Wavelength, $\\lambda$ [nm]")
    ax.set_ylabel("Intensity, Arbitrary Units [-]")

def create_figure() -> tuple[Figure, Axes]:
    fig: Figure = Figure()
    axs: Axes   = fig.add_subplot(111)

    # Set left x limits to something greater than zero so the secondary axis doesn't encounter a
    # divide by zero error
    axs.set_xlim(100, 200)

    set_axis_labels(axs)

    return fig, axs

class MolecularSimulationGUI:
    """
    The GUI.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root: tk.Tk = root

        root.geometry("800x600")
        root.resizable(True, True)
        root.title("Diatomic Molecular Simulation")

        self.create_widgets()

    def create_widgets(self) -> None:
        """
        Create widgets.
        """

        # Frames -----------------------------------------------------------------------------------

        # Frame to contain the table and plot
        self.frame_top: ttk.Frame = ttk.Frame(self.root)
        self.frame_top.pack(side=tk.TOP)

        # Frame for input boxes
        frame_input: ttk.Frame = ttk.Frame(self.root)
        frame_input.pack(side=tk.BOTTOM)

        # Frame for the table
        frame_table: ttk.Frame = ttk.Frame(self.frame_top)
        frame_table.pack(side=tk.LEFT)

        # Frame for the plot
        frame_plot: ttk.Frame = ttk.Frame(self.frame_top)
        frame_plot.pack(side=tk.RIGHT)

        # Entries ----------------------------------------------------------------------------------

        ttk.Label(frame_input, text="Temperature [K]:").grid(row=0, column=0)
        self.temperature: tk.DoubleVar = tk.DoubleVar(value=DEFAULT_TEMPERATURE)
        ttk.Entry(frame_input, textvariable=self.temperature).grid(row=1, column=0)

        ttk.Label(frame_input, text="Pressure [Pa]:").grid(row=0, column=1)
        self.pressure: tk.DoubleVar = tk.DoubleVar(value=DEFAULT_PRESSURE)
        ttk.Entry(frame_input, textvariable=self.pressure).grid(row=1, column=1)

        # Comboboxes -------------------------------------------------------------------------------

        ttk.Label(frame_input, text="Simulation Type:").grid(row=0, column=2)
        self.simulation: tk.StringVar = tk.StringVar(value=DEFAULT_SIMTYPE)
        (ttk.Combobox(frame_input, textvariable=self.simulation, values=("Absorption", "Emission"))
        .grid(row=1, column=2))

        ttk.Label(frame_input, text="Plot Type:").grid(row=0, column=3, padx=5, pady=5)
        self.plot_type = tk.StringVar(value=DEFAULT_PLOTTYPE)
        (ttk.Combobox(frame_input, textvariable=self.plot_type, values=("Line", "Convolution"))
        .grid(row=1, column=3))

        # Button -----------------------------------------------------------------------------------

        (ttk.Button(frame_input, text="Run Simulation", command=self.run_simulation)
        .grid(row=3, column=0, columnspan=4))

        # Plot -------------------------------------------------------------------------------------

        # Draw the initial figure and axes with no date present
        self.fig: Figure
        self.axs: Axes
        self.fig, self.axs = create_figure()
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.frame_top)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack()

        # Map plot types to functions
        self.map_functions = {
            "Line": plot.plot_line,
            "Convolution": plot.plot_conv,
        }

    def run_simulation(self):
        """
        Runs a simulation instance.
        """

        # Grab the temperature, pressure, and simulation type directly from the input fields
        temperature: float = self.temperature.get()
        pressure:    float = self.pressure.get()
        # Convert to uppercase to use as a key for the SimType enum
        sim_type:    str   = self.simulation.get().upper()

        bands: list[tuple[int, int]] = [(2, 0)]

        molecule:   Molecule   = Molecule("o2", 'o', 'o')
        simulation: Simulation = Simulation(molecule, temperature, pressure, np.arange(0, 36),
                                            "b3su", "x3sg", bands, SimType[sim_type])

        colors: list[str] = get_colors("small", bands)

        # Clear the previously plotted data and reset the axis labels
        self.axs.clear()
        set_axis_labels(self.axs)

        # Choose the plotting function based on the selected plot type
        plot_type:     str             = self.plot_type.get()
        plot_function: Callable | None = self.map_functions.get(plot_type)

        if plot_function:
            plot_function(self.axs, simulation, colors)
        else:
            print(f"Plot type '{plot_type}' is not recognized.")

        self.axs.legend()
        self.plot_canvas.draw()

def main() -> None:
    """
    Runs the program.
    """

    root: tk.Tk = tk.Tk()
    MolecularSimulationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
