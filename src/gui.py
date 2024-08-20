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

DEFAULT_PLOTTYPE: str = "Line"
DEFAULT_SIMTYPE:  str = "Absorption"

def set_axis_labels(ax: Axes) -> None:
    secax = ax.secondary_xaxis("top", functions=(plot.wavenum_to_wavelen, plot.wavenum_to_wavelen))
    secax.set_xlabel("Wavenumber, $\\nu$ [cm$^{-1}$]")

    ax.set_xlabel("Wavelength, $\\lambda$ [nm]")
    ax.set_ylabel("Intensity, Arbitrary Units [-]")

def create_figure() -> tuple[Figure, Axes]:
    fig: Figure = Figure()
    axs: Axes   = fig.add_subplot(111)

    # Set the left x-limit to something greater than zero so the secondary axis doesn't encounter a
    # divide by zero error before any data is actually plotted
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
        root.title("Diatomic Molecular Simulation")

        # Center the window on the screen
        screen_width:  int = self.root.winfo_screenwidth()
        screen_height: int = self.root.winfo_screenheight()

        window_height: int = 600
        window_width:  int = 1200

        x_offset: int = int((screen_width / 2) - (window_width / 2))
        y_offset: int = int((screen_height / 2) - (window_height / 2))

        self.root.geometry(f"{window_width}x{window_height}+{x_offset}+{y_offset}")

        self.create_widgets()

    def create_widgets(self) -> None:
        """
        Create widgets.
        """

        # Frames -----------------------------------------------------------------------------------

        # Frame for input boxes
        self.frame_input: ttk.Frame = ttk.Frame(self.root)
        self.frame_input.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # Frame for the table and plot
        self.frame_main: ttk.Frame = ttk.Frame(self.root)
        self.frame_main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Frame for the table
        self.frame_table: ttk.Frame = ttk.Frame(self.frame_main)
        self.frame_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame for the plot
        self.frame_plot: ttk.Frame = ttk.Frame(self.frame_main)
        self.frame_plot.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Entries ----------------------------------------------------------------------------------

        ttk.Label(self.frame_input, text="Temperature [K]:").pack(side=tk.LEFT, padx=5, pady=5)
        self.temperature: tk.DoubleVar = tk.DoubleVar(value=DEFAULT_TEMPERATURE)
        ttk.Entry(self.frame_input, textvariable=self.temperature).pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Label(self.frame_input, text="Pressure [Pa]:").pack(side=tk.LEFT, padx=5, pady=5)
        self.pressure: tk.DoubleVar = tk.DoubleVar(value=DEFAULT_PRESSURE)
        ttk.Entry(self.frame_input, textvariable=self.pressure).pack(side=tk.LEFT, padx=5, pady=5)

        # Comboboxes -------------------------------------------------------------------------------

        ttk.Label(self.frame_input, text="Simulation Type:").pack(side=tk.LEFT, padx=5, pady=5)
        self.simulation: tk.StringVar = tk.StringVar(value=DEFAULT_SIMTYPE)
        ttk.Combobox(self.frame_input, textvariable=self.simulation, values=("Absorption", "Emission")).pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Label(self.frame_input, text="Plot Type:").pack(side=tk.LEFT, padx=5, pady=5)
        self.plot_type: tk.StringVar = tk.StringVar(value=DEFAULT_PLOTTYPE)
        ttk.Combobox(self.frame_input, textvariable=self.plot_type, values=("Line", "Convolution")).pack(side=tk.LEFT, padx=5, pady=5)

        # Button -----------------------------------------------------------------------------------

        ttk.Button(self.frame_input, text="Run Simulation", command=self.run_simulation).pack(side=tk.LEFT, padx=5, pady=5)

        # Table ------------------------------------------------------------------------------------

        columns: list[str] = ["rot_qn_up", "rot_qn_lo", "branch_idx_up", "branch_idx_lo"]

        self.table: ttk.Treeview = ttk.Treeview(self.frame_table, columns=columns, show="headings")

        # Set column headings
        self.table.heading("rot_qn_up", text="J'")
        self.table.heading("rot_qn_lo", text="J''")
        self.table.heading("branch_idx_up", text="n'")
        self.table.heading("branch_idx_lo", text="n''")

        # Set default column widths
        self.table.column("rot_qn_up", width=50)
        self.table.column("rot_qn_lo", width=50)
        self.table.column("branch_idx_up", width=50)
        self.table.column("branch_idx_lo", width=50)

        self.table.pack(fill=tk.BOTH, expand=True)

        # Plot -------------------------------------------------------------------------------------

        # Draw the initial figure and axes with no data present
        self.fig: Figure
        self.axs: Axes
        self.fig, self.axs = create_figure()
        self.plot_canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(self.fig, master=self.frame_plot)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Map plot types to functions
        self.map_functions: dict[str, Callable] = {
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

        # Test pushing data to table
        self.table.delete(*self.table.get_children())

        data: list[list[str]] = [[f"{line.rot_qn_up}", f"{line.rot_qn_lo}",
                                  f"{line.branch_idx_up}", f"{line.branch_idx_lo}"]
                                 for line in simulation.vib_bands[0].lines]

        for item in data:
            self.table.insert('', tk.END, values=item)

def main() -> None:
    """
    Runs the program.
    """

    root: tk.Tk = tk.Tk()
    MolecularSimulationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
