# module gui
"""
Testing GUI functionality.
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from typing import Callable
import warnings

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
import numpy as np
import pandas as pd
from pandastable import Table

from atom import Atom
from colors import get_colors
from molecule import Molecule
import plot
from sim import Sim
from simtype import SimType
from state import State
import utils

# NOTE: 11/04/24 - I think an internal function within pandastable is using .fillna or a related
# function that emits "FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is
# deprecated and will change in a future version."
#
# The fixes in the linked thread don't seem to work, which is why I think the issue is internal to
# pandastable itself. For now, just disable the warning.
# https://stackoverflow.com/questions/77900971/pandas-futurewarning-downcasting-object-dtype-arrays-on-fillna-ffill-bfill.
pd.set_option("future.no_silent_downcasting", True)
# NOTE: 11/06/24 - More warnings for d_type conversions that aren't yet fixed in a release build of
# pandastable, see: https://github.com/dmnfarrell/pandastable/issues/251.
warnings.simplefilter(action="ignore", category=FutureWarning)

DEFAULT_LINES: int = 40

DEFAULT_TEMPERATURE: float = 300.0  # [K]
DEFAULT_PRESSURE: float = 101325.0  # [Pa]

DEFAULT_BANDS: str = "0-0"
DEFAULT_PLOTTYPE: str = "Line"
DEFAULT_SIMTYPE: str = "Absorption"


class GUI:
    """
    The GUI.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root: tk.Tk = root

        self.root.title("Diatomic Molecular Simulation")

        # Center the window on the screen.
        screen_width: int = self.root.winfo_screenwidth()
        screen_height: int = self.root.winfo_screenheight()

        window_height: int = 800
        window_width: int = 1600

        x_offset: int = int((screen_width / 2) - (window_width / 2))
        y_offset: int = int((screen_height / 2) - (window_height / 2))

        self.root.geometry(f"{window_width}x{window_height}+{x_offset}+{y_offset}")

        self.create_widgets()

    def create_widgets(self) -> None:
        """
        Create widgets.
        """

        # FRAMES -----------------------------------------------------------------------------------

        # Main frames for input boxes, entries, combo boxes, the table, and the plot.
        self.frame_above: ttk.Frame = ttk.Frame(self.root)
        self.frame_above.pack(side=tk.TOP, fill=tk.X)

        self.frame_above_bands: ttk.Frame = ttk.Frame(self.frame_above)
        self.frame_above_bands.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=5)

        self.frame_above_run: ttk.Frame = ttk.Frame(self.frame_above)
        self.frame_above_run.pack(side=tk.RIGHT, fill=tk.X, padx=5, pady=5)

        self.frame_input: ttk.Frame = ttk.Frame(self.root)
        self.frame_input.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.frame_input_entries: ttk.Frame = ttk.Frame(self.frame_input)
        self.frame_input_entries.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=5)

        self.frame_input_combos: ttk.Frame = ttk.Frame(self.frame_input)
        self.frame_input_combos.pack(side=tk.RIGHT, fill=tk.X, padx=5, pady=5)

        self.frame_main: ttk.Frame = ttk.Frame(self.root)
        self.frame_main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.frame_table: ttk.Frame = ttk.Frame(self.frame_main)
        self.frame_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.frame_plot: ttk.Frame = ttk.Frame(self.frame_main)
        self.frame_plot.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ABOVE ------------------------------------------------------------------------------------

        # Entry for choosing bands.
        self.band_ranges = tk.StringVar(value=DEFAULT_BANDS)
        ttk.Label(self.frame_above_bands, text="Band Ranges (format: v'-v''):").grid(
            row=0, column=0, padx=5, pady=5
        )
        ttk.Entry(self.frame_above_bands, textvariable=self.band_ranges, width=50).grid(
            row=0, column=1, columnspan=3, padx=5, pady=5
        )

        # Selection for number of rotational lines.
        self.num_lines = tk.IntVar(value=DEFAULT_LINES)
        ttk.Label(self.frame_above_run, text="Rotational Lines:").grid(
            row=0, column=0, padx=5, pady=5
        )
        ttk.Entry(self.frame_above_run, textvariable=self.num_lines).grid(
            row=0, column=1, padx=5, pady=5
        )

        # Button for running the simulation.
        ttk.Button(self.frame_above_run, text="Run Simulation", command=self.add_simulation).grid(
            row=0, column=2, padx=5, pady=5
        )

        ttk.Button(self.frame_above_run, text="Open File", command=self.add_sample).grid(
            row=0, column=3, padx=5, pady=5
        )

        # ENTRIES ----------------------------------------------------------------------------------

        # Show the entry for equilibrium temperature by default.
        self.temp = tk.DoubleVar(value=DEFAULT_TEMPERATURE)

        self.label_temp = ttk.Label(self.frame_input_entries, text="Temperature [K]:")
        self.label_temp.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.entry_temp = ttk.Entry(self.frame_input_entries, textvariable=self.temp)
        self.entry_temp.grid(row=0, column=1, padx=5, pady=5)

        # Nonequilibrium entries, all of which are hidden by default.
        self.temp_trn = tk.DoubleVar(value=DEFAULT_TEMPERATURE)
        self.temp_elc = tk.DoubleVar(value=DEFAULT_TEMPERATURE)
        self.temp_vib = tk.DoubleVar(value=DEFAULT_TEMPERATURE)
        self.temp_rot = tk.DoubleVar(value=DEFAULT_TEMPERATURE)

        self.label_temp_trn = ttk.Label(self.frame_input_entries, text="Translational Temp [K]:")
        self.entry_temp_trn = ttk.Entry(self.frame_input_entries, textvariable=self.temp_trn)
        self.label_temp_elc = ttk.Label(self.frame_input_entries, text="Electronic Temp [K]:")
        self.entry_temp_elc = ttk.Entry(self.frame_input_entries, textvariable=self.temp_elc)
        self.label_temp_vib = ttk.Label(self.frame_input_entries, text="Vibrational Temp [K]:")
        self.entry_temp_vib = ttk.Entry(self.frame_input_entries, textvariable=self.temp_vib)
        self.label_temp_rot = ttk.Label(self.frame_input_entries, text="Rotational Temp [K]:")
        self.entry_temp_rot = ttk.Entry(self.frame_input_entries, textvariable=self.temp_rot)

        # Entries for pressure and bands.
        self.pressure = tk.DoubleVar(value=DEFAULT_PRESSURE)
        ttk.Label(self.frame_input_entries, text="Pressure [Pa]:").grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        ttk.Entry(self.frame_input_entries, textvariable=self.pressure).grid(
            row=1, column=1, padx=5, pady=5
        )

        # COMBOS -----------------------------------------------------------------------------------

        # Combo boxes for temperature mode, simulation type, and plot type.
        self.temp_type = tk.StringVar(value="Equilibrium")
        ttk.Label(self.frame_input_combos, text="Temperature Type:").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        mode_combobox = ttk.Combobox(
            self.frame_input_combos,
            textvariable=self.temp_type,
            values=("Equilibrium", "Nonequilibrium"),
        )
        mode_combobox.grid(row=0, column=1, padx=5, pady=5)

        # Updates the visible boxes based on the mode the user selects.
        mode_combobox.bind("<<ComboboxSelected>>", self.switch_temp_mode)

        self.sim_type = tk.StringVar(value=DEFAULT_SIMTYPE)
        ttk.Label(self.frame_input_combos, text="Simulation Type:").grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        ttk.Combobox(
            self.frame_input_combos, textvariable=self.sim_type, values=("Absorption", "Emission")
        ).grid(row=1, column=1, padx=5, pady=5)

        self.plot_type = tk.StringVar(value=DEFAULT_PLOTTYPE)
        ttk.Label(self.frame_input_combos, text="Plot Type:").grid(
            row=2, column=0, padx=5, pady=5, sticky="w"
        )
        ttk.Combobox(
            self.frame_input_combos,
            textvariable=self.plot_type,
            values=(
                "Line",
                "Line Info",
                "Convolve Separate",
                "Convolve All",
                "Instrument Separate",
            ),
        ).grid(row=2, column=1, padx=5, pady=5)

        # TABLE ------------------------------------------------------------------------------------

        # Notebook (tabs) for holding multiple tables, one for each specified vibrational band.
        self.notebook = ttk.Notebook(self.frame_table)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Initialize the table with an empty dataframe so that nothing is shown until a simulation
        # is run by the user.
        frame_notebook: ttk.Frame = ttk.Frame(self.notebook)
        table: Table = Table(
            frame_notebook,
            dataframe=pd.DataFrame(),
            showtoolbar=True,
            showstatusbar=True,
            editable=False,
        )
        table.show()
        self.notebook.add(frame_notebook, text="v'-v''")

        # PLOT -------------------------------------------------------------------------------------

        # Draw the initial figure and axes with no data present.
        self.fig: Figure
        self.axs: Axes
        self.fig, self.axs = create_figure()
        self.plot_canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(self.fig, master=self.frame_plot)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Show the matplotlib toolbar at the bottom of the plot.
        self.toolbar: NavigationToolbar2Tk = NavigationToolbar2Tk(self.plot_canvas)

        # Map plot types to functions.
        self.map_functions: dict[str, Callable] = {
            "Line": plot.plot_line,
            "Line Info": plot.plot_line_info,
            "Convolve Separate": plot.plot_conv_sep,
            "Convolve All": plot.plot_conv_all,
            "Instrument Separate": plot.plot_inst_sep,
        }

    def add_sample(self) -> None:
        """
        Testing.
        """

        full_path: str = filedialog.askopenfilename(initialdir="../data/samples")
        filename: str = full_path.split("/")[-1]

        if full_path:
            try:
                df: pd.DataFrame = pd.read_csv(full_path)
            except ValueError:
                messagebox.showerror("Error", "Data is improperly formatted.")
                return
        else:
            return

        frame_notebook: ttk.Frame = ttk.Frame(self.notebook)
        table: Table = Table(
            frame_notebook, dataframe=df, showtoolbar=True, showstatusbar=True, editable=False
        )
        table.show()
        self.notebook.add(frame_notebook, text=filename)

        plot_sample(self.axs, df, filename, "black")
        self.axs.legend()
        self.plot_canvas.draw()

    def switch_temp_mode(self, event=None) -> None:
        """
        Switches between equilibrium and nonequilibrium temperature modes.
        """

        if self.temp_type.get() == "Nonequilibrium":
            # Remove equilibrium entry.
            self.label_temp.grid_forget()
            self.entry_temp.grid_forget()

            # Place all nonequilibrium entries.
            self.label_temp_trn.grid(row=0, column=0, padx=5, pady=5, sticky="w")
            self.entry_temp_trn.grid(row=0, column=1, padx=5, pady=5)
            self.label_temp_elc.grid(row=0, column=2, padx=5, pady=5, sticky="w")
            self.entry_temp_elc.grid(row=0, column=3, padx=5, pady=5)
            self.label_temp_vib.grid(row=0, column=4, padx=5, pady=5, sticky="w")
            self.entry_temp_vib.grid(row=0, column=5, padx=5, pady=5)
            self.label_temp_rot.grid(row=0, column=6, padx=5, pady=5, sticky="w")
            self.entry_temp_rot.grid(row=0, column=7, padx=5, pady=5)
        else:
            # Remove all nonequilibrium entries.
            self.label_temp_trn.grid_forget()
            self.entry_temp_trn.grid_forget()
            self.label_temp_elc.grid_forget()
            self.entry_temp_elc.grid_forget()
            self.label_temp_vib.grid_forget()
            self.entry_temp_vib.grid_forget()
            self.label_temp_rot.grid_forget()
            self.entry_temp_rot.grid_forget()

            # Place equilibrium entry.
            self.label_temp.grid(row=0, column=0, padx=5, pady=5, sticky="w")
            self.entry_temp.grid(row=0, column=1, padx=5, pady=5)

    def parse_band_ranges(self) -> list[tuple[int, int]]:
        """
        Convert comma-separated user input of the form 0-0, 0-2, etc. into valid vibrational bands.
        """

        band_ranges_str: str = self.band_ranges.get()

        bands: list[tuple[int, int]] = []

        for range_str in band_ranges_str.split(","):
            range_str: str = range_str.strip()

            if "-" in range_str:
                try:
                    lower_band: int
                    upper_band: int
                    lower_band, upper_band = map(int, range_str.split("-"))
                    bands.append((lower_band, upper_band))
                except ValueError:
                    messagebox.showinfo("Info", f"Invalid band range format: {range_str}")
            else:
                messagebox.showinfo("Info", f"Invalid band range format: {range_str}")

        return bands

    def add_simulation(self) -> None:
        """
        Runs a simulation instance.
        """

        # TODO: 11/06/24:
        # - Create an option for switching between equilibrium and non-equilibrium.
        # - Add the ability to plot sample data with simulated data.
        # - Allow multiple types of data to be plotted at the same time.
        # - Have separate temperature, pressure, and plots for each band.
        # - Make another tab for LIF calculations.
        # - Create options to save and load simulations.

        # First check which mode the simulation is in and set the temperatures accordingly.
        temp_trn = temp_elc = temp_vib = temp_rot = self.temp.get()
        if self.temp_type.get() == "Nonequilibrium":
            temp_trn = self.temp_trn.get()
            temp_elc = self.temp_elc.get()
            temp_vib = self.temp_vib.get()
            temp_rot = self.temp_rot.get()

        # Grab the pressure data directly from the input fields.
        pres: float = self.pressure.get()
        # Convert the simulation type to uppercase to use as a key for the SimType enum.
        sim_type: SimType = SimType[self.sim_type.get().upper()]
        # Get the list of upper and lower vibrational bands from the user input.
        bands: list[tuple[int, int]] = self.parse_band_ranges()
        # Maximum number of rotational lines to simulate.
        rot_lvls: np.ndarray = np.arange(0, self.num_lines.get())

        molecule: Molecule = Molecule(name="O2", atom_1=Atom("O"), atom_2=Atom("O"))

        state_up: State = State(name="B3Su-", spin_multiplicity=3, molecule=molecule)
        state_lo: State = State(name="X3Sg-", spin_multiplicity=3, molecule=molecule)

        sim: Sim = Sim(
            sim_type=sim_type,
            molecule=molecule,
            state_up=state_up,
            state_lo=state_lo,
            rot_lvls=rot_lvls,
            temp_trn=temp_trn,
            temp_elc=temp_elc,
            temp_vib=temp_vib,
            temp_rot=temp_rot,
            pressure=pres,
            bands=bands,
        )

        colors: list[str] = get_colors(bands)

        # Clear the previously plotted data and reset the axis labels.
        self.axs.clear()
        set_axis_labels(self.axs)

        # Choose the plotting function based on the selected plot type.
        plot_type: str = self.plot_type.get()
        plot_function: Callable | None = self.map_functions.get(plot_type)

        if plot_function is not None:
            if plot_function.__name__ in ("plot_inst_sep", "plot_inst_all"):
                plot_function(self.axs, sim, colors, 10)
            else:
                plot_function(self.axs, sim, colors)
        else:
            messagebox.showinfo("Info", f"Plot type '{plot_type}' is not recognized.")

        self.axs.legend()
        self.plot_canvas.draw()

        # Clear previous tabs.
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)

        # Each vibrational band has a separate tab associated with it, each tab gets updated
        # separately.
        for i, band in enumerate(bands):
            data: list[dict[str, float | int | str]] = [
                {
                    "Wavelength": utils.wavenum_to_wavelen(line.wavenumber),
                    "Wavenumber": line.wavenumber,
                    "Intensity": line.intensity,
                    "J'": line.j_qn_up,
                    "J''": line.j_qn_lo,
                    "N'": line.n_qn_up,
                    "N''": line.n_qn_lo,
                    "Branch": line.branch_name + str(line.branch_idx_up) + str(line.branch_idx_lo),
                }
                for line in sim.bands[i].lines
            ]

            df: pd.DataFrame = pd.DataFrame(data)

            frame_notebook: ttk.Frame = ttk.Frame(self.notebook)
            table: Table = Table(
                frame_notebook, dataframe=df, showtoolbar=True, showstatusbar=True, editable=False
            )
            table.show()
            self.notebook.add(frame_notebook, text=f"{band[0]}-{band[1]}")


def set_axis_labels(ax: Axes) -> None:
    """
    Sets the main x-label to wavelength and adds a secondary wavenumber x-axis.
    """

    def conversion_fn(x):
        """
        A robust conversion from wavenumbers to wavelength that avoids divide by zero errors.
        """

        x = np.array(x, float)
        near_zero: np.ndarray = np.isclose(x, 0)

        x[near_zero] = np.inf
        x[~near_zero] = 1 / x[~near_zero]

        return x * 1e7

    secax = ax.secondary_xaxis("top", functions=(conversion_fn, conversion_fn))
    secax.set_xlabel("Wavenumber, $\\nu$ [cm$^{-1}$]")

    ax.set_xlabel("Wavelength, $\\lambda$ [nm]")
    ax.set_ylabel("Intensity, $I$ [a.u.]")


def create_figure() -> tuple[Figure, Axes]:
    """
    Initialize a blank figure with arbitrary limits.
    """

    fig: Figure = Figure()
    axs: Axes = fig.add_subplot(111)

    # Set the left x-limit to something greater than zero so the secondary axis doesn't encounter a
    # divide by zero error before any data is actually plotted.
    axs.set_xlim(100, 200)

    set_axis_labels(axs)

    return fig, axs


def plot_sample(axs: Axes, df: pd.DataFrame, label: str, color: str) -> None:
    """
    Plots sample data.
    """

    wavelengths = utils.wavenum_to_wavelen(df["wavenumber"])
    intensities = df["intensity"]

    axs.plot(wavelengths, intensities / intensities.max(), label=label, color=color)


def main() -> None:
    """
    Entry point.
    """

    root: tk.Tk = tk.Tk()
    GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
