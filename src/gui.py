# module gui
"""A GUI built using PySide6 with a native table view for DataFrames."""

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPoint, QRect, Qt
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import plot
import utils
from atom import Atom
from colors import get_colors
from molecule import Molecule
from sim import Sim
from simtype import SimType
from state import State

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

DEFAULT_LINES: int = 40
DEFAULT_GRANULARITY: int = int(1e4)

DEFAULT_TEMPERATURE: float = 300.0  # [K]
DEFAULT_PRESSURE: float = 101325.0  # [Pa]
DEFAULT_BROADENING: float = 0.0  # [nm]

DEFAULT_BANDS: str = "0-0"
DEFAULT_PLOTTYPE: str = "Line"
DEFAULT_SIMTYPE: str = "Absorption"


class MyDoubleSpinBox(QDoubleSpinBox):
    """A custom double spin box.

    Allows for arbitrarily large or small input values, high decimal precision, and scientific
    notation.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(-1e99, 1e99)
        self.setDecimals(6)
        self.setKeyboardTracking(False)

    def valueFromText(self, text: str) -> float:
        try:
            return float(text)
        except ValueError:
            return 0.0

    def textFromValue(self, value: float) -> str:
        return f"{value:g}"

    def validate(self, text: str, pos: int):
        # Allow empty input.
        if text == "":
            return (QValidator.State.Intermediate, text, pos)
        try:
            # Try converting to float.
            float(text)
            return (QValidator.State.Acceptable, text, pos)
        except ValueError:
            # If the text contains an 'e' or 'E', it might be a partial scientific notation.
            if "e" in text.lower():
                parts = text.lower().split("e")
                # Allow cases like "1e", "1e-", or "1e+".
                if len(parts) == 2 and (parts[1] == "" or parts[1] in ["-", "+"]):
                    return (QValidator.State.Intermediate, text, pos)
            return (QValidator.State.Invalid, text, pos)


class MyTable(QAbstractTableModel):
    """A simple model to interface a Qt view with a DataFrame."""

    def __init__(self, df: pl.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df

    def rowCount(self, parent=QModelIndex()):
        return self._df.height

    def columnCount(self, parent=QModelIndex()):
        return self._df.width

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._df[index.row(), index.column()]
            return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        if orientation == Qt.Orientation.Vertical:
            return str(section)
        return None


def create_dataframe_tab(df: pl.DataFrame, tab_label: str) -> QWidget:
    """Create a QWidget containing a QTableView to display the DataFrame."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    table_view = QTableView()
    model = MyTable(df)
    table_view.setModel(model)
    table_view.resizeColumnsToContents()
    layout.addWidget(table_view)

    return widget


class GUI(QMainWindow):
    """The GUI implemented with PySide6."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Diatomic Molecular Simulation")
        self.resize(1600, 800)
        self.center()
        self.init_ui()

    def center(self) -> None:
        """Center the window on the screen."""
        qr: QRect = self.frameGeometry()
        qp: QPoint = self.screen().availableGeometry().center()
        qr.moveCenter(qp)
        self.move(qr.topLeft())

    def init_ui(self) -> None:
        """Initialize the user interface."""
        central_widget: QWidget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout: QVBoxLayout = QVBoxLayout(central_widget)

        # Top panel (above)
        top_panel: QWidget = self.create_top_panel()
        main_layout.addWidget(top_panel)

        # Main panel (table and plot)
        main_panel: QWidget = self.create_main_panel()
        main_layout.addWidget(main_panel, stretch=1)

        # Bottom panel (input entries and combos)
        bottom_panel: QWidget = self.create_bottom_panel()
        main_layout.addWidget(bottom_panel)

    def create_top_panel(self) -> QWidget:
        """Create the top panel with band ranges, broadening, granularity, and run controls."""
        top_widget: QWidget = QWidget()
        layout: QHBoxLayout = QHBoxLayout(top_widget)

        # --- Group 1: Bands and broadening checkboxes ---
        group_bands: QGroupBox = QGroupBox("Bands")
        bands_layout: QVBoxLayout = QVBoxLayout(group_bands)

        # Row: Band ranges entry.
        band_range_layout = QHBoxLayout()
        band_range_label = QLabel("Band Ranges (format: v'-v''):")
        self.band_ranges_line_edit = QLineEdit(DEFAULT_BANDS)
        band_range_layout.addWidget(band_range_label)
        band_range_layout.addWidget(self.band_ranges_line_edit)
        bands_layout.addLayout(band_range_layout)

        # Row: Broadening checkboxes.
        checkbox_layout = QHBoxLayout()
        self.checkbox_instrument = QCheckBox("Instrument Broadening")
        self.checkbox_instrument.setChecked(True)
        self.checkbox_doppler = QCheckBox("Doppler Broadening")
        self.checkbox_doppler.setChecked(True)
        self.checkbox_natural = QCheckBox("Natural Broadening")
        self.checkbox_natural.setChecked(True)
        self.checkbox_collisional = QCheckBox("Collisional Broadening")
        self.checkbox_collisional.setChecked(True)
        self.checkbox_predissociation = QCheckBox("Predissociation Broadening")
        self.checkbox_predissociation.setChecked(True)
        checkbox_layout.addWidget(self.checkbox_instrument)
        checkbox_layout.addWidget(self.checkbox_doppler)
        checkbox_layout.addWidget(self.checkbox_natural)
        checkbox_layout.addWidget(self.checkbox_collisional)
        checkbox_layout.addWidget(self.checkbox_predissociation)
        bands_layout.addLayout(checkbox_layout)
        layout.addWidget(group_bands)

        # --- Group 2: Instrument Broadening value ---
        group_inst_broadening = QGroupBox("Instrument Broadening [nm]")
        inst_layout = QHBoxLayout(group_inst_broadening)
        self.inst_broadening_spinbox = MyDoubleSpinBox()
        self.inst_broadening_spinbox.setValue(DEFAULT_BROADENING)
        inst_layout.addWidget(self.inst_broadening_spinbox)
        layout.addWidget(group_inst_broadening)

        # --- Group 3: Granularity ---
        group_granularity = QGroupBox("Granularity")
        gran_layout = QHBoxLayout(group_granularity)
        self.granularity_spinbox = QSpinBox()
        self.granularity_spinbox.setMaximum(10000000)
        self.granularity_spinbox.setValue(DEFAULT_GRANULARITY)
        gran_layout.addWidget(self.granularity_spinbox)
        layout.addWidget(group_granularity)

        # --- Group 4: Rotational Lines and action buttons ---
        group_run = QGroupBox("Run Simulation")
        run_layout = QHBoxLayout(group_run)
        run_layout.addWidget(QLabel("Rotational Lines:"))
        self.num_lines_spinbox = QSpinBox()
        self.num_lines_spinbox.setMaximum(10000)
        self.num_lines_spinbox.setValue(DEFAULT_LINES)
        run_layout.addWidget(self.num_lines_spinbox)
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.add_simulation)
        run_layout.addWidget(self.run_button)
        self.open_button = QPushButton("Open File")
        self.open_button.clicked.connect(self.add_sample)
        run_layout.addWidget(self.open_button)
        layout.addWidget(group_run)

        return top_widget

    def create_main_panel(self) -> QWidget:
        """Create the main panel with table tabs on the left and a plot on the right."""
        main_widget = QWidget()
        layout = QHBoxLayout(main_widget)

        # --- Left side: QTabWidget for tables ---
        self.tab_widget = QTabWidget()
        # Add an initial empty tab (using an empty DataFrame)
        empty_df = pl.DataFrame()
        empty_tab = create_dataframe_tab(empty_df, "v'-v''")
        self.tab_widget.addTab(empty_tab, "v'-v''")
        layout.addWidget(self.tab_widget, stretch=1)

        # --- Right side: Plot area ---
        self.plot_widget = QWidget()
        plot_layout = QVBoxLayout(self.plot_widget)
        self.fig, self.axs = create_figure()
        self.plot_canvas = FigureCanvas(self.fig)
        plot_layout.addWidget(self.plot_canvas)
        self.toolbar = NavigationToolbar(self.plot_canvas, self.plot_widget)
        plot_layout.addWidget(self.toolbar)
        layout.addWidget(self.plot_widget, stretch=2)

        return main_widget

    def create_bottom_panel(self) -> QWidget:
        """Create the bottom panel with temperature, pressure, and combo selections."""
        bottom_widget = QWidget()
        layout = QHBoxLayout(bottom_widget)

        # --- Left: Input entries for temperature and pressure ---
        entries_widget = QWidget()
        entries_layout = QGridLayout(entries_widget)

        # Row 0: Equilibrium temperature.
        self.temp_label = QLabel("Temperature [K]:")
        self.temp_spinbox = MyDoubleSpinBox()
        self.temp_spinbox.setValue(DEFAULT_TEMPERATURE)
        entries_layout.addWidget(self.temp_label, 0, 0)
        entries_layout.addWidget(self.temp_spinbox, 0, 1)

        # Nonequilibrium temperatures (hidden initially)
        self.temp_trn_label = QLabel("Translational Temp [K]:")
        self.temp_trn_spinbox = MyDoubleSpinBox()
        self.temp_trn_spinbox.setValue(DEFAULT_TEMPERATURE)
        self.temp_elc_label = QLabel("Electronic Temp [K]:")
        self.temp_elc_spinbox = MyDoubleSpinBox()
        self.temp_elc_spinbox.setValue(DEFAULT_TEMPERATURE)
        self.temp_vib_label = QLabel("Vibrational Temp [K]:")
        self.temp_vib_spinbox = MyDoubleSpinBox()
        self.temp_vib_spinbox.setValue(DEFAULT_TEMPERATURE)
        self.temp_rot_label = QLabel("Rotational Temp [K]:")
        self.temp_rot_spinbox = MyDoubleSpinBox()
        self.temp_rot_spinbox.setValue(DEFAULT_TEMPERATURE)

        for w in (
            self.temp_trn_label,
            self.temp_trn_spinbox,
            self.temp_elc_label,
            self.temp_elc_spinbox,
            self.temp_vib_label,
            self.temp_vib_spinbox,
            self.temp_rot_label,
            self.temp_rot_spinbox,
        ):
            w.hide()

        entries_layout.addWidget(self.temp_trn_label, 0, 2)
        entries_layout.addWidget(self.temp_trn_spinbox, 0, 3)
        entries_layout.addWidget(self.temp_elc_label, 0, 4)
        entries_layout.addWidget(self.temp_elc_spinbox, 0, 5)
        entries_layout.addWidget(self.temp_vib_label, 0, 6)
        entries_layout.addWidget(self.temp_vib_spinbox, 0, 7)
        entries_layout.addWidget(self.temp_rot_label, 0, 8)
        entries_layout.addWidget(self.temp_rot_spinbox, 0, 9)

        # Row 1: Pressure.
        pressure_label = QLabel("Pressure [Pa]:")
        self.pressure_spinbox = MyDoubleSpinBox()
        self.pressure_spinbox.setValue(DEFAULT_PRESSURE)
        entries_layout.addWidget(pressure_label, 1, 0)
        entries_layout.addWidget(self.pressure_spinbox, 1, 1)

        layout.addWidget(entries_widget)

        # --- Right: Combo boxes ---
        combos_widget = QWidget()
        combos_layout = QGridLayout(combos_widget)

        # Temperature Type.
        temp_type_label = QLabel("Temperature Type:")
        self.temp_type_combo = QComboBox()
        self.temp_type_combo.addItems(["Equilibrium", "Nonequilibrium"])
        self.temp_type_combo.currentTextChanged.connect(self.switch_temp_mode)
        combos_layout.addWidget(temp_type_label, 0, 0)
        combos_layout.addWidget(self.temp_type_combo, 0, 1)

        # Simulation Type.
        sim_type_label = QLabel("Simulation Type:")
        self.sim_type_combo = QComboBox()
        self.sim_type_combo.addItems(["Absorption", "Emission"])
        combos_layout.addWidget(sim_type_label, 1, 0)
        combos_layout.addWidget(self.sim_type_combo, 1, 1)

        # Plot Type.
        plot_type_label = QLabel("Plot Type:")
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Line", "Line Info", "Convolve Separate", "Convolve All"])
        combos_layout.addWidget(plot_type_label, 2, 0)
        combos_layout.addWidget(self.plot_type_combo, 2, 1)

        layout.addWidget(combos_widget)

        return bottom_widget

    def add_sample(self) -> None:
        """Open a CSV file and adds a new tab showing its contents."""
        filename, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Open File",
            dir=str(Path("..", "data", "samples")),
            filter="CSV Files (*.csv);;All Files (*)",
        )
        if filename:
            try:
                df = pl.read_csv(filename)
            except ValueError:
                QMessageBox.critical(self, "Error", "Data is improperly formatted.")
                return
        else:
            return

        new_tab = create_dataframe_tab(df, Path(filename).name)
        self.tab_widget.addTab(new_tab, Path(filename).name)

        plot_sample(self.axs, df, Path(filename).name, "black")
        self.axs.legend()
        self.plot_canvas.draw()

    def switch_temp_mode(self) -> None:
        """Switch between equilibrium and nonequilibrium temperature modes."""
        if self.temp_type_combo.currentText() == "Nonequilibrium":
            self.temp_label.hide()
            self.temp_spinbox.hide()
            self.temp_trn_label.show()
            self.temp_trn_spinbox.show()
            self.temp_elc_label.show()
            self.temp_elc_spinbox.show()
            self.temp_vib_label.show()
            self.temp_vib_spinbox.show()
            self.temp_rot_label.show()
            self.temp_rot_spinbox.show()
        else:
            self.temp_trn_label.hide()
            self.temp_trn_spinbox.hide()
            self.temp_elc_label.hide()
            self.temp_elc_spinbox.hide()
            self.temp_vib_label.hide()
            self.temp_vib_spinbox.hide()
            self.temp_rot_label.hide()
            self.temp_rot_spinbox.hide()
            self.temp_label.show()
            self.temp_spinbox.show()

    def parse_band_ranges(self) -> list[tuple[int, int]]:
        """Parse comma-separated band ranges from user input."""
        band_ranges_str: str = self.band_ranges_line_edit.text()
        bands: list[tuple[int, int]] = []

        for range_str in band_ranges_str.split(","):
            if "-" in range_str.strip():
                try:
                    lower_band, upper_band = map(int, range_str.split("-"))
                    bands.append((lower_band, upper_band))
                except ValueError:
                    QMessageBox.information(
                        self,
                        "Info",
                        f"Invalid band range format: {range_str}",
                        QMessageBox.StandardButton.Ok,
                    )
            else:
                QMessageBox.information(
                    self,
                    "Info",
                    f"Invalid band range format: {range_str}",
                    QMessageBox.StandardButton.Ok,
                )

        return bands

    def add_simulation(self) -> None:
        """Run a simulation instance and update the plot and table tabs."""
        start_time: float = time.time()

        # Determine temperatures based on mode.
        temp: float = self.temp_spinbox.value()
        temp_trn = temp_elc = temp_vib = temp_rot = temp
        if self.temp_type_combo.currentText() == "Nonequilibrium":
            temp_trn = self.temp_trn_spinbox.value()
            temp_elc = self.temp_elc_spinbox.value()
            temp_vib = self.temp_vib_spinbox.value()
            temp_rot = self.temp_rot_spinbox.value()

        pres: float = self.pressure_spinbox.value()
        sim_type: SimType = SimType[self.sim_type_combo.currentText().upper()]
        bands: list[tuple[int, int]] = self.parse_band_ranges()
        rot_lvls = np.arange(0, self.num_lines_spinbox.value())

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

        print(f"Time to create sim: {time.time() - start_time} s")
        start_plot_time: float = time.time()

        colors: list[str] = get_colors(bands)
        self.axs.clear()
        set_axis_labels(self.axs)

        # Map plot types to functions.
        map_functions: dict[str, Callable] = {
            "Line": plot.plot_line,
            "Line Info": plot.plot_line_info,
            "Convolve Separate": plot.plot_conv_sep,
            "Convolve All": plot.plot_conv_all,
        }
        plot_type: str = self.plot_type_combo.currentText()
        plot_function: Callable | None = map_functions.get(plot_type)

        # Check which FWHM parameters the user has selected.
        fwhm_selections: dict[str, bool] = {
            "instrument": self.checkbox_instrument.isChecked(),
            "doppler": self.checkbox_doppler.isChecked(),
            "natural": self.checkbox_natural.isChecked(),
            "collisional": self.checkbox_collisional.isChecked(),
            "predissociation": self.checkbox_predissociation.isChecked(),
        }

        if plot_function is not None:
            if plot_function.__name__ in ("plot_conv_sep", "plot_conv_all"):
                # The instrument broadening FWHM passed here is in [nm], it will get converted to
                # [1/cm] in the FWHM function of the Line class.
                plot_function(
                    self.axs,
                    sim,
                    colors,
                    fwhm_selections,
                    self.inst_broadening_spinbox.value(),
                    self.granularity_spinbox.value(),
                )
            else:
                plot_function(self.axs, sim, colors)
        else:
            QMessageBox.information(
                self,
                "Info",
                f"Plot type '{plot_type}' is not recognized.",
                QMessageBox.StandardButton.Ok,
            )

        self.axs.legend()
        self.plot_canvas.draw()

        print(f"Time to create plot: {time.time() - start_plot_time} s")
        start_table_time: float = time.time()

        # Clear previous tabs.
        while self.tab_widget.count() > 0:
            self.tab_widget.removeTab(0)

        # Create a new tab for each vibrational band.
        for i, band in enumerate(bands):
            df: pl.DataFrame = pl.DataFrame(
                [
                    {
                        "Wavelength": utils.wavenum_to_wavelen(line.wavenumber),
                        "Wavenumber": line.wavenumber,
                        "Intensity": line.intensity,
                        "J'": line.j_qn_up,
                        "J''": line.j_qn_lo,
                        "N'": line.n_qn_up,
                        "N''": line.n_qn_lo,
                        "Branch": f"{line.branch_name}{line.branch_idx_up}{line.branch_idx_lo}",
                    }
                    for line in sim.bands[i].lines
                ]
            )

            tab_name: str = f"{band[0]}-{band[1]}"
            new_tab: QWidget = create_dataframe_tab(df, tab_name)
            self.tab_widget.addTab(new_tab, tab_name)

        # TODO: 25/03/10 - Creating tables seems to be the bottleneck in terms of speed, especially
        #       when there are a large number of vibrational bands being displayed. I think this
        #       is due to iterating over each Line, which means there's not much I can do.
        print(f"Time to create table: {time.time() - start_table_time} s")
        print(f"Total time: {time.time() - start_time} s\n")


def set_axis_labels(ax: Axes) -> None:
    """Set the main x-label to wavelength and adds a secondary wavenumber x-axis."""

    def conversion_fn(x):
        x = np.array(x, float)
        near_zero = np.isclose(x, 0)
        x[near_zero] = np.inf
        x[~near_zero] = 1 / x[~near_zero]
        return x * 1e7

    secax = ax.secondary_xaxis("top", functions=(conversion_fn, conversion_fn))
    secax.set_xlabel("Wavenumber, $\\nu$ [cm$^{-1}$]")
    ax.set_xlabel("Wavelength, $\\lambda$ [nm]")
    ax.set_ylabel("Intensity, $I$ [a.u.]")


def create_figure() -> tuple[Figure, Axes]:
    """Initialize a blank figure with arbitrary limits."""
    fig: Figure = Figure()
    axs: Axes = fig.add_subplot(111)
    axs.set_xlim(100, 200)
    set_axis_labels(axs)

    return fig, axs


def plot_sample(axs: Axes, df: pl.DataFrame, label: str, color: str) -> None:
    """Plot sample data."""
    wavelengths: NDArray[np.float64] = utils.wavenum_to_wavelen(df["wavenumber"].to_numpy())
    intensities: NDArray[np.float64] = df["intensity"].to_numpy()
    axs.plot(wavelengths, intensities / intensities.max(), label=label, color=color)


def main() -> None:
    """Entry point."""
    app: QApplication = QApplication(sys.argv)
    window: GUI = GUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
