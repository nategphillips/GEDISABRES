# module main.py
"""Contains code for the GUI."""

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

import contextlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pyqtgraph as pg
import qdarktheme
from PySide6 import QtCore
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    Qt,
)
from PySide6.QtGui import QIcon, QValidator
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTabBar,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import constants
import data_path
import lif
import plot
import utils
from atom import Atom
from colors import get_colors
from molecule import Molecule
from sim import Sim
from sim_params import (
    BroadeningBools,
    InstrumentParams,
    LaserParams,
    PlotBools,
    PlotParams,
    ShiftBools,
    ShiftParams,
    TemperatureParams,
)
from sim_props import ConstantsType, InversionSymmetry, ReflectionSymmetry, SimType, TermSymbol
from state import State

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

DEFAULT_J_MAX = 40
DEFAULT_RESOLUTION = int(1e4)

DEFAULT_PRESSURE = 101325.0
DEFAULT_BROADENING = 0.0

DEFAULT_BANDS = "0-0"

DEFAULT_MOLECULE = Molecule(Atom(16, "O"), Atom(16, "O"))
DEFAULT_STATE_UP = State(
    DEFAULT_MOLECULE,
    "B",
    3,
    TermSymbol.SIGMA,
    InversionSymmetry.UNGERADE,
    ReflectionSymmetry.MINUS,
    ConstantsType.PERLEVEL,
)
DEFAULT_STATE_LO = State(
    DEFAULT_MOLECULE,
    "X",
    3,
    TermSymbol.SIGMA,
    InversionSymmetry.GERADE,
    ReflectionSymmetry.MINUS,
    ConstantsType.PERLEVEL,
)
DEFAULT_SIM = Sim(
    sim_type=SimType.ABSORPTION,
    molecule=DEFAULT_MOLECULE,
    state_up=DEFAULT_STATE_UP,
    state_lo=DEFAULT_STATE_LO,
    j_qn_up_max=DEFAULT_J_MAX,
    pressure=DEFAULT_PRESSURE,
    bands_input=[(0, 0)],
)

O2_MOLECULE = Molecule(Atom(16, "O"), Atom(16, "O"))
NO_MOLECULE = Molecule(Atom(14, "N"), Atom(16, "O"))
OH_MOLECULE = Molecule(Atom(16, "O"), Atom(1, "H"))

MOLECULAR_PRESETS = [
    {
        "name": "16O16O B-X",
        "molecule": O2_MOLECULE,
        "state_up": State(
            O2_MOLECULE,
            "B",
            3,
            TermSymbol.SIGMA,
            InversionSymmetry.UNGERADE,
            ReflectionSymmetry.MINUS,
            ConstantsType.PERLEVEL,
        ),
        "state_lo": State(
            O2_MOLECULE,
            "X",
            3,
            TermSymbol.SIGMA,
            InversionSymmetry.GERADE,
            ReflectionSymmetry.MINUS,
            ConstantsType.PERLEVEL,
        ),
    },
    {
        "name": "14N16O A-X",
        "molecule": NO_MOLECULE,
        "state_up": State(
            NO_MOLECULE,
            "A",
            2,
            TermSymbol.SIGMA,
            InversionSymmetry.NONE,
            ReflectionSymmetry.PLUS,
            ConstantsType.PERLEVEL,
        ),
        "state_lo": State(
            NO_MOLECULE,
            "X",
            2,
            TermSymbol.PI,
            InversionSymmetry.NONE,
            ReflectionSymmetry.NONE,
            ConstantsType.PERLEVEL,
        ),
    },
    {
        "name": "14N16O B-X",
        "molecule": NO_MOLECULE,
        "state_up": State(
            NO_MOLECULE,
            "B",
            2,
            TermSymbol.PI,
            InversionSymmetry.NONE,
            ReflectionSymmetry.NONE,
            ConstantsType.PERLEVEL,
        ),
        "state_lo": State(
            NO_MOLECULE,
            "X",
            2,
            TermSymbol.PI,
            InversionSymmetry.NONE,
            ReflectionSymmetry.NONE,
            ConstantsType.PERLEVEL,
        ),
    },
    {
        "name": "16O1H A-X",
        "molecule": OH_MOLECULE,
        "state_up": State(
            OH_MOLECULE,
            "A",
            2,
            TermSymbol.SIGMA,
            InversionSymmetry.NONE,
            ReflectionSymmetry.PLUS,
            ConstantsType.DUNHAM,
        ),
        "state_lo": State(
            OH_MOLECULE,
            "X",
            2,
            TermSymbol.PI,
            InversionSymmetry.NONE,
            ReflectionSymmetry.NONE,
            ConstantsType.DUNHAM,
        ),
    },
    {
        "name": "Custom",
        "molecule": DEFAULT_MOLECULE,
        "state_up": DEFAULT_STATE_UP,
        "state_lo": DEFAULT_STATE_LO,
    },
]


class MyDoubleSpinBox(QDoubleSpinBox):
    """A custom double spin box.

    Allows for arbitrarily large or small input values, high decimal precision, and scientific
    notation.
    """

    def __init__(self) -> None:
        """Initialize class variables."""
        super().__init__()
        self.setRange(-1e99, 1e99)
        self.setDecimals(6)
        self.setKeyboardTracking(False)

    def valueFromText(self, text: str) -> float:  # noqa: N802
        """Reads text from the input field and converts it to a float.

        Args:
            text: Input text.

        Returns:
            Converted value.
        """
        try:
            return float(text)
        except ValueError:
            return 0.0

    def textFromValue(self, value: float) -> str:  # noqa: N802
        """Displays the input parameter using f-strings.

        Args:
            value: Input float.

        Returns:
            Displayed text.
        """
        return f"{value:g}"

    def validate(self, text: str, pos: int) -> tuple[QValidator.State, str, int]:
        """Checks user input and returns the correct state.

        Args:
            text: Input text.
            pos: Position in the string.

        Returns:
            The current state, input text, and string position.
        """
        if text == "":
            return (QValidator.State.Intermediate, text, pos)

        try:
            float(text)
            return (QValidator.State.Acceptable, text, pos)
        except ValueError:
            # If the string cannot immediately be converted to a float, first check for scientific
            # notation and return an intermediate state if found.
            if "e" in text.lower():
                parts = text.lower().split("e")
                num_parts = 2

                if len(parts) == num_parts and (parts[1] == "" or parts[1] in ["-", "+"]):
                    return (QValidator.State.Intermediate, text, pos)

            return (QValidator.State.Invalid, text, pos)


class MyTable(QAbstractTableModel):
    """A simple model to interface a Qt view with a Polars DataFrame."""

    def __init__(self, df: pl.DataFrame) -> None:
        """Initialize class variables.

        Args:
            df: A Polars `DataFrame`.
        """
        super().__init__()
        self.df = df

    def rowCount(self, _: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        """Get the height of the table.

        Args:
            _: Parent class used to locate data. Defaults to QModelIndex().

        Returns:
            Table height.
        """
        return self.df.height

    def columnCount(self, _: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        """Get the width of the table.

        Args:
            _: Parent class used to locate data. Defaults to QModelIndex().

        Returns:
            Table width.
        """
        return self.df.width

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> str | None:
        """Validates and formats data displayed in the table.

        Args:
            index: Parent class used to locate data.
            role: Data rendered as text. Defaults to Qt.ItemDataRole.DisplayRole.

        Returns:
            The formatted text.
        """
        if not index.isValid():
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            value: int | float | str = self.df[index.row(), index.column()]
            column_name = self.df.columns[index.column()]

            # NOTE: 25/04/10 - This only changes the values displayed to the user using the built-in
            #       table view. If the table is exported, the underlying dataframe is used instead,
            #       which retains the full-precision values calculated by the simulation.
            if isinstance(value, float):
                if "Intensity" in column_name:
                    return f"{value:.4e}"
                return f"{value:.4f}"

            return str(value)
        return None

    def headerData(  # noqa: N802
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ) -> str | None:
        """Sets data for the given role and section in the header with the specified orientation.

        Args:
            section: For horizontal headers, the section number corresponds to the column number.
                For vertical headers, the section number corresponds to the row number.
            orientation: The header orientation.
            role: Data rendered as text. Defaults to Qt.ItemDataRole.DisplayRole.

        Returns:
            Header data.
        """
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self.df.columns[section])
        if orientation == Qt.Orientation.Vertical:
            return str(section)
        return None


def create_dataframe_tab(df: pl.DataFrame, _: str) -> QWidget:
    """Create a QWidget containing a QTableView to display the DataFrame.

    Args:
        df: Data to be displayed in the tab.
        _: The tab label.

    Returns:
        A QWidget containing a QTableView to display the DataFrame.
    """
    widget = QWidget()
    layout = QVBoxLayout(widget)
    table_view = QTableView()
    model = MyTable(df)
    table_view.setModel(model)

    # TODO: 25/04/10 - Enabling column resizing dramatically increases the time it takes to render
    #       tables with even a moderate number of bands. Keeping this disabled unless there's a
    #       faster way to achieve the same thing.

    # table_view.resizeColumnsToContents()

    layout.addWidget(table_view)

    return widget


class PresetSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Simulation")
        self.setModal(True)

        self.selected_preset = None

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Choose a preset, or build your own."))

        self.preset_combo = QComboBox()
        for preset in MOLECULAR_PRESETS:
            self.preset_combo.addItem(preset["name"])
        layout.addWidget(self.preset_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self):
        selected_index = self.preset_combo.currentIndex()
        self.selected_preset = MOLECULAR_PRESETS[selected_index]
        super().accept()


class ParametersDialog(QDialog):
    def __init__(self, tab, context_name=""):
        super().__init__(tab)
        self.setWindowTitle(f"{context_name} Parameters")

        # Prevent modification of the main window while the dialog box is open.
        self.setModal(True)
        self.resize(600, 400)
        self.tab = tab

        self.atom_1_mass = QSpinBox()
        self.atom_1_mass.setRange(0, 400)
        self.atom_1_symbol = QLineEdit()

        atom1_row = QWidget()
        atom1_layout = QHBoxLayout(atom1_row)
        atom1_layout.setContentsMargins(0, 0, 0, 0)
        atom1_layout.addWidget(self.atom_1_mass)
        atom1_layout.addWidget(self.atom_1_symbol)

        self.atom_2_mass = QSpinBox()
        self.atom_2_mass.setRange(0, 400)
        self.atom_2_symbol = QLineEdit()

        atom2_row = QWidget()
        atom2_layout = QHBoxLayout(atom2_row)
        atom2_layout.setContentsMargins(0, 0, 0, 0)
        atom2_layout.addWidget(self.atom_2_mass)
        atom2_layout.addWidget(self.atom_2_symbol)

        molecule_form = QFormLayout()
        molecule_form.addRow(QLabel(f"<b>Molecule: {tab.molecule.name}</b>"))
        molecule_form.addRow("Atom 1 (A, symbol):", atom1_row)
        molecule_form.addRow("Atom 2 (A, symbol):", atom2_row)

        self.letter_up = QLineEdit()
        self.spin_multiplicity_up = QLineEdit()
        self.term_symbol_up = QComboBox()
        self.inversion_symmetry_up = QComboBox()
        self.reflection_symmetry_up = QComboBox()
        self.constants_type_up = QComboBox()

        term_symbols = [t.name for t in TermSymbol]
        inv_syms = [i.name for i in InversionSymmetry]
        ref_syms = [r.name for r in ReflectionSymmetry]
        const_types = [c.name for c in ConstantsType]

        self.term_symbol_up.addItems(term_symbols)
        self.inversion_symmetry_up.addItems(inv_syms)
        self.reflection_symmetry_up.addItems(ref_syms)
        self.constants_type_up.addItems(const_types)

        state_up_form = QFormLayout()
        state_up_form.addRow(QLabel("<b>Upper State</b>"))
        state_up_form.addRow("Letter:", self.letter_up)
        state_up_form.addRow("Spin Mult:", self.spin_multiplicity_up)
        state_up_form.addRow("Term Symbol:", self.term_symbol_up)
        state_up_form.addRow("Inversion:", self.inversion_symmetry_up)
        state_up_form.addRow("Reflection:", self.reflection_symmetry_up)
        state_up_form.addRow("Constants:", self.constants_type_up)

        self.letter_lo = QLineEdit()
        self.spin_multiplicity_lo = QLineEdit()
        self.term_symbol_lo = QComboBox()
        self.inversion_symmetry_lo = QComboBox()
        self.reflection_symmetry_lo = QComboBox()
        self.constants_type_lo = QComboBox()

        self.term_symbol_lo.addItems(term_symbols)
        self.inversion_symmetry_lo.addItems(inv_syms)
        self.reflection_symmetry_lo.addItems(ref_syms)
        self.constants_type_lo.addItems(const_types)

        state_lo_form = QFormLayout()
        state_lo_form.addRow(QLabel("<b>Lower State</b>"))
        state_lo_form.addRow("Letter:", self.letter_lo)
        state_lo_form.addRow("Spin Mult:", self.spin_multiplicity_lo)
        state_lo_form.addRow("Term Symbol:", self.term_symbol_lo)
        state_lo_form.addRow("Inversion:", self.inversion_symmetry_lo)
        state_lo_form.addRow("Reflection:", self.reflection_symmetry_lo)
        state_lo_form.addRow("Constants:", self.constants_type_lo)

        self.sim_type = QComboBox()
        self.sim_type.addItems([s.name for s in SimType])
        self.temp_trn = QLineEdit()
        self.temp_elc = QLineEdit()
        self.temp_vib = QLineEdit()
        self.temp_rot = QLineEdit()
        self.pressure = QLineEdit()

        sim_form = QFormLayout()
        sim_form.addRow(QLabel("<b>Simulation</b>"))
        sim_form.addRow("Type:", self.sim_type)
        sim_form.addRow("T_trn [K]:", self.temp_trn)
        sim_form.addRow("T_elc [K]:", self.temp_elc)
        sim_form.addRow("T_vib [K]:", self.temp_vib)
        sim_form.addRow("T_rot [K]:", self.temp_rot)
        sim_form.addRow("Pressure [Pa]:", self.pressure)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(molecule_form)

        columns = QHBoxLayout()
        columns.addLayout(state_up_form)
        columns.addLayout(state_lo_form)
        columns.addLayout(sim_form)
        main_layout.addLayout(columns)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

        # Set values from the parent tab.
        self.atom_1_symbol.setText(tab.molecule.atom_1.chemical_symbol)
        self.atom_2_symbol.setText(tab.molecule.atom_2.chemical_symbol)
        self.atom_1_mass.setValue(tab.molecule.atom_1.atomic_mass_number)
        self.atom_2_mass.setValue(tab.molecule.atom_2.atomic_mass_number)
        self.letter_up.setText(tab.state_up.letter)
        self.letter_lo.setText(tab.state_lo.letter)
        self.spin_multiplicity_up.setText(str(tab.state_up.spin_multiplicity))
        self.spin_multiplicity_lo.setText(str(tab.state_lo.spin_multiplicity))
        self.inversion_symmetry_up.setCurrentText(tab.state_up.inversion_symmetry.name)
        self.inversion_symmetry_lo.setCurrentText(tab.state_lo.inversion_symmetry.name)
        self.reflection_symmetry_up.setCurrentText(tab.state_up.reflection_symmetry.name)
        self.reflection_symmetry_lo.setCurrentText(tab.state_lo.reflection_symmetry.name)
        self.constants_type_up.setCurrentText(tab.state_up.constants_type.name)
        self.constants_type_lo.setCurrentText(tab.state_lo.constants_type.name)
        self.term_symbol_up.setCurrentText(tab.state_up.term_symbol.name)
        self.term_symbol_lo.setCurrentText(tab.state_lo.term_symbol.name)
        self.sim_type.setCurrentText(tab.sim.sim_type.name)
        self.temp_trn.setText(str(tab.sim.temp_params.translational))
        self.temp_elc.setText(str(tab.sim.temp_params.electronic))
        self.temp_vib.setText(str(tab.sim.temp_params.vibrational))
        self.temp_rot.setText(str(tab.sim.temp_params.rotational))
        self.pressure.setText(str(tab.sim.pressure))

    def accept(self):
        self.tab.molecule = Molecule(
            Atom(self.atom_1_mass.value(), self.atom_1_symbol.text()),
            Atom(self.atom_2_mass.value(), self.atom_2_symbol.text()),
        )

        self.tab.state_up = State(
            self.tab.molecule,
            self.letter_up.text(),
            int(self.spin_multiplicity_up.text()),
            TermSymbol[self.term_symbol_up.currentText()],
            InversionSymmetry[self.inversion_symmetry_up.currentText()],
            ReflectionSymmetry[self.reflection_symmetry_up.currentText()],
            ConstantsType[self.constants_type_up.currentText()],
        )

        self.tab.state_lo = State(
            self.tab.molecule,
            self.letter_lo.text(),
            int(self.spin_multiplicity_lo.text()),
            TermSymbol[self.term_symbol_lo.currentText()],
            InversionSymmetry[self.inversion_symmetry_lo.currentText()],
            ReflectionSymmetry[self.reflection_symmetry_lo.currentText()],
            ConstantsType[self.constants_type_lo.currentText()],
        )

        temp_params = TemperatureParams(
            translational=float(self.temp_trn.text()),
            electronic=float(self.temp_elc.text()),
            vibrational=float(self.temp_vib.text()),
            rotational=float(self.temp_rot.text()),
        )

        self.tab.sim = Sim(
            sim_type=SimType[self.sim_type.currentText()],
            molecule=self.tab.molecule,
            state_up=self.tab.state_up,
            state_lo=self.tab.state_lo,
            j_qn_up_max=self.tab.maxj_spinbox.value(),
            pressure=float(self.pressure.text()),
            bands_input=self.tab.get_current_bands(),
            temp_params=temp_params,
            laser_params=self.tab.sim.laser_params,
            inst_params=self.tab.sim.inst_params,
            shift_params=self.tab.sim.shift_params,
            shift_bools=self.tab.sim.shift_bools,
            broad_bools=self.tab.sim.broad_bools,
        )

        super().accept()


class CustomTab(QWidget):
    def __init__(self, parent_tab_widget=None):
        super().__init__()

        self.is_first_run: bool = True

        self.parent_tab_widget = parent_tab_widget

        self.molecule = DEFAULT_MOLECULE
        self.state_up = DEFAULT_STATE_UP
        self.state_lo = DEFAULT_STATE_LO
        self.sim = DEFAULT_SIM

        main_layout = QVBoxLayout(self)

        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)

        self.btn_params = QPushButton("Parameters")
        self.btn_params.clicked.connect(self.open_parameters_dialog)
        main_layout.addWidget(self.btn_params)

        group_bands = QGroupBox("Bands")
        group_bands.setFixedWidth(300)
        bands_layout = QVBoxLayout(group_bands)

        band_selection_layout = QHBoxLayout()
        self.radio_specific_bands = QRadioButton("Specific Bands")
        self.radio_specific_bands.setChecked(True)
        self.radio_band_ranges = QRadioButton("Band Ranges")
        band_selection_layout.addWidget(self.radio_specific_bands)
        band_selection_layout.addWidget(self.radio_band_ranges)
        bands_layout.addLayout(band_selection_layout)

        self.radio_specific_bands.toggled.connect(self.toggle_band_input_method)
        self.radio_specific_bands.toggled.connect(self.update_sim_objects)
        self.radio_band_ranges.toggled.connect(self.update_sim_objects)

        self.specific_bands_container = QWidget()
        self.specific_bands_container.setFixedHeight(60)
        specific_bands_layout = QVBoxLayout(self.specific_bands_container)
        specific_bands_layout.setContentsMargins(0, 0, 0, 0)

        band_layout = QHBoxLayout()
        band_label = QLabel("Bands:")
        self.band_line_edit = QLineEdit(DEFAULT_BANDS)
        self.band_line_edit.textChanged.connect(self.update_sim_objects)
        band_layout.addWidget(band_label)
        band_layout.addWidget(self.band_line_edit)
        specific_bands_layout.addLayout(band_layout)

        self.band_ranges_container = QWidget()
        self.band_ranges_container.setFixedHeight(60)
        band_ranges_layout = QGridLayout(self.band_ranges_container)
        band_ranges_layout.setContentsMargins(0, 0, 0, 0)

        v_up_min_label = QLabel("v' min:")
        self.v_up_min_spinbox = QSpinBox()
        self.v_up_min_spinbox.setRange(0, 99)
        self.v_up_min_spinbox.setValue(0)
        self.v_up_min_spinbox.valueChanged.connect(self.update_sim_objects)

        v_up_max_label = QLabel("v' max:")
        self.v_up_max_spinbox = QSpinBox()
        self.v_up_max_spinbox.setRange(0, 99)
        self.v_up_max_spinbox.setValue(5)
        self.v_up_max_spinbox.valueChanged.connect(self.update_sim_objects)

        v_lo_min_label = QLabel("v'' min:")
        self.v_lo_min_spinbox = QSpinBox()
        self.v_lo_min_spinbox.setRange(0, 99)
        self.v_lo_min_spinbox.setValue(5)
        self.v_lo_min_spinbox.valueChanged.connect(self.update_sim_objects)

        v_lo_max_label = QLabel("v'' max:")
        self.v_lo_max_spinbox = QSpinBox()
        self.v_lo_max_spinbox.setRange(0, 99)
        self.v_lo_max_spinbox.setValue(5)
        self.v_lo_max_spinbox.valueChanged.connect(self.update_sim_objects)

        band_ranges_layout.addWidget(v_up_min_label, 0, 0)
        band_ranges_layout.addWidget(self.v_up_min_spinbox, 0, 1)
        band_ranges_layout.addWidget(v_up_max_label, 0, 2)
        band_ranges_layout.addWidget(self.v_up_max_spinbox, 0, 3)

        band_ranges_layout.addWidget(v_lo_min_label, 1, 0)
        band_ranges_layout.addWidget(self.v_lo_min_spinbox, 1, 1)
        band_ranges_layout.addWidget(v_lo_max_label, 1, 2)
        band_ranges_layout.addWidget(self.v_lo_max_spinbox, 1, 3)

        self.band_ranges_container.hide()

        bands_layout.addWidget(self.specific_bands_container)
        bands_layout.addWidget(self.band_ranges_container)
        controls_layout.addWidget(group_bands)

        group_maxj = QGroupBox("Max J'")
        maxj_layout = QHBoxLayout(group_maxj)
        self.maxj_spinbox = QSpinBox()
        self.maxj_spinbox.setMaximum(10000)
        self.maxj_spinbox.setValue(DEFAULT_J_MAX)
        self.maxj_spinbox.valueChanged.connect(self.update_sim_objects)
        maxj_layout.addWidget(self.maxj_spinbox)
        controls_layout.addWidget(group_maxj)

        group_resolution = QGroupBox("Resolution")
        resolution_layout = QHBoxLayout(group_resolution)
        self.resolution_spinbox = QSpinBox()
        self.resolution_spinbox.setMaximum(10000000)
        self.resolution_spinbox.setValue(DEFAULT_RESOLUTION)
        resolution_layout.addWidget(self.resolution_spinbox)
        controls_layout.addWidget(group_resolution)

        group_plot_type = QGroupBox("Plot Type")
        plot_type_layout = QHBoxLayout(group_plot_type)
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(
            ["Line", "Line Info", "Continuous Separate", "Continuous All"]
        )
        plot_type_layout.addWidget(self.plot_type_combo)
        controls_layout.addWidget(group_plot_type)

        group_limits = QGroupBox("Plot Limits")
        limits_layout = QVBoxLayout(group_limits)
        limit_params_layout = QVBoxLayout()

        limit_min_layout = QHBoxLayout()
        limit_min_layout.addWidget(QLabel("Minimum:"))
        self.limit_min_spinbox = MyDoubleSpinBox()
        self.limit_min_spinbox.setValue(0.0)
        self.limit_min_spinbox.setSuffix(" [nm]")
        self.limit_min_spinbox.valueChanged.connect(self.update_sim_objects)
        limit_min_layout.addWidget(self.limit_min_spinbox)
        limit_params_layout.addLayout(limit_min_layout)

        limit_max_layout = QHBoxLayout()
        limit_max_layout.addWidget(QLabel("Maximum:"))
        self.limit_max_spinbox = MyDoubleSpinBox()
        self.limit_max_spinbox.setValue(10000.0)
        self.limit_max_spinbox.setSuffix(" [nm]")
        self.limit_max_spinbox.valueChanged.connect(self.update_sim_objects)
        limit_max_layout.addWidget(self.limit_max_spinbox)
        limit_params_layout.addLayout(limit_max_layout)

        limit_checkbox_layout = QHBoxLayout()
        self.checkbox_plot_limits = QCheckBox("Enabled")
        self.checkbox_plot_limits.toggled.connect(self.update_sim_objects)
        limit_checkbox_layout.addWidget(self.checkbox_plot_limits)

        limits_layout.addLayout(limit_params_layout)
        limits_layout.addLayout(limit_checkbox_layout)
        controls_layout.addWidget(group_limits)

        group_shift = QGroupBox("Line Shift")
        shift_layout = QVBoxLayout(group_shift)
        shift_params_layout = QHBoxLayout()

        coll_shift_a_layout = QVBoxLayout()
        coll_shift_a_layout.addWidget(QLabel("a"))
        self.coll_shift_a_spinbox = MyDoubleSpinBox()
        self.coll_shift_a_spinbox.setValue(0.0)
        self.coll_shift_a_spinbox.setSuffix(" [cm⁻¹]")
        self.coll_shift_a_spinbox.valueChanged.connect(self.update_sim_objects)
        coll_shift_a_layout.addWidget(self.coll_shift_a_spinbox)
        shift_params_layout.addLayout(coll_shift_a_layout)

        coll_shift_b_layout = QVBoxLayout()
        coll_shift_b_layout.addWidget(QLabel("b"))
        self.coll_shift_b_spinbox = MyDoubleSpinBox()
        self.coll_shift_b_spinbox.setValue(0.0)
        self.coll_shift_b_spinbox.setSuffix(" [-]")
        self.coll_shift_b_spinbox.valueChanged.connect(self.update_sim_objects)
        coll_shift_b_layout.addWidget(self.coll_shift_b_spinbox)
        shift_params_layout.addLayout(coll_shift_b_layout)

        shift_layout.addLayout(shift_params_layout)

        shift_checkbox_layout = QHBoxLayout()
        self.checkbox_collisional_shift = QCheckBox("Collisional")
        self.checkbox_doppler_shift = QCheckBox("Doppler")

        checkboxes = [
            self.checkbox_collisional_shift,
            self.checkbox_doppler_shift,
        ]

        for i, cb in enumerate(checkboxes):
            cb.toggled.connect(self.update_sim_objects)
            shift_checkbox_layout.addWidget(cb)

        shift_layout.addLayout(shift_checkbox_layout)
        controls_layout.addWidget(group_shift)

        group_broadening = QGroupBox("Broadening")
        broadening_layout = QVBoxLayout(group_broadening)
        broadening_params_layout = QHBoxLayout()

        inst_broad_gauss_layout = QVBoxLayout()
        inst_broad_gauss_layout.addWidget(QLabel("Gauss. FWHM"))
        self.inst_broad_gauss_spinbox = MyDoubleSpinBox()
        self.inst_broad_gauss_spinbox.setValue(DEFAULT_BROADENING)
        self.inst_broad_gauss_spinbox.setSuffix(" [nm]")
        self.inst_broad_gauss_spinbox.valueChanged.connect(self.update_sim_objects)
        inst_broad_gauss_layout.addWidget(self.inst_broad_gauss_spinbox)
        broadening_params_layout.addLayout(inst_broad_gauss_layout)

        inst_broad_loren_layout = QVBoxLayout()
        inst_broad_loren_layout.addWidget(QLabel("Loren. FWHM"))
        self.inst_broad_loren_spinbox = MyDoubleSpinBox()
        self.inst_broad_loren_spinbox.setValue(DEFAULT_BROADENING)
        self.inst_broad_loren_spinbox.setSuffix(" [nm]")
        self.inst_broad_loren_spinbox.valueChanged.connect(self.update_sim_objects)
        inst_broad_loren_layout.addWidget(self.inst_broad_loren_spinbox)
        broadening_params_layout.addLayout(inst_broad_loren_layout)

        laser_power_layout = QVBoxLayout()
        laser_power_layout.addWidget(QLabel("Laser Power"))
        self.laser_power_spinbox = MyDoubleSpinBox()
        self.laser_power_spinbox.setValue(0.0)
        self.laser_power_spinbox.setSuffix(" [W]")
        self.laser_power_spinbox.valueChanged.connect(self.update_sim_objects)
        laser_power_layout.addWidget(self.laser_power_spinbox)
        broadening_params_layout.addLayout(laser_power_layout)

        beam_diameter_layout = QVBoxLayout()
        beam_diameter_layout.addWidget(QLabel("Beam Diameter"))
        self.beam_diameter_spinbox = MyDoubleSpinBox()
        self.beam_diameter_spinbox.setValue(1.0)
        self.beam_diameter_spinbox.setSuffix(" [mm]")
        self.beam_diameter_spinbox.valueChanged.connect(self.update_sim_objects)
        beam_diameter_layout.addWidget(self.beam_diameter_spinbox)
        broadening_params_layout.addLayout(beam_diameter_layout)

        transit_layout = QVBoxLayout()
        transit_layout.addWidget(QLabel("Molecule Velocity"))
        self.transit_spinbox = MyDoubleSpinBox()
        self.transit_spinbox.setValue(0.0)
        self.transit_spinbox.setSuffix(" [m/s]")
        self.transit_spinbox.valueChanged.connect(self.update_sim_objects)
        transit_layout.addWidget(self.transit_spinbox)
        broadening_params_layout.addLayout(transit_layout)

        broadening_layout.addLayout(broadening_params_layout)

        checkbox_layout = QHBoxLayout()
        self.checkbox_instrument = QCheckBox("Instrument")
        self.checkbox_doppler = QCheckBox("Doppler")
        self.checkbox_natural = QCheckBox("Natural")
        self.checkbox_collisional = QCheckBox("Collisional")
        self.checkbox_predissociation = QCheckBox("Predissociation")
        self.checkbox_power = QCheckBox("Power")
        self.checkbox_transit = QCheckBox("Transit")

        checkboxes = [
            self.checkbox_instrument,
            self.checkbox_doppler,
            self.checkbox_natural,
            self.checkbox_collisional,
            self.checkbox_predissociation,
            self.checkbox_power,
            self.checkbox_transit,
        ]

        for i, cb in enumerate(checkboxes):
            cb.setChecked(True)
            cb.toggled.connect(self.update_sim_objects)
            checkbox_layout.addWidget(cb)

        broadening_layout.addLayout(checkbox_layout)
        controls_layout.addWidget(group_broadening)

        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation)
        actions_layout.addWidget(self.run_button)

        self.open_sample_button = QPushButton("Open Sample")
        self.open_sample_button.clicked.connect(self.open_sample)
        actions_layout.addWidget(self.open_sample_button)

        self.export_button = QPushButton("Export Table")
        self.export_button.clicked.connect(self.export_current_table)
        actions_layout.addWidget(self.export_button)

        controls_layout.addWidget(actions_group)
        main_layout.addWidget(controls_widget)

        plot_table_widget = QWidget()
        plot_table_layout = QHBoxLayout(plot_table_widget)

        self.table_tab_widget = QTabWidget()
        empty_df = pl.DataFrame()
        page = create_dataframe_tab(empty_df, "v'-v''")
        self.table_tab_widget.addTab(page, "v'-v''")
        plot_table_layout.addWidget(self.table_tab_widget, stretch=1)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.addLegend(offset=(0, 1))
        self.plot_widget.setAxisItems({"top": WavelengthAxis(orientation="top")})
        self.plot_widget.setLabel("top", "Wavelength, λ [nm]")
        self.plot_widget.setLabel("bottom", "Wavenumber, ν [cm⁻¹]")
        self.plot_widget.setLabel("left", "Intensity, I [a.u.]")
        self.plot_widget.setLabel("right", "")
        self.plot_widget.setXRange(40000, 50000)
        plot_table_layout.addWidget(self.plot_widget, stretch=2)

        main_layout.addWidget(plot_table_widget, stretch=1)

    def update_tab_name(self):
        if self.parent_tab_widget is not None:
            current_index = self.parent_tab_widget.indexOf(self)
            if current_index != -1:
                tab_name = f"{self.molecule.name} ({self.state_up.letter}→{self.state_lo.letter})"
                self.parent_tab_widget.setTabText(current_index, tab_name)

    def toggle_band_input_method(self, checked: bool) -> None:
        if checked:
            self.specific_bands_container.show()
            self.band_ranges_container.hide()
        else:
            self.specific_bands_container.hide()
            self.band_ranges_container.show()

    def get_current_bands(self) -> list[tuple[int, int]]:
        if self.radio_specific_bands.isChecked():
            return self.parse_specific_bands()
        return self.generate_band_ranges()

    def parse_specific_bands(self) -> list[tuple[int, int]]:
        band_ranges_str = self.band_line_edit.text()
        bands: list[tuple[int, int]] = []

        for range_str in band_ranges_str.split(","):
            range_str = range_str.strip()
            if "-" in range_str:
                try:
                    v_up, v_lo = map(int, range_str.split("-"))
                    bands.append((v_up, v_lo))
                except ValueError:
                    continue

        return bands if bands else [(0, 0)]

    def generate_band_ranges(self) -> list[tuple[int, int]]:
        v_up_min = self.v_up_min_spinbox.value()
        v_up_max = self.v_up_max_spinbox.value()
        v_lo_min = self.v_lo_min_spinbox.value()
        v_lo_max = self.v_lo_max_spinbox.value()

        if v_up_min > v_up_max or v_lo_min > v_lo_max:
            return [(0, 0)]

        if (v_up_min == v_up_max) and (v_lo_min == v_lo_max):
            return [(v_up_min, v_lo_min)]
        if v_up_min == v_up_max:
            return [(v_up_min, v_lo) for v_lo in range(v_lo_min, v_lo_max + 1)]
        if v_lo_min == v_lo_max:
            return [(v_up, v_lo_min) for v_up in range(v_up_min, v_up_max + 1)]
        return [
            (v_up, v_lo)
            for v_up in range(v_up_min, v_up_max + 1)
            for v_lo in range(v_lo_min, v_lo_max + 1)
        ]

    def update_sim_objects(self) -> None:
        current_bands = self.get_current_bands()

        inst_params = InstrumentParams(
            gauss_fwhm_wl=self.inst_broad_gauss_spinbox.value(),
            loren_fwhm_wl=self.inst_broad_loren_spinbox.value(),
        )

        laser_params = LaserParams(
            power_w=self.laser_power_spinbox.value(),
            beam_diameter_mm=self.beam_diameter_spinbox.value(),
            molecule_velocity_ms=self.transit_spinbox.value(),
        )

        shift_params = ShiftParams(
            collisional_a=self.coll_shift_a_spinbox.value(),
            collisional_b=self.coll_shift_b_spinbox.value(),
        )

        shift_bools = ShiftBools(
            collisional=self.checkbox_collisional_shift.isChecked(),
            doppler=self.checkbox_doppler_shift.isChecked(),
        )

        broad_bools = BroadeningBools(
            collisional=self.checkbox_collisional.isChecked(),
            doppler=self.checkbox_doppler.isChecked(),
            instrument=self.checkbox_instrument.isChecked(),
            natural=self.checkbox_natural.isChecked(),
            power=self.checkbox_power.isChecked(),
            predissociation=self.checkbox_predissociation.isChecked(),
            transit=self.checkbox_transit.isChecked(),
        )

        plot_bools = PlotBools(limits=self.checkbox_plot_limits.isChecked())

        plot_params = PlotParams(
            limit_min=self.limit_min_spinbox.value(), limit_max=self.limit_max_spinbox.value()
        )

        # Only update the values that are controlled by the tab specifically.
        self.sim = Sim(
            sim_type=self.sim.sim_type,
            molecule=self.molecule,
            state_up=self.state_up,
            state_lo=self.state_lo,
            j_qn_up_max=self.maxj_spinbox.value(),
            pressure=self.sim.pressure,
            bands_input=current_bands,
            temp_params=self.sim.temp_params,
            laser_params=laser_params,
            inst_params=inst_params,
            shift_bools=shift_bools,
            shift_params=shift_params,
            broad_bools=broad_bools,
            plot_bools=plot_bools,
            plot_params=plot_params,
        )

    def run_simulation(self) -> None:
        """Run a simulation instance for this specific tab."""
        bands = self.get_current_bands()
        colors = get_colors(len(bands))

        self.plot_widget.clear()

        map_functions: dict[str, Callable] = {
            "Line": plot.plot_line,
            "Line Info": plot.plot_line_info,
            "Continuous Separate": plot.plot_cont_sep,
            "Continuous All": plot.plot_cont_all,
        }
        plot_type = self.plot_type_combo.currentText()
        plot_function: Callable | None = map_functions.get(plot_type)

        if plot_function is not None:
            if plot_function.__name__ in ("plot_cont_sep", "plot_cont_all"):
                plot_function(
                    self.plot_widget,
                    self.sim,
                    colors,
                    self.resolution_spinbox.value(),
                )
            else:
                plot_function(self.plot_widget, self.sim, colors)
        else:
            QMessageBox.information(
                self,
                "Info",
                f"Plot type '{plot_type}' is not recognized.",
                QMessageBox.StandardButton.Ok,
            )

        # TODO: 25/08/18 - Might want to auto range if the bands are updated.
        if self.is_first_run:
            self.plot_widget.autoRange()
            self.is_first_run = False

        while self.table_tab_widget.count() > 0:
            self.table_tab_widget.removeTab(0)

        for i, band in enumerate(bands):
            df = pl.DataFrame(
                [
                    {
                        "Wavelength": utils.wavenum_to_wavelen(line.wavenumber),
                        "Wavenumber": line.wavenumber,
                        "Intensity": line.intensity,
                        "J'": f"{line.j_qn_up:.1f}",
                        "J''": f"{line.j_qn_lo:.1f}",
                        "N'": f"{line.n_qn_up:.1f}",
                        "N''": f"{line.n_qn_lo:.1f}",
                        "ΔJ Branch": f"{line.branch_name_j}{line.branch_idx_up}{line.branch_idx_lo}",
                        "ΔN Branch": f"{line.branch_name_n}{line.branch_idx_up}{line.branch_idx_lo}",
                    }
                    for line in self.sim.bands[i].lines
                ]
            )

            tab_name = f"{band[0]}-{band[1]}"
            new_tab = create_dataframe_tab(df, tab_name)
            self.table_tab_widget.addTab(new_tab, tab_name)

    def export_current_table(self) -> None:
        """Export the currently displayed table to a CSV file."""
        current_widget = self.table_tab_widget.currentWidget()

        table_view = current_widget.findChild(QTableView)

        if table_view is None:
            QMessageBox.information(
                self,
                "Error",
                "No table view found in this tab.",
                QMessageBox.StandardButton.Ok,
            )
            return

        model: MyTable = table_view.model()  # pyright: ignore[reportAssignmentType]
        if not hasattr(model, "df"):
            QMessageBox.information(
                self,
                "Error",
                "This table does not support CSV export.",
                QMessageBox.StandardButton.Ok,
            )
            return

        df = model.df

        filename, _ = QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV Files (*.csv)")
        if filename:
            try:
                df.write_csv(filename)
                QMessageBox.information(
                    self, "Success", "Table exported successfully.", QMessageBox.StandardButton.Ok
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"An error occurred: {e}")

    def open_sample(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Open Sample File",
            dir=str(data_path.get_data_path("data", "samples")),
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

        display_name = Path(filename).name

        new_table_tab = create_dataframe_tab(df, display_name)
        self.table_tab_widget.addTab(new_table_tab, display_name)

        if "wavenumber" in df.columns:
            x_values = df["wavenumber"].to_numpy()
            value_type = "wavenumber"
        elif "wavelength" in df.columns:
            x_values = df["wavelength"].to_numpy()
            value_type = "wavelength"
        else:
            QMessageBox.critical(self, "Error", "No 'wavenumber' or 'wavelength' column found.")
            return

        try:
            intensities = df["intensity"].to_numpy()
        except pl.exceptions.ColumnNotFoundError:
            QMessageBox.critical(self, "Error", "No 'intensity' column found.")
            return

        plot.plot_sample(self.plot_widget, x_values, intensities, display_name, value_type)

    def open_parameters_dialog(self):
        dialog = ParametersDialog(self, context_name=self.molecule.name)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.update_sim_objects()
            self.update_tab_name()


class LIFTab(QWidget):
    def __init__(self, parent_tab_widget: QTabWidget):
        super().__init__()
        self.parent_tab_widget = parent_tab_widget

        layout = QVBoxLayout(self)

        sim_group = QGroupBox("Parent Simulation")
        sim_layout = QHBoxLayout(sim_group)
        self.sim_selector = QComboBox()
        sim_layout.addWidget(QLabel("Use tab:"))
        sim_layout.addWidget(self.sim_selector)
        layout.addWidget(sim_group)

        line_group = QGroupBox("Line Selection")
        line_layout = QHBoxLayout(line_group)
        self.branch_name_j = QComboBox()
        self.branch_name_j.addItems(["P", "Q", "R"])
        self.branch_idx_lo = QSpinBox()
        self.branch_idx_lo.setRange(1, 3)
        self.n_qn_lo = QSpinBox()
        self.n_qn_lo.setRange(0, 200)

        line_layout.addWidget(QLabel("Branch (ΔJ):"))
        line_layout.addWidget(self.branch_name_j)
        line_layout.addWidget(QLabel("Branch index:"))
        line_layout.addWidget(self.branch_idx_lo)
        line_layout.addWidget(QLabel("N'':"))
        line_layout.addWidget(self.n_qn_lo)
        layout.addWidget(line_group)

        laser_group = QGroupBox("Laser Pulse Parameters")
        laser_layout = QHBoxLayout(laser_group)

        self.pulse_center = MyDoubleSpinBox()
        self.pulse_center.setSuffix(" [ns]")
        self.pulse_width = MyDoubleSpinBox()
        self.pulse_width.setSuffix(" [ns]")
        self.fluence = MyDoubleSpinBox()
        self.fluence.setSuffix(" [J/cm²]")

        self.pulse_center.setValue(30.0)
        self.pulse_width.setValue(20.0)
        self.fluence.setValue(25e-3)

        self.max_time = MyDoubleSpinBox()
        self.max_time.setSuffix(" [ns]")
        self.n_points = QSpinBox()
        self.n_points.setMaximum(1000000)

        self.max_time.setValue(60.0)
        self.n_points.setValue(1000)

        laser_layout.addWidget(QLabel("Center:"))
        laser_layout.addWidget(self.pulse_center)
        laser_layout.addWidget(QLabel("Width (FWHM):"))
        laser_layout.addWidget(self.pulse_width)
        laser_layout.addWidget(QLabel("Fluence:"))
        laser_layout.addWidget(self.fluence)
        laser_layout.addWidget(QLabel("Max time:"))
        laser_layout.addWidget(self.max_time)
        laser_layout.addWidget(QLabel("Points:"))
        laser_layout.addWidget(self.n_points)
        layout.addWidget(laser_group)

        button_group = QGroupBox()
        button_layout = QHBoxLayout(button_group)
        self.lifspec_btn = QPushButton("Populations vs. Time")
        self.lifspec_btn.clicked.connect(self.populations_vs_time)
        button_layout.addWidget(self.lifspec_btn)

        self.lifspec_btn = QPushButton("LIF Spectra vs. Time")
        self.lifspec_btn.clicked.connect(self.lif_spectra_vs_time)
        button_layout.addWidget(self.lifspec_btn)
        layout.addWidget(button_group)

        self.plot = pg.PlotWidget()
        self.slice_plot = pg.PlotWidget()
        self.slice_plot.hide()

        self._lif_img_item = None
        self._lif_cbar = None

        self._time_cursor = None

        layout.addWidget(self.slice_plot, stretch=0)
        layout.addWidget(self.plot, stretch=1)

        self.refresh_parent_tabs()

    def refresh_parent_tabs(self):
        self.sim_selector.clear()
        for idx in range(self.parent_tab_widget.count()):
            tab = self.parent_tab_widget.widget(idx)
            if hasattr(tab, "sim"):
                self.sim_selector.addItem(self.parent_tab_widget.tabText(idx), userData=idx)

    def get_parent_tab(self):
        idx = self.sim_selector.currentData()
        if idx is None:
            return None
        return self.parent_tab_widget.widget(idx)

    def get_parent_sim(self):
        idx = self.sim_selector.currentData()
        if idx is None:
            return None
        tab = self.parent_tab_widget.widget(idx)
        return getattr(tab, "sim", None)

    def setup_lif(self):
        pumped_sim = self.get_parent_sim()

        if pumped_sim is None:
            raise ValueError("No pumped simulation.")

        branch_name_j = self.branch_name_j.currentText()
        branch_idx_lo = int(self.branch_idx_lo.value())
        n_qn_lo = int(self.n_qn_lo.value())

        # TODO: 26/02/02 - Actually add dialog boxes here to tell the user what's happened.
        try:
            pumped_line = lif.find_line(pumped_sim, branch_name_j, branch_idx_lo, n_qn_lo)
        except ValueError as e:
            print(str(e))

        # For a given v', get the maximum value of v'' (account for 0-indexing).
        v_qn_lo_max = lif.get_v_qn_lo_max(pumped_sim)
        band_range = [(pumped_sim.bands[0].v_qn_up, v_qn_lo) for v_qn_lo in range(v_qn_lo_max + 1)]

        # The LIF emission simulation is exactly the same as the absorption line simulation except
        # for the simulation type and bands simulated.
        emission_sim = Sim(
            sim_type=SimType.LIF,
            molecule=pumped_sim.molecule,
            state_up=pumped_sim.state_up,
            state_lo=pumped_sim.state_lo,
            j_qn_up_max=pumped_sim.j_qn_up_max,
            pressure=pumped_sim.pressure,
            bands_input=band_range,
            temp_params=pumped_sim.temp_params,
            laser_params=pumped_sim.laser_params,
            inst_params=pumped_sim.inst_params,
            shift_params=pumped_sim.shift_params,
            shift_bools=pumped_sim.shift_bools,
            broad_bools=pumped_sim.broad_bools,
            plot_bools=pumped_sim.plot_bools,
            plot_params=pumped_sim.plot_params,
            pumped_line=pumped_line,
        )

        # Spinbox inputs are in [ns], so convert back to [s].
        laser_params = lif.LIFLaserParams(
            self.pulse_center.value() * 1e-9,
            self.pulse_width.value() * 1e-9,
            self.fluence.value(),
        )

        # Max time also has to be converted from [ns] to [s].
        t_eval = np.linspace(0.0, self.max_time.value() * 1e-9, self.n_points.value())

        return emission_sim, pumped_sim, pumped_line, laser_params, t_eval

    def populations_vs_time(self):
        if self.slice_plot is not None:
            self.slice_plot.hide()

        emission_sim, pumped_sim, pumped_line, laser_params, t_eval = self.setup_lif()

        rate_params = lif.time_independent_rates(emission_sim, pumped_sim, pumped_line)
        n1_hat, n2_hat, n3_hat = lif.simulate(rate_params, laser_params, pumped_line, t_eval)

        # Normalize the signal with respect to N2.
        sf = lif.get_signal(t_eval, n2_hat, rate_params)
        sf /= n2_hat.max()

        # Normalize the laser with respect to itself.
        il = lif.laser_intensity(t_eval, laser_params)
        il /= il.max()

        colors = get_colors(5)

        # Show time in [ns].
        time_ns = t_eval * 1e9

        self.plot.clear()
        self._clear_lif_extras()

        # Reset the top axis since the spectrum plot will mess this up otherwise.
        self.plot.setAxisItems({"top": pg.AxisItem(orientation="top")})
        self.plot.setLabel("top", "")
        self.plot.setLabel("bottom", "Time, t [ns]")
        self.plot.setLabel("left", "N_1, N_3, I_l (Normalized)")
        self.plot.setLabel("right", "N_2, S_f (Normalized)")
        self.plot.addLegend(offset=(0, 1))

        # Left axis.
        self.plot.plot(time_ns, n1_hat, name="N1", pen=pg.mkPen(colors[0], width=1))
        self.plot.plot(time_ns, n3_hat, name="N3", pen=pg.mkPen(colors[1], width=1))
        self.plot.plot(time_ns, il, name="I_l", pen=pg.mkPen(colors[2], width=1))

        # Right axis.
        self.plot.plot(
            time_ns,
            n2_hat,
            name="N2",
            pen=pg.mkPen(colors[3], style=pg.QtCore.Qt.PenStyle.DashLine, width=1),
        )
        self.plot.plot(
            time_ns,
            sf,
            name="S_f",
            pen=pg.mkPen(colors[4], style=pg.QtCore.Qt.PenStyle.DashLine, width=1),
        )
        self.plot.autoRange()

    def lif_spectra_vs_time(self):
        emission_sim, pumped_sim, pumped_line, laser_params, t_eval = self.setup_lif()

        parent_tab = self.get_parent_tab()
        if parent_tab is not None and hasattr(parent_tab, "resolution_spinbox"):
            granularity = int(parent_tab.resolution_spinbox.value())  # pyright: ignore[reportAttributeAccessIssue]
        else:
            granularity = DEFAULT_RESOLUTION

        total_number_density = pumped_sim.pressure / (
            constants.BOLTZ * pumped_sim.temp_params.translational
        )

        # N_{1, 0}, the lower state number density of the pumped line.
        number_density_lo = (
            total_number_density
            * pumped_sim.elc_boltz_frac[1]
            * pumped_line.band.vib_boltz_frac[1]
            * pumped_line.rot_boltz_frac[1]
        )

        rate_params = lif.time_independent_rates(emission_sim, pumped_sim, pumped_line)
        _, n2_hat, _ = lif.simulate(rate_params, laser_params, pumped_line, t_eval)

        n2 = n2_hat * number_density_lo

        wavenumbers_line = np.concatenate([band.wavenumbers_line() for band in emission_sim.bands])
        inst_broadening = max(emission_sim.bands[0].lines[0].fwhm_instrument())
        padding = 10.0 * max(inst_broadening, 2.0)

        grid_min = wavenumbers_line.min() - padding
        grid_max = wavenumbers_line.max() + padding

        wavenumbers_cont = np.linspace(grid_min, grid_max, granularity, dtype=np.float64)

        # Intensity per upper number density.
        intensities_cont = np.zeros_like(wavenumbers_cont)

        for band in emission_sim.bands:
            intensities_cont += band.intensities_cont(wavenumbers_cont)

        # Intensity vs. time and wavelength.
        i_t_wn = n2[None, :] * intensities_cont[:, None]

        time_ns = t_eval * 1e9

        self.slice_plot.show()
        self.slice_plot.clear()
        self.slice_plot.setAxisItems({"top": WavelengthAxis(orientation="top")})
        self.slice_plot.setLabel("top", "Wavelength, λ [nm]")
        self.slice_plot.setLabel("bottom", "Wavenumber, ν [cm⁻¹]")
        self.slice_plot.setLabel("left", "Intensity, I [a.u.]")
        self.slice_plot.setLabel("right", "")

        if self._time_cursor is not None:
            with contextlib.suppress(Exception):
                self.plot.removeItem(self._time_cursor)

        self.plot.clear()
        self._clear_lif_extras()

        # NOTE: 26/02/10 - The grid used to create the intensities at each point is linearly spaced
        #       in wavenumber coordinates. Any pixel-based image created using this data will have
        #       one pixel per wavenumber spacing, and therefore cannot easily be transformed into
        #       wavelength coordinates for plotting without squashing/stretching individual pixels.
        #       To avoid this, just plot the spectrum in wavenumber coordinates and create a second
        #       axis for wavelength on top.
        self.plot.setAxisItems({"top": WavelengthAxis(orientation="top")})
        self.plot.setLabel("top", "Wavelength, λ [nm]")
        self.plot.setLabel("bottom", "Wavenumber, ν [cm⁻¹]")
        self.plot.setLabel("left", "Time, t [ns]")
        self.plot.setLabel("right", "")

        self._lif_img_item = pg.ImageItem()
        self._lif_img_item.setImage(i_t_wn, autoLevels=False)
        self._lif_img_item.setLevels([i_t_wn.min(), i_t_wn.max()])

        x0, x1 = float(wavenumbers_cont.min()), float(wavenumbers_cont.max())
        y0, y1 = float(time_ns.min()), float(time_ns.max())
        self._lif_img_item.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))

        self.plot.addItem(self._lif_img_item)

        self._time_cursor = pg.InfiniteLine(
            pos=float(time_ns[len(time_ns) // 2]),
            angle=0,
            movable=True,
            pen=pg.mkPen("w", width=2),
        )
        self.plot.addItem(self._time_cursor)

        # Plot an empty set of data that can be updated later.
        slice_curve = self.slice_plot.plot([], [], pen=pg.mkPen("w", width=1))

        def update_slice():
            y = self._time_cursor.value()  # pyright: ignore[reportOptionalMemberAccess]
            idx = np.argmin(np.abs(time_ns - y))
            slice_curve.setData(wavenumbers_cont, i_t_wn[:, idx])
            self.slice_plot.setTitle(f"Spectrum at t = {time_ns[idx]:.3f} ns")
            self.slice_plot.setYRange(0, i_t_wn.max())

        self._time_cursor.sigPositionChanged.connect(update_slice)
        update_slice()

        self._lif_cbar = self.plot.addColorBar(
            self._lif_img_item, colorMap="magma", label="Intensity"
        )

        self.plot.autoRange()

    def _clear_lif_extras(self):
        if self._lif_cbar is not None:
            with contextlib.suppress(Exception):
                self.plot.removeItem(self._lif_cbar)
            with contextlib.suppress(Exception):
                self._lif_cbar.setParentItem(None)
            with contextlib.suppress(Exception):
                self._lif_cbar.hide()
            self._lif_cbar = None

        if self._lif_img_item is not None:
            with contextlib.suppress(Exception):
                self.plot.removeItem(self._lif_img_item)
            self._lif_img_item = None


class AllSimulationsTab(QWidget):
    def __init__(self, parent_tab_widget=None):
        super().__init__()

        self.is_first_run = True

        self.parent_tab_widget = parent_tab_widget

        main_layout = QVBoxLayout(self)

        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)

        resolution_group = QGroupBox("Resolution")
        resolution_layout = QHBoxLayout(resolution_group)
        self.resolution_spinbox = QSpinBox()
        self.resolution_spinbox.setMaximum(10000000)
        self.resolution_spinbox.setValue(DEFAULT_RESOLUTION)
        resolution_layout.addWidget(self.resolution_spinbox)
        controls_layout.addWidget(resolution_group)

        plot_group = QGroupBox("Plot Type")
        plot_layout = QHBoxLayout(plot_group)
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(
            ["Line", "Line Info", "Continuous Separate", "Continuous All"]
        )
        plot_layout.addWidget(self.plot_type_combo)
        controls_layout.addWidget(plot_group)

        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        self.run_button: QPushButton = QPushButton("Run All Simulations")
        self.run_button.clicked.connect(self.run_all_simulations)
        actions_layout.addWidget(self.run_button)
        controls_layout.addWidget(actions_group)

        main_layout.addWidget(controls_widget)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.addLegend(offset=(0, 1))
        self.plot_widget.setAxisItems({"top": WavelengthAxis(orientation="top")})
        self.plot_widget.setLabel("top", "Wavelength, λ [nm]")
        self.plot_widget.setLabel("bottom", "Wavenumber, ν [cm⁻¹]")
        self.plot_widget.setLabel("left", "Intensity, I [a.u.]")
        self.plot_widget.setLabel("right", "")
        self.plot_widget.setXRange(40000, 50000)
        main_layout.addWidget(self.plot_widget, stretch=1)

    def run_all_simulations(self):
        if self.parent_tab_widget is None:
            return

        self.plot_widget.clear()

        custom_tabs: list[CustomTab] = []

        for idx in range(self.parent_tab_widget.count()):
            tab = self.parent_tab_widget.widget(idx)
            if isinstance(tab, CustomTab):
                custom_tabs.append(tab)

        if not custom_tabs:
            QMessageBox.information(
                self,
                "No Simulations",
                "No simulation tabs found to plot.",
                QMessageBox.StandardButton.Ok,
            )
            return

        map_functions = {
            "Line": plot.plot_line,
            "Line Info": plot.plot_line_info,
            "Continuous Separate": plot.plot_cont_sep,
            "Continuous All": plot.plot_cont_all,
        }
        plot_type = self.plot_type_combo.currentText()
        plot_function = map_functions.get(plot_type)

        def max_intensity_line():
            intensities_line = np.array([], dtype=np.float64)

            for tab in custom_tabs:
                _, ins = tab.sim.all_line_data()
                intensities_line = np.concatenate((intensities_line, ins))

            return intensities_line.max()

        def max_intensity_cont_sep(granularity):
            continuous_data: list[NDArray[np.float64]] = []
            max_intensity = 0.0

            for tab in custom_tabs:
                for band in tab.sim.bands:
                    intensities_cont = band.intensities_cont(
                        band.wavenumbers_cont(granularity),
                    )

                    continuous_data.append(intensities_cont)

                    max_intensity = max(max_intensity, intensities_cont.max())

            return max_intensity

        def max_intensity_cont_all():
            intensities_cont = np.array([], dtype=np.float64)

            for tab in custom_tabs:
                _, ins = tab.sim.all_cont_data(
                    self.resolution_spinbox.value(),
                )
                intensities_cont = np.concatenate((intensities_cont, ins))

            return intensities_cont.max()

        # TODO: 25/08/01 - Adding all bands together needs to create a new wavenumber axis
        #       common to all simulations and then add the contributions of all simulations to the
        #       plot.

        all_sim_colors = get_colors(len(custom_tabs))

        for idx, tab in enumerate(custom_tabs):
            if plot_function is not None:
                if plot_function.__name__ == "plot_cont_sep":
                    resolution = self.resolution_spinbox.value()
                    plot_function(
                        self.plot_widget,
                        tab.sim,
                        all_sim_colors,
                        resolution,
                        max_intensity_cont_sep(resolution),
                        idx,
                    )
                elif plot_function.__name__ == "plot_cont_all":
                    plot_function(
                        self.plot_widget,
                        tab.sim,
                        all_sim_colors,
                        self.resolution_spinbox.value(),
                        max_intensity_cont_all(),
                        idx,
                    )
                else:
                    plot_function(
                        self.plot_widget, tab.sim, all_sim_colors, max_intensity_line(), idx
                    )

        if self.is_first_run:
            self.plot_widget.autoRange()
            self.is_first_run = False


class GUI(QMainWindow):
    def __init__(self):
        super().__init__()

        pg.setConfigOption("background", (32, 33, 36))
        pg.setConfigOption("foreground", "w")

        self.setWindowTitle("GEDISABRES")
        self.resize(1600, 800)
        self.init_ui()
        self.showMaximized()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)

        self.new_tab_button = QPushButton("Add Simulation")
        self.new_tab_button.clicked.connect(self.add_tab)
        toolbar_layout.addWidget(self.new_tab_button)

        main_layout.addWidget(toolbar_widget)

        tab_panel = self.create_tab_panel()
        main_layout.addWidget(tab_panel)

    def create_tab_panel(self):
        self.molecule_tab_widget = QTabWidget(movable=True, tabsClosable=True)

        all_sims_tab = AllSimulationsTab(parent_tab_widget=self.molecule_tab_widget)
        self.molecule_tab_widget.addTab(all_sims_tab, "All Simulations")
        self.molecule_tab_widget.tabBar().setTabButton(0, QTabBar.ButtonPosition.RightSide, None)

        self.lif_tab = LIFTab(parent_tab_widget=self.molecule_tab_widget)
        self.molecule_tab_widget.addTab(self.lif_tab, "LIF")
        self.molecule_tab_widget.tabBar().setTabButton(1, QTabBar.ButtonPosition.RightSide, None)

        self.molecule_tab_widget.tabBar().setMovable(False)

        for preset in MOLECULAR_PRESETS[:1]:
            tab = CustomTab(parent_tab_widget=self.molecule_tab_widget)
            tab.molecule = preset["molecule"]
            tab.state_up = preset["state_up"]
            tab.state_lo = preset["state_lo"]
            tab.update_sim_objects()

            self.molecule_tab_widget.addTab(tab, preset["name"])
            tab.update_tab_name()
            tab.run_simulation()

        self.molecule_tab_widget.setCurrentIndex(2)
        self.molecule_tab_widget.tabCloseRequested.connect(self.close_tab)

        self.lif_tab.refresh_parent_tabs()

        return self.molecule_tab_widget

    def close_tab(self, index):
        self.molecule_tab_widget.removeTab(index)
        self.lif_tab.refresh_parent_tabs()

    def add_tab(self):
        dialog = PresetSelectionDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_preset:
            preset = dialog.selected_preset
            tab = CustomTab(parent_tab_widget=self.molecule_tab_widget)

            tab.molecule = preset["molecule"]
            tab.state_up = preset["state_up"]
            tab.state_lo = preset["state_lo"]
            tab.update_sim_objects()

            tab_count = self.molecule_tab_widget.count()
            self.molecule_tab_widget.addTab(tab, preset["name"])
            tab.update_tab_name()

            self.molecule_tab_widget.setCurrentIndex(tab_count)
            tab.run_simulation()

            self.lif_tab.refresh_parent_tabs()


class WavelengthAxis(pg.AxisItem):
    """A custom x-axis displaying wavelengths."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize class variables."""
        super().__init__(*args, **kwargs)

    def tickStrings(self, wavenumbers: list[float], *_) -> list[str]:  # noqa: N802
        """Return the wavelength strings that are placed next to ticks.

        Args:
            wavenumbers: List of wavenumber values.

        Returns:
            List of wavelength values placed next to ticks.
        """
        strings: list[str] = []

        for wavenumber in wavenumbers:
            if wavenumber != 0:
                wavelength = utils.wavenum_to_wavelen(wavenumber)
                strings.append(f"{wavelength:.1f}")
            else:
                strings.append("∞")

        return strings


def main() -> None:
    """Entry point."""
    app = QApplication(sys.argv)
    qdarktheme.setup_theme()

    app_icon = QIcon(str(data_path.get_data_path("img", "icon.ico")))
    app.setWindowIcon(app_icon)

    gui = GUI()
    gui.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
