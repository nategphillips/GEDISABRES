# module main.py
"""A simulation of the Schumann-Runge bands of molecular oxygen written in Python."""

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

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyqtgraph as pg
import qdarktheme
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

import plot
import utils
from atom import Atom
from colors import get_colors
from enums import ConstantsType, InversionSymmetry, ReflectionSymmetry, SimType, TermSymbol
from molecule import Molecule
from sim import Sim
from state import State

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

DEFAULT_J_MAX: int = 40
DEFAULT_RESOLUTION: int = int(1e4)

DEFAULT_TEMPERATURE: float = 300.0  # [K]
DEFAULT_PRESSURE: float = 101325.0  # [Pa]
DEFAULT_BROADENING: float = 0.0  # [nm]

DEFAULT_BANDS: str = "0-0"
DEFAULT_PLOTTYPE: str = "Line"
DEFAULT_SIMTYPE: str = "Absorption"

DEFAULT_MOLECULE: Molecule = Molecule("O2", Atom("O"), Atom("O"))
DEFAULT_STATE_UP: State = State(
    DEFAULT_MOLECULE,
    "B",
    3,
    TermSymbol.SIGMA,
    InversionSymmetry.UNGERADE,
    ReflectionSymmetry.MINUS,
    ConstantsType.PERLEVEL,
)
DEFAULT_STATE_LO: State = State(
    DEFAULT_MOLECULE,
    "X",
    3,
    TermSymbol.SIGMA,
    InversionSymmetry.GERADE,
    ReflectionSymmetry.MINUS,
    ConstantsType.PERLEVEL,
)
DEFAULT_SIM: Sim = Sim(
    SimType.ABSORPTION,
    DEFAULT_MOLECULE,
    DEFAULT_STATE_UP,
    DEFAULT_STATE_LO,
    40,
    300,
    300,
    300,
    300,
    101325,
    [(0, 0)],
)

MOLECULAR_PRESETS = [
    {
        "name": "O2 B-X",
        "molecule": Molecule("O2", Atom("O"), Atom("O")),
        "state_up": State(
            Molecule("O2", Atom("O"), Atom("O")),
            "B",
            3,
            TermSymbol.SIGMA,
            InversionSymmetry.UNGERADE,
            ReflectionSymmetry.MINUS,
            ConstantsType.PERLEVEL,
        ),
        "state_lo": State(
            Molecule("O2", Atom("O"), Atom("O")),
            "X",
            3,
            TermSymbol.SIGMA,
            InversionSymmetry.GERADE,
            ReflectionSymmetry.MINUS,
            ConstantsType.PERLEVEL,
        ),
    },
    {
        "name": "NO A-X",
        "molecule": Molecule("NO", Atom("N"), Atom("O")),
        "state_up": State(
            Molecule("NO", Atom("N"), Atom("O")),
            "A",
            2,
            TermSymbol.SIGMA,
            InversionSymmetry.NONE,
            ReflectionSymmetry.PLUS,
            ConstantsType.PERLEVEL,
        ),
        "state_lo": State(
            Molecule("NO", Atom("N"), Atom("O")),
            "X",
            2,
            TermSymbol.PI,
            InversionSymmetry.NONE,
            ReflectionSymmetry.NONE,
            ConstantsType.PERLEVEL,
        ),
    },
    {
        "name": "NO B-X",
        "molecule": Molecule("NO", Atom("N"), Atom("O")),
        "state_up": State(
            Molecule("NO", Atom("N"), Atom("O")),
            "B",
            2,
            TermSymbol.PI,
            InversionSymmetry.NONE,
            ReflectionSymmetry.NONE,
            ConstantsType.PERLEVEL,
        ),
        "state_lo": State(
            Molecule("NO", Atom("N"), Atom("O")),
            "X",
            2,
            TermSymbol.PI,
            InversionSymmetry.NONE,
            ReflectionSymmetry.NONE,
            ConstantsType.PERLEVEL,
        ),
    },
    {
        "name": "OH A-X",
        "molecule": Molecule("OH", Atom("O"), Atom("H")),
        "state_up": State(
            Molecule("OH", Atom("O"), Atom("H")),
            "A",
            2,
            TermSymbol.SIGMA,
            InversionSymmetry.NONE,
            ReflectionSymmetry.PLUS,
            ConstantsType.DUNHAM,
        ),
        "state_lo": State(
            Molecule("OH", Atom("O"), Atom("H")),
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
            text (str): Input text.

        Returns:
            float: Converted value.
        """
        try:
            return float(text)
        except ValueError:
            return 0.0

    def textFromValue(self, value: float) -> str:  # noqa: N802
        """Displays the input parameter using f-strings.

        Args:
            value (float): Input float.

        Returns:
            str: Displayed text.
        """
        return f"{value:g}"

    def validate(self, text: str, pos: int) -> tuple[QValidator.State, str, int]:
        """Checks user input and returns the correct state.

        Args:
            text (str): Input text.
            pos (int): Position in the string.

        Returns:
            tuple[QValidator.State, str, int]: The current state, input text, and string position.
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
                parts: list[str] = text.lower().split("e")
                num_parts: int = 2
                if len(parts) == num_parts and (parts[1] == "" or parts[1] in ["-", "+"]):
                    return (QValidator.State.Intermediate, text, pos)
            return (QValidator.State.Invalid, text, pos)


class MyTable(QAbstractTableModel):
    """A simple model to interface a Qt view with a Polars DataFrame."""

    def __init__(self, df: pl.DataFrame) -> None:
        """Initialize class variables.

        Args:
            df (pl.DataFrame): A Polars `DataFrame`.
        """
        super().__init__()
        self.df: pl.DataFrame = df

    def rowCount(self, _: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        """Get the height of the table.

        Args:
            _ (QModelIndex, optional): Parent class used to locate data. Defaults to QModelIndex().

        Returns:
            int: Table height.
        """
        return self.df.height

    def columnCount(self, _: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        """Get the width of the table.

        Args:
            _ (QModelIndex, optional): Parent class used to locate data. Defaults to QModelIndex().

        Returns:
            int: Table width.
        """
        return self.df.width

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> str | None:
        """Validates and formats data displayed in the table.

        Args:
            index (QModelIndex): Parent class used to locate data.
            role (int, optional): Data rendered as text. Defaults to Qt.ItemDataRole.DisplayRole.

        Returns:
            str | None: The formatted text.
        """
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            value: int | float | str = self.df[index.row(), index.column()]
            column_name: str = self.df.columns[index.column()]

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
            section (int): For horizontal headers, the section number corresponds to the column
                number. For vertical headers, the section number corresponds to the row number.
            orientation (Qt.Orientation): The header orientation.
            role (int, optional): Data rendered as text. Defaults to Qt.ItemDataRole.DisplayRole.

        Returns:
            str | None: Header data.
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
        df (pl.DataFrame): Data to be displayed in the tab.
        _ (str): The tab label.

    Returns:
        QWidget: A QWidget containing a QTableView to display the DataFrame.
    """
    widget: QWidget = QWidget()
    layout: QVBoxLayout = QVBoxLayout(widget)
    table_view: QTableView = QTableView()
    model: MyTable = MyTable(df)
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

        self.name = QLineEdit()
        self.atom_1 = QLineEdit()
        self.atom_2 = QLineEdit()

        molecule_form = QFormLayout()
        molecule_form.addRow("Molecule Name:", self.name)
        molecule_form.addRow("Atom 1:", self.atom_1)
        molecule_form.addRow("Atom 2:", self.atom_2)

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
        self.name.setText(tab.molecule.name)
        self.atom_1.setText(tab.molecule.atom_1.name)
        self.atom_2.setText(tab.molecule.atom_2.name)
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
        self.temp_trn.setText(str(tab.sim.temp_trn))
        self.temp_elc.setText(str(tab.sim.temp_elc))
        self.temp_vib.setText(str(tab.sim.temp_vib))
        self.temp_rot.setText(str(tab.sim.temp_rot))
        self.pressure.setText(str(tab.sim.pressure))

    def accept(self):
        # TODO: 25/07/31 - Just create completely new objects each time. This is certainly not the
        #       most efficient thing to do, but it ensures that everything is updated properly.
        self.tab.molecule = Molecule(
            self.name.text(), Atom(self.atom_1.text()), Atom(self.atom_2.text())
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

        self.tab.sim = Sim(
            SimType[self.sim_type.currentText()],
            self.tab.molecule,
            self.tab.state_up,
            self.tab.state_lo,
            self.tab.maxj_spinbox.value(),
            float(self.temp_trn.text()),
            float(self.temp_elc.text()),
            float(self.temp_vib.text()),
            float(self.temp_rot.text()),
            float(self.pressure.text()),
            self.tab.get_current_bands(),
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

        group_bands: QGroupBox = QGroupBox("Bands")
        group_bands.setFixedWidth(300)
        bands_layout: QVBoxLayout = QVBoxLayout(group_bands)

        band_selection_layout: QHBoxLayout = QHBoxLayout()
        self.radio_specific_bands: QRadioButton = QRadioButton("Specific Bands")
        self.radio_specific_bands.setChecked(True)
        self.radio_band_ranges: QRadioButton = QRadioButton("Band Ranges")
        band_selection_layout.addWidget(self.radio_specific_bands)
        band_selection_layout.addWidget(self.radio_band_ranges)
        bands_layout.addLayout(band_selection_layout)

        self.radio_specific_bands.toggled.connect(self.toggle_band_input_method)
        self.radio_specific_bands.toggled.connect(self.update_sim_objects)
        self.radio_band_ranges.toggled.connect(self.update_sim_objects)

        self.specific_bands_container: QWidget = QWidget()
        self.specific_bands_container.setFixedHeight(60)
        specific_bands_layout: QVBoxLayout = QVBoxLayout(self.specific_bands_container)
        specific_bands_layout.setContentsMargins(0, 0, 0, 0)

        band_layout: QHBoxLayout = QHBoxLayout()
        band_label: QLabel = QLabel("Bands:")
        self.band_line_edit: QLineEdit = QLineEdit(DEFAULT_BANDS)
        self.band_line_edit.textChanged.connect(self.update_sim_objects)
        band_layout.addWidget(band_label)
        band_layout.addWidget(self.band_line_edit)
        specific_bands_layout.addLayout(band_layout)

        self.band_ranges_container: QWidget = QWidget()
        self.band_ranges_container.setFixedHeight(60)
        band_ranges_layout: QGridLayout = QGridLayout(self.band_ranges_container)
        band_ranges_layout.setContentsMargins(0, 0, 0, 0)

        v_up_min_label: QLabel = QLabel("v' min:")
        self.v_up_min_spinbox: QSpinBox = QSpinBox()
        self.v_up_min_spinbox.setRange(0, 99)
        self.v_up_min_spinbox.setValue(0)
        self.v_up_min_spinbox.valueChanged.connect(self.update_sim_objects)

        v_up_max_label: QLabel = QLabel("v' max:")
        self.v_up_max_spinbox: QSpinBox = QSpinBox()
        self.v_up_max_spinbox.setRange(0, 99)
        self.v_up_max_spinbox.setValue(5)
        self.v_up_max_spinbox.valueChanged.connect(self.update_sim_objects)

        v_lo_min_label: QLabel = QLabel("v'' min:")
        self.v_lo_min_spinbox: QSpinBox = QSpinBox()
        self.v_lo_min_spinbox.setRange(0, 99)
        self.v_lo_min_spinbox.setValue(5)
        self.v_lo_min_spinbox.valueChanged.connect(self.update_sim_objects)

        v_lo_max_label: QLabel = QLabel("v'' max:")
        self.v_lo_max_spinbox: QSpinBox = QSpinBox()
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

        group_maxj: QGroupBox = QGroupBox("Max J'")
        maxj_layout: QHBoxLayout = QHBoxLayout(group_maxj)
        self.maxj_spinbox: QSpinBox = QSpinBox()
        self.maxj_spinbox.setMaximum(10000)
        self.maxj_spinbox.setValue(DEFAULT_J_MAX)
        self.maxj_spinbox.valueChanged.connect(self.update_sim_objects)
        maxj_layout.addWidget(self.maxj_spinbox)
        controls_layout.addWidget(group_maxj)

        group_resolution: QGroupBox = QGroupBox("Resolution")
        resolution_layout: QHBoxLayout = QHBoxLayout(group_resolution)
        self.resolution_spinbox: QSpinBox = QSpinBox()
        self.resolution_spinbox.setMaximum(10000000)
        self.resolution_spinbox.setValue(DEFAULT_RESOLUTION)
        resolution_layout.addWidget(self.resolution_spinbox)
        controls_layout.addWidget(group_resolution)

        group_plot_type: QGroupBox = QGroupBox("Plot Type")
        plot_type_layout: QHBoxLayout = QHBoxLayout(group_plot_type)
        self.plot_type_combo: QComboBox = QComboBox()
        self.plot_type_combo.addItems(["Line", "Line Info", "Convolve Separate", "Convolve All"])
        plot_type_layout.addWidget(self.plot_type_combo)
        controls_layout.addWidget(group_plot_type)

        group_broadening: QGroupBox = QGroupBox("Instrument Broadening [nm]")
        broadening_layout: QVBoxLayout = QVBoxLayout(group_broadening)

        self.inst_broadening_spinbox: MyDoubleSpinBox = MyDoubleSpinBox()
        self.inst_broadening_spinbox.setValue(DEFAULT_BROADENING)
        self.inst_broadening_spinbox.valueChanged.connect(self.update_sim_objects)
        broadening_layout.addWidget(self.inst_broadening_spinbox)

        checkbox_layout: QHBoxLayout = QHBoxLayout()
        self.checkbox_instrument: QCheckBox = QCheckBox("Instrument FWHM")
        self.checkbox_doppler: QCheckBox = QCheckBox("Doppler")
        self.checkbox_natural: QCheckBox = QCheckBox("Natural")
        self.checkbox_collisional: QCheckBox = QCheckBox("Collisional")
        self.checkbox_predissociation: QCheckBox = QCheckBox("Predissociation")

        checkboxes = [
            self.checkbox_instrument,
            self.checkbox_doppler,
            self.checkbox_natural,
            self.checkbox_collisional,
            self.checkbox_predissociation,
        ]

        for i, cb in enumerate(checkboxes):
            cb.setChecked(True)
            cb.toggled.connect(self.update_sim_objects)
            checkbox_layout.addWidget(cb)

        broadening_layout.addLayout(checkbox_layout)
        controls_layout.addWidget(group_broadening)

        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        self.run_button: QPushButton = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation)
        actions_layout.addWidget(self.run_button)

        self.open_sample_button: QPushButton = QPushButton("Open Sample")
        self.open_sample_button.clicked.connect(self.open_sample)
        actions_layout.addWidget(self.open_sample_button)

        self.export_button: QPushButton = QPushButton("Export Table")
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
        self.plot_widget.setAxisItems({"top": WavenumberAxis(orientation="top")})
        self.plot_widget.setLabel("top", "Wavenumber, ν [cm⁻¹]")
        self.plot_widget.setLabel("bottom", "Wavelength, λ [nm]")
        self.plot_widget.setLabel("left", "Intensity, I [a.u.]")
        self.plot_widget.setLabel("right", "Intensity, I [a.u.]")
        self.plot_widget.setXRange(100, 200)
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
        band_ranges_str: str = self.band_line_edit.text()
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
        v_up_min: int = self.v_up_min_spinbox.value()
        v_up_max: int = self.v_up_max_spinbox.value()
        v_lo_min: int = self.v_lo_min_spinbox.value()
        v_lo_max: int = self.v_lo_max_spinbox.value()

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

        # Only update the values that are controlled by the tab specifically.
        self.sim = Sim(
            self.sim.sim_type,
            self.molecule,
            self.state_up,
            self.state_lo,
            self.maxj_spinbox.value(),
            self.sim.temp_trn,
            self.sim.temp_elc,
            self.sim.temp_vib,
            self.sim.temp_rot,
            self.sim.pressure,
            current_bands,
            inst_broadening_wl=self.inst_broadening_spinbox.value(),
        )

    def run_simulation(self) -> None:
        """Run a simulation instance for this specific tab."""
        bands = self.get_current_bands()
        colors: list[str] = get_colors(bands)

        self.plot_widget.clear()

        map_functions: dict[str, Callable] = {
            "Line": plot.plot_line,
            "Line Info": plot.plot_line_info,
            "Convolve Separate": plot.plot_conv_sep,
            "Convolve All": plot.plot_conv_all,
        }
        plot_type: str = self.plot_type_combo.currentText()
        plot_function: Callable | None = map_functions.get(plot_type)

        fwhm_selections: dict[str, bool] = {
            "instrument": self.checkbox_instrument.isChecked(),
            "doppler": self.checkbox_doppler.isChecked(),
            "natural": self.checkbox_natural.isChecked(),
            "collisional": self.checkbox_collisional.isChecked(),
            "predissociation": self.checkbox_predissociation.isChecked(),
        }

        if plot_function is not None:
            if plot_function.__name__ in ("plot_conv_sep", "plot_conv_all"):
                plot_function(
                    self.plot_widget,
                    self.sim,
                    colors,
                    fwhm_selections,
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
            df: pl.DataFrame = pl.DataFrame(
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

            tab_name: str = f"{band[0]}-{band[1]}"
            new_tab: QWidget = create_dataframe_tab(df, tab_name)
            self.table_tab_widget.addTab(new_tab, tab_name)

    def export_current_table(self) -> None:
        """Export the currently displayed table to a CSV file."""
        current_widget: QWidget = self.table_tab_widget.currentWidget()

        table_view = current_widget.findChild(QTableView)

        if table_view is None:
            QMessageBox.information(
                self,
                "Error",
                "No table view found in this tab.",
                QMessageBox.StandardButton.Ok,
            )
            return

        model: MyTable = table_view.model()
        if not hasattr(model, "df"):
            QMessageBox.information(
                self,
                "Error",
                "This table does not support CSV export.",
                QMessageBox.StandardButton.Ok,
            )
            return

        df: pl.DataFrame = model.df

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
            dir=str(utils.get_data_path("data", "samples")),
            filter="CSV Files (*.csv);;All Files (*)",
        )
        if filename:
            try:
                df: pl.DataFrame = pl.read_csv(filename)
            except ValueError:
                QMessageBox.critical(self, "Error", "Data is improperly formatted.")
                return
        else:
            return

        display_name: str = Path(filename).name

        new_table_tab: QWidget = create_dataframe_tab(df, display_name)
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
        except pl.ColumnNotFoundError:
            QMessageBox.critical(self, "Error", "No 'intensity' column found.")
            return

        plot.plot_sample(self.plot_widget, x_values, intensities, display_name, value_type)

    def open_parameters_dialog(self):
        dialog = ParametersDialog(self, context_name=self.molecule.name)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.update_sim_objects()
            self.update_tab_name()


class AllSimulationsTab(QWidget):
    def __init__(self, parent_tab_widget=None):
        super().__init__()

        self.is_first_run: bool = True

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
        self.plot_type_combo.addItems(["Line", "Line Info", "Convolve Separate", "Convolve All"])
        plot_layout.addWidget(self.plot_type_combo)
        controls_layout.addWidget(plot_group)

        broadening_group = QGroupBox("Instrument Broadening [nm]")
        broadening_layout = QVBoxLayout(broadening_group)

        self.inst_broadening_spinbox = MyDoubleSpinBox()
        self.inst_broadening_spinbox.setValue(DEFAULT_BROADENING)
        broadening_layout.addWidget(self.inst_broadening_spinbox)

        checkbox_layout = QHBoxLayout()
        self.checkbox_instrument = QCheckBox("Instrument FWHM")
        self.checkbox_doppler = QCheckBox("Doppler")
        self.checkbox_natural = QCheckBox("Natural")
        self.checkbox_collisional = QCheckBox("Collisional")
        self.checkbox_predissociation = QCheckBox("Predissociation")

        checkboxes = [
            self.checkbox_instrument,
            self.checkbox_doppler,
            self.checkbox_natural,
            self.checkbox_collisional,
            self.checkbox_predissociation,
        ]

        for cb in checkboxes:
            cb.setChecked(True)
            checkbox_layout.addWidget(cb)

        broadening_layout.addLayout(checkbox_layout)
        controls_layout.addWidget(broadening_group)

        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        self.run_button: QPushButton = QPushButton("Run All Simulations")
        self.run_button.clicked.connect(self.run_all_simulations)
        actions_layout.addWidget(self.run_button)
        controls_layout.addWidget(actions_group)

        main_layout.addWidget(controls_widget)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.addLegend(offset=(0, 1))
        self.plot_widget.setAxisItems({"top": WavenumberAxis(orientation="top")})
        self.plot_widget.setLabel("top", "Wavenumber, ν [cm⁻¹]")
        self.plot_widget.setLabel("bottom", "Wavelength, λ [nm]")
        self.plot_widget.setLabel("left", "Intensity, I [a.u.]")
        self.plot_widget.setLabel("right", "Intensity, I [a.u.]")
        self.plot_widget.setXRange(100, 200)
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
            "Convolve Separate": plot.plot_conv_sep,
            "Convolve All": plot.plot_conv_all,
        }
        plot_type = self.plot_type_combo.currentText()
        plot_function = map_functions.get(plot_type)

        # TODO: 25/08/01 - Broadening parameters should be pulled from each individual tab instead.

        fwhm_selections = {
            "instrument": self.checkbox_instrument.isChecked(),
            "doppler": self.checkbox_doppler.isChecked(),
            "natural": self.checkbox_natural.isChecked(),
            "collisional": self.checkbox_collisional.isChecked(),
            "predissociation": self.checkbox_predissociation.isChecked(),
        }

        def max_intensity_line():
            intensities_line: NDArray[np.float64] = np.array([])

            for tab in custom_tabs:
                _, ins = tab.sim.all_line_data()
                intensities_line = np.concatenate((intensities_line, ins))

            return intensities_line.max()

        def max_intensity_conv_sep(granularity):
            convolved_data: list[NDArray[np.float64]] = []
            max_intensity: float = 0.0

            for tab in custom_tabs:
                for band in tab.sim.bands:
                    intensities_conv = band.intensities_conv(
                        fwhm_selections,
                        band.wavenumbers_conv(granularity),
                    )

                    convolved_data.append(intensities_conv)

                    max_intensity = max(max_intensity, intensities_conv.max())

            return max_intensity

        def max_intensity_conv_all():
            intensities_conv: NDArray[np.float64] = np.array([])

            for tab in custom_tabs:
                _, ins = tab.sim.all_conv_data(
                    fwhm_selections,
                    self.resolution_spinbox.value(),
                )
                intensities_conv = np.concatenate((intensities_conv, ins))

            return intensities_conv.max()

        # TODO: 25/08/01 - Convolving all bands together needs to create a new wavenumber axis
        #       common to all simulations and then add the contributions of all simulations to the
        #       plot.

        num_tabs = len(custom_tabs)
        all_sim_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:num_tabs]

        for idx, tab in enumerate(custom_tabs):
            if plot_function is not None:
                if plot_function.__name__ == "plot_conv_sep":
                    resolution = self.resolution_spinbox.value()
                    plot_function(
                        self.plot_widget,
                        tab.sim,
                        all_sim_colors,
                        fwhm_selections,
                        resolution,
                        max_intensity_conv_sep(resolution),
                        idx,
                    )
                elif plot_function.__name__ == "plot_conv_all":
                    plot_function(
                        self.plot_widget,
                        tab.sim,
                        all_sim_colors,
                        fwhm_selections,
                        self.resolution_spinbox.value(),
                        max_intensity_conv_all(),
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

        self.setWindowTitle("pyGEONOSIS")
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

        tab_panel: QWidget = self.create_tab_panel()
        main_layout.addWidget(tab_panel)

    def create_tab_panel(self):
        self.molecule_tab_widget = QTabWidget(movable=True, tabsClosable=True)

        all_sims_tab = AllSimulationsTab(parent_tab_widget=self.molecule_tab_widget)
        self.molecule_tab_widget.addTab(all_sims_tab, "All Simulations")
        self.molecule_tab_widget.tabBar().setTabButton(0, QTabBar.ButtonPosition.RightSide, None)
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

        self.molecule_tab_widget.setCurrentIndex(1)
        self.molecule_tab_widget.tabCloseRequested.connect(self.close_tab)

        return self.molecule_tab_widget

    def close_tab(self, index):
        if self.molecule_tab_widget.count() > 2:
            self.molecule_tab_widget.removeTab(index)
        else:
            QMessageBox.information(
                self,
                "Cannot Close Tab",
                "At least one simulation tab must remain open.",
                QMessageBox.StandardButton.Ok,
            )

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


class WavenumberAxis(pg.AxisItem):
    """A custom x-axis displaying wavenumbers."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize class variables."""
        super().__init__(*args, **kwargs)

    def tickStrings(self, wavelengths: list[float], *_) -> list[str]:  # noqa: N802
        """Return the wavenumber strings that are placed next to ticks.

        Args:
            wavelengths (list[float]): List of wavelength values.

        Returns:
            list[str]: List of wavenumber values placed next to ticks.
        """
        strings: list[str] = []

        for wavelength in wavelengths:
            if wavelength != 0:
                wavenumber: float = utils.wavenum_to_wavelen(wavelength)
                strings.append(f"{wavenumber:.1f}")
            else:
                strings.append("∞")

        return strings


def main() -> None:
    """Entry point."""
    app: QApplication = QApplication(sys.argv)
    qdarktheme.setup_theme()

    app_icon: QIcon = QIcon(str(utils.get_data_path("img", "icon.ico")))
    app.setWindowIcon(app_icon)

    gui: GUI = GUI()
    gui.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
