# module data_path.py
"""Ensure the correct data path is returned for during development and for PyInstaller."""

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

import sys
from pathlib import Path


def get_data_path(*relative_path_parts) -> Path:
    """Get the correct data path, accounting for PyInstaller executable.

    Returns:
        A relative path if developing, the absolute path to the bundle folder if using PyInstaller.
    """
    if getattr(sys, "frozen", False):
        base_path = Path(sys._MEIPASS)  # pyright: ignore[reportAttributeAccessIssue]
    else:
        base_path = Path(__file__).resolve().parent.parent

    return base_path.joinpath(*relative_path_parts)
