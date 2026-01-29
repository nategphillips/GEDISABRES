# module fwf_reader.py
"""Functions for reading a fixed-width file into a Polars LazyFrame."""

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

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path


def is_data_line(line: str) -> bool:
    """Checks if the line contains valid data.

    Args:
        line: A fixed-width input line.

    Returns:
        True if the line is valid, False otherwise.
    """
    return not line.startswith("#")


def slice_fwf(line: str, column_names: list[tuple[str, int, int]]) -> dict:
    """Convert a fixed-width line into a dictionary.

    Args:
        line: A fixed-width input line.
        column_names: Column headers from the file.

    Returns:
        A dictionary containing the line data.
    """
    return {name: line[a:b] for name, a, b in column_names}


def create_lazyframe(file: Path, column_names: list[tuple[str, int, int]]) -> pl.LazyFrame:
    """Create a Polars LazyFrame from dictionary data.

    Args:
        file: The fixed-width data file.
        column_names: Column headers from the file.

    Returns:
        A LazyFrame containing the file data.
    """
    # Strip trailing whitespace from the end of each valid line.
    with open(file, encoding="utf-8", errors="strict") as f:
        rows = [slice_fwf(line.rstrip("\n"), column_names) for line in f if is_data_line(line)]

    lf = pl.LazyFrame(rows)

    # Remove leading and trailing whitespace from each entry.
    return lf.with_columns(pl.all().cast(pl.Utf8).str.strip_chars())
