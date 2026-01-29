# module lookup.py
"""Generate lookup tables for the AMS 2020 and NUBASE 2020 databases."""

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

import polars as pl

import data_path
import fwf_reader

# Column headers and data are from the Atomic Mass Evaluation 2020 mass_1.mas20 database file.
MASS_COLS: list[tuple[str, int, int]] = [
    # a1
    ("cc", 0, 1),
    # i3
    ("NZ", 1, 4),
    # i5
    ("N", 4, 9),
    # i5
    ("Z", 9, 14),
    # i5
    ("A", 14, 19),
    # 1x, then a3
    ("el", 20, 23),
    # a4
    ("o", 23, 27),
    # 1x, then f14.6
    ("mass_excess", 28, 42),
    # f12.6
    ("mass_excess_unc", 42, 54),
    # f13.5
    ("binding", 54, 67),
    # 1x, then f10.5
    ("binding_unc", 68, 78),
    # 1x, then a2
    ("B", 79, 81),
    # f13.5
    ("beta", 81, 94),
    # f11.5
    ("beta_unc", 94, 105),
    # 1x, then i3
    ("atomic_mass_first", 106, 109),
    # 1x, then 13.6
    ("atomic_mass_second", 110, 123),
    # f12.6
    ("atomic_mass_unc", 123, 135),
]

# Column headers and data are from the NUBASE 2020 nubase_4.mas20 database file.
SPIN_COLS = [
    # a3
    ("AAA", 0, 3),
    # 1x, then a4
    ("ZZZi", 4, 8),
    # 3x, then a5
    ("A El", 11, 16),
    # a1
    ("s", 16, 17),
    # 1x, then f13.6
    ("Mass #", 18, 31),
    # f11.6
    ("dMass #", 31, 42),
    # f12.6
    ("Exc #", 42, 54),
    # f11.6
    ("dE #", 54, 65),
    # a2
    ("Orig", 65, 67),
    # a1
    ("Isom.Unc", 67, 68),
    # a1
    ("Isom.Unv", 68, 69),
    # f9.4
    ("T #", 69, 78),
    # a2
    ("unit T", 78, 80),
    # 1x, then a7
    ("dT", 81, 88),
    # a14
    ("Jpi */#/T=", 88, 102),
    # a2
    ("Ensdf year", 102, 104),
    # 10x, then a4
    ("Discovery", 114, 118),
    # 1x, then a90
    ("BR", 119, 209),
]


def mass_lookup() -> dict[str, float]:
    """Create a lookup table for atomic masses using the AME 2020 database.

    Returns:
        A dictionary containing (element, mass) pairs.
    """
    lf = fwf_reader.create_lazyframe(data_path.get_data_path("data", "mass_1.mas20"), MASS_COLS)

    # Stuff for the mass LazyFrame.
    # Create an `A El` column to match what's found in NUBASE, it also makes grabbing isotopes easy.
    lf = lf.with_columns(
        [
            # Atomic number + element name.
            (pl.col("A") + pl.col("el")).alias("A El"),
            # Atomic weight in [amu].
            # The leading three digits of the masses in [amu] are listed in a different column than
            # the remaining digits. By removing the decimal in the second column and adding a
            # decimal in between the two columns, we're also converting from [µ-amu] to [amu].
            (
                pl.col("atomic_mass_first")
                + "."
                # Some masses are estimated, which are denoted with a `#` character, so we must
                # remove those to get a valid float value.
                + pl.col("atomic_mass_second").str.strip_chars("#").str.replace(r"\.", "")
            )
            .cast(pl.Float64)
            .alias("amu"),
        ]
    )

    return dict(lf.collect().select("A El", "amu").iter_rows())


def spin_lookup() -> dict[str, str]:
    """Create a lookup table for atomic spins using the NUBASE 2020 database.

    Returns:
        A dictionary containing (element, spin) pairs.
    """
    lf = fwf_reader.create_lazyframe(data_path.get_data_path("data", "nubase_4.mas20"), SPIN_COLS)
    # Stuff for the spin LazyFrame.
    # Drop all elements that are not in the ground state (i = 0), i.e., isomers (i = 1,2),
    # levels (i = 3,4), resonance (i = 5), and IAS (i = 8,9).
    lf = lf.filter(pl.col("ZZZi").str.slice(3, 1) == "0")

    # The `Jpi */#/T=` field may contain something like `1/2+*`, `3/2-#`, `(1/2+)`, `2(+)`, or
    # `0+ T=1`. There are some even weirder ones like `(1-,2-)`, but I'm not dealing with those for
    # now.
    jpi_clean = (
        pl.col("Jpi */#/T=")
        # Remove any trailing isospin labels `T=...`.
        .str.replace(r"\s*T=.*$", "")
        # Remove all parentheses, both around the spin and the parity.
        .str.replace_all(r"[()]", "")
        # Remove `*` and `#` markers denoting how the spin was measured.
        .str.replace_all(r"[*#]", "")
        .str.strip_chars()
    )

    lf = lf.with_columns(
        [
            jpi_clean.alias("Jpi_clean"),
            # Get the spin as everything up to the first `+` or `-`. This isn't perfect because of
            # cases like `(1-,2-)`, but it's good enough for now.
            jpi_clean.str.extract(r"^\s*([^+-]+)", 1).alias("spin"),
            jpi_clean.str.extract(r"([+-])", 1).alias("parity"),
        ]
    ).drop("Jpi_clean")

    return dict(lf.collect().select("A El", "spin").iter_rows())
