# DiatomicSpectra.jl
"""A package for simulating the rovibronic spectra of diatomic molecules."""

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

"""
    DiatomicSpectra 

A package for simulating the rovibronic spectra of diatomic molecules.
"""
module DiatomicSpectra

# TODO:
#   × Look into using Runic.jl for formatting
#   × Add package tests and benchmarks
#   × Ensure type stability with JET.jl / Cthulhu.jl / DispatchDoctor.jl
#   × Add automated checks using Aqua.jl
#   × Write docstrings
#   × Dimensionalize variables using Unitful.jl
#   × Use values from PhysicalConstants.jl where possible

greet() = print("Hello World!")

end # module DiatomicSpectra
