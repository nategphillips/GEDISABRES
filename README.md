# pyGEONOSIS

Python GEnerated Oxygen and Nitric Oxide SImulated Spectra (pyGEONOSIS) is a tool for simulating the Schumann–Runge (B-X) bands of molecular oxygen, and eventually the $\gamma$ (A-X) bands of nitric oxide. Built using NumPy, Polars, PySide6, PyQtGraph, and SciPy, pyGEONOSIS is designed to be easily understood and modified.

The capabilities of this tool are briefly summarized below. More detailed theory and notation are explained in the included document.

## Background

### Rotational Hamiltonian

The rotational Hamiltonian used for both the $B^3\Sigma_u^-$ and $X^3\Sigma_g^-$ states is

$$
H = H_{r} + H_{ss} + H_{sr},
$$

where

$$
\begin{aligned}
    H_{r}  &= B\mathbf{N}^{2} - D\mathbf{N}^{4} \\
    H_{ss} &= \frac{2}{3}\lambda(3S_{z}^{2} - \mathbf{S}^{2}) \\
    H_{sr} &= \gamma\mathbf{N}\cdot\mathbf{S}.
\end{aligned}
$$

Upper state constants are taken from [*Molecular Spectroscopic Constants of O2: The Upper State of the Schumann–Runge Bands*](https://doi.org/10.1016/0022-2852(86)90196-7) by Cheung et al. (1986). Ground state constants are taken from [*High Resolution Spectral Analysis of Oxygen: IV*](https://doi.org/10.1063/1.4900510) by Yu et al. (2014).

### Spectral Broadening

Convolutions include the effects of both Gaussian and Lorentzian broadening mechanisms. Each type of broadening can be toggled on or off individually.

#### Gaussian

- Thermal Doppler broadening
- Instrument broadening

#### Lorentzian

- Pressure broadening
- Natural broadening
- Predissociation broadening

### Rotational Lines

By default, the maximum upper state rotational quantum number is set to $J'_\text{max} = 40$. Currently, predissociation factors are computed for each rotational line using a polynomial fit that is valid up to $v = 21$ and $J = 40$. Therefore, it is recommended not to exceed $J = 40$ by a large margin if accuracy is to be preserved.

### Equilibrium & Non-equilibrium

In general, the electronic, vibrational, and rotational Boltzmann partition functions are computed assuming Boltzmann population distributions. For equilibrium simulations, the input temperature for all states (translational, electronic, vibrational, and rotational) is the same. Non-equilibrium simulations have different temperatures specified for each state, but the population distributions within each state remain Boltzmann.

### Plot Types

Four plot types are currently implemented:

- Line
  - Each rotational line is plotted at its exact wavenumber.
- Line Info
  - Information is printed above each line for easier identification.
- Convolve Separate
  - The rotational lines within a single vibrational band are convolved.
- Convolve All
  - All vibrational bands are convolved together.

## Validation with Experimental Data

Data for molecular oxygen from the [Harvard-Smithsonian Center for Astrophysics](https://www.cfa.harvard.edu/) is used to verify the output of pyGEONOSIS. Data for the Schumann–Runge bands is available [here](https://lweb.cfa.harvard.edu/amp/ampdata/o2pub92/S-R.html) at 300 K. The (2, 0) and (6, 0) experimental band data were chosen since they contain lines belonging to multiple vibrational bands. The specific bands chosen to emulate the experimental data are shown in the at the top left of the GUI in each image.

### (2, 0) Band Validation

![(2, 0) Band Validation](img/example_20.webp)

### (6, 0) Band Validation

![(6, 0) Band Validation](img/example_60.webp)

## Installation

### From an Executable

Go to the [releases tab](https://github.com/nategphillips/pyGEONOSIS/releases) of this repository and download the latest compressed archive specific to your operating system. Extract the contents and run the binary file contained inside.

### From Source

On your PC, navigate to a directory of your choice and run

```bash
git clone https://github.com/nategphillips/pyGEONOSIS
```

to download the source code directly. This repository uses the [uv](https://github.com/astral-sh/uv) package manager for Python. Install uv using the instructions on their [website](https://docs.astral.sh/uv/). Once uv is installed, the packages required for pyGEONOSIS can be added in a virtual environment by navigating to the repository's root directory and installing the dependencies:

```bash
uv sync
```

Then, navigate to the `src/` directory and run the GUI:

```bash
uv run ./main.py
```

## Roadmap

### GUI Functionality

- [x] Switch to PyQtGraph instead of Matplotlib for improved plot performance
- [x] Add the ability to export rotational line data from the built-in spreadsheet
- [x] Make a custom GUI icon
- [ ] Build a more intuitive interface for adding/removing simulated and experimental data, especially once multiple molecules are added
- [ ] Design and implement a GUI for LIF computations, including estimated fluorescence yield and the ability to search for rotational line overlaps

### Physics

- [ ] Add support for more diatomic molecules, starting with $\text{NO}$
- [ ] Implement electronic spectra for atomic species
- [ ] Include the ability to view and edit the rotational Hamiltonian on a per-term basis
- [ ] Include non-Boltzmann temperature distributions

### Packaging

- [x] Package a pre-compiled binary to improve user experience
- [x] Implement a splash screen to show that the program is loading
- [x] Create an icon executable file (Windows only for now)

## License and Copyright

Copyright (C) 2023-2025 Nathan G. Phillips

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
