# module convolve

from __future__ import annotations
from typing import TYPE_CHECKING

from scipy.special import wofz # pylint: disable=no-name-in-module
import numpy as np

import constants as cn

if TYPE_CHECKING:
    from simulation import Simulation

def convolve_inst(wavenumbers_conv: np.ndarray, intensities_conv: np.ndarray,
                  broadening: float) -> np.ndarray:
    intensities_inst = np.zeros_like(wavenumbers_conv)

    for wave, intn in zip(wavenumbers_conv, intensities_conv):
        intensities_inst += intn * instrument_fn(wavenumbers_conv, wave, broadening)

    return intensities_inst

def convolve_brod(sim: Simulation, lines: np.ndarray, wavenumbers_line: np.ndarray,
                  intensities_line: np.ndarray, wavenumbers_conv: np.ndarray) -> np.ndarray:
    intensities_conv = np.zeros_like(wavenumbers_conv)
    natural, collide = broadening_params(sim)

    for idx, (wave, intn) in enumerate(zip(wavenumbers_line, intensities_line)):
        intensities_conv += intn * broadening_fn(sim, lines, wavenumbers_conv, wave, idx, natural,
                                                 collide)

    return intensities_conv

def instrument_fn(convolved_wavenumbers: np.ndarray, wavenumber_peak: float,
                  broadening: float) -> np.ndarray:
    return np.exp(- 0.5 * (convolved_wavenumbers - wavenumber_peak)**2 / broadening**2) / \
           (broadening * np.sqrt(2 * np.pi))

def broadening_fn(sim: Simulation, lines: np.ndarray, convolved_wavenumbers: np.ndarray,
                  wavenumber_peak: float, line_idx: int, natural: float,
                  collide: float) -> np.ndarray:

    doppler = wavenumber_peak * \
              np.sqrt(cn.BOLTZ * sim.temp / (sim.molecule.molecular_mass * (cn.LIGHT / 1e2)**2))

    prediss = lines[line_idx].predissociation()

    gauss = doppler
    loren = natural + collide + prediss

    fadd = ((convolved_wavenumbers - wavenumber_peak) + 1j * loren) / (gauss * np.sqrt(2))

    return np.real(wofz(fadd)) / (gauss * np.sqrt(2 * np.pi))

def broadening_params(sim: Simulation) -> tuple[float, float]:
    natural = sim.state_lo.cross_section**2 * \
              np.sqrt(8 / (np.pi * sim.molecule.reduced_mass * cn.BOLTZ * sim.temp)) / 4

    collide = (sim.pres * 10) * sim.state_lo.cross_section**2 * \
              np.sqrt(8 / (np.pi * sim.molecule.reduced_mass * cn.BOLTZ * sim.temp)) / 2

    return natural, collide
