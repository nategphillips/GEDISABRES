# module convolution
'''
Convolves the quantized spectral data by applying thermal doppler broadening, pressure broadening,
natural broadening, and predissociation broadening.
'''

from scipy.special import wofz # pylint: disable=no-name-in-module
import numpy as np

import constants as cn
import input as inp

def convolve(wavenumbers, intensities, temp, pres, lines):
    new_wavenumbers = np.linspace(wavenumbers.min(), wavenumbers.max(), inp.CONV_GRAN)
    new_intensities = np.zeros_like(new_wavenumbers)

    for idx, (wave, intn) in enumerate(zip(wavenumbers, intensities)):
        new_intensities += intn * broadening_fn(new_wavenumbers, wave, temp, pres, idx, lines)

    new_intensities /= new_intensities.max()

    return new_wavenumbers, new_intensities

def placeholder(wavenumbers, intensities, broadening):
    new_intensities = np.zeros_like(wavenumbers)

    for wave, intn in zip(wavenumbers, intensities):
        new_intensities += intn * instrument_fn(wavenumbers, wave, broadening)

    return new_intensities

def instrument_fn(convolved_wavenumbers, wavenumber_peak, broadening):
    return np.exp(- 0.5 * (convolved_wavenumbers - wavenumber_peak)**2 / broadening**2) / \
           (broadening * np.sqrt(2 * np.pi))

def broadening_fn(convolved_wavenumbers: np.ndarray, wavenumber_peak: float, temp: float, pres: float,
             idx, lines) -> float:
    # TODO: 11/19/23 this function needs to be reworked since I also want to include the ability to
    #                convolve with an instrument function - ideally it takes in a convolution type
    #                and broadening parameters

    # natural (Lorentzian)
    natural = cn.CROSS_SEC**2 * np.sqrt(8 / (np.pi * cn.MASS_REDUCED * cn.BOLTZ * temp)) / 4

    # doppler (Gaussian)
    doppler = wavenumber_peak * np.sqrt((cn.BOLTZ * temp) / (cn.MASS_MOLECULE * (cn.LIGHT / 1e2)**2))

    # collisional (Lorentzian)
    # convert pressure in [N/m^2] to pressure in [dyne/cm^2]
    collide = (pres * 10) * cn.CROSS_SEC**2 * np.sqrt(8 / (np.pi * cn.MASS_REDUCED * cn.BOLTZ * temp)) / 2

    # predissociation (Lorentzian)
    prediss = lines[idx].predissociation()

    # TODO: this might be wrong, not sure if the parameters just add together or what
    gauss = doppler
    loren = natural + collide + prediss

    # Faddeeva function
    fadd = ((convolved_wavenumbers - wavenumber_peak) + 1j * loren) / (gauss * np.sqrt(2))

    return np.real(wofz(fadd)) / (gauss * np.sqrt(2 * np.pi))
