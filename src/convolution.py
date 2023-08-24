# module convolution
'''
Convolves the quantized spectral data by applying thermal doppler broadening, pressure broadening,
natural broadening, and predissociation broadening.
'''

from scipy.special import wofz # pylint: disable=no-name-in-module
import numpy as np

import constants as cn
import input as inp

def convolve(convolved_wavenumbers: np.ndarray, wavenumber_peak: float,
             temp: float, pres: float) -> float:
    '''
    Each step of the convolution involves the calculation of several broadening parameters, which
    depend on several factors. After these are calculated, this returns the Voigt probability
    distribution function.

    Args:
        convolved_wavenumbers (np.ndarray): array of all valid wavenumbers in the current plot
        wavenumber_peak (float): current quantized wavenumber
        temp (float): temperature
        pres (float): pressure

    Returns:
        float: Voigt probability distribution function
    '''

    # TODO: these calculations should be removed from the function considering that the mass and
    # such don't change for each iteration of the convolution.

    # Mass of molecular oxygen [kg]
    m_o2 = (2 * 15.999) / cn.AVOGD / 1e3
    # Collisional cross section of O2 with O2 (ground state radius) [cm]
    sigma_ab = np.pi * (cn.X_RAD + cn.X_RAD)**2
    # Reduced mass [kg]
    mu_ab = (m_o2 * m_o2) / (m_o2 + m_o2)

    # Natural [1/cm]
    gamma_n = sigma_ab**2 * np.sqrt(8 / (np.pi * mu_ab * cn.BOLTZ * temp)) / 4

    # Doppler [1/cm]
    sigma_v = wavenumber_peak * np.sqrt((cn.BOLTZ * temp) / (m_o2 * (cn.LIGHT / 1e2)**2))

    # Collision [1/cm]
    # Convert pressure in N/m^2 to pressure in dyne/cm^2
    gamma_v = (pres * 10) * sigma_ab**2 * np.sqrt(8 / (np.pi * mu_ab * cn.BOLTZ * temp)) / 2

    gamma = np.sqrt(gamma_n**2 + gamma_v**2)

    # Faddeeva function
    fadd = ((convolved_wavenumbers - wavenumber_peak) + 1j * gamma) / (sigma_v * np.sqrt(2))

    return np.real(wofz(fadd)) / (sigma_v * np.sqrt(2 * np.pi))

def convolved_data(wavenumbers: np.ndarray, intensities: np.ndarray,
                   temp: float, pres: float) -> tuple[np.ndarray, np.ndarray]:
    '''
    Generates the final convolved data.

    Args:
        wavenumbers (np.ndarray): all quantized wavenumbers
        intensities (np.ndarray): matching intensities
        temp (float): temperature
        pres (float): pressure

    Returns:
        tuple[np.ndarray, np.ndarray]: convolved wavenumbers and intensities (both continuous)
    '''

    # Generate a fine-grained x-axis for plotting
    convolved_wavenumbers = np.linspace(wavenumbers.min(), wavenumbers.max(), inp.CONV_GRAN)
    convolved_intensities = np.zeros_like(convolved_wavenumbers)

    # Convolve wavenumber peaks with chosen probability density function
    for wavenumber_peak, intensity_peak in zip(wavenumbers, intensities):
        convolved_intensities += intensity_peak * convolve(convolved_wavenumbers, wavenumber_peak,
                                                            temp, pres)
    convolved_intensities /= convolved_intensities.max()

    return convolved_wavenumbers, convolved_intensities
