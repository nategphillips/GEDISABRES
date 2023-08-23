# module convolution
'''
Convolves the data if necessary.
'''

from scipy.special import wofz # pylint: disable=no-name-in-module
import numpy as np

import constants as cn

def convolve(convolved_wavenumbers: np.ndarray, wavenumber_peaks: float, temperature: float,
             pressure: float) -> float:
    # Mass of molecular oxygen [kg]
    m_o2 = (2 * 15.999) / cn.AVOGD / 1e3
    # Collisional cross section of O2 with O2 (ground state radius) [cm]
    sigma_ab = np.pi * (cn.X_RAD + cn.X_RAD)**2
    # Reduced mass [kg]
    mu_ab = (m_o2 * m_o2) / (m_o2 + m_o2)

    # Natural [1/cm]
    gamma_n = sigma_ab**2 * np.sqrt(8 / (np.pi * mu_ab * cn.BOLTZ * temperature)) / 4

    # Doppler [1/cm]
    sigma_v = wavenumber_peaks * np.sqrt((cn.BOLTZ * temperature) / (m_o2 * (cn.LIGHT / 1e2)**2))

    # Collision [1/cm]
    # Convert pressure in N/m^2 to pressure in dyne/cm^2
    gamma_v = (pressure * 10) * sigma_ab**2 * np.sqrt(8 / (np.pi * mu_ab * cn.BOLTZ * temperature)) / 2

    gamma = np.sqrt(gamma_n**2 + gamma_v**2)

    # Faddeeva function
    fadd = ((convolved_wavenumbers - wavenumber_peaks) + 1j * gamma) / (sigma_v * np.sqrt(2))

    return np.real(wofz(fadd)) / (sigma_v * np.sqrt(2 * np.pi))

def convolved_data(wavenumbers, intensities, temperature: float,
                   pressure: float) -> tuple:
    # Generate a fine-grained x-axis for plotting
    convolved_wavenumbers = np.linspace(min(wavenumbers), max(wavenumbers), 10000)
    convolved_intensities = np.zeros_like(convolved_wavenumbers)

    # Convolve wavenumber peaks with chosen probability density function
    for wavenumber_peaks, intensity_peaks in zip(wavenumbers, intensities):
        convolved_intensities += intensity_peaks * convolve(convolved_wavenumbers, wavenumber_peaks,
                                                            temperature, pressure)
    convolved_intensities /= max(convolved_intensities)

    return convolved_wavenumbers, convolved_intensities
