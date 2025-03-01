# module utils
"""Contains useful utility functions."""


def n_to_j(n_qn: int, branch_idx: int) -> int:
    """Convert from N to J."""
    # For Hund's case (b), spin multiplicity 3.
    match branch_idx:
        case 1:
            # F1: J = N + 1
            return n_qn + 1
        case 2:
            # F2: J = N
            return n_qn
        case 3:
            # F3: J = N - 1
            return n_qn - 1
        case _:
            raise ValueError(f"Unknown branch index: {branch_idx}.")


def wavenum_to_wavelen(x):
    """Convert wavenumbers to wavelengths and vice versa."""
    return 1.0 / x * 1e7


def bandwidth_wavelen_to_wavenum(center_wl: float, fwhm_wl: float):
    """Convert a FWHM bandwidth from [nm] to [1/cm] given a center wavelength.

    Note that this is not a linear approximation, so it is accurate for large FWHM parameters. See
    https://toolbox.lightcon.com/tools/bandwidthconverter for details.
    """
    return 1e7 * fwhm_wl / (center_wl**2 - fwhm_wl**2 / 4)
