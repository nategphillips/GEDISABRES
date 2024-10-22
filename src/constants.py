# Avodagro constant [1/mol]
AVOGD: float = 6.02214076e23
# Boltzmann constant [J/K]
BOLTZ: float = 1.380649e-23
# Speed of light [cm/s]
LIGHT: float = 2.99792458e10
# Planck constant [J*s]
PLANC: float = 6.62607015e-34

# Atomic masses [g/mol]
ATOMIC_MASSES: dict[str, float] = {"O": 15.999}

# Internuclear distance [m]
# Data from NIST Chemistry WebBook
INTERNUCLEAR_DISTANCE: dict[str, dict[str, float]] = {
    "O2": {"X3Sg-": 1.20752e-10, "B3Su-": 1.6042e-10}
}
