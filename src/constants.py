import pandas as pd

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

# Molecular constants [1/cm]
CONSTS_UP: dict[str, dict[int, float]] = pd.read_csv("../data/constants/O2/B3Su-.csv").to_dict()
CONSTS_LO: dict[str, dict[int, float]] = pd.read_csv("../data/constants/O2/X3Sg-.csv").to_dict()

# Molecular constants [1/cm]
MOLECULAR_CONSTANTS: dict[str, dict[str, dict[str, float]]] = {
    "O2": {
        "X3Sg-": {
            "T_e": 0.0,
            "w_e": 1580.193,
            "we_xe": 11.981,
            "we_ye": 0.04747,
            "we_ze": -0.001273,
            "B_e": 1.4456,
            "alpha_e": 0.01593,
            "gamma_e": 0.0,
            "delta_e": 0.0,
            "D_e": 4.839e-6,
            "beta_e": 0.0,
            "H_e": 0.0,
            "lamda": 1.9847511,
            "gamma": -0.00842536,
        },
        "B3Su-": {
            "T_e": 49793.28,
            "w_e": 709.31,
            "we_xe": 10.65,
            "we_ye": -0.139,
            "we_ze": 0.0,
            "B_e": 0.81902,
            "alpha_e": 0.01206,
            "gamma_e": -5.56e-4,
            "delta_e": 0.0,
            "D_e": 4.55e-6,
            "beta_e": 0.22e-6,
            "H_e": 0.0,
            "lamda": 1.5,
            "gamma": -0.04,
        },
    }
}
