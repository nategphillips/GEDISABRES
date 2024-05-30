# module atom

import pandas as pd

import constants as cn

class Atom:
    def __init__(self, name: str) -> None:
        self.name: str   = name
        self.mass: float = (pd.read_csv('../data/atomic_masses.csv', index_col=0)
                            ['mass'][self.name] / cn.AVOGD / 1e3)
