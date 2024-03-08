# module state

import pandas as pd
import numpy as np

class State:
    def __init__(self, name: str, consts: pd.DataFrame) -> None:
        self.name:          str              = name
        self.consts:        dict[str, float] = consts.loc[self.name].to_dict()
        self.cross_section: float            = np.pi * (2 * self.consts['rad'])**2
