# module terms.py
"""Contains a function used for vibrational term calculations."""

# Copyright (C) 2023-2025 Nathan G. Phillips

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import TYPE_CHECKING

import numpy as np

import enums
from constants import DUNHAM_MAP
from state import State

if TYPE_CHECKING:
    from numpy.typing import NDArray


def dunham_helper(state: State, v_qn: int, dunham_column: int):
    dunham: NDArray[np.float64] = np.loadtxt(
        f"../data/{state.molecule.name}/dunham/{state.name}.csv", delimiter=","
    )
    g_consts = dunham[:, dunham_column]

    result: float = 0.0

    for idx, const in enumerate(g_consts):
        result += const * (v_qn + 0.5) ** idx

    return result


def get_g(state: State, v_qn: int, constants_type: enums.ConstantsType) -> float:
    if constants_type == enums.ConstantsType.DUNHAM:
        return dunham_helper(state, v_qn, DUNHAM_MAP["G"])

    return state.constants["G"][v_qn]


def get_b(state: State, v_qn: int, constants_type: enums.ConstantsType) -> float:
    if constants_type == enums.ConstantsType.DUNHAM:
        return dunham_helper(state, v_qn, DUNHAM_MAP["B"])

    return state.constants["B"][v_qn]


def get_d(state: State, v_qn: int, constants_type: enums.ConstantsType) -> float:
    if constants_type == enums.ConstantsType.DUNHAM:
        return dunham_helper(state, v_qn, DUNHAM_MAP["D"])

    return state.constants["D"][v_qn]
