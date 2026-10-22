from __future__ import annotations

import numpy as np
from jmetal.core.operator import Mutation
from jmetal.core.solution import FloatSolution


class FastPolynomialMutation(Mutation[FloatSolution]):
    """Vectorized polynomial mutation — Python for-loop を numpy に置換。"""

    def __init__(self, probability: float, distribution_index: float = 20.0):
        super().__init__(probability=probability)
        self.distribution_index = distribution_index
        self._lb: np.ndarray | None = None
        self._ub: np.ndarray | None = None

    def execute(self, solution: FloatSolution) -> FloatSolution:
        if self._lb is None:
            self._lb = np.asarray(solution.lower_bound)
            self._ub = np.asarray(solution.upper_bound)
        y = np.asarray(solution.variables)
        n = len(y)

        # 変異確率は通常 1/D なので当たりは平均1変数程度。全変数分の
        # polynomial 計算（np.power ×4）を避け、当たった添字だけ計算する。
        idx = np.nonzero(np.random.random(n) <= self.probability)[0]
        if idx.size == 0:
            solution.variables = y
            return solution

        lb = self._lb[idx]
        ub = self._ub[idx]
        ys = y[idx]

        dy      = ub - lb
        safe_dy = np.where(dy > 0, dy, 1.0)

        delta1 = (ys - lb) / safe_dy
        delta2 = (ub - ys) / safe_dy

        rnd     = np.random.random(idx.size)
        eta     = self.distribution_index
        mut_pow = 1.0 / (eta + 1.0)

        val1    = 2.0 * rnd + (1.0 - 2.0 * rnd) * np.power(1.0 - delta1, eta + 1.0)
        deltaq1 = np.power(np.maximum(0.0, val1), mut_pow) - 1.0

        val2    = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * np.power(1.0 - delta2, eta + 1.0)
        deltaq2 = 1.0 - np.power(np.maximum(0.0, val2), mut_pow)

        deltaq = np.where(rnd <= 0.5, deltaq1, deltaq2)
        y_mut  = np.clip(ys + deltaq * dy, lb, ub)
        y_mut  = np.where(dy == 0, lb, y_mut)

        out = np.array(y, dtype=float)
        out[idx] = y_mut
        solution.variables = out
        return solution

    def get_number_of_parents(self) -> int:
        return 1

    def get_number_of_children(self) -> int:
        return 1

    def get_name(self) -> str:
        return "Fast polynomial mutation"
