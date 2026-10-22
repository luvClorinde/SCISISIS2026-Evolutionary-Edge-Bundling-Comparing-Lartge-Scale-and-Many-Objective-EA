from __future__ import annotations

import random
from typing import List

import numpy as np
from jmetal.core.operator import Crossover
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import DifferentialEvolutionCrossover


class FastDECrossover(DifferentialEvolutionCrossover):
    """deepcopy を回避し、DE変異をnumpy化した高速版。

    境界は ndarray でキャッシュする。FloatSolution の lower_bound / upper_bound
    は Python リストのため、np.clip に毎回渡すと呼び出しごとに D 要素の
    リスト→配列変換が走り、大規模グラフではそれだけで支配的なコストになる。
    """

    _lb: np.ndarray | None = None
    _ub: np.ndarray | None = None

    def execute(self, parents):
        current = self.current_individual

        if self._lb is None:
            self._lb = np.asarray(current.lower_bound, dtype=float)
            self._ub = np.asarray(current.upper_bound, dtype=float)

        n = len(current.variables)
        rand_idx = random.randint(0, n - 1)

        current_vars = np.asarray(current.variables)
        new_vars = (
            np.asarray(parents[2].variables)
            + self.F * (np.asarray(parents[0].variables) - np.asarray(parents[1].variables))
        )

        mask = np.random.random(n) < self.CR
        mask[rand_idx] = True

        result = np.clip(np.where(mask, new_vars, current_vars), self._lb, self._ub)

        child = object.__new__(FloatSolution)
        child.lower_bound = current.lower_bound
        child.upper_bound = current.upper_bound
        child.objectives  = [0.0] * len(current.objectives)
        child.constraints = []
        child.attributes  = {}
        child.variables   = result

        return [child]


class FastSBXCrossover(Crossover[FloatSolution, FloatSolution]):
    """SBXCrossover without deepcopy.

    jMetal の SBXCrossover は execute() の冒頭で親個体を deepcopy するが、
    lower_bound / upper_bound（変数次元数の大きなリスト）まで毎回コピーされるため
    大規模グラフで著しく遅くなる。
    本クラスでは子個体を直接生成し、variables のみ ndarray.copy() でコピーする。
    """

    __EPS = 1.0e-14

    def __init__(self, probability: float, distribution_index: float = 20.0):
        super().__init__(probability=probability)
        self.distribution_index = distribution_index
        self._lb: np.ndarray | None = None
        self._ub: np.ndarray | None = None

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        if self._lb is None:
            self._lb = np.asarray(parents[0].lower_bound)
            self._ub = np.asarray(parents[0].upper_bound)
        x1 = np.asarray(parents[0].variables)
        x2 = np.asarray(parents[1].variables)
        lb = self._lb
        ub = self._ub
        n  = len(x1)

        out0, out1 = x1.copy(), x2.copy()

        if random.random() <= self.probability:
            do_sbx = (np.random.random(n) <= 0.5) & (np.abs(x1 - x2) > self.__EPS)
            y1 = np.minimum(x1, x2)
            y2 = np.maximum(x1, x2)
            dy = np.where(do_sbx, y2 - y1, 1.0)  # ゼロ除算防止

            rand = np.random.random(n)
            eta = self.distribution_index + 1.0

            # c1
            beta1  = 1.0 + 2.0 * (y1 - lb) / dy
            alpha1 = 2.0 - np.power(beta1, -eta)
            betaq1 = np.where(rand <= 1.0 / alpha1,
                np.power(rand * alpha1, 1.0 / eta),
                np.power(1.0 / (2.0 - rand * alpha1), 1.0 / eta))
            c1 = np.clip(0.5 * (y1 + y2 - betaq1 * dy), lb, ub)

            # c2（同じrandを使う）
            beta2  = 1.0 + 2.0 * (ub - y2) / dy
            alpha2 = 2.0 - np.power(beta2, -eta)
            betaq2 = np.where(rand <= 1.0 / alpha2,
                np.power(rand * alpha2, 1.0 / eta),
                np.power(1.0 / (2.0 - rand * alpha2), 1.0 / eta))
            c2 = np.clip(0.5 * (y1 + y2 + betaq2 * dy), lb, ub)

            swap = np.random.random(n) <= 0.5
            out0 = np.where(do_sbx, np.where(swap, c2, c1), x1)
            out1 = np.where(do_sbx, np.where(swap, c1, c2), x2)

        child0 = object.__new__(FloatSolution)
        child0.lower_bound = parents[0].lower_bound
        child0.upper_bound = parents[0].upper_bound
        child0.objectives  = [0.0] * len(parents[0].objectives)
        child0.constraints = []
        child0.attributes  = {}
        child0.variables   = out0

        child1 = object.__new__(FloatSolution)
        child1.lower_bound = parents[1].lower_bound
        child1.upper_bound = parents[1].upper_bound
        child1.objectives  = [0.0] * len(parents[1].objectives)
        child1.constraints = []
        child1.attributes  = {}
        child1.variables   = out1

        return [child0, child1]

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return "Fast SBX crossover"
