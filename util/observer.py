"""Custom observers used during optimisation runs."""

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

import numpy as np
from jmetal.core.observer import Observer
from jmetal.core.quality_indicator import HyperVolume
from jmetal.util.termination_criterion import TerminationCriterion
import csv


@dataclass
class _Record:
    """Container for observer data."""

    iteration: int
    values: Union[float, List[float]]

class ValueObserver(Observer):
    """
    逐次最良目的値を記録するObserver
    """
    def __init__(self) -> None:
        super().__init__()
        self._records: List[_Record] = []

    @property
    def iterations(self) -> List[int]:
        return [r.iteration for r in self._records]

    @property
    def best_objectives(self) -> List[Union[float, List[float]]]:
        return [r.values for r in self._records]

    def update(self, *args, **kwargs):
        evaluations = kwargs.get('EVALUATIONS')
        solutions = kwargs.get('SOLUTIONS')
        if solutions is None:
            return
        best = self._extract_best_values(solutions)
        self._records.append(_Record(evaluations, best))

    def _extract_best_values(self, solutions: Union[Sequence[Any], Any]) -> Union[float, List[float]]:
        """Return the best objective value(s) from the given solutions."""
        if isinstance(solutions, list):
            if hasattr(solutions[0], "objectives") and len(solutions[0].objectives) > 1:
                return [min(s.objectives[i] for s in solutions) for i in range(len(solutions[0].objectives))]
            return min(s.objectives[0] for s in solutions)

        if hasattr(solutions, "objectives") and len(solutions.objectives) > 1:
            return list(solutions.objectives)
        return solutions.objectives[0]

    def to_csv(self, path: str):
        """
        記録データをCSV保存
        """
        with open(path, mode="w", newline="") as f:
            writer = csv.writer(f)
            for rec in self._records:
                if isinstance(rec.values, list):
                    writer.writerow([rec.iteration] + rec.values)
                else:
                    writer.writerow([rec.iteration, rec.values])


def _non_dominated_objectives(solutions: Sequence[Any]) -> np.ndarray:
    """Return the objective matrix of the non-dominated subset (numpy O(n²m)).

    jmetal の get_non_dominated_solutions は Python の逐次アーカイブで、
    定常状態アルゴリズム（MOEAD等）が毎評価 notify する場合に全実行時間を
    支配するため使わない。
    """
    obj = np.array([s.objectives for s in solutions], dtype=np.float64)
    diff = obj[:, None, :] - obj[None, :, :]                      # (n, n, m)
    dominates = (diff <= 0).all(axis=2) & (diff < 0).any(axis=2)  # i dominates j
    return obj[~dominates.any(axis=0)]


class HyperVolumeObserver(Observer):
    """毎世代HVを記録するObserver。

    参照点は問題の最適化方向から自動設定:
      MINIMIZE → 1.0
      MAXIMIZE → 0.0  (objectives は内部でnegate済みなので -obj ≤ 0 が保証される)

    min_evaluation_interval: 前回記録からこの評価数が経過するまで記録を
    スキップする。定常状態アルゴリズム（MOEAD 等、1評価ごとに notify）では
    集団サイズ程度を指定しないと HV 計算が実行時間を支配する。
    """

    def __init__(self, min_evaluation_interval: int = 0) -> None:
        super().__init__()
        self._records: List[_Record] = []
        self._hv_calculator: Optional[HyperVolume] = None
        self._min_interval = min_evaluation_interval
        self._last_recorded: Optional[int] = None

    @property
    def iterations(self) -> List[int]:
        return [r.iteration for r in self._records]

    @property
    def hypervolumes(self) -> List[float]:
        return [r.values for r in self._records]

    def _build_calculator(self, problem) -> None:
        ref_point = [
            1.0 if d == problem.MINIMIZE else 0.0
            for d in problem.obj_directions
        ]
        self._hv_calculator = HyperVolume(reference_point=ref_point)

    def update(self, *args, **kwargs) -> None:
        problem = kwargs.get("PROBLEM")
        solutions = kwargs.get("SOLUTIONS")
        evaluations = kwargs.get("EVALUATIONS")

        if not solutions or problem is None:
            return

        if (
            self._last_recorded is not None
            and evaluations is not None
            and evaluations - self._last_recorded < self._min_interval
        ):
            return

        if self._hv_calculator is None:
            self._build_calculator(problem)

        obj_array = _non_dominated_objectives(solutions)
        if obj_array.size == 0:
            return

        hv_value = self._hv_calculator.compute(obj_array)
        self._records.append(_Record(evaluations, hv_value))
        self._last_recorded = evaluations

    def to_csv(self, path: str) -> None:
        with open(path, mode="w", newline="") as f:
            writer = csv.writer(f)
            for rec in self._records:
                writer.writerow([rec.iteration, rec.values])


class StoppingByAny(TerminationCriterion):
    """複数の終了条件のOR結合。いずれかが成立したら停止する。"""

    def __init__(self, *criteria: TerminationCriterion) -> None:
        super().__init__()
        self.criteria = list(criteria)

    def update(self, *args, **kwargs) -> None:
        for criterion in self.criteria:
            criterion.update(*args, **kwargs)

    @property
    def is_met(self) -> bool:
        return any(criterion.is_met for criterion in self.criteria)

    @property
    def met_criteria(self) -> List[str]:
        """成立した条件のクラス名一覧（停止理由の記録用）。"""
        return [type(c).__name__ for c in self.criteria if c.is_met]


class StoppingByHVStall(TerminationCriterion):
    """HVの停滞で停止する終了条件。

    HyperVolumeObserver の記録系列を直接参照する（HVの二重計算を避ける）。
    直近 patience+1 記録のHVの相対変動幅が rel_tol 以下なら停滞と判定する。
    rel_tol=0.0（既定）は「同一HVが patience 記録続いたら停止」の意味。
    相対判定なのでHVの絶対スケール（1e-6等）に依存しない。

    ウィンドウ内のHVがすべて0以下の間は判定しない。解が参照点の内側に
    まだ入っていないだけで、停滞とは区別できないため。
    """

    def __init__(
        self,
        hv_observer: "HyperVolumeObserver",
        patience: int = 20,
        rel_tol: float = 0.0,
    ) -> None:
        super().__init__()
        self.hv_observer = hv_observer
        self.patience = patience
        self.rel_tol = rel_tol

    def update(self, *args, **kwargs) -> None:
        # HV計算は HyperVolumeObserver 側で行われるため何もしない。
        # is_met が呼ばれた時点の記録系列（今世代分を含む）を参照する。
        pass

    @property
    def is_met(self) -> bool:
        hvs = self.hv_observer.hypervolumes
        if len(hvs) < self.patience + 1:
            return False
        window = hvs[-(self.patience + 1):]
        hi = max(window)
        if hi <= 0.0:
            return False
        return (hi - min(window)) <= self.rel_tol * hi


class SaveResultsObserver(Observer):
    """Observer that saves the results of another observer to CSV."""

    def __init__(self, csv_file: str, observer_instance: ValueObserver) -> None:
        super().__init__()
        self.csv_file = csv_file
        self.observer_instance = observer_instance

    def update(self, *args, **kwargs):  # noqa: D401
        """Write current observer data to ``csv_file``."""
        with open(self.csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            for iter_, values in zip(
                self.observer_instance.iterations,
                self.observer_instance.best_objectives,
            ):
                row = [iter_] + (values if isinstance(values, list) else [values])
                writer.writerow(row)

