from __future__ import annotations

from typing import List, TypeVar

import numpy as np
from jmetal.util.comparator import SolutionAttributeComparator
from jmetal.util.ranking import Ranking

S = TypeVar("S")


class NumpyNonDominatedRanking(Ranking[List[S]]):
    """Non-dominated ranking using numpy broadcasting instead of O(n²) Python comparator calls."""

    def compute_ranking(self, solutions: List[S], k: int = None):
        if not solutions:
            self.ranked_sublists = []
            return self.ranked_sublists

        n = len(solutions)
        obj = np.array([s.objectives for s in solutions], dtype=np.float64)  # (n, m)

        # dominates[i, j] = True iff solution i dominates solution j (lower is better)
        diff = obj[:, None, :] - obj[None, :, :]          # (n, n, m)
        dominates = (diff <= 0).all(axis=2) & (diff < 0).any(axis=2)  # (n, n)

        dominating_count = dominates.sum(axis=0).tolist()  # how many solutions dominate each j
        dominated_by = [dominates[i].nonzero()[0].tolist() for i in range(n)]  # what i dominates

        current_front = [i for i in range(n) if dominating_count[i] == 0]
        for i in current_front:
            solutions[i].attributes["dominance_ranking"] = 0

        self.ranked_sublists = []
        front_index = 0
        total_count = 0

        while current_front:
            self.ranked_sublists.append([solutions[i] for i in current_front])
            total_count += len(current_front)
            if k and total_count >= k:
                break

            next_front = []
            for p in current_front:
                for q in dominated_by[p]:
                    dominating_count[q] -= 1
                    if dominating_count[q] == 0:
                        next_front.append(q)
                        solutions[q].attributes["dominance_ranking"] = front_index + 1

            current_front = next_front
            front_index += 1

        return self.ranked_sublists

    @classmethod
    def get_comparator(cls):
        return SolutionAttributeComparator("dominance_ranking")
