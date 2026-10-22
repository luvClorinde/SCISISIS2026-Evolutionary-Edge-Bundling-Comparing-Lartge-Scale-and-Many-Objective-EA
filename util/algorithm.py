from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    _HAS_CUPY = False
from jmetal.algorithm.multiobjective.moead import MOEAD
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.nsgaiii import (
    NSGAIII,
    UniformReferenceDirectionFactory,
    associate_to_niches,
    compute_niche_count,
    get_extreme_points,
    get_nadir_point,
    niching,
)
from jmetal.core.algorithm import Algorithm
from jmetal.core.operator import Crossover, Mutation
from jmetal.core.problem import FloatProblem
from jmetal.core.quality_indicator import HyperVolume
from jmetal.core.solution import FloatSolution
from jmetal.util.evaluator import Evaluator
from jmetal.util.termination_criterion import TerminationCriterion

from util.crossover import FastSBXCrossover
from util.mutation import FastPolynomialMutation
from jmetal.operator.replacement import (
    RankingAndDensityEstimatorReplacement,
    RemovalPolicyType,
)
from jmetal.util.aggregation_function import AggregationFunction
from jmetal.util.density_estimator import CrowdingDistanceDensityEstimator
from jmetal.util.point import IdealPoint
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from util.ranking import NumpyNonDominatedRanking


# ─── BatchEvaluator ───────────────────────────────────────────────────────────

class BatchEvaluator(Evaluator):
    """jMetal evaluator that batches via ``problem.evaluate_batch`` when available.

    Pass as ``population_evaluator=`` to any jMetal generational algorithm
    (NSGA-II/III, SPEA2, …) so each offspring generation becomes one GPU batch.
    Falls back to per-solution ``problem.evaluate`` otherwise.
    """

    def evaluate(self, solution_list, problem):
        return _evaluate_solutions(problem, solution_list)


# ─── FastNSGAII ───────────────────────────────────────────────────────────────

class FastNSGAII(NSGAII):
    """NSGA-II with numpy-vectorized non-dominated ranking (no Python comparator loop)."""

    def replacement(self, population, offspring_population):
        ranking = NumpyNonDominatedRanking()
        density_estimator = CrowdingDistanceDensityEstimator()
        r = RankingAndDensityEstimatorReplacement(
            ranking, density_estimator, RemovalPolicyType.ONE_SHOT
        )
        return r.replace(population, offspring_population)


# ─── FastNSGAIII ──────────────────────────────────────────────────────────────

class FastNSGAIII(NSGAIII):
    """NSGA-III with numpy-vectorized non-dominated ranking.

    jMetal's replacement/result instantiate FastNonDominatedRanking (Python
    pairwise comparator, O(n²) interpreted loops) every generation; this
    subclass swaps in NumpyNonDominatedRanking. The environmental-selection
    body is otherwise copied verbatim from jMetal (same reference-point
    association and niching helpers).
    """

    def replacement(self, population, offspring_population):
        F = np.array([s.objectives for s in population])

        self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
        self.worst_point = np.max(np.vstack((self.worst_point, F)), axis=0)

        ranking = NumpyNonDominatedRanking()
        ranking.compute_ranking(population + offspring_population, k=self.population_size)

        fronts, non_dominated = ranking.ranked_sublists, ranking.get_subfront(0)

        self.extreme_points = get_extreme_points(
            F=np.array([s.objectives for s in non_dominated]),
            n_objs=self.problem.number_of_objectives(),
            ideal_point=self.ideal_point,
            extreme_points=self.extreme_points,
        )

        worst_of_population = np.max(F, axis=0)
        worst_of_front = np.max(np.array([s.objectives for s in non_dominated]), axis=0)

        nadir_point = get_nadir_point(
            extreme_points=self.extreme_points,
            ideal_point=self.ideal_point,
            worst_point=self.worst_point,
            worst_of_population=worst_of_population,
            worst_of_front=worst_of_front,
        )

        pop = np.concatenate(ranking.ranked_sublists)
        F = np.array([s.objectives for s in pop])

        counter = 0
        for i in range(len(fronts)):
            for j in range(len(fronts[i])):
                fronts[i][j] = counter
                counter += 1
        last_front = np.array(fronts[-1])

        niche_of_individuals, dist_to_niche = associate_to_niches(
            F=F, niches=self.reference_directions, ideal_point=self.ideal_point, nadir_point=nadir_point
        )

        if len(pop) > self.population_size:
            if len(fronts) == 1:
                until_last_front = np.array([], dtype=int)
                niche_count = np.zeros(len(self.reference_directions), dtype=int)
                n_remaining = self.population_size
            else:
                until_last_front = np.concatenate(fronts[:-1])
                niche_count = compute_niche_count(
                    len(self.reference_directions), niche_of_individuals[until_last_front]
                )
                n_remaining = self.population_size - len(until_last_front)

            S_idx = niching(
                pop=pop[last_front],
                n_remaining=n_remaining,
                niche_count=niche_count,
                niche_of_individuals=niche_of_individuals[last_front],
                dist_to_niche=dist_to_niche[last_front],
            )

            survivors_idx = np.concatenate((until_last_front, last_front[S_idx].tolist()))
            pop = pop[survivors_idx]

        return list(pop)

    def result(self):
        ranking = NumpyNonDominatedRanking()
        ranking.compute_ranking(self.solutions, k=self.population_size)
        return ranking.get_subfront(0)

    def reproduction(self, mating_population):
        """Vectorised SBX + polynomial mutation over the whole mating pool.

        Same per-entry arithmetic and distributions as jMetal's pairwise loop
        with FastSBXCrossover / FastPolynomialMutation (RNG consumption order
        differs). Falls back to the generic loop for any other operator
        combination or an odd offspring size.
        """
        cx = self.crossover_operator
        mt = self.mutation_operator
        n_off = self.offspring_population_size
        if (
            not isinstance(cx, FastSBXCrossover)
            or not isinstance(mt, FastPolynomialMutation)
            or n_off % 2 != 0
            or len(mating_population) < n_off
        ):
            return super().reproduction(mating_population)

        template = mating_population[0]
        lb = np.asarray(template.lower_bound, dtype=float)
        ub = np.asarray(template.upper_bound, dtype=float)
        p1 = np.array(
            [np.asarray(mating_population[i].variables, dtype=float)
             for i in range(0, n_off, 2)]
        )
        p2 = np.array(
            [np.asarray(mating_population[i + 1].variables, dtype=float)
             for i in range(0, n_off, 2)]
        )

        if _HAS_CUPY:
            lb_g, ub_g = cp.asarray(lb), cp.asarray(ub)
            out0, out1 = _sbx_pairs_matrix(
                cp.asarray(p1), cp.asarray(p2), lb_g, ub_g,
                cx.probability, cx.distribution_index,
            )
            all_decs_g = cp.empty((n_off, p1.shape[1]))
            all_decs_g[0::2] = out0
            all_decs_g[1::2] = out1
            all_decs = cp.asnumpy(
                _poly_mut_matrix(all_decs_g, lb_g, ub_g, mt.probability, mt.distribution_index)
            )
        else:
            out0, out1 = _sbx_pairs_matrix(
                p1, p2, lb, ub, cx.probability, cx.distribution_index
            )
            all_decs = np.empty((n_off, p1.shape[1]))
            all_decs[0::2] = out0
            all_decs[1::2] = out1
            all_decs = _poly_mut_matrix(all_decs, lb, ub, mt.probability, mt.distribution_index)

        n_obj = len(template.objectives)
        offspring = []
        for k in range(n_off):
            child = object.__new__(FloatSolution)
            child.lower_bound = template.lower_bound
            child.upper_bound = template.upper_bound
            child.objectives = [0.0] * n_obj
            child.constraints = []
            child.attributes = {}
            child.variables = all_decs[k]
            offspring.append(child)
        return offspring


# ─── FastMOEAD ────────────────────────────────────────────────────────────────

class FastMOEAD(MOEAD):
    """MOEAD with lightweight solution copy in neighbourhood replacement.

    jMetal deep-copies the offspring (including the D-length bound lists and
    attribute dict) once per replaced neighbour; for large D this dominates
    the step cost. Only ``variables`` needs an actual copy — bounds are
    shared read-only.
    """

    def update_current_subproblem_neighborhood(self, new_solution, population):
        permuted_neighbors_indexes = self.generate_permutation_of_neighbors(
            self.current_subproblem
        )

        # Score the whole neighbourhood in two vectorised calls when the
        # aggregation function supports it (bit-identical to the per-k loop;
        # replacement has no effect on later f1 scores because each k appears
        # once and only its own objectives are read).
        batch_fn = getattr(self.fitness_function, "compute_batch", None)
        if batch_fn is not None:
            idx = np.fromiter(permuted_neighbors_indexes, dtype=int)
            weights = np.asarray(self.neighbourhood.weight_vectors, dtype=float)[idx]
            pop_objs = np.array([population[k].objectives for k in idx], dtype=float)
            f1 = batch_fn(pop_objs, weights)
            f2 = batch_fn(
                np.broadcast_to(
                    np.asarray(new_solution.objectives, dtype=float), weights.shape
                ),
                weights,
            )
            replacements = 0
            for pos, k in enumerate(idx):
                if f2[pos] < f1[pos]:
                    population[int(k)] = _lmea_copy_solution(new_solution)
                    replacements += 1
                if replacements >= self.max_number_of_replaced_solutions:
                    break
            return population

        replacements = 0
        for k in permuted_neighbors_indexes:
            f1 = self.fitness_function.compute(
                population[k].objectives, self.neighbourhood.weight_vectors[k]
            )
            f2 = self.fitness_function.compute(
                new_solution.objectives, self.neighbourhood.weight_vectors[k]
            )

            if f2 < f1:
                population[k] = _lmea_copy_solution(new_solution)
                replacements += 1

            if replacements >= self.max_number_of_replaced_solutions:
                break

        return population


# ─── NormalizedTschebycheff ────────────────────────────────────────────────────
#
# jmetalpy's Tschebycheff only translates by the running ideal point; it never
# rescales by objective range. That is fine on benchmarks like DTLZ/LSMOP where
# every objective already lives on a comparable scale, but this project's four
# objectives do not (MELD/PQ are normalised to ~[0,1], MOA/EDD are raw grid-based
# sums with a much larger range). Without rescaling, the largest-scale objective
# dominates every subproblem's max-term regardless of the weight vector, which
# defeats MOEA/D's decomposition. This tracks a running ideal (min) and nadir
# (max) estimate per objective and normalises to [0,1] before the standard
# Tchebycheff computation.

class NormalizedTschebycheff(AggregationFunction):
    """Tchebycheff aggregation with running ideal/nadir normalisation."""

    def __init__(self, dimension: int):
        self.ideal_point = IdealPoint(dimension)
        self._nadir = np.full(dimension, -np.inf)

    def update(self, vector: list) -> None:
        self.ideal_point.update(vector)
        self._nadir = np.maximum(self._nadir, np.asarray(vector, dtype=float))

    def compute(self, vector: list, weight_vector: list) -> float:
        ideal = np.asarray(self.ideal_point.point, dtype=float)
        span = np.where(self._nadir - ideal > 1e-12, self._nadir - ideal, 1.0)
        diff = np.abs((np.asarray(vector, dtype=float) - ideal) / span)
        w = np.asarray(weight_vector, dtype=float)
        feval = np.where(w == 0, 0.0001 * diff, diff * w)
        return float(np.max(feval))

    def compute_batch(self, vectors: np.ndarray, weight_vectors: np.ndarray) -> np.ndarray:
        """Row-wise compute(): (K, M) objective rows × (K, M) weight rows → (K,).

        Same arithmetic as compute(), so results are bit-identical to calling
        it in a loop. Used by FastMOEAD to score a whole neighbourhood at once.
        """
        ideal = np.asarray(self.ideal_point.point, dtype=float)
        span = np.where(self._nadir - ideal > 1e-12, self._nadir - ideal, 1.0)
        diff = np.abs((np.asarray(vectors, dtype=float) - ideal) / span)
        w = np.asarray(weight_vectors, dtype=float)
        feval = np.where(w == 0, 0.0001 * diff, diff * w)
        return feval.max(axis=1)


# ─── LMEA: Decision Variable Clustering-Based Evolutionary Algorithm ──────────
#
# Zhang et al., "A Decision Variable Clustering-Based Evolutionary Algorithm
# for Large-Scale Many-Objective Optimization," IEEE TEVC, 2018.
#
# platEMO-faithful implementation (VariableClustering / CorrelationAnalysis /
# ConvergenceOptimization / DistributionOptimization).

# ── Solution helpers ──────────────────────────────────────────────────────────

def _lmea_copy_solution(sol: FloatSolution) -> FloatSolution:
    new = object.__new__(FloatSolution)
    new.lower_bound = sol.lower_bound
    new.upper_bound = sol.upper_bound
    new.objectives = list(sol.objectives)
    new.constraints = list(sol.constraints) if sol.constraints else []
    new.attributes = {}
    new.variables = np.asarray(sol.variables).copy()
    return new


def _evaluate_decs(
    problem: FloatProblem,
    decs: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> np.ndarray:
    """Evaluate a (B, D) matrix of decision vectors → (B, M) objectives.

    Uses ``problem.evaluate_batch`` (single batched GPU pass) when the problem
    provides it; otherwise falls back to per-solution ``problem.evaluate`` so
    problems without batch support (e.g. jMetal benchmarks) keep working.
    Both paths advance ``problem.evaluation`` by B.
    """
    batch_fn = getattr(problem, "evaluate_batch", None)
    if batch_fn is not None:
        return np.asarray(batch_fn(decs), dtype=float)
    n_obj = problem.number_of_objectives()
    out = np.empty((len(decs), n_obj), dtype=float)
    lb_list, ub_list = list(lb), list(ub)
    for k, dec in enumerate(decs):
        sol = FloatSolution(lb_list, ub_list, n_obj)
        sol.variables = np.asarray(dec, dtype=float).copy()
        out[k] = problem.evaluate(sol).objectives
    return out


def _evaluate_solutions(
    problem: FloatProblem, solution_list: List[FloatSolution]
) -> List[FloatSolution]:
    """Evaluate solutions in-place, batching via ``evaluate_batch`` if available."""
    if not solution_list:
        return solution_list
    batch_fn = getattr(problem, "evaluate_batch", None)
    if batch_fn is None:
        return [problem.evaluate(sol) for sol in solution_list]
    decs = np.array([np.asarray(s.variables, dtype=float) for s in solution_list])
    for sol, objs in zip(solution_list, batch_fn(decs)):
        sol.objectives = list(objs)
    return solution_list


# ── Genetic operators on variable subsets (matrix / whole-population) ────────

# 行チャンク1回あたりの一時配列サイズ上限（要素数; float64で128KB）。
# メモリ保護に加え、チャンクをL2キャッシュに収めるのが狙い。実測では
# 2M要素（16MB）にするとメモリ帯域律速になり per-row ループより遅くなる
# （16k: 0.75s / 2M: 1.17s @ (500行, g=14,500) SBX）。
_MATRIX_OP_ELEMS = 16_000


def _sbx_values_matrix(
    x1: np.ndarray,
    x2: np.ndarray,
    lb_s: np.ndarray,
    ub_s: np.ndarray,
    eta: float = 20.0,
    rand_fn: Optional[Callable] = None,
) -> np.ndarray:
    """Row-wise SBX (single child, platEMO form) on (C, g) parent-value matrices.

    Same arithmetic as the former per-individual `_sbx_on_indices`, applied to
    every row at once. numpy 入力では temporaries が _MATRIX_OP_ELEMS 以下に
    なるよう行チャンク処理、cupy 入力では全行を1パスで処理する。
    rand_fn は等価性テスト用の乱数注入フック（shape → 一様乱数）。
    """
    EPS = 1e-14
    xp = cp.get_array_module(x1) if _HAS_CUPY else np
    rand = rand_fn if rand_fn is not None else xp.random.random
    n_rows, g = x1.shape
    out = xp.empty_like(x1)
    rows = n_rows if xp is not np else max(1, _MATRIX_OP_ELEMS // max(1, g))
    eta1 = eta + 1.0
    for s in range(0, n_rows, rows):
        a = x1[s:s + rows]
        b = x2[s:s + rows]
        do_sbx = (rand(a.shape) <= 0.5) & (xp.abs(a - b) > EPS)
        y1 = xp.minimum(a, b)
        y2 = xp.maximum(a, b)
        dy = xp.where(do_sbx, y2 - y1, 1.0)
        rnd = rand(a.shape)

        beta1 = 1.0 + 2.0 * (y1 - lb_s) / dy
        alpha1 = 2.0 - xp.power(xp.maximum(beta1, 1e-10), -eta1)
        betaq1 = xp.where(
            rnd <= 1.0 / alpha1,
            xp.power(xp.maximum(rnd * alpha1, 0.0), 1.0 / eta1),
            xp.power(xp.maximum(1.0 / (2.0 - rnd * alpha1), 0.0), 1.0 / eta1),
        )
        c1 = xp.clip(0.5 * (y1 + y2 - betaq1 * dy), lb_s, ub_s)
        out[s:s + rows] = xp.where(do_sbx, c1, a)
    return out


def _poly_mut_matrix(
    vals: np.ndarray,
    lb_s: np.ndarray,
    ub_s: np.ndarray,
    prob: float,
    eta: float = 20.0,
    rand_fn: Optional[Callable] = None,
) -> np.ndarray:
    """Sparse polynomial mutation on a (C, g) value matrix (numpy/cupy 両対応).

    prob is typically 1/D, so only the hit entries (≈ C·g·prob of C·g) get the
    polynomial computation; the Bernoulli draw is the only full-width cost.
    Same per-entry arithmetic as FastPolynomialMutation / the former
    `_poly_mut_on_indices`.
    rand_fn は等価性テスト用の乱数注入フック（shape → 一様乱数）。
    """
    xp = cp.get_array_module(vals) if _HAS_CUPY else np
    rand = rand_fn if rand_fn is not None else xp.random.random
    out = vals.copy()
    site = rand(out.shape) <= prob
    r, c = xp.nonzero(site)
    if r.size == 0:
        return out

    lb_v = lb_s[c]
    ub_v = ub_s[c]
    y = out[r, c]
    dy = ub_v - lb_v
    safe_dy = xp.where(dy > 0, dy, 1.0)
    delta1 = (y - lb_v) / safe_dy
    delta2 = (ub_v - y) / safe_dy
    rnd = rand(r.size)
    mut_pow = 1.0 / (eta + 1.0)

    val1 = 2.0 * rnd + (1.0 - 2.0 * rnd) * xp.power(1.0 - delta1, eta + 1.0)
    dq1 = xp.power(xp.maximum(0.0, val1), mut_pow) - 1.0
    val2 = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * xp.power(1.0 - delta2, eta + 1.0)
    dq2 = 1.0 - xp.power(xp.maximum(0.0, val2), mut_pow)
    deltaq = xp.where(rnd <= 0.5, dq1, dq2)
    y_mut = xp.clip(y + deltaq * dy, lb_v, ub_v)
    y_mut = xp.where(dy == 0, lb_v, y_mut)

    out[r, c] = y_mut
    return out


def _sbx_pairs_matrix(
    p1: np.ndarray,
    p2: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    probability: float,
    distribution_index: float,
    rand_fn: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised FastSBXCrossover over parent pairs: (P, D)×2 → two (P, D) children.

    Row-wise identical arithmetic to FastSBXCrossover.execute: per-pair
    crossover gate, per-variable 50% SBX mask, shared rand for c1/c2,
    per-variable swap. The polynomial terms run only on the do_sbx entries
    (~45% density) via flat gather; numpy 入力では行チャンクで temporaries を
    抑え、cupy 入力では全行1パスで処理する。
    rand_fn は等価性テスト用の乱数注入フック（shape → 一様乱数）。
    """
    EPS = 1e-14
    xp = cp.get_array_module(p1) if _HAS_CUPY else np
    rand = rand_fn if rand_fn is not None else xp.random.random
    n_pairs, D = p1.shape
    out0 = p1.copy()
    out1 = p2.copy()
    do_cross = rand(n_pairs) <= probability
    eta = distribution_index + 1.0
    rows = n_pairs if xp is not np else max(1, _MATRIX_OP_ELEMS // max(1, D))
    for s in range(0, n_pairs, rows):
        x1 = p1[s:s + rows]
        x2 = p2[s:s + rows]
        do_sbx = (
            (rand(x1.shape) <= 0.5)
            & (xp.abs(x1 - x2) > EPS)
            & do_cross[s:s + rows, None]
        )
        r, c = xp.nonzero(do_sbx)
        if r.size == 0:
            continue
        a = x1[r, c]
        b = x2[r, c]
        lb_v = lb[c]
        ub_v = ub[c]
        y1 = xp.minimum(a, b)
        y2 = xp.maximum(a, b)
        dy = y2 - y1  # gathered entries all have |a-b| > EPS → dy > 0
        rnd = rand(r.size)

        beta1 = 1.0 + 2.0 * (y1 - lb_v) / dy
        alpha1 = 2.0 - xp.power(beta1, -eta)
        betaq1 = xp.where(
            rnd <= 1.0 / alpha1,
            xp.power(rnd * alpha1, 1.0 / eta),
            xp.power(1.0 / (2.0 - rnd * alpha1), 1.0 / eta),
        )
        c1v = xp.clip(0.5 * (y1 + y2 - betaq1 * dy), lb_v, ub_v)

        beta2 = 1.0 + 2.0 * (ub_v - y2) / dy
        alpha2 = 2.0 - xp.power(beta2, -eta)
        betaq2 = xp.where(
            rnd <= 1.0 / alpha2,
            xp.power(rnd * alpha2, 1.0 / eta),
            xp.power(1.0 / (2.0 - rnd * alpha2), 1.0 / eta),
        )
        c2v = xp.clip(0.5 * (y1 + y2 + betaq2 * dy), lb_v, ub_v)

        swap = rand(r.size) <= 0.5
        out0[r + s, c] = xp.where(swap, c2v, c1v)
        out1[r + s, c] = xp.where(swap, c1v, c2v)
    return out0, out1


# ── GPU dispatch helpers for the matrix operators ────────────────────────────

def _sbx_mut_values(
    x1: np.ndarray,
    x2: np.ndarray,
    lb_s: np.ndarray,
    ub_s: np.ndarray,
    crossover_eta: float,
    pm: float,
    mutation_eta: float,
) -> np.ndarray:
    """SBX（単一子）→ 多項式変異 を GPU があれば GPU 上で実行して numpy で返す。"""
    if _HAS_CUPY:
        lb_g = cp.asarray(lb_s)
        ub_g = cp.asarray(ub_s)
        vals = _sbx_values_matrix(cp.asarray(x1), cp.asarray(x2), lb_g, ub_g, crossover_eta)
        vals = _poly_mut_matrix(vals, lb_g, ub_g, pm, mutation_eta)
        return cp.asnumpy(vals)
    vals = _sbx_values_matrix(x1, x2, lb_s, ub_s, crossover_eta)
    return _poly_mut_matrix(vals, lb_s, ub_s, pm, mutation_eta)


# ── Convergence score & tournament (platEMO: calCon / TournamentSelection) ───

def _cal_con(population: List[FloatSolution]) -> np.ndarray:
    """platEMO calCon: frontNo*(sum_range) + sum_obj  (lower = better).

    Objectives are min-max normalised per objective before summing so that a
    large-scale objective (e.g. EDD) cannot dominate the tie-break just because
    of its raw magnitude relative to the others (e.g. MELD/PQ).
    """
    ranking = NumpyNonDominatedRanking()
    ranking.compute_ranking(population)
    front_nums = np.array(
        [s.attributes["dominance_ranking"] + 1 for s in population], dtype=float
    )
    obj = np.array([s.objectives for s in population], dtype=np.float64)
    obj_min = obj.min(axis=0)
    obj_max = obj.max(axis=0)
    obj_span = np.where(obj_max - obj_min > 1e-12, obj_max - obj_min, 1.0)
    sum_obj = ((obj - obj_min) / obj_span).sum(axis=1)
    rng = float(sum_obj.max() - sum_obj.min())
    return front_nums * rng + sum_obj


def _tournament_select_indices(con: np.ndarray, n: int) -> np.ndarray:
    """Binary tournament on calCon (lower wins). Returns n winner indices."""
    N = len(con)
    a = np.random.randint(N, size=n)
    b = np.random.randint(N, size=n)
    return np.where(con[a] <= con[b], a, b)


# ── Environmental selection helpers ──────────────────────────────────────────

def _lmea_crowding_distances(solutions: List[FloatSolution]) -> np.ndarray:
    N = len(solutions)
    if N <= 2:
        return np.full(N, np.inf)
    obj = np.array([s.objectives for s in solutions], dtype=np.float64)
    M = obj.shape[1]
    dist = np.zeros(N)
    for m in range(M):
        order = np.argsort(obj[:, m])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        rng = obj[order[-1], m] - obj[order[0], m]
        if rng < 1e-10:
            continue
        for i in range(1, N - 1):
            dist[order[i]] += (obj[order[i + 1], m] - obj[order[i - 1], m]) / rng
    return dist


def _lmea_environmental_select(
    combined: List[FloatSolution],
    ranked_sublists: List[List[FloatSolution]],
    N: int,
) -> List[FloatSolution]:
    """NSGA-II-style: fill fronts, crowding distance for the last front."""
    selected: List[FloatSolution] = []
    for front in ranked_sublists:
        if len(selected) + len(front) <= N:
            selected.extend(front)
        else:
            needed = N - len(selected)
            cd = _lmea_crowding_distances(front)
            order = np.argsort(-cd)
            selected.extend([front[int(k)] for k in order[:needed]])
            break
    return selected[:N]


# ── VariableClustering (platEMO-faithful) ────────────────────────────────────

def variable_clustering(
    problem: FloatProblem,
    population: List[FloatSolution],
    nSel: int = 5,
    nPer: int = 50,
    eval_batch_rows: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """Divide decision variables into CV (convergence) and DV (diversity).

    Follows platEMO VariableClustering:
    - Normalises using non-dominated front min/max
    - Computes RMSE of line fit and angle for each (variable, solution) pair
    - k-means on D×nSel angle matrix
    - CV candidates = mean RMSE < 1e-2, refined by angle cluster

    All D×nSel×nPer perturbed solutions are evaluated in batches of
    ``eval_batch_rows`` decision vectors via ``_evaluate_decs`` (single GPU
    pass per batch when the problem provides ``evaluate_batch``), and the
    per-(variable, solution) line fits are done with one stacked SVD.

    Returns (DV, CV) index arrays.
    """
    D = problem.number_of_variables()
    M = problem.number_of_objectives()
    lb = np.asarray(population[0].lower_bound, dtype=float)
    ub = np.asarray(population[0].upper_bound, dtype=float)
    n_pop = len(population)

    # Non-dominated front range for normalisation
    ranking = NumpyNonDominatedRanking()
    ranking.compute_ranking(population)
    obj_all = np.array([s.objectives for s in population], dtype=np.float64)
    nd_mask = np.array([s.attributes["dominance_ranking"] == 0 for s in population])
    nd_obj = obj_all[nd_mask] if nd_mask.any() else obj_all
    fmin = nd_obj.min(axis=0)
    fmax = nd_obj.max(axis=0)
    frange = np.where(fmax - fmin > 0, fmax - fmin, 1.0)

    # Same nSel solutions for all variables (platEMO: Sample = randi(N,1,nSel))
    nSel_eff = min(nSel, n_pop)
    sel_idx = np.random.choice(n_pop, size=nSel_eff, replace=False)
    base_decs = np.array(
        [np.asarray(population[s].variables, dtype=float) for s in sel_idx]
    )  # (nSel, D)

    angles = np.full((D, nSel_eff), 90.0)
    rmse   = np.zeros((D, nSel_eff))
    normal = np.ones(M)  # hyperplane f1+...+fM=1 normal direction
    normal_len = np.linalg.norm(normal) + 1e-12

    rows_per_var = nSel_eff * nPer
    vars_per_chunk = max(1, eval_batch_rows // rows_per_var)
    # 摂動 decs は GPU 上で構築する（基準解と1変数しか違わない全長ベクトルを
    # CPU で実体化して PCIe 転送するコストを丸ごと回避）。乱数は従来どおり
    # np.random から同順で引くので、同一シードなら CPU 構築と一致する。
    use_gpu = _HAS_CUPY and getattr(problem, "evaluate_batch", None) is not None
    xp = cp if use_gpu else np
    base_x = xp.asarray(base_decs)
    for start in range(0, D, vars_per_chunk):
        var_idx = np.arange(start, min(start + vars_per_chunk, D))
        n_var = len(var_idx)

        # (n_var, nSel, nPer, D): copies of the sampled solutions with one
        # variable resampled uniformly per perturbation
        decs = xp.broadcast_to(
            base_x[None, :, None, :], (n_var, nSel_eff, nPer, D)
        ).copy()
        for c, i in enumerate(var_idx):
            decs[c, :, :, i] = xp.asarray(
                np.random.uniform(lb[i], ub[i], size=(nSel_eff, nPer))
            )

        objs = _evaluate_decs(problem, decs.reshape(-1, D), lb, ub)
        SP_norm = (objs.reshape(n_var, nSel_eff, nPer, M) - fmin) / frange
        SP_c = SP_norm - SP_norm.mean(axis=2, keepdims=True)

        flat = SP_c.reshape(n_var * nSel_eff, nPer, M)
        nonzero = ~np.all(np.abs(flat) <= 1e-8, axis=(1, 2))
        rmse_flat = np.zeros(n_var * nSel_eff)
        ang_flat = np.full(n_var * nSel_eff, 90.0)
        if nonzero.any():
            try:
                _, _, Vt = np.linalg.svd(flat[nonzero], full_matrices=False)
                direction = Vt[:, 0, :]
                direction = direction / (
                    np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12
                )
                # RMSE = sqrt(sum of squared perpendicular distances)
                proj = np.einsum("bpm,bm->bp", flat[nonzero], direction)
                residuals = flat[nonzero] - proj[:, :, None] * direction[:, None, :]
                rmse_flat[nonzero] = np.sqrt((residuals ** 2).sum(axis=(1, 2)))
                # Angle between fitted direction and hyperplane normal (degrees)
                cos_a = np.abs(direction @ normal) / normal_len
                ang_flat[nonzero] = (
                    np.arccos(np.clip(cos_a, 0.0, 1.0)) / np.pi * 180.0
                )
            except np.linalg.LinAlgError:
                pass  # keep angle=90, rmse=0 (matches the per-pair fallback)
        rmse[var_idx] = rmse_flat.reshape(n_var, nSel_eff)
        angles[var_idx] = ang_flat.reshape(n_var, nSel_eff)

        done = var_idx[-1] + 1
        if done % max(1, D // 10) < vars_per_chunk or done == D:
            print(f"  clustering: {done}/{D} vars, evals={problem.evaluation}")

    # CV candidates: mean RMSE < 1e-2 (platEMO: VariableKind = mean(RMSE,2)<1e-2)
    variable_kind = rmse.mean(axis=1) < 1e-2  # shape (D,), True = CV candidate

    # k-means on D×nSel angle matrix (each variable = nSel-dim point)
    km = KMeans(n_clusters=2, n_init=10, random_state=0)
    result = km.fit_predict(angles)  # cluster label per variable

    # Intersection logic: if CV candidates span both clusters, keep smaller-angle cluster
    cv_in_0 = variable_kind & (result == 0)
    cv_in_1 = variable_kind & (result == 1)
    if cv_in_0.any() and cv_in_1.any():
        if angles[cv_in_0].mean() <= angles[cv_in_1].mean():
            variable_kind = variable_kind & (result == 0)
        else:
            variable_kind = variable_kind & (result == 1)

    CV = np.where(variable_kind)[0]
    DV = np.where(~variable_kind)[0]
    return DV, CV


# ── CorrelationAnalysis (platEMO-faithful) ───────────────────────────────────

def interaction_analysis(
    problem: FloatProblem,
    population: List[FloatSolution],
    CV: np.ndarray,
    nCor: int = 5,
    batch_tests: int = 256,
    wave_start: Optional[int] = None,
) -> List[np.ndarray]:
    """Group CV variables into interacting subgroups.

    Follows platEMO CorrelationAnalysis:
    - 3 new evaluations per test (reuses sol.objectives as f(a1,b1))
    - Bidirectional condition: any(delta1 * delta2 < 0)

    Tests are evaluated in waves (3 decision vectors per test) via
    ``_evaluate_decs``. A group's remaining tests are skipped once any test
    in an earlier wave hit, so compared with the sequential early-exit
    version this evaluates at most one extra wave per group.

    ``wave_start`` を指定すると最初のウェーブを小さく始めて ×4 ずつ
    ``batch_tests`` まで広げる（適応ウェーブ）。本問題のように最初の
    数テストでヒットする（全変数が相互作用する）場合、固定幅 256 で
    走らせるより1変数あたりの無駄な評価が桁違いに減る。判定は「いずれ
    かのテストがヒットしたか」なので、ウェーブ幅は結果の分布を変えず、
    platEMO の逐次早期打ち切りにより近い動作になる。
    ``wave_start=None`` は従来どおり固定幅 ``batch_tests``。
    """
    lb = np.asarray(population[0].lower_bound, dtype=float)
    ub = np.asarray(population[0].upper_bound, dtype=float)
    n_pop = len(population)
    pop_decs = np.array(
        [np.asarray(s.variables, dtype=float) for s in population]
    )
    pop_objs = np.array([s.objectives for s in population], dtype=float)
    # テスト用 decs（1テスト=3行の全長ベクトル）は GPU 上で構築する。
    # 母集団は analysis 中に更新されないので一度だけ転送すればよい。
    use_gpu = _HAS_CUPY and getattr(problem, "evaluate_batch", None) is not None
    xp = cp if use_gpu else np
    pop_decs_x = xp.asarray(pop_decs)

    subCVs: List[List[int]] = []
    for count, v in enumerate(CV, start=1):
        v = int(v)

        # Flat test list in platEMO order: per group, per member u, nCor trials
        test_group: List[int] = []
        test_u: List[int] = []
        for g_idx, group in enumerate(subCVs):
            for u in group:
                test_group.extend([g_idx] * nCor)
                test_u.extend([int(u)] * nCor)
        test_group_arr = np.asarray(test_group, dtype=int)
        test_u_arr = np.asarray(test_u, dtype=int)

        cor_group_indices: set[int] = set()
        n_groups = len(subCVs)
        total_tests = len(test_u_arr)
        pos = 0
        wave = batch_tests if wave_start is None else min(wave_start, batch_tests)
        while pos < total_tests:
            if len(cor_group_indices) == n_groups:
                break
            sel = slice(pos, pos + wave)
            pos += wave
            if wave_start is not None:
                wave = min(wave * 4, batch_tests)
            groups = test_group_arr[sel]
            keep = ~np.isin(groups, list(cor_group_indices))
            if not keep.any():
                continue
            groups = groups[keep]
            us = test_u_arr[sel][keep]
            n_t = len(us)

            sols = np.random.randint(n_pop, size=n_t)
            a2 = np.random.uniform(lb[v], ub[v], size=n_t)
            b2 = np.random.uniform(lb[us], ub[us])

            # 3 evaluations per test; f(a1,b1) reuses the stored objectives
            # rows: a2b1, a1b2, a2b2（GPU 上で構築）
            decs = xp.repeat(pop_decs_x[xp.asarray(sols)], 3, axis=0)
            rows = xp.asarray(np.arange(n_t) * 3)
            us_x = xp.asarray(us)
            a2_x = xp.asarray(a2)
            b2_x = xp.asarray(b2)
            decs[rows, v] = a2_x
            decs[rows + 1, us_x] = b2_x
            decs[rows + 2, v] = a2_x
            decs[rows + 2, us_x] = b2_x

            objs = _evaluate_decs(problem, decs, lb, ub).reshape(n_t, 3, -1)
            delta1 = objs[:, 0] - pop_objs[sols]        # f(a2,b1) − f(a1,b1)
            delta2 = objs[:, 2] - objs[:, 1]            # f(a2,b2) − f(a1,b2)
            hit = np.any(delta1 * delta2 < 0, axis=1)
            cor_group_indices.update(int(g) for g in groups[hit])

        if count % 100 == 0:
            print(
                f"  interaction: {count}/{len(CV)} vars, "
                f"{len(subCVs)} groups, evals={problem.evaluation}"
            )

        if not cor_group_indices:
            subCVs.append([v])
        else:
            merged: List[int] = [v]
            new_subCVs: List[List[int]] = []
            for g_idx, group in enumerate(subCVs):
                if g_idx in cor_group_indices:
                    merged.extend(group)
                else:
                    new_subCVs.append(group)
            new_subCVs.append(merged)
            subCVs = new_subCVs

    return [np.array(g) for g in subCVs]


# ── ConvergenceOptimization (platEMO-faithful) ───────────────────────────────

def convergence_optimization(
    problem: FloatProblem,
    population: List[FloatSolution],
    subCVs: List[np.ndarray],
    crossover_eta: float = 20.0,
    mutation_probability: Optional[float] = None,
    mutation_eta: float = 20.0,
    should_stop: Optional[Callable[[], bool]] = None,
    on_round: Optional[Callable[[], None]] = None,
) -> List[FloatSolution]:
    """platEMO ConvergenceOptimization.

    For each subgroup (repeated len(subgroup) times):
      - Tournament selection on calCon → 2N parents
      - SBX between p1/p2 pairs for subgroup vars, base = population[k]
      - Pairwise replacement if offspring calCon improves

    One inner round costs N evaluations; with a large subgroup a single call
    can dwarf the whole evaluation budget, so `should_stop` is checked before
    every round and `on_round` (e.g. observer notification) after every round.
    """
    N = len(population)
    D = problem.number_of_variables()
    lb = np.asarray(population[0].lower_bound)
    ub = np.asarray(population[0].upper_bound)
    pm = mutation_probability if mutation_probability is not None else 1.0 / D

    con = _cal_con(population)
    pop_decs = np.array([s.variables for s in population], dtype=float)

    for group in subCVs:
        if len(group) == 0:
            continue
        lb_g = lb[group]
        ub_g = ub[group]
        for _ in range(len(group)):  # platEMO: inner loop runs len(group) times
            if should_stop is not None and should_stop():
                return population
            pool = _tournament_select_indices(con, 2 * N)
            p1_decs = pop_decs[pool[:N]]
            p2_decs = pop_decs[pool[N:]]

            off_decs = pop_decs.copy()
            off_decs[:, group] = _sbx_mut_values(
                p1_decs[:, group], p2_decs[:, group],
                lb_g, ub_g, crossover_eta, pm, mutation_eta,
            )

            off_objs = _evaluate_decs(problem, off_decs, lb, ub)
            offspring: List[FloatSolution] = []
            for k in range(N):
                child = _lmea_copy_solution(population[k])
                child.variables = off_decs[k].copy()
                child.objectives = list(off_objs[k])
                offspring.append(child)

            # calCon on combined [P; offspring] for consistent front ranking
            all_con = _cal_con(population + offspring)
            par_con = all_con[:N]
            off_con = all_con[N:]

            for k in range(N):
                if off_con[k] < par_con[k]:
                    population[k] = offspring[k]
                    pop_decs[k] = off_decs[k]
                    con[k] = off_con[k]
                else:
                    con[k] = par_con[k]

            if on_round is not None:
                on_round()

    return population


# ── DistributionOptimization (platEMO-faithful) ──────────────────────────────

def diversity_optimization(
    problem: FloatProblem,
    population: List[FloatSolution],
    DV: np.ndarray,
    crossover_eta: float = 20.0,
    mutation_probability: Optional[float] = None,
    mutation_eta: float = 20.0,
) -> List[FloatSolution]:
    """platEMO DistributionOptimization.

    Base solutions: tournament-selected by calCon.
    DV variables: replaced by SBX of random parent pairs.
    Environmental selection: angle-based (as in platEMO EnvironmentalSelection).
    """
    N = len(population)
    D = problem.number_of_variables()
    lb = np.asarray(population[0].lower_bound)
    ub = np.asarray(population[0].upper_bound)
    pm = mutation_probability if mutation_probability is not None else 1.0 / D

    if len(DV) == 0:
        return population

    con = _cal_con(population)
    base_idx = _tournament_select_indices(con, N)
    pop_decs = np.array([s.variables for s in population], dtype=float)
    off_decs = pop_decs[base_idx].copy()

    rand_idx1 = np.random.randint(N, size=N)
    rand_idx2 = np.random.randint(N, size=N)
    lb_dv = lb[DV]
    ub_dv = ub[DV]
    off_decs[:, DV] = _sbx_mut_values(
        pop_decs[rand_idx1][:, DV], pop_decs[rand_idx2][:, DV],
        lb_dv, ub_dv, crossover_eta, pm, mutation_eta,
    )

    off_objs = _evaluate_decs(problem, off_decs, lb, ub)
    offspring: List[FloatSolution] = []
    for k in range(N):
        child = _lmea_copy_solution(population[base_idx[k]])
        child.variables = off_decs[k].copy()
        child.objectives = list(off_objs[k])
        offspring.append(child)

    # Angle-based environmental selection (platEMO EnvironmentalSelection)
    combined = population + offspring
    ranking = NumpyNonDominatedRanking()
    ranking.compute_ranking(combined)
    fronts = ranking.ranked_sublists

    Q: List[FloatSolution] = []
    last_front: List[FloatSolution] = []
    for front in fronts:
        if len(Q) + len(front) <= N:
            Q.extend(front)
        else:
            last_front = list(front)
            break
    else:
        return Q[:N]

    if not Q and last_front:
        idx = np.random.randint(len(last_front))
        Q.append(last_front.pop(idx))

    while len(Q) < N and last_front:
        obj_Q = np.array([s.objectives for s in Q], dtype=np.float64)
        obj_c = np.array([s.objectives for s in last_front], dtype=np.float64)
        norm_Q = np.linalg.norm(obj_Q, axis=1, keepdims=True) + 1e-12
        norm_c = np.linalg.norm(obj_c, axis=1, keepdims=True) + 1e-12
        cos_mat = np.clip((obj_c @ obj_Q.T) / (norm_c * norm_Q.T), -1.0, 1.0)
        angle_mat = np.arccos(cos_mat)
        min_angles = angle_mat.min(axis=1)
        best = int(np.argmax(min_angles))
        Q.append(last_front[best])
        last_front.pop(best)

    return Q[:N]


# ── Main algorithm class ──────────────────────────────────────────────────────

class LMEA(Algorithm[FloatSolution, List[FloatSolution]]):
    """Large-scale Many-Objective Evolutionary Algorithm (platEMO-faithful).

    Main loop: 10 × ConvergenceOptimization → M × DistributionOptimization.
    Default params match platEMO: nSel=5, nPer=50, nCor=5.
    Inherits jMetal Algorithm so .observable is available for observer registration.
    """

    def __init__(
        self,
        problem: FloatProblem,
        population_size: int,
        max_evaluations: int,
        nSel: int = 5,
        nPer: int = 50,
        nCor: int = 5,
        crossover: Optional[Crossover] = None,
        mutation: Optional[Mutation] = None,
        crossover_distribution_index: float = 20.0,
        mutation_probability: Optional[float] = None,
        mutation_distribution_index: float = 20.0,
        analyse_interactions: bool = True,
        interaction_wave_start: Optional[int] = 16,
        dva_cache_path: Optional[Union[str, Path]] = None,
        termination_criterion: Optional[TerminationCriterion] = None,
    ):
        super().__init__()
        self.problem = problem
        self.population_size = population_size
        self.max_evaluations = max_evaluations
        self.nSel = nSel
        self.nPer = nPer
        self.nCor = nCor
        # Additional stop condition checked alongside the evaluation budget
        # (e.g. HV stagnation). Registered as observer so it receives
        # EVALUATIONS/SOLUTIONS on every update_progress call.
        self.termination_criterion = termination_criterion
        if termination_criterion is not None:
            self.observable.register(termination_criterion)
        # When False, skip the O(|CV|²) pairwise interaction analysis and put
        # every convergence variable into a single subgroup. Use this when the
        # objectives couple all variables anyway, so the analysis can only
        # return one group and its evaluation cost buys nothing.
        self.analyse_interactions = analyse_interactions
        # 相互作用分析の適応ウェーブ初期幅（None で従来の固定幅 256）
        self.interaction_wave_start = interaction_wave_start
        # DV/CV/subCVs はグラフ（と目的関数設定）にのみ依存するので、
        # パスを与えると初回実行の結果を保存し以降の試行で再利用する。
        self.dva_cache_path = Path(dva_cache_path) if dva_cache_path is not None else None
        self.dva_cache_status = "disabled"
        # Operator objects take priority; fall back to eta/probability params
        self.crossover_eta = (
            crossover.distribution_index if crossover is not None
            else crossover_distribution_index
        )
        self.mutation_eta = (
            mutation.distribution_index if mutation is not None
            else mutation_distribution_index
        )
        self._mutation_probability = (
            mutation.probability if mutation is not None
            else mutation_probability
        )
        # Internal state populated during init_progress
        self._DV: Optional[np.ndarray] = None
        self._CV: Optional[np.ndarray] = None
        self._subCVs: Optional[List[np.ndarray]] = None
        self._M: int = 0
        self._pm: float = 0.0
        # Evaluations spent on variable clustering + interaction analysis are
        # excluded from the search budget (max_evaluations governs only the
        # convergence/diversity optimisation loop below).
        self._preprocessing_evaluations: int = 0

    # ------------------------------------------------------------------
    # jMetal Algorithm abstract interface
    # ------------------------------------------------------------------

    def create_initial_solutions(self) -> List[FloatSolution]:
        return [self.problem.create_solution() for _ in range(self.population_size)]

    def evaluate(self, solution_list: List[FloatSolution]) -> List[FloatSolution]:
        return _evaluate_solutions(self.problem, solution_list)

    def init_progress(self) -> None:
        self._M = self.problem.number_of_objectives()
        self._pm = (
            self._mutation_probability
            if self._mutation_probability is not None
            else 1.0 / self.problem.number_of_variables()
        )
        pre_start = self.problem.evaluation
        if self._load_dva_cache():
            print(f"LMEA: DVA cache hit → {self.dva_cache_path}")
        else:
            print("LMEA: variable clustering...")
            self._DV, CV = variable_clustering(self.problem, self.solutions, self.nSel, self.nPer)
            self._CV = CV
            print(f"  → CV: {len(CV)} vars, DV: {len(self._DV)} vars")
            if self.analyse_interactions:
                print("LMEA: interaction analysis (may take a while)...")
                self._subCVs = interaction_analysis(
                    self.problem, self.solutions, CV, self.nCor,
                    wave_start=self.interaction_wave_start,
                )
            else:
                print("LMEA: interaction analysis skipped → single CV subgroup")
                self._subCVs = [CV.copy()] if len(CV) > 0 else []
            self._save_dva_cache()
        print(f"  → {len(self._subCVs)} subgroups in CV")
        self._preprocessing_evaluations = self.problem.evaluation - pre_start
        self.evaluations = self.problem.evaluation - self._preprocessing_evaluations

    def _load_dva_cache(self) -> bool:
        path = self.dva_cache_path
        if path is None or not path.exists():
            return False
        D = self.problem.number_of_variables()
        try:
            data = np.load(path)
            DV, CV = data["DV"], data["CV"]
            sizes, flat = data["subcv_sizes"], data["subcv_flat"]
        except Exception as exc:
            print(f"LMEA: DVA cache unreadable ({exc}) → recompute")
            return False
        if len(DV) + len(CV) != D or sizes.sum() != len(flat):
            print("LMEA: DVA cache shape mismatch → recompute")
            return False
        self._DV, self._CV = DV, CV
        bounds = np.cumsum(sizes)[:-1]
        self._subCVs = [g for g in np.split(flat, bounds) if len(g) > 0]
        self.dva_cache_status = "hit"
        return True

    def _save_dva_cache(self) -> None:
        path = self.dva_cache_path
        if path is None:
            return
        subCVs = self._subCVs or []
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            DV=np.asarray(self._DV, dtype=np.int64),
            CV=np.asarray(self._CV, dtype=np.int64),
            subcv_sizes=np.array([len(g) for g in subCVs], dtype=np.int64),
            subcv_flat=(
                np.concatenate([np.asarray(g, dtype=np.int64) for g in subCVs])
                if subCVs else np.empty(0, dtype=np.int64)
            ),
        )
        self.dva_cache_status = "saved"

    def stopping_condition_is_met(self) -> bool:
        if (self.problem.evaluation - self._preprocessing_evaluations) >= self.max_evaluations:
            return True
        return self.termination_criterion is not None and self.termination_criterion.is_met

    def step(self) -> None:
        for _ in range(10):
            if self.stopping_condition_is_met():
                break
            self.solutions = convergence_optimization(
                self.problem, self.solutions, self._subCVs,
                self.crossover_eta, self._pm, self.mutation_eta,
                should_stop=self.stopping_condition_is_met,
                on_round=self.update_progress,
            )
        for _ in range(self._M):
            if self.stopping_condition_is_met():
                break
            self.solutions = diversity_optimization(
                self.problem, self.solutions, self._DV,
                self.crossover_eta, self._pm, self.mutation_eta,
            )
            self.update_progress()

    def update_progress(self) -> None:
        self.evaluations = self.problem.evaluation - self._preprocessing_evaluations
        self.observable.notify_all(**self.observable_data())

    def observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }

    def result(self) -> List[FloatSolution]:
        return self.solutions

    def get_name(self) -> str:
        return "LMEA"


# ─── MOEA/DVA ─────────────────────────────────────────────────────────────────
#
# Ma et al., "A Multiobjective Evolutionary Algorithm Based on Decision
# Variable Analyses for Multiobjective Optimization Problems With
# Large-Scale Variables," IEEE TEC, 2016.
#
# Algorithms implemented:
#   moeadva_cva           (Algorithm 1 — Control Variable Analysis)
#   moeadva_ida           (Algorithm 2/3 — Interdependence Analysis)
#   moeadva_subcomponent  (Algorithm 5 — Subcomponent Optimizer)
#   moeadva_phase2        (Phase 2 — MOEA/D Tchebycheff)
#   MOEADVA               (main class)

def _moeadva_new_solution(problem: FloatProblem, dec: np.ndarray) -> FloatSolution:
    sol = FloatSolution(
        problem.lower_bound, problem.upper_bound, problem.number_of_objectives()
    )
    sol.variables = np.asarray(dec, dtype=float)
    return problem.evaluate(sol)


def _moeadva_solution_from(
    problem: FloatProblem, dec: np.ndarray, objectives: np.ndarray, copy: bool = True
) -> FloatSolution:
    """Wrap an already-evaluated decision vector without re-evaluating.

    ``copy=False`` stores ``dec`` itself (a view when ``dec`` is a row of a
    larger matrix) — only for callers that materialise the survivors
    themselves before the backing matrix should be released (LSMOF's
    weight-optimisation archive); see the paragraph below for why views are
    otherwise a memory hazard.

    Bypasses ``FloatSolution.__init__`` (→ ``Solution.__init__``), which builds
    ``self.variables = [[] for _ in range(D)]`` — a D-sized list of throwaway
    empty lists — purely to be overwritten by the ``np.asarray`` line below.
    At D in the thousands and tens of thousands of calls per run (this helper
    is the main per-individual solution constructor for CCGDE3 and LSMOF),
    that list comprehension dominates wall-clock time (profiled at ~75% of
    total runtime). ``object.__new__`` + manual field assignment is the same
    fast-construction pattern already used by ``_lmea_copy_solution`` and
    LMOCSO's ``_lmocso_copy`` elsewhere in this module.

    ``variables`` is an explicit **copy** (``np.array``, not ``np.asarray``).
    Callers pass ``dec`` as one row of a whole-batch ``(N, D)`` matrix; a view
    would keep that entire matrix alive for as long as any single solution
    built from it survives in the population. With GDE3/NSGA-style elitism a
    few survivors from every round persist for many rounds, so each round's
    278 MB batch (usmig, D≈87k, N=400) stayed pinned and the process grew to
    ~13 GB before the OOM killer stopped it. A copy costs ~D·8 bytes per
    solution and lets each batch buffer be freed at the end of its round.
    """
    sol = object.__new__(FloatSolution)
    sol.lower_bound = problem.lower_bound
    sol.upper_bound = problem.upper_bound
    sol.variables = np.array(dec, dtype=float) if copy else np.asarray(dec, dtype=float)
    sol.objectives = list(objectives)
    sol.constraints = []
    sol.attributes = {}
    return sol


def _moeadva_count_fronts(obj_matrix: np.ndarray) -> int:
    """NDSort: count number of non-dominated fronts."""
    n = obj_matrix.shape[0]
    diff = obj_matrix[:, None, :] - obj_matrix[None, :, :]
    dominates = (diff <= 0).all(axis=2) & (diff < 0).any(axis=2)
    remaining = list(range(n))
    n_fronts = 0
    while remaining:
        r = np.array(remaining)
        sub = dominates[np.ix_(r, r)]
        dom_count = sub.sum(axis=0)
        current = r[dom_count == 0]
        for idx in current:
            remaining.remove(idx)
        n_fronts += 1
    return n_fronts


def moeadva_cva(
    problem: FloatProblem,
    nca: int,
    eval_batch_rows: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """Algorithm 1: Control Variable Analysis.

    Classify each variable as DiverIndex (position) or ConverIndex (distance).
    n_fronts == NCA → complete dominance chain → ConverIndex; else DiverIndex.

    The nca sample solutions of many variables are evaluated together in
    batches of ``eval_batch_rows`` decision vectors via ``_evaluate_decs``.
    """
    D = problem.number_of_variables()
    M = problem.number_of_objectives()
    lb = np.array(problem.lower_bound)
    ub = np.array(problem.upper_bound)
    diver: List[int] = []
    conver: List[int] = []

    vars_per_chunk = max(1, eval_batch_rows // nca)
    for start in range(0, D, vars_per_chunk):
        var_idx = range(start, min(start + vars_per_chunk, D))
        decs = np.empty((len(var_idx), nca, D))
        for c, i in enumerate(var_idx):
            x = np.random.uniform(lb, ub)
            decs[c] = np.tile(x, (nca, 1))
            spacing = (ub[i] - lb[i]) / nca
            decs[c, :, i] = np.clip(
                np.arange(nca) * spacing + lb[i] + np.random.uniform(0, spacing, nca),
                lb[i], ub[i],
            )

        objs = _evaluate_decs(problem, decs.reshape(-1, D), lb, ub)
        objs = objs.reshape(len(var_idx), nca, M)
        for c, i in enumerate(var_idx):
            if _moeadva_count_fronts(objs[c]) == nca:
                conver.append(i)
            else:
                diver.append(i)

        done = var_idx[-1] + 1
        if done % max(1, D // 10) < vars_per_chunk or done == D:
            print(f"  CVA: {done}/{D}  diver={len(diver)} conver={len(conver)} "
                  f"evals={problem.evaluations}")

    return np.array(diver, dtype=int), np.array(conver, dtype=int)


def moeadva_ida(
    problem: FloatProblem,
    population: List[FloatSolution],
    nia: int,
    conver_idx: np.ndarray,
    pair_chunk: int = 1024,
) -> List[np.ndarray]:
    """Interdependence Analysis over ALL D*(D-1)/2 variable pairs (PlatEMO style).

    Sign-based criterion: Δᵢf|_{xⱼ=b1} · Δᵢf|_{xⱼ=b2} < 0 for any objective.
    Population is updated in-place when a strictly dominating solution is found.
    Only ConverIndex variables are partitioned into subcomponents.

    Pairs are processed in waves of ``pair_chunk``: each wave evaluates one
    trial (3 decision vectors) for every still-undecided pair in the chunk via
    ``_evaluate_decs``, so the per-pair early exit is kept at trial
    granularity. Base vectors come from the population state at wave start;
    dominating replacements are applied between waves.
    """
    D = problem.number_of_variables()
    C = len(conver_idx)
    if C == 0:
        return []

    lb = np.array(problem.lower_bound)
    ub = np.array(problem.upper_bound)
    N = len(population)
    M = problem.number_of_objectives()
    conver_set = set(conver_idx.tolist())

    pop_decs = np.array([np.asarray(s.variables, dtype=float) for s in population])
    pop_objs = np.array([s.objectives for s in population], dtype=float)

    interact = np.zeros((D, D), dtype=bool)
    np.fill_diagonal(interact, True)

    total_pairs = D * (D - 1) // 2
    done_pairs = 0
    report_every = max(1, total_pairs // 20)

    def _replace(idx: int, dec: np.ndarray, objectives: np.ndarray) -> None:
        pop_decs[idx] = dec
        pop_objs[idx] = objectives
        population[idx] = _moeadva_solution_from(problem, dec.copy(), objectives)

    def _process(pair_i: np.ndarray, pair_j: np.ndarray) -> None:
        nonlocal done_pairs
        active_i, active_j = pair_i, pair_j
        for _ in range(nia):
            n_t = len(active_i)
            if n_t == 0:
                break
            xl_idx = np.random.randint(N, size=n_t)
            a2 = np.random.uniform(lb[active_i], ub[active_i])
            b2 = np.random.uniform(lb[active_j], ub[active_j])
            xl_obj = pop_objs[xl_idx]

            # rows per test: xi (var i resampled), xj, xij
            decs = np.repeat(pop_decs[xl_idx], 3, axis=0)
            r0 = np.arange(n_t) * 3
            decs[r0, active_i] = a2
            decs[r0 + 1, active_j] = b2
            decs[r0 + 2, active_i] = a2
            decs[r0 + 2, active_j] = b2

            objs = _evaluate_decs(problem, decs, lb, ub).reshape(n_t, 3, M)
            fi, fj, fij = objs[:, 0], objs[:, 1], objs[:, 2]

            hit = np.any((fi - xl_obj) * (fij - fj) < 0, axis=1)
            interact[active_i[hit], active_j[hit]] = True
            interact[active_j[hit], active_i[hit]] = True

            # Update population when strictly better solution found
            # (ConverIndex only). Dominating the current objectives implies
            # dominating the wave-start snapshot, so pre-filter on the
            # snapshot and resolve the few candidates sequentially.
            cand = (
                np.all(fj <= xl_obj, axis=1) & np.any(fj < xl_obj, axis=1)
                | (np.all(fi <= xl_obj, axis=1) & np.any(fi < xl_obj, axis=1))
                | (np.all(fij <= xl_obj, axis=1) & np.any(fij < xl_obj, axis=1))
            )
            for k in np.where(cand)[0]:
                idx = int(xl_idx[k])
                cur = pop_objs[idx]
                if (active_j[k] in conver_set
                        and np.all(fj[k] <= cur) and np.any(fj[k] < cur)):
                    _replace(idx, decs[r0[k] + 1], fj[k])
                    cur = pop_objs[idx]
                if (active_i[k] in conver_set
                        and np.all(fi[k] <= cur) and np.any(fi[k] < cur)):
                    _replace(idx, decs[r0[k]], fi[k])
                    cur = pop_objs[idx]
                if (active_i[k] in conver_set and active_j[k] in conver_set
                        and np.all(fij[k] <= cur) and np.any(fij[k] < cur)):
                    _replace(idx, decs[r0[k] + 2], fij[k])

            keep = ~hit
            active_i, active_j = active_i[keep], active_j[keep]

        prev = done_pairs
        done_pairs += len(pair_i)
        if done_pairs // report_every != prev // report_every:
            print(f"  IDA: {done_pairs}/{total_pairs} pairs, "
                  f"evals={problem.evaluations}")

    buf_i: List[np.ndarray] = []
    buf_j: List[np.ndarray] = []
    buffered = 0
    for i in range(D - 1):
        js = np.arange(i + 1, D)
        buf_i.append(np.full(len(js), i))
        buf_j.append(js)
        buffered += len(js)
        while buffered >= pair_chunk:
            pi = np.concatenate(buf_i)
            pj = np.concatenate(buf_j)
            _process(pi[:pair_chunk], pj[:pair_chunk])
            buf_i, buf_j = [pi[pair_chunk:]], [pj[pair_chunk:]]
            buffered -= pair_chunk
    if buffered:
        _process(np.concatenate(buf_i), np.concatenate(buf_j))

    # BFS to build subcomponents from ConverIndex variables only
    visited = {v: False for v in conver_idx.tolist()}
    subcomponents: List[np.ndarray] = []
    for start in conver_idx.tolist():
        if not visited[start]:
            component: List[int] = []
            queue = [start]
            visited[start] = True
            while queue:
                node = queue.pop(0)
                component.append(node)
                for nb in conver_idx.tolist():
                    if not visited[nb] and interact[node, nb]:
                        visited[nb] = True
                        queue.append(nb)
            subcomponents.append(np.array(component, dtype=int))

    return subcomponents


def _moeadva_lhc_population(
    problem: FloatProblem,
    n: int,
    diver_idx: np.ndarray,
    conver_idx: np.ndarray,
) -> List[FloatSolution]:
    """Initial population following PlatEMO MOEA-DVA convention.

    DiverIndex == 1 : evenly spaced (guarantees PF coverage)
    DiverIndex 2-4  : Latin Hypercube
    DiverIndex > 4  : random uniform
    ConverIndex     : random uniform
    """
    D = problem.number_of_variables()
    lb = np.array(problem.lower_bound)
    ub = np.array(problem.upper_bound)
    decs = np.random.uniform(lb, ub, (n, D))
    nd = len(diver_idx)
    if nd == 1:
        var = diver_idx[0]
        decs[:, var] = lb[var] + np.arange(n) / (n - 1) * (ub[var] - lb[var])
    elif nd <= 4:
        for var in diver_idx:
            perm = np.random.permutation(n)
            decs[:, var] = lb[var] + (perm + np.random.uniform(0, 1, n)) / n * (ub[var] - lb[var])
    objs = _evaluate_decs(problem, decs, lb, ub)
    return [
        _moeadva_solution_from(problem, decs[k], objs[k]) for k in range(n)
    ]


def _moeadva_neighbourhood(
    population: List[FloatSolution],
    diver_idx: np.ndarray,
    k: int,
) -> np.ndarray:
    """k-NN neighbourhood in DiverIndex subspace."""
    N = len(population)
    if len(diver_idx) == 0:
        return np.array([
            np.random.choice([j for j in range(N) if j != i], k, replace=False)
            for i in range(N)
        ])
    decs = np.array([np.asarray(s.variables)[diver_idx] for s in population])
    dist = cdist(decs, decs)
    np.fill_diagonal(dist, np.inf)
    return np.argsort(dist, axis=1)[:, :k]


def moeadva_subcomponent(
    population: List[FloatSolution],
    problem: FloatProblem,
    neighbour: np.ndarray,
    indices: np.ndarray,
    de_f: float,
    mutation_distribution_index: float = 20.0,
    mutation_probability: Optional[float] = None,
) -> List[FloatSolution]:
    """Subcomponent optimizer on variable subset `indices` (PlatEMO style).

    Mutation : DE/current-to-rand  donor = x + F*(r1 - r2)
    Crossover: binomial on `indices` with CR = D/|indices|/2
    Mutation : polynomial restricted to `indices` only
    Replacement: sum-of-objectives (greedy), objectives min-max normalised
    per-objective (computed once from the current population) so a large-scale
    objective (e.g. EDD) cannot dominate the greedy sum just because of its raw
    magnitude relative to the others (e.g. MELD/PQ).

    All N offspring are generated from the population snapshot at call entry
    and evaluated in one ``_evaluate_decs`` batch (matches PlatEMO's
    vectorised SubOptimize); replacement stays per-individual greedy.
    """
    N = len(population)
    D = problem.number_of_variables()
    k_nb = neighbour.shape[1]
    lb = np.array(problem.lower_bound)
    ub = np.array(problem.upper_bound)
    lb_s = lb[indices]
    ub_s = ub[indices]
    cr = min(1.0, D / len(indices) / 2)
    eta_m = mutation_distribution_index
    mut_pow = 1.0 / (eta_m + 1)
    pm = mutation_probability if mutation_probability is not None else 1.0 / D

    pop_obj = np.array([s.objectives for s in population], dtype=np.float64)
    obj_min = pop_obj.min(axis=0)
    obj_max = pop_obj.max(axis=0)
    obj_span = np.where(obj_max - obj_min > 1e-12, obj_max - obj_min, 1.0)

    pop_decs = np.array([np.asarray(s.variables, dtype=float) for s in population])
    off_decs = np.empty_like(pop_decs)

    for i in range(N):
        if np.random.random() < 0.9:
            p = neighbour[i, np.random.choice(k_nb, 2, replace=False)]
        else:
            p = np.random.choice(N, 2, replace=False)

        x = pop_decs[i].copy()

        # DE/current-to-rand: base is current individual x
        donor = x + de_f * (pop_decs[p[0]] - pop_decs[p[1]])

        # Binomial crossover on indices only
        mask = np.random.random(len(indices)) < cr
        mask[np.random.randint(len(indices))] = True
        new_dec = x.copy()
        new_dec[indices] = np.clip(
            np.where(mask, donor[indices], x[indices]), lb_s, ub_s
        )

        # Polynomial mutation restricted to indices only
        sub = new_dec[indices].copy()
        rnd_m = np.random.random(len(indices))
        do_mut = np.random.random(len(indices)) < pm
        if do_mut.any():
            dy = ub_s - lb_s
            safe_dy = np.where(dy > 0, dy, 1.0)
            d1 = np.where(dy > 0, (sub - lb_s) / safe_dy, 0.5)
            d2 = np.where(dy > 0, (ub_s - sub) / safe_dy, 0.5)
            v1 = 2 * rnd_m + (1 - 2 * rnd_m) * (1 - d1) ** (eta_m + 1)
            v2 = 2 * (1 - rnd_m) + 2 * (rnd_m - 0.5) * (1 - d2) ** (eta_m + 1)
            dq = np.where(rnd_m <= 0.5,
                          np.power(np.maximum(v1, 0), mut_pow) - 1,
                          1 - np.power(np.maximum(v2, 0), mut_pow))
            new_dec[indices] = np.where(do_mut, np.clip(sub + dq * dy, lb_s, ub_s), sub)

        off_decs[i] = new_dec

    off_objs = _evaluate_decs(problem, off_decs, lb, ub)
    off_ns = ((off_objs - obj_min) / obj_span).sum(axis=1)
    pop_ns = ((pop_obj - obj_min) / obj_span).sum(axis=1)
    for i in np.where(off_ns < pop_ns)[0]:
        population[i] = _moeadva_solution_from(problem, off_decs[i], off_objs[i])

    return population


class MOEADVA(Algorithm[FloatSolution, List[FloatSolution]]):
    """MOEA/DVA — Ma et al. IEEE TEC 2016 (PlatEMO-faithful implementation).

    Structure:
      1. CVA  — classify each variable as DiverIndex or ConverIndex
      2. Init — evenly-spaced population on DiverIndex; random on ConverIndex
      3. IDA  — detect interactions over all D*(D-1)/2 pairs; form subcomponents
      4. Loop — run SubcomponentOptimizer for ALL remaining evaluations

    No Phase 2 (MOEA/D): PlatEMO's reference implementation runs only the
    subcomponent loop until the evaluation budget is exhausted.
    Inherits jMetal Algorithm so .observable is available for observer registration.
    """

    def __init__(
        self,
        problem: FloatProblem,
        population_size: int,
        max_evaluations: int,
        nca: int = 20,
        nia: int = 6,
        de_f: float = 0.5,
        mutation: Optional[Mutation] = None,
        mutation_distribution_index: float = 20.0,
        mutation_probability: Optional[float] = None,
        termination_criterion: Optional[TerminationCriterion] = None,
    ):
        super().__init__()
        self.problem = problem
        self.population_size = population_size
        self.max_evaluations = max_evaluations
        self.nca = nca
        self.nia = nia
        self.de_f = de_f
        # Additional stop condition checked alongside the evaluation budget
        # (e.g. HV stagnation).
        self.termination_criterion = termination_criterion
        if termination_criterion is not None:
            self.observable.register(termination_criterion)
        # Mutation operator takes priority; fall back to explicit params
        self.mutation_eta = (
            mutation.distribution_index if mutation is not None
            else mutation_distribution_index
        )
        self._mutation_probability = (
            mutation.probability if mutation is not None
            else mutation_probability
        )
        # Internal state populated during init_progress
        self._diver_idx: Optional[np.ndarray] = None
        self._conver_idx: Optional[np.ndarray] = None
        self._neighbour: Optional[np.ndarray] = None
        self._subcomponents: Optional[List[np.ndarray]] = None
        # Evaluations spent on CVA + IDA are excluded from the search budget
        # (max_evaluations governs only the subcomponent-optimisation loop).
        self._preprocessing_evaluations: int = 0
        self._cva_evaluations: int = 0
        self._ida_evaluations: int = 0

    # ------------------------------------------------------------------
    # jMetal Algorithm abstract interface
    # ------------------------------------------------------------------

    def create_initial_solutions(self) -> List[FloatSolution]:
        # Population is created in init_progress (depends on CVA result)
        return []

    def evaluate(self, solution_list: List[FloatSolution]) -> List[FloatSolution]:
        # No-op: initial list is empty; real evaluation happens inside init_progress
        return solution_list

    def init_progress(self) -> None:
        N = self.population_size

        print(f"MOEADVA: CVA  D={self.problem.number_of_variables()}  NCA={self.nca}")
        cva_start = self.problem.evaluations
        self._diver_idx, self._conver_idx = moeadva_cva(self.problem, self.nca)
        cva_cost = self.problem.evaluations - cva_start
        print(f"  → diver={len(self._diver_idx)}  conver={len(self._conver_idx)}")

        print(f"MOEADVA: initial population  N={N}")
        self.solutions = _moeadva_lhc_population(
            self.problem, N, self._diver_idx, self._conver_idx
        )
        k_nb = max(2, int(np.ceil(N / 10)))
        self._neighbour = _moeadva_neighbourhood(self.solutions, self._diver_idx, k_nb)

        print(f"MOEADVA: IDA  D={self.problem.number_of_variables()}  NIA={self.nia}")
        ida_start = self.problem.evaluations
        self._subcomponents = moeadva_ida(
            self.problem, self.solutions, self.nia, self._conver_idx
        )
        ida_cost = self.problem.evaluations - ida_start
        print(f"  → {len(self._subcomponents)} subcomponent(s)  evals={self.problem.evaluations}")

        self._cva_evaluations = cva_cost
        self._ida_evaluations = ida_cost
        self._preprocessing_evaluations = cva_cost + ida_cost
        self.evaluations = self.problem.evaluations - self._preprocessing_evaluations

    def stopping_condition_is_met(self) -> bool:
        if not self._subcomponents:
            # No ConverIndex subcomponents means step() has nothing to
            # optimise and never advances evaluations; without this guard
            # the run loop would spin forever.
            return True
        if (self.problem.evaluations - self._preprocessing_evaluations) >= self.max_evaluations:
            return True
        return self.termination_criterion is not None and self.termination_criterion.is_met

    def step(self) -> None:
        for subcomp in self._subcomponents:
            if self.stopping_condition_is_met():
                break
            self.solutions = moeadva_subcomponent(
                self.solutions, self.problem, self._neighbour, subcomp, self.de_f,
                mutation_distribution_index=self.mutation_eta,
                mutation_probability=self._mutation_probability,
            )

    def update_progress(self) -> None:
        self.evaluations = self.problem.evaluations - self._preprocessing_evaluations
        self.observable.notify_all(**self.observable_data())

    def observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }

    def result(self) -> List[FloatSolution]:
        return self.solutions

    def get_name(self) -> str:
        return "MOEADVA"


# ─── LMOCSO: Large-Scale Multiobjective Competitive Swarm Optimizer ───────────
#
# Tian et al., "Efficient Large-Scale Multiobjective Optimization Based on a
# Competitive Swarm Optimizer," IEEE TCYB, vol. 50, no. 8, 2020.
#
# Faithful to PlatEMO v4.x LMOCSO.m / Operator.m / EnvironmentalSelection.m.
# Key deviations from the paper that PlatEMO introduces:
#   1. SDE fitness: objectives normalised to [0,1] before SDE computation.
#   2. r1/r2: one scalar per particle (same across all D dimensions), not component-wise.
#   3. Environmental selection: first keeps only Pareto rank-1 solutions, then APD.
#   4. Progress ratio theta = (FE/maxFE)^2 (evaluation-count based, alpha fixed at 2).
#   5. Offspring = N/2 updated losers + N/2 unchanged winners, polynomial mutation on all N.


def _lmocso_sde_fitness(obj: np.ndarray) -> np.ndarray:
    """SDE-based fitness.

    fitness[i] = min_{j≠i} ||max(0, obj[j] - obj[i])||
    Larger value = more isolated = BETTER particle (winner in competition).
    Expects ``obj`` already normalised to [0,1] per objective (PlatEMO convention);
    see the per-generation min-max normalisation in ``_lmocso_operator``.
    """
    # diff[i,j,m] = obj[j,m] - obj[i,m]
    diff = obj[None, :, :] - obj[:, None, :]                    # (N, N, M)
    sde  = np.sqrt((np.maximum(0.0, diff) ** 2).sum(axis=2))    # (N, N)
    np.fill_diagonal(sde, np.inf)
    return sde.min(axis=1)                                       # (N,)


def _lmocso_ref_vectors(N: int, M: int) -> np.ndarray:
    """Das-Dennis uniform reference vectors on the unit simplex; count ≈ N."""
    from math import comb

    best_H, best_diff = 1, abs(M - N)
    for H in range(1, 500):
        c = comb(H + M - 1, M - 1)
        d = abs(c - N)
        if d < best_diff:
            best_diff, best_H = d, H
        if c > 10 * max(N, 100):
            break

    def _gen(m: int, total: int):
        if m == 1:
            yield [total]
        else:
            for v in range(total + 1):
                for rest in _gen(m - 1, total - v):
                    yield [v] + rest

    return np.array(list(_gen(M, best_H)), dtype=float) / best_H


def _lmocso_copy(sol: FloatSolution) -> FloatSolution:
    """Shallow-structure copy of FloatSolution, including velocity attribute."""
    new = object.__new__(FloatSolution)
    new.lower_bound = sol.lower_bound
    new.upper_bound = sol.upper_bound
    new.objectives  = list(sol.objectives)
    new.constraints = list(sol.constraints) if sol.constraints else []
    new.attributes  = {}
    if "velocity" in sol.attributes:
        new.attributes["velocity"] = sol.attributes["velocity"].copy()
    new.variables = np.asarray(sol.variables).copy()
    return new


def _lmocso_poly_mut_batch(
    decs: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    prob: float,
    eta: float = 20.0,
    rand_fn: Optional[Callable] = None,
) -> np.ndarray:
    """Vectorised polynomial mutation on (N, D) decision matrix (PlatEMO Operator.m).

    prob is typically 1/D, so only the hit entries (≈N of N×D) get the
    polynomial computation; the Bernoulli draw is the only full-width cost.
    Lower/upper halves are disjoint per entry, so evaluating both branches
    from the pre-mutation value is equivalent to the sequential form.
    numpy/cupy 両対応。rand_fn は等価性テスト用の乱数注入フック。
    """
    xp = cp.get_array_module(decs) if _HAS_CUPY else np
    rand = rand_fn if rand_fn is not None else xp.random.random
    out = decs.copy()
    site = rand(out.shape) < prob
    r, c = xp.nonzero(site)
    if r.size == 0:
        return out

    lb_v = lb[c]
    ub_v = ub[c]
    y = out[r, c]
    dy = ub_v - lb_v
    safe_dy = xp.where(dy > 1e-12, dy, 1.0)
    mu = rand(r.size)

    # Lower half: mu <= 0.5
    d1 = xp.clip((y - lb_v) / safe_dy, 0.0, 1.0)
    v1 = xp.maximum(2.0 * mu + (1.0 - 2.0 * mu) * (1.0 - d1) ** (eta + 1), 0.0)
    y_low = xp.clip(y + (xp.power(v1, 1.0 / (eta + 1)) - 1.0) * dy, lb_v, ub_v)

    # Upper half: mu > 0.5
    d2 = xp.clip((ub_v - y) / safe_dy, 0.0, 1.0)
    v2 = xp.maximum(2.0 * (1.0 - mu) + 2.0 * (mu - 0.5) * (1.0 - d2) ** (eta + 1), 0.0)
    y_up = xp.clip(y + (1.0 - xp.power(v2, 1.0 / (eta + 1))) * dy, lb_v, ub_v)

    out[r, c] = xp.where(mu <= 0.5, y_low, y_up)
    return out


def _lmocso_operator(
    population: List[FloatSolution],
    lb: np.ndarray,
    ub: np.ndarray,
    pm: float,
    eta: float,
) -> List[FloatSolution]:
    """PlatEMO Operator.m / LMOCSO.m main loop: CSO update on floor(P/2)*2 individuals.

    Verified against PlatEMO v4.x source (BIMK/PlatEMO, LMOCSO/LMOCSO.m):
    ``Rank = randperm(length(Population), floor(length(Population)/2)*2)`` —
    pairing is WITHOUT replacement and sized by the CURRENT population length
    P (population size is allowed to fluctuate; see ``_lmocso_env_select``,
    which does not pad back to N). PlatEMO guards the degenerate P<2 case
    with ``Rank = [1,1]`` (pair the sole individual with itself) instead of
    crashing; that guard is reproduced below.
    Loser (smaller SDE fitness) learns from Winner via Eq. (6) with
    per-particle scalar r1/r2. Winner position is passed through unchanged.
    Polynomial mutation applied to all offspring.
    Returns list of FloatSolution with objectives UNSET (caller must evaluate).
    """
    P = len(population)
    D = len(lb)

    obj      = np.array([s.objectives for s in population], dtype=np.float64)
    obj_min  = obj.min(axis=0)
    obj_max  = obj.max(axis=0)
    obj_span = np.where(obj_max - obj_min > 1e-12, obj_max - obj_min, 1.0)
    obj_norm = (obj - obj_min) / obj_span
    fitness  = _lmocso_sde_fitness(obj_norm)

    if P >= 2:
        n_pairs = P // 2
        perm    = np.random.permutation(P)[: 2 * n_pairs]
        cand_a  = perm[:n_pairs]
        cand_b  = perm[n_pairs:]
    else:
        # PlatEMO: Rank = [1,1] when length(Population) < 2
        n_pairs = 1
        cand_a  = np.zeros(1, dtype=int)
        cand_b  = np.zeros(1, dtype=int)

    # Swap so that loser < winner in fitness (>=: a is better → swap)
    swap       = fitness[cand_a] >= fitness[cand_b]
    loser_idx  = np.where(swap, cand_b, cand_a)
    winner_idx = np.where(swap, cand_a, cand_b)

    # Extract decision vectors and velocities
    l_decs = np.array([population[i].variables for i in loser_idx],  dtype=float)
    w_decs = np.array([population[i].variables for i in winner_idx], dtype=float)
    l_vels = np.array(
        [population[i].attributes.get("velocity", np.zeros(D)) for i in loser_idx], dtype=float
    )
    w_vels = np.array(
        [population[i].attributes.get("velocity", np.zeros(D)) for i in winner_idx], dtype=float
    )

    # Eq. (6) — per-particle scalar r1/r2 broadcast across D (PlatEMO: repmat(rand(N,1),1,D))
    # GPU があれば速度更新〜変異まで GPU 上で計算して最後に一括転送する。
    xp = cp if _HAS_CUPY else np
    if _HAS_CUPY:
        l_decs, w_decs = cp.asarray(l_decs), cp.asarray(w_decs)
        l_vels, w_vels = cp.asarray(l_vels), cp.asarray(w_vels)
        lb_x, ub_x = cp.asarray(lb), cp.asarray(ub)
    else:
        lb_x, ub_x = lb, ub
    r1 = xp.random.random((n_pairs, 1))
    r2 = xp.random.random((n_pairs, 1))
    off_vel  = r1 * l_vels + r2 * (w_decs - l_decs)
    off_decs_loser = l_decs + off_vel + r1 * (off_vel - l_vels)

    # Stack: N/2 updated losers | N/2 winners
    all_decs = xp.vstack([off_decs_loser, w_decs])
    all_vels = xp.vstack([off_vel,        w_vels])

    # Clamp then mutate
    all_decs = xp.clip(all_decs, lb_x, ub_x)
    all_decs = _lmocso_poly_mut_batch(all_decs, lb_x, ub_x, pm, eta)
    if _HAS_CUPY:
        all_decs = cp.asnumpy(all_decs)
        all_vels = cp.asnumpy(all_vels)

    # Build solution objects (objectives left unset; caller evaluates)
    template = population[0]
    offspring: List[FloatSolution] = []
    for k in range(2 * n_pairs):
        sol = _lmocso_copy(template)
        sol.variables              = all_decs[k].copy()
        sol.attributes["velocity"] = all_vels[k].copy()
        offspring.append(sol)

    return offspring  # length = 2 * n_pairs ≤ P


def _lmocso_env_select(
    population: List[FloatSolution],
    R: np.ndarray,
    theta: float,
) -> List[FloatSolution]:
    """PlatEMO EnvironmentalSelection.m: Pareto filter → APD selection.

    Verified against PlatEMO v4.x source (BIMK/PlatEMO, LMOCSO/EnvironmentalSelection.m):
    1. Keep only Pareto rank-1 solutions from the combined pool (rank-1 is
       never empty for a non-empty input, by definition of non-domination).
    2. Translate objectives to non-negative.
    3. Associate each solution to its closest reference vector by angle.
    4. Select one solution per reference vector with minimum APD.
    5. Return exactly what step 4 produced — PlatEMO does NOT pad or trim to
       N (``Population = Population(Next(Next~=0))``); reference vectors with
       no associated individual are simply dropped, so the returned
       population size can fluctuate below ``N`` across generations. This is
       intentional PlatEMO behaviour, not a bug — see ``_lmocso_operator``'s
       ``P<2`` guard, which is PlatEMO's actual mechanism for tolerating a
       shrunk population instead of preventing the shrinkage itself.
    """
    M = R.shape[1]

    # --- Step 1: Pareto rank-1 filter ---
    ranking = NumpyNonDominatedRanking()
    ranking.compute_ranking(population)
    nd = [s for s in population if s.attributes.get("dominance_ranking", 0) == 0]

    n_nd = len(nd)
    obj  = np.array([s.objectives for s in nd], dtype=np.float64)

    # --- Step 2: Translate to non-negative ---
    obj_t = obj - obj.min(axis=0, keepdims=True)

    # --- Step 3: Unit vectors ---
    obj_norms = np.linalg.norm(obj_t, axis=1, keepdims=True) + 1e-12
    obj_unit  = obj_t / obj_norms                              # (n_nd, M)
    R_norms   = np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
    R_unit    = R / R_norms                                    # (K, M)

    cos_angle = np.clip(obj_unit @ R_unit.T, -1.0, 1.0)       # (n_nd, K)
    angle     = np.arccos(cos_angle)                           # (n_nd, K)

    assoc       = np.argmin(angle, axis=1)                     # (n_nd,)
    angle_assoc = angle[np.arange(n_nd), assoc]                # (n_nd,)

    # Min angle between any two distinct reference vectors (per reference vector)
    K       = len(R)
    cos_rr  = np.clip(R_unit @ R_unit.T, -1.0, 1.0)
    np.fill_diagonal(cos_rr, -2.0)
    gamma   = np.arccos(cos_rr.max(axis=1)) + 1e-12            # (K,)

    # APD = (1 + M*theta * angle_assoc / gamma[assoc]) * ||f||
    apd = (1.0 + M * theta * angle_assoc / gamma[assoc]) * obj_norms.squeeze()  # (n_nd,)

    # --- Step 4/5: One best per reference vector; unassigned vectors are
    # dropped (no padding/trimming — matches PlatEMO's Next(Next~=0)) ---
    selected_idx: List[int] = []
    for r in range(K):
        in_r = np.where(assoc == r)[0]
        if len(in_r) == 0:
            continue
        selected_idx.append(int(in_r[np.argmin(apd[in_r])]))

    return [nd[i] for i in selected_idx]


class LMOCSO(Algorithm[FloatSolution, List[FloatSolution]]):
    """Large-Scale Multiobjective Competitive Swarm Optimizer (PlatEMO-faithful).

    Main loop (verified against PlatEMO v4.x LMOCSO.m/Operator.m/EnvironmentalSelection.m):
      1. SDE fitness → random pairing (no replacement, sized by current P) →
         CSO update (Eq. 6) → polynomial mutation → floor(P/2)*2 offspring
      2. RVEA-style environmental selection on P ∪ P' (Pareto filter + APD)
    Population size is NOT kept fixed at ``population_size`` — PlatEMO's
    EnvironmentalSelection drops reference vectors with no associated
    individual rather than padding, so the population can shrink (down to 1,
    handled by a dedicated guard in ``_lmocso_operator``) and later recover.
    Reference vectors: Das-Dennis lattice with count ≈ population_size.
    """

    def __init__(
        self,
        problem: FloatProblem,
        population_size: int,
        max_evaluations: int,
        mutation_probability: Optional[float] = None,
        mutation_distribution_index: float = 20.0,
        termination_criterion: Optional[TerminationCriterion] = None,
    ):
        super().__init__()
        self.problem          = problem
        self.population_size  = population_size
        self.max_evaluations  = max_evaluations
        self._mutation_probability = mutation_probability
        self.mutation_eta     = mutation_distribution_index
        # Additional stop condition checked alongside the evaluation budget
        # (e.g. HV stagnation).
        self.termination_criterion = termination_criterion
        if termination_criterion is not None:
            self.observable.register(termination_criterion)
        self._R: Optional[np.ndarray] = None
        self._pm: float = 0.0
        # Population size fluctuates (see class docstring); tracked for
        # reporting since it is diagnostically relevant on this problem.
        self._min_population_size: Optional[int] = None

    # ── jMetal Algorithm interface ─────────────────────────────────────────────

    def create_initial_solutions(self) -> List[FloatSolution]:
        D = self.problem.number_of_variables()
        solutions = [self.problem.create_solution() for _ in range(self.population_size)]
        for sol in solutions:
            sol.attributes["velocity"] = np.zeros(D)
        return solutions

    def evaluate(self, solution_list: List[FloatSolution]) -> List[FloatSolution]:
        return _evaluate_solutions(self.problem, solution_list)

    def init_progress(self) -> None:
        N = self.population_size
        M = self.problem.number_of_objectives()
        D = self.problem.number_of_variables()
        self._R  = _lmocso_ref_vectors(N, M)
        self._pm = (
            self._mutation_probability
            if self._mutation_probability is not None
            else 1.0 / D
        )
        # PlatEMO applies EnvironmentalSelection to the freshly-initialised
        # population too (main.m), which can immediately shrink it below N.
        theta = (self.problem.evaluation / self.max_evaluations) ** 2
        self.solutions = _lmocso_env_select(self.solutions, self._R, theta)
        self._min_population_size = len(self.solutions)
        self.evaluations = self.problem.evaluation

    def stopping_condition_is_met(self) -> bool:
        if self.problem.evaluation >= self.max_evaluations:
            return True
        return self.termination_criterion is not None and self.termination_criterion.is_met

    def step(self) -> None:
        lb = np.asarray(self.problem.lower_bound)
        ub = np.asarray(self.problem.upper_bound)

        # Progress ratio theta = (FE/maxFE)^2 (PlatEMO convention, alpha=2 fixed)
        theta = (self.problem.evaluation / self.max_evaluations) ** 2

        # Generate offspring via CSO operator (pair count derived from the
        # current, possibly-shrunk population size — see _lmocso_operator)
        offspring = _lmocso_operator(self.solutions, lb, ub, self._pm, self.mutation_eta)

        # Evaluate offspring (objectives were unset)
        _evaluate_solutions(self.problem, offspring)

        # Environmental selection on P ∪ P'
        self.solutions = _lmocso_env_select(self.solutions + offspring, self._R, theta)
        self._min_population_size = min(self._min_population_size, len(self.solutions))

    def update_progress(self) -> None:
        self.evaluations = self.problem.evaluation
        self.observable.notify_all(**self.observable_data())

    def observable_data(self) -> dict:
        return {
            "PROBLEM":        self.problem,
            "EVALUATIONS":    self.evaluations,
            "SOLUTIONS":      self.result(),
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }

    def result(self) -> List[FloatSolution]:
        return self.solutions

    def get_name(self) -> str:
        return "LMOCSO"


# ─── Shared NSGA-II / NSGA-III selection helpers (CCGDE3 / LSMOF) ─────────────

_ND_BLOCK = 1024


def _dominated_by(cands: np.ndarray, obj: np.ndarray, block: int = _ND_BLOCK) -> np.ndarray:
    """(n,) bool: row j of ``obj`` is dominated by at least one row of ``cands``.

    Row-blocked over ``cands`` so the transient is ``(block, n, M)`` instead of
    ``(len(cands), n, M)``; with ``n`` in the tens of thousands (LSMOF's
    weight-optimisation archive) the unblocked form is several GB per call.
    """
    dominated = np.zeros(len(obj), dtype=bool)
    for s in range(0, len(cands), block):
        diff = cands[s:s + block, None, :] - obj[None, :, :]
        dominated |= ((diff <= 0).all(axis=2) & (diff < 0).any(axis=2)).any(axis=0)
    return dominated


def _nd_filter_mask_array(obj: np.ndarray) -> np.ndarray:
    """Boolean mask of the non-dominated rows of an (n, M) objective matrix."""
    if len(obj) == 0:
        return np.zeros(0, dtype=bool)
    return ~_dominated_by(obj, obj)


def _nd_merge_masks(old_objs: np.ndarray, new_objs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Non-dominated masks over the union ``[old; new]`` when ``old`` is
    already mutually non-dominated (an archive maintained by this filter).

    Equivalent to ``_nd_filter_mask_array(np.concatenate([old, new]))`` split
    back into its two halves: an old row can only be dominated by a new row,
    a new row by an old row or another new row. Costs O(n_old · n_new) instead
    of O((n_old + n_new)²), and never materialises the concatenated matrix.
    """
    keep_old = ~_dominated_by(new_objs, old_objs)
    keep_new = ~(_dominated_by(old_objs, new_objs) | _dominated_by(new_objs, new_objs))
    return keep_old, keep_new


def _nsga2_select(pool: List[FloatSolution], N: int) -> List[FloatSolution]:
    """platEMO-style basic NSGA-II EnvironmentalSelection: fronts + crowding trim.

    Shared by CCGDE3 (``EnvironmentalSelection.m`` / the trim step inside
    ``GDE3_EnvironmentalSelection.m``) and, indirectly, by LSMOF's reference
    -solution sampling (``EnvironmentalSelection(Population, wD)`` selects a
    representative subset, not a size-N population, but the same routine
    applies once N is set to the desired subset size).
    """
    if len(pool) <= N:
        return list(pool)
    ranking = NumpyNonDominatedRanking()
    ranking.compute_ranking(pool, k=N)
    return _lmea_environmental_select(pool, ranking.ranked_sublists, N)


def _full_front_and_crowding(population: List[FloatSolution]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-individual (front_no, crowding_distance), matching platEMO's
    ``NDSort(Objs, inf)`` + ``CrowdingDistance(Objs, FrontNo)`` (crowding is
    computed independently within each front, not across mixed fronts).
    """
    ranking = NumpyNonDominatedRanking()
    ranking.compute_ranking(list(population))
    id_to_pos = {id(s): i for i, s in enumerate(population)}
    front_no = np.empty(len(population), dtype=int)
    crowd = np.empty(len(population), dtype=float)
    for f_idx, front in enumerate(ranking.ranked_sublists):
        cd = _lmea_crowding_distances(front)
        for local_i, sol in enumerate(front):
            pos = id_to_pos[id(sol)]
            front_no[pos] = f_idx
            crowd[pos] = cd[local_i]
    return front_no, crowd


def _nsga2_tournament(front_no: np.ndarray, crowd_dis: np.ndarray, n: int) -> np.ndarray:
    """platEMO TournamentSelection(2, n, FrontNo, -CrowdDis): lower FrontNo wins,
    ties broken by higher CrowdDis. Draws with replacement (as does randi in
    platEMO's implementation).
    """
    N = len(front_no)
    a = np.random.randint(N, size=n)
    b = np.random.randint(N, size=n)
    a_wins = (front_no[a] < front_no[b]) | (
        (front_no[a] == front_no[b]) & (crowd_dis[a] > crowd_dis[b])
    )
    return np.where(a_wins, a, b)


def _nsga3_replace(
    combined: List[FloatSolution],
    reference_directions: np.ndarray,
    ideal_point: np.ndarray,
    worst_point: np.ndarray,
    extreme_points: Optional[np.ndarray],
    n_objs: int,
    population_size: int,
) -> Tuple[List[FloatSolution], np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """NSGA-III environmental selection as a free function.

    Body copied from ``FastNSGAIII.replacement`` (jMetal's NSGAIII.replacement
    with NumpyNonDominatedRanking swapped in) but taking a single already-
    combined pool and explicit ideal/worst/extreme-point state instead of
    reading/writing ``self``. Used by LSMOF so both the phase-1 archive merge
    (``Population = EnvironmentalSelection([Population, Archive], N)``) and
    the phase-2 generational replacement can share one reference-direction
    niching routine — the "NSGA3-based" instantiation the caller asked for,
    in place of platEMO's literal crowding-distance EnvironmentalSelection.
    Callers persist and thread through the returned (ideal, worst, extreme)
    state across calls, exactly as ``self.ideal_point`` etc. persist across
    generations on FastNSGAIII.
    """
    F = np.array([s.objectives for s in combined], dtype=float)
    ideal_point = np.min(np.vstack((ideal_point, F)), axis=0)
    worst_point = np.max(np.vstack((worst_point, F)), axis=0)

    ranking = NumpyNonDominatedRanking()
    ranking.compute_ranking(combined, k=population_size)
    fronts, non_dominated = ranking.ranked_sublists, ranking.get_subfront(0)

    extreme_points = get_extreme_points(
        F=np.array([s.objectives for s in non_dominated]),
        n_objs=n_objs,
        ideal_point=ideal_point,
        extreme_points=extreme_points,
    )

    worst_of_population = np.max(F, axis=0)
    worst_of_front = np.max(np.array([s.objectives for s in non_dominated]), axis=0)
    nadir_point = get_nadir_point(
        extreme_points=extreme_points,
        ideal_point=ideal_point,
        worst_point=worst_point,
        worst_of_population=worst_of_population,
        worst_of_front=worst_of_front,
    )

    pop = np.concatenate(ranking.ranked_sublists)
    F = np.array([s.objectives for s in pop])

    counter = 0
    for i in range(len(fronts)):
        for j in range(len(fronts[i])):
            fronts[i][j] = counter
            counter += 1
    last_front = np.array(fronts[-1])

    niche_of_individuals, dist_to_niche = associate_to_niches(
        F=F, niches=reference_directions, ideal_point=ideal_point, nadir_point=nadir_point
    )

    if len(pop) > population_size:
        if len(fronts) == 1:
            until_last_front = np.array([], dtype=int)
            niche_count = np.zeros(len(reference_directions), dtype=int)
            n_remaining = population_size
        else:
            until_last_front = np.concatenate(fronts[:-1])
            niche_count = compute_niche_count(
                len(reference_directions), niche_of_individuals[until_last_front]
            )
            n_remaining = population_size - len(until_last_front)

        S_idx = niching(
            pop=pop[last_front],
            n_remaining=n_remaining,
            niche_count=niche_count,
            niche_of_individuals=niche_of_individuals[last_front],
            dist_to_niche=dist_to_niche[last_front],
        )

        survivors_idx = np.concatenate((until_last_front, last_front[S_idx].tolist()))
        pop = pop[survivors_idx]

    return list(pop), ideal_point, worst_point, extreme_points


# ─── CCGDE3: Cooperative Coevolutionary GDE3 ───────────────────────────────────
#
# Antonio & Coello Coello, "Use of Cooperative Coevolution for Solving Large
# Scale Multi-objective Optimization Problems," IEEE CEC, 2013.
#
# platEMO-faithful port of Algorithms/Multi-objective optimization/CCGDE3
# (CCGDE3.m / CCDE.m / GDE3_EnvironmentalSelection.m / EnvironmentalSelection.m).
# Deviations from the platEMO source, documented where they occur:
#   1. ``Dec2 = Population1.decs`` in the published CCGDE3.m (group-2 branch)
#      reads back the wrong subpopulation's decision matrix — looks like a
#      copy-paste slip from the group-1 branch just above it. This port uses
#      ``Population2.decs`` there, matching the group-1 branch's own pattern.
#   2. The population passed to ``GDE3_EnvironmentalSelection`` and to the
#      final ``CCDE`` mutation step is evaluated in one batched GPU call per
#      group per round (``_evaluate_decs``) instead of platEMO's implicit
#      per-``Problem.Evaluation`` calls — same objective values, no change in
#      selection outcome.

def _ccde_values_matrix(
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    lb_s: np.ndarray,
    ub_s: np.ndarray,
    cr: float,
    f: float,
    rand_fn: Optional[Callable] = None,
) -> np.ndarray:
    """CCDE.m differential mutation: ``Offspring = Parent1``; on a per-entry
    Bernoulli(cr) mask, ``Offspring += f*(Parent2-Parent3)``; clipped to bounds.
    """
    xp = cp.get_array_module(x1) if _HAS_CUPY else np
    rand = rand_fn if rand_fn is not None else xp.random.random
    site = rand(x1.shape) < cr
    out = xp.where(site, x1 + f * (x2 - x3), x1)
    return xp.clip(out, lb_s, ub_s)


def _ccde_offspring(
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    lb_s: np.ndarray,
    ub_s: np.ndarray,
    cr: float,
    f: float,
    pm: float,
    mutation_eta: float,
) -> np.ndarray:
    """CCDE.m mutation → polynomial mutation, GPU-dispatched like ``_sbx_mut_values``.

    Reused verbatim by LSMOF's phase-2 sub-optimizer (platEMO's ``OperatorDE``
    has the identical DE-then-polynomial-mutation structure, just with
    CR=1/F=0.5 defaults over the full decision vector instead of CCDE's
    CR=0.5/F=0.5 over one variable subgroup).
    """
    if _HAS_CUPY:
        lb_g, ub_g = cp.asarray(lb_s), cp.asarray(ub_s)
        vals = _ccde_values_matrix(cp.asarray(x1), cp.asarray(x2), cp.asarray(x3), lb_g, ub_g, cr, f)
        vals = _poly_mut_matrix(vals, lb_g, ub_g, pm, mutation_eta)
        return cp.asnumpy(vals)
    vals = _ccde_values_matrix(x1, x2, x3, lb_s, ub_s, cr, f)
    return _poly_mut_matrix(vals, lb_s, ub_s, pm, mutation_eta)


def _gde3_select(
    population: List[FloatSolution],
    offspring: List[FloatSolution],
) -> List[FloatSolution]:
    """GDE3_EnvironmentalSelection.m with the constraint-handling terms dropped
    (this problem has no constraints, so ``feasibleP``/``feasibleO`` are always
    True and those terms degenerate away):

      - paired (same-index) comparison: offspring weakly dominates parent in
        every objective → replace the parent;
      - mutually non-dominated pair → keep both (append offspring to the pool);
      - trim the combined pool back to ``len(population)`` by non-dominated
        sorting + crowding distance on the last front.
    """
    N = len(population)
    pop_obj = np.array([s.objectives for s in population], dtype=float)
    off_obj = np.array([s.objectives for s in offspring], dtype=float)

    updated = np.all(pop_obj >= off_obj, axis=1)
    added = np.any(pop_obj < off_obj, axis=1) & np.any(pop_obj > off_obj, axis=1)

    new_pop = list(population)
    for i in np.where(updated)[0]:
        new_pop[i] = offspring[i]
    extra = [offspring[i] for i in np.where(added)[0]]

    pool = new_pop + extra
    if len(pool) <= N:
        return pool
    return _nsga2_select(pool, N)


def _ccgde3_assemble(
    own_vals: np.ndarray,
    own_idx: np.ndarray,
    context_pool: np.ndarray,
    context_idx: np.ndarray,
    D: int,
) -> np.ndarray:
    """platEMO GetInd: place ``own_vals`` at ``own_idx`` columns; fill the
    remaining columns (``context_idx``) with rows sampled *with replacement*
    from ``context_pool`` (the collaborator subpopulation's own columns).
    """
    n = len(own_vals)
    full = np.empty((n, D), dtype=float)
    full[:, own_idx] = own_vals
    if len(context_idx) > 0:
        rows = np.random.randint(len(context_pool), size=n)
        full[:, context_idx] = context_pool[rows]
    return full


class CCGDE3(Algorithm[FloatSolution, List[FloatSolution]]):
    """Cooperative Coevolutionary GDE3 (platEMO-faithful).

    Random grouping splits the D decision variables into 2 co-evolving
    subpopulations (platEMO hardcodes exactly 2 — ``NumEsp = 2`` — throughout
    CCGDE3.m; every later step is written for exactly two groups, so this
    port keeps that structure rather than generalising it). Each subpopulation
    has ``ceil(subpop_fraction * population_size)`` individuals (platEMO:
    ``numSPop = ceil(0.8*N)``) and evolves its own variable subgroup via DE
    (CCDE), collaborating with the other group through a randomly-sampled
    context vector — the current elite (front-1) archive of the *other*
    subpopulation once the main loop starts. The externally-tracked
    ``population_size``-sized population is rebuilt every round via basic
    NSGA-II selection over the union of both subpopulations.
    Inherits jMetal Algorithm so ``.observable`` is available for observer
    registration.
    """

    def __init__(
        self,
        problem: FloatProblem,
        population_size: int,
        max_evaluations: int,
        subpop_fraction: float = 0.8,
        crossover_cr: float = 0.5,
        crossover_f: float = 0.5,
        mutation_probability: Optional[float] = None,
        mutation_distribution_index: float = 20.0,
        gmax: int = 1,
        termination_criterion: Optional[TerminationCriterion] = None,
    ):
        super().__init__()
        self.problem = problem
        self.population_size = population_size
        self.max_evaluations = max_evaluations
        self.subpop_fraction = subpop_fraction
        self.crossover_cr = crossover_cr
        self.crossover_f = crossover_f
        self._mutation_probability = mutation_probability
        self.mutation_eta = mutation_distribution_index
        self.gmax = gmax
        self.termination_criterion = termination_criterion
        if termination_criterion is not None:
            self.observable.register(termination_criterion)

        # Internal state populated during init_progress
        self._group_idx: Optional[List[np.ndarray]] = None
        self._num_spop: int = 0
        self._sub_pop: Optional[List[List[FloatSolution]]] = None
        self._sub_pm: Optional[List[float]] = None

    # ------------------------------------------------------------------
    # jMetal Algorithm abstract interface
    # ------------------------------------------------------------------

    def create_initial_solutions(self) -> List[FloatSolution]:
        # Population is built in init_progress (random grouping + two
        # subpopulations come first).
        return []

    def evaluate(self, solution_list: List[FloatSolution]) -> List[FloatSolution]:
        return solution_list

    def init_progress(self) -> None:
        D = self.problem.number_of_variables()
        lb = np.asarray(self.problem.lower_bound, dtype=float)
        ub = np.asarray(self.problem.upper_bound, dtype=float)

        # Random grouping into 2 subgroups (platEMO: varsPerGroup = floor(D/NumEsp))
        var_per_group = D // 2
        group_id = np.concatenate(
            [np.zeros(var_per_group, dtype=int), np.ones(D - var_per_group, dtype=int)]
        )
        group_id = np.random.permutation(group_id)
        self._group_idx = [np.where(group_id == 0)[0], np.where(group_id == 1)[0]]

        self._num_spop = int(np.ceil(self.subpop_fraction * self.population_size))
        self._sub_pm = [
            (self._mutation_probability if self._mutation_probability is not None
             else 1.0 / max(1, len(self._group_idx[j])))
            for j in range(2)
        ]

        decs = np.random.uniform(lb, ub, size=(self._num_spop, D))
        sub_decs = [decs[:, self._group_idx[j]] for j in range(2)]

        sub_pop: List[List[FloatSolution]] = []
        for j in range(2):
            other = 1 - j
            full = _ccgde3_assemble(
                sub_decs[j], self._group_idx[j], sub_decs[other], self._group_idx[other], D
            )
            objs = _evaluate_decs(self.problem, full, lb, ub)
            sub_pop.append(
                [_moeadva_solution_from(self.problem, full[k], objs[k]) for k in range(self._num_spop)]
            )
        self._sub_pop = sub_pop

        self.solutions = _nsga2_select(sub_pop[0] + sub_pop[1], self.population_size)
        self.evaluations = self.problem.evaluation

    def stopping_condition_is_met(self) -> bool:
        if self.problem.evaluation >= self.max_evaluations:
            return True
        return self.termination_criterion is not None and self.termination_criterion.is_met

    def step(self) -> None:
        D = self.problem.number_of_variables()
        lb = np.asarray(self.problem.lower_bound, dtype=float)
        ub = np.asarray(self.problem.upper_bound, dtype=float)

        # Elite (front-1) context archive of each subpopulation, snapshotted
        # once per round and shared by both groups' offspring generation
        # within this round (platEMO computes NDsubDec1/2 once, before the
        # j=1..NumEsp loop, and reuses them for both j).
        nd_ctx: List[np.ndarray] = []
        for j in range(2):
            ranking = NumpyNonDominatedRanking()
            ranking.compute_ranking(list(self._sub_pop[j]))
            nd = ranking.get_subfront(0)
            decs = np.array([s.variables for s in nd], dtype=float)
            nd_ctx.append(decs[:, self._group_idx[j]])

        for j in range(2):
            other = 1 - j
            group = self._group_idx[j]
            for _ in range(self.gmax):
                if self.stopping_condition_is_met():
                    return
                pop_decs = np.array([s.variables for s in self._sub_pop[j]], dtype=float)
                own = pop_decs[:, group]
                r1 = own[np.random.randint(self._num_spop, size=self._num_spop)]
                r2 = own[np.random.randint(self._num_spop, size=self._num_spop)]
                off_own = _ccde_offspring(
                    own, r1, r2, lb[group], ub[group],
                    self.crossover_cr, self.crossover_f, self._sub_pm[j], self.mutation_eta,
                )
                off_full = _ccgde3_assemble(
                    off_own, group, nd_ctx[other], self._group_idx[other], D
                )
                off_objs = _evaluate_decs(self.problem, off_full, lb, ub)
                offspring = [
                    _moeadva_solution_from(self.problem, off_full[k], off_objs[k])
                    for k in range(self._num_spop)
                ]
                self._sub_pop[j] = _gde3_select(self._sub_pop[j], offspring)
                self.solutions = _nsga2_select(
                    self._sub_pop[0] + self._sub_pop[1], self.population_size
                )
                self.update_progress()

    def update_progress(self) -> None:
        self.evaluations = self.problem.evaluation
        self.observable.notify_all(**self.observable_data())

    def observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }

    def result(self) -> List[FloatSolution]:
        return self.solutions

    def get_name(self) -> str:
        return "CCGDE3"


# ─── LSMOF: Large-Scale Multi-objective Optimization Framework ────────────────
#
# He, Li, Tian, Zhang, Cheng, Jin & Yao, "Accelerating Large-Scale
# Multi-objective Optimization via Problem Reformulation," IEEE TEVC, 2019.
#
# platEMO-faithful port of Algorithms/Multi-objective optimization/LSMOF
# (LSMOF.m / WeightOptimization.m / subNSGAII.m / EnvironmentalSelection.m),
# instantiated with NSGA-III (reference-direction niching) as the base MOEA
# in place of platEMO's default crowding-distance NSGA-II — both the phase-1
# archive merge and the phase-2 generational loop reuse ``_nsga3_replace``
# (the same routine backing NSGA3.py's FastNSGAIII) in place of platEMO's
# EnvironmentalSelection.m. Deviations, documented where they occur:
#   1. Decoded candidate solutions (``lower + w*direction``,
#      ``upper - w*direction``) are clipped to [lower, upper] before
#      evaluation. platEMO's benchmarks (DTLZ/LSMOP/WFG) tolerate a
#      Problem.Evaluation call slightly outside the box; this project's GPU
#      grid kernels (EDD / crossing-count) index fixed-size buffers sized
#      from the control-point bound and do not.
#   2. The inner weight-vector DE (WeightOptimization.m's hand-rolled
#      DE/rand/1/bin over the 2*wD-dim weight space) updates the whole
#      generation from a pre-generation snapshot in one batched GPU
#      evaluation, instead of platEMO's sequential in-place sweep where
#      individual i's mutation can already see individual i-1's replacement
#      within the same generation. Standard synchronous-vs-asynchronous DE
#      distinction, statistically equivalent, and the only way to keep this
#      phase's cost from being dominated by per-individual Python/GPU
#      round-trips (each generation needs SubN*2*wD real Problem.Evaluation
#      rows regardless of update ordering).
#   3. Phase-2 mating uses uniform-random parent triples instead of
#      platEMO's FrontNo/CrowdDis tournament: canonical NSGA-III does not
#      use crowding-biased mating (diversity is enforced entirely at the
#      niching/replacement step), and this project's own NSGA3.py already
#      departs from strict per-paper mating in the same direction.
#   4. Only platEMO's literal invoked default, Operator=2 (DE), is ported
#      for the phase-2 sub-optimizer — ``ParameterSet(10,30,2)`` actually
#      runs DE regardless of the source comment's "default 1" (GA) claim.
#   5. ``WeightOptimization``'s internal wD-reference-solution sampling
#      (``RefPop = EnvironmentalSelection(Population,wD)``) stays basic
#      NSGA-II crowding selection — a minor "pick wD well-spread anchor
#      points" utility shared verbatim with CCGDE3 (``_nsga2_select``), not
#      part of the "NSGA3-based" framework instantiation being requested.

def _lsmof_reference_directions(ref_decs: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """WeightOptimization.m direction vectors: unit vectors from the lower
    bound corner toward each reference solution, and from each reference
    solution toward the upper bound corner. Returns (2*wD, D).
    """
    lower_dir = ref_decs - lb
    upper_dir = ub - ref_decs
    lower_norm = np.linalg.norm(lower_dir, axis=1, keepdims=True)
    upper_norm = np.linalg.norm(upper_dir, axis=1, keepdims=True)
    lower_unit = lower_dir / np.where(lower_norm > 1e-12, lower_norm, 1.0)
    upper_unit = upper_dir / np.where(upper_norm > 1e-12, upper_norm, 1.0)
    return np.vstack([lower_unit, upper_unit])


def _lsmof_decode(
    w: np.ndarray, direct: np.ndarray, lb: np.ndarray, ub: np.ndarray, wD: int
) -> np.ndarray:
    """fitfunc's PopDec: (K, 2*wD) weight rows → (K, 2*wD, D) decision rows.

    Row k, direction i<wD:  lower + w[k,i]  * direct[i]   (grow from lower)
    Row k, direction i>=wD: upper - w[k,i]  * direct[i]   (shrink from upper)
    Clipped to [lb, ub] — see module deviation note (1) above.
    """
    # Assembled in place: the (K, 2*wD, D) block is ~270 MB for air
    # (K=30, wD=10, D≈57k), so the obvious ``concatenate`` + ``clip``
    # formulation would hold three such blocks at once.
    K = w.shape[0]
    decs = np.empty((K, 2 * wD, lb.shape[0]), dtype=float)
    np.multiply(w[:, :wD, None], direct[None, :wD, :], out=decs[:, :wD])
    decs[:, :wD] += lb[None, None, :]
    np.multiply(w[:, wD:, None], direct[None, wD:, :], out=decs[:, wD:])
    np.subtract(ub[None, None, :], decs[:, wD:], out=decs[:, wD:])
    np.clip(decs, lb, ub, out=decs)
    return decs


def _lsmof_weight_optimization(
    problem: FloatProblem,
    population: List[FloatSolution],
    wD: int,
    subN: int,
    generations: int,
    de_beta_range: Tuple[float, float] = (0.2, 0.8),
    de_pcr: float = 0.2,
    should_stop: Optional[Callable[[], bool]] = None,
    on_round: Optional[Callable[[], None]] = None,
) -> List[FloatSolution]:
    """WeightOptimization.m: search the low-dimensional (2*wD) weight space
    with DE, maximising the hypervolume of each individual's decoded
    2*wD-solution front against the current population's nadir point.

    Returns the non-dominated archive of every real decision vector
    evaluated during the call (platEMO: ``Arc = [Arc, temp]`` — the union of
    the initial-round archive and the incrementally re-filtered DE-loop
    archive; both are subsumed by re-filtering one running archive after
    every generation, since a point dominated by anything ever evaluated
    stays dominated by the final union too).
    """
    D = problem.number_of_variables()
    lb = np.asarray(problem.lower_bound, dtype=float)
    ub = np.asarray(problem.upper_bound, dtype=float)

    pop_obj = np.array([s.objectives for s in population], dtype=float)
    reference = pop_obj.max(axis=0)
    hv_calc = HyperVolume(reference_point=list(reference))

    ref_pop = _nsga2_select(population, wD)
    ref_decs = np.array([s.variables for s in ref_pop], dtype=float)
    if len(ref_decs) < wD:
        pad = ref_decs[np.random.randint(len(ref_decs), size=wD - len(ref_decs))]
        ref_decs = np.vstack([ref_decs, pad])
    direct = _lsmof_reference_directions(ref_decs, lb, ub)
    wmax = 0.5 * float(np.linalg.norm(ub - lb))

    def _decode_eval(w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        K = w.shape[0]
        decs = _lsmof_decode(w, direct, lb, ub, wD)
        flat_objs = _evaluate_decs(problem, decs.reshape(K * 2 * wD, D), lb, ub)
        objs = flat_objs.reshape(K, 2 * wD, -1)
        fitness = np.empty(K)
        for i in range(K):
            front = objs[i][_nd_filter_mask_array(objs[i])]
            fitness[i] = -hv_calc.compute(front) if len(front) else 0.0
        return fitness, decs.reshape(K * 2 * wD, D), flat_objs

    w = np.random.uniform(0.0, wmax, size=(subN, 2 * wD))
    fitness, decs0, objs0 = _decode_eval(w)
    keep0 = _nd_filter_mask_array(objs0)
    archive = _RowArchive(decs0, objs0, keep0)
    del decs0, objs0

    idx_all = np.arange(subN)
    for _ in range(generations):
        if should_stop is not None and should_stop():
            break

        a = np.empty(subN, dtype=int)
        b = np.empty(subN, dtype=int)
        c = np.empty(subN, dtype=int)
        for i in range(subN):
            others = np.delete(idx_all, i)
            pick = others[np.random.choice(len(others), 3, replace=False)]
            a[i], b[i], c[i] = pick

        beta = np.random.uniform(de_beta_range[0], de_beta_range[1], size=(subN, 2 * wD))
        y = np.clip(w[a] + beta * (w[b] - w[c]), 0.0, wmax)
        j0 = np.random.randint(2 * wD, size=subN)
        cross = np.random.random((subN, 2 * wD)) <= de_pcr
        cross[np.arange(subN), j0] = True
        z = np.where(cross, y, w)

        fit_z, decs_z, objs_z = _decode_eval(z)
        archive.merge(decs_z, objs_z)
        del decs_z

        improved = fit_z < fitness
        w[improved] = z[improved]
        fitness[improved] = fit_z[improved]

        if on_round is not None:
            on_round()

    # Solutions are handed out as *views* into the archive buffer (no copy):
    # the caller only needs objectives to run environmental selection, and
    # must materialise (copy) the variables of the few survivors it keeps —
    # see ``LSMOF.step``. Copying every row here would double the archive's
    # footprint (several GB on air) at the moment of peak memory use.
    decs, objs = archive.rows()
    return [
        _moeadva_solution_from(problem, decs[i], objs[i], copy=False)
        for i in range(len(decs))
    ]


class _RowArchive:
    """Non-dominated archive of (decision row, objective row) pairs kept in a
    single pre-grown buffer.

    Used by ``_lsmof_weight_optimization``. Its archive grows by
    ``subN * 2 * wD`` (600) candidate rows per DE generation for ~85
    generations, and on air each row is D≈57k doubles (455 KB). The previous
    ``concatenate`` + boolean-index formulation copied the whole archive
    twice per generation (up to ~3× the archive resident at once, >10 GB),
    and the all-pairs non-dominated filter added an (n, n, M) diff on top.
    Here new rows are written into spare capacity (amortised doubling),
    dominated rows are compacted in place, and dominance is checked
    incrementally (``_nd_merge_masks``), so the resident footprint stays
    ~1× the live archive.
    """

    def __init__(self, decs: np.ndarray, objs: np.ndarray, keep: np.ndarray):
        idx = np.flatnonzero(keep)
        self._n = len(idx)
        self._decs = np.empty((max(2 * self._n, 1024), decs.shape[1]), dtype=float)
        np.take(decs, idx, axis=0, out=self._decs[:self._n])
        self._objs = objs[idx]

    def merge(self, new_decs: np.ndarray, new_objs: np.ndarray) -> None:
        keep_old, keep_new = _nd_merge_masks(self._objs[:self._n], new_objs)
        self._compact(keep_old)
        idx_new = np.flatnonzero(keep_new)
        self._reserve(self._n + len(idx_new))
        np.take(new_decs, idx_new, axis=0, out=self._decs[self._n:self._n + len(idx_new)])
        self._objs = np.concatenate([self._objs[:self._n], new_objs[idx_new]], axis=0)
        self._n += len(idx_new)

    def _compact(self, keep: np.ndarray) -> None:
        """Drop dominated rows, moving survivors down in place (front-to-back,
        so every move targets an index ≤ its source and nothing live is
        overwritten). Dominated old rows are rare, so this is usually a no-op."""
        if keep.all():
            return
        dst = 0
        for src in np.flatnonzero(keep):
            if src != dst:
                self._decs[dst] = self._decs[src]
            dst += 1
        self._objs = self._objs[keep]
        self._n = dst

    def _reserve(self, n: int) -> None:
        if n <= self._decs.shape[0]:
            return
        grown = np.empty((max(n, 2 * self._decs.shape[0]), self._decs.shape[1]), dtype=float)
        grown[:self._n] = self._decs[:self._n]
        self._decs = grown  # untouched capacity is never resident (no page commit)

    def rows(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._decs[:self._n], self._objs[:self._n]


def _lsmof_sub_nsga3_step(
    problem: FloatProblem,
    population: List[FloatSolution],
    reference_directions: np.ndarray,
    nsga3_state: Dict[str, Any],
    crossover_cr: float,
    crossover_f: float,
    mutation_probability: float,
    mutation_distribution_index: float,
) -> List[FloatSolution]:
    """Phase-2 sub-optimizer: OperatorDE-style DE offspring (CR=1, F=0.5
    defaults) from uniform-random parent triples (module deviation note 3),
    followed by NSGA-III reference-direction niching in place of platEMO's
    crowding-distance EnvironmentalSelection (the "NSGA3-based" swap).
    """
    N = len(population)
    lb = np.asarray(problem.lower_bound, dtype=float)
    ub = np.asarray(problem.upper_bound, dtype=float)

    decs = np.array([s.variables for s in population], dtype=float)
    p1 = decs[np.random.randint(N, size=N)]
    p2 = decs[np.random.randint(N, size=N)]
    p3 = decs[np.random.randint(N, size=N)]

    off_decs = _ccde_offspring(
        p1, p2, p3, lb, ub,
        crossover_cr, crossover_f, mutation_probability, mutation_distribution_index,
    )
    off_objs = _evaluate_decs(problem, off_decs, lb, ub)
    offspring = [_moeadva_solution_from(problem, off_decs[k], off_objs[k]) for k in range(N)]

    selected, ideal, worst, extreme = _nsga3_replace(
        population + offspring, reference_directions,
        nsga3_state["ideal"], nsga3_state["worst"], nsga3_state["extreme"],
        problem.number_of_objectives(), N,
    )
    nsga3_state["ideal"], nsga3_state["worst"], nsga3_state["extreme"] = ideal, worst, extreme
    return selected


class LSMOF(Algorithm[FloatSolution, List[FloatSolution]]):
    """Large-Scale Multi-objective Optimization Framework — NSGA-III instantiation.

    Phase 1 (first ``phase1_fraction`` of the evaluation budget, platEMO
    default 0.6): repeatedly collapse the search to a 2*wD-dim weight space
    via ``_lsmof_weight_optimization`` and merge the resulting real-decision
    archive back into the tracked population with NSGA-III replacement.
    Phase 2 (remaining budget): a standard full-D generational NSGA-III loop
    (DE offspring + reference-direction niching) — platEMO's ``subNSGAII``,
    instantiated with NSGA-III per this port's "NSGA3-based" request in
    place of platEMO's own NSGA-II sub-optimizer.
    Inherits jMetal Algorithm so ``.observable`` is available for observer
    registration.
    """

    def __init__(
        self,
        problem: FloatProblem,
        population_size: int,
        max_evaluations: int,
        wD: int = 10,
        subN: int = 30,
        phase1_fraction: float = 0.6,
        weight_opt_evaluation_fraction: float = 0.05,
        reference_direction_partitions: int = 12,
        crossover_cr: float = 1.0,
        crossover_f: float = 0.5,
        mutation_probability: Optional[float] = None,
        mutation_distribution_index: float = 20.0,
        termination_criterion: Optional[TerminationCriterion] = None,
    ):
        super().__init__()
        self.problem = problem
        self.population_size = population_size
        self.max_evaluations = max_evaluations
        self.wD = wD
        self.subN = subN
        self.phase1_fraction = phase1_fraction
        self.weight_opt_evaluation_fraction = weight_opt_evaluation_fraction
        self.reference_direction_partitions = reference_direction_partitions
        self.crossover_cr = crossover_cr
        self.crossover_f = crossover_f
        self._mutation_probability = mutation_probability
        self.mutation_eta = mutation_distribution_index
        self.termination_criterion = termination_criterion
        if termination_criterion is not None:
            self.observable.register(termination_criterion)

        # Internal state populated during init_progress
        self._generations: int = 1
        self._pm: float = 0.0
        self._reference_directions: Optional[np.ndarray] = None
        self._nsga3_state: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # jMetal Algorithm abstract interface
    # ------------------------------------------------------------------

    def create_initial_solutions(self) -> List[FloatSolution]:
        return [self.problem.create_solution() for _ in range(self.population_size)]

    def evaluate(self, solution_list: List[FloatSolution]) -> List[FloatSolution]:
        return _evaluate_solutions(self.problem, solution_list)

    def init_progress(self) -> None:
        D = self.problem.number_of_variables()
        M = self.problem.number_of_objectives()
        self._pm = (
            self._mutation_probability if self._mutation_probability is not None
            else 1.0 / D
        )
        rdf = UniformReferenceDirectionFactory(
            n_dim=M, n_partitions=self.reference_direction_partitions
        )
        self._reference_directions = rdf.compute()
        self._nsga3_state = {
            "ideal": np.full(M, np.inf),
            "worst": np.full(M, -np.inf),
            "extreme": None,
        }
        # platEMO: G = ceil(maxFE*0.05 / (SubN*2*wD)) — DE generations per
        # WeightOptimization call, calibrated so one call costs roughly
        # weight_opt_evaluation_fraction of the total evaluation budget.
        self._generations = max(
            1,
            int(np.ceil(
                self.max_evaluations * self.weight_opt_evaluation_fraction
                / (self.subN * 2 * self.wD)
            )),
        )
        self.evaluations = self.problem.evaluation

    def stopping_condition_is_met(self) -> bool:
        if self.problem.evaluation >= self.max_evaluations:
            return True
        return self.termination_criterion is not None and self.termination_criterion.is_met

    def step(self) -> None:
        if self.problem.evaluation < self.phase1_fraction * self.max_evaluations:
            archive = _lsmof_weight_optimization(
                self.problem, self.solutions, self.wD, self.subN, self._generations,
                should_stop=self.stopping_condition_is_met,
                on_round=self.update_progress,
            )
            selected, ideal, worst, extreme = _nsga3_replace(
                self.solutions + archive, self._reference_directions,
                self._nsga3_state["ideal"], self._nsga3_state["worst"], self._nsga3_state["extreme"],
                self.problem.number_of_objectives(), self.population_size,
            )
            self._nsga3_state["ideal"] = ideal
            self._nsga3_state["worst"] = worst
            self._nsga3_state["extreme"] = extreme
            # Archive solutions carry views into the weight-optimisation
            # buffer (see _lsmof_weight_optimization); copy the survivors so
            # the buffer — several GB on air — is released with ``archive``.
            for s in selected:
                if isinstance(s.variables, np.ndarray) and s.variables.base is not None:
                    s.variables = np.array(s.variables, dtype=float)
            del archive
            self.solutions = selected
        else:
            self.solutions = _lsmof_sub_nsga3_step(
                self.problem, self.solutions, self._reference_directions, self._nsga3_state,
                self.crossover_cr, self.crossover_f, self._pm, self.mutation_eta,
            )
        self.update_progress()

    def update_progress(self) -> None:
        self.evaluations = self.problem.evaluation
        self.observable.notify_all(**self.observable_data())

    def observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }

    def result(self) -> List[FloatSolution]:
        return self.solutions

    def get_name(self) -> str:
        return "LSMOF"
