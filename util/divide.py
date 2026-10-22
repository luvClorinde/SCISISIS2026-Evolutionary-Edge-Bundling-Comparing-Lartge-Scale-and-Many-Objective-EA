from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Union

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from util.module import EdgeBundlingModule

DivideFn = Callable[["EdgeBundlingModule", int], List]


@dataclass
class Partition:
    """エッジ分類用パーティション（Approach 2）"""
    edge_indices: np.ndarray  # shape (k,) — full_module.edges へのインデックス
    G: nx.Graph
    pos: np.ndarray           # (n_sub_nodes, 2)
    label: str = ""


@dataclass
class CPPartition:
    """CP分類用パーティション（Approach 1）"""
    var_indices: np.ndarray            # フル変数ベクトル (n_edges*n_control,) へのインデックス
    affected_edge_indices: np.ndarray  # var_indices から逆引きしたエッジインデックス
    label: str = ""


AnyPartition = Union[Partition, CPPartition]


# ─── 構築ヘルパー ────────────────────────────────────────────────────────────

def _build_partition(
    full_module: "EdgeBundlingModule", edge_indices: np.ndarray, label: str = ""
) -> Partition:
    """edge_indices から Partition（G, pos 含む）を構築する共通ヘルパー"""
    sub_edges_orig = full_module.edges[edge_indices]
    node_ids = np.unique(sub_edges_orig)
    id_map = {int(old): new for new, old in enumerate(node_ids)}

    sub_G = nx.Graph()
    sub_G.add_nodes_from(range(len(node_ids)))
    for u, v in sub_edges_orig:
        sub_G.add_edge(id_map[int(u)], id_map[int(v)])

    sub_pos = full_module.pos[node_ids]
    return Partition(edge_indices=edge_indices, G=sub_G, pos=sub_pos, label=label)


def _build_cp_partition(
    full_module: "EdgeBundlingModule", var_indices: np.ndarray, label: str = ""
) -> CPPartition:
    """var_indices から CPPartition を構築する共通ヘルパー"""
    affected_edge_indices = np.unique(var_indices // full_module.n_control)
    return CPPartition(
        var_indices=var_indices,
        affected_edge_indices=affected_edge_indices,
        label=label,
    )


# ─── サブモジュール生成・マージユーティリティ ──────────────────────────────────

def make_sub_module(full_module: "EdgeBundlingModule", partition: Partition) -> "EdgeBundlingModule":
    """Partition が持つ G と pos から部分 EdgeBundlingModule を返す"""
    from util.module import EdgeBundlingModule
    canvas = (full_module.xlim, full_module.ylim, full_module.x_show, full_module.y_show)
    return EdgeBundlingModule(partition.G, partition.pos, full_module.n_control, canvas)


def assemble_full_var(
    full_module: "EdgeBundlingModule",
    partitions: List[Partition],
    sub_vars_list: List[np.ndarray],
) -> np.ndarray:
    """Partition 用: 各パーティションの変数を結合して完全変数ベクトルを作る"""
    nc = full_module.n_control
    full_var = np.zeros(full_module.n_edges * nc)
    for partition, sub_var in zip(partitions, sub_vars_list):
        for local_i, global_i in enumerate(partition.edge_indices):
            full_var[global_i * nc : (global_i + 1) * nc] = \
                sub_var[local_i * nc : (local_i + 1) * nc]
    return full_var


def assemble_cp_var(
    full_module: "EdgeBundlingModule",
    cp_partitions: List[CPPartition],
    sub_vars_list: List[np.ndarray],
) -> np.ndarray:
    """CPPartition 用: 各クラスタの変数を var_indices の位置に配置して完全変数ベクトルを作る"""
    full_var = np.zeros(full_module.n_edges * full_module.n_control)
    for cp_part, sub_var in zip(cp_partitions, sub_vars_list):
        full_var[cp_part.var_indices] = sub_var
    return full_var


# ─── 分割アルゴリズム ────────────────────────────────────────────────────────

def divide_random(full_module: "EdgeBundlingModule", n_parts: int, seed: int = 0) -> List[Partition]:
    """エッジをランダムに n_parts 均等分割するベースライン（Approach 2）"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(full_module.n_edges)
    parts = np.array_split(idx, n_parts)
    return [
        _build_partition(full_module, np.sort(p), label=f"rand_{i}")
        for i, p in enumerate(parts)
    ]


def divide_random_cp(full_module: "EdgeBundlingModule", n_parts: int, seed: int = 0) -> List[CPPartition]:
    """CPをランダムに n_parts 均等分割するベースライン（Approach 1）"""
    rng = np.random.default_rng(seed)
    n_total = full_module.n_edges * full_module.n_control
    idx = rng.permutation(n_total)
    parts = np.array_split(idx, n_parts)
    return [
        _build_cp_partition(full_module, np.sort(p), label=f"rand_cp_{i}")
        for i, p in enumerate(parts)
    ]


def divide_metis_cp(
    full_module: "EdgeBundlingModule", n_parts: int, k_neighbors: int = 5
) -> List[CPPartition]:
    """
    CP を METIS でパーティショニング（Approach 1）。
    グラフ構造:
      - 同エッジ内の隣接CP をチェーン接続
      - 他エッジとの空間的に近い CP を k-NN で接続
    """
    import pymetis
    from scipy.spatial import cKDTree

    n_control = full_module.n_control
    n_edges   = full_module.n_edges
    n_total   = n_edges * n_control

    # CP の2D座標: (n_edges*n_control, 2)
    cp_pos = full_module.pos_np[:, 1:-1, :].reshape(-1, 2)

    # 隣接リスト構築
    adj: list[set[int]] = [set() for _ in range(n_total)]

    # 同エッジ内の隣接CP をチェーン接続
    for e in range(n_edges):
        for k in range(n_control - 1):
            i = e * n_control + k
            j = e * n_control + k + 1
            adj[i].add(j)
            adj[j].add(i)

    # 他エッジとの近傍接続 (k-NN)
    tree = cKDTree(cp_pos)
    _, nn_idx = tree.query(cp_pos, k=k_neighbors + 1)  # +1: 自身を含む

    for i in range(n_total):
        edge_i = i // n_control
        for j in map(int, nn_idx[i, 1:]):
            if j // n_control != edge_i:
                adj[i].add(j)
                adj[j].add(i)

    _, labels = pymetis.part_graph(n_parts, adjacency=[list(a) for a in adj])
    labels = np.array(labels)

    return [
        _build_cp_partition(full_module, np.where(labels == k)[0], label=f"metis_cp_{k}")
        for k in range(n_parts)
    ]


def divide_kmeans_cp(full_module: "EdgeBundlingModule", n_parts: int, seed: int = 0) -> List[CPPartition]:
    """
    CP の2D座標を特徴量として K-means クラスタリング（Approach 1）。
    var_index = edge_idx * n_control + cp_idx に対応する位置座標を使用。
    """
    from scipy.cluster.vq import kmeans2

    # (n_edges, n_control+2, 2) → endpoints を除いた CP のみ → (n_edges*n_control, 2)
    cp_pos = full_module.pos_np[:, 1:-1, :].reshape(-1, 2).astype(np.float64)

    _, labels = kmeans2(cp_pos, n_parts, seed=seed, minit="points")

    return [
        _build_cp_partition(full_module, np.where(labels == k)[0], label=f"kmeans_cp_{k}")
        for k in range(n_parts)
    ]


def divide_meanshift(full_module: "EdgeBundlingModule", n_parts: int) -> List[Partition]:
    """
    エッジ重心（端点 + CP の平均）を特徴量として Mean Shift クラスタリング（Approach 2）。
    n_parts はバンド幅推定のヒント。実際のクラスタ数は自動決定。
    """
    from sklearn.cluster import MeanShift, estimate_bandwidth

    # (n_edges, n_control+2, 2) → 各エッジの全点平均 → (n_edges, 2)
    edge_centroids = full_module.pos_np.mean(axis=1).astype(np.float64)

    bandwidth = estimate_bandwidth(edge_centroids, quantile=1.0 / n_parts)
    labels = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit_predict(edge_centroids)

    unique_labels = np.unique(labels)
    return [
        _build_partition(full_module, np.where(labels == k)[0], label=f"meanshift_{k}")
        for k in unique_labels
    ]


def divide_meanshift_cp(full_module: "EdgeBundlingModule", n_parts: int) -> List[CPPartition]:
    """
    CP の2D座標を特徴量として Mean Shift クラスタリング（Approach 1）。
    n_parts はバンド幅推定のヒント。実際のクラスタ数は自動決定。
    """
    from sklearn.cluster import MeanShift, estimate_bandwidth

    # (n_edges, n_control+2, 2) → endpoints を除いた CP のみ → (n_edges*n_control, 2)
    cp_pos = full_module.pos_np[:, 1:-1, :].reshape(-1, 2).astype(np.float64)

    bandwidth = estimate_bandwidth(cp_pos, quantile=1.0 / n_parts)
    labels = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit_predict(cp_pos)

    unique_labels = np.unique(labels)
    return [
        _build_cp_partition(full_module, np.where(labels == k)[0], label=f"meanshift_cp_{k}")
        for k in unique_labels
    ]


def _edge_lengths(full_module: "EdgeBundlingModule") -> np.ndarray:
    """各エッジの端点間ユークリッド距離 (n_edges,)"""
    start = full_module.pos_np[:, 0, :]   # (n_edges, 2)
    end   = full_module.pos_np[:, -1, :]  # (n_edges, 2)
    return np.linalg.norm(end - start, axis=1)


def divide_length_range(full_module: "EdgeBundlingModule", n_parts: int) -> List[Partition]:
    """
    エッジ長の相対値（最短=0%, 最長=100%）で n 等幅ビンに分類（Approach 2）。
    空ビンはスキップするため実際のパーティション数は n_parts 以下になることがある。
    """
    lengths = _edge_lengths(full_module)
    l_min, l_max = lengths.min(), lengths.max()
    normalized = (lengths - l_min) / (l_max - l_min)
    bin_idx = np.floor(normalized * n_parts).astype(int).clip(0, n_parts - 1)

    return [
        _build_partition(full_module, np.where(bin_idx == k)[0], label=f"len_range_{k}")
        for k in np.unique(bin_idx)
    ]


def divide_length_rank(full_module: "EdgeBundlingModule", n_parts: int) -> List[Partition]:
    """
    エッジ長でソートして n 等個数分割（Approach 2）。
    各パーティションはほぼ同数のエッジを持つ（短い順に並ぶ）。
    """
    lengths = _edge_lengths(full_module)
    sorted_idx = np.argsort(lengths)
    parts = np.array_split(sorted_idx, n_parts)
    return [
        _build_partition(full_module, np.sort(p), label=f"len_rank_{i}")
        for i, p in enumerate(parts)
    ]


def divide_grid_cp(full_module: "EdgeBundlingModule", n_parts: int) -> List[CPPartition]:
    """
    描画領域を ceil(sqrt(n_parts)) × ceil(n_parts / n_cols) のグリッドに分割し、
    CP が属するセルでクラスタリング（Approach 1）。
    n_parts=4 → 2×2、n_parts=9 → 3×3。空セルはスキップ。
    """
    import math
    n_cols = math.ceil(math.sqrt(n_parts))
    n_rows = math.ceil(n_parts / n_cols)

    cp_pos = full_module.pos_np[:, 1:-1, :].reshape(-1, 2)  # (n_edges*n_control, 2)
    cell_w = full_module.xlim / n_cols
    cell_h = full_module.ylim / n_rows

    col_idx = np.clip((cp_pos[:, 0] / cell_w).astype(int), 0, n_cols - 1)
    row_idx = np.clip((cp_pos[:, 1] / cell_h).astype(int), 0, n_rows - 1)
    cell_id = row_idx * n_cols + col_idx

    return [
        _build_cp_partition(full_module, np.where(cell_id == k)[0], label=f"grid_{k}")
        for k in np.unique(cell_id)
    ]


def divide_link_community(
    full_module: "EdgeBundlingModule", n_parts: int,
    max_iter: int = 500, delta: float = 1e-3, seed: int = 0, n_init: int = 10,
) -> List[Partition]:
    """
    Ball-Karrer-Newman (arXiv:1104.3590) のリンクコミュニティ EM アルゴリズム（Approach 2）。
    各エッジの帰属確率 q_ij(z) = θ_iz·θ_jz / Σ_z θ_iz·θ_jz を EM で推定し
    argmax_z q_ij(z) でエッジをハード分割する。n_init 回の乱数初期化で最高対数尤度の解を採用。
    """
    rng = np.random.default_rng(seed)

    node_list = np.array(sorted(full_module.G.nodes()), dtype=int)
    n_nodes = len(node_list)
    node_to_idx = {int(v): i for i, v in enumerate(node_list)}

    src = np.array([node_to_idx[int(u)] for u, v in full_module.edges], dtype=int)
    dst = np.array([node_to_idx[int(v)] for u, v in full_module.edges], dtype=int)
    K = n_parts

    best_ll = -np.inf
    best_labels = np.zeros(len(src), dtype=int)

    for _ in range(n_init):
        # k_mat[i, z]: 頂点 i のカラー z エッジ端数の期待値
        k_mat = rng.random((n_nodes, K)) + 1e-6
        ll_new = -np.inf

        for _ in range(max_iter):
            kappa = np.maximum(k_mat.sum(axis=0), 1e-12)    # (K,)
            theta = k_mat / np.sqrt(kappa)                  # (n_nodes, K)

            prod = theta[src] * theta[dst]                  # (m, K)
            D = np.maximum(prod.sum(axis=1, keepdims=True), 1e-300)
            q = prod / D                                    # (m, K)

            ll_prev, ll_new = ll_new, float(np.log(D).sum())

            k_new = np.zeros_like(k_mat)
            np.add.at(k_new, src, q)
            np.add.at(k_new, dst, q)
            k_new[k_new < delta] = 0.0
            k_mat = k_new

            if abs(ll_new - ll_prev) < 1e-4:
                break

        if ll_new > best_ll:
            best_ll = ll_new
            best_labels = q.argmax(axis=1).astype(int)

    result = []
    for z in range(K):
        idx = np.where(best_labels == z)[0]
        if len(idx) == 0:
            continue
        result.append(_build_partition(full_module, idx, label=f"link_community_{z}"))

    return result


def divide_line_graph(full_module: "EdgeBundlingModule", n_parts: int, seed: int = 0) -> List[Partition]:
    """
    Cluster-Based Edge Bundling の手法（Approach 2）。
    元グラフの Line Graph（エッジ → ノード、共有ノードを持つエッジ同士を接続）を構築し、
    モジュラリティベースのコミュニティ検出でエッジをクラスタリングする。
    """
    LG = nx.line_graph(full_module.G)

    # edge → index 逆引きマップ（向き不問）
    edge_to_idx = {
        frozenset([int(u), int(v)]): i
        for i, (u, v) in enumerate(full_module.edges)
    }

    communities = nx.community.greedy_modularity_communities(LG, best_n=n_parts)

    result = []
    for k, comm in enumerate(sorted(communities, key=len, reverse=True)):
        indices = np.array([edge_to_idx[frozenset([int(u), int(v)])] for u, v in comm])
        result.append(_build_partition(full_module, np.sort(indices), label=f"linegraph_{k}"))

    return result
