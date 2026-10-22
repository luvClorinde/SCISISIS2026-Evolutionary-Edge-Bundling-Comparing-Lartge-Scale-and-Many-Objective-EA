"""
util/evaluation.py — 定量評価モジュール

共通入力フォーマット:
    paths  : List[np.ndarray]  各エッジの折れ線座標列 (n_pts_i, 2)
    pos    : np.ndarray        ノード座標 (n_nodes, 2)
    G      : networkx.Graph    元グラフ（エッジ順は list(G.edges()) と一致）

変換ヘルパー:
    fdeb_to_paths(pts)           FDEB bundle() 出力 → paths
    ctrl_to_paths(module, vars)  進化的アルゴリズムの解変数 → paths

メトリクス:
    ink_ratio(paths, pos, G, canvas)
    distortion(paths, pos, G)
    ambiguity(paths, pos, G, ...)
"""

from __future__ import annotations

import io
from typing import List, Tuple

import networkx as nx
import numpy as np

# ============================================================
# CUDA カーネル：セグメントごとにラスタライズを並列実行
# 1スレッド = 1セグメント。各スレッドが担当セグメントのピクセルを書き込む。
# ============================================================
_RASTERIZE_SRC = r"""
extern "C" __global__ void rasterize(
    float* __restrict__ occup,
    float* __restrict__ angle_arr,
    const float* __restrict__ segs,   /* (n_segs, 6): x0,y0,x1,y1,angle,z */
    const int n_segs,
    const int W,   /* w + 1 */
    const int H,   /* h + 1 */
    const int E    /* n_edges */
) {
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= n_segs) return;

    const float x0 = segs[sid*6+0], y0 = segs[sid*6+1];
    const float x1 = segs[sid*6+2], y1 = segs[sid*6+3];
    const float a  = segs[sid*6+4];
    const int   z  = (int)segs[sid*6+5];

    const float dx = x1 - x0, dy = y1 - y0;
    const int seg_len = max(1, (int)ceilf(sqrtf(dx*dx + dy*dy) / 0.4f));

    int prev_px = -2, prev_py = -2;
    for (int i = 0; i < seg_len; i++) {
        float t = (seg_len > 1) ? (float)i / (float)(seg_len - 1) : 0.0f;
        int px = min(W-1, max(0, (int)floorf(x0 + t * dx)));
        int py = min(H-1, max(0, (int)floorf(y0 + t * dy)));
        if (px == prev_px && py == prev_py) continue;
        prev_px = px; prev_py = py;
        int idx = px * H * E + py * E + z;
        occup[idx]     = 1.0f;
        angle_arr[idx] = a;
    }
}
"""


# ============================================================
# デフォルト定数（experiments.py に準拠）
# ============================================================
_SCALE          = 0.25
_ANGLE_DEG      = 10.0
_SP_CUTOFF      = 5


# ============================================================
# 変換ヘルパー
# ============================================================

def fdeb_to_paths(pts: np.ndarray) -> List[np.ndarray]:
    """FDEB bundle() 出力 (n_edges, n_pts, 2) → List[np.ndarray]"""
    return [pts[i] for i in range(pts.shape[0])]


def ctrl_to_paths(module, variables, n_samples: int = 100) -> List[np.ndarray]:
    """
    進化的アルゴリズムの解変数 → paths

    module    : EdgeBundlingModule
    variables : solution.variables (1D array-like)
    """
    try:
        import cupy as cp
        v = cp.asarray(variables, dtype=cp.float64)
    except Exception:
        v = np.asarray(variables, dtype=float)

    # move_attract (move_attract 系) か move_ver (move_ver 系) を自動判定
    n_vars = len(v)
    if hasattr(module, 'move_attract') and n_vars % 3 == 0:
        ctrl = module.move_attract(v)
    else:
        ctrl = module.move_ver(v)

    try:
        ctrl = ctrl.get()           # CuPy → NumPy
    except AttributeError:
        pass

    orig_n = module.n_samples
    module.n_samples = n_samples
    module.B = module._EdgeBundlingModule__bernstein(n_samples)
    sampled = module.sample_bezier_edges(ctrl)   # (n_edges, n_samples, 2)
    module.n_samples = orig_n
    module.B = module._EdgeBundlingModule__bernstein(orig_n)

    return [sampled[i] for i in range(sampled.shape[0])]


# ============================================================
# Ink Ratio
# ============================================================

def ink_ratio(
    paths: List[np.ndarray],
    pos: np.ndarray,
    G: nx.Graph,
    canvas: Tuple[int, int, int, int],
    dpi: int = 100,
    edge_color: str = "black",
    alpha: float = 1.0,
    edge_width: float = 1.0,
) -> float:
    """
    ink ratio = 暗ピクセル数(bundled) / 暗ピクセル数(straight-line)

    Returns
    -------
    float  (> 1 = インク増、< 1 = インク減)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    xlim, ylim, x_show, y_show = canvas
    ratio_wh = xlim / ylim
    figsize = (x_show, x_show / ratio_wh) if ratio_wh >= 1 else (x_show * ratio_wh, y_show)

    def _render(draw_fn) -> np.ndarray:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor="white")
        ax.set_xlim(0, xlim)
        ax.set_ylim(ylim, 0)
        ax.axis("off")
        ax.set_facecolor("white")
        draw_fn(ax)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        return np.array(Image.open(buf).convert("L"))

    def _draw_bundled(ax):
        for pts in paths:
            ax.plot(pts[:, 0], pts[:, 1], "-",
                    color=edge_color, alpha=alpha, linewidth=edge_width)

    def _draw_straight(ax):
        for u, v in G.edges():
            ax.plot([pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]], "-",
                    color=edge_color, alpha=alpha, linewidth=edge_width)

    bundled_dark  = ((_render(_draw_bundled))  < 255).sum()
    straight_dark = ((_render(_draw_straight)) < 255).sum()

    return float(bundled_dark) / float(straight_dark) if straight_dark > 0 else float("nan")


# ============================================================
# Distortion
# ============================================================

def distortion(
    paths: List[np.ndarray],
    pos: np.ndarray,
    G: nx.Graph,
) -> Tuple[float, float, float, float, float]:
    """
    各エッジの折れ線長 / 直線長 (distortion)

    Returns
    -------
    (min, mean, std, var, median)
    """
    dists = []
    for i, (u, v) in enumerate(G.edges()):
        pts = paths[i]
        l_straight = float(np.linalg.norm(pos[u] - pos[v]))
        if l_straight < 1e-10:
            continue
        if len(pts) <= 2:
            dists.append(1.0)
        else:
            l_poly = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            dists.append(l_poly / l_straight)

    d = np.array(dists)
    return float(d.min()), float(d.mean()), float(d.std()), float(d.var()), float(np.median(d))


# ============================================================
# Ambiguity
# ============================================================

def ambiguity(
    paths: List[np.ndarray],
    pos: np.ndarray,
    G: nx.Graph,
    scale: float = _SCALE,
    angle_thresh_deg: float = _ANGLE_DEG,
    sp_cutoff: int = _SP_CUTOFF,
) -> List[float]:
    """
    ピクセルベースのambiguity（experiments.py のロジックに準拠）。
    CuPy が利用可能なら GPU で高速化。

    Returns
    -------
    List[float]: cutoff=1..sp_cutoff それぞれのambiguity値
    """
    try:
        import cupy as cp
        return _ambiguity_gpu(paths, pos, G, scale, angle_thresh_deg, sp_cutoff, cp)
    except Exception:
        return _ambiguity_cpu(paths, pos, G, scale, angle_thresh_deg, sp_cutoff)


def _build_segments(paths: List[np.ndarray], scale: float) -> np.ndarray:
    """
    全エッジのセグメントを (n_segs, 6) の float32 配列にまとめる。
    全パスが同じ長さ（FDEB）の場合は完全ベクトル化、それ以外は逐次処理。
    columns: x0, y0, x1, y1, angle, edge_z
    """
    lengths = [len(p) for p in paths]
    if len(set(lengths)) == 1 and lengths[0] >= 2:
        # 均一長パス（FDEB）: 完全ベクトル化
        pts_all = np.stack(paths) * scale          # (E, L, 2)
        x0s = pts_all[:, :-1, 0].ravel()           # (E*(L-1),)
        y0s = pts_all[:, :-1, 1].ravel()
        x1s = pts_all[:, 1:,  0].ravel()
        y1s = pts_all[:, 1:,  1].ravel()
        angles = np.arctan2(y0s - y1s, x0s - x1s)
        angles[angles < 0] += 2 * np.pi
        n_segs_per_edge = lengths[0] - 1
        zs = np.repeat(np.arange(len(paths), dtype=np.float32), n_segs_per_edge)
        return np.column_stack([x0s, y0s, x1s, y1s, angles, zs]).astype(np.float32)

    # 可変長パス（EPB）: Python ループ（行数が不定のため）
    rows = []
    for z, pts in enumerate(paths):
        p = pts * scale
        x0s, y0s = p[:-1, 0], p[:-1, 1]
        x1s, y1s = p[1:,  0], p[1:,  1]
        angles = np.arctan2(y0s - y1s, x0s - x1s)
        angles[angles < 0] += 2 * np.pi
        zz = np.full(len(x0s), float(z), dtype=np.float32)
        rows.append(np.column_stack([x0s, y0s, x1s, y1s, angles, zz]))
    return np.concatenate(rows).astype(np.float32) if rows else np.empty((0, 6), dtype=np.float32)


def _angle_check_and_register(occup, angle_arr, edges, angle_thresh):
    """2エッジピクセルをベクトル化、3+エッジをループ処理して edgeAmbSet を返す。"""
    edgeEdgeSet = {(s, t): set() for s, t in edges}
    edgeAmbSet  = {}
    for s, t in edges:
        edgeAmbSet[(s, t)] = {t}
        edgeAmbSet[(t, s)] = {s}

    def _register(e1, e2, a):
        s1, t1 = edges[e1]; s2, t2 = edges[e2]
        edgeEdgeSet[(s1, t1)].add(e2); edgeEdgeSet[(s2, t2)].add(e1)
        if a < np.pi - a:
            edgeAmbSet[(s1, t1)].add(t2); edgeAmbSet[(t1, s1)].add(s2)
            edgeAmbSet[(t2, s2)].add(s1); edgeAmbSet[(s2, t2)].add(t1)
        else:
            edgeAmbSet[(s1, t1)].add(s2); edgeAmbSet[(t1, s1)].add(t2)
            edgeAmbSet[(t2, s2)].add(t1); edgeAmbSet[(s2, t2)].add(s1)

    count = occup.sum(axis=2)

    # 2エッジ → ベクトル化
    px2, py2 = np.where(count == 2)
    if len(px2) > 0:
        occ2 = occup[px2, py2, :]
        ang2 = angle_arr[px2, py2, :]
        _, nz_col = np.where(occ2 > 0.5)
        e_pairs = nz_col.reshape(-1, 2)
        e1s, e2s = e_pairs[:, 0], e_pairs[:, 1]
        a1s = ang2[np.arange(len(px2)), e1s]
        a2s = ang2[np.arange(len(px2)), e2s]
        diff  = np.minimum(np.mod(a1s - a2s, 2*np.pi), np.mod(a2s - a1s, 2*np.pi))
        hit   = np.minimum(diff, np.pi - diff) < angle_thresh
        for e1, e2, a in zip(e1s[hit].tolist(), e2s[hit].tolist(), diff[hit].tolist()):
            _register(e1, e2, a)

    # 3エッジ以上 → Python ループ（稀）
    px3, py3 = np.where(count > 2)
    for px, py in zip(px3.tolist(), py3.tolist()):
        zs = np.where(occup[px, py, :] > 0.5)[0].tolist()
        for ii in range(len(zs)):
            for jj in range(ii + 1, len(zs)):
                e1, e2 = zs[ii], zs[jj]
                a1, a2 = float(angle_arr[px, py, e1]), float(angle_arr[px, py, e2])
                a = min(np.mod(a1-a2, 2*np.pi), np.mod(a2-a1, 2*np.pi))
                if min(a, np.pi - a) < angle_thresh:
                    _register(e1, e2, a)

    return edgeAmbSet


def _compute_ambiguity_ratio(edgeAmbSet, edges, G, sp_cutoff):
    """edgeAmbSet から cutoff=1..sp_cutoff の ambiguity 値リストを返す。"""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path as csgraph_sp

    nodes = list(G.nodes())
    n = len(nodes)
    node_idx = {v: i for i, v in enumerate(nodes)}

    # BFS 距離行列（scipy CSR → C実装、純Python BFS より大幅高速）
    row_i, col_i = [], []
    for u, v in G.edges():
        i, j = node_idx[u], node_idx[v]
        row_i += [i, j]; col_i += [j, i]
    csr = csr_matrix(
        (np.ones(len(row_i), dtype=np.float32), (row_i, col_i)), shape=(n, n)
    )
    dist_mat = csgraph_sp(csr, directed=False, unweighted=True)  # (n,n), inf=非連結

    # 全 perceived-neighbor ペア (src_idx, tgt_idx) を収集
    src_list, tgt_list = [], []
    for s, t in edges:
        si, ti = node_idx[s], node_idx[t]
        for v in edgeAmbSet[(s, t)]:
            if v != s:
                src_list.append(si); tgt_list.append(node_idx[v])
        for v in edgeAmbSet[(t, s)]:
            if v != t:
                src_list.append(ti); tgt_list.append(node_idx[v])

    if not src_list:
        return [0.0] * sp_cutoff

    # 重複排除（vReachable/vUnreachable の set 意味論に対応）
    pairs   = np.unique(np.column_stack([src_list, tgt_list]), axis=0)
    src_arr = pairs[:, 0]
    tgt_arr = pairs[:, 1]
    dists   = dist_mat[src_arr, tgt_arr]   # (n_pairs,)

    result = []
    for cutoff in range(1, sp_cutoff + 1):
        reachable = dists <= float(cutoff)
        n_reach   = np.zeros(n, dtype=np.int64)
        n_unreach = np.zeros(n, dtype=np.int64)
        np.add.at(n_reach,   src_arr[reachable],  1)
        np.add.at(n_unreach, src_arr[~reachable], 1)
        total = n_reach + n_unreach
        ratio = np.where(total > 0, n_unreach / total, 0.0)
        result.append(float(ratio.mean()))
    return result


def _ambiguity_gpu(paths, pos, G, scale, angle_thresh_deg, sp_cutoff, cp):
    """GPU 実装: ラスタライズを CUDA カーネルで、畳み込みを cupyx.ndimage で処理。"""
    import cupyx.scipy.ndimage as cpnd

    angle_thresh = 2 * np.pi * angle_thresh_deg / 360.0
    edges = list(G.edges())
    n_edges = len(edges)

    all_x = np.concatenate([p[:, 0] for p in paths]) * scale
    all_y = np.concatenate([p[:, 1] for p in paths]) * scale
    w = int(np.ceil(all_x.max())) + 1
    h = int(np.ceil(all_y.max())) + 1
    W, H = w + 1, h + 1

    # ---- ラスタライズ（CUDA カーネル）----
    segs_np = _build_segments(paths, scale)
    n_segs  = len(segs_np)

    occup_g     = cp.zeros((W, H, n_edges), dtype=cp.float32)
    angle_arr_g = cp.zeros((W, H, n_edges), dtype=cp.float32)
    segs_g      = cp.asarray(segs_np)

    kernel = cp.RawKernel(_RASTERIZE_SRC, "rasterize")
    block  = 256
    grid   = (n_segs + block - 1) // block
    kernel(
        (grid,), (block,),
        (occup_g, angle_arr_g, segs_g,
         np.int32(n_segs), np.int32(W), np.int32(H), np.int32(n_edges))
    )

    # ---- 畳み込み（cupyx.ndimage）----
    kernel3d = cp.ones((3, 3, 1), dtype=cp.float32)
    c  = cpnd.convolve(occup_g,     kernel3d, mode="constant")
    c2 = cpnd.convolve(angle_arr_g, kernel3d, mode="constant")
    multi = c > 1
    c2[multi] /= c[multi]
    c[multi] = 1.0

    # ---- ノードマスク ----
    node_px = cp.clip(cp.asarray((pos[:, 0] * scale).astype(np.int32)), 0, w)
    node_py = cp.clip(cp.asarray((pos[:, 1] * scale).astype(np.int32)), 0, h)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx_ = cp.clip(node_px + dx, 0, w)
            ny_ = cp.clip(node_py + dy, 0, h)
            c[nx_, ny_, :]  = 0.0
            c2[nx_, ny_, :] = 0.0

    # CPU へ転送して角度チェック・ambiguity 計算
    occup_np     = c.get()
    angle_arr_np = c2.get()

    edgeAmbSet = _angle_check_and_register(occup_np, angle_arr_np, edges, angle_thresh)
    return _compute_ambiguity_ratio(edgeAmbSet, edges, G, sp_cutoff)


def _ambiguity_cpu(paths, pos, G, scale, angle_thresh_deg, sp_cutoff):
    """CPU フォールバック実装。"""
    from scipy.ndimage import convolve as ndimage_convolve

    angle_thresh = 2 * np.pi * angle_thresh_deg / 360.0
    edges = list(G.edges())
    n_edges = len(edges)

    all_x = np.concatenate([p[:, 0] for p in paths]) * scale
    all_y = np.concatenate([p[:, 1] for p in paths]) * scale
    w = int(np.ceil(all_x.max())) + 1
    h = int(np.ceil(all_y.max())) + 1

    occup     = np.zeros((w+1, h+1, n_edges), dtype=np.float32)
    angle_arr = np.zeros((w+1, h+1, n_edges), dtype=np.float32)

    for z, pts in enumerate(paths):
        xs = pts[:, 0] * scale
        ys = pts[:, 1] * scale
        for i in range(len(xs) - 1):
            x0, y0, x1, y1 = xs[i], ys[i], xs[i+1], ys[i+1]
            seg_len = max(1, int(np.ceil(np.hypot(x1-x0, y1-y0) / 0.4)))
            a = float(np.arctan2(y0-y1, x0-x1))
            if a < 0:
                a += 2 * np.pi
            px = np.clip(np.floor(np.linspace(x0, x1, seg_len)).astype(np.int32), 0, w)
            py = np.clip(np.floor(np.linspace(y0, y1, seg_len)).astype(np.int32), 0, h)
            keep = np.concatenate([[True], (np.diff(px) != 0) | (np.diff(py) != 0)])
            occup[px[keep], py[keep], z]     = 1.0
            angle_arr[px[keep], py[keep], z] = a

    kernel3d = np.ones((3, 3, 1), dtype=np.float32)
    c  = ndimage_convolve(occup,     kernel3d, mode="constant")
    c2 = ndimage_convolve(angle_arr, kernel3d, mode="constant")
    multi = c > 1
    c2[multi] /= c[multi]
    c[multi] = 1.0

    node_px = np.clip((pos[:, 0] * scale).astype(int), 0, w)
    node_py = np.clip((pos[:, 1] * scale).astype(int), 0, h)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx_ = np.clip(node_px + dx, 0, w)
            ny_ = np.clip(node_py + dy, 0, h)
            c[nx_, ny_, :]  = 0.0
            c2[nx_, ny_, :] = 0.0

    edgeAmbSet = _angle_check_and_register(c, c2, edges, angle_thresh)
    return _compute_ambiguity_ratio(edgeAmbSet, edges, G, sp_cutoff)


# ============================================================
# まとめて実行するユーティリティ
# ============================================================

def evaluate_all(
    paths: List[np.ndarray],
    pos: np.ndarray,
    G: nx.Graph,
    canvas: Tuple[int, int, int, int],
    scale: float = _SCALE,
    angle_thresh_deg: float = _ANGLE_DEG,
    sp_cutoff: int = _SP_CUTOFF,
    verbose: bool = True,
) -> dict:
    """
    ink_ratio / distortion / ambiguity をまとめて算出。

    Returns
    -------
    dict with keys: "ink_ratio", "distortion", "ambiguity"
    """
    if verbose:
        print("  Computing ink ratio...")
    ir = ink_ratio(paths, pos, G, canvas)

    if verbose:
        print("  Computing distortion...")
    dist = distortion(paths, pos, G)

    if verbose:
        print("  Computing ambiguity...")
    amb = ambiguity(paths, pos, G, scale=scale,
                    angle_thresh_deg=angle_thresh_deg, sp_cutoff=sp_cutoff)

    if verbose:
        print(f"  ink_ratio   : {ir:.4f}")
        print(f"  distortion  : mean={dist[1]:.4f}, std={dist[2]:.4f}")
        print(f"  ambiguity   : {[f'{v:.4f}' for v in amb]}")

    return {"ink_ratio": ir, "distortion": dist, "ambiguity": amb}
