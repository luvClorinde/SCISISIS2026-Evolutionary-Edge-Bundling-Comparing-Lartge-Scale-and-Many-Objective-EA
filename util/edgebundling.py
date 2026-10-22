from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

_COMPAT_KERNEL_SRC = r"""
extern "C" __global__ void compute_compat(
    const float* src,
    const float* tgt,
    const float* vecs,
    const float* lens,
    const float* mids,
    const float* norms,
    const float* lens2,
    bool*        mask,
    const int    n,
    const float  threshold
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;

    if (i == j) { mask[i * n + j] = false; return; }

    const float eps = 1e-10f;

    /* Angle */
    float angle = fabsf(norms[i*2]*norms[j*2] + norms[i*2+1]*norms[j*2+1]);

    /* Scale */
    float li = lens[i], lj = lens[j];
    float lmin = fminf(li, lj), lmax = fmaxf(li, lj);
    float scale = 2.0f / (lmax/fmaxf(lmin,eps) + lmin/fmaxf(lmax,eps));

    /* Position */
    float dmx = mids[i*2] - mids[j*2], dmy = mids[i*2+1] - mids[j*2+1];
    float d = sqrtf(dmx*dmx + dmy*dmy);
    float position = (li + lj) / (li + lj + 2.0f * d);

    /* Visibility ij: project edge-j endpoints onto edge-i */
    float abx = vecs[i*2], aby = vecs[i*2+1], ab2 = lens2[i];
    float asx = src[j*2]-src[i*2], asy = src[j*2+1]-src[i*2+1];
    float atx = tgt[j*2]-src[i*2], aty = tgt[j*2+1]-src[i*2+1];
    float ts = (asx*abx + asy*aby) / fmaxf(ab2, eps);
    float tt = (atx*abx + aty*aby) / fmaxf(ab2, eps);
    float p0x = src[i*2]+ts*abx, p0y = src[i*2+1]+ts*aby;
    float p1x = src[i*2]+tt*abx, p1y = src[i*2+1]+tt*aby;
    float mx  = (p0x+p1x)*0.5f,  my  = (p0y+p1y)*0.5f;
    float den = sqrtf((p1x-p0x)*(p1x-p0x) + (p1y-p0y)*(p1y-p0y));
    float num = 2.0f*sqrtf((mx-mids[i*2])*(mx-mids[i*2]) + (my-mids[i*2+1])*(my-mids[i*2+1]));
    float vis_ij = (den < eps) ? 0.0f : fmaxf(0.0f, 1.0f - num/den);

    /* Visibility ji: project edge-i endpoints onto edge-j */
    float bjx = vecs[j*2], bjy = vecs[j*2+1], ab2j = lens2[j];
    float bsx = src[i*2]-src[j*2], bsy = src[i*2+1]-src[j*2+1];
    float btx = tgt[i*2]-src[j*2], bty = tgt[i*2+1]-src[j*2+1];
    float ts2 = (bsx*bjx + bsy*bjy) / fmaxf(ab2j, eps);
    float tt2 = (btx*bjx + bty*bjy) / fmaxf(ab2j, eps);
    float q0x = src[j*2]+ts2*bjx, q0y = src[j*2+1]+ts2*bjy;
    float q1x = src[j*2]+tt2*bjx, q1y = src[j*2+1]+tt2*bjy;
    float mx2  = (q0x+q1x)*0.5f,  my2  = (q0y+q1y)*0.5f;
    float den2 = sqrtf((q1x-q0x)*(q1x-q0x) + (q1y-q0y)*(q1y-q0y));
    float num2 = 2.0f*sqrtf((mx2-mids[j*2])*(mx2-mids[j*2]) + (my2-mids[j*2+1])*(my2-mids[j*2+1]));
    float vis_ji = (den2 < eps) ? 0.0f : fmaxf(0.0f, 1.0f - num2/den2);

    float compat = angle * scale * position * (vis_ij + vis_ji) * 0.5f;
    mask[i * n + j] = (compat >= threshold);
}
"""


@dataclass(frozen=True)
class FDEBParameters:
    """
    論文 Holten & van Wijk (2009) のデフォルト値に準拠。
    cycles 回のサイクルで subdivision → force simulation を繰り返す。
    各サイクル後: step_size *= step_decay, iterations *= iteration_rate
    """
    compatibility_threshold: float = 0.6   # 互換ペアの閾値
    cycles: int = 6                         # サイクル数 C
    iterations: int = 60                    # 初期反復数 I（サイクル毎に減衰）
    iteration_rate: float = 2 / 3          # 反復数の減衰率（論文値）
    step_size: float = 0.04                 # 初期ステップサイズ S（論文値）
    step_decay: float = 0.5                 # ステップサイズ減衰率（論文値: ×0.5）
    K: float = 0.1                          # バネ定数（論文値）

    @property
    def n_control(self) -> int:
        """cycles 回の細分化後の内部制御点数 = 2**cycles - 1"""
        return 2 ** self.cycles - 1


class ForceEdgeBundling:
    """
    Force-Directed Edge Bundling (Holten & van Wijk, 2009) の NumPy 実装。
    論文通り、各サイクルで subdivision → force simulation を実施。

    Usage
    -----
    params = FDEBParameters()
    bundler = ForceEdgeBundling(params)
    pts = bundler.bundle(pos, edges)   # (n_edges, 2**cycles + 1, 2)
    """

    def __init__(self, params: FDEBParameters = FDEBParameters()):
        self.p = params

    # ------------------------------------------------------------------
    # 互換リスト構築（全ペアを NumPy でベクトル化）
    # ------------------------------------------------------------------

    def _build_compat_lists(
        self, pos: np.ndarray, edges: np.ndarray
    ) -> List[List[int]]:
        """
        互換リストを CUDA カーネルで構築。
        n×n ペアを全スレッド並列に1パスで計算し、中間配列を生成しない。
        """
        import cupy as cp

        kernel = cp.RawKernel(_COMPAT_KERNEL_SRC, "compute_compat")

        n = len(edges)
        src_np   = pos[edges[:, 0]].astype(np.float32)
        tgt_np   = pos[edges[:, 1]].astype(np.float32)
        vecs_np  = (tgt_np - src_np)
        lens_np  = np.linalg.norm(vecs_np, axis=1).astype(np.float32)
        mids_np  = ((src_np + tgt_np) / 2).astype(np.float32)
        norms_np = (vecs_np / np.maximum(lens_np[:, None], 1e-10)).astype(np.float32)
        lens2_np = (lens_np ** 2).astype(np.float32)

        src_g   = cp.asarray(src_np)
        tgt_g   = cp.asarray(tgt_np)
        vecs_g  = cp.asarray(vecs_np.astype(np.float32))
        lens_g  = cp.asarray(lens_np)
        mids_g  = cp.asarray(mids_np)
        norms_g = cp.asarray(norms_np)
        lens2_g = cp.asarray(lens2_np)
        mask_g  = cp.empty((n, n), dtype=cp.bool_)

        block = 32
        grid  = ((n + block - 1) // block, (n + block - 1) // block)
        kernel(
            grid, (block, block),
            (src_g, tgt_g, vecs_g, lens_g, mids_g, norms_g, lens2_g,
             mask_g, np.int32(n), np.float32(self.p.compatibility_threshold))
        )

        mask_np = mask_g.get()
        return [list(np.where(mask_np[i])[0]) for i in range(n)]

    # ------------------------------------------------------------------
    # 分割・力適用 (GPU)
    # ------------------------------------------------------------------

    @staticmethod
    def _subdivide(pts):
        """(n, m, 2) → (n, 2m-1, 2): 隣接点の中点を挿入。cupy/numpy 両対応。"""
        import cupy as cp
        xp = cp.get_array_module(pts)
        out = xp.empty((pts.shape[0], 2 * pts.shape[1] - 1, 2), dtype=pts.dtype)
        out[:, 0::2, :] = pts
        out[:, 1::2, :] = (pts[:, :-1, :] + pts[:, 1:, :]) / 2.0
        return out

    @staticmethod
    def _build_W(compat_lists: List[List[int]], n: int):
        """compat_lists から正規化済み疎引力行列 W と has_compat マスクを構築。"""
        import cupy as cp
        import cupyx.scipy.sparse as cpsparse

        rows, cols, data = [], [], []
        for i, compat in enumerate(compat_lists):
            if compat:
                w = 1.0 / len(compat)
                for j in compat:
                    rows.append(i)
                    cols.append(j)
                    data.append(w)

        if rows:
            W = cpsparse.csr_matrix(
                (cp.array(data, dtype=np.float64),
                 (cp.array(rows, dtype=np.int32), cp.array(cols, dtype=np.int32))),
                shape=(n, n),
            )
        else:
            W = cpsparse.csr_matrix((n, n), dtype=np.float64)

        has_compat = cp.array([len(c) > 0 for c in compat_lists])  # (n,) bool
        return W, has_compat

    @staticmethod
    def _apply_forces_gpu(pts_g, W, has_compat, pos_g, edges_g, K: float, step: float):
        new = pts_g.copy()
        if pts_g.shape[1] <= 2:
            return new

        n, m, _ = pts_g.shape

        # バネ力
        target = (pts_g[:, :-2, :] + pts_g[:, 2:, :]) / 2.0
        new[:, 1:-1, :] += (target - pts_g[:, 1:-1, :]) * K * step

        # 引力: W @ pts_inner で全エッジの平均位置を一括計算
        # has_compat が False の行は W が零行 → attracted=0 になるため
        # (0 - pts_inner) という誤った力が生じる。マスクで無効化する。
        pts_inner_flat = pts_g[:, 1:-1, :].reshape(n, -1)        # (n, (m-2)*2)
        attracted = W @ pts_inner_flat                             # (n, (m-2)*2)
        delta = (attracted - pts_inner_flat).reshape(n, m - 2, 2)
        delta[~has_compat] = 0.0
        new[:, 1:-1, :] += step * delta

        # 端点を固定
        new[:, 0,  :] = pos_g[edges_g[:, 0]]
        new[:, -1, :] = pos_g[edges_g[:, 1]]
        return new

    # ------------------------------------------------------------------
    # 公開インターフェース
    # ------------------------------------------------------------------

    def bundle(
        self,
        pos: np.ndarray,
        edges: np.ndarray,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        pos     : (n_nodes, 2)  ノード座標
        edges   : (n_edges, 2)  エッジ (int インデックス)
        verbose : 進捗を標準出力に表示するか

        Returns
        -------
        np.ndarray  shape (n_edges, 2**cycles + 1, 2)
        """
        # 座標を [0,1] に正規化（力の大きさを座標スケールに依存させない）
        pos_min = pos.min(axis=0)
        pos_scale = max((pos.max(axis=0) - pos_min).max(), 1e-10)
        pos_norm = (pos - pos_min) / pos_scale

        import cupy as cp

        if verbose:
            print("  Building compatibility lists...")
        compat_lists = self._build_compat_lists(pos_norm, edges)
        if verbose:
            n_compat = sum(len(c) for c in compat_lists)
            print(f"  Compatible pairs: {n_compat // 2}")

        if verbose:
            print("  Building W matrix...")
        W, has_compat = self._build_W(compat_lists, len(edges))

        # GPU へ転送
        pos_g   = cp.asarray(pos_norm)
        edges_g = cp.asarray(edges)
        pts_g   = cp.stack([pos_g[edges_g[:, 0]], pos_g[edges_g[:, 1]]], axis=1)

        step = self.p.step_size
        iters = float(self.p.iterations)

        for c in range(self.p.cycles):
            # 1. 細分化（サイクル毎に制御点を倍増）
            pts_g = self._subdivide(pts_g)

            if verbose:
                n_pts = pts_g.shape[1]
                print(f"  Cycle {c + 1}/{self.p.cycles} "
                      f"(pts={n_pts}, iters={round(iters)}, step={step:.4f})...",
                      end=" ", flush=True)

            # 2. 力シミュレーション（全反復を GPU 上で実行）
            for _ in range(round(iters)):
                pts_g = self._apply_forces_gpu(
                    pts_g, W, has_compat, pos_g, edges_g, self.p.K, step
                )

            # 3. パラメータ減衰（論文値）
            step  *= self.p.step_decay
            iters *= self.p.iteration_rate

            if verbose:
                print("done")

        # CPU へ転送して元のスケールに戻す
        pts = pts_g.get()
        return pts * pos_scale + pos_min

    def show_bundled(
        self,
        pts: np.ndarray,
        canvas: Tuple[int, int, int, int],
        save_path: Optional[str] = None,
        edge_color: str = "black",
        alpha: float = 0.15,
        edge_width: float = 1.0,
        bg_color: str = "white",
        dpi: int = 300,
        n_samples: int = 100,
    ) -> None:
        """
        FDEB出力を PCHIP で描画する。
        EdgeBundlingModule の cluster_control_points を経由しない。

        Parameters
        ----------
        pts    : bundle() の戻り値 (n_edges, n_pts, 2)
        canvas : (xlim, ylim, x_show, y_show)
        """
        xlim, ylim, x_show, y_show = canvas
        ratio = xlim / ylim
        figsize = (x_show, x_show / ratio) if ratio >= 1 else (x_show * ratio, y_show)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=bg_color)
        ax.set_xlim(0, xlim)
        ax.set_ylim(ylim, 0)
        ax.axis("off")
        ax.set_facecolor(bg_color)

        m = pts.shape[1]
        ar = np.arange(m)
        t = np.linspace(0, m - 1, n_samples)

        for edge in pts:
            cs_x = PchipInterpolator(ar, edge[:, 0])
            cs_y = PchipInterpolator(ar, edge[:, 1])
            ax.plot(cs_x(t), cs_y(t), "-", color=edge_color, alpha=alpha, linewidth=edge_width)

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
            plt.close()


# ===========================================================================
# Edge-Path Bundling  (Wallinger et al., IEEE TVCG 2021)
# ===========================================================================

@dataclass(frozen=True)
class EPBParameters:
    k: float = 2.0    # 迂回許容比率: path_length <= k × euclidean_distance
    d: float = 2.0    # エッジ重みの指数: weight = dist^d
    smoothing: int = 2  # 中点挿入の回数（1 = 挿入なし）


class EdgePathBundling:
    """
    Edge-Path Bundling (Wallinger et al., IEEE TVCG 2021).

    Greedy algorithm: processes edges in descending weight order.
    For each edge, finds a shortest alternative path (Dijkstra on the original
    graph minus the current edge). Locks path edges so they serve as routing
    substrate but are not re-processed as source edges.

    Usage
    -----
    bundler = EdgePathBundling(EPBParameters())
    paths = bundler.bundle(pos, G)   # List[np.ndarray], one (n_pts, 2) per edge
    bundler.show_bundled(paths, canvas, save_path="out.png")
    """

    def __init__(self, params: EPBParameters = EPBParameters()):
        self.p = params

    def bundle(self, pos: np.ndarray, G) -> List[np.ndarray]:
        """
        Parameters
        ----------
        pos : (n_nodes, 2)  node coordinates (integer-indexed)
        G   : networkx.Graph

        Returns
        -------
        List[np.ndarray]: one (n_pts, 2) polyline per edge (same order as G.edges())
        """
        import scipy.sparse as sp
        import scipy.sparse.csgraph as csgraph

        p = self.p
        n_nodes = pos.shape[0]
        _INF = 1e18

        # Pre-build CSR (both directions) with weight = dist^d
        rows, cols, data = [], [], []
        edge_dist: dict = {}

        for u, v in G.edges():
            dist = float(np.linalg.norm(pos[u] - pos[v]))
            w = dist ** p.d
            rows += [u, v]
            cols += [v, u]
            data += [w, w]
            edge_dist[(min(u, v), max(u, v))] = dist

        csr = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        csr.sort_indices()

        # Pre-compute data-array index for each directed edge (u→v)
        edge_data_idx: dict = {}
        for u, v in G.edges():
            for a, b in [(u, v), (v, u)]:
                start = int(csr.indptr[a])
                offset = int(np.searchsorted(csr.indices[start:int(csr.indptr[a + 1])], b))
                edge_data_idx[(a, b)] = start + offset

        # Sort edges by weight descending (heavy edges bundled first)
        edges_sorted = sorted(
            G.edges(),
            key=lambda e: edge_dist[(min(e[0], e[1]), max(e[0], e[1]))] ** p.d,
            reverse=True,
        )

        lock_set: set = set()
        bundled: dict = {}

        for u, v in edges_sorted:
            edge_key = (min(u, v), max(u, v))
            if edge_key in lock_set:
                continue

            # Temporarily exclude this edge from routing
            idx_uv = edge_data_idx[(u, v)]
            idx_vu = edge_data_idx[(v, u)]
            saved_uv = csr.data[idx_uv]
            saved_vu = csr.data[idx_vu]
            csr.data[idx_uv] = _INF
            csr.data[idx_vu] = _INF

            dists, pred = csgraph.dijkstra(
                csr, directed=True, indices=u, return_predecessors=True
            )

            if np.isinf(dists[v]):
                csr.data[idx_uv] = saved_uv
                csr.data[idx_vu] = saved_vu
                continue

            # Reconstruct node path from predecessor array
            node_path = []
            cur = v
            valid = True
            while cur != u:
                node_path.append(int(cur))
                cur = pred[cur]
                if cur < 0:
                    valid = False
                    break
            if not valid:
                csr.data[idx_uv] = saved_uv
                csr.data[idx_vu] = saved_vu
                continue
            node_path.append(int(u))
            node_path.reverse()

            # Detour check against euclidean distances
            path_len = sum(
                float(np.linalg.norm(pos[node_path[i]] - pos[node_path[i + 1]]))
                for i in range(len(node_path) - 1)
            )
            if path_len > p.k * edge_dist[edge_key]:
                csr.data[idx_uv] = saved_uv
                csr.data[idx_vu] = saved_vu
                continue

            # Success: lock path edges (still traversable, excluded from outer loop)
            for i in range(len(node_path) - 1):
                a, b = min(node_path[i], node_path[i + 1]), max(node_path[i], node_path[i + 1])
                lock_set.add((a, b))

            bundled[edge_key] = node_path
            # Edge (u,v) stays at INF in CSR (permanently skipped)

        # Build output polylines
        result = []
        for u, v in G.edges():
            edge_key = (min(u, v), max(u, v))
            pts = pos[np.array(bundled[edge_key])] if edge_key in bundled else pos[np.array([u, v])]
            for _ in range(p.smoothing - 1):
                pts = self._insert_midpoints(pts)
            result.append(pts)

        return result

    @staticmethod
    def _insert_midpoints(pts: np.ndarray) -> np.ndarray:
        """Insert midpoint between every consecutive pair of points."""
        out = np.empty((2 * len(pts) - 1, 2), dtype=pts.dtype)
        out[0::2] = pts
        out[1::2] = (pts[:-1] + pts[1:]) / 2.0
        return out

    @staticmethod
    def _eval_bezier(ctrl: np.ndarray, n_samples: int) -> np.ndarray:
        """
        de Casteljau algorithm, vectorized over t.
        ctrl : (m, 2) control points
        returns (n_samples, 2) points on the Bezier curve
        """
        t = np.linspace(0.0, 1.0, n_samples)          # (S,)
        p = np.tile(ctrl[np.newaxis], (n_samples, 1, 1)).astype(float)  # (S, m, 2)
        t3 = t[:, np.newaxis, np.newaxis]              # (S, 1, 1)
        while p.shape[1] > 1:
            p = (1.0 - t3) * p[:, :-1] + t3 * p[:, 1:]
        return p[:, 0, :]                              # (S, 2)

    def show_bundled(
        self,
        paths: List[np.ndarray],
        canvas: Tuple[int, int, int, int],
        save_path: Optional[str] = None,
        edge_color: str = "black",
        alpha: float = 0.15,
        edge_width: float = 1.0,
        bg_color: str = "white",
        dpi: int = 300,
        n_samples: int = 100,
    ) -> None:
        """
        Parameters
        ----------
        paths  : bundle() の戻り値 List[np.ndarray]
        canvas : (xlim, ylim, x_show, y_show)
        """
        xlim, ylim, x_show, y_show = canvas
        ratio = xlim / ylim
        figsize = (x_show, x_show / ratio) if ratio >= 1 else (x_show * ratio, y_show)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=bg_color)
        ax.set_xlim(0, xlim)
        ax.set_ylim(ylim, 0)
        ax.axis("off")
        ax.set_facecolor(bg_color)

        for pts in paths:
            if len(pts) < 2:
                continue
            if len(pts) == 2:
                ax.plot(pts[:, 0], pts[:, 1], "-",
                        color=edge_color, alpha=alpha, linewidth=edge_width)
            else:
                curve = self._eval_bezier(pts, n_samples)
                ax.plot(curve[:, 0], curve[:, 1], "-",
                        color=edge_color, alpha=alpha, linewidth=edge_width)

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
            plt.close()
