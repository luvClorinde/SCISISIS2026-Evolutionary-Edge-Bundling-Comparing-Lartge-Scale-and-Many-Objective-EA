"""Utility class implementing GPU based metrics for edge bundling."""

from __future__ import annotations

import numpy as np
import cupy as cp

bresenham_kernel = cp.RawKernel(r'''
extern "C" __global__
void bresenham(const float* edges, int* grid, int width, int height, int edge_count, int nodes_per_edge, int cell_width, int cell_height, int total_segs) {
    int seg_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg_idx >= total_segs) return;

    int n_segs    = nodes_per_edge - 1;
    int edge_idx  = seg_idx / n_segs;
    int seg_in_edge = seg_idx % n_segs;

    const float* edge = &edges[edge_idx * nodes_per_edge * 2];
    int i = seg_in_edge;

    float x0 = edge[i * 2];
    float y0 = edge[i * 2 + 1];
    float x1 = edge[(i + 1) * 2];
    float y1 = edge[(i + 1) * 2 + 1];

    int ix0 = int(round(x0));
    int iy0 = int(round(y0));
    int ix1 = int(round(x1));
    int iy1 = int(round(y1));

    int dx = abs(ix1 - ix0);
    int dy = abs(iy1 - iy0);
    int sx = (ix0 < ix1) ? 1 : -1;
    int sy = (iy0 < iy1) ? 1 : -1;
    int err = dx - dy;

    bool skip_first = (seg_in_edge > 0);

    while (true) {
        if (!skip_first && ix0 >= 0 && ix0 < width && iy0 >= 0 && iy0 < height) {
            int gx = ix0 / cell_width;
            int gy = iy0 / cell_height;

            atomicAdd(&grid[gy * (width / cell_width) + gx], 1);
        }
        skip_first = false;

        if (ix0 == ix1 && iy0 == iy1) break;

        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; ix0 += sx; }
        if (e2 < dx) { err += dx; iy0 += sy; }
    }
}
''', 'bresenham')

bresenham_batch_kernel = cp.RawKernel(r'''
extern "C" __global__
void bresenham_batch(const float* edges, int* grid, int width, int height, int edge_count, int nodes_per_edge, int cell_width, int cell_height, int total_segs) {
    int seg_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg_idx >= total_segs) return;

    int n_segs = nodes_per_edge - 1;
    int segs_per_layout = edge_count * n_segs;
    int layout = seg_idx / segs_per_layout;
    int rem = seg_idx - layout * segs_per_layout;
    int edge_idx = rem / n_segs;
    int seg_in_edge = rem % n_segs;

    const float* edge = &edges[((long long)layout * edge_count + edge_idx) * nodes_per_edge * 2];
    int cells = (width / cell_width) * (height / cell_height);
    int* g = &grid[(long long)layout * cells];
    int i = seg_in_edge;

    float x0 = edge[i * 2];
    float y0 = edge[i * 2 + 1];
    float x1 = edge[(i + 1) * 2];
    float y1 = edge[(i + 1) * 2 + 1];

    int ix0 = int(round(x0));
    int iy0 = int(round(y0));
    int ix1 = int(round(x1));
    int iy1 = int(round(y1));

    int dx = abs(ix1 - ix0);
    int dy = abs(iy1 - iy0);
    int sx = (ix0 < ix1) ? 1 : -1;
    int sy = (iy0 < iy1) ? 1 : -1;
    int err = dx - dy;

    bool skip_first = (seg_in_edge > 0);

    while (true) {
        if (!skip_first && ix0 >= 0 && ix0 < width && iy0 >= 0 && iy0 < height) {
            int gx = ix0 / cell_width;
            int gy = iy0 / cell_height;

            atomicAdd(&g[gy * (width / cell_width) + gx], 1);
        }
        skip_first = false;

        if (ix0 == ix1 && iy0 == iy1) break;

        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; ix0 += sx; }
        if (e2 < dx) { err += dx; iy0 += sy; }
    }
}
''', 'bresenham_batch')

crossing_kernel = cp.RawKernel(r'''
extern "C" __device__
float orient(float ax, float ay, float bx, float by, float cx, float cy) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

extern "C" __device__
bool segments_cross(float ax0, float ay0, float ax1, float ay1,
                    float bx0, float by0, float bx1, float by1) {
    float o1 = orient(ax0, ay0, ax1, ay1, bx0, by0);
    float o2 = orient(ax0, ay0, ax1, ay1, bx1, by1);
    float o3 = orient(bx0, by0, bx1, by1, ax0, ay0);
    float o4 = orient(bx0, by0, bx1, by1, ax1, ay1);
    return (o1 * o2 < 0.0f) && (o3 * o4 < 0.0f);
}

extern "C" __global__
void count_crossings(const float* edges, int n_edges, int nodes_per_edge, int* out_counts) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;

    int cnt = 0;
    int npt = nodes_per_edge;
    int base1 = e * npt * 2;

    for (int other = e + 1; other < n_edges; ++other) {
        int base2 = other * npt * 2;
        for (int i = 0; i < npt - 1; ++i) {
            float ax0 = edges[base1 + i * 2];
            float ay0 = edges[base1 + i * 2 + 1];
            float ax1 = edges[base1 + (i + 1) * 2];
            float ay1 = edges[base1 + (i + 1) * 2 + 1];
            for (int j = 0; j < npt - 1; ++j) {
                float bx0 = edges[base2 + j * 2];
                float by0 = edges[base2 + j * 2 + 1];
                float bx1 = edges[base2 + (j + 1) * 2];
                float by1 = edges[base2 + (j + 1) * 2 + 1];
                if (segments_cross(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1)) {
                    cnt += 1;
                }
            }
        }
    }

    out_counts[e] = cnt;
}
''', 'count_crossings')

class MyFunc:
    def __init__(self, G, pos, c, param = (1000, 1000, 10, 10), grid_size=1, bound=None):
        self.n_n = len(G.nodes())
        self.n_e = len(G.edges())
        self.G = G
        self.pos = pos
        self.c = c
        self.xlim = param[0]
        self.ylim = param[1]
        self.x_show = param[2]
        self.y_show = param[3]
        self.size = (self.ylim, self.xlim)
        self.grid_size = grid_size
        self.grid = cp.zeros((self.size[0] // self.grid_size, self.size[1] // self.grid_size), dtype=cp.int32)
        self.grid_f = cp.zeros_like(self.grid, dtype=cp.float32)
        self._threads_per_block = 512
        self.bound = bound
        self.pq_max: float | None = None
        self.edd_straight: float | None = None

    def total_Length(self, pos_np):
        """
        総延長を比較する
        """
        pos_tl = pos_np - np.roll(pos_np, 1, axis=1)
        pos_tl = np.sqrt(pos_tl[:, :, 0] ** 2 + pos_tl[:, :, 1] ** 2)
        return pos_tl[:, 1:].sum()


    def meld(self, pos_np, nomalization=True):
        """
        pos_np:
            改良版の座標配列。
        """
        direct = pos_np[:, 0] - pos_np[:, -1]
        direct_len = cp.sqrt(direct[:, 0] ** 2 + direct[:, 1] ** 2)
        diff = pos_np[:, 1:] - pos_np[:, :-1]
        seg_len = cp.sqrt(diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2)
        seg_sum = cp.sum(seg_len, axis=1)
        if nomalization:
            # 迂回割合 (経路長に占める余剰分)。direct_len <= seg_sum なので [0, 1) に収まる
            pos_meld = cp.abs(1 - direct_len / seg_sum)
        else:
            pos_meld = cp.abs(seg_sum - direct_len)
        return pos_meld.sum() / self.n_e
    
    def meld_batch(self, pos: cp.ndarray, nomalization=True) -> cp.ndarray:
        """meld のバッチ版。pos: (B, n_e, n_pt, 2) → (B,)

        注: サイズ2〜4程度の最終軸への cupy sum(axis=-1) は極端に遅い縮約
        カーネルに落ちる（air 実測 12ms 対 0.3ms）ため、座標軸・セグメント軸
        の縮約は要素演算と逐次加算で展開している（相対誤差 ~1e-16 のみ）。
        """
        direct = pos[..., 0, :] - pos[..., -1, :]
        direct_len = cp.sqrt(direct[..., 0] ** 2 + direct[..., 1] ** 2)
        diff = pos[..., 1:, :] - pos[..., :-1, :]
        seg = cp.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)  # (B, n_e, n_pt-1)
        seg_sum = seg[..., 0]
        for k in range(1, seg.shape[-1]):
            seg_sum = seg_sum + seg[..., k]
        if nomalization:
            pos_meld = cp.abs(1 - direct_len / seg_sum)
        else:
            pos_meld = cp.abs(seg_sum - direct_len)
        return pos_meld.sum(axis=-1) / self.n_e

    def _edd(self, pos_np: np.ndarray) -> cp.ndarray:
        """Calculate edge density using a GPU-based Bresenham kernel."""
        grid = self.grid
        grid.fill(0)
        edges_flat = pos_np.astype(cp.float32).ravel()
        _, num_points, _ = pos_np.shape
        n_segs = self.n_e * (num_points - 1)
        blocks = (n_segs + self._threads_per_block - 1) // self._threads_per_block
        bresenham_kernel(
            (blocks,),
            (self._threads_per_block,),
            (
                edges_flat,
                grid,
                self.size[1],
                self.size[0],
                self.n_e,
                num_points,
                self.grid_size,
                self.grid_size,
                n_segs,
            ),
        )
        return grid

    def _edd_batch(self, pos: cp.ndarray) -> cp.ndarray:
        """_edd のバッチ版。pos: (B, n_e, n_pt, 2) → grids (B, gh, gw)"""
        batch, _, num_points, _ = pos.shape
        gh, gw = self.grid.shape
        grids = cp.zeros((batch, gh, gw), dtype=cp.int32)
        edges_flat = pos.astype(cp.float32).ravel()
        total_segs = batch * self.n_e * (num_points - 1)
        blocks = (total_segs + self._threads_per_block - 1) // self._threads_per_block
        bresenham_batch_kernel(
            (blocks,),
            (self._threads_per_block,),
            (
                edges_flat,
                grids,
                self.size[1],
                self.size[0],
                self.n_e,
                num_points,
                self.grid_size,
                self.grid_size,
                total_segs,
            ),
        )
        return grids

    def _moa(self, grid: cp.ndarray) -> float:
        """Calculate mean occupancy ratio of the density grid."""
        return cp.count_nonzero(grid) / grid.size

    def _build_straight_pos(self) -> cp.ndarray:
        """全エッジの制御点を端点間の等分内挿に置いた直線配置（非バンドル基準）。"""
        edges = list(self.G.edges())
        pos_straight = np.empty((self.n_e, self.c + 2, 2))
        for i, (u, v) in enumerate(edges):
            pu = self.pos[u]
            pv = self.pos[v]
            for k in range(self.c + 2):
                t = k / (self.c + 1)
                pos_straight[i, k] = pu + t * (pv - pu)
        return cp.array(pos_straight)

    def _compute_edd_straight(self) -> float:
        """Grid-density variance of the straight-line (unbundled) configuration.

        Used as the anchor of the soft normalisation
        edd2 / (edd2 + edd_straight): the straight state maps to 0.5, more
        concentrated ink approaches 1, more uniform ink falls below 0.5.
        Anchoring at an attainable reference keeps the normalised values in a
        usable range, unlike the previous unattainable theoretical-max
        denominator (zig-zag ink concentrated in one cell) which squashed
        everything to ~1e-6.
        """
        grid_f = self._edd(self._build_straight_pos()).astype(cp.float32)
        grid_f -= grid_f.mean()
        return float(cp.sum(cp.square(grid_f)) / self.size[0] / self.size[1])

    def moa_edd(self, pos_np: np.ndarray, normalization=True) -> tuple[float, float]:
        """Return mean occupancy and squared edge density for positions."""
        if normalization and self.bound is not None and self.edd_straight is None:
            self.edd_straight = self._compute_edd_straight()
        grid = self._edd(pos_np)
        moa = self._moa(grid)
        cp.copyto(self.grid_f, grid, casting='unsafe')
        self.grid_f -= cp.mean(self.grid_f)
        edd2 = cp.sum(cp.square(self.grid_f)) / self.size[0] / self.size[1]
        if normalization and self.bound is not None and self.edd_straight > 0:
            edd2 = edd2 / (edd2 + self.edd_straight)
        return moa, edd2

    def moa_edd_batch(self, pos: cp.ndarray, normalization=True) -> tuple[cp.ndarray, cp.ndarray]:
        """moa_edd のバッチ版。pos: (B, n_e, n_pt, 2) → (moa (B,), edd2 (B,))"""
        if normalization and self.bound is not None and self.edd_straight is None:
            self.edd_straight = self._compute_edd_straight()
        grids = self._edd_batch(pos)
        batch = grids.shape[0]
        flat = grids.reshape(batch, -1)
        moa = cp.count_nonzero(flat, axis=1) / flat.shape[1]
        grid_f = flat.astype(cp.float32)
        grid_f -= grid_f.mean(axis=1, keepdims=True)
        edd2 = cp.sum(cp.square(grid_f), axis=1) / self.size[0] / self.size[1]
        if normalization and self.bound is not None and self.edd_straight > 0:
            edd2 = edd2 / (edd2 + self.edd_straight)
        return moa, edd2

    def _pq_core(self, pos_n: cp.ndarray) -> cp.floating:
        """Raw zig-zag sum without normalization."""
        diff = pos_n[:, 1:] - pos_n[:, :-1]
        A = cp.arctan2(diff[:, :, 1], diff[:, :, 0])
        A -= A[:, [0]]
        delta = A[:, 1:] - A[:, :-1]
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        gamma = cp.sign(delta[:, 1:]) != cp.sign(delta[:, :-1])
        values = gamma.astype(cp.float64) * cp.abs(delta[:, 1:])
        values = cp.where(cp.abs(values) < 1e-14, 0.0, values)
        return cp.sum(values)

    def _build_pq_max_pos(self) -> cp.ndarray:
        """Build alternating ±bound CP configuration that maximizes PQ."""
        edges = list(self.G.edges())
        pos_max = np.empty((self.n_e, self.c + 2, 2))
        for i, (u, v) in enumerate(edges):
            pu = self.pos[u]
            pv = self.pos[v]
            edge_vec = pv - pu
            edge_len = np.linalg.norm(edge_vec)
            perp = np.array([-edge_vec[1], edge_vec[0]])
            if edge_len > 1e-12:
                perp /= edge_len
            pos_max[i, 0] = pu
            pos_max[i, -1] = pv
            for k in range(1, self.c + 1):
                t = k / (self.c + 1)
                sign = 1 if k % 2 == 1 else -1
                pos_max[i, k] = pu + t * edge_vec + sign * self.bound * perp
        return cp.array(pos_max)

    def path_quality(self, pos_n: np.ndarray, normalization=True) -> float:
        """Return zig-zag metric of the given control points."""
        raw = self._pq_core(pos_n)
        if normalization and self.bound is not None:
            if self.pq_max is None:
                self.pq_max = float(self._pq_core(self._build_pq_max_pos()))
            if self.pq_max > 0:
                return raw / self.pq_max
        return raw

    def _pq_core_batch(self, pos: cp.ndarray) -> cp.ndarray:
        """_pq_core のバッチ版。pos: (B, n_e, n_pt, 2) → (B,)"""
        diff = pos[..., 1:, :] - pos[..., :-1, :]
        A = cp.arctan2(diff[..., 1], diff[..., 0])
        A = A - A[..., :1]
        delta = A[..., 1:] - A[..., :-1]
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        gamma = cp.sign(delta[..., 1:]) != cp.sign(delta[..., :-1])
        values = gamma.astype(cp.float64) * cp.abs(delta[..., 1:])
        values = cp.where(cp.abs(values) < 1e-14, 0.0, values)
        return values.sum(axis=(-2, -1))

    def path_quality_batch(self, pos: cp.ndarray, normalization=True) -> cp.ndarray:
        """path_quality のバッチ版。pos: (B, n_e, n_pt, 2) → (B,)"""
        raw = self._pq_core_batch(pos)
        if normalization and self.bound is not None:
            if self.pq_max is None:
                self.pq_max = float(self._pq_core(self._build_pq_max_pos()))
            if self.pq_max > 0:
                return raw / self.pq_max
        return raw


    def crossing_count_cuda(self, pos_np: np.ndarray) -> int:
        """Count edge intersections using a CUDA kernel.

        Parameters
        ----------
        pos_np : np.ndarray
            Edge coordinate array of shape ``(n_e, c+2, 2)``.
        """
        edges_flat = cp.array(pos_np, dtype=cp.float32).reshape(-1)
        counts = cp.zeros(self.n_e, dtype=cp.int32)
        _, num_points, _ = pos_np.shape
        blocks = (self.n_e + self._threads_per_block - 1) // self._threads_per_block
        crossing_kernel(
            (blocks,),
            (self._threads_per_block,),
            (
                edges_flat,
                self.n_e,
                num_points,
                counts,
            ),
        )
        return int(cp.sum(counts).get())

    def crossing_count_sweepline(self, pos_np: np.ndarray) -> int:
        """Count edge crossings using a simple sweep line algorithm.

        This implementation avoids checking all edge pairs by only
        comparing segments whose ``x`` ranges overlap. It yields the
        same result as :meth:`crossing_count` but typically runs faster
        than the brute-force approach.

        Parameters
        ----------
        pos_np : np.ndarray
            Edge coordinate array of shape ``(n_e, c+2, 2)``.
        """

        def _orient(ax, ay, bx, by, cx, cy):
            return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

        def _intersects(s1, s2) -> bool:
            a1x, a1y, a2x, a2y, e1 = s1
            b1x, b1y, b2x, b2y, e2 = s2
            if e1 == e2:
                return False
            o1 = _orient(a1x, a1y, a2x, a2y, b1x, b1y)
            o2 = _orient(a1x, a1y, a2x, a2y, b2x, b2y)
            if o1 * o2 >= 0:
                return False
            o3 = _orient(b1x, b1y, b2x, b2y, a1x, a1y)
            o4 = _orient(b1x, b1y, b2x, b2y, a2x, a2y)
            return o3 * o4 < 0

        segs = []
        n_e, n_pt, _ = pos_np.shape
        for e in range(n_e):
            for i in range(n_pt - 1):
                x0, y0 = pos_np[e, i]
                x1, y1 = pos_np[e, i + 1]
                segs.append([x0, y0, x1, y1, e])

        events = []
        for idx, (x0, y0, x1, y1, _) in enumerate(segs):
            events.append((min(x0, x1), 0, idx))
            events.append((max(x0, x1), 1, idx))
        events.sort()

        active: list[int] = []
        count = 0
        for _, typ, idx in events:
            if typ == 0:  # segment enters sweep line
                for j in active:
                    if _intersects(segs[idx], segs[j]):
                        count += 1
                active.append(idx)
            else:  # segment leaves sweep line
                if idx in active:
                    active.remove(idx)
        return count
    
    # ---------------------------
    # Bezier helpers / wrappers
    # ---------------------------
    def sample_bezier(self, ctrl_pts: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """
        Sample Bezier curves for each edge from control points.

        Parameters
        ----------
        ctrl_pts : np.ndarray
            Control points array shape (n_e, m, 2)
        n_samples : int
            Number of samples per edge

        Returns
        -------
        np.ndarray
            Sampled points shape (n_e, n_samples, 2)
        """
        n_e, m, _ = ctrl_pts.shape
        t = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
        n = m - 1
        # Bernstein basis (m, n_samples)
        from math import comb
        B = np.vstack([comb(n, k) * (t**k) * ((1 - t)**(n - k)) for k in range(n + 1)])  # (m, n_samples)
        samples = np.empty((n_e, n_samples, 2), dtype=np.float64)
        for e in range(n_e):
            # ctrl_pts[e] -> (m,2), produce (2, n_samples) then transpose
            curve = ctrl_pts[e].T @ B  # (2, n_samples)
            samples[e] = curve.T
        return samples
