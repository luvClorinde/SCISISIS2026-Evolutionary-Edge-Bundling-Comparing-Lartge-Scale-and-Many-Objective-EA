import networkx as nx
import numpy as np
import cupy as cp
from os import PathLike
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Sequence, Union, Iterable
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from math import sin, cos, comb
from util.compatibility import compatibility_gpu as comp

PathInput = Union[str, Path, PathLike[str]]

# --- Bezier sampling helper moved out of the class (module level) ---


class EdgeBundlingModule:
    """
    ネットワーク可視化と制御点最適化をまとめるクラス
    """
    def __init__(
        self,
        G: nx.Graph,
        pos: np.ndarray,
        n_control: int,
        canvas: Tuple[int, int, int, int] = (1000, 1000, 10, 10),
        n_samples: int = 10
    ):
        """
        Args:
            G: networkx グラフ
            pos: ノード座標 (n_nodes, 2)
            n_control: 各エッジ内のコントロールポイント数
            canvas: (xlim, ylim, x_show, y_show)
        """
        self.G = G
        self.pos = pos
        self.n_control = n_control
        self.xlim, self.ylim, self.x_show, self.y_show = canvas
        self.n_nodes = len(G.nodes())
        self.n_edges = len(G.edges())
        self.edges = np.array(G.edges())
        self.pos_np = self.divide_control_points()[1]  # 初期コントロール点配置
        self.pos_cp = cp.asarray(self.pos_np)  # GPU用コピー
        edge_vec = self.pos[self.edges[:, 1]] - self.pos[self.edges[:, 0]]
        ninty = np.deg2rad(90)
        rot = np.array([[cos(ninty), -sin(ninty)], [sin(ninty), cos(ninty)]])
        self.suichoku = (edge_vec @ rot.T)
        self.suichoku /= np.linalg.norm(self.suichoku, axis=1, keepdims=True)
        self.suichoku_cp = cp.asarray(self.suichoku)
        #self.angle_for_compmax = self.__angle_for_compmax()
        self.n_samples = n_samples
        self.B = self.__bernstein(n_samples)

    def _prepare_axes(self, ax: Optional[plt.Axes], bg_color: str, dpi: int, title: Optional[str]) -> plt.Axes:
        """Create or configure matplotlib Axes for drawing."""
        if ax is None:
            ratio = self.xlim / self.ylim
            figsize = (10, 10 / ratio) if ratio >= 1 else (10 * ratio, 10)
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=bg_color)
        ax.set_xlim(0, self.xlim)
        ax.set_ylim(self.ylim, 0)
        ax.axis("off")
        ax.set_facecolor(bg_color)
        if title:
            ax.set_title(title)
        return ax

    def _finalize_plot(self, ax: Optional[plt.Axes], save_path: Optional[str], dpi: int) -> None:
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
            plt.close()

    def tic(self):
        import time
        self._start_time = time.perf_counter()

    def toc(self, reset: bool = False, show: bool = True) -> float:
        import time
        if hasattr(self, '_start_time'):
            elapsed = time.perf_counter() - self._start_time
            if show:
                print(f"time: {elapsed} sec")
            if reset:
                self._start_time = time.perf_counter()
            return elapsed
        else:
            print("tic() was not called.")

    def show_graph(
        self,
        G: Optional[nx.Graph] = None,
        pos: Optional[np.ndarray] = None,
        with_nodes: bool = False,
        node_color: str = "blue",
        edge_color: str = "black",
        node_size: float = 30,
        edge_width: float = 1,
        save_path: Optional[str] = None,
        alpha: float = 1.0,
        bg_color: str = "white",
        dpi: int = 300,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None
    ):
        """
        グラフ描画（networkx＋matplotlib）

        Args:
            G: 描画用networkxグラフ（デフォルトはself.G）
            pos: ノード座標（デフォルトはself.pos）
            with_nodes: ノード表示フラグ
            node_color, edge_color, node_size, edge_width: 描画パラメータ
            save_path: 画像保存パス
            alpha, bg_color, dpi: 見た目調整
            ax: 外部matplotlib Axes（省略時は新規生成）
            title: タイトル
        """
        G = G if G is not None else self.G
        pos = pos if pos is not None else self.pos

        ax = self._prepare_axes(ax, bg_color, dpi, title)

        if with_nodes:
            nx.draw_networkx_nodes(G, {i: pos[i] for i in G.nodes()}, ax=ax, node_color=node_color, node_size=node_size)
        nx.draw_networkx_edges(G, {i: pos[i] for i in G.nodes()}, ax=ax, edge_color=edge_color, width=edge_width, alpha=alpha)
        self._finalize_plot(ax, save_path, dpi)

    def show_graph_edges_np(
        self,
        edges_pos: np.ndarray,
        edge_color: str = "black",
        edge_width: float = 1.0,
        save_path: Optional[str] = None,
        alpha: float = 1.0,
        bg_color: str = "white",
        dpi: int = 300,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None
    ):
        """
        複数エッジ配列(n_edges, n_pts, 2)を単純線分描画
        """
        ax = self._prepare_axes(ax, bg_color, dpi, title)
        for edge in edges_pos:
            ax.plot(edge[:, 0], edge[:, 1], '-', color=edge_color, alpha=alpha, linewidth=edge_width)
        self._finalize_plot(ax, save_path, dpi)

    def show_graph_smooth(
        self,
        edges_pos: np.ndarray,
        method: str = "PCHIP",
        threshold: int = 0,
        edge_color: str = "black",
        edge_width: float = 1.0,
        save_path: Optional[str] = None,
        alpha: float = 1.0,
        bg_color: str = "white",
        dpi: int = 300,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        n_samples: int = 100,
    ):
        """
        曲線補間によるエッジ描画（PCHIP/CubicSpline/BEZIER）

        - method="BEZIER" のときは cluster_control_points を行わず、
          コントロール点から直接ベジェ曲線をサンプリングして描画します。
        - 非ベジェ描画ではクラスタリング後に補間して描画します。
        """
        method_upper = method.upper()

        # BEZIER のときはクラスタリングをスキップして直接サンプリング
        if method_upper == "BEZIER":
            ax = self._prepare_axes(ax, bg_color, dpi, title)
            # edges_pos expected shape (n_edges, m, 2)
            samples = self.sample_bezier_edges(edges_pos)
            for e in range(samples.shape[0]):
                ax.plot(samples[e, :, 0], samples[e, :, 1], '-', color=edge_color, alpha=alpha, linewidth=edge_width)
            self._finalize_plot(ax, save_path, dpi)
            return

        # 非ベジェ描画はクラスタリングして補間
        edges_pos = self.cluster_control_points(edges_pos, threshold)
        ax = self._prepare_axes(ax, bg_color, dpi, title)

        # 多くのエッジで同じコントロール点数なら t を再利用
        if len(edges_pos) > 0:
            m_len = len(edges_pos[0])
            t_common = np.linspace(0, m_len - 1, 100)

        for edge in edges_pos:
            m = len(edge)
            t = t_common if m == m_len else np.linspace(0, m - 1, 100)
            ar = np.arange(m)
            if method_upper == "PCHIP":
                cs_x = PchipInterpolator(ar, edge[:, 0])
                cs_y = PchipInterpolator(ar, edge[:, 1])
            else:
                cs_x = CubicSpline(ar, edge[:, 0])
                cs_y = CubicSpline(ar, edge[:, 1])
            ax.plot(cs_x(t), cs_y(t), '-', color=edge_color, alpha=alpha, linewidth=edge_width)

        self._finalize_plot(ax, save_path, dpi)

    def _format_annotation_values(
        self, labels: Sequence[str], values: Iterable[float] | None
    ) -> str:
        """Create a multi-line string summarising metric values."""

        if values is None:
            return ""

        try:
            values_list = list(values)
        except TypeError:
            values_list = [values]

        lines: List[str] = []
        for label, raw in zip(labels, values_list):
            try:
                num = float(raw)
            except (TypeError, ValueError):
                text = str(raw)
            else:
                integer = int(num)
                text = str(integer) if abs(num - integer) < 1e-9 else f"{num:.6f}"
            lines.append(f"{label}: {text}")

        return "\n".join(lines)

    def annotate_metrics(
        self, path: PathInput, labels: Sequence[str], values: Iterable[float] | None
    ) -> None:
        """Format evaluation values and annotate the specified image."""

        annotation = self._format_annotation_values(labels, values)
        self.annotate_image(path, annotation)

    def annotate_image(self, path: PathInput, text: str) -> None:
        """Overlay evaluation text onto an existing image file."""

        from PIL import Image, ImageDraw, ImageFont

        target = Path(path)
        if not target.exists():
            print(f"annotate failed: file not found: {target}")
            return

        with Image.open(target) as base_image:
            image = base_image.convert("RGBA")

        font_size = 28
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(image)
        lines = text.splitlines() if text else [""]
        widths: List[int] = []
        heights: List[int] = []
        for line in lines:
            if hasattr(draw, "textbbox"):
                bbox = draw.textbbox((0, 0), line, font=font)
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
            else:
                width, height = draw.textsize(line, font=font)
            widths.append(width)
            heights.append(height)

        max_width = max(widths) if widths else 0
        line_spacing = max(2, int(font_size * 0.2))
        total_height = sum(heights) + line_spacing * (len(lines) - 1)
        padding = 10

        overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(
            [0, 0, max_width + padding * 2, total_height + padding],
            fill=(255, 255, 255, 200),
        )
        composed = Image.alpha_composite(image, overlay)

        text_draw = ImageDraw.Draw(composed)
        y = padding // 2
        for idx, line in enumerate(lines):
            text_draw.text((padding, y), line, fill="black", font=font)
            y += heights[idx] + line_spacing

        composed.convert("RGB").save(target)

    def divide_control_points(self) -> Tuple[nx.Graph, np.ndarray]:
        """
        エッジごとにコントロールポイントを等分配置し、新グラフ＋配列返却

        Returns:
            new_G: コントロール点追加済みnetworkx.Graph
            control_points: (n_edges, n_control+2, 2) ndarray
        """
        G_new = self.G.copy()
        control_points = np.empty((self.n_edges, self.n_control + 2, 2))
        # もとのエッジ除去＋等分点追加
        for i, (u, v) in enumerate(self.edges):
            G_new.remove_edge(u, v)
            control_points[i, 0] = self.pos[u]
            control_points[i, -1] = self.pos[v]
            for k in range(1, self.n_control + 1):
                control_points[i, k] = self.pos[u] + (self.pos[v] - self.pos[u]) * k / (self.n_control + 1)
        for i, (u, v) in enumerate(self.edges):
            nodes = [u] + [self.n_nodes + i * self.n_control + k for k in range(self.n_control)] + [v]
            for s, t in zip(nodes[:-1], nodes[1:]):
                G_new.add_edge(s, t)
        return G_new, control_points

    def control_points_to_pos(self, control_points: np.ndarray) -> np.ndarray:
        """
        コントロール点配列(n_edges, n_control+2, 2) → 全ノード座標(ndarray)
        """
        flat_points = np.delete(control_points, [0, -1], axis=1).reshape(-1, 2)
        return np.vstack([self.pos, flat_points])

    def pos_to_control_points(self, pos_full: np.ndarray) -> np.ndarray:
        """
        全ノード座標(ndarray) → コントロール点配列(n_edges, n_control+2, 2)
        """
        control_points_flat = pos_full[self.n_nodes:].reshape(self.n_edges, self.n_control, 2)
        control_points = np.empty((self.n_edges, self.n_control + 2, 2))
        control_points[:, 1:-1] = control_points_flat
        control_points[:, 0] = self.pos[self.edges[:, 0]]
        control_points[:, -1] = self.pos[self.edges[:, 1]]
        return control_points
    
    def move_ver(self, v):
        """
            numpyのブロードキャストで実装しようの会
        """
        v_mat = v.reshape(self.n_edges, self.n_control)
        suichoku_exp = self.suichoku_cp[:, None, :]
        delta = v_mat[:, :, None] * suichoku_exp
        pos_new = self.pos_cp.copy()
        pos_new[:, 1:1+self.n_control, :] += delta
        return pos_new

    def move_ver_batch(self, v):
        """move_ver のバッチ版。

        v: cupy (B, n_edges * n_control) → (B, n_edges, n_control+2, 2)
        """
        batch = v.shape[0]
        v_mat = v.reshape(batch, self.n_edges, self.n_control)
        delta = v_mat[:, :, :, None] * self.suichoku_cp[None, :, None, :]
        pos_new = cp.broadcast_to(self.pos_cp, (batch,) + self.pos_cp.shape).copy()
        pos_new[:, :, 1:1 + self.n_control, :] += delta
        return pos_new

    def move_attract(self, v: np.ndarray) -> np.ndarray:
        """
        Attractorベースのエッジ変形（numpy vectorized）

        v: shape (3*N,) — [x0, y0, r0, x1, y1, r1, ...]
           各Attractorは中心座標(x, y)と影響半径rを持つ
        戻り値: pos_np shape (n_edges, n_control+2, 2) — move_ver と同じ形式

        各CPについて:
        - 影響範囲内(distance < r)のAttractorが存在すれば、最近傍のAttractorへスナップ
        - 範囲内のAttractorがなければ初期配置のまま
        """
        attractors = v.reshape(-1, 3)           # (N, 3)
        attractor_xy = attractors[:, :2]        # (N, 2)
        attractor_r  = attractors[:, 2]         # (N,)

        pos_new = self.pos_cp.copy()            # (n_edges, n_control+2, 2)
        cp_flat = pos_new[:, 1:self.n_control + 1, :].reshape(-1, 2)  # (E*M, 2)

        # 距離計算: (E*M, N)
        diff      = cp_flat[:, None, :] - attractor_xy[None, :, :]  # (E*M, N, 2)
        distances = cp.linalg.norm(diff, axis=2)                     # (E*M, N)

        # スナップ判定
        in_range  = distances < attractor_r[None, :]                 # (E*M, N) bool
        masked    = cp.where(in_range, distances, cp.inf)            # 範囲外→inf
        nearest_idx  = cp.argmin(masked, axis=1)                     # (E*M,)
        has_attractor = in_range.any(axis=1)                         # (E*M,) bool

        # 座標上書き（範囲内のみスナップ）
        snapped = attractor_xy[nearest_idx]                          # (E*M, 2)
        cp_new  = cp.where(has_attractor[:, None], snapped, cp_flat) # (E*M, 2)

        pos_new[:, 1:self.n_control + 1, :] = cp_new.reshape(self.n_edges, self.n_control, 2)
        return pos_new

    def __point_side_of_line(self, A, B, P):
        # ベクトル計算を用いたクロス積
        cross_product = (B[0] - A[0]) * (P[1] - A[1]) - (B[1] - A[1]) * (P[0] - A[0])
        if cross_product > 0:
            return 1  # 左側
        elif cross_product < 0:
            return -1  # 右側
        else:
            return 0  # 直線上
    
    def __angle_for_compmax(self):
        compatibility = self.get_compatibility_matrix()
        angle = []
        for i, c in enumerate(compatibility):
            if max(c) == 0:
                angle.append(0)
            else:
                target_edge = np.argmax(c)
                angle.append(self.__point_side_of_line(self.pos[self.edges[i][0]], self.pos[self.edges[i][1]], (self.pos[self.edges[target_edge][0]] + self.pos[self.edges[target_edge][1]]) / 2))
        return angle
    def get_compatibility_matrix(self):
        return comp(self.edges, self.pos)
    def v_for_compmax(self, v):
        angle = self.angle_for_compmax
        for i, j in enumerate(angle):
            for k in range(self.n_control):
                v[i * self.n_control + k] *= j
        return v
    
    def move_for_compmax(self, v):
        return self.move_ver(self.v_for_compmax(v.copy()))
    
    def cluster_control_points(
        self,
        pos_np: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """
        コントロールポイントを距離閾値でクラスタリングし、各クラスタを重心へ移動
        """
        pos_full = self.control_points_to_pos(pos_np)
        control_pts = pos_full[self.n_nodes:]
        distance_matrix = pdist(control_pts)
        linkage_matrix = linkage(distance_matrix, method='single')
        labels = fcluster(linkage_matrix, threshold, criterion='distance')
        centroids = {label: control_pts[labels == label].mean(axis=0) for label in np.unique(labels)}
        for i, label in enumerate(labels):
            control_pts[i] = centroids[label]
        pos_full[self.n_nodes:] = control_pts
        return self.pos_to_control_points(pos_full)
    
    def __bernstein(self, n_samples):
        m = self.n_control + 2
        t = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
        n = m - 1
        B = np.vstack([comb(n, k) * (t**k) * ((1 - t)**(n - k)) for k in range(n + 1)])  # (m, n_samples)
        return B

    def sample_bezier_edges(self, ctrl_pts: np.ndarray) -> np.ndarray:
        """
        Sample Bezier curves for multiple edges using batch computation.

        Parameters
        ----------
        ctrl_pts : np.ndarray
            Control points array shape (n_edges, m, 2)

        Returns
        -------
        np.ndarray
            Sampled points shape (n_edges, n_samples, 2)
        """
        # ctrl_pts: (n_edges, m, 2)
        # self.B: (m, n_samples)
        # Reshape ctrl_pts to (n_edges, 2, m) for matrix multiplication
        ctrl_pts_transposed = np.transpose(ctrl_pts, (0, 2, 1))  # (n_edges, 2, m)
        # Batch matrix multiplication
        curves = np.matmul(ctrl_pts_transposed, self.B)  # (n_edges, 2, n_samples)
        # Transpose back to desired shape (n_edges, n_samples, 2)
        return np.transpose(curves, (0, 2, 1))

# --- 外部ユーティリティも柔軟に呼べるイメージ例 ---
def sort_solutions_by_objective(solutions: Sequence[Any], index: int = 0) -> List[Any]:
    """ jMetal系のソリューションリストを指定objectiveでソート """
    return sorted(solutions, key=lambda s: s.objectives[index])

def save_solutions_to_txt(solutions: Sequence[Any], var_path: str, fun_path: str):
    from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
    print_function_values_to_file(solutions, fun_path)
    print_variables_to_file(solutions, var_path)

def text_to_ndarray(text : str):
    array = np.loadtxt(text)
    return array
