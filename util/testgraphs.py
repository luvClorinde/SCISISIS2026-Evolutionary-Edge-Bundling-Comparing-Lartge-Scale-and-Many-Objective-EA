import networkx as nx
import numpy as np
import random
import pandas as pd
from pathlib import Path
from typing import Tuple, Union

XLIM_DEFAULT = 1000
YLIM_DEFAULT = 1000

def random_graph_names(prefix: str = "") -> list:
    """
    data_set/random 内の合成データセット名を昇順で返す。
    prefix="waxman" / "gthreshold" で系列を絞り込める。
    ファイル追加・削除が自動で反映されるよう、固定リストではなくディレクトリを都度走査する。
    """
    d = Path(__file__).parent.parent / "data_set" / "random"
    names = sorted(p.stem.removeprefix("Node_") for p in d.glob("Node_*.csv"))
    return [n for n in names if n.startswith(prefix)]

def get_graph(name: Union[int, str], xlim: int = XLIM_DEFAULT, ylim: int = YLIM_DEFAULT):
    """
    サンプルグラフの取得関数。
    Args:
        name: グラフタイプ（intまたはstr）
        xlim: 描画範囲x
        ylim: 描画範囲y
    Returns:
        (G, pos) or (G, pos, param) 形式
    """
    mapping = {
        1: graph1,
        2: graph2,
        3: graph3,
        4: graph4,
        5: graph5,
        "jpair": japan,
        "usair": usairline,
        "air": airline,
        "usmig": usmigration,
        "wiki": wikivote,
        "ds0": ds0,
        "ds1": ds1,
        "ds2": ds2,
        "ds4": ds4,
        "mafia": mafia,
    }
    if isinstance(name, str) and name.startswith(("waxman", "gthreshold")):
        return synthetic(name)
    if name not in mapping:
        raise ValueError(f"Unknown graph name: {name}")
    func = mapping[name]
    XLIM_DEFAULT = xlim
    YLIM_DEFAULT = ylim
    return func()

def graph1(xlim: int = XLIM_DEFAULT, ylim: int = YLIM_DEFAULT, yrang: float = 0.2, yplace: float = 0.28, xrang: float = 0.2, n_n: int = 100) -> Tuple[nx.Graph, np.ndarray]:
    """
    平行なサンプルグラフ
    Returns:
        G: networkx.Graph
        pos: np.ndarray [n_n, 2]
    """
    if n_n % 4 != 0:
        raise ValueError("n_n must be a multiple of 4.")
    G = nx.Graph()
    G.add_nodes_from(range(n_n))
    G.add_edges_from((i, i + 1) for i in range(0, n_n, 2))

    random.seed(1)
    pos = np.empty((n_n, 2))
    for i in range(0, n_n, 4):
        pos[i] = [random.random() * xlim * xrang, (random.random() * yrang + yplace) * ylim]
        pos[i + 1] = [xlim - (random.random() * xlim * xrang), (random.random() * yrang + yplace) * ylim]
        pos[i + 2] = [random.random() * xlim * xrang, ylim - ((random.random() * yrang + yplace) * ylim)]
        pos[i + 3] = [xlim - (random.random() * xlim * xrang), ylim - ((random.random() * yrang + yplace) * ylim)]
    return G, pos, (xlim, ylim, 10, 10)

def graph2(xlim: int = XLIM_DEFAULT, ylim: int = YLIM_DEFAULT) -> Tuple[nx.Graph, np.ndarray]:
    """
    三角形のサンプルグラフ
    """
    n_n = 101
    G = nx.Graph()
    G.add_nodes_from(range(n_n))
    G.add_edges_from((0, i) for i in range(1, n_n))

    pos = np.empty((n_n, 2))
    pos[0] = [xlim / 2, 0]
    d = xlim / (n_n - 1)
    for i in range(1, n_n):
        pos[i] = [d * (i - 1), ylim]
    return G, pos, (xlim, ylim, 10, 10)

def graph3(xlim: int = XLIM_DEFAULT, ylim: int = YLIM_DEFAULT) -> Tuple[nx.Graph, np.ndarray]:
    """
    クロス型のサンプルグラフ
    """
    n_n = 100
    G = nx.Graph()
    G.add_nodes_from(range(n_n))
    G.add_edges_from((i, n_n - (i + 1)) for i in range(n_n // 2))

    d = xlim / n_n * 2
    pos = np.empty((n_n, 2))
    for i in range(n_n // 2):
        pos[i] = [d * i, 0]
        pos[n_n - (i + 1)] = [xlim - d * i, ylim]
    return G, pos, (xlim, ylim, 10, 10)

def graph4(xlim: int = XLIM_DEFAULT, ylim: int = YLIM_DEFAULT) -> Tuple[nx.Graph, np.ndarray]:
    """
    三角形が二つならんだサンプルグラフ
    """
    n_n = 102
    G = nx.Graph()
    G.add_nodes_from(range(n_n))
    G.add_edges_from((0, i) for i in range(1, n_n // 2))
    G.add_edges_from((n_n // 2, i) for i in range(n_n // 2 + 1, n_n))

    pos = np.empty((n_n, 2))
    x_temp = 0
    d = xlim // (n_n - 2)
    pos[0] = [xlim // 4, 0]
    pos[n_n // 2] = [xlim // 4 * 3, 0]
    for i in range(n_n):
        if i not in (0, n_n // 2):
            pos[i] = [x_temp, ylim]
            x_temp += d
    return G, pos, (xlim, ylim, 10, 10)

def graph5(xlim: int = XLIM_DEFAULT, ylim: int = YLIM_DEFAULT) -> Tuple[nx.Graph, np.ndarray]:
    """
    4つの辺からなる閉路状のサンプルグラフ
    """
    n_n = 100
    G = nx.Graph()
    G.add_nodes_from(range(n_n))
    for i in range(n_n):
        if i not in (0, n_n // 2):
            G.add_edge(i, n_n - i)

    pos = np.empty((n_n, 2))
    dx = xlim // (n_n // 4)
    dy = ylim // (n_n // 4)
    for i in range(0, n_n // 4):
        pos[i] = [0, i * dy]
    for i in range(n_n // 4, n_n // 2):
        pos[i] = [(i - n_n / 4) * dx, ylim]
    for i in range(n_n // 2, n_n // 4 * 3):
        pos[i] = [xlim, ylim - (i - n_n / 2) * dy]
    for i in range(n_n // 4 * 3, n_n):
        pos[i] = [xlim - (i - n_n / 4 * 3) * dx, 0]
    return G, pos, (xlim, ylim, 10, 10)

def airline(node_file: str = "../data_set/Node_AirPorts.csv",
            edge_file: str = "../data_set/Edge_AirPorts.csv") -> Tuple[nx.Graph, np.ndarray, Tuple[int, int, int, int]]:
    """
    世界の航空路線グラフ
    """
    xlim, ylim, x_show, y_show = 1440, 535, 8, 3
    G, pos = _load_graph_from_csv(node_file, edge_file, swap_xy=True)
    pos[:, 1] = ylim - pos[:, 1]  # 上下反転
    return G, pos, (xlim, ylim, x_show, y_show)

def usairline(node_file: str = "../data_set/Node_AmericaAirLines.csv", edge_file: str = "../data_set/Edge_AmericaAirLines.csv") -> Tuple[nx.Graph, np.ndarray, Tuple[int, int, int, int]]:
    """
    アメリカの航空路線グラフ
    """
    xlim, ylim, x_show, y_show = 700, 350, 10, 5
    G, pos = _load_graph_from_csv(node_file, edge_file)
    return G, pos, (xlim, ylim, x_show, y_show)

def japan(node_file: str = "../data_set/Node_Japan.csv", edge_file: str = "../data_set/Edge_Japan.csv") -> Tuple[nx.Graph, np.ndarray, Tuple[int, int, int, int]]:
    """
    日本の航空路線グラフ
    """
    xlim, ylim, x_show, y_show = 800, 800, 10, 10
    G, pos = _load_graph_from_csv(node_file, edge_file)
    pos -= pos.min(axis=0) - 20
    return G, pos, (xlim, ylim, x_show, y_show)

def usmigration(node_file: str = "../data_set/Node_AmericaMigrations.csv", edge_file: str = "../data_set/Edge_AmericaMigrations.csv") -> Tuple[nx.Graph, np.ndarray, Tuple[int, int, int, int]]:
    """
    アメリカの移民グラフ
    """
    xlim, ylim, x_show, y_show = 725, 335, 7, 3
    G, pos = _load_graph_from_csv(node_file, edge_file)
    pos -= pos.min(axis=0) - 20
    return G, pos, (xlim, ylim, x_show, y_show)

def wikivote(node_file: str = "../data_set/Node_WikiVote.csv", edge_file: str = "../data_set/Edge_WikiVote.csv") -> Tuple[nx.Graph, np.ndarray, Tuple[int, int, int, int]]:
    """
    WikiVote グラフ
    """
    xlim, ylim, x_show, y_show = 2615, 2930, 9, 10
    G, pos = _load_graph_from_csv(node_file, edge_file)
    return G, pos, (xlim, ylim, x_show, y_show)

def synthetic(name: str) -> Tuple[nx.Graph, np.ndarray, Tuple[int, int, int, int]]:
    """
    ランダム幾何系モデルで生成した合成グラフ（n=250、座標は[20,980]^2、最大連結成分のみ）
    - "waxman_a012_s1" 等: Waxmanモデル beta=0.5、alpha∈{0.10,0.12,0.15} × seed∈{1..5}
    - "gthreshold_t080_s1" 等: geographical_threshold、theta∈{120,80,60} × seed∈{1..5}
    name → data_set/random/{Node,Edge}_<name>.csv（座標は[10,490]^2程度）
    """
    xlim, ylim, x_show, y_show = 500, 500, 10, 10
    G, pos = _load_graph_from_csv(f"../data_set/random/Node_{name}.csv", f"../data_set/random/Edge_{name}.csv")
    return G, pos, (xlim, ylim, x_show, y_show)

def mafia(node_file: str = "../data_set/Node_Mafia.csv", edge_file: str = "../data_set/Edge_Mafia.csv") -> Tuple[nx.Graph, np.ndarray, Tuple[int, int, int, int]]:
    """
    マフィア人物関係グラフ（ノードは経度緯度で位置指定）
    """
    scale = 14  # 経度緯度(度)を他データセットと近いスケール(数百〜千)に揃えるための倍率
    x_show, y_show = 10, 4
    G, pos = _load_graph_from_named_csv(node_file, edge_file)
    pos = _jitter_duplicate_positions(pos)  # 座標が完全一致するノードがあると零ベクトルのエッジができるため
    pos -= pos.min(axis=0)
    pos *= scale
    pos += 20
    xlim, ylim = int(pos[:, 0].max()) + 20, int(pos[:, 1].max()) + 20
    return G, pos, (xlim, ylim, x_show, y_show)

def _jitter_duplicate_positions(pos: np.ndarray, jitter: float = 1e-4, seed: int = 1) -> np.ndarray:
    """
    座標が完全一致するノードにランダムノイズを加える
    （Wallinger et al. 2025 が GTREE の重複座標整列対策として行ったのと同様の処理）
    """
    pos = pos.copy()
    rng = np.random.default_rng(seed)
    _, inverse, counts = np.unique(pos, axis=0, return_inverse=True, return_counts=True)
    for group in np.where(counts > 1)[0]:
        idx = np.where(inverse == group)[0]
        pos[idx] += rng.uniform(-jitter, jitter, size=(len(idx), 2))
    return pos

def _load_graphml(graphml_file: str) -> Tuple[nx.Graph, np.ndarray]:
    """
    GraphMLファイルからグラフと座標を読み込む
    """
    G_raw = nx.read_graphml(graphml_file)
    if G_raw.is_directed():
        G_raw = G_raw.to_undirected()
    mapping = {n: i for i, n in enumerate(G_raw.nodes())}
    G = nx.relabel_nodes(G_raw, mapping)
    pos = np.empty((len(mapping), 2))
    for n_id, data in G_raw.nodes(data=True):
        pos[mapping[n_id]] = [data['x'], data['y']]
    return G, pos

def ds0(graphml_file: str = "../data_set/ds/ds0.graphml") -> Tuple[nx.Graph, np.ndarray, Tuple[int, int, int, int]]:
    xlim, ylim, x_show, y_show = 610, 630, 6, 6
    G, pos = _load_graphml(graphml_file)
    return G, pos, (xlim, ylim, x_show, y_show)

def ds1(graphml_file: str = "../data_set/ds/ds1.graphml") -> Tuple[nx.Graph, np.ndarray, Tuple[int, int, int, int]]:
    xlim, ylim, x_show, y_show = 610, 625, 6, 6
    G, pos = _load_graphml(graphml_file)
    return G, pos, (xlim, ylim, x_show, y_show)

def ds2(graphml_file: str = "../data_set/ds/ds2.graphml") -> Tuple[nx.Graph, np.ndarray, Tuple[int, int, int, int]]:
    xlim, ylim, x_show, y_show = 610, 630, 6, 6
    G, pos = _load_graphml(graphml_file)
    return G, pos, (xlim, ylim, x_show, y_show)

def ds4(graphml_file: str = "../data_set/ds/ds4.graphml") -> Tuple[nx.Graph, np.ndarray, Tuple[int, int, int, int]]:
    xlim, ylim, x_show, y_show = 610, 610, 6, 6
    G, pos = _load_graphml(graphml_file)
    return G, pos, (xlim, ylim, x_show, y_show)

def _load_graph_from_csv(node_file: str, edge_file: str, swap_xy: bool = False) -> Tuple[nx.Graph, np.ndarray]:
    """
    CSVファイルからグラフと座標を読み込む
    """
    G = nx.Graph()
    d1 = pd.read_csv(node_file)
    d2 = pd.read_csv(edge_file)
    G.add_nodes_from(range(len(d1)))
    G.add_edges_from((int(d2.loc[i, 'source']) - 1, int(d2.loc[i, 'target']) - 1) for i in range(len(d2)))
    if swap_xy:
        pos = np.array([[d1.loc[i, 'y'], d1.loc[i, 'x']] for i in range(len(d1))])
    else:
        pos = np.array([[d1.loc[i, 'x'], d1.loc[i, 'y']] for i in range(len(d1))])
    return G, pos

def _load_graph_from_named_csv(node_file: str, edge_file: str) -> Tuple[nx.Graph, np.ndarray]:
    """
    ノード名（文字列）で source/target を参照する形式のCSVからグラフと座標を読み込む
    """
    d1 = pd.read_csv(node_file)
    d2 = pd.read_csv(edge_file)
    name_to_idx = {name: i for i, name in enumerate(d1['NODE'])}
    G = nx.Graph()
    G.add_nodes_from(range(len(d1)))
    G.add_edges_from((name_to_idx[s], name_to_idx[t]) for s, t in zip(d2['Source'], d2['Target']))
    pos = d1[['LonX', 'LatY']].to_numpy(dtype=float)
    return G, pos
