# -*- coding: utf-8 -*-
# tda_core.py (logic)
import pandas as pd
import numpy as np
import io
from joblib import Parallel, delayed
from scipy.sparse.csgraph import connected_components, dijkstra
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import networkx as nx
from gudhi.cover_complex import MapperComplex
import matplotlib.pyplot as plt
import os, warnings, random

# Hygiene
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(100); random.seed(100)

# ---------- IO ----------
def read_csv_bytes(content: bytes) -> pd.DataFrame:
    """Parse CSV from raw bytes: try pyarrow -> utf8 -> gbk."""
    if not content:
        return pd.DataFrame()
    bio = io.BytesIO(content)
    try:
        return pd.read_csv(bio, engine="pyarrow")
    except Exception:
        bio.seek(0)
    try:
        return pd.read_csv(bio, low_memory=False, on_bad_lines="skip")
    except Exception:
        bio.seek(0)
    return pd.read_csv(bio, encoding="gbk", low_memory=False, on_bad_lines="skip")

def numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    return df.select_dtypes(include=[np.number]).copy()

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Column","Missing","MissingRate(%)","Dtype"])
    miss = df.isna().sum()
    rate = (miss / len(df) * 100).round(2)
    return (pd.DataFrame({"Column": df.columns, "Missing": miss.values, "MissingRate(%)": rate.values,
                          "Dtype": [str(t) for t in df.dtypes]})
            .sort_values(["Missing","MissingRate(%)"], ascending=[False,False])
            .reset_index(drop=True))

def top_missing_bar_mpl(df: pd.DataFrame, top_k: int = 15):
    """Return a Matplotlib Figure for Shiny @render.plot."""
    fig = plt.figure(figsize=(8, 3.6))
    ax = fig.add_subplot(111)
    if df.empty:
        ax.set_title("(No data loaded)")
        return fig
    ms = missing_summary(df)
    ms_top = ms[ms["Missing"]>0].head(top_k)
    if ms_top.empty:
        ax.set_title("No missing values")
        return fig
    ax.bar(ms_top["Column"], ms_top["Missing"])
    ax.set_ylabel("Missing")
    ax.set_xlabel("Column")
    ax.set_title(f"Top {len(ms_top)} columns with most missing values")
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    fig.tight_layout()
    return fig

def p_hist_mpl(p_values: np.ndarray):
    fig = plt.figure(figsize=(8, 3.6))
    ax = fig.add_subplot(111)
    if p_values.size == 0:
        ax.set_title("No p-values yet")
        return fig
    ax.hist(p_values, bins=20, range=(0,1))
    ax.set_xlabel("p-value")
    ax.set_ylabel("Count")
    ax.set_title("p-value distribution (pipeline)")
    fig.tight_layout()
    return fig

# ---------- Mapper & pipeline ----------
def _cdf01(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x)+1, dtype=float)
    return (ranks - 0.5) / len(x)

class AdaptiveKMeans:
    def __init__(self, base_n_clusters=5, random_state=0, n_init=10):
        self.base_n_clusters = int(base_n_clusters)
        self.random_state = random_state
        self.n_init = int(n_init)
    def fit_predict(self, X, y=None):
        m = X.shape[0]
        if m <= 1: return np.zeros(m, dtype=int)
        k = min(self.base_n_clusters, m)
        if k <= 1: return np.zeros(m, dtype=int)
        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=self.n_init)
        return km.fit_predict(X)

def preprocess_pool(A: pd.DataFrame, B: pd.DataFrame, knn_k=20, use_2d_filter=True):
    A = numeric_df(A); B = numeric_df(B)
    n1, n2 = len(A), len(B)
    pool_df = pd.concat([A, B], axis=0, ignore_index=True)
    if pool_df.empty:
        return np.empty((0,0)), np.empty((0,1)), np.array([], int), np.array([], int)
    coords = StandardScaler().fit_transform(pool_df).astype(np.float32)
    pca1 = PCA(n_components=1, random_state=0).fit_transform(coords).ravel()
    nn = NearestNeighbors(n_neighbors=min(knn_k, max(2, len(coords))), metric="euclidean").fit(coords)
    dists, _ = nn.kneighbors(coords)
    knnd = dists[:, -1]
    pca1u = _cdf01(pca1); knndu = _cdf01(knnd)
    fil = (np.vstack([pca1u, knndu]).T if use_2d_filter else pca1u.reshape(-1,1)).astype(np.float32)
    idx_A = np.arange(0, n1, dtype=int); idx_B = np.arange(n1, n1+n2, dtype=int)
    return coords, fil, idx_A, idx_B

def build_mapper_from_indices(coords_full, fil_full, idx, res, gain, n_clusters=4, km_random_state=0):
    if coords_full.size == 0:
        return None, np.empty((0,1))
    sub_coords = coords_full[idx, :]; sub_fil = fil_full[idx, :]
    n_filters = sub_fil.shape[1]
    filter_bnds = np.array([[0.0, 1.0]] * n_filters, dtype=np.float32)
    mapper = MapperComplex(
        filter_bnds=filter_bnds,
        resolutions=np.array([res] * n_filters, dtype=int),
        gains=np.array([gain] * n_filters, dtype=float),
        clustering=AdaptiveKMeans(base_n_clusters=n_clusters, random_state=km_random_state, n_init=10),
        input_type="point cloud",
    )
    mapper.fit(sub_coords, filters=sub_fil, colors=sub_fil)
    return mapper, sub_fil

TOPO_TYPES = ["connected_components","downbranch","upbranch","loop"]

def _mapper2networkx_gudhi(M):
    st = M.mapper_ if hasattr(M, "mapper_") else M.simplex_tree_
    G = nx.Graph()
    for (splx, _) in st.get_skeleton(1):
        if len(splx) == 1: G.add_node(splx[0])
        elif len(splx) == 2: G.add_edge(splx[0], splx[1])
    return G

def compute_topological_features_statmapper(M, func=None, func_type="data", topo_type="downbranch", threshold=0.0):
    mapper = M.mapper_ if hasattr(M, "mapper_") else M.simplex_tree_
    node_info = M.node_info_
    num_nodes = len(node_info)

    if func is None:
        A = np.zeros((num_nodes, num_nodes), dtype=float)
        for (splx, _) in mapper.get_skeleton(1):
            if len(splx) == 2:
                A[splx[0], splx[1]] = 1.0; A[splx[1], splx[0]] = 1.0
        dij = dijkstra(A, directed=False); D = np.where(np.isinf(dij), 0.0, dij)
        func = list(-D.max(axis=1)); func_type = "node"

    if func_type == "data":
        function = [np.mean([func[i] for i in node_info[v]["indices"]]) for v in range(num_nodes)]
    else:
        function = list(func)

    dgm, bnd = [], []

    if topo_type == "connected_components":
        A = np.zeros((num_nodes, num_nodes), dtype=float)
        for (splx, _) in mapper.get_skeleton(1):
            if len(splx) == 2:
                A[splx[0], splx[1]] = 1.0; A[splx[1], splx[0]] = 1.0
        _, labels = connected_components(A, directed=False)
        for cc in np.unique(labels):
            pts = np.where(labels == cc)[0]
            vals = [function[p] for p in pts]
            if abs(min(vals) - max(vals)) >= threshold:
                dgm.append((0, (min(vals), max(vals)))); bnd.append(list(pts))

    elif topo_type in ("downbranch", "upbranch"):
        f = np.array(function, dtype=float)
        if topo_type == "upbranch": f = -f
        A = np.zeros((num_nodes, num_nodes), dtype=float)
        for (splx, _) in mapper.get_skeleton(1):
            if len(splx) == 2:
                A[splx[0], splx[1]] = 1.0; A[splx[1], splx[0]] = 1.0
        order = np.argsort(f); rank = np.empty(num_nodes, dtype=int); rank[order] = np.arange(num_nodes)
        def find(i, parent): return i if parent[i] == i else find(parent[i], parent)
        def union(i, j, parent):
            if f[i] <= f[j]: parent[j] = i
            else: parent[i] = j
        parent = -np.ones(num_nodes, dtype=int)
        diag, comp, seen = {}, {}, {}
        for t in range(num_nodes):
            u = order[t]
            nbrs = np.where(A[u, :] == 1.0)[0]
            lower = [v for v in nbrs if rank[v] <= t] if nbrs.size > 0 else []
            if not lower:
                parent[u] = u; continue
            neigh_pars = [find(v, parent) for v in lower]
            g = neigh_pars[np.argmin([f[w] for w in neigh_pars])]
            pg = find(g, parent); parent[u] = pg
            for v in lower:
                pv = find(v, parent)
                if pg != pv:
                    pp = pg if f[pg] > f[pv] else pv
                    comp[pp] = []
                    for w in order[:t]:
                        if find(w, parent) == pp and w not in seen:
                            seen[w] = True; comp[pp].append(w)
                    comp[pp].append(u); diag[pp] = u
                    union(pg, pv, parent)
                else:
                    if len(nbrs) == len(lower):
                        comp[pg] = []
                        for w in order[:t+1]:
                            if find(w, parent) == pg and w not in seen:
                                seen[w] = True; comp[pg].append(w)
                        comp[pg].append(u); diag[pg] = u
        for key, val in diag.items():
            if topo_type == "downbranch": dgm.append((0, (f[key], f[val])))
            else: dgm.append((0, (-f[val], -f[key])))
            bnd.append(comp[key])

    elif topo_type == "loop":
        G = _mapper2networkx_gudhi(M)
        for pts in nx.cycle_basis(G):
            vals = [function[p] for p in pts]
            if abs(min(vals) - max(vals)) >= 0.0:
                dgm.append((1, (min(vals), max(vals)))); bnd.append(list(pts))

    return dgm, bnd

def _pairs_from_dgm(dgm):
    if not dgm: return np.empty((0,2), dtype=np.float64)
    return np.array([[bd[0], bd[1]] for dim, bd in dgm if dim <= 1], dtype=np.float64, order="C")

def _persist_vec(P, k=200):
    if P.size == 0: return np.zeros(k, dtype=np.float64)
    pers = P[:,1]-P[:,0]
    if pers.size > k:
        idx = np.argpartition(pers, -k)[-k:]
        v = pers[idx]; v.sort(); v = v[::-1]
    else:
        v = np.sort(pers)[::-1]
        if v.size < k: v = np.pad(v, (0, k-v.size))
    return np.ascontiguousarray(v, dtype=np.float64)

def bottleneck_statmapper(MF1, MF2, topo_type="connected_components", proxy_k=200):
    (M1, fil1), (M2, fil2) = MF1, MF2
    f1 = fil1[:,0] if fil1.ndim>1 else fil1
    f2 = fil2[:,0] if fil2.ndim>1 else fil2
    dgm1, _ = compute_topological_features_statmapper(M1, func=f1, func_type="data", topo_type=topo_type)
    dgm2, _ = compute_topological_features_statmapper(M2, func=f2, func_type="data", topo_type=topo_type)
    P = _pairs_from_dgm(dgm1); Q = _pairs_from_dgm(dgm2)
    if P.size==0 and Q.size==0: return 0.0
    if P.size==0 or Q.size==0: return float("inf")
    v1 = _persist_vec(P, k=proxy_k); v2 = _persist_vec(Q, k=proxy_k)
    return float(np.max(np.abs(v1-v2)))
