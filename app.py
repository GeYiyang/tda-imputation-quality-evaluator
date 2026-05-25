# -*- coding: utf-8 -*-
# app.py — Imputation Quality Evaluator (Shiny for Python)
# - Matplotlib backend forced to Agg
# - Mapper & Pipeline split; cached mapper figures
# - tda_core calls with safe fallbacks (build_mapper, preprocess_filters_single, etc.)

import os
# force non-interactive backend before importing pyplot anywhere
os.environ["MPLBACKEND"] = "Agg"

from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
from datetime import datetime
import io

# === core logic module ===
import tda_core as core  # keep your own file

# ------------------------------------------------------------
# Fallback helpers
# ------------------------------------------------------------
def _from_core(name):
    return getattr(core, name, None)

# 1) read_csv_bytes
_read_csv_bytes_core = _from_core("read_csv_bytes")
if _read_csv_bytes_core is None:
    def read_csv_bytes(b: bytes) -> pd.DataFrame:
        try:
            return pd.read_csv(io.BytesIO(b))
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(b), encoding="gbk")
else:
    read_csv_bytes = _read_csv_bytes_core

# 2) missing_summary
_missing_summary_core = _from_core("missing_summary")
if _missing_summary_core is None:
    def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["Column", "Missing", "MissingRate(%)", "Dtype"])
        miss = df.isna().sum()
        rate = (miss / len(df) * 100).round(2)
        return (
            pd.DataFrame({
                "Column": df.columns,
                "Missing": miss.values,
                "MissingRate(%)": rate.values,
                "Dtype": [str(t) for t in df.dtypes],
            })
            .sort_values(["Missing", "MissingRate(%)"], ascending=[False, False])
            .reset_index(drop=True)
        )
else:
    missing_summary = _missing_summary_core

# 3) top_missing_bar_mpl (matplotlib)
_top_missing_bar_mpl_core = _from_core("top_missing_bar_mpl")
if _top_missing_bar_mpl_core is None:
    def top_missing_bar_mpl(df: pd.DataFrame, top_k: int = 15):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 3.6))
        ax = fig.gca()
        if df.empty:
            ax.set_title("(No data loaded)")
            return fig
        ms = missing_summary(df)
        ms_top = ms[ms["Missing"] > 0].head(top_k)
        if ms_top.empty:
            ax.set_title("No missing values")
            return fig
        ax.bar(ms_top["Column"], ms_top["Missing"])
        ax.set_title(f"Top {len(ms_top)} columns with most missing")
        ax.set_xticklabels(ms_top["Column"], rotation=30, ha="right")
        return fig
else:
    top_missing_bar_mpl = _top_missing_bar_mpl_core

# 4) preprocess_filters_single
_preprocess_filters_single_core = _from_core("preprocess_filters_single")
if _preprocess_filters_single_core is None:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    def _cdf01_local(x):
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(x) + 1, dtype=float)
        return (ranks - 0.5) / len(x)

    def preprocess_filters_single(df: pd.DataFrame, use_2d: bool = True, knn_k: int = 20):
        X = df.select_dtypes(include=[np.number]).values.astype(np.float32)
        if X.size == 0:
            return np.empty((0, 0), dtype=np.float32), np.empty((0, 1), dtype=np.float32)
        coords = StandardScaler().fit_transform(X).astype(np.float32)
        pca1 = PCA(n_components=1, random_state=0).fit_transform(coords).ravel()
        n_neighbors = min(max(2, len(coords)), int(knn_k))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(coords)
        dists, _ = nn.kneighbors(coords)
        knnd = dists[:, -1]
        pca1u = _cdf01_local(pca1)
        knndu = _cdf01_local(knnd)
        filters = (np.vstack([pca1u, knndu]).T if use_2d else pca1u.reshape(-1, 1)).astype(np.float32)
        return coords, filters
else:
    preprocess_filters_single = _preprocess_filters_single_core

# 5) build_mapper & plot_mapper_pruned
_build_mapper_core = _from_core("build_mapper")
_plot_mapper_pruned_core = _from_core("plot_mapper_pruned")

if _build_mapper_core is None or _plot_mapper_pruned_core is None:
    # local mapper pieces
    from sklearn.cluster import KMeans
    import networkx as nx

    class AdaptiveKMeans:
        def __init__(self, base_n_clusters=4, random_state=0, n_init=10):
            self.base_n_clusters = int(base_n_clusters)
            self.random_state = random_state
            self.n_init = int(n_init)

        def fit_predict(self, X, y=None):
            m = X.shape[0]
            if m <= 1:
                return np.zeros(m, dtype=int)
            k = min(self.base_n_clusters, m)
            if k <= 1:
                return np.zeros(m, dtype=int)
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=self.n_init)
            return km.fit_predict(X)

    def build_mapper(coords: np.ndarray, filters: np.ndarray, res: int, gain: float, n_clusters: int):
        if coords.size == 0:
            return None
        try:
            from gudhi.cover_complex import MapperComplex
        except Exception as e:
            return None  # will be handled by plot function
        n_filters = filters.shape[1]
        filter_bnds = np.array([[0.0, 1.0]] * n_filters, dtype=np.float32)
        mapper = MapperComplex(
            filter_bnds=filter_bnds,
            resolutions=np.array([res] * n_filters, dtype=int),
            gains=np.array([gain] * n_filters, dtype=float),
            clustering=AdaptiveKMeans(base_n_clusters=n_clusters, random_state=0, n_init=10),
            input_type="point cloud",
        )
        mapper.fit(coords, filters=filters, colors=filters)
        return mapper

    def _mapper_to_networkx(M) -> nx.Graph:
        G = nx.Graph()
        try:
            st = M.mapper_ if hasattr(M, "mapper_") else M.simplex_tree_
            for (splx, _) in st.get_skeleton(1):
                if len(splx) == 1: G.add_node(splx[0])
                elif len(splx) == 2: G.add_edge(splx[0], splx[1])
        except Exception:
            pass
        return G

    def _node_color_and_size_dict(M, color_dim: int = 0):
        cs = {}
        for k, info in M.node_info_.items():
            cnt = info.get("size", len(info.get("indices", [])))
            size = 20.0 + 2.0 * np.sqrt(max(1.0, float(cnt)))
            c = np.asarray(info.get("colors", 0.0))
            if c.ndim == 0:
                val = float(c)
            elif c.ndim == 1:
                val = float(c[color_dim]) if c.size > color_dim else float(np.mean(c))
            elif c.ndim == 2:
                val = float(np.mean(c[:, color_dim])) if c.shape[1] > color_dim else float(np.mean(c))
            else:
                val = float(np.mean(c))
            cs[k] = (val, size)
        vals = np.array([v for v, _ in cs.values()], dtype=float)
        if np.isfinite(vals).any():
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            norm = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vals)
            for i, k in enumerate(cs.keys()):
                cs[k] = (float(norm[i]), cs[k][1])
        return cs

    def plot_mapper_pruned(M, title: str = "Mapper Graph", color_dim: int = 0, layout: str = "spring",
                           min_component_size: int = 3, drop_isolates: bool = True):
        import matplotlib.pyplot as plt
        import networkx as nx
        fig = plt.figure(figsize=(8, 6))
        if M is None:
            plt.title(f"{title} (no data)")
            return fig
        G = _mapper_to_networkx(M)
        cs = _node_color_and_size_dict(M, color_dim=color_dim)
        Gp = G.copy()
        if drop_isolates:
            Gp.remove_nodes_from(list(nx.isolates(Gp)))
        if min_component_size and min_component_size > 1:
            for comp in list(nx.connected_components(Gp)):
                if len(comp) < min_component_size:
                    Gp.remove_nodes_from(comp)
        if Gp.number_of_nodes() == 0:
            Gp = G
        nodes = list(Gp.nodes())
        colors = np.array([cs[n][0] for n in nodes], dtype=float) if nodes else []
        sizes  = np.array([cs[n][1] for n in nodes], dtype=float) if nodes else []
        pos = nx.kamada_kawai_layout(Gp) if layout == "kamada" else nx.spring_layout(Gp, seed=0, iterations=150)
        nx.draw_networkx_edges(Gp, pos, width=1.0, alpha=0.5)
        if len(nodes):
            sc = nx.draw_networkx_nodes(Gp, pos, nodelist=nodes, node_size=sizes, node_color=colors,
                                        cmap="viridis", linewidths=0.3, edgecolors="black", alpha=0.95)
            plt.colorbar(sc, label=f"Filter dim {color_dim}")
        plt.title(title); plt.axis("off"); plt.tight_layout()
        return fig
else:
    build_mapper = _build_mapper_core
    plot_mapper_pruned = _plot_mapper_pruned_core

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
app_ui = ui.page_fluid(
    ui.tags.style((
        """
        body {
            background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 45%, #ecfeff 100%);
            color: #0f172a;
            font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
        }

        h2 {
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 18px;
            color: #0f172a;
        }

        .card {
            background: rgba(255, 255, 255, 0.92) !important;
            border: 1px solid rgba(148, 163, 184, 0.35) !important;
            border-radius: 18px !important;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08) !important;
            backdrop-filter: blur(10px);
        }

        .card-header {
            background: transparent !important;
            border-bottom: 1px solid rgba(148, 163, 184, 0.25) !important;
            font-weight: 700;
            color: #1e293b;
        }

        .sidebar {
            background: rgba(255, 255, 255, 0.88) !important;
            border-right: 1px solid rgba(148, 163, 184, 0.35);
            box-shadow: 8px 0 24px rgba(15, 23, 42, 0.05);
        }

        .shiny-input-container > label {
            color: #334155;
            font-weight: 650;
            font-size: 13px;
        }

        .btn-primary {
            background: linear-gradient(135deg, #2563eb, #4f46e5) !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.28);
        }

        .btn-secondary {
            background: linear-gradient(135deg, #0f766e, #0891b2) !important;
            border: none !important;
            border-radius: 12px !important;
            color: white !important;
            font-weight: 700 !important;
            box-shadow: 0 8px 18px rgba(8, 145, 178, 0.24);
        }

        .btn-danger {
            background: linear-gradient(135deg, #dc2626, #f97316) !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            box-shadow: 0 8px 18px rgba(220, 38, 38, 0.22);
        }

        .nav-tabs .nav-link {
            border: none !important;
            color: #475569;
            font-weight: 650;
        }

        .nav-tabs .nav-link.active {
            color: #1d4ed8 !important;
            background: white !important;
            border-radius: 12px 12px 0 0 !important;
            box-shadow: 0 -4px 16px rgba(15, 23, 42, 0.04);
        }

        table {
            font-size: 13px;
            border-radius: 12px;
            overflow: hidden;
        }

        .logbox {
            height: 240px;
            overflow-y: auto;
            background: #0f172a;
            color: #dbeafe;
            border: 1px solid #1e293b;
            padding: 12px;
            border-radius: 14px;
            white-space: pre-wrap;
            font-family: Consolas, 'Courier New', monospace;
            font-size: 12px;
        }

        .kpi-wrap {
            padding: 10px 0;
        }

        .kpi-wrap span {
            display: block;
            margin-bottom: 4px;
        }

        .muted {
            color: #64748b;
            font-size: 13px;
            font-weight: 600;
        }

        progress {
            accent-color: #2563eb;
        }
        """
    )),
    ui.h2("TDA-Based Imputation Quality Evaluator"),

    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Upload & Controls"),
            ui.input_text("dataset_name", "Dataset name (optional)", placeholder="e.g. ESR_MCAR20"),
            ui.input_file("file_raw", "Upload Raw Data (CSV)", accept=[".csv"], multiple=False),
            ui.input_file("file_cc", "Upload Complete Case (CSV)", accept=[".csv"], multiple=False),
            ui.input_file("file_imp", "Upload Imputed Data (CSV)", accept=[".csv"], multiple=False),
            ui.hr(),
            ui.input_checkbox_group(
                "na_tokens", "Tokens treated as NA",
                choices=["", "NA", "N/A", "NaN", "NULL", "-999", "?"],
                selected=["", "NA", "N/A", "NaN"]
            ),
            ui.input_checkbox("drop_na_preview", "Drop NA rows in preview", value=False),
            ui.hr(),
            ui.h5("Mapper parameters"),
            ui.input_slider("res", "Resolution", 2, 50, 10),
            ui.input_slider("gain", "Gain", 0.1, 1.0, 0.3, step=0.05),
            ui.input_checkbox("drop_isolates", "Remove isolated clusters", True),
            ui.input_slider("min_comp", "Min component size (prune)", 1, 20, 3),
            ui.input_select("layout", "Layout", {"spring": "Spring", "kamada": "Kamada-Kawai"}, selected="spring"),
            ui.input_numeric("mapper_max_samples", "Max rows for mapper", 2000, min=200, step=200),
            ui.hr(),
            ui.h5("Pipeline parameters"),
            ui.p(
                "Please refer to the Parameter Guide before running the pipeline.",
                class_="muted"
            ),
            ui.input_text("res_list", "Resolution list", value="5,8,10,15"),
            ui.input_text("gain_list", "Gain list", value="0.3,0.4"),
            ui.input_numeric("num_perm", "#Permutations (per combo)", 99, min=9, step=10),
            ui.input_numeric("knn_k", "k for kNN distance", 20, min=2),
            ui.input_numeric("n_clusters", "Base clusters per bin", 4, min=2),
            ui.input_numeric("n_jobs", "Parallel jobs", 4, min=1),
            ui.hr(),
            ui.row(
                ui.column(6, ui.input_action_button("btn_run_mapper", "Render Mapper", class_="btn-secondary")),
                ui.column(6, ui.input_action_button("btn_run_pipeline", "Start Pipeline", class_="btn-primary")),
            ),
            ui.row(
                ui.column(
                    12,
                    ui.input_action_button(
                        "btn_stop",
                        "Stop After Current Combo",
                        class_="btn-danger"
                    )
                ),
            ),
            ui.hr(),
            ui.p(ui.span("Status:", class_="muted"), ui.output_text("status_text")),
            ui.output_ui("progress_bar"),
            ui.hr(),
            ui.card(
                ui.card_header("Debug / Upload status"),
                ui.output_text_verbatim("debug_status")
            ),
        ),

        ui.navset_tab(
            ui.nav_panel(
                "Data Overview",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Dimensions & File Info"),
                        ui.output_text("dims_raw"), ui.br(),
                        ui.output_text("dims_cc"), ui.br(),
                        ui.output_text("dims_imp"),
                        ui.hr(),
                        ui.card_header("Raw Missing Summary"),
                        ui.output_text("raw_missing_kpi"),
                    ),
                    ui.card(
                        ui.card_header("Preview — first 10 rows and 12 columns"),
                        ui.input_radio_buttons(
                            "preview_choice", "Dataset",
                            {"raw": "Raw", "cc": "Complete", "imp": "Imputed"},
                            selected="cc", inline=True
                        ),
                        ui.output_table("preview_table"),
                    ),
                    col_widths=(4, 8)
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Missing Value Summary — top 15 missing columns"),
                        ui.input_radio_buttons(
                            "miss_choice", "Dataset",
                            {"raw": "Raw", "cc": "Complete", "imp": "Imputed"},
                            selected="cc", inline=True
                        ),
                        ui.output_table("missing_table"),
                    ),
                    ui.card(
                        ui.card_header("Most Missing Columns (Barplot)"),
                        ui.input_radio_buttons(
                            "miss_plot_choice", "Dataset",
                            {"raw": "Raw", "cc": "Complete", "imp": "Imputed"},
                            selected="cc", inline=True
                        ),
                        ui.output_plot("missing_bar", height="360px"),
                    ),
                )
            ),
            ui.nav_panel(
                "Mapper Graph",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Complete Case - Mapper"),
                        ui.output_plot("mapper_cc", height="520px"),
                        ui.output_text("mapper_cc_stats"),
                    ),
                    ui.card(
                        ui.card_header("Imputed Data - Mapper"),
                        ui.output_plot("mapper_imp", height="520px"),
                        ui.output_text("mapper_imp_stats"),
                    ),
                )
            ),
            ui.nav_panel(
                "Pipeline Monitor",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Parameter Queue & Progress"),
                        ui.output_table("queue_table"),
                        ui.br(),
                        ui.output_text("progress_text"),
                    ),
                    ui.card(
                        ui.card_header("Live Log"),
                        ui.tags.div(ui.output_text_verbatim("live_log"), class_="logbox"),
                    ),
                ),
                ui.card(
                    ui.card_header("Streaming Results Table"),
                    ui.output_table("results_table"),
                )
            ),
            ui.nav_panel(
                "Evaluation Results",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("KPI - Significant Counts (p<0.05)"),
                        ui.tags.div(
                            ui.tags.div(ui.span("Significant (p<0.05)", class_="muted"), ui.output_text("kpi_sig"), class_="kpi-wrap"),
                            ui.tags.div(ui.span("Total runs", class_="muted"), ui.output_text("kpi_tot"), class_="kpi-wrap"),
                            ui.tags.div(ui.span("Significance rate", class_="muted"), ui.output_text("kpi_rate"), class_="kpi-wrap"),
                        ),
                    ),
                    ui.card(
                        ui.card_header("p-value Distribution"),
                        ui.output_plot("p_hist", height="360px"),
                    ),
                )
            ),
        )
    )
)

# ------------------------------------------------------------
# Server
# ------------------------------------------------------------
def server(input, output, session):
    # Upload bytes only
    raw_bytes = reactive.Value(b"")
    cc_bytes  = reactive.Value(b"")
    imp_bytes = reactive.Value(b"")

    # Parsed DataFrames
    df_raw = reactive.Value(pd.DataFrame())
    df_cc  = reactive.Value(pd.DataFrame())
    df_imp = reactive.Value(pd.DataFrame())

    # States
    started_mapper  = reactive.Value(False)
    started_pipe    = reactive.Value(False)
    is_running      = reactive.Value(False)
    progress        = reactive.Value(0.0)
    live_log_lines  = reactive.Value([])

    # Mapper cache
    mapper_trigger   = reactive.Value(0)
    rendering_mapper = reactive.Value(False)
    mapper_fig_cc    = reactive.Value(None)
    mapper_fig_imp   = reactive.Value(None)

    # Debug counters
    click_mapper_ct  = reactive.Value(0)
    click_pipe_ct    = reactive.Value(0)

    # Pipeline results & queue
    results_df = reactive.Value(pd.DataFrame(columns=[
        "timestamp", "res", "gain", "dataset_pair", "metric", "p_value", "significant", "elapsed_s", "note"
    ]))
    param_grid = reactive.Value(pd.DataFrame(columns=["res", "gain", "status"]))

    # Helpers
    def _append_log(line: str):
        lines = list(live_log_lines()); ts = datetime.now().strftime("%H:%M:%S")
        lines.append(f"[{ts}] {line}"); live_log_lines.set(lines)

    def _wait_df(msg="Waiting for action..."):
        return pd.DataFrame({"Message": [msg]})

    def _parse_list(txt: str, cast):
        if not str(txt).strip(): return []
        out = []
        for t in str(txt).replace(";", ",").split(","):
            tt = t.strip()
            if tt: out.append(cast(tt))
        return out

    # read upload via datapath (stable)
    def _read_upload_file(item) -> bytes:
        if not item: return b""
        info = item[0]
        path = info.get("datapath") if isinstance(info, dict) else getattr(info, "datapath", None)
        if path and os.path.exists(path):
            with open(path, "rb") as fh:
                return fh.read()
        return info.read() if hasattr(info, "read") else b""

    @reactive.effect
    @reactive.event(input.file_raw)
    def _on_raw_upload():
        try:
            data = _read_upload_file(input.file_raw())
            if data:
                raw_bytes.set(data)
                _append_log(f"[UPLOAD] Raw ({len(data)} bytes)")
        except Exception as e:
            _append_log(f"[UPLOAD RAW][ERROR] {type(e).__name__}: {e}")

    @reactive.effect
    @reactive.event(input.file_cc)
    def _on_cc_upload():
        try:
            data = _read_upload_file(input.file_cc())
            if data:
                cc_bytes.set(data)
                _append_log(f"[UPLOAD] Complete ({len(data)} bytes)")
        except Exception as e:
            _append_log(f"[UPLOAD CC][ERROR] {type(e).__name__}: {e}")

    @reactive.effect
    @reactive.event(input.file_imp)
    def _on_imp_upload():
        try:
            data = _read_upload_file(input.file_imp())
            if data:
                imp_bytes.set(data)
                _append_log(f"[UPLOAD] Imputed ({len(data)} bytes)")
        except Exception as e:
            _append_log(f"[UPLOAD IMP][ERROR] {type(e).__name__}: {e}")

    # Status / Debug
    @output
    @render.ui
    def progress_bar():
        return ui.tags.progress(value=int(progress() * 100), max=100, style="width:100%;height:16px;")

    @output
    @render.text
    def status_text():
        if is_running(): return "Pipeline: Running"
        if started_pipe(): return "Pipeline: Ready"
        if started_mapper(): return "Mapper: Ready"
        return "Idle"

    @output
    @render.text
    def dims_raw():

        if not df_raw().empty:
            return f"Raw: {df_raw().shape[0]} rows x {df_raw().shape[1]} columns"

        if raw_bytes():
            return "Raw: Uploaded"

        return "Raw: Not loaded"

    @output
    @render.text
    def dims_cc():
        if not df_cc().empty: return f"Complete: {df_cc().shape[0]} rows x {df_cc().shape[1]} columns"
        return "Complete: Uploaded (not parsed yet)" if cc_bytes() else "Complete: Not loaded"

    @output
    @render.text
    def dims_imp():
        if not df_imp().empty: return f"Imputed: {df_imp().shape[0]} rows x {df_imp().shape[1]} columns"
        return "Imputed: Uploaded (not parsed yet)" if imp_bytes() else "Imputed: Not loaded"
    
    @output
    @render.text
    def raw_missing_kpi():
        d = df_raw()

        if d.empty:
            return "Raw data has not been parsed yet."

        total_cells = d.shape[0] * d.shape[1]
        missing_cells = int(d.isna().sum().sum())
        missing_rate = missing_cells / total_cells * 100 if total_cells > 0 else 0

        cols_with_missing = int((d.isna().sum() > 0).sum())

        return (
            f"Missing cells: {missing_cells:,}\n"
            f"Missing rate: {missing_rate:.2f}%\n"
            f"Columns with missing values: {cols_with_missing:,}"
        )

    @output
    @render.text
    def debug_status():
        return (
            f"raw_bytes={len(raw_bytes() or b''):,} | "
            f"cc_bytes={len(cc_bytes() or b''):,} | "
            f"imp_bytes={len(imp_bytes() or b''):,}\n\n"
            f"RenderMapper clicks={click_mapper_ct()} | "
            f"StartPipeline clicks={click_pipe_ct()}"
        )

    # Overview
    def _ensure_raw_parsed_if_needed():
        if df_raw().empty and raw_bytes() and input.preview_choice() == "raw":
            try:
                df_raw.set(read_csv_bytes(raw_bytes()))
            except Exception as e:
                _append_log(f"[PARSE RAW][ERROR] {type(e).__name__}: {e}")

    def _get_preview_df():
        _ensure_raw_parsed_if_needed()
        choice = input.preview_choice()
        _df = {"raw": df_raw(), "cc": df_cc(), "imp": df_imp()}[choice].copy()
        if _df.empty: return pd.DataFrame({"Message": ["Please upload data"]})
        if input.drop_na_preview(): _df = _df.dropna()
        return _df.iloc[:10, :12]

    @output
    @render.table
    def preview_table():
        if df_cc().empty and df_imp().empty and df_raw().empty:
            return _wait_df("Click 'Render Mapper' or 'Start Pipeline' first.")
        return _get_preview_df()

    @output
    @render.table
    def missing_table():

        if df_cc().empty and df_imp().empty and df_raw().empty:
            return _wait_df("No parsed data yet.")

        choice = input.miss_choice()

        _df = {
            "raw": df_raw(),
            "cc": df_cc(),
            "imp": df_imp()
        }[choice]

        ms = missing_summary(_df)

        # Only keep columns with missing values
        ms = ms[ms["Missing"] > 0]

        # Show only top 15
        ms = ms.head(15)

        if ms.empty:
            return pd.DataFrame({
                "Message": ["No missing values detected."]
            })

        return ms

    @output
    @render.plot
    def missing_bar():
        import matplotlib.pyplot as plt
        if df_cc().empty and df_imp().empty and df_raw().empty:
            fig = plt.figure(figsize=(8, 3.6)); plt.title("No parsed data yet"); return fig
        choice = input.miss_plot_choice()
        df_sel = {"raw": df_raw(), "cc": df_cc(), "imp": df_imp()}[choice]
        return top_missing_bar_mpl(df_sel)

    # Mapper
    @reactive.effect
    @reactive.event(input.btn_run_mapper)
    def _on_render_mapper():
        click_mapper_ct.set(click_mapper_ct() + 1)
        _append_log("[CLICK] Render Mapper")
        need = []
        if not cc_bytes(): need.append("Complete")
        if not imp_bytes(): need.append("Imputed")
        if need:
            _append_log("Please upload " + " & ".join(need) + " before rendering mapper.")
            return
        try:
            if df_raw().empty and raw_bytes():
                df_raw.set(read_csv_bytes(raw_bytes()))

            if df_cc().empty:
                df_cc.set(read_csv_bytes(cc_bytes()))

            if df_imp().empty:
                df_imp.set(read_csv_bytes(imp_bytes()))
            started_mapper.set(True)
            mapper_trigger.set(mapper_trigger() + 1)
        except Exception as e:
            _append_log(f"[PARSE][ERROR] {type(e).__name__}: {e}")

    @reactive.effect
    @reactive.event(mapper_trigger)
    def _compute_mapper_plots():
        import matplotlib.pyplot as plt
        if not started_mapper(): return
        if rendering_mapper(): return
        rendering_mapper.set(True)
        try:
            # CC
            if df_cc().empty:
                mapper_fig_cc.set(None)
            else:
                d = df_cc()
                max_n = int(input.mapper_max_samples())
                if len(d) > max_n: d = d.sample(max_n, random_state=1)
                coords, fil = preprocess_filters_single(d, use_2d=True, knn_k=int(input.knn_k()))
                M = build_mapper(coords, fil, res=int(input.res()), gain=float(input.gain()),
                                 n_clusters=int(input.n_clusters()))
                fig_cc = plot_mapper_pruned(
                    M, title="Complete Case - Mapper",
                    color_dim=0, layout=input.layout(),
                    min_component_size=int(input.min_comp()), drop_isolates=bool(input.drop_isolates())
                )
                mapper_fig_cc.set(fig_cc)
            # Imputed
            if df_imp().empty:
                mapper_fig_imp.set(None)
            else:
                d2 = df_imp()
                max_n = int(input.mapper_max_samples())
                if len(d2) > max_n: d2 = d2.sample(max_n, random_state=1)
                coords2, fil2 = preprocess_filters_single(d2, use_2d=True, knn_k=int(input.knn_k()))
                M2 = build_mapper(coords2, fil2, res=int(input.res()), gain=float(input.gain()),
                                  n_clusters=int(input.n_clusters()))
                fig_imp = plot_mapper_pruned(
                    M2, title="Imputed Data - Mapper",
                    color_dim=0, layout=input.layout(),
                    min_component_size=int(input.min_comp()), drop_isolates=bool(input.drop_isolates())
                )
                mapper_fig_imp.set(fig_imp)

            _append_log("Mapper plots rendered.")
        except Exception as e:
            msg = str(e)
            if len(msg) > 120: msg = msg[:117] + "..."
            err_fig = plt.figure(figsize=(8, 4)); plt.title(f"Mapper error: {msg}")
            mapper_fig_cc.set(err_fig); mapper_fig_imp.set(err_fig)
        finally:
            rendering_mapper.set(False)

    @output
    @render.plot
    def mapper_cc():
        import matplotlib.pyplot as plt
        fig = mapper_fig_cc()
        if fig is None:
            f = plt.figure(figsize=(8, 4)); plt.title("Click 'Render Mapper'"); return f
        return fig

    @output
    @render.plot
    def mapper_imp():
        import matplotlib.pyplot as plt
        fig = mapper_fig_imp()
        if fig is None:
            f = plt.figure(figsize=(8, 4)); plt.title("Click 'Render Mapper'"); return f
        return fig

    @output
    @render.text
    def mapper_cc_stats():
        d = df_cc();  return "No data loaded" if d.empty else f"Rows: {len(d)} (plot may sample)"

    @output
    @render.text
    def mapper_imp_stats():
        d = df_imp();  return "No data loaded" if d.empty else f"Rows: {len(d)} (plot may sample)"

    # Pipeline
    @output
    @render.table
    def queue_table(): return param_grid()

    @output
    @render.text
    def progress_text(): return f"Completed: {int(progress() * 100)}%"

    @output
    @render.text
    def live_log(): return "\n".join(live_log_lines())

    @output
    @render.table
    def results_table():
        df = results_df()
        if df.empty: return pd.DataFrame({"Message": ["No results yet. Click 'Start Pipeline'."]})
        return df

    @output
    @render.text
    def kpi_sig():
        df = results_df();  return "0" if df.empty else str(int((df["p_value"] < 0.05).sum()))

    @output
    @render.text
    def kpi_tot():
        return str(len(results_df()))
    
    @output
    @render.text
    def kpi_rate():
        df = results_df()

        if df.empty:
            return "0.00%"

        total = len(df)

        if total == 0:
            return "0.00%"

        sig_count = int((df["p_value"] < 0.05).sum())
        rate = sig_count / total * 100

        return f"{rate:.2f}%"

    @output
    @render.plot
    def p_hist():
        import matplotlib.pyplot as plt
        df = results_df()
        if df.empty:
            fig = plt.figure(figsize=(8, 3.6)); plt.title("No p-values yet"); return fig
        vals, bins = np.asarray(df["p_value"].values, dtype=float), 20
        fig = plt.figure(figsize=(8, 3.6))
        ax = fig.gca()
        ax.hist(vals, bins=bins, range=(0, 1))
        ax.set_title("p-value distribution"); ax.set_xlim(0, 1)
        return fig

    @reactive.effect
    @reactive.event(input.btn_run_pipeline)
    def _on_start_pipeline():
        click_pipe_ct.set(click_pipe_ct() + 1)
        _append_log("[CLICK] Start Pipeline")

        need = []
        if not cc_bytes(): need.append("Complete")
        if not imp_bytes(): need.append("Imputed")
        if need:
            _append_log("Please upload " + " & ".join(need) + " before starting pipeline.")
            return

        # Check essential core functions
        missing = [name for name in
                   ["preprocess_pool", "build_mapper_from_indices", "bottleneck_statmapper", "TOPO_TYPES"]
                   if not hasattr(core, name)]
        if missing:
            _append_log(f"[PIPELINE][ERROR] tda_core missing: {', '.join(missing)}. "
                        f"Please add these to tda_core.py or let me add fallbacks.")
            return

        try:
            if df_raw().empty and raw_bytes():
                df_raw.set(read_csv_bytes(raw_bytes()))

            if df_cc().empty:
                df_cc.set(read_csv_bytes(cc_bytes()))

            if df_imp().empty:
                df_imp.set(read_csv_bytes(imp_bytes()))
        except Exception as e:
            _append_log(f"[PARSE][ERROR] {type(e).__name__}: {e}")
            return

        started_mapper.set(True)
        started_pipe.set(True)
        is_running.set(False)
        progress.set(0)
        live_log_lines.set([])
        results_df.set(results_df().iloc[0:0])

        rlist = _parse_list(input.res_list(), int)
        glist = _parse_list(input.gain_list(), float)
        grid = pd.DataFrame({
            "res":  [r for r in rlist for g in glist],
            "gain": [g for r in rlist for g in glist],
        })
        if grid.empty:
            started_pipe.set(False)
            _append_log("Parameter grid is empty - check Resolution/Gain lists.")
            return
        grid["status"] = "Pending"
        param_grid.set(grid)
        _append_log(f"Pipeline armed: {len(grid)} combos.")
        reactive.invalidate_later(0.4)
        is_running.set(True)

    @reactive.effect
    @reactive.event(input.btn_stop)
    def _on_stop():
        is_running.set(False)
        started_pipe.set(False)
        progress.set(0)
        _append_log("Stop requested. Current combo will finish first.")
        if len(param_grid()) > 0:
            g = param_grid().copy()
            g["status"] = "Pending"
            param_grid.set(g)

    @reactive.effect
    def _runner_tick():
        reactive.invalidate_later(0.8)
        if not is_running(): return
        grid = param_grid().copy()
        if grid.empty:
            is_running.set(False); progress.set(1); return
        pending_idx = grid.index[grid["status"] == "Pending"]
        if len(pending_idx) == 0:
            is_running.set(False); progress.set(1); _append_log("All parameter combos completed."); return

        i = pending_idx[0]
        res = int(grid.at[i, "res"]); gain = float(grid.at[i, "gain"])
        grid.at[i, "status"] = "Running"; param_grid.set(grid)

        try:
            coords, fil, idx_A, idx_B = core.preprocess_pool(df_cc(), df_imp(),
                                                             knn_k=int(input.knn_k()), use_2d_filter=True)
            if coords.size == 0: raise RuntimeError("No numeric columns found in uploaded datasets.")
            M_A, F_A = core.build_mapper_from_indices(coords, fil, idx_A, res, gain, int(input.n_clusters()), 0)
            M_B, F_B = core.build_mapper_from_indices(coords, fil, idx_B, res, gain, int(input.n_clusters()), 0)
            parts_obs = [core.bottleneck_statmapper((M_A, F_A), (M_B, F_B), topo) for topo in core.TOPO_TYPES]
            Tobs = float(max(parts_obs))

            num_perm = int(input.num_perm())
            n_jobs_val = int(input.n_jobs())
            n_clusters_val = int(input.n_clusters())

            n1 = len(idx_A)
            n = len(coords)

            from joblib import Parallel, delayed

            def _one_perm(seed):
                rng = np.random.default_rng(seed)
                perm = rng.permutation(np.arange(n))

                A_idx = perm[:n1]
                B_idx = perm[n1:]

                MA, FA = core.build_mapper_from_indices(
                    coords, fil, A_idx, res, gain, n_clusters_val, 0
                )

                MB, FB = core.build_mapper_from_indices(
                    coords, fil, B_idx, res, gain, n_clusters_val, 0
                )

                T = max(
                    core.bottleneck_statmapper((MA, FA), (MB, FB), topo)
                    for topo in core.TOPO_TYPES
                )

                return T

            perm_T = Parallel(
                n_jobs=n_jobs_val,
                backend="threading"
            )(
                delayed(_one_perm)(seed)
                for seed in range(num_perm)
            )
            perm_T = np.array(perm_T)
            p_max = float((np.sum(perm_T >= Tobs) + 1) / (num_perm + 1))

            _append_log(f"res={res}, gain={gain} -> Tobs={Tobs:.4f}, p={p_max:.4f}")
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "res": res, "gain": gain, "dataset_pair": "Complete vs Imputed",
                "metric": Tobs, "p_value": p_max, "significant": (p_max < 0.05),
                "elapsed_s": np.nan, "note": "pipeline",
            }
            results_df.set(pd.concat([results_df(), pd.DataFrame([row])], ignore_index=True))
            grid.at[i, "status"] = "Done"; param_grid.set(grid)
        except Exception as e:
            _append_log(f"Error at res={res}, gain={gain}: {e}")
            grid.at[i, "status"] = "Failed"; param_grid.set(grid)
        finally:
            done = (grid["status"] == "Done").sum(); total = len(grid); progress.set(done/total if total else 1)

app = App(app_ui, server)
