"""
MatchInsight — EPL Match Outcome Prediction Dashboard
Team 16 | CMPS344 Applied Data Science

Run with:
    streamlit run dashboard.py

Reads live results from your pipeline outputs:
    results/model_results.csv
    results/<model>/metrics.json
    results/<model>/confusion.csv
    results/feature_selection/permutation_importance.png
    results/feature_selection/shap_summary.png
    results/feature_selection/rfecv_curve.png
    data/matches/preprocessed_train.csv
    data/matches/preprocessed_test.csv
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from PIL import Image

# ─── Paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR            = Path("results")
MODEL_RESULTS_PATH     = RESULTS_DIR / "model_results.csv"
TRAIN_PATH             = Path("data/matches/preprocessed_train.csv")
TEST_PATH              = Path("data/matches/preprocessed_test.csv")
FEATURES_LINEAR_PATH   = Path("data/matches/features_linear.csv")
FEATURES_TREES_PATH    = Path("data/matches/features_trees.csv")
PERM_IMP_IMG           = RESULTS_DIR / "feature_selection" / "permutation_importance.png"
SHAP_IMG               = RESULTS_DIR / "feature_selection" / "shap_summary.png"
RFECV_IMG              = RESULTS_DIR / "feature_selection" / "rfecv_curve.png"

MODEL_NAMES = [
    "home_win_baseline",
    "logistic_regression",
    "svm_rbf",
    "knn",
    "gaussian_nb",
    "random_forest",
    "gradient_boosting",
]

TARGET_LABELS = {0: "Away Win", 1: "Draw", 2: "Home Win"}
TARGET_COLORS = {"Away Win": "#4d9fff", "Draw": "#f5a623", "Home Win": "#22c97e"}

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="monospace", size=12, color="#c8d0e8"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.1)"),
    margin=dict(l=16, r=16, t=36, b=16),
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MatchInsight · EPL Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700&family=IBM+Plex+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #0a0d12; }
section[data-testid="stSidebar"] { background: #111520 !important; border-right: 1px solid rgba(255,255,255,0.07); }
section[data-testid="stSidebar"] * { color: #c8d0e8 !important; }

h1, h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; }

.metric-card {
    background: #1a2030; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 18px 20px; text-align: center;
}
.metric-card .label {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    text-transform: uppercase; letter-spacing: 1px; color: #5a6480; margin-bottom: 6px;
}
.metric-card .value {
    font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 700;
}
.metric-card .sub { font-size: 11px; color: #5a6480; margin-top: 4px; }

.section-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    text-transform: uppercase; letter-spacing: 1.2px; color: #5a6480;
    margin-bottom: 4px;
}
.card {
    background: #1a2030; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 20px 22px; margin-bottom: 16px;
}
.stSelectbox label, .stRadio label { color: #8a94b0 !important; font-size: 12px !important; }
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ─── Data loaders (cached) ───────────────────────────────────────────────────

@st.cache_data
def load_model_results() -> pd.DataFrame | None:
    if not MODEL_RESULTS_PATH.exists():
        return None
    df = pd.read_csv(MODEL_RESULTS_PATH)
    return df


@st.cache_data
def load_model_detail(model_name: str) -> dict | None:
    path = RESULTS_DIR / model_name / "metrics.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


@st.cache_data
def load_confusion(model_name: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / model_name / "confusion.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_train() -> pd.DataFrame | None:
    if not TRAIN_PATH.exists():
        return None
    return pd.read_csv(TRAIN_PATH)


@st.cache_data
def load_test() -> pd.DataFrame | None:
    if not TEST_PATH.exists():
        return None
    return pd.read_csv(TEST_PATH)


@st.cache_data
def load_features_linear() -> list[str]:
    if not FEATURES_LINEAR_PATH.exists():
        return []
    df = pd.read_csv(FEATURES_LINEAR_PATH)
    return [c for c in df.columns if c != "FTR"]


@st.cache_data
def load_features_trees() -> list[str]:
    if not FEATURES_TREES_PATH.exists():
        return []
    df = pd.read_csv(FEATURES_TREES_PATH)
    return [c for c in df.columns if c != "FTR"]


def metric_card(label: str, value: str, sub: str = "", color: str = "#22c97e"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value" style="color:{color};">{value}</div>
        <div class="sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def check_results_exist() -> bool:
    return MODEL_RESULTS_PATH.exists()


def check_data_exists() -> bool:
    return TRAIN_PATH.exists()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ MatchInsight")
    st.markdown(
        "<div style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#5a6480;margin-bottom:20px;'>"
        "EPL Outcome Predictor · Team 16<br>CMPS344 Applied Data Science</div>",
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        ["Overview", "EDA", "Model Comparison", "Feature Analysis", "Business Insights"],
        label_visibility="collapsed",
    )

    st.divider()

    results_ok = check_results_exist()
    data_ok = check_data_exists()

    st.markdown("**Pipeline status**")
    st.markdown(
        f"{'✅' if data_ok else '⚠️'} Training data",
        unsafe_allow_html=False,
    )
    st.markdown(f"{'✅' if results_ok else '⚠️'} Model results")

    linear_feats = load_features_linear()
    tree_feats = load_features_trees()
    st.markdown(f"{'✅' if linear_feats else '⚠️'} Linear features ({len(linear_feats)})")
    st.markdown(f"{'✅' if tree_feats else '⚠️'} Tree features ({len(tree_feats)})")

    if not results_ok or not data_ok:
        st.warning("Run `python pipeline.py` to generate results.", icon="⚠️")

    st.divider()
    if st.button("🔄 Clear cache & refresh"):
        st.cache_data.clear()
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("EPL Match Outcome Predictor")
    st.markdown(
        "<div style='color:#8a94b0;font-size:14px;margin-bottom:28px;'>"
        "Predicting Premier League results (H / D / A) from pre-match features across 10 seasons."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Top metrics ────────────────────────────────────────────────────────────
    results = load_model_results()
    train = load_train()
    test = load_test()

    total_matches = (len(train) + len(test)) if (train is not None and test is not None) else "—"
    n_train = len(train) if train is not None else "—"
    n_test  = len(test)  if test is not None else "—"

    best_acc   = "—"
    best_f1    = "—"
    best_model = "—"
    n_models   = len(MODEL_NAMES)

    if results is not None:
        best_row   = results.sort_values("macro_f1", ascending=False).iloc[0]
        best_acc   = f"{best_row['accuracy']*100:.1f}%"
        best_f1    = f"{best_row['macro_f1']:.3f}"
        best_model = best_row["model"].replace("_", " ").title()

    cols = st.columns(6)
    with cols[0]: metric_card("Total Matches", str(total_matches), "10 EPL seasons", "#22c97e")
    with cols[1]: metric_card("Train / Test", f"{n_train} / {n_test}", "Temporal split", "#4d9fff")
    with cols[2]: metric_card("Models Trained", str(n_models), "incl. baseline", "#9b7fff")
    with cols[3]: metric_card("Best Accuracy", best_acc, best_model, "#22c97e")
    with cols[4]: metric_card("Best Macro F1", best_f1, best_model, "#f5a623")
    with cols[5]: metric_card("Data Sources", "3", "Stats · ELO · xG", "#4d9fff")

    st.divider()

    # ── Class distribution (from real train data) ──────────────────────────────
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader("Target class distribution")
        if train is not None and "FTR" in train.columns:
            counts = train["FTR"].value_counts().sort_index()
            label_map = {0: "Away Win (A)", 1: "Draw (D)", 2: "Home Win (H)"}
            color_map = {"Away Win (A)": "#4d9fff", "Draw (D)": "#f5a623", "Home Win (H)": "#22c97e"}
            labels = [label_map.get(i, str(i)) for i in counts.index]
            fig = go.Figure(go.Bar(
                x=labels, y=counts.values,
                marker_color=[color_map.get(l, "#8a94b0") for l in labels],
                text=[f"{v}<br>{v/counts.sum()*100:.1f}%" for v in counts.values],
                textposition="outside",
            ))
            fig.update_layout(**PLOTLY_THEME, title="Training set class distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run pipeline to see class distribution from real training data.")

    with col_b:
        st.subheader("Model performance summary")
        if results is not None:
            fig = go.Figure()
            results_sorted = results.sort_values("macro_f1", ascending=True)
            colors = ["#22c97e" if r == results_sorted.iloc[-1]["model"] else "#4d9fff"
                      for r in results_sorted["model"]]
            fig.add_trace(go.Bar(
                y=results_sorted["model"].str.replace("_", " "),
                x=results_sorted["macro_f1"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:.3f}" for v in results_sorted["macro_f1"]],
                textposition="outside",
                name="Macro F1",
            ))
            fig.update_layout(**PLOTLY_THEME, title="Test macro F1 — all models",
                              xaxis_range=[0, results_sorted["macro_f1"].max() * 1.2])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run pipeline to see model results.")

    # ── Pipeline diagram ───────────────────────────────────────────────────────
    st.subheader("Pipeline architecture")
    steps = [
        ("football-data.co.uk", "Season CSVs", "#1a2030"),
        ("Understat", "xG per match", "#1a2030"),
        ("ClubElo", "ELO ratings", "#1a2030"),
        ("collect.py", "Merge & combine", "#1f2840"),
        ("validate.py", "8 checks", "#1f2840"),
        ("engineer.py", "ELO · Odds · Form · xG", "#1f2840"),
        ("preprocess.py", "Split · Encode · Scale", "#1f2840"),
        ("_select.py", "VIF · RFECV · SHAP", "#1f2840"),
        ("train.py", "7 classifiers", "#0f6e40"),
    ]
    pipe_html = "<div style='display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:8px 0;'>"
    for i, (name, detail, bg) in enumerate(steps):
        border = "1px solid #22c97e" if bg == "#0f6e40" else "1px solid rgba(255,255,255,0.1)"
        pipe_html += (
            f"<div style='background:{bg};border:{border};border-radius:8px;padding:8px 14px;"
            f"font-family:IBM Plex Mono,monospace;font-size:11px;'>"
            f"<div style='color:#5a6480;font-size:9px;text-transform:uppercase;'>{detail}</div>"
            f"<div style='color:#e8edf8;font-weight:500;margin-top:2px;'>{name}</div></div>"
        )
        if i < len(steps) - 1:
            pipe_html += "<span style='color:#5a6480;font-size:18px;'>→</span>"
    pipe_html += "</div>"
    st.markdown(pipe_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    st.markdown(
        "<div style='color:#8a94b0;font-size:14px;margin-bottom:24px;'>"
        "Live analysis from <code>collected.csv</code> (raw) and "
        "<code>preprocessed_train.csv</code> (engineered features).</div>",
        unsafe_allow_html=True,
    )

    # ── Load data sources ─────────────────────────────────────────────────────
    COLLECTED_PATH = Path("data/matches/collected.csv")

    @st.cache_data
    def load_collected():
        if not COLLECTED_PATH.exists():
            return None
        df = pd.read_csv(COLLECTED_PATH)
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        return df

    raw   = load_collected()
    train = load_train()
    trees_feats = load_features_trees()

    LABEL_MAP  = {0: "Away Win", 1: "Draw", 2: "Home Win"}
    FTR_COLORS = {"Away Win": "#4d9fff", "Draw": "#f5a623", "Home Win": "#22c97e",
                  "A": "#4d9fff", "D": "#f5a623", "H": "#22c97e"}

    if raw is None and train is None:
        st.error("No data found. Run `python pipeline.py` first.")
        st.stop()

    # ── Summary cards ─────────────────────────────────────────────────────────
    total     = len(raw)   if raw   is not None else len(train)
    n_seasons = raw["Season"].nunique() if raw is not None else "—"
    n_teams   = raw["HomeTeam"].nunique() if raw is not None else "—"
    n_feats   = len(trees_feats) if trees_feats else (
        len([c for c in train.select_dtypes("number").columns if c != "FTR"])
        if train is not None else "—"
    )

    cols = st.columns(4)
    with cols[0]: metric_card("Total Matches",  str(total),     "across all seasons",   "#22c97e")
    with cols[1]: metric_card("Seasons",         str(n_seasons), "EPL seasons",          "#4d9fff")
    with cols[2]: metric_card("Teams",           str(n_teams),   "unique clubs",         "#9b7fff")
    with cols[3]: metric_card("Engineered Feats",str(n_feats),   "after feature select", "#f5a623")

    st.divider()

    eda_tab1, eda_tab2, eda_tab3, eda_tab4, eda_tab5 = st.tabs([
        "Outcomes", "Feature Explorer", "Correlations", "Domain Signals", "Data Quality"
    ])

    # ══════════════════════════════════════════════════════════════
    # TAB 1 – OUTCOMES
    # ══════════════════════════════════════════════════════════════
    with eda_tab1:
        src = raw if raw is not None else train

        # Overall distribution
        if raw is not None:
            counts = raw["FTR"].value_counts()
            total_r = counts.sum()
            df_dist = pd.DataFrame({
                "Outcome": counts.index,
                "Count":   counts.values,
                "Pct":     (counts.values / total_r * 100).round(1),
            })
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = go.Figure(go.Bar(
                    x=df_dist["Outcome"], y=df_dist["Count"],
                    marker_color=[FTR_COLORS.get(o, "#8a94b0") for o in df_dist["Outcome"]],
                    text=[f"{r}<br>{p}%" for r, p in zip(df_dist["Count"], df_dist["Pct"])],
                    textposition="outside",
                ))
                fig.update_layout(**PLOTLY_THEME, showlegend=False,
                                  title="Overall match outcome distribution",
                                  yaxis_range=[0, df_dist["Count"].max() * 1.2])
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig2 = px.pie(df_dist, names="Outcome", values="Count",
                              color="Outcome",
                              color_discrete_map={"H": "#22c97e", "D": "#f5a623", "A": "#4d9fff"},
                              hole=0.5)
                fig2.update_layout(**PLOTLY_THEME, showlegend=True)
                st.plotly_chart(fig2, use_container_width=True)

        # Per-season outcome breakdown
        if raw is not None and "Season" in raw.columns:
            st.subheader("Outcome rate per season")
            season_counts = (
                raw.groupby(["Season", "FTR"]).size()
                .unstack(fill_value=0)
                .apply(lambda r: r / r.sum() * 100, axis=1)
            )
            fig_s = go.Figure()
            for outcome, color in [("H", "#22c97e"), ("D", "#f5a623"), ("A", "#4d9fff")]:
                if outcome in season_counts.columns:
                    fig_s.add_trace(go.Scatter(
                        x=season_counts.index, y=season_counts[outcome].round(1),
                        mode="lines+markers", name={"H": "Home Win", "D": "Draw", "A": "Away Win"}[outcome],
                        line=dict(color=color, width=2),
                        marker=dict(color=color, size=7),
                    ))
            fig_s.update_layout(**PLOTLY_THEME,
                                title="Outcome rate (%) per EPL season",
                                yaxis_title="Rate (%)",
                                legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_s, use_container_width=True)

        # Goals distribution
        if raw is not None and "FTHG" in raw.columns:
            st.subheader("Goals distribution")
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig_g = go.Figure()
                fig_g.add_trace(go.Histogram(x=raw["FTHG"], name="Home goals",
                                             marker_color="#22c97e", opacity=0.75, nbinsx=12))
                fig_g.add_trace(go.Histogram(x=raw["FTAG"], name="Away goals",
                                             marker_color="#4d9fff", opacity=0.75, nbinsx=12))
                fig_g.update_layout(**PLOTLY_THEME, barmode="overlay",
                                    title="Full-time goals per match",
                                    legend=dict(bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_g, use_container_width=True)
            with col_g2:
                avg_goals = raw.groupby("Season")[["FTHG", "FTAG"]].mean().reset_index()
                fig_avg = go.Figure()
                fig_avg.add_trace(go.Bar(x=avg_goals["Season"], y=avg_goals["FTHG"].round(2),
                                         name="Home avg", marker_color="#22c97e"))
                fig_avg.add_trace(go.Bar(x=avg_goals["Season"], y=avg_goals["FTAG"].round(2),
                                         name="Away avg", marker_color="#4d9fff"))
                fig_avg.update_layout(**PLOTLY_THEME, barmode="group",
                                      title="Average goals per match per season",
                                      legend=dict(bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_avg, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 2 – FEATURE EXPLORER
    # ══════════════════════════════════════════════════════════════
    with eda_tab2:
        if train is None:
            st.info("Run pipeline to see engineered features.")
        else:
            numeric_cols = train.select_dtypes(include="number").columns.tolist()
            feature_cols = [c for c in numeric_cols if c != "FTR"]

            col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
            with col_s1:
                sel_feat = st.selectbox("Select feature", sorted(feature_cols), key="feat_sel")
            with col_s2:
                chart_type = st.selectbox("Chart type", ["Histogram", "Box plot", "Violin"], key="chart_type")
            with col_s3:
                split_outcome = st.toggle("Split by outcome", value=True, key="split_tog")

            lmap = {0: "Away Win", 1: "Draw", 2: "Home Win"}
            cmap = {"Away Win": "#4d9fff", "Draw": "#f5a623", "Home Win": "#22c97e"}

            if sel_feat:
                fig = go.Figure()
                classes = sorted(train["FTR"].dropna().unique()) if split_outcome and "FTR" in train.columns else [None]

                for cls in classes:
                    subset = (train[train["FTR"] == cls][sel_feat].dropna()
                              if cls is not None else train[sel_feat].dropna())
                    label = lmap.get(int(cls), str(cls)) if cls is not None else sel_feat
                    color = cmap.get(label, "#4d9fff")

                    if chart_type == "Histogram":
                        fig.add_trace(go.Histogram(x=subset, name=label, opacity=0.72,
                                                   nbinsx=40, marker_color=color))
                        fig.update_layout(barmode="overlay")
                    elif chart_type == "Box plot":
                        _h = color.lstrip("#")
                        _fc = f"rgba({int(_h[0:2],16)},{int(_h[2:4],16)},{int(_h[4:6],16)},0.25)"
                        fig.add_trace(go.Box(y=subset, name=label,
                                             marker_color=color, line_color=color,
                                             fillcolor=_fc))
                    else:
                        fig.add_trace(go.Violin(y=subset, name=label, fillcolor=color,
                                                line_color=color, opacity=0.7,
                                                box_visible=True, meanline_visible=True))

                fig.update_layout(**PLOTLY_THEME,
                                  title=f"{sel_feat} — {chart_type.lower()}",
                                  legend=dict(bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig, use_container_width=True)

            # Feature stats table
            st.subheader("Feature summary statistics")
            stats = train[feature_cols].describe().T.round(4)
            stats.index.name = "Feature"
            st.dataframe(stats, use_container_width=True, height=400)

    # ══════════════════════════════════════════════════════════════
    # TAB 3 – CORRELATIONS
    # ══════════════════════════════════════════════════════════════
    with eda_tab3:
        if train is None:
            st.info("Run pipeline to see correlations.")
        else:
            numeric_cols = train.select_dtypes(include="number").columns.tolist()
            feature_cols = [c for c in numeric_cols if c != "FTR"]

            if "FTR" in train.columns:
                corr_with_target = (
                    train[feature_cols + ["FTR"]].corr()["FTR"].drop("FTR").sort_values()
                )

                _n_feats = len(corr_with_target)
                if _n_feats < 2:
                    top_n = _n_feats
                else:
                    _slider_max = max(_n_feats, 11)
                    _slider_min = min(10, _n_feats - 1)
                    top_n = st.slider("Number of features to show", _slider_min, _slider_max,
                                      min(30, _n_feats), key="corr_slider")
                half  = top_n // 2
                corr_show = pd.concat([corr_with_target.head(half),
                                       corr_with_target.tail(top_n - half)])

                fig_c = go.Figure(go.Bar(
                    y=corr_show.index, x=corr_show.values, orientation="h",
                    marker_color=["#22c97e" if v >= 0 else "#e85a5a" for v in corr_show.values],
                    text=[f"{v:+.3f}" for v in corr_show.values], textposition="outside",
                ))
                fig_c.update_layout(**PLOTLY_THEME,
                    title=f"Pearson correlation with FTR — top {top_n} features",
                    height=max(500, top_n * 24))
                fig_c.update_xaxes(range=[-0.5, 0.5], gridcolor="rgba(255,255,255,0.07)")
                fig_c.update_yaxes(gridcolor="rgba(255,255,255,0.07)")
                st.plotly_chart(fig_c, use_container_width=True)

                # Heatmap
                st.subheader("Pairwise correlation heatmap")
                top15 = corr_with_target.abs().sort_values(ascending=False).head(15).index.tolist()
                hmap_feats = st.multiselect("Features for heatmap (default: top 15 by |corr|)",
                                            feature_cols, default=top15, key="hmap_sel")
                if len(hmap_feats) >= 2:
                    cm = train[hmap_feats].corr()
                    fig_h = px.imshow(
                        cm, text_auto=".2f", aspect="auto",
                        color_continuous_scale=[[0,"#e85a5a"],[0.5,"#1a2030"],[1,"#22c97e"]],
                        zmin=-1, zmax=1,
                    )
                    fig_h.update_layout(
                        **PLOTLY_THEME,
                        height=max(420, len(hmap_feats) * 32),
                        title="Feature correlation matrix",
                    )
                    st.plotly_chart(fig_h, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 4 – DOMAIN SIGNALS
    # ══════════════════════════════════════════════════════════════
    with eda_tab4:
        if raw is None:
            st.info("`collected.csv` not found. Run pipeline to enable domain signal charts.")
        else:
            # ELO differential by outcome
            if "H_ELO" in raw.columns and "A_ELO" in raw.columns:
                st.subheader("ELO differential by outcome")
                raw["ELO_DIFF_RAW"] = raw["H_ELO"].astype(float) - raw["A_ELO"].astype(float)
                fig_elo = go.Figure()
                for outcome, color in [("H","#22c97e"),("D","#f5a623"),("A","#4d9fff")]:
                    subset = raw[raw["FTR"] == outcome]["ELO_DIFF_RAW"].dropna()
                    fig_elo.add_trace(go.Violin(
                        y=subset, name={"H":"Home Win","D":"Draw","A":"Away Win"}[outcome],
                        fillcolor=color, line_color=color, opacity=0.65,
                        box_visible=True, meanline_visible=True,
                    ))
                fig_elo.update_layout(**PLOTLY_THEME,
                                      title="ELO differential (Home − Away) distribution by outcome",
                                      legend=dict(bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_elo, use_container_width=True)

                # ELO diff bins → outcome rate
                st.subheader("Home win rate by ELO differential bracket")
                raw["ELO_BIN"] = pd.cut(raw["ELO_DIFF_RAW"],
                                         bins=[-800,-200,-100,-50,0,50,100,200,800],
                                         labels=["<-200","-200:-100","-100:-50","-50:0",
                                                 "0:50","50:100","100:200",">200"])
                bin_rates = (raw.groupby("ELO_BIN", observed=True)["FTR"]
                               .value_counts(normalize=True).unstack(fill_value=0) * 100)
                fig_bin = go.Figure()
                for col_k, color in [("H","#22c97e"),("D","#f5a623"),("A","#4d9fff")]:
                    if col_k in bin_rates.columns:
                        fig_bin.add_trace(go.Bar(
                            x=bin_rates.index.astype(str), y=bin_rates[col_k].round(1),
                            name={"H":"Home Win","D":"Draw","A":"Away Win"}[col_k],
                            marker_color=color,
                        ))
                fig_bin.update_layout(**PLOTLY_THEME, barmode="stack",
                                       title="Outcome rate (%) by ELO differential bracket",
                                       yaxis_title="%",
                                       legend=dict(bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_bin, use_container_width=True)

            # Betting odds signals
            if "B365H" in raw.columns:
                st.subheader("Implied win probability from betting odds")
                raw_odds = raw.copy()
                raw_odds["IMP_H"] = (1 / raw_odds["B365H"])
                raw_odds["IMP_D"] = (1 / raw_odds["B365D"])
                raw_odds["IMP_A"] = (1 / raw_odds["B365A"])
                overround = raw_odds["IMP_H"] + raw_odds["IMP_D"] + raw_odds["IMP_A"]
                raw_odds["IMP_H_NORM"] = raw_odds["IMP_H"] / overround
                raw_odds["IMP_A_NORM"] = raw_odds["IMP_A"] / overround

                col_o1, col_o2 = st.columns(2)
                with col_o1:
                    fig_imp = go.Figure()
                    for outcome, col_k, color in [
                        ("Home Win","IMP_H_NORM","#22c97e"),
                        ("Draw","IMP_D","#f5a623"),
                        ("Away Win","IMP_A_NORM","#4d9fff"),
                    ]:
                        if col_k in raw_odds.columns:
                            fig_imp.add_trace(go.Histogram(
                                x=raw_odds[col_k], name=outcome, opacity=0.7,
                                nbinsx=30, marker_color=color,
                            ))
                    fig_imp.update_layout(**PLOTLY_THEME, barmode="overlay",
                                          title="Implied probability distribution (odds-normalised)",
                                          legend=dict(bgcolor="rgba(0,0,0,0)"))
                    st.plotly_chart(fig_imp, use_container_width=True)
                with col_o2:
                    avg_imp = (raw_odds.groupby("FTR")[["IMP_H_NORM","IMP_A_NORM"]]
                                       .mean().reset_index())
                    avg_imp["label"] = avg_imp["FTR"].map({"H":"Home Win","D":"Draw","A":"Away Win"})
                    fig_avg_imp = go.Figure()
                    fig_avg_imp.add_trace(go.Bar(
                        x=avg_imp["label"], y=avg_imp["IMP_H_NORM"].round(3),
                        name="Avg implied home prob", marker_color="#22c97e",
                    ))
                    fig_avg_imp.update_layout(**PLOTLY_THEME,
                                              title="Avg implied home win prob by actual outcome",
                                              yaxis_range=[0,1])
                    st.plotly_chart(fig_avg_imp, use_container_width=True)

            # xG signals
            if "HxG" in raw.columns:
                st.subheader("xG distribution by outcome")
                fig_xg = go.Figure()
                _xg_labels = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
                for outcome, color in [("H","#22c97e"),("D","#f5a623"),("A","#4d9fff")]:
                    sub = raw[raw["FTR"] == outcome]
                    lbl = _xg_labels[outcome]
                    fig_xg.add_trace(go.Box(
                        y=sub["HxG"], name=f"Home xG — {lbl}",
                        marker_color=color, line_color=color,
                    ))
                    fig_xg.add_trace(go.Box(
                        y=sub["AxG"], name=f"Away xG — {lbl}",
                        marker_color=color, line_color=color, visible="legendonly",
                    ))
                fig_xg.update_layout(**PLOTLY_THEME,
                                     title="Home xG distribution by match outcome",
                                     legend=dict(bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_xg, use_container_width=True)

            # Shots on target
            if "HST" in raw.columns:
                st.subheader("Shots on target vs outcome")
                raw["SOT_DIFF"] = raw["HST"].astype(float) - raw["AST"].astype(float)
                sot_avg = raw.groupby("FTR")["SOT_DIFF"].mean().reset_index()
                sot_avg["label"] = sot_avg["FTR"].map({"H":"Home Win","D":"Draw","A":"Away Win"})
                fig_sot = go.Figure(go.Bar(
                    x=sot_avg["label"], y=sot_avg["SOT_DIFF"].round(2),
                    marker_color=["#22c97e","#f5a623","#4d9fff"],
                    text=sot_avg["SOT_DIFF"].round(2), textposition="outside",
                ))
                fig_sot.update_layout(**PLOTLY_THEME,
                                       title="Average shots-on-target differential (Home − Away) by outcome")
                st.plotly_chart(fig_sot, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 5 – DATA QUALITY
    # ══════════════════════════════════════════════════════════════
    with eda_tab5:
        for label, df_q in [("Raw collected data", raw), ("Preprocessed train", train)]:
            if df_q is None:
                continue
            st.subheader(f"{label} — {df_q.shape[0]:,} rows × {df_q.shape[1]} columns")
            null_counts = df_q.isna().sum()
            null_counts = null_counts[null_counts > 0]
            if null_counts.empty:
                st.success(f"No missing values in {label}.")
            else:
                fig_null = go.Figure(go.Bar(
                    x=null_counts.index, y=null_counts.values, marker_color="#e85a5a",
                    text=[f"{v} ({v/len(df_q)*100:.1f}%)" for v in null_counts.values],
                    textposition="outside",
                ))
                fig_null.update_layout(**PLOTLY_THEME,
                                        title=f"Missing values — {label}")
                st.plotly_chart(fig_null, use_container_width=True)

            dup_count = df_q.duplicated().sum()
            if dup_count:
                st.warning(f"{dup_count} duplicate rows found.")
            else:
                st.success(f"No duplicate rows in {label}.")

            with st.expander(f"Data types — {label}"):
                dtype_df = pd.DataFrame({
                    "Column": df_q.dtypes.index,
                    "Type":   df_q.dtypes.astype(str).values,
                    "Non-null": df_q.notna().sum().values,
                    "Unique":   df_q.nunique().values,
                })
                st.dataframe(dtype_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.markdown(
        "<div style='color:#8a94b0;font-size:14px;margin-bottom:24px;'>"
        "Results read from <code>results/model_results.csv</code> and per-model "
        "<code>metrics.json</code> / <code>confusion.csv</code>.</div>",
        unsafe_allow_html=True,
    )

    results = load_model_results()
    if results is None:
        st.error("No results found. Run `python pipeline.py` to generate model outputs.")
        st.stop()

    # ── Metric selector ────────────────────────────────────────────────────────
    metric_options = {
        "Test Accuracy": "accuracy",
        "Macro F1": "macro_f1",
        "Macro Precision": "macro_precision",
        "Macro Recall": "macro_recall",
    }
    if "val_macro_f1" in results.columns:
        metric_options["Validation Macro F1"] = "val_macro_f1"
    if "val_accuracy" in results.columns:
        metric_options["Validation Accuracy"] = "val_accuracy"

    selected_metric_label = st.selectbox("Primary metric", list(metric_options.keys()), key="metric_sel")
    selected_metric = metric_options[selected_metric_label]

    # ── Bar chart ──────────────────────────────────────────────────────────────
    results_sorted = results.sort_values(selected_metric, ascending=True)
    best_val = results_sorted[selected_metric].max()
    colors = ["#22c97e" if v == best_val else "#4d9fff" for v in results_sorted[selected_metric]]

    fmt = ".1%" if "accuracy" in selected_metric else ".3f"
    fig = go.Figure(go.Bar(
        y=results_sorted["model"].str.replace("_", " "),
        x=results_sorted[selected_metric],
        orientation="h",
        marker_color=colors,
        text=[f"{v:{fmt}}" for v in results_sorted[selected_metric]],
        textposition="outside",
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        title=f"{selected_metric_label} — all models",
        height=350,
        xaxis_range=[0, best_val * 1.18],
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Radar chart (multi-metric) ─────────────────────────────────────────────
    st.subheader("Multi-metric radar comparison")
    radar_metrics = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    radar_labels = ["Accuracy", "Macro F1", "Precision", "Recall"]
    available_radar = [m for m in radar_metrics if m in results.columns]
    available_labels = [radar_labels[radar_metrics.index(m)] for m in available_radar]

    fig_radar = go.Figure()
    palette = ["#22c97e", "#4d9fff", "#9b7fff", "#f5a623", "#e85a5a", "#2dd4bf", "#ff7eb3"]

    def hex_to_rgba(hex_color: str, alpha: float = 0.05) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    for i, (_, row) in enumerate(results.iterrows()):
        vals = [row[m] for m in available_radar]
        vals_closed = vals + [vals[0]]
        labels_closed = available_labels + [available_labels[0]]
        color = palette[i % len(palette)]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_closed, theta=labels_closed,
            name=row["model"].replace("_", " "),
            line=dict(color=color, width=2),
            fill="toself", fillcolor=hex_to_rgba(color, 0.05),
            opacity=0.85,
        ))
    fig_radar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8d0e8", size=11),
        polar=dict(
            bgcolor="rgba(26,32,48,0.8)",
            angularaxis=dict(color="#5a6480"),
            radialaxis=dict(color="#5a6480", range=[0, 1], showticklabels=True,
                            gridcolor="rgba(255,255,255,0.1)"),
        ),
        legend=dict(bgcolor="rgba(26,32,48,0.7)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        title="All models — multi-metric overview",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # ── Full table ─────────────────────────────────────────────────────────────
    st.subheader("Full results table")
    display_cols = ["model", "accuracy", "macro_f1", "macro_precision", "macro_recall"]
    if "val_macro_f1" in results.columns:
        display_cols.append("val_macro_f1")

    display_df = results[display_cols].copy()
    display_df["model"] = display_df["model"].str.replace("_", " ")

    format_dict = {
        "accuracy": "{:.4f}",
        "macro_f1": "{:.4f}",
        "macro_precision": "{:.4f}",
        "macro_recall": "{:.4f}",
    }
    if "val_macro_f1" in display_df.columns:
        format_dict["val_macro_f1"] = "{:.4f}"

    best_idx = display_df["macro_f1"].idxmax()

    def highlight_best(row):
        if row.name == best_idx:
            return ["background-color: rgba(34,201,126,0.08); color: #22c97e"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_df.style.apply(highlight_best, axis=1).format(format_dict),
        use_container_width=True,
        height=300,
    )

    st.divider()

    # ── Per-model deep-dive ─────────────────────────────────────────────────────
    st.subheader("Per-model deep dive")
    selected_model = st.selectbox(
        "Select model",
        [m for m in MODEL_NAMES if (RESULTS_DIR / m / "metrics.json").exists()],
        format_func=lambda x: x.replace("_", " ").title(),
        key="model_detail_sel",
    )

    detail = load_model_detail(selected_model)
    cm_df  = load_confusion(selected_model)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Hyperparameters & metrics**")
        if detail:
            metrics = detail.get("metrics", {})
            best_params = detail.get("best_params", {})

            m_cols = st.columns(2)
            with m_cols[0]:
                metric_card("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%", "test set", "#22c97e")
            with m_cols[1]:
                metric_card("Macro F1", f"{metrics.get('macro_f1', 0):.4f}", "test set", "#4d9fff")

            st.markdown("**Best hyperparameters:**")
            if best_params:
                param_clean = {k.replace("model__", ""): v for k, v in best_params.items()}
                param_df = pd.DataFrame(list(param_clean.items()), columns=["Parameter", "Value"])
                st.dataframe(param_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No hyperparameters (baseline / fixed model).")
        else:
            st.info(f"No metrics.json found for `{selected_model}`.")

    with col2:
        st.markdown("**Confusion matrix (test set)**")
        if cm_df is not None:
            label_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
            n = len(cm_df)
            labels = [label_map.get(i, str(i)) for i in range(n)]
            z = cm_df.values

            total = z.sum()
            z_pct = z / total * 100

            # Normalize 0-1 for colour intensity, then pick text colour per cell
            z_norm = z / z.max() if z.max() > 0 else z
            # bright cells (norm > 0.45) get dark text; dark cells get light text
            font_colors = [
                ["#0a0d12" if z_norm[i][j] > 0.45 else "#e8edf8" for j in range(n)]
                for i in range(n)
            ]
            text_vals = [[f"{z[i][j]}<br>({z_pct[i][j]:.1f}%)" for j in range(n)] for i in range(n)]

            fig_cm = go.Figure(go.Heatmap(
                z=z,
                x=[f"Pred: {l}" for l in labels],
                y=[f"Act: {l}" for l in labels],
                text=text_vals,
                texttemplate="%{text}",
                colorscale=[
                    [0.0, "#111827"],
                    [0.4, "#1a4a88"],
                    [1.0, "#22c97e"],
                ],
                showscale=False,
            ))
            # Overlay invisible scatter for per-cell font colour control
            for i in range(n):
                for j in range(n):
                    fig_cm.add_annotation(
                        x=f"Pred: {labels[j]}",
                        y=f"Act: {labels[i]}",
                        text=text_vals[i][j].replace("<br>", "<br>"),
                        showarrow=False,
                        font=dict(color=font_colors[i][j], size=13, family="IBM Plex Mono"),
                        align="center",
                    )
            # Hide default heatmap text (we use annotations instead)
            fig_cm.update_traces(texttemplate="")
            fig_cm.update_layout(
                **PLOTLY_THEME,
                title=f"Confusion matrix — {selected_model.replace('_', ' ')}",
                height=380,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            # Per-class recall / precision
            class_prec, class_rec, class_f1 = [], [], []
            for i in range(n):
                tp = z[i, i]
                prec = tp / z[:, i].sum() if z[:, i].sum() > 0 else 0
                rec  = tp / z[i, :].sum() if z[i, :].sum() > 0 else 0
                f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                class_prec.append(prec); class_rec.append(rec); class_f1.append(f1)

            per_class_df = pd.DataFrame({
                "Class": labels,
                "Precision": [f"{v:.3f}" for v in class_prec],
                "Recall":    [f"{v:.3f}" for v in class_rec],
                "F1":        [f"{v:.3f}" for v in class_f1],
            })
            st.dataframe(per_class_df, use_container_width=True, hide_index=True)
        else:
            st.info(f"No confusion.csv found for `{selected_model}`.")

    # ── Validation vs Test ─────────────────────────────────────────────────────
    if "val_macro_f1" in results.columns:
        st.divider()
        st.subheader("Validation vs test macro F1 — overfitting check")
        fig_vt = go.Figure()
        fig_vt.add_trace(go.Scatter(
            x=results["model"].str.replace("_", " "),
            y=results["val_macro_f1"],
            mode="lines+markers", name="Validation F1",
            marker=dict(color="#4d9fff", size=8),
            line=dict(color="#4d9fff", width=2),
        ))
        fig_vt.add_trace(go.Scatter(
            x=results["model"].str.replace("_", " "),
            y=results["macro_f1"],
            mode="lines+markers", name="Test F1",
            marker=dict(color="#22c97e", size=8),
            line=dict(color="#22c97e", width=2, dash="dot"),
        ))
        fig_vt.update_layout(
            **PLOTLY_THEME, title="Val vs test macro F1 (gap = overfitting signal)",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_vt, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FEATURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Feature Analysis":
    st.title("Feature Analysis")
    st.markdown(
        "<div style='color:#8a94b0;font-size:14px;margin-bottom:24px;'>"
        "Feature selection outputs from <code>results/feature_selection/</code> — "
        "permutation importance, SHAP values, and RFECV curve.</div>",
        unsafe_allow_html=True,
    )

    linear_feats = load_features_linear()
    tree_feats   = load_features_trees()

    # ── Feature set overview ───────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        metric_card("Linear model features", str(len(linear_feats)), "selected by RFECV", "#4d9fff")
    with col2:
        metric_card("Tree model features", str(len(tree_feats)), "selected by perm. importance", "#22c97e")

    if linear_feats or tree_feats:
        tab1, tab2 = st.tabs(["Linear features (RFECV)", "Tree features (permutation importance)"])
        with tab1:
            if linear_feats:
                cols = st.columns(3)
                for i, f in enumerate(linear_feats):
                    with cols[i % 3]:
                        st.markdown(
                            f"<div style='background:#1a2030;border:1px solid rgba(255,255,255,0.08);"
                            f"border-radius:8px;padding:8px 12px;font-family:IBM Plex Mono,monospace;"
                            f"font-size:12px;color:#4d9fff;margin-bottom:6px;'>{f}</div>",
                            unsafe_allow_html=True,
                        )
            else:
                st.info("No features_linear.csv found.")
        with tab2:
            if tree_feats:
                cols = st.columns(3)
                for i, f in enumerate(tree_feats):
                    with cols[i % 3]:
                        st.markdown(
                            f"<div style='background:#1a2030;border:1px solid rgba(255,255,255,0.08);"
                            f"border-radius:8px;padding:8px 12px;font-family:IBM Plex Mono,monospace;"
                            f"font-size:12px;color:#22c97e;margin-bottom:6px;'>{f}</div>",
                            unsafe_allow_html=True,
                        )
            else:
                st.info("No features_trees.csv found.")

    st.divider()

    # ── Images from feature selection ─────────────────────────────────────────
    img_col1, img_col2 = st.columns(2)

    with img_col1:
        st.subheader("Permutation importance")
        if PERM_IMP_IMG.exists():
            st.image(str(PERM_IMP_IMG), use_container_width=True,
                     caption="Mean decrease in accuracy — Random Forest (20 repeats)")
        else:
            st.info(f"Image not found: `{PERM_IMP_IMG}`")

    with img_col2:
        st.subheader("RFECV curve")
        if RFECV_IMG.exists():
            st.image(str(RFECV_IMG), use_container_width=True,
                     caption="Cross-validated macro F1 vs number of features")
        else:
            st.info(f"Image not found: `{RFECV_IMG}`")

    st.subheader("SHAP summary (XGBoost)")
    if SHAP_IMG.exists():
        st.image(str(SHAP_IMG), use_container_width=True,
                 caption="SHAP values indicating directional feature impact on predictions")
    else:
        st.info(f"Image not found: `{SHAP_IMG}`")

    # ── Live correlation check from training data ──────────────────────────────
    train = load_train()
    if train is not None and tree_feats:
        st.divider()
        st.subheader("High-correlation pairs (|r| > 0.85)")
        avail = [f for f in tree_feats if f in train.columns]
        if len(avail) >= 2:
            corr_matrix = train[avail].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            pairs = [
                (col, row, upper.loc[row, col])
                for col in upper.columns
                for row in upper.index
                if pd.notna(upper.loc[row, col]) and upper.loc[row, col] > 0.85
            ]
            if pairs:
                pairs_df = pd.DataFrame(pairs, columns=["Feature A", "Feature B", "|r|"])
                pairs_df = pairs_df.sort_values("|r|", ascending=False)
                st.dataframe(
                    pairs_df.style.background_gradient(subset=["|r|"], cmap="RdYlGn_r"),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.success("No remaining high-correlation pairs in the final tree feature set.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BUSINESS INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Business Insights":
    st.title("Business Insights")
    st.markdown(
        "<div style='color:#8a94b0;font-size:14px;margin-bottom:24px;'>"
        "Translating model performance into actionable intelligence for sports analytics stakeholders.</div>",
        unsafe_allow_html=True,
    )

    results = load_model_results()

    # ── Lift over baseline ─────────────────────────────────────────────────────
    if results is not None and "home_win_baseline" in results["model"].values:
        baseline_row = results[results["model"] == "home_win_baseline"].iloc[0]
        best_row = results.sort_values("macro_f1", ascending=False).iloc[0]

        acc_lift = (best_row["accuracy"] - baseline_row["accuracy"]) / baseline_row["accuracy"] * 100
        f1_lift  = best_row["macro_f1"] - baseline_row["macro_f1"]

        st.subheader("Model value over baseline")
        cols = st.columns(4)
        with cols[0]:
            metric_card("Best model", best_row["model"].replace("_", " ").title(), "", "#22c97e")
        with cols[1]:
            metric_card("Accuracy lift", f"+{acc_lift:.1f}%", "vs. always-home-win baseline", "#22c97e")
        with cols[2]:
            metric_card("F1 lift", f"+{f1_lift:.3f}", "macro F1 improvement", "#4d9fff")
        with cols[3]:
            metric_card("Baseline accuracy", f"{baseline_row['accuracy']*100:.1f}%", "home win rate", "#5a6480")

        st.divider()

    # ── Business metric: draw prediction ──────────────────────────────────────
    st.subheader("Per-class recall analysis (best model)")
    best_model_name = (
        results.sort_values("macro_f1", ascending=False).iloc[0]["model"]
        if results is not None else None
    )
    if best_model_name:
        cm_df = load_confusion(best_model_name)
        if cm_df is not None:
            label_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
            z = cm_df.values
            n = len(z)
            recalls = {}
            for i in range(n):
                denom = z[i, :].sum()
                recalls[label_map.get(i, str(i))] = z[i, i] / denom if denom > 0 else 0

            fig_recall = go.Figure(go.Bar(
                x=list(recalls.keys()), y=list(recalls.values()),
                marker_color=["#4d9fff", "#f5a623", "#22c97e"],
                text=[f"{v:.1%}" for v in recalls.values()],
                textposition="outside",
            ))
            fig_recall.update_layout(
                **PLOTLY_THEME,
                title=f"Per-class recall — {best_model_name.replace('_', ' ')} (test set)",
                yaxis_tickformat=".0%", yaxis_range=[0, 1],
            )
            st.plotly_chart(fig_recall, use_container_width=True)

    st.divider()

    # ── Stakeholder cards ──────────────────────────────────────────────────────
    st.subheader("Stakeholder applications")

    insights = [
        {
            "title": "Football Clubs — Tactical Preparation",
            "color": "#22c97e",
            "body": (
                "Use outcome probabilities as a pre-match intelligence layer. "
                "When the model assigns >65% win probability to the opponent, "
                "coaching staff can flag the fixture for additional defensive analysis. "
                "ELO differential and league position gap are directly interpretable metrics "
                "for quantifying relative competitive standing ahead of a match."
            ),
            "tags": [("Actionable", "#22c97e"), ("Scouting", "#4d9fff")],
        },
        {
            "title": "Sports Broadcasters — Narrative Generation",
            "color": "#4d9fff",
            "body": (
                "The ELO_MARKET_DIFF feature — the gap between ELO-derived and "
                "bookmaker-implied win probabilities — surfaces fixtures where the model "
                "disagrees with market consensus. These matches have historically shown "
                "higher upset rates and provide compelling pre-match narrative angles "
                "for data-driven coverage."
            ),
            "tags": [("Pre-match content", "#4d9fff"), ("Upset detection", "#9b7fff")],
        },
        {
            "title": "Analytics Platforms — Confidence Scoring",
            "color": "#9b7fff",
            "body": (
                "Gradient Boosting probabilities can be Platt-calibrated and served as "
                "three-class confidence scores via API. The model performs most reliably "
                "on decisive fixtures (large ELO gaps) — confidence intervals should be "
                "widened for closely matched contests. Draw predictions carry the lowest "
                "confidence across all models (recall ≈ 20–25%)."
            ),
            "tags": [("API-ready", "#9b7fff"), ("Calibration needed", "#f5a623")],
        },
    ]

    for ins in insights:
        def _tag_span(tag: str, color: str) -> str:
            h = color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return (
                f"<span style='background:rgba({r},{g},{b},0.12);color:{color};"
                f"font-size:11px;padding:3px 10px;border-radius:10px;"
                f"font-family:IBM Plex Mono,monospace;margin-right:6px;'>{tag}</span>"
            )

        tag_html = "".join(_tag_span(t, c) for t, c in ins["tags"])
        color = ins["color"]
        title = ins["title"]
        body  = ins["body"]
        st.markdown(
            f"<div style='background:#1a2030;border:1px solid rgba(255,255,255,0.08);"
            f"border-left:3px solid {color};border-radius:14px;padding:20px 22px;margin-bottom:16px;'>"
            f"<div style='font-family:Syne,sans-serif;font-weight:700;font-size:16px;"
            f"color:{color};margin-bottom:10px;'>{title}</div>"
            f"<div style='font-size:13px;color:#8a94b0;line-height:1.7;margin-bottom:12px;'>{body}</div>"
            f"{tag_html}</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Limitations ────────────────────────────────────────────────────────────
    st.subheader("Limitations & future work")
    col_lim, col_fut = st.columns(2)
    with col_lim:
        st.markdown("**Current limitations**")
        for item in [
            "Draw class consistently underperforms across all models",
            "No injury, suspension, or confirmed lineup data used",
            "Newly promoted teams have sparse early-season rolling features",
            "Model does not account for mid-season manager changes",
            "Football's inherent randomness caps achievable accuracy",
        ]:
            st.markdown(f"<div style='color:#e85a5a;font-size:13px;padding:4px 0;'>– {item}</div>",
                        unsafe_allow_html=True)
    with col_fut:
        st.markdown("**Future improvements**")
        for item in [
            "Integrate Transfermarkt squad value data (already validated in Phase 2)",
            "Add expected lineup features from pre-match data",
            "Experiment with XGBoost / LightGBM ensembles",
            "Calibrate output probabilities (Platt scaling)",
            "Deploy as live API with automated per-matchday refresh",
        ]:
            st.markdown(f"<div style='color:#22c97e;font-size:13px;padding:4px 0;'>+ {item}</div>",
                        unsafe_allow_html=True)