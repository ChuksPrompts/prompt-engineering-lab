"""
dashboard.py
============
LLM Prompt Benchmark System — Streamlit Dashboard
Project: P7 · prompt-engineering-lab by ChuksForge

Usage:
    pip install streamlit
    streamlit run dashboard.py
    # Opens at http://localhost:8501
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import streamlit as st
except ImportError:
    print("Install Streamlit: pip install streamlit")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

RESULTS_DIR = Path("results")

PALETTE = {
    "bg":      "#0f1117", "surface": "#161820", "border": "#1e2130",
    "text":    "#f0f2f8", "muted":   "#5a6080", "accent": "#e8ff47",
    "blue":    "#47c8ff", "purple":  "#b847ff", "orange": "#ff8c47",
    "green":   "#47ffb2", "red":     "#ff4776",
}

MODEL_COLORS = {
    "GPT-4o-mini":       "#47c8ff",
    "GPT-4o":            "#e8ff47",
    "Claude Haiku":      "#ff8c47",
    "Claude Sonnet 4.6": "#b847ff",
    "Mistral 7B":        "#47ffb2",
    "Llama 3 8B":        "#ff4776",
}

TASK_COLORS = {
    "summarization": "#47c8ff",
    "qa":            "#e8ff47",
    "reasoning":     "#b847ff",
    "coding":        "#47ffb2",
}


# ── Data loading ─────────────────────────────────────────────

@st.cache_data
def load_data():
    results_path = RESULTS_DIR / "results.csv"
    if not results_path.exists():
        return None, None, None, None

    df  = pd.read_csv(results_path)
    df  = df[df["error"].isna() | (df["error"] == "")].copy()
    lb  = pd.read_csv(RESULTS_DIR / "leaderboard.csv") if (RESULTS_DIR / "leaderboard.csv").exists() else pd.DataFrame()
    clb = pd.read_csv(RESULTS_DIR / "cost_leaderboard.csv") if (RESULTS_DIR / "cost_leaderboard.csv").exists() else pd.DataFrame()
    ptl = pd.read_csv(RESULTS_DIR / "leaderboard_per_task.csv") if (RESULTS_DIR / "leaderboard_per_task.csv").exists() else pd.DataFrame()
    return df, lb, clb, ptl


def setup_mpl():
    if not HAS_MPL:
        return
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],   "axes.facecolor":  PALETTE["surface"],
        "axes.edgecolor":   PALETTE["border"],"axes.labelcolor": PALETTE["text"],
        "axes.titlecolor":  PALETTE["text"],  "xtick.color":     PALETTE["muted"],
        "ytick.color":      PALETTE["muted"], "text.color":      PALETTE["text"],
        "grid.color":       PALETTE["border"],"grid.linewidth":  0.5,
        "font.family":      "monospace",      "font.size":       9,
        "figure.dpi":       120,
    })


# ── Chart helpers ────────────────────────────────────────────

def leaderboard_chart(lb):
    if not HAS_MPL or lb.empty:
        return None
    lb_sorted = lb.sort_values("task_score", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(lb_sorted)*0.6)))
    colors = [MODEL_COLORS.get(m, PALETTE["blue"]) for m in lb_sorted["model"]]
    bars = ax.barh(lb_sorted["model"], lb_sorted["task_score"],
                   color=colors, height=0.55, alpha=0.9)
    for bar, val in zip(bars, lb_sorted["task_score"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8, color=PALETTE["text"])
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Avg Task Score (0–1)")
    ax.set_title("MODEL LEADERBOARD", loc="left", fontsize=10, fontweight="bold")
    ax.axvline(lb_sorted["task_score"].mean(), color=PALETTE["muted"],
               linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def cost_quality_scatter(df):
    if not HAS_MPL or df.empty:
        return None
    model_avg = df.groupby("model")[["cost_usd","task_score"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in model_avg.iterrows():
        color = MODEL_COLORS.get(row["model"], PALETTE["blue"])
        ax.scatter(row["cost_usd"] * 1000, row["task_score"],
                   color=color, s=200, zorder=4, edgecolors="white", linewidths=0.5)
        ax.annotate(row["model"], (row["cost_usd"]*1000, row["task_score"]),
                    textcoords="offset points", xytext=(8, 3),
                    color=color, fontsize=8, fontweight="bold")
    ax.set_xlabel("Avg Cost per Run ($ × 10⁻³)")
    ax.set_ylabel("Avg Task Score")
    ax.set_title("COST vs QUALITY — Per Model (avg across all tasks & prompts)", loc="left", fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def task_heatmap(df):
    if not HAS_MPL or df.empty:
        return None
    pivot = df.groupby(["model","task"])["task_score"].mean().unstack(fill_value=0).round(3)
    fig, ax = plt.subplots(figsize=(8, max(3, len(pivot)*0.7)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=15, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i,j]
            color = "black" if val > 0.6 else PALETTE["text"]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("SCORE HEATMAP — Model × Task", loc="left", fontsize=10)
    plt.tight_layout()
    return fig


def strategy_chart(ptl):
    if not HAS_MPL or ptl.empty:
        return None
    strat_avg = ptl.groupby(["prompt_strategy","task"])["task_score"].mean().unstack(fill_value=0).round(3)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(strat_avg.index))
    n = len(strat_avg.columns)
    w = 0.7 / max(n, 1)
    task_color_list = [TASK_COLORS.get(c, PALETTE["blue"]) for c in strat_avg.columns]
    for i, task in enumerate(strat_avg.columns):
        offset = (i - n/2) * w + w/2
        ax.bar(x + offset, strat_avg[task], width=w*0.9,
               color=task_color_list[i], alpha=0.85, label=task)
    ax.set_xticks(x)
    ax.set_xticklabels(strat_avg.index, rotation=10, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Avg Task Score")
    ax.set_title("PROMPT STRATEGY COMPARISON — By Task", loc="left", fontsize=10)
    ax.legend(framealpha=0, fontsize=8, ncol=4)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def quality_per_dollar_chart(clb):
    if not HAS_MPL or clb.empty:
        return None
    clb_sorted = clb.sort_values("quality_per_dollar", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(clb_sorted)*0.6)))
    colors = [MODEL_COLORS.get(m, PALETTE["blue"]) for m in clb_sorted["model"]]
    bars = ax.barh(clb_sorted["model"], clb_sorted["quality_per_dollar"],
                   color=colors, height=0.55, alpha=0.9)
    for bar, val in zip(bars, clb_sorted["quality_per_dollar"]):
        ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va="center", fontsize=8, color=PALETTE["text"])
    ax.set_xlabel("Quality per Dollar (higher = better value)")
    ax.set_title("COST EFFICIENCY — Quality Score per $1 Spent", loc="left", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ── Main dashboard ───────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="LLM Prompt Benchmark System",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    .main { background-color: #0f1117; }
    .stMetric { background: #161820; border: 1px solid #1e2130; padding: 12px; border-radius: 4px; }
    h1, h2, h3 { font-family: monospace; }
    </style>
    """, unsafe_allow_html=True)

    st.title("📊 LLM Prompt Benchmark System")
    st.caption("P7 · prompt-engineering-lab — Multi-task, multi-model evaluation with cost analysis")

    setup_mpl()
    df, lb, clb, ptl = load_data()

    if df is None:
        st.error("No results found. Run `python run_benchmark.py` first.")
        st.code("python run_benchmark.py --quick --models openai")
        st.stop()

    # ── Sidebar filters ──────────────────────────────────────
    with st.sidebar:
        st.header("Filters")

        all_models = sorted(df["model"].unique().tolist())
        selected_models = st.multiselect("Models", all_models, default=all_models)

        all_tasks = sorted(df["task"].unique().tolist())
        selected_tasks = st.multiselect("Tasks", all_tasks, default=all_tasks)

        all_strats = sorted(df["prompt_strategy"].unique().tolist())
        selected_strats = st.multiselect("Strategies", all_strats, default=all_strats)

        st.divider()
        st.caption(f"Total runs: {len(df)}")
        st.caption(f"Models: {df['model'].nunique()}")
        st.caption(f"Tasks: {df['task'].nunique()}")

    # Apply filters
    mask = (
        df["model"].isin(selected_models) &
        df["task"].isin(selected_tasks) &
        df["prompt_strategy"].isin(selected_strats)
    )
    df_f = df[mask]

    if df_f.empty:
        st.warning("No data matches current filters.")
        st.stop()

    # ── Top metrics ──────────────────────────────────────────
    st.subheader("Summary")
    col1, col2, col3, col4, col5 = st.columns(5)

    best_model = df_f.groupby("model")["task_score"].mean().idxmax()
    best_score = df_f.groupby("model")["task_score"].mean().max()

    col1.metric("Best Model", best_model, f"{best_score:.3f} avg score")
    col2.metric("Avg Score (all)", f"{df_f['task_score'].mean():.3f}")
    col3.metric("Total Cost", f"${df_f['cost_usd'].sum():.4f}")
    col4.metric("Avg Latency", f"{df_f['latency_s'].mean():.2f}s")

    if not clb.empty and not clb[clb["model"].isin(selected_models)].empty:
        best_value = clb[clb["model"].isin(selected_models)].sort_values("quality_per_dollar", ascending=False).iloc[0]
        col5.metric("Best Value", best_value["model"], f"{best_value['quality_per_dollar']:.1f} q/$")

    st.divider()

    # ── Tab layout ───────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Leaderboard",
        "💰 Cost Analysis",
        "📋 Task Breakdown",
        "🔬 Strategy Comparison",
        "🔍 Raw Results",
    ])

    # ── Tab 1: Leaderboard ───────────────────────────────────
    with tab1:
        col_a, col_b = st.columns([1.2, 1])

        with col_a:
            st.subheader("Overall Rankings")
            if not lb.empty:
                lb_filtered = lb[lb["model"].isin(selected_models)].copy()
                lb_display = lb_filtered[["rank","model","task_score","quality_per_dollar","latency_s"]].copy()
                lb_display.columns = ["Rank","Model","Avg Score","Quality/$","Latency (s)"]
                lb_display = lb_display.round(4)
                st.dataframe(lb_display, use_container_width=True, hide_index=True)

        with col_b:
            st.subheader("Score Chart")
            if not lb.empty:
                lb_f = lb[lb["model"].isin(selected_models)]
                fig = leaderboard_chart(lb_f)
                if fig:
                    st.pyplot(fig)
                    plt.close()

        st.subheader("Score Heatmap (Model × Task)")
        fig = task_heatmap(df_f)
        if fig:
            st.pyplot(fig)
            plt.close()

    # ── Tab 2: Cost Analysis ─────────────────────────────────
    with tab2:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Cost vs Quality")
            fig = cost_quality_scatter(df_f)
            if fig:
                st.pyplot(fig)
                plt.close()

        with col_b:
            st.subheader("Quality per Dollar")
            if not clb.empty:
                clb_f = clb[clb["model"].isin(selected_models)]
                fig = quality_per_dollar_chart(clb_f)
                if fig:
                    st.pyplot(fig)
                    plt.close()

        st.subheader("Cost Summary Table")
        cost_summary = (
            df_f.groupby("model")
            .agg(
                avg_score=("task_score", "mean"),
                total_cost=("cost_usd", "sum"),
                avg_cost_per_run=("cost_usd", "mean"),
                avg_quality_per_dollar=("quality_per_dollar", "mean"),
                total_runs=("task_score", "count"),
            )
            .round(6)
            .reset_index()
            .sort_values("avg_quality_per_dollar", ascending=False)
        )
        cost_summary.columns = ["Model","Avg Score","Total Cost ($)","Avg Cost/Run ($)","Quality/$","Runs"]
        st.dataframe(cost_summary, use_container_width=True, hide_index=True)

    # ── Tab 3: Task Breakdown ────────────────────────────────
    with tab3:
        st.subheader("Performance by Task")

        task_summary = (
            df_f.groupby(["task","model"])["task_score"]
            .mean().round(3).unstack(fill_value=0)
        )
        st.dataframe(task_summary.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1),
                     use_container_width=True)

        st.subheader("Task Score Distribution")
        for task in selected_tasks:
            task_df = df_f[df_f["task"] == task]
            if task_df.empty:
                continue
            model_scores = task_df.groupby("model")["task_score"].mean().sort_values(ascending=False)
            cols = st.columns(len(model_scores))
            for col, (model, score) in zip(cols, model_scores.items()):
                col.metric(model, f"{score:.3f}", label_visibility="visible")

    # ── Tab 4: Strategy Comparison ───────────────────────────
    with tab4:
        st.subheader("Prompt Strategy Comparison")

        if not ptl.empty:
            ptl_f = ptl[ptl["model"].isin(selected_models) & ptl["task"].isin(selected_tasks)]
            fig = strategy_chart(ptl_f)
            if fig:
                st.pyplot(fig)
                plt.close()

        st.subheader("Strategy Rankings per Task")
        if not ptl.empty:
            ptl_f2 = ptl[ptl["model"].isin(selected_models) & ptl["task"].isin(selected_tasks)]
            strat_task = (
                ptl_f2.groupby(["prompt_strategy","task"])["task_score"]
                .mean().round(3).unstack(fill_value=0)
            )
            st.dataframe(
                strat_task.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1),
                use_container_width=True,
            )

    # ── Tab 5: Raw Results ───────────────────────────────────
    with tab5:
        st.subheader("Raw Results Explorer")

        col_f1, col_f2, col_f3 = st.columns(3)
        task_filter_raw   = col_f1.selectbox("Task", ["All"] + sorted(df_f["task"].unique().tolist()))
        model_filter_raw  = col_f2.selectbox("Model", ["All"] + sorted(df_f["model"].unique().tolist()))
        strat_filter_raw  = col_f3.selectbox("Strategy", ["All"] + sorted(df_f["prompt_strategy"].unique().tolist()))

        df_raw = df_f.copy()
        if task_filter_raw  != "All": df_raw = df_raw[df_raw["task"]  == task_filter_raw]
        if model_filter_raw != "All": df_raw = df_raw[df_raw["model"] == model_filter_raw]
        if strat_filter_raw != "All": df_raw = df_raw[df_raw["prompt_strategy"] == strat_filter_raw]

        display_cols = [c for c in ["task","case_id","model","prompt_strategy",
                                     "task_score","cost_usd","quality_per_dollar",
                                     "latency_s","output"] if c in df_raw.columns]
        st.dataframe(df_raw[display_cols].round(4), use_container_width=True, hide_index=True)

        st.caption(f"Showing {len(df_raw)} rows")

        # Expand to see full output
        if not df_raw.empty:
            st.subheader("Inspect Output")
            row_idx = st.number_input("Row index", 0, len(df_raw)-1, 0)
            row = df_raw.iloc[int(row_idx)]
            st.write(f"**Task:** {row.get('task')} | **Case:** {row.get('case_id')} | **Model:** {row.get('model')} | **Score:** {row.get('task_score', 0):.3f}")
            st.text_area("Output", value=str(row.get("output", "")), height=200, disabled=True)


if __name__ == "__main__":
    main()
