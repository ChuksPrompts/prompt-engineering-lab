"""
visualize.py

Summarization Benchmark — Chart Generator
Project: P1 · prompt-engineering-lab

Generates publication-quality charts from results.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

RESULTS_DIR = Path("results")


# ── Style ──────────────────────────────────────

PALETTE = {
    "bg": "#0f1117",
    "surface": "#161820",
    "border": "#1e2130",
    "text": "#f0f2f8",
    "muted": "#5a6080",
    "accent": "#e8ff47",
    "blue": "#47c8ff",
    "purple": "#b847ff",
    "orange": "#ff8c47",
    "green": "#47ffb2",
    "red": "#ff4776",
}

MODEL_COLORS = {
    "GPT-4o-mini": PALETTE["blue"],
    "GPT-4o": PALETTE["accent"],
    "Claude Haiku": PALETTE["orange"],
    "Claude Sonnet 4.6": PALETTE["purple"],
    "Gemini 1.5 Flash": PALETTE["green"],
    "Gemini 1.5 Pro": PALETTE["red"],
}


def setup_style():

    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["surface"],
            "axes.edgecolor": PALETTE["border"],
            "axes.labelcolor": PALETTE["text"],
            "axes.titlecolor": PALETTE["text"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "text.color": PALETTE["text"],
            "grid.color": PALETTE["border"],
            "grid.linewidth": 0.5,
            "font.family": "monospace",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "figure.dpi": 150,
        }
    )


# ── Data Loading ───────────────────────────────


def load_data(results_path: Path):

    df = pd.read_csv(results_path)

    if "error" in df.columns:
        df_clean = df[df["error"].isna() | (df["error"] == "")].copy()
    else:
        df_clean = df.copy()

    metric_cols = [
        "rouge1",
        "rouge2",
        "rougeL",
        "bertscore_f1",
        "compression_ratio",
        "latency_s",
        "word_count",
    ]

    available = [c for c in metric_cols if c in df_clean.columns]

    lb = (
        df_clean.groupby(["model", "prompt_id", "prompt_strategy"])[available]
        .mean()
        .round(4)
        .reset_index()
    )

    if all(c in lb.columns for c in ["rouge1", "rouge2", "rougeL", "bertscore_f1"]):

        lb["composite_score"] = (
            0.2 * lb["rouge1"]
            + 0.2 * lb["rouge2"]
            + 0.2 * lb["rougeL"]
            + 0.4 * lb["bertscore_f1"]
        ).round(4)

    return df_clean, lb


# ── Chart 1: Leaderboard ───────────────────────


def chart_leaderboard(lb, out_dir):

    if "composite_score" not in lb.columns:
        return

    model_scores = (
        lb.groupby("model")["composite_score"].mean().sort_values().reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [MODEL_COLORS.get(m, PALETTE["blue"]) for m in model_scores["model"]]

    bars = ax.barh(
        model_scores["model"], model_scores["composite_score"], color=colors, height=0.55
    )

    for bar, val in zip(bars, model_scores["composite_score"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center")

    ax.set_xlabel("Composite Score")
    ax.set_title("Model Leaderboard")

    plt.tight_layout()

    path = out_dir / "chart_leaderboard.png"
    plt.savefig(path)
    plt.close()

    print("✓", path)


# ── Chart 2: Heatmap ───────────────────────────


def chart_heatmap(lb, out_dir):

    metrics = ["rouge1", "rouge2", "rougeL", "bertscore_f1", "composite_score"]

    available = [m for m in metrics if m in lb.columns]

    if not available:
        return

    model_metric = lb.groupby("model")[available].mean()

    fig, ax = plt.subplots(figsize=(10, 4))

    data = model_metric.values

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(available)))
    ax.set_xticklabels([m.upper() for m in available])

    ax.set_yticks(range(len(model_metric.index)))
    ax.set_yticklabels(model_metric.index)

    plt.colorbar(im)

    path = out_dir / "chart_heatmap.png"

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    print("✓", path)


# ── Chart 3: Prompt Strategy ───────────────────


def chart_prompt_strategies(lb, out_dir):

    if "composite_score" not in lb.columns:
        return

    strat_scores = (
        lb.groupby(["prompt_strategy", "model"])["composite_score"]
        .mean()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    n_models = len(strat_scores.columns)
    x = np.arange(len(strat_scores.index))

    width = 0.8 / n_models

    for i, model in enumerate(strat_scores.columns):

        offset = (i - n_models / 2) * width + width / 2

        ax.bar(
            x + offset,
            strat_scores[model],
            width=width,
            label=model,
            color=MODEL_COLORS.get(model, PALETTE["blue"]),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(strat_scores.index, rotation=20)

    ax.set_ylabel("Composite Score")

    ax.legend()

    plt.tight_layout()

    path = out_dir / "chart_prompt_strategies.png"
    plt.savefig(path)
    plt.close()

    print("✓", path)


# ── Chart 4: Latency vs Quality ────────────────


def chart_latency_quality(lb, out_dir):

    if "composite_score" not in lb.columns:
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    for _, row in lb.iterrows():

        ax.scatter(
            row["latency_s"],
            row["composite_score"],
            color=MODEL_COLORS.get(row["model"], PALETTE["blue"]),
            s=60,
        )

    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Composite Score")

    plt.tight_layout()

    path = out_dir / "chart_latency_quality.png"
    plt.savefig(path)
    plt.close()

    print("✓", path)


# ── Chart 5: Radar ─────────────────────────────


def chart_radar(lb, out_dir):

    metrics = ["rouge1", "rouge2", "rougeL", "bertscore_f1"]

    available = [m for m in metrics if m in lb.columns]

    if len(available) < 3:
        return

    model_scores = lb.groupby("model")[available].mean()

    N = len(available)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for model, row in model_scores.iterrows():

        values = row[available].tolist()
        values += values[:1]

        color = MODEL_COLORS.get(model, PALETTE["blue"])

        ax.plot(angles, values, label=model, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper() for m in available])

    ax.legend()

    path = out_dir / "chart_radar.png"

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    print("✓", path)


# ── Master chart ───────────────────────────────


def chart_master(lb, df, out_dir):

    if "composite_score" not in lb.columns:
        return

    fig = plt.figure(figsize=(16, 10))

    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    model_scores = lb.groupby("model")["composite_score"].mean().sort_values()

    ax1.barh(
        model_scores.index,
        model_scores.values,
        color=[MODEL_COLORS.get(m, PALETTE["blue"]) for m in model_scores.index],
    )

    ax1.set_title("Leaderboard")

    model_metric = lb.groupby("model")[["rouge1", "rouge2", "rougeL"]].mean()

    model_metric.plot(kind="bar", ax=ax2)

    ax2.set_title("Metrics")

    model_avg = lb.groupby("model")[["latency_s", "composite_score"]].mean()

    for model, row in model_avg.iterrows():

        ax3.scatter(row["latency_s"], row["composite_score"])

        ax3.annotate(model, (row["latency_s"], row["composite_score"]))

    ax3.set_title("Latency vs Quality")

    best_prompts = lb.loc[lb.groupby("model")["composite_score"].idxmax()]

    ax4.barh(best_prompts["model"], best_prompts["composite_score"])

    ax4.set_title("Best Prompt per Model")

    path = out_dir / "charts.png"

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    print("✓", path)


# ── Runner ─────────────────────────────────────


def generate_all_charts(results_path=None):

    results_path = results_path or RESULTS_DIR / "results.csv"

    if not results_path.exists():
        raise FileNotFoundError("No results file found")

    setup_style()

    RESULTS_DIR.mkdir(exist_ok=True)

    df, lb = load_data(results_path)

    print("Generating charts...")

    chart_leaderboard(lb, RESULTS_DIR)
    chart_heatmap(lb, RESULTS_DIR)
    chart_prompt_strategies(lb, RESULTS_DIR)
    chart_latency_quality(lb, RESULTS_DIR)
    chart_radar(lb, RESULTS_DIR)
    chart_master(lb, df, RESULTS_DIR)

    print("Charts saved to", RESULTS_DIR)


# ── Entry ──────────────────────────────────────


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input", type=str, help="Path to results CSV (default: results/results.csv)"
    )

    args = parser.parse_args()

    generate_all_charts(Path(args.input) if args.input else None)
    