"""
visualize.py
============
LLM Prompt Benchmark System — Static Chart Generator
Project: P7 · prompt-engineering-lab

Generates charts from results/ for README and notebook.
Run after run_benchmark.py.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

RESULTS_DIR = Path("results")

PALETTE = {
    "bg":"#0f1117","surface":"#161820","border":"#1e2130",
    "text":"#f0f2f8","muted":"#5a6080","accent":"#e8ff47",
    "blue":"#47c8ff","purple":"#b847ff","orange":"#ff8c47",
    "green":"#47ffb2","red":"#ff4776",
}
MODEL_COLORS = {
    "GPT-4o-mini":"#47c8ff","GPT-4o":"#e8ff47",
    "Claude Haiku":"#ff8c47","Claude Sonnet 4.6":"#b847ff",
    "Mistral 7B":"#47ffb2","Llama 3 8B":"#ff4776",
}
TASK_COLORS = {
    "summarization":"#47c8ff","qa":"#e8ff47",
    "reasoning":"#b847ff","coding":"#47ffb2",
}

def setup_style():
    plt.rcParams.update({
        "figure.facecolor":PALETTE["bg"],"axes.facecolor":PALETTE["surface"],
        "axes.edgecolor":PALETTE["border"],"axes.labelcolor":PALETTE["text"],
        "axes.titlecolor":PALETTE["text"],"xtick.color":PALETTE["muted"],
        "ytick.color":PALETTE["muted"],"text.color":PALETTE["text"],
        "grid.color":PALETTE["border"],"grid.linewidth":0.5,
        "font.family":"monospace","font.size":9,
        "axes.titlesize":10,"axes.titleweight":"bold",
        "figure.dpi":150,
    })

def load_data():
    df  = pd.read_csv(RESULTS_DIR/"results.csv")
    df  = df[df["error"].isna()|(df["error"]=="")].copy()
    lb  = pd.read_csv(RESULTS_DIR/"leaderboard.csv")
    clb = pd.read_csv(RESULTS_DIR/"cost_leaderboard.csv")
    ptl = pd.read_csv(RESULTS_DIR/"leaderboard_per_task.csv")
    return df, lb, clb, ptl


def chart_master(df, lb, clb, ptl, out):
    fig = plt.figure(figsize=(16,10))
    fig.patch.set_facecolor(PALETTE["bg"])
    gs = GridSpec(2,2,figure=fig,hspace=0.45,wspace=0.35)
    ax1,ax2 = fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])
    ax3,ax4 = fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])

    # Panel 1: Leaderboard
    lb_s = lb.sort_values("task_score",ascending=True)
    colors = [MODEL_COLORS.get(m,PALETTE["blue"]) for m in lb_s["model"]]
    bars = ax1.barh(lb_s["model"],lb_s["task_score"],color=colors,height=0.55,alpha=0.9)
    for bar,val in zip(bars,lb_s["task_score"]):
        ax1.text(val+0.01,bar.get_y()+bar.get_height()/2,f"{val:.3f}",va="center",fontsize=8,color=PALETTE["text"])
    ax1.set_xlim(0,1.2); ax1.set_xlabel("Avg Task Score")
    ax1.set_title("LEADERBOARD",loc="left",fontsize=9)
    ax1.grid(axis="x",alpha=0.3); ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # Panel 2: Task heatmap
    pivot = df.groupby(["model","task"])["task_score"].mean().unstack(fill_value=0).round(3)
    im = ax2.imshow(pivot.values,cmap="RdYlGn",aspect="auto",vmin=0,vmax=1)
    ax2.set_xticks(range(len(pivot.columns))); ax2.set_xticklabels(pivot.columns,rotation=15,ha="right",fontsize=8)
    ax2.set_yticks(range(len(pivot.index))); ax2.set_yticklabels(pivot.index,fontsize=8)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val=pivot.values[i,j]; col="black" if val>0.6 else PALETTE["text"]
            ax2.text(j,i,f"{val:.2f}",ha="center",va="center",color=col,fontsize=8)
    plt.colorbar(im,ax=ax2,fraction=0.04); ax2.set_title("SCORE HEATMAP",loc="left",fontsize=9)

    # Panel 3: Cost vs quality scatter
    model_avg = df.groupby("model")[["cost_usd","task_score"]].mean().reset_index()
    for _,row in model_avg.iterrows():
        color = MODEL_COLORS.get(row["model"],PALETTE["blue"])
        ax3.scatter(row["cost_usd"]*1000,row["task_score"],color=color,s=150,zorder=4,edgecolors="white",linewidths=0.5)
        ax3.annotate(row["model"],(row["cost_usd"]*1000,row["task_score"]),
                     textcoords="offset points",xytext=(6,3),color=color,fontsize=7,fontweight="bold")
    ax3.set_xlabel("Avg Cost ($ ×10⁻³)"); ax3.set_ylabel("Avg Score")
    ax3.set_title("COST vs QUALITY",loc="left",fontsize=9)
    ax3.grid(alpha=0.3); ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    # Panel 4: Quality per dollar
    clb_s = clb.sort_values("quality_per_dollar",ascending=True)
    colors4 = [MODEL_COLORS.get(m,PALETTE["blue"]) for m in clb_s["model"]]
    bars4 = ax4.barh(clb_s["model"],clb_s["quality_per_dollar"],color=colors4,height=0.55,alpha=0.9)
    for bar,val in zip(bars4,clb_s["quality_per_dollar"]):
        ax4.text(val+0.05,bar.get_y()+bar.get_height()/2,f"{val:.1f}",va="center",fontsize=8,color=PALETTE["text"])
    ax4.set_xlabel("Quality Score per $1 Spent")
    ax4.set_title("COST EFFICIENCY",loc="left",fontsize=9)
    ax4.grid(axis="x",alpha=0.3); ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

    fig.suptitle("LLM PROMPT BENCHMARK SYSTEM — Results Overview",
                 fontsize=13,fontweight="bold",color=PALETTE["text"],y=0.99)
    path = out/"charts.png"
    plt.savefig(path,bbox_inches="tight",facecolor=PALETTE["bg"],dpi=150)
    plt.close(); print(f"  ✓ {path}")


def generate_all_charts():
    if not (RESULTS_DIR/"results.csv").exists():
        raise FileNotFoundError("Run run_benchmark.py first.")
    setup_style()
    df,lb,clb,ptl = load_data()
    print(f"\n Generating charts ({len(df)} rows)...\n")
    chart_master(df,lb,clb,ptl,RESULTS_DIR)
    print(f"\n Charts saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    generate_all_charts()
