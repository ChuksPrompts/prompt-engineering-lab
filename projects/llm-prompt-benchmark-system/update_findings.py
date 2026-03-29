"""
update_findings.py
==================
LLM Prompt Benchmark System — Auto-populate findings
Project: P7 · prompt-engineering-lab by ChuksForge
"""

from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results")
README_PATH = Path("README.md")


def load():
    df  = pd.read_csv(RESULTS_DIR/"results.csv")
    lb  = pd.read_csv(RESULTS_DIR/"leaderboard.csv")
    clb = pd.read_csv(RESULTS_DIR/"cost_leaderboard.csv")
    ptl = pd.read_csv(RESULTS_DIR/"leaderboard_per_task.csv")
    df  = df[df["error"].isna()|(df["error"]=="")].copy()
    return df, lb, clb, ptl


def build_leaderboard_table(lb, clb):
    lb_merged = lb.merge(clb[["model","quality_per_dollar"]], on="model", how="left")
    lb_merged = lb_merged.sort_values("task_score", ascending=False).reset_index(drop=True)
    rows = [
        "| Rank | Model | Avg Score | Quality/$ | Avg Cost/Run | Avg Latency |",
        "|------|-------|-----------|-----------|--------------|-------------|",
    ]
    for rank, (_, row) in enumerate(lb_merged.iterrows(), 1):
        rows.append(
            f"| {rank} | {row['model']} "
            f"| {row.get('task_score',0):.3f} "
            f"| {row.get('quality_per_dollar',0):.1f} "
            f"| ${row.get('cost_usd',0)*1000:.4f}×10⁻³ "
            f"| {row.get('latency_s',0):.2f}s |"
        )
    return "\n".join(rows)


def build_task_table(df):
    pivot = df.groupby(["task","model"])["task_score"].mean().round(3).unstack()
    rows = ["| Task | " + " | ".join(pivot.columns) + " |",
            "|------|" + "|".join(["------"]*len(pivot.columns)) + "|"]
    for task, row in pivot.iterrows():
        rows.append("| " + task + " | " + " | ".join(f"{v:.3f}" for v in row.values) + " |")
    return "\n".join(rows)


def update_readme(lb_table, task_table):
    content = README_PATH.read_text(encoding="utf-8")
    for start, end, replacement in [
        ("| Rank | Model | Avg Score |", "\n\n*Run", lb_table),
        ("| Task |", "\n\n---", task_table),
    ]:
        si = content.find(start); ei = content.find(end, si)
        if si != -1 and ei != -1:
            content = content[:si] + replacement + content[ei:]
    README_PATH.write_text(content, encoding="utf-8")
    print("  README.md updated.")


def print_findings(df, lb, clb, ptl):
    best_model = lb.iloc[0]["model"]; best_score = lb.iloc[0]["task_score"]
    best_value = clb.sort_values("quality_per_dollar", ascending=False).iloc[0]
    task_avg   = df.groupby("task")["task_score"].mean().sort_values(ascending=False)
    best_task  = task_avg.index[0]; best_task_score = task_avg.iloc[0]
    hardest    = task_avg.index[-1]; hardest_score  = task_avg.iloc[-1]
    strat_avg  = ptl.groupby("prompt_strategy")["task_score"].mean().sort_values(ascending=False)
    best_strat = strat_avg.index[0]; best_strat_score = strat_avg.iloc[0]
    lat_avg    = df.groupby("model")["latency_s"].mean().sort_values()
    fastest    = lat_avg.index[0]; fastest_lat = lat_avg.iloc[0]

    print("\n" + "="*62)
    print("COPY THIS INTO NOTEBOOK KEY FINDINGS CELL:")
    print("="*62)
    print(f"""
## Key Findings

1. **Best model overall:** `{best_model}` — avg score {best_score:.3f} across all tasks
2. **Best cost efficiency:** `{best_value['model']}` — {best_value['quality_per_dollar']:.1f} quality points per $1
3. **Easiest task for models:** `{best_task}` — avg score {best_task_score:.3f}
4. **Hardest task for models:** `{hardest}` — avg score {hardest_score:.3f}
5. **Best prompt strategy:** `{best_strat}` — avg score {best_strat_score:.3f}
6. **Fastest model:** `{fastest}` — {fastest_lat:.2f}s avg latency
7. **Key insight:** [Fill in after reviewing cost vs quality scatter]

---
*Dashboard: `streamlit run dashboard.py`*
""")


def main():
    print("Loading results...")
    df, lb, clb, ptl = load()
    print(f"  {len(df)} rows | {df['model'].nunique()} models | {df['task'].nunique()} tasks\n")
    print("Building README tables...")
    update_readme(build_leaderboard_table(lb, clb), build_task_table(df))
    print_findings(df, lb, clb, ptl)
    print("\nDone.")

if __name__ == "__main__":
    main()
