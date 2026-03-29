"""
run_benchmark.py
================
LLM Prompt Benchmark System — Benchmark Runner
Project: P7 · prompt-engineering-lab by ChuksForge

Usage:
    python run_benchmark.py                          # full benchmark
    python run_benchmark.py --models openai          # one provider
    python run_benchmark.py --tasks summarization,qa # specific tasks
    python run_benchmark.py --strategies zero_shot   # one strategy
    python run_benchmark.py --quick                  # 2 cases/task, 1 strategy
"""

import os
import re
import time
import logging
import argparse
from pathlib import Path

import pandas as pd

from evaluation import evaluate, BenchmarkResult
from tasks.task_definitions import TASKS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")

MODELS = {
    "openai": {
        "gpt-4o-mini": {"provider": "openai",     "label": "GPT-4o-mini"},
        "gpt-4o":      {"provider": "openai",     "label": "GPT-4o"},
    },
    "anthropic": {
        "claude-haiku-4-5-20251001":    {"provider": "anthropic", "label": "Claude Haiku"},
        "claude-sonnet-4-6": {"provider": "anthropic", "label": "Claude Sonnet 4.6"},
    },
    "openrouter": {
        "google/gemini-2.0-flash-001":  {"provider": "openrouter", "label": "Gemini 2.0 Flash"},
        "meta-llama/llama-3-8b-instruct": {"provider": "openrouter", "label": "Llama 3 8B"},
    },
}


# ── Clients ──────────────────────────────────────────────────

def init_clients(model_filter):
    clients, active = {}, set()
    needed = set()
    for grp, grp_models in MODELS.items():
        if model_filter and grp not in model_filter:
            continue
        for _, meta in grp_models.items():
            needed.add(meta["provider"])

    from openai import OpenAI
    import anthropic as ant

    if "openai" in needed and os.environ.get("OPENAI_API_KEY"):
        try:
            clients["openai"] = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            active.add("openai"); logger.info("  openai ready")
        except Exception as e:
            logger.warning(f"  openai: {e}")

    if "anthropic" in needed and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            clients["anthropic"] = ant.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            active.add("anthropic"); logger.info("  anthropic ready")
        except Exception as e:
            logger.warning(f"  anthropic: {e}")

    if "openrouter" in needed and os.environ.get("OPENROUTER_API_KEY"):
        try:
            clients["openrouter"] = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
            active.add("openrouter"); logger.info("  openrouter ready")
        except Exception as e:
            logger.warning(f"  openrouter: {e}")

    return clients, active


def call_model(provider, model_id, prompt, clients):
    for attempt in range(3):
        try:
            t0 = time.time()
            if provider in ("openai", "openrouter"):
                resp = clients[provider].chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2, max_tokens=600,
                )
                usage = resp.usage
                return (
                    resp.choices[0].message.content.strip(),
                    time.time() - t0,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                )
            elif provider == "anthropic":
                resp = clients[provider].messages.create(
                    model=model_id, max_tokens=600, temperature=0.2,
                    messages=[{"role": "user", "content": prompt}],
                )
                return (
                    resp.content[0].text.strip(),
                    time.time() - t0,
                    resp.usage.input_tokens,
                    resp.usage.output_tokens,
                )
        except Exception as e:
            if attempt == 2:
                raise
            logger.warning(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(2)


# ── Prompt filling ───────────────────────────────────────────

def fill_prompt(template: str, case: dict) -> str:
    """Fill a prompt template with case variables."""
    result = template
    # Common variables
    for key in ("input", "question", "context", "code"):
        val = case.get(key, "")
        result = result.replace(f"{{{key}}}", str(val))
    return result


# ── Leaderboard building ─────────────────────────────────────

def build_leaderboard(df: pd.DataFrame):
    df_clean = df[df["error"].isna() | (df["error"] == "")]

    # Per task
    per_task = (
        df_clean.groupby(["model", "task", "prompt_strategy"])
        [["task_score", "cost_usd", "quality_per_dollar", "latency_s"]]
        .mean().round(4).reset_index()
    )
    per_task.to_csv(RESULTS_DIR / "leaderboard_per_task.csv", index=False)

    # Overall composite (avg task_score across all tasks)
    overall = (
        df_clean.groupby("model")
        [["task_score", "cost_usd", "quality_per_dollar", "latency_s"]]
        .mean().round(4).reset_index()
        .sort_values("task_score", ascending=False)
    )
    overall.insert(0, "rank", range(1, len(overall)+1))
    overall.to_csv(RESULTS_DIR / "leaderboard.csv", index=False)
    logger.info("  leaderboard.csv saved")

    # Cost efficiency leaderboard
    cost_lb = (
        df_clean.groupby("model")
        [["quality_per_dollar", "cost_usd", "task_score"]]
        .mean().round(6).reset_index()
        .sort_values("quality_per_dollar", ascending=False)
    )
    cost_lb.to_csv(RESULTS_DIR / "cost_leaderboard.csv", index=False)
    logger.info("  cost_leaderboard.csv saved")


# ── Main loop ────────────────────────────────────────────────

def run_benchmark(
    model_filter=None,
    task_filter=None,
    strategy_filter=None,
    quick=False,
):
    RESULTS_DIR.mkdir(exist_ok=True)
    clients, active = init_clients(model_filter)
    if not clients:
        raise RuntimeError("No API clients initialized.")

    tasks_to_run = {
        k: v for k, v in TASKS.items()
        if not task_filter or k in task_filter
    }

    all_results = []
    total = errors = 0

    for task_name, task in tasks_to_run.items():
        cases = task["cases"][:2] if quick else task["cases"]
        prompts = task["prompts"]

        if strategy_filter:
            prompts = {k: v for k, v in prompts.items() if k in strategy_filter}
        if quick:
            prompts = {list(prompts.keys())[0]: list(prompts.values())[0]}

        for case in cases:
            for strat_name, template in prompts.items():
                prompt = fill_prompt(template, case)

                for grp, grp_models in MODELS.items():
                    if model_filter and grp not in model_filter:
                        continue
                    for model_id, mmeta in grp_models.items():
                        provider = mmeta["provider"]
                        if provider not in active:
                            continue

                        logger.info(
                            f"  [{task_name}] {case['id']} × {strat_name} → {mmeta['label']}"
                        )

                        try:
                            output, latency, ptok, ctok = call_model(
                                provider, model_id, prompt, clients
                            )
                            result = evaluate(
                                task=task_name,
                                case=case,
                                output=output,
                                model=mmeta["label"],
                                prompt_strategy=strat_name,
                                prompt_tokens=ptok,
                                completion_tokens=ctok,
                                latency_s=latency,
                            )
                            total += 1

                        except Exception as e:
                            logger.error(f"    FAILED: {e}")
                            result = BenchmarkResult(
                                task=task_name, case_id=case["id"],
                                model=mmeta["label"], prompt_strategy=strat_name,
                                prompt_tokens=0, completion_tokens=0,
                                output="", latency_s=0.0, error=str(e),
                            )
                            errors += 1

                        all_results.append(result.to_dict())
                        time.sleep(0.3)

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "results.csv", index=False)
    logger.info(f"\n  results.csv saved ({total} runs, {errors} errors)")
    build_leaderboard(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",     type=str)
    parser.add_argument("--tasks",      type=str)
    parser.add_argument("--strategies", type=str)
    parser.add_argument("--quick",      action="store_true")
    args = parser.parse_args()

    run_benchmark(
        model_filter    = args.models.split(",")     if args.models     else None,
        task_filter     = args.tasks.split(",")      if args.tasks      else None,
        strategy_filter = args.strategies.split(",") if args.strategies else None,
        quick           = args.quick,
    )
