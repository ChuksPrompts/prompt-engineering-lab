"""
run_experiment.py

Summarization Benchmark — Experiment Runner
Project: P1 · prompt-engineering-lab

Usage:
python run_experiment.py                         # full run (all models, all prompts)
python run_experiment.py --models openai         # single model
python run_experiment.py --prompts P01,P05,P07   # specific prompts
python run_experiment.py --articles A01,A02      # specific articles
python run_experiment.py --quick                 # fast subset for testing
python run_experiment.py --llm-judge             # enable LLM-as-judge scoring

Outputs:
results/results.csv     — full results table
results/leaderboard.csv — aggregated model×prompt leaderboard
"""

import os
import time
import argparse
import logging
import re
from pathlib import Path
from datetime import datetime

import pandas as pd

from evaluation import evaluate_summary, BenchmarkResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

MODELS = {
    "openai": {
        "gpt-4o-mini": {"provider": "openai", "label": "GPT-4o-mini"},
        "gpt-4o": {"provider": "openai", "label": "GPT-4o"},
    },
    "anthropic": {
        "claude-haiku-4-5-20251001": {"provider": "anthropic", "label": "Claude Haiku"},
        "claude-sonnet-4-6": {"provider": "anthropic", "label": "Claude Sonnet 4.6"},
    },
    "google": {
        "gemini-1.5-flash": {"provider": "google", "label": "Gemini 1.5 Flash"},
        "gemini-1.5-pro": {"provider": "google", "label": "Gemini 1.5 Pro"},
    },
}

RESULTS_DIR = Path("results")
DATA_DIR = Path("data")
PROMPTS_FILE = Path("prompts/prompts.txt")

# ──────────────────────────────────────────────
# Prompt loading
# ──────────────────────────────────────────────


def load_prompts(prompts_file: Path) -> dict:
    prompts = {}
    current_id = None
    current_strategy = ""
    current_lines = []

    with open(prompts_file, encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            # Skip P_EVAL block
            if line.startswith("## [P_EVAL]"):
                current_id = None
                continue

            # Skip comment-only lines that aren't inside a prompt block
            if line.startswith("#") and not line.startswith("## [P"):
                continue

            # New prompt header: ## [P01] baseline_zero_shot
            m = re.match(r'^##\s+\[(\w+)\]\s+(\S+)', line)
            if m:
                # Save previous prompt
                if current_id and current_id != "P_EVAL":
                    prompts[current_id] = {
                        "strategy": current_strategy,
                        "template": "\n".join(current_lines).strip(),
                    }
                current_id = m.group(1)
                current_strategy = ""
                current_lines = []
                continue

            if current_id is None:
                continue

            # Metadata lines
            if line.startswith("strategy:"):
                current_strategy = line.split(":", 1)[1].strip()
                continue
            if line.startswith("expected_strength:"):
                continue  # skip, not needed

            # Everything else is template content
            current_lines.append(line)

    # Save last prompt
    if current_id and current_id != "P_EVAL":
        prompts[current_id] = {
            "strategy": current_strategy,
            "template": "\n".join(current_lines).strip(),
        }

    return prompts


def fill_prompt(template: str, text: str) -> str:
    """Replace template variables."""

    word_count = len(re.findall(r"\b\w+\b", text))
    target_words = max(10, round(word_count * 0.10))

    result = template.replace("{{TEXT}}", text)
    result = result.replace("{{WORD_COUNT}}", str(word_count))
    result = result.replace("{{TARGET_WORDS}}", str(target_words))

    return result


# ──────────────────────────────────────────────
# Model clients
# ──────────────────────────────────────────────


def get_openai_client():
    from openai import OpenAI

    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def get_anthropic_client():
    import anthropic

    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def get_google_client():
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    return genai


def call_model(provider: str, model_id: str, prompt: str, clients: dict):
    """
    Call a model and return (response_text, latency).
    """

    for attempt in range(3):

        try:
            t0 = time.time()

            if provider == "openai":

                client = clients["openai"]

                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500,
                )

                text = resp.choices[0].message.content.strip()

            elif provider == "anthropic":

                client = clients["anthropic"]

                resp = client.messages.create(
                    model=model_id,
                    max_tokens=500,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}],
                )

                text = resp.content[0].text.strip()

            elif provider == "google":

                genai = clients["google"]
                model = genai.GenerativeModel(model_id)

                resp = model.generate_content(prompt)
                text = resp.text.strip()

            else:
                raise ValueError(f"Unknown provider: {provider}")

            latency = time.time() - t0

            return text, latency

        except Exception as e:

            if attempt == 2:
                raise

            logger.warning(f"Attempt {attempt+1} failed ({e}), retrying in 2s")
            time.sleep(2)


# ──────────────────────────────────────────────
# Extract summary
# ──────────────────────────────────────────────


def extract_summary_text(raw_output: str, prompt_id: str):

    if not raw_output:
        return ""

    if "FINAL SUMMARY:" in raw_output:
        return raw_output.split("FINAL SUMMARY:")[-1].strip()

    if "ONE-LINE SUMMARY:" in raw_output:

        for line in raw_output.split("\n"):

            if line.startswith("ONE-LINE SUMMARY:"):
                return line.replace("ONE-LINE SUMMARY:", "").strip()

    if "GENERAL PUBLIC SUMMARY" in raw_output:

        parts = raw_output.split("GENERAL PUBLIC SUMMARY")

        if len(parts) > 1:
            text = parts[1].split("\n\n")[0]
            return re.sub(r'^\s*\(.*?\)\s*:', '', text).strip()

    return raw_output.strip()


# ──────────────────────────────────────────────
# Experiment loop
# ──────────────────────────────────────────────


def run_experiment(
    model_filter=None,
    prompt_filter=None,
    article_filter=None,
    run_llm_judge=False,
    quick=False,
):

    RESULTS_DIR.mkdir(exist_ok=True)

    articles = pd.read_csv(DATA_DIR / "articles.csv")
    prompts = load_prompts(PROMPTS_FILE)

    if quick:
        articles = articles.head(2)
        prompt_filter = prompt_filter or ["P01", "P05", "P07"]
        logger.info("Quick mode: 2 articles, 3 prompts")

    if article_filter:
        articles = articles[articles["id"].isin(article_filter)]

    if prompt_filter:
        prompts = {k: v for k, v in prompts.items() if k in prompt_filter}

    clients = {}
    active_providers = set()

    needed_providers = set()

    for group_name, model_group in MODELS.items():

        if model_filter and group_name not in model_filter:
            continue

        for _, meta in model_group.items():
            needed_providers.add(meta["provider"])

    if "openai" in needed_providers and os.environ.get("OPENAI_API_KEY"):
        clients["openai"] = get_openai_client()
        active_providers.add("openai")
        logger.info("OpenAI client initialized")

    if "anthropic" in needed_providers and os.environ.get("ANTHROPIC_API_KEY"):
        clients["anthropic"] = get_anthropic_client()
        active_providers.add("anthropic")
        logger.info("Anthropic client initialized")

    if "google" in needed_providers and os.environ.get("GOOGLE_API_KEY"):
        clients["google"] = get_google_client()
        active_providers.add("google")
        logger.info("Google client initialized")

    if not clients:
        raise RuntimeError("No API clients initialized.")

    judge_client = clients.get("openai") if run_llm_judge else None

    all_results = []
    total_runs = 0
    errors = 0

    for _, article in articles.iterrows():

        for pid, prompt_meta in prompts.items():

            for group_name, model_group in MODELS.items():

                if model_filter and group_name not in model_filter:
                    continue

                for model_id, model_meta in model_group.items():

                    provider = model_meta["provider"]

                    if provider not in active_providers:
                        continue

                    logger.info(f"[{article['id']}] {pid} → {model_meta['label']}")

                    filled = fill_prompt(prompt_meta["template"], article["text"])

                    try:

                        raw_output, latency = call_model(
                            provider, model_id, filled, clients
                        )

                        summary = extract_summary_text(raw_output, pid)

                        scores = evaluate_summary(
                            summary=summary,
                            reference=article["reference_summary"],
                            original=article["text"],
                            judge_client=judge_client,
                            run_bertscore=True,
                            run_llm_judge=run_llm_judge,
                        )

                        result = BenchmarkResult(
                            article_id=article["id"],
                            model=model_meta["label"],
                            prompt_id=pid,
                            prompt_strategy=prompt_meta["strategy"],
                            summary=summary,
                            latency_s=latency,
                            scores=scores,
                        )

                        total_runs += 1

                    except Exception as e:

                        logger.error(f"FAILED: {e}")

                        result = BenchmarkResult(
                            article_id=article["id"],
                            model=model_meta["label"],
                            prompt_id=pid,
                            prompt_strategy=prompt_meta["strategy"],
                            summary="",
                            latency_s=0.0,
                            error=str(e),
                        )

                        errors += 1

                    all_results.append(result.to_dict())

                    time.sleep(0.3)

    if not all_results:
        raise RuntimeError(
            f"No results generated. Articles={len(articles)}, Prompts={len(prompts)}, Providers={active_providers}"
        )

    df = pd.DataFrame(all_results)

    results_path = RESULTS_DIR / "results.csv"
    df.to_csv(results_path, index=False)

    logger.info(f"Results saved → {results_path}")

    leaderboard = build_leaderboard(df)

    lb_path = RESULTS_DIR / "leaderboard.csv"
    leaderboard.to_csv(lb_path, index=False)

    logger.info(f"Leaderboard → {lb_path}")

    return df


# ──────────────────────────────────────────────
# Leaderboard
# ──────────────────────────────────────────────


def build_leaderboard(df):

    metric_cols = [
        "rouge1",
        "rouge2",
        "rougeL",
        "bertscore_f1",
        "flesch_kincaid_grade",
        "compression_ratio",
        "word_count",
        "latency_s",
    ]

    available = [c for c in metric_cols if c in df.columns]

    if "error" in df.columns:
        df_clean = df[df["error"].isna() | (df["error"] == "")]
    else:
        df_clean = df

    lb = (
        df_clean.groupby(["model", "prompt_id", "prompt_strategy"])[available]
        .mean()
        .round(4)
        .reset_index()
    )

    if "rouge1" in lb.columns:
        lb = lb.sort_values("rouge1", ascending=False)

    if all(c in lb.columns for c in ["rouge1", "rouge2", "rougeL", "bertscore_f1"]):

        lb["composite_score"] = (
            0.2 * lb["rouge1"]
            + 0.2 * lb["rouge2"]
            + 0.2 * lb["rougeL"]
            + 0.4 * lb["bertscore_f1"]
        ).round(4)

        lb = lb.sort_values("composite_score", ascending=False)

    lb.insert(0, "rank", range(1, len(lb) + 1))

    return lb


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run summarization benchmark")

    parser.add_argument("--models", type=str)
    parser.add_argument("--prompts", type=str)
    parser.add_argument("--articles", type=str)
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--quick", action="store_true")

    args = parser.parse_args()

    model_f = args.models.split(",") if args.models else None
    prompt_f = args.prompts.split(",") if args.prompts else None
    article_f = args.articles.split(",") if args.articles else None

    run_experiment(
        model_filter=model_f,
        prompt_filter=prompt_f,
        article_filter=article_f,
        run_llm_judge=args.llm_judge,
        quick=args.quick,
    )
    