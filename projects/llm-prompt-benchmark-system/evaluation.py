"""
evaluation.py
=============
LLM Prompt Benchmark System — Evaluation Engine
Project: P7 · prompt-engineering-lab

Task-specific scorers:
  score_summarization  — ROUGE-1 + ROUGE-L composite
  score_qa             — keyword-based factual accuracy
  score_reasoning      — answer correctness + step detection
  score_coding         — keyword presence + structure quality

All scorers return a float 0.0–1.0.
"""

import re
import math
import logging
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    # Identity
    task: str
    case_id: str
    model: str
    prompt_strategy: str
    # I/O
    prompt_tokens: int
    completion_tokens: int
    output: str
    latency_s: float
    # Scores
    task_score: float = 0.0       # primary task metric (0-1)
    cost_usd: float = 0.0         # total API cost
    cost_per_quality: float = 0.0 # cost / task_score
    quality_per_dollar: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ── Tokenization helpers ─────────────────────────────────────

def _tok(text: str) -> list:
    return re.findall(r"\b\w+\b", text.lower())

def _ngrams(tokens, n):
    counts = {}
    for i in range(len(tokens) - n + 1):
        g = tuple(tokens[i:i+n])
        counts[g] = counts.get(g, 0) + 1
    return counts

def _rouge_n(hyp, ref, n):
    h, r = _tok(hyp), _tok(ref)
    hg, rg = _ngrams(h, n), _ngrams(r, n)
    overlap = sum(min(hg.get(g,0), rg[g]) for g in rg)
    rt = sum(rg.values()); ht = sum(hg.values())
    rec = overlap/rt if rt else 0.0
    pre = overlap/ht if ht else 0.0
    return round(2*pre*rec/(pre+rec), 4) if (pre+rec) else 0.0

def _lcs(x, y):
    m, n = len(x), len(y)
    if not m or not n: return 0
    prev = [0]*(n+1)
    for i in range(1, m+1):
        curr = [0]*(n+1)
        for j in range(1, n+1):
            curr[j] = prev[j-1]+1 if x[i-1]==y[j-1] else max(curr[j-1], prev[j])
        prev = curr
    return prev[n]

def _rouge_l(hyp, ref):
    h, r = _tok(hyp), _tok(ref)
    lcs = _lcs(h, r)
    rec = lcs/len(r) if r else 0.0
    pre = lcs/len(h) if h else 0.0
    return round(2*pre*rec/(pre+rec), 4) if (pre+rec) else 0.0


# ── Task scorers ─────────────────────────────────────────────

def score_summarization(output: str, case: dict) -> float:
    """ROUGE-1 + ROUGE-L composite."""
    reference = case.get("reference", "")
    if not reference:
        return 0.0
    r1 = _rouge_n(output, reference, 1)
    rl = _rouge_l(output, reference)
    return round((r1 + rl) / 2, 4)


def score_qa(output: str, case: dict) -> float:
    """
    Factual accuracy: checks if answer keywords appear in output.
    Returns 1.0 if primary answer found, 0.5 if partial, 0.0 if missing.
    """
    answer_keywords = case.get("answer_keywords", [])
    if not answer_keywords:
        return 0.0

    output_lower = output.lower()
    hits = sum(1 for kw in answer_keywords if kw.lower() in output_lower)

    if hits == 0:
        return 0.0
    elif hits >= len(answer_keywords):
        return 1.0
    else:
        return 0.5


def score_reasoning(output: str, case: dict) -> float:
    """
    Reasoning quality: answer correctness (0.6) + step presence (0.4).
    """
    answer_keywords = case.get("answer_keywords", [])
    output_lower = output.lower()

    # Answer correctness
    answer_hits = sum(1 for kw in answer_keywords if kw.lower() in output_lower)
    answer_score = 1.0 if answer_hits > 0 else 0.0

    # Step validity: did the model show reasoning steps?
    step_signals = [
        r'\b(step|first|second|third|therefore|because|since|thus)\b',
        r'\d+\.',             # numbered steps
        r'(so|hence|thus),',  # logical connectors
        r'(let|let me|we|if)',
    ]
    step_count = sum(bool(re.search(p, output_lower)) for p in step_signals)
    step_score = min(1.0, step_count / 3)

    return round(0.6 * answer_score + 0.4 * step_score, 4)


def score_coding(output: str, case: dict) -> float:
    """
    Code quality: keyword presence (0.5) + structure quality (0.5).
    """
    answer_keywords = case.get("answer_keywords", [])
    output_lower = output.lower()

    # Keyword presence (function definition, key logic)
    kw_hits = sum(1 for kw in answer_keywords if kw.lower() in output_lower)
    kw_score = min(1.0, kw_hits / max(len(answer_keywords), 1))

    # Structure quality signals
    structure_signals = [
        r'def \w+\(',                  # function definition
        r'""".*?"""',                  # docstring
        r'#\s+\w',                     # comments
        r'return\s+',                  # return statement
        r'if.*:',                      # conditional
        r'for.*:',                     # loop
    ]
    struct_count = sum(bool(re.search(p, output, re.DOTALL)) for p in structure_signals)
    struct_score = min(1.0, struct_count / 4)

    # Test cases presence (bonus for test_driven strategy)
    has_tests = bool(re.search(r'(assert|def test_|==)', output))
    test_bonus = 0.1 if has_tests else 0.0

    return round(min(1.0, 0.5 * kw_score + 0.4 * struct_score + test_bonus), 4)


# ── Dispatcher ───────────────────────────────────────────────

SCORERS = {
    "summarization": score_summarization,
    "qa":            score_qa,
    "reasoning":     score_reasoning,
    "coding":        score_coding,
}


def evaluate(
    task: str,
    case: dict,
    output: str,
    model: str,
    prompt_strategy: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_s: float,
) -> BenchmarkResult:
    """
    Evaluate one model output against a task case.
    Returns a fully-populated BenchmarkResult.
    """
    from costs import calculate_cost, cost_per_quality, quality_per_dollar

    scorer = SCORERS.get(task)
    task_score = scorer(output, case) if scorer else 0.0

    cost = calculate_cost(model, prompt_tokens, completion_tokens)
    cpq  = cost_per_quality(cost.total_cost_usd, task_score)
    qpd  = quality_per_dollar(cost.total_cost_usd, task_score)

    return BenchmarkResult(
        task=task,
        case_id=case["id"],
        model=model,
        prompt_strategy=prompt_strategy,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        output=output,
        latency_s=round(latency_s, 3),
        task_score=task_score,
        cost_usd=cost.total_cost_usd,
        cost_per_quality=cpq if cpq != float("inf") else 999.0,
        quality_per_dollar=qpd if qpd != float("inf") else 0.0,
    )


# ── Self-test ────────────────────────────────────────────────

if __name__ == "__main__":
    from tasks.task_definitions import TASKS

    for task_name, task in TASKS.items():
        case = task["cases"][0]
        # Simulate a decent output
        if task_name == "summarization":
            output = case["reference"]
        elif task_name == "qa":
            output = f"The answer is {case['answer']}"
        elif task_name == "reasoning":
            output = f"Step 1: analyze. Step 2: calculate. Therefore {case['answer']}"
        else:
            output = f"def {case['id'].lower()}(x):\n    # solution\n    return result"

        result = evaluate(
            task=task_name, case=case, output=output,
            model="GPT-4o-mini", prompt_strategy="test",
            prompt_tokens=200, completion_tokens=100, latency_s=1.0,
        )
        print(f"{task_name:15s}  score={result.task_score:.3f}  cost=${result.cost_usd:.6f}  qpd={result.quality_per_dollar:.2f}")
