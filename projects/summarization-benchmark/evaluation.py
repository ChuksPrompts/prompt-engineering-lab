"""
evaluation.py

Summarization Benchmark — Evaluation Engine
Project: P1 · prompt-engineering-lab

Computes:

- ROUGE-1, ROUGE-2, ROUGE-L  (lexical overlap)
- BERTScore F1               (semantic similarity)
- Flesch-Kincaid Grade Level (readability)
- Compression Ratio          (length efficiency)
- LLM-as-Judge scores        (faithfulness, coverage, conciseness, fluency, coherence)
"""

import re
import json
import math
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class EvalScores:
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    bertscore_f1: float = 0.0
    flesch_kincaid_grade: float = 0.0
    compression_ratio: float = 0.0
    word_count: int = 0

    # LLM judge metrics
    judge_faithfulness: float = 0.0
    judge_coverage: float = 0.0
    judge_conciseness: float = 0.0
    judge_fluency: float = 0.0
    judge_coherence: float = 0.0
    judge_overall: float = 0.0
    judge_rationale: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkResult:
    article_id: str
    model: str
    prompt_id: str
    prompt_strategy: str
    summary: str
    latency_s: float
    scores: EvalScores = field(default_factory=EvalScores)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "article_id": self.article_id,
            "model": self.model,
            "prompt_id": self.prompt_id,
            "prompt_strategy": self.prompt_strategy,
            "summary": self.summary,
            "latency_s": round(self.latency_s, 3),
            "error": self.error,
        }
        d.update(self.scores.to_dict())
        return d


# ──────────────────────────────────────────────
# ROUGE (pure Python)
# ──────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _ngrams(tokens: list[str], n: int) -> dict:
    counts = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i+n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def _rouge_n(hypothesis: str, reference: str, n: int) -> dict:

    hyp_tokens = _tokenize(hypothesis)
    ref_tokens = _tokenize(reference)

    hyp_ngrams = _ngrams(hyp_tokens, n)
    ref_ngrams = _ngrams(ref_tokens, n)

    overlap = sum(min(hyp_ngrams.get(g, 0), ref_ngrams[g]) for g in ref_ngrams)

    ref_total = sum(ref_ngrams.values())
    hyp_total = sum(hyp_ngrams.values())

    recall = overlap / ref_total if ref_total else 0.0
    precision = overlap / hyp_total if hyp_total else 0.0

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(x: list, y: list) -> int:
    """Dynamic programming LCS for ROUGE-L."""
    m, n = len(x), len(y)

    if m == 0 or n == 0:
        return 0

    prev = [0] * (n + 1)

    for i in range(1, m + 1):

        curr = [0] * (n + 1)

        for j in range(1, n + 1):

            if x[i-1] == y[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(curr[j-1], prev[j])

        prev = curr

    return prev[n]


def _rouge_l(hypothesis: str, reference: str) -> dict:

    hyp_tokens = _tokenize(hypothesis)
    ref_tokens = _tokenize(reference)

    lcs = _lcs_length(hyp_tokens, ref_tokens)

    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    precision = lcs / len(hyp_tokens) if hyp_tokens else 0.0

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_rouge(hypothesis: str, reference: str) -> dict:

    return {
        "rouge1": round(_rouge_n(hypothesis, reference, 1)["f1"], 4),
        "rouge2": round(_rouge_n(hypothesis, reference, 2)["f1"], 4),
        "rougeL": round(_rouge_l(hypothesis, reference)["f1"], 4),
    }


# ──────────────────────────────────────────────
# BERTScore
# ──────────────────────────────────────────────

def _tfidf_vector(text: str, vocab: list[str]) -> list[float]:

    tokens = _tokenize(text)

    tf = {t: tokens.count(t) / len(tokens) for t in set(tokens)} if tokens else {}

    return [tf.get(w, 0.0) for w in vocab]


def _cosine(a: list[float], b: list[float]) -> float:

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x**2 for x in a))
    norm_b = math.sqrt(sum(x**2 for x in b))

    return dot / (norm_a * norm_b) if norm_a * norm_b else 0.0


def compute_bertscore(hypothesis: str, reference: str) -> float:

    try:

        from bert_score import score as bert_score_fn

        P, R, F1 = bert_score_fn(
            [hypothesis],
            [reference],
            lang="en",
            verbose=False,
            model_type="distilbert-base-uncased"
        )

        return round(F1[0].item(), 4)

    except ImportError:

        vocab = list(set(_tokenize(hypothesis + " " + reference)))

        vec_h = _tfidf_vector(hypothesis, vocab)
        vec_r = _tfidf_vector(reference, vocab)

        return round(_cosine(vec_h, vec_r), 4)


# ──────────────────────────────────────────────
# Readability
# ──────────────────────────────────────────────

def compute_flesch_kincaid(text: str) -> float:

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    words = _tokenize(text)

    if not sentences or not words:
        return 0.0

    def count_syllables(word: str) -> int:
        word = word.lower()
        count = len(re.findall(r"[aeiouy]+", word))

        if word.endswith("e") and len(word) > 2:
            count -= 1

        return max(1, count)

    total_syllables = sum(count_syllables(w) for w in words)

    asl = len(words) / len(sentences)
    asw = total_syllables / len(words)

    grade = 0.39 * asl + 11.8 * asw - 15.59

    return round(grade, 2)


# ──────────────────────────────────────────────
# Compression Ratio
# ──────────────────────────────────────────────

def compute_compression_ratio(original: str, summary: str) -> float:

    orig_words = len(_tokenize(original))
    sum_words = len(_tokenize(summary))

    return round(sum_words / orig_words, 4) if orig_words else 0.0


# ──────────────────────────────────────────────
# LLM Judge
# ──────────────────────────────────────────────

JUDGE_PROMPT_TEMPLATE = """You are an expert at evaluating text summaries.

ORIGINAL TEXT:
{original}

SUMMARY:
{summary}

Score from 1-5:

faithfulness
coverage
conciseness
fluency
coherence

Respond ONLY with JSON:
{
"faithfulness": <score>,
"coverage": <score>,
"conciseness": <score>,
"fluency": <score>,
"coherence": <score>,
"overall": <average>,
"brief_rationale": "<short explanation>"
}
"""


def compute_llm_judge(original, summary, judge_client, judge_model="gpt-4o-mini"):

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        original=original[:3000],
        summary=summary
    )

    try:

        response = judge_client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )

        raw = response.choices[0].message.content.strip()

        raw = re.sub(r"`(?:json)?", "", raw).strip().rstrip("`")

        data = json.loads(raw)

        return {
            "faithfulness": float(data.get("faithfulness", 0)),
            "coverage": float(data.get("coverage", 0)),
            "conciseness": float(data.get("conciseness", 0)),
            "fluency": float(data.get("fluency", 0)),
            "coherence": float(data.get("coherence", 0)),
            "overall": float(data.get("overall", 0)),
            "rationale": data.get("brief_rationale", ""),
        }

    except Exception as e:

        logger.warning(f"LLM judge failed: {e}")

        return {
            "faithfulness": 0.0,
            "coverage": 0.0,
            "conciseness": 0.0,
            "fluency": 0.0,
            "coherence": 0.0,
            "overall": 0.0,
            "rationale": "",
        }


# ──────────────────────────────────────────────
# Master evaluation
# ──────────────────────────────────────────────

def evaluate_summary(
    summary: str,
    reference: str,
    original: str,
    judge_client=None,
    judge_model="gpt-4o-mini",
    run_bertscore=True,
    run_llm_judge=False,
) -> EvalScores:

    scores = EvalScores()

    rouge = compute_rouge(summary, reference)

    scores.rouge1 = rouge["rouge1"]
    scores.rouge2 = rouge["rouge2"]
    scores.rougeL = rouge["rougeL"]

    if run_bertscore:
        scores.bertscore_f1 = compute_bertscore(summary, reference)

    scores.flesch_kincaid_grade = compute_flesch_kincaid(summary)

    scores.compression_ratio = compute_compression_ratio(original, summary)

    scores.word_count = len(_tokenize(summary))

    if run_llm_judge and judge_client:

        judge = compute_llm_judge(original, summary, judge_client, judge_model)

        scores.judge_faithfulness = judge["faithfulness"]
        scores.judge_coverage = judge["coverage"]
        scores.judge_conciseness = judge["conciseness"]
        scores.judge_fluency = judge["fluency"]
        scores.judge_coherence = judge["coherence"]
        scores.judge_overall = judge["overall"]
        scores.judge_rationale = judge["rationale"]

    return scores


# ──────────────────────────────────────────────
# Self test
# ──────────────────────────────────────────────

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    test_original = (
        "Scientists at MIT have developed a new battery technology using sodium ions "
        "instead of lithium. The new batteries are cheaper to manufacture, use more "
        "abundant materials, and can be charged 40% faster than current lithium-ion batteries."
    )

    test_summary = (
        "MIT researchers created sodium-ion batteries that charge 40% faster "
        "and cost less than lithium batteries."
    )

    test_reference = (
        "MIT developed cheaper sodium-ion batteries that charge faster than lithium batteries."
    )

    scores = evaluate_summary(
        summary=test_summary,
        reference=test_reference,
        original=test_original,
    )

    print("\nSelf-test results:\n")

    for k, v in scores.to_dict().items():
        if v:
            print(f"{k:30s} {v}")
