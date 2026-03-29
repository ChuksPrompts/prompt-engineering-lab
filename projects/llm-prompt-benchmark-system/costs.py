"""
costs.py
========
LLM Prompt Benchmark System — Cost Calculator
Project: P7 · prompt-engineering-lab by ChuksForge

Token pricing per million tokens (input / output).
Updated: March 2026 — verify current prices at provider docs.

Usage:
    from costs import calculate_cost, PRICING

    cost = calculate_cost("gpt-4o-mini", prompt_tokens=500, completion_tokens=150)
    # Returns cost in USD
"""

from dataclasses import dataclass

# ── Pricing table (USD per million tokens) ──────────────────
# Format: { model_label: (input_per_M, output_per_M) }

PRICING = {
    # OpenAI
    "GPT-4o-mini":           (0.15,   0.60),
    "GPT-4o":                (2.50,  10.00),
    # Anthropic
    "Claude Haiku":          (0.25,   1.25),
    "Claude Sonnet 4.6":     (3.00,  15.00),
    # OpenRouter (approximate — varies by route)
    "Mistral 7B":            (0.07,   0.07),
    "Llama 3 8B":            (0.05,   0.08),
    "Llama 3 70B":           (0.52,   0.75),
    "Gemini Flash":          (0.075,  0.30),
}


@dataclass
class CostResult:
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


def calculate_cost(
    model_label: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> CostResult:
    """
    Calculate the USD cost for a single API call.

    Args:
        model_label:       Label matching a key in PRICING (e.g. "GPT-4o-mini")
        prompt_tokens:     Number of input/prompt tokens
        completion_tokens: Number of output/completion tokens

    Returns:
        CostResult with breakdown and total cost in USD
    """
    pricing = PRICING.get(model_label, (1.0, 1.0))  # fallback to $1/M if unknown
    input_cost  = (prompt_tokens     / 1_000_000) * pricing[0]
    output_cost = (completion_tokens / 1_000_000) * pricing[1]

    return CostResult(
        model=model_label,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        input_cost_usd=round(input_cost, 8),
        output_cost_usd=round(output_cost, 8),
        total_cost_usd=round(input_cost + output_cost, 8),
    )


def cost_per_quality(total_cost_usd: float, quality_score: float) -> float:
    """
    Cost efficiency metric: cost per unit of quality.
    Lower = better value.

    cost_per_quality = total_cost_usd / quality_score
    If quality_score is 0, returns infinity.
    """
    if quality_score <= 0:
        return float("inf")
    return round(total_cost_usd / quality_score, 8)


def quality_per_dollar(total_cost_usd: float, quality_score: float) -> float:
    """
    Inverted cost efficiency: quality per dollar spent.
    Higher = better value.
    """
    if total_cost_usd <= 0:
        return float("inf")
    return round(quality_score / total_cost_usd, 4)


if __name__ == "__main__":
    print("Token pricing (USD per 1M tokens):\n")
    print(f"{'Model':25s}  {'Input':>10s}  {'Output':>10s}")
    print("-" * 50)
    for model, (inp, out) in PRICING.items():
        print(f"{model:25s}  ${inp:>9.3f}  ${out:>9.3f}")

    print("\nExample: 1000 prompt + 200 completion tokens")
    for model in ["GPT-4o-mini", "GPT-4o", "Claude Haiku", "Claude Sonnet 4.6"]:
        result = calculate_cost(model, 1000, 200)
        print(f"  {model:25s}  ${result.total_cost_usd:.6f}")
