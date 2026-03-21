from __future__ import annotations

import random
from time import perf_counter

from ..entities import Model, GenerationResult


def generate_random_baseline(
    *,
    prompt: str,
    model: Model,
    system_prompt: str | None = None,
) -> GenerationResult:
    """
    uniform random baseline for with fixed labels: ["A", "B", "C", "D"]
    """
    labels = ["A", "B", "C", "D"]

    started = perf_counter()
    chosen_label = random.choice(labels)
    latency_sec = perf_counter() - started

    return GenerationResult(
        text=chosen_label,
        provider="random_baseline",
        model_name=model.model_name,
        finish_reason="random_choice",
        usage_prompt_tokens=None,
        usage_completion_tokens=1,
        raw_response={
            "candidate_labels": labels,
            "latency_sec": round(latency_sec, 6),
        },
    )