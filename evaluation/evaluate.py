from __future__ import annotations

from collections import Counter

from benchmark.entities import Model


def make_summary(results: list[dict], model: Model) -> dict:
    with_gold = [r for r in results if r["gold_label"] is not None]
    parse_counts = Counter(r["parse_status"] for r in results)

    summary = {
        "n_items": len(results),
        "n_with_gold": len(with_gold),
        "n_correct": sum(bool(r["is_correct"]) for r in with_gold),
        "accuracy": (
            sum(bool(r["is_correct"]) for r in with_gold) / len(with_gold)
            if with_gold
            else None
        ),
        "parse_status_counts": dict(parse_counts),
        "parse_ok_rate": (
            sum(r["parse_status"] == "ok" for r in results) / len(results)
            if results
            else None
        ),
        "provider": model.provider,
        "model_name": model.model_name,
    }

    return summary


    # TODO EVALUATION BY CATEGORY, PHENOMENON ETC.

