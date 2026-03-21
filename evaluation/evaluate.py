from __future__ import annotations

from collections import Counter

from benchmark.entities import Model


def _subset_metrics(rows: list[dict]) -> dict:
    """
    computes statistics for a subset of evaluated items (e.g. only genre="beletrie")
    """
    parse_counts = Counter(r["parse_status"] for r in rows)
    n_correct = sum(bool(r["is_correct"]) for r in rows)

    return {
        "n_items": len(rows),
        "n_correct": n_correct,
        "accuracy": (n_correct / len(rows)) if rows else None, # TODO this accuracy will include items that were not parsed well, maybe change?
        "parse_status_counts": dict(parse_counts),
        "parse_ok_rate": (
            sum(r["parse_status"] == "ok" for r in rows) / len(rows)
            if rows else None
        ),
    }


def _group_metrics(results: list[dict], key_fn) -> dict:
    """
    group rows by a key extracted from each row, then compute metrics for each group with _subset_metrics
    """
    groups: dict[str, list[dict]] = {}

    for row in results:
        key = key_fn(row)

        if key is None or key == "":
            key = "__MISSING__"

        key = str(key)
        groups.setdefault(key, []).append(row)

    return {
        key: _subset_metrics(rows)
        for key, rows in groups.items()
    }



def _count_values(rows: list[dict], function) -> dict:
    """
    counts values using a given function

    usage:
    _count_values(rowa, lambda x: x.get("parsed_type")) - count distractors (but also can be used to smth else)
    """
    counts = Counter()

    for row in rows:
        value = function(row)
        if value is None or value == "":
            value = "__MISSING__"
        counts[str(value)] += 1

    return dict(counts)


def make_summary(results: list[dict], model: Model) -> dict:

    # apply the metrics function to the whole dataset
    general_results = _subset_metrics(results)

    summary = {
        **general_results,
        "provider": model.provider,
        "model_name": model.model_name,

        "n_model_errors": sum(r["parse_status"] == "model_error" for r in results),
        "n_parse_invalid": sum(r["parse_status"] == "invalid" for r in results),

        "parsed_type_counts": _count_values(results, lambda r: r.get("parsed_type")),

        # apply the metrics function on each grouping
        "by_phenomenon": _group_metrics(
            results,
            lambda r: r.get("item_meta", {}).get("phenomenon"),
        ),
        "by_category": _group_metrics(
            results,
            lambda r: r.get("item_meta", {}).get("category"),
        ),
        "by_creation_method": _group_metrics(
            results,
            lambda r: r.get("item_meta", {}).get("creation_method"),
        ),
        "by_genre": _group_metrics(
            results,
            lambda r: r.get("item_meta", {}).get("metadata", {}).get("genre"),
        ),
        "by_trigger": _group_metrics(
            results,
            lambda r: r.get("item_meta", {}).get("metadata", {}).get("trigger"),
        ),
    }

    # TODO if I run it on 3 phenomena separately, by_phenomenon doesn't make sense;
    # but if i run it on the whole benchmark, trigger and genre statistic can overlap...
    # maybe implement a way to count the by_genre, by_trigger separately for each phenomenon? 
    # for now i will leave it as it is, and later add a way to count it for phenomena separately

    return summary
