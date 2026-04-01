from __future__ import annotations

from collections import Counter

from benchmark.entities import Model


def _subset_metrics(rows: list[dict]) -> dict:
    parse_counts = Counter(r.get("parse_status") for r in rows)
    parsed_type_counts = Counter(r.get("parsed_type") or "__MISSING__" for r in rows)
    gold_type_counts = Counter(r.get("gold_type") or "__MISSING__" for r in rows)

    n_correct = sum(bool(r.get("is_correct")) for r in rows)
    n_items = len(rows)

    return {
        "n_items": n_items,
        "n_correct": n_correct,
        "accuracy": (n_correct / n_items) if n_items else None,
        "parse_status_counts": dict(parse_counts),
        "parse_ok_rate": (
            sum(r.get("parse_status") == "ok" for r in rows) / n_items
            if n_items else None
        ),
        "parsed_type_counts": dict(parsed_type_counts),
        "gold_type_counts": dict(gold_type_counts),
    }


def _safe_key(value) -> str:
    if value is None or value == "":
        return "__MISSING__"
    return str(value)


def _group_rows(rows: list[dict], key_fn) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}

    for row in rows:
        key = _safe_key(key_fn(row))
        groups.setdefault(key, []).append(row)

    return groups


def _group_metrics(rows: list[dict], key_fn) -> dict:
    groups = _group_rows(rows, key_fn)
    return {
        key: _subset_metrics(group_rows)
        for key, group_rows in groups.items()
    }


def _nested_group_metrics(rows: list[dict], outer_key_fn, inner_key_fn) -> dict:
    outer_groups = _group_rows(rows, outer_key_fn)

    return {
        outer_key: {
            "overall": _subset_metrics(outer_rows),
            "by_inner": _group_metrics(outer_rows, inner_key_fn),
        }
        for outer_key, outer_rows in outer_groups.items()
    }


def _item_meta(row: dict) -> dict:
    return row.get("item_meta", {})


def _metadata(row: dict) -> dict:
    return _item_meta(row).get("metadata", {})


def _phenomenon(row: dict):
    return _item_meta(row).get("phenomenon")


def _category(row: dict):
    return _item_meta(row).get("category")


def _creation_method(row: dict):
    return _item_meta(row).get("creation_method")


def _genre(row: dict):
    return _metadata(row).get("genre")


def _trigger(row: dict):
    return _metadata(row).get("trigger")


def _cue(row: dict):
    return _metadata(row).get("cue")


def _case(row: dict):
    return _metadata(row).get("case")


def _topic(row: dict):
    return _metadata(row).get("topic")


def _subtype(row: dict):
    return _metadata(row).get("subtype")


def _negation(row: dict):
    return _metadata(row).get("negation")


def _has_apposition(row: dict):
    return _metadata(row).get("has_apposition")


def _filter_rows(rows: list[dict], predicate) -> list[dict]:
    return [row for row in rows if predicate(row)]


def _is_scalar(row: dict) -> bool:
    category = _category(row)
    return isinstance(category, str) and category.startswith("scalar-")


def _is_scalar_frequency(row: dict) -> bool:
    return _category(row) == "scalar-frequency"


def _is_scalar_quantifier(row: dict) -> bool:
    return _category(row) == "scalar-quantifier"


def _information_structure_summary(rows: list[dict]) -> dict:
    return {
        "overall": _subset_metrics(rows),
        "by_cue": _group_metrics(rows, _cue),
        "by_case": _group_metrics(rows, _case),
        "by_cue_then_case": _nested_group_metrics(rows, _cue, _case),
    }


def _presupposition_summary(rows: list[dict]) -> dict:
    by_category_rows = _group_rows(rows, _category)

    category_then_genre = {
        category: {
            "overall": _subset_metrics(category_rows),
            "by_genre": _group_metrics(category_rows, _genre),
        }
        for category, category_rows in by_category_rows.items()
    }

    category_then_trigger = {
        category: {
            "overall": _subset_metrics(category_rows),
            "by_trigger": _group_metrics(category_rows, _trigger),
        }
        for category, category_rows in by_category_rows.items()
    }

    implicative_rows = by_category_rows.get("implicative", [])
    possessive_rows = by_category_rows.get("possessive", [])

    special_metadata = {}

    if implicative_rows:
        special_metadata["implicative_by_negation"] = _group_metrics(
            implicative_rows,
            _negation,
        )

    if possessive_rows:
        special_metadata["possessive_by_has_apposition"] = _group_metrics(
            possessive_rows,
            _has_apposition,
        )

    return {
        "overall": _subset_metrics(rows),
        "by_category": _group_metrics(rows, _category),
        "by_genre": _group_metrics(rows, _genre),
        "by_category_then_genre": category_then_genre,
        "by_category_then_trigger": category_then_trigger,
        "special_metadata": special_metadata,
    }


def _implicature_summary(rows: list[dict]) -> dict:
    indirect_rows = _filter_rows(rows, lambda r: _category(r) == "indirect-answer")
    scalar_rows = _filter_rows(rows, _is_scalar)
    frequency_rows = _filter_rows(rows, _is_scalar_frequency)
    quantifier_rows = _filter_rows(rows, _is_scalar_quantifier)

    grouped_views = {
        "indirect_answer": _subset_metrics(indirect_rows),
        "scalar_implicatures": _subset_metrics(scalar_rows),
        "frequency": _subset_metrics(frequency_rows),
        "quantifier": _subset_metrics(quantifier_rows),
    }

    out = {
        "overall": _subset_metrics(rows),
        "grouped_views": grouped_views,
        "indirect_answer": {
            "overall": _subset_metrics(indirect_rows),
            "by_topic": _group_metrics(indirect_rows, _topic),
        },
        "scalar_implicatures": {
            "overall": _subset_metrics(scalar_rows),
            "by_subtype": _group_metrics(scalar_rows, _subtype),
        },
        "scalar_frequency": {
            "overall": _subset_metrics(frequency_rows),
            "by_subtype": _group_metrics(frequency_rows, _subtype),
        },
        "scalar_quantifier": {
            "overall": _subset_metrics(quantifier_rows),
            "by_subtype": _group_metrics(quantifier_rows, _subtype),
        },
    }

    return out


def make_summary(results: list[dict], model: Model) -> dict:
    overall = _subset_metrics(results)

    summary = {
        **overall,
        "provider": model.provider,
        "model_name": model.model_name,
        "n_model_errors": sum(r.get("parse_status") == "model_error" for r in results),
        "n_parse_invalid": sum(r.get("parse_status") == "invalid" for r in results),
        "by_phenomenon": _group_metrics(results, _phenomenon),
        "phenomenon_details": {},
    }

    phenomenon_rows = _group_rows(results, _phenomenon)

    information_structure_rows = phenomenon_rows.get("information_structure", [])
    if information_structure_rows:
        summary["phenomenon_details"]["information_structure"] = _information_structure_summary(
            information_structure_rows
        )

    presupposition_rows = phenomenon_rows.get("presupposition", [])
    if presupposition_rows:
        summary["phenomenon_details"]["presupposition"] = _presupposition_summary(
            presupposition_rows
        )

    implicature_rows = phenomenon_rows.get("implicature", [])
    if implicature_rows:
        summary["phenomenon_details"]["implicature"] = _implicature_summary(
            implicature_rows
        )

    return summary