from __future__ import annotations

from pathlib import Path

from evaluation.evaluate import make_summary
from evaluation.parser import parse_mcq_answer
from benchmark.model_client import generate
from benchmark.entities import Model
from utils.helpers import (
    get_gold_label,
    get_valid_labels,
    load_dataset,
    save_json,
    save_jsonl,
)
from utils.prompt_builder import build_prompt


def safe_model_dir_name(model: Model) -> str:
    """
    create a safe directory name for a model name.
    """
    return model.model_name.replace("/", "__").replace("\\", "__").strip()


def extract_item_metadata(item: dict) -> dict:
    return {
        "creation_method": item.get("creation_method"),
        "phenomenon": item.get("phenomenon"),
        "category": item.get("category"),
        "metadata": item.get("metadata", {}),
    }


def get_option_type_by_label(item: dict, label: str | None) -> str | None:
    if label is None:
        return None

    for option in item.get("options", []):
        if option.get("label") == label:
            return option.get("type")

    return None


def load_benchmark_items(
    data_path: str | Path,
    limit: int | None = None,
) -> list[dict]:
    items = load_dataset(data_path)

    if limit is not None:
        items = items[:limit]

    return items


def run_single_item(
    item: dict,
    model: Model,
    system_prompt: str | None = None,
) -> dict:
    item_id = item.get("id")
    item_meta = extract_item_metadata(item)
    valid_labels = get_valid_labels(item)
    gold_label = get_gold_label(item)
    gold_type = get_option_type_by_label(item, gold_label)
    prompt = build_prompt(item)

    try:
        gen_result = generate(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
        )

        parse_result = parse_mcq_answer(gen_result.text, valid_labels)
        parsed_type = get_option_type_by_label(item, parse_result.label)

        is_correct = None
        if gold_label is not None:
            is_correct = parse_result.label == gold_label

        return {
            "id": item_id,
            "item_meta": item_meta,
            "provider": gen_result.provider,
            "model_name": gen_result.model_name,
            "prompt": prompt,
            "raw_output": gen_result.text,
            "finish_reason": gen_result.finish_reason,
            "usage_prompt_tokens": gen_result.usage_prompt_tokens,
            "usage_completion_tokens": gen_result.usage_completion_tokens,
            "parsed_label": parse_result.label,
            "parsed_type": parsed_type,
            "parse_status": parse_result.status,
            "parse_matched_text": parse_result.matched_text,
            "gold_label": gold_label,
            "gold_type": gold_type,
            "is_correct": is_correct,
        }

    except Exception as e:
        return {
            "id": item_id,
            "item_meta": item_meta,
            "provider": model.provider,
            "model_name": model.model_name,
            "prompt": prompt,
            "raw_output": None,
            "finish_reason": "error",
            "usage_prompt_tokens": None,
            "usage_completion_tokens": None,
            "parsed_label": None,
            "parsed_type": None,
            "parse_status": "model_error",
            "parse_matched_text": None,
            "gold_label": gold_label,
            "gold_type": gold_type,
            "is_correct": False,
            "error": f"{type(e).__name__}: {e}",
        }


def run_items(
    items: list[dict],
    model: Model,
    system_prompt: str | None = None,
) -> list[dict]:
    results: list[dict] = []

    for i, item in enumerate(items, start=1):
        row = run_single_item(
            item=item,
            model=model,
            system_prompt=system_prompt,
        )
        results.append(row)

        print(
            f"[{i}/{len(items)}] "
            f"id={row['id']} "
            f"parse={row['parse_status']} "
            f"pred={row['parsed_label']} "
            f"gold={row['gold_label']} "
            f"correct={row['is_correct']}"
        )

    return results


def save_run_outputs(
    results: list[dict],
    summary: dict,
    model: Model,
    out_dir: str | Path = "runs",
) -> Path:
    run_dir = Path(out_dir) / safe_model_dir_name(model)
    run_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(run_dir / "predictions.jsonl", results)
    save_json(run_dir / "summary.json", summary)

    return run_dir


def run_benchmark(
    *,
    data_path: str | Path,
    model: Model,
    out_dir: str | Path = "runs",
    system_prompt: str | None = None,
    limit: int | None = None,
) -> tuple[list[dict], dict]:
    """
    run one model on one benchmark dataset and save predictions plus summary.
    """
    items = load_benchmark_items(
        data_path=data_path,
        limit=limit,
    )
    results = run_items(
        items=items,
        model=model,
        system_prompt=system_prompt,
    )
    summary = make_summary(results, model)
    save_run_outputs(
        results=results,
        summary=summary,
        model=model,
        out_dir=out_dir,
    )

    return results, summary
