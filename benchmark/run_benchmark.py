from __future__ import annotations

from pathlib import Path
import argparse
import json

from evaluation.evaluate import make_summary
from evaluation.parser import parse_mcq_answer
from benchmark.model_client import Model, generate
from utils.helpers import (
    get_gold_label,
    get_valid_labels,
    load_dataset,
    save_json,
    save_jsonl,
)
from utils.prompt_builder import build_prompt

def _safe_model_dir_name(model: Model) -> str:
    """
    create a safe directory name for a model (replacing slashes)

    example:
    - Qwen/Qwen2.5-3B-Instruct -> Qwen__Qwen2.5-3B-Instruct
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

def run_benchmark(
    *,
    data_path: str | Path,
    model: Model,
    out_dir: str | Path = "runs",
    system_prompt: str | None = None,
    limit: int | None = None,
) -> tuple[list[dict], dict]:
    """
    run one model on one benchmark dataset, sequentially.

    for each item:
    - build prompt
    - generate model output
    - parse answer
    - compare to the correct option
    - store result

    save to out_dir:
    - predictions.jsonl
    - summary.json
    """
    items = load_dataset(data_path)

    if limit is not None:
        items = items[:limit]

    results: list[dict] = []

    for i, item in enumerate(items, start=1):
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

            row = {
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
            row = {
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

        results.append(row)

        print(
            f"[{i}/{len(items)}] "
            f"id={item_id} "
            f"parse={row['parse_status']} "
            f"pred={row['parsed_label']} "
            f"gold={row['gold_label']} "
            f"correct={row['is_correct']}"
        )

    summary = make_summary(results, model)

    # make the run dir inside of the out directory with the name of the used model
    run_dir = Path(out_dir) / _safe_model_dir_name(model)
    run_dir.mkdir(parents=True, exist_ok=True)

    # save predictions and summary to run_dir
    save_jsonl(run_dir / "predictions.jsonl", results)
    save_json(run_dir / "summary.json", summary)

    return results, summary



def main() -> None:
    argparser = argparse.ArgumentParser(description="run a single model on a single benchmark dataset")

    argparser.add_argument(
        "--data",
        required=True,
        help="path to benchmark dataset (.json or .jsonl).",
    )
    argparser.add_argument(
        "--provider",
        required=True,
        help="model provider (e.g. openai, anthropic, google, hf_local)",
    )
    argparser.add_argument(
        "--model",
        required=True,
        help="model name, e.g. gpt-4.1-mini or Qwen/Qwen2.5-3B-Instruct",
    )
    argparser.add_argument(
        "--outdir",
        default="runs",
        help="root directory where run subdirectories will be created.",
    )
    argparser.add_argument(
        "--system-prompt",
        default=None,
        help="system prompt string (or path to .txt file)",
    )
    argparser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="limit on number of items to run.",
    )
    argparser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="maximum number of output tokens",
    )
    argparser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="sampling temperature",
    )
    argparser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="optional top-p value",
    )
    argparser.add_argument(
        "--extra",
        default=None,
        help='optional json string for provider-specific extra params',
    )

    args = argparser.parse_args()

    system_prompt = args.system_prompt
    if system_prompt is not None:
        possible_path = Path(system_prompt)
        if possible_path.exists() and possible_path.is_file():
            system_prompt = possible_path.read_text(encoding="utf-8")

    extra = {}
    if args.extra:
        extra = json.loads(args.extra)

    model = Model(
        provider=args.provider,
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        extra=extra,
    )

    _, summary = run_benchmark(
        data_path=args.data,
        model=model,
        out_dir=args.outdir,
        system_prompt=system_prompt,
        limit=args.limit,
    )

    print("\nfinished running the benchmark.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()