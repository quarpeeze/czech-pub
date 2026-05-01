from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


import json
import traceback

from dotenv import load_dotenv

from benchmark.core import run_benchmark
from benchmark.entities import Model

# load env vars from .env
load_dotenv()

EVAL_PATH = r"data\eval_general\eval\randomized\eval_czech_pub.jsonl"
DEV_PATH = r"data\eval_general\dev\randomized\dev_czech_pub.jsonl"

# for run_all_models, default to the full eval split
DATA_PATH = Path(EVAL_PATH)
OUT_DIR = Path(r"runs\eval\all_models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT_PATH = Path(r"prompts\system_prompt.txt")

with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()

# find evaluation process in notebooks/run_all_models

BENCHMARK_MODEL_LIST = [
    # baselines
    {"provider": "random_baseline", "model_name": "random_uniform", "tier": "baseline"},  # random baseline
    {"provider": "openai", "model_name": "gpt-5.4-nano", "tier": "baseline"},
    {"provider": "hf_local", "model_name": "Qwen/Qwen2.5-0.5B-Instruct", "tier": "baseline"},

    # closed - strong
    {"provider": "openai", "model_name": "gpt-5.4-mini", "tier": "strong_closed"},
    {"provider": "anthropic", "model_name": "claude-sonnet-4-6", "tier": "strong_closed"},
    {"provider": "google", "model_name": "gemini-2.5-flash", "tier": "strong_closed"},

    # closed - frontier
    {"provider": "openai", "model_name": "gpt-5.4", "tier": "frontier_closed"},
    {"provider": "anthropic", "model_name": "claude-opus-4-6", "tier": "frontier_closed"},
    {"provider": "google", "model_name": "gemini-2.5-pro", "tier": "frontier_closed"},

    # open - strong
    {"provider": "hf_local", "model_name": "google/gemma-3-4b-it", "tier": "local_open"},
    {"provider": "hf_local", "model_name": "Qwen/Qwen2.5-7B-Instruct", "tier": "local_open"},
    {"provider": "hf_local", "model_name": "meta-llama/Llama-3.1-8B-Instruct", "tier": "local_open"},
    {"provider": "hf_local", "model_name": "mistralai/Ministral-8B-Instruct-2410", "tier": "local_open"},
    {"provider": "hf_local", "model_name": "Qwen/Qwen2.5-14B-Instruct", "tier": "local_open"},
    {"provider": "hf_local", "model_name": "google/gemma-3-12b-it", "tier": "local_open"},

    # open - heavy
    {"provider": "hf_local", "model_name": "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "tier": "local_open_heavy"},
    {"provider": "hf_local", "model_name": "Qwen/Qwen2.5-32B-Instruct", "tier": "local_open_heavy"},
    # {"provider": "hf_local", "model_name": "meta-llama/Llama-3.3-70B-Instruct", "tier": "local_open_heavy"},
]


def select_models(
    models: list[dict] = BENCHMARK_MODEL_LIST,
    *,
    tier: str | None = None,
    provider: str | None = None,
) -> list[dict]:
    """
    selects a list of models to run, based on provider or tier
    e.g. run everything from OpenAI or run all from closed frontier models
    """
    selected = models

    if tier is not None:
        selected = [m for m in selected if m["tier"] == tier]

    if provider is not None:
        selected = [m for m in selected if m["provider"] == provider]

    return selected


def build_model(model_spec: dict) -> Model:
    """builds model from a short dict; max_tokens, temperature, top_p can be added as (k,v) pairs"""
    return Model(
        provider=model_spec["provider"],
        model_name=model_spec["model_name"],
        max_tokens=model_spec.get("max_tokens", 48),
        temperature=model_spec.get("temperature", 0.0),
        top_p=model_spec.get("top_p"),
        extra=model_spec.get("extra", {}),
    )


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_model_set(
    *,
    models: list[dict],
    data_path: str | Path,
    out_dir: str | Path = "runs",
    system_prompt: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """runs the predefined set of models on the given items dataset"""

    data_path = Path(data_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("starting benchmark run")
    print(f"dataset: {data_path}")
    print(f"out dir: {out_dir}")
    print(f"models to run: {len(models)}")
    print("=" * 100)

    summaries: list[dict] = []

    for idx, m in enumerate(models, start=1):
        print("\n" + "#" * 100)
        print(f"[model {idx}/{len(models)}]")
        print(f"provider: {m['provider']}")
        print(f"model:    {m['model_name']}")
        print(f"tier:     {m.get('tier')}")
        print("#" * 100)

        model = build_model(m)

        try:
            _, summary = run_benchmark(
                data_path=data_path,
                model=model,
                out_dir=out_dir,
                system_prompt=system_prompt,
                limit=limit,
            )

            record = {
                "provider": model.provider,
                "model_name": model.model_name,
                "tier": m.get("tier"),
                "status": "ok",
                "summary": summary,
            }

            print("\nmodel finished successfully")
            print(f"accuracy: {summary.get('accuracy')}")
            print(f"n_correct: {summary.get('n_correct')} / {summary.get('n_items')}")

        except Exception as e:
            record = {
                "provider": model.provider,
                "model_name": model.model_name,
                "tier": m.get("tier"),
                "status": "failed",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            }

            print("\nmodel failed")
            print(f"{type(e).__name__}: {e}")

        summaries.append(record)

        # save aggregate summaries after every model
        save_json(out_dir / "all_summaries.json", summaries)

        print("\naggregate summary file updated:")
        print(out_dir / "all_summaries.json")

    print("\n" + "=" * 100)
    print("all selected models finished")
    print("=" * 100)

    return summaries


def run_selected_models(
    *,
    data_path: str | Path,
    out_dir: str | Path = "runs/run_all",
    tier: str | None = None,
    provider: str | None = None,
    system_prompt: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """
    first select models, then run them.
    """
    models = select_models(
        tier=tier,
        provider=provider,
    )

    return run_model_set(
        models=models,
        data_path=data_path,
        out_dir=out_dir,
        system_prompt=system_prompt,
        limit=limit,
    )


def run_all() -> list[dict]:
    return run_selected_models(
        data_path=DATA_PATH,
        out_dir=OUT_DIR,
        system_prompt=SYSTEM_PROMPT,
        limit=None,
    )


if __name__ == "__main__":
    run_all()
