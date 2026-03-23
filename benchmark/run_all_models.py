from __future__ import annotations

from pathlib import Path

from benchmark.model_client import Model
from benchmark.run_benchmark import run_benchmark

# find evaluation process in notebooks/run_all_models

BENCHMARK_MODEL_LIST = [
    # baselines
    {"provider": "random_baseline", "model_name": "random_uniform", "tier": "baseline"}, # random baseline
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
    {"provider": "hf_local", "model_name": "Qwen/Qwen2.5-32B-Instruct", "tier": "local_open_heavy"}
    #{"provider": "hf_local", "model_name": "meta-llama/Llama-3.3-70B-Instruct", "tier": "local_open_heavy"},
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
    """ builds model from a short dict; max_tokens, temperature, top_p can be added as (k,v) pairs"""
    return Model(
        provider=model_spec["provider"],
        model_name=model_spec["model_name"],
        max_tokens=model_spec.get("max_tokens", 64),
        temperature=model_spec.get("temperature", 0.0),
        top_p=model_spec.get("top_p"),
        extra=model_spec.get("extra", {}),
    )


def run_model_set(
    *,
    models: list[dict],
    data_path: str | Path,
    out_dir: str | Path = "runs",
    system_prompt: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """ runs the predefined set of models on the given items dataset"""

    summaries = []

    for m in models:
        model = build_model(m)

        print(f"running: {model.provider} | {model.model_name}")

        run_results, summary = run_benchmark(
            data_path=Path(data_path),
            model=model,
            out_dir=Path(out_dir),
            system_prompt=system_prompt,
            limit=limit,
        )

        summaries.append(
            {
                "provider": model.provider,
                "model_name": model.model_name,
                "tier": m.get("tier"),
                "summary": summary,
            }
        )

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