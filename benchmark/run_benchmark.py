from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.core import run_benchmark
from benchmark.entities import Model
from utils.prompt_builder import load_system_prompt


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
        default=load_system_prompt(),
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
