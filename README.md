# CZECH-PUB

CZECH-PUB is a benchmark for evaluating pragmatic reasoning in Czech. This repository contains the benchmark dataset, reusable benchmark code, evaluation utilities, and demo materials.

The benchmark covers three pragmatic phenomena:

- Implicature
- Presupposition
- Information Structure

The task is formulated as multiple-choice question answering. Each item contains a context, an utterance or other prompt-relevant input, answer options, the correct answer, and metadata describing the evaluated phenomenon and category.

## What CZECH-PUB Evaluates

CZECH-PUB tests whether a model can recover context-dependent meaning in Czech. The benchmark includes:

- **Implicature**: indirect answers, relevance-based inferences, and scalar implicatures.
- **Presupposition**: inferences triggered by lexical or structural cues.
- **Information Structure**: discourse-sensitive Czech word order, especially topic-focus structure.

## Dataset Construction

The dataset was assembled from three sources and construction strategies:

- **Presupposition** items were collected from Czech SYN2025 corpus data using trigger-based patterns. The items are based on Czech presupposition triggers and were manually filtered and revised into multiple-choice benchmark questions.
- **Implicature** items include indirect-answer and relevance implicatures translated from CIRCA into Czech, with pragmatic localization and manual review. The implicature subset also includes scalar implicatures mined from OpenSubtitles2023 and SYN2025.
- **Information Structure** items were created through controlled generation from simple Czech sentence structures with help of an LLM. These items test whether a model can select the appropriate Czech word order for the discourse context.

The final benchmark contains **1,920 items** across the three main phenomena.

## Data

- `data/eval` - Main evaluation set consisting of 1,920 items.
- `data/dev` - Sample of 180 items for human evaluation.


The repository also includes [`item_schema.json`](item_schema.json), which documents the item fields used in the benchmark format.

## Repository Structure

The main repository layout is:

- `benchmark/` - Core benchmark code, model dispatch, and runnable benchmark scripts.
- `evaluation/` - Answer parsing, metric aggregation, and human-readable summary generation.
- `utils/` - Dataset IO and prompt-building helpers.
- `prompts/` - Prompt templates for the three benchmark phenomena + the shared system prompt.
- `data/` - Main benchmark datasets.
- `demo/` - Provides an overview of the CZECH-PUB dataset and step-by-step demo of evaluation pipeline.
- `tests/` - Tests.


## Demo

The `demo/` folder contains:

- `demo/dataset_overview.ipynb` - Quick inspection of the merged `eval` and `dev` datasets, with simple counts and prompt-formatted examples.
- `demo/demo_run_benchmark.ipynb` - Step-by-step notebook showing the benchmark flow: load items, inspect a prompt, run a model, build the summary, and create a readable analysis.
- `demo/render_to_txt.py` - Script for rendering benchmark items into plain-text test sheets (human-readable format).
- `demo/txt/raw/` - Rendered text files without answers.
- `demo/txt/with_answers/` - Rendered text files with answer keys.

These materials are useful for quickly inspecting benchmark items and trying the benchmark workflow before running a larger experiment.

## Running the Benchmark

The reusable benchmark flow is implemented in `benchmark/core.py` and exposed through the CLI script `benchmark/run_benchmark.py`.


### Minimal CLI Run

The smallest command to run the benchmark requires only the dataset path, provider, and model name. Default system prompt is taken from `prompts/system_prompt.txt`.

Use the commands below to run benchmark evaluation on a model loaded from HugingFace.
If `python` is not available on your system, use `python3` instead.

```console
python benchmark/run_benchmark.py --data data/dev/dev_czech_pub.json --provider hf_local --model Qwen/Qwen2.5-0.5B-Instruct
```

Useful optional flags:

```console
python benchmark/run_benchmark.py --data data/dev/dev_czech_pub.json --provider hf_local --model Qwen/Qwen2.5-0.5B-Instruct --outdir runs/test --limit 10 --max-tokens 16 --temperature 0.0
```

Another HuggingFace example:

```console
python benchmark/run_benchmark.py --data data/eval/eval_czech_pub.json --provider hf_local --model google/gemma-3-4b-it --limit 20
```

### Python Usage

The same workflow can be used directly from Python:

```python
from benchmark.core import load_benchmark_items, run_items, save_run_outputs
from benchmark.entities import Model
from evaluation.evaluate import make_summary
from utils.prompt_builder import load_system_prompt

items = load_benchmark_items("data/dev/dev_czech_pub.json", limit=10)

model = Model(
    provider="hf_local",
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    max_tokens=16,
    temperature=0.0,
)

results = run_items(
    items=items,
    model=model,
    system_prompt=load_system_prompt(),
)

summary = make_summary(results, model)
run_dir = save_run_outputs(results, summary, model, out_dir="runs")
```

### Multi-Model Script

The repository also includes `benchmark/run_all_models.py`, which defines a predefined model list and runs multiple benchmark configurations in sequence.

## Data Flow

The actual benchmark flow is:

1. Load benchmark items from `data/dev` or `data/eval`.
2. Build a phenomenon-specific prompt for each item using `utils/prompt_builder.py`.
3. Run the selected model through `benchmark/model_client.py`.
4. Parse the model answer with `evaluation/parser.py`.
5. Aggregate prediction rows into a summary with `evaluation/evaluate.py`.
6. Convert the summary into a short readable text report with `evaluation/analyze.py`.
7. Write outputs to the run directory.

By default, benchmark outputs are written under `runs/`. Each run is stored in:

- `runs/<safe_model_name>/predictions.jsonl`
- `runs/<safe_model_name>/summary.json`
- `runs/<safe_model_name>/analysis.txt`


## Evaluation Utilities

The evaluation utilities can also be used independently:

- `evaluation/evaluate.py` builds a structured summary from benchmark prediction rows (predictions.jsonl).
- `evaluation/analyze.py` turns a summary JSON file into a short human-readable text report.


## Prompting

- Prompt templates are stored in `prompts/`.
- The shared system prompt is stored in `prompts/system_prompt.txt`.
- Benchmark item rendering is handled by `utils/prompt_builder.py`.


Author: Yevhenii Karpizenkov