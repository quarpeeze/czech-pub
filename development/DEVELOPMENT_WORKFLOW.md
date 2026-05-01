# Development Workflow

This document gives a practical overview of the `development/` directory and how it was used while building the benchmark data.

## Directory Overview

### `data`

`development/data` contains benchmark data in multiple intermediate and final versions, stored in `.json` format. The directory is organized by phenomenon and includes merged evaluation files in `development/data/eval_general/`.

### `notebooks`

`development/notebooks` contains:

- `llm_prompting/` with test queries to LLM providers.
- `data_processing/` with Python notebooks that operate on raw `.json` data and help unify the data format across phenomena into a single format ready for evaluation.

### `run_all_models.py`

`development/run_all_models.py` is the script that was used to evaluate multiple models on the benchmark.

### `runs`

`development/runs` stores benchmark outputs for model runs. It contains model predictions and structured summaries for the `dev` and `eval` splits.

### `results`

`development/results` stores corresponding human-readable text summaries derived from the evaluation summaries in `development/runs`.

## Workflow by Phenomenon

### Implicatures

- `development/notebooks/data_processing/circa-translation.ipynb`
  processes the CIRCA dataset, samples it, applies filtering steps, and manages the data through the translation and adaptation pipeline. This process was only partially handled inside the repository, so it is not fully end-to-end in Python.

- `development/notebooks/data_processing/process_scalars_to_eval.ipynb`
  processes mined scalar implicature data into the unified benchmark format.

Relevant data folders:

- `development/data/implicature/circa_google/`
- `development/data/implicature/scalar/`
- `development/data/implicature/eval/`

### Information Structure

- `development/notebooks/data_processing/information_structure_processing.ipynb`
  demonstrates how generated scenarios are expanded into full benchmark items. The practical flow is `generated -> master -> eval`.

Relevant data folders:

- `development/data/information-structure/generated/`
- `development/data/information-structure/master/`
- `development/data/information-structure/eval/`

### Presupposition

- `development/notebooks/data_processing/presuppositions_processing.ipynb`
  helps process raw parsed presupposition data from the `initial -> eval` state.

Relevant data folders:

- `development/data/presupposition/initial/`
- `development/data/presupposition/eval/`

## Helper Notebook

- `development/notebooks/data_processing/schema_and_json_converters.ipynb`
  contains helper code for schema handling and JSON conversion.

## Unified Evaluation Data

The merged evaluation data used for final benchmark assembly is stored in:

- `development/data/eval_general/dev/`
- `development/data/eval_general/eval/`

These folders contain merged files across phenomena, including randomized and not-randomized variants as well as phenomenon-specific source files used during assembly.

## Important Disclaimer

The Python code in `development/` does not fully cover the complete data creation process.

A substantial part of the benchmark creation was handled manually, especially during corpus mining, filtering, revision, translation/adaptation decisions, and item validation. The notebooks and data files in `development/` should therefore be read as a practical record of how the data was processed and unified across phenomena, not as a complete end-to-end reproduction of every creation step.
