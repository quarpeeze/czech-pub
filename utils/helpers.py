from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """
    load dataset from .json or .jsonl.

    supported shapes:
    - JSON list of items
    - JSON object with key "items" containing a list
    - JSONL with one item per line
    """
    path = Path(path)

    if path.suffix.lower() == ".jsonl":
        items: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        return data["items"]

    raise ValueError("Dataset must be either a list or a dict with key 'items'.")


def save_json(path: str | Path, obj: Any) -> None:
    """save object as formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """save list of dicts as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def normalize(s: str) -> str:
    """normalize any string."""
    return " ".join(str(s).strip().split()).lower()


def get_valid_labels(item: dict[str, Any]) -> list[str]:
    """
    extract option labels from item["options"].
    works according to the unified schema in /item_schema.json

    expects options like:
    [{"label": "A", ...}, {"label": "B", ...}]
    """
    options = item.get("options", [])
    labels: list[str] = []

    for option in options:
        label = option.get("label")
        if not label:
            raise ValueError(f"Item {item.get('id')} has an option without 'label'.")
        labels.append(str(label).strip())

    if not labels:
        raise ValueError(f"Item {item.get('id')} has no options.")

    return labels


def get_gold_label(item: dict[str, Any]) -> str | None:
    """
    extract gold label (correct answer) from the item

    currently gold label key name according to the item_schema.json is: "gold_label"
    """
    #TODO pozdeji zmenit tohle
    key = "gold_label"
    value = item.get(key)
    if value:
        return str(value).strip()
    return None