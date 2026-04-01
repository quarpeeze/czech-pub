from pathlib import Path
from typing import Any


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

def load_prompt_template(phenomenon: str) -> str:
    path = PROMPTS_DIR / f"{phenomenon}.txt"

    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")

    return path.read_text(encoding="utf-8").strip()


def format_options_block(options: list[dict[str, str]]) -> str:
    lines = []

    for option in options:
        label = option["label"]
        text = option["text"]
        lines.append(f"{label}. {text}")

    return "\n".join(lines)


def prepare_prompt_vars(item: dict[str, Any]) -> dict[str, str]:
    required = ["context", "utterance", "options", "phenomenon"]
    for key in required:
        if key not in item:
            raise KeyError(f"missing required item field: '{key}'") # check missing

    meta_general = item.get("meta_general", {})
    meta_special = item.get("meta_special", {})

    return {
        "context": item["context"],
        "utterance": item["utterance"],
        "options": format_options_block(item["options"]),
        "phenomenon": item["phenomenon"],
        "trigger": meta_special.get("trigger", ""),
        "genre": meta_general.get("genre", ""),
        "corpus": meta_general.get("corpus", ""),
    }


def build_prompt(item: dict[str, Any]) -> str:
    prompt_vars = prepare_prompt_vars(item)
    phenomenon = item["phenomenon"]

    template = load_prompt_template(phenomenon)

    try:
        return template.format(**prompt_vars)
    except KeyError as e:
        missing = e.args[0]
        raise KeyError(
            f"missing variable '{missing}' while rendering "
            f"prompt for phenomenon '{phenomenon}'"
        ) from e