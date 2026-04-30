import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.helpers import load_dataset
from utils.prompt_builder import build_prompt

# define ins and outs
SOURCE_DIR = ROOT / "development" / "data" / "eval_general"
OUTPUT_DIR = ROOT / "demo" / "txt"
ITEM_SEPARATOR = "\n\n\n\n"


def get_correct_option(item: dict) -> dict:
    gold_label = item["gold_label"]

    for option in item["options"]:
        if option["label"] == gold_label:
            return option

    raise ValueError(f"could not find gold option for item {item.get('id')}")


def render_items(items: list[dict], title: str, include_answers: bool) -> str:
    rendered_items = []

    for index, item in enumerate(items, start=1):
        prompt = build_prompt(item).strip()
        lines = [
            f"{title} {index}",
            item["id"],
            "",
            prompt,
        ]

        if include_answers:
            correct_option = get_correct_option(item)
            lines.extend(
                [
                    "",
                    f"Correct answer: {correct_option['label']}",
                    correct_option["text"],
                ]
            )

        rendered_items.append("\n".join(lines))

    return ITEM_SEPARATOR.join(rendered_items) + "\n"


def write_rendered_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    # load the 600 presupposition items
    presupposition_items = load_dataset(
        SOURCE_DIR / "presupposition-merged-600-quotes-fixed.json"
    )

    # load the 600 implicature items as 500 indirect answers + 100 scalars
    implicature_items = load_dataset(
        SOURCE_DIR / "eval_final_circa_indirect_answer_500.json"
    )
    implicature_items += load_dataset(SOURCE_DIR / "implicature-scalar-eval-100.json")

    # load the 720 information structure items
    information_structure_items = load_dataset(SOURCE_DIR / "inf_structure_eval_720.json")

    write_rendered_file(
        OUTPUT_DIR / "raw" / "presupposition_600.txt",
        render_items(presupposition_items, "Item", include_answers=False),
    )
    write_rendered_file(
        OUTPUT_DIR / "raw" / "implicature_600.txt",
        render_items(implicature_items, "Item", include_answers=False),
    )
    write_rendered_file(
        OUTPUT_DIR / "raw" / "information_structure_720.txt",
        render_items(information_structure_items, "Item", include_answers=False),
    )

    write_rendered_file(
        OUTPUT_DIR / "with_answers" / "presupposition_600.txt",
        render_items(presupposition_items, "Item", include_answers=True),
    )
    write_rendered_file(
        OUTPUT_DIR / "with_answers" / "implicature_600.txt",
        render_items(implicature_items, "Item", include_answers=True),
    )
    write_rendered_file(
        OUTPUT_DIR / "with_answers" / "information_structure_720.txt",
        render_items(information_structure_items, "Item", include_answers=True),
    )

    print(f"wrote {len(presupposition_items)} presupposition items")
    print(f"wrote {len(implicature_items)} implicature items")
    print(f"wrote {len(information_structure_items)} information structure items")
    print(f"output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
