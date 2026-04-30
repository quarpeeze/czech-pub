from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def format_accuracy(metrics: dict) -> str:
    accuracy = metrics.get("accuracy")
    n_correct = metrics.get("n_correct")
    n_items = metrics.get("n_items")

    if accuracy is None:
        return f"n/a ({n_correct}/{n_items})"

    return f"{accuracy:.1%} ({n_correct}/{n_items})"


def make_group_lines(groups: dict, label: str) -> list[str]:
    lines = [label]

    for name, metrics in groups.items():
        lines.append(f"- {name}: {format_accuracy(metrics)}")

    return lines


def build_report(summary: dict) -> str:
    lines: list[str] = []

    provider = summary.get("provider", "unknown")
    model_name = summary.get("model_name", "unknown")

    lines.append("benchmark analysis")
    lines.append("")
    lines.append(f"model: {model_name}")
    lines.append(f"provider: {provider}")
    lines.append(f"overall accuracy: {format_accuracy(summary)}")
    lines.append("")

    by_phenomenon = summary.get("by_phenomenon", {})
    if by_phenomenon:
        lines.extend(make_group_lines(by_phenomenon, "accuracy by phenomenon:"))
        lines.append("")

    details = summary.get("phenomenon_details", {})

    presupposition = details.get("presupposition", {})
    if presupposition:
        lines.append("presupposition breakdown:")
        lines.append(
            f"- overall: {format_accuracy(presupposition.get('overall', {}))}"
        )
        if presupposition.get("by_category"):
            lines.extend(make_group_lines(presupposition["by_category"], "by category:"))
        if presupposition.get("by_genre"):
            lines.extend(make_group_lines(presupposition["by_genre"], "by genre:"))
        lines.append("")

    implicature = details.get("implicature", {})
    if implicature:
        lines.append("implicature breakdown:")
        lines.append(f"- overall: {format_accuracy(implicature.get('overall', {}))}")

        grouped_views = implicature.get("grouped_views", {})
        if grouped_views:
            lines.append("- grouped views:")
            for name, metrics in grouped_views.items():
                lines.append(f"  {name}: {format_accuracy(metrics)}")

        indirect_answer = implicature.get("indirect_answer", {})
        if indirect_answer.get("by_topic"):
            lines.append("- indirect answer by topic:")
            for name, metrics in indirect_answer["by_topic"].items():
                lines.append(f"  {name}: {format_accuracy(metrics)}")

        scalar_implicatures = implicature.get("scalar_implicatures", {})
        if scalar_implicatures.get("by_subtype"):
            lines.append("- scalar implicatures by subtype:")
            for name, metrics in scalar_implicatures["by_subtype"].items():
                lines.append(f"  {name}: {format_accuracy(metrics)}")
        lines.append("")

    information_structure = details.get("information_structure", {})
    if information_structure:
        lines.append("information structure breakdown:")
        lines.append(
            f"- overall: {format_accuracy(information_structure.get('overall', {}))}"
        )
        if information_structure.get("by_cue"):
            lines.append("- by cue:")
            for name, metrics in information_structure["by_cue"].items():
                lines.append(f"  {name}: {format_accuracy(metrics)}")
        if information_structure.get("by_case"):
            lines.append("- by case:")
            for name, metrics in information_structure["by_case"].items():
                lines.append(f"  {name}: {format_accuracy(metrics)}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="turn an evaluation summary json into a short human-readable txt report"
    )
    parser.add_argument("--input", required=True, help="path to summary.json")
    parser.add_argument("--output", required=True, help="path to output .txt file")
    args = parser.parse_args()

    summary = load_summary(args.input)
    report = build_report(summary)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print(f"wrote analysis to {output_path}")


if __name__ == "__main__":
    main()
