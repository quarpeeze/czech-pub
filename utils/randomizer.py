import json
import random
from copy import deepcopy
from collections import Counter

LABELS = ["A", "B", "C", "D"]  # labels of mcqa answer options (always 4)


def validate_item(item: dict, correct_label_field: str = "gold_label") -> list[str]:
    """
    validate one item, return a list of errors
    empty list = all good
    """
    errors = []

    item_id = item.get("id", "<missing-id>")

    # required top-level fields
    for field in ["id", "options", correct_label_field]:
        if field not in item:
            errors.append(f"Item '{item_id}': missing required field '{field}'.")

    if "options" not in item:
        return errors

    options = item["options"]

    if not isinstance(options, list):
        errors.append(f"Item '{item_id}': 'options' must be a list.")
        return errors

    if len(options) != 4:
        errors.append(f"Item '{item_id}': expected exactly 4 options, got {len(options)}.")
        return errors

    labels = []
    type_text_pairs = []

    for i, opt in enumerate(options):
        if not isinstance(opt, dict):
            errors.append(f"Item '{item_id}': option at index {i} is not a dict.")
            continue

        for key in ["label", "type", "text"]:
            if key not in opt:
                errors.append(f"Item '{item_id}': option at index {i} missing field '{key}'.")

        if "label" in opt:
            labels.append(opt["label"])

        if "type" in opt and "text" in opt:
            type_text_pairs.append((opt["type"], opt["text"]))

    # labels must be unique
    if len(labels) != len(set(labels)):
        errors.append(f"Item '{item_id}': option labels are not unique: {labels}")

    # correct label must exist among option labels
    if correct_label_field in item and item[correct_label_field] not in labels:
        errors.append(
            f"Item '{item_id}': {correct_label_field}='{item[correct_label_field]}' "
            f"not found among option labels {labels}."
        )

    # matching later depends on unique (type, text)
    if len(type_text_pairs) != len(set(type_text_pairs)):
        errors.append(
            f"Item '{item_id}': duplicate (type, text) pairs found in options; "
            f"correct option recovery would be ambiguous."
        )

    return errors


def validate_dataset(items: list[dict], correct_label_field: str = "gold_label") -> list[str]:
    """
    validate the full dataset, return a flat list of errors
    empty list = dataset is valid
    """
    errors = []

    if not isinstance(items, list):
        return ["Input dataset must be a list of items."]

    ids = []

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"Dataset item at index {idx} is not a dict.")
            continue

        errors.extend(validate_item(item, correct_label_field=correct_label_field))

        if "id" in item:
            ids.append(item["id"])

    id_counts = Counter(ids)
    duplicate_ids = [item_id for item_id, count in id_counts.items() if count > 1]
    for dup_id in duplicate_ids:
        errors.append(f"Duplicate item id found: '{dup_id}'.")

    return errors


def shuffle_dataset_items(items: list[dict], master_seed: int) -> list[dict]:
    """
    shuffle dataset item order reproducibly with the master seed
    """
    items = deepcopy(items)
    rng = random.Random(master_seed)
    rng.shuffle(items)
    return items


def shuffle_options_in_item(
    item: dict,
    master_seed: int,
    correct_label_field: str = "gold_label",
) -> dict:
    """
    shuffle options reproducibly using the master seed and item id
    """
    item = deepcopy(item)

    old_correct_label = item[correct_label_field]
    old_options = item["options"]

    # find the original correct option before shuffling
    original_correct = next(
        (opt for opt in old_options if opt["label"] == old_correct_label),
        None
    )

    if original_correct is None:
        raise ValueError(
            f"Item '{item['id']}': {correct_label_field}='{old_correct_label}' does not match any option."
        )

    # shuffle reproducibly using master seed + stable item id
    rng = random.Random(f"{master_seed}::{item['id']}")
    shuffled_options = deepcopy(old_options)
    rng.shuffle(shuffled_options)

    # relabel shuffled options to A/B/C/D
    for i, opt in enumerate(shuffled_options):
        opt["label"] = LABELS[i]

    # find where the original correct option ended up
    new_correct_label = None
    for opt in shuffled_options:
        if (
            opt["type"] == original_correct["type"]
            and opt["text"] == original_correct["text"]
        ):
            new_correct_label = opt["label"]
            break

    if new_correct_label is None:
        raise ValueError(
            f"Item '{item['id']}': could not recover the correct option after shuffling."
        )

    item["options"] = shuffled_options
    item[correct_label_field] = new_correct_label

    return item


def randomize_dataset(
    items: list[dict],
    master_seed: int,
    correct_label_field: str = "gold_label",
) -> list[dict]:
    """
    1. shuffle options inside each item
    2. shuffle the dataset items themselves
    """
    randomized_items = [
        shuffle_options_in_item(item, master_seed, correct_label_field=correct_label_field)
        for item in items
    ]

    randomized_items = shuffle_dataset_items(randomized_items, master_seed)
    return randomized_items


def verify_randomized_dataset(
    original_items: list[dict],
    randomized_items: list[dict],
    correct_label_field: str = "gold_label",
) -> list[str]:
    """
    check that randomization kept content intact
    return a list of problems, empty list = looks good
    """
    problems = []

    if len(original_items) != len(randomized_items):
        problems.append(
            f"Dataset size changed: original={len(original_items)}, randomized={len(randomized_items)}."
        )
        return problems

    original_by_id = {item["id"]: item for item in original_items}
    randomized_by_id = {item["id"]: item for item in randomized_items}

    if set(original_by_id.keys()) != set(randomized_by_id.keys()):
        problems.append("Set of item IDs changed after randomization.")
        return problems

    for item_id in original_by_id:
        original = original_by_id[item_id]
        randomized = randomized_by_id[item_id]

        # all option contents should stay the same, ignoring displayed labels/order
        original_pairs = sorted((opt["type"], opt["text"]) for opt in original["options"])
        randomized_pairs = sorted((opt["type"], opt["text"]) for opt in randomized["options"])

        if original_pairs != randomized_pairs:
            problems.append(f"Item '{item_id}': option contents changed after randomization.")

        original_gold = next(
            (opt for opt in original["options"] if opt["label"] == original[correct_label_field]),
            None
        )
        randomized_gold = next(
            (opt for opt in randomized["options"] if opt["label"] == randomized[correct_label_field]),
            None
        )

        if original_gold is None:
            problems.append(f"Item '{item_id}': original gold option missing.")
            continue

        if randomized_gold is None:
            problems.append(f"Item '{item_id}': randomized gold option missing.")
            continue

        if (
            original_gold["type"] != randomized_gold["type"]
            or original_gold["text"] != randomized_gold["text"]
        ):
            problems.append(f"Item '{item_id}': gold option meaning changed after randomization.")

    return problems


def export_jsonl(items: list[dict], output_path: str) -> None:
    """
    export dataset as jsonl, one item per line
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    input_path = ""
    output_path = ""
    master_seed = 42
    correct_label_field = "gold_label"

    with open(input_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list):
        raise ValueError("Input JSON must contain a list of dataset items.")

    errors = validate_dataset(items, correct_label_field=correct_label_field)
    if errors:
        raise ValueError(
            "Dataset validation failed. First errors:\n" + "\n".join(errors[:20])
        )

    randomized_items = randomize_dataset(
        items,
        master_seed=master_seed,
        correct_label_field=correct_label_field,
    )

    problems = verify_randomized_dataset(
        items,
        randomized_items,
        correct_label_field=correct_label_field,
    )
    if problems:
        raise ValueError(
            "Post-randomization verification failed. First problems:\n" + "\n".join(problems[:20])
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(randomized_items, f, ensure_ascii=False, indent=2)

    print(f"Saved randomized dataset to: {output_path}")
    print(f"Master seed: {master_seed}")


if __name__ == "__main__":
    main()