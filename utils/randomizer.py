import json
import random
from copy import deepcopy

LABELS = ["A", "B", "C", "D"] # labels of mcqa answer options (always 4)

def shuffle_dataset_items(items: list[dict], master_seed: int) -> list[dict]:
    """
    shuffle the order of dataset items reproducibly using the master seed
    """
    items = deepcopy(items)
    rng = random.Random(master_seed)
    rng.shuffle(items)
    return items

def shuffle_options_in_item(item: dict, master_seed: int) -> dict:
    """
    shuffle options using the main seed (masterseed)
    """
    item = deepcopy(item)

    old_correct_label = item["correct_option"]
    old_options = item["options"]

    # find the original correct option before shuffling
    original_correct = next(
        (opt for opt in old_options if opt["label"] == old_correct_label),
        None
    )

    if original_correct is None:
        raise ValueError(f"item '{item['id']}': correct_option='{old_correct_label}' is none ")

    # shuffle options, currently i pass masterseed + id of the item (since random.Random accepts strs also)
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
        raise ValueError(f"Item '{item['id']}': could not recover the correct option after shuffling.")

    item["options"] = shuffled_options
    item["correct_option"] = new_correct_label

    return item


def randomize_dataset(items: list[dict], master_seed: int) -> list[dict]:
    """
    1. shuffle options within each item
    2. shuffle the dataset items themselves
    """
    randomized_items = [
        shuffle_options_in_item(item, master_seed)
        for item in items
    ]

    randomized_items = shuffle_dataset_items(randomized_items, master_seed)
    return randomized_items


def main():
    input_path = ""
    output_path = ""
    master_seed = 42

    with open(input_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list):
        raise ValueError("input json must contain a list of dataset items.")

    randomized_items = randomize_dataset(items, master_seed)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(randomized_items, f, ensure_ascii=False, indent=2)

    print(f"saved randomized dataset to: {output_path}")
    print(f"master seed: {master_seed}")


if __name__ == "__main__":
    main()