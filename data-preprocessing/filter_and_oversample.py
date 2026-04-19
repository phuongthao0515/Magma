"""
Filter to 4 evaluatable actions, split balanced val, oversample rare actions in train.

Input:  som-reduced/word_som_reduced_100.json (full dataset before split)
Output: som-reduced-100/train_4actions.json
        som-reduced-100/val_4actions.json

Usage:
    python /home/thaole/thao_le/Magma/data-preprocessing/filter_and_oversample.py
"""

import json
import random
from collections import Counter

# ============ CONFIGURATION ============
INPUT_JSON = "/home/thaole/thao_le/Magma/datasets/agentnet/word/som-reduced/word_som_reduced_100.json"
TRAIN_OUTPUT = "/home/thaole/thao_le/Magma/datasets/agentnet/word/som-reduced-100/train_4actions.json"
VAL_OUTPUT = "/home/thaole/thao_le/Magma/datasets/agentnet/word/som-reduced-100/val_4actions.json"

KEEP_ACTIONS = {"CLICK", "TYPE", "DOUBLE_CLICK", "RIGHT_CLICK"}

# Balanced val: samples per action
VAL_PER_ACTION = 50

# Oversampling multipliers for training
TRAIN_MULTIPLIERS = {
    "CLICK": 1,
    "TYPE": 3,
    "DOUBLE_CLICK": 5,
    "RIGHT_CLICK": 5,
}

SEED = 42
# =======================================


def get_action(sample):
    try:
        parsed = json.loads(sample["conversations"][1]["value"])
        return parsed.get("ACTION")
    except (json.JSONDecodeError, KeyError):
        return None


def main():
    random.seed(SEED)

    with open(INPUT_JSON) as f:
        all_data = json.load(f)
    print(f"Total samples: {len(all_data)}")

    # Filter to 4 actions
    by_action = {}
    for sample in all_data:
        action = get_action(sample)
        if action in KEEP_ACTIONS:
            by_action.setdefault(action, []).append(sample)

    total_filtered = sum(len(v) for v in by_action.values())
    print(f"\nAfter filtering to {len(KEEP_ACTIONS)} actions: {total_filtered}")
    for action in sorted(KEEP_ACTIONS):
        count = len(by_action.get(action, []))
        print(f"  {action:>15}: {count:>5} ({count/total_filtered*100:.1f}%)")

    # Split: take VAL_PER_ACTION from each action for val, rest goes to train
    val_samples = []
    train_by_action = {}

    for action, samples in by_action.items():
        random.shuffle(samples)
        n_val = min(VAL_PER_ACTION, len(samples))
        val_samples.extend(samples[:n_val])
        train_by_action[action] = samples[n_val:]

    random.shuffle(val_samples)

    print(f"\nVAL ({len(val_samples)} samples, {VAL_PER_ACTION} per action):")
    val_counts = Counter(get_action(s) for s in val_samples)
    for action, count in val_counts.most_common():
        print(f"  {action:>15}: {count:>5} ({count/len(val_samples)*100:.1f}%)")

    # Oversample training
    train_output = []
    for action, samples in train_by_action.items():
        mult = TRAIN_MULTIPLIERS.get(action, 1)
        train_output.extend(samples * mult)

    random.shuffle(train_output)

    total_train = len(train_output)
    print(f"\nTRAIN ({total_train} samples, after oversampling):")
    train_counts = Counter(get_action(s) for s in train_output)
    for action, count in train_counts.most_common():
        print(f"  {action:>15}: {count:>5} ({count/total_train*100:.1f}%)")

    # Save
    with open(TRAIN_OUTPUT, "w") as f:
        json.dump(train_output, f, indent=2)
    with open(VAL_OUTPUT, "w") as f:
        json.dump(val_samples, f, indent=2)

    print(f"\nSaved:")
    print(f"  Train: {TRAIN_OUTPUT} ({total_train} samples)")
    print(f"  Val:   {VAL_OUTPUT} ({len(val_samples)} samples)")


if __name__ == "__main__":
    main()
