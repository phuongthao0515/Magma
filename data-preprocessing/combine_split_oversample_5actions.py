"""
Combine Word + Excel + PowerPoint, split balanced val, oversample rare actions in train.

Pipeline:
1. Load all 3 app JSONs (with mark coordinates already injected) — 5 actions (incl. TERMINATE)
2. Split balanced val: 50 per action per app = 750 val samples (5 actions × 3 apps × 50)
3. Oversample rare actions in train: TERMINATE ×7, RIGHT_CLICK ×10, DOUBLE_CLICK ×10, TYPE ×5
4. Shuffle and save

Output goes to a new folder (3_apps_combined_5actions/) to avoid overwriting
the existing 4-actions train/val files.

Usage:
    python data-preprocessing/combine_split_oversample_5actions.py
"""

import json
import random
from collections import Counter

# ============ CONFIGURATION ============
INPUT_JSONS = {
    "word": "/home/thaole/thao_le/Magma/datasets/agentnet/word_5actions/word_final_yolo_ocr/word_som_with_marks_list.json",
    "excel": "/home/thaole/thao_le/Magma/datasets/agentnet/excel_5actions/excel_final_yolo_ocr/excel_som_with_marks_list.json",
    "powerpoint": "/home/thaole/thao_le/Magma/datasets/agentnet/powerpoint_5actions/powerpoint_final_yolo_ocr/powerpoint_som_with_marks_list.json",
}

OUTPUT_DIR = "/home/thaole/thao_le/Magma/datasets/agentnet/3_apps_combined_5actions"
TRAIN_OUTPUT = f"{OUTPUT_DIR}/train_3_apps_5actions.json"
VAL_OUTPUT = f"{OUTPUT_DIR}/val_3_apps_5actions.json"

# Balanced val: samples per action per app
VAL_PER_ACTION_PER_APP = 50

# Oversampling multipliers for training
TRAIN_MULTIPLIERS = {
    "CLICK": 1,
    "TYPE": 5,
    "DOUBLE_CLICK": 10,
    "RIGHT_CLICK": 10,
    "TERMINATE": 7,
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
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load all apps
    all_samples = {}  # app_name → list of samples
    for app_name, path in INPUT_JSONS.items():
        if not os.path.exists(path):
            print(f"WARNING: {app_name} not found at {path}, skipping")
            continue
        with open(path) as f:
            data = json.load(f)
        all_samples[app_name] = data
        print(f"  Loaded {app_name}: {len(data)} samples")

    # Step 2: Split val — 50 per action per app (balanced)
    val_samples = []
    train_samples = []

    for app_name, samples in all_samples.items():
        # Group by action
        by_action = {}
        for s in samples:
            action = get_action(s)
            by_action.setdefault(action, []).append(s)

        app_val = []
        app_train = []

        for action, action_samples in by_action.items():
            random.shuffle(action_samples)
            n_val = min(VAL_PER_ACTION_PER_APP, len(action_samples))
            app_val.extend(action_samples[:n_val])
            app_train.extend(action_samples[n_val:])

        val_samples.extend(app_val)
        train_samples.extend(app_train)

        # Per-app val stats
        val_actions = Counter(get_action(s) for s in app_val)
        print(f"\n  {app_name} val ({len(app_val)}):")
        for a, c in val_actions.most_common():
            print(f"    {a:>15}: {c}")

    random.shuffle(val_samples)

    # Step 3: Oversample train
    train_by_action = {}
    for s in train_samples:
        action = get_action(s)
        train_by_action.setdefault(action, []).append(s)

    print(f"\n{'=' * 60}")
    print(f"  TRAIN BEFORE OVERSAMPLING")
    print(f"{'=' * 60}")
    for action in sorted(train_by_action.keys()):
        count = len(train_by_action[action])
        mult = TRAIN_MULTIPLIERS.get(action, 1)
        print(f"    {action:>15}: {count:>5} × {mult} = {count * mult}")

    oversampled = []
    for action, samples in train_by_action.items():
        mult = TRAIN_MULTIPLIERS.get(action, 1)
        oversampled.extend(samples * mult)

    random.shuffle(oversampled)

    # Step 4: Print summary
    total_train = len(oversampled)
    total_val = len(val_samples)

    print(f"\n{'=' * 60}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 60}")

    print(f"\n  VAL ({total_val} samples):")
    val_counts = Counter(get_action(s) for s in val_samples)
    for a, c in val_counts.most_common():
        print(f"    {a:>15}: {c:>5} ({c/total_val*100:.1f}%)")

    print(f"\n  TRAIN ({total_train} samples, after oversampling):")
    train_counts = Counter(get_action(s) for s in oversampled)
    for a, c in train_counts.most_common():
        print(f"    {a:>15}: {c:>5} ({c/total_train*100:.1f}%)")

    # Per-app distribution in train
    print(f"\n  TRAIN per app:")
    for app_name in all_samples.keys():
        prefix = f"{app_name}_"
        app_count = sum(1 for s in oversampled if s["id"].startswith(prefix))
        print(f"    {app_name:>12}: {app_count}")

    # Step 5: Save
    with open(TRAIN_OUTPUT, "w") as f:
        json.dump(oversampled, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved train: {TRAIN_OUTPUT} ({total_train} samples)")

    with open(VAL_OUTPUT, "w") as f:
        json.dump(val_samples, f, indent=2, ensure_ascii=False)
    print(f"  Saved val:   {VAL_OUTPUT} ({total_val} samples)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
