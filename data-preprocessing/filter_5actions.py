"""
Pre-filter mind2web JSONs to 5 target actions before SoM preprocessing.
Saves YOLO+OCR time by dropping samples that would be discarded anyway.

For Excel/PowerPoint: also removes redundant/incorrect steps (not filtered
during their mind2web generation, unlike Word).

Usage:
    python data-preprocessing/filter_5actions.py --app excel
    python data-preprocessing/filter_5actions.py --app powerpoint
    python data-preprocessing/filter_5actions.py --app word
    python data-preprocessing/filter_5actions.py --app all
"""

import argparse
import json
import os
from collections import Counter

KEEP_ACTIONS = {"CLICK", "TYPE", "DOUBLE_CLICK", "RIGHT_CLICK", "TERMINATE"}

APP_CONFIGS = {
    "word": {
        "mind2web": "/home/thaole/thao_le/Magma/datasets/agentnet/word_5actions/word_mind2web_style_5actions.json",
        "jsonl": "/home/thaole/thao_le/Magma/datasets/agentnet/word/word_tasks.jsonl",
        "output": "/home/thaole/thao_le/Magma/datasets/agentnet/word_5actions/word_5actions.json",
        "filter_bad_steps": False,  # Word already filtered by add_word_terminate.py
    },
    "excel": {
        "mind2web": "/home/thaole/thao_le/Magma/datasets/agentnet/excel/excel_windows_mind2web_style.json",
        "jsonl": "/home/thaole/thao_le/Magma/datasets/agentnet/excel/excel_windows_samples.jsonl",
        "output": "/home/thaole/thao_le/Magma/datasets/agentnet/excel_5actions/excel_5actions.json",
        "filter_bad_steps": True,  # Teammate's script didn't filter these
    },
    "powerpoint": {
        "mind2web": "/home/thaole/thao_le/Magma/datasets/agentnet/powerpoint/powerpoint_windows_mind2web_style.json",
        "jsonl": "/home/thaole/thao_le/Magma/datasets/agentnet/powerpoint/powerpoint_windows_samples.jsonl",
        "output": "/home/thaole/thao_le/Magma/datasets/agentnet/powerpoint_5actions/powerpoint_5actions.json",
        "filter_bad_steps": True,  # Teammate's script didn't filter these
    },
}


def build_bad_step_images(jsonl_path):
    """Build set of image names from redundant/incorrect steps.

    Each trajectory step has a unique screenshot, so image name
    uniquely identifies the step.
    """
    bad_images = set()
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            for step in entry["traj"]:
                is_correct = step["value"].get("last_step_correct", True)
                is_redundant = step["value"].get("last_step_redundant", False)
                if not is_correct or is_redundant:
                    bad_images.add(step["image"])
    return bad_images


def get_action(sample):
    try:
        parsed = json.loads(sample["conversations"][1]["value"])
        return parsed.get("ACTION")
    except (json.JSONDecodeError, KeyError):
        return None


def process_app(app_name, config):
    print(f"\n{'=' * 60}")
    print(f"  {app_name.upper()}")
    print(f"{'=' * 60}")

    os.makedirs(os.path.dirname(config["output"]), exist_ok=True)

    with open(config["mind2web"]) as f:
        samples = json.load(f)
    print(f"  Input: {len(samples)} samples")

    # Show action distribution before
    before = Counter(get_action(s) for s in samples)
    print(f"  Actions before:")
    for a, c in before.most_common():
        tag = " <<<" if a in KEEP_ACTIONS else ""
        print(f"    {a:>15}: {c:>5}{tag}")

    # Build bad step index if needed
    bad_images = set()
    if config["filter_bad_steps"]:
        bad_images = build_bad_step_images(config["jsonl"])
        print(f"  Bad step images (redundant/incorrect): {len(bad_images)}")

    # Filter
    filtered = []
    skipped_action = 0
    skipped_bad = 0
    for sample in samples:
        action = get_action(sample)

        if action not in KEEP_ACTIONS:
            skipped_action += 1
            continue

        if config["filter_bad_steps"] and sample["image"] in bad_images:
            skipped_bad += 1
            continue

        filtered.append(sample)

    # Show results
    print(f"\n  Skipped (wrong action): {skipped_action}")
    if config["filter_bad_steps"]:
        print(f"  Skipped (redundant/incorrect): {skipped_bad}")
    print(f"  Output: {len(filtered)} samples")

    if filtered:
        after = Counter(get_action(s) for s in filtered)
        print(f"  Actions after:")
        for a, c in after.most_common():
            print(f"    {a:>15}: {c:>5} ({c/len(filtered)*100:.1f}%)")
        unique_images = len(set(s["image"] for s in filtered))
        print(f"  Unique images (YOLO+OCR will process): {unique_images}")
    else:
        print(f"  WARNING: No samples remaining after filter!")

    with open(config["output"], "w") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {config['output']}")


def main():
    parser = argparse.ArgumentParser(description="Pre-filter mind2web to 5 actions")
    parser.add_argument("--app", choices=["word", "excel", "powerpoint", "all"], default="all")
    args = parser.parse_args()

    apps = list(APP_CONFIGS.keys()) if args.app == "all" else [args.app]
    for app_name in apps:
        process_app(app_name, APP_CONFIGS[app_name])

    print(f"\n{'=' * 60}")
    print("  Next step: update preprocess_office_som.py mind2web paths")
    print("  to point to the *_5actions.json files, then run:")
    print('    python "data-preprocessing/SoM annotations/preprocess_office_som.py" --app word')
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
