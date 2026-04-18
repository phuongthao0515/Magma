"""
Inject mark coordinates + OCR text into training prompts.

Reads {app}_som.json + mark_metadata.json, injects a mark reference list
into each sample's prompt (after "What is the next action?"), outputs new JSON.

Uses Magma's pretrained format: Mark X at [x,y]
Coordinates are quantized to a GRID_SIZE x GRID_SIZE grid.

Usage:
    python data-preprocessing/inject_mark_coords_5actions.py --app word
    python data-preprocessing/inject_mark_coords_5actions.py --app excel
    python data-preprocessing/inject_mark_coords_5actions.py --app powerpoint
    python data-preprocessing/inject_mark_coords_5actions.py --app all
"""

import argparse
import json
import os
from collections import Counter

# ============ CONFIGURATION ============
APP_CONFIGS = {
    "word": {
        "som_json": "/home/thaole/thao_le/Magma/datasets/agentnet/word_5actions/word_final_yolo_ocr/word_som.json",
        "metadata": "/home/thaole/thao_le/Magma/datasets/agentnet/word_5actions/word_final_yolo_ocr/mark_metadata.json",
        "output": "/home/thaole/thao_le/Magma/datasets/agentnet/word_5actions/word_final_yolo_ocr/word_som_with_marks_list.json",
    },
    "excel": {
        "som_json": "/home/thaole/thao_le/Magma/datasets/agentnet/excel_5actions/excel_final_yolo_ocr/excel_som.json",
        "metadata": "/home/thaole/thao_le/Magma/datasets/agentnet/excel_5actions/excel_final_yolo_ocr/mark_metadata.json",
        "output": "/home/thaole/thao_le/Magma/datasets/agentnet/excel_5actions/excel_final_yolo_ocr/excel_som_with_marks_list.json",
    },
    "powerpoint": {
        "som_json": "/home/thaole/thao_le/Magma/datasets/agentnet/powerpoint_5actions/powerpoint_final_yolo_ocr/powerpoint_som.json",
        "metadata": "/home/thaole/thao_le/Magma/datasets/agentnet/powerpoint_5actions/powerpoint_final_yolo_ocr/mark_metadata.json",
        "output": "/home/thaole/thao_le/Magma/datasets/agentnet/powerpoint_5actions/powerpoint_final_yolo_ocr/powerpoint_som_with_marks_list.json",
    },
}

GRID_SIZE = 100  # quantize norm coordinates to [0, GRID_SIZE-1]

INSERTION_ANCHOR = "What is the next action?\n"

# Line to tell the model this is a reference, not ground truth
REFERENCE_NOTE = "The following mark positions and labels are approximate references to help you identify elements. Always verify by checking the image."
# =======================================


def is_clean_ocr(text):
    """Check if OCR text is clean enough to include in the prompt."""
    if not text or not text.strip():
        return False

    # Reject single character
    if len(text.strip()) < 2:
        return False

    # Reject non-ASCII (garbled unicode like euro signs)
    if not text.isascii():
        return False

    # Count "allowed" characters: letters, digits, common punctuation
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?()/-@&+_=")
    allowed_count = sum(1 for c in text if c in allowed)
    if len(text) > 0 and allowed_count / len(text) < 0.85:
        return False

    # For short text, require at least one letter
    if len(text.strip()) <= 4 and not any(c.isalpha() for c in text):
        return False

    # Reject if too many harsh special characters
    harsh = set("#*{}[]<>~^|\\`")
    harsh_count = sum(1 for c in text if c in harsh)
    if len(text) > 0 and harsh_count / len(text) > 0.3:
        return False

    return True


def format_marks_text(marks_dict, grid_size=GRID_SIZE):
    """Format mark metadata into a text string for prompt injection.

    Uses Magma pretrained format: Mark X at [gx,gy]
    Optionally appends clean OCR text: Mark X at [gx,gy] "File"

    Returns: (mark_text_string, stats_dict)
    """
    entries = []
    stats = {"total": 0, "ocr_used": 0, "ocr_filtered": 0, "ocr_null": 0}

    for mark_id in sorted(marks_dict.keys(), key=int):
        mark = marks_dict[mark_id]
        stats["total"] += 1

        # Quantize normalized center to grid
        norm_x, norm_y = mark["norm_center"]
        gx = min(int(norm_x * grid_size), grid_size - 1)
        gy = min(int(norm_y * grid_size), grid_size - 1)

        entry = f"Mark {mark_id} at [{gx},{gy}]"

        # Add OCR text if clean
        ocr_text = mark.get("ocr_text")
        if ocr_text is None:
            stats["ocr_null"] += 1
        elif is_clean_ocr(ocr_text):
            # Strip internal quotes to avoid format issues
            clean = ocr_text.strip().replace('"', "").replace("'", "")
            entry += f' "{clean}"'
            stats["ocr_used"] += 1
        else:
            stats["ocr_filtered"] += 1

        entries.append(entry)

    # Join with period-space (Magma pretrained separator)
    mark_text = ". ".join(entries)
    return mark_text, stats


def inject_into_prompt(user_value, mark_text):
    """Insert mark reference text after the 'What is the next action?' anchor."""
    if INSERTION_ANCHOR not in user_value:
        return user_value, False

    injection = f"\n{REFERENCE_NOTE}\n{mark_text}\n"
    new_value = user_value.replace(INSERTION_ANCHOR, INSERTION_ANCHOR + injection, 1)
    return new_value, True


def process_app(app_name, config):
    print(f"\n{'=' * 60}")
    print(f"  {app_name.upper()} — Inject Mark Coordinates")
    print(f"{'=' * 60}")

    # Load inputs
    with open(config["som_json"]) as f:
        samples = json.load(f)
    print(f"  Samples: {len(samples)}")

    with open(config["metadata"]) as f:
        metadata = json.load(f)
    print(f"  Metadata: {len(metadata)} images")

    # Process
    output = []
    total_stats = Counter()
    anchor_missing = 0
    metadata_missing = 0

    for sample in samples:
        image_name = sample["image"]
        user_value = sample["conversations"][0]["value"]

        if image_name not in metadata:
            # No metadata — keep prompt unchanged
            metadata_missing += 1
            output.append(sample)
            continue

        marks = metadata[image_name]["marks"]
        mark_text, stats = format_marks_text(marks)

        for k, v in stats.items():
            total_stats[k] += v

        new_value, found = inject_into_prompt(user_value, mark_text)
        if not found:
            anchor_missing += 1

        output.append({
            "id": sample["id"],
            "image": sample["image"],
            "conversations": [
                {"from": "user", "value": new_value},
                sample["conversations"][1],
            ],
        })

    # Save
    with open(config["output"], "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Stats
    print(f"\n  Results:")
    print(f"    Output samples:    {len(output)}")
    print(f"    Metadata missing:  {metadata_missing}")
    print(f"    Anchor missing:    {anchor_missing}")
    print(f"    Total marks:       {total_stats['total']}")
    print(f"    OCR text used:     {total_stats['ocr_used']}")
    print(f"    OCR text filtered: {total_stats['ocr_filtered']}")
    print(f"    OCR text null:     {total_stats['ocr_null']}")

    if total_stats["total"] > 0:
        avg = total_stats["total"] / max(len(output) - metadata_missing, 1)
        print(f"    Avg marks/sample:  {avg:.1f}")

    # Estimate token overhead
    avg_marks = total_stats["total"] / max(len(output) - metadata_missing, 1)
    est_tokens = avg_marks * 9  # ~9 tokens per mark with format
    print(f"    Est. token overhead: ~{est_tokens:.0f} tokens/sample")
    print(f"\n  Saved: {config['output']}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Inject mark coordinates into training prompts")
    parser.add_argument("--app", choices=["word", "excel", "powerpoint", "all"], default="all")
    args = parser.parse_args()

    apps = list(APP_CONFIGS.keys()) if args.app == "all" else [args.app]

    for app_name in apps:
        config = APP_CONFIGS[app_name]

        if not os.path.exists(config["som_json"]):
            print(f"WARNING: Skipping {app_name} — {config['som_json']} not found")
            continue
        if not os.path.exists(config["metadata"]):
            print(f"WARNING: Skipping {app_name} — {config['metadata']} not found")
            continue

        process_app(app_name, config)

    print(f"\nNext steps:")
    print(f"  1. Run filter_and_oversample.py on the *_som_with_marks.json files")
    print(f"  2. Update data_configs YAML to point to new train/val JSONs")
    print(f"  3. Set model_max_length=2560 in run.sh")


if __name__ == "__main__":
    main()
