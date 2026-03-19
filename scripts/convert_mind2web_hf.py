"""
Convert MagmaAI/Magma-Mind2Web-SoM from HuggingFace arrow format
to the JSON + images layout expected by the training pipeline.

Output structure:
    datasets/mind2web/
        mind2web_train.json
        mind2web_val.json
        images/
            mind2web_000000.png
            mind2web_000001.png
            ...

Each JSON entry:
    {
        "id": "mind2web_000000",
        "image": "images/mind2web_000000.png",
        "conversations": [
            {"from": "human", "value": "<image>\n...instruction..."},
            {"from": "gpt",   "value": "{\"ACTION\": ..., \"MARK\": ..., \"VALUE\": ...}"}
        ]
    }

Usage:
    python evaluation/convert_mind2web_hf.py
"""

import os
import json
from datasets import load_dataset
from tqdm import tqdm

# ── Paths (absolute) ─────────────────────────────────────────────────────────
PROJECT_ROOT = "/home/sonnguyen/thaole/magma/Magma"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "datasets", "mind2web")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")

HF_DATASET = "MagmaAI/Magma-Mind2Web-SoM"   

# Train / val split ratio (the HF dataset only has a "train" split;
# we carve out 10 % for validation).
VAL_RATIO = 0.1


def convert_split(dataset, split_name: str, start_idx: int = 0):
    """Convert a HuggingFace dataset split to JSON + saved images.

    Returns the list of JSON records and the next available index.
    """
    records = []
    idx = start_idx
    for sample in tqdm(dataset, desc=f"Converting {split_name}"):
        sample_id = sample.get("id", f"mind2web_{idx:06d}")
        image_filename = f"mind2web_{idx:06d}.png"
        image_relpath = os.path.join("images", image_filename)
        image_abspath = os.path.join(IMAGE_DIR, image_filename)

        # Save image
        image = sample["image"]
        if image is not None:
            image.convert("RGB").save(image_abspath)

        # Build conversations (pass through if already in the right shape)
        if "conversations" in sample:
            convs = sample["conversations"]
            # HF datasets may store conversations as list[dict] or list[list].
            # Normalise to list[dict] with "from" / "value" keys.
            if isinstance(convs, list) and len(convs) > 0:
                if isinstance(convs[0], dict):
                    pass  # already correct
                else:
                    # Fallback: assume [user_text, assistant_text]
                    convs = [
                        {"from": "human", "value": str(convs[0])},
                        {"from": "gpt", "value": str(convs[1])},
                    ]
        else:
            # Build from separate fields if present
            user_text = sample.get("question", sample.get("prompt", ""))
            assistant_text = sample.get("answer", sample.get("response", ""))
            convs = [
                {"from": "human", "value": f"<image>\n{user_text}"},
                {"from": "gpt", "value": assistant_text},
            ]

        records.append({
            "id": sample_id,
            "image": image_relpath,
            "conversations": convs,
        })
        idx += 1

    return records, idx


def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # Load from HuggingFace
    print(f"Loading dataset: {HF_DATASET}")
    available_splits = load_dataset(HF_DATASET).keys()
    print(f"Available splits: {list(available_splits)}")

    if "train" in available_splits and "validation" in available_splits:
        # Both splits already present
        train_ds = load_dataset(HF_DATASET, split="train")
        val_ds = load_dataset(HF_DATASET, split="validation")
        print(f"Train size: {len(train_ds)},  Val size: {len(val_ds)}")
        train_records, next_idx = convert_split(train_ds, "train", start_idx=0)
        val_records, _ = convert_split(val_ds, "val", start_idx=next_idx)

    elif "train" in available_splits:
        # Only train split — carve out a validation portion
        full_ds = load_dataset(HF_DATASET, split="train")
        n_val = max(1, int(len(full_ds) * VAL_RATIO))
        n_train = len(full_ds) - n_val
        print(f"Total: {len(full_ds)},  Train: {n_train},  Val: {n_val}")
        split = full_ds.train_test_split(test_size=n_val, seed=42)
        train_records, next_idx = convert_split(split["train"], "train", start_idx=0)
        val_records, _ = convert_split(split["test"], "val", start_idx=next_idx)

    else:
        # Single unnamed split
        full_ds = load_dataset(HF_DATASET, split=list(available_splits)[0])
        n_val = max(1, int(len(full_ds) * VAL_RATIO))
        n_train = len(full_ds) - n_val
        print(f"Total: {len(full_ds)},  Train: {n_train},  Val: {n_val}")
        split = full_ds.train_test_split(test_size=n_val, seed=42)
        train_records, next_idx = convert_split(split["train"], "train", start_idx=0)
        val_records, _ = convert_split(split["test"], "val", start_idx=next_idx)

    # Write JSON files
    train_json = os.path.join(OUTPUT_DIR, "mind2web_train.json")
    val_json = os.path.join(OUTPUT_DIR, "mind2web_val.json")

    with open(train_json, "w") as f:
        json.dump(train_records, f, indent=2)
    with open(val_json, "w") as f:
        json.dump(val_records, f, indent=2)

    print(f"\nDone!")
    print(f"  Train: {train_json}  ({len(train_records)} samples)")
    print(f"  Val:   {val_json}  ({len(val_records)} samples)")
    print(f"  Images: {IMAGE_DIR}")


if __name__ == "__main__":
    main()
