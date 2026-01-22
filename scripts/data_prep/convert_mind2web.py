"""
Convert Mind2Web Arrow format (HuggingFace) to JSON + Images (Magma format)

Usage:
    python scripts/data_prep/convert_mind2web.py

Input:  datasets/mind2web/*.arrow
Output: datasets/mind2web/mind2web_train.json
        datasets/mind2web/images/*.png
"""

import os
import json
from datasets import load_from_disk
from tqdm import tqdm
from pathlib import Path


def convert_mind2web(
    arrow_path: str = "datasets/mind2web",
    output_json: str = "datasets/mind2web/mind2web_train.json",
    output_images: str = "datasets/mind2web/images",
    max_samples: int = None  # Set to limit samples for testing
):
    """
    Convert Mind2Web from Arrow format to Magma's JSON + Images format.

    Args:
        arrow_path: Path to Arrow dataset folder
        output_json: Output JSON file path
        output_images: Output images folder path
        max_samples: Max samples to convert (None = all)
    """

    # Create images folder
    os.makedirs(output_images, exist_ok=True)
    print(f"Output images folder: {output_images}")

    # Load Arrow dataset (saved with save_to_disk)
    print(f"Loading Arrow dataset from: {arrow_path}")
    dataset = load_from_disk(arrow_path)
    print(f"Dataset structure: {dataset}")

    # Dataset is loaded directly (not DatasetDict)
    data = dataset
    print(f"Dataset loaded with {len(data)} samples")

    # Limit samples if specified
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))
        print(f"Limited to {len(data)} samples")

    # Convert each sample
    train_data = []
    print("\nConverting samples...")

    for idx, item in enumerate(tqdm(data)):
        try:
            # Generate image filename
            img_filename = f"mind2web_{idx:06d}.png"
            img_path = os.path.join(output_images, img_filename)

            # Save image
            if item['image'] is not None:
                item['image'].save(img_path)
            else:
                print(f"Warning: Sample {idx} has no image, skipping...")
                continue

            # Create JSON entry (Magma format)
            entry = {
                "id": item.get('id', f"mind2web_{idx:06d}"),
                "image": f"images/{img_filename}",
                "conversations": item['conversations']
            }
            train_data.append(entry)

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    # Save JSON
    print(f"\nSaving JSON to: {output_json}")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "=" * 50)
    print("Conversion Complete!")
    print("=" * 50)
    print(f"Total samples converted: {len(train_data)}")
    print(f"JSON file: {output_json}")
    print(f"Images folder: {output_images}")
    print(f"Images count: {len(os.listdir(output_images))}")

    # Show sample entry
    if train_data:
        print("\nSample entry:")
        print(json.dumps(train_data[0], indent=2))

    return train_data


if __name__ == "__main__":
    convert_mind2web()
