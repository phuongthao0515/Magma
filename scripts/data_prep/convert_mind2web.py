"""
Convert Mind2Web from HuggingFace to JSON + Images (Magma format)

Usage:
    python scripts/data_prep/convert_mind2web.py
    python scripts/data_prep/convert_mind2web.py --max_samples 100  # For testing
    python scripts/data_prep/convert_mind2web.py --split train      # Specific split
    python scripts/data_prep/convert_mind2web.py --val_ratio 0.1    # 10% validation

Input:  Downloads from HuggingFace: MagmaAI/Magma-Mind2Web-SoM
Output: datasets/mind2web/mind2web_train.json
        datasets/mind2web/mind2web_val.json (if --val_ratio > 0)
        datasets/mind2web/images/*.png
"""

import os
import json
import argparse
import random
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from pathlib import Path


def download_mind2web(
    dataset_name: str = "MagmaAI/Magma-Mind2Web-SoM",
    cache_dir: str = "datasets/mind2web/raw",
    split: str = "train"
):
    """
    Download Mind2Web dataset from HuggingFace.

    Args:
        dataset_name: HuggingFace dataset name
        cache_dir: Directory to cache the downloaded dataset
        split: Dataset split to download (train, test, etc.)

    Returns:
        The loaded dataset
    """
    print("=" * 50)
    print("Downloading Mind2Web from HuggingFace")
    print("=" * 50)
    print(f"Dataset: {dataset_name}")
    print(f"Split: {split}")
    print(f"Cache dir: {cache_dir}")
    print()

    os.makedirs(cache_dir, exist_ok=True)

    # Check if already downloaded
    # local_path = os.path.join(cache_dir, split)
    local_path = "dataset/mind2web"
    if os.path.exists(local_path):
        print(f"Found cached dataset at: {local_path}")
        try:
            dataset = load_from_disk(local_path)
            print(f"Loaded cached dataset with {len(dataset)} samples")
            return dataset
        except Exception as e:
            print(f"Failed to load cached dataset: {e}")
            print("Re-downloading...")

    # Download from HuggingFace
    print(f"Downloading from HuggingFace: {dataset_name}...")
    try:
        dataset = load_dataset(dataset_name, split=split)
        print(f"Downloaded {len(dataset)} samples")

        print(f"Caching dataset to: {local_path}")
        dataset.save_to_disk(local_path)
        print("Dataset cached successfully!")

        return dataset
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nTrying alternative dataset names...")

        alternative_names = [
            "osunlp/Mind2Web",
            "mind2web/mind2web",
        ]
        for alt_name in alternative_names:
            try:
                print(f"Trying: {alt_name}")
                dataset = load_dataset(alt_name, split=split)
                print(f"Success! Downloaded {len(dataset)} samples")
                dataset.save_to_disk(local_path)
                return dataset
            except Exception:
                continue

        raise RuntimeError(
            f"Failed to download Mind2Web dataset. "
            f"Please download manually and place in {cache_dir}"
        )


def convert_mind2web(
    dataset=None,
    output_dir: str = "datasets/mind2web",
    output_json: str = None,
    output_images: str = None,
    max_samples: int = None,
    split: str = "train",
    val_ratio: float = 0.0,
    seed: int = 42
):
    """
    Convert Mind2Web from HuggingFace format to Magma's JSON + Images format.

    Args:
        dataset: Pre-loaded dataset (if None, will download)
        output_dir: Base output directory
        output_json: Output JSON file path (auto-generated if None)
        output_images: Output images folder path (auto-generated if None)
        max_samples: Max samples to convert (None = all)
        split: Dataset split name for output file naming
        val_ratio: Ratio of data to use for validation (0.0 to 1.0)
        seed: Random seed for train/val split
    """
    if output_json is None:
        output_json = os.path.join(output_dir, f"mind2web_{split}.json")
    if output_images is None:
        output_images = os.path.join(output_dir, "images")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_images, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    print(f"Output JSON: {output_json}")
    print(f"Output images: {output_images}")
    if val_ratio > 0:
        print(f"Validation ratio: {val_ratio}")

    if dataset is None:
        dataset = download_mind2web(split=split)

    data = dataset
    print(f"\nDataset loaded with {len(data)} samples")

    if max_samples:
        data = data.select(range(min(max_samples, len(data))))
        print(f"Limited to {len(data)} samples")

    # Convert each sample
    all_data = []
    skipped = 0
    print("\nConverting samples...")

    for idx, item in enumerate(tqdm(data)):
        try:
            # Generate image filename
            img_filename = f"mind2web_{idx:06d}.png"
            img_path = os.path.join(output_images, img_filename)

            # Save image
            if 'image' in item and item['image'] is not None:
                item['image'].save(img_path)
            else:
                print(f"Warning: Sample {idx} has no image, skipping...")
                skipped += 1
                continue

            # Create JSON entry (Magma format)
            entry = {
                "id": item.get('id', f"mind2web_{idx:06d}"),
                "image": f"images/{img_filename}",
                "conversations": item['conversations']
            }
            all_data.append(entry)

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            skipped += 1
            continue

    # Split into train/val if val_ratio > 0
    if val_ratio > 0:
        random.seed(seed)
        random.shuffle(all_data)

        val_size = int(len(all_data) * val_ratio)
        val_data = all_data[:val_size]
        train_data = all_data[val_size:]

        # Save train JSON
        train_json = os.path.join(output_dir, "mind2web_train.json")
        print(f"\nSaving train JSON to: {train_json}")
        with open(train_json, "w", encoding="utf-8") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

        # Save val JSON
        val_json = os.path.join(output_dir, "mind2web_val.json")
        print(f"Saving val JSON to: {val_json}")
        with open(val_json, "w", encoding="utf-8") as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)

        print(f"\nTrain samples: {len(train_data)}")
        print(f"Val samples: {len(val_data)}")
    else:
        train_data = all_data
        # Save JSON
        print(f"\nSaving JSON to: {output_json}")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "=" * 50)
    print("Conversion Complete!")
    print("=" * 50)
    print(f"Total samples converted: {len(all_data)}")
    print(f"Samples skipped: {skipped}")
    print(f"Images folder: {output_images}")
    print(f"Images count: {len(os.listdir(output_images))}")

    # Show sample entry
    if all_data:
        print("\nSample entry:")
        print(json.dumps(all_data[0], indent=2))

    # Verify config file
    config_path = "data_configs/mind2web.yaml"
    if os.path.exists(config_path):
        print(f"\nConfig file exists: {config_path}")
        print("You can start training with:")
        print("  bash scripts/finetune/finetune_mind2web_qlora.sh")
    else:
        print(f"\nWarning: Config file not found at {config_path}")
        print("Make sure the config points to the correct data paths.")

    return train_data


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert Mind2Web dataset to Magma format"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="MagmaAI/Magma-Mind2Web-SoM",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to download and convert"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/mind2web",
        help="Output directory for converted data"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to convert (for testing)"
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download and use existing cached data"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio of data for validation (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("Mind2Web Data Preparation")
    print("=" * 50)
    print(f"Dataset: {args.dataset_name}")
    print(f"Split: {args.split}")
    print(f"Output: {args.output_dir}")
    print(f"Val ratio: {args.val_ratio} ({args.val_ratio*100:.0f}%)")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print()

    # Download and convert
    if args.skip_download:
        # Try to load from cache
        cache_path = os.path.join(args.output_dir, "raw", args.split)
        if os.path.exists(cache_path):
            dataset = load_from_disk(cache_path)
        else:
            raise FileNotFoundError(
                f"No cached dataset found at {cache_path}. "
                f"Run without --skip_download first."
            )
    else:
        dataset = download_mind2web(
            dataset_name=args.dataset_name,
            cache_dir=os.path.join(args.output_dir, "raw"),
            split=args.split
        )

    convert_mind2web(
        dataset=dataset,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        split=args.split,
        val_ratio=args.val_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
