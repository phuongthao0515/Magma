"""
Split Word SoM dataset into train/val (80/20) with random seed.

Input:  word_mind2web_som_dense_iou0.1.json
Output: word_train.json, word_val.json

Usage:
    cd /home/thaole/thao_le/Magma
    python data-preprocessing/split_train_val.py
"""

import json
import random

SEED = 42
INPUT = "/home/thaole/thao_le/Magma/datasets/agentnet/word/som-reduced/word_som_reduced.json"
TRAIN_OUT = "/home/thaole/thao_le/Magma/datasets/agentnet/word/som-reduced/train.json"
VAL_OUT = "/home/thaole/thao_le/Magma/datasets/agentnet/word/som-reduced/val.json"
SPLIT_RATIO = 0.8


def main():
    random.seed(SEED)

    with open(INPUT) as f:
        data = json.load(f)

    indices = list(range(len(data)))
    random.shuffle(indices)

    split = int(len(data) * SPLIT_RATIO)
    train_indices = sorted(indices[:split])
    val_indices = sorted(indices[split:])

    train = [data[i] for i in train_indices]
    val = [data[i] for i in val_indices]

    with open(TRAIN_OUT, "w") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)

    with open(VAL_OUT, "w") as f:
        json.dump(val, f, indent=2, ensure_ascii=False)

    print(f"Seed:  {SEED}")
    print(f"Total: {len(data)}")
    print(f"Train: {len(train)} ({TRAIN_OUT})")
    print(f"Val:   {len(val)} ({VAL_OUT})")


if __name__ == "__main__":
    main()
