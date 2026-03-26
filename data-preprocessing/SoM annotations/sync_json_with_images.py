"""
Sync JSON with images — removes entries with missing images OR too many marks (>150).
Also deletes the dense image files from disk.

Usage:
    cd /home/thaole/thao_le/Magma
    python "data-preprocessing/SoM annotations/sync_json_with_images.py"
"""

import json
import os

JSON_PATH = "/home/thaole/thao_le/Magma/datasets/agentnet/word/som-reduced/word_som_reduced.json"
IMAGE_DIR = "/home/thaole/thao_le/Magma/datasets/agentnet/word/som-reduced"
MAX_MARKS = 150


def main():
    with open(JSON_PATH) as f:
        data = json.load(f)

    images_on_disk = set(os.listdir(IMAGE_DIR))

    # Find max MARK per image to identify dense images
    image_max_mark = {}
    for e in data:
        action = json.loads(e["conversations"][1]["value"])
        mark = action.get("MARK")
        if mark != "None" and isinstance(mark, int):
            img = e["image"]
            image_max_mark[img] = max(image_max_mark.get(img, 0), mark)

    dense_images = {img for img, m in image_max_mark.items() if m + 1 > MAX_MARKS}

    before = len(data)

    # Filter: keep only if image exists AND not too dense
    filtered = [
        e for e in data
        if e["image"] in images_on_disk and e["image"] not in dense_images
    ]

    removed_missing = sum(1 for e in data if e["image"] not in images_on_disk)
    removed_dense = sum(1 for e in data if e["image"] in dense_images)

    # Delete dense images from disk
    deleted_files = 0
    for img in dense_images:
        path = os.path.join(IMAGE_DIR, img)
        if os.path.exists(path):
            os.remove(path)
            deleted_files += 1

    with open(JSON_PATH, "w") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"Before:              {before} samples")
    print(f"After:               {len(filtered)} samples")
    print(f"Removed (missing):   {removed_missing}")
    print(f"Removed (>{MAX_MARKS} marks): {removed_dense}")
    print(f"Deleted image files: {deleted_files}")
    print(f"Dense images found:  {len(dense_images)}")


if __name__ == "__main__":
    main()
