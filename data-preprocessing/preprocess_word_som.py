"""
Run SoM (YOLO/OmniParser + MarkHelper) on Word dataset images and fill MARK IDs.

Pipeline:
1. Read word_mind2web_style.json (samples) + word_tasks.jsonl (coordinates)
2. For each unique image: YOLO (OmniParser) → detect UI elements → annotate with marks
3. Match action coordinates to nearest mark → fill MARK ID
4. Save SoM-annotated images + updated JSON

Actions that get MARK:
  - CLICK, DOUBLE_CLICK, RIGHT_CLICK, MIDDLE_CLICK: use step's own coordinates
  - TYPE: use previous step's click coordinates (clicked on the text field)
  - DRAG: use start coordinates
Actions that keep MARK="None":
  - SCROLL, HSCROLL, PRESS, HOTKEY, MOVE, TERMINATE (no target element)

Usage:
    cd /home/thaole/thao_le/Magma
    python data-preprocessing/preprocess_word_som.py
"""

import os
import sys
import json
import re
import numpy as np
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm

# ============ CONFIGURATION ============
WORD_JSONL = "/home/thaole/thao_le/Magma/datasets/agentnet/word/word_tasks.jsonl"
WORD_MIND2WEB = "/home/thaole/thao_le/Magma/datasets/agentnet/word/word_mind2web_style.json"
IMAGE_DIR = "/home/thaole/thao_le/Magma/datasets/agentnet/office_images"
OUTPUT_IMAGE_DIR = "/home/thaole/thao_le/Magma/datasets/agentnet/word/word_images_som"
OUTPUT_JSON = "/home/thaole/thao_le/Magma/datasets/agentnet/word/word_mind2web_som.json"

WEIGHTS_DIR = "/home/thaole/thao_le/Magma/weights"
YOLO_MODEL_PATH = os.path.join(WEIGHTS_DIR, "icon_detect", "model.pt")
BOX_THRESHOLD = 0.05
# =======================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agents.ui_agent.util.som import MarkHelper, plot_boxes_with_marks


def detect_ui_elements(image, yolo_model):
    """Detect UI elements with YOLO (OmniParser).
    Returns list of (y, x, h, w) normalized bboxes.
    """
    w, h = image.size
    result = yolo_model.predict(source=image, conf=BOX_THRESHOLD, iou=0.1, verbose=False)
    bboxes = []
    for box in result[0].boxes.xyxy:
        x1, y1, x2, y2 = box.tolist()
        bboxes.append((y1 / h, x1 / w, (y2 - y1) / h, (x2 - x1) / w))
    return bboxes


def match_click_to_mark(click_x, click_y, bboxes):
    """Match normalized (x, y) to nearest bbox.
    Returns (mark_id, distance).
    """
    if not bboxes:
        return None, float("inf")
    # First: click inside bbox (pick smallest)
    best_mark = None
    best_area = float("inf")
    for i, (y, x, h, w) in enumerate(bboxes):
        if x <= click_x <= x + w and y <= click_y <= y + h:
            area = h * w
            if area < best_area:
                best_mark = i
                best_area = area
    if best_mark is not None:
        return best_mark, 0.0
    # Second: nearest center
    best_distance = float("inf")
    for i, (y, x, h, w) in enumerate(bboxes):
        cx, cy = x + w / 2, y + h / 2
        dist = np.sqrt((click_x - cx) ** 2 + (click_y - cy) ** 2)
        if dist < best_distance:
            best_distance = dist
            best_mark = i
    return best_mark, best_distance


def extract_coordinates(code):
    """Extract (x, y) coordinates from pyautogui code."""
    match = re.search(r"(?:click|Click)\(x=([\d.]+),\s*y=([\d.]+)\)", code)
    if match:
        return float(match.group(1)), float(match.group(2))
    match = re.search(r"moveTo\(x=([\d.]+),\s*y=([\d.]+)\)", code)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None


def build_image_coords_map():
    """Build mapping: image_name -> coordinates from JSONL trajectories."""
    image_coords = {}
    image_prev_coords = {}

    with open(WORD_JSONL) as f:
        for line in f:
            entry = json.loads(line)
            prev_coords = None
            for step in entry["traj"]:
                code = step["value"].get("code", "")
                image = step["image"]
                coords = extract_coordinates(code)

                if coords:
                    image_coords[image] = coords
                image_prev_coords[image] = prev_coords

                if coords:
                    prev_coords = coords

    return image_coords, image_prev_coords


ACTIONS_NEED_MARK = {"CLICK", "DOUBLE_CLICK", "RIGHT_CLICK", "MIDDLE_CLICK", "TYPE", "DRAG"}


def main():
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"YOLO weights not found at: {YOLO_MODEL_PATH}")
    print(f"YOLO weights: {YOLO_MODEL_PATH}")

    # Step 1: Build coordinate mappings
    print("\nBuilding coordinate mappings from JSONL...")
    image_coords, image_prev_coords = build_image_coords_map()
    print(f"  Images with own coords: {len(image_coords)}")
    print(f"  Images with prev coords: {sum(1 for v in image_prev_coords.values() if v)}")

    # Step 2: Load samples
    print(f"\nLoading: {WORD_MIND2WEB}")
    with open(WORD_MIND2WEB) as f:
        samples = json.load(f)
    print(f"  Total samples: {len(samples)}")

    # Step 3: Load YOLO model
    print("\nLoading YOLO (OmniParser)...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    mark_helper = MarkHelper()
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    # Step 4: Process images and fill MARK
    image_cache = {}  # image_name -> (bboxes, som_image_name)
    output_data = []
    stats = {
        "total": 0,
        "matched_inside": 0,
        "matched_nearest": 0,
        "no_coords": 0,
        "no_elements": 0,
        "too_far": 0,
        "mark_not_needed": 0,
    }

    for sample in tqdm(samples, desc="Processing"):
        stats["total"] += 1
        image_name = sample["image"]
        action = json.loads(sample["conversations"][1]["value"])
        action_type = action["ACTION"]

        # Run YOLO on image (with caching)
        if image_name in image_cache:
            bboxes, som_image_name = image_cache[image_name]
        else:
            image_path = os.path.join(IMAGE_DIR, image_name)
            image = Image.open(image_path).convert("RGB")
            bboxes = detect_ui_elements(image, yolo_model)

            som_image_name = f"word_som_{image_name}"
            if bboxes:
                som_image = plot_boxes_with_marks(
                    image.copy(), bboxes, mark_helper,
                    edgecolor=(255, 0, 0), linewidth=2,
                    normalized_to_pixel=True, add_mark=True,
                )
                som_image.save(os.path.join(OUTPUT_IMAGE_DIR, som_image_name))
            else:
                stats["no_elements"] += 1
                image.save(os.path.join(OUTPUT_IMAGE_DIR, som_image_name))

            image_cache[image_name] = (bboxes, som_image_name)

        # Determine MARK
        mark = "None"
        if action_type in ACTIONS_NEED_MARK and bboxes:
            coords = None
            if action_type == "TYPE":
                coords = image_prev_coords.get(image_name)
                if not coords:
                    coords = image_coords.get(image_name)
            elif action_type == "DRAG":
                coords = image_coords.get(image_name)
            else:
                coords = image_coords.get(image_name)

            if coords:
                mark_id, distance = match_click_to_mark(coords[0], coords[1], bboxes)
                if mark_id is not None:
                    if distance == 0.0:
                        stats["matched_inside"] += 1
                        mark = mark_id
                    elif distance < 0.05:
                        stats["matched_nearest"] += 1
                        mark = mark_id
                    else:
                        stats["too_far"] += 1
            else:
                stats["no_coords"] += 1
        else:
            stats["mark_not_needed"] += 1

        action["MARK"] = mark if mark != "None" else "None"

        output_data.append({
            "id": sample["id"],
            "image": som_image_name,
            "conversations": [
                sample["conversations"][0],
                {"from": "assistant", "value": json.dumps(action)},
            ],
        })

    # Save
    print(f"\nWriting: {OUTPUT_JSON}")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"  Total samples:             {stats['total']}")
    print(f"  Matched inside bbox:       {stats['matched_inside']}")
    print(f"  Matched nearest (<5%):     {stats['matched_nearest']}")
    print(f"  Too far from any bbox:     {stats['too_far']}")
    print(f"  No coordinates available:  {stats['no_coords']}")
    print(f"  No elements detected:      {stats['no_elements']}")
    print(f"  MARK not needed (action):  {stats['mark_not_needed']}")
    print(f"  Output samples:            {len(output_data)}")
    if image_cache:
        avg = np.mean([len(v[0]) for v in image_cache.values()])
        print(f"  Avg elements per image:    {avg:.1f}")
    print(f"  SoM images: {OUTPUT_IMAGE_DIR}")
    print(f"  JSON: {OUTPUT_JSON}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
