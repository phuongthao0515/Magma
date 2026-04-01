"""
Run SoM (OmniParser v2 + MarkHelper) on Word dataset images and fill MARK IDs.

Uses OmniParser v2 icon detection model (microsoft/OmniParser-v2.0) to detect UI elements.
OCR is not used — OmniParser's YOLO model detects interactive elements including text buttons.

Pipeline:
1. Read word_mind2web_style.json (samples) + word_tasks.jsonl (coordinates)
2. For each unique image: OmniParser → detect UI elements → annotate with marks
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
import torch
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm

# ============ CONFIGURATION ============
WORD_JSONL = "/home/thaole/thao_le/Magma/datasets/agentnet/word/word_tasks.jsonl"
WORD_MIND2WEB = "/home/thaole/thao_le/Magma/datasets/agentnet/word/word_mind2web_style.json"
IMAGE_DIR = "/home/thaole/thao_le/Magma/datasets/agentnet/office_images"
OUTPUT_IMAGE_DIR = "/home/thaole/thao_le/Magma/datasets/agentnet/word/som-reduced-100"
OUTPUT_JSON = "/home/thaole/thao_le/Magma/datasets/agentnet/word/som-reduced/word_som_reduced_100.json"

WEIGHTS_DIR = "/home/thaole/thao_le/Magma/weights"
OMNIPARSER_MODEL_PATH = os.path.join(WEIGHTS_DIR, "icon_detect", "model.pt")
BOX_THRESHOLD = 0.1
MIN_BOX_AREA = 0.0
OVERLAP_IOU_THRESHOLD = 0.7
# =======================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from agents.ui_agent.util.som import MarkHelper, plot_boxes_with_marks


def remove_overlap(boxes, iou_threshold):
    """Remove overlapping boxes, keeping the smaller one.
    Copied from utils.py to avoid OCR import at module level.
    Input/output: tensor of shape (N, 4) in xyxy normalized format.
    """
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    boxes = boxes.tolist()
    filtered = []
    for i, box1 in enumerate(boxes):
        is_valid = True
        for j, box2 in enumerate(boxes):
            if i != j and IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid = False
                break
        if is_valid:
            filtered.append(box1)
    return torch.tensor(filtered) if filtered else torch.zeros(0, 4)


def detect_ui_elements(image, yolo_model, box_threshold=None, overlap_iou_threshold=None, min_box_area=None):
    """Detect UI elements with YOLO (OmniParser).
    Applies overlap removal (keeps smaller box) and taskbar filter.
    Returns list of (y, x, h, w) normalized bboxes.

    Args:
        box_threshold: YOLO confidence threshold (default: module BOX_THRESHOLD=0.1)
        overlap_iou_threshold: IoU threshold for overlap removal (default: module OVERLAP_IOU_THRESHOLD=0.7)
        min_box_area: minimum normalized box area to keep (default: module MIN_BOX_AREA=0.0)
    """
    _box_threshold = box_threshold if box_threshold is not None else BOX_THRESHOLD
    _overlap_iou = overlap_iou_threshold if overlap_iou_threshold is not None else OVERLAP_IOU_THRESHOLD
    _min_area = min_box_area if min_box_area is not None else MIN_BOX_AREA

    w, h = image.size
    result = yolo_model.predict(source=image, conf=_box_threshold, iou=0.1, verbose=False)

    raw_boxes = result[0].boxes.xyxy
    if len(raw_boxes) == 0:
        return []

    # Normalize to [0, 1]
    xyxy_norm = raw_boxes / torch.Tensor([w, h, w, h]).to(raw_boxes.device)

    # Remove overlapping boxes (keep smaller box when IoU > threshold)
    xyxy_norm = remove_overlap(xyxy_norm, iou_threshold=_overlap_iou)

    # Convert xyxy -> (y, x, h, w), apply area filter and taskbar filter
    bboxes = []
    for box in xyxy_norm.tolist():
        x1, y1, x2, y2 = box
        bh, bw = y2 - y1, x2 - x1
        if bh * bw >= _min_area:
            center_y = (y1 + y2) / 2
            if center_y <= 0.93:  # Filter taskbar
                bboxes.append((y1, x1, bh, bw))
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
    if not os.path.exists(OMNIPARSER_MODEL_PATH):
        raise FileNotFoundError(f"YOLO weights not found at: {OMNIPARSER_MODEL_PATH}")
    print(f"YOLO weights: {OMNIPARSER_MODEL_PATH}")

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
    yolo_model = YOLO(OMNIPARSER_MODEL_PATH)
    mark_helper = MarkHelper()
    mark_helper.min_font_size = 12
    mark_helper.max_font_size = 14
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    # Step 4: Process images and fill MARK
    image_cache = {}  # image_name -> (bboxes, som_image_name)
    output_data = []
    all_distances = []  # Track match distances for quality measurement
    stats = {
        "total": 0,
        "matched_inside": 0,
        "matched_nearest": 0,
        "no_coords": 0,
        "no_elements": 0,
        "mark_not_needed": 0,
    }

    for idx, sample in enumerate(tqdm(samples, desc="Processing")):
        stats["total"] += 1
        image_name = sample["image"]
        action = json.loads(sample["conversations"][1]["value"])
        action_type = action["ACTION"]

        # Run YOLO on image (with caching)
        if image_name in image_cache:
            bboxes, som_image_name = image_cache[image_name]
        else:
            image_path = os.path.join(IMAGE_DIR, image_name)
            if not os.path.exists(image_path):
                stats["no_elements"] += 1
                image_cache[image_name] = ([], f"word_som_{image_name}")
                continue
            try:
                image = Image.open(image_path).convert("RGB")
                bboxes = detect_ui_elements(image, yolo_model)
            except Exception as e:
                print(f"  WARNING: Failed on {image_name}: {e}")
                image_cache[image_name] = ([], f"word_som_{image_name}")
                continue

            som_image_name = f"word_som_{image_name}"
            if bboxes:
                som_image = plot_boxes_with_marks(
                    image.copy(), bboxes, mark_helper,
                    edgecolor=(255, 0, 0), linewidth=1,
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
                    all_distances.append(distance)
                    if distance == 0.0:
                        stats["matched_inside"] += 1
                    else:
                        stats["matched_nearest"] += 1
                    mark = mark_id
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

        # Periodic save every 1000 samples
        if (idx + 1) % 1000 == 0:
            print(f"\n  Checkpoint: saving {len(output_data)} samples...")
            with open(OUTPUT_JSON, "w") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Save
    print(f"\nWriting: {OUTPUT_JSON}")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"  Total samples:             {stats['total']}")
    print(f"  Matched inside bbox:       {stats['matched_inside']}")
    print(f"  Matched nearest (<5%):     {stats['matched_nearest']}")
    print(f"  No coordinates available:  {stats['no_coords']}")
    print(f"  No elements detected:      {stats['no_elements']}")
    print(f"  MARK not needed (action):  {stats['mark_not_needed']}")
    print(f"  Output samples:            {len(output_data)}")
    if image_cache:
        avg = np.mean([len(v[0]) for v in image_cache.values()])
        print(f"  Avg elements per image:    {avg:.1f}")
    if all_distances:
        print(f"  Match distance stats:")
        print(f"    Mean:   {np.mean(all_distances):.4f}")
        print(f"    Median: {np.median(all_distances):.4f}")
        print(f"    <1%:    {100*sum(1 for d in all_distances if d < 0.01)/len(all_distances):.1f}%")
        print(f"    <2%:    {100*sum(1 for d in all_distances if d < 0.02)/len(all_distances):.1f}%")
        print(f"    <5%:    {100*sum(1 for d in all_distances if d < 0.05)/len(all_distances):.1f}%")
    print(f"  SoM images: {OUTPUT_IMAGE_DIR}")
    print(f"  JSON: {OUTPUT_JSON}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
