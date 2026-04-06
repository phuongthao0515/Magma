"""
Run SoM (OmniParser v2 YOLO + EasyOCR + MarkHelper) on Office dataset images.

Processes Word, Excel, and PowerPoint data into separate output folders.
Each app gets its own SoM images, combined JSON, and mark metadata.

Pipeline per app:
1. Load mind2web JSON + build coordinate maps
2. For each unique image: YOLO + OCR → merge → annotate with marks
3. Match action coordinates to nearest mark → fill MARK ID
4. Save SoM images + updated JSON + mark_metadata.json

Coordinate extraction (all apps use the same source):
  - All: extract coords from JSONL pyautogui code field
  - TYPE: uses previous step's click coords (the text field)

Prerequisites:
  - Run filter_4actions_mind2web.py first to filter to 4 actions and remove
    redundant/incorrect steps

Usage:
    cd /home/thaole/thao_le/Magma
    python "data-preprocessing/SoM annotations/preprocess_office_som.py" --app word
    python "data-preprocessing/SoM annotations/preprocess_office_som.py" --app excel
    python "data-preprocessing/SoM annotations/preprocess_office_som.py" --app powerpoint
    python "data-preprocessing/SoM annotations/preprocess_office_som.py" --app all
"""

import argparse
import json
import os
import re
import sys

import easyocr
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# ============ CONFIGURATION ============
APP_CONFIGS = {
    "word": {
        "jsonl": "/home/thaole/thao_le/Magma/datasets/agentnet/word/word_tasks.jsonl",
        "mind2web": "/home/thaole/thao_le/Magma/datasets/agentnet/word/word_4actions.json",
        "output_dir": "/home/thaole/thao_le/Magma/datasets/agentnet/word/word_final_yolo_ocr",
    },
    "excel": {
        "jsonl": "/home/thaole/thao_le/Magma/datasets/agentnet/excel/excel_windows_samples.jsonl",
        "mind2web": "/home/thaole/thao_le/Magma/datasets/agentnet/excel/excel_4actions.json",
        "output_dir": "/home/thaole/thao_le/Magma/datasets/agentnet/excel/excel_final_yolo_ocr",
    },
    "powerpoint": {
        "jsonl": "/home/thaole/thao_le/Magma/datasets/agentnet/powerpoint/powerpoint_windows_samples.jsonl",
        "mind2web": "/home/thaole/thao_le/Magma/datasets/agentnet/powerpoint/powerpoint_4actions.json",
        "output_dir": "/home/thaole/thao_le/Magma/datasets/agentnet/powerpoint/powerpoint_final_yolo_ocr",
    },
}

IMAGE_DIR = "/home/thaole/thao_le/Magma/datasets/agentnet/office_images"
WEIGHTS_DIR = "/home/thaole/thao_le/Magma/weights"
OMNIPARSER_MODEL_PATH = os.path.join(WEIGHTS_DIR, "icon_detect", "model.pt")

# Detection parameters
BOX_THRESHOLD = 0.25  # matched to test pipeline (stricter, cleaner with OCR)
OVERLAP_IOU_THRESHOLD = 0.7
MIN_BOX_AREA = 0.0
MAX_MARKS = 100
MARK_FONT_MIN = 12
MARK_FONT_MAX = 14

# OCR parameters
OCR_TEXT_THRESHOLD = 0.9
OCR_MAX_AREA_RATIO = 0.05  # filter OCR boxes > 5% of image area

# Actions that need coordinate → mark matching (all 4 kept actions need it)
ACTIONS_NEED_MARK = {"CLICK", "DOUBLE_CLICK", "RIGHT_CLICK", "TYPE"}
# =======================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from agents.ui_agent.util.som import MarkHelper, plot_boxes_with_marks


# ==================== OCR FUNCTIONS ====================

def run_ocr(image, ocr_reader):
    """Run EasyOCR on image, return structured element list in pixel xyxy."""
    image_np = np.array(image)
    width, height = image.size

    result = ocr_reader.readtext(
        image_np, paragraph=False, text_threshold=OCR_TEXT_THRESHOLD
    )

    ocr_elements = []
    for item in result:
        coord = item[0]  # corner format [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        text = item[1]
        # Convert corners to xyxy (use min/max to handle rotated text)
        all_x = [pt[0] for pt in coord]
        all_y = [pt[1] for pt in coord]
        x1, y1 = int(min(all_x)), int(min(all_y))
        x2, y2 = int(max(all_x)), int(max(all_y))

        # Skip degenerate boxes
        if x2 <= x1 or y2 <= y1:
            continue

        # Filter taskbar (same as YOLO filter)
        center_y_norm = ((y1 + y2) / 2) / height
        if center_y_norm > 0.93:
            continue

        # Filter large boxes
        area = (x2 - x1) * (y2 - y1)
        if area >= OCR_MAX_AREA_RATIO * width * height:
            continue

        ocr_elements.append({
            "type": "text",
            "bbox": [x1, y1, x2, y2],
            "interactivity": False,
            "content": text,
        })

    return ocr_elements


# ==================== OVERLAP REMOVAL ====================

def _box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection_area(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def _iou(box1, box2):
    inter = _intersection_area(box1, box2)
    union = _box_area(box1) + _box_area(box2) - inter + 1e-6
    return inter / union


def _is_inside(inner, outer, threshold=0.80):
    """Check if inner box is mostly inside outer box."""
    inter = _intersection_area(inner, outer)
    inner_area = _box_area(inner)
    if inner_area == 0:
        return False
    return inter / inner_area >= threshold


def remove_overlap_yolo(boxes_elem, iou_threshold):
    """Remove overlapping YOLO boxes, keeping the smaller one."""
    if not boxes_elem:
        return []
    filtered = []
    for i, box_i in enumerate(boxes_elem):
        is_valid = True
        for j, box_j in enumerate(boxes_elem):
            if i != j and _iou(box_i["bbox"], box_j["bbox"]) > iou_threshold:
                if _box_area(box_i["bbox"]) > _box_area(box_j["bbox"]):
                    is_valid = False
                    break
        if is_valid:
            filtered.append(box_i)
    return filtered


def remove_overlap_new(yolo_boxes, iou_threshold, ocr_boxes):
    """OCR-aware overlap removal (from test_annotate.ipynb).

    Merges YOLO icon detections with OCR text detections:
    - If OCR text is inside a YOLO icon: absorb text into the icon
    - If YOLO icon is inside OCR text: drop the icon
    - Otherwise: keep both
    """
    if not yolo_boxes and not ocr_boxes:
        return []

    # Start with OCR boxes
    output = list(ocr_boxes)

    # Deduplicate YOLO boxes among themselves
    yolo_deduped = remove_overlap_yolo(yolo_boxes, iou_threshold)

    for yolo_box in yolo_deduped:
        absorbed = False
        for i, ocr_box in enumerate(output):
            if ocr_box["type"] != "text":
                continue

            # OCR text inside YOLO icon → absorb text, keep YOLO
            if _is_inside(ocr_box["bbox"], yolo_box["bbox"]):
                yolo_box["content"] = ocr_box["content"]
                output.pop(i)
                output.append(yolo_box)
                absorbed = True
                break

            # YOLO icon inside OCR text → drop YOLO
            if _is_inside(yolo_box["bbox"], ocr_box["bbox"]):
                absorbed = True
                break

        if not absorbed:
            output.append(yolo_box)

    return output


# ==================== DETECTION ====================

def detect_ui_elements_yolo(image, yolo_model):
    """Run YOLO detection, return elements in pixel xyxy format."""
    w, h = image.size
    result = yolo_model.predict(source=image, conf=BOX_THRESHOLD, iou=0.1, verbose=False)

    raw_boxes = result[0].boxes.xyxy
    if len(raw_boxes) == 0:
        return []

    elements = []
    for box in raw_boxes.tolist():
        x1, y1, x2, y2 = box
        # Filter taskbar
        center_y_norm = ((y1 + y2) / 2) / h
        if center_y_norm > 0.93:
            continue
        # Filter tiny boxes
        bh_norm = (y2 - y1) / h
        bw_norm = (x2 - x1) / w
        if bh_norm * bw_norm < MIN_BOX_AREA:
            continue

        elements.append({
            "type": "icon",
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "interactivity": True,
            "content": None,
        })

    return elements


def detect_ui_elements_with_ocr(image, yolo_model, ocr_reader):
    """Combined YOLO + OCR detection pipeline.

    Returns:
        bboxes_norm: list of (y, x, h, w) normalized — for MarkHelper drawing
        filtered_elements: list of element dicts with bbox/content — for metadata
    """
    width, height = image.size

    # 1. Run OCR
    ocr_elements = run_ocr(image, ocr_reader)

    # 2. Run YOLO
    yolo_elements = detect_ui_elements_yolo(image, yolo_model)

    # 3. OCR-aware merge
    filtered_elements = remove_overlap_new(yolo_elements, OVERLAP_IOU_THRESHOLD, ocr_elements)

    if not filtered_elements:
        return [], []

    # 4. Convert to normalized (y, x, h, w) for MarkHelper — skip degenerate boxes
    bboxes_norm = []
    valid_elements = []
    for elem in filtered_elements:
        x1, y1, x2, y2 = elem["bbox"]
        if x2 <= x1 or y2 <= y1:
            continue
        bboxes_norm.append((
            y1 / height,
            x1 / width,
            (y2 - y1) / height,
            (x2 - x1) / width,
        ))
        valid_elements.append(elem)
    filtered_elements = valid_elements

    # 5. Truncate to MAX_MARKS
    if len(bboxes_norm) > MAX_MARKS:
        bboxes_norm = bboxes_norm[:MAX_MARKS]
        filtered_elements = filtered_elements[:MAX_MARKS]

    return bboxes_norm, filtered_elements


# ==================== COORDINATE EXTRACTION ====================

def extract_coordinates_from_code(code):
    """Extract (x, y) coordinates from pyautogui code string."""
    match = re.search(r"(?:click|Click|doubleClick|rightClick|middleClick)\(x=([\d.]+),\s*y=([\d.]+)\)", code)
    if match:
        return float(match.group(1)), float(match.group(2))
    match = re.search(r"moveTo\(x=([\d.]+),\s*y=([\d.]+)\)", code)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None



def build_image_coords_map(jsonl_path):
    """Build image_name → coordinates from JSONL trajectories (Word-style)."""
    image_coords = {}
    image_prev_coords = {}

    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            prev_coords = None
            for step in entry["traj"]:
                code = step["value"].get("code", "")
                image = step["image"]
                coords = extract_coordinates_from_code(code)

                if coords:
                    image_coords[image] = coords
                image_prev_coords[image] = prev_coords

                if coords:
                    prev_coords = coords

    return image_coords, image_prev_coords



# ==================== MARK MATCHING ====================

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


# ==================== OUTPUT NORMALIZATION ====================

def normalize_action_output(action):
    """Normalize action dict: null→'None', coords→removed from MARK."""
    mark = action.get("MARK")
    if mark is None or mark == "None" or isinstance(mark, list):
        action["MARK"] = "None"
    # int mark stays as-is

    value = action.get("VALUE")
    if value is None:
        action["VALUE"] = "None"

    return action


def build_mark_metadata(bboxes_norm, filtered_elements, width, height):
    """Build mark metadata dict: mark_id → {center, bbox, ocr_text}."""
    marks = {}
    for idx, (elem, (y, x, h, w)) in enumerate(zip(filtered_elements, bboxes_norm)):
        # Pixel coordinates
        px1, py1, px2, py2 = elem["bbox"]
        cx = (px1 + px2) // 2
        cy = (py1 + py2) // 2
        marks[str(idx)] = {
            "center": [cx, cy],
            "bbox": [px1, py1, px2, py2],
            "norm_center": [round(x + w / 2, 4), round(y + h / 2, 4)],
            "ocr_text": elem.get("content"),
        }
    return marks


# ==================== MAIN PROCESSING ====================

def process_app(app_name, config, yolo_model, ocr_reader):
    """Process a single app (word/excel/powerpoint)."""
    print(f"\n{'=' * 70}")
    print(f"  Processing: {app_name.upper()}")
    print(f"{'=' * 70}")

    output_dir = config["output_dir"]
    output_image_dir = os.path.join(output_dir, "images")
    output_json = os.path.join(output_dir, f"{app_name}_som.json")
    output_metadata = os.path.join(output_dir, "mark_metadata.json")
    os.makedirs(output_image_dir, exist_ok=True)

    # Step 1: Build coordinate maps from JSONL (same source for all apps)
    print("  Building coordinate maps from JSONL...")
    image_coords, image_prev_coords = build_image_coords_map(config["jsonl"])
    print(f"    Images with coords: {len(image_coords)}")

    # Step 2: Load pre-filtered mind2web samples (4 actions only)
    print(f"  Loading: {config['mind2web']}")
    with open(config["mind2web"]) as f:
        samples = json.load(f)
    print(f"    Total samples: {len(samples)}")

    # Step 3: Process
    mark_helper = MarkHelper()
    mark_helper.min_font_size = MARK_FONT_MIN
    mark_helper.max_font_size = MARK_FONT_MAX

    image_cache = {}  # image_name → (bboxes_norm, som_image_name) or None
    output_data = []
    all_distances = []
    start_idx = 0
    stats = {
        "total": 0,
        "matched_inside": 0,
        "matched_nearest": 0,
        "no_coords": 0,
        "no_elements": 0,
        "mark_not_needed": 0,
    }

    # Resume: load previous progress if exists
    resume_file = os.path.join(output_dir, f"{app_name}_resume_idx.txt")

    if os.path.exists(output_metadata):
        with open(output_metadata) as f:
            mark_metadata = json.load(f)
        print(f"    Resumed metadata: {len(mark_metadata)} images already processed")
    else:
        mark_metadata = {}

    if os.path.exists(resume_file) and os.path.exists(output_json):
        with open(resume_file) as f:
            start_idx = int(f.read().strip())
        with open(output_json) as f:
            output_data = json.load(f)
        print(f"    Resumed: {len(output_data)} samples saved, continuing from index {start_idx}")
    else:
        output_data = []

    for idx in tqdm(range(start_idx, len(samples)), desc=f"  {app_name}", initial=start_idx, total=len(samples)):
        sample = samples[idx]
        stats["total"] += 1
        image_name = sample["image"]

        # Parse action
        action = json.loads(sample["conversations"][1]["value"])
        action_type = action.get("ACTION", "")

        # Run detection on image (with caching)
        som_image_name = f"{app_name}_som_{image_name}"

        if image_name in image_cache:
            cached = image_cache[image_name]
            if cached is None:
                continue
            bboxes_norm, som_image_name = cached
        elif som_image_name in mark_metadata:
            # Resume: image was already processed in a previous run
            meta = mark_metadata[som_image_name]
            bboxes_norm = []
            for mark_id in sorted(meta["marks"].keys(), key=int):
                m = meta["marks"][mark_id]
                img_w, img_h = meta["image_size"]
                px1, py1, px2, py2 = m["bbox"]
                bboxes_norm.append((
                    py1 / img_h, px1 / img_w,
                    (py2 - py1) / img_h, (px2 - px1) / img_w,
                ))
            image_cache[image_name] = (bboxes_norm, som_image_name)
        else:
            image_path = os.path.join(IMAGE_DIR, image_name)
            if not os.path.exists(image_path):
                stats["no_elements"] += 1
                image_cache[image_name] = None
                continue

            try:
                image = Image.open(image_path).convert("RGB")
                bboxes_norm, filtered_elements = detect_ui_elements_with_ocr(
                    image, yolo_model, ocr_reader
                )
            except Exception as e:
                print(f"    WARNING: Failed on {image_name}: {e}")
                stats["no_elements"] += 1
                image_cache[image_name] = None
                continue

            if bboxes_norm:
                # Draw SoM marks
                som_image = plot_boxes_with_marks(
                    image.copy(), bboxes_norm, mark_helper,
                    edgecolor=(255, 0, 0), linewidth=1,
                    normalized_to_pixel=True, add_mark=True,
                )
                som_image.save(os.path.join(output_image_dir, som_image_name))

                # Build mark metadata
                width, height = image.size
                marks_info = build_mark_metadata(
                    bboxes_norm, filtered_elements, width, height
                )
                mark_metadata[som_image_name] = {
                    "num_marks": len(bboxes_norm),
                    "image_size": [width, height],
                    "marks": marks_info,
                }
            else:
                stats["no_elements"] += 1
                image_cache[image_name] = None
                continue

            image_cache[image_name] = (bboxes_norm, som_image_name)

        # Determine MARK
        mark = "None"
        if action_type in ACTIONS_NEED_MARK and bboxes_norm:
            coords = None

            if action_type == "TYPE":
                # TYPE: use previous step's click (the text field)
                coords = image_prev_coords.get(image_name)
                if not coords:
                    coords = image_coords.get(image_name)
            else:
                # CLICK/DOUBLE_CLICK/RIGHT_CLICK: use this step's coords
                coords = image_coords.get(image_name)

            if coords:
                mark_id, distance = match_click_to_mark(coords[0], coords[1], bboxes_norm)
                if mark_id is not None:
                    all_distances.append(distance)
                    if distance == 0.0:
                        stats["matched_inside"] += 1
                    else:
                        stats["matched_nearest"] += 1
                    mark = mark_id
            else:
                stats["no_coords"] += 1
        elif action_type not in ACTIONS_NEED_MARK:
            stats["mark_not_needed"] += 1

        # Build normalized output
        action["MARK"] = mark if mark != "None" else "None"
        action = normalize_action_output(action)

        output_data.append({
            "id": f"{app_name}_{idx}",
            "image": som_image_name,
            "conversations": [
                sample["conversations"][0],
                {"from": "assistant", "value": json.dumps(action)},
            ],
        })

        # Periodic checkpoint — save output JSON, metadata, and resume index
        if (idx + 1) % 500 == 0:
            print(f"    Checkpoint at {idx+1}: {len(output_data)} samples, {len(mark_metadata)} images...")
            with open(output_json, "w") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            with open(output_metadata, "w") as f:
                json.dump(mark_metadata, f, indent=2, ensure_ascii=False)
            with open(resume_file, "w") as f:
                f.write(str(idx + 1))

    # Save final outputs
    print(f"\n  Writing: {output_json}")
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"  Writing: {output_metadata}")
    with open(output_metadata, "w") as f:
        json.dump(mark_metadata, f, indent=2, ensure_ascii=False)

    # Clean up resume file — completed successfully
    if os.path.exists(resume_file):
        os.remove(resume_file)
        print(f"  Cleaned up resume file")

    # Print stats
    print(f"\n  {'=' * 60}")
    print(f"  {app_name.upper()} RESULTS")
    print(f"  {'=' * 60}")
    print(f"    Total samples:             {stats['total']}")
    print(f"    Matched inside bbox:       {stats['matched_inside']}")
    print(f"    Matched nearest:           {stats['matched_nearest']}")
    print(f"    No coordinates:            {stats['no_coords']}")
    print(f"    No elements detected:      {stats['no_elements']}")
    print(f"    MARK not needed (action):  {stats['mark_not_needed']}")
    print(f"    Output samples:            {len(output_data)}")
    valid_caches = [v for v in image_cache.values() if v is not None]
    if valid_caches:
        avg = np.mean([len(v[0]) for v in valid_caches])
        print(f"    Avg elements per image:    {avg:.1f}")
    if all_distances:
        print(f"    Match distance stats:")
        print(f"      Mean:   {np.mean(all_distances):.4f}")
        print(f"      Median: {np.median(all_distances):.4f}")
        print(f"      <1%:    {100 * sum(1 for d in all_distances if d < 0.01) / len(all_distances):.1f}%")
        print(f"      <2%:    {100 * sum(1 for d in all_distances if d < 0.02) / len(all_distances):.1f}%")
        print(f"      <5%:    {100 * sum(1 for d in all_distances if d < 0.05) / len(all_distances):.1f}%")
    print(f"    SoM images: {output_image_dir}")
    print(f"    JSON: {output_json}")
    print(f"    Metadata: {output_metadata}")
    print(f"  {'=' * 60}")

    return output_data, stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess Office apps for SoM training")
    parser.add_argument(
        "--app",
        choices=["word", "excel", "powerpoint", "all"],
        default="all",
        help="Which app to process (default: all)",
    )
    args = parser.parse_args()

    # Validate
    if not os.path.exists(OMNIPARSER_MODEL_PATH):
        raise FileNotFoundError(f"YOLO weights not found: {OMNIPARSER_MODEL_PATH}")

    # Load models once
    print("Loading YOLO (OmniParser)...")
    yolo_model = YOLO(OMNIPARSER_MODEL_PATH)
    print("Loading EasyOCR...")
    ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    print("Models loaded!\n")

    # Process selected app(s)
    apps = list(APP_CONFIGS.keys()) if args.app == "all" else [args.app]

    all_stats = {}
    for app_name in apps:
        config = APP_CONFIGS[app_name]

        # Check input files exist
        if not os.path.exists(config["mind2web"]):
            print(f"WARNING: Skipping {app_name} — mind2web file not found: {config['mind2web']}")
            continue
        if not os.path.exists(config["jsonl"]):
            print(f"WARNING: Skipping {app_name} — JSONL file not found: {config['jsonl']}")
            continue

        _, stats = process_app(app_name, config, yolo_model, ocr_reader)
        all_stats[app_name] = stats

    # Overall summary
    if len(all_stats) > 1:
        print(f"\n{'=' * 70}")
        print("OVERALL SUMMARY")
        print(f"{'=' * 70}")
        for app, stats in all_stats.items():
            matched = stats["matched_inside"] + stats["matched_nearest"]
            print(f"  {app:>12}: {stats['total']} samples, {matched} matched")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
