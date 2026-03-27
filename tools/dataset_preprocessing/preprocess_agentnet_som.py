"""
Preprocess AgentNet CLICK samples with SoM (YOLO + EasyOCR + MarkHelper).

Pipeline:
1. Read original AgentNet JSONL with pyautogui (x, y) coordinates
2. Run YOLO (icon detection) + EasyOCR (text detection) on each image
3. Merge detections, remove overlaps, annotate with MarkHelper (same style as Mind2Web)
4. Match click (x, y) to nearest bounding box → assign MARK ID
5. Save SoM-annotated image + Mind2Web-style JSON

Usage:
    cd /home/thaole/thao_le/Magma
    python data-preprocessing/preprocess_agentnet_som.py
"""

import os
import sys
import json
import re
import numpy as np
import torch
import easyocr
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm

# ============ CONFIGURATION ============
AGENTNET_JSONL = "/home/thaole/Downloads/agentnet_win_mac_18k.jsonl"
IMAGE_DIR = "/home/thaole/thao_le/Magma/datasets/agentnet/office_images"
OUTPUT_IMAGE_DIR = "/home/thaole/thao_le/Magma/datasets/agentnet/images_som"
OUTPUT_JSON = "/home/thaole/thao_le/Magma/datasets/agentnet/agentnet_som_2000.json"
MAX_SAMPLES = 2000

# OmniParser YOLO weights (downloaded from microsoft/OmniParser-v2.0)
WEIGHTS_DIR = "/home/thaole/thao_le/Magma/weights"
YOLO_MODEL_PATH = os.path.join(WEIGHTS_DIR, "icon_detect", "model.pt")
BOX_THRESHOLD = 0.05
IOU_THRESHOLD = 0.7
# =======================================

# Add project root for MarkHelper import
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agents.ui_agent.util.som import MarkHelper, plot_boxes_with_marks


def download_weights():
    """Download OmniParser-v2.0 weights if not present."""
    if os.path.exists(YOLO_MODEL_PATH):
        print(f"YOLO weights found at: {YOLO_MODEL_PATH}")
        return
    print("Downloading OmniParser-v2.0 weights...")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="microsoft/OmniParser-v2.0", local_dir=WEIGHTS_DIR)
    print(f"Weights downloaded to: {WEIGHTS_DIR}")


def detect_yolo(image, yolo_model):
    """Detect UI elements (icons, buttons) with YOLO.
    Returns list of (y, x, h, w) normalized bboxes.
    """
    w, h = image.size
    result = yolo_model.predict(source=image, conf=BOX_THRESHOLD, iou=0.1, verbose=False)
    boxes_pixel = result[0].boxes.xyxy

    bboxes = []
    for box in boxes_pixel:
        x1, y1, x2, y2 = box.tolist()
        bboxes.append((y1 / h, x1 / w, (y2 - y1) / h, (x2 - x1) / w))
    return bboxes


def detect_ocr(image, ocr_reader):
    """Detect text elements with EasyOCR.
    Returns list of (y, x, h, w) normalized bboxes and texts.
    """
    w, h = image.size
    image_np = np.array(image)
    results = ocr_reader.readtext(image_np, text_threshold=0.5)

    bboxes = []
    texts = []
    for coords, text, conf in results:
        x1, y1 = coords[0]
        x2, y2 = coords[2]
        bboxes.append((y1 / h, x1 / w, (y2 - y1) / h, (x2 - x1) / w))
        texts.append(text)
    return bboxes, texts


def remove_overlaps(yolo_bboxes, ocr_bboxes, iou_threshold=0.7):
    """Merge YOLO + OCR bboxes, removing overlapping ones.
    OCR bboxes are prioritized. All bboxes in (y, x, h, w) normalized format.
    """
    def box_iou(b1, b2):
        y1, x1, h1, w1 = b1
        y2, x2, h2, w2 = b2
        yi = max(y1, y2)
        xi = max(x1, x2)
        ya = min(y1 + h1, y2 + h2)
        xa = min(x1 + w1, x2 + w2)
        inter = max(0, ya - yi) * max(0, xa - xi)
        area1 = h1 * w1
        area2 = h2 * w2
        union = area1 + area2 - inter + 1e-6
        return max(inter / union, inter / (area1 + 1e-6), inter / (area2 + 1e-6))

    merged = list(ocr_bboxes)
    for yolo_box in yolo_bboxes:
        if yolo_box[2] * yolo_box[3] <= 0:
            continue
        if not any(box_iou(yolo_box, existing) > iou_threshold for existing in merged):
            merged.append(yolo_box)
    return merged


def match_click_to_mark(click_x, click_y, bboxes):
    """Match normalized (x, y) click to nearest bbox.

    Args:
        click_x, click_y: Normalized click coordinates (0-1)
        bboxes: List of (y, x, h, w) normalized bboxes

    Returns:
        mark_id: Index of matched bbox, or None
        distance: 0.0 if inside, else distance to nearest center
    """
    if not bboxes:
        return None, float('inf')

    # First pass: click inside bbox (pick smallest)
    best_mark = None
    best_area = float('inf')
    for i, (y, x, h, w) in enumerate(bboxes):
        if x <= click_x <= x + w and y <= click_y <= y + h:
            area = h * w
            if area < best_area:
                best_mark = i
                best_area = area
    if best_mark is not None:
        return best_mark, 0.0

    # Second pass: nearest center
    best_distance = float('inf')
    for i, (y, x, h, w) in enumerate(bboxes):
        cx, cy = x + w / 2, y + h / 2
        dist = np.sqrt((click_x - cx) ** 2 + (click_y - cy) ** 2)
        if dist < best_distance:
            best_distance = dist
            best_mark = i
    return best_mark, best_distance


def build_mind2web_prompt(task_description, previous_actions=""):
    """Build Mind2Web-style instruction prompt."""
    prompt = (
        "<image>\n"
        "Imagine that you are imitating humans doing web navigation for a task step by step. "
        "At each stage, you can see the webpage like humans by a screenshot and know the previous actions "
        "before the current step decided by yourself through recorded history. You need to decide on the "
        "following action to take. You can click an element with the mouse, select an option, or type text "
        "with the keyboard. The output format should be a dictionary like: \n"
        '{"ACTION": "CLICK" or "TYPE" or "SELECT", "MARK": a numeric id, e.g., 5, '
        '"VALUE": a string value for the action if applicable, otherwise None}.\n'
        f"You are asked to complete the following task: {task_description}. "
        f"The previous actions you have taken: \n"
        f"{previous_actions if previous_actions else '(None)'}\n"
        "For your convenience, I have labeled the candidates with numeric marks and "
        "bounding boxes on the screenshot. What is the next action you would take?\n"
    )
    return prompt


def extract_click_samples(jsonl_path, max_samples):
    """Extract CLICK samples with coordinates from AgentNet JSONL."""
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            if len(samples) >= max_samples:
                break
            data = json.loads(line)
            task = data['instruction']
            prev_actions = []

            for step in data['traj']:
                if len(samples) >= max_samples:
                    break

                code = step['value'].get('code', '')
                if 'pyautogui.click(' not in code:
                    action_desc = step['value'].get('action', '')
                    if action_desc:
                        prev_actions.append(action_desc)
                    continue

                match = re.search(r'x=([\d.]+),\s*y=([\d.]+)', code)
                if not match:
                    continue

                click_x = float(match.group(1))
                click_y = float(match.group(2))
                image_name = step['image']
                image_path = os.path.join(IMAGE_DIR, image_name)

                if not os.path.exists(image_path):
                    continue

                samples.append({
                    'task_id': data['task_id'],
                    'task': task,
                    'image_name': image_name,
                    'image_path': image_path,
                    'click_x': click_x,
                    'click_y': click_y,
                    'action_description': step['value'].get('action', ''),
                    'previous_actions': '; '.join(prev_actions) if prev_actions else '',
                })

                action_desc = step['value'].get('action', '')
                if action_desc:
                    prev_actions.append(action_desc)

    return samples


def main():
    download_weights()

    print(f"Extracting up to {MAX_SAMPLES} CLICK samples from AgentNet...")
    samples = extract_click_samples(AGENTNET_JSONL, MAX_SAMPLES)
    print(f"Extracted {len(samples)} CLICK samples")

    # Load models
    print("Loading YOLO model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("Initializing EasyOCR...")
    ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)

    mark_helper = MarkHelper()
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    output_data = []
    stats = {
        'total': 0,
        'matched_inside': 0,
        'matched_nearest': 0,
        'no_elements': 0,
        'too_far': 0,
    }

    # Cache: avoid re-processing the same image
    # image_name -> (bboxes, som_image_name)
    image_cache = {}

    for idx, sample in enumerate(tqdm(samples, desc="Processing")):
        stats['total'] += 1
        image_name = sample['image_name']

        if image_name in image_cache:
            bboxes, som_image_name = image_cache[image_name]
        else:
            image = Image.open(sample['image_path']).convert('RGB')

            # Detect elements
            yolo_bboxes = detect_yolo(image, yolo_model)
            ocr_bboxes, ocr_texts = detect_ocr(image, ocr_reader)

            # Merge and remove overlaps
            bboxes = remove_overlaps(yolo_bboxes, ocr_bboxes, IOU_THRESHOLD)

            if len(bboxes) == 0:
                stats['no_elements'] += 1
                continue

            # Annotate with MarkHelper (same style as Mind2Web)
            som_image = plot_boxes_with_marks(
                image.copy(), bboxes, mark_helper,
                edgecolor=(255, 0, 0), linewidth=2,
                normalized_to_pixel=True, add_mark=True
            )

            # Save annotated image
            som_image_name = f"agentnet_som_{idx:06d}.png"
            som_image_path = os.path.join(OUTPUT_IMAGE_DIR, som_image_name)
            som_image.save(som_image_path)

            image_cache[image_name] = (bboxes, som_image_name)

        # Match click to mark
        mark_id, distance = match_click_to_mark(
            sample['click_x'], sample['click_y'], bboxes
        )

        if mark_id is None:
            stats['no_elements'] += 1
            continue

        if distance == 0.0:
            stats['matched_inside'] += 1
        elif distance < 0.05:
            stats['matched_nearest'] += 1
        else:
            stats['too_far'] += 1

        # Build Mind2Web-style entry
        prompt = build_mind2web_prompt(sample['task'], sample['previous_actions'])
        gt = json.dumps({"ACTION": "CLICK", "MARK": mark_id, "VALUE": None})

        entry = {
            "id": f"agentnet_{idx:06d}",
            "image": f"images_som/{som_image_name}",
            "conversations": [
                {"from": "user", "value": prompt},
                {"from": "assistant", "value": gt},
            ],
        }
        output_data.append(entry)

    # Save output JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Total CLICK samples processed: {stats['total']}")
    print(f"  Click inside bbox (exact match): {stats['matched_inside']}")
    print(f"  Click near bbox (<5% distance):  {stats['matched_nearest']}")
    print(f"  Click far from any bbox (>5%):   {stats['too_far']}")
    print(f"  No elements detected:            {stats['no_elements']}")
    print(f"  Output samples:                  {len(output_data)}")
    if image_cache:
        avg_elements = np.mean([len(v[0]) for v in image_cache.values()])
        print(f"  Avg elements per image:          {avg_elements:.1f}")
    print(f"\nSoM images saved to: {OUTPUT_IMAGE_DIR}")
    print(f"JSON saved to: {OUTPUT_JSON}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
