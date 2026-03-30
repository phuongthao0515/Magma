"""
Step 1: Run OmniParser on screenshots to generate SoM annotated images.

After running this, manually inspect the annotated images to fill in expected
MARK IDs in the test cases JSON.

Usage:
    python /home/thaole/thao_le/Magma/inference/annotate_screenshots.py
"""

import json
import os
import sys

from PIL import Image
from ultralytics import YOLO

# ============ CONFIGURATION ============
PROJECT_ROOT = "/home/thaole/thao_le/Magma"
INPUT_DIR = "/home/thaole/thao_le/Magma/inference/tests/screenshots"
PROMPTS_DIR = "/home/thaole/thao_le/Magma/inference/tests/prompts"
OUTPUT_DIR = "/home/thaole/thao_le/Magma/inference/tests/annotated"
TEST_CASES_JSON = "/home/thaole/thao_le/Magma/inference/tests/test_cases.json"
OMNIPARSER_MODEL_PATH = "/home/thaole/thao_le/Magma/weights/icon_detect/model.pt"

# OmniParser params — override these for denser/sparser detection
BOX_THRESHOLD = 0.1        # YOLO confidence (lower = more detections)
OVERLAP_IOU = 0.7          # overlap removal (lower = keep more boxes)
MAX_MARKS = 100            # cap mark count
MARK_FONT_MIN = 12
MARK_FONT_MAX = 14
# =======================================

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data-preprocessing", "SoM annotations"))

from preprocess_word_som import detect_ui_elements
from agents.ui_agent.util.som import MarkHelper, plot_boxes_with_marks


def annotate_single(image_path, yolo_model):
    """Annotate one screenshot. Returns annotated image + mark coordinate mappings."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    bboxes = detect_ui_elements(
        image, yolo_model,
        box_threshold=BOX_THRESHOLD,
        overlap_iou_threshold=OVERLAP_IOU,
    )

    if MAX_MARKS and len(bboxes) > MAX_MARKS:
        bboxes = bboxes[:MAX_MARKS]

    if not bboxes:
        return image.copy(), {}, 0

    mark_helper = MarkHelper()
    mark_helper.min_font_size = MARK_FONT_MIN
    mark_helper.max_font_size = MARK_FONT_MAX

    annotated = plot_boxes_with_marks(
        image.copy(), bboxes, mark_helper,
        edgecolor=(255, 0, 0), linewidth=1,
        normalized_to_pixel=True, add_mark=True,
    )

    mark_to_center = {}
    for idx, (y_norm, x_norm, h_norm, w_norm) in enumerate(bboxes):
        x1 = int(x_norm * width)
        y1 = int(y_norm * height)
        x2 = int((x_norm + w_norm) * width)
        y2 = int((y_norm + h_norm) * height)
        mark_to_center[idx] = {
            "center": ((x1 + x2) // 2, (y1 + y2) // 2),
            "bbox": (x1, y1, x2, y2),
        }

    return annotated, mark_to_center, len(bboxes)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading YOLO model: {OMNIPARSER_MODEL_PATH}")
    yolo_model = YOLO(OMNIPARSER_MODEL_PATH)

    extensions = (".png", ".jpg", ".jpeg", ".bmp")
    images = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(extensions)
    ])
    print(f"Found {len(images)} screenshots in {INPUT_DIR}")
    print(f"Params: box_threshold={BOX_THRESHOLD}, overlap_iou={OVERLAP_IOU}, max_marks={MAX_MARKS}")

    all_marks_info = {}

    for img_name in images:
        img_path = os.path.join(INPUT_DIR, img_name)
        print(f"\n  Processing: {img_name}")

        annotated, mark_to_center, num_marks = annotate_single(img_path, yolo_model)
        print(f"    Marks: {num_marks}")

        annotated.save(os.path.join(OUTPUT_DIR, img_name))

        serializable = {}
        for mark_id, info in mark_to_center.items():
            serializable[str(mark_id)] = {
                "center_x": info["center"][0],
                "center_y": info["center"][1],
                "bbox": list(info["bbox"]),
            }
        all_marks_info[img_name] = {
            "num_marks": num_marks,
            "marks": serializable,
        }

    # Save mark coordinate mapping
    marks_json_path = os.path.join(OUTPUT_DIR, "marks_info.json")
    with open(marks_json_path, "w") as f:
        json.dump(all_marks_info, f, indent=2)

    # Read prompts from PROMPTS_DIR and generate test cases
    template = []
    for img_name in images:
        img_stem = os.path.splitext(img_name)[0]
        prompts_list = []

        # Find prompt folder matching this image (folder name contains image stem)
        for folder_name in sorted(os.listdir(PROMPTS_DIR)):
            folder_path = os.path.join(PROMPTS_DIR, folder_name)
            if os.path.isdir(folder_path) and img_stem in folder_name:
                prompt_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".txt"))
                for pf in prompt_files:
                    with open(os.path.join(folder_path, pf)) as f:
                        prompt_text = f.read().strip()
                    if prompt_text:
                        prompts_list.append({
                            "prompt": prompt_text,
                            "expected": {"ACTION": "CLICK", "MARK": 0, "VALUE": "None"},
                        })
                break

        if not prompts_list:
            print(f"  WARNING: no prompts found for {img_name} in {PROMPTS_DIR}")

        template.append({
            "image": os.path.join(INPUT_DIR, img_name),
            "annotated_image": os.path.join(OUTPUT_DIR, img_name),
            "num_marks": all_marks_info[img_name]["num_marks"],
            "prompts": prompts_list,
        })

    test_cases_path = TEST_CASES_JSON
    with open(test_cases_path, "w") as f:
        json.dump(template, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Annotated images: {OUTPUT_DIR}")
    print(f"Mark coordinates: {marks_json_path}")
    print(f"Test cases:       {test_cases_path}")
    print(f"")
    print(f"Next steps:")
    print(f"  1. Open annotated images to see mark numbers")
    print(f"  2. Edit {test_cases_path} — fill in prompts + expected ACTION/MARK/VALUE")
    print(f"  3. Run: python /home/thaole/thao_le/Magma/inference/run_e2e.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
