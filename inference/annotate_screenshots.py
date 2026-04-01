"""
Run OmniParser on test case screenshots to generate SoM annotated images.

Reads from testcase/ folder structure:
    testcase/
      test0/ -> input_0.png + prompt_1.txt, prompt_2.txt, ...
      test1/ -> input_1.png + prompt_1.txt, prompt_2.txt, ...
      ...

Outputs:
    annotated/ -> SoM annotated images + marks_info.json
    test_cases.json -> all images + prompts ready for evaluation

Usage:
    cd /home/thaole/thao_le/Magma
    python inference/annotate_screenshots.py
"""

import json
import os
import sys

from PIL import Image
from ultralytics import YOLO

# ============ CONFIGURATION ============
PROJECT_ROOT = "/home/thaole/thao_le/Magma"
TESTCASE_DIR = "/home/thaole/thao_le/Magma/inference/tests/testcase"
OUTPUT_DIR = "/home/thaole/thao_le/Magma/inference/tests/annotated"
TEST_CASES_JSON = "/home/thaole/thao_le/Magma/inference/tests/test_cases.json"
OMNIPARSER_MODEL_PATH = "/home/thaole/thao_le/Magma/weights/icon_detect/model.pt"

# OmniParser params
BOX_THRESHOLD = 0.1
OVERLAP_IOU = 0.7
MAX_MARKS = 100
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

    # Scan testcase/ folder for test subfolders
    test_folders = sorted([
        d for d in os.listdir(TESTCASE_DIR)
        if os.path.isdir(os.path.join(TESTCASE_DIR, d))
    ])
    print(f"Found {len(test_folders)} test folders in {TESTCASE_DIR}")
    print(f"Params: box_threshold={BOX_THRESHOLD}, overlap_iou={OVERLAP_IOU}, max_marks={MAX_MARKS}")

    all_marks_info = {}
    template = []
    total_prompts = 0

    for folder_name in test_folders:
        folder_path = os.path.join(TESTCASE_DIR, folder_name)

        # Find the image in this folder
        extensions = (".png", ".jpg", ".jpeg", ".bmp")
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]
        if not images:
            print(f"  WARNING: no image found in {folder_name}, skipping")
            continue
        img_name = images[0]
        img_path = os.path.join(folder_path, img_name)

        print(f"\n  [{folder_name}] {img_name}")

        # Annotate
        annotated, mark_to_center, num_marks = annotate_single(img_path, yolo_model)
        print(f"    Marks: {num_marks}")

        # Save annotated image (use folder name as filename for clarity)
        annotated_name = f"{folder_name}.png"
        annotated.save(os.path.join(OUTPUT_DIR, annotated_name))

        # Save marks info
        serializable = {}
        for mark_id, info in mark_to_center.items():
            serializable[str(mark_id)] = {
                "center_x": info["center"][0],
                "center_y": info["center"][1],
                "bbox": list(info["bbox"]),
            }
        all_marks_info[annotated_name] = {
            "num_marks": num_marks,
            "marks": serializable,
        }

        # Read prompts from the same folder
        prompt_files = sorted([
            f for f in os.listdir(folder_path)
            if f.startswith("prompt") and f.endswith(".txt")
        ])
        prompts_list = []
        for pf in prompt_files:
            with open(os.path.join(folder_path, pf)) as f:
                prompt_text = f.read().strip()
            if prompt_text:
                prompts_list.append({
                    "prompt": prompt_text,
                    "expected": {"ACTION": "CLICK", "MARK": 0, "VALUE": "None"},
                })

        total_prompts += len(prompts_list)
        print(f"    Prompts: {len(prompts_list)}")

        template.append({
            "test_folder": folder_name,
            "image": img_path,
            "annotated_image": os.path.join(OUTPUT_DIR, annotated_name),
            "num_marks": num_marks,
            "prompts": prompts_list,
        })

    # Save marks info
    marks_json_path = os.path.join(OUTPUT_DIR, "marks_info.json")
    with open(marks_json_path, "w") as f:
        json.dump(all_marks_info, f, indent=2)

    # Save test cases
    with open(TEST_CASES_JSON, "w") as f:
        json.dump(template, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Total: {len(test_folders)} images, {total_prompts} prompts")
    print(f"Annotated images: {OUTPUT_DIR}")
    print(f"Mark coordinates: {marks_json_path}")
    print(f"Test cases:       {TEST_CASES_JSON}")
    print(f"")
    print(f"Next steps:")
    print(f"  1. Open annotated images to see mark numbers")
    print(f"  2. Edit {TEST_CASES_JSON} — fill in expected ACTION/MARK/VALUE")
    print(f"  3. Run: python /home/thaole/thao_le/Magma/inference/run_e2e.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
