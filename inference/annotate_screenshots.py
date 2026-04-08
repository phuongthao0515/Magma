"""
Run YOLO + OCR on test case screenshots to generate SoM annotated images.
Uses the SAME detection pipeline as preprocess_office_som.py for consistency.

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

import easyocr
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# ============ CONFIGURATION ============
PROJECT_ROOT = "/home/thaole/thao_le/Magma"
TESTCASE_DIR = "/home/thaole/thao_le/Magma/inference/tests/testcase"
OUTPUT_DIR = "/home/thaole/thao_le/Magma/inference/tests/annotated"
TEST_CASES_JSON = "/home/thaole/thao_le/Magma/inference/tests/test_cases.json"
OMNIPARSER_MODEL_PATH = "/home/thaole/thao_le/Magma/weights/icon_detect/model.pt"

# Same params as preprocess_office_som.py
BOX_THRESHOLD = 0.25
OVERLAP_IOU_THRESHOLD = 0.7
MAX_MARKS = 100
MARK_FONT_MIN = 12
MARK_FONT_MAX = 14
OCR_TEXT_THRESHOLD = 0.9
OCR_MAX_AREA_RATIO = 0.05
GRID_SIZE = 100
# =======================================

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data-preprocessing", "SoM annotations"))

from agents.ui_agent.util.som import MarkHelper, plot_boxes_with_marks
from preprocess_office_som import (
    detect_ui_elements_with_ocr,
    build_mark_metadata,
)


def format_marks_text(marks_dict, grid_size=GRID_SIZE):
    """Format marks into text for prompt injection. Same logic as inject_mark_coords.py."""
    entries = []
    for mark_id in sorted(marks_dict.keys(), key=int):
        mark = marks_dict[mark_id]
        norm_x, norm_y = mark["norm_center"]
        gx = min(int(norm_x * grid_size), grid_size - 1)
        gy = min(int(norm_y * grid_size), grid_size - 1)
        entry = f"Mark {mark_id} at [{gx},{gy}]"

        ocr_text = mark.get("ocr_text")
        if ocr_text and ocr_text.strip() and len(ocr_text.strip()) >= 2 and ocr_text.isascii():
            clean = ocr_text.strip().replace('"', '').replace("'", '')
            if clean:
                entry += f' "{clean}"'

        entries.append(entry)

    return ". ".join(entries)


def annotate_single(image_path, yolo_model, ocr_reader):
    """Annotate one screenshot with YOLO+OCR (same pipeline as training)."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    bboxes_norm, filtered_elements = detect_ui_elements_with_ocr(
        image, yolo_model, ocr_reader
    )

    if not bboxes_norm:
        return image.copy(), {}, "", 0

    mark_helper = MarkHelper()
    mark_helper.min_font_size = MARK_FONT_MIN
    mark_helper.max_font_size = MARK_FONT_MAX

    annotated = plot_boxes_with_marks(
        image.copy(), bboxes_norm, mark_helper,
        edgecolor=(255, 0, 0), linewidth=1,
        normalized_to_pixel=True, add_mark=True,
    )

    # Build metadata in same format as preprocess_office_som.py
    marks_info = build_mark_metadata(bboxes_norm, filtered_elements, width, height)

    # Build mark text for prompt injection
    marks_text = format_marks_text(marks_info)

    return annotated, marks_info, marks_text, len(bboxes_norm)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading YOLO model: {OMNIPARSER_MODEL_PATH}")
    yolo_model = YOLO(OMNIPARSER_MODEL_PATH)
    print(f"Loading EasyOCR...")
    ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    print(f"Models loaded!")

    # Scan testcase/ folder for test subfolders
    test_folders = sorted([
        d for d in os.listdir(TESTCASE_DIR)
        if os.path.isdir(os.path.join(TESTCASE_DIR, d))
    ])
    print(f"Found {len(test_folders)} test folders in {TESTCASE_DIR}")
    print(f"Params: box_threshold={BOX_THRESHOLD}, max_marks={MAX_MARKS}, ocr_threshold={OCR_TEXT_THRESHOLD}")

    all_marks_info = {}
    all_marks_text = {}  # image_name -> mark text for prompt injection
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

        # Annotate with YOLO+OCR
        annotated, marks_info, marks_text, num_marks = annotate_single(
            img_path, yolo_model, ocr_reader
        )
        print(f"    Marks: {num_marks}")

        # Save annotated image
        annotated_name = f"{folder_name}.png"
        annotated.save(os.path.join(OUTPUT_DIR, annotated_name))

        # Save marks info (same format as preprocess_office_som.py)
        image_obj = Image.open(img_path)
        width, height = image_obj.size
        all_marks_info[annotated_name] = {
            "num_marks": num_marks,
            "image_size": [width, height],
            "marks": marks_info,
        }
        all_marks_text[annotated_name] = marks_text

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
            "marks_text": marks_text,
            "prompts": prompts_list,
        })

    # Save marks info
    marks_json_path = os.path.join(OUTPUT_DIR, "marks_info.json")
    with open(marks_json_path, "w") as f:
        json.dump(all_marks_info, f, indent=2, ensure_ascii=False)

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
    print(f"  3. Run: python inference/run_tests.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
