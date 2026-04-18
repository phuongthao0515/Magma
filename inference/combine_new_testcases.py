"""
Combine testcase_new/ into test_cases_new.json in exact training format.

Output format matches training data (conversations with full prompt + marks):
  {
    "id": "test_new_1_0",
    "image": "test1.png",
    "conversations": [
      {"from": "user", "value": "<image>\nImagine that you are...\\nWhat is the next action?\\n\\n...marks..."},
      {"from": "assistant", "value": "{\"ACTION\": \"CLICK\", \"MARK\": 53, \"VALUE\": \"None\"}"}
    ]
  }

Usage:
    cd /home/thaole/thao_le/Magma
    python inference/tests/combine_new_testcases.py
"""

import json
import os

TESTCASE_DIR = "/home/thaole/thao_le/Magma/inference/tests/word/Word"
OUTPUT_JSON = "/home/thaole/thao_le/Magma/inference/tests/test_cases_word_v2.json"
GRID_SIZE = 100

REFERENCE_NOTE = "The following mark positions and labels are approximate references to help you identify elements. Always verify by checking the image."


def is_clean_ocr(text):
    if not text or not text.strip():
        return False
    if len(text.strip()) < 2:
        return False
    if not text.isascii():
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?()/-@&+_=")
    allowed_count = sum(1 for c in text if c in allowed)
    if len(text) > 0 and allowed_count / len(text) < 0.85:
        return False
    return True


def build_marks_text(marks_dict, image_size):
    w, h = image_size
    entries = []
    for mark_id in sorted(marks_dict.keys(), key=int):
        m = marks_dict[mark_id]
        if "norm_center" in m:
            norm_x, norm_y = m["norm_center"]
        else:
            norm_x = m.get("center_x", 0) / w if w > 0 else 0
            norm_y = m.get("center_y", 0) / h if h > 0 else 0
        gx = min(int(norm_x * GRID_SIZE), GRID_SIZE - 1)
        gy = min(int(norm_y * GRID_SIZE), GRID_SIZE - 1)
        entry = f"Mark {mark_id} at [{gx},{gy}]"
        ocr_text = m.get("ocr_text")
        if ocr_text and is_clean_ocr(ocr_text):
            clean = ocr_text.strip().replace('"', '').replace("'", '')
            if clean:
                entry += f' "{clean}"'
        entries.append(entry)
    return ". ".join(entries)


def build_user_prompt(task_prompt, marks_text):
    """Build the user prompt in exact training format."""
    return (
        "<image>\n"
        "Imagine that you are imitating humans doing GUI navigation step by step.\n\n"
        "You can perform actions such as CLICK, DOUBLE_CLICK, RIGHT_CLICK, MIDDLE_CLICK, "
        "MOVE, DRAG, SCROLL, HSCROLL, TYPE, PRESS, HOTKEY.\n\n"
        "Output format must be:\n"
        '{"ACTION": action_type, "MARK": numeric_id, "VALUE": text_or_null}\n\n'
        f"Task: {task_prompt}\n\n"
        "Previous actions:\nNone\n\n"
        "For your convenience, UI elements are labeled with numeric marks.\n\n"
        "What is the next action?\n\n"
        f"{REFERENCE_NOTE}\n"
        f"{marks_text}\n"
    )


def main():
    all_samples = []

    for i in range(0, 100):
        tc_path = os.path.join(TESTCASE_DIR, f"test_case_{i}.json")
        mi_path = os.path.join(TESTCASE_DIR, f"marks_info_{i}.json")
        img_path = os.path.join(TESTCASE_DIR, f"test{i}.png")

        if not os.path.exists(tc_path):
            break

        tc = json.load(open(tc_path))
        mi = json.load(open(mi_path))

        mi_key = list(mi.keys())[0]
        marks_data = mi[mi_key]
        marks = marks_data["marks"]

        if "image_size" in marks_data:
            image_size = marks_data["image_size"]
        else:
            from PIL import Image
            img = Image.open(img_path)
            image_size = list(img.size)

        marks_text = build_marks_text(marks, image_size)

        prompt_count = 0
        for item in tc:
            for prompt_info in item["prompts"]:
                user_value = build_user_prompt(prompt_info["prompt"], marks_text)
                assistant_value = json.dumps(prompt_info["expected"])

                all_samples.append({
                    "id": f"test_word_{i}_{prompt_count}",
                    "image": img_path,
                    "conversations": [
                        {"from": "user", "value": user_value},
                        {"from": "assistant", "value": assistant_value},
                    ],
                })
                prompt_count += 1

        print(f"  test{i}: {prompt_count} prompts, {len(marks)} marks")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print(f"\nTotal: {len(all_samples)} samples")
    print(f"Saved: {OUTPUT_JSON}")
    print(f"\nCan be evaluated with eval_checkpoints.py by setting:")
    print(f"  VAL_JSON = '{OUTPUT_JSON}'")
    print(f"  IMAGE_DIR = '{TESTCASE_DIR}'")


if __name__ == "__main__":
    main()
