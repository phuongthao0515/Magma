"""
Generate coordinate-based training data for Word tasks from AgentNet JSONL.

Uses raw screenshots (no SoM) with COORDINATE [x, y] instead of MARK ID.
Coordinates come directly from pyautogui code in the JSONL (already normalized 0-1).

Reuses: word_tasks.jsonl (already filtered for Word, Windows, images exist)
Images: datasets/agentnet/office_images/ (raw screenshots, no SoM marks)
Output: datasets/agentnet/word/word_coordinate_style.json

Usage:
    cd /home/thaole/thao_le/Magma
    python data-preprocessing/Coordinates/generate_word_coordinate.py
"""

import json
import os
import re
from collections import Counter

WORD_JSONL = "/home/thaole/thao_le/Magma/datasets/agentnet/word/word_tasks.jsonl"
IMAGE_DIR = "/home/thaole/thao_le/Magma/datasets/agentnet/office_images"
OUTPUT_JSON = "/home/thaole/thao_le/Magma/datasets/agentnet/word/word_coordinate_style.json"

ACTIONS_NEED_COORD = {"CLICK", "DOUBLE_CLICK", "RIGHT_CLICK", "MIDDLE_CLICK", "TYPE", "DRAG"}


def extract_coordinates(code):
    """Extract normalized (x, y) from pyautogui code. Already 0-1 range."""
    match = re.search(r"(?:click|Click)\(x=([\d.]+),\s*y=([\d.]+)\)", code)
    if match:
        return float(match.group(1)), float(match.group(2))
    match = re.search(r"moveTo\(x=([\d.]+),\s*y=([\d.]+)\)", code)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None


def extract_drag_end(code):
    """Extract drag end coordinates from pyautogui.dragTo()."""
    match = re.search(r"dragTo\(x=([\d.]+),\s*y=([\d.]+)", code)
    if match:
        return [round(float(match.group(1)), 3), round(float(match.group(2)), 3)]
    return None


def parse_action(code):
    """Parse pyautogui code into action dict with COORDINATE placeholder."""
    if "computer.terminate" in code:
        return {"ACTION": "TERMINATE", "COORDINATE": "None", "VALUE": "None"}

    if "pyautogui.write(" in code:
        messages = re.findall(r"write\(message=['\"](.+?)['\"]", code)
        text = "".join(messages)
        if text:
            return {"ACTION": "TYPE", "COORDINATE": "None", "VALUE": text}

    if "pyautogui.hotkey(" in code:
        match = re.search(r"hotkey\(\[?([^\])\]]+)\]?\)", code)
        if match:
            keys = [k.strip().strip("'\"") for k in match.group(1).split(",")]
            return {"ACTION": "HOTKEY", "COORDINATE": "None", "VALUE": keys}
        return {"ACTION": "HOTKEY", "COORDINATE": "None", "VALUE": "None"}

    if "pyautogui.moveTo(" in code and "pyautogui.dragTo(" in code:
        end = extract_drag_end(code)
        return {"ACTION": "DRAG", "COORDINATE": "None", "VALUE": end if end else "None"}

    if "pyautogui.doubleClick(" in code:
        return {"ACTION": "DOUBLE_CLICK", "COORDINATE": "None", "VALUE": "None"}

    if "pyautogui.rightClick(" in code:
        return {"ACTION": "RIGHT_CLICK", "COORDINATE": "None", "VALUE": "None"}

    if "pyautogui.middleClick(" in code:
        return {"ACTION": "MIDDLE_CLICK", "COORDINATE": "None", "VALUE": "None"}

    if "computer.tripleClick(" in code:
        return {"ACTION": "CLICK", "COORDINATE": "None", "VALUE": "None"}

    if "pyautogui.hscroll(" in code:
        return {"ACTION": "HSCROLL", "COORDINATE": "None", "VALUE": "None"}

    if "pyautogui.scroll(" in code:
        return {"ACTION": "SCROLL", "COORDINATE": "None", "VALUE": "None"}

    if "pyautogui.press(" in code:
        match = re.search(r"press\(['\"](\w+)['\"]", code)
        value = match.group(1) if match else "None"
        return {"ACTION": "PRESS", "COORDINATE": "None", "VALUE": value}

    if "pyautogui.click(" in code:
        return {"ACTION": "CLICK", "COORDINATE": "None", "VALUE": "None"}

    if "pyautogui.moveTo(" in code:
        return {"ACTION": "MOVE", "COORDINATE": "None", "VALUE": "None"}

    return {"ACTION": "CLICK", "COORDINATE": "None", "VALUE": "None"}


def build_prompt(task_description, previous_actions):
    if previous_actions:
        prev_text = "\n".join(f"- {a}" for a in previous_actions)
    else:
        prev_text = "None"

    return (
        "<image>\n"
        "Imagine that you are imitating humans doing GUI navigation step by step.\n\n"
        "You can perform actions such as CLICK, DOUBLE_CLICK, RIGHT_CLICK, "
        "MIDDLE_CLICK, MOVE, DRAG, SCROLL, HSCROLL, TYPE, PRESS, HOTKEY.\n\n"
        "Output format must be:\n"
        '{"ACTION": action_type, "COORDINATE": [x, y], "VALUE": text_or_null}\n\n'
        f"Task: {task_description}\n\n"
        f"Previous actions:\n{prev_text}\n\n"
        "What is the next action?\n"
    )


def main():
    print(f"Reading: {WORD_JSONL}")
    images_on_disk = set(os.listdir(IMAGE_DIR))
    print(f"Images on disk: {len(images_on_disk)}")

    task_count = 0
    sample_idx = 0
    entries = []
    action_counts = Counter()
    skipped = Counter()

    with open(WORD_JSONL) as f:
        for line in f:
            entry = json.loads(line)
            task = entry["instruction"]
            previous_actions = []
            prev_coords = None
            task_count += 1

            for step in entry["traj"]:
                code = step["value"].get("code", "")
                action_desc = step["value"].get("action", "")
                correct = step["value"].get("last_step_correct", True)
                redundant = step["value"].get("last_step_redundant", False)
                image = step["image"]

                step_coords = extract_coordinates(code)

                # Filter incorrect/redundant
                if not correct or redundant:
                    skipped["incorrect_or_redundant"] += 1
                    if action_desc:
                        previous_actions.append(action_desc)
                    if step_coords:
                        prev_coords = step_coords
                    continue

                action = parse_action(code)

                if action["ACTION"] == "TERMINATE":
                    skipped["terminate"] += 1
                    continue

                if image not in images_on_disk:
                    skipped["missing_image"] += 1
                    if action_desc:
                        previous_actions.append(action_desc)
                    if step_coords:
                        prev_coords = step_coords
                    continue

                # Fill COORDINATE from pyautogui code
                if action["ACTION"] in ACTIONS_NEED_COORD:
                    coords = None
                    if action["ACTION"] == "TYPE":
                        # TYPE: prefer own click coords (the text field click),
                        # fallback to previous step's coords
                        coords = step_coords
                        if not coords:
                            coords = prev_coords
                    elif action["ACTION"] == "DRAG":
                        # DRAG: use moveTo coords (drag start), not click coords
                        move_match = re.search(r"moveTo\(x=([\d.]+),\s*y=([\d.]+)\)", code)
                        if move_match:
                            coords = (float(move_match.group(1)), float(move_match.group(2)))
                        else:
                            coords = step_coords
                    else:
                        # CLICK, DOUBLE_CLICK, RIGHT_CLICK, MIDDLE_CLICK
                        coords = step_coords

                    if coords:
                        action["COORDINATE"] = [round(coords[0], 3), round(coords[1], 3)]
                    else:
                        skipped["no_coordinates"] += 1
                        continue

                action_counts[action["ACTION"]] += 1
                prompt = build_prompt(task, previous_actions)

                entries.append({
                    "id": f"AgentNet_Coord_{sample_idx}",
                    "image": image,
                    "conversations": [
                        {"from": "user", "value": prompt},
                        {"from": "assistant", "value": json.dumps(action)},
                    ],
                })
                sample_idx += 1

                if action_desc:
                    previous_actions.append(action_desc)
                if step_coords:
                    prev_coords = step_coords

            if task_count % 100 == 0:
                print(f"  Processed {task_count} tasks, {sample_idx} samples...")

    has_coord = sum(
        1 for e in entries
        if json.loads(e["conversations"][1]["value"])["COORDINATE"] != "None"
    )

    print(f"\nWriting: {OUTPUT_JSON}")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"  Tasks:                {task_count}")
    print(f"  Samples:              {len(entries)}")
    print(f"    with COORDINATE:    {has_coord}")
    print(f"    without COORDINATE: {len(entries) - has_coord}")
    print(f"  Skipped:")
    for reason, count in skipped.most_common():
        print(f"    {reason:30s}: {count}")
    print(f"  Actions:")
    for action, count in action_counts.most_common():
        print(f"    {action:15s}: {count:6d} ({100 * count / len(entries):.1f}%)")
    print(f"  Output: {OUTPUT_JSON}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
