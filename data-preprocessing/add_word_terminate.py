"""
Build a TERMINATE-augmented copy of word_mind2web_style.json.

Word's mind2web file was generated with --include-terminate disabled (default),
leaving 0 TERMINATE samples. This script extracts TERMINATE steps from the
already-Word-filtered word_tasks.jsonl, appends them to a COPY of the existing
word_mind2web_style.json (preserving all original sample IDs), and writes the
result to a new dedicated folder.

IMPORTANT: This script DOES NOT overwrite any existing files. It only creates
new files in a new folder so the existing Word pipeline stays intact.

Source (read-only, never modified):
    datasets/agentnet/word/word_tasks.jsonl
    datasets/agentnet/word/word_mind2web_style.json

Output (new folder):
    datasets/agentnet/word_5actions/word_mind2web_style_5actions.json

Usage:
    cd /home/thaole/thao_le/Magma
    python data-preprocessing/add_word_terminate.py
"""

import json
import os
import re
from collections import Counter

ROOT = "/home/thaole/thao_le/Magma"
WORD_JSONL = f"{ROOT}/datasets/agentnet/word/word_tasks.jsonl"
EXISTING_JSON = f"{ROOT}/datasets/agentnet/word/word_mind2web_style.json"
IMAGE_DIR = f"{ROOT}/datasets/agentnet/office_images"

# New dedicated folder — does NOT touch the existing word/ folder
OUTPUT_DIR = f"{ROOT}/datasets/agentnet/word_5actions"
OUTPUT_JSON = f"{OUTPUT_DIR}/word_mind2web_style_5actions.json"


# Inlined from generate_mind2web_from_jsonl.py (avoids importing from a
# path that contains a space, which breaks some IDE static analysis).
# Logic is byte-identical to the source.

def parse_action(code):
    if "computer.terminate" in code:
        return {"ACTION": "TERMINATE", "MARK": "None", "VALUE": "None"}
    if "pyautogui.write(" in code:
        messages = re.findall(r"write\(message=['\"](.+?)['\"]", code)
        text = "".join(messages)
        if text:
            return {"ACTION": "TYPE", "MARK": "None", "VALUE": text}
    if "pyautogui.hotkey(" in code:
        match = re.search(r"hotkey\(\[?([^\])\]]+)\]?\)", code)
        if match:
            keys_str = match.group(1)
            keys = [k.strip().strip("'\"") for k in keys_str.split(",")]
            return {"ACTION": "HOTKEY", "MARK": "None", "VALUE": keys}
        return {"ACTION": "HOTKEY", "MARK": "None", "VALUE": "None"}
    if "pyautogui.moveTo(" in code and "pyautogui.dragTo(" in code:
        start_match = re.search(r"moveTo\(x=([\d.]+),\s*y=([\d.]+)\)", code)
        end_match = re.search(r"dragTo\(x=([\d.]+),\s*y=([\d.]+)", code)
        if start_match and end_match:
            start = [float(start_match.group(1)), float(start_match.group(2))]
            end = [float(end_match.group(1)), float(end_match.group(2))]
            return {"ACTION": "DRAG", "MARK": "None", "VALUE": {"start": start, "end": end}}
        return {"ACTION": "DRAG", "MARK": "None", "VALUE": "None"}
    if "pyautogui.doubleClick(" in code:
        return {"ACTION": "DOUBLE_CLICK", "MARK": "None", "VALUE": "None"}
    if "pyautogui.rightClick(" in code:
        return {"ACTION": "RIGHT_CLICK", "MARK": "None", "VALUE": "None"}
    if "pyautogui.middleClick(" in code:
        return {"ACTION": "MIDDLE_CLICK", "MARK": "None", "VALUE": "None"}
    if "computer.tripleClick(" in code:
        return {"ACTION": "CLICK", "MARK": "None", "VALUE": "None"}
    if "pyautogui.hscroll(" in code:
        return {"ACTION": "HSCROLL", "MARK": "None", "VALUE": "None"}
    if "pyautogui.scroll(" in code:
        return {"ACTION": "SCROLL", "MARK": "None", "VALUE": "None"}
    if "pyautogui.press(" in code:
        match = re.search(r"press\(['\"](\w+)['\"]", code)
        value = match.group(1) if match else "None"
        return {"ACTION": "PRESS", "MARK": "None", "VALUE": value}
    if "pyautogui.click(" in code:
        return {"ACTION": "CLICK", "MARK": "None", "VALUE": "None"}
    if "pyautogui.moveTo(" in code:
        return {"ACTION": "MOVE", "MARK": "None", "VALUE": "None"}
    return {"ACTION": "CLICK", "MARK": "None", "VALUE": "None"}


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
        '{"ACTION": action_type, "MARK": numeric_id, "VALUE": text_or_null}\n\n'
        f"Task: {task_description}\n\n"
        f"Previous actions:\n{prev_text}\n\n"
        "For your convenience, UI elements are labeled with numeric marks.\n\n"
        "What is the next action?\n"
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load existing Word samples (read-only — preserved identically in output)
    with open(EXISTING_JSON) as f:
        existing = json.load(f)
    existing_images = {s["image"] for s in existing}
    max_idx = max(
        int(s["id"].rsplit("_", 1)[-1])
        for s in existing
        if s["id"].startswith("AgentNet_SoM_")
    )
    next_idx = max_idx + 1
    print(f"Existing Word samples:        {len(existing)}")
    print(f"Max existing sample ID index: {max_idx}")
    print(f"New TERMINATE IDs start at:   AgentNet_SoM_{next_idx}")

    # 2. Snapshot images on disk (match filter_agentnet_word.py's on-disk check)
    on_disk = set(os.listdir(IMAGE_DIR))
    print(f"Images on disk in office_images/: {len(on_disk)}")

    # 3. Walk Word tasks, extract TERMINATE steps only
    added = []
    stats = Counter()
    with open(WORD_JSONL) as f:
        for line in f:
            entry = json.loads(line)
            task = entry["instruction"]
            previous_actions = []

            for step in entry["traj"]:
                code = step["value"].get("code", "")
                desc = step["value"].get("action", "")
                correct = step["value"].get("last_step_correct", True)
                redundant = step["value"].get("last_step_redundant", False)
                image = step["image"]

                # Same filter as generate_mind2web_from_jsonl.py default:
                # drop incorrect/redundant steps (but keep their action in history)
                if not correct or redundant:
                    if desc:
                        previous_actions.append(desc)
                    stats["skipped_incorrect_or_redundant"] += 1
                    continue

                action = parse_action(code)

                # Only TERMINATE is appended; everything else already exists
                if action["ACTION"] != "TERMINATE":
                    if desc:
                        previous_actions.append(desc)
                    stats["skipped_non_terminate"] += 1
                    continue

                # Defensive: skip if image already in existing samples
                if image in existing_images:
                    stats["skipped_duplicate_image"] += 1
                    if desc:
                        previous_actions.append(desc)
                    continue

                # Match filter_agentnet_word.py's on-disk check
                if image not in on_disk:
                    stats["skipped_missing_image"] += 1
                    if desc:
                        previous_actions.append(desc)
                    continue

                added.append({
                    "id": f"AgentNet_SoM_{next_idx}",
                    "image": image,
                    "conversations": [
                        {"from": "user", "value": build_prompt(task, previous_actions)},
                        {"from": "assistant", "value": json.dumps(action)},
                    ],
                })
                next_idx += 1
                stats["added_terminate"] += 1

                if desc:
                    previous_actions.append(desc)

    # 4. Combine (existing preserved verbatim) and write to NEW folder
    combined = existing + added
    with open(OUTPUT_JSON, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    # 5. Report
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    for k, v in stats.most_common():
        print(f"  {k:35s}: {v}")
    print(f"\n  Existing (preserved verbatim):     {len(existing)}")
    print(f"  TERMINATE appended:                {len(added)}")
    print(f"  New total:                         {len(combined)}")

    term_count = sum(
        1 for s in combined
        if json.loads(s["conversations"][1]["value"])["ACTION"] == "TERMINATE"
    )
    action_dist = Counter(
        json.loads(s["conversations"][1]["value"])["ACTION"] for s in combined
    )
    print(f"  TERMINATE in output:               {term_count}")
    print(f"\n  Action distribution in output:")
    for a, c in action_dist.most_common():
        print(f"    {a:>15s}: {c:>5d}")
    print(f"\n  Output: {OUTPUT_JSON}")
    print(f"\n  Source files NOT modified:")
    print(f"    {EXISTING_JSON}")
    print(f"    {WORD_JSONL}")
    print("=" * 60)


if __name__ == "__main__":
    main()
