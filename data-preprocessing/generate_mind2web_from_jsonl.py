"""
Generate mind2web-SoM style JSON from AgentNet raw JSONL (source of truth).

Each trajectory step becomes one independent sample:
  - id: unique sample ID
  - image: screenshot filename
  - conversations: [user prompt, assistant action]

Filtering (default):
  - Exclude incorrect steps (last_step_correct=False)
  - Exclude redundant steps (last_step_redundant=True)
  - Exclude TERMINATE actions (no visual grounding)

Flags:
  --include-incorrect   Include incorrect/redundant steps
  --include-terminate   Include TERMINATE actions

Input:  agentnet_win_mac_18k.jsonl
Output: agentnet_mind2web_style_full.json

Usage:
    cd /home/thaole/thao_le/Magma
    python data-preprocessing/generate_mind2web_from_jsonl.py
    python data-preprocessing/generate_mind2web_from_jsonl.py --include-terminate
    python data-preprocessing/generate_mind2web_from_jsonl.py --include-incorrect --include-terminate
"""

import argparse
import json
import re
from collections import Counter

AGENTNET_JSONL = "/home/thaole/thao_le/Magma/datasets/agentnet/agentnet_win_mac_18k.jsonl"
OUTPUT_JSON = "/home/thaole/thao_le/Magma/datasets/agentnet/agentnet_mind2web_style_full.json"


def parse_action(code):
    """Parse pyautogui code into mind2web action format.

    Handles combo code blocks (e.g. write+press) by prioritizing the most
    meaningful action: TYPE > HOTKEY > DRAG > PRESS > CLICK > SCROLL.

    Returns: {"ACTION": str, "MARK": "None", "VALUE": ...}
    Uses string "None" (not JSON null) to match mind2web-SoM format.
    """
    if "computer.terminate" in code:
        return {"ACTION": "TERMINATE", "MARK": "None", "VALUE": "None"}

    # TYPE: pyautogui.write() with non-empty message — highest priority for combos
    # because typing is the main intent (press('enter') is just confirmation)
    if "pyautogui.write(" in code:
        # Find ALL write messages and concatenate (some steps have multiple writes)
        messages = re.findall(r"write\(message=['\"](.+?)['\"]", code)
        text = "".join(messages)
        if text:  # Only TYPE if there's actual text
            return {"ACTION": "TYPE", "MARK": "None", "VALUE": text}

    # HOTKEY: ctrl+c, cmd+v, etc.
    if "pyautogui.hotkey(" in code:
        match = re.search(r"hotkey\(\[?([^\])\]]+)\]?\)", code)
        if match:
            keys_str = match.group(1)
            keys = [k.strip().strip("'\"") for k in keys_str.split(",")]
            return {"ACTION": "HOTKEY", "MARK": "None", "VALUE": keys}
        return {"ACTION": "HOTKEY", "MARK": "None", "VALUE": "None"}

    # DRAG: moveTo + dragTo
    if "pyautogui.moveTo(" in code and "pyautogui.dragTo(" in code:
        start_match = re.search(r"moveTo\(x=([\d.]+),\s*y=([\d.]+)\)", code)
        end_match = re.search(r"dragTo\(x=([\d.]+),\s*y=([\d.]+)", code)
        if start_match and end_match:
            start = [float(start_match.group(1)), float(start_match.group(2))]
            end = [float(end_match.group(1)), float(end_match.group(2))]
            return {"ACTION": "DRAG", "MARK": "None", "VALUE": {"start": start, "end": end}}
        return {"ACTION": "DRAG", "MARK": "None", "VALUE": "None"}

    # DOUBLE_CLICK
    if "pyautogui.doubleClick(" in code:
        return {"ACTION": "DOUBLE_CLICK", "MARK": "None", "VALUE": "None"}

    # RIGHT_CLICK
    if "pyautogui.rightClick(" in code:
        return {"ACTION": "RIGHT_CLICK", "MARK": "None", "VALUE": "None"}

    # MIDDLE_CLICK
    if "pyautogui.middleClick(" in code:
        return {"ACTION": "MIDDLE_CLICK", "MARK": "None", "VALUE": "None"}

    # TRIPLE_CLICK → map to CLICK
    if "computer.tripleClick(" in code:
        return {"ACTION": "CLICK", "MARK": "None", "VALUE": "None"}

    # HSCROLL
    if "pyautogui.hscroll(" in code:
        return {"ACTION": "HSCROLL", "MARK": "None", "VALUE": "None"}

    # SCROLL
    if "pyautogui.scroll(" in code:
        return {"ACTION": "SCROLL", "MARK": "None", "VALUE": "None"}

    # PRESS
    if "pyautogui.press(" in code:
        match = re.search(r"press\(['\"](\w+)['\"]", code)
        value = match.group(1) if match else "None"
        return {"ACTION": "PRESS", "MARK": "None", "VALUE": value}

    # CLICK (most common)
    if "pyautogui.click(" in code:
        return {"ACTION": "CLICK", "MARK": "None", "VALUE": "None"}

    # MOVE (moveTo without dragTo)
    if "pyautogui.moveTo(" in code:
        return {"ACTION": "MOVE", "MARK": "None", "VALUE": "None"}

    # Fallback
    return {"ACTION": "CLICK", "MARK": "None", "VALUE": "None"}


def build_prompt(task_description, previous_actions):
    """Build mind2web-SoM style prompt."""
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
    parser = argparse.ArgumentParser(description="Generate mind2web JSON from AgentNet JSONL")
    parser.add_argument("--include-incorrect", action="store_true",
                        help="Include incorrect/redundant steps (default: exclude)")
    parser.add_argument("--include-terminate", action="store_true",
                        help="Include TERMINATE actions (default: exclude)")
    args = parser.parse_args()

    print(f"Reading: {AGENTNET_JSONL}")
    print(f"Filters: exclude_incorrect={not args.include_incorrect}, exclude_terminate={not args.include_terminate}")

    task_count = 0
    sample_idx = 0
    mind2web_entries = []
    action_counts = Counter()
    skipped = Counter()

    with open(AGENTNET_JSONL) as f:
        for line in f:
            entry = json.loads(line)
            task = entry["instruction"]
            previous_actions = []
            task_count += 1

            for step in entry["traj"]:
                code = step["value"].get("code", "")
                action_desc = step["value"].get("action", "")
                correct = step["value"].get("last_step_correct", True)
                redundant = step["value"].get("last_step_redundant", False)
                image = step["image"]

                # Filter incorrect/redundant steps
                if not args.include_incorrect and (not correct or redundant):
                    skipped["incorrect_or_redundant"] += 1
                    # Still add to previous_actions so context stays accurate
                    if action_desc:
                        previous_actions.append(action_desc)
                    continue

                action = parse_action(code)

                # Filter TERMINATE
                if not args.include_terminate and action["ACTION"] == "TERMINATE":
                    skipped["terminate"] += 1
                    continue

                action_counts[action["ACTION"]] += 1
                prompt = build_prompt(task, previous_actions)

                mind2web_entries.append({
                    "id": f"AgentNet_SoM_{sample_idx}",
                    "image": image,
                    "conversations": [
                        {"from": "user", "value": prompt},
                        {"from": "assistant", "value": json.dumps(action)},
                    ],
                })
                sample_idx += 1

                # Accumulate previous actions for next step
                if action_desc:
                    previous_actions.append(action_desc)

            if task_count % 2000 == 0:
                print(f"  Processed {task_count} tasks, {sample_idx} samples...")

    print(f"\nWriting: {OUTPUT_JSON}")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(mind2web_entries, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"  Tasks:            {task_count}")
    print(f"  Samples output:   {len(mind2web_entries)}")
    print(f"  Skipped:")
    for reason, count in skipped.most_common():
        print(f"    {reason:30s}: {count}")
    print(f"  Action distribution:")
    for action, count in action_counts.most_common():
        print(f"    {action:15s}: {count:6d} ({100 * count / len(mind2web_entries):.1f}%)")
    print(f"  Output: {OUTPUT_JSON}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
