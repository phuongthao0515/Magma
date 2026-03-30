"""
Convert model output (ACTION/MARK/VALUE) + mark coordinates to pyautogui commands.

Reads test results JSON and the marks_info.json to produce executable pyautogui code.

Usage:
    python /home/thaole/thao_le/Magma/inference/convert_to_pyautogui.py
"""

import json
import os

# ============ CONFIGURATION ============
RESULTS_PATH = "/home/thaole/thao_le/Magma/inference/tests/results/test_results.json"
MARKS_INFO_PATH = "/home/thaole/thao_le/Magma/inference/tests/annotated/marks_info.json"
OUTPUT_PATH = "/home/thaole/thao_le/Magma/inference/tests/results/pyautogui_commands.json"
# =======================================


def action_to_pyautogui(prediction, mark_coords):
    """Convert a single prediction to a pyautogui command string + dict."""
    action = prediction.get("ACTION", "").upper()
    mark = prediction.get("MARK")
    value = prediction.get("VALUE")

    # Resolve mark to pixel coordinates
    coord = None
    if mark is not None and str(mark) != "None":
        mark_str = str(mark)
        if mark_str in mark_coords:
            coord = (mark_coords[mark_str]["center_x"], mark_coords[mark_str]["center_y"])

    if action == "CLICK" and coord:
        return {
            "command": f"pyautogui.click(x={coord[0]}, y={coord[1]})",
            "action": "click",
            "x": coord[0],
            "y": coord[1],
        }

    elif action == "DOUBLE_CLICK" and coord:
        return {
            "command": f"pyautogui.doubleClick(x={coord[0]}, y={coord[1]})",
            "action": "doubleClick",
            "x": coord[0],
            "y": coord[1],
        }

    elif action == "RIGHT_CLICK" and coord:
        return {
            "command": f"pyautogui.rightClick(x={coord[0]}, y={coord[1]})",
            "action": "rightClick",
            "x": coord[0],
            "y": coord[1],
        }

    elif action == "TYPE" and coord and value and value != "None":
        return {
            "command": f"pyautogui.click(x={coord[0]}, y={coord[1]})\npyautogui.write({repr(value)}, interval=0.02)",
            "action": "type",
            "x": coord[0],
            "y": coord[1],
            "value": value,
        }

    elif action == "SCROLL":
        amount = -3 if value in (None, "None", "down") else 3
        cmd = f"pyautogui.scroll({amount})"
        if coord:
            cmd = f"pyautogui.scroll({amount}, x={coord[0]}, y={coord[1]})"
        return {
            "command": cmd,
            "action": "scroll",
            "amount": amount,
        }

    elif action == "HOTKEY" and value and value != "None":
        keys = [k.strip() for k in value.replace("+", ",").split(",")]
        keys_str = ", ".join(repr(k) for k in keys)
        return {
            "command": f"pyautogui.hotkey({keys_str})",
            "action": "hotkey",
            "keys": keys,
        }

    elif action == "PRESS" and value and value != "None":
        return {
            "command": f"pyautogui.press({repr(value)})",
            "action": "press",
            "key": value,
        }

    elif action == "DRAG" and coord and value and value != "None":
        return {
            "command": f"pyautogui.moveTo(x={coord[0]}, y={coord[1]})\npyautogui.drag({value})",
            "action": "drag",
            "x": coord[0],
            "y": coord[1],
            "value": value,
        }

    return {
        "command": f"# Could not convert: ACTION={action}, MARK={mark}, VALUE={value}",
        "action": "unknown",
        "error": True,
    }


def main():
    print(f"Loading results: {RESULTS_PATH}")
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    print(f"Loading mark coordinates: {MARKS_INFO_PATH}")
    with open(MARKS_INFO_PATH) as f:
        all_marks = json.load(f)

    results = data["results"]
    commands = []

    for r in results:
        image_name = os.path.basename(r["image"])
        mark_coords = all_marks.get(image_name, {}).get("marks", {})

        pyautogui_cmd = action_to_pyautogui(r["prediction"], mark_coords)

        commands.append({
            "index": r["index"],
            "image": r["image"],
            "prompt": r["prompt"],
            "prediction": r["prediction"],
            "overall_match": r["overall_match"],
            **pyautogui_cmd,
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(commands, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Generated {len(commands)} pyautogui commands")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"{'='*60}")

    # Print all commands
    for cmd in commands:
        status = "PASS" if cmd.get("overall_match") else "FAIL"
        print(f"\n[{status}] {cmd['prompt'][:60]}")
        print(f"  {cmd['command']}")


if __name__ == "__main__":
    main()
