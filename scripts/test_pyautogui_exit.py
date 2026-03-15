"""Execute hard-coded model actions with PyAutoGUI."""

import importlib
import json
import sys
import time
from typing import Any, Dict, Tuple


# Hard-coded model output JSON (list of actions).
MODEL_OUTPUT_JSON = """
[
    {"ACTION": "CLICK", "MARK": 1, "VALUE": null},
    {"ACTION": "TYPE", "MARK": 2, "VALUE": "hello"}
]
"""

# Every mark_id maps to one screen coordinate.
MARK_COORDS: Dict[int, Tuple[int, int]] = {
    1: (1780, 13),
    2: (135, 46),
}

# Set to True to re-capture coordinates for marks on your current screen.
CALIBRATE_COORDS = False
CALIBRATION_MARKS = [1, 2]


def execute_action(pyautogui, action_item: Dict[str, Any]) -> int:
    action = str(action_item.get("ACTION", "")).upper()
    mark = action_item.get("MARK")
    value = action_item.get("VALUE")

    if action not in {"CLICK", "TYPE"}:
        print(f"Unsupported ACTION: {action}")
        return 2

    if mark not in MARK_COORDS:
        print(f"Unknown MARK id: {mark}")
        return 2

    x, y = MARK_COORDS[mark]
    pyautogui.moveTo(x, y, duration=0.2)
    if action == "CLICK":
        pyautogui.click()
        time.sleep(2.0)

    if action == "TYPE":
        if value in (None, "None", ""):
            print("TYPE action requires VALUE")
            return 2
        pyautogui.write(str(value), interval=0.02)
    elif value not in (None, "None", ""):
        print("VALUE is ignored for CLICK action")

    print(f"Executed {action} on MARK {mark} at ({x}, {y})")
    return 0


def calibrate_mark_coords(pyautogui, mark_ids: list[int]) -> Dict[int, Tuple[int, int]]:
    captured: Dict[int, Tuple[int, int]] = {}
    print("Calibration mode enabled.")
    print("For each mark: move mouse to target and wait for capture.")

    for mark_id in mark_ids:
        print(f"MARK {mark_id}: capturing in 3 seconds...")
        time.sleep(3.0)
        pos = pyautogui.position()
        captured[mark_id] = (int(pos.x), int(pos.y))
        print(f"Captured MARK {mark_id}: {captured[mark_id]}")

    print("Copy these coordinates into MARK_COORDS:")
    print(json.dumps(captured, indent=2))
    return captured


def main() -> int:
    try:
        pyautogui: Any = importlib.import_module("pyautogui")
    except ModuleNotFoundError:
        print("Missing dependency: pyautogui")
        print("Install with: pip install pyautogui")
        return 2

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.1

    if CALIBRATE_COORDS:
        calibrate_mark_coords(pyautogui, CALIBRATION_MARKS)
        return 0

    try:
        model_output = json.loads(MODEL_OUTPUT_JSON)
    except json.JSONDecodeError as exc:
        print(f"Invalid MODEL_OUTPUT_JSON: {exc}")
        return 2

    if isinstance(model_output, dict):
        actions = [model_output]
    elif isinstance(model_output, list):
        actions = model_output
    else:
        print("MODEL_OUTPUT_JSON must decode to a JSON object or array")
        return 2

    if not actions:
        print("No actions to execute")
        return 2

    delay_seconds = 3.0
    print(f"Starting in {delay_seconds:.1f}s. Move mouse to top-left corner to abort.")
    time.sleep(delay_seconds)

    try:
        for idx, action_item in enumerate(actions, start=1):
            if not isinstance(action_item, dict):
                print(f"Action #{idx} is not a JSON object")
                return 2
            rc = execute_action(pyautogui, action_item)
            if rc != 0:
                return rc
            time.sleep(0.2)
        return 0

    except pyautogui.FailSafeException:
        print("Aborted by fail-safe (mouse moved to top-left corner).")
        return 130


if __name__ == "__main__":
    sys.exit(main())
