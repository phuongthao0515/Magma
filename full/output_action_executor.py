"""Execute JSON actions from user input via PyAutoGUI."""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


def _extract_target_xy(action: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    coords = action.get("coordinates")
    if not isinstance(coords, dict):
        coords = action.get("coordinate")
    if isinstance(coords, dict):
        x = coords.get("x")
        y = coords.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return int(x), int(y)

    bbox = action.get("bbox") or action.get("bounding_box")
    if isinstance(bbox, dict):
        x1 = bbox.get("x1")
        y1 = bbox.get("y1")
        x2 = bbox.get("x2")
        y2 = bbox.get("y2")
        if (
            isinstance(x1, (int, float))
            and isinstance(y1, (int, float))
            and isinstance(x2, (int, float))
            and isinstance(y2, (int, float))
        ):
            return int((x1 + x2) / 2), int((y1 + y2) / 2)

    if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    return None, None


def execute_action(pyautogui, action: Dict[str, Any]) -> int:
    action_type = str(action.get("predicted_action") or action.get("ACTION") or "").upper()
    value = action.get("value") if "value" in action else action.get("VALUE")

    x, y = _extract_target_xy(action)
    if x is not None and y is not None:
        pyautogui.moveTo(x, y, duration=0.2)

    if action_type == "CLICK":
        if x is None or y is None:
            print("CLICK requires coordinates or bbox")
            return 2
        pyautogui.click()
        print(f"Executed CLICK at ({x}, {y})")
        return 0

    if action_type in {"TYPE", "SELECT"}:
        if x is None or y is None:
            print(f"{action_type} requires coordinates or bbox")
            return 2
        if value in (None, "", "None"):
            print(f"{action_type} requires value")
            return 2
        pyautogui.click()
        pyautogui.write(str(value), interval=0.02)
        print(f"Executed {action_type} at ({x}, {y}) with value={value}")
        return 0

    if action_type == "SCROLL":
        scroll_amount = 0
        if isinstance(value, (int, float)):
            scroll_amount = int(value)
        elif isinstance(value, str):
            if value.lower() in {"up", "scroll_up"}:
                scroll_amount = 400
            elif value.lower() in {"down", "scroll_down"}:
                scroll_amount = -400
        if scroll_amount == 0:
            scroll_amount = -400
        pyautogui.scroll(scroll_amount)
        print(f"Executed SCROLL amount={scroll_amount}")
        return 0

    print(f"Unsupported action type: {action_type}")
    return 2


def normalize_actions(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        if "actions" in payload and isinstance(payload["actions"], list):
            return [item for item in payload["actions"] if isinstance(item, dict)]
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def main() -> int:
    try:
        import pyautogui
    except ModuleNotFoundError:
        print("Missing dependency: pyautogui. Install with: pip install pyautogui")
        return 2

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.1

    print("Paste an action JSON, then press Enter to execute it.")
    print("Accepted formats: single action object, list of action objects, or {'actions': [...]}.")
    print("Type 'exit' or 'quit' then Enter to stop.")
    print("Move mouse to top-left corner to trigger fail-safe abort.")

    while True:
        try:
            user_input = input("action> ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("Stopped by user.")
                return 0

            if not user_input:
                print("Please enter a JSON action payload.")
                continue

            try:
                payload = json.loads(user_input)
            except json.JSONDecodeError as exc:
                print(f"Invalid JSON: {exc}")
                continue

            actions = normalize_actions(payload)
            if not actions:
                print("No valid actions found in input payload.")
                continue

            print(f"Executing {len(actions)} action(s) from user input")
            for action in actions:
                time.sleep(2)
                rc = execute_action(pyautogui, action)
                if rc != 0:
                    print("Action execution failed; waiting for next input.")
                    break

        except pyautogui.FailSafeException:
            print("Aborted by fail-safe (mouse moved to top-left corner).")
            return 130
        except KeyboardInterrupt:
            print("Stopped by user.")
            return 0
        except Exception as exc:
            print(f"Loop error: {exc}")
            continue


if __name__ == "__main__":
    sys.exit(main())
