"""Capture screen + prompt and write input files to local synced folder.

Workflow:
1) User types a prompt and presses Enter.
2) Script captures full screen to input.png.
3) Script writes prompt text to prompt.txt.
4) Script copies both files to the configured local input sync folder.


"""

from __future__ import annotations

import sys
import shutil
import time
from pathlib import Path


WORK_DIR = Path(__file__).resolve().parent
INPUT_IMAGE_PATH = WORK_DIR / "input.png"
PROMPT_PATH = WORK_DIR / "prompt.txt"
SYNC_ROOT = WORK_DIR / "drive_sync"
INPUT_SYNC_DIR = SYNC_ROOT / "input"
CAPTURE_DELAY_SECONDS = 2


def capture_screen(image_path: Path) -> None:
    try:
        import pyautogui
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: pyautogui. Install with: pip install pyautogui") from exc

    screenshot = pyautogui.screenshot()
    screenshot.save(str(image_path))


def run_once(prompt_text: str) -> None:
    INPUT_SYNC_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Waiting {CAPTURE_DELAY_SECONDS} second(s) before capture...")
    time.sleep(CAPTURE_DELAY_SECONDS)
    capture_screen(INPUT_IMAGE_PATH)
    PROMPT_PATH.write_text(prompt_text.strip() + "\n", encoding="utf-8")
    shutil.copy2(INPUT_IMAGE_PATH, INPUT_SYNC_DIR / "input.png")
    shutil.copy2(PROMPT_PATH, INPUT_SYNC_DIR / "prompt.txt")
    print(f"Wrote input.png to: {INPUT_SYNC_DIR / 'input.png'}")
    print(f"Wrote prompt.txt to: {INPUT_SYNC_DIR / 'prompt.txt'}")


def main() -> int:
    print("Enter your prompt and press Enter.")
    print("Type 'exit' or 'quit' to stop.")
    print(f"Input sync directory: {INPUT_SYNC_DIR}")

    while True:
        try:
            prompt_text = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nStopping.")
            return 0

        if not prompt_text:
            print("Prompt is empty. Try again.")
            continue

        if prompt_text.lower() in {"exit", "quit"}:
            return 0

        try:
            run_once(prompt_text)
            print("Capture + local sync write complete.\n")
        except Exception as exc:
            print(f"Failed: {exc}")
            return 2


if __name__ == "__main__":
    sys.exit(main())
