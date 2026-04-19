"""
Run inference + evaluation + pyautogui conversion (skips annotation).

Use this after test_cases.json is already filled in.

Usage:
    python /home/thaole/thao_le/Magma/inference/run_e2e.py
"""

import subprocess
import sys

SCRIPTS = [
    "/home/thaole/thao_le/Magma/inference/run_tests.py",
    "/home/thaole/thao_le/Magma/inference/convert_to_pyautogui.py",
]


def main():
    for script in SCRIPTS:
        print(f"\n{'='*60}")
        print(f"Running: {script}")
        print(f"{'='*60}\n")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"\nFailed at: {script} (exit code {result.returncode})")
            sys.exit(result.returncode)

    print(f"\n{'='*60}")
    print("All steps complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
