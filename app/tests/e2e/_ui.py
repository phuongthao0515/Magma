"""Playwright helpers for driving the React frontend.

Selectors verified against app/client/src/pages/home/index.tsx:
  - prompt input: the single AntD <Input> (role "textbox")
  - start button: text "Start Task"
  - terminal banner: AntD Alert "Task <completed|failed|cancelled> — N step(s) executed"
    (the dash is an em dash, U+2014; we match the leading word only)
"""

from __future__ import annotations

import os
import re

# done -> "completed", failed -> "failed", cancelled -> "cancelled"
_TERMINAL_RE = re.compile(r"Task (completed|failed|cancelled)")
_DEFAULT_TIMEOUT_MS = int(os.environ.get("E2E_TIMEOUT_MS", "300000"))


def step(msg: str) -> None:
    """Print a clearly-marked step line (shows up in `pytest -s` output)."""
    print(f"[e2e] >> {msg}", flush=True)


def start_task(page, url: str, prompt: str, type_delay_ms: int = 40) -> None:
    step(f"opening frontend: {url}")
    page.goto(url)
    box = page.get_by_role("textbox").first
    box.click()
    step("typing the prompt into the input (watch the field)...")
    box.press_sequentially(prompt, delay=type_delay_ms)  # visible, char-by-char
    page.wait_for_timeout(400)
    step("clicking 'Start Task'")
    page.get_by_role("button", name="Start Task").click()


def wait_for_terminal(page, timeout_ms: int = _DEFAULT_TIMEOUT_MS) -> str:
    """Block until the completion/failure/cancel banner appears; return its text."""
    banner = page.get_by_text(_TERMINAL_RE).first
    banner.wait_for(timeout=timeout_ms)
    return banner.inner_text()


def capture_notification(page, timeout_ms: int = 6000):
    """Return the top-right toast text, or None if it didn't appear in time.

    Best-effort: the AntD notification auto-dismisses after ~4.5s, so call this
    immediately after the completion banner is detected.
    """
    try:
        notice = page.locator(".ant-notification-notice").first
        notice.wait_for(state="visible", timeout=timeout_ms)
        return notice.inner_text()
    except Exception:
        return None


def capture_history_row(page, prompt_substring: str, status: str = "done", timeout_ms: int = 8000):
    """Return the Task History row matching prompt_substring + status (or None).

    With the history cleared before the run, this run's task is the only row.
    """
    try:
        row = (
            page.locator("tr.ant-table-row")
            .filter(has_text=prompt_substring)
            .filter(has_text=status)
            .first
        )
        row.wait_for(state="visible", timeout=timeout_ms)
        row.scroll_into_view_if_needed()
        return row.inner_text()
    except Exception:
        return None


def timeline_step_count(page) -> int:
    return page.locator(".ant-timeline-item").count()
