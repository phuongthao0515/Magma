"""E2E Test A (Word): single action then terminate.

Flow:
  1. Open a fresh Word document via COM.
  2. Drive the real frontend with Playwright: type the prompt, click "Start Task".
  3. The real local agent screenshots Word and executes the model's action
     (pyautogui types the sentence into the document).
  4. The model emits TERMINATE -> the task reaches status "done".
  5. Assert the OS-level effect via Office COM (the typed text is present).

Works with the current server default max_steps=2 (turn 0 = type, turn 1 =
TERMINATE). The whole run is screen-recorded to artifacts/.
"""

import time

import pytest

import _office
import _ui

pytestmark = pytest.mark.e2e

# One action (type a sentence), then the model should TERMINATE. Tune freely.
WORD_PROMPT = 'Type "DOME: Parameter-Efficient Adaptation of Vision-Language Models with Set-of-Mark Prompting for Desktop Office UI Automation."'

# A distinctive chunk of the typed text we expect to find in the document body.
EXPECTED_SUBSTR = "Parameter-Efficient Adaptation"


def test_word_single_action_then_terminate(web_frontend, clean_history, page, word_app, record):
    app, doc = word_app
    _ui.step("[1] Word document opened via COM")
    _office.bring_to_front("word", app)

    _ui.start_task(page, web_frontend, WORD_PROMPT)
    _ui.step("[2] task created on HF and dispatched to the local agent")

    # Give the frontend a moment to dispatch to the agent, then make Word the
    # foreground window so the agent screenshots Word (not the browser).
    time.sleep(1.5)
    _ui.step("[3] bringing Word to the foreground for the agent's screenshot")
    _office.bring_to_front("word", app)

    _ui.step("[4] waiting for agent: screenshot -> HF -> execute -> TERMINATE ...")
    banner = _ui.wait_for_terminal(page)
    _ui.step(f"[5] task stopped; completion alert = {banner!r}")
    assert "completed" in banner.lower(), (
        f"Task did not complete (banner={banner!r}). "
        "The model may not have emitted TERMINATE after one action."
    )

    # Bring the browser forward immediately so BOTH the persistent completion
    # alert AND the transient top-right notification toast are visible in the
    # recording (Word was on top during the agent's screenshots). The toast
    # auto-dismisses ~4.5s after the task stops, so we act right away.
    _ui.step("[6] showing the completion alert + notification in the browser")
    page.bring_to_front()
    toast = _ui.capture_notification(page)
    if toast:
        _ui.step(f"[6a] notification toast captured: {toast!r}")
        assert "completed" in toast.lower(), f"Unexpected notification text: {toast!r}"
    else:
        _ui.step("[6a] note: toast not captured (it may have auto-dismissed)")
    page.wait_for_timeout(4000)

    # Task History (cleared at the start) now shows only THIS run's task, as "done".
    _ui.step("[7] checking the Task History shows this task as done")
    row_text = _ui.capture_history_row(page, "DOME", status="done")
    if row_text:
        _ui.step(f"[7a] history row: {row_text!r}")
    else:
        _ui.step("[7a] note: completed history row not found yet (refresh delay?)")
    page.wait_for_timeout(2000)

    # COM source-of-truth: the typed text landed in the document body.
    _ui.step("[8] verifying the typed text is in the Word document (COM)")
    text = doc.Content.Text
    assert EXPECTED_SUBSTR.lower() in text.lower(), (
        f"Typed text not found in Word document. Expected substring "
        f"{EXPECTED_SUBSTR!r}; document starts with {text[:200]!r}"
    )
    _ui.step("[9] PASS: typed text found in Word")
