import time

import pytest

import _office
import _ui

pytestmark = pytest.mark.e2e

PPT_PROMPT = "Click the New Slide button on the Home tab to add a slide, then stop."
HISTORY_MATCH = "New Slide"


def test_powerpoint_single_action_then_terminate(
    web_frontend, clean_history, page, ppt_app, record
):
    app, prs = ppt_app
    initial = prs.Slides.Count
    _ui.step(f"[1] PowerPoint opened via COM (slides={initial})")
    _office.bring_to_front("powerpoint", app)

    _ui.start_task(page, web_frontend, PPT_PROMPT)
    _ui.step("[2] task created on HF and dispatched to the local agent")

    time.sleep(1.5)
    _ui.step("[3] bringing PowerPoint to the foreground for the agent's screenshot")
    _office.bring_to_front("powerpoint", app)

    _ui.step("[4] waiting for agent: screenshot -> HF -> execute -> TERMINATE ...")
    banner = _ui.wait_for_terminal(page)
    _ui.step(f"[5] task stopped; completion alert = {banner!r}")
    assert "completed" in banner.lower(), (
        f"Task did not complete (banner={banner!r}). "
        "The model may not have emitted TERMINATE after one action."
    )

    _ui.step("[6] showing the completion alert + notification in the browser")
    page.bring_to_front()
    toast = _ui.capture_notification(page)
    if toast:
        _ui.step(f"[6a] notification toast captured: {toast!r}")
        assert "completed" in toast.lower(), f"Unexpected notification text: {toast!r}"
    else:
        _ui.step("[6a] note: toast not captured (it may have auto-dismissed)")
    page.wait_for_timeout(4000)

    _ui.step("[7] checking the Task History shows this task as done")
    row_text = _ui.capture_history_row(page, HISTORY_MATCH, status="done")
    if row_text:
        _ui.step(f"[7a] history row: {row_text!r}")
    else:
        _ui.step("[7a] note: completed history row not found yet (refresh delay?)")
    page.wait_for_timeout(2000)

    _ui.step("[8] verifying a new slide was added (COM)")
    final = prs.Slides.Count
    assert final > initial, f"No slide added (count {final} <= {initial})"
    _ui.step(f"[9] PASS: slide count {initial} -> {final}")
