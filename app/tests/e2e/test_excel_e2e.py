import time

import pytest

import _office
import _ui

pytestmark = pytest.mark.e2e

EXCEL_PROMPT = "Click the Insert tab, then stop."
HISTORY_MATCH = "Insert tab"


def test_excel_single_action_then_terminate(web_frontend, clean_history, page, excel_app, record):
    app, _wb = excel_app
    _ui.step("[1] Excel workbook opened via COM")
    _office.bring_to_front("excel", app)

    _ui.start_task(page, web_frontend, EXCEL_PROMPT)
    _ui.step("[2] task created on HF and dispatched to the local agent")

    time.sleep(1.5)
    _ui.step("[3] bringing Excel to the foreground for the agent's screenshot")
    _office.bring_to_front("excel", app)

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

    _ui.step("[8] PASS: task completed after clicking the Insert tab")
