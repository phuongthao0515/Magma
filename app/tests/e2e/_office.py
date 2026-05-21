"""Office COM helpers + window-foreground control for e2e tests (Windows only).

Why the foreground control matters: the agent's ``pyautogui.screenshot()``
captures the primary display, and ``pyautogui.click()`` clicks the visible
pixels. So the target Office window must be on top and maximized when the agent
acts — not the browser. ``bring_to_front`` enforces that.
"""

from __future__ import annotations

import ctypes
import time

import pythoncom
import win32com.client

_user32 = ctypes.windll.user32
_kernel32 = ctypes.windll.kernel32

# Main top-level window class per Office app.
_WINDOW_CLASS = {"word": "OpusApp", "excel": "XLMAIN", "powerpoint": "PPTFrameClass"}
_SW_RESTORE = 9
_SW_MAXIMIZE = 3


def _co_init() -> None:
    try:
        pythoncom.CoInitialize()
    except Exception:
        pass


# --- open / close ---------------------------------------------------------
def open_word():
    _co_init()
    app = win32com.client.DispatchEx("Word.Application")
    app.Visible = True
    doc = app.Documents.Add()
    return app, doc


def close_word(app, doc) -> None:
    try:
        doc.Close(SaveChanges=0)  # 0 = wdDoNotSaveChanges
    except Exception:
        pass
    try:
        app.Quit()
    except Exception:
        pass


def open_excel():
    _co_init()
    app = win32com.client.DispatchEx("Excel.Application")
    app.Visible = True
    app.DisplayAlerts = False
    wb = app.Workbooks.Add()
    return app, wb


def close_excel(app, wb) -> None:
    try:
        wb.Close(SaveChanges=False)
    except Exception:
        pass
    try:
        app.Quit()
    except Exception:
        pass


def open_powerpoint():
    _co_init()
    app = win32com.client.DispatchEx("PowerPoint.Application")
    app.Visible = True  # PowerPoint refuses to automate while invisible
    prs = app.Presentations.Add()  # WithWindow defaults to msoTrue
    try:
        layout = prs.SlideMaster.CustomLayouts(1)
        prs.Slides.AddSlide(1, layout)
    except Exception:
        prs.Slides.Add(1, 12)  # 12 = ppLayoutBlank (legacy API fallback)
    return app, prs


def close_powerpoint(app, prs) -> None:
    try:
        prs.Close()
    except Exception:
        pass
    try:
        app.Quit()
    except Exception:
        pass


# --- window foreground ----------------------------------------------------
def _hwnd_from_com(app_key: str, com_app) -> int:
    if com_app is None:
        return 0
    try:
        if app_key == "excel":
            return int(com_app.Hwnd)
        if app_key == "powerpoint":
            return int(com_app.HWND)
    except Exception:
        return 0
    return 0


def _find_window(app_key: str) -> int:
    return _user32.FindWindowW(_WINDOW_CLASS[app_key], None)


def _force_foreground(hwnd: int) -> None:
    """Bring hwnd to the foreground, working around the Windows foreground lock."""
    _user32.ShowWindow(hwnd, _SW_RESTORE)
    _user32.ShowWindow(hwnd, _SW_MAXIMIZE)
    if _user32.GetForegroundWindow() == hwnd:
        return
    cur = _kernel32.GetCurrentThreadId()
    fg = _user32.GetForegroundWindow()
    fg_thread = _user32.GetWindowThreadProcessId(fg, None) if fg else 0
    tgt_thread = _user32.GetWindowThreadProcessId(hwnd, None)
    attached_fg = bool(fg_thread) and _user32.AttachThreadInput(fg_thread, cur, True)
    attached_tgt = bool(tgt_thread) and _user32.AttachThreadInput(tgt_thread, cur, True)
    try:
        _user32.BringWindowToTop(hwnd)
        _user32.SetForegroundWindow(hwnd)
    finally:
        if attached_fg:
            _user32.AttachThreadInput(fg_thread, cur, False)
        if attached_tgt:
            _user32.AttachThreadInput(tgt_thread, cur, False)


def bring_to_front(app_key: str, com_app=None, retries: int = 10, pause: float = 0.3) -> bool:
    """Make the Office window the topmost, maximized, foreground window."""
    # Word and PowerPoint expose Application.Activate(); Excel does not.
    if com_app is not None:
        try:
            com_app.Activate()
        except Exception:
            pass
    hwnd = 0
    for _ in range(retries):
        hwnd = _hwnd_from_com(app_key, com_app) or _find_window(app_key)
        if hwnd:
            break
        time.sleep(pause)
    if not hwnd:
        return False
    try:
        _force_foreground(hwnd)
    except Exception:
        pass
    time.sleep(0.4)
    return True
