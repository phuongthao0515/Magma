"""Pytest fixtures for the end-to-end (e2e) suite.

These tests drive the REAL system on a real Windows desktop:
  frontend (Vite :4000) -> HF backend -> local agent (:8010) -> pyautogui -> Office

Nothing here modifies existing project files. The suite is opt-in via the `e2e`
marker and is skipped automatically off-Windows or when Playwright is missing.

Run from the `app/` directory:
    python -m pytest tests/e2e -m e2e -v -s
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Make sibling helper modules (_office, _recorder, _ui) importable regardless of
# pytest's import mode.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

APP_ROOT = _HERE.parents[1]  # .../app
AGENT_DIR = APP_ROOT / "agent"
CLIENT_DIR = APP_ROOT / "client"
EXECUTOR_PY = AGENT_DIR / "executor.py"
SCRIPTS_DIR = _HERE / "scripts"
ARTIFACTS_DIR = _HERE / "artifacts"

# --- configuration (all overridable via environment) ---------------------
FRONTEND_URL = os.environ.get("E2E_FRONTEND_URL", "http://localhost:4000")
AGENT_PORT = int(os.environ.get("E2E_AGENT_PORT", "8010"))
AGENT_URL = os.environ.get("E2E_AGENT_URL", f"http://localhost:{AGENT_PORT}")
BACKEND_URL = os.environ.get("DOME_SERVER_URL", "https://wafair-dome.hf.space")
AGENT_DELAY = os.environ.get("E2E_AGENT_DELAY", "3")
# HF cold start can take a minute+, so the terminal-state wait is generous.
TERMINAL_TIMEOUT_MS = int(os.environ.get("E2E_TIMEOUT_MS", "300000"))
AUTOSTART = os.environ.get("E2E_AUTOSTART", "1") != "0"
HEADLESS = os.environ.get("E2E_HEADLESS", "0") == "1"
SLOW_MO = int(os.environ.get("E2E_SLOWMO_MS", "200"))
FPS = int(os.environ.get("E2E_FPS", "8"))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "e2e: end-to-end tests (real agent + frontend + Office; Windows only, opt-in)",
    )


# --- guard fixtures ------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _require_windows():
    if sys.platform != "win32":
        pytest.skip("e2e tests require Windows + Microsoft Office")


@pytest.fixture(scope="session", autouse=True)
def _require_playwright():
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        pytest.skip(
            "playwright not installed. From app/:\n"
            "  pip install -r tests/e2e/requirements-e2e.txt\n"
            "  playwright install chromium"
        )


# --- service availability helpers ----------------------------------------
def _url_ok(url: str, timeout: float = 3.0) -> bool:
    import httpx

    try:
        return httpx.get(url, timeout=timeout).status_code < 500
    except Exception:
        return False


def _wait_url(url: str, timeout: float, interval: float = 1.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _url_ok(url):
            return True
        time.sleep(interval)
    return False


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _start_service(name: str) -> subprocess.Popen:
    """Start "agent" or "frontend" via the platform launch script (run.ps1/run.sh)."""
    env = {
        **os.environ,
        "DOME_SERVER_URL": BACKEND_URL,
        "VITE_API_URL": BACKEND_URL,
        "VITE_BACKEND_URL": BACKEND_URL,
        "VITE_AGENT_URL": AGENT_URL,
    }
    if sys.platform == "win32":
        script = SCRIPTS_DIR / "run.ps1"
        cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), name]
    else:
        script = SCRIPTS_DIR / "run.sh"
        cmd = ["bash", str(script), name]
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    logfile = open(ARTIFACTS_DIR / f"{name}.log", "w", encoding="utf-8")  # noqa: SIM115
    print(f"[e2e] starting {name} via {script.name} {name}; logs -> artifacts/{name}.log")
    return subprocess.Popen(cmd, env=env, stdout=logfile, stderr=subprocess.STDOUT)


# --- process fixtures (reuse if already running, else auto-start) ---------
@pytest.fixture(scope="session")
def agent_server():
    health = f"{AGENT_URL}/health"
    if _url_ok(health):
        print(f"[e2e] reusing agent already running at {AGENT_URL}")
        yield AGENT_URL
        return
    if not AUTOSTART:
        pytest.skip(
            f"Agent not running at {AGENT_URL} (E2E_AUTOSTART=0). Start it manually:\n"
            f"  Windows: powershell -ExecutionPolicy Bypass -File tests/e2e/scripts/run.ps1 agent\n"
            f"  Linux:   bash tests/e2e/scripts/run.sh agent"
        )
    proc = _start_service("agent")
    try:
        if not _wait_url(health, timeout=60):
            _kill_tree(proc)
            pytest.fail(f"Agent did not become healthy at {health}")
        yield AGENT_URL
    finally:
        _kill_tree(proc)


@pytest.fixture(scope="session")
def web_frontend(agent_server):
    if _url_ok(FRONTEND_URL):
        print(f"[e2e] reusing frontend already running at {FRONTEND_URL}")
        yield FRONTEND_URL
        return
    if not AUTOSTART:
        pytest.skip(
            f"Frontend not running at {FRONTEND_URL} (E2E_AUTOSTART=0). Start it manually:\n"
            f"  Windows: powershell -ExecutionPolicy Bypass -File tests/e2e/scripts/run.ps1 frontend\n"
            f"  Linux:   bash tests/e2e/scripts/run.sh frontend"
        )
    proc = _start_service("frontend")
    try:
        if not _wait_url(FRONTEND_URL, timeout=120):
            _kill_tree(proc)
            pytest.fail(f"Frontend did not start at {FRONTEND_URL}")
        yield FRONTEND_URL
    finally:
        _kill_tree(proc)


# --- browser fixtures -----------------------------------------------------
@pytest.fixture(scope="session")
def _browser():
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    try:
        browser = pw.chromium.launch(headless=HEADLESS, slow_mo=SLOW_MO)
    except Exception as exc:
        pw.stop()
        pytest.skip(f"Could not launch Chromium (run 'playwright install chromium'): {exc}")
    yield browser
    browser.close()
    pw.stop()


@pytest.fixture
def page(_browser, web_frontend):
    # Grant notifications so the app's desktop-notification path can also fire.
    context = _browser.new_context(
        viewport={"width": 1366, "height": 850},
        permissions=["notifications"],
    )
    pg = context.new_page()
    yield pg
    context.close()


# --- screen recorder (per test) ------------------------------------------
@pytest.fixture
def record(request):
    """Record the desktop for the duration of the test -> artifacts/<name>.mp4.

    Listed LAST in test signatures so it never starts if an earlier fixture
    skips. Recording failures are non-fatal.
    """
    from _recorder import ScreenRecorder

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = request.node.name.replace("/", "_").replace("\\", "_").replace(":", "_")
    rec = ScreenRecorder(ARTIFACTS_DIR / f"{safe}.mp4", fps=FPS)
    rec.start()
    yield rec
    path = rec.stop()
    if path:
        print(f"[e2e] recording saved: {path}")


# --- backend history ------------------------------------------------------
@pytest.fixture
def clean_history():
    """Delete existing backend tasks so Task History shows only this run's task."""
    import httpx

    try:
        with httpx.Client(base_url=BACKEND_URL, timeout=30.0) as c:
            data = (c.get("/api/v1/tasks").json() or {}).get("data") or []
            for t in data:
                tid = t.get("id")
                if tid:
                    try:
                        c.delete(f"/api/v1/tasks/{tid}")
                    except Exception:
                        pass
            print(f"[e2e] cleared {len(data)} existing task(s) from history")
    except Exception as exc:  # noqa: BLE001
        print(f"[e2e] could not clear history: {exc}")
    yield


# --- Office app fixtures --------------------------------------------------
@pytest.fixture
def word_app():
    import _office

    app, doc = _office.open_word()
    try:
        yield app, doc
    finally:
        _office.close_word(app, doc)


@pytest.fixture
def excel_app():
    import _office

    app, wb = _office.open_excel()
    try:
        yield app, wb
    finally:
        _office.close_excel(app, wb)


@pytest.fixture
def ppt_app():
    import _office

    app, prs = _office.open_powerpoint()
    try:
        yield app, prs
    finally:
        _office.close_powerpoint(app, prs)
