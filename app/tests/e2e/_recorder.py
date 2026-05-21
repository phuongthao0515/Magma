"""Pure-Python screen recorder for e2e tests.

Records the primary display to an .mp4 using PIL (frame grab) + OpenCV (encode).
No ffmpeg required. The grab region matches what the agent's
``pyautogui.screenshot()`` sees (primary monitor), so the recording is exactly
the input the model was given.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageGrab


class ScreenRecorder:
    """Background screen recorder. Use start()/stop() or as a context manager."""

    def __init__(self, out_path: str | Path, fps: int = 8) -> None:
        self.out_path = Path(out_path)
        self.fps = max(1, fps)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def __enter__(self) -> "ScreenRecorder":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def start(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="screen-recorder", daemon=True)
        self._thread.start()

    def stop(self) -> Path | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
        if self._error is not None:
            print(f"[recorder] WARNING: recording failed: {self._error}")
            return None
        return self.out_path

    def _open_writer(self, size: tuple[int, int]) -> "cv2.VideoWriter":
        writer = cv2.VideoWriter(str(self.out_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size)
        if not writer.isOpened():
            # Fallback codec/container if mp4v is unavailable in this OpenCV build.
            self.out_path = self.out_path.with_suffix(".avi")
            writer = cv2.VideoWriter(
                str(self.out_path), cv2.VideoWriter_fourcc(*"XVID"), self.fps, size
            )
        return writer

    def _run(self) -> None:
        try:
            first = ImageGrab.grab()  # primary monitor
            w, h = first.size
            writer = self._open_writer((w, h))
            if not writer.isOpened():
                raise RuntimeError("OpenCV VideoWriter could not be opened (no codec available).")
            period = 1.0 / self.fps
            while not self._stop.is_set():
                t0 = time.time()
                frame = cv2.cvtColor(np.array(ImageGrab.grab()), cv2.COLOR_RGB2BGR)
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h))
                writer.write(frame)
                remaining = period - (time.time() - t0)
                if remaining > 0:
                    time.sleep(remaining)
            writer.release()
        except Exception as exc:  # noqa: BLE001 - recording is best-effort
            self._error = exc
