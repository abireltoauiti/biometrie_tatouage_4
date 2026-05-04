"""core/utils.py — Shared helpers used across all modules."""

import cv2
import os
import time
import numpy as np
from typing import Optional
from config import FRAME_RESIZE_WIDTH


def resize_frame(frame: np.ndarray) -> np.ndarray:
    """Resize frame to FRAME_RESIZE_WIDTH keeping aspect ratio."""
    if FRAME_RESIZE_WIDTH <= 0:
        return frame
    h, w = frame.shape[:2]
    if w == FRAME_RESIZE_WIDTH:
        return frame
    scale = FRAME_RESIZE_WIDTH / w
    return cv2.resize(frame, (FRAME_RESIZE_WIDTH, int(h * scale)))


def to_gray_equalised(frame: np.ndarray) -> np.ndarray:
    """BGR → grayscale + CLAHE equalisation (better in low light & varied backgrounds)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def safe_filename(name: str, timestamp: str) -> str:
    """'Aziz', '2025-01-15 10:30:00'  →  'Aziz_2025-01-15_10-30-00.jpg'"""
    safe_ts = timestamp.replace(":", "-").replace(" ", "_")
    return f"{name}_{safe_ts}.jpg"


def ensure_dirs(*paths: str) -> None:
    """Create directories if they don't exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


class FPSCounter:
    """Rolling 1-second FPS counter."""
    def __init__(self):
        self._count = 0
        self._start = time.time()
        self.fps    = 0.0

    def tick(self) -> Optional[float]:
        self._count += 1
        elapsed = time.time() - self._start
        if elapsed >= 1.0:
            self.fps    = self._count / elapsed
            self._count = 0
            self._start = time.time()
            return self.fps
        return None
