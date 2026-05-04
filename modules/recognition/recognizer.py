"""
modules/recognition/recognizer.py — LBPH recognition.
"""
import cv2, os, pickle
import numpy as np
from typing import Tuple
from config import TRAINER_PATH, LABELS_PATH, CONFIDENCE_MIN, CONFIDENCE_THRESHOLD
from modules.database.db import get_person_status

ROI_SIZE = (100, 100)
RecognitionResult = Tuple[str, str, str, float]


class FaceRecognizer:
    def __init__(self):
        self._recognizer = cv2.face.LBPHFaceRecognizer_create()
        self._trained    = False
        self._labels     = {}

        if os.path.isfile(TRAINER_PATH):
            self._recognizer.read(TRAINER_PATH)
            self._trained = True
            print(f"[Recognizer] LBPH model loaded OK")
        else:
            print(f"[Recognizer] WARNING: models/trainner.yml not found")

        if os.path.isfile(LABELS_PATH):
            with open(LABELS_PATH, "rb") as f:
                raw = pickle.load(f)
            self._labels = {v: k for k, v in raw.items()}
            print(f"[Recognizer] Labels: {self._labels}")
        else:
            print(f"[Recognizer] WARNING: models/labels.pickle not found")

    def predict(self, roi_gray: np.ndarray) -> RecognitionResult:
        """
        Returns (name, category, alert_level, confidence_display)
        confidence_display: raw LBPH score (lower = better match)
        Shown on bounding box as-is so it always has a meaningful value.
        """
        if not self._trained or not self._labels:
            return "UNKNOWN", "UNKNOWN", "ALARM", 0.0

        try:
            roi_resized          = cv2.resize(roi_gray, ROI_SIZE)
            label_id, confidence = self._recognizer.predict(roi_resized)
            confidence           = float(confidence)
        except cv2.error:
            return "UNKNOWN", "UNKNOWN", "ALARM", 0.0

        # Use values from config.py
        if confidence < CONFIDENCE_MIN or confidence > CONFIDENCE_THRESHOLD:
            return "UNKNOWN", "UNKNOWN", "ALARM", confidence

        raw_name = self._labels.get(label_id)
        if raw_name is None:
            return "UNKNOWN", "UNKNOWN", "ALARM", confidence

        # DB lookup — try lowercase first, then title case
        status       = get_person_status(raw_name)
        display_name = raw_name
        if status is None:
            title  = raw_name.replace("-", " ").title()
            status = get_person_status(title)
            if status is not None:
                display_name = title

        # Return raw confidence score (lower = better for LBPH)
        if status == "authorized":
            return display_name, "AUTHORIZED", "LOG_ONLY", confidence
        elif status == "unauthorized":
            return display_name, "UNAUTHORIZED", "ALERT", confidence
        else:
            return display_name, "UNKNOWN", "ALARM", confidence

    @property
    def is_ready(self) -> bool:
        return self._trained and bool(self._labels)
