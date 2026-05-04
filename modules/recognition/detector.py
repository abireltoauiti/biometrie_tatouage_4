"""
modules/recognition/detector.py — Haar Cascade detector.
"""
import cv2, os, numpy as np
from typing import List, Tuple
from config import CASCADE_PATH, SCALE_FACTOR, MIN_NEIGHBORS, MIN_FACE_SIZE

FaceBoxes = List[Tuple[int, int, int, int]]

class FaceDetector:
    def __init__(self):
        if not os.path.isfile(CASCADE_PATH):
            raise FileNotFoundError(f"Cascade not found: {CASCADE_PATH}")
        self._cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load cascade: {CASCADE_PATH}")
        print(f"[Detector] Haar Cascade loaded OK")

    def detect(self, gray: np.ndarray) -> FaceBoxes:
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS, minSize=MIN_FACE_SIZE)
        if len(faces) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for x,y,w,h in faces]
