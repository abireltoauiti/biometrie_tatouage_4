"""
scripts/diagnose.py — Diagnose recognition accuracy.

Shows the actual confidence values your model produces for each person.
Run this BEFORE adjusting the threshold.

Usage:
    python scripts/diagnose.py

Stand in front of the camera when prompted for each person.
"""

import cv2
import sys
import os
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (CASCADE_PATH, TRAINER_PATH, LABELS_PATH,
                    CAMERA_INDEX, SCALE_FACTOR, MIN_NEIGHBORS)


def test_live():
    # Load cascade
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        print("[ERROR] Cascade not found"); sys.exit(1)

    # Load model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if not os.path.isfile(TRAINER_PATH):
        print("[ERROR] models/trainner.yml not found — run train_model.py first")
        sys.exit(1)
    recognizer.read(TRAINER_PATH)

    # Load labels
    with open(LABELS_PATH, "rb") as f:
        raw = pickle.load(f)
    labels = {v: k for k, v in raw.items()}   # {id: name}
    print(f"\n[Diagnose] Label map: {labels}")
    print("[Diagnose] Press SPACE to capture a reading, Q to quit\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    readings = []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Do NOT equalise here — match training conditions exactly
        faces = cascade.detectMultiScale(gray, SCALE_FACTOR, MIN_NEIGHBORS,
                                          minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            # Resize to match training size
            roi_resized = cv2.resize(roi, (100, 100))
            id_, conf   = recognizer.predict(roi_resized)
            name        = labels.get(id_, f"id={id_}")

            color = (0, 255, 0) if conf < 85 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name}  conf={conf:.1f}",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2, cv2.LINE_AA)

        cv2.putText(frame, "SPACE=capture reading  Q=quit",
                    (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Diagnose — press SPACE to record confidence", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    roi = gray[y:y+h, x:x+w]
                    roi_resized = cv2.resize(roi, (100, 100))
                    id_, conf = recognizer.predict(roi_resized)
                    name = labels.get(id_, f"id={id_}")
                    readings.append((name, conf))
                    print(f"  Reading: predicted='{name}'  confidence={conf:.1f}")

    cap.release()
    cv2.destroyAllWindows()

    if readings:
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        for name in set(n for n,_ in readings):
            confs = [c for n,c in readings if n==name]
            print(f"  '{name}':  min={min(confs):.1f}  max={max(confs):.1f}  avg={sum(confs)/len(confs):.1f}  ({len(confs)} readings)")
        print()
        print("RECOMMENDATION:")
        all_confs = [c for _,c in readings]
        print(f"  Set CONFIDENCE_THRESHOLD = {int(min(all_confs) + (max(all_confs)-min(all_confs))*0.6)}")
        print("  (Adjust manually based on what you see above)")


if __name__ == "__main__":
    test_live()
