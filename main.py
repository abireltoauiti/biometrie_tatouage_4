"""
main.py — Smart Surveillance Platform entry point.
Run:  python main.py
"""

import cv2
import os
import time
import threading
from datetime import datetime

from config import (
    CAMERA_INDEX, CAMERA_ID, CAPTURE_COOLDOWN,
    CAPTURES_AUTH, CAPTURES_UNAUTH, CAPTURES_UNK,
    SEED_PERSONS, ALERT_COLORS,
)
from core.utils                       import resize_frame, to_gray_equalised, FPSCounter, ensure_dirs, safe_filename
from modules.recognition.detector    import FaceDetector
from modules.recognition.recognizer  import FaceRecognizer
from modules.watermark.watermark     import add_visible_watermark, compute_sha256
from modules.database.db             import init_db, insert_person_if_missing, insert_event
from ui.dashboard                    import SurveillanceDashboard

SAVE_DIR_MAP = {
    "AUTHORIZED":   CAPTURES_AUTH,
    "UNAUTHORIZED": CAPTURES_UNAUTH,
    "UNKNOWN":      CAPTURES_UNK,
}


def process_frame(frame, detector, recognizer, cooldowns):
    events  = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    frame = resize_frame(frame)
    gray  = to_gray_equalised(frame)
    boxes = detector.detect(gray)

    for (x, y, w, h) in boxes:
        roi_gray = gray[y:y+h, x:x+w]
        name, category, alert_level, confidence = recognizer.predict(roi_gray)

        # Draw bounding box
        color = ALERT_COLORS.get(alert_level, (200, 200, 200))
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Label with name + confidence
        label = f"{name}  {confidence:.0f}"
        cv2.putText(frame, label, (x+1, y-9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        # Category badge bottom of box
        cv2.rectangle(frame, (x, y+h-20), (x+w, y+h), color, -1)
        cv2.putText(frame, category, (x+4, y+h-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)

        # Cooldown
        key = f"{name}_{category}"
        if time.time() - cooldowns.get(key, 0) < CAPTURE_COOLDOWN:
            continue
        cooldowns[key] = time.time()

        events.append({
            "frame":       frame.copy(),
            "name":        name,
            "category":    category,
            "alert_level": alert_level,
            "confidence":  confidence,
            "timestamp":   now_str,
            "save_dir":    SAVE_DIR_MAP.get(category, CAPTURES_UNK),
        })

    return frame, events


def save_event(ev: dict, dashboard: SurveillanceDashboard):
    metadata = {
        "timestamp":   ev["timestamp"],
        "camera_id":   CAMERA_ID,
        "person_name": ev["name"],
        "category":    ev["category"],
        "alert_level": ev["alert_level"],
    }
    watermarked = add_visible_watermark(ev["frame"], metadata)
    filename    = safe_filename(ev["name"], ev["timestamp"])
    img_path    = os.path.normpath(os.path.join(ev["save_dir"], filename))

    cv2.imwrite(img_path, watermarked, [cv2.IMWRITE_JPEG_QUALITY, 95])
    sha = compute_sha256(img_path)

    insert_event(ev["name"], ev["category"], CAMERA_ID,
                 ev["timestamp"], img_path, sha, ev["alert_level"])

    print(f"  [{ev['alert_level']:8s}] {ev['name']:<15s} | {ev['category']:<12s} | {filename}")

    if dashboard:
        dashboard.after(0, lambda p=img_path: dashboard.update_last_capture(p))
        dashboard.after(0, lambda: dashboard.log_event(
            ev["timestamp"], ev["category"],
            ev["name"], ev["alert_level"]))


def run():
    init_db()
    for p in SEED_PERSONS:
        insert_person_if_missing(p["name"], p["status"])
    ensure_dirs(CAPTURES_AUTH, CAPTURES_UNAUTH, CAPTURES_UNK)

    detector   = FaceDetector()
    recognizer = FaceRecognizer()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cooldowns = {}
    fps_ctr   = FPSCounter()
    alive     = {"running": True}

    def on_quit():
        alive["running"] = False

    dashboard = SurveillanceDashboard(on_quit=on_quit)

    print("\n[Main] Platform running — close the window to stop\n")

    def tick():
        if not alive["running"]:
            return
        ret, frame = cap.read()
        if not ret:
            dashboard.after(100, tick)
            return

        annotated, events = process_frame(frame, detector, recognizer, cooldowns)

        for ev in events:
            threading.Thread(target=save_event, args=(ev, dashboard),
                             daemon=True).start()
            # Update status immediately on detection
            dashboard.after(0, lambda e=ev: dashboard.update_status(
                e["name"], e["category"], e["alert_level"],
                confidence=e["confidence"]))

        new_fps = fps_ctr.tick()
        if new_fps:
            dashboard.after(0, lambda f=new_fps: dashboard.update_status(
                dashboard.var_person.get(),
                dashboard.var_category.get(),
                dashboard.var_alert.get(),
                fps=f))

        dashboard.update_frame(annotated)
        dashboard.after(30, tick)

    dashboard.after(0, tick)
    dashboard.mainloop()
    cap.release()
    print("[Main] Stopped.")


if __name__ == "__main__":
    run()
