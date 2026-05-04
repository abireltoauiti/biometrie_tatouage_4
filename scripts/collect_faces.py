"""
scripts/collect_faces.py — Collect face CROPS for LBPH training.
Saves 200x200 face crops (not full frames).
"""
import cv2, os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CASCADE_PATH, CAMERA_INDEX, IMAGE_DIR
from modules.database.db import init_db, insert_person_if_missing

N_SAMPLES = 80

def collect(name, role, n=N_SAMPLES):
    label      = name.replace(" ", "-").lower()
    person_dir = os.path.join(IMAGE_DIR, label)
    os.makedirs(person_dir, exist_ok=True)
    init_db()
    insert_person_if_missing(name,  role)
    insert_person_if_missing(label, role)

    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        print("[ERROR] Cascade not found"); sys.exit(1)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera"); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"\n[Collect] '{name}' ({role}) — saving face crops")
    print("  Sit 40cm from camera · good light · look straight · press Q to stop\n")

    count = 0
    skip  = 0

    while count < n:
        ret, frame = cap.read()
        if not ret: break
        skip += 1
        if skip % 3 != 0: continue

        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_e = cv2.equalizeHist(gray)
        faces  = cascade.detectMultiScale(gray_e, 1.2, 6, minSize=(80, 80))

        for (x, y, w, h) in faces:
            pad = int(max(w, h) * 0.2)
            fh, fw = frame.shape[:2]
            x1 = max(0, x-pad); y1 = max(0, y-pad)
            x2 = min(fw, x+w+pad); y2 = min(fh, y+h+pad)

            # Save grayscale crop (LBPH works on grayscale)
            crop_gray = gray[y1:y2, x1:x2]
            crop_gray = cv2.resize(crop_gray, (100, 100))
            cv2.imwrite(os.path.join(person_dir, f"{count+1:04d}.jpg"), crop_gray)
            count += 1

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,220,80), 2)
            cv2.putText(frame, f"{name} [{count}/{n}]",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0,220,80), 2)
            if count >= n: break

        bar = int((count/n)*frame.shape[1])
        cv2.rectangle(frame,(0,frame.shape[0]-8),(bar,frame.shape[0]),(0,200,80),-1)
        cv2.putText(frame, f"Saved: {count}/{n}  Q=stop",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.imshow(f"Collecting: {name}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[Collect] Saved {count} grayscale crops → images/{label}/")
    print("  Next: python scripts/train_model.py")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--role", required=True, choices=["authorized","unauthorized"])
    p.add_argument("--samples", type=int, default=N_SAMPLES)
    a = p.parse_args()
    collect(a.name, a.role, a.samples)
