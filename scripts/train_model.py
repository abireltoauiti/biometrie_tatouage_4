"""
scripts/train_model.py — Version minimale stable.
Meme preprocessing que recognizer.py : juste resize(100,100).
"""
import cv2, os, sys, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR, IMAGE_DIR, TRAINER_PATH, LABELS_PATH
from modules.database.db import init_db, insert_person_if_missing

def train():
    os.makedirs(os.path.dirname(TRAINER_PATH), exist_ok=True)
    if not os.path.isdir(IMAGE_DIR):
        print("[ERROR] images/ not found."); sys.exit(1)
    init_db()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    x_train, y_labels, label_ids = [], [], {}
    current_id = 0

    person_dirs = sorted([d for d in os.listdir(IMAGE_DIR)
                          if os.path.isdir(os.path.join(IMAGE_DIR, d))])
    if not person_dirs:
        print("[ERROR] No person folders."); sys.exit(1)

    print(f"\n[Train] Training for {len(person_dirs)} persons...\n")
    for person_name in person_dirs:
        person_path = os.path.join(IMAGE_DIR, person_name)
        files = [f for f in os.listdir(person_path)
                 if f.lower().endswith((".jpg",".jpeg",".png"))]
        if not files:
            continue
        insert_person_if_missing(person_name, "authorized")
        label_ids[person_name] = current_id
        id_ = current_id
        current_id += 1
        loaded = 0
        for fname in files:
            img = cv2.imread(os.path.join(person_path, fname), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.resize(img, (100, 100))   # juste resize, rien d'autre
            x_train.append(img)
            y_labels.append(id_)
            loaded += 1
        print(f"[Train] '{person_name}' (id={id_}) -> {loaded} images")

    if not x_train:
        print("[ERROR] No training data."); sys.exit(1)

    print(f"\n[Train] Training on {len(x_train)} images...")
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save(TRAINER_PATH)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(label_ids, f)
    print(f"[Train] Done. Labels: {label_ids}")
    print("[Train] Run: python web/app.py")

if __name__ == "__main__":
    train()
