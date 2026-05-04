import cv2
import os

# ===== CONFIG =====
PERSON_NAME = "abir"   # change ici: abir / douaa / unknown
SAVE_DIR = f"test_images/{PERSON_NAME}"
NUM_IMAGES = 80

# ==================

os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not detected")
    exit()

count = 0
print(f"📸 Capturing {NUM_IMAGES} images for '{PERSON_NAME}'...")
print("👉 Press SPACE to capture | Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture Test Images", frame)

    key = cv2.waitKey(1)

    if key == ord(' '):  # SPACE
        img_path = os.path.join(SAVE_DIR, f"{count:04d}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved: {img_path}")
        count += 1

        if count >= NUM_IMAGES:
            break

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"✅ Done! {count} images saved in {SAVE_DIR}")