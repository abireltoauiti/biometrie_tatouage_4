

"""
config.py — All settings in one place.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Paths ───────────────────────────────────────────────────────────────
CASCADE_PATH   = os.path.join(BASE_DIR, "cascades", "haarcascade_frontalface_alt2.xml")
TRAINER_PATH   = os.path.join(BASE_DIR, "models",   "trainner.yml")
LABELS_PATH    = os.path.join(BASE_DIR, "models",   "labels.pickle")

IMAGE_DIR      = os.path.join(BASE_DIR, "images")        # training data
TEST_IMAGE_DIR = os.path.join(BASE_DIR, "test_images")   # evaluation data

DB_PATH        = os.path.join(BASE_DIR, "surveillance.db")

CAPTURES_DIR   = os.path.join(BASE_DIR, "captures")
CAPTURES_AUTH  = os.path.join(CAPTURES_DIR, "authorized")
CAPTURES_UNAUTH= os.path.join(CAPTURES_DIR, "unauthorized")
CAPTURES_UNK   = os.path.join(CAPTURES_DIR, "unknown")

# ── Camera ──────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
CAMERA_ID    = "CAM-01"

# ── Face detection ────────────────────────────────────────────────────────────
SCALE_FACTOR  = 1.2
MIN_NEIGHBORS = 5
MIN_FACE_SIZE = (30, 30)

# ── Face recognition ────────────────────────────────────────────────────
CONFIDENCE_MIN       = 4
CONFIDENCE_THRESHOLD = 130

# ── Capture behaviour ─────────────────────────────────────────────────────────
CAPTURE_COOLDOWN   = 5      # seconds between saves for same person
FRAME_RESIZE_WIDTH = 640    # resize frame before processing (0 = no resize)
DATASET_SAMPLES    = 150    # photos collected per person

# ── Watermark ───────────────────────────────────────────────────────────
WM_FONT_SCALE  = 0.45
WM_THICKNESS   = 1
WM_LINE_HEIGHT = 16
WM_MARGIN      = 8

# ── Alert colours ───────────────────────────────────────────────────────
ALERT_COLORS = {
    "LOG_ONLY": (0, 200, 80),
    "ALERT":    (0, 140, 255),
    "ALARM":    (50,  50, 255),
}

# ── Seed persons ────────────────────────────────────────────────────────
SEED_PERSONS = [
    {"name": "Abir",  "status": "authorized"},
    {"name": "Douaa", "status": "unauthorized"},
]
