"""
modules/verification/verify.py — Check if a capture has been tampered with.

Usage:
    python -m modules.verification.verify captures/unauthorized/Yassine_xxx.jpg
    python -m modules.verification.verify        # interactive prompt
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from modules.watermark.watermark import compute_sha256
from modules.database.db         import get_event_by_image_path


def verify_capture(image_path: str, silent: bool = False) -> bool:
    def log(msg):
        if not silent:
            print(msg)

    if not os.path.isfile(image_path):
        log(f"[Verify] File not found: {image_path}")
        return False

    current_hash = compute_sha256(image_path)
    log(f"[Verify] Current  SHA-256 : {current_hash}")

    event = get_event_by_image_path(image_path)
    if event is None:
        log("[Verify] No database record found for this image.")
        return False

    stored_hash = event["hash_sha256"]
    log(f"[Verify] Stored   SHA-256 : {stored_hash}")
    log(f"[Verify] Person   : {event['person_name']}  |  "
        f"Category: {event['category']}  |  "
        f"Time: {event['timestamp']}")

    if current_hash == stored_hash:
        log("[Verify] VALID — image not modified since capture.")
        return True
    else:
        log("[Verify] MODIFIED — hash mismatch! Possible tampering.")
        return False


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) >= 2 else input("Image path: ").strip()
    sys.exit(0 if verify_capture(path) else 1)
