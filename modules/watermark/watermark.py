"""
modules/watermark/watermark.py — DCT watermarking + SHA-256 integrity.

Two-layer authenticity system:
  1. DCT invisible watermark — embeds a numeric signature in frequency
     coefficients. Survives JPEG compression (unlike LSB).
  2. SHA-256 hash — stored in database, verifiable anytime.
  3. Visible overlay — human-readable proof on the image.

DCT method:
  - Convert image to YCrCb, work on Y (luminance) channel
  - Divide into 8x8 blocks
  - Apply DCT to each block
  - Embed watermark bits in mid-frequency coefficients (robust to JPEG)
  - Inverse DCT to reconstruct
  - Strength controlled by EMBED_STRENGTH (higher = more robust, more visible)
"""

import cv2
import numpy as np
import hashlib
import struct
from typing import Optional
from config import WM_FONT_SCALE, WM_THICKNESS, WM_LINE_HEIGHT, WM_MARGIN, ALERT_COLORS

# DCT watermark settings
EMBED_STRENGTH = 40    # higher = more robust to JPEG, but can slightly affect image quality
BLOCK_SIZE     = 8     # DCT block size (standard 8x8)
WM_COEFF_POS   = (3, 4)  # mid-frequency coefficient position in 8x8 block
REPEAT_BITS    = 7     # each bit is embedded several times and recovered by majority vote


# ── DCT Watermark ─────────────────────────────────────────────────────────────

def _text_to_bits(text: str) -> list:
    """Convert string to list of bits."""
    bits = []
    for char in text:
        byte = ord(char)
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def _bits_to_text(bits: list) -> str:
    """Convert list of bits back to string."""
    chars = []
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        if len(byte_bits) < 8:
            break
        byte = 0
        for b in byte_bits:
            byte = (byte << 1) | b
        if byte == 0:
            break
        try:
            chars.append(chr(byte))
        except Exception:
            break
    return ''.join(chars)


def _block_positions(height: int, width: int) -> list:
    """Return all 8x8 block top-left positions in scan order."""
    return [
        (row, col)
        for row in range(0, height - BLOCK_SIZE + 1, BLOCK_SIZE)
        for col in range(0, width - BLOCK_SIZE + 1, BLOCK_SIZE)
    ]


def _effective_repeat(num_bits: int, total_blocks: int) -> int:
    """
    Choose a repeat factor that fits in the image.
    Large images use REPEAT_BITS; small images fall back automatically.
    """
    if num_bits <= 0:
        return 1
    return max(1, min(REPEAT_BITS, total_blocks // num_bits))


def _embed_bit_qim(coeff: float, bit: int) -> float:
    """
    Quantization Index Modulation (QIM) parity embedding.
    bit 0 -> even quantization index, bit 1 -> odd quantization index.
    This is usually more stable than checking fractional remainders.
    """
    q = int(np.round(coeff / EMBED_STRENGTH))

    # Force parity: even for 0, odd for 1
    if (q % 2) != bit:
        if coeff >= 0:
            q += 1
        else:
            q -= 1

    return float(q * EMBED_STRENGTH)


def _extract_bit_qim(coeff: float) -> int:
    """Extract one bit from a QIM-parity-embedded DCT coefficient."""
    q = int(np.round(coeff / EMBED_STRENGTH))
    return q % 2


def embed_dct_watermark(image: np.ndarray, watermark_text: str) -> np.ndarray:
    """
    Embed a text watermark using DCT mid-frequency coefficients.

    Robustness improvement:
    each bit is embedded multiple times in different 8x8 blocks. During
    extraction, the bit is recovered with majority voting. This improves
    resistance to JPEG compression, blur and noise compared with embedding
    each bit only once.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image provided to embed_dct_watermark")

    # Work on luminance channel (Y in YCrCb)
    ycrcb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float32)

    h, w = Y.shape
    bits = _text_to_bits(watermark_text)
    positions = _block_positions(h, w)
    repeat = _effective_repeat(len(bits), len(positions))

    if len(bits) > len(positions):
        raise ValueError(
            f"Image too small for watermark: need {len(bits)} DCT blocks, "
            f"available {len(positions)}. Use a larger image or shorter text."
        )

    pos_idx = 0
    r, c = WM_COEFF_POS

    for bit in bits:
        for _ in range(repeat):
            if pos_idx >= len(positions):
                break

            row, col = positions[pos_idx]
            block = Y[row:row + BLOCK_SIZE, col:col + BLOCK_SIZE]
            dct_block = cv2.dct(block)

            dct_block[r, c] = _embed_bit_qim(dct_block[r, c], bit)

            Y[row:row + BLOCK_SIZE, col:col + BLOCK_SIZE] = cv2.idct(dct_block)
            pos_idx += 1

    # Clip and reconstruct
    ycrcb[:, :, 0] = np.clip(Y, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def extract_dct_watermark(image: np.ndarray, num_chars: int = 20) -> str:
    """
    Extract the DCT watermark from an image.

    Uses the same redundancy strategy as embedding: each bit is read multiple
    times and recovered by majority vote.
    """
    if image is None or image.size == 0:
        return ""

    ycrcb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float32)

    h, w = Y.shape
    needed_bits = num_chars * 8
    positions = _block_positions(h, w)
    repeat = _effective_repeat(needed_bits, len(positions))

    bits = []
    pos_idx = 0
    r, c = WM_COEFF_POS

    for _ in range(needed_bits):
        votes = []

        for _ in range(repeat):
            if pos_idx >= len(positions):
                break

            row, col = positions[pos_idx]
            block = Y[row:row + BLOCK_SIZE, col:col + BLOCK_SIZE]
            dct_block = cv2.dct(block)
            votes.append(_extract_bit_qim(dct_block[r, c]))
            pos_idx += 1

        if not votes:
            break

        # Majority vote. In a tie, default to 0 to reduce random characters.
        ones = sum(votes)
        zeros = len(votes) - ones
        bits.append(1 if ones > zeros else 0)

    return _bits_to_text(bits)


# ── Visible overlay ───────────────────────────────────────────────────────────

def add_visible_watermark(image: np.ndarray, metadata: dict) -> np.ndarray:
    """
    Draw visible watermark banner on image.
    Applied AFTER DCT embedding.
    """
    img   = image.copy()
    h, w  = img.shape[:2]
    color = ALERT_COLORS.get(metadata.get("alert_level", ""), (255, 255, 255))

    lines = [
        f"[CAMERA]  {metadata.get('camera_id',   'N/A')}",
        f"[TIME]    {metadata.get('timestamp',   'N/A')}",
        f"[PERSON]  {metadata.get('person_name', 'UNKNOWN')}",
        f"[STATUS]  {metadata.get('category',    'N/A')}",
        f"[ALERT]   {metadata.get('alert_level', 'N/A')}",
        f"[METHOD]  DCT+SHA256",
    ]

    font     = cv2.FONT_HERSHEY_SIMPLEX
    band_top = h - len(lines) * WM_LINE_HEIGHT - WM_MARGIN * 2

    overlay  = img.copy()
    cv2.rectangle(overlay, (0, band_top), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    for i, line in enumerate(lines):
        y = band_top + WM_MARGIN + (i + 1) * WM_LINE_HEIGHT
        cv2.putText(img, line, (WM_MARGIN+1, y+1),
                    font, WM_FONT_SCALE, (0,0,0), WM_THICKNESS+1, cv2.LINE_AA)
        cv2.putText(img, line, (WM_MARGIN, y),
                    font, WM_FONT_SCALE, color, WM_THICKNESS, cv2.LINE_AA)

    badge = "DCT WATERMARKED"
    (tw, th), _ = cv2.getTextSize(badge, font, 0.35, 1)
    cv2.putText(img, badge, (w-tw-WM_MARGIN, th+WM_MARGIN),
                font, 0.35, color, 1, cv2.LINE_AA)
    return img


def watermark_capture(image: np.ndarray, metadata: dict) -> np.ndarray:
    """
    Full watermarking pipeline:
      1. Embed DCT watermark (invisible, JPEG-robust)
      2. Add visible overlay

    Returns fully watermarked image.
    """
    # Build short watermark string for DCT embedding
    ts   = metadata.get("timestamp", "")[-8:].replace(":", "")  # HHMMSS
    name = metadata.get("person_name", "UNK")[:6].upper()
    cam  = metadata.get("camera_id", "CAM")[-3:]
    wm_text = f"{cam}{name}{ts}"   # e.g. "01ABIRUNKN143052"

    # Step 1: DCT embed
    dct_watermarked = embed_dct_watermark(image, wm_text)

    # Step 2: Visible overlay
    final = add_visible_watermark(dct_watermarked, metadata)

    return final


# ── SHA-256 ───────────────────────────────────────────────────────────────────

def compute_sha256(image_path: str) -> str:
    """Compute SHA-256 hash of file on disk."""
    sha256 = hashlib.sha256()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
