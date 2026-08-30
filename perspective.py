"""
perspective.py  Document Perspective Correction and MRZ Strip Isolation
=======================================================================
Automatically detects the four corners of a passport/ID page, deskews it
via perspective transform, and optionally isolates the MRZ strip for OCR.

Pipeline:
  1. Grayscale + Gaussian blur + Canny edge detection.
  2. Find the largest quadrilateral contour (approxPolyDP).
  3. Order corners: top-left, top-right, bottom-right, bottom-left.
  4. Apply cv2.getPerspectiveTransform + cv2.warpPerspective.
  5. Crop the bottom ~22% height strip for targeted MRZ OCR.

Fallback:
  If a clear quadrilateral is not found, returns the original image unchanged.
  All functions are safe to call unconditionally.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MRZ_STRIP_FRACTION: float = 0.22


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Sort four 2-D points into (top-left, top-right, bottom-right, bottom-left)."""
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_document_corners(image_np: np.ndarray) -> Optional[np.ndarray]:
    """Try to locate the four corners of a document card in the image.

    Args:
        image_np: RGB numpy array (H, W, 3).

    Returns:
        Ordered (4, 2) float32 corner array, or None if detection fails.
    """
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        h, w = image_np.shape[:2]
        min_area = (h * w) * 0.10

        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                corners = _order_corners(approx)
                logger.info("Document corners detected.")
                return corners

        logger.info("No clear quadrilateral found; skipping perspective correction.")
        return None

    except Exception as exc:
        logger.warning("Corner detection failed: %s", exc)
        return None


def deskew_document(image_np: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp a document image to a rectangular, upright view using its four corners.

    Args:
        image_np: RGB numpy array (H, W, 3).
        corners:  Ordered (4, 2) float32 corner array (TL, TR, BR, BL).

    Returns:
        Perspective-corrected RGB numpy array.
    """
    tl, tr, br, bl = corners
    w_top = float(np.linalg.norm(tr - tl))
    w_bot = float(np.linalg.norm(br - bl))
    out_w = int(max(w_top, w_bot))
    h_left = float(np.linalg.norm(bl - tl))
    h_right = float(np.linalg.norm(br - tr))
    out_h = int(max(h_left, h_right))

    if out_w < 50 or out_h < 50:
        logger.warning("Computed output dimensions too small; skipping warp.")
        return image_np

    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image_np, M, (out_w, out_h))
    logger.info("Perspective correction applied: output %dx%d", out_w, out_h)
    return warped


def isolate_mrz_strip(image_np: np.ndarray) -> np.ndarray:
    """Crop the bottom fraction of a document image to isolate the MRZ zone.

    TD3 passports have MRZ in approximately the bottom 22% of the data page.

    Args:
        image_np: RGB numpy array (H, W, 3).

    Returns:
        Cropped RGB numpy array containing the MRZ region.
    """
    h = image_np.shape[0]
    strip_top = max(0, int(h * (1.0 - MRZ_STRIP_FRACTION)))
    mrz_strip = image_np[strip_top:, :]
    logger.info("MRZ strip isolated: rows %d-%d of %d.", strip_top, h, h)
    return mrz_strip


def auto_deskew(image_np: np.ndarray) -> tuple:
    """Full deskew pipeline: detect corners, warp, return corrected image.

    Falls back to the original image if corners cannot be found.

    Args:
        image_np: RGB numpy array (H, W, 3).

    Returns:
        Tuple (corrected_np, was_corrected) where was_corrected is a bool.
    """
    corners = detect_document_corners(image_np)
    if corners is None:
        return image_np, False
    corrected = deskew_document(image_np, corners)
    return corrected, True
