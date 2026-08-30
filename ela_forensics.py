"""
ela_forensics.py  Error Level Analysis (ELA) for Document Forensics
====================================================================
ELA is a HEURISTIC technique that highlights regions of an image where
the JPEG compression artifact level differs significantly from the rest.
Consistent compression artifacts throughout a genuinely unmodified JPEG
produce a roughly uniform error map; spliced or composited regions may
show anomalously high or low error levels.

IMPORTANT DISCLAIMER:
  ELA produces an *indicator*, NOT proof of forgery.
  A high mean-error score means "elevated compression-artifact score"
  and warrants further investigation -- it does not confirm tampering.
  Factors such as re-saves, re-edits, social-media re-encoding,
  and image-quality settings can all produce elevated ELA scores on
  unmodified images.

ELA on lossless sources (PNG, BMP, TIFF):
  ELA is most meaningful on JPEG images that have already been compressed.
  When the source is a lossless format (PNG etc.), we create a JPEG baseline
  by saving at the chosen quality level; the resulting error map reflects
  the *first* quantisation, not a difference between compressions.
  This is a weaker signal and the UI should display the associated caveat.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CALIBRATE_ME  -- tune against your own sample images.
# This threshold is the mean absolute pixel error (0-255 scale, post-scaling)
# above which we flag the image as having an "elevated compression-artifact
# score." Increase it to reduce false positives on legitimately re-saved docs;
# decrease it to catch subtler splice artefacts.
# ---------------------------------------------------------------------------
ELA_ALERT_THRESHOLD: float = 12.0


def perform_ela(
    image_path: str,
    quality: int = 90,
    scale: int = 15,
) -> dict:
    """Run Error Level Analysis on a document image.

    Saves the source image to an in-memory JPEG buffer at *quality*, then
    computes the pixel-wise scaled absolute difference between the original
    and the re-compressed version:

        E(x, y) = scale * |I_original(x,y) - I_recompressed(x,y)|

    A color heatmap (JET colormap via OpenCV) makes high-error regions
    visually salient.

    Args:
        image_path: Absolute or relative path to the document image
                    (JPEG, PNG, BMP, TIFF, WebP all accepted).
        quality:    JPEG re-compression quality (1-95). Lower quality
                    amplifies errors; 90 is a standard starting point.
        scale:      Amplification factor applied before clamping to 0-255.
                    Increasing this makes subtle differences more visible.

    Returns:
        A dict with keys:
          'mean_error'  (float)         - Mean absolute pixel error (0-255 scale).
          'alert'       (bool)          - True if mean_error > ELA_ALERT_THRESHOLD.
          'heatmap_bgr' (np.ndarray)    - H x W x 3 BGR heatmap, ready for cv2/st.
          'ela_gray'    (np.ndarray)    - H x W grayscale error image.
          'caveat'      (str)           - Non-empty if source was lossless format
                                         (signal strength caveat for the UI).
          'error'       (str)           - Non-empty if ELA could not run at all.
    """
    result: dict = {
        "mean_error": 0.0,
        "alert": False,
        "heatmap_bgr": None,
        "ela_gray": None,
        "caveat": "",
        "error": "",
    }

    try:
        path = Path(image_path)
        if not path.exists():
            result["error"] = f"Image file not found: {image_path}"
            logger.error(result["error"])
            return result

        suffix = path.suffix.lower()
        lossless_formats = {".png", ".bmp", ".tiff", ".tif", ".webp"}
        if suffix in lossless_formats:
            result["caveat"] = (
                f"Source format is {suffix.upper()} (lossless). "
                "ELA baseline is created from the first JPEG compression, "
                "which is a weaker signal than detecting re-saves of an "
                "already-JPEG image. Interpret the heatmap with caution."
            )
            logger.info("ELA caveat applied for lossless source: %s", suffix)

        # Load the original image
        try:
            original_pil = Image.open(image_path).convert("RGB")
        except Exception as exc:
            result["error"] = f"Could not open image with Pillow: {exc}"
            logger.error(result["error"])
            return result

        original_np = np.array(original_pil, dtype=np.float32)

        # Re-save to in-memory JPEG buffer and reload
        buffer = io.BytesIO()
        original_pil.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        try:
            recompressed_pil = Image.open(buffer).convert("RGB")
        except Exception as exc:
            result["error"] = f"Could not decode JPEG from in-memory buffer: {exc}"
            logger.error(result["error"])
            return result

        recompressed_np = np.array(recompressed_pil, dtype=np.float32)

        # Compute error map: E(x,y) = scale * |orig - recomp|, clamp 0-255
        ela_float = np.abs(original_np - recompressed_np) * scale
        ela_clipped = np.clip(ela_float, 0, 255).astype(np.uint8)

        # Grayscale version (max across channels to capture any channel error)
        ela_gray = np.max(ela_clipped, axis=2)

        # Colormap for visual display (JET: blue=low, red=high)
        heatmap_bgr = cv2.applyColorMap(ela_gray, cv2.COLORMAP_JET)

        mean_error = float(np.mean(ela_gray))

        result["mean_error"] = mean_error
        result["alert"] = mean_error > ELA_ALERT_THRESHOLD
        result["heatmap_bgr"] = heatmap_bgr
        result["ela_gray"] = ela_gray

        logger.info(
            "ELA complete -- mean_error=%.2f, alert=%s",
            mean_error,
            result["alert"],
        )

    except Exception as exc:
        result["error"] = f"Unexpected ELA error: {exc}"
        logger.error("ELA pipeline error: %s", exc, exc_info=True)

    return result


def heatmap_to_rgb(heatmap_bgr: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR heatmap to RGB for display in Streamlit/PIL.

    Args:
        heatmap_bgr: H x W x 3 BGR numpy array from cv2.applyColorMap.

    Returns:
        H x W x 3 RGB numpy array.
    """
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
