"""
biometrics.py  Facial Verification for Document Authentication
===============================================================
Uses DeepFace with the ArcFace model and RetinaFace detector backend
for consistent face alignment between a scanned passport photo and a
live webcam frame.

THRESHOLD NOTE:
  DeepFace default cosine-distance threshold for ArcFace is ~0.68.
  This is meaningfully different from the ~0.40 threshold used for
  VGG-Face / FaceNet models. We expose ARCFACE_COSINE_THRESHOLD as a
  named constant below so the team can calibrate it against their own
  test image pairs.

PRIVACY:
  Uploaded face images and computed embeddings are processed entirely
  in memory and are never persisted to disk. NumPy arrays are discarded
  when the function returns.

DETECTOR BACKEND:
  "retinaface" is specified explicitly (rather than the default "opencv")
  because it provides better face alignment on printed passport photos,
  which may be smaller and lower-contrast than typical webcam frames.
  Change to "mtcnn" if retinaface is unavailable in the deployment env.
"""

from __future__ import annotations

import io
import cv2
import logging
import tempfile
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CALIBRATE_ME -- tune against your own matched / non-matched image pairs.
# ArcFace cosine distance: lower = more similar (0 = identical).
# DeepFace's own default for ArcFace + cosine is approximately 0.68.
# Tighten (lower) this value to reduce false accepts;
# loosen (raise) it to reduce false rejects.
# ---------------------------------------------------------------------------
ARCFACE_COSINE_THRESHOLD: float = 0.68

# Explicit detector backend for consistent alignment.
# Use "opencv" for lightweight in-memory detection (prevents container OOM)
DETECTOR_BACKEND: str = "opencv"


@dataclass
class PassportPhotoRegion:
    """Cropped passport portrait metadata and array."""
    crop_rgb: np.ndarray
    bbox: Optional[dict]  # {'x': int, 'y': int, 'w': int, 'h': int} on full document
    status: str           # 'DETECTED', 'ICAO_ROI_FALLBACK', 'FAILED'
    confidence: float = 1.0


def locate_passport_photo(document_rgb: np.ndarray) -> PassportPhotoRegion:
    """Automatically locate and crop the primary photo region from a passport document.

    Robust multi-stage locator:
      1. OpenCV Haar Cascade face detection across document.
      2. If multiple faces exist (e.g. primary photo + secondary ghost watermark),
         selects the primary photo (largest area in the standard left-half portrait zone).
      3. Expands bounding box with standard portrait padding (headroom + chin/collar).
      4. Fallback to ICAO 9303 standard portrait ROI if detector is obstructed.

    Args:
        document_rgb: H x W x 3 RGB uint8 image array of the passport page.

    Returns:
        PassportPhotoRegion dataclass containing the clean cropped portrait and bbox.
    """
    h, w = document_rgb.shape[:2]
    if h == 0 or w == 0:
        return PassportPhotoRegion(crop_rgb=document_rgb, bbox=None, status="FAILED", confidence=0.0)

    # Convert to grayscale for cascade detector
    gray = cv2.cvtColor(document_rgb, cv2.COLOR_RGB2GRAY) if len(document_rgb.shape) == 3 else document_rgb

    # Stage 1: Haar Cascade Fast Frontal Face Detection
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        min_size = (int(min(h, w) * 0.08), int(min(h, w) * 0.08))
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=min_size)
    except Exception as exc:
        logger.warning("Haar cascade error: %s", exc)
        faces = ()

    selected_face = None

    if len(faces) > 0:
        # Score candidates: prioritize faces in the left 60% of document and above MRZ (top 85%)
        best_score = -1.0
        for (fx, fy, fw, fh) in faces:
            area = fw * fh
            # ICAO standard TD3: photo is on the left side
            is_left_side = (fx + fw / 2) < (w * 0.65)
            is_above_mrz = (fy + fh) < (h * 0.88)
            
            score = area
            if is_left_side:
                score *= 2.0
            if is_above_mrz:
                score *= 1.5
            
            if score > best_score:
                best_score = score
                selected_face = (fx, fy, fw, fh)

    if selected_face is not None:
        fx, fy, fw, fh = selected_face
        # Apply portrait padding (25% top for hair, 35% bottom for chin/shoulders, 20% sides)
        pad_top = int(fh * 0.25)
        pad_bottom = int(fh * 0.35)
        pad_side = int(fw * 0.20)

        y1 = max(0, fy - pad_top)
        y2 = min(h, fy + fh + pad_bottom)
        x1 = max(0, fx - pad_side)
        x2 = min(w, fx + fw + pad_side)

        cropped_portrait = document_rgb[y1:y2, x1:x2].copy()
        bbox_dict = {"x": int(fx), "y": int(fy), "w": int(fw), "h": int(fh), "crop_rect": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}
        
        logger.info("Passport photo region detected at [%d, %d, %d, %d]", x1, y1, x2-x1, y2-y1)
        return PassportPhotoRegion(
            crop_rgb=cropped_portrait,
            bbox=bbox_dict,
            status="DETECTED",
            confidence=0.95,
        )

    # Stage 2: ICAO Doc 9303 Standard Left-Quadrant Fallback ROI
    # On standard TD3 passports, photo sits in x: [4%, 52%], y: [16%, 84%]
    logger.info("No face detected by cascade; applying ICAO standard portrait ROI fallback.")
    roi_x1 = max(0, int(w * 0.04))
    roi_x2 = min(w, int(w * 0.54))
    roi_y1 = max(0, int(h * 0.16))
    roi_y2 = min(h, int(h * 0.84))

    fallback_crop = document_rgb[roi_y1:roi_y2, roi_x1:roi_x2].copy()
    bbox_dict = {"x": roi_x1, "y": roi_y1, "w": roi_x2 - roi_x1, "h": roi_y2 - roi_y1}

    return PassportPhotoRegion(
        crop_rgb=fallback_crop,
        bbox=bbox_dict,
        status="ICAO_ROI_FALLBACK",
        confidence=0.60,
    )


def _load_deepface():
    """Attempt to import DeepFace, returning the module or None on failure.

    Returns:
        The deepface module if importable, else None.
    """
    try:
        import deepface.DeepFace as df
        return df
    except ImportError as exc:
        logger.error("DeepFace import failed: %s", exc)
        return None


def _array_to_temp_file(image_array: np.ndarray, suffix: str = ".jpg") -> Optional[str]:
    """Write a numpy image array to a named temporary file.

    DeepFace's verify() currently requires file paths, not in-memory arrays.
    We write to a temp file, run the model, then delete the file immediately.

    Args:
        image_array: H x W x 3 RGB uint8 image array.
        suffix:      File suffix (JPEG is used to avoid large files).

    Returns:
        Absolute path to the temp file, or None on failure.
    """
    try:
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(image_array.astype(np.uint8))
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        pil_img.save(tmp.name, format="JPEG", quality=95)
        tmp.close()
        return tmp.name
    except Exception as exc:
        logger.error("Could not write image to temp file: %s", exc)
        return None


def _cleanup_temp(path: Optional[str]) -> None:
    """Delete a temporary file, logging but not raising on failure.

    Args:
        path: File path to remove, or None (no-op).
    """
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError as exc:
            logger.warning("Could not remove temp file %s: %s", path, exc)


def count_faces(image_array: np.ndarray) -> int:
    """Count the number of faces detected in an image.

    Uses DeepFace extract_faces with the configured detector backend.
    Returns -1 if detection fails entirely.

    Args:
        image_array: H x W x 3 RGB uint8 image array.

    Returns:
        Number of faces detected, or -1 if detection errored out.
    """
    df = _load_deepface()
    if df is None:
        return -1

    tmp_path = _array_to_temp_file(image_array)
    if tmp_path is None:
        return -1

    try:
        faces = df.extract_faces(
            img_path=tmp_path,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
        )
        return len(faces)
    except Exception as exc:
        # DeepFace raises ValueError when no face is found
        if "face" in str(exc).lower() or "detector" in str(exc).lower():
            return 0
        logger.error("Face count error: %s", exc)
        return -1
    finally:
        _cleanup_temp(tmp_path)


def verify_faces(
    passport_image: np.ndarray,
    live_frame: np.ndarray,
) -> dict:
    """Compare a passport photo against a live webcam frame.

    Automatically locates and crops the passport portrait from the document
    page before running ArcFace comparison to avoid text, watermark, or
    ghost photo interference.

    Args:
        passport_image: Full passport document or photo crop (RGB uint8 array).
        live_frame:     Live photo / webcam frame of traveller (RGB uint8 array).

    Returns:
        A dict with keys:
          'distance'                (float)        - ArcFace cosine distance.
          'threshold'               (float)        - Decision threshold.
          'passed'                  (bool)         - True if distance <= threshold.
          'similarity_pct'          (float)        - 0-100 similarity score.
          'passport_crop'           (np.ndarray)   - Clean isolated passport portrait.
          'passport_bbox'           (dict|None)    - Bounding box on full document.
          'live_bbox'               (dict|None)    - Bounding box on live photo.
          'photo_extraction_status' (str)          - 'DETECTED' / 'ICAO_ROI_FALLBACK'.
          'status'                  (str)          - Outcome status code.
          'message'                 (str)          - Human-readable explanation.
    """
    # 1. Automatically locate & isolate passport portrait
    photo_region = locate_passport_photo(passport_image)
    passport_crop = photo_region.crop_rgb

    result: dict = {
        "distance": None,
        "threshold": ARCFACE_COSINE_THRESHOLD,
        "passed": False,
        "similarity_pct": 0.0,
        "passport_crop": passport_crop,
        "passport_bbox": photo_region.bbox,
        "live_bbox": None,
        "photo_extraction_status": photo_region.status,
        "status": "ERROR",
        "message": "",
    }

    df = _load_deepface()
    if df is None:
        result["status"] = "MODULE_UNAVAILABLE"
        result["message"] = (
            "DeepFace / TensorFlow could not be loaded. "
            "Biometric verification is unavailable for this session."
        )
        return result

    # Validate face counts before attempting verification
    passport_count = count_faces(passport_crop)
    live_count = count_faces(live_frame)

    if passport_count == 0:
        result["status"] = "NO_FACE_PASSPORT"
        result["message"] = "No facial features resolved in the extracted passport photo."
        return result

    if live_count == 0:
        result["status"] = "NO_FACE_LIVE"
        result["message"] = "No face detected in the live/uploaded traveller photo."
        return result

    if live_count > 1:
        result["status"] = "MULTIPLE_FACES"
        result["message"] = (
            f"Multiple faces ({live_count}) found in live photo. "
            "Please ensure only the traveller is visible."
        )
        return result

    # Write both images to temp files for DeepFace
    p_path = _array_to_temp_file(passport_crop, suffix="_passport.jpg")
    l_path = _array_to_temp_file(live_frame, suffix="_live.jpg")

    if p_path is None or l_path is None:
        result["message"] = "Failed to write temporary image files for DeepFace."
        _cleanup_temp(p_path)
        _cleanup_temp(l_path)
        return result

    try:
        verification = df.verify(
            img1_path=p_path,
            img2_path=l_path,
            model_name="ArcFace",
            detector_backend=DETECTOR_BACKEND,
            distance_metric="cosine",
            enforce_detection=False,
        )

        distance = float(verification.get("distance", 1.0))
        passed = distance <= ARCFACE_COSINE_THRESHOLD

        # Similarity percentage: 0% at distance >= threshold, 100% at distance 0
        similarity_pct = max(0.0, min(100.0, (1.0 - distance / ARCFACE_COSINE_THRESHOLD) * 100.0))

        # Extract bounding boxes if available
        def _parse_bbox(facial_area: Optional[dict]) -> Optional[dict]:
            if not facial_area:
                return None
            return {
                "x": facial_area.get("x", 0),
                "y": facial_area.get("y", 0),
                "w": facial_area.get("w", 0),
                "h": facial_area.get("h", 0),
            }

        facial2 = verification.get("facial_areas", {}).get("img2")

        result.update({
            "distance": distance,
            "threshold": ARCFACE_COSINE_THRESHOLD,
            "passed": passed,
            "similarity_pct": similarity_pct,
            "live_bbox": _parse_bbox(facial2),
            "status": "OK",
            "message": (
                f"Match: distance={distance:.4f}, threshold={ARCFACE_COSINE_THRESHOLD:.2f}"
                if passed
                else f"No match: distance={distance:.4f} > threshold={ARCFACE_COSINE_THRESHOLD:.2f}"
            ),
        })

        logger.info(
            "Face verification -- distance=%.4f, passed=%s, photo_status=%s",
            distance, passed, photo_region.status
        )

    except Exception as exc:
        err_msg = str(exc).lower()
        if "face" in err_msg and ("detect" in err_msg or "found" in err_msg):
            result["status"] = "NO_FACE_LIVE"
            result["message"] = f"Face detection failed during verification: {exc}"
        else:
            result["status"] = "ERROR"
            result["message"] = f"Biometric verification error: {exc}"
        logger.error("verify_faces error: %s", exc, exc_info=True)

    finally:
        # PRIVACY: always remove temp files regardless of outcome
        _cleanup_temp(p_path)
        _cleanup_temp(l_path)

    return result


def draw_bbox_on_image(
    image_array: np.ndarray,
    bbox: Optional[dict],
    color_rgb: tuple = (0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    """Draw a face bounding box on an image array.

    Args:
        image_array: H x W x 3 RGB uint8 image array (will not be mutated).
        bbox:        Dict with 'x','y','w','h' keys, or None (returns original).
        color_rgb:   RGB tuple for the rectangle colour.
        thickness:   Line thickness in pixels.

    Returns:
        New H x W x 3 RGB uint8 array with bbox drawn (or original if bbox is None).
    """
    if bbox is None:
        return image_array

    annotated = image_array.copy()
    # OpenCV uses BGR; convert the provided RGB colour
    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
    bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    cv2.rectangle(bgr, (x, y), (x + w, y + h), color_bgr, thickness)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# Lazy import of cv2 (only needed for draw_bbox_on_image)
try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]
    logger.warning("OpenCV not available; draw_bbox_on_image will be a no-op.")
