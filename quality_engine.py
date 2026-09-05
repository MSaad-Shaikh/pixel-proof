"""
quality_engine.py  Document Image Quality Assessment Engine
============================================================
Pre-flight visual validation for passport and ID inspection.

Checks performed:
  1. Blur Detection (Laplacian variance)
  2. Glare and Overexposure Detection (Specular reflection / high-luminance saturation)
  3. Low-Light and Underexposure Detection (Mean luminance and dark pixel ratio)
  4. Cropping, Framing and Orientation Analysis (Boundary clearance and aspect ratio)

Zero Streamlit imports - purely functional and UI-agnostic.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Calibration Thresholds (CALIBRATE_ME)
# ---------------------------------------------------------------------------
# Laplacian variance threshold below which an image is considered blurry
BLUR_THRESHOLD_STRICT: float = 80.0
BLUR_THRESHOLD_ACCEPTABLE: float = 40.0

# Glare threshold: percentage of pixels with luminance >= 250
GLARE_PIXEL_INTENSITY: int = 250
GLARE_MAX_RATIO_ALERT: float = 0.04  # 4% of pixels saturated

# Low light thresholds
MIN_MEAN_BRIGHTNESS: float = 65.0    # 0-255 grayscale mean
MAX_DARK_PIXEL_RATIO: float = 0.35   # fraction of pixels with luminance < 30


@dataclass
class QualityReport:
    """Structured document image quality report."""
    passed: bool
    blur_score: float
    is_blurry: bool
    glare_ratio: float
    has_glare: bool
    mean_brightness: float
    is_underexposed: bool
    is_clipped: bool
    aspect_ratio: float
    orientation_status: str
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    quality_grade: str = "GOOD"  # EXCELLENT, GOOD, FAIR, POOR


def detect_blur(gray_image: np.ndarray) -> Tuple[float, bool, str]:
    """Compute blur metric using Laplacian variance.

    Args:
        gray_image: Grayscale uint8 2D numpy array.

    Returns:
        Tuple of (laplacian_variance, is_blurry, explanation).
    """
    score = float(cv2.Laplacian(gray_image, cv2.CV_64F).var())
    is_blurry = score < BLUR_THRESHOLD_ACCEPTABLE
    
    if score >= BLUR_THRESHOLD_STRICT:
        msg = f"Sharp focus (Laplacian variance: {score:.1f})"
    elif score >= BLUR_THRESHOLD_ACCEPTABLE:
        msg = f"Moderate sharpness (Laplacian variance: {score:.1f} - acceptable for OCR)"
    else:
        msg = f"Blurry image detected (Laplacian variance: {score:.1f} < {BLUR_THRESHOLD_ACCEPTABLE:.0f}). Text/MRZ details may be compromised."
    
    return score, is_blurry, msg


def detect_glare(gray_image: np.ndarray) -> Tuple[float, bool, str]:
    """Detect specular reflections and overexposure hotspots.

    Args:
        gray_image: Grayscale uint8 2D numpy array.

    Returns:
        Tuple of (saturated_pixel_ratio, has_glare, explanation).
    """
    total_pixels = gray_image.size
    if total_pixels == 0:
        return 0.0, False, "Empty image"

    saturated_count = int(np.sum(gray_image >= GLARE_PIXEL_INTENSITY))
    glare_ratio = float(saturated_count / total_pixels)
    has_glare = glare_ratio > GLARE_MAX_RATIO_ALERT

    if has_glare:
        msg = f"High glare detected: {glare_ratio*100:.1f}% saturated pixels (Threshold: {GLARE_MAX_RATIO_ALERT*100:.0f}%). Reflections can obscure MRZ check digits."
    else:
        msg = f"Glare level normal ({glare_ratio*100:.2f}% saturated)"

    return glare_ratio, has_glare, msg


def detect_low_light(gray_image: np.ndarray) -> Tuple[float, float, bool, str]:
    """Detect underexposure or insufficient illumination.

    Args:
        gray_image: Grayscale uint8 2D numpy array.

    Returns:
        Tuple of (mean_brightness, dark_ratio, is_underexposed, explanation).
    """
    mean_brightness = float(np.mean(gray_image))
    dark_pixels = int(np.sum(gray_image < 30))
    dark_ratio = float(dark_pixels / gray_image.size) if gray_image.size > 0 else 0.0

    is_underexposed = (mean_brightness < MIN_MEAN_BRIGHTNESS) or (dark_ratio > MAX_DARK_PIXEL_RATIO)

    if is_underexposed:
        msg = f"Low-light / underexposed: Mean luminance {mean_brightness:.1f}/255 ({dark_ratio*100:.1f}% shadow pixels). OCR contrast is degraded."
    else:
        msg = f"Illumination normal (Mean luminance: {mean_brightness:.1f}/255)"

    return mean_brightness, dark_ratio, is_underexposed, msg


def analyze_framing_and_borders(image_rgb: np.ndarray) -> Tuple[float, str, bool, str]:
    """Examine image framing, aspect ratio, and boundary margin clearances.

    Args:
        image_rgb: RGB uint8 3D numpy array.

    Returns:
        Tuple of (aspect_ratio, orientation_string, is_clipped, explanation).
    """
    h, w = image_rgb.shape[:2]
    if h == 0 or w == 0:
        return 1.0, "UNKNOWN", True, "Empty image"

    aspect_ratio = float(w / h)
    
    # Typical standard ICAO ID-3 (passport bio page) is approx 1.42 landscape or 0.70 portrait
    if aspect_ratio > 1.1:
        orientation = "Landscape"
    elif aspect_ratio < 0.9:
        orientation = "Portrait"
    else:
        orientation = "Square / Near-square"

    # Boundary check: inspect edge pixels to see if document is clipped right against edges
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if len(image_rgb.shape) == 3 else image_rgb
    margin = max(4, int(min(h, w) * 0.02))  # 2% edge margin

    # Sample borders
    top_edge = gray[:margin, :]
    bottom_edge = gray[-margin:, :]
    left_edge = gray[:, :margin]
    right_edge = gray[:, -margin:]

    # If edges have very high variance or text-like high contrast, document might be tightly cropped/clipped
    edge_vars = [float(np.var(edge)) for edge in (top_edge, bottom_edge, left_edge, right_edge)]
    max_edge_var = max(edge_vars)
    
    # High variance at the very edge often indicates text/MRZ cutoff
    is_clipped = max_edge_var > 1200.0

    if is_clipped:
        msg = "Tight framing/clipping detected near image borders. Ensure full document corners and MRZ lines are inside the frame."
    else:
        msg = f"Framing acceptable ({w}x{h}, {orientation})"

    return aspect_ratio, orientation, is_clipped, msg


def assess_document_quality(image_input) -> QualityReport:
    """Perform complete pre-flight document quality assessment.

    Args:
        image_input: File path (str), PIL Image, or RGB numpy array.

    Returns:
        QualityReport with all diagnostic scores and actionable recommendations.
    """
    # Normalize input to RGB numpy array
    if isinstance(image_input, str):
        bgr = cv2.imread(image_input)
        if bgr is None:
            raise ValueError(f"Could not load image: {image_input}")
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif hasattr(image_input, 'convert'):
        img_rgb = np.array(image_input.convert('RGB'))
    elif isinstance(image_input, np.ndarray):
        img_rgb = image_input
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Grayscale conversion
    if len(img_rgb.shape) == 3 and img_rgb.shape[2] == 3:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb

    # Run checks
    blur_score, is_blurry, blur_msg = detect_blur(gray)
    glare_ratio, has_glare, glare_msg = detect_glare(gray)
    mean_bright, dark_ratio, is_underexposed, light_msg = detect_low_light(gray)
    aspect_ratio, orientation, is_clipped, frame_msg = analyze_framing_and_borders(img_rgb)

    issues: list[str] = []
    recommendations: list[str] = []

    if is_blurry:
        issues.append("Image is blurry or out of focus")
        recommendations.append("Hold the camera steady or tap to refocus directly on document text.")

    if has_glare:
        issues.append("Reflective glare / white flash overexposure detected")
        recommendations.append("Tilt the document slightly or move away from overhead glare/direct flash.")

    if is_underexposed:
        issues.append("Low ambient lighting / dark image")
        recommendations.append("Increase ambient room lighting or turn on diffuse light.")

    if is_clipped:
        issues.append("Document text or MRZ may be cut off near margins")
        recommendations.append("Include a small border margin around all 4 edges of the document.")

    # Grade computation
    fail_count = int(is_blurry) + int(has_glare) + int(is_underexposed) + int(is_clipped)
    if fail_count == 0 and blur_score > BLUR_THRESHOLD_STRICT and 80 < mean_bright < 200:
        grade = "EXCELLENT"
        passed = True
    elif fail_count == 0:
        grade = "GOOD"
        passed = True
    elif fail_count == 1:
        grade = "FAIR"
        passed = True
    else:
        grade = "POOR"
        passed = False

    return QualityReport(
        passed=passed,
        blur_score=blur_score,
        is_blurry=is_blurry,
        glare_ratio=glare_ratio,
        has_glare=has_glare,
        mean_brightness=mean_bright,
        is_underexposed=is_underexposed,
        is_clipped=is_clipped,
        aspect_ratio=aspect_ratio,
        orientation_status=orientation,
        issues=issues,
        recommendations=recommendations,
        quality_grade=grade,
    )


# ---------------------------------------------------------------------------
# Image Pre-processing and Transformation Utilities
# ---------------------------------------------------------------------------

def rotate_image(image_rgb: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate image by arbitrary angle in degrees clockwise.

    Supports both fast 90/180/270 degree cardinal rotations and arbitrary
    floating-point angle deskewing without clipping corners.

    Args:
        image_rgb:     H x W x 3 RGB uint8 image array.
        angle_degrees: Clockwise rotation angle in degrees (e.g. -180 to +180).

    Returns:
        Rotated H' x W' x 3 RGB uint8 array.
    """
    normalized_deg = float(angle_degrees % 360)
    if normalized_deg == 0.0:
        return image_rgb

    # Fast lossless exact 90-deg transforms
    if normalized_deg == 90.0:
        return cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)
    elif normalized_deg == 180.0:
        return cv2.rotate(image_rgb, cv2.ROTATE_180)
    elif normalized_deg == 270.0:
        return cv2.rotate(image_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Arbitrary angle rotation using Pillow with bounding box expansion and anti-aliasing
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(image_rgb)
    # PIL rotate uses counter-clockwise positive degrees, so pass -angle for clockwise
    rotated_pil = pil_img.rotate(-normalized_deg, resample=PILImage.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    return np.array(rotated_pil)


def crop_image_normalized(image_rgb: np.ndarray, top_pct: float, bottom_pct: float, left_pct: float, right_pct: float) -> np.ndarray:
    """Crop image using normalized margins (0.0 - 1.0)."""
    h, w = image_rgb.shape[:2]
    y1 = max(0, min(h - 1, int(top_pct * h)))
    y2 = max(y1 + 1, min(h, int((1.0 - bottom_pct) * h)))
    x1 = max(0, min(w - 1, int(left_pct * w)))
    x2 = max(x1 + 1, min(w, int((1.0 - right_pct) * w)))
    return image_rgb[y1:y2, x1:x2]
