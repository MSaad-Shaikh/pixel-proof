"""
tests/test_quality_engine.py
============================
Unit tests for the document quality assessment engine.
"""

import sys
import os
import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quality_engine import (
    detect_blur,
    detect_glare,
    detect_low_light,
    analyze_framing_and_borders,
    assess_document_quality,
    rotate_image,
    crop_image_normalized,
    QualityReport,
)


@pytest.fixture
def synthetic_sharp_image() -> np.ndarray:
    """Create a synthetic high-frequency sharp image with text-like patterns."""
    img = np.zeros((300, 400), dtype=np.uint8) + 128
    # Add high-contrast text-like lines
    for i in range(20, 280, 20):
        img[i:i+4, 20:380] = 20
        img[i+6:i+10, 20:380] = 240
    return img


class TestQualityEngine:
    def test_detect_blur_sharp_vs_blurry(self, synthetic_sharp_image):
        score_sharp, is_blurry_sharp, _ = detect_blur(synthetic_sharp_image)
        assert score_sharp > 80.0
        assert is_blurry_sharp is False

        # Apply severe Gaussian blur
        blurry_img = cv2.GaussianBlur(synthetic_sharp_image, (25, 25), 0)
        score_blurry, is_blurry, msg = detect_blur(blurry_img)
        assert score_blurry < 40.0
        assert is_blurry is True
        assert "Blurry" in msg

    def test_detect_glare(self, synthetic_sharp_image):
        # Normal image without saturated glare
        ratio, has_glare, _ = detect_glare(synthetic_sharp_image)
        assert has_glare is False

        # Introduce high glare spot (e.g. 15% of pixels set to 255)
        glare_img = synthetic_sharp_image.copy()
        glare_img[50:150, 50:200] = 255
        ratio_glare, has_glare, msg = detect_glare(glare_img)
        assert ratio_glare > 0.04
        assert has_glare is True
        assert "glare" in msg.lower()

    def test_detect_low_light(self, synthetic_sharp_image):
        # Normal image
        mean_b, dark_r, is_dark, _ = detect_low_light(synthetic_sharp_image)
        assert is_dark is False
        assert mean_b > 65.0

        # Underexposed dark image
        dark_img = (synthetic_sharp_image.astype(np.float32) * 0.15).astype(np.uint8)
        mean_dark, _, is_dark, msg = detect_low_light(dark_img)
        assert is_dark is True
        assert mean_dark < 65.0
        assert "underexposed" in msg.lower()

    def test_rotate_image(self, synthetic_sharp_image):
        rgb = cv2.cvtColor(synthetic_sharp_image, cv2.COLOR_GRAY2RGB)
        h, w = rgb.shape[:2]
        rot90 = rotate_image(rgb, 90)
        assert rot90.shape[:2] == (w, h)
        rot180 = rotate_image(rgb, 180)
        assert rot180.shape[:2] == (h, w)
        # Test arbitrary angle rotation (e.g. 15 deg tilt)
        rot15 = rotate_image(rgb, 15.0)
        assert rot15.shape[0] > 0
        assert rot15.shape[1] > 0
        # Test negative angle (e.g. -45 deg)
        rot_neg = rotate_image(rgb, -45.0)
        assert rot_neg.shape[0] > 0

    def test_crop_image_normalized(self, synthetic_sharp_image):
        rgb = cv2.cvtColor(synthetic_sharp_image, cv2.COLOR_GRAY2RGB)
        h, w = rgb.shape[:2]
        cropped = crop_image_normalized(rgb, 0.1, 0.1, 0.1, 0.1)
        assert cropped.shape[0] < h
        assert cropped.shape[1] < w

    def test_assess_document_quality_end_to_end(self, synthetic_sharp_image):
        rgb = cv2.cvtColor(synthetic_sharp_image, cv2.COLOR_GRAY2RGB)
        report = assess_document_quality(rgb)
        assert isinstance(report, QualityReport)
        assert report.passed is True
        assert report.quality_grade in ("EXCELLENT", "GOOD")
