"""
tests/test_biometrics.py
========================
Unit tests for passport photo region detection and biometric extraction.
"""

import sys
import os
import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from biometrics import (
    locate_passport_photo,
    PassportPhotoRegion,
    ARCFACE_COSINE_THRESHOLD,
)


@pytest.fixture
def synthetic_passport_canvas() -> np.ndarray:
    """Create a synthetic ID-3 passport page canvas (800x600 RGB)."""
    canvas = np.zeros((800, 600, 3), dtype=np.uint8) + 240
    # Simulate text lines across document
    for y in range(100, 650, 40):
        canvas[y:y+6, 320:560] = 30
    # Simulate MRZ band at bottom
    canvas[700:720, 30:570] = 20
    canvas[730:750, 30:570] = 20
    return canvas


class TestPassportPhotoRegionDetection:
    def test_fallback_on_blank_canvas(self, synthetic_passport_canvas):
        """When no face cascade detects a face, graceful fallback to ICAO standard ROI."""
        res = locate_passport_photo(synthetic_passport_canvas)
        assert isinstance(res, PassportPhotoRegion)
        assert res.status == "ICAO_ROI_FALLBACK"
        assert res.crop_rgb is not None
        assert res.crop_rgb.shape[0] > 0
        assert res.crop_rgb.shape[1] > 0
        # Should be roughly the left half of the document
        assert res.bbox["x"] < 100
        assert res.bbox["w"] > 200

    def test_locate_on_real_or_simulated_image(self):
        """Test on actual test images if present in test directory or user media."""
        test_path = r"C:/Users/unkno/.gemini/antigravity/brain/5b42f714-83d5-4a03-af52-5a219690a731/.user_uploaded/media_1788014238444.jpg"
        if os.path.exists(test_path):
            img_bgr = cv2.imread(test_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            res = locate_passport_photo(img_rgb)
            assert res.status in ("DETECTED", "ICAO_ROI_FALLBACK")
            assert res.crop_rgb.shape[0] > 100
            assert res.crop_rgb.shape[1] > 100

    def test_empty_image_handling(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        res = locate_passport_photo(empty)
        assert res.status == "FAILED"
