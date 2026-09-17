"""
Unit tests for morphological defect detection, Gabor wavelets, and classification.
"""

import pytest
import numpy as np
import cv2

from visioninspect.config import MorphologicalConfig, GaborConfig, DefectClass
from visioninspect.detector.morphological_detector import MorphologicalDefectDetector
from visioninspect.detector.classifier import DefectClassifier
from visioninspect.features.edges_contours import ContourAnalyzer


@pytest.fixture
def image_with_crack():
    """Generates an image with a clear dark crack."""
    img = np.full((200, 200, 3), 180, dtype=np.uint8)
    # Draw winding dark crack
    pts = [(30, 40), (60, 55), (90, 50), (130, 80), (160, 85)]
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], (20, 20, 20), 4)
    return img


@pytest.fixture
def image_with_pit():
    """Generates an image with a compact circular dark pit."""
    img = np.full((200, 200, 3), 180, dtype=np.uint8)
    cv2.circle(img, (100, 100), 12, (20, 20, 20), -1)
    return img


def test_morphological_crack_detection(image_with_crack):
    config = MorphologicalConfig(min_defect_area_px=20)
    detector = MorphologicalDefectDetector(morph_config=config)
    res = detector.detect(image_with_crack)

    binary_mask = res["binary_mask"]
    assert np.any(binary_mask > 0), "Detector should segment crack pixels"

    contours = ContourAnalyzer.extract_contours(binary_mask)
    assert len(contours) >= 1

    metrics = ContourAnalyzer.compute_metrics(contours[0], contour_id=1)
    assert metrics is not None
    assert metrics.area_px >= 20

    defect_class, conf, _ = DefectClassifier.classify_defect(
        metrics=metrics,
        original_image=image_with_crack,
        dark_residual=res["blackhat_map"],
        bright_residual=res["tophat_map"]
    )
    assert defect_class in (DefectClass.CRACK, DefectClass.SCRATCH)
    assert conf > 0.5


def test_morphological_pit_detection(image_with_pit):
    config = MorphologicalConfig(min_defect_area_px=20)
    detector = MorphologicalDefectDetector(morph_config=config)
    res = detector.detect(image_with_pit)

    binary_mask = res["binary_mask"]
    assert np.any(binary_mask > 0), "Detector should segment pit pixels"

    contours = ContourAnalyzer.extract_contours(binary_mask)
    assert len(contours) == 1

    metrics = ContourAnalyzer.compute_metrics(contours[0], contour_id=1)
    assert metrics is not None
    # A circle should have high circularity (> 0.6)
    assert metrics.circularity >= 0.55

    defect_class, conf, _ = DefectClassifier.classify_defect(
        metrics=metrics,
        original_image=image_with_pit,
        dark_residual=res["blackhat_map"],
        bright_residual=res["tophat_map"]
    )
    assert defect_class == DefectClass.PIT


def test_clean_surface_no_false_positives():
    """Verifies that a uniform defect-free surface produces zero detected contours."""
    clean_img = np.full((200, 200, 3), 160, dtype=np.uint8)
    detector = MorphologicalDefectDetector()
    res = detector.detect(clean_img)

    binary_mask = res["binary_mask"]
    assert np.sum(binary_mask) == 0, "Pristine surface must not produce false positive detections"
