"""
Unit tests for dimensional metrology, physical scaling, and tolerance severity assessment.
"""

import pytest
import numpy as np

from visioninspect.config import MetrologyConfig, DefectSeverity, DefectClass
from visioninspect.metrology.measurement import MetrologyEngine
from visioninspect.features.edges_contours import ContourMetrics


@pytest.fixture
def mock_contour_metrics():
    """Generates synthetic ContourMetrics."""
    return ContourMetrics(
        contour_id=1,
        area_px=1000.0,
        perimeter_px=150.0,
        circularity=0.4,
        aspect_ratio=3.0,
        solidity=0.85,
        extent=0.6,
        eccentricity=0.8,
        bbox_xywh=(10, 10, 60, 20),
        min_rect_box=np.array([[10, 10], [70, 10], [70, 30], [10, 30]]),
        min_rect_dims=(20.0, 60.0),
        centroid=(40, 20),
        raw_contour=np.array([[[10, 10]], [[70, 10]], [[70, 30]], [[10, 30]]])
    )


def test_pixel_to_metric_conversion(mock_contour_metrics):
    # 1 pixel = 0.05 mm -> area: 1000 * 0.0025 = 2.5 mm²
    config = MetrologyConfig(pixel_to_mm_ratio=0.05)
    engine = MetrologyEngine(config)

    report = engine.measure_defect(
        metrics=mock_contour_metrics,
        defect_class=DefectClass.SCRATCH,
        confidence=0.92,
        reasons={}
    )

    assert report.area_mm2 == pytest.approx(2.5, rel=1e-3)
    assert report.length_mm == pytest.approx(60.0 * 0.05, rel=1e-3)
    assert report.width_mm == pytest.approx(20.0 * 0.05, rel=1e-3)
    assert report.perimeter_mm == pytest.approx(150.0 * 0.05, rel=1e-3)


def test_crack_is_critical_by_policy(mock_contour_metrics):
    engine = MetrologyEngine()
    report = engine.measure_defect(
        metrics=mock_contour_metrics,
        defect_class=DefectClass.CRACK,
        confidence=0.95,
        reasons={}
    )
    assert report.severity == DefectSeverity.CRITICAL


def test_pass_fail_summary():
    engine = MetrologyEngine()
    # Sample with zero defects should PASS
    pass_summary = engine.summarize_sample("clean_01", [], 12.5)
    assert pass_summary.overall_status == "PASS"
    assert pass_summary.total_defects == 0
