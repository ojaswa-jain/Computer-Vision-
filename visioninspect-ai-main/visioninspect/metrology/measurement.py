"""
Dimensional metrology, metric calibration, and tolerance-based quality assessment.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from visioninspect.config import MetrologyConfig, DefectSeverity, DefectClass
from visioninspect.features.edges_contours import ContourMetrics


@dataclass
class DefectReport:
    """Detailed metrological telemetry for a single detected defect."""
    defect_id: int
    defect_class: DefectClass
    confidence: float
    severity: DefectSeverity
    area_px: float
    area_mm2: float
    perimeter_px: float
    perimeter_mm: float
    length_px: float
    length_mm: float
    width_px: float
    width_mm: float
    aspect_ratio: float
    circularity: float
    centroid: Tuple[int, int]
    bbox_xywh: Tuple[int, int, int, int]
    min_rect_box: np.ndarray
    reasons: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes report to dictionary."""
        return {
            "defect_id": self.defect_id,
            "defect_class": self.defect_class.value,
            "confidence": round(self.confidence, 3),
            "severity": self.severity.value,
            "area_px": round(self.area_px, 2),
            "area_mm2": round(self.area_mm2, 4),
            "perimeter_px": round(self.perimeter_px, 2),
            "perimeter_mm": round(self.perimeter_mm, 3),
            "length_px": round(self.length_px, 2),
            "length_mm": round(self.length_mm, 3),
            "width_px": round(self.width_px, 2),
            "width_mm": round(self.width_mm, 3),
            "aspect_ratio": round(self.aspect_ratio, 2),
            "circularity": round(self.circularity, 3),
            "centroid": list(self.centroid),
            "bbox_xywh": list(self.bbox_xywh),
            "reasons": self.reasons,
        }


@dataclass
class SampleInspectionResult:
    """Summary of inspection findings across a single sample."""
    sample_name: str
    overall_status: str  # "PASS" or "FAIL"
    total_defects: int
    defects: List[DefectReport]
    severity_breakdown: Dict[str, int]
    class_breakdown: Dict[str, int]
    total_defect_area_mm2: float
    processing_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_name": self.sample_name,
            "overall_status": self.overall_status,
            "total_defects": self.total_defects,
            "severity_breakdown": self.severity_breakdown,
            "class_breakdown": self.class_breakdown,
            "total_defect_area_mm2": round(self.total_defect_area_mm2, 4),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "defects": [d.to_dict() for d in self.defects],
        }


class MetrologyEngine:
    """Converts geometric pixel measurements to physical units and grades severity."""

    def __init__(self, config: Optional[MetrologyConfig] = None):
        self.config = config or MetrologyConfig()

    def evaluate_severity(
        self,
        area_mm2: float,
        length_mm: float,
        defect_class: DefectClass
    ) -> DefectSeverity:
        """
        Determines severity based on dimensional tolerance rules:
        - CRITICAL: Cracks, or defects exceeding critical area / length threshold
        - MAJOR: Area exceeds major threshold
        - MINOR: Area exceeds minor threshold
        - PASS: Below minor defect threshold
        """
        if defect_class.value in self.config.critical_defect_classes:
            return DefectSeverity.CRITICAL

        if (area_mm2 >= self.config.critical_area_mm2_threshold or
                length_mm >= self.config.critical_length_mm_threshold):
            return DefectSeverity.CRITICAL

        if area_mm2 >= self.config.major_area_mm2_threshold:
            return DefectSeverity.MAJOR

        if area_mm2 >= self.config.minor_area_mm2_threshold:
            return DefectSeverity.MINOR

        return DefectSeverity.PASS

    def measure_defect(
        self,
        metrics: ContourMetrics,
        defect_class: DefectClass,
        confidence: float,
        reasons: Dict[str, Any]
    ) -> DefectReport:
        """
        Transforms pixel contour metrics to physical units and compiles defect telemetry.
        """
        px_to_mm = self.config.pixel_to_mm_ratio
        px2_to_mm2 = px_to_mm * px_to_mm

        area_mm2 = metrics.area_px * px2_to_mm2
        perimeter_mm = metrics.perimeter_px * px_to_mm
        
        dim_minor_px, dim_major_px = metrics.min_rect_dims
        width_mm = dim_minor_px * px_to_mm
        length_mm = dim_major_px * px_to_mm

        severity = self.evaluate_severity(area_mm2, length_mm, defect_class)

        return DefectReport(
            defect_id=metrics.contour_id,
            defect_class=defect_class,
            confidence=confidence,
            severity=severity,
            area_px=metrics.area_px,
            area_mm2=area_mm2,
            perimeter_px=metrics.perimeter_px,
            perimeter_mm=perimeter_mm,
            length_px=dim_major_px,
            length_mm=length_mm,
            width_px=dim_minor_px,
            width_mm=width_mm,
            aspect_ratio=metrics.aspect_ratio,
            circularity=metrics.circularity,
            centroid=metrics.centroid,
            bbox_xywh=metrics.bbox_xywh,
            min_rect_box=metrics.min_rect_box,
            reasons=reasons,
        )

    def summarize_sample(
        self,
        sample_name: str,
        defects: List[DefectReport],
        elapsed_time_ms: float
    ) -> SampleInspectionResult:
        """
        Summarizes individual defect reports into an overall part Pass/Fail determination.
        """
        severity_counts = {sev.value: 0 for sev in DefectSeverity}
        class_counts = {cls.value: 0 for cls in DefectClass}
        total_area = 0.0

        for d in defects:
            severity_counts[d.severity.value] += 1
            class_counts[d.defect_class.value] += 1
            total_area += d.area_mm2

        # Overall part grading logic:
        # Fails if any CRITICAL or MAJOR defect is detected, or if > 3 MINOR defects are present.
        has_critical = severity_counts[DefectSeverity.CRITICAL.value] > 0
        has_major = severity_counts[DefectSeverity.MAJOR.value] > 0
        excessive_minor = severity_counts[DefectSeverity.MINOR.value] > 3

        overall_status = "FAIL" if (has_critical or has_major or excessive_minor) else "PASS"

        return SampleInspectionResult(
            sample_name=sample_name,
            overall_status=overall_status,
            total_defects=len(defects),
            defects=defects,
            severity_breakdown=severity_counts,
            class_breakdown=class_counts,
            total_defect_area_mm2=total_area,
            processing_time_ms=elapsed_time_ms,
        )
