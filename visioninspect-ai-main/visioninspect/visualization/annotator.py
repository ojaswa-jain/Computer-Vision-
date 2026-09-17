"""
Industrial inspection HUD and high-contrast defect visual annotator.
"""

from typing import List, Tuple, Dict
import cv2
import numpy as np

from visioninspect.config import DefectSeverity
from visioninspect.metrology.measurement import DefectReport, SampleInspectionResult


class VisualAnnotator:
    """Renders industrial inspection overlays, defect bounding annotations, and telemetry HUD."""

    # BGR Color Palette
    SEVERITY_COLORS: Dict[DefectSeverity, Tuple[int, int, int]] = {
        DefectSeverity.CRITICAL: (0, 0, 230),    # Bright Red
        DefectSeverity.MAJOR: (0, 140, 255),    # Vibrant Orange
        DefectSeverity.MINOR: (0, 220, 255),    # Vibrant Yellow
        DefectSeverity.PASS: (50, 205, 50),     # Lime Green
    }

    @classmethod
    def render_overlay(
        cls,
        image: np.ndarray,
        inspection_result: SampleInspectionResult,
        binary_mask: np.ndarray = None
    ) -> np.ndarray:
        """
        Renders complete multi-layer diagnostic overlay:
        1. Translucent mask overlay
        2. Oriented and axis-aligned bounding boxes
        3. Defect badges and callout text
        4. Top telemetry HUD
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]

        # 1. Semi-transparent defect highlight mask
        if binary_mask is not None and np.any(binary_mask > 0):
            colored_mask = np.zeros_like(annotated)
            colored_mask[binary_mask > 0] = (0, 50, 200)  # Red translucent tint
            annotated = cv2.addWeighted(annotated, 0.85, colored_mask, 0.15, 0)

        # 2. Draw each defect contour and bounding box
        for defect in inspection_result.defects:
            color = cls.SEVERITY_COLORS.get(defect.severity, (255, 255, 255))

            # Draw minimum area rotated rectangle
            if defect.min_rect_box is not None and len(defect.min_rect_box) == 4:
                cv2.drawContours(annotated, [defect.min_rect_box], 0, color, 2, cv2.LINE_AA)

            # Draw centroid marker
            cx, cy = defect.centroid
            cv2.circle(annotated, (cx, cy), 4, color, -1, cv2.LINE_AA)
            cv2.circle(annotated, (cx, cy), 7, (255, 255, 255), 1, cv2.LINE_AA)

            # Defect callout badge
            bx, by, bw, bh = defect.bbox_xywh
            label = f"#{defect.defect_id} {defect.defect_class.value} [{defect.severity.value}]"
            dim_label = f"{defect.length_mm:.2f}x{defect.width_mm:.2f}mm"

            # Badge background
            label_y = max(20, by - 8)
            cv2.putText(annotated, label, (bx, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated, label, (bx, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            cv2.putText(annotated, dim_label, (bx, label_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)

        # 3. Top telemetry HUD
        hud_height = 48
        hud_bg = np.zeros((hud_height, w, 3), dtype=np.uint8)
        
        status_color = (0, 200, 0) if inspection_result.overall_status == "PASS" else (0, 0, 220)
        cv2.rectangle(hud_bg, (0, 0), (w, hud_height), (25, 25, 30), -1)
        cv2.line(hud_bg, (0, hud_height - 1), (w, hud_height - 1), (60, 60, 70), 1)

        # Status badge
        cv2.rectangle(hud_bg, (10, 8), (110, 40), status_color, -1)
        cv2.putText(hud_bg, inspection_result.overall_status, (22, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Telemetry metrics
        hud_text = (
            f"Sample: {inspection_result.sample_name} | "
            f"Defects: {inspection_result.total_defects} | "
            f"Crit: {inspection_result.severity_breakdown.get('CRITICAL', 0)} "
            f"Maj: {inspection_result.severity_breakdown.get('MAJOR', 0)} "
            f"Min: {inspection_result.severity_breakdown.get('MINOR', 0)} | "
            f"Area: {inspection_result.total_defect_area_mm2:.2f} mm2 | "
            f"Time: {inspection_result.processing_time_ms:.1f}ms"
        )
        cv2.putText(hud_bg, hud_text, (125, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

        # Composite HUD with annotated image
        final_view = np.vstack([hud_bg, annotated])
        return final_view
