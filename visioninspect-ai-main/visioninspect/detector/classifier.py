"""
Rule-based and geometric decision engine for classifying segmented defects
into standardized industrial taxonomies.
"""

from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np

from visioninspect.config import DefectClass
from visioninspect.features.edges_contours import ContourMetrics


class DefectClassifier:
    """
    Classifies surface flaws based on topological invariants, geometric aspect ratios,
    circularity, boundary tortuosity, and local photometric contrast.
    """

    @classmethod
    def classify_defect(
        cls,
        metrics: ContourMetrics,
        original_image: np.ndarray,
        dark_residual: Optional[np.ndarray] = None,
        bright_residual: Optional[np.ndarray] = None
    ) -> Tuple[DefectClass, float, Dict[str, Any]]:
        """
        Evaluates contour geometric and photometric properties to assign
        defect taxonomy, confidence, and attribute breakdown.
        """
        # Extract local region of interest (ROI)
        x, y, w, h = metrics.bbox_xywh
        h_img, w_img = original_image.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)

        roi = original_image[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

        # Compute local contrast relative to surroundings
        mean_intensity = float(np.mean(gray_roi))
        dim_minor, dim_major = metrics.min_rect_dims
        length_to_width = dim_major / max(1.0, dim_minor)

        reasons = {}

        # 1. Pits / Voids / Pinholes:
        # High circularity, compact, roughly isotropic dimensions
        if metrics.circularity >= 0.55 and length_to_width < 2.5:
            confidence = min(0.98, 0.60 + (metrics.circularity * 0.38))
            reasons["rule"] = "High circularity and low aspect ratio"
            return DefectClass.PIT, confidence, reasons

        # 2. Scratches:
        # High elongation / aspect ratio, linear profile, relatively high solidity along oriented axis
        if length_to_width >= 3.5 and metrics.circularity < 0.35:
            # Check linearity / thinness
            if dim_minor < 25.0:
                confidence = min(0.97, 0.70 + (min(length_to_width, 10.0) / 10.0) * 0.25)
                reasons["rule"] = "High length-to-width ratio and narrow cross-section"
                return DefectClass.SCRATCH, confidence, reasons

        # 3. Cracks:
        # Elongated but non-linear (tortuous / winding), lower solidity and low circularity
        if metrics.circularity < 0.30 and (metrics.solidity < 0.70 or length_to_width >= 2.5):
            confidence = min(0.96, 0.65 + (1.0 - metrics.solidity) * 0.30)
            reasons["rule"] = "Low solidity, irregular perimeter, and high boundary tortuosity"
            return DefectClass.CRACK, confidence, reasons

        # 4. Foreign Particles / Inclusions:
        # If local bright residual is dominant or local pixel intensity is significantly elevated
        if bright_residual is not None:
            bright_roi = bright_residual[y1:y2, x1:x2]
            if np.mean(bright_roi) > 40:
                confidence = 0.88
                reasons["rule"] = "Dominant high-reflectance optical signature"
                return DefectClass.PARTICLE, confidence, reasons

        # 5. Surface Blemish / Discoloration:
        # Medium circularity, diffuse shape, moderate area
        if metrics.area_px > 100 and metrics.circularity >= 0.25:
            confidence = 0.82
            reasons["rule"] = "Diffuse boundary with moderate area"
            return DefectClass.BLEMISH, confidence, reasons

        # Fallback default classification
        confidence = 0.65
        reasons["rule"] = "Geometric profile matches general abrasive scratch"
        return DefectClass.SCRATCH, confidence, reasons
