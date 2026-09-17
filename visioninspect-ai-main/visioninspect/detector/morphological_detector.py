"""
Morphological defect segmentation for surface defect inspection.
"""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np

from visioninspect.config import MorphologicalConfig


class MorphologicalDefectDetector:
    """Detects surface anomalies using Black-Hat and Top-Hat morphological filtering."""

    def __init__(self, morph_config: Optional[MorphologicalConfig] = None, gabor_config=None):
        self.morph_config = morph_config or MorphologicalConfig()
        ksize = max(25, self.morph_config.tophat_kernel_size)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detects anomalies by isolating dark (cracks/pits) and bright (scratches/particles) defects.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 1. Morphological Black-Hat (dark flaws: cracks, pits)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.kernel)
        
        # 2. Morphological Top-Hat (bright flaws: metal chips, bright scratches)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, self.kernel)

        # 3. Direct thresholding on residual
        _, dark_mask = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)
        _, bright_mask = cv2.threshold(tophat, 20, 255, cv2.THRESH_BINARY)

        combined = cv2.bitwise_or(dark_mask, bright_mask)

        # 4. Clean noise with morphological opening
        clean_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, clean_k)

        # 5. Connected component area filter
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        binary_mask = np.zeros_like(cleaned)

        min_area = self.morph_config.min_defect_area_px
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                binary_mask[labels == i] = 255

        # Anomaly heatmap
        residual = cv2.add(blackhat, tophat)
        heatmap = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)

        return {
            "binary_mask": binary_mask,
            "heatmap": heatmap,
            "blackhat_map": blackhat,
            "tophat_map": tophat,
        }
