"""
Edge feature extraction, morphological gradients, and geometric contour analysis.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np


@dataclass
class ContourMetrics:
    """Quantitative geometric measurements for an extracted contour."""
    contour_id: int
    area_px: float
    perimeter_px: float
    circularity: float
    aspect_ratio: float
    solidity: float
    extent: float
    eccentricity: float
    bbox_xywh: Tuple[int, int, int, int]
    min_rect_box: np.ndarray  # 4x2 points
    min_rect_dims: Tuple[float, float]  # (width, length)
    centroid: Tuple[int, int]
    raw_contour: np.ndarray


class EdgeFeatureExtractor:
    """Extracts multi-scale edge gradients and adaptive Canny boundaries."""

    @staticmethod
    def auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
        """
        Computes Canny edges with thresholds automatically derived from the median pixel intensity.
        Avoids brittle manual threshold tuning.
        """
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        return cv2.Canny(image, lower, upper)

    @staticmethod
    def morphological_gradient(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Morphological gradient = Dilation(I) - Erosion(I).
        Reveals sharp intensity transitions along defect boundaries.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

    @staticmethod
    def sobel_magnitude(image: np.ndarray, ksize: int = 3) -> np.ndarray:
        """Computes gradient magnitude using Sobel operators along X and Y."""
        grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=ksize)
        magnitude = cv2.magnitude(grad_x, grad_y)
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


class ContourAnalyzer:
    """Analyzes topological boundaries and computes geometric shape descriptors."""

    @staticmethod
    def extract_contours(binary_mask: np.ndarray) -> List[np.ndarray]:
        """Finds external contours from a binary defect mask."""
        contours, _ = cv2.findContours(
            binary_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return list(contours)

    @classmethod
    def compute_metrics(cls, contour: np.ndarray, contour_id: int = 0) -> Optional[ContourMetrics]:
        """Calculates rich geometric descriptors for a given contour."""
        area = cv2.contourArea(contour)
        if area <= 0:
            return None

        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter <= 0:
            return None

        # Circularity / Compactness: 4 * pi * Area / Perimeter^2
        # (1.0 for a perfect circle, lower for elongated or jagged cracks)
        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        circularity = min(1.0, circularity)

        # Standard axis-aligned bounding box
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 1.0

        # Minimum area rotated rectangle
        min_rect = cv2.minAreaRect(contour)
        (center_x, center_y), (rect_w, rect_h), angle = min_rect
        box_points = cv2.boxPoints(min_rect).astype(np.int32)
        
        # Consistent width (minor) and length (major)
        dim_minor = min(rect_w, rect_h)
        dim_major = max(rect_w, rect_h)

        # Solidity: Area / Convex Hull Area
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0.0

        # Extent: Area / Bounding Box Area
        bbox_area = w * h
        extent = float(area) / bbox_area if bbox_area > 0 else 0.0

        # Eccentricity via image moments
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        mu20 = moments["mu20"]
        mu02 = moments["mu02"]
        mu11 = moments["mu11"]
        delta = np.sqrt(4 * (mu11 ** 2) + ((mu20 - mu02) ** 2))
        denominator = mu20 + mu02 + delta
        if denominator > 0 and (mu20 + mu02 - delta) >= 0:
            eccentricity = np.sqrt(1.0 - ((mu20 + mu02 - delta) / denominator))
        else:
            eccentricity = 0.0

        return ContourMetrics(
            contour_id=contour_id,
            area_px=float(area),
            perimeter_px=float(perimeter),
            circularity=float(circularity),
            aspect_ratio=float(aspect_ratio),
            solidity=float(solidity),
            extent=float(extent),
            eccentricity=float(eccentricity),
            bbox_xywh=(x, y, w, h),
            min_rect_box=box_points,
            min_rect_dims=(float(dim_minor), float(dim_major)),
            centroid=(cx, cy),
            raw_contour=contour
        )
