"""
Multi-stage diagnostic comparison cards and visual inspection panels.
"""

from typing import Tuple
import cv2
import numpy as np

from visioninspect.metrology.measurement import SampleInspectionResult
from visioninspect.visualization.annotator import VisualAnnotator


class DiagnosticCardGenerator:
    """Combines inspection pipeline intermediate stages into side-by-side diagnostic cards."""

    @staticmethod
    def _add_stage_title(image: np.ndarray, title: str) -> np.ndarray:
        """Adds banner title to a stage image."""
        img_out = image.copy()
        if len(img_out.shape) == 2:
            img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)

        h, w = img_out.shape[:2]
        title_h = 32
        banner = np.zeros((title_h, w, 3), dtype=np.uint8)
        banner[:] = (35, 35, 40)
        cv2.putText(banner, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.line(banner, (0, title_h - 1), (w, title_h - 1), (80, 80, 90), 1)

        return np.vstack([banner, img_out])

    @classmethod
    def create_quad_panel(
        cls,
        original_image: np.ndarray,
        enhanced_image: np.ndarray,
        heatmap: np.ndarray,
        annotated_image: np.ndarray,
        target_cell_size: Tuple[int, int] = (400, 400)
    ) -> np.ndarray:
        """
        Creates a 2x2 grid diagnostic panel:
        [ (1) Original Input        | (2) Preprocessed / CLAHE  ]
        [ (3) Anomaly Heatmap (Jet) | (4) Annotated Inspection  ]
        """
        cell_w, cell_h = target_cell_size

        # Resize each cell to uniform dimensions
        cell1 = cv2.resize(original_image, (cell_w, cell_h))
        cell2 = cv2.resize(enhanced_image, (cell_w, cell_h))

        # Colorize heatmap with JET colormap
        if len(heatmap.shape) == 2:
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        else:
            heatmap_colored = heatmap
        cell3 = cv2.resize(heatmap_colored, (cell_w, cell_h))

        # Annotated image
        cell4 = cv2.resize(annotated_image, (cell_w, cell_h))

        # Title banners
        titled1 = cls._add_stage_title(cell1, "1. Raw Surface Ingestion")
        titled2 = cls._add_stage_title(cell2, "2. CLAHE & Illumination Conditioning")
        titled3 = cls._add_stage_title(cell3, "3. Morphological & Gabor Energy Heatmap")
        titled4 = cls._add_stage_title(cell4, "4. Metrology & Severity Classification")

        # Stack into 2x2 grid
        top_row = np.hstack([titled1, titled2])
        bottom_row = np.hstack([titled3, titled4])
        quad_panel = np.vstack([top_row, bottom_row])

        return quad_panel
