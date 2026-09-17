"""
Master pipeline integrating preprocessing, morphological segmentation,
contour metrology, classification, and visual rendering.
"""

import time
import os
from typing import Tuple, Dict, Any, Optional, Union
import cv2
import numpy as np

from visioninspect.config import InspectionConfig
from visioninspect.preprocessing.enhancement import ImageEnhancer
from visioninspect.features.edges_contours import ContourAnalyzer
from visioninspect.detector.morphological_detector import MorphologicalDefectDetector
from visioninspect.detector.classifier import DefectClassifier
from visioninspect.metrology.measurement import MetrologyEngine, SampleInspectionResult
from visioninspect.visualization.annotator import VisualAnnotator
from visioninspect.visualization.summary_card import DiagnosticCardGenerator


class InspectionPipeline:
    """End-to-end industrial visual inspection pipeline."""

    def __init__(self, config: Optional[InspectionConfig] = None):
        self.config = config or InspectionConfig()
        self.enhancer = ImageEnhancer(self.config.preprocessing)
        self.detector = MorphologicalDefectDetector(self.config.morphology, self.config.gabor)
        self.metrology = MetrologyEngine(self.config.metrology)

    def inspect(
        self,
        image_input: Union[str, np.ndarray],
        sample_name: Optional[str] = None
    ) -> Tuple[SampleInspectionResult, Dict[str, np.ndarray]]:
        """
        Executes full inspection workflow on an input image path or numpy array:
        1. Ingestion
        2. Conditioning & Enhancement
        3. Multi-modal Anomaly Segmentation
        4. Contour Extraction & Shape Metrics
        5. Defect Classification & Metrology
        6. Visual Rendering & Diagnostic Panel Assembly
        """
        start_time = time.perf_counter()

        # 1. Image Ingestion
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Inspection target image not found: {image_input}")
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Failed to decode image from path: {image_input}")
            name = sample_name or os.path.basename(image_input)
        else:
            image = image_input
            name = sample_name or "sample_memory"

        # 2. Conditioning & Preprocessing
        enhanced = self.enhancer.process(image)

        # 3. Defect Segmentation
        detection_data = self.detector.detect(enhanced)
        binary_mask = detection_data["binary_mask"]
        heatmap = detection_data["heatmap"]
        dark_residual = detection_data["blackhat_map"]
        bright_residual = detection_data["tophat_map"]

        # 4. Contour Boundary Extraction & Shape Analysis
        raw_contours = ContourAnalyzer.extract_contours(binary_mask)
        defect_reports = []

        defect_counter = 1
        for cnt in raw_contours:
            metrics = ContourAnalyzer.compute_metrics(cnt, contour_id=defect_counter)
            if metrics is None:
                continue

            # Classify defect
            defect_class, confidence, reasons = DefectClassifier.classify_defect(
                metrics=metrics,
                original_image=image,
                dark_residual=dark_residual,
                bright_residual=bright_residual
            )

            # Measure defect and grade severity
            report = self.metrology.measure_defect(
                metrics=metrics,
                defect_class=defect_class,
                confidence=confidence,
                reasons=reasons
            )
            defect_reports.append(report)
            defect_counter += 1

        # Calculate processing latency
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 5. Summarize sample PASS / FAIL verdict
        summary = self.metrology.summarize_sample(
            sample_name=name,
            defects=defect_reports,
            elapsed_time_ms=elapsed_ms
        )

        # 6. Render Visual Overlays & Diagnostic Panels
        annotated_image = VisualAnnotator.render_overlay(
            image=image,
            inspection_result=summary,
            binary_mask=binary_mask
        )

        quad_panel = DiagnosticCardGenerator.create_quad_panel(
            original_image=image,
            enhanced_image=enhanced,
            heatmap=heatmap,
            annotated_image=annotated_image
        )

        output_images = {
            "original": image,
            "enhanced": enhanced,
            "binary_mask": binary_mask,
            "heatmap": heatmap,
            "annotated": annotated_image,
            "quad_panel": quad_panel,
        }

        return summary, output_images
