"""
Quantitative evaluation benchmark: Pixel IoU, Dice/F1, Precision, Recall, and confusion metrics.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np


@dataclass
class EvaluationMetrics:
    """Quantitative performance metrics comparing prediction mask against ground truth."""
    sample_name: str
    pixel_iou: float
    dice_coefficient: float
    precision: float
    recall: float
    specificity: float
    accuracy: float
    true_positive_pixels: int
    false_positive_pixels: int
    false_negative_pixels: int
    true_negative_pixels: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_name": self.sample_name,
            "pixel_iou": round(self.pixel_iou, 4),
            "dice_coefficient": round(self.dice_coefficient, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "specificity": round(self.specificity, 4),
            "accuracy": round(self.accuracy, 4),
            "tp_px": self.true_positive_pixels,
            "fp_px": self.false_positive_pixels,
            "fn_px": self.false_negative_pixels,
            "tn_px": self.true_negative_pixels,
        }


class BenchmarkEvaluator:
    """Evaluates predicted defect segmentation masks against annotated ground truth binary masks."""

    @staticmethod
    def evaluate_sample(
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        sample_name: str = "sample"
    ) -> EvaluationMetrics:
        """
        Computes pixel-level confusion matrix and derived segmentation benchmark scores.
        """
        if pred_mask.shape != gt_mask.shape:
            raise ValueError(f"Shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}")

        bin_pred = (pred_mask > 0).astype(np.uint8)
        bin_gt = (gt_mask > 0).astype(np.uint8)

        tp = int(np.sum((bin_pred == 1) & (bin_gt == 1)))
        fp = int(np.sum((bin_pred == 1) & (bin_gt == 0)))
        fn = int(np.sum((bin_pred == 0) & (bin_gt == 1)))
        tn = int(np.sum((bin_pred == 0) & (bin_gt == 0)))

        # Intersection over Union (IoU)
        intersection = tp
        union = tp + fp + fn
        iou = float(intersection) / union if union > 0 else (1.0 if (tp + fp + fn) == 0 else 0.0)

        # Dice / F1
        dice = (2.0 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0

        # Precision & Recall
        precision = float(tp) / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
        recall = float(tp) / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)

        # Specificity
        specificity = float(tn) / (tn + fp) if (tn + fp) > 0 else 1.0

        # Overall Pixel Accuracy
        total_pixels = tp + fp + fn + tn
        accuracy = float(tp + tn) / total_pixels if total_pixels > 0 else 1.0

        return EvaluationMetrics(
            sample_name=sample_name,
            pixel_iou=iou,
            dice_coefficient=dice,
            precision=precision,
            recall=recall,
            specificity=specificity,
            accuracy=accuracy,
            true_positive_pixels=tp,
            false_positive_pixels=fp,
            false_negative_pixels=fn,
            true_negative_pixels=tn,
        )

    @classmethod
    def aggregate_metrics(cls, metrics_list: List[EvaluationMetrics]) -> Dict[str, Any]:
        """Averages metrics across a test set of ground-truth annotated samples."""
        if not metrics_list:
            return {}

        n = len(metrics_list)
        return {
            "num_samples": n,
            "mean_pixel_iou": round(sum(m.pixel_iou for m in metrics_list) / n, 4),
            "mean_dice_f1": round(sum(m.dice_coefficient for m in metrics_list) / n, 4),
            "mean_precision": round(sum(m.precision for m in metrics_list) / n, 4),
            "mean_recall": round(sum(m.recall for m in metrics_list) / n, 4),
            "mean_specificity": round(sum(m.specificity for m in metrics_list) / n, 4),
            "mean_accuracy": round(sum(m.accuracy for m in metrics_list) / n, 4),
        }
