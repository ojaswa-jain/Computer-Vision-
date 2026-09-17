"""
Color space transformations and chromatic irregularity extraction.
"""

from typing import Tuple, Dict
import cv2
import numpy as np


class ColorSpaceProcessor:
    """Manages color space transformations and chromatic anomaly detection."""

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """Converts BGR image to single-channel 8-bit grayscale."""
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def to_lab(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decomposes BGR image into CIE L* (luminance), a* (green-red), b* (blue-yellow)."""
        if len(image.shape) == 2:
            l = image
            a = np.full_like(image, 128)
            b = np.full_like(image, 128)
            return l, a, b
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        return cv2.split(lab)

    @staticmethod
    def to_hsv(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decomposes BGR image into Hue, Saturation, and Value components."""
        if len(image.shape) == 2:
            v = image
            h = np.zeros_like(image)
            s = np.zeros_like(image)
            return h, s, v
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return cv2.split(hsv)

    @staticmethod
    def compute_chromatic_distance(image: np.ndarray, reference_bgr: Tuple[int, int, int] = None) -> np.ndarray:
        """
        Computes Euclidean color distance in LAB space against an expected surface reference color
        or against the median surface baseline.
        Highlights oxidation spots, thermal discoloration, and chemical stains.
        """
        if len(image.shape) == 2:
            return np.zeros_like(image, dtype=np.uint8)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        if reference_bgr is not None:
            ref_mat = np.uint8([[reference_bgr]])
            ref_lab = cv2.cvtColor(ref_mat, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
        else:
            # Robust median baseline across the sample
            ref_lab = np.median(lab.reshape(-1, 3), axis=0)

        delta_e = np.sqrt(np.sum((lab - ref_lab) ** 2, axis=2))
        norm_delta_e = cv2.normalize(delta_e, None, 0, 255, cv2.NORM_MINMAX)
        return norm_delta_e.astype(np.uint8)

    @classmethod
    def extract_color_channels(cls, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extracts comprehensive channel dictionary for multi-modal feature inspection."""
        gray = cls.to_grayscale(image)
        l, a, b = cls.to_lab(image)
        h, s, v = cls.to_hsv(image)

        return {
            "gray": gray,
            "lab_l": l,
            "lab_a": a,
            "lab_b": b,
            "hsv_h": h,
            "hsv_s": s,
            "hsv_v": v,
        }
