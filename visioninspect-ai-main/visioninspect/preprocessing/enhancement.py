"""
Image enhancement module implementing illumination normalization,
bilateral edge-preserving filtering, and dynamic range adjustment.
"""

from typing import Optional
import cv2
import numpy as np
from visioninspect.config import PreprocessingConfig


class ImageEnhancer:
    """Handles illumination normalization, noise attenuation, and contrast enhancement."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size
        )

    def apply_gamma_correction(self, image: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
        """
        Apply non-linear power-law gamma transformation: I_out = 255 * (I_in / 255) ** (1 / gamma).
        """
        g = gamma if gamma is not None else self.config.gamma
        if abs(g - 1.0) < 1e-4:
            return image

        inv_gamma = 1.0 / g
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    def apply_clahe(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization to a grayscale image.
        Prevents noise over-amplification in uniform regions while enhancing local micro-contrast.
        """
        if len(gray_image.shape) != 2:
            raise ValueError("CLAHE requires a single-channel 2D grayscale image")
        return self._clahe.apply(gray_image)

    def apply_bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Applies bilateral filtering: smooths surface texture and sensor noise while
        preserving sharp defect boundaries.
        """
        return cv2.bilateralFilter(
            src=image,
            d=self.config.bilateral_d,
            sigmaColor=self.config.bilateral_sigma_color,
            sigmaSpace=self.config.bilateral_sigma_space
        )

    def estimate_background_illumination(self, gray_image: np.ndarray, kernel_size: int = 51) -> np.ndarray:
        """
        Estimates the low-frequency background illumination field using large morphological closing.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)

    def normalize_illumination(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Corrects spatial illumination gradients by dividing by estimated background.
        I_norm = (I / I_bg) * mean(I_bg)
        """
        bg = self.estimate_background_illumination(gray_image)
        bg_float = bg.astype(np.float32) + 1e-5
        img_float = gray_image.astype(np.float32)

        mean_bg = np.mean(bg_float)
        norm = (img_float / bg_float) * mean_bg
        return np.clip(norm, 0, 255).astype(np.uint8)

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Executes full preprocessing pipeline:
        Gamma -> Bilateral Filter -> Grayscale / LAB Luminance CLAHE.
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid or empty input image provided to ImageEnhancer")

        # 1. Non-linear gamma adjustment
        corrected = self.apply_gamma_correction(image)

        # 2. Edge-preserving bilateral filter
        filtered = self.apply_bilateral_filter(corrected)

        # 3. Channel-wise or luminance CLAHE
        if len(filtered.shape) == 3:
            lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_enhanced = self.apply_clahe(l_channel)
            enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            return self.apply_clahe(filtered)
