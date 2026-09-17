"""
Multi-scale, multi-orientation 2D Gabor wavelet filter bank for surface texture analysis.
"""

from typing import List, Optional
import cv2
import numpy as np
from visioninspect.config import GaborConfig


class GaborFeatureExtractor:
    """
    Constructs a 2D Gabor filter bank to decompose image textures across
    spatial frequencies and orientations, highlighting texture discontinuities and weave defects.
    """

    def __init__(self, config: Optional[GaborConfig] = None):
        self.config = config or GaborConfig()
        self.kernels = self._build_filter_bank()

    def _build_filter_bank(self) -> List[np.ndarray]:
        """Precomputes 2D Gabor kernels for all configured orientation angles."""
        kernels = []
        ksize = self.config.ksize
        if ksize % 2 == 0:
            ksize += 1

        for theta_deg in self.config.theta_angles:
            theta_rad = np.deg2rad(theta_deg)
            kernel = cv2.getGaborKernel(
                ksize=(ksize, ksize),
                sigma=self.config.sigma,
                theta=theta_rad,
                lambd=self.config.lambd,
                gamma=self.config.gamma,
                psi=self.config.psi,
                ktype=cv2.CV_32F
            )
            # Normalize kernel so DC response is zero-centered
            kernel /= 1.5 * kernel.sum() if kernel.sum() != 0 else 1.0
            kernels.append(kernel)
        return kernels

    def filter_image(self, gray_image: np.ndarray) -> List[np.ndarray]:
        """Applies each Gabor filter in the bank to the input grayscale image."""
        responses = []
        for kernel in self.kernels:
            filtered = cv2.filter2D(gray_image, cv2.CV_32F, kernel)
            responses.append(filtered)
        return responses

    def compute_energy_magnitude(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Computes maximum energy response across all filter orientations:
        E(x, y) = max_theta | response_theta(x, y) |
        """
        responses = self.filter_image(gray_image)
        if not responses:
            return np.zeros_like(gray_image, dtype=np.uint8)

        # Compute element-wise maximum across all orientation responses
        abs_responses = [np.abs(r) for r in responses]
        max_response = np.maximum.reduce(abs_responses)

        # If variation is negligible, no texture anomaly exists
        if np.ptp(max_response) < 1e-2:
            return np.zeros_like(gray_image, dtype=np.uint8)

        # Normalize to 0-255 uint8 range
        norm = cv2.normalize(max_response, None, 0, 255, cv2.NORM_MINMAX)
        return norm.astype(np.uint8)

    def detect_texture_anomaly_mask(self, gray_image: np.ndarray, threshold_std: float = 2.5) -> np.ndarray:
        """
        Segments texture anomalies where Gabor energy deviates significantly
        from the surface mean texture response.
        """
        energy = self.compute_energy_magnitude(gray_image).astype(np.float32)
        if np.max(energy) == 0 or np.ptp(energy) < 15:
            return np.zeros_like(gray_image, dtype=np.uint8)

        mean_val = np.mean(energy)
        std_val = np.std(energy)

        if std_val < 2.0:
            return np.zeros_like(gray_image, dtype=np.uint8)

        anomaly_threshold = mean_val + (threshold_std * std_val)
        mask = (energy > anomaly_threshold).astype(np.uint8) * 255

        # Clean noise via morphological opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
