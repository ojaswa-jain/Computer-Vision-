"""
Unit tests for image enhancement and color space preprocessing.
"""

import pytest
import numpy as np
import cv2

from visioninspect.config import PreprocessingConfig
from visioninspect.preprocessing.enhancement import ImageEnhancer
from visioninspect.preprocessing.color_spaces import ColorSpaceProcessor


@pytest.fixture
def sample_bgr_image():
    """Returns a deterministic 100x100 BGR test image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = (120, 140, 160)
    # Add high contrast region
    img[30:50, 30:50] = (20, 30, 40)
    return img


@pytest.fixture
def sample_gray_image():
    """Returns a deterministic 100x100 grayscale test image."""
    img = np.full((100, 100), 128, dtype=np.uint8)
    img[20:40, 20:40] = 30
    return img


def test_gamma_correction_identity(sample_gray_image):
    enhancer = ImageEnhancer(PreprocessingConfig(gamma=1.0))
    result = enhancer.apply_gamma_correction(sample_gray_image)
    assert np.array_equal(result, sample_gray_image)


def test_gamma_correction_brightening(sample_gray_image):
    enhancer = ImageEnhancer(PreprocessingConfig(gamma=2.0))
    result = enhancer.apply_gamma_correction(sample_gray_image)
    # Gamma > 1 should brighten the image
    assert np.mean(result) > np.mean(sample_gray_image)


def test_clahe_enhancement(sample_gray_image):
    enhancer = ImageEnhancer()
    enhanced = enhancer.apply_clahe(sample_gray_image)
    assert enhanced.shape == sample_gray_image.shape
    assert enhanced.dtype == np.uint8
    # Standard deviation should increase due to local contrast expansion
    assert np.std(enhanced) >= np.std(sample_gray_image)


def test_bilateral_filtering(sample_bgr_image):
    enhancer = ImageEnhancer()
    filtered = enhancer.apply_bilateral_filter(sample_bgr_image)
    assert filtered.shape == sample_bgr_image.shape
    assert filtered.dtype == np.uint8


def test_color_space_decomposition(sample_bgr_image):
    channels = ColorSpaceProcessor.extract_color_channels(sample_bgr_image)
    expected_keys = {"gray", "lab_l", "lab_a", "lab_b", "hsv_h", "hsv_s", "hsv_v"}
    assert set(channels.keys()) == expected_keys
    for k, v in channels.items():
        assert v.shape == (100, 100)
        assert v.dtype == np.uint8


def test_chromatic_distance(sample_bgr_image):
    dist = ColorSpaceProcessor.compute_chromatic_distance(sample_bgr_image)
    assert dist.shape == (100, 100)
    assert dist.dtype == np.uint8
