"""
Configuration schemas, enumeration types, and threshold defaults for VisionInspect-AI.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Tuple


class DefectSeverity(str, Enum):
    PASS = "PASS"
    MINOR = "MINOR"
    MAJOR = "MAJOR"
    CRITICAL = "CRITICAL"


class DefectClass(str, Enum):
    CRACK = "Crack"
    SCRATCH = "Scratch"
    PIT = "Pit/Void"
    BLEMISH = "Discoloration/Blemish"
    PARTICLE = "Foreign Particle"
    NORMAL = "Normal"


@dataclass
class PreprocessingConfig:
    """Parameters for illumination correction and filtering."""
    clahe_clip_limit: float = 2.5
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    bilateral_d: int = 7
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0
    gamma: float = 1.0


@dataclass
class MorphologicalConfig:
    """Parameters for dual Top-Hat / Black-Hat morphological operations."""
    tophat_kernel_size: int = 15
    blackhat_kernel_size: int = 15
    kernel_shape: str = "ellipse"  # "ellipse", "rect", "cross"
    min_defect_area_px: int = 25
    max_defect_area_px: int = 250000
    morph_open_size: int = 3
    morph_close_size: int = 5


@dataclass
class GaborConfig:
    """Parameters for 2D Gabor wavelet filter bank."""
    enabled: bool = True
    ksize: int = 21
    sigma: float = 4.0
    theta_angles: Tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)  # degrees
    lambd: float = 10.0
    gamma: float = 0.5
    psi: float = 0.0


@dataclass
class MetrologyConfig:
    """Dimensional calibration and tolerance grading thresholds."""
    pixel_to_mm_ratio: float = 0.05  # 1 pixel = 0.05 mm (default 20 px = 1 mm)
    
    # Severity thresholds in millimeters
    minor_area_mm2_threshold: float = 0.15
    major_area_mm2_threshold: float = 1.00
    critical_area_mm2_threshold: float = 4.00
    
    critical_length_mm_threshold: float = 5.0
    critical_defect_classes: Tuple[str, ...] = (DefectClass.CRACK.value,)


@dataclass
class InspectionConfig:
    """Master configuration aggregate for the inspection pipeline."""
    profile_name: str = "default_metal"
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    morphology: MorphologicalConfig = field(default_factory=MorphologicalConfig)
    gabor: GaborConfig = field(default_factory=GaborConfig)
    metrology: MetrologyConfig = field(default_factory=MetrologyConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InspectionConfig":
        """Instantiate config from dictionary or JSON."""
        prep_data = data.get("preprocessing", {})
        morph_data = data.get("morphology", {})
        gabor_data = data.get("gabor", {})
        metr_data = data.get("metrology", {})

        return cls(
            profile_name=data.get("profile_name", "custom"),
            preprocessing=PreprocessingConfig(**prep_data),
            morphology=MorphologicalConfig(**morph_data),
            gabor=GaborConfig(**gabor_data),
            metrology=MetrologyConfig(**metr_data),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "profile_name": self.profile_name,
            "preprocessing": {
                "clahe_clip_limit": self.preprocessing.clahe_clip_limit,
                "clahe_tile_grid_size": list(self.preprocessing.clahe_tile_grid_size),
                "bilateral_d": self.preprocessing.bilateral_d,
                "bilateral_sigma_color": self.preprocessing.bilateral_sigma_color,
                "bilateral_sigma_space": self.preprocessing.bilateral_sigma_space,
                "gamma": self.preprocessing.gamma,
            },
            "morphology": {
                "tophat_kernel_size": self.morphology.tophat_kernel_size,
                "blackhat_kernel_size": self.morphology.blackhat_kernel_size,
                "kernel_shape": self.morphology.kernel_shape,
                "min_defect_area_px": self.morphology.min_defect_area_px,
                "max_defect_area_px": self.morphology.max_defect_area_px,
                "morph_open_size": self.morphology.morph_open_size,
                "morph_close_size": self.morphology.morph_close_size,
            },
            "gabor": {
                "enabled": self.gabor.enabled,
                "ksize": self.gabor.ksize,
                "sigma": self.gabor.sigma,
                "theta_angles": list(self.gabor.theta_angles),
                "lambd": self.gabor.lambd,
                "gamma": self.gabor.gamma,
                "psi": self.gabor.psi,
            },
            "metrology": {
                "pixel_to_mm_ratio": self.metrology.pixel_to_mm_ratio,
                "minor_area_mm2_threshold": self.metrology.minor_area_mm2_threshold,
                "major_area_mm2_threshold": self.metrology.major_area_mm2_threshold,
                "critical_area_mm2_threshold": self.metrology.critical_area_mm2_threshold,
                "critical_length_mm_threshold": self.metrology.critical_length_mm_threshold,
                "critical_defect_classes": list(self.metrology.critical_defect_classes),
            },
        }
