"""Core data models for the lacunae analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


DEFAULT_LOWER_VOLUME_UM3 = 100.0
DEFAULT_UPPER_VOLUME_UM3 = 2000.0
DEFAULT_INCLUDE_EDGE_LACUNAE = False
DEFAULT_EDGE_WIDTH = 2
SUPPORTED_INTENSITY_UNITS = ("raw", "bmd", "hu", "mu")
DEFAULT_AIM_INTENSITY_UNIT = "bmd"
IntensityUnit = Literal["raw", "bmd", "hu", "mu"]


@dataclass(slots=True)
class ScanInput:
    image_path: Path
    output_dir: Path | None = None
    intensity_unit: IntensityUnit = DEFAULT_AIM_INTENSITY_UNIT
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ThresholdSettings:
    manual_threshold: float = 0.0
    use_otsu: bool = True
    use_peak_threshold: bool = True


@dataclass(slots=True)
class DensityFilterSettings:
    lower_volume_um3: float = DEFAULT_LOWER_VOLUME_UM3
    upper_volume_um3: float | None = DEFAULT_UPPER_VOLUME_UM3
    include_edge_lacunae: bool = DEFAULT_INCLUDE_EDGE_LACUNAE
    edge_width: int = DEFAULT_EDGE_WIDTH


@dataclass(slots=True)
class BatchRun:
    scans: list[ScanInput] = field(default_factory=list)


@dataclass(slots=True)
class LoadedScan:
    source_path: Path
    voxel_data: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    units: str
    density_slope: float
    density_intercept: float
    metadata: dict[str, Any] = field(default_factory=dict)
