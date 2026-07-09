"""Core data models for the lacunae analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ScanInput:
    image_path: Path
    output_dir: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ThresholdSettings:
    manual_threshold: float = 0.0
    use_otsu: bool = True
    use_peak_threshold: bool = True


@dataclass(slots=True)
class DensityFilterSettings:
    lower_volume_um3: float = 0.0
    upper_volume_um3: float | None = None


@dataclass(slots=True)
class BatchRun:
    scans: list[ScanInput] = field(default_factory=list)

