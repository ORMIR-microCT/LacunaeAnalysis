"""Input and output helpers for scan loading."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from .models import LoadedScan


def _import_py_aimio() -> Any:
    try:
        return import_module("py_aimio")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Unable to import py_aimio from aimio-py. Install aimio-py in the "
            "active environment to enable AIMS read/write support."
        ) from error


def _extract_triplet(
    metadata: dict[str, Any],
    keys: tuple[str, ...],
    *,
    default: tuple[float, float, float] | None = None,
    label: str,
) -> tuple[float, float, float]:
    values = None
    matched_key = None
    for key in keys:
        if key in metadata:
            values = metadata[key]
            matched_key = key
            break

    if values is None:
        if default is not None:
            return default
        keys_text = "`, `".join(keys)
        raise ValueError(f"AIMS metadata is missing required `{label}` geometry; checked `{keys_text}`.")

    triplet = tuple(float(value) for value in values)
    if len(triplet) != 3:
        raise ValueError(f"AIMS metadata `{matched_key}` must contain exactly three values.")
    return triplet


def _density_equation(py_aimio: Any, metadata: dict[str, Any]) -> tuple[float, float]:
    processing_log = metadata.get("processing_log_raw", metadata.get("processing_log", ""))
    if isinstance(processing_log, dict):
        processing_log = py_aimio.dict_to_log(processing_log)

    try:
        slope, intercept = py_aimio.get_aim_density_equation(processing_log)
    except Exception:
        slope = metadata.get("density_slope", 1.0)
        intercept = metadata.get("density_intercept", 0.0)

    return float(slope), float(intercept)


def convert_to_density(
    voxel_data: np.ndarray,
    slope: float | None,
    intercept: float | None,
) -> np.ndarray:
    if slope is None or intercept is None:
        raise ValueError("Missing density calibration required for density conversion.")

    return voxel_data.astype(np.float64, copy=False) * float(slope) + float(intercept)


def load_aim_as_density(path: str | Path) -> LoadedScan:
    source_path = Path(path)
    if source_path.suffix.lower() != ".aim":
        raise ValueError(f"Unsupported input extension: {source_path.suffix}")

    py_aimio = _import_py_aimio()
    density_data, metadata = py_aimio.read_aim(str(source_path), density=True)
    metadata = dict(metadata)
    slope, intercept = _density_equation(py_aimio, metadata)

    spacing = _extract_triplet(
        metadata,
        ("spacing", "voxel_size", "voxel_size_mm", "element_size", "element_size_mm"),
        label="spacing",
    )
    origin = _extract_triplet(
        metadata,
        ("origin", "position", "offset"),
        default=(0.0, 0.0, 0.0),
        label="origin",
    )

    return LoadedScan(
        source_path=source_path,
        voxel_data=np.asarray(density_data, dtype=np.float64),
        spacing=spacing,
        origin=origin,
        units=str(metadata.get("unit", "BMD")),
        density_slope=slope,
        density_intercept=intercept,
        metadata=metadata,
    )


def write_binary_mask(path: str | Path, scan: LoadedScan, mask: np.ndarray) -> None:
    output_path = Path(path)
    if output_path.suffix.lower() != ".aim":
        raise ValueError(f"Unsupported output extension: {output_path.suffix}")

    if mask.shape != scan.voxel_data.shape:
        raise ValueError("Binary mask shape must match the loaded scan voxel grid.")

    py_aimio = _import_py_aimio()
    mask_metadata = dict(scan.metadata)
    mask_metadata.update(
        {
            "density_slope": scan.density_slope,
            "density_intercept": scan.density_intercept,
            "units": "binary_mask",
        }
    )
    binary_mask = np.asarray(mask, dtype=np.uint8)
    py_aimio.write_aim(
        str(output_path),
        binary_mask,
        meta=mask_metadata,
        unit="native",
    )
