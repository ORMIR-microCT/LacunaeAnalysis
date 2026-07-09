"""Input and output helpers for scan loading."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from .models import LoadedScan


def _import_aimio() -> Any:
    errors: list[Exception] = []
    for module_name in ("aimio", "aimio_py"):
        try:
            return import_module(module_name)
        except ModuleNotFoundError as error:
            errors.append(error)

    raise ModuleNotFoundError(
        "Unable to import aimio-py. Install the package in the active environment "
        "to enable AIMS read/write support."
    ) from errors[-1]


def _extract_metadata(raw_scan: Any) -> dict[str, Any]:
    metadata = getattr(raw_scan, "meta", None)
    if metadata is None:
        metadata = getattr(raw_scan, "metadata", None)
    if metadata is None:
        return {}
    return dict(metadata)


def _extract_triplet(raw_scan: Any, attribute: str) -> tuple[float, float, float]:
    values = getattr(raw_scan, attribute, None)
    if values is None:
        raise ValueError(f"AIMS scan is missing required `{attribute}` geometry.")
    triplet = tuple(float(value) for value in values)
    if len(triplet) != 3:
        raise ValueError(f"AIMS scan `{attribute}` must contain exactly three values.")
    return triplet


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

    aimio = _import_aimio()
    raw_scan = aimio.read(source_path)
    metadata = _extract_metadata(raw_scan)
    slope = metadata.get("density_slope")
    intercept = metadata.get("density_intercept")
    density_data = convert_to_density(
        np.asarray(raw_scan.data),
        slope=slope,
        intercept=intercept,
    )

    return LoadedScan(
        source_path=source_path,
        voxel_data=density_data,
        spacing=_extract_triplet(raw_scan, "spacing"),
        origin=_extract_triplet(raw_scan, "origin"),
        units="density",
        density_slope=float(slope),
        density_intercept=float(intercept),
        metadata=metadata,
    )


def write_binary_mask(path: str | Path, scan: LoadedScan, mask: np.ndarray) -> None:
    output_path = Path(path)
    if output_path.suffix.lower() != ".aim":
        raise ValueError(f"Unsupported output extension: {output_path.suffix}")

    if mask.shape != scan.voxel_data.shape:
        raise ValueError("Binary mask shape must match the loaded scan voxel grid.")

    aimio = _import_aimio()
    mask_metadata = dict(scan.metadata)
    mask_metadata.update(
        {
            "density_slope": scan.density_slope,
            "density_intercept": scan.density_intercept,
            "units": "binary_mask",
        }
    )
    binary_mask = np.asarray(mask, dtype=np.uint8)
    aimio.write(
        output_path,
        binary_mask,
        spacing=scan.spacing,
        origin=scan.origin,
        meta=mask_metadata,
    )
