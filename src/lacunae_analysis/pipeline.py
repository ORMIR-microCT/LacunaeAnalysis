"""Shared single-scan and batch pipeline entry points."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .io import load_aim_as_density
from .metrics import analyze_lacuna_density
from .models import BatchRun, DensityFilterSettings, ScanInput, ThresholdSettings
from .segmentation import segment_lacunae
from .thresholding import compute_threshold


def _resolve_threshold_settings(settings: ThresholdSettings | None) -> ThresholdSettings:
    return settings if settings is not None else ThresholdSettings()


def _resolve_density_filter_settings(
    settings: DensityFilterSettings | None,
) -> DensityFilterSettings:
    if settings is not None:
        return settings

    return DensityFilterSettings(lower_volume_um3=200.0, upper_volume_um3=1500.0)


def run_single_scan(
    scan_input: ScanInput,
    threshold_settings: ThresholdSettings | None = None,
    density_filter_settings: DensityFilterSettings | None = None,
    bone_sigma: float = 10.0,
    lacuna_sigma: float = 1.2,
    spacing_length_unit: str = "mm",
    return_images: bool = True,
) -> dict[str, Any]:
    """Run the density-space lacunae workflow for one scan."""
    threshold_settings = _resolve_threshold_settings(threshold_settings)
    density_filter_settings = _resolve_density_filter_settings(density_filter_settings)

    loaded_scan = load_aim_as_density(scan_input.image_path)
    threshold_results = compute_threshold(
        loaded_scan,
        manual_threshold=threshold_settings.manual_threshold,
    )
    segmentation_results = segment_lacunae(
        loaded_scan,
        threshold=float(threshold_results["selected_threshold"]),
        bone_sigma=bone_sigma,
        lacuna_sigma=lacuna_sigma,
    )
    density_results = analyze_lacuna_density(
        segmentation_results["lacuna_binary_sitk"],
        segmentation_results["bone_mask_sitk"],
        lower_volume_um3=density_filter_settings.lower_volume_um3,
        upper_volume_um3=density_filter_settings.upper_volume_um3,
        spacing_length_unit=spacing_length_unit,
        return_images=return_images,
    )

    return {
        "scan_input": scan_input,
        "loaded_scan": loaded_scan,
        "threshold_results": threshold_results,
        "segmentation_results": segmentation_results,
        "density_results": density_results,
        "summary": density_results["summary"],
    }


def run_batch(
    batch_run: BatchRun,
    threshold_settings: ThresholdSettings | None = None,
    density_filter_settings: DensityFilterSettings | None = None,
    bone_sigma: float = 10.0,
    lacuna_sigma: float = 1.2,
    spacing_length_unit: str = "mm",
    return_images: bool = False,
) -> pd.DataFrame:
    """Run the single-scan pipeline over a batch of inputs."""
    rows: list[dict[str, Any]] = []

    for scan_input in batch_run.scans:
        base_row: dict[str, Any] = {
            "scan_name": scan_input.image_path.stem,
            "image_path": str(scan_input.image_path),
            **scan_input.metadata,
        }
        try:
            results = run_single_scan(
                scan_input,
                threshold_settings=threshold_settings,
                density_filter_settings=density_filter_settings,
                bone_sigma=bone_sigma,
                lacuna_sigma=lacuna_sigma,
                spacing_length_unit=spacing_length_unit,
                return_images=return_images,
            )
        except Exception as error:
            rows.append(
                {
                    **base_row,
                    "status": "error",
                    "error": str(error),
                }
            )
            continue

        rows.append(
            {
                **base_row,
                "status": "ok",
                **results["summary"],
                **results["threshold_results"],
            }
        )

    return pd.DataFrame(rows)
