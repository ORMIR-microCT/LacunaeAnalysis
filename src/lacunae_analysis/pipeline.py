"""Shared single-scan and batch pipeline entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .diagnostics import (
    build_threshold_figure,
    save_threshold_plot,
    write_batch_summary_csv,
    write_batch_summary_json,
    write_component_table_csv,
    write_scan_summary_csv,
    write_scan_summary_json,
)
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


def _threshold_settings_from_config(config: dict[str, Any]) -> ThresholdSettings:
    threshold_config = dict(config.get("thresholding", {}))
    method = str(threshold_config.get("method", "peak"))
    manual_threshold = float(threshold_config.get("manual_threshold", 0.0) or 0.0)
    use_otsu = bool(threshold_config.get("use_otsu", method == "otsu"))
    use_peak_threshold = bool(
        threshold_config.get("use_peak_threshold", threshold_config.get("method", "peak") == "peak")
    )
    if manual_threshold != 0.0:
        use_otsu = False
        use_peak_threshold = False
    return ThresholdSettings(
        manual_threshold=manual_threshold,
        use_otsu=use_otsu,
        use_peak_threshold=use_peak_threshold,
    )


def _density_filter_settings_from_config(config: dict[str, Any]) -> DensityFilterSettings:
    density_config = dict(config.get("density_filter", {}))
    return DensityFilterSettings(
        lower_volume_um3=float(density_config.get("lower_volume_um3", 200.0) or 0.0),
        upper_volume_um3=(
            float(density_config["upper_volume_um3"])
            if density_config.get("upper_volume_um3") is not None
            else None
        ),
    )


def _scan_metadata_from_config(config: dict[str, Any]) -> dict[str, Any]:
    scan_config = dict(config.get("scan", {}))
    metadata = dict(scan_config.get("metadata", {}))
    for key, value in scan_config.items():
        if key not in {"image_path", "output_dir", "metadata"}:
            metadata[key] = value
    return metadata


def _resolve_scan_output_dir(base_output_dir: Path, scan_config: dict[str, Any], default_name: str) -> Path:
    configured = scan_config.get("output_dir")
    if configured is None:
        return base_output_dir / default_name

    configured_path = Path(configured)
    if configured_path.is_absolute():
        return configured_path
    return base_output_dir / configured_path


def _resolve_batch_scan_path(input_dir: Path, image_path_value: str) -> Path:
    image_path = Path(image_path_value)
    if image_path.is_absolute():
        return image_path

    direct_candidate = input_dir / image_path
    if image_path.parent != Path(".") or direct_candidate.exists():
        return direct_candidate

    return input_dir / image_path.name


def summarize_batch_results(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a stable batch summary table from per-scan result rows."""
    table = pd.DataFrame(rows)
    preferred_columns = ["scan_name", "image_path", "status", "error"]
    present_preferred = [column for column in preferred_columns if column in table.columns]
    remaining_columns = [column for column in table.columns if column not in preferred_columns]
    ordered_columns = present_preferred + remaining_columns
    if not ordered_columns:
        return table
    return table.reindex(columns=ordered_columns)


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


def run_single_scan_job(
    input_path: str | Path,
    config: dict[str, Any] | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run one scan and persist deterministic diagnostics."""
    resolved_config = {} if config is None else dict(config)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = run_single_scan(
        ScanInput(
            image_path=Path(input_path),
            output_dir=output_path,
            metadata=_scan_metadata_from_config(resolved_config),
        ),
        threshold_settings=_threshold_settings_from_config(resolved_config),
        density_filter_settings=_density_filter_settings_from_config(resolved_config),
        return_images=True,
    )

    threshold_figure = build_threshold_figure(results["threshold_results"])
    save_threshold_plot(output_path / "thresholds.png", threshold_figure)
    write_scan_summary_csv(output_path / "scan_summary.csv", results)
    write_scan_summary_json(output_path / "scan_summary.json", results, resolved_config)
    write_component_table_csv(
        output_path / "component_table.csv",
        results["density_results"]["component_table"],
    )
    return results


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

    return summarize_batch_results(rows)


def run_batch_job(
    input_dir: str | Path,
    config: dict[str, Any] | None,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Run the single-scan job over a batch and persist aggregate summaries."""
    resolved_config = {} if config is None else dict(config)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    scan_entries = list(resolved_config.get("batch", {}).get("scans", []))

    if scan_entries:
        planned_scans = [
            (
                _resolve_batch_scan_path(input_path, str(entry["image_path"])),
                dict(entry),
            )
            for entry in scan_entries
        ]
    else:
        planned_scans = [(path, {}) for path in sorted(input_path.glob("*.aim"))]

    for image_path, scan_config in planned_scans:
        scan_output_dir = _resolve_scan_output_dir(output_path, scan_config, image_path.stem)
        scan_run_config = dict(resolved_config)
        scan_run_config["scan"] = {
            **dict(resolved_config.get("scan", {})),
            **{key: value for key, value in scan_config.items() if key != "image_path"},
        }

        base_row: dict[str, Any] = {
            "scan_name": image_path.stem,
            "image_path": str(image_path),
            **{
                key: value
                for key, value in scan_config.items()
                if key not in {"image_path", "output_dir"}
            },
        }
        try:
            results = run_single_scan_job(image_path, scan_run_config, scan_output_dir)
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

    table = summarize_batch_results(rows)
    write_batch_summary_csv(output_path / "batch_summary.csv", table)
    write_batch_summary_json(output_path / "batch_summary.json", table, resolved_config)
    return table
