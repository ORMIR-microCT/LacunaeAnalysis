"""Deterministic diagnostics writers for lacunae analysis runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def _ensure_parent(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_threshold_figure(threshold_results: Mapping[str, Any]):
    import matplotlib.pyplot as plt

    labels = ["otsu_threshold", "peak_threshold", "manual_threshold", "selected_threshold"]
    values = [float(threshold_results.get(label, 0.0) or 0.0) for label in labels]
    display_labels = ["Otsu", "Peak", "Manual", "Selected"]

    figure, axis = plt.subplots(figsize=(8, 4))
    bars = axis.bar(display_labels, values, color=["#d62728", "#9467bd", "#2ca02c", "#1f77b4"])
    axis.set_title("Threshold Summary")
    axis.set_ylabel("Threshold")

    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + (bar.get_width() / 2.0),
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()
    return figure


def save_threshold_plot(output_path: str | Path, figure) -> None:
    import matplotlib.pyplot as plt

    output_path = _ensure_parent(output_path)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def build_scan_summary_row(results: Mapping[str, Any]) -> dict[str, Any]:
    scan_input = results["scan_input"]
    segmentation_results = results["segmentation_results"]
    density_results = results["density_results"]

    return {
        "scan_name": scan_input.image_path.stem,
        "image_path": str(scan_input.image_path),
        **results["summary"],
        **results["threshold_results"],
        "bone_voxels": segmentation_results["bone_voxels"],
        "lacuna_voxels": segmentation_results["lacuna_voxels"],
        "bone_volume": segmentation_results["bone_volume"],
        "lacuna_volume": segmentation_results["lacuna_volume"],
        "lower_volume_um3": density_results["lower_volume_um3"],
        "upper_volume_um3": density_results["upper_volume_um3"],
        "lower_voxel_threshold": density_results["lower_voxel_threshold"],
        "upper_voxel_threshold": density_results["upper_voxel_threshold"],
    }


def write_scan_summary_csv(output_path: str | Path, results: Mapping[str, Any]) -> None:
    pd.DataFrame([build_scan_summary_row(results)]).to_csv(_ensure_parent(output_path), index=False)


def write_scan_summary_json(
    output_path: str | Path,
    results: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    scan_summary = {
        "scan_name": results["scan_input"].image_path.stem,
        "image_path": str(results["scan_input"].image_path),
        "summary": dict(results["summary"]),
        "threshold_results": dict(results["threshold_results"]),
        "segmentation_summary": {
            "bone_voxels": results["segmentation_results"]["bone_voxels"],
            "lacuna_voxels": results["segmentation_results"]["lacuna_voxels"],
            "bone_volume": results["segmentation_results"]["bone_volume"],
            "lacuna_volume": results["segmentation_results"]["lacuna_volume"],
        },
        "density_settings": {
            "lower_volume_um3": results["density_results"]["lower_volume_um3"],
            "upper_volume_um3": results["density_results"]["upper_volume_um3"],
            "lower_voxel_threshold": results["density_results"]["lower_voxel_threshold"],
            "upper_voxel_threshold": results["density_results"]["upper_voxel_threshold"],
        },
        "config": dict(config),
    }
    _ensure_parent(output_path).write_text(json.dumps(scan_summary, indent=2), encoding="utf-8")


def write_component_table_csv(output_path: str | Path, component_table: pd.DataFrame) -> None:
    component_table.to_csv(_ensure_parent(output_path), index=False)


def write_batch_summary_csv(output_path: str | Path, table: pd.DataFrame) -> None:
    table.to_csv(_ensure_parent(output_path), index=False)


def write_batch_summary_json(
    output_path: str | Path,
    table: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    payload = {
        "config": dict(config),
        "rows": table.to_dict(orient="records"),
    }
    _ensure_parent(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
