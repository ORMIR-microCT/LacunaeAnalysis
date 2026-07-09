"""Connected-component metrics for lacunae density analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk

from .morphology import measure_lacuna_morphology


def _as_binary_sitk_image(image_like: sitk.Image | np.ndarray) -> sitk.Image:
    if isinstance(image_like, sitk.Image):
        return sitk.Cast(image_like > 0, sitk.sitkUInt8)

    return sitk.GetImageFromArray(np.asarray(image_like, dtype=np.uint8))


def get_voxel_volume(image: sitk.Image) -> float:
    """Return physical voxel volume in image spacing units cubed."""
    return float(np.prod(image.GetSpacing()))


def um3_to_image_volume(volume_um3: float, spacing_length_unit: str = "mm") -> float:
    """
    Convert a volume in um^3 to the image volume units implied by spacing.

    Examples
    --------
    spacing_length_unit='mm' -> returns mm^3
    spacing_length_unit='um' -> returns um^3
    """
    unit_to_mm = {
        "mm": 1.0,
        "um": 1e-3,
        "µm": 1e-3,
        "nm": 1e-6,
    }
    if spacing_length_unit not in unit_to_mm:
        raise ValueError(f"Unsupported spacing_length_unit: {spacing_length_unit}")

    mm_per_unit = unit_to_mm[spacing_length_unit]
    image_units_per_um = 1e-3 / mm_per_unit
    return float(volume_um3 * image_units_per_um**3)


def binary_array_to_sitk(binary_array: np.ndarray, reference_image: sitk.Image) -> sitk.Image:
    """Convert a binary NumPy array to SimpleITK, preserving metadata."""
    out = sitk.GetImageFromArray(np.asarray(binary_array, dtype=np.uint8))
    out.CopyInformation(reference_image)
    return out


def label_and_measure_components(binary_image: sitk.Image) -> tuple[sitk.Image, np.ndarray, pd.DataFrame]:
    """
    Label connected components and return:
    - labeled SimpleITK image
    - labeled NumPy array
    - component table with label, voxel_count, and physical volume
    """
    labeled_image = sitk.ConnectedComponent(binary_image > 0)
    labeled_array = sitk.GetArrayFromImage(labeled_image)

    counts = np.bincount(labeled_array.ravel())
    voxel_volume = get_voxel_volume(binary_image)

    if len(counts) <= 1:
        table = pd.DataFrame(columns=["label", "voxel_count", "volume"])
        return labeled_image, labeled_array, table

    labels = np.arange(1, len(counts))
    voxel_counts = counts[1:]
    volumes = voxel_counts * voxel_volume

    table = pd.DataFrame(
        {
            "label": labels.astype(int),
            "voxel_count": voxel_counts.astype(int),
            "volume": volumes.astype(float),
        }
    )

    return labeled_image, labeled_array, table


def filter_component_table(
    component_table: pd.DataFrame,
    lower_volume: float,
    upper_volume: float | None,
    voxel_volume: float,
) -> tuple[pd.DataFrame, int, int | None]:
    """
    Filter connected components by physical volume.
    Returns filtered table plus equivalent voxel thresholds.
    """
    lower_voxel_threshold = int(np.ceil(lower_volume / voxel_volume))
    upper_voxel_threshold = (
        int(np.floor(upper_volume / voxel_volume)) if upper_volume is not None else None
    )

    if component_table.empty:
        return component_table.copy(), lower_voxel_threshold, upper_voxel_threshold

    keep_mask = component_table["volume"] >= lower_volume
    if upper_volume is not None:
        keep_mask &= component_table["volume"] <= upper_volume

    filtered = component_table[keep_mask].copy()
    filtered = filtered.sort_values("voxel_count", ascending=False).reset_index(drop=True)
    return filtered, lower_voxel_threshold, upper_voxel_threshold


def build_filtered_binary(labeled_array: np.ndarray, kept_labels: np.ndarray | list[int]) -> np.ndarray:
    """Build a binary image containing only the selected connected-component labels."""
    if len(kept_labels) == 0:
        return np.zeros_like(labeled_array, dtype=np.uint8)

    keep = np.zeros(int(labeled_array.max()) + 1, dtype=bool)
    keep[np.asarray(kept_labels, dtype=int)] = True
    return keep[labeled_array].astype(np.uint8)


def summarize_lacuna_density(
    filtered_binary_image: sitk.Image,
    bone_mask_image: sitk.Image,
    component_table: pd.DataFrame,
) -> dict[str, float | int]:
    """
    Summarize lacuna density statistics inside the bone mask.
    """
    voxel_volume = get_voxel_volume(filtered_binary_image)

    bone_voxels = int(sitk.GetArrayViewFromImage(bone_mask_image).sum())
    lacuna_voxels = int(sitk.GetArrayViewFromImage(filtered_binary_image).sum())

    bone_volume = bone_voxels * voxel_volume
    lacuna_volume = lacuna_voxels * voxel_volume
    n_lacunae = int(len(component_table))

    summary = {
        "n_lacunae": n_lacunae,
        "bone_voxels": bone_voxels,
        "lacuna_voxels": lacuna_voxels,
        "voxel_volume": float(voxel_volume),
        "bone_volume": float(bone_volume),
        "lacuna_volume": float(lacuna_volume),
        "mean_lacuna_volume": float(component_table["volume"].mean()) if n_lacunae > 0 else 0.0,
        "median_lacuna_volume": float(component_table["volume"].median()) if n_lacunae > 0 else 0.0,
        "lacuna_density_per_volume": float(n_lacunae / bone_volume) if bone_volume > 0 else 0.0,
        "lacuna_volume_fraction": float(lacuna_volume / bone_volume) if bone_volume > 0 else 0.0,
    }
    if "surface_area_um2" in component_table.columns:
        summary["mean_lacuna_surface_area_um2"] = (
            float(component_table["surface_area_um2"].mean()) if n_lacunae > 0 else 0.0
        )
        summary["median_lacuna_surface_area_um2"] = (
            float(component_table["surface_area_um2"].median()) if n_lacunae > 0 else 0.0
        )
    if "lacuna_stretch" in component_table.columns:
        summary["mean_lacuna_stretch"] = (
            float(component_table["lacuna_stretch"].mean()) if n_lacunae > 0 else 0.0
        )
        summary["median_lacuna_stretch"] = (
            float(component_table["lacuna_stretch"].median()) if n_lacunae > 0 else 0.0
        )
    if "lacuna_oblateness" in component_table.columns:
        summary["mean_lacuna_oblateness"] = (
            float(component_table["lacuna_oblateness"].mean()) if n_lacunae > 0 else 0.0
        )
        summary["median_lacuna_oblateness"] = (
            float(component_table["lacuna_oblateness"].median()) if n_lacunae > 0 else 0.0
        )
    return summary


def analyze_lacuna_density(
    lacuna_binary_input: sitk.Image | np.ndarray,
    bone_mask_input: sitk.Image | np.ndarray,
    lower_volume_um3: float = 200.0,
    upper_volume_um3: float | None = 1500.0,
    spacing_length_unit: str = "mm",
    return_images: bool = True,
) -> dict[str, Any]:
    """
    Filter connected lacunae by size and compute density statistics.
    """
    lacuna_sitk = _as_binary_sitk_image(lacuna_binary_input)
    bone_sitk = _as_binary_sitk_image(bone_mask_input)

    voxel_volume = get_voxel_volume(lacuna_sitk)

    lower_volume = um3_to_image_volume(lower_volume_um3, spacing_length_unit)
    upper_volume = (
        um3_to_image_volume(upper_volume_um3, spacing_length_unit)
        if upper_volume_um3 is not None
        else None
    )

    labeled_image, labeled_array, component_table = label_and_measure_components(lacuna_sitk)

    filtered_table, lower_voxel_threshold, upper_voxel_threshold = filter_component_table(
        component_table,
        lower_volume=lower_volume,
        upper_volume=upper_volume,
        voxel_volume=voxel_volume,
    )

    filtered_binary_array = build_filtered_binary(
        labeled_array,
        filtered_table["label"].to_numpy() if not filtered_table.empty else [],
    )
    filtered_table = measure_lacuna_morphology(
        labeled_array,
        filtered_table,
        spacing=lacuna_sitk.GetSpacing(),
        spacing_length_unit=spacing_length_unit,
    )

    filtered_binary_image = binary_array_to_sitk(filtered_binary_array, lacuna_sitk)

    summary = summarize_lacuna_density(
        filtered_binary_image=filtered_binary_image,
        bone_mask_image=bone_sitk,
        component_table=filtered_table,
    )

    if spacing_length_unit == "mm":
        summary["lacuna_density_per_mm3"] = summary["lacuna_density_per_volume"]
    else:
        unit_to_mm = {"mm": 1.0, "um": 1e-3, "µm": 1e-3, "nm": 1e-6}
        mm_per_unit = unit_to_mm[spacing_length_unit]
        summary["lacuna_density_per_mm3"] = summary["lacuna_density_per_volume"] / (mm_per_unit**3)

    results: dict[str, Any] = {
        "summary": summary,
        "component_table": filtered_table,
        "lower_volume_um3": float(lower_volume_um3),
        "upper_volume_um3": float(upper_volume_um3) if upper_volume_um3 is not None else None,
        "lower_volume_image_units": float(lower_volume),
        "upper_volume_image_units": float(upper_volume) if upper_volume is not None else None,
        "lower_voxel_threshold": int(lower_voxel_threshold),
        "upper_voxel_threshold": int(upper_voxel_threshold) if upper_voxel_threshold is not None else None,
        "spacing_length_unit": spacing_length_unit,
        "voxel_volume_image_units": float(voxel_volume),
    }

    if return_images:
        results["labeled_lacunae_sitk"] = labeled_image
        results["filtered_lacuna_binary_sitk"] = filtered_binary_image
        results["filtered_lacuna_binary_array"] = filtered_binary_array

    return results
