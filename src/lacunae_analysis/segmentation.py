"""Segmentation utilities for lacunae analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from .models import LoadedScan


def _scan_to_sitk(scan: LoadedScan) -> sitk.Image:
    image = sitk.GetImageFromArray(np.asarray(scan.voxel_data, dtype=np.float32))
    image.SetSpacing(tuple(float(value) for value in scan.spacing))
    image.SetOrigin(tuple(float(value) for value in scan.origin))
    return image


def voxel_sigma_to_physical_sigma(image: sitk.Image, sigma_vox: float) -> float:
    """Convert Gaussian sigma from voxel units to physical units for SimpleITK."""
    spacing = np.array(image.GetSpacing(), dtype=float)
    return float(float(sigma_vox) * np.mean(spacing))


def create_bone_mask(
    image: sitk.Image,
    threshold: float,
    large_gaussian_kernel: float = 10.0,
    max_intensity: float | None = None,
) -> sitk.Image:
    """Create binary bone mask by blurring and thresholding."""
    if max_intensity is None:
        max_intensity = float(sitk.GetArrayViewFromImage(image).max())

    sigma_physical = voxel_sigma_to_physical_sigma(image, large_gaussian_kernel)
    blurred = sitk.SmoothingRecursiveGaussian(image, sigma_physical)
    return sitk.BinaryThreshold(
        blurred,
        lowerThreshold=float(threshold),
        upperThreshold=float(max_intensity),
        insideValue=1,
        outsideValue=0,
    )


def isolate_lacuna_binary(
    image: sitk.Image,
    bone_mask: sitk.Image,
    threshold: float,
    small_gaussian_kernel: float = 1.2,
    max_intensity: float | None = None,
) -> sitk.Image:
    """Create binary image of isolated lacunae."""
    if max_intensity is None:
        max_intensity = float(sitk.GetArrayViewFromImage(image).max())

    sigma_physical = voxel_sigma_to_physical_sigma(image, small_gaussian_kernel)
    inverted = sitk.InvertIntensity(image, maximum=float(max_intensity))
    smoothed = sitk.SmoothingRecursiveGaussian(inverted, sigma_physical)
    masked = sitk.Mask(smoothed, bone_mask)

    return sitk.BinaryThreshold(
        masked,
        lowerThreshold=float(max_intensity - float(threshold)),
        upperThreshold=float(max_intensity),
        insideValue=1,
        outsideValue=0,
    )


def calculate_volume(binary_mask: sitk.Image) -> float:
    """Calculate physical volume of a binary mask."""
    voxel_volume = float(np.prod(binary_mask.GetSpacing()))
    num_voxels = int(sitk.GetArrayViewFromImage(binary_mask).sum())
    return voxel_volume * num_voxels


def segment_lacunae(
    scan: LoadedScan,
    threshold: float,
    bone_sigma: float,
    lacuna_sigma: float,
) -> dict[str, Any]:
    """Run notebook-aligned lacuna segmentation on a loaded scan."""
    image_array = np.asarray(scan.voxel_data, dtype=float)
    sitk_img = _scan_to_sitk(scan)
    max_intensity = float(image_array.max())
    threshold = float(threshold)

    bone_mask = create_bone_mask(
        sitk_img,
        threshold=threshold,
        large_gaussian_kernel=bone_sigma,
        max_intensity=max_intensity,
    )
    lacuna_binary = isolate_lacuna_binary(
        sitk_img,
        bone_mask=bone_mask,
        threshold=threshold,
        small_gaussian_kernel=lacuna_sigma,
        max_intensity=max_intensity,
    )

    bone_voxels = int(sitk.GetArrayViewFromImage(bone_mask).sum())
    lacuna_voxels = int(sitk.GetArrayViewFromImage(lacuna_binary).sum())

    return {
        "input_sitk": sitk_img,
        "input_array": image_array,
        "bone_mask_sitk": bone_mask,
        "lacuna_binary_sitk": lacuna_binary,
        "bone_mask_array": sitk.GetArrayFromImage(bone_mask),
        "lacuna_binary_array": sitk.GetArrayFromImage(lacuna_binary),
        "threshold": threshold,
        "max_intensity": max_intensity,
        "inverted_threshold": float(max_intensity - threshold),
        "bone_voxels": bone_voxels,
        "lacuna_voxels": lacuna_voxels,
        "bone_volume": calculate_volume(bone_mask),
        "lacuna_volume": calculate_volume(lacuna_binary),
        "spacing": sitk_img.GetSpacing(),
    }


def plot_lacuna_segmentation(
    results: dict[str, Any],
    density_results: dict[str, Any] | None = None,
    slice_index: int | None = None,
    pmin: float = 1,
    pmax: float = 99,
    output_path: str | Path | None = None,
    show: bool = True,
) -> Any:
    """Diagnostic visualization of segmentation and volume-filtered lacunae."""
    import matplotlib.pyplot as plt

    image = results["input_array"]
    bone = results.get("bone_mask_array")
    lacuna = results.get("lacuna_binary_array")
    filtered_lacuna = None

    if bone is None:
        bone = sitk.GetArrayFromImage(results["bone_mask_sitk"])
    if lacuna is None:
        lacuna = sitk.GetArrayFromImage(results["lacuna_binary_sitk"])
    if density_results is not None:
        filtered_lacuna = density_results.get("filtered_lacuna_binary_array")
        if filtered_lacuna is None and "filtered_lacuna_binary_sitk" in density_results:
            filtered_lacuna = sitk.GetArrayFromImage(density_results["filtered_lacuna_binary_sitk"])

    if slice_index is None:
        slice_index = int(image.shape[0] // 2)

    vmin, vmax = np.percentile(image, [pmin, pmax])
    figure, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(image[slice_index], cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Original (z={slice_index})")

    axes[1].imshow(bone[slice_index], cmap="gray")
    axes[1].set_title("Bone mask")

    axes[2].imshow(lacuna[slice_index], cmap="gray")
    axes[2].set_title("Lacuna binary")

    final_lacuna = filtered_lacuna if filtered_lacuna is not None else lacuna
    axes[3].imshow(final_lacuna[slice_index], cmap="gray")
    axes[3].set_title("Volume-filtered lacunae" if filtered_lacuna is not None else "Lacuna binary")

    for axis in axes:
        axis.axis("off")

    figure.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return figure
