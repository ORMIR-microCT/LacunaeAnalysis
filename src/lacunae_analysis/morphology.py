"""Morphological measurements for filtered lacuna components."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk


def _spacing_to_um(spacing: tuple[float, float, float], spacing_length_unit: str) -> np.ndarray:
    unit_to_um = {
        "mm": 1000.0,
        "um": 1.0,
        "µm": 1.0,
        "nm": 0.001,
    }
    if spacing_length_unit not in unit_to_um:
        raise ValueError(f"Unsupported spacing_length_unit: {spacing_length_unit}")
    spacing_xyz = np.asarray(spacing, dtype=float) * unit_to_um[spacing_length_unit]
    return np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], dtype=float)


def _spacing_xyz_to_um(spacing: tuple[float, float, float], spacing_length_unit: str) -> np.ndarray:
    unit_to_um = {
        "mm": 1000.0,
        "um": 1.0,
        "µm": 1.0,
        "nm": 0.001,
    }
    if spacing_length_unit not in unit_to_um:
        raise ValueError(f"Unsupported spacing_length_unit: {spacing_length_unit}")
    return np.asarray(spacing, dtype=float) * unit_to_um[spacing_length_unit]


def _is_border_component(coords_zyx: np.ndarray, shape: tuple[int, int, int], edge_width: int) -> int:
    lower_edge = np.any(coords_zyx < edge_width)
    upper_bounds = np.asarray(shape, dtype=int) - int(edge_width)
    upper_edge = np.any(coords_zyx >= upper_bounds)
    return int(bool(lower_edge or upper_edge))


def _component_surface_area_um2(component_mask: np.ndarray, spacing_xyz_um: np.ndarray) -> float:
    if int(component_mask.sum()) == 0:
        return 0.0

    image = sitk.GetImageFromArray(component_mask.astype(np.uint8))
    image.SetSpacing(tuple(float(value) for value in spacing_xyz_um))
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.ComputePerimeterOn()
    stats.Execute(image)
    if not stats.HasLabel(1):
        return 0.0
    return float(stats.GetPerimeter(1))


def _principal_axes(coords_zyx: np.ndarray, spacing_zyx_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(coords_zyx) < 2:
        return np.zeros(3, dtype=float), np.eye(3, dtype=float)

    physical_zyx = coords_zyx.astype(float) * spacing_zyx_um
    centered = physical_zyx - physical_zyx.mean(axis=0)
    covariance = np.cov(centered.T)
    if not np.all(np.isfinite(covariance)):
        return np.zeros(3, dtype=float), np.eye(3, dtype=float)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    return eigenvalues, eigenvectors


def _shape_from_eigenvalues(eigenvalues: np.ndarray) -> dict[str, float]:
    axis_radii = np.sqrt(np.maximum(eigenvalues, 0.0))
    major = float(axis_radii[0])
    intermediate = float(axis_radii[1])
    minor = float(axis_radii[2])

    if major == 0.0:
        stretch = 0.0
        axis_ratio = 0.0
    else:
        stretch = (major - minor) / major
        axis_ratio = minor / major

    denominator = major - minor
    if denominator == 0.0:
        oblateness = 0.0
    else:
        oblateness = 2.0 * ((intermediate - minor) / denominator) - 1.0

    return {
        "lacuna_stretch": float(stretch),
        "lacuna_oblateness": float(oblateness),
        "major_axis_radius_um": major,
        "minor_axis_radius_um": minor,
        "minor_to_major_axis_ratio": float(axis_ratio),
    }


def _orientation_xyz(eigenvectors_zyx: np.ndarray, index: int) -> dict[str, float]:
    vector_zyx = eigenvectors_zyx[:, index]
    return {
        f"orientation_{index + 1}_x": float(vector_zyx[2]),
        f"orientation_{index + 1}_y": float(vector_zyx[1]),
        f"orientation_{index + 1}_z": float(vector_zyx[0]),
    }


def measure_lacuna_morphology(
    labeled_array: np.ndarray,
    component_table: pd.DataFrame,
    spacing: tuple[float, float, float],
    spacing_length_unit: str = "mm",
    edge_width: int = 2,
) -> pd.DataFrame:
    """Append morphology metrics to a connected-component table."""
    measured = component_table.copy()
    morphology_columns: dict[str, list[Any]] = {
        "centroid_x_um": [],
        "centroid_y_um": [],
        "centroid_z_um": [],
        "volume_um3": [],
        "surface_area_um2": [],
        "surface_area_to_volume_ratio": [],
        "lacuna_stretch": [],
        "lacuna_oblateness": [],
        "orientation_1_x": [],
        "orientation_1_y": [],
        "orientation_1_z": [],
        "orientation_2_x": [],
        "orientation_2_y": [],
        "orientation_2_z": [],
        "major_axis_radius_um": [],
        "minor_axis_radius_um": [],
        "minor_to_major_axis_ratio": [],
        "border": [],
    }

    spacing_zyx_um = _spacing_to_um(spacing, spacing_length_unit)
    spacing_xyz_um = _spacing_xyz_to_um(spacing, spacing_length_unit)
    voxel_volume_um3 = float(np.prod(spacing_zyx_um))
    label_array = np.asarray(labeled_array)

    for label in measured["label"].to_numpy(dtype=int):
        component_mask = label_array == label
        coords_zyx = np.argwhere(component_mask)
        voxel_count = int(coords_zyx.shape[0])
        volume_um3 = float(voxel_count * voxel_volume_um3)

        if voxel_count == 0:
            centroid_xyz_um = np.array([math.nan, math.nan, math.nan], dtype=float)
            surface_area = 0.0
            eigenvalues, eigenvectors = np.zeros(3, dtype=float), np.eye(3, dtype=float)
        else:
            centroid_zyx_um = coords_zyx.mean(axis=0) * spacing_zyx_um
            centroid_xyz_um = np.array([centroid_zyx_um[2], centroid_zyx_um[1], centroid_zyx_um[0]])
            surface_area = _component_surface_area_um2(component_mask, spacing_xyz_um)
            eigenvalues, eigenvectors = _principal_axes(coords_zyx, spacing_zyx_um)

        shape_metrics = _shape_from_eigenvalues(eigenvalues)
        orientation_1 = _orientation_xyz(eigenvectors, 0)
        orientation_2 = _orientation_xyz(eigenvectors, 1)

        morphology_columns["centroid_x_um"].append(float(centroid_xyz_um[0]))
        morphology_columns["centroid_y_um"].append(float(centroid_xyz_um[1]))
        morphology_columns["centroid_z_um"].append(float(centroid_xyz_um[2]))
        morphology_columns["volume_um3"].append(volume_um3)
        morphology_columns["surface_area_um2"].append(surface_area)
        morphology_columns["surface_area_to_volume_ratio"].append(
            float(surface_area / volume_um3) if volume_um3 > 0.0 else 0.0
        )
        for key, value in shape_metrics.items():
            morphology_columns[key].append(value)
        for key, value in orientation_1.items():
            morphology_columns[key].append(value)
        for key, value in orientation_2.items():
            morphology_columns[key].append(value)
        morphology_columns["border"].append(_is_border_component(coords_zyx, label_array.shape, edge_width))

    for column, values in morphology_columns.items():
        measured[column] = values

    return measured
