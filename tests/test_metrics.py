from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import pytest

from lacunae_analysis.metrics import (
    analyze_lacuna_density,
    binary_array_to_sitk,
    build_filtered_binary,
    filter_component_table,
    get_voxel_volume,
    label_and_measure_components,
    summarize_lacuna_density,
    um3_to_image_volume,
)


def make_binary_image(
    array: np.ndarray,
    spacing: tuple[float, float, float] = (0.01, 0.01, 0.01),
) -> sitk.Image:
    image = sitk.GetImageFromArray(np.asarray(array, dtype=np.uint8))
    image.SetSpacing(spacing)
    image.SetOrigin((1.0, 2.0, 3.0))
    return image


def test_get_voxel_volume_multiplies_spacing() -> None:
    image = make_binary_image(np.zeros((1, 1, 1), dtype=np.uint8), spacing=(1.2, 1.2, 1.2))

    assert get_voxel_volume(image) == pytest.approx(1.728)


def test_um3_to_image_volume_supports_mm_and_um() -> None:
    assert um3_to_image_volume(200.0, spacing_length_unit="mm") == pytest.approx(2e-7)
    assert um3_to_image_volume(200.0, spacing_length_unit="um") == pytest.approx(200.0)


def test_binary_array_to_sitk_preserves_reference_metadata() -> None:
    reference = make_binary_image(np.zeros((2, 2, 2), dtype=np.uint8), spacing=(0.5, 0.6, 0.7))

    converted = binary_array_to_sitk(np.ones((2, 2, 2), dtype=bool), reference)

    assert converted.GetSpacing() == reference.GetSpacing()
    assert converted.GetOrigin() == reference.GetOrigin()
    assert sitk.GetArrayFromImage(converted).dtype == np.uint8


def test_label_and_measure_components_returns_component_table() -> None:
    binary = make_binary_image(
        np.array(
            [
                [[1, 1, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
            ],
            dtype=np.uint8,
        )
    )

    labeled_image, labeled_array, table = label_and_measure_components(binary)

    assert sitk.GetArrayFromImage(labeled_image).shape == labeled_array.shape
    assert table["label"].tolist() == [1, 2]
    assert table["voxel_count"].tolist() == [2, 2]
    assert table["volume"].tolist() == pytest.approx([2e-6, 2e-6])


def test_filter_component_table_converts_volume_thresholds() -> None:
    component_table = pd.DataFrame(
        [
            {"label": 1, "voxel_count": 1, "volume": 1.0},
            {"label": 2, "voxel_count": 3, "volume": 3.0},
            {"label": 3, "voxel_count": 2, "volume": 2.0},
        ]
    )

    filtered, lower_voxels, upper_voxels = filter_component_table(
        component_table,
        lower_volume=1.5,
        upper_volume=3.4,
        voxel_volume=1.0,
    )

    assert (lower_voxels, upper_voxels) == (2, 3)
    assert filtered["label"].tolist() == [2, 3]


def test_build_filtered_binary_keeps_selected_labels() -> None:
    labeled = np.array([[[0, 1, 2], [2, 0, 3]]], dtype=np.uint16)

    filtered = build_filtered_binary(labeled, kept_labels=[2, 3])

    assert filtered.tolist() == [[[0, 0, 1], [1, 0, 1]]]


def test_summarize_lacuna_density_reports_density_statistics() -> None:
    filtered_binary = make_binary_image(np.array([[[1, 0], [0, 1]]], dtype=np.uint8))
    bone_mask = make_binary_image(np.ones((1, 2, 2), dtype=np.uint8))
    component_table = pd.DataFrame(
        [
            {"label": 1, "voxel_count": 1, "volume": 1e-6},
            {"label": 2, "voxel_count": 1, "volume": 1e-6},
        ]
    )

    summary = summarize_lacuna_density(filtered_binary, bone_mask, component_table)

    assert summary["n_lacunae"] == 2
    assert summary["bone_voxels"] == 4
    assert summary["lacuna_voxels"] == 2
    assert summary["voxel_volume"] == pytest.approx(1e-6)
    assert summary["bone_volume"] == pytest.approx(4e-6)
    assert summary["lacuna_volume"] == pytest.approx(2e-6)
    assert summary["mean_lacuna_volume"] == pytest.approx(1e-6)
    assert summary["median_lacuna_volume"] == pytest.approx(1e-6)
    assert summary["lacuna_density_per_volume"] == pytest.approx(500000.0)
    assert summary["lacuna_volume_fraction"] == pytest.approx(0.5)


def test_analyze_lacuna_density_filters_components_and_returns_summary() -> None:
    lacuna_array = np.zeros((6, 6, 6), dtype=np.uint8)
    lacuna_array[2, 2, 2:4] = 1
    lacuna_binary = make_binary_image(lacuna_array, spacing=(1.0, 1.0, 1.0))
    bone_mask = make_binary_image(np.ones((6, 6, 6), dtype=np.uint8), spacing=(1.0, 1.0, 1.0))

    results = analyze_lacuna_density(
        lacuna_binary,
        bone_mask,
        lower_volume_um3=1.5,
        upper_volume_um3=3.0,
        spacing_length_unit="um",
    )

    assert results["component_table"]["voxel_count"].tolist() == [2]
    assert "surface_area_um2" in results["component_table"].columns
    assert "lacuna_stretch" in results["component_table"].columns
    assert "mean_lacuna_surface_area_um2" in results["summary"]
    assert results["lower_voxel_threshold"] == 2
    assert results["upper_voxel_threshold"] == 3
    assert results["filtered_lacuna_binary_array"].sum() == 2
    assert results["summary"]["n_lacunae"] == 1
    assert results["summary"]["lacuna_density_per_mm3"] > 0.0


def test_analyze_lacuna_density_excludes_edge_lacunae_by_default() -> None:
    lacuna_binary = make_binary_image(
        np.array(
            [
                [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            dtype=np.uint8,
        ),
        spacing=(1.0, 1.0, 1.0),
    )
    bone_mask = make_binary_image(np.ones((3, 4, 4), dtype=np.uint8), spacing=(1.0, 1.0, 1.0))

    results = analyze_lacuna_density(
        lacuna_binary,
        bone_mask,
        lower_volume_um3=0.0,
        upper_volume_um3=None,
        spacing_length_unit="um",
        edge_width=1,
    )

    assert results["component_table"]["label"].tolist() == [2]
    assert results["component_table"]["border"].tolist() == [0]
    assert results["summary"]["n_lacunae"] == 1
    assert results["summary"]["excluded_border_count"] == 1
    assert results["filtered_lacuna_binary_array"].sum() == 3


def test_analyze_lacuna_density_can_include_edge_lacunae() -> None:
    lacuna_binary = make_binary_image(
        np.array(
            [
                [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 1], [0, 1, 0]],
            ],
            dtype=np.uint8,
        ),
        spacing=(1.0, 1.0, 1.0),
    )
    bone_mask = make_binary_image(np.ones((2, 3, 3), dtype=np.uint8), spacing=(1.0, 1.0, 1.0))

    results = analyze_lacuna_density(
        lacuna_binary,
        bone_mask,
        lower_volume_um3=0.0,
        upper_volume_um3=None,
        spacing_length_unit="um",
        include_edge_lacunae=True,
        edge_width=1,
    )

    assert results["component_table"]["label"].tolist() == [2, 1]
    assert results["component_table"]["border"].tolist() == [1, 1]
    assert results["summary"]["n_lacunae"] == 2
    assert results["summary"]["excluded_border_count"] == 0
