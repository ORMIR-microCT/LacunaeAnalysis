from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from lacunae_analysis.models import LoadedScan
from lacunae_analysis.segmentation import (
    calculate_volume,
    plot_lacuna_segmentation,
    segment_lacunae,
    voxel_sigma_to_physical_sigma,
)


def make_scan(data: np.ndarray, spacing: tuple[float, float, float] = (0.5, 0.5, 0.5)) -> LoadedScan:
    return LoadedScan(
        source_path=Path('scan.aim'),
        voxel_data=np.asarray(data, dtype=float),
        spacing=spacing,
        origin=(0.0, 0.0, 0.0),
        units='density',
        density_slope=1.0,
        density_intercept=0.0,
    )


def test_calculate_volume_uses_spacing() -> None:
    image = sitk.GetImageFromArray(np.ones((2, 2, 2), dtype=np.uint8))
    image.SetSpacing((0.5, 0.5, 0.5))

    assert calculate_volume(image) == 1.0


def test_voxel_sigma_to_physical_sigma_uses_mean_spacing() -> None:
    image = sitk.GetImageFromArray(np.zeros((2, 2, 2), dtype=np.float32))
    image.SetSpacing((1.0, 2.0, 3.0))

    sigma = voxel_sigma_to_physical_sigma(image, 1.5)

    assert sigma == 3.0


def test_segment_lacunae_returns_binary_masks_and_volumes() -> None:
    data = np.full((4, 4, 4), 300.0, dtype=float)
    data[2, 2, 2] = 100.0
    scan = make_scan(data)

    results = segment_lacunae(scan, threshold=200.0, bone_sigma=1.0, lacuna_sigma=0.01)

    assert results['threshold'] == 200.0
    assert results['inverted_threshold'] == 100.0
    assert results['bone_mask_array'].dtype == np.uint8
    assert results['lacuna_binary_array'].dtype == np.uint8
    assert results['bone_voxels'] == 64
    assert results['lacuna_voxels'] == 1
    assert results['bone_volume'] == 8.0
    assert results['lacuna_volume'] == 0.125


def test_plot_lacuna_segmentation_uses_filtered_lacunae_for_final_panel(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    segmentation_results = {
        "input_array": np.arange(27, dtype=float).reshape((3, 3, 3)),
        "bone_mask_array": np.ones((3, 3, 3), dtype=np.uint8),
        "lacuna_binary_array": np.ones((3, 3, 3), dtype=np.uint8),
    }
    filtered_lacunae = np.zeros((3, 3, 3), dtype=np.uint8)
    filtered_lacunae[1, 1, 1] = 1

    figure = plot_lacuna_segmentation(
        segmentation_results,
        density_results={"filtered_lacuna_binary_array": filtered_lacunae},
        slice_index=1,
        output_path=tmp_path / "segmentation_diagnostic.png",
        show=False,
    )

    try:
        assert figure.axes[3].get_title() == "Volume-filtered lacunae"
        assert (tmp_path / "segmentation_diagnostic.png").exists()
    finally:
        plt.close(figure)
