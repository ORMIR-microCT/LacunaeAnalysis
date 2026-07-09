from __future__ import annotations

from pathlib import Path

import numpy as np

from lacunae_analysis.models import LoadedScan
from lacunae_analysis.thresholding import compute_threshold, find_local_peaks_1d, smooth_histogram


def make_scan(data: np.ndarray) -> LoadedScan:
    return LoadedScan(
        source_path=Path('scan.aim'),
        voxel_data=np.asarray(data, dtype=float),
        spacing=(1.2, 1.2, 1.2),
        origin=(0.0, 0.0, 0.0),
        units='density',
        density_slope=1.0,
        density_intercept=0.0,
    )


def test_smooth_histogram_preserves_shape() -> None:
    hist = np.array([0, 1, 4, 1, 0], dtype=float)

    smoothed = smooth_histogram(hist, window=3)

    assert smoothed.shape == hist.shape


def test_find_local_peaks_enforces_min_distance() -> None:
    signal = np.array([0.0, 3.0, 0.0, 2.5, 0.0, 5.0, 0.0])

    peaks = find_local_peaks_1d(signal, min_distance=3)

    assert peaks.tolist() == [1, 5]


def test_compute_threshold_returns_density_thresholds_and_manual_value() -> None:
    scan = make_scan(np.array([[[0.0, 0.0, 90.0, 95.0, 100.0, 105.0, 110.0, 280.0, 285.0, 290.0, 295.0, 300.0]]]))

    results = compute_threshold(scan, manual_threshold=215.0)

    assert results['manual_threshold'] == 215.0
    assert results['selected_threshold'] == 215.0
    assert results['n_voxels_used'] == 10
    assert 90.0 <= results['otsu_threshold'] <= 300.0
    assert 90.0 <= results['peak_threshold'] <= 300.0
