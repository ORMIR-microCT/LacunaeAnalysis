"""Thresholding utilities for density-space lacunae analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import SimpleITK as sitk

from .models import LoadedScan


DEFAULT_BINS = 500
DEFAULT_SMOOTH_WINDOW = 7
DEFAULT_MIN_PEAK_HEIGHT_FRAC = 0.01
DEFAULT_MIN_PEAK_DISTANCE = 8


def _scan_to_sitk(scan: LoadedScan) -> sitk.Image:
    image = sitk.GetImageFromArray(np.asarray(scan.voxel_data, dtype=np.float32))
    image.SetSpacing(tuple(float(value) for value in scan.spacing))
    image.SetOrigin(tuple(float(value) for value in scan.origin))
    return image


def get_otsu_threshold(image: sitk.Image) -> float:
    """Otsu threshold using SimpleITK, restricted to nonzero voxels."""
    mask = image > 0
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetMaskValue(1)
    otsu_filter.Execute(image, mask)
    return float(otsu_filter.GetThreshold())


def smooth_histogram(hist: np.ndarray, window: int = DEFAULT_SMOOTH_WINDOW) -> np.ndarray:
    """Simple moving-average smoothing."""
    if window <= 1:
        return np.asarray(hist, dtype=float)

    kernel = np.ones(window, dtype=float) / window
    return np.convolve(np.asarray(hist, dtype=float), kernel, mode="same")


def find_local_peaks_1d(
    y: np.ndarray,
    min_height: float | None = None,
    min_distance: int = 5,
) -> np.ndarray:
    """NumPy-based local peak finder."""
    values = np.asarray(y)

    if values.ndim != 1 or len(values) < 3:
        return np.array([], dtype=int)

    peaks = np.where((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]))[0] + 1

    if min_height is not None:
        peaks = peaks[values[peaks] >= min_height]

    if len(peaks) == 0:
        return peaks

    order = np.argsort(values[peaks])[::-1]
    selected: list[int] = []

    for index in peaks[order]:
        if all(abs(int(index) - chosen) >= min_distance for chosen in selected):
            selected.append(int(index))

    return np.array(sorted(selected), dtype=int)


def get_peak_threshold(
    vi_array: np.ndarray,
    bins: int = DEFAULT_BINS,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    min_peak_height_frac: float = DEFAULT_MIN_PEAK_HEIGHT_FRAC,
    min_peak_distance: int = DEFAULT_MIN_PEAK_DISTANCE,
    ax: Any | None = None,
    check_fit: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Estimate threshold from the highest-intensity meaningful histogram peak."""
    values = np.asarray(vi_array, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("Input intensity array is empty.")

    hist, bin_edges = np.histogram(
        values,
        bins=bins,
        range=(np.min(values), np.max(values)),
    )
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    hist_smooth = smooth_histogram(hist, window=smooth_window)

    min_height = np.max(hist_smooth) * min_peak_height_frac
    peaks = find_local_peaks_1d(
        hist_smooth,
        min_height=min_height,
        min_distance=min_peak_distance,
    )

    if len(peaks) == 0:
        peak_idx = int(np.argmax(hist_smooth))
    else:
        peak_idx = int(peaks[-1])

    peak_height = hist_smooth[peak_idx]
    frac = 0.25
    left_idx = peak_idx
    while left_idx > 0 and hist_smooth[left_idx] > frac * peak_height:
        left_idx -= 1

    peak_thresh = float(bin_centres[left_idx])

    if check_fit and ax is not None:
        ax.plot(bin_centres, hist_smooth, "k-", linewidth=1.5, label="Smoothed histogram")

        if len(peaks) > 0:
            ax.plot(
                bin_centres[peaks],
                hist_smooth[peaks],
                "x",
                color="orange",
                label="Detected peaks",
            )

        ax.axvline(
            bin_centres[peak_idx],
            color="gray",
            linestyle=":",
            linewidth=1.5,
            label=f"Chosen peak = {bin_centres[peak_idx]:.1f}",
        )

    debug = {
        "hist": hist,
        "hist_smooth": hist_smooth,
        "bin_edges": bin_edges,
        "bin_centres": bin_centres,
        "peaks": peaks,
        "chosen_peak_index": peak_idx,
    }
    return peak_thresh, debug


def create_hist(
    scan: LoadedScan,
    manual_threshold: float = 0.0,
    check_fit: bool = False,
    sample_size: int | None = None,
    bins: int = DEFAULT_BINS,
    ignore_below: float | None = None,
    show: bool = True,
) -> tuple[dict[str, float | int], Any, Any, dict[str, Any]]:
    """Generate intensity histogram and notebook-aligned thresholds."""
    import matplotlib.pyplot as plt

    image_array = np.asarray(scan.voxel_data, dtype=float)
    sitk_img = _scan_to_sitk(scan)

    voxel_intensities = image_array.ravel()
    vi = voxel_intensities[voxel_intensities != 0]

    if ignore_below is not None:
        vi = vi[vi > ignore_below]

    if vi.size == 0:
        raise ValueError("No voxels remain after filtering.")

    if sample_size is not None and vi.size > sample_size:
        rng = np.random.default_rng(42)
        vi = rng.choice(vi, size=sample_size, replace=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("Histogram of Voxel Intensities")
    ax.set_xlabel("Voxel Intensity")
    ax.set_ylabel("Frequency")
    ax.hist(
        vi,
        bins=bins,
        range=(np.min(vi), np.max(vi)),
        color="blue",
        alpha=0.5,
        label="Histogram",
    )

    otsu_thresh = get_otsu_threshold(sitk_img)
    peak_thresh, debug = get_peak_threshold(vi, bins=bins, ax=ax, check_fit=check_fit)

    ax.axvline(
        otsu_thresh,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Otsu = {otsu_thresh:.1f}",
    )

    if manual_threshold != 0:
        ax.axvline(
            float(manual_threshold),
            color="green",
            linestyle="dashed",
            linewidth=2,
            label=f"Manual = {float(manual_threshold):.1f}",
        )

    ax.axvline(
        peak_thresh,
        color="purple",
        linestyle="dashed",
        linewidth=2,
        label=f"Peak-based = {peak_thresh:.1f}",
    )
    ax.legend(frameon=False)

    if show:
        plt.show()

    results = {
        "otsu_threshold": float(otsu_thresh),
        "manual_threshold": float(manual_threshold),
        "peak_threshold": float(peak_thresh),
        "n_voxels_used": int(vi.size),
    }
    return results, fig, ax, debug


def _with_selected_threshold(
    results: dict[str, float | int],
    manual_threshold: float = 0.0,
) -> dict[str, float | int]:
    if manual_threshold != 0.0:
        results["selected_threshold"] = float(manual_threshold)
    else:
        results["selected_threshold"] = float(results["peak_threshold"])
    return results


def compute_threshold_with_figure(
    scan: LoadedScan,
    manual_threshold: float = 0.0,
) -> tuple[dict[str, float | int], Any]:
    """Compute thresholds and return the annotated histogram figure."""
    results, fig, _, _ = create_hist(
        scan,
        manual_threshold=manual_threshold,
        check_fit=True,
        show=False,
    )
    return _with_selected_threshold(results, manual_threshold=manual_threshold), fig


def compute_threshold(scan: LoadedScan, manual_threshold: float = 0.0) -> dict[str, float | int]:
    """Compute notebook-aligned thresholds for a loaded scan."""
    import matplotlib.pyplot as plt

    results, fig, _, _ = create_hist(scan, manual_threshold=manual_threshold, show=False)
    plt.close(fig)
    return _with_selected_threshold(results, manual_threshold=manual_threshold)
