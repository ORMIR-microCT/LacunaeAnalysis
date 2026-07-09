from __future__ import annotations

import numpy as np
import pandas as pd

from lacunae_analysis.morphology import measure_lacuna_morphology


def test_measure_lacuna_morphology_adds_shape_metrics_for_filtered_labels() -> None:
    labeled = np.zeros((7, 7, 7), dtype=np.uint16)
    labeled[2:5, 3, 3] = 1
    labeled[3, 3:5, 3] = 1
    component_table = pd.DataFrame({"label": [1], "voxel_count": [4], "volume": [4e-9]})

    measured = measure_lacuna_morphology(
        labeled,
        component_table,
        spacing=(0.001, 0.001, 0.001),
        spacing_length_unit="mm",
    )

    row = measured.iloc[0]
    assert row["volume_um3"] == 4.0
    assert row["centroid_x_um"] == 3.0
    assert row["centroid_y_um"] == 3.25
    assert row["centroid_z_um"] == 3.0
    assert row["surface_area_um2"] > 0.0
    assert row["surface_area_to_volume_ratio"] > 0.0
    assert row["major_axis_radius_um"] >= row["minor_axis_radius_um"]
    assert 0.0 <= row["minor_to_major_axis_ratio"] <= 1.0
    assert 0.0 <= row["lacuna_stretch"] <= 1.0
    assert row["border"] == 0


def test_measure_lacuna_morphology_flags_border_components() -> None:
    labeled = np.zeros((4, 4, 4), dtype=np.uint16)
    labeled[0, 1, 1] = 1
    component_table = pd.DataFrame({"label": [1], "voxel_count": [1], "volume": [1e-9]})

    measured = measure_lacuna_morphology(
        labeled,
        component_table,
        spacing=(0.001, 0.001, 0.001),
        spacing_length_unit="mm",
        edge_width=1,
    )

    assert measured.loc[0, "border"] == 1
