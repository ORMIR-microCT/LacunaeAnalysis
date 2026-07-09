from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lacunae_analysis.io import convert_to_density, load_aim_as_density, write_binary_mask
from lacunae_analysis.models import LoadedScan


def test_convert_to_density_applies_linear_calibration() -> None:
    raw = np.array([0.0, 10.0, 20.0], dtype=float)

    converted = convert_to_density(raw, slope=0.5, intercept=100.0)

    assert converted.tolist() == [100.0, 105.0, 110.0]


def test_convert_to_density_requires_calibration() -> None:
    raw = np.array([1.0, 2.0, 3.0], dtype=float)

    with pytest.raises(ValueError, match="density calibration"):
        convert_to_density(raw, slope=None, intercept=None)


def test_load_aim_as_density_rejects_non_aim_input(tmp_path) -> None:
    path = tmp_path / "scan.nii"
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="Unsupported input extension"):
        load_aim_as_density(path)


def test_load_aim_as_density_converts_voxels_and_preserves_geometry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "scan.aim"
    path.write_bytes(b"aim")
    raw_scan = SimpleNamespace(
        data=np.array([[1, 2], [3, 4]], dtype=np.int16),
        spacing=(1.2, 1.3, 1.4),
        origin=(0.1, 0.2, 0.3),
        meta={
            "density_slope": 2.5,
            "density_intercept": 10.0,
            "scanner": "test-rig",
        },
    )

    def fake_import() -> object:
        return SimpleNamespace(read=lambda input_path: raw_scan)

    monkeypatch.setattr("lacunae_analysis.io._import_aimio", fake_import)

    loaded = load_aim_as_density(path)

    assert isinstance(loaded, LoadedScan)
    assert loaded.source_path == path
    assert np.array_equal(loaded.voxel_data, np.array([[12.5, 15.0], [17.5, 20.0]]))
    assert loaded.spacing == (1.2, 1.3, 1.4)
    assert loaded.origin == (0.1, 0.2, 0.3)
    assert loaded.units == "density"
    assert loaded.density_slope == 2.5
    assert loaded.density_intercept == 10.0
    assert loaded.metadata["scanner"] == "test-rig"


def test_write_binary_mask_uses_aimio_with_scan_geometry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "mask.aim"
    scan = LoadedScan(
        source_path=tmp_path / "scan.aim",
        voxel_data=np.array([[12.0, 15.0]], dtype=float),
        spacing=(1.0, 2.0, 3.0),
        origin=(4.0, 5.0, 6.0),
        units="density",
        density_slope=2.0,
        density_intercept=8.0,
        metadata={"sample_id": "A1"},
    )
    mask = np.array([[True, False]])
    recorded: dict[str, object] = {}

    def fake_write(output_path, data, *, spacing, origin, meta) -> None:
        recorded["output_path"] = output_path
        recorded["data"] = data
        recorded["spacing"] = spacing
        recorded["origin"] = origin
        recorded["meta"] = meta

    def fake_import() -> object:
        return SimpleNamespace(write=fake_write)

    monkeypatch.setattr("lacunae_analysis.io._import_aimio", fake_import)

    write_binary_mask(path, scan=scan, mask=mask)

    assert recorded["output_path"] == path
    assert np.array_equal(recorded["data"], np.array([[1, 0]], dtype=np.uint8))
    assert recorded["spacing"] == (1.0, 2.0, 3.0)
    assert recorded["origin"] == (4.0, 5.0, 6.0)
    assert recorded["meta"] == {
        "sample_id": "A1",
        "density_slope": 2.0,
        "density_intercept": 8.0,
        "units": "binary_mask",
    }
