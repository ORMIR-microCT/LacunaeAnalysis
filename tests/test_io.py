from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lacunae_analysis.io import convert_to_density, load_aim, load_aim_as_density, load_scan, write_binary_mask
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


def test_load_aim_defaults_to_bmd_and_preserves_geometry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "scan.aim"
    path.write_bytes(b"aim")
    recorded: dict[str, object] = {}
    density_data = np.array([[12.5, 15.0], [17.5, 20.0]])
    metadata = {
        "spacing": (1.2, 1.3, 1.4),
        "origin": (0.1, 0.2, 0.3),
        "unit": "BMD",
        "processing_log_raw": "calibration text",
        "scanner": "test-rig",
    }

    class FakeAimIO:
        @staticmethod
        def read_aim(input_path: str, density: bool = False, hu: bool = False):
            recorded["input_path"] = input_path
            recorded["density"] = density
            recorded["hu"] = hu
            return density_data, metadata

        @staticmethod
        def get_aim_density_equation(processing_log: str) -> tuple[float, float]:
            recorded["processing_log"] = processing_log
            return 2.5, 10.0

        @staticmethod
        def dict_to_log(processing_log: dict) -> str:
            return str(processing_log)

    monkeypatch.setattr("lacunae_analysis.io._import_py_aimio", lambda: FakeAimIO)

    loaded = load_aim(path)

    assert recorded == {
        "input_path": str(path),
        "density": True,
        "hu": False,
        "processing_log": "calibration text",
    }
    assert isinstance(loaded, LoadedScan)
    assert loaded.source_path == path
    assert np.array_equal(loaded.voxel_data, density_data)
    assert loaded.spacing == (1.2, 1.3, 1.4)
    assert loaded.origin == (0.1, 0.2, 0.3)
    assert loaded.units == "BMD"
    assert loaded.density_slope == 2.5
    assert loaded.density_intercept == 10.0
    assert loaded.metadata["scanner"] == "test-rig"


def test_load_aim_can_return_raw_greyscale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "scan.aim"
    path.write_bytes(b"aim")
    recorded: dict[str, object] = {}
    raw_data = np.array([[10, 20], [30, 40]], dtype=np.int16)
    metadata = {
        "spacing": (1.0, 1.0, 1.0),
        "origin": (0.0, 0.0, 0.0),
        "processing_log_raw": "calibration text",
    }

    class FakeAimIO:
        @staticmethod
        def read_aim(input_path: str, density: bool = False, hu: bool = False):
            recorded["input_path"] = input_path
            recorded["density"] = density
            recorded["hu"] = hu
            return raw_data, metadata

        @staticmethod
        def get_aim_density_equation(processing_log: str) -> tuple[float, float]:
            return 2.5, 10.0

    monkeypatch.setattr("lacunae_analysis.io._import_py_aimio", lambda: FakeAimIO)

    loaded = load_aim(path, intensity_unit="raw")

    assert recorded == {"input_path": str(path), "density": False, "hu": False}
    assert np.array_equal(loaded.voxel_data, raw_data.astype(np.float64))
    assert loaded.units == "raw"


def test_load_aim_can_return_hu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "scan.aim"
    path.write_bytes(b"aim")
    recorded: dict[str, object] = {}
    hu_data = np.array([[-1000.0, 0.0], [500.0, 1200.0]])
    metadata = {
        "spacing": (1.0, 1.0, 1.0),
        "origin": (0.0, 0.0, 0.0),
        "unit": "HU",
        "processing_log_raw": "calibration text",
    }

    class FakeAimIO:
        @staticmethod
        def read_aim(input_path: str, density: bool = False, hu: bool = False):
            recorded["input_path"] = input_path
            recorded["density"] = density
            recorded["hu"] = hu
            return hu_data, metadata

        @staticmethod
        def get_aim_density_equation(processing_log: str) -> tuple[float, float]:
            return 2.5, 10.0

    monkeypatch.setattr("lacunae_analysis.io._import_py_aimio", lambda: FakeAimIO)

    loaded = load_aim(path, intensity_unit="HU")

    assert recorded == {"input_path": str(path), "density": False, "hu": True}
    assert np.array_equal(loaded.voxel_data, hu_data)
    assert loaded.units == "HU"


def test_load_aim_reports_unsupported_mu_for_aim(tmp_path) -> None:
    path = tmp_path / "scan.aim"
    path.write_bytes(b"aim")

    with pytest.raises(NotImplementedError, match="linear attenuation"):
        load_aim(path, intensity_unit="mu")


def test_load_scan_dispatches_aim_with_selected_intensity_unit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "scan.aim"
    path.write_bytes(b"aim")
    recorded: dict[str, object] = {}

    def fake_load_aim(input_path: str | Path, intensity_unit: str | None = None) -> LoadedScan:
        recorded["input_path"] = input_path
        recorded["intensity_unit"] = intensity_unit
        return LoadedScan(
            source_path=Path(input_path),
            voxel_data=np.ones((1, 1, 1)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            units=str(intensity_unit),
            density_slope=1.0,
            density_intercept=0.0,
        )

    monkeypatch.setattr("lacunae_analysis.io.load_aim", fake_load_aim)

    load_scan(path, intensity_unit="raw")

    assert recorded == {"input_path": path, "intensity_unit": "raw"}


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

    def fake_write_aim(output_path, data, meta=None, unit=None) -> None:
        recorded["output_path"] = output_path
        recorded["data"] = data
        recorded["meta"] = meta
        recorded["unit"] = unit

    class FakeAimIO:
        write_aim = staticmethod(fake_write_aim)

    monkeypatch.setattr("lacunae_analysis.io._import_py_aimio", lambda: FakeAimIO)

    write_binary_mask(path, scan=scan, mask=mask)

    assert recorded["output_path"] == str(path)
    assert np.array_equal(recorded["data"], np.array([[1, 0]], dtype=np.uint8))
    assert recorded["meta"] == {
        "sample_id": "A1",
        "density_slope": 2.0,
        "density_intercept": 8.0,
        "units": "binary_mask",
    }
    assert recorded["unit"] == "native"
