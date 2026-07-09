from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lacunae_analysis.models import BatchRun, DensityFilterSettings, LoadedScan, ScanInput, ThresholdSettings
from lacunae_analysis.pipeline import run_batch, run_batch_job, run_single_scan, summarize_batch_results


def make_scan() -> LoadedScan:
    return LoadedScan(
        source_path=Path("/tmp/sample.aim"),
        voxel_data=np.ones((2, 2, 2), dtype=float),
        spacing=(0.01, 0.01, 0.01),
        origin=(0.0, 0.0, 0.0),
        units="density",
        density_slope=1.0,
        density_intercept=0.0,
    )


def test_run_single_scan_chains_loading_threshold_segmentation_and_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    scan = make_scan()

    def fake_load(path: str | Path) -> LoadedScan:
        calls.append(("load", Path(path)))
        return scan

    def fake_threshold(loaded_scan: LoadedScan, manual_threshold: float = 0.0) -> dict[str, float]:
        calls.append(("threshold", loaded_scan, manual_threshold))
        return {"selected_threshold": 222.0, "manual_threshold": manual_threshold}

    def fake_segment(
        loaded_scan: LoadedScan,
        threshold: float,
        bone_sigma: float,
        lacuna_sigma: float,
    ) -> dict[str, object]:
        calls.append(("segment", loaded_scan, threshold, bone_sigma, lacuna_sigma))
        return {
            "lacuna_binary_sitk": "lacuna-image",
            "bone_mask_sitk": "bone-image",
        }

    def fake_analyze(
        lacuna_binary_input: object,
        bone_mask_input: object,
        lower_volume_um3: float,
        upper_volume_um3: float,
        spacing_length_unit: str,
        return_images: bool,
    ) -> dict[str, object]:
        calls.append(
            (
                "analyze",
                lacuna_binary_input,
                bone_mask_input,
                lower_volume_um3,
                upper_volume_um3,
                spacing_length_unit,
                return_images,
            )
        )
        return {"summary": {"n_lacunae": 4, "lacuna_density_per_mm3": 12.5}}

    monkeypatch.setattr("lacunae_analysis.pipeline.load_aim_as_density", fake_load)
    monkeypatch.setattr("lacunae_analysis.pipeline.compute_threshold", fake_threshold)
    monkeypatch.setattr("lacunae_analysis.pipeline.segment_lacunae", fake_segment)
    monkeypatch.setattr("lacunae_analysis.pipeline.analyze_lacuna_density", fake_analyze)

    results = run_single_scan(
        ScanInput(image_path=Path("/tmp/sample.aim")),
        threshold_settings=ThresholdSettings(manual_threshold=205.0),
        density_filter_settings=DensityFilterSettings(lower_volume_um3=200.0, upper_volume_um3=1500.0),
        bone_sigma=10.0,
        lacuna_sigma=1.2,
    )

    assert results["summary"] == {"n_lacunae": 4, "lacuna_density_per_mm3": 12.5}
    assert [call[0] for call in calls] == ["load", "threshold", "segment", "analyze"]
    assert calls[1][2] == 205.0
    assert calls[2][2:] == (222.0, 10.0, 1.2)
    assert calls[3][1:6] == ("lacuna-image", "bone-image", 200.0, 1500.0, "mm")


def test_run_batch_builds_rows_from_run_single_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []

    def fake_run_single_scan(scan_input: ScanInput, **_: object) -> dict[str, object]:
        calls.append(scan_input.image_path)
        if scan_input.image_path.stem == "broken":
            raise RuntimeError("bad scan")
        return {
            "summary": {"n_lacunae": 2, "lacuna_density_per_mm3": 4.5},
            "threshold_results": {"selected_threshold": 210.0},
        }

    monkeypatch.setattr("lacunae_analysis.pipeline.run_single_scan", fake_run_single_scan)

    batch = BatchRun(
        scans=[
            ScanInput(image_path=Path("/tmp/alpha.aim"), metadata={"group": "control"}),
            ScanInput(image_path=Path("/tmp/broken.aim")),
            ScanInput(image_path=Path("/tmp/beta.aim"), metadata={"group": "treated"}),
        ]
    )

    table = run_batch(batch)

    assert calls == [Path("/tmp/alpha.aim"), Path("/tmp/broken.aim"), Path("/tmp/beta.aim")]
    assert isinstance(table, pd.DataFrame)
    assert table[["scan_name", "status"]].to_dict(orient="records") == [
        {"scan_name": "alpha", "status": "ok"},
        {"scan_name": "broken", "status": "error"},
        {"scan_name": "beta", "status": "ok"},
    ]
    assert table.loc[table["scan_name"] == "alpha", "group"].item() == "control"
    assert table.loc[table["scan_name"] == "beta", "lacuna_density_per_mm3"].item() == 4.5


def test_summarize_batch_results_returns_stable_dataframe() -> None:
    frame = summarize_batch_results(
        [
            {"scan_name": "alpha", "image_path": "/tmp/alpha.aim", "status": "ok", "n_lacunae": 10},
            {"scan_name": "beta", "image_path": "/tmp/beta.aim", "status": "error", "error": "bad scan"},
        ]
    )

    assert isinstance(frame, pd.DataFrame)
    assert frame.columns.tolist()[:4] == ["scan_name", "image_path", "status", "error"]
    assert frame["scan_name"].tolist() == ["alpha", "beta"]
    assert frame.loc[frame["scan_name"] == "alpha", "n_lacunae"].item() == 10
    assert pd.isna(frame.loc[frame["scan_name"] == "alpha", "error"].item())
    assert frame.loc[frame["scan_name"] == "beta", "error"].item() == "bad scan"


def test_run_batch_job_writes_batch_summary_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    output_dir.mkdir()

    for scan_name in ("alpha", "beta"):
        (input_dir / f"{scan_name}.aim").write_text("fake")

    single_scan_calls: list[tuple[Path, Path, dict[str, object]]] = []

    def fake_run_single_scan_job(
        input_path: str | Path,
        config: dict[str, object] | None,
        output_path: str | Path,
    ) -> dict[str, object]:
        config_dict = {} if config is None else dict(config)
        single_scan_calls.append((Path(input_path), Path(output_path), config_dict))
        return {
            "summary": {"n_lacunae": 2, "lacuna_density_per_mm3": 4.5},
            "threshold_results": {"selected_threshold": 210.0},
        }

    written_csv: list[tuple[Path, pd.DataFrame]] = []
    written_json: list[tuple[Path, pd.DataFrame, dict[str, object]]] = []

    def fake_write_batch_summary_csv(path: Path, table: pd.DataFrame) -> None:
        written_csv.append((path, table.copy()))

    def fake_write_batch_summary_json(
        path: Path,
        table: pd.DataFrame,
        config: dict[str, object],
    ) -> None:
        written_json.append((path, table.copy(), dict(config)))

    monkeypatch.setattr("lacunae_analysis.pipeline.run_single_scan_job", fake_run_single_scan_job)
    monkeypatch.setattr("lacunae_analysis.pipeline.write_batch_summary_csv", fake_write_batch_summary_csv)
    monkeypatch.setattr("lacunae_analysis.pipeline.write_batch_summary_json", fake_write_batch_summary_json)

    config = {
        "scan": {"species": "mouse"},
        "batch": {
            "scans": [
                {"image_path": "alpha.aim", "group": "control"},
                {"image_path": "beta.aim", "group": "treated", "output_dir": "nested/beta-run"},
            ]
        },
    }

    table = run_batch_job(input_dir, config, output_dir)

    assert table["scan_name"].tolist() == ["alpha", "beta"]
    assert table["group"].tolist() == ["control", "treated"]
    assert single_scan_calls == [
        (
            input_dir / "alpha.aim",
            output_dir / "alpha",
            {"scan": {"species": "mouse", "group": "control"}, "batch": config["batch"]},
        ),
        (
            input_dir / "beta.aim",
            output_dir / "nested/beta-run",
            {"scan": {"species": "mouse", "group": "treated", "output_dir": "nested/beta-run"}, "batch": config["batch"]},
        ),
    ]
    assert written_csv and written_json
    assert written_csv[0][0] == output_dir / "batch_summary.csv"
    assert written_json[0][0] == output_dir / "batch_summary.json"
    assert written_csv[0][1]["selected_threshold"].tolist() == [210.0, 210.0]
