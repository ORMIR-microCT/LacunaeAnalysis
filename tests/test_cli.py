from __future__ import annotations

from pathlib import Path

import yaml

from lacunae_analysis import cli


def test_cli_has_single_and_batch_subcommands() -> None:
    parser = cli.build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert {"single", "batch"} <= set(choices)


def test_main_single_delegates_to_pipeline(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[Path, dict[str, object], Path]] = []

    def fake_run_single(input_path: Path, config: dict[str, object], output_dir: Path) -> dict[str, object]:
        calls.append((input_path, config, output_dir))
        return {"status": "ok"}

    config_path = tmp_path / "single.yaml"
    config = {"thresholding": {"manual_threshold": 205.0}}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.setattr(cli, "run_single_scan_job", fake_run_single)

    exit_code = cli.main(
        [
            "single",
            str(tmp_path / "scan.aim"),
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            tmp_path / "scan.aim",
            config,
            tmp_path / "out",
        )
    ]


def test_main_batch_delegates_to_pipeline(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[Path, dict[str, object], Path]] = []

    def fake_run_batch(input_dir: Path, config: dict[str, object], output_dir: Path) -> dict[str, object]:
        calls.append((input_dir, config, output_dir))
        return {"status": "ok"}

    config_path = tmp_path / "batch.yaml"
    config = {"density_filter": {"lower_volume_um3": 200.0}}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.setattr(cli, "run_batch_job", fake_run_batch)

    exit_code = cli.main(
        [
            "batch",
            str(tmp_path / "inputs"),
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            tmp_path / "inputs",
            config,
            tmp_path / "out",
        )
    ]
