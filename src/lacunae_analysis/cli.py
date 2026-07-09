"""Command-line entry point for lacunae analysis."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import yaml

def run_single_scan_job(input_path: Path, config: dict[str, object], output_dir: Path):
    from .pipeline import run_single_scan_job as pipeline_run_single_scan_job

    return pipeline_run_single_scan_job(input_path, config, output_dir)


def run_batch_job(input_dir: Path, config: dict[str, object], output_dir: Path):
    from .pipeline import run_batch_job as pipeline_run_batch_job

    return pipeline_run_batch_job(input_dir, config, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lacunae-analysis")
    parser.add_argument(
        "--version",
        action="version",
        version="lacunae-analysis 0.1.0",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    single_parser = subparsers.add_parser("single")
    single_parser.add_argument("input_path")
    single_parser.add_argument("--config", required=True)
    single_parser.add_argument("--output-dir", required=True)

    batch_parser = subparsers.add_parser("batch")
    batch_parser.add_argument("input_dir")
    batch_parser.add_argument("--config", required=True)
    batch_parser.add_argument("--output-dir", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if args.command == "single":
        run_single_scan_job(Path(args.input_path), config, Path(args.output_dir))
        return 0

    run_batch_job(Path(args.input_dir), config, Path(args.output_dir))
    return 0
