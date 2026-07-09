"""Command-line entry point for lacunae analysis."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lacunae-analysis")
    parser.add_argument(
        "--version",
        action="version",
        version="lacunae-analysis 0.1.0",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0

