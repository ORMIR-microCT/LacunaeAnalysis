# Lacunae Analysis

Lacunae Analysis is a small Python package for exploring lacunae segmentation workflows for microCT scans.

This repository is being migrated into an installable `src/` layout so the workflow can be reused from Python code and the command line.

## Quick Start

Install the package in your active environment:

```bash
pip install -e .
```

Run a single scan with the example configuration:

```bash
lacunae-analysis single /path/to/scan.aim --config configs/single_scan_example.yaml --output-dir outputs/sample
```

Run a batch directory with the batch example configuration:

```bash
lacunae-analysis batch /path/to/scan_dir --config configs/batch_example.yaml --output-dir outputs/batch_run
```

For a package-based exploratory workflow, open `lacunae_analysis_workflow_microct.ipynb` or `notebooks/lacunae_analysis_walkthrough.ipynb`.

## Layout

- `src/lacunae_analysis/` — package code
- `configs/` — example YAML inputs for single-scan and batch runs
- `notebooks/` — package-consuming walkthrough notebook
- `tests/` — import and pipeline checks

## Development

Install the project with your preferred Python environment tooling, then run:

```bash
pytest tests/test_imports.py tests/test_pipeline.py -v
```
