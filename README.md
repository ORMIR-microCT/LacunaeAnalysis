# Lacunae Analysis

Lacunae Analysis is a small Python package for exploring lacunae segmentation workflows for microCT scans.

This repository is being migrated into an installable `src/` layout so the workflow can be reused from Python code and the command line.

## Quick Start

Create and activate the conda environment first:

```bash
conda create -n lacunae-analysis python=3.10
conda activate lacunae-analysis
```

Install the package in the active environment:

```bash
pip install -e .
```

Run a single scan with the example configuration:

```bash
lacunae-analysis single /path/to/scan.aim --config configs/single_scan_example.yaml --output-dir outputs/sample
```

The single-scan output includes `component_table.csv`, which reports filtered lacuna components with volume, surface area, surface-area-to-volume ratio, centroid, border flag, PCA-based axis radii, orientation vectors, stretch, and oblateness.

Edge-touching lacunae are excluded by default. To retain them, set `include_edge_lacunae: true` under `density_filter` in the YAML config.

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
