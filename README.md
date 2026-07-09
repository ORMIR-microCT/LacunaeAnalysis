# Lacunae Analysis

Lacunae Analysis is a small Python package for exploring the lacunae segmentation workflow developed in `Osteobaddies_lacuna_analysis.ipynb`.

This repository is being migrated into an installable `src/` layout so the workflow can be reused from Python code and the command line.

## Layout

- `src/lacunae_analysis/` — package code
- `configs/` — example YAML inputs for single-scan and batch runs
- `tests/` — import and packaging checks

## Development

Install the project with your preferred Python environment tooling, then run:

```bash
pytest tests/test_imports.py -v
```

