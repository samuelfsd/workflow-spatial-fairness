# Spatial Fairness Workflow

> A research workflow for spatial fairness analysis focused on **group fairness bias identification** and **MAUP mitigation** using density-based spatial partitioning.

## Overview

This project provides a workflow to analyze and mitigate spatial biases in machine learning and statistical models.

### Key Objectives

...

## Repository Structure

```
├── datasets/          # Spatial datasets and documentation
├── notebooks/         # Exploratory analysis and visualization notebooks
├── src/               # Source code
│   ├── clustering/    # Density-based spatial partitioning implementations
│   │   ├── dbscan.py
│   │   ├── hdbscan.py
│   │   └── graphs.py
│   ├── metrics/       # Fairness and spatial bias metrics
│   │   ├── group_fairness.py
│   │   ├── spatial_metrics.py
│   │   └── maup.py
│   └── main.py
├── README.md
└── pyproject.toml
```

## How to run

### Prerequisites

- Python >= 3.12
- See `pyproject.toml` for dependencies

### Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management and virtual environments.

```bash
# Sync dependencies and create/update the virtual environment
uv sync
```

### Run Workflow

```bash
# Run using the project's virtual environment
uv run python src/main.py
```

Alternatively, activate the virtual environment first:

```bash
source .venv/bin/activate
python src/main.py
```

## Datasets

See [`datasets/README.md`](datasets/README.md) for detailed descriptions of each dataset.

## Citation

If you use this workflow in your research, please cite this repository.

## License

To be defined.
