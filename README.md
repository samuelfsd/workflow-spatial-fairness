# Spatial Fairness Workflow

Research workflow for auditing spatial fairness with the original spatial scan setup and an HDBSCAN-based comparison.

The current pipeline reproduces the authors' experiments with SUL/MeanVar and adds an organic density-based alternative:

```text
Authors: KMeans seeds -> square scan regions -> SUL
This work: HDBSCAN clusters -> organic regions -> SUL
```

## Repository Structure

```text
datasets/
  old/                 # Current CSV inputs: LAR, Crime, Semisynth, Synth_fair, Synth_unfair
src/
  main.py              # CLI entry point
  data_loading.py      # Dataset normalization and default experiment parameters
  experiments.py       # Experiment runner, logging, CSV output
  regions.py           # R-tree, grid, KMeans scan regions, overlap filtering
  visualization.py     # Folium maps with HDBSCAN convex hulls
  clustering/
    hdbscan.py         # HDBSCAN spatial partitioning
  metrics/
    group_fairness.py  # SUL, MeanVar, Monte Carlo thresholding
tests/                 # Unit tests for data loading, metrics, regions, HDBSCAN, maps
```

## Setup

Requires Python 3.12 and `uv`.

```bash
uv sync
```

Run tests:

```bash
uv run python -m unittest discover -s tests
```

## Running Experiments

The CLI has four commands:

```bash
uv run python src/main.py unrestricted --dataset lar --out outputs --no-maps
uv run python src/main.py one-partitioning --dataset crime --out outputs --maps
uv run python src/main.py multiple-partitionings --dataset semisynth --out outputs
uv run python src/main.py all --out outputs --no-maps
```

Use `--maps` to generate Folium HTML maps and `--quiet` to suppress progress logs. For faster exploratory runs, reduce Monte Carlo worlds or HDBSCAN settings:

```bash
uv run python src/main.py unrestricted \
  --dataset crime \
  --n-alt-worlds 20 \
  --hdbscan-fracs 0.001,0.002,0.005 \
  --out outputs \
  --maps
```

## Experiment Modes

- `unrestricted`: reproduces the authors' unrestricted scan with 100 KMeans seeds and dataset-specific square radii, then compares against HDBSCAN clusters evaluated with SUL.
- `one-partitioning`: runs fixed grid partitioning, SUL, MeanVar, Monte Carlo thresholds, and HDBSCAN comparison. Defaults include LAR `100x50` and `25x12`, Crime `20x20`.
- `multiple-partitionings`: generates 100 random rectangular grids with dimensions sampled from `10..40` and reports MeanVar stability.
- `all`: runs the default reproduction suite across the main datasets.

## Outputs

CSV outputs are written per dataset to avoid overwriting previous runs:

```text
outputs/unrestricted_lar_regions.csv
outputs/unrestricted_crime_regions.csv
outputs/hdbscan_lar_comparison.csv
outputs/hdbscan_crime_comparison.csv
outputs/one_partitioning_lar.csv
outputs/multiple_partitionings_semisynth.csv
```

Maps are written under `outputs/maps/`, for example:

```text
outputs/maps/unrestricted_lar.html
outputs/maps/unrestricted_crime.html
```

Key output columns:

- `max_sul`: strongest spatial unfairness likelihood found.
- `signif_threshold`: Monte Carlo threshold for significance.
- `significant_regions`: number of regions or clusters above the threshold.
- `best_region_n`, `best_region_rate`: size and local positive rate of the strongest region.
- `noise_rate`: HDBSCAN-only share of points not assigned to any cluster.

Use `max_sul >= signif_threshold` as the basic significance check.

## Datasets

Current inputs live in `datasets/old/`:

- `LAR.csv`: outcome column `action_taken`, with `3` normalized to `0`.
- `Crime.csv`: outcome column `pred`.
- `Semisynth.csv`, `Synth_fair.csv`, `Synth_unfair.csv`: outcome column `label`.

See [datasets/README.md](datasets/README.md) for dataset organization notes.

## Notes

MeanVar is kept as a baseline from prior work and is most meaningful on non-overlapping partitions. The main comparison for the proposed approach is HDBSCAN-generated regions evaluated with SUL and Monte Carlo significance testing.
