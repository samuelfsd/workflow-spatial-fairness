# Datasets

This directory contains spatial datasets used for the spatial fairness workflow.

Each dataset should be placed in its own subdirectory with a consistent structure:

```
datasets/
├── <dataset-name>/
│   ├── data/              # Raw and processed data files
│   ├── metadata.yaml      # Dataset description, variables, and provenance
│   └── README.md          # Dataset-specific documentation
```

## Adding a New Dataset

1. Create a new subdirectory: `datasets/<dataset-name>/`
2. Add raw data to `datasets/<dataset-name>/data/`
3. Create `metadata.yaml` following the template below
4. Update this README with a brief description

### metadata.yaml template

```yaml
name: dataset-name
description: Short description of the dataset
source: Data source URL or reference
spatial_coverage: Geographic extent
temporal_coverage: Time period
files:
  - filename: data/file.csv
    description: What this file contains
    format: csv|shp|geojson|parquet
variables:
  - name: outcome
    description: Target variable
    type: binary|continuous|categorical
  - name: sensitive_attr
    description: Sensitive attribute for fairness analysis
    type: categorical
sensitive_attributes:
  - sensitive_attr
spatial_columns:
  latitude: lat
  longitude: lon
  geometry: geometry  # for geospatial files
```

## Data Notes

- Do not commit raw data files to Git. Use `.gitignore` and store large datasets externally (e.g., Zenodo, Google Drive, institutional server).
- Keep a `data/.gitkeep` if the directory needs to be tracked while data is ignored.
