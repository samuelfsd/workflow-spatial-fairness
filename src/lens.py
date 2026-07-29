"""Case-study lenses over the organically discovered clusters.

A lens does NOT define regions — it filters clusters the pipeline already found
to a geographic window, for an illustrative case study (e.g. greater Los Angeles,
backed by redlining literature). It is illustration, never validation.
"""

from __future__ import annotations

import pandas as pd

# lon_min, lat_min, lon_max, lat_max — greater Los Angeles.
GREATER_LA_BBOX = (-118.95, 33.60, -117.60, 34.40)


def clusters_in_bbox(regions: list[dict], df: pd.DataFrame, bbox: tuple[float, float, float, float]) -> list[int]:
    """Return the labels of clusters whose centroid falls inside `bbox`."""
    lon_min, lat_min, lon_max, lat_max = bbox
    selected = []
    for region in regions:
        subset = df.iloc[region["points"]]
        lat = float(subset["lat"].mean())
        lon = float(subset["lon"].mean())
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            selected.append(region["cluster_label"])
    return selected
