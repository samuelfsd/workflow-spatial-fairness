"""The project's single color language, shared by folium maps and matplotlib figures.

Two palettes, two jobs, never mixed:

- **Status (detection class)** — 🔴 negative / 🟢 positive / ⚪ nothing detected.
  Reserved: these three never encode anything else, and they only ever describe a
  *cluster* after metrics + Monte Carlo (see `CONTEXT.md`). Because they are status
  colors they always ship with a legend naming the class, never color alone.
- **Categorical** — identity slots, assigned in **fixed order, never cycled**; the
  ordering is itself the CVD-safety mechanism, so slots are taken from the front,
  not chosen by taste. Notably this is what balance charts use: positives vs
  negatives must NOT be drawn red/green, or point outcome would collide with the
  detection class of a cluster.
"""

from __future__ import annotations

# Status palette — the detection class (fixed by CONTEXT.md, asserted in tests).
COLOR_NEGATIVE = "#d03b3b"
COLOR_POSITIVE = "#0ca30c"
COLOR_NEUTRAL = "#bdbdbd"
COLOR_NOISE = "#999999"
COLOR_NOT_EVALUATED = "#333333"

DETECTION_COLORS = {
    "negative": COLOR_NEGATIVE,
    "positive": COLOR_POSITIVE,
    "neutral": COLOR_NEUTRAL,
}

DETECTION_LABELS = {
    "negative": "injustiça negativa (significativa)",
    "positive": "injustiça positiva (significativa)",
    "neutral": "nada detectado",
}

# Categorical palette — fixed order; take slots from the front.
CATEGORICAL = (
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
)

# Raw-outcome point colors on the stage-1 map: this is the raw data layer, where
# no metric has run yet, and the doc explains the distinction inline.
POINT_POSITIVE = "#2ca25f"
POINT_NEGATIVE = "#de2d26"

# Chart ink and surface (recessive grid/axes, text never wears a series color).
SURFACE = "#ffffff"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#6b6a66"
INK_MUTED = "#898781"
GRID = "#e4e3de"
