"""Shared presentation typography for analytical matplotlib figures."""

from __future__ import annotations

from matplotlib.figure import Figure
from matplotlib.text import Text


MIN_PRESENTATION_FONT_SIZE = 12.0
TITLE_FONT_SIZE = 14.0


def apply_presentation_style(figure: Figure) -> Figure:
    """Guarantee the repository's projection-safe minimum text size."""
    for text in figure.findobj(match=Text):
        if text.get_fontsize() < MIN_PRESENTATION_FONT_SIZE:
            text.set_fontsize(MIN_PRESENTATION_FONT_SIZE)
    for axis in figure.axes:
        axis.tick_params(labelsize=MIN_PRESENTATION_FONT_SIZE)
        axis.xaxis.label.set_fontsize(MIN_PRESENTATION_FONT_SIZE)
        axis.yaxis.label.set_fontsize(MIN_PRESENTATION_FONT_SIZE)
        if axis.title.get_text():
            axis.title.set_fontsize(max(axis.title.get_fontsize(), TITLE_FONT_SIZE))
        legend = axis.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(MIN_PRESENTATION_FONT_SIZE)
    return figure
