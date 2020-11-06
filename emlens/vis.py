import numpy as np
from scipy import sparse
from adjustText import adjust_text
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import textwrap


def repel_labels(
    ax,
    x,
    y,
    labels,
    color="#4d4d4d",
    label_width=30,
    arrow_shrink=5,
    text_params={},
    adjust_text_params={},
):
    """
    Add text labels to the points. The position of text will be automatically adjusted to avoid overlap.
    """
    txt_list = []
    for i, label in enumerate(labels):
        txt = ax.text(
            x[i], y[i], textwrap.fill(label, label_width), **text_params, zorder=100
        )
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground="w")])
        txt_list += [txt]

    if "precision" is not adjust_text_params:
        adjust_text_params["precision"] = 0.1

    adjust_text(
        txt_list,
        arrowprops=dict(
            arrowstyle="-",
            linestyle=":",
            shrinkA=1,
            shrinkB=arrow_shrink,
            connectionstyle="arc3",
            color=color,
        ),
        zorder=99,
        **adjust_text_params
    )
    return ax
