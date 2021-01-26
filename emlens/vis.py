import textwrap

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from scipy import sparse


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
    """Add text labels to the points. The position of text will be
    automatically adjusted to avoid overlap.

    Parameters
    ----------
    ax : matplotlib axis
    x : numpy.array
    y : numpy.array
    labels : list of str
        Text labels.
    color : str (Optional; Default "#4d4d4d")
        Hex code for the color of lines
    label_width : int (Optional; Default 30)
        Maximum length of labels. If the text exceeds this length,
        it is automatically folded
    arrow_shring : int (Optional; Default 5)
        Determine the margin betwen point and dashed line.
        This parameter is passed to `arrowprops` of `adjust_text`
    text_param : dict (Optional; Default {})
        parametrs that are passed to `ax.text`
    adjust_text_params : dict (Optional; Default {})
        parametrs that are passed to `adjust_text``

    Returns
    ------
    ax : matplotlib ax
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
