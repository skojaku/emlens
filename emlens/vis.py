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

    :param ax: matplotlib axis
    :type ax: matplotlib.axis
    :param x: x coordinate
    :type x: numpy.array
    :param y: y coordinate
    :type y: numpy.array
    :param labels: Maximum length of labels. If the text exceeds this length, it is automatically folded
    :type labels: numpy.ndarry
    :param color: color of font, defaults to "#4d4d4d"
    :type color: str, optional
    :param label_width: label maximum width, defaults to 30
    :type label_width: int, optional
    :param arrow_shrink: size of arrow, defaults to 5
    :type arrow_shrink: int, optional
    :param text_params: parameter for ax.text, defaults to {}
    :type text_params: dict, optional
    :param adjust_text_params: parameter for adjust_text, defaults to {}
    :type adjust_text_params: dict, optional
    :return: matplotlib axis
    :rtype: matplotlib.ax
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
