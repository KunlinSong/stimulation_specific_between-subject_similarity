#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Kunlin SONG"
__copyright__ = "Copyright (c) 2024 Kunlin SONG"
__license__ = "MIT"
__email__ = "kunlinsongcode@gmail.com"


from typing import Callable, Literal, NamedTuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D, art3d
from scipy.stats import chi2

from ._test import ConfidenceInterval

__all__ = [
    "plot_pca_2d",
    "plot_pca_3d",
    "plot_similarity_heatmap",
    "plot_distribution_barplot",
    "plot_clustering_scores",
    "plot_brain",
]

CMAP = "rainbow"
HEATMAP_CMAP = "seismic"
LEGEND_PARAMS = {
    "loc": "upper right",
}

DataInfo = NamedTuple(
    "DataInfo",
    [
        ("min", float),
        ("max", float),
        ("diff", float),
    ],
)


def _get_data_info(data_dict: dict[str, list[np.ndarray]], axis: int) -> DataInfo:
    """
    Get the minimum, maximum, and difference between the maximum and
    minimum values of the data along the specified axis.

    Args:
        data_dict (dict[str, list[np.ndarray]]): A dictionary where the
            keys represent labels and the values are lists of numpy arrays.
        axis (int): The axis along which to get the data info.

    Returns:
        DataInfo: A named tuple containing the minimum, maximum, and
            difference between the maximum and minimum values of the data
            along the specified axis.
    """
    min_val = np.inf
    max_val = -np.inf
    for v in data_dict.values():
        v = np.concatenate(v, axis=0)
        min_val = min(min_val, np.min(v[:, axis]))
        max_val = max(max_val, np.max(v[:, axis]))
    diff = max_val - min_val
    return DataInfo(min_val, max_val, diff)


def _get_ellipse(data: np.ndarray, confidence_level: float) -> Ellipse:
    """
    Get the ellipse for the given data and confidence level.

    Args:
        data (np.ndarray): A 2D numpy array representing the data.
        confidence_level (float): The confidence level.

    Returns:
        Ellipse: An ellipse object.
    """
    cov = np.cov(data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * np.sqrt(chi2.ppf(confidence_level, 2) * eigenvalues)
    return Ellipse(xy=np.mean(data, axis=0), width=width, height=height, angle=angle)


def get_lim(data_info: DataInfo) -> tuple[float, float]:
    FACTOR = 0.618
    diff = data_info.diff * FACTOR
    return data_info.min - diff, data_info.max + diff


def plot_pca_2d(
    data_dict: dict[str, list[np.ndarray]],
    title: str,
    ax: plt.Axes,
    confidence_level: Optional[float] = None,
    legend: bool = False,
    ticks: bool = False,
) -> plt.Axes:
    """
    Plot a 2D scatter plot of the data points projected onto the first
    two principal components (PC1 and PC2).

    Parameters:
        data_dict (dict[str, list[np.ndarray]]): A dictionary containing
            the data points for each category.
            The keys represent the category labels, and the values are
            lists of numpy arrays representing the data points.
        title (str): The title of the plot.
        ax (plt.Axes): The matplotlib Axes object to plot on.
        confidence_level (Optional[float], optional): The confidence
        level for plotting confidence ellipses around the data points.
            If None, no confidence ellipses will be plotted. Defaults to None.
        legend (bool, optional): Whether to show the legend. Defaults to False.
        ticks (bool, optional): Whether to show the ticks on the
            x and y axes. Defaults to False.

    Returns:
        plt.Axes: The matplotlib Axes object with the plot.

    """
    colors = plt.cm.get_cmap(CMAP, len(data_dict))
    x_info = _get_data_info(data_dict, 0)
    y_info = _get_data_info(data_dict, 1)
    x_lim = get_lim(x_info)
    y_lim = get_lim(y_info)

    for color_idx, (k, v) in enumerate(data_dict.items()):
        v = np.concatenate(v, axis=0)
        ax.scatter(v[:, 0], v[:, 1], c=colors(color_idx), label=k)

        if confidence_level is not None:
            ellipse = _get_ellipse(v, confidence_level)
            ellipse.set_edgecolor(colors(color_idx))
            ellipse.set_facecolor("none")
            ax.add_artist(ellipse)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    if legend:
        ax.legend(**LEGEND_PARAMS)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    return ax


def plot_pca_3d(
    data_dict: dict[str, list[np.ndarray]],
    title: str,
    ax: Axes3D,
    confidence_level: Optional[float] = None,
    legend: bool = False,
    ticks: bool = False,
) -> plt.Axes:
    """
    Plot a 3D scatter plot of PCA data.

    Args:
        data_dict (dict[str, list[np.ndarray]]): A dictionary containing
            the data for each category.
            The keys represent the category names, and the values are
            lists of numpy arrays representing the data points.
        title (str): The title of the plot.
        ax (Axes3D): The 3D axes object to plot on.
        confidence_level (Optional[float], optional): The confidence
            level for plotting ellipses. Defaults to None.
        legend (bool, optional): Whether to show the legend. Defaults to
            False.
        ticks (bool, optional): Whether to show the ticks on the axes.
            Defaults to False.

    Returns:
        plt.Axes: The plotted axes object.
    """
    colors = plt.cm.get_cmap(CMAP, len(data_dict))
    x_info = _get_data_info(data_dict, 0)
    y_info = _get_data_info(data_dict, 1)
    z_info = _get_data_info(data_dict, 2)
    x_lim = get_lim(x_info)
    y_lim = get_lim(y_info)
    z_lim = get_lim(z_info)
    for color_idx, (k, v) in enumerate(data_dict.items()):
        v = np.concatenate(v, axis=0)
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], c=colors(color_idx), label=k)
        if confidence_level is not None:
            for axis in ("x", "y", "z"):
                match axis:
                    case "x":
                        dims = [1, 2]
                        z_pos = x_lim[0]
                    case "y":
                        dims = [0, 2]
                        z_pos = y_lim[1]
                    case "z":
                        dims = [0, 1]
                        z_pos = z_lim[0]
                data = v[:, dims]
                ellipse = _get_ellipse(data, confidence_level)
                ellipse.set_edgecolor(colors(color_idx))
                ellipse.set_facecolor("none")
                ax.add_artist(ellipse)
                art3d.patch_2d_to_3d(ellipse, z=z_pos, zdir=axis)

    if legend:
        ax.legend(**LEGEND_PARAMS)
    if not ticks:
        ax.set_xticklabels(
            ["" for _ in range(len(ax.get_xticks()))],
        )
        ax.set_yticklabels(
            ["" for _ in range(len(ax.get_yticks()))],
        )
        ax.set_zticklabels(
            ["" for _ in range(len(ax.get_zticks()))],
        )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    return ax


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    title: str,
    ax: plt.Axes,
    slices_dict: dict[str, slice] | None = None,
    split_x: bool = False,
    split_y: bool = False,
    v_lim: tuple[float, float] = (-1, 1),
    cbar: bool = False,
):
    """
    Plot a heatmap of similarity matrix.

    Args:
        similarity_matrix (np.ndarray): The similarity matrix to be
            plotted.
        title (str): The title of the plot.
        ax (plt.Axes): The axes object to plot the heatmap on.
        slices_dict (dict[str, slice] | None, optional): A dictionary
            mapping labels to slices of the matrix.
            Defaults to None.
        split_x (bool, optional): Whether to split the x-axis. Defaults
            to False.
        split_y (bool, optional): Whether to split the y-axis. Defaults
            to False.
        v_lim (tuple[float, float], optional): The limits of the colorbar.
            Defaults to (-1, 1).
        cbar (bool, optional): Whether to show the colorbar. Defaults to
            False.

    Returns:
        plt.Axes: The axes object with the heatmap plotted.
    """
    sns.heatmap(
        data=similarity_matrix,
        cmap=HEATMAP_CMAP,
        ax=ax,
        cbar=cbar,
        vmin=v_lim[0],
        vmax=v_lim[1],
        square=True,
    )
    ax.set_title(title)
    if slices_dict is not None:
        splits = []
        for label_slice in slices_dict.values():
            splits.append(label_slice.start)
            splits.append(label_slice.stop)
        splits = list(set(splits))
        splits.remove(0)
        splits.remove(len(similarity_matrix))
        labels = []
        ticks = []
        for label, label_slice in slices_dict.items():
            labels.append(label)
            ticks.append((label_slice.start + label_slice.stop) / 2)
        if split_x:
            for split in splits:
                ax.axvline(split, color="black", linewidth=1)
            ax.set_xticks(ticks, labels)
        if split_y:
            for split in splits:
                ax.axhline(split, color="black", linewidth=1)
            ax.set_yticks(ticks, labels)
    return ax


_Label = str
_Statistic = float
_Distribution = Optional[np.ndarray]
_ConfidenceInterval = Optional[dict[Literal["*", "**", "***"], ConfidenceInterval]]

_ShowLevel = int
_Group1 = int
_Group2 = int


def _plot_bar(
    statistic_dict: dict[int, tuple[_Label, _Statistic]], ax: plt.Axes, ncolors: int
) -> tuple[plt.Axes, list]:
    colors = plt.cm.get_cmap(CMAP, ncolors)
    bars = []
    for i, (k, v) in enumerate(statistic_dict.items()):
        color = colors(i % ncolors)
        bar = ax.bar(k, v[1], width=1, color=color, label=v[0])
        if i // ncolors == 0:
            bars.append(bar)
    return ax, bars


def _plot_distribution(
    distribution_dict: dict[int, _Distribution], ax: plt.Axes, ncolors: int
) -> plt.Axes:
    RANDOM_SEED = 42
    DIFF = 0.4
    random_generator = np.random.default_rng(RANDOM_SEED)
    colors = plt.cm.get_cmap(CMAP, ncolors)
    for i, (k, v) in enumerate(distribution_dict.items()):
        color = colors(i % ncolors)
        bias = random_generator.uniform(-DIFF, DIFF, size=v.size)
        ax.scatter(k + bias, v, color=color, s=1, alpha=0.2)
    return ax


def _plot_confidence_interval(
    confidence_interval_dict: dict[int, tuple[_Statistic, _ConfidenceInterval]],
    ax: plt.Axes,
    show_confidence_interval: list[Literal["*", "**", "***"]],
) -> plt.Axes:
    def process_confidence_interval(
        statistic: float, confidence_interval: ConfidenceInterval
    ) -> tuple[float, float]:
        return statistic - confidence_interval.low, confidence_interval.high - statistic

    for k, v in confidence_interval_dict.items():
        for confidence_level in v[1].keys():
            if confidence_level in show_confidence_interval:
                low, high = process_confidence_interval(v[0], v[1][confidence_level])
                match confidence_level:
                    case "*":
                        color = "black"
                        factor = 3
                    case "**":
                        color = "black"
                        factor = 2
                    case "***":
                        color = "red"
                        factor = 1
                    case _:
                        raise ValueError("Invalid confidence level")

                ax.errorbar(
                    k,
                    v[0],
                    yerr=[[low], [high]],
                    color=color,
                    capsize=1 * (0.618**factor),
                )
    return ax


def _get_sorted_levels(
    confidence_levels: list[Literal["*", "**", "***"]]
) -> list[Literal["*", "**", "***"]]:
    sorted_levels = sorted(confidence_levels, key=lambda x: x.count("*"), reverse=True)
    return sorted_levels


def _plot_group_difference(
    data_diff: float,
    confidence_interval_dict: dict[int, tuple[_Statistic, _ConfidenceInterval]],
    show_confidence_interval: Optional[list[Literal["*", "**", "***"]]],
    show_group_difference: list[tuple[_Group1, _Group2, _ShowLevel]],
    ax: plt.Axes,
) -> plt.Axes:
    diff = data_diff * 0.025
    level_diff = data_diff * 0.05

    def _get_group_pos_function() -> Callable:
        if show_confidence_interval is None:
            get_group_pos = lambda idx: confidence_interval_dict[idx][0]
        else:
            sorted_levels = _get_sorted_levels(show_confidence_interval)
            maximum_level = sorted_levels[0]
            get_group_pos = lambda idx: confidence_interval_dict[idx][1][maximum_level]
        return get_group_pos

    def _get_difference_label(group_1: int, group_2: int) -> str:
        confidence_intervals_1 = confidence_interval_dict[group_1][1]
        confidence_intervals_2 = confidence_interval_dict[group_2][1]
        levels_1 = confidence_intervals_1.keys()
        levels_2 = confidence_intervals_2.keys()
        common_levels = list(set(levels_1).intersection(set(levels_2)))
        for level in _get_sorted_levels(common_levels):
            level_interval_1 = confidence_intervals_1[level]
            level_interval_2 = confidence_intervals_2[level]
            if (level_interval_1.low > level_interval_2.high) or (
                level_interval_2.low > level_interval_1.high
            ):
                return level
        return "NS"

    get_group_pos = _get_group_pos_function()
    for group_1, group_2, show_level in show_group_difference:
        pos_1 = get_group_pos(group_1) + diff
        pos_2 = get_group_pos(group_2) + diff
        group_inside_lst = [
            group
            for group in confidence_interval_dict.keys()
            if group_1 <= group <= group_2
        ]
        group_positions = [get_group_pos(group) for group in group_inside_lst]
        pos_top = max(group_positions) + diff + level_diff * show_level
        ax.plot(
            [group_1, group_1, group_2, group_2],
            [pos_1, pos_top, pos_top, pos_2],
            color="black",
        )
        ax.text(
            (group_1 + group_2) / 2,
            pos_top,
            _get_difference_label(group_1, group_2),
            ha="center",
            va="bottom",
            color="black",
        )


def plot_distribution_barplot(
    data_dict: dict[int, tuple[_Label, _Statistic, _Distribution, _ConfidenceInterval]],
    ncolors: int,
    title: str,
    ax: plt.Axes,
    legend: bool = False,
    show_labels: Optional[dict[float, str]] = None,
    show_distribution: bool = False,
    show_confidence_interval: Optional[list[float]] = None,
    show_group_difference: Optional[list[tuple[_Group1, _Group2, _ShowLevel]]] = None,
) -> plt.Axes:
    """
    Plot a distribution barplot.

    Args:
        data_dict (dict[int, tuple[_Label, _Statistic, _Distribution, _ConfidenceInterval]]):
            A dictionary containing the data for each group.
            The keys are integers representing the group IDs.
            The values are tuples containing the label, statistic,
            distribution, and confidence interval for each group.
        ncolors (int): The number of colors to use for the bars.
        title (str): The title of the plot.
        ax (plt.Axes): The matplotlib Axes object to plot on.
        legend (bool, optional): Whether to show the legend. Defaults to
            False.
        show_labels (Optional[dict[float, str]], optional): A dictionary
            mapping x-axis tick positions to labels.
            Defaults to None.
        show_distribution (bool, optional): Whether to show the
            distribution plot. Defaults to False.
        show_confidence_interval (Optional[list[float]], optional): A
            list of confidence interval levels to show.
            Defaults to None.
        show_group_difference (Optional[list[tuple[_Group1, _Group2, _ShowLevel]]], optional):
            A list of tuples specifying the groups to compare and the
            level of difference to show.
            Defaults to None.

    Returns:
        plt.Axes: The matplotlib Axes object with the plot.

    """
    statistic_dict = {k: (v[0], v[1]) for k, v in data_dict.items()}
    ax, bars = _plot_bar(statistic_dict, ax, ncolors)
    if show_distribution:
        distribution_dict = {k: v[2] for k, v in data_dict.items()}
        ax = _plot_distribution(distribution_dict, ax, ncolors)
    if show_confidence_interval is not None:
        confidence_interval_dict = {k: (v[1], v[3]) for k, v in data_dict.items()}
        ax = _plot_confidence_interval(
            confidence_interval_dict, ax, show_confidence_interval
        )
    if show_group_difference is not None:
        data_diff = max(statistic_dict.values(), key=lambda x: x[1])[1]
        confidence_interval_dict = {k: (v[1], v[3]) for k, v in data_dict.items()}
        ax = _plot_group_difference(
            data_diff,
            confidence_interval_dict,
            show_confidence_interval,
            show_group_difference,
            ax,
        )
    if show_labels is not None:
        ticks, labels = zip(*show_labels.items())
        ax.set_xticks(ticks, labels)
    if legend:
        ax.legend(bars, **LEGEND_PARAMS)
    ax.set_title(title)
    return ax


_NClusteringGroups = int
_ClusteringScore = float


def plot_clustering_scores(
    data_dict: dict[str, dict[_NClusteringGroups, _ClusteringScore]],
    title: str,
    score_name: str,
    ax: plt.Axes,
) -> plt.Axes:
    """
    Plot clustering scores.

    Args:
        data_dict (dict[str, dict[_NClusteringGroups, _ClusteringScore]]):
            A dictionary containing clustering scores for different labels.
        title (str): The title of the plot.
        score_name (str): The name of the clustering score.
        ax (plt.Axes): The matplotlib Axes object to plot on.

    Returns:
        plt.Axes: The matplotlib Axes object with the plot.

    """
    colors = plt.cm.get_cmap(CMAP, len(data_dict))
    for i, (label, scores_dict) in enumerate(data_dict.items()):
        x, y = zip(*scores_dict.items())
        ax.plot(x, y, color=colors(i), label=label)
    ax.set_title(title)
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel(score_name)
    ax.legend(**LEGEND_PARAMS)
    return ax


def plot_brain(
    data: np.ndarray,
    axis: Literal["x", "y", "z"],
    idx: int,
    ax: plt.Axes,
    title: str,
    roi_mask: Optional[np.ndarray] = None,
) -> plt.Axes:
    """
    Plot a brain image along a specific axis.

    Args:
        data (np.ndarray): The 3D brain image data.
        axis (Literal["x", "y", "z"]): The axis along which to plot the
            brain image.
        idx (int): The index of the slice along the specified axis.
        ax (plt.Axes): The matplotlib Axes object to plot the image on.
        title (str): The title of the plot.
        roi_mask (Optional[np.ndarray], optional): The binary mask
            indicating regions of interest. Defaults to None.

    Returns:
        plt.Axes: The matplotlib Axes object with the plotted image.
    """
    CMAP_ROI = LinearSegmentedColormap.from_list(
        "Red-Black-Blue", ["lightsteelblue", "blue", "black", "red", "lightcoral"]
    )
    CMAP_BASIC = "gray"
    PERCENTILE = 99.9
    assert data.ndim == 3, "The input data must be 3D"
    persentile_high = np.nanpercentile(data, PERCENTILE)
    persentile_low = np.nanpercentile(data, 100 - PERCENTILE)
    v_lim = max(abs(persentile_low), abs(persentile_high))
    v_kwargs = {"vmin": -v_lim, "vmax": v_lim}
    match axis:
        case "x":
            data = data[idx, :, :].copy()
            if roi_mask is not None:
                roi_mask = roi_mask[idx, :, :].copy()
        case "y":
            data = data[:, idx, :].copy()
            if roi_mask is not None:
                roi_mask = roi_mask[:, idx, :].copy()
        case "z":
            data = data[:, :, idx].copy()
            if roi_mask is not None:
                roi_mask = roi_mask[:, :, idx].copy()
        case _:
            raise ValueError("Invalid axis")
    ax.imshow(data, cmap=CMAP_ROI, **v_kwargs)
    if roi_mask is not None:
        data[roi_mask] = np.nan
        ax.imshow(data, cmap=CMAP_BASIC, **v_kwargs)
    ax.set_title(title)
    ax.axis("off")
    return ax
