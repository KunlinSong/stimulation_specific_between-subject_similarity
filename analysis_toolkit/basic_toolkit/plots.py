# MIT License
#
# Copyright (c) 2024 Kunlin SONG
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""A toolkit for plotting."""

from dataclasses import fields
from statistics import mean

import numpy as np
from scipy.stats import chi2
from sklearn.preprocessing import minmax_scale

from analysis_toolkit.basic_toolkit.types import (
    _Annotation,
    _AnnotationLine,
    _AnnotationText,
    _BootstrapTestResult,
    _ConfidenceInterval,
    _EllipseParams,
    _StarMap,
    _StarsMapping,
)

__all__ = [
    "confidence_ellipse",
    "min_max_scale",
]


def confidence_ellipse(
    x: np.ndarray, y: np.ndarray, confidence_level: float = 0.95
) -> _EllipseParams:
    """Computes the confidence ellipse of the data.

    We use the chi-squared distribution to compute the confidence ellipse
    of the data.  The confidence level is the probability that the
    confidence ellipse contains the true mean.  The returned ellipse
    parameters include the center, width, height, and angle in degrees.

    Args:
        x: The x-coordinates of the data.
        y: The y-coordinates of the data.
        confidence_level: The confidence level.  Defaults to 0.95.

    Returns:
        The ellipse parameters, including the center, width, height, and
        angle in degrees.
    """
    chi2_value = chi2.ppf(confidence_level, 2)
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * np.sqrt(chi2_value * eigenvalues)
    return _EllipseParams(
        x=x.mean(),
        y=y.mean(),
        width=width,
        height=height,
        angle=angle,
    )


def min_max_scale(
    data: np.ndarray, feature_range: tuple[float, float] = (-1.0, 1.0)
) -> np.ndarray:
    """Scales the data using min-max scaling.

    Args:
        data: The input data, with shape (n_samples, n_features).
        feature_range: The feature range.  Defaults to (-1., 1.).

    Returns:
        The scaled data, with shape (n_samples, n_features).
    """
    return minmax_scale(data, feature_range=feature_range, axis=0)


def annotate_test(
    x: list[float, float],
    test_results: list[_BootstrapTestResult, _BootstrapTestResult],
    inner_maximal_y: float,
    offset: float = 0.05,
    line_height: float = 0.01,
) -> _Annotation:
    """Annotates the test results.

    We use the stars mapping to annotate the test results.  The stars of
    highest significant level will be returned as the text.  If it is not
    significant, "NS" will be returned.  The returned annotation includes
    the line and the text for matplotlib plotting.

    Args:
        x: The 2 x-coordinates.
        test_results: The 2 test results.
        inner_maximal_y: The maximal y-coordinate between the 2
            x-coordinates, including the 2 x-coordinates.
        offset: The offset between the line start and the inner maximal y.
        line_height: The height of the 2 lines at the end of the line.
            Defaults to 0.01.


    Returns:
        The annotation for the test results.
    """
    text_x = mean(x)
    x_1, x_2 = x
    line_x = [x_1, x_1, x_2, x_2]
    line_start_y = inner_maximal_y + offset
    text_y = line_start_y + line_height
    line_y = [line_start_y, text_y, text_y, line_start_y]

    res_1, res_2 = test_results
    stars_mapping: list[_StarMap] = [
        getattr(_StarsMapping(), field.name) for field in fields(_StarsMapping)
    ]
    stars_mapping = sorted(stars_mapping, reverse=True)
    text = "NS"
    for star_map in stars_mapping:
        star_res_1: _ConfidenceInterval = getattr(
            res_1.confidence_intervals, star_map.name
        )
        star_res_2: _ConfidenceInterval = getattr(
            res_2.confidence_intervals, star_map.name
        )
        if star_res_1.low > star_res_2.high:
            text = star_map.text
            break
        if star_res_2.low > star_res_1.high:
            text = star_map.text
            break
    return _Annotation(
        line=_AnnotationLine(x=line_x, y=line_y),
        text=_AnnotationText(x=text_x, y=text_y, text=text),
    )
