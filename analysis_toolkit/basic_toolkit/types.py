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
"""Types for toolkit modules."""


from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generator,
    Literal,
    NamedTuple,
    Optional,
    TypedDict,
    Union,
    overload,
)

import numpy as np

__all__ = []


_Stimulation = str


@dataclass
class _ConfidenceInterval:
    """A dataclass for confidence interval.

    Attributes:
        low: The lower bound of the confidence interval.
        high: The upper bound of the confidence interval.
    """

    low: float
    high: float


@dataclass
class _StarMap:
    """A dataclass for the mapping of stars.

    Attributes:
        name: The name of the starmap.  For example, "one_star".
        level: The level of the stars.
        text: The text of the stars.
        confidence_level: The confidence level of the stars in string
            format.
    """

    name: str
    level: int
    text: str
    confidence_level: str

    def __lt__(self, other: "_StarMap") -> bool:
        return self.level < other.level


@dataclass
class _StarsMapping:
    """A dataclass for the mapping of stars.

    Attributes:
        one_star: The mapping for confidence level 0.95.
        two_stars: The mapping for confidence level 0.99.
        three_stars: The mapping for confidence level 0.999.
        four_stars: The mapping for confidence level 0.9999.
    """

    one_star = _StarMap(
        name="one_star", level=1, text="*", confidence_level="0.95"
    )
    two_stars = _StarMap(
        name="two_stars", level=2, text="**", confidence_level="0.99"
    )
    three_stars = _StarMap(
        name="three_stars", level=3, text="***", confidence_level="0.999"
    )
    four_stars = _StarMap(
        name="four_stars", level=4, text="****", confidence_level="0.9999"
    )


@dataclass
class _ConfidenceIntervalDict:
    """A dataclass for confidence interval dictionary.

    We use n_stars as the key for the confidence interval.

    Attributes:
        one_star: The confidence interval for one star, whose level is
            0.95.
        two_stars: The confidence interval for two stars, whose level is
            0.99.
        three_stars: The confidence interval for three stars, whose level
            is 0.999.
        four_stars: The confidence interval for four stars, whose level
            is 0.9999.
    """

    one_star: _ConfidenceInterval
    two_stars: _ConfidenceInterval
    three_stars: _ConfidenceInterval
    four_stars: _ConfidenceInterval
    _stars_mapping = _StarsMapping()


@dataclass
class _BootstrapTestResult:
    """A dataclass for the result of bootstrap testing.

    Attributes:
        statistic: The value of the statistic.
        distribution: The distribution of the bootstrap resamples.
        confidence_intervals: The confidence intervals for the statistic.
    """

    statistic: float
    distribution: np.ndarray
    confidence_intervals: _ConfidenceIntervalDict


@dataclass
class _KMeansScores:
    """A dataclass for the scores of k-means clustering.

    Attributes:
        inertia: The inertia of the clustering.
        silhouette: The silhouette score of the clustering.
    """

    inertia: float
    silhouette: float


@dataclass
class _KMeansResult:
    """A dataclass for the result of k-means clustering.

    Attributes:
        n_clusters: The number of clusters.
        labels: The cluster labels.
        centers: The cluster centers.
        scores: The scores of the clustering.
    """

    n_clusters: int
    labels: np.ndarray
    centers: np.ndarray
    scores: _KMeansScores


@dataclass
class _EllipseParams:
    """A dataclass for the parameters of an ellipse.

    Attributes:
        center_x: The x-coordinate of the center of the ellipse.
        center_y: The y-coordinate of the center of the ellipse.
        width: Total length (diameter) of the ellipse along the x-axis.
        height: Total length (diameter) of the ellipse along the y-axis.
        angle: The angle in degrees anti-clockwise from the x-axis.
    """

    center_x: float
    center_y: float
    width: float
    height: float
    angle: float


@dataclass
class _AnnotationLine:
    """A dataclass for the line of testing annotation.

    Attributes:
        x: The x-coordinates of the line.
        y: The y-coordinates of the line.
    """

    x: list[float, float, float, float]
    y: list[float, float, float, float]


@dataclass
class _AnnotationText:
    """A dataclass for the text of testing annotation.

    Attributes:
        x: The x-coordinate of the text.
        y: The y-coordinate of the text.
        text: The content of the text. Basically the stars.
    """

    x: float
    y: float
    text: str


@dataclass
class _Annotation:
    """A dataclass for the annotation of testing.

    Attributes:
        line: The line of the annotation.
        text: The text of the annotation.
    """

    line: _AnnotationLine
    text: _AnnotationText
