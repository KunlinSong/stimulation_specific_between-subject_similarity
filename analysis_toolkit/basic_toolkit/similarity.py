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
"""A basic toolkit for similarity computation."""


from functools import wraps

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import pearsonr

__all__ = [
    "cosine_similarity",
    "pearson_correlation_coefficient",
]


def _similarity_function(func):
    """A decorator for similarity functions.

    We use this decorator to ensure that the input arrays have the same
    shape, and then flatten them and remove NaN values before passing
    them to the similarity function.

    Args:
        func: A similarity function.

    Returns:
        The wrapped similarity function.

    Raises:
        AssertionError: If the input arrays have different shapes.
    """

    @wraps(func)
    def wrapper(x: np.ndarray, y: np.ndarray, *args, **kwargs):
        assert x.shape == y.shape, (
            "Input arrays must have the same shape, but got "
            f"[x: {x.shape}] and [y: {y.shape}]."
        )
        x, y = x.flatten(), y.flatten()
        xy = np.stack([x, y], axis=0)
        xy = xy[:, ~np.isnan(xy).any(axis=0)]
        x, y = xy
        return func(x, y, *args, **kwargs)

    return wrapper


@_similarity_function
def pearson_correlation_coefficient(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """Compute the Pearson correlation coefficient.

    Args:
        x: An N-dimensional array.
        y: Another N-dimensional array with the same shape as `x`.

    Returns:
        The Pearson correlation coefficient, ranging from -1 to 1.

    Raises:
        AssertionError: If the input arrays have different shapes.
    """
    try:
        return pearsonr(x, y)[0]
    except AssertionError as error:
        raise ValueError("Invalid arrays") from error


@_similarity_function
def cosine_similarity(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """Compute the cosine similarity.

    We use the cosine distance to compute the cosine similarity.  The
    cosine distance is defined as 1 - cosine similarity.

    Args:
        x: An N-dimensional array.
        y: Another N-dimensional array with the same shape as `x`.

    Returns:
        The cosine similarity, ranging from -1 to 1.

    Raises:
        AssertionError: If the input arrays have different shapes.
    """
    return 1 - cosine_distance(x, y)
