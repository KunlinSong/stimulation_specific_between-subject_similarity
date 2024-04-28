"""A specific project toolkit for similarity computation."""

from functools import partial
from typing import Callable, Literal, TypedDict

import numpy as np

from ._data_modifier import calculate_gradient, fft, spatial_average
from ._similarity_func import (
    cosine_similarity,
    pearson_correlation_coefficient,
)

__all__ = [
    "cs",
    "pcc",
]
_PreprocessMethod = Literal["FFT", "Gradient", "Spatial Average"]


class _FFTKwargs(TypedDict):
    part: Literal["real", "imaginary", "both"]


class _GradientKwargs(TypedDict):
    axis: int | Literal["x", "y", "z"] | None
    keep_raw: bool


class _SpatialAverageKwargs(TypedDict):
    kernel_size: int
    sigma: float


def _get_preprocess_func(
    name: _PreprocessMethod,
    func_kwargs: _FFTKwargs | _GradientKwargs | _SpatialAverageKwargs | None,
) -> Callable:
    match name:
        case "FFT":
            func = fft
        case "Gradient":
            func = calculate_gradient
        case "Spatial Average":
            func = spatial_average
        case _:
            raise ValueError(f"Invalid preprocess method: {name}")
    if func_kwargs is not None:
        func = partial(func, **func_kwargs)
    return func


def _preprocess(
    x: np.ndarray,
    y: np.ndarray,
    preprocess_method: _PreprocessMethod,
    preprocess_kwargs: (
        _FFTKwargs | _GradientKwargs | _SpatialAverageKwargs | None
    ),
) -> tuple[np.ndarray, np.ndarray]:
    preprocess_func = _get_preprocess_func(
        name=preprocess_method, func_kwargs=preprocess_kwargs
    )
    return preprocess_func(x), preprocess_func(y)


def pcc(
    x: np.ndarray,
    y: np.ndarray,
    preprocess_method: _PreprocessMethod | None = None,
    preprocess_kwargs: (
        _FFTKwargs | _GradientKwargs | _SpatialAverageKwargs | None
    ) = None,
) -> float:
    """Compute the Pearson correlation coefficient between two arrays.

    We use the Pearson correlation coefficient to compute the correlation
    between two arrays.  Before computing the correlation, the input
    arrays can be preprocessed with the specified method and its
    arguments.  The Pearson correlation coefficient ranges from -1 to 1.

    Args:
        x: An N-dimensional array.
        y: Another N-dimensional array with the same shape as x.
        preprocess_method: The method to preprocess the input arrays.
          It can be one of the following: "FFT", "Gradient",
          "Spatial Average", or None.  If None, no preprocessing is
          performed.  Defaults to None.
        preprocess_kwargs: The keyword arguments for the preprocess
          method.  Defaults to None.

    Returns:
        The Pearson correlation coefficient, ranging from -1 to 1.

    Raises:
        AssertionError: If the input arrays have different shapes.
        ValueError: If the preprocess method is invalid.
    """
    if preprocess_method is not None:
        x, y = _preprocess(x, y, preprocess_method, preprocess_kwargs)
    return pearson_correlation_coefficient(x=x, y=y)


def cs(
    x: np.ndarray,
    y: np.ndarray,
    preprocess_method: _PreprocessMethod | None = None,
    preprocess_kwargs: (
        _FFTKwargs | _GradientKwargs | _SpatialAverageKwargs | None
    ) = None,
) -> float:
    """Compute the cosine similarity between two arrays.

    We use the cosine similarity to compute the similarity between two
    arrays.  Before computing the similarity, the input
    arrays can be preprocessed with the specified method and its
    arguments.  The cosine similarity ranges from -1 to 1.

    Args:
        x: An N-dimensional array.
        y: Another N-dimensional array with the same shape as x.
        preprocess_method: The method to preprocess the input arrays.
          It can be one of the following: "FFT", "Gradient",
          "Spatial Average", or None.  If None, no preprocessing is
          performed.  Defaults to None.
        preprocess_kwargs: The keyword arguments for the preprocess
          method.  Defaults to None.

    Returns:
        The cosine similarity, ranging from -1 to 1.

    Raises:
        AssertionError: If the input arrays have different shapes.
        ValueError: If the preprocess method is invalid.
    """
    if preprocess_method is not None:
        x, y = _preprocess(x, y, preprocess_method, preprocess_kwargs)
    return cosine_similarity(x=x, y=y)
