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
""""""


from functools import partial

import numpy as np

from analysis_toolkit.basic_toolkit.types import (
    _FFTKwargs,
    _GradientKwargs,
    _PreprocessMethod,
    _SpatialAverageKwargs,
)

from . import toolkit as toolkit


def _preprocess_data(
    data: np.ndarray,
    preprocess_method: _PreprocessMethod,
    preprocess_kwargs: _FFTKwargs | _GradientKwargs | _SpatialAverageKwargs,
) -> np.ndarray:
    valid_methods = ["FFT", "Gradient", "Spatial Average"]
    get_func = lambda func: (
        partial(func, **preprocess_kwargs)
        if (preprocess_kwargs is not None)
        else func
    )
    match preprocess_method:
        case "FFT":
            preprocess_func = get_func(toolkit.data_modifier.fft)
        case "Gradient":
            preprocess_func = get_func(
                toolkit.data_modifier.calculate_gradient
            )
        case "Spatial Average":
            preprocess_func = get_func(toolkit.data_modifier.spatial_average)
        case _:
            raise ValueError(
                f"Precprocess method must be one of {valid_methods}, "
                f"but got {preprocess_method}."
            )
    return preprocess_func(data)


def pearson_correlation_coefficient(
    x: np.ndarray,
    y: np.ndarray,
    *,
    preprocess_method: _PreprocessMethod | None = None,
    preprocess_kwargs: (
        _FFTKwargs | _GradientKwargs | _SpatialAverageKwargs | None
    ) = None,
) -> float:
    preprocess = partial(
        _preprocess_data,
        preprocess_method=preprocess_method,
        preprocess_kwargs=preprocess_kwargs,
    )
    x, y = preprocess(x), preprocess(y)
    return toolkit.similarity.pearson_correlation_coefficient(x, y)


def cosine_similarity(
    x: np.ndarray,
    y: np.ndarray,
    *,
    preprocess_method: _PreprocessMethod | None,
) -> float:
    preprocess = partial(
        _preprocess_data,
        preprocess_method=preprocess_method,
        preprocess_kwargs=None,
    )
    x, y = preprocess(x), preprocess(y)
    return toolkit.similarity.cosine_similarity(x, y)
