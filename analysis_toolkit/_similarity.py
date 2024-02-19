#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Kunlin SONG"
__copyright__ = "Copyright (c) 2024 Kunlin SONG"
__license__ = "MIT"
__email__ = "kunlinsongcode@gmail.com"


from typing import Callable, Literal, NamedTuple, Optional, Union, overload

import numpy as np
import scipy.spatial.distance as ssd
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve
from scipy.stats import pearsonr

__all__ = [
    "cosine_similarity",
    "pearson_correlation_coefficient",
]


"""
  The kernel size and sigma for spatial averaging and local pearson
correlation coefficient."""
KERNEL_SIZE = 3
SIGMA = 1.0

ConvolvePreprocessResult = NamedTuple(
    "ConvolvePreprocessResult",
    [("data", np.ndarray), ("kernel", np.ndarray)],
)

ConvolvePreprocessResultList = NamedTuple(
    "ConvolvePreprocessResultList",
    [("data_lst", list[np.ndarray]), ("kernel", np.ndarray)],
)


@overload
def _convolve_preprocess(
    data: np.ndarray,
    kernel_size: int = 3,
    sigma: float = 1.0,
) -> ConvolvePreprocessResult: ...


@overload
def _convolve_preprocess(
    data_lst: list[np.ndarray],
    kernel_size: int = 3,
    sigma: float = 1.0,
) -> ConvolvePreprocessResultList: ...


def _convolve_preprocess(
    *,
    data: np.ndarray = None,
    data_lst: list[np.ndarray] = None,
    kernel_size: int = 3,
    sigma: float = 1.0,
) -> Union[ConvolvePreprocessResult, ConvolvePreprocessResultList]:
    if output_res := (data is not None):
        x = data.copy()
    elif output_res_lst := (data_lst is not None):
        for data in data_lst:
            assert (
                data.shape == data_lst[0].shape
            ), "The shapes of the arrays must be the same"
        x: np.ndarray = data_lst[0].copy()
    else:
        raise ValueError("Either data or data_lst must be provided")
    kernel = np.zeros((kernel_size,) * x.ndim)
    center = (kernel_size - 1) // 2
    kernel[(center,) * x.ndim] = 1
    kernel = gaussian_filter(kernel, sigma=sigma)

    def process_data(data: np.ndarray) -> np.ndarray:
        data = np.nan_to_num(x=data, copy=True, nan=0.0)
        data = np.pad(data, pad_width=center, mode="constant", constant_values=0)
        return data

    if output_res:
        data = process_data(data)
        return ConvolvePreprocessResult(data=data, kernel=kernel)
    elif output_res_lst:
        data_lst = [process_data(data) for data in data_lst]
        return ConvolvePreprocessResultList(data_lst=data_lst, kernel=kernel)


def calculate_similarity(
    x: np.ndarray, y: np.ndarray, similarity_func: Callable
) -> float:
    """
    Calculates the similarity between two arrays using a given similarity
    function.

    Args:
        x (np.ndarray): The first input array.
        y (np.ndarray): The second input array.
            similarity_func (Callable): The similarity function to use.

    Returns:
        float: The similarity score calculated by the similarity function
            between the two arrays.
    """
    assert x.shape == y.shape, "The shapes of the arrays must be the same"
    x, y = x.copy().flatten(), y.copy().flatten()
    data = np.stack((x, y), axis=0)
    data = data[:, ~np.any(np.isnan(data), axis=0)]
    x, y = data
    return similarity_func(x, y)


def do_fft(x: np.ndarray) -> np.ndarray:
    """
    Perform Fast Fourier Transform on the input array.

    Parameters:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Array containing the real and imaginary parts of the
            FFT result.
    """
    x = np.nan_to_num(x=x, copy=True, nan=0.0)
    x = np.fft.fftshift(np.fft.fftn(x))
    return np.stack((np.real(x), np.imag(x)), axis=-1)


def do_gradient(x: np.ndarray) -> np.ndarray:
    """
    Calculate the gradient of an array along each dimension.

    Parameters:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Array containing the gradient along each dimension.
    """
    gradient_array = np.gradient(x)
    gradient_array.append(x)
    return np.stack(gradient_array, axis=-1)


def do_spatial_average(
    x: np.ndarray, kernel_size: int = 3, sigma: float = 1.0
) -> np.ndarray:
    """
    Perform spatial averaging on the input array.

    Args:
        x (np.ndarray): The input array.
        kernel_size (int, optional): The size of the kernel for
            convolution. Defaults to 3.
        sigma (float, optional): The standard deviation of the Gaussian
            kernel. Defaults to 1.0.

    Returns:
        np.ndarray: The result of spatial averaging.
    """
    res = _convolve_preprocess(data=x, kernel_size=kernel_size, sigma=sigma)
    return convolve(res.data, res.kernel, mode="valid")


PreprocessResult = NamedTuple(
    "PreprocessResult", [("x", np.ndarray), ("y", np.ndarray)]
)


def _preprocess(
    x: np.ndarray,
    y: np.ndarray,
    preprocess_method: Optional[Literal["FFT", "Gradient", "Spatial Average"]] = None,
) -> PreprocessResult:
    if preprocess_method is not None:
        match preprocess_method:
            case "FFT":
                preprocess_func = do_fft
            case "Gradient":
                preprocess_func = do_gradient
            case "Spatial Average":
                preprocess_func = lambda x: do_spatial_average(
                    x=x, kernel_size=KERNEL_SIZE, sigma=SIGMA
                )
            case _:
                raise ValueError("Invalid preprocess method")
        x, y = preprocess_func(x), preprocess_func(y)

    return PreprocessResult(x=x, y=y)


def pearson_correlation_coefficient(
    x: np.ndarray,
    y: np.ndarray,
    preprocess_method: Optional[Literal["FFT", "Gradient", "Spatial Average"]] = None,
) -> float:
    """
    Calculate the Pearson correlation coefficient between two arrays.

    Args:
        x (np.ndarray): The first input array.
        y (np.ndarray): The second input array.
        preprocess_method (Optional[Literal["FFT", "Gradient"]], optional):
            The preprocessing method to be applied to the flattened arrays.
            Defaults to None.

    Returns:
        float: The Pearson correlation coefficient between the two arrays.
    """
    x, y = _preprocess(x, y, preprocess_method)
    pearson_correlation_coefficient_func = lambda x, y: pearsonr(x, y).statistic
    return calculate_similarity(
        x=x, y=y, similarity_func=pearson_correlation_coefficient_func
    )


def cosine_similarity(
    x: np.ndarray,
    y: np.ndarray,
    preprocess_method: Optional[Literal["FFT", "Gradient", "Spatial Average"]] = None,
) -> float:
    """
    Calculate the cosine similarity between two arrays.

    Args:
        x (np.ndarray): The first vector.
        y (np.ndarray): The second vector.
        preprocess_method (Optional[Literal["FFT", "Gradient"]], optional):
            The preprocessing method to be applied to the flattened
            arrays. Defaults to None.

    Returns:
        float: The cosine similarity between the two arrays.
    """
    x, y = _preprocess(x, y, preprocess_method)
    cosine_similarity_func = lambda x, y: 1 - ssd.cosine(x, y)
    return calculate_similarity(x=x, y=y, similarity_func=cosine_similarity_func)
