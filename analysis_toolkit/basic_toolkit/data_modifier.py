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
"""A basic toolkit for data modification."""


import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve

from analysis_toolkit.basic_toolkit.types import Literal

__all__ = [
    "calculate_gradient",
    "fft",
    "spatial_average",
]


def fft(
    data: np.ndarray,
    part: Literal["real", "imaginary", "both"] = "both",
) -> np.ndarray:
    """Computes the fast Fourier transform of the data.

    We use n-dimensional fast Fourier transform (FFT) to compute the
    frequency-domain representation of the data.  The NaN values in the
    data are replaced with 0.0.  The zero-frequency component is shifted
    to the center of the spectrum.  The returned part of the frequency-
    domain representation of the data is determined by the part argument.

    Args:
        data (np.ndarray): The input data.
        part (Literal["real", "imaginary", "both"], optional): The part
            of the returned data.  Defaults to "both".

    Returns:
        The frequency-domain representation of the data.  The returned
        part of the frequency-domain representation of the data is
        determined by the part argument.

    Raises:
        ValueError: If the part argument is invalid.
    """
    data = np.nan_to_num(data, copy=True, nan=0.0)
    data = np.fft.fftshift(np.fft.fftn(data))
    match part:
        case "real":
            return np.real(data)
        case "imaginary":
            return np.imag(data)
        case "both":
            return np.stack([np.real(data), np.imag(data)], axis=-1)
        case _:
            raise ValueError("Invalid part.")


def calculate_gradient(
    data: np.ndarray,
    axis: int | Literal["x", "y", "z"] | None = None,
    keep_raw: bool = False,
) -> np.ndarray:
    """Computes the gradient of the data.

    We use the numpy.gradient function to compute the gradient of the
    data.  The returned gradient of the data is determined by the axis
    argument.  The raw data can be kept in the returned array by setting
    the keep_raw argument to True.

    Args:
        data: The input data.
        axis: The axis along which the gradient is computed.  Defaults to
          None.
        keep_raw: Whether to keep the raw data in the returned array.

    Returns:
        The gradient of the data.  The returned gradient of the data is
        determined by the axis argument.  The raw data can be kept in the
        returned array by setting the keep_raw argument to True.

    Raises:
        ValueError: If the axis argument is invalid.
    """
    gradient_array_lst: list[np.ndarray] = np.gradient(data)
    if isinstance(axis, int):
        gradient_array_lst = [gradient_array_lst[axis]]
    elif isinstance(axis, str):
        if axis not in ["x", "y", "z"]:
            raise ValueError("Invalid axis.")
        axis = ["x", "y", "z"].index(axis)
        gradient_array_lst = [gradient_array_lst[axis]]
    elif axis is not None:
        raise ValueError("Invalid axis.")
    if keep_raw:
        gradient_array_lst.append(data)
    return np.stack(gradient_array_lst, axis=-1)


def spatial_average(
    data: np.ndarray,
    kernel_size: int = 3,
    sigma: float = 1.0,
) -> np.ndarray:
    """Performs spatial averaging on the data.

    We use the scipy.signal.convolve function to perform spatial averaging
    on the data.  The NaN values in the data are replaced with 0.0.  The
    kernel for convolution is distributed according to the Gaussian
    distribution with the given sigma.  We normalize the total weight of
    the kernel to 1.0

    Args:
        data: The input data.
        kernel_size: The size of the kernel for convolution.  Defaults
          to 3.
        sigma: The standard deviation of the Gaussian kernel.  Defaults
          to 1.0.

    Returns:
        The result of spatial averaging.
    """
    kernel = np.zeros((kernel_size,) * data.ndim)
    kernel_center = (kernel_size - 1) // 2
    kernel[(kernel_center,) * data.ndim] = 1
    kernel = gaussian_filter(kernel, sigma=sigma)

    data = np.nan_to_num(data, copy=True, nan=0.0)
    data = np.pad(
        data, pad_width=kernel_center, mode="constant", constant_values=0
    )
    return convolve(data, kernel, mode="valid")
