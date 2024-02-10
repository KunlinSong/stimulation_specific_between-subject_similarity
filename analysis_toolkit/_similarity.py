from typing import Callable, Literal, NamedTuple, Optional

import numpy as np
import scipy.spatial.distance as ssd
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve
from scipy.stats import pearsonr

KERNEL_SIZE = 3
SIGMA = 1.0

ConvolvePreprossResult = NamedTuple(
    "ConvolvePreprossResult",
    [("x", np.ndarray), ("y", Optional[np.ndarray]), ("kernel", np.ndarray)],
)


def _convolve_preprocess(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    kernel_size: int = 3,
    sigma: float = 1.0,
) -> ConvolvePreprossResult:
    if y is not None:
        assert x.shape == y.shape, "The shapes of the arrays must be the same"
    kernel = np.zeros((kernel_size,) * x.ndim)
    center = (kernel_size - 1) // 2
    kernel[(center,) * x.ndim] = 1
    kernel = gaussian_filter(kernel, sigma=sigma)

    def process_data(data: np.ndarray) -> np.ndarray:
        data = np.nan_to_num(x=data, copy=True, nan=0.0)
        data = np.pad(data, pad_width=center, mode="constant", constant_values=0)
        return data

    x = process_data(x)
    if y is not None:
        y = process_data(y)
    return ConvolvePreprossResult(x=x, y=y, kernel=kernel)


def calculate_similarity(
    x: np.ndarray, y: np.ndarray, similarity_func: Callable
) -> float:
    assert x.shape == y.shape, "The shapes of the arrays must be the same"
    x, y = x.copy().flatten(), y.copy().flatten()
    data = np.stack((x, y), axis=0)
    data = data[:, ~np.any(np.isnan(data), axis=0)]
    x, y = data
    return similarity_func(x, y)


def do_fft(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x=x, copy=True, nan=0.0)
    x = np.fft.fftshift(np.fft.fftn(x))
    return np.stack((np.real(x), np.imag(x)), axis=-1)


def do_gradient(x: np.ndarray) -> np.ndarray:
    gradient_array = np.gradient(x)
    return np.stack(gradient_array, axis=-1)


def do_spatial_average(
    x: np.ndarray, kernel_size: int = 3, sigma: float = 1.0
) -> np.ndarray:
    res = _convolve_preprocess(x, kernel_size=kernel_size, sigma=sigma)
    return convolve(res.x, res.kernel, mode="valid")


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
    preprocess_method: Optional[Literal["FFT", "Gradient"]] = None,
) -> float:
    x, y = _preprocess(x, y, preprocess_method)
    pearson_correlation_coefficient_func = lambda x, y: pearsonr(x, y).statistic
    return calculate_similarity(
        x=x, y=y, similarity_func=pearson_correlation_coefficient_func
    )


def cosine_similarity(
    x: np.ndarray,
    y: np.ndarray,
    preprocess_method: Optional[Literal["FFT", "Gradient"]] = None,
) -> float:
    x, y = _preprocess(x, y, preprocess_method)
    cosine_similarity_func = lambda x, y: 1 - ssd.cosine(x, y)
    return calculate_similarity(x=x, y=y, similarity_func=cosine_similarity_func)


def local_pearson_correlation_coefficient(
    x: np.ndarray,
    y: np.ndarray,
    kernel_size: int = 3,
    sigma: float = 1.0,
):
    res = _convolve_preprocess(x, y, kernel_size=kernel_size, sigma=sigma)

    def get_mu(data: np.ndarray) -> np.ndarray:
        return convolve(data, res.kernel, mode="valid")

    mu_x, mu_y = get_mu(res.x), get_mu(res.y)
    var_x = get_mu(res.x**2) - mu_x**2
    var_y = get_mu(res.y**2) - mu_y**2
    cov_xy = get_mu(res.x * res.y) - mu_x * mu_y
    den = np.sqrt(var_x * var_y)
    return cov_xy / (den + np.finfo(den.dtype).eps)
