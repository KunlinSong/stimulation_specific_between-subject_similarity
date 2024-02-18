#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Kunlin SONG"
__copyright__ = "Copyright (c) 2024 Kunlin SONG"
__license__ = "MIT"
__email__ = "kunlinsongcode@gmail.com"


import numpy as np
from sklearn.decomposition import PCA

__all__ = ["pca_decomposition", "pca_decomposition_2d", "pca_decomposition_3d"]


def pca_decomposition(
    data_dict: dict[str, list[np.ndarray]], n_components: int
) -> tuple[dict[str, list[np.ndarray]], PCA]:
    """
    Perform PCA decomposition on the input data dictionary. The
    decomposition is performed to reduce the data to the specified
    number of components.

    Args:
        data_dict (dict[str, list[np.ndarray]]): A dictionary where the
            keys represent labels and the values are lists of numpy arrays.

    Returns:
        tuple[dict[str, list[np.ndarray]], PCA]: A tuple containing the
            transformed data dictionary and the PCA object.

    Raises:
        None

    Examples:
        >>> data = {
        ...     'label1': [np.array([1, 2, 3]), np.array([4, 5, 6])],
        ...     'label2': [np.array([7, 8, 9]), np.array([10, 11, 12])]
        ... }
        >>> n_components = 2
        >>> transformed_data, pca = pca_decomposition(data, n_components)
    """
    pca = PCA(n_components=n_components)
    v_lst = []
    for v in data_dict.values():
        v_lst.append(np.stack(v, axis=0))
    v = np.concatenate(v_lst, axis=0)
    v = v.reshape(v.shape[0], -1)
    not_nan_indices = ~np.any(np.isnan(v), axis=0)
    v = v[:, not_nan_indices]
    pca.fit(v)
    new_data_dict = {}
    for k, v in data_dict.items():
        new_data = [pca.transform(x.reshape(1, -1)[:, not_nan_indices]) for x in v]
        new_data_dict[k] = new_data
    return new_data_dict, pca


def pca_decomposition_2d(
    data_dict: dict[str, list[np.ndarray]]
) -> tuple[dict[str, list[np.ndarray]], PCA]:
    """
    Perform PCA decomposition on the input data dictionary. The
    decomposition is performed to reduce the data to 2D.

    Args:
        data_dict (dict[str, list[np.ndarray]]): A dictionary where the
            keys represent labels and the values are lists of numpy arrays.

    Returns:
        tuple[dict[str, list[np.ndarray]], PCA]: A tuple containing the
            transformed data dictionary and the PCA object.
    """
    n_components = 2
    return pca_decomposition(data_dict, n_components)


def pca_decomposition_3d(
    data_dict: dict[str, list[np.ndarray]]
) -> tuple[dict[str, list[np.ndarray]], PCA]:
    """
    Perform PCA decomposition on the input data dictionary. The
    decomposition is performed to reduce the data to 3D.

    Args:
        data_dict (dict[str, list[np.ndarray]]): A dictionary where the
            keys represent labels and the values are lists of numpy arrays.

    Returns:
        tuple[dict[str, list[np.ndarray]], PCA]: A tuple containing the
            transformed data dictionary and the PCA object.
    """
    n_components = 3
    return pca_decomposition(data_dict, n_components)
