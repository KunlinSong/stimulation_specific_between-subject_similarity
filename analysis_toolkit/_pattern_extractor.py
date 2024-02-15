#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Kunlin SONG"
__copyright__ = "Copyright (c) 2024 Kunlin SONG"
__license__ = "MIT"
__email__ = "kunlinsongcode@gmail.com"


import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import FastICA

__all__ = [
    "extract_mean_pattern",
]


def extract_mean_pattern(data_lst: list[np.ndarray]) -> np.ndarray:
    """
    Extracts the mean pattern from a list of numpy arrays.

    Args:
        data_lst (list[np.ndarray]): A list of numpy arrays representing
            the data of subjects.

    Returns:
        np.ndarray: The mean pattern of subjects.

    """
    return np.mean(data_lst, axis=0)
