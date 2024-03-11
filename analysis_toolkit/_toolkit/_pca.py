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

"""Module for PCA decompositon"""

import numpy as np
from sklearn.decomposition import PCA as skPCA

from ._types import Literal


class PCA(skPCA):

    def __init__(self, n_components: int):
        super().__init__(n_components=n_components)
        self._not_nan_indices = None

    @staticmethod
    def _flatten_input(x: np.ndarray, batch_first: bool = False) -> np.ndarray:
        if batch_first:
            if x.ndim == 2:
                return x
            else:
                return x.reshape(x.shape[0], -1)
        else:
            return x.reshape(1, -1)

    def _get_not_nan_values(self, x: np.ndarray) -> np.ndarray:
        if self._not_nan_indices is None:
            raise ValueError("The model has not been fitted yet.")
        return x[:, self._not_nan_indices]

    def fit(self, x: np.ndarray):
        x_flattened = self._flatten_input(x, batch_first=True)
        self._not_nan_indices = ~np.any(np.isnan(x_flattened), axis=0)
        x_not_nan = self._get_not_nan_values(x_flattened)
        super().fit(x_not_nan)

    def transform(self, x: np.ndarray, batch_first: bool = True) -> np.ndarray:
        x_flattened = self._flatten_input(x, batch_first=batch_first)
        print(x_flattened.shape)
        x_not_nan = self._get_not_nan_values(x_flattened)
        return super().transform(x_not_nan)
