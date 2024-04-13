"""Module for PCA decompositon"""

import numpy as np
from sklearn.decomposition import PCA as skPCA


class PCA(skPCA):
    """A class for principal components analysis.

    Attributes:
        _not_nan_indices: The indices with not nan values in the data
        fitted to the model.
    """

    def __init__(self, n_components: int):
        """Initializes the instance based on number of components.

        Args:
            n_components: The number of principal components.
        """
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
        """Fits the model to the data.

        Args:
            x: The input data.
        """
        x_flattened = self._flatten_input(x, batch_first=True)
        self._not_nan_indices = ~np.any(np.isnan(x_flattened), axis=0)
        x_not_nan = self._get_not_nan_values(x_flattened)
        super().fit(x_not_nan)

    def transform(self, x: np.ndarray, batch_first: bool = True) -> np.ndarray:
        """Transforms the input data.

        The input data is transformed to the principal components space.
        If the batch_first is True, the input data is assumed to be in
        the form of (n_batch, features, features, ...). Otherwise, the
        input data is assumed to be in the form of (features, features,
        ...). The returned data is in the form of (n_batch, n_components)
        if the batch_first is True, otherwise, the returned data is in
        the form of (1, n_components).

        Args:
            x: The input data.
            batch_first: Whether the input data is in the form of
              (batch, features, features, ...). Defaults to True.

        Returns:
            The transformed data.
        """

        x_flattened = self._flatten_input(x, batch_first=batch_first)
        print(x_flattened.shape)
        x_not_nan = self._get_not_nan_values(x_flattened)
        return super().transform(x_not_nan)
