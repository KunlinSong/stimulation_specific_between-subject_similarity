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
"""K-means clustering toolkit."""


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from analysis_toolkit.basic_toolkit.types import _KMeansResult, _KMeansScores

__all__ = [
    "kmeans_cluster",
    "get_optimal_n_clusters",
]


def kmeans_cluster(
    data: np.ndarray,
    n_clusters: int,
    n_init: int = "auto",
) -> _KMeansResult:
    """Performs K-means clustering on the data.

    We use the K-means algorithm to cluster the data into n_clusters
    clusters.  The number of initializations is determined by the n_init
    argument.  The random state is fixed to 42.  The returned result
    includes the cluster labels, the cluster centers, and the scores of
    the clustering.

    Args:
        data: The input data with shape (n_samples, n_features, ...).
        n_clusters: The number of clusters.
        n_init: The number of initializations.  Defaults to "auto".

    Returns:
        The result of the K-means clustering, including the cluster
        labels, the cluster centers, and the scores of the clustering.
        The scores include the inertia and the silhouette score.
    """
    data = data.copy()
    n_samples = data.shape[0]
    data = data.reshape(n_samples, -1)
    data = data[:, ~np.isnan(data).any(axis=0)]
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42).fit(
        data
    )

    if n_clusters == 1 or n_clusters == n_samples:
        silhouette = np.nan
    else:
        silhouette = silhouette_score(data, kmeans.labels_)
    scores = _KMeansScores(inertia=kmeans.inertia_, silhouette=silhouette)
    return _KMeansResult(
        n_clusters=n_clusters,
        labels=kmeans.labels_,
        centers=kmeans.cluster_centers_,
        scores=scores,
    )


def get_optimal_n_clusters(results: list[_KMeansResult]) -> int:
    """Returns the optimal number of clusters based on the silhouette
    score.

    Args:
        results: The results of K-means clustering.

    Returns:
        The optimal number of clusters based on the silhouette score.

    Raises:
        ValueError: If no valid silhouette score is found.
    """
    results = [
        result for result in results if not np.isnan(result.scores.silhouette)
    ]
    if not results:
        raise ValueError("No valid silhouette score.")
    optimal_result = max(results, key=lambda result: result.scores.silhouette)
    return optimal_result.n_clusters
