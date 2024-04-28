"""K-means clustering toolkit."""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

__all__ = [
    "kmeans_cluster",
    "get_optimal_n_clusters",
]


@dataclass
class _KMeansScores:
    """A dataclass for the scores of k-means clustering.

    Attributes:
        inertia: The inertia of the clustering.
        silhouette: The silhouette score of the clustering.
    """

    inertia: float
    silhouette: float


@dataclass
class _KMeansResult:
    """A dataclass for the result of k-means clustering.

    Attributes:
        n_clusters: The number of clusters.
        labels: The cluster labels.
        centers: The cluster centers.
        scores: The scores of the clustering.
    """

    n_clusters: int
    labels: np.ndarray
    centers: np.ndarray
    scores: _KMeansScores


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
