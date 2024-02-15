from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RANDOM_SEED = 42


@dataclass
class Score:
    silhouette_score: float
    inertia: float


@dataclass
class ClusterResult:
    n_clusters: int
    labels: np.ndarray
    centers: np.ndarray
    score: Score


def kmeans_clustering(data: np.ndarray, n_clusters: int) -> ClusterResult:
    """
    Perform k-means clustering on the given data.

    Args:
        data (np.ndarray): The input data for clustering.
        n_clusters (int): The number of clusters to create.

    Returns:
        ClusterResult: An object containing the clustering results,
            including the cluster labels, cluster centers, and clustering
            score.
    """
    data = data.copy()
    data = data.reshape(data.shape[0], -1)
    data = data[:, ~np.any(np.isnan(data), axis=0)]
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init="auto")
    kmeans.fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    inertia = kmeans.inertia_

    if n_clusters == 1 or n_clusters == data.shape[0]:
        silhouette = np.nan
    else:
        silhouette = silhouette_score(data, labels)
    score = Score(silhouette_score=silhouette, inertia=inertia)
    return ClusterResult(
        n_clusters=n_clusters, labels=labels, centers=centers, score=score
    )


def get_optimal_n_clusters(cluster_results_dict: dict[int, ClusterResult]) -> int:
    """
    Calculates the optimal number of clusters based on the silhouette score.

    Args:
        cluster_results_dict (dict[int, ClusterResult]): A dictionary
            containing the cluster results,  where the keys are the number
            of clusters and the values are the corresponding ClusterResult
            objects.

    Returns:
        int: The optimal number of clusters based on the silhouette score.
    """
    max_score = -np.inf
    optimal_n_clusters = 0
    for n_clusters, cluster_result in cluster_results_dict.items():
        if (score := cluster_result.score.silhouette_score) > max_score:
            max_score = score
            optimal_n_clusters = n_clusters
    return optimal_n_clusters
