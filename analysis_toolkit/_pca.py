import numpy as np
from sklearn.decomposition import PCA


def pca_decomposition(
    data_dict: dict[str, list[np.ndarray]], n_components: int
) -> dict[str, list[np.ndarray]]:
    """
    Perform PCA decomposition on the input data dictionary. The
    decomposition is performed to reduce the data to the specified
    number of components.

    Args:
        data_dict (dict[str, list[np.ndarray]]): A dictionary where the
            keys represent labels and the values are lists of numpy arrays.

    Returns:
        dict[str, list[np.ndarray]]: A dictionary where the keys
            represent labels and the values are lists of transformed numpy
            arrays.

    """
    pca = PCA(n_components=n_components)
    v_lst = []
    for v in data_dict.values():
        v_lst.append(np.stack(v, axis=0))
    v = np.concatenate(v_lst, axis=0)
    v = v.reshape(v.shape[0], -1)
    pca.fit(v)
    new_data_dict = {}
    for k, v in data_dict.items():
        new_data = [pca.transform(x.reshape(1, -1)) for x in v]
        new_data_dict[k] = new_data
    return new_data_dict


def pca_decomposition_2d(
    data_dict: dict[str, list[np.ndarray]]
) -> dict[str, list[np.ndarray]]:
    """
    Perform PCA decomposition on the input data dictionary. The
    decomposition is performed to reduce the data to 2D.

    Args:
        data_dict (dict[str, list[np.ndarray]]): A dictionary where the
            keys represent labels and the values are lists of numpy arrays.

    Returns:
        dict[str, list[np.ndarray]]: A dictionary where the keys
            represent labels and the values are lists of transformed numpy
            arrays.
    """
    n_components = 2
    return pca_decomposition(data_dict, n_components)


def pca_decomposition_3d(
    data_dict: dict[str, list[np.ndarray]]
) -> dict[str, list[np.ndarray]]:
    """
    Perform PCA decomposition on the input data dictionary. The
    decomposition is performed to reduce the data to 3D.

    Args:
        data_dict (dict[str, list[np.ndarray]]): A dictionary where the
            keys represent labels and the values are lists of numpy arrays.

    Returns:
        dict[str, list[np.ndarray]]: A dictionary where the keys
            represent labels and the values are lists of transformed numpy
            arrays.
    """
    n_components = 3
    return pca_decomposition(data_dict, n_components)
