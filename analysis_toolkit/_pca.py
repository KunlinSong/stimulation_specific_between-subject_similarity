import numpy as np
from sklearn.decomposition import PCA


def pca_decomposition(
    data_dict: dict[str, list[np.ndarray]]
) -> dict[str, list[np.ndarray]]:
    """
    Perform PCA decomposition on the input data dictionary.

    Args:
        data_dict (dict[str, list[np.ndarray]]): A dictionary where the
        keys represent labels and the values are lists of numpy arrays.

    Returns:
        dict[str, list[np.ndarray]]: A dictionary where the keys
        represent labels and the values are lists of transformed numpy
        arrays.

    """
    pca = PCA(n_components=2)
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
