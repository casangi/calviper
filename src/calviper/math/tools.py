import numba

import numpy as np

from typing import Union, List

def encode(antennas: Union[List[str], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode the antenna names into a list of unique integers identifiers.
    :param antennas: list of antennas to encode
    :return: (np.ndarray) Array of unique integers identifiers.
    """
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()

    names = np.unique(antennas)

    encoder.fit(names)

    return encoder.transform(antennas), encoder.classes_


@numba.njit()
def build_visibility_matrix(array: np.ndarray, index_a: np.ndarray, index_b: np.ndarray) -> np.ndarray:
    """
    Build a visibility matrix from a visibility array with zeros for autocorrelation.
    :param index_b:
    :param index_a:
    :param array: (numpy.ndarray) visibility array
    :return: (np.ndarray) Visibility matrix of dimension N x N.
    """
    # Get the full array length
    size = index_a.shape[0]

    # Calculate the N X N matrix size needed
    dimension = np.unique(index_a).shape[0] + 1

    # Build matrix
    matrix_ = np.zeros((dimension, dimension), dtype=np.complex64)

    for m in range(size):
        i = index_a[m]
        j = index_b[m]

        if i == j:
            continue

        matrix_[i, j] = array[i]
        matrix_[j, i] = np.conj(array[i])

    return matrix_