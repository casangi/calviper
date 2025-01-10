import numba

import numpy as np

import toolviper.utils.logger as logger

from typing import Union, List


def ravel(array: np.ndarray):
    """
    Convert parameter array to array of polarization correlation matrices.
    :param array: parameter array
    :return: np.ndarray of polarization correlation matrices
    """

    # Array should be something like (n_times, ..., XX, XY, YX, YY)
    if array.ndim < 2:
        logger.error("Unravel failed because array has too few dimensions.")
        return None

    shape_ = (*array.shape[:-1], 2, 2)

    return array.reshape(shape_)


def unravel(array: np.ndarray) -> Union[np.ndarray, None]:
    """
        Convert array of polarization correlation matrices to parameter array.
        :param array of polarization correlation matrices
        :return: parameter array
        """

    # Array should be something like (n_times, ..., [ [XX, XY],
    #                                                 [YX, YY] ])
    if array.ndim < 3:
        logger.error("Unravel failed because array has too few dimensions.")
        return None

    shape_ = (*array.shape[:-2], 4)

    return array.reshape(shape_)


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
    array_ = array.flatten()

    # Calculate the N X N matrix size needed
    #
    # For the moment this will only work for the simplest case, n_chan=1, n_pol=1

    dimension = np.unique(index_a).shape[0]

    # Build matrix
    matrix_ = np.zeros((dimension, dimension), dtype=np.complex64)

    # This only works with n_channel = 0, n_polarization=0 at the moment.
    for m in range(size):
        i = index_a[m]
        j = index_b[m]

        if i == j:
            continue

        matrix_[i, j] = array_[i]
        matrix_[j, i] = np.conj(array_[i])

    return matrix_
