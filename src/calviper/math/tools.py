import numba

import numpy as np

import toolviper.utils.logger as logger

from typing import Union, List, Any

from sklearn.preprocessing import LabelEncoder


@numba.njit
def to_baseline(array):
    baseline = 0
    n_frequency, n_polarization, n_antenna = array.shape

    n_baseline = int(n_antenna * (n_antenna - 1) * 0.5)
    print(n_antenna)

    n_array = np.zeros((n_baseline, n_frequency, n_polarization), dtype=np.complex64)

    for ant1 in range(n_antenna):
        for ant2 in range(ant1, n_antenna):
            for frequency in range(n_frequency):
                for polarization in range(n_polarization):
                    n_array[baseline, frequency, polarization] = array[frequency, polarization, ant1] * array[
                        frequency, polarization, ant2]

            baseline += 1

    return n_array


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


def encode(antennas: Union[List[str], np.ndarray]) -> tuple[LabelEncoder, Any]:
    """
    Encode the antenna names into a list of unique integers identifiers.
    :param antennas: list of antennas to encode
    :return: (np.ndarray) Array of unique integers identifiers.
    """
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()

    names = np.unique(antennas)

    encoder.fit(names)

    #return encoder.transform(antennas), encoder.classes_
    return encoder, encoder.classes_


def compute_index(ant, chan, pol, shape=None):
    """
        This function provides a way to index the flattened visilibity array, which
        will be needed by the code to build the parameter matrix.
    """
    if shape is None:
        logger.error("Unravel failed because shape is not provided.")
        return None

    n_ant, n_chan, n_pol = shape

    # index = ([antenna-block] ==> antenna * n_channels * n_polarizations) +
    #         ([channel-block] ==> chan * n_pol) +
    #         ([polarization-block] ==> polarization)
    return (ant * n_chan * n_pol) + (chan * n_pol) + pol


@numba.njit()
def build_visibility_matrix_singleton(array: np.ndarray, index_a: np.ndarray, index_b: np.ndarray) -> np.ndarray:
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


@numba.njit()
def build_visibility_matrix(array: np.ndarray, index_a: np.ndarray, index_b: np.ndarray) -> np.ndarray:
    """
    Build a visibility matrix from a visibility array with zeros for autocorrelation.
    :param index_b:
    :param index_a:
    :param array: (numpy.ndarray) visibility array
    :return: (np.ndarray) Visibility matrix of dimension N x N.
    """
    # Get the full array length. This should be the number of baselines, so maybe we could just take that.
    size = index_a.shape[0]

    # Determine the number of unique antennas
    n_antennas = np.union1d(index_a, index_b).shape[0]

    # Dimensions are (n_baselines, n_channel, n_polarization) but we are replacing the first index with n_antenna
    _, n_channel, n_polarization = array.shape

    # Build matrix
    # -- Building a matrix that works better with the regression algorithm. I would really prefer not to
    # -- completely flip the axes but at the C-level, using jit(), it should fine fine for the time being.
    # -- This also allows for effective vectorization in the solver.
    matrix_ = np.zeros((n_channel, n_polarization, n_antennas, n_antennas), dtype=np.complex64)

    # This only works with n_channel = 0, n_polarization=0 at the moment.
    for baseline in range(size):
        for channel in range(n_channel):
            for polarization in range(n_polarization):

                i = index_a[baseline]
                j = index_b[baseline]

                if i == j:
                    continue

                matrix_[channel, polarization, i, j] = array[baseline, channel, polarization]
                matrix_[channel, polarization, j, i] = np.conj(array[baseline, channel, polarization])

    return matrix_


#@numba.njit()
def build_visibility_matrix_full(array: np.ndarray, index_a: np.ndarray, index_b: np.ndarray) -> np.ndarray:
    """
    Build a visibility matrix from a visibility array with zeros for autocorrelation.
    :param index_b:
    :param index_a:
    :param array: (numpy.ndarray) visibility array
    :return: (np.ndarray) Visibility matrix of dimension N x N.
    """
    # Get the full array length
    size = index_a.shape[0]

    # array.shape = (n_baseline, n_channel, n_polarization)
    array_ = array.flatten()

    # Calculate the N X N matrix size needed
    #
    # For the moment this will only work for the simplest case, n_chan=1, n_pol=1
    n_antennas = np.unique(index_a).shape[0]
    dimension = n_antennas * np.prod(array.shape[1:])

    # Dimensions are (n_baselines, n_channel, n_polarization) but we are replacing the first index with n_antenna
    _, n_channel, n_polarization = array.shape

    shape = (n_antennas, n_channel, n_polarization)
    print(f"Shape: {shape}")

    # Build matrix
    matrix_ = np.zeros((dimension, dimension), dtype=np.complex64)

    # This only works with n_channel = 0, n_polarization=0 at the moment.
    for baseline in range(size):
        for channel in range(n_channel):
            for polarization in range(n_polarization):

                i = compute_index(index_a[baseline], channel, polarization, shape=shape)
                j = compute_index(index_b[baseline], channel, polarization, shape=shape)
                print(
                    f"baseline={baseline}\t({index_a[baseline]}, {index_b[baseline]})\tchannel={channel}\tpolarization={polarization}\t(i={i}, j={j})\tvis: {array_[baseline]}")

                if i == j:
                    continue

                matrix_[i, j] = array_[baseline]
                matrix_[j, i] = np.conj(array_[baseline])

    return matrix_
