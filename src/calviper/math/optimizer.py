import itertools
import numba

import numpy as np


class MeanSquaredError:

    def __init__(self, alpha: float = 1e-3):
        self.alpha = alpha

    @staticmethod
    def gradient_(target: np.ndarray, model: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        # cache_ = target, model.conj()
        # numerator_ = np.matmul(cache_, parameter)
        # denominator_ = np.matmul(model * model.conj(), parameter * parameter.conj())
        n_time, n_channel, n_polarization, n_antennas, n_antennas = target.shape
        target = target.reshape(n_time, n_channel, 2, 2, n_antennas, n_antennas)
        model = model.reshape(n_time, n_channel, 2, 2, n_antennas, n_antennas)

        # einsum should work here *only* because we made the  diagonal of the model zeros
        numerator_ = np.einsum('tcpqij,tcqj,tcpqij->tcpi', target, parameter, model.conj())
        denominator_ = np.einsum('tcqj,tcqj,tcpqij,tcpqij->tcpi', parameter, parameter.conj(), model, model.conj())
        gradient_ = (numerator_ / denominator_) - parameter

        #print(f"gradient is {gradient_.shape}")
        #print(f"@-Gradient(Numerator): {numerator_[0, 0, 0, 1]} {numerator_[0, 0, 1, 1]}")
        #print(f"@-Gradient(Denominator): {denominator_[0, 0, 0, 1]} {denominator_[0, 0, 1, 1]}")
        #print(f"@-Gradient(Inner): {gradient_[0, 0, 0, 1]} {gradient_[0, 0, 1, 1]}")

        return gradient_

    @staticmethod
    #@numba.njit
    def gradient(target: np.ndarray, model: np.ndarray, parameter: np.ndarray) -> np.ndarray:

        n_time, n_channel, n_polarization, n_antennas, n_antennas = target.shape
        target = target.reshape(n_time, n_channel, 2, 2, n_antennas, n_antennas)
        model = model.reshape(n_time, n_channel, 2, 2, n_antennas, n_antennas)

        numerator_ = np.zeros((n_time, n_channel, 2, n_antennas), dtype=np.complex64)
        denominator_ = np.zeros((n_time, n_channel, 2, n_antennas), dtype=np.complex64)

        # polarizations per baseline are in the order [XX, XY, YX, YY] I think ... so
        # for p, q in itertools.product([0, 1], [0, 1]):
        for antenna_i in range(n_antennas):
            for antenna_j in range(n_antennas):
                for p in [0, 1]:
                    for q in [0, 1]:
                        if antenna_i == antenna_j:
                            continue

                        numerator_[0, 0, p, antenna_i] += target[0, 0, p, q, antenna_i, antenna_j] * parameter[0, 0, q, antenna_j] * model[0, 0, p, q, antenna_i, antenna_j].conj()
                        denominator_[0, 0, p, antenna_i] += parameter[0, 0, q, antenna_j] * parameter[0, 0, q, antenna_j].conj() * model[0, 0, p, q, antenna_i, antenna_j].conj() * model[0, 0, p, q, antenna_i, antenna_j]

        gradient_ = (numerator_ / denominator_).conj() - parameter
        #print(f"gradient(X): {gradient_[0, 0, 0, 0]}\tparameter: {parameter[0, 0, 0, 0]}")
        #print(f"gradient(Y): {gradient_[0, 0, 1, 0]}\tparameter: {parameter[0, 0, 1, 0]}")

        return gradient_

    @staticmethod
    def loss(y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the mean squared error as defined in https://en.wikipedia.org/wiki/Mean_squared_error
        :param y: Observed values.
        :param y_pred: Predicted values.
        :return: Mean squared error.
        """

        return np.mean(np.power(np.abs(y_pred.flatten()) - np.abs(y.flatten()), 2))

    def step(self, parameter: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        parameter += self.alpha * gradient

        return parameter
