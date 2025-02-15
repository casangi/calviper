import numpy as np


class MeanSquaredError:

    def __init__(self, alpha: float = 1e-3):
        self.alpha = alpha

    @staticmethod
    def gradient(target: np.ndarray, model: np.ndarray, parameter: np.ndarray) -> np.ndarray:
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
    def loss(y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the mean squared error as defined in https://en.wikipedia.org/wiki/Mean_squared_error
        :param y: Observed values.
        :param y_pred: Predicted values.
        :return: Mean squared error.
        """
        #square_difference = np.power(np.abs(y_pred) - np.abs(y), 2)
        #for p in range(square_difference.shape[2]):
        #    for ant1 in range(square_difference.shape[3]):
        #        for ant2 in range(square_difference.shape[4]):
        #            print(f"polarization: {p}, ant1: {ant1}, ant2: {ant2}: value: {square_difference[0, 0, p, ant1, ant2]}")
        #print(y_pred[0, 0, 1, :])

        return np.mean(np.power(np.abs(y_pred.flatten()) - np.abs(y.flatten()), 2))

    def step(self, parameter: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        parameter = parameter + self.alpha * gradient

        return parameter
