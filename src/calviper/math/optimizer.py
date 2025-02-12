import numpy as np


class MeanSquaredError:

    def __init__(self, alpha: float = 1e-3):
        self.alpha = alpha

    @staticmethod
    def gradient(target: np.ndarray, model: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        # cache_ = target, model.conj()
        # numerator_ = np.matmul(cache_, parameter)
        # denominator_ = np.matmul(model * model.conj(), parameter * parameter.conj())

        # einsum should work here *only* because we made the  diagonal of the model zeros
        numerator_ = np.einsum('tcpij,tcpj,tcpij->tcpi', target, parameter, model.conj())
        denominator_ = np.einsum('tcpj,tcpj,tcpij,tcpij->tcpi', parameter, parameter.conj(), model, model.conj())
        gradient_ = (numerator_ / denominator_) - parameter

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
        parameter = parameter + self.alpha * gradient

        return parameter
