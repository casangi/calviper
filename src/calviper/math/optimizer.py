import numpy as np

class MeanSquaredError:

    def __init__(self, alpha: float = 1e-3):
        self.alpha = alpha

    @staticmethod
    def gradient(target: np.ndarray, model: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        cache_ = target * target.T

        numerator_ = np.matmul(cache_, parameter)
        denominator_ = np.matmul(model * model.conj(), parameter * parameter.conj())

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
        return np.mean(np.square(y_pred - y))

    def step(self, parameter: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        parameter = parameter + self.alpha * gradient

        return parameter
