import numpy as np

import toolviper.utils.logger as logger


def mean_squared_error(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the mean squared error as defined in https://en.wikipedia.org/wiki/Mean_squared_error
    :param y: Observed values.
    :param y_pred: Predicted values.
    :return: Mean squared error.
    """
    return np.mean(np.square(y_pred - y))


def root_mean_squared_error(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the mean squared error as defined in https://en.wikipedia.org/wiki/Root_mean_square_deviation
    :param y: Observed values.
    :param y_pred: Predicted values.
    :return: Mean squared error.
    """
    return np.sqrt(np.mean(np.square(y_pred - y)))


def mean_absolute_error(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
        Calculates the mean absolute error as defined in https://en.wikipedia.org/wiki/Mean_absolute_error. This method
        tends to be more robust to outliers in the data.
        :param y: Observed values.
        :param y_pred: Predicted values.
        :return: Mean absolute error.
    """
    return np.mean(np.abs(y_pred - y))


def huber_loss(y: np.ndarray, y_pred: np.ndarray, delta: float) -> float:
    """
        Calculates the Huber loss as defined in https://en.wikipedia.org/wiki/Huber_loss. This loss function combines
        the best features of the MSE and the MAE.
        :param y:
        :param y_pred:
        :param delta:
        :return:
    """
    condition = np.abs(y - y_pred) < delta

    r = np.where(
        condition,
        0.5 * (y - y_pred) ** 2,  # True case
        delta * (np.abs(y - y_pred) - 0.5 * delta)  # Otherwise case
    )

    try:
        return np.sum(r) / y.shape[0]

    except ZeroDivisionError:
        logger.error("Divide by zero error due to shape of observed values array.")
        return np.nan