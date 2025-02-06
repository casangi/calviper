import numpy as np
import toolviper.utils.logger as logger

from calviper.math.optimizer import MeanSquaredError


class LeastSquaresSolver:

    def __init__(self):
        # public variables
        self.losses = None
        self.parameter = None

        self.optimizer = None

        # Private variables
        self.model_ = None

    def predict(self):
        parameter_matrix_ = np.identity(self.parameter.shape[0]) * self.parameter
        cache_ = np.dot(self.model_, parameter_matrix_)

        return np.dot(parameter_matrix_.conj(), cache_)

    def solve(self, vis, iterations, optimizer=MeanSquaredError(), stopping=1e-3):
        # This is an attempt to do the solving in a vectorized way

        self.parameter = 0.1 * np.ones(vis.shape, dtype=complex)

        # Generate point source model
        if self.model_ is None:
            self.model_ = (1.0 + 1j * 0.0) * np.ones_like(vis, dtype=complex)
            np.fill_diagonal(self.model_, complex(0., 0.))

        self.losses = []

        for n in range(iterations):
            # Fill this in when I figure out the most optimal way to calculate the error given the
            # input data structure.
            # self.losses.append(optimizer.loss(y, y_pred))

            gradient_ = optimizer.gradient(
                target=vis,
                model=self.model_,
                parameter=self.parameter
            )

            self.parameter = optimizer.step(
                parameter=self.parameter,
                gradient=gradient_
            )

            y_pred = self.predict()

            self.losses.append(optimizer.loss(y_pred, vis))

            if self.losses[-1] < stopping:
                logger.info(f"Iteration: ({n})\tStopping criterion reached: {self.losses[-1]}")
                break

        return self.parameter
