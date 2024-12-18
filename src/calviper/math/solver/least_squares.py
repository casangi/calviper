import numpy as np
import toolviper.utils.logger as logger

from calviper.math.optimizer import MeanSquaredError
from calviper.math.loss import mean_squared_error as mse


class LeastSquaresSolver:

    def __init__(self):
        # public variables
        self.losses = None
        self.parameter = None

        self.optimizer = None

        # Private variables
        self.model_ = None

    def _solve(self, vis, iterations, loss=mse, optimizer=None, alpha=0.1):
        # The visibility matrix should be square so this will work. To
        # for an initial guess gains vector.
        _gains = 0.1 * np.ones(vis.shape[1], dtype=complex)
        _step = np.zeros(vis.shape[1], dtype=complex)

        # Generate point source model
        _model = (1.0 + 1j * 0.0) * np.ones_like(vis, dtype=complex)

        loss_ = 0.0 + 1j * 0.0

        self.losses = []

        for n in range(iterations):
            _gains_matrix = np.outer(_gains, _gains.conj())
            np.fill_diagonal(_gains_matrix, complex(0, 0))

            vis_pred = _gains_matrix * _model

            loss_ = loss(vis, vis_pred)
            self.losses.append(np.abs(loss_))

            # (start) Here is where the step function is
            for i in range(vis.shape[0]):
                _numerator = 0.0 + 0.0j
                _denominator = 0.0 + 0.0j

                for j in range(vis.shape[1]):
                    if i != j:
                        _numerator += vis[i, j] * _gains[j] * _model.conj()[i, j]
                        _denominator += _gains[j] * _gains[j].conj() * _model[i, j] * _model[i, j].conj()

                _step[i] = (_numerator / _denominator) - _gains[i]

                print(f"step({i}): {_step[i]}")
                _gains[i] = _gains[i] + alpha * _step[i]

        return _gains

    def predict(self):
        parameter_matrix_ = np.identity(self.parameter.shape[0]) * self.parameter
        cache_ = np.dot(self.model_, parameter_matrix_)

        return np.dot(parameter_matrix_.conj(), cache_)

    def solve(self, vis, iterations, optimizer=MeanSquaredError(), stopping=1e-3):
        # This is an attempt to do the solving in a vectorized way

        self.parameter = 0.1 * np.ones(vis.shape[1], dtype=complex)

        # Generate point source model
        if self.model_ is None:
            self.model_ = (1.0 + 1j * 0.0) * np.ones_like(vis, dtype=complex)
            np.fill_diagonal(self.model_, complex(0., 0.))


        self.losses = []

        for n in range(iterations):
            # Fill this in when I figure out the most optimal way to calculate the error given the
            # input data structure.
            #self.losses.append(optimizer.loss(y, y_pred))

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