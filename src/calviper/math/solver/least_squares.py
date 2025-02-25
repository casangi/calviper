import numba

import numpy as np
import toolviper.utils.logger as logger

from calviper.math.optimizer import MeanSquaredError


# from calviper.math.loss import mean_squared_error as mse


class LeastSquaresSolver:

    def __init__(self):
        # public variables
        self.losses = None
        self.parameter = None

        self.optimizer = None

        # Private variables
        self.model_ = None

    '''
    def _solve(self, vis, iterations, loss=mse, optimizer=None, alpha=0.1):
        # **** Deprecated ****
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
    '''

    def predict_(self):

        n_channel, n_polarizations, n_antennas = self.parameter.shape
        parameter_matrix_ = np.identity(n_antennas) * self.parameter
        cache_ = np.dot(self.model_, parameter_matrix_)

        return np.dot(parameter_matrix_.conj(), cache_)

    @staticmethod
    #@numba.njit()
    def predict(model_, parameter)->np.ndarray:
        n_time, n_channel, n_polarizations, n_antennas = parameter.shape
        prediction = np.zeros_like(model_)

        # This can definitely be optimized, but I just want to test for now.
        for time in range(n_time):
            for channel in range(n_channel):
                for polarization in range(n_polarizations):
                    for i in range(n_antennas):
                        for j in range(n_antennas):
                            if i == j:
                                continue

                            prediction[time, channel, polarization, i, j] = parameter[time, channel, polarization, i] * model_[time, channel, polarization, i, j] * np.conj(parameter[time, channel, polarization, j])
                            #print(f"({time}, {channel}, {polarization}): prediction: [{i}, {j}] {parameter[time, channel, polarization, i]}")


        return prediction

    def solve(self, vis, iterations, optimizer=MeanSquaredError(), stopping=1e-3):
        # This is an attempt to do the solving in a vectorized way

        # Unpack the shape
        n_times, n_channel, n_polarization, n_antenna1, n_antenna2 = vis.shape

        assert n_antenna1 == n_antenna2, logger.error("Antenna indices don't match")

        self.parameter = np.tile(0.1 * np.ones(n_antenna1, dtype=np.complex64), reps=[n_times, n_channel, int(np.sqrt(n_polarization)), 1])
        #print(f"\n@Creation(param): pol(X): {self.parameter[0, 0, 0, 1]}\tpol(Y): {self.parameter[0, 0, 1, 1]}\n")
        # Generate point source model
        if self.model_ is None:
            self.model_ = (1.0 + 1j * 0.0) * np.ones_like(vis, dtype=np.complex64)

            # numpy.fill_diagonal doesn't fill tensors in the way I had hoped, ie. for shape = (m, n, i, j)
            # the fill is done for m == n == i == j, which is not what we want. Instead, we want
            # i == j for each (m. n). The following is my attempt to fix this.
            anti_eye = np.ones((n_antenna1, n_antenna2), dtype=np.complex64)

            eye = np.identity(n_antenna1, dtype=np.complex64)
            np.fill_diagonal(anti_eye, np.complex64(1., 0.))

            self.model_ = self.model_

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

            #print(f"gradient: {gradient_.mean()}")

            self.parameter = optimizer.step(
                parameter=self.parameter,
                gradient=gradient_
            )

            y_pred = self.predict(self.model_, self.parameter)

            self.losses.append(optimizer.loss(y_pred, vis))

            if n % (iterations // 10) == 0:
                logger.info(f"iteration: {n}\tloss: {np.abs(self.losses[-1])}")

            if self.losses[-1] < stopping:
                logger.info(f"Iteration: ({n})\tStopping criterion reached: {self.losses[-1]}")
                break

        return self.parameter
