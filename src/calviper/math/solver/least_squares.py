import numpy as np

from typing import Any

from numpy import dtype, floating
from calviper.math.loss import mean_squared_error as mse


def solve(vis, iterations, loss=mse, optimizer=None, alpha=0.1) -> tuple[np.ndarray[Any, dtype[floating[Any]]], np.ndarray[Any, dtype[Any]]]:
    # The visibility matrix should be square so this will work. To
    # for an initial guess gains vector.
    _gains = 0.1 * np.ones(vis.shape[1], dtype=complex)
    _step = np.zeros(vis.shape[1], dtype=complex)

    # Generate point source model
    _model = (1.0 + 1j * 0.0) * np.ones_like(vis, dtype=complex)

    _loss = 0.0 + 1j * 0.0

    _tracker = []

    for n in range(iterations):
        _gains_matrix = np.outer(_gains, _gains.conj())
        np.fill_diagonal(_gains_matrix, complex(0, 0))

        vis_pred = _gains_matrix * _model

        _loss = loss(vis, vis_pred)
        _tracker.append(np.abs(_loss))

        # (start) Here is where the step function is
        for i in range(vis.shape[0]):
            _numerator = 0.0 + 0.0j
            _denominator = 0.0 + 0.0j

            for j in range(vis.shape[1]):
                if i != j:
                    _numerator += vis[i, j] * _gains[j] * _model.conj()[i, j]
                    _denominator += _gains[j] * _gains[j].conj() * _model[i, j] * _model[i, j].conj()

            _step[i] = (_numerator / _denominator) - _gains[i]

            _gains[i] = _gains[i] + alpha * _step[i]

    return _gains, np.array(_tracker)
