import numpy as np


def solve(vis, iterations, loss=mse, optimizer=None, alpha=0.1):
    # The visibility matrix should be square so this will work. To
    # for an initial guess gains vector.
    # _gains = np.random.uniform(0, 1, vis.shape[1])
    _gains = 0.1 * np.ones(vis.shape[1], dtype=complex)
    # _numerator = np.zeros_like(_gains, dtype=complex)
    # _denomenator = np.zeros_like(_gains, dtype=complex)

    # Generate point source model
    _model = np.ones_like(vis)

    _loss = 0.0
    _step = 0.0

    _tracker = []

    for n in range(iterations):
        _gains_matrix = np.outer(_gains, _gains.conj())
        vis_pred = np.matmul(
            _gains.conj(),
            np.matmul(_model, _gains_matrix)
        )

        _loss = loss(vis, vis_pred)
        _tracker.append(np.abs(_loss))

        # (start) Here is where the step function is
        for i in range(vis.shape[1]):
            _numerator = 0.0 + 0.0j
            _denomenator = 0.0 + 0.0j

            for j in range(vis.shape[1]):
                if i != j:
                    _numerator += vis[0, i, j] * _gains[j] * _model.conj()[0, i, j]
                    _denomenator += _gains[j] * _gains[j].conj() * _model[0, i, j] * _model[0, i, j].conj()

            _step = (_numerator / _denomenator) - _gains[i]

            _gains[i] = _gains[i] + alpha * _step

        # (end)
        if n % 10 == 0:
            print(np.abs(_gains))

    return _tracker