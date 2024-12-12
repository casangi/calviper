import numpy as np


def generate_gains(n_ant):
    amplitude = np.random.uniform(0.5, 2.5, n_ant)
    phase = np.random.uniform(0., np.pi, n_ant)

    gains = amplitude * np.cos(phase) + 1j * amplitude * np.sin(phase)

    return np.identity(n_ant) * gains


def generate_vis(n_ant, gains, samples=100, noise=0.1):
    noise = np.random.uniform(0, noise, n_ant) + 1j * np.random.uniform(0, noise, n_ant)

    model = np.empty([n_ant, n_ant], dtype=complex)
    model.fill(complex(1, 0))

    # linear transformation time!
    # V = G^H M G

    vis = np.matmul(model, gains)
    vis = np.matmul(gains.conj(), vis)

    return vis