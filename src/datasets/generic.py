from math import ceil
from typing import Tuple
import numpy as np

# Aliases
Dataset = Tuple[np.ndarray, np.ndarray]


def make_two_spirals(n_points: int, squeeze: int = 720, noise: float = 0,
                     seed: int = None) -> Dataset:
    """ Returns the two spirals dataset. """

    n_points = n_points // 2
    points = np.sqrt(np.random.rand(n_points, 1)) * squeeze * (2 * np.pi) / 360
    d1x = -np.cos(points) * points
    d1y = np.sin(points) * points
    x, y = (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
            np.hstack((np.zeros(n_points), np.ones(n_points))))

    x = x / x.max(axis=0)
    x += np.random.uniform(-1., 1., size=(n_points * 2, 2)) * noise
    x = x / x.max(axis=0)
    y = y.reshape(-1, 1)

    return x, y


def make_checkerboard(n_points: int, n_clusters: int = 6, noise: float = 0,
                      resolution: int = 1000, seed: int = None) -> Dataset:
    """ Returns checkerboard dataset """

    def _get_checkerboard(resolution: int, n_clusters: int) -> np.ndarray:
        checkerboard = np.indices((n_clusters, n_clusters)).sum(axis=0) % 2
        scale_factor = int(ceil(resolution / n_clusters))
        return np.repeat(np.repeat(checkerboard, scale_factor, axis=1),
                         scale_factor, axis=0)

    def _get_random_points(n_points: int) -> np.ndarray:
        return np.random.uniform(0, resolution, size=(n_points, 2)).astype(int)

    x = _get_random_points(n_points)
    classes = _get_checkerboard(resolution, n_clusters)
    y = classes[x[:, 0], x[:, 1]]

    x = x.astype(float) / resolution
    x += np.random.uniform(-1., 1., size=(n_points, 2)) * noise
    x = x / x.max(axis=0)
    y = y.reshape(-1, 1)

    return (x * 2 - 1), y
