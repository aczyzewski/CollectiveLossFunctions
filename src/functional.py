import torch
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from functools import reduce

EPSILON = 1e-9

def mish(input):
    """ Applies the mish function element-wise:
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        See additional documentation for mish class.
    """
    return input * torch.tanh(F.softplus(input))


def kl_divergence(predictions: Tensor, values: Tensor) -> Tensor:
    """ Calculates the Kullback-Leibler divergence between two distributions

    Args:
        predictions: 2-D tensor with class predictions of processed instances
        values: 2-D tensor with class labels of nearest neigbors of processed
        instances

    Returns:
        1-D tensor with KL-divergencies of neighborhoods of processed instances

    """
    _kl_divergence = lambda p, q: torch.sum(p * torch.log(p/q + EPSILON))

    output_vector = []

    # TODO: Should be implemented as tensor operation
    for i, vector in enumerate(values):
        vals, counts = torch.unique(vector, return_counts=True)
        probablity_vector = counts * 1. / torch.sum(counts)
        output_vector.append(_kl_divergence(predictions[i], probablity_vector))

    return torch.Tensor(output_vector).reshape(-1, 1)


def entropy(values: Tensor, distances: Tensor = None, use_weights: bool = None) -> Tensor:
    """ Calculates entropy independently for each vector in a tensor
        Returns results as 1-D tensor

        Args:
             values: 2-D tensor with class labels of nearest neighbors of processed instances
             distances: tensor with distances from nearest neighbors of processed instances
             use_weights: if True use weighted entropy

        Returns:
            1-D tensor with entropies of neighborhoods of processed instances
    """

    _entropy = lambda vector: torch.abs(-1. * torch.sum(torch.Tensor(
        [probability * torch.log2(probability) for probability in vector])))

    output_vector = []

    if use_weights:

        assert distances is not None, "Distances required for weighted entropy!"

        for vector in values:
            vals, counts = torch.unique(vector, return_counts=True)
            class_distances = torch.Tensor([(1 / distances[values == val]).sum() for val in vals])
            probability_vector = F.softmax(class_distances, dim=0)
            output_vector.append(_entropy(probability_vector))
    else:
        for vector in values:
            _, counts = torch.unique(vector, return_counts=True)
            probablity_vector = counts * 1. / torch.sum(counts)
            output_vector.append(_entropy(probablity_vector))

    return torch.Tensor(output_vector).reshape(-1, 1)


def theil(values: Tensor) -> Tensor:
    """ Computes the Theil index of the inequality of distribution 
        (https://en.wikipedia.org/wiki/Theil_index)
        Returns the value of the Theil index of the distribution """

    output_vector = []

    for vector in values:
        _, counts = torch.unique(vector, return_counts=True)
        mi = torch.mean(counts.float())

        if mi == 0:
            theil_index = 0
        else:
            theil_index = (1 / len(counts)) * torch.sum((counts / mi) * torch.log(counts / mi))

        output_vector.append(theil_index)

    return torch.Tensor(output_vector).reshape(-1, 1)


def gini(values: Tensor) -> Tensor:
    """ Computes the value of the Gini index of a distribution
        Returns results as a 1-D tensor """

    output_vector = []

    for vector in values:

        if torch.mean(vector) == 0:
            gini_index = 0
        else:
            mean_abs_difference = torch.mean(torch.abs(torch.Tensor(np.subtract.outer(vector, vector))))
            relative_mean_abs_difference = mean_abs_difference / torch.mean(vector)
            gini_index = 0.5 * relative_mean_abs_difference

        output_vector.append(gini_index)

    return torch.Tensor(output_vector).reshape(-1, 1)


def atkinson(values: Tensor) -> Tensor:
    """ Computes the value of the Atkinson index: https://en.wikipedia.org/wiki/Atkinson_index
        Assumes that the epsilon=1
        Return results as a 1-D tensor """

    output_vector = []

    for vector in values:
        _, counts = torch.unique(vector, return_counts=True)
        mi = torch.mean(counts.float())
        N = len(counts)

        atkinson_index = 1 - (1/mi) * (reduce(lambda x, y: x * y, counts) ** (1/N))

        output_vector.append(atkinson_index)

    return torch.Tensor(output_vector).reshape(-1, 1)
