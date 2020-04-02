import torch
import torch.nn.functional as F
import numpy as np

from torch import Tensor

def mish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    '''
    return input * torch.tanh(F.softplus(input))


def entropy(values: Tensor) -> Tensor:
    """ Calculates entropy independently for each vector in a tensor
        Returns results as 1-D tensor """
    _entropy = lambda vector: -1. * torch.sum(torch.Tensor(
        [probability * torch.log2(probability) for probability in vector]))

    output_vector = []

    for vector in values:
        _, counts = torch.unique(vector, return_counts=True)
        probablity_vector = counts * 1. / torch.sum(counts)
        output_vector.append(_entropy(probablity_vector))

    return torch.Tensor(output_vector).reshape(-1, 1)


def theil(values: Tensor) -> Tensor:
    """ Computes the Theil index of the inequality of distribution (https://en.wikipedia.org/wiki/Theil_index)
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

    return torch.Tensor(output_vector).reshape(-1,1)


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

    return torch.Tensor(output_vector).reshape(-1,1)
