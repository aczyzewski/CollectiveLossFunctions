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
