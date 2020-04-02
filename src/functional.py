import torch
import torch.nn.functional as F

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


