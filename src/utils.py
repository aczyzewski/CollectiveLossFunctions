from typing import List, Tuple, Callable, Union, Any
from functools import reduce
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn, Tensor

from .mish import Mish


def split_data(x: np.array, y: np.array, test_size: float = 0.2,
               val_size: float = 0.2, random_state: int = None
               ) -> List[Tuple[np.array]]:
    """ Splits data into train/val/test subsets of a given size """

    train_x, test_x, train_y, test_y = \
        train_test_split(x, y, test_size=test_size, random_state=random_state)

    train_x, val_x, train_y, val_y = \
        train_test_split(train_x, train_y, test_size=val_size, random_state=random_state)

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def get_reduction_method(reduction_type: str
                         ) -> Callable[[Tensor], Union[Tensor, float]]:
    """ Reduces a tensor according to reduction type. This function should be
        probably replaced by some internal torch function written in C

        See:
        https://github.com/pytorch/pytorch/blob/master/torch/nn/_reduction.py
    """

    reduction_methods = {
        'mean': lambda x: x.mean(),
        'sum': lambda x: x.sum(),
        'none': lambda x: x
    }

    if reduction_type not in reduction_methods:
        raise KeyError('Invalid reduction type.')

    return reduction_methods[reduction_type]


def get_initialization_by_name(name: str) -> Any:
    """ Returns network weights initialization method
        by a given name """

    methods = {
        "uniform": nn.init.uniform_,
        "normal": nn.init.normal_,
        "eye": nn.init.eye_,
        "xavier_uniform": nn.init.xavier_uniform_,
        "xavier": nn.init.xavier_uniform_,
        "xavier_normal": nn.init.xavier_normal_,
        "kaiming_uniform": nn.init.kaiming_uniform_,
        "kaiming": nn.init.kaiming_uniform_,
        "kaiming_normal": nn.init.kaiming_normal_,
        "he": nn.init.kaiming_normal_,
        "orthogonal": nn.init.orthogonal_,
    }

    if name not in methods.keys():
        raise KeyError("Given initialization method name doesn\'t exist \
                        or it is not supported.")

    return methods[name]


def get_activation_by_name(name: str) -> Any:
    """ Returns activation method by a given name """

    methods = {
        "elu": nn.ELU,
        "hardshrink": nn.Hardshrink,
        "hardtanh": nn.Hardtanh,
        "leakyrelu": nn.LeakyReLU,
        "logsigmoid": nn.LogSigmoid,
        "prelu": nn.PReLU,
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "rrelu": nn.RReLU,
        "selu": nn.SELU,
        "sigmoid": nn.Sigmoid,
        "softplus": nn.Softplus,
        "logsoftmax": nn.LogSoftmax,
        "softshrink": nn.Softshrink,
        "softsign": nn.Softsign,
        "tanh": nn.Tanh,
        "tanhshrink": nn.Tanhshrink,
        "softmin": nn.Softmin,
        "softmax": nn.Softmax,
        "mish": Mish,
    }

    if name not in methods.keys():
        raise KeyError("Given activation function name doesn\'t exist \
                        or it is not supported.")

    return methods[name]


def convert_logtis_to_class_distribution(a: Tensor, no_classes: int) -> Tensor:
    """ Converts each of of a given tensor to probablility class distribution

        Example:
            >>> no_classes = 4
            >>> a = [[0, 1, 2, 3], [1, 1, 2, 2], [0, 0, 0, 3]]
            >>> convert_tensor_to_class_distribution(a, no_classes)
            [[0.25, 0.25, 0.25, 0.25],
             [0.,   0.75, 0.25, 0.  ],
             [0.75, 0.    0.    0.25]]

    """

    # FIXME: Slow!
    output = []
    for row in a.numpy():
        counts = Counter(row.astype('int'))
        output.append([counts[idx] for idx in range(no_classes)])

    return Tensor(output) / a.shape[1]


def iterparams(params):
    """ Iterate over all possible combination of given parameters """

    keys = list(params.keys())
    counters = [len(param) for param in params.values()]
    indices = [0] * len(keys)
    num_loops = reduce(lambda x, y: x * y, counters)

    for _ in range(num_loops):
        for c_idx in range(len(indices) - 1, 0, -1):
            if indices[c_idx] == len(params[keys[c_idx]]):
                indices[c_idx] = 0
                indices[c_idx - 1] += 1

        attributes = []
        for key, value, c_idx in zip(keys, params.values(), indices):
            attributes.append((key, value[c_idx]))
        yield dict(attributes)

        indices[-1] += 1

