from typing import List, Dict, Tuple, Callable, Union, Any

from torch import nn, Tensor
import matplotlib.pyplot as plt

from .mish import Mish


def plot_values(values: Dict[str, List[float]], xlabel: str = 'Epoch',
                ylabel: str = 'Loss', size: Tuple[int, int] = (12, 6)):
    """ Plots multiple lines on the same plot """

    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(size))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--")

    for title, items in values.items():
        plt.plot(items, label=title)

    plt.legend()
    plt.show()


def get_reduction_method(reduction_type: str
                         ) -> Callable[[Tensor], Union[Tensor, float]]:
    """ Reduces a tensor according to reduction type. This function should be
        probably replaced by some internal torch function written in C
        (See: https://github.com/pytorch/pytorch/blob/master/torch/nn/_reduction.py) """

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
