
from typing import List, Dict, Tuple, Callable, Union

import matplotlib.pyplot as plt
from torch import Tensor


def plot_values(values: Dict[str, List[float]],
                xlabel: str = 'Epoch', ylabel: str = 'Loss',
                size: Tuple[int, int] = (12, 6)):
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


def get_reduction_method(reduction_type: str) -> Callable[[Tensor], Union[Tensor, float]]:
    """ Reduces a tensor according to reduction type. This function should be
        probably replaced by some internal torch function written in C
        (See: https://github.com/pytorch/pytorch/blob/master/torch/nn/_reduction.py) """

    reduction_methods = {
        'mean': lambda x: x.mean(),
        'sum': lambda x: x.sum(),
        'none': lambda x: x
    }

    return reduction_methods[reduction_type]
