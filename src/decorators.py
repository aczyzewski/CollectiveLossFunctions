from typing import Callable, Union
import torch
from torch import Tensor
from .utils import get_reduction_method

# Aliases
LossFunction = Callable[[Tensor, Tensor], Tensor]


def lossfunction(func: LossFunction) -> LossFunction:
    """ A decorator that helps to maintain valid input/output shapes
        of tensors in order to ensure valid tensor operations. """

    def wrapper(prediction: Tensor, target: Tensor, *args, **kwargs
                ) -> Union[float, Tensor]:

        assert prediction.shape[0] == target.shape[0], 'Invalid target or \
            prediction shape!'

        reduction_method = kwargs.get('reduction', 'mean')
        kwargs['reduction'] = 'none'

        loss = func(prediction, target, *args, **kwargs)
        assert loss.shape == target.shape, 'Invalid loss shape!'
        assert torch.isnan(loss).sum() == 0, "Calculated loss \
            contains NaN values"
        assert (loss < 0).sum() == 0, "Negative loss function!"

        reduction_method = get_reduction_method(reduction_method)
        return reduction_method(loss)

    return wrapper
