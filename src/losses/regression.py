from typing import Callable, Union

import torch
from torch import Tensor

from src.neighboorhood import AbstractKNN
from src.utils import get_reduction_method
from src.decorators import lossfunction
from src.functional import scaled_variance

# Type aliases
LossFunction = Callable[[Tensor, Tensor], Tensor]

# Constants
EPSILON = 1e-9


# --- Basic loss functions
def MSE() -> LossFunction:
    """ Mean Square Loss implementation (torch.nn.MSELoss) """

    @lossfunction
    def mse(prediction: Tensor, target: Tensor, reduction: str = 'mean'
            ) -> Union[float, Tensor]:

        loss = torch.pow(target - prediction, 2)
        reduction_method = get_reduction_method(reduction)
        return reduction_method(loss)

    return mse


def MAE() -> LossFunction:
    """ Mean Absolute Error implementation (torch.nn.L1Loss) """

    @lossfunction
    def mae(prediction: Tensor, target: Tensor, reduction: str = 'mean'
            ) -> Union[float, Tensor]:

        loss = torch.abs(target - prediction)
        reduction_method = get_reduction_method(reduction)
        return reduction_method(loss)

    return mae


# Variance-weighted loss functions
def VarianceWeightedMSE(knn: AbstractKNN) -> LossFunction:
    """ Mean Square Loss implementation """

    mse = MSE()

    @lossfunction
    def var_weighted_mse(prediction: Tensor, target: Tensor, inputs: Tensor,
                     reduction: str = 'mean') -> Union[float, Tensor]:

        # Average target value in the neighborhood
        _, _, values = knn.get(inputs.numpy(), exclude_query=True)
        variance = scaled_variance(target, Tensor(values)) / knn.k

        # Caculate loss
        base_loss = mse(prediction, target, reduction='none')
        loss = base_loss * torch.exp(-variance)
        reduction_method = get_reduction_method(reduction)
        return reduction_method(loss)

    return var_weighted_mse


def VarianceWeightedMAE(knn: AbstractKNN) -> LossFunction:
    """ Mean Square Loss implementation """

    mae = MAE()

    @lossfunction
    def var_weighted_mae(prediction: Tensor, target: Tensor, inputs: Tensor,
                     reduction: str = 'mean') -> Union[float, Tensor]:

        # Average target value in the neighborhood
        _, _, values = knn.get(inputs.numpy(), exclude_query=True)
        variance = scaled_variance(target, Tensor(values)) / knn.k

        # Caculate loss
        base_loss = mae(prediction, target, reduction='none')
        loss = base_loss * torch.exp(-variance)
        reduction_method = get_reduction_method(reduction)
        return reduction_method(loss)

    return var_weighted_mae


# Collective loss functions
def CollectiveMSE(knn: AbstractKNN, alpha: float = 0.5) -> LossFunction:
    """ Mean Square Loss implementation """

    @lossfunction
    def collective_mse(prediction: Tensor, target: Tensor, inputs: Tensor,
                       reduction: str = 'mean') -> Union[float, Tensor]:

        # Average target value in the neighborhood
        _, _, values = knn.get(inputs.numpy(), exclude_query=True)
        avg_value = (Tensor(values).sum(dim=1) / knn.k).reshape(-1, 1)

        loss = (torch.pow(target - prediction, 2)
                + alpha * torch.pow(prediction - avg_value, 2))

        reduction_method = get_reduction_method(reduction)
        return reduction_method(loss)

    return collective_mse


def CollectiveMAE(knn: AbstractKNN, alpha: float = 0.5) -> LossFunction:
    """ Mean Square Loss implementation """

    @lossfunction
    def collective_mae(prediction: Tensor, target: Tensor, inputs: Tensor,
                       reduction: str = 'mean') -> Union[float, Tensor]:

        # Average target value in the neighborhood
        _, _, values = knn.get(inputs.numpy(), exclude_query=True)
        avg_value = (Tensor(values).sum(dim=1) / knn.k).reshape(-1, 1)

        loss = (torch.abs(target - prediction)
                + alpha * torch.abs(prediction - avg_value))

        reduction_method = get_reduction_method(reduction)
        return reduction_method(loss)

    return collective_mae
