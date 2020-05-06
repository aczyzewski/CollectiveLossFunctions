from typing import Callable, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from src.functional import entropy, kl_divergence
from src.neighboorhood import AbstractKNN
from src.utils import get_reduction_method
from src.decorators import lossfunction

# Type aliases
LossFunction = Callable[[Tensor, Tensor], Tensor]

# Constants
EPSILON = 1e-9


# --- Basic loss functions
def HingeLoss() -> LossFunction:
    """ Pure Hinge Loss implementation """

    @lossfunction
    def hinge_loss(prediction: Tensor, target: Tensor,
                   reduction: str = 'mean') -> Union[float, Tensor]:

        loss = torch.max(
            Tensor([0.]),
            Tensor([1.]) - prediction * target
        )

        reduction_method = get_reduction_method(reduction)
        return reduction_method(loss)

    return hinge_loss


def SquaredHingeLoss() -> LossFunction:

    hingle_loss = HingeLoss()

    @lossfunction
    def squared_hinge_loss(prediction: Tensor, target: Tensor,
                           reduction: str = 'mean') -> Union[float, Tensor]:

        loss = torch.pow(
            hingle_loss(prediction, target, reduction='none'),
            exponent=2
        )
        reduction_method = get_reduction_method(reduction)
        return reduction_method(loss)

    return squared_hinge_loss


def BinaryCrossEntropy() -> LossFunction:

    @lossfunction
    def binary_cross_entropy(prediction: Tensor, target: Tensor,
                             reduction: str = 'mean') -> Union[float, Tensor]:

        loss = (
            - target * torch.log2(prediction + EPSILON)
            - (Tensor([1.]) - target)
            * torch.log2(Tensor([1.]) - prediction + EPSILON)
        )

        reduction_method = get_reduction_method(reduction)
        return reduction_method(loss)

    return binary_cross_entropy


def LogisticLoss() -> LossFunction:

    @lossfunction
    def logistic_loss(prediction: Tensor, target: Tensor,
                      reduction: str = 'mean') -> Union[float, Tensor]:

        cached_log = 1.4426950408889634     # = 1 / ln(2)
        loss = (
            Tensor([cached_log])
            * torch.log(
                Tensor([1.])
                + torch.exp(- target * prediction)
                + EPSILON
            )
        )
        reduction_method = get_reduction_method(reduction)
        return reduction_method(loss)

    return logistic_loss


def ExponentialLoss(beta: float = 0.5) -> LossFunction:

    @lossfunction
    def exponential_loss(prediction: Tensor, target: Tensor,
                         reduction: str = 'mean') -> Union[float, Tensor]:

        loss = torch.exp(- beta * prediction * target)
        reduction_method = get_reduction_method(reduction)
        return reduction_method(loss)

    return exponential_loss


# --- Entropy-weighted loss functions
def EntropyWeightedBinaryLoss(base_loss_function: LossFunction,
                              knn: AbstractKNN) -> LossFunction:
    """ Calculates pure loss function (`base_loss_fuction`)
        weighted by the entropy of a neighboorhood """

    @lossfunction
    def entropy_weighted_bin_loss(prediction: Tensor, target: Tensor,
                                  inputs: Tensor, reduction: str = 'mean'
                                  ) -> Union[float, Tensor]:

        _, _, classes = knn.get(inputs.numpy(), exclude_query=True)

        entropies = entropy(Tensor(classes))
        assert entropies.shape == target.shape, 'Invalid entropies shape!'

        base_loss = base_loss_function(prediction, target, reduction='none')
        assert base_loss.shape == target.shape, 'Invalid base loss shape!'

        loss = torch.exp(-1 * entropies) * base_loss

        reduction_method = get_reduction_method(reduction)
        return reduction_method(loss)

    return entropy_weighted_bin_loss


# --- Entropy-regularized loss functions
def EntropyRegularizedBinaryLoss(base_loss_function: LossFunction,
                                 knn: AbstractKNN) -> LossFunction:

    """ Calculates pure loss function (`base_loss_fuction`) and
        adds Kullback-Leibler divergence of two distributions to the
        final loss """

    @lossfunction
    def entropy_regularized_bin_loss(prediction: Tensor, target: Tensor,
                                     inputs: Tensor, pred_class_dist: Tensor,
                                     reduction: str = 'mean'
                                     ) -> Union[float, Tensor]:

        base_loss = base_loss_function(prediction, target, reduction='none')

        pred_class_dist = F.sigmoid(pred_class_dist)
        _, _, classes = knn.get(inputs.numpy(), exclude_query=True)
        kl_div_score = kl_divergence(pred_class_dist, Tensor(classes))

        assert kl_div_score.shape == target.shape, \
            'Invalid KL divergence output shape!'

        regularized_loss = base_loss + kl_div_score
        regularized_loss = torch.max(Tensor([0.]), regularized_loss)
        reduction_method = get_reduction_method(reduction)
        return reduction_method(regularized_loss)

    return entropy_regularized_bin_loss


# --- Collective loss functions
def CollectiveHingeLoss(knn: AbstractKNN, alpha: float = 0.5) -> LossFunction:
    """ Collective Hinge Loss implementation """

    @lossfunction
    def collective_hinge_loss(prediction: Tensor, target: Tensor,
                              inputs: Tensor, reduction: str = 'mean'
                              ) -> Union[float, Tensor]:

        # Get and return average class
        _, _, classes = knn.get(inputs.numpy(), exclude_query=True)
        avg_class = (Tensor(classes).sum(dim=1) / knn.k).reshape(-1, 1)
        assert avg_class.shape == target.shape, 'Invalid avg class shape!'

        loss = torch.max(
            torch.zeros_like(prediction),
            Tensor([1.]) - prediction * target
            - alpha * (torch.ones_like(avg_class) - target * avg_class)
        )

        reduction_method = get_reduction_method(reduction)
        return reduction_method(loss)

    return collective_hinge_loss


def CollectiveSquaredHingeLoss(knn: AbstractKNN, alpha: float = 0.5
                               ) -> LossFunction:
    """ Collective Hinge Loss implementation """

    collective_hinge_loss = CollectiveHingeLoss(knn, alpha)

    @lossfunction
    def collective_squared_hinge_loss(prediction: Tensor, target: Tensor,
                                      inputs: Tensor, reduction: str = 'mean'
                                      ) -> Union[float, Tensor]:

        loss = torch.pow(
            collective_hinge_loss(prediction, target, inputs,
                                  reduction='none'),
            exponent=2
        )

        reduction_method = get_reduction_method(reduction)
        return reduction_method(loss)

    return collective_squared_hinge_loss


def CollectiveBinaryCrossEntropy(knn: AbstractKNN, alpha: float = 0.5
                                 ) -> LossFunction:
    """ Collective Hinge Loss implementation """

    @lossfunction
    def collective_bin_cross_entropy(prediction: Tensor, target: Tensor,
                                     inputs: Tensor, reduction: str = 'mean'
                                     ) -> Union[float, Tensor]:

        # Get and return average class
        _, _, classes = knn.get(inputs.numpy(), exclude_query=True)
        avg_class = (Tensor(classes).sum(dim=1) / knn.k).reshape(-1, 1)
        assert avg_class.shape == target.shape, 'Invalid avg class shape!'

        loss = (
            - target * torch.log2(prediction + EPSILON)
            - (Tensor([1.]) - target)
            * torch.log2(Tensor([1.]) - prediction + EPSILON)

            + alpha * (
                - target * torch.log2(avg_class + EPSILON)
                - (Tensor([1.]) - target)
                * torch.log2(Tensor([1.]) - avg_class + EPSILON)
            )
        )
        reduction_method = get_reduction_method(reduction)
        return reduction_method(loss)

    return collective_bin_cross_entropy


def CollectiveLogisticLoss(knn: AbstractKNN, alpha: float = 0.5
                           ) -> LossFunction:

    @lossfunction
    def collective_logistic_loss(prediction: Tensor, target: Tensor,
                                 inputs: Tensor, reduction: str = 'mean'
                                 ) -> Union[float, Tensor]:

        # Get and return average class
        _, _, classes = knn.get(inputs.numpy(), exclude_query=True)
        avg_class = (Tensor(classes).sum(dim=1) / knn.k).reshape(-1, 1)
        assert avg_class.shape == target.shape, 'Invalid avg class shape!'

        cached_log = 1.4426950408889634     # = 1 / ln(2)
        loss = (
            Tensor([cached_log])
            * torch.log(
                Tensor([1.])
                + torch.exp(-prediction * (target + alpha * avg_class))
            )
        )

        reduction_method = get_reduction_method(reduction)
        return reduction_method(loss)

    return collective_logistic_loss


def CollectiveExponentialLoss(knn: AbstractKNN, alpha: float = 0.5,
                              beta: float = 0.5) -> LossFunction:

    @lossfunction
    def collective_exponential_loss(prediction: Tensor, target: Tensor,
                                    inputs: Tensor, reduction: str = 'mean'
                                    ) -> Union[float, Tensor]:

        # Get and return average class
        _, _, classes = knn.get(inputs.numpy(), exclude_query=True)
        avg_class = (Tensor(classes).sum(dim=1) / knn.k).reshape(-1, 1)
        assert avg_class.shape == target.shape, 'Invalid avg class shape!'

        loss = torch.exp(- beta * prediction * (target + alpha * avg_class))
        reduction_method = get_reduction_method(reduction)
        return reduction_method(loss)

    return collective_exponential_loss
