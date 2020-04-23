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
EPSILON = 1e-16


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

        prediction = F.softmax(prediction, dim=1)

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
        loss = (
            Tensor([1.]) / torch.log(Tensor([2.]))
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

        _, _, classes = knn.get(inputs.numpy(), exclude_query=True)

        # FIXME: Improve: find cleaner and more efficient way to convert
        #        a vector into probability distribution of 2 classes
        _k = float(knn.k)
        classes = Tensor(classes.astype('float32'))
        pos_examples = torch.sum(classes == 1.0, dim=1, dtype=torch.float32)
        neg_examples = _k - pos_examples

        nn_distribution = torch.stack((neg_examples, pos_examples), dim=1)
        nn_distribution /= _k

        assert pred_class_dist.shape == nn_distribution.shape, \
            'Invalid distibution shape!'

        kl_div_score = kl_divergence(pred_class_dist,
                                     nn_distribution + EPSILON)

        assert kl_div_score.shape == target.shape, \
            'Invalid KL divergence output shape!'

        reguralized_loss = base_loss + kl_div_score

        reduction_method = get_reduction_method(reduction)
        return reduction_method(reguralized_loss)

    return entropy_regularized_bin_loss

# --- Collective loss functions

def CollectiveHingeLoss(knn: AbstractKNN, alpha: float = 0.5) -> LossFunction:
    """ Collective Hinge Loss implementation """

    @lossfunction
    def collective_hinge_loss(prediction: Tensor, target: Tensor,
                              inputs: Tensor, reduction: str = 'mean'
                              ) -> Union[float, Tensor]:

        _, _, classes = knn.get(inputs.numpy(), exclude_query=True)

        # Get and return most frequent class
        knn_classes, _ = torch.mode(Tensor(classes), dim=1)
        knn_classes = knn_classes.reshape(-1, 1)
        assert knn_classes.shape == target.shape, 'Invalid knn classes shape!'

        loss = torch.max(
            torch.zeros_like(prediction),
            Tensor([1.]) - prediction * target
            - alpha * (torch.ones_like(knn_classes) - target * knn_classes)
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

        _, _, classes = knn.get(inputs.numpy(), exclude_query=True)

        # Get and return most frequent class
        knn_classes, _ = torch.mode(Tensor(classes), dim=1)
        knn_classes = knn_classes.reshape(-1, 1)
        assert knn_classes.shape == target.shape, 'Invalid knn classes shape!'

        prediction = F.softmax(prediction, dim=1)
        loss = (
            - target * torch.log2(prediction)
            - (Tensor([1.]) - target)
            * torch.log2(Tensor([1.]) - prediction + EPSILON)
            + alpha * (
                - target * torch.log2(knn_classes + EPSILON)
                - (Tensor([1.]) - target)
                * torch.log2(Tensor([1.]) - knn_classes + EPSILON)
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

        _, _, classes = knn.get(inputs.numpy(), exclude_query=True)

        # Get and return most frequent class
        knn_classes, _ = torch.mode(Tensor(classes), dim=1)
        knn_classes = knn_classes.reshape(-1, 1)
        assert knn_classes.shape == target.shape, 'Invalid knn classes shape!'

        # FIXME: Cache the result of torch.log(Tensor([2.])
        loss = (
            Tensor([1.]) / torch.log(Tensor([2.]))
            * torch.log(
                Tensor([1.])
                + torch.exp(- target * prediction)
                - alpha * torch.exp(-target * knn_classes)
                + EPSILON
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

        _, _, classes = knn.get(inputs.numpy(), exclude_query=True)

        # Get and return most frequent class
        knn_classes, _ = torch.mode(Tensor(classes), dim=1)
        knn_classes = knn_classes.reshape(-1, 1)
        assert knn_classes.shape == target.shape, 'Invalid knn classes shape!'

        loss = (torch.exp(- beta * prediction * target)
                - alpha * torch.exp(- prediction * knn_classes))

        reduction_method = get_reduction_method(reduction)
        return reduction_method(loss)

    return collective_exponential_loss
