import torch
from torch import Tensor
from typing import Callable, Union

from src.functional import entropy, kl_divergence
from src.neighboorhood import AbstractKNN
from src.utils import get_reduction_method

# Type aliases
LossFunction = Callable[[Tensor, Tensor], Tensor]

# --- Basic loss functions

def HingeLoss() -> LossFunction:
    """ Pure Hinge Loss implementation """

    def hinge_loss(prediction: Tensor, target: Tensor,
                   reduction: str = 'mean') -> float:

        loss = torch.max(
            torch.zeros_like(prediction),
            1.0 - prediction * target
        )
        reduction_method = get_reduction_method(reduction)
        return reduction_method(loss)

    return hinge_loss

# --- Entropy-weighted loss functions

def EntropyWeightedBinaryLoss(base_loss_function: LossFunction,
                              knn: AbstractKNN) -> LossFunction:
    """ Calculates pure loss function (`base_loss_fuction`)
        weighted by the entropy of a neighboorhood """

    def entropy_weighted_bin_loss(prediction: Tensor, target: Tensor,
                                  inputs: Tensor, reduction: str = 'mean'
                                  ) -> Union[float, Tensor]:

        assert prediction.shape == target.shape, 'Invalid target shape!'
        _, _, classes = knn.get(inputs.numpy(), exclude_query=True)

        entropies = entropy(Tensor(classes))
        assert entropies.shape == target.shape, 'Invalid entropies shape!'

        base_loss = base_loss_function(prediction, target, reduction='none')
        assert base_loss.shape == target.shape, 'Invalid base loss shape!'

        weighted_loss = torch.exp(-1 * entropies) * base_loss
        assert weighted_loss.shape == target.shape, 'Invalid weighted_loss shape!'

        reduction_method = get_reduction_method(reduction)
        return reduction_method(weighted_loss)

    return entropy_weighted_bin_loss

# --- Entropy-regularized loss functions

def EntropyRegularizedBinaryLoss(base_loss_function: LossFunction,
                                 knn: AbstractKNN) -> LossFunction:

    """ Calculates pure loss function (`base_loss_fuction`) and
        adds Kullback-Leibler divergence of two distributions to the
        final loss """

    def entropy_regularized_bin_loss(prediction: Tensor, target: Tensor,
                                     inputs: Tensor, prediction_distribution: Tensor,
                                     reduction: str = 'mean') -> Union[float, Tensor]:

        assert prediction.shape == target.shape, 'Invalid target shape!'

        base_loss = base_loss_function(prediction, target, reduction='none')
        assert base_loss.shape == target.shape, 'Invalid base loss shape!'

        _, _, classes = knn.get(inputs.numpy(), exclude_query=True)

        # FIXME: Immprove
        _k = float(knn.k)
        pos_examples = torch.sum(Tensor(classes.astype('float32')) == 1.0, dim=1, dtype=torch.float32)
        neg_examples = _k - pos_examples

        nn_distribution = torch.stack((neg_examples, pos_examples), dim=1)
        nn_distribution /= _k
        assert prediction_distribution.shape == nn_distribution.shape, 'Invalid distibution shape!'

        kl_div_score = kl_divergence(prediction_distribution, nn_distribution)
        assert kl_div_score.shape == target.shape, 'Invalid KL divergence output shape!'

        reguralized_loss = base_loss + kl_div_score
        assert reguralized_loss.shape == target.shape, 'Invalid weighted_loss shape!'

        reduction_method = get_reduction_method(reduction)
        return reduction_method(reguralized_loss)

    return entropy_regularized_bin_loss

# --- Collective loss functions

def CollectiveHingeLoss(knn: AbstractKNN, alpha: float) -> LossFunction:
    """ Collective Hinge Loss implementation """

    def collective_hinge_loss(prediction: Tensor, target: Tensor,
                              inputs: Tensor, reduction: str = 'mean') -> float:

        _, _, classes = knn.get(inputs.numpy(), exclude_query=True)

        # Get and return most frequent class
        knn_classes, _ = torch.mode(Tensor(classes), dim=1)
        knn_classes = knn_classes.reshape(-1, 1)
        assert knn_classes.shape == target.shape, 'Invalid knn classes shape!'

        loss = torch.max(
            torch.zeros_like(prediction),
            1.0 - prediction * target - alpha * (torch.ones_like(knn_classes) - target * knn_classes)
        )

        reduction_method = get_reduction_method(reduction)
        return reduction_method(loss)

    return collective_hinge_loss
