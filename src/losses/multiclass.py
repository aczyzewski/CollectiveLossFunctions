from typing import Callable, Union

import torch
from torch import nn, Tensor
from torch.nn.functional import softmax

from src.neighboorhood import AbstractKNN
from src.utils import get_reduction_method, \
    convert_logits_to_class_distribution
from src.decorators import lossfunction
from src.functional import entropy, kl_divergence

# Type aliases
LossFunction = Callable[[Tensor, Tensor], Tensor]

# Constants
EPSILON = 1e-12


def CrossEntropy() -> LossFunction:

    logsoftmax = nn.LogSoftmax(dim=1)

    @lossfunction
    def cross_entropy(prediction: Tensor, target: Tensor,
                      reduction: str = 'mean') -> Union[float, Tensor]:

        # Convert target vector into probabilty distribution
        n_classes = prediction.shape[1]
        target = convert_logits_to_class_distribution(target, n_classes)

        # Apply softmax on model output
        prediction = logsoftmax(prediction)

        # Calculate loss
        loss = - torch.sum(target * prediction, dim=1).reshape(-1, 1)
        reduction_method = get_reduction_method(reduction)
        return reduction_method(loss)

    return cross_entropy


def WeightedCrossEntropy(knn: AbstractKNN) -> LossFunction:

    logsoftmax = nn.LogSoftmax(dim=1)

    @lossfunction
    def weighted_cross_entropy(prediction: Tensor, target: Tensor,
                               inputs: Tensor, reduction: str = 'mean'
                               ) -> Union[float, Tensor]:

        n_classes = prediction.shape[1]

        # Retrieve nearest points and their calsses
        _, _, nn_classes = knn.get(inputs.numpy(), exclude_query=True)
        nn_classes = Tensor(nn_classes)
        nn_class_entropy = entropy(nn_classes)

        # Convert target vector into probabilty distribution
        target = convert_logits_to_class_distribution(target, n_classes)

        # Apply softmax on model output
        prediction = logsoftmax(prediction)

        loss = (-torch.sum(target * prediction, dim=1).reshape(-1, 1)
                * torch.exp(-nn_class_entropy))

        reduction_method = get_reduction_method(reduction)
        return reduction_method(loss)

    return weighted_cross_entropy


def RegularizedCrossEntropy(knn: AbstractKNN) -> LossFunction:

    @lossfunction
    def regularized_cross_entropy(prediction: Tensor, target: Tensor,
                                  inputs: Tensor, reduction: str = 'mean'
                                  ) -> Union[float, Tensor]:

        n_classes = prediction.shape[1]

        # Retrieve nearest points and their calsses
        _, _, nn_classes = knn.get(inputs.numpy(), exclude_query=True)
        nn_classes = Tensor(nn_classes)

        # Convert target vector into probabilty distribution
        target = convert_logits_to_class_distribution(target, n_classes)

        # Calcuate KL-Divergence
        prediction = softmax(prediction, dim=1)
        kl_divergence_score = kl_divergence(prediction, nn_classes, n_classes)
        prediction = torch.log(prediction)

        # Calculate loss
        loss = (-torch.sum(target * prediction, dim=1).reshape(-1, 1)
                + kl_divergence_score)

        reduction_method = get_reduction_method(reduction)
        return reduction_method(loss)

    return regularized_cross_entropy


def CollectiveCrossEntropy(knn: AbstractKNN, alpha: float = 0.5
                           ) -> LossFunction:

    logsoftmax = nn.LogSoftmax(dim=1)

    @lossfunction
    def collective_cross_entropy(prediction: Tensor, target: Tensor,
                                 inputs: Tensor, reduction: str = 'mean'
                                 ) -> Union[float, Tensor]:

        n_classes = prediction.shape[1]

        # Retrieve nearest points and their calsses
        _, _, nn_classes = knn.get(inputs.numpy(), exclude_query=True)
        nn_classes = Tensor(nn_classes)
        nn_class_disribution = convert_logits_to_class_distribution(nn_classes,
                                                                    n_classes)

        # Convert target vector into probabilty distribution
        target = convert_logits_to_class_distribution(target, n_classes)

        # Apply softmax on model output
        prediction = logsoftmax(prediction)

        loss = - torch.sum(target * prediction
                           + alpha * nn_class_disribution * prediction,
                           dim=1).reshape(-1, 1)

        reduction_method = get_reduction_method(reduction)
        return reduction_method(loss)

    return collective_cross_entropy
