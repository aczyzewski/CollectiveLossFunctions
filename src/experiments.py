import os
import datetime
from enum import Enum
from typing import Callable, List, Tuple, Any

import torch
import numpy as np
from tqdm import tqdm

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader


# Custom aliases
LossFunction = Callable[[Tensor, Tensor], Tensor]


class LossFuncType(Enum):
    """ A simple enumerator of loss function types """
    BASIC = 0,      # Basic loss function
    ENTR_W = 1,     # Entropy-weighted
    ENTR_R = 2,     # Entropy-regularized
    CLF = 3         # Collective


def training_step(model: Module, criterion: LossFunction, optimizer: Optimizer,
                  inputs: Tensor, indicies: Tensor, labels: Tensor,
                  loss_type: LossFuncType = LossFuncType.BASIC,
                  knn_use_indicies: bool = False) -> float:
    """ Performs single train step. This function should be compatible
        with any loss function type.

        Non-trivial params:
            loss_type: determines which arguments should be passed to a given
                loss function (criterion)
    """

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = None

    if loss_type == LossFuncType.BASIC:
        loss = criterion(outputs, labels)

    elif loss_type in [LossFuncType.ENTR_W, LossFuncType.CLF]:
        inputs = indicies if knn_use_indicies else inputs
        loss = criterion(outputs, labels, inputs)

    elif loss_type == LossFuncType.ENTR_R:
        inputs = indicies if knn_use_indicies else inputs
        pred_class_dist = model.stored_output
        loss = criterion(outputs, labels, inputs, pred_class_dist)

    loss.backward()
    optimizer.step()

    return loss.item()


def validation_step(model: Module, criterion: LossFunction, inputs: Tensor,
                    indicies: Tensor, labels: Tensor,
                    loss_type: LossFuncType = LossFuncType.BASIC,
                    knn_use_indicies: bool = False) -> float:
    """ Forward-propagation without calculating gradients. This function
        should be compatible with any loss function type.

        Non-trivial params:
            loss_type: determines which arguments should be passed to a given
                loss function (criterion)
    """

    loss = None
    with torch.no_grad():
        outputs = model(inputs)

        if loss_type == LossFuncType.BASIC:
            loss = criterion(outputs, labels)

        elif loss_type in [LossFuncType.ENTR_W, LossFuncType.CLF]:
            inputs = indicies if knn_use_indicies else inputs
            loss = criterion(outputs, labels, inputs)

        elif loss_type == LossFuncType.ENTR_R:
            inputs = indicies if knn_use_indicies else inputs
            model_last_layer_distr = model.stored_output
            loss = criterion(outputs, labels, inputs, model_last_layer_distr)

    return loss.item()


def run(name: str, optimizer: Optimizer, criterion: LossFunction,
        model: Module, train_dataloader: DataLoader,  epochs: int = 100,
        valid_dataloader: DataLoader = None, early_stopping: int = None,
        neptune_logger: Any = None, tqdm_description: str = None,
        return_best_model: bool = True, knn_use_indicies: bool = False,
        loss_type: LossFuncType = LossFuncType.BASIC,
        output_directory: str = 'results/models'
        ) -> Tuple[Module, List[float], List[float]]:

    """ Runs standard training loop.

        Non-trivial arguments:
            neptune_logger: pass an neptune experiment object to track
                the experiment using neptune.ai service
            knn_use_indicies: some of the loss fuctions use inputs
                (examples) to retireve the nearest neighboorhood of a point.
                Provided kNN wrappers are capable of caching partial results
                which can be accessed using indicies, not raw vectors.
    """

    training_loss_history, validation_loss_history = [], []
    best_model = None
    best_validation_loss = float('inf')
    no_change_counter = 0

    # Warnings
    if valid_dataloader is None and (early_stopping or return_best_model):
        print("Warning: Early stoping and/or returning only best model won't\
                work if the validation set is not defined.")

    # Start training
    epochs_bar = tqdm(range(epochs), desc=tqdm_description)
    for epoch in epochs_bar:

        training_loss, validation_loss = [], []

        # Training loop
        for idx, inputs, labels in train_dataloader:
            loss = training_step(model, criterion, optimizer, inputs, idx,
                                 labels, loss_type, knn_use_indicies)
            training_loss.append(loss)

        mean_train_loss = np.mean(training_loss)
        training_loss_history.append(mean_train_loss)

        # Log train_loss
        if neptune_logger is not None:
            neptune_logger.log_metric(f'train_loss', mean_train_loss)

        # Validation loop
        if valid_dataloader is not None:
            for idx, inputs, labels in valid_dataloader:
                loss = validation_step(model, criterion, inputs, idx,
                                       labels, loss_type, knn_use_indicies)
                validation_loss.append(loss)

            mean_validation_loss = np.mean(validation_loss)
            validation_loss_history.append(mean_validation_loss)
            epochs_bar.set_postfix({
                'val loss': round(mean_validation_loss, 3)
            })

            # Log valid_loss
            if neptune_logger is not None:
                neptune_logger.log_metric(f'valid_loss',
                                          mean_validation_loss)

            # Tracking current best loss (early-stopping)
            if mean_validation_loss < best_validation_loss:
                best_validation_loss = mean_validation_loss
                best_model = model
                no_change_counter = -1

            no_change_counter += 1

        # Early stopping
        if early_stopping is not None and no_change_counter >= early_stopping:
            print('Warning: Earlystopping.')
            break

    # Overwrite current model if return_best_model is set to True
    if return_best_model and best_model is not None:
        model = best_model

    # Save the model on the disk/W&B
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    model_name = datetime.datetime.now().strftime("%d_%m_%y-%H-%m-%S") \
        + f'{name}_model.pt'
    model_path = os.path.join(output_directory, model_name)
    torch.save(model.state_dict(), model_path)

    if neptune_logger is not None:
        neptune_logger.log_artifact(model_path)

    return model, training_loss_history, validation_loss_history
