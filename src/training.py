import os
import datetime
from typing import Callable, List, Tuple, Any, Dict

import numpy as np
from tqdm import tqdm

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, accuracy_score, \
    f1_score

from src.experiments import LossFuncType

# Custom aliases
LossFunction = Callable[[Tensor, Tensor], Tensor]
Metric = Callable[[np.ndarray, np.ndarray], float]


def training_step(model: Module, criterion: LossFunction, optimizer: Optimizer,
                  inputs: Tensor, indices: Tensor, labels: Tensor,
                  loss_type: LossFuncType = LossFuncType.BASIC,
                  knn_use_indices: bool = False) -> float:
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
        inputs = indices if knn_use_indices else inputs
        loss = criterion(outputs, labels, inputs)

    elif loss_type == LossFuncType.ENTR_R:
        inputs = indices if knn_use_indices else inputs
        pred_class_dist = model.stored_output
        loss = criterion(outputs, labels, inputs, pred_class_dist)

    loss.backward()
    optimizer.step()

    return loss.item()


def validation_step(model: Module, criterion: LossFunction, inputs: Tensor,
                    indicies: Tensor, labels: Tensor,
                    loss_type: LossFuncType = LossFuncType.BASIC,
                    knn_use_indices: bool = False) -> float:
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
            inputs = indicies if knn_use_indices else inputs
            loss = criterion(outputs, labels, inputs)

        elif loss_type == LossFuncType.ENTR_R:
            inputs = indicies if knn_use_indices else inputs
            model_last_layer_distr = model.stored_output
            loss = criterion(outputs, labels, inputs, model_last_layer_distr)

    return loss.item()


def run(name: str, optimizer: Optimizer, criterion: LossFunction,
        model: Module, train_dataloader: DataLoader, epochs: int = 100,
        valid_dataloader: DataLoader = None, early_stopping: int = None,
        test_data_x: Tensor = None, test_data_y: Tensor = None,
        neptune_logger: Any = None, tqdm_description: str = None,
        return_best_model: bool = True, knn_use_indices: bool = False,
        eval_freq: int = None, loss_type: LossFuncType = LossFuncType.BASIC,
        output_directory: str = 'results/models'
        ) -> Tuple[Module, List[float], List[float]]:

    """ Runs standard training loop.

        Non-trivial arguments:
            neptune_logger: pass an neptune experiment object to track
                the experiment using neptune.ai service
            knn_use_indices: some of the loss functions use inputs
                (examples) to retrieve the nearest neighboorhood of a point.
                Provided kNN wrappers are capable of caching partial results
                which can be accessed using indices, not raw vectors.
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
                                 labels, loss_type, knn_use_indices)
            training_loss.append(loss)

        mean_train_loss = np.mean(training_loss)
        training_loss_history.append(mean_train_loss)

        # Log train_loss
        if neptune_logger is not None:
            neptune_logger.log_metric('train_loss', mean_train_loss)

        # Validation loop
        if valid_dataloader is not None:
            for idx, inputs, labels in valid_dataloader:
                loss = validation_step(model, criterion, inputs, idx,
                                       labels, loss_type, knn_use_indices)
                validation_loss.append(loss)

            mean_validation_loss = np.mean(validation_loss)
            validation_loss_history.append(mean_validation_loss)
            epochs_bar.set_postfix({
                'val loss': round(mean_validation_loss, 3)
            })

            # Log valid_loss
            if neptune_logger is not None:
                neptune_logger.log_metric('valid_loss', mean_validation_loss)

            # Tracking current best loss (early-stopping)
            if best_validation_loss - mean_validation_loss > 0.003:
                best_validation_loss = mean_validation_loss
                best_model = f'{name}_temp_best_model.p5'
                torch.save(model.state_dict(), best_model)
                no_change_counter = -1

            no_change_counter += 1

        # Evaluate current state of the model
        if eval_freq is not None and epoch > 0 and epoch % eval_freq == 0:
            metrics = evaluate_binary(model, test_data_x, test_data_y)

            if neptune_logger is not None:
                for metric, value in metrics.items():
                    neptune_logger.log_metric(metric, value)

        # Early stopping
        if early_stopping is not None and no_change_counter >= early_stopping:
            print('Warning: Earlystopping.')
            break

    # Overwrite current model if return_best_model is set to True
    if return_best_model and best_model is not None:
        model.load_state_dict(torch.load(best_model))

    # Save the model on the disk/W&B
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    model_name = datetime.datetime.now().strftime("%d_%m_%y-%H-%m-%S") \
        + f'{name}_model.pt'
    model_path = os.path.join(output_directory, model_name)
    torch.save(model.state_dict(), model_path)

    if neptune_logger is not None:
        neptune_logger.log_artifact(model_path)

    # Clean-up
    if best_model is not None and os.path.isfile(best_model):
        os.remove(best_model)

    return model, training_loss_history, validation_loss_history


def evaluate_binary(model: Module, test_x: Tensor, test_y: Tensor,
                    metrics: Dict[str, Metric] = None) -> Dict[str, float]:

    """ Evaluates the given model. If the model uses Tanh as output
        activation the function will decrese the threshold from 0.5
        to 0. It's also makes sure that the target is in range [0, 1] """

    # FIXME: assert length(test_y.unique()) == 2

    # Set-up metrics
    default_metrics = {
        'precision': precision_score,
        'recall': recall_score,
        'accuracy': accuracy_score,
        'f1_score': f1_score
    }

    if metrics is None:
        metrics = default_metrics

    # Set threshold
    # FIXME: This solution supports Sigmoid and Tanh only.
    #        The threshold should be based on `test_y` vector.
    #        E. g. test_y.unique() == [-1, 1] -> thr = 0.
    #              test_y.unique() = [0, 1] -> th = 0.5
    # threshold = test_y.unique().mean()

    threshold = 0.5
    last_layer_type = type(list(model.modules())[-1])
    if last_layer_type is torch.nn.Tanh:
        threshold = 0

    # Targets and predictions should always be in range [0, 1]
    target = torch.max(test_y.clone(), Tensor([0.])).reshape(-1).int()
    predictions = (model(test_x) > threshold).reshape(-1).int()

    return {metric_name: metric(target, predictions)
            for metric_name, metric in metrics.items()}
