import os
from enum import Enum
from datetime import datetime
from collections import defaultdict
from typing import Callable, List, Tuple, Any

import wandb
import torch
import numpy as np
from tqdm import tqdm

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader


def training_step(
		model: Module, criterion: Callable[[Tensor, Tensor], Tensor], optimizer: Optimizer, 
		inputs: Tensor, labels: Tensor, knn_loss: bool = False) -> float:
	""" Performs single train step """

	optimizer.zero_grad()

	outputs = model(inputs)
	loss = criterion(outputs, labels) if not knn_loss else criterion(outputs, labels, inputs)

	loss.backward()
	optimizer.step()

	return loss.item()


def validation_step(		
		model: Module, criterion: Callable[[Tensor, Tensor], Tensor], inputs: Tensor,
		labels: Tensor, knn_loss: bool = False) -> float:
	""" Forward-propagation without calculating gradients """

	with torch.no_grad():
		outputs = model(inputs)
		loss = criterion(outputs, labels) if not knn_loss else criterion(outputs, labels, inputs)

	return loss.item()


def run_training_loop(
		optimizer: Optimizer = None, criterion: Callable[[Tensor, Tensor], Tensor] = None,
		model: Module = None, trainloader: DataLoader = None, valloader: DataLoader = None,
		epochs: int = 100, early_stopping: int = None, return_best_model: bool = True,		
		knn_loss: bool = False, tensorboard_path: str = '../tensorboard', use_wandb: bool = True,
		wandb_project_name: str = 'collective_loss_functions', tqdm_description: str = None
	) -> Tuple[Module, List[float], List[float]]:

	training_loss_history, validation_loss_history = [], []
	best_model = None
	best_val_loss = float('inf')
	no_change_counter = 0

	# Safe-checks
	if trainloader is None:
		print('Error: Training set is required! Exiting ...')
		return 

	if valloader is None and (early_stopping or return_best_model):
		print("Warning: Early stoping and/or returning only best model won't work if the validation set is not defined.")

	# Start logging
	if use_wandb:
		wandb.init(project=wandb_project_name)
		wandb.watch(model)

	# Start training
	epochs_bar = tqdm(range(epochs), desc=tqdm_description)
	for epoch in epochs_bar: 

		training_loss, validation_loss = [], []

		# Training loop
		for i, (inputs, labels) in enumerate(trainloader, 0):
			loss = training_step(model, criterion, optimizer, inputs, labels, knn_loss)
			training_loss.append(loss)
		mean_train_loss = np.mean(training_loss)
		training_loss_history.append(mean_train_loss)

		# Validation loop
		if valloader is not None:
			for i, (inputs, labels) in enumerate(valloader, 0):
				loss = validation_step(model, criterion, inputs, labels, knn_loss)
				validation_loss.append(loss)
			mean_validation_loss = np.mean(validation_loss)
			validation_loss_history.append(mean_validation_loss)
			epochs_bar.set_postfix({'val loss': round(mean_validation_loss, 3)})
	
			if mean_validation_loss < best_val_loss:
				best_val_loss = mean_validation_loss
				best_model = model
				no_change_counter = 0
			else:
				no_change_counter += 1

		# Log outputs
		if use_wandb:
			wandb.log({
				'Epoch': epoch + 1,
				'Train loss': mean_train_loss,
				'Validation loss': mean_validation_loss
			})

		# Early stopping
		if early_stopping is not None and no_change_counter >= early_stopping:
			print('Warning: Earlystopping.')
			break

	# Overwrite current model if return_best_model is set to True
	if return_best_model and best_model is not None:
		model = best_model

	# Save the model on the disk/W&B
	if use_wandb:
		model_name = datetime.datetime.now().strftime("%d_%m_%y-%H-%m-%S") + '_model.pt'
		torch.save(model.state_dict(), os.path.join(wandb.run.dir, model_name))

	return model, training_loss_history, validation_loss_history
