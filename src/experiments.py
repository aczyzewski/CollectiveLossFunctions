import wandb
import torch
import numpy as np

from enum import Enum
from tqdm import tqdm
from collections import defaultdict
from typing import Callable, List
from datetime import datetime

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

from typing import Any

class TrainingCallbackType(Enum):
	ON_EPOCH_START = 0,
	ON_EPOCH_END = 1,

	def __eq__(self, value):
		return self.value == value

def training_step(model, criterion, optimizer, inputs, labels, knn_loss: bool = False) -> float:
	""" Performs single train step """

	# Remove past gradients
	optimizer.zero_grad()

	# Forward
	outputs = model(inputs)
	loss = criterion(outputs, labels) if not knn_loss else criterion(outputs, labels, inputs)

	# Backward
	loss.backward()
	optimizer.step()

	return loss.item()

def validation_step(model, criterion, inputs, labels, knn_loss: bool = False) -> float:
	""" Forward-propagation without calculating gradients """
	with torch.no_grad():
		outputs = model(inputs)
		loss = criterion(outputs, labels) if not knn_loss else criterion(outputs, labels, inputs)
	return loss.item()

def run_training_loop(
	
	# Mandatory arguments
	optimizer: Optimizer = None,
	criterion: Callable[[Tensor, Tensor], Tensor] = None,
	model: Module = None,
	trainloader: DataLoader = None,
	valloader: DataLoader = None,
	
	# Default arguments
	epochs: int = 100,
	callbacks: List[Callable[[Any], bool]] = defaultdict(list),
	
	# Loss related
	knn_loss: bool = False,
	
	# Logs
	tensorboard_path: str = '../tensorboard',
	use_wandb: bool = True,
	wandb_project_name: str = 'collective_loss_functions'
	
) -> (Module, List[float], List[float]):
	pass

	# Collect losses
	training_loss_history, validation_loss_history = [], []

	# W&B:
	if use_wandb:
		wandb.init(project=wandb_project_name)
		wandb.watch(model)

	# Mail loop
	for epoch in tqdm(range(epochs)): 

		training_loss, validation_loss = [], []

		# Callbacks (EPOCH_START)
		force_stop_by_callbacks = [callback(model, optimizer, criterion, training_loss, validation_loss) for callback in callbacks[TrainingCallbackType.ON_EPOCH_START.value]]
		if any(force_stop_by_callbacks): break

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

		# W&B
		if use_wandb:
			wandb.log({
				'Epoch': epoch + 1,
				'Train loss': mean_train_loss,
				'Validation loss': mean_validation_loss
			})

		# Callbacks (EPOCH_END)
		force_stop_by_callbacks = [callback(model, optimizer, criterion, training_loss, validation_loss) for callback in callbacks[TrainingCallbackType.ON_EPOCH_END.value]]
		if any(force_stop_by_callbacks): break

	if use_wandb:
		model_name = datetime.datetime.now().strftime("%d_%m_%y-%H-%m-%S") + '_model.pt'
		torch.save(model.state_dict(), os.path.join(wandb.run.dir, model_name))

	return model, training_loss_history, validation_loss_history
	
	