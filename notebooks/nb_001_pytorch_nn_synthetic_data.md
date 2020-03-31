---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Task:
 -  Utworzenie prostego notatnika umożliwiającego: 
  * wygenerowanie syntetycznego zbioru danych
  * uruchomienie procesu uczenia
  * wizualizację tego procesu


---

```python
%load_ext autoreload
%autoreload 2
```

```python
import sys
sys.path.insert(0, '..')
```

```python
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Any, Tuple, Callable
```

```python
import torch
from torch import nn, Tensor, FloatTensor
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
```

```python
# Custom modules

from src.data.generator import SyntheticDataGenerator
from src.networks import ExampleNeuralNetwork
from src.losses import MSE
from src.utils import plot_loss
```

```python
# Reproducible results
seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
```

```python
# Create TB writter
writer = SummaryWriter(log_dir='../tensorboard/')
```

```python
# Generate data
num_points = 120
x, y = SyntheticDataGenerator().generate(num_points)
x, y = FloatTensor(x).reshape(num_points, 1), FloatTensor(y).reshape(num_points, 1)
```

```python
# Load custom NN and Loss
model = ExampleNeuralNetwork()
criterion = MSE()
```

```python
# Hyperparameters
learning_rate = 0.1
num_epochs = 512
optimizer = SGD(model.parameters(), lr=learning_rate)
```

```python
loss_values = []
for epoch in range(num_epochs):
    
    # Forward pass
    outputs = model(x)
    
    # Calculate loss
    loss = criterion(outputs, y)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Save loss
    loss_values.append(loss.item())
    writer.add_scalar('Loss/train', loss.item(), epoch)
       
writer.close()
plot_loss(loss_values)
```
