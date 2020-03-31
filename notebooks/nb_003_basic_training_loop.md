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

# Basic Training Loop - example

```python
%load_ext autoreload
%autoreload 2
```

```python
import sys
sys.path.insert(0, '..')
```

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from src.experiments import run_training_loop
from src.networks import CustomNeuralNetwork
from src.utils import plot_values

from torch.optim import SGD
from torch.nn import MSELoss
from torch.utils.data import TensorDataset, DataLoader
```

```python
# Load and normalize the trainig data
data = load_boston()
x, y = data['data'], data['target']
x = MinMaxScaler().fit_transform(x)
```

```python
# Convert data into Tensors
x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
y = torch.tensor(y.tolist(), dtype=torch.float32).reshape(-1, 1)
```

```python
# Split data into train/test sets
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
train_dataloader = DataLoader(TensorDataset(train_x, train_y), batch_size=32)
test_dataloader = DataLoader(TensorDataset(test_x, test_y), batch_size=32)
```

```python
# Define model and hyperparameters
model = CustomNeuralNetwork(layers=[13, 16, 4, 1], hidden_activations="sigmoid")
model
```

```python
optimizer = SGD(model.parameters(), lr=0.01)
criterion = MSELoss()
```

```python
# Start training
model, train_loss, test_loss = run_training_loop(
    optimizer, criterion, model, train_dataloader, test_dataloader, epochs=200, use_wandb=False
)
```

```python
# Plot values
plot_values({
    'train_loss': train_loss,
    'test_loss': test_loss,
})
```
