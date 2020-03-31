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

# KNN MSE Loss - example

```python pycharm={"is_executing": false}
%load_ext autoreload
%autoreload 2
```

```python pycharm={"is_executing": false}
import sys
sys.path.insert(0, '..')
```

```python pycharm={"is_executing": false}
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from torch.optim import SGD
from torch.nn import MSELoss
from torch.utils.data import TensorDataset, DataLoader
```

```python pycharm={"is_executing": true}
from src.experiments import run_training_loop
from src.networks import CustomNeuralNetwork
from src.losses import KNNMSELoss
from src.utils import plot_values
```

```python
# Load and normalize the trainig data
data = load_boston()
x, y = data['data'], data['target']
x = MinMaxScaler().fit_transform(x)
```

```python
# Split data into train/test sets
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)
```

```python
x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
y = torch.tensor(y.tolist(), dtype=torch.float32).reshape(-1, 1)

train_x = torch.from_numpy(np.ascontiguousarray(train_x, dtype=np.float32))
train_y = torch.tensor(train_y.tolist(), dtype=torch.float32).reshape(-1,1)

test_x = torch.from_numpy(np.ascontiguousarray(test_x, dtype=np.float32))
test_y = torch.tensor(test_y.tolist(), dtype=torch.float32).reshape(-1,1)

val_x = torch.from_numpy(np.ascontiguousarray(val_x, dtype=np.float32))
val_y = torch.tensor(val_y.tolist(), dtype=torch.float32).reshape(-1,1)
```

```python
train_dataloader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
val_dataloader = DataLoader(TensorDataset(test_x, test_y), batch_size=32)
test_dataloader = DataLoader(TensorDataset(test_x, test_y), batch_size=1)
```

```python
# Grid search
losses = [('MSELoss', MSELoss(), False)]
for k in [1, 3, 5]:
    losses.append((f'KNNMSELoss (K={k})', KNNMSELoss(x, y, k=k), True))
```

```python
from tqdm import tqdm_notebook as tqdm

# Training loop
output_data = []

for lossname, lossfunc, knn_loss in losses:
    
    model = CustomNeuralNetwork(layers=[13, 16, 4, 1], hidden_activations="relu")
    optimizer = SGD(model.parameters(), lr=0.001)
    criterion = lossfunc
    
    model, train_loss, test_loss = run_training_loop(
        optimizer, 
        criterion, 
        model, 
        train_dataloader, 
        val_dataloader, 
        epochs=500, 
        use_wandb=False, 
        knn_loss=knn_loss, 
        tqdm_description=lossname
    )
    
    output_data.append((model, train_loss, test_loss))
```

```python
# Base line (HingeLoss)
model, tloss, vloss = output_data[0]
plot_values({'val': vloss})
```

```python
# KNN Hinge Loss
plt.figure(figsize=(18, 18))
for idx, (model, tloss, vloss) in enumerate(output_data[1:]):
    plt.subplot(3, 3, idx + 1)
    plt.grid('on')
    #plt.plot(tloss, label='train')
    plt.plot(vloss, label='val')
    plt.legend()
    plt.title(losses[idx + 1][0])
```

```python
plot_all_data = {}
for idx, (model, tloss, vloss) in enumerate(output_data[1:]):
    plot_all_data[losses[idx + 1][0] + "_val"] = vloss
plot_values(plot_all_data, size=(17, 14))
```

```python
# Evaluate output
for idx, (model, tloss, vloss) in enumerate(output_data):
    model_name = losses[idx][0]
    gt, predictions = [], []
    print(f'{model_name} / mse: {MSELoss()(model(test_x), test_y)}')
```

```python

```
