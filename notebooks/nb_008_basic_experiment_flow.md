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

```python
%load_ext autoreload
%autoreload 2
```

```python
import sys
import warnings
from collections import defaultdict
```

```python
sys.path.insert(0, '..')
warnings.filterwarnings('ignore')
```

```python
import numpy as np
from umap import UMAP

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
```

```python
from src.datasets import UCIDatabase
from src.losses import KNNHingeLoss, HingeLoss
from src.preprocessing import transform
from src.experiments import run_training_loop
from src.networks import CustomNeuralNetwork
```

```python
datasets = UCIDatabase(output_directory='../datasets')
```

```python
bin_classifiation_datasets = [
    'Phishing Websites',
    'Breast Cancer Wisconsin (Diagnostic)',
    'Bank Marketing',
    'Skin Segmentation',
    'Adult'
]
```

```python
# Define layers of NNs
params_models_layers = [
    [12, 16, 32],
    [6, 16, 32, 54, 32, 16],
    [4, 12, 128, 64, 64, 32, 48]
]
```

```python
# Define hidden activations
params_hidden_activation = [
    "sigmoid",
    "relu",
    "mish"
]
```

```python
# Other parameters

# Learning
batch_size = 1024
learning_rate = 0.01
max_epochs = 1

# Data
test_subset_size = 0.2
val_subset_size = 0.2

# Output
results_output_path = 'results/008_experiment_results.csv'
```

```python
results = []
for dataset in bin_classifiation_datasets:
    
    # Prepare data
    x, y = datasets.get_by_name(dataset).load()
    x, y = transform(x, y)
    x = UMAP().fit_transform(x) 
    
    num_features = x.shape[1]
    
    # Split data into test/val/train
    x, y = torch.tensor(x), torch.tensor(y)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_subset_size)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=val_subset_size)
        
    # Convert subsets into DataLoaders
    train_dataloader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size)
    val_dataloader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)
    test_dataloader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size)
    
    # Define criterions
    params_criterions = [
        (KNNHingeLoss(x, y), True),
        (HingeLoss(), False)
    ]

    # Run the grid search
    for model_configuration in params_models_layers:
        for hidden_activation in params_hidden_activation:
            for criterion, is_knn_loss in params_criterions:
                
                # Prepare the model
                layers = [num_features] + model_configuration + [1]
                model = CustomNeuralNetwork(layers, hidden_activation, "sigmoid")
                    
                # Define other hyper-parameters
                optimizer = Adam(model.parameters(), lr=learning_rate)
                model, train_loss, val_loss = run_training_loop(
                    optimizer, criterion, model, train_dataloader, val_dataloader, epochs=max_epochs,
                    knn_loss=is_knn_loss, use_wandb=False
                )
                
                # Evaluate the model
                metrics = defaultdict(list)
                with torch.no_grad():
                    for examples, labels in test_dataloader:
                        predictions = model(examples) > 0.5
                        metrics['precision'].append(precision_score(labels, predictions))
                        metrics['recall'].append(recall_score(labels, predictions))
                        metrics['f1'].append(f1_score(labels, predictions))
                        
                # Average metrics from all batches
                for key in metrics.keys():
                    metrics[key] = sum(metrics[key]) / len(metrics[key])

                # Save results
                results.append((dataset, train_loss, val_loss, model, model_configuration, hidden_activation, criterion.__name__, metrics))      
```

```python
results
```

```python

```
