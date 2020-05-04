---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
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
import string
import warnings
import asyncio
from joblib import Parallel, delayed
from typing import List, Any, Dict
```

```python
sys.path.insert(0, '..')
warnings.filterwarnings('ignore')
```

```python
import umap
import neptune
import numpy as np
import pandas as pd
from torch import nn
from torch.optim import Adam
from torch.nn import Module
from torch.utils.data import DataLoader
```

```python
import src.datasets as datasets
import src.losses as lossfunc
import src.utils as utils
import src.experiments as trainingloop
from src.neighboorhood import FaissKNN
from src.preprocessing import transform
from src.networks import CustomNeuralNetwork
```

# Binary classification

```python
# Parameters
GLOBAL_PARAMS = {
    'use_umap': False,
    'batch_size': 32,
    'output_type': 'binary'
}
```

```python
EXPERIMENT_PARAMS = {
    'learning_rate': [0.01],
    'layers': [[12, 24, 2]],
    'hidden_activations': ["relu"],
    'output_activations': ["sigmoid"],
    'epochs': [256],
    'early_stopping': [32]
}
```

```python
# Define list o functions
functions_type = trainingloop.LossFuncType.BASIC
basic_functions = [
    
    # Basic functions
    ['hinge_loss', lossfunc.HingeLoss(), False],
    ['sq_hinge_loss', lossfunc.SquaredHingeLoss(), False],
#     ['bce', lossfunc.BinaryCrossEntropy(), True],
#     ['log_loss', lossfunc.HingeLoss(), False],
#     ['exp_loss', lossfunc.HingeLoss(), False],

]

neptune.init('aczyzewski/clfbin')
for function_name, criterion, std_range in basic_functions:
    
    # Base loss functions. Inputs: (pred, target)
    for dataset_name in datasets.get_datasets('binary')[:1]:
        
        simplified_dataset_name = datasets.simplify_dataset_name(dataset_name)        
        
        # Preprocess the data
        x, y = datasets.load_dataset(dataset_name, transform, use_umap=USE_UMAP)
        y[y == 0] = -1 if not std_range else 0
        knn = FaissKNN(x, y, precompute=True)
        
        # Dataloaders
        (train_x, train_y), (val_x, val_y), (test_x, test_y) = utils.split_data(x, y)
        train_dataloader = datasets.convert_to_dataloader(train_x, train_y, batch_size=32)
        valid_dataloader = datasets.convert_to_dataloader(val_x, val_y, batch_size=32)
        
        # Prepare an experiment
        for PARAMS in utils.iterparams(EXPERIMENT_PARAMS):
            
            experiment_name = f'{simplified_dataset_name}_{function_name}'
            ALL_PARAMS = {**PARAMS, **GLOBAL_PARAMS}
            tags = [function_name, simplified_dataset_name, 'binary']
            experiment = neptune.create_experiment(name=experiment_name, tags=tags, params=ALL_PARAMS,
                                                   upload_source_files=['../src/losses/binary.py'])

            # Parameters
            layers = [x.shape[1]] + PARAMS['layers'] + [1]
            model = CustomNeuralNetwork(layers, PARAMS['hidden_activations'], PARAMS['output_activations'])
            optimizer = Adam(model.parameters(), lr=PARAMS['learning_rate'])


            trainingloop.run(experiment_name, optimizer, criterion, model, train_dataloader,
                         PARAMS['epochs'], valid_dataloader, early_stopping=PARAMS['early_stopping'], 
                         neptune_logger=neptune, knn_use_indicies=True)

            experiment.stop()
    
```

```python

```
