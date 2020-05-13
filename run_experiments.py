import sys
import string
import warnings
import asyncio
from joblib import Parallel, delayed
from typing import List, Any, Dict

import umap
import neptune
import numpy as np
import pandas as pd
from torch import nn
from torch.optim import Adam
from torch.nn import Module
from torch.utils.data import DataLoader

import src.datasets as datasets
import src.losses as lossfunc
import src.utils as utils
import src.experiments as trainingloop
from src.experiments import LossFuncType as ftype
from src.neighboorhood import FaissKNN
from src.preprocessing import transform
from src.networks import CustomNeuralNetwork

warnings.filterwarnings('ignore')


loss_functions = {
    'hinge_loss': {
        ftype.BASIC: lossfunc.HingeLoss,
        ftype.CLF: lossfunc.CollectiveHingeLoss
     },

    'sq_hinge_loss': {
        ftype.BASIC: lossfunc.SquaredHingeLoss,
        ftype.CLF: lossfunc.CollectiveSquaredHingeLoss
     },

    'bce': {
        ftype.BASIC: lossfunc.BinaryCrossEntropy,
        ftype.CLF: lossfunc.CollectiveBinaryCrossEntropy
     },

    'log_loss': {
        ftype.BASIC: lossfunc.LogisticLoss,
        ftype.CLF: lossfunc.CollectiveLogisticLoss
     },

    'exp_loss': {
        ftype.BASIC: lossfunc.ExponentialLoss,
        ftype.CLF: lossfunc.CollectiveExponentialLoss
     }
}    

std_target_range = set(['bce'])

# Parameters
GLOBAL_PARAMS = {
    'use_umap': [False],
    'batch_size': [32],
    'dataset': datasets.get_datasets('binary'),
    'function_type': [ftype.BASIC, ftype.ENTR_R, ftype.ENTR_W, ftype.CLF],
    'function_name': list(loss_functions.keys()),
    'knn_k': [3]
}

EXPERIMENT_PARAMS = {
    'learning_rate': [0.01],
    'layers': [[12, 24, 2]],
    'hidden_activations': ["relu"],
    'output_activations': ["sigmoid"],
    'epochs': [256],
    'early_stopping': [32]
}

neptune.init('clfmsc2020/experiments')

for G_PARAMS in utils.iterparams(GLOBAL_PARAMS):
    
    output_type = 'binary'
    function_type = G_PARAMS['function_type']
    function_type_name = function_type.name.lower()
    function_name = G_PARAMS['function_name']
    dataset_name = G_PARAMS['dataset']

    simplified_dataset_name = datasets.simplify_dataset_name(dataset_name)        

    # Preprocess the data
    x, y = datasets.load_dataset(dataset_name, transform, use_umap=G_PARAMS['use_umap'])
    knn = FaissKNN(x, y, precompute=True, k=G_PARAMS['knn_k'])
    if function_name not in std_target_range:
        y[y == 0] = -1

    # Dataloaders
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = utils.split_data(x, y)
    train_dataloader = datasets.convert_to_dataloader(train_x, train_y, batch_size=G_PARAMS['batch_size'])
    valid_dataloader = datasets.convert_to_dataloader(val_x, val_y, batch_size=G_PARAMS['batch_size'])
    
    # Initialize criterion
    criterion = None

    # Initialize criterion
    if function_type == ftype.BASIC:
        criterion = loss_functions[function_name][function_type]()

    elif function_type == ftype.ENTR_R:
        base_criterion = loss_functions[function_name][ftype.BASIC]()
        criterion = lossfunc.EntropyRegularizedBinaryLoss(base_criterion, knn)

    elif function_type == ftype.ENTR_W:
        base_criterion = loss_functions[function_name][ftype.BASIC]()
        criterion = lossfunc.EntropyWeightedBinaryLoss(base_criterion, knn)

    elif function_type == ftype.CLF:
        cfunction = loss_functions[function_name][function_type]
        criterion = cfunction(knn, 0.5)  # Fixed parameters (alpha, beta)

    else:
        raise Exception('Invalid function type!')

    # Start experiments
    for PARAMS in utils.iterparams(EXPERIMENT_PARAMS):

        # Gather all necessary values
        experiment_name = f'{simplified_dataset_name}_{function_name}_{function_type_name}'
        ALL_PARAMS = {**PARAMS, **G_PARAMS}
        tags = [function_name, simplified_dataset_name, output_type, function_type_name, 'test_2']

        # Register an experiment
        experiment = neptune.create_experiment(name=experiment_name, tags=tags, params=ALL_PARAMS,
                                               upload_source_files=['src/losses/binary.py',
                                                                    __file__])

        # Configure modules
        layers = [x.shape[1]] + PARAMS['layers'] + [1]
        model = CustomNeuralNetwork(layers, PARAMS['hidden_activations'], PARAMS['output_activations'])
        optimizer = Adam(model.parameters(), lr=PARAMS['learning_rate'])

        # Run an experiment
        trainingloop.run(experiment_name, optimizer, criterion, model, train_dataloader,
                     PARAMS['epochs'], valid_dataloader, early_stopping=PARAMS['early_stopping'], 
                     neptune_logger=neptune, knn_use_indicies=True, loss_type=function_type)

        experiment.stop()
