# Collective Loss Functions (2020)

import argparse

from torch import Tensor
from torch.optim import Adam

import neptune
import numpy as np

from src.utils import iterparams, split_data
from src.preprocessing import transform
from src.neighboorhood import FaissKNN
from src.networks import CustomNeuralNetwork
from src.training import run, evaluate_binary
from src.datasets import load_dataset, convert_to_dataloader, \
    simplify_dataset_name
from src.experiments import load_config_file, get_loss_function, \
    LossFuncType


def print_logo() -> None:
    print("""
          ____ _     _____   ____   ___ ____   ___
         / ___| |   |  ___| |___ \\ / _ \\___ \\ / _ \\
        | |   | |   | |_      __) | | | |__) | | | |
        | |___| |___|  _|    / __/| |_| / __/| |_| |
         \\____|_____|_|     |_____|\\___/_____|\\___/
     """)


def parse_arguments() -> argparse.Namespace:
    """ Handles CLI arguments """

    parser = argparse.ArgumentParser()

    parser.add_argument('config', help='Path to a config (.yaml) file.')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable print statements.')
    parser.add_argument('-o', '--output', default='results',
                        help='Set the output path.')
    parser.add_argument('-n', '--neptune', action='store_true',
                        help='Enable neptune.ai logger.')
    return parser.parse_args()


def run_experiments(args: argparse.Namespace, project_id: str) -> None:
    """ Wraps all modules into a single ML pipeline and performs
        set of predefined experiments. """

    def _debug(text: str) -> None:
        """ Prints debug messages """
        if args.debug or args.verbose:
            print(f'INFO: {text}')

    if args.neptune:
        neptune.init(project_id)
        _debug('neptune.ai logger was initialized.')

    _c_data, _c_neighbourhood, _c_experiment = \
        load_config_file(args.config)

    # Notice:
    #    "_c_*" (configuration)  a range of values for a given section
    #    "_p_*" (parameters)     a single value for a given section

    # Some additional checks
    assert all([item in ['basic', 'entr_r', 'entr_w', 'collective']
                for item in _c_experiment['FUNCTION_TYPE']]), 'Invalid loss type!'
    assert not ('bce' in _c_experiment['FUNCTION_NAME'] and
                'tanh' in _c_experiment['OUTPUT_ACTIVATIONS']), \
                'BCE cannot be combined with TanH!'

    # Iterate over datasets
    for _p_data in iterparams(_c_data):
        _debug(f'Current dataset: {_p_data.DATASET}')

        # Load a dataset
        x, y = load_dataset(_p_data.DATASET, transform,
                            use_umap=_p_data.USE_UMAP)

        (train_x, train_y), (val_x, val_y), (test_x, test_y) = \
            split_data(x, y, val_size=_p_data.VAL_SIZE, test_size=_p_data.TEST_SIZE)

        # Adjust the negative class
        if _p_data.NEG_CLASS != 0:
            for subset in [train_y, val_y, test_y]:
                subset[subset == 0] = _p_data.NEG_CLASS

        # Create dataloaders
        train_dataloader = convert_to_dataloader(train_x, train_y,
                                                 batch_size=_p_data.BATCH_SIZE)

        valid_dataloader = convert_to_dataloader(val_x, val_y,
                                                 batch_size=_p_data.BATCH_SIZE,
                                                 startidx=train_x.shape[0])

        test_data_x, test_data_y = Tensor(test_x), Tensor(test_y)

        # Iterate over different sizes of the neighbourhood
        for _p_nn in iterparams(_c_neighbourhood):

            KNN = None
            if _p_nn.K is not None and _p_nn.K > 0:
                train_val_x = np.concatenate((train_x, val_x), axis=0)
                train_val_y = np.concatenate((train_y, val_y), axis=0)
                KNN = FaissKNN(train_val_x, train_val_y,
                               precompute=True, k=_p_nn.K)
                _debug(f'The KNN module was initialized (k = {_p_nn.K}).')

            for _p_exp in iterparams(_c_experiment):

                # Get the loss function
                f_name, f_type = _p_exp.FUNCTION_NAME, _p_exp.FUNCTION_TYPE
                loss = get_loss_function(f_name, 'basic', False)()

                if f_type.startswith('entr'):
                    _wrapper = get_loss_function(f_name, f_type, False)
                    loss = _wrapper(loss, KNN)

                if f_type == 'collective':
                    _loss = get_loss_function(f_name, f_type, False)
                    loss = _loss(KNN, 0.5)

                # Set-up the network
                n_layers = len(_p_exp.LAYERS)
                predefined_layers = _p_exp.LAYERS.copy()
                if f_type == 'entr_r':
                    predefined_layers.append(2)
                layers = [x.shape[1]] + predefined_layers + [1]
                model = CustomNeuralNetwork(layers, _p_exp.HIDDEN_ACTIVATIONS,
                                            _p_exp.OUTPUT_ACTIVATIONS)
                optimizer = Adam(model.parameters(), lr=_p_exp.LEARNING_RATE)

                # Generate experiment name
                unified_dataset_name = simplify_dataset_name(_p_data.DATASET)
                experiment_name = f'{unified_dataset_name}_{f_name}_{f_type}'
                _debug(f'Experiment name: {experiment_name}')
                f_type_enum = LossFuncType.from_string(f_type)

                # Neptune.ai
                experiment = None
                if args.neptune:
                    params = {**_p_data, **_p_nn, **_p_exp}
                    params['N_LAYERS'] = n_layers
                    tags = [f_name, f_type, unified_dataset_name]

                    experiment = neptune.create_experiment(
                        name=experiment_name, tags=tags, params=params,
                        upload_source_files=[args.config, 'src/losses/*.py',
                                             __file__])

                # Training phrase
                logger = neptune if args.neptune else None
                model, training_loss_history, validation_loss_history = run(
                    experiment_name, optimizer, loss, model, train_dataloader,
                    _p_exp.EPOCHS, valid_dataloader, test_data_x=test_data_x,
                    test_data_y=test_data_y, eval_freq=_p_exp.EVAL,
                    early_stopping=_p_exp.EARLY_STOPPING, neptune_logger=logger,
                    knn_use_indices=True, loss_type=f_type_enum)

                # Evaluation phrase
                metrics = evaluate_binary(model, test_data_x, test_data_y)
                _debug(f'Done! Metrics: \n{metrics}\n')

                if args.neptune:
                    for metric, value in metrics.items():
                        neptune.log_metric(f'final_{metric}', value)
                    experiment.stop()


if __name__ == '__main__':

    args = parse_arguments()
    project_id = 'clfmsc2020/experimentsv2' if not args.debug \
        else 'clfmsc2020/debugv2'

    if args.debug or args.verbose:
        print_logo()
        print('--- [Given arguments] ---')
        for key, value in vars(args).items():
            print(f'  {key}: {value}')
        print()

    run_experiments(args, project_id)
