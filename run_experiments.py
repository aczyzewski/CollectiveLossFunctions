import warnings
import argparse

import neptune
import numpy as np

import torch
from torch import Tensor
from torch.optim import Adam

import src.datasets as datasets
import src.losses as lossfunc
import src.utils as utils
import src.experiments as helpers
import src.training as trainingloop

from src.experiments import LossFuncType as ftype
from src.neighboorhood import FaissKNN
from src.preprocessing import transform
from src.networks import CustomNeuralNetwork

warnings.filterwarnings('ignore')


def parse_arguments() -> argparse.Namespace:
    """ Handles CLI arguments """

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to config (.yaml) file.')
    parser.add_argument('-d', '--debug', help='Enable debug mode.',
                        action='store_true')
    parser.add_argument('-v', '--verbose', help='Enable print statements.',
                        action='store_true')
    parser.add_argument('-o', '--output', help='Output path.',
                        default='results')
    parser.add_argument('-nn', '--noneptune', dest='useneptune', default=True,
                        action='store_false', help='Disable neptune.ai logger.')
    return parser.parse_args()


def run_experiments(args: argparse.Namespace,
                    neptuneai_project_id: str = 'clfmsc2020/experiments'
                    ) -> None:
    """ Runs experiments """

    def _debug(text: str) -> None:
        """ Prints debugs statemets only if args.debug or arg.verbose
            flag was set to True """
        if args.debug or args.verbose:
            print(f'[INFO] {text}')

    if args.useneptune:
        _debug('Neptune.AI was enabled.')
        neptune.init(neptuneai_project_id)

    _debug(f'Current configuration file path: {args.config}')
    data_yaml_params, knn_yaml_params, exp_yaml_params = \
        helpers.load_config_file(args.config)

    # -- LEVEL 0: CACHE DATASET
    for data_params in utils.iterparams(data_yaml_params):

        x, y = datasets.load_dataset(data_params.DATASET, transform,
                                     use_umap=data_params.USE_UMAP)
        (train_x, train_y), (val_x, val_y), (test_x, test_y) = utils.split_data(
            x, y, val_size=data_params.VAL_SIZE, test_size=data_params.TEST_SIZE)
        _debug(f'Current dataset: {data_params.DATASET}\n')

        # TODO: output_shape = ...

        # -- LEVEL 1: CACHE KNN
        for knn_params in utils.iterparams(knn_yaml_params):

            knn = None
            if knn_params.K is not None and knn_params.K > 0:
                train_val_x = np.concatenate((train_x, val_x), axis=0)
                train_val_y = np.concatenate((train_y, val_y), axis=0)
                knn = FaissKNN(train_val_x, train_val_y,
                               precompute=True, k=knn_params.K)
                _debug(f'kNN wrapper was initialized (k = {knn_params.K}).\n')

            # -- LEVEL 2: RUN EXPERIMENTS
            for exp_params in utils.iterparams(exp_yaml_params):

                # EXCEPTIONS
                if exp_params.FUNCTION_NAME == 'bce' and \
                   exp_params.OUTPUT_ACTIVATIONS == 'tanh':
                    _debug('An exception has occurred (BCE + TanH)')
                    continue

                # Criterion
                criterion = None
                hinge_target_range = False

                criterion_name = exp_params.FUNCTION_NAME
                criterion_type = helpers.LossFuncType.from_string(exp_params.FUNCTION_TYPE)
                n_layers = len(exp_params.LAYERS)
                _debug(f'Criterion: {criterion_name} (type: {criterion_type})')

                if criterion_type == ftype.BASIC:
                    hinge_target_range, loss_function = helpers.get_loss_function(
                        criterion_name, criterion_type)
                    criterion = loss_function()

                elif criterion_type == ftype.ENTR_R:
                    assert knn is not None, 'kNN wrapper is not initialized!'
                    hinge_target_range, base_loss = helpers.get_loss_function(
                        criterion_name, ftype.BASIC)
                    criterion = lossfunc.EntropyRegularizedBinaryLoss(base_loss(), knn)

                elif criterion_type == ftype.ENTR_W:
                    assert knn is not None, 'kNN wrapper is not initialized!'
                    hinge_target_range, base_loss = helpers.get_loss_function(
                        criterion_name, ftype.BASIC)
                    criterion = lossfunc.EntropyWeightedBinaryLoss(base_loss(), knn)

                elif criterion_type == ftype.CLF:
                    assert knn is not None, 'kNN wrapper is not initialized!'
                    hinge_target_range, loss_function = helpers.get_loss_function(
                        criterion_name, ftype.CLF)
                    criterion = loss_function(knn, 0.5)  # FIXME: Fixed params (alpha, beta)

                assert criterion is not None, 'Criterion was not initialized!'

                # Change target range
                target_train_y = np.copy(train_y)
                target_val_y = np.copy(val_y)
                target_test_y = np.copy(test_y)

                if hinge_target_range:
                    _debug('Negative class: 0 -> -1')
                    target_train_y[train_y == 0] = -1
                    target_val_y[val_y == 0] = -1
                    target_test_y[test_y == 0] = -1

                # Convert the subsets into DataLoaders
                train_dataloader = datasets.convert_to_dataloader(
                    train_x, target_train_y, batch_size=data_params.BATCH_SIZE)

                valid_dataloader = datasets.convert_to_dataloader(
                    val_x, target_val_y, batch_size=data_params.BATCH_SIZE,
                    startidx=train_x.shape[0])

                test_data_x, test_data_y = Tensor(test_x), Tensor(test_y)

                # Prepare the experiment
                all_params = {**data_params, **knn_params, **exp_params}
                _debug(f'Paramters: \n {all_params}')

                unified_dataset_name = datasets.simplify_dataset_name(data_params.DATASET)
                experiment_name = f'{unified_dataset_name}_{exp_params.FUNCTION_NAME}_{exp_params.FUNCTION_TYPE}'
                _debug(f'Experiment name: {experiment_name}')

                # Set-up neptue.ai experiment
                experiment = None
                if args.useneptune:
                    tags = [exp_params.FUNCTION_NAME, unified_dataset_name, data_params.PROBLEM,
                            exp_params.FUNCTION_TYPE, exp_params.OUTPUT_ACTIVATIONS]

                    all_params['N_LAYERS'] = n_layers
                    experiment = neptune.create_experiment(
                        name=experiment_name, tags=tags, params=all_params,
                        upload_source_files=[args.config, 'src/losses/*.py', __file__])

                # Input shape
                input_dim = x.shape[1]

                # Layers
                predefined_layers = exp_params.LAYERS.copy()
                if criterion_type == ftype.ENTR_R:
                    predefined_layers.append(2)

                # Output shape (#FIXME)
                output_dim = 1

                layers = [input_dim] + predefined_layers + [output_dim]

                model = CustomNeuralNetwork(layers, exp_params.HIDDEN_ACTIVATIONS,
                                            exp_params.OUTPUT_ACTIVATIONS)
                optimizer = Adam(model.parameters(), lr=exp_params.LEARNING_RATE)

                # Run an experiment
                _debug('Starting the training ...')
                logger = neptune if args.useneptune else None
                model, training_loss_history, validation_loss_history = \
                    trainingloop.run(experiment_name, optimizer, criterion, model,
                                     train_dataloader, exp_params.EPOCHS, valid_dataloader,
                                     test_data_x=test_data_x, test_data_y=test_data_y,
                                     eval_freq=exp_params.EVAL,
                                     early_stopping=exp_params.EARLY_STOPPING,
                                     neptune_logger=logger, knn_use_indices=True,
                                     loss_type=criterion_type)

                # Evaluate the model
                metrics = trainingloop.evaluate_binary(model, test_data_x,
                                                       test_data_y)
                _debug(f'Done! The evaluation results: \n{metrics}\n')

                if args.useneptune:
                    for metric, value in metrics.items():
                        neptune.log_metric(f'final_{metric}', value)
                    experiment.stop()

                # TODO: Save the output on the disk (args.output)


if __name__ == '__main__':
    args = parse_arguments()
    print(args.debug)
    project_id = 'clfmsc2020/experiments' if not args.debug else \
        'aczyzewski/customruns'
    run_experiments(args, project_id)
