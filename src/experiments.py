from __future__ import annotations

import os
from enum import Enum
from typing import Dict, Any, Tuple, Callable, Union

import yaml
from torch import Tensor

import src.losses as loss
from src.utils import dotdict

# Aliases
LossFunction = Callable[[Tensor, Tensor], Tensor]


class LossFuncType(Enum):
    """ A simple enumerator of loss function types """
    BASIC = 0,      # Basic loss function
    ENTR_W = 1,     # Entropy-weighted
    ENTR_R = 2,     # Entropy-regularized
    CLF = 3         # Collective

    @staticmethod
    def from_string(value: str) -> LossFuncType:
        """ Converts string type of funtion into enum """

        mapping = {
            'basic': LossFuncType.BASIC,
            'entr_r': LossFuncType.ENTR_R,
            'entr_w': LossFuncType.ENTR_W,
            'collective': LossFuncType.CLF
        }
        return mapping[value]

    @staticmethod
    def to_string(value: LossFuncType):
        """ Converts LossFuncType into string """

        mapping = {
            LossFuncType.BASIC: 'basic',
            LossFuncType.ENTR_R: 'entr_r',
            LossFuncType.ENTR_W: 'entr_w',
            LossFuncType.CLF: 'collective'
        }
        return mapping[value]


def get_loss_function(fname: str, ftype: Union[LossFuncType, str]
                      ) -> Tuple[bool, LossFunction]:
    """ Retruns loss function of the given name and type and boolean
        value that determine target range of the criterion. """

    _std_range_funcs = set(['bce'])
    _str_func_mapping = {
        'hinge_loss': {
            ftype.BASIC: loss.HingeLoss,
            ftype.CLF: loss.CollectiveHingeLoss
        },
        'sq_hinge_loss': {
            ftype.BASIC: loss.SquaredHingeLoss,
            ftype.CLF: loss.CollectiveSquaredHingeLoss
        },
        'bce': {
            ftype.BASIC: loss.BinaryCrossEntropy,
            ftype.CLF: loss.CollectiveBinaryCrossEntropy
        },
        'log_loss': {
            ftype.BASIC: loss.LogisticLoss,
            ftype.CLF: loss.CollectiveLogisticLoss
        },
        'exp_loss': {
            ftype.BASIC: loss.ExponentialLoss,
            ftype.CLF: loss.CollectiveExponentialLoss
        }
    }

    if isinstance(ftype, str):
        ftype = LossFuncType.from_string(ftype)

    extended_target_range = fname not in _std_range_funcs
    functon = _str_func_mapping[fname][ftype]

    return extended_target_range, functon


def load_config_file(path: str) -> Tuple[dict, dict, dict]:
    """ Loads and converts YAML config into dotdict """

    if not os.path.isfile(path):
        raise FileNotFoundError('Invalid config file path!')

    def _item_to_list(values: Dict[str, Any]) -> Dict[str, list]:
        return {key: [value] if not isinstance(value, list) else value
                for key, value in values.items()}

    with open(path, 'r') as file:
        parameters = yaml.load(file, Loader=yaml.Loader)

    data_params, exp_params, knn_params = parameters['DATA'], \
        parameters['KNN'], parameters['EXPERIMENT']

    return _item_to_list(data_params), _item_to_list(exp_params), \
        _item_to_list(knn_params)
