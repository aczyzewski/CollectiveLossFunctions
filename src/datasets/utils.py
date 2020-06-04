from typing import List, Tuple, Callable
import numpy as np
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from src.datasets import UCIDatabase, generic

# Custom aliases
Pipeline = Callable[[np.array, np.array], Tuple[np.array, np.array]]


DATABASE = UCIDatabase(output_directory='datasets')
DATASETS = {
    'binary': [
        'Phishing Websites',
        'Breast Cancer Wisconsin (Diagnostic)',
        'Bank Marketing',
        'Adult',
        'Skin Segmentation'
    ],

    'multiclass': [],

    'regression': [
        'Online News Popularity',
        'Bike Sharing Dataset',
        'Optical Interconnection Network',
        'Communities and Crime',
        'BlogFeedback'
    ]
}

GENERIC_DATASETS = {
    'two_spirals': [generic.make_two_spirals, {'n_points': 3200, 'noise': 0.15}],
    'checkerboard': [generic.make_checkerboard, {'n_points': 3200, 'noise': 0.03}]
}

def get_datasets(category: str) -> List[str]:
    """ Returns the list of available datastes of a given category """

    if category not in DATASETS.keys():
        raise KeyError(f'Dataset category not found. \
            Available catgories are: {", ".join(DATASETS.keys())}')

    return DATASETS[category]


def load_dataset(name: str, preprocesing: Pipeline = None, **kwargs
                 ) -> Tuple[np.array, np.array]:
    """ Loads dataset from a disk """

    # Generic datasets
    if name in GENERIC_DATASETS.keys():
        print('Warning: Generic dataset.')
        method, params = GENERIC_DATASETS[name]
        x, y = method(**params)
        return x, y

    # UCI dataset
    x, y = DATABASE.get_by_name(name).load()
    if preprocesing is None:
        print('Warning: preprocessing method wasn\'t defined! Loading RAW data.')
    else:
        x, y = preprocesing(x, y, **kwargs)
    return x, y


def convert_to_dataloader(x: np.array, y: np.array, startidx: int = 0, **kwargs,
                          ) -> List[str]:
    """ Converts a given dataset into DataLoader object """

    indices = np.arange(x.shape[0]) + startidx
    idx, x, y = Tensor(indices), Tensor(x), Tensor(y)
    dataset = TensorDataset(idx, x, y)
    return DataLoader(dataset, **kwargs)


def simplify_dataset_name(value: str) -> str:

    translation = {
        ord(' '): ord('_'),
        ord('('): None,
        ord(')'): None
    }

    return value.translate(translation).lower()
