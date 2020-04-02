from typing import Callable, Tuple

import pandas as pd
import pandas.api.types as ptypes


def binary_dataset_loader(func: Callable[[str], Tuple[pd.DataFrame, pd.DataFrame]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Decorator designed to check outputs of binary datasets loaders """

    def wrapper(*args, **kwargs):

        x, y = func(*args, **kwargs)
        target_values = y.unique()
        num_target_values = len(target_values)

        assert num_target_values == 2, f'{func.__name__}: Target column has more than 2 unique values!'
        assert ptypes.is_numeric_dtype(y), f'{func.__name__}: Target column is not numeric!'
        assert 0 in target_values and 1 in target_values, f'{func.__name__}: Target columns has different values than 0 and 1!'

        return x, y

    return wrapper