
from typing import Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def basic_pipeline(x: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Transforms inuput data """

    # Numeric features
    numeric_features = [column for column in x.columns if is_numeric_dtype(x[column])]
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('var_thr', VarianceThreshold(0.02))
    ])

    # Categorical features
    categorical_features = [column for column in x.columns if is_string_dtype(x[column])]
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine them together
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
    ])

    x = preprocessor.fit_transform(x)
    return x, y