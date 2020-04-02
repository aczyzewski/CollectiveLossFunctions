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

# Data preprocessing pipeline

```python
%load_ext autoreload
%autoreload 2
```

```python
import sys
sys.path.insert(0, '..')
```

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from pandas.api.types import is_numeric_dtype, is_string_dtype

from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
```

```python
from src.datasets import UCIDatabase
datasets_db = UCIDatabase(output_directory='../datasets', verbose=True)
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
for ds_name in bin_classifiation_datasets:
    
    x, y = datasets_db.get(lambda x: x.name == ds_name, first_only=True).load()
    
    # Numeric features
    numeric_features = [column for column in x.columns if is_numeric_dtype(x[column])]
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # CHECK
        ('scaler', StandardScaler()),
        ('var_thr', VarianceThreshold(0.5))
    ])

    # Categorical features
    categorical_features = [column for column in x.columns if is_string_dtype(x[column])]
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # CHECK
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine them together
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
    ])

    x = preprocessor.fit_transform(x)
```

```python

```
