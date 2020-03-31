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

# Task
Wybrać z UCI MLR 5 zbiorów danych do klasyfikacji binarnej + prosty interfejs odczytu

```python
%load_ext autoreload
%autoreload 2
```

```python
import sys
sys.path.insert(0, '..')
```

```python
from src.data.uci.database import UCIDatabase
```

```python
bin_classifiation_datasets = [
    'Phishing Websites',
    'Breast Cancer Wisconsin (Diagnostic)',
    'Bank Marketing',
    'Dota2 Games Results',
    'Adult'
]
```

```python
db = UCIDatabase(output_directory='../datasets', verbose=True)
```

```python
for dataset_name in bin_classifiation_datasets:
    dataset = db.get(lambda x: x.name == dataset_name, first_only=True)
    data = dataset.to_df()
    print(f'Dataset: {dataset.name} | Shape: {data.shape}')
```

```python

```
