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
from src.datasets import UCIDatabase
from src.preprocessing import basic_pipeline
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
    x, y = basic_pipeline(x, y)
```
