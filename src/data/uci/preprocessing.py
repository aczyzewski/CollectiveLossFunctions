import os
from os.path import join as joinpath

import gzip
import zipfile
import pandas as pd
import numpy as np

from .utils import load_arff_file, iterfile

# Important: Very last column of each dataset is the target column.

def phishing_websites(location: str) -> pd.DataFrame:
    """ Phishing Websites """
    
    filename = 'Training Dataset.arff'
    data, _ = load_arff_file(joinpath(location, filename))
    return pd.DataFrame(data)

def breast_cancer_wisconsin_diag(location: str) -> pd.DataFrame:
    """ Breast Cancer Wisconsin (Diagnostic) """
    
    columns = ['id', 'target']
    data_type = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']

    for prefix in ['mean', 'sd', 'worst']:
        for dtype in data_type:
            columns.append(f'{prefix}_{dtype}')
            
    return pd.read_csv(joinpath(location, 'wdbc.data'), names=columns)

def bank_marketing(location: str) -> pd.DataFrame:
    """ Bank Marketing """
    
    data = joinpath(location, 'data')
    files = ['bank.zip', 'bank-additional.zip']
    for file in files:
        with zipfile.ZipFile(joinpath(location, file), 'r') as zipref:
            zipref.extractall(data)
    return pd.read_csv(joinpath(data, 'bank-additional', 'bank-additional.csv'), sep=';')

def higgs(location: str) -> pd.DataFrame:
    """ HIGGS """
    
    columns = ["target", "lepton pT", "lepton eta", "lepton phi", "missing energy magnitude", "missing energy phi", \
           "jet 1 pt", "jet 1 eta", "jet 1 phi", "jet 1 b-tag", "jet 2 pt", "jet 2 eta", "jet 2 phi", "jet 2 b-tag", \
           "jet 3 pt", "jet 3 eta", "jet 3 phi", "jet 3 b-tag", "jet 4 pt", "jet 4 eta", "jet 4 phi", "jet 4 b-tag", \
           "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"]

    gzfile = joinpath(location, 'HIGGS.csv.gz')
    with gzip.open(gzfile) as f:
        features_train = pd.read_csv(f, names=columns)
    return features_train[columns[1:] + [columns[0]]]

def adult(location: str) -> pd.DataFrame:
    """ Adult """
    
    columns = []
    for idx, row in enumerate(iterfile(joinpath(location, 'adult.names'), filters=[lambda x: x.startswith('|')])):
        if idx == 0: continue
        column, *_ = row.split(':')
        columns.append(column)
    columns.append('target')
    
    test_data = pd.read_csv(joinpath(location, 'adult.test'), skiprows=1, names=columns)
    test_data.dropna(inplace=True)
    test_data.replace(r'^\s+\?$', np.nan, regex=True, inplace=True)
    
    data = pd.read_csv(joinpath(location, 'adult.data'), names=columns)
    data.replace(r'^\s+\?$', np.nan, regex=True, inplace=True)
    
    return pd.concat((data, test_data))

    