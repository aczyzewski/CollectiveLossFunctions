import os
from os.path import join as joinpath
from typing import Tuple

import gzip
import zipfile
import pandas as pd
import numpy as np

from .utils import load_arff_file, iterfile
from .decorators import binary_dataset_loader, regression_dataset_loader


@binary_dataset_loader
def _phishing_websites(location: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Phishing Websites """
    
    # Load arff file
    filename = 'Training Dataset.arff'
    data, _ = load_arff_file(joinpath(location, filename))

    # Split data into x, y
    df = pd.DataFrame(data).astype('int')
    x, y = df.iloc[:,: -1], df.iloc[:, -1]

    # Convert target from [-1, 1] to [0, 1]
    y = y.apply(lambda x: 0 if x < 0 else 1)

    return x, y


@binary_dataset_loader
def _breast_cancer_wisconsin_diag(location: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Breast Cancer Wisconsin (Diagnostic) """
    
    columns = ['id', 'target']
    data_type = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']

    # Compute proper names of the columns
    for prefix in ['mean', 'sd', 'worst']:
        for dtype in data_type:
            columns.append(f'{prefix}_{dtype}')
    
    # Read dataframe and split data into x, y
    df = pd.read_csv(joinpath(location, 'wdbc.data'), names=columns)
    y, x = df.pop('target'), df

    # Convert target into [0, 1]
    conv_to_class = {'M': 0, 'B': 1} 
    y = y.apply(lambda x: conv_to_class[x])

    return x, y


@binary_dataset_loader
def _bank_marketing(location: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Bank Marketing """
    
    # Unpack files
    data = joinpath(location, 'data')
    files = ['bank.zip', 'bank-additional.zip']
    if not os.path.isfile(joinpath(data, 'bank-additional', 'bank-additional.csv')):
        for file in files:
            with zipfile.ZipFile(joinpath(location, file), 'r') as zipref:
                zipref.extractall(data)

    # Read and split the data into x, y
    df = pd.read_csv(joinpath(data, 'bank-additional', 'bank-additional.csv'), sep=';')
    x, y = df.iloc[:,: -1], df.iloc[:, -1]

    # Convert the target into [0, 1]
    conv_to_class = {'no': 0, 'yes': 1}
    y = y.apply(lambda x: conv_to_class[x])

    return x, y


@binary_dataset_loader
def _adult(location: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Adult """
    
    # Read the data
    columns = []
    for idx, row in enumerate(iterfile(joinpath(location, 'adult.names'), filters=[lambda x: x.startswith('|')])):
        if idx == 0: continue
        column, *_ = row.split(':')
        columns.append(column.strip())
    columns.append('target')
    
    # Read train data
    train_data = pd.read_csv(joinpath(location, 'adult.data'), sep=', ', names=columns, engine='python')
    train_data.replace(r'^\s+\?$', np.nan, regex=True, inplace=True)

    # Read test data
    test_data = pd.read_csv(joinpath(location, 'adult.test'), sep=', ', skiprows=1, names=columns, engine='python')
    test_data.dropna(inplace=True)
    test_data.replace(r'^\s+\?$', np.nan, regex=True, inplace=True)
    
    # Split data into x, y
    df = pd.concat((train_data, test_data))

    # 48k rows -> 43k rows
    df =  df.dropna() 
    x, y = df.iloc[:,: -1], df.iloc[:, -1]

    # Convert target into [0, 1]
    conv_to_class = {'<=50K': 0, '<=50K.': 0, '>50K': 1, '>50K.': 1}
    y = y.apply(lambda x: conv_to_class[x.strip()])

    return x, y


@binary_dataset_loader
def _skin_segmentation(location: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Skin Segmentation """

    # Load the data and split into x, y
    datafile = 'Skin_NonSkin.txt'
    df = pd.read_csv(joinpath(location, datafile), sep='\t')
    x, y = df.iloc[:,: -1], df.iloc[:, -1]

    # Shift the target from [1, 2] to [0, 1]
    y = y - 1

    return x, y


@regression_dataset_loader
def _online_news_popularity(location: str):
    """ Online News Popularity """ 

    # Unzip the files
    unzip_directory = joinpath(location, 'data')
    with zipfile.ZipFile(joinpath(location, 'OnlineNewsPopularity.zip'), 'r') as zipref:
        zipref.extractall(unzip_directory)
        
    # Read and split the data into x, y
    data = pd.read_csv(joinpath(unzip_directory, 'OnlineNewsPopularity', 'OnlineNewsPopularity.csv'))
    x, y = data.iloc[:,1: -1], data.iloc[:, -1]
    
    return x, y 
