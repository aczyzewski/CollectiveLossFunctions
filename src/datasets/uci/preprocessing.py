import os
from os.path import join as joinpath
from typing import Tuple

import zipfile
import pandas as pd
import numpy as np

from .utils import load_arff_file, iterfile
from .decorators import binary_dataset_loader, regression_dataset_loader

# Custom aliases
Dataset = Tuple[pd.DataFrame, pd.DataFrame]


@binary_dataset_loader
def _phishing_websites(location: str) -> Dataset:
    """ Phishing Websites """

    # Load arff file
    filename = 'Training Dataset.arff'
    data, _ = load_arff_file(joinpath(location, filename))

    # Split data into x, y
    df = pd.DataFrame(data).astype('int')
    x, y = df.iloc[:, :-1], df.iloc[:, -1]

    # Convert target from [-1, 1] to [0, 1]
    y = y.apply(lambda x: 0 if x < 0 else 1)

    return x, y


@binary_dataset_loader
def _breast_cancer_wisconsin_diag(location: str) -> Dataset:
    """ Breast Cancer Wisconsin (Diagnostic) """

    columns = ['id', 'target']
    data_type = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
                 'compactness', 'concavity', 'concave_points', 'symmetry',
                 'fractal_dimension']

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
def _bank_marketing(location: str) -> Dataset:
    """ Bank Marketing """

    # Unpack files
    data = joinpath(location, 'data')
    files = ['bank.zip', 'bank-additional.zip']
    datafile_path = joinpath(data, 'bank-additional', 'bank-additional.csv')
    if not os.path.isfile(datafile_path):
        for file in files:
            with zipfile.ZipFile(joinpath(location, file), 'r') as zipref:
                zipref.extractall(data)

    # Read and split the data into x, y
    df = pd.read_csv(datafile_path, sep=';')
    x, y = df.iloc[:, : -1], df.iloc[:, -1]

    # Convert the target into [0, 1]
    conv_to_class = {'no': 0, 'yes': 1}
    y = y.apply(lambda x: conv_to_class[x])

    return x, y


@binary_dataset_loader
def _adult(location: str) -> Dataset:
    """ Adult """

    # Read the data
    columns = []
    lines = iterfile(joinpath(location, 'adult.names'),
                     filters=[lambda x: x.startswith('|')])

    for idx, row in enumerate(lines):
        if idx == 0:
            continue
        column, *_ = row.split(':')
        columns.append(column.strip())
    columns.append('target')

    # Read train data
    train_data = pd.read_csv(joinpath(location, 'adult.data'), sep=', ',
                             names=columns, engine='python')
    train_data.replace(r'^\s+\?$', np.nan, regex=True, inplace=True)

    # Read test data
    test_data = pd.read_csv(joinpath(location, 'adult.test'), sep=', ',
                            skiprows=1, names=columns, engine='python')
    test_data.dropna(inplace=True)
    test_data.replace(r'^\s+\?$', np.nan, regex=True, inplace=True)

    # Split data into x, y
    df = pd.concat((train_data, test_data), axis=0)

    # 48k rows -> 43k rows
    df = df.dropna()
    x, y = df.iloc[:, :-1], df.iloc[:, -1]

    # Convert target into [0, 1]
    conv_to_class = {'<=50K': 0, '<=50K.': 0, '>50K': 1, '>50K.': 1}
    y = y.apply(lambda x: conv_to_class[x.strip()])

    return x, y


@binary_dataset_loader
def _skin_segmentation(location: str) -> Dataset:
    """ Skin Segmentation """

    # Load the data and split into x, y
    datafile = 'Skin_NonSkin.txt'
    df = pd.read_csv(joinpath(location, datafile), sep='\t')
    x, y = df.iloc[:, :-1], df.iloc[:, -1] - 1

    return x, y


@regression_dataset_loader
def _online_news_popularity(location: str) -> Dataset:
    """ Online News Popularity """

    # Unzip the files
    unzip_directory = joinpath(location, 'data')
    zipfile_path = joinpath(location, 'OnlineNewsPopularity.zip'),
    datafile_path = joinpath(unzip_directory, 'OnlineNewsPopularity',
                             'OnlineNewsPopularity.csv')

    with zipfile.ZipFile(zipfile_path, 'r') as zipref:
        zipref.extractall(unzip_directory)

    # Read and split the data into x, y
    data = pd.read_csv(datafile_path)
    x, y = data.iloc[:, 1: -1], data.iloc[:, -1]

    return x, y


@regression_dataset_loader
def _bike_sharing(location: str) -> Dataset:
    """ Bike Sharing Dataset """

    # Unzip files
    unzip_path = joinpath(location, 'data')
    zipfile_path = joinpath(location, 'Bike-Sharing-Dataset.zip')

    with zipfile.ZipFile(zipfile_path, 'r') as zipref:
        zipref.extractall(unzip_path)

    # Remove redundant columns
    data = pd.read_csv(joinpath(unzip_path, 'hour.csv'))
    data = data.drop(columns=['instant', 'dteday', 'casual', 'registered'])
    x, y = data.iloc[:, :-1], data.iloc[:, -1]

    return x, y


@regression_dataset_loader
def _optical_interconnection_network(location) -> Dataset:
    """ Optical Interconnection Network """

    datafile_path = joinpath(location, 'optical_interconnection_network.csv')
    data = pd.read_csv(datafile_path, sep=';', decimal=',')
    data = data.iloc[:, :-5]
    x, y = data.iloc[:, :-1], data.iloc[:, -1]

    return x, y


@regression_dataset_loader
def _communities_and_crime(location) -> Dataset:
    """ Communities and Crime """

    # Read data and retrieve columns names
    datafile_path = joinpath(location, 'communities.names')
    content = iterfile(datafile_path,
                       filters=[lambda x: not x.startswith('@attribute')])

    columns = list(map(lambda x: x.split(' ')[1], content))
    data = pd.read_csv(joinpath(location, 'communities.data'), header=None)
    data.columns = columns
    data = data.replace('?', np.nan)

    # Redundant columns
    columns_to_delete = ['fold']

    # Remove columns with more than 20% null values (24 columns)
    for column_name, null_values in zip(data.columns, data.isna().sum()):
        null_ratio = null_values / data.shape[0]
        if null_ratio > 0.2:
            columns_to_delete.append(column_name)
    data = data.drop(columns=columns_to_delete)

    # Remove rows with null values (1 row)
    data = data.dropna()

    # Split
    x, y = data.iloc[:, :-1], data.iloc[:, -1]

    return x, y


@regression_dataset_loader
def _blogfeedback(location: str) -> Dataset:
    """ BlogFeedback """

    # Unzip files
    unzip_path = joinpath(location, 'data')
    datafile_path = joinpath(location, 'BlogFeedback.zip')
    with zipfile.ZipFile(datafile_path, 'r') as zipref:
        zipref.extractall(unzip_path)

    # Merge all csv files
    csv_files = filter(lambda x: x.endswith('.csv'), os.listdir(unzip_path))
    paths = list(map(lambda x: joinpath(unzip_path, x), csv_files))
    data = pd.read_csv(paths[0], header=None)
    for csv in paths[1:]:
        single_csv_data = pd.read_csv(csv, header=None)
        data = pd.concat((data, single_csv_data), axis=0)

    # Split
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    return x, y
