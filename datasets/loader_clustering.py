# ===============================================================================
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_svmlight_file
from sklearn.model_selection import train_test_split

from .loader_utils import retrieve


def epsilon_50K(dataset_dir: Path) -> bool:
    """
    Epsilon dataset
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html

    Classification task. n_classes = 2.
    epsilon_50K x cluster dataset (50000, 2001)
    """
    dataset_name = 'epsilon_50K'
    os.makedirs(dataset_dir, exist_ok=True)

    url= 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary' \
                '/epsilon_normalized.bz2'
    local_url = os.path.join(dataset_dir, os.path.basename(url))

    num_train = 50000
    if not os.path.isfile(local_url):
        logging.info(f'Started loading {dataset_name}')
        retrieve(url, local_url)
    logging.info(f'{dataset_name} is loaded, started parsing...')
    X_train, y_train = load_svmlight_file(local_url,
                                        dtype=np.float32)

    X_train = X_train.toarray()[:num_train]
    y_train = y_train[:num_train]
    y_train[y_train <= 0] = 0

    filename = f'{dataset_name}_{name}.npy'
    np.save(os.path.join(dataset_dir, filename), data)
    return True


def cifar_cluster(dataset_dir: Path) -> bool:
    """
    Cifar dataset from LIBSVM Datasets (
    https://www.cs.toronto.edu/~kriz/cifar.html#cifar)
    TaskType: Classification
    cifar x cluster dataset (50000, 3073)
    """
    dataset_name = 'cifar'
    os.makedirs(dataset_dir, exist_ok=True)

    url= 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/cifar10.bz2'
    local_url = os.path.join(dataset_dir, os.path.basename(url))

    if not os.path.isfile(local_url):
        logging.info(f'Started loading {dataset_name}')
        retrieve(url, local_url)
    logging.info(f'{dataset_name} is loaded, started parsing...')
    X_train, y_train = load_svmlight_file(local_url,
                                        dtype=np.float32)

    X_train = X_train.toarray()
    y_train[y_train <= 0] = 0

    for data, name in zip((X_train, y_train),
                          ('x_train', 'y_train')):
        filename = f'{dataset_name}_{name}.npy'
        np.save(os.path.join(dataset_dir, filename), data)
    return True


def hepmass_10K(dataset_dir: Path) -> bool:
    """
    HEPMASS dataset from UCI machine learning repository (
    https://archive.ics.uci.edu/ml/datasets/HEPMASS).
    
    Cludtering task. n_classes = 2.
    hepmass_10K X cluster dataset (10000, 29)
    """
    dataset_name = 'hepmass_10K'
    os.makedirs(dataset_dir, exist_ok=True)

    url_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00347/all_train.csv.gz'

    local_url_train = os.path.join(dataset_dir, os.path.basename(url_train))

    if not os.path.isfile(local_url_train):
        logging.info(f'Started loading {dataset_name}, train')
        retrieve(url_train, local_url_train)
    logging.info(f'{dataset_name} is loaded, started parsing...')

    nrows_train, dtype = 10000, np.float32
    data_train: Any = pd.read_csv(local_url_train, delimiter=",",
                            compression="gzip", dtype=dtype,
                            nrows=nrows_train)

    x_train = np.ascontiguousarray(data_train.values[:nrows_train, 1:], dtype=dtype)
    y_train = np.ascontiguousarray(data_train.values[:nrows_train, 0], dtype=dtype)

    for data, name in zip((x_train, y_train),
                          ('x_train', 'y_train')):
        filename = f'{dataset_name}_{name}.npy'
        np.save(os.path.join(dataset_dir, filename), data)
    logging.info(f'dataset {dataset_name} is ready.')
    return True
