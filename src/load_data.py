import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris


def get_raw_dataset(dataname, train_ratio=0.8):
    if dataname not in ['cancer', 'iris', 'auto', 'gas']:
        raise RuntimeError('Unsupported dataset')
    elif dataname == 'cancer':
        return get_raw_cancer(train_ratio)
    elif dataname == 'iris':
        return get_raw_iris(train_ratio)
    elif dataname == 'auto':
        return get_raw_auto(train_ratio)
    elif dataname == 'gas':
        return get_raw_gas(train_ratio)


def get_processed_dataset(dataname, train_ratio=0.8):
    if dataname not in ['cancer', 'iris', 'auto', 'gas']:
        raise RuntimeError('Unsupported dataset')
    else:
        dataset = np.loadtxt('./data/processed/' + dataname + '.txt')
        np.random.shuffle(dataset)
        train_len = int(dataset.shape[0] * train_ratio)
        trainset = dataset[:train_len, :]
        testset = dataset[train_len:, :]
        return trainset, testset


def get_synthetic_data(rho, train_ratio=0.8):
    if rho not in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, -0.2, -0.4, -0.6, -0.8, -1.0]:
        raise RuntimeError('Unsupported ρ value')
    else:
        dataset = np.loadtxt('./data/synthetic/rho=' + str(rho) + '.txt')
        np.random.shuffle(dataset)
        train_len = int(dataset.shape[0] * train_ratio)
        trainset = dataset[:train_len, :]
        testset = dataset[train_len:, :]
        return trainset, testset


def normalize(X):
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    normalized_X = (X - min_vals) / (max_vals - min_vals)
    return normalized_X


def get_raw_cancer(train_ratio):
    cancer = load_breast_cancer()

    data = cancer.data
    target = cancer.target

    total_len = len(target)
    train_len = int(total_len * train_ratio)

    indices = np.random.permutation(total_len)
    shuffled_x = data[indices]
    shuffled_y = target[indices]

    train_x = shuffled_x[:train_len, :]
    train_y = shuffled_y[:train_len]
    test_x = shuffled_x[train_len:, :]
    test_y = shuffled_y[train_len:]

    return train_x, train_y, test_x, test_y


def get_raw_iris(train_ratio):
    iris = load_iris()

    data = iris.data
    target = iris.target

    total_len = len(target)
    train_len = int(total_len * train_ratio)

    indices = np.random.permutation(total_len)
    shuffled_x = data[indices]
    shuffled_y = target[indices]

    train_x = shuffled_x[:train_len, :]
    train_y = shuffled_y[:train_len]
    test_x = shuffled_x[train_len:, :]
    test_y = shuffled_y[train_len:]

    return train_x, train_y, test_x, test_y


def get_raw_auto(train_ratio):
    auto = np.array(pd.read_csv('./data/raw/auto.csv', usecols=range(7)))
    np.random.shuffle(auto)
    total_len = auto.shape[0]
    train_len = int(total_len * train_ratio)
    train_x = auto[:train_len, 1:]
    train_y = auto[:train_len, 0]
    test_x = auto[train_len:, 1:]
    test_y = auto[train_len:, 0]
    return train_x, train_y, test_x, test_y


def get_raw_gas(train_ratio):
    gas = np.loadtxt('./data/raw/gas_40000.txt')
    total_len = gas.shape[0]
    train_len = int(total_len * train_ratio)
    train_x = gas[:train_len, 2:]
    train_y = gas[:train_len, 0]
    test_x = gas[train_len:, 2:]
    test_y = gas[train_len:, 0]
    return train_x, train_y, test_x, test_y
