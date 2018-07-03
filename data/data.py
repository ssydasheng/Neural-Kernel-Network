import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

import numpy as np
import gpflowSlim as gpfs
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.image as mpimg
import copy

from .hparams import HParams
from .register import register

DATA_PATH = os.path.join(root_path, 'data/DATA')
DATASETS = dict(
    boston='housing.data',
    concrete='concrete.data',
    energy='energy.data',
    kin8nm='kin8nm.data',
    naval='naval.data',
    power_plant='power_plant.data',
    wine='wine.data',
    yacht='yacht_hydrodynamics.data',
)


def pca_split(data, target=None):
    """
    split data according to the result of pca projecting to 1-d dimension.
    :param data: array of shape [N, d]
    :param target: array of shape [N]
    """
    if data.shape[0] < 10000:
        d = data
    else:
        indices = np.random.permutation(range(data.shape[0]))
        d = data[indices][:10000]

    pca = PCA(n_components=1)
    pca.fit(d)
    projected = pca.transform(data)

    ind = list(np.argsort(projected.squeeze()))
    len_ = len(ind)
    test_ind = np.array(ind[: len_ // 15] + ind[-len_ // 15:])
    train_ind = np.array(ind[len_//15: -len_ // 15])
    test_ind = test_ind[np.random.permutation(len(test_ind))]
    train_ind = train_ind[np.random.permutation(len(train_ind))]

    assert len(test_ind) + len(train_ind) == len_, 'train set and test set should add to the whole set'
    assert set(test_ind) - set(train_ind) == set(test_ind), 'train set and test set should be exclusive'

    xtrain, ytrain = data[train_ind], target[train_ind]
    xtest, ytest = data[test_ind], target[test_ind]
    return xtrain, xtest, ytrain, ytest


@register('uci_woval_pca')
def load_uci_pca(dataset_name, seed=1):
    """Split train/test based on PCA principle direction."""
    data = np.loadtxt(os.path.join(DATA_PATH, DATASETS[dataset_name]))
    data = data.astype(gpfs.settings.float_type)

    x, y = data[:, :-1], data[:, -1]
    x_t, x_v, y_t, y_v = pca_split(x, y)

    x_t, x_v, _, _ = standardize(x_t, x_v)
    y_t, y_v, _, train_std = standardize(y_t, y_v)
    hparams = HParams(
        x_train=x_t,
        x_test=x_v,
        y_train=y_t,
        y_test=y_v,
        std_y_train=train_std
    )
    return hparams


@register('uci_woval_pca_train_val_split')
def load_uci_woval_pca_train_val_split(dataset_name, seed=1):
    """Split sub-train/valid based on PCA principle direction,
    after spliting train/test based on PCA."""
    data = np.loadtxt(os.path.join(DATA_PATH, DATASETS[dataset_name]))
    data = data.astype(gpfs.settings.float_type)

    x, y = data[:, :-1], data[:, -1]
    x_t, x_v, y_t, y_v = pca_split(x, y)
    x_t, x_v, y_t, y_v = pca_split(x_t, y_t)

    x_t, x_v, _, _ = standardize(x_t, x_v)
    y_t, y_v, _, train_std = standardize(y_t, y_v)
    hparams = HParams(
        x_train=x_t,
        x_test=x_v,
        y_train=y_t,
        y_test=y_v,
        std_y_train=train_std
    )
    return hparams


@register('uci_woval')
def load_uci(dataset_name, seed=1):
    """Split train/test randomly."""
    data = np.loadtxt(os.path.join(DATA_PATH, DATASETS[dataset_name]))
    data = data.astype(gpfs.settings.float_type)

    x, y = data[:, :-1], data[:, -1]
    x_t, x_v, y_t, y_v = train_test_split(x, y, test_size=.1, random_state=seed)

    x_t, x_v, _, _ = standardize(x_t, x_v)
    y_t, y_v, _, train_std = standardize(y_t, y_v)
    hparams = HParams(
        x_train=x_t,
        x_test=x_v,
        y_train=y_t,
        y_test=y_v,
        std_y_train=train_std
    )
    return hparams


def standardize(data_train, *args):
    """
    Standardize a dataset to have zero mean and unit standard deviation.

    :param data_train: 2-D Numpy array. Training data.
    :param data_test: 2-D Numpy array. Test data.

    :return: (train_set, test_set, mean, std), The standardized dataset and
        their mean and standard deviation before processing.
    """
    std = np.std(data_train, 0, keepdims=True)
    std[std == 0] = 1
    mean = np.mean(data_train, 0, keepdims=True)
    data_train_standardized = (data_train - mean) / std
    output = [data_train_standardized]
    for d in args:
        dd = (d - mean) / std
        output.append(dd)
    output.append(mean)
    output.append(std)
    return output


@register('texture')
def load_texture(img_name):
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    rgb = mpimg.imread(os.path.join(DATA_PATH, img_name+'.png'))
    gray = rgb2gray(rgb)

    nx1, nx2 = gray.shape
    x_train1 = np.arange(nx1) / 224
    x_train2 = np.arange(nx2) / 224
    y_train = copy.copy(gray)
    y_train[60:120, 80:160] = np.random.randn(60, 80) * 1e3
    x_test1 = np.arange(60, 120) / 224
    x_test2 = np.arange(80, 160) / 224
    y_test = gray[60:120, 80:160]
    gt = copy.copy(gray)
    mask = np.zeros_like(gray, dtype=np.int32)
    mask[60:120, 80:160] = 1
    hparams = HParams(
        x_train1=np.expand_dims(x_train1, 1),
        x_train2=np.expand_dims(x_train2, 1),
        y_train=y_train,
        x_test1=np.expand_dims(x_test1, 1),
        x_test2=np.expand_dims(x_test2, 1),
        y_test=y_test,
        gt=gt,
        mask=mask,
    )
    return hparams
