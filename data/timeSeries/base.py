import numpy as np
nax = np.newaxis
import os
import scipy.linalg

from .datasets import *


def load_data(name):
    if name == 'airline':
        X, y = airline.load_X_y()
    elif name == 'eeg_single':
        X, y = eeg.load_one_channel()
    elif name == 'eeg_all':
        X, y = eeg.load_all_channels()
    elif name == 'methane':
        X, y = methane.read_data()
    elif name == 'sea_level_monthly':
        X, y = sea_level.get_X_y('monthly')
    elif name == 'sea_level_annual':
        X, y = sea_level.get_X_y('annual')
    elif name == 'solar':
        X, y = solar.get_X_y()
    elif name == 'mauna':
        X, y = mauna.get_Xy()
    else:
        fname = 'data/%s.mat' % name
        if not os.path.exists(fname):
            raise RuntimeError("Couldn't find dataset: %s" % name)
        X, y = scipy.io.loadmat(fname)

    # make sure X and y are in matrix form
    if X.ndim == 1:
        X = X[:, nax]
    if y.ndim == 1:
        y = y[:, nax]

    return X, y

if __name__ == '__main__':
    load_data('airline')
