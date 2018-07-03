#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.cluster import MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft

__all__ = [
    'get_kemans_init',
    'median_distance_local',
    'median_distance_global',
    'mean_distance_global',
    'mean_distance_local',
]


def get_kemans_init(x, k_centers):
    if x.shape[0] > 10000:
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation[:10000]]

    kmeans = MiniBatchKMeans(n_clusters=k_centers, batch_size=k_centers*10).fit(x)
    return kmeans.cluster_centers_


def median_distance_global(x):
    """
    get the median of distances between x.
    :param x: shape of [n, d]
    :return: float
    """
    if x.shape[0] > 10000:
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation[:10000]]
    x_col = np.expand_dims(x, 1)
    x_row = np.expand_dims(x, 0)
    dis_a = np.sqrt(np.sum((x_col - x_row) ** 2, -1)) # [n, n]
    return np.median(dis_a)

def mean_distance_global(x):
    """
    get the median of distances between x.
    :param x: shape of [n, d]
    :return: float
    """
    if x.shape[0] > 10000:
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation[:10000]]
    x_col = np.expand_dims(x, 1)
    x_row = np.expand_dims(x, 0)
    dis_a = np.sqrt(np.sum((x_col - x_row) ** 2, -1)) # [n, n]
    return np.mean(dis_a)


def median_distance_local(x):
    """
    get the median of distances between x.
    :param x: shape of [n, d]
    :return: float
    """
    if x.shape[0] > 10000:
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation[:10000]]
    x_col = np.expand_dims(x, 1)
    x_row = np.expand_dims(x, 0)
    dis_a = np.abs(x_col - x_row) # [n, n, d]
    dis_a = np.reshape(dis_a, [-1, dis_a.shape[-1]])
    return np.median(dis_a, 0) * (x.shape[1] ** 0.5)

def mean_distance_local(x):
    """
    get the median of distances between x.
    :param x: shape of [n, d]
    :return: float
    """
    if x.shape[0] > 10000:
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation[:10000]]
    x_col = np.expand_dims(x, 1)
    x_row = np.expand_dims(x, 0)
    dis_a = np.abs(x_col - x_row) # [n, n, d]
    dis_a = np.reshape(dis_a, [-1, dis_a.shape[-1]])
    return np.mean(dis_a, 0)


def _test_median_global():
    x = np.random.normal(size=[20000, 2])
    print(median_distance_global(x))

def _test_median_local():
    x = np.eye(20)
    print(median_distance_local(x))

def FFT(func):
    # import matplotlib.pyplot as plt
    # t = np.arange(10)
    # sp = np.fft.rfft(np.sin(t))
    # freq = np.fft.rfftfreq(t.shape[-1])
    # plt.plot(freq, sp.real)
    # plt.show(block=True)
    #

    N = 60
    # sample spacing
    T = 2
    x = np.linspace(- N * T / 2, N * T / 2, N)
    yf = np.fft.rfft(func(x))
    xf = np.fft.rfftfreq(N)

    plt.scatter(xf, np.abs(yf))
    plt.show()

if __name__ == '__main__':
    FFT(lambda x: np.cos(x / 2))
    # FFT(lambda x: np.exp(-(x)**2 / 1.))
    # FFT(lambda x:np.sin(50.0 * 2.0 * np.pi * x))