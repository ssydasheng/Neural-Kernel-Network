import numpy as np
import abc
import tensorflow as tf
import gpflowSlim as gfs


FLOAT_TYPE = gfs.settings.float_type


def stybtang(x):
    if len(x.shape) == 1:
        return 0.5 * tf.reduce_sum(tf.pow(x, 4) - 16 * tf.pow(x, 2) + 5 * x)
    else:
        return 0.5 * tf.reduce_sum(tf.pow(x, 4) - 16 * tf.pow(x, 2) + 5 * x, axis=1)


def michalewicz(x):
    assert len(x.shape) == 2, 'x input must be 2 dimensional array'
    indx = tf.constant(list(range(1, 1 + int(x.shape[1]))), dtype=FLOAT_TYPE)
    indx = tf.expand_dims(indx, 0)
    return -tf.reduce_sum(tf.sin(x) * tf.sin(x * indx / np.pi) ** (2 * 10), axis=-1)


def michalewicz(x):
    assert len(x.shape) == 2, 'x input must be 2 dimensional array'
    indx = tf.constant(list(range(1, 1 + int(x.shape[1]))), dtype=FLOAT_TYPE)
    indx = tf.expand_dims(indx, 0)
    return -tf.reduce_sum(tf.sin(x) * tf.sin(x**2 * indx / np.pi) ** (2 * 10), axis=-1)


class Function:
    @abc.abstractclassmethod
    def min(self):
        pass

    @abc.abstractclassmethod
    def max(self):
        pass

    @abc.abstractclassmethod
    def func(self, x):
        pass

    @abc.abstractclassmethod
    def name(self):
        pass

class Stybtang(Function):
    def __init__(self, num_dims):
        self.num_dims = num_dims

    @property
    def min(self):
        return -4. * np.ones([self.num_dims], dtype=FLOAT_TYPE)

    @property
    def max(self):
        return 4. * np.ones([self.num_dims], dtype=FLOAT_TYPE)

    def func(self, x):
        return stybtang(x)

    @property
    def name(self):
        return 'Stybtang'


class Michalewicz(Function):
    def __init__(self, num_dims):
        self.num_dims = num_dims

    @property
    def min(self):
        return 0. * np.ones([self.num_dims], dtype=FLOAT_TYPE)

    @property
    def max(self):
        return np.pi * np.ones([self.num_dims], dtype=FLOAT_TYPE)

    def func(self, x):
        return michalewicz(x)

    @property
    def name(self):
        return 'Michalewicz'


class Stybtang_transform(Function):
    def __init__(self, num_dims, dim_groups):
        self.num_dims = num_dims
        Q = np.zeros((num_dims, num_dims), dtype=FLOAT_TYPE)
        for group in dim_groups:
            d = len(group)
            if d > 1:
                A = np.random.randn(d, d).astype(FLOAT_TYPE)
                Q_d, _ = np.linalg.qr(A)
                Q[np.ix_(group, group)] = Q_d
            else:
                Q[group[0], group[0]] = 1.
        self.Q = Q
        global_opt = -2.903534 * np.ones([num_dims, 1], dtype=FLOAT_TYPE)
        self.inverse_opt = np.matmul(np.transpose(Q), global_opt)
        self.inverse_opt = np.squeeze(self.inverse_opt)

    @property
    def min(self):
        return self.inverse_opt - 3

    @property
    def max(self):
        return self.inverse_opt + 3

    def func(self, x):
        return stybtang(tf.matmul(x, tf.transpose(self.Q)))

    @property
    def name(self):
        return 'Stybtang_transform'