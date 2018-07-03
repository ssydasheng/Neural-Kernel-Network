from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import time

import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import tensorflow as tf
import sympy as sp
import gpflowSlim as gfs
from gpflowSlim.neural_kernel_network import NKNWrapper, NeuralKernelNetwork

from utils.create_logger import create_logger
from data.timeSeries.base import load_data
from utils.functions import median_distance_local
from kernels import KernelWrapper

tf.set_random_seed(123)
np.random.seed(123)
start_time = time.time()

# Training settings
parser = argparse.ArgumentParser(description='Neural-Kernel-Network')
parser.add_argument('--name', type=str, default='airline')
parser.add_argument('--kern', type=str, default='nkn')
args = parser.parse_args()

############################## load and normalize data ##############################
X, y = load_data(args.name)
x_train, y_train = X.astype(gfs.settings.float_type), y.astype(gfs.settings.float_type)
print ('# Training Data:', x_train.shape[0])
mean_x, std_x = np.mean(x_train), np.std(x_train)
mean_y, std_y = np.mean(y_train), np.std(y_train)

normalized_x_train = (x_train - mean_x) / std_x
normalized_y_train = (y_train - mean_y) / std_y
inputs, targets = tf.constant(normalized_x_train), tf.constant(normalized_y_train)
x_train_extended = tf.concat(
    [inputs, inputs - np.min(normalized_x_train) + np.max(normalized_x_train)], axis=0)

logger = create_logger('results/time-series/', args.name, __file__)
print = logger.info

############################## setup parameters ##############################
epochs = 20000
plot_interval=5000
print_interval=100

############################## build NKN ##############################
ls = median_distance_local(normalized_x_train).astype('float32')
ls[abs(ls) < 1e-6] = 1.
input_dim = 1

kernel = dict(
    nkn=[
        {'name': 'Linear',   'params': {'input_dim': input_dim, 'name':'k0'}},
        {'name': 'Periodic', 'params': {'input_dim': input_dim, 'period': ls, 'lengthscales': ls, 'name':'k1'}},
        {'name': 'ExpQuad',  'params': {'input_dim': input_dim, 'lengthscales': ls / 4.0, 'name':'k2'}},
        {'name': 'RatQuad',  'params': {'input_dim': input_dim, 'alpha': 0.2, 'lengthscales': ls * 2.0, 'name':'k3'}},
        {'name': 'Linear',   'params': {'input_dim': input_dim, 'name':'k4'}},
        {'name': 'RatQuad',  'params': {'input_dim': input_dim, 'alpha': 0.1, 'lengthscales': ls, 'name':'k5'}},
        {'name': 'ExpQuad',  'params': {'input_dim': input_dim, 'lengthscales': ls, 'name':'k6'}},
        {'name': 'Periodic', 'params': {'input_dim': input_dim, 'period': ls / 4.0, 'lengthscales': ls / 4.0, 'name':'k7'}}],
    heuristic=[
        {'name': 'Linear',   'params': {'input_dim': input_dim, 'name': 'k0'}},
        {'name': 'Periodic', 'params': {'input_dim': input_dim, 'period': ls, 'lengthscales': ls, 'name': 'k1'}},
        {'name': 'ExpQuad',  'params': {'input_dim': input_dim, 'lengthscales': ls, 'name': 'k2'}}]
)[args.kern]

wrapper = dict(
    nkn=[
        {'name': 'Linear',  'params': {'input_dim': 8, 'output_dim': 8, 'name': 'layer1'}},
        {'name': 'Product', 'params': {'input_dim': 8, 'step': 2,       'name': 'layer2'}},
        {'name': 'Linear',  'params': {'input_dim': 4, 'output_dim': 4, 'name': 'layer3'}},
        {'name': 'Product', 'params': {'input_dim': 4, 'step': 2,       'name': 'layer4'}},
        {'name': 'Linear',  'params': {'input_dim': 2, 'output_dim': 1, 'name': 'layer5'}}],
    heuristic=[
        {'name': 'Linear', 'params': {'input_dim': 3, 'output_dim': 1,  'name': 'layer1'}}]
)[args.kern]
wrapper = NKNWrapper(wrapper)

nkn = NeuralKernelNetwork(1, KernelWrapper(kernel), wrapper)
model = gfs.models.GPR(inputs, targets, nkn, name='model')


############################## build graph ##############################
optimizer = tf.train.AdamOptimizer(1e-3)
loss = model.objective
infer = optimizer.minimize(loss)
variance = model.likelihood.variance
pred_mu, pred_cov = model.predict_y(x_train_extended)

############################## setup plotting region ##############################
xmin_t, xmax_t, ymin_t, ymax_t = np.min(x_train), np.max(x_train), np.min(y_train), np.max(y_train)
xmin, xmax = xmin_t, xmax_t + (xmax_t - xmin_t)/2.0
ymin, ymax = ymin_t + (ymin_t - ymax_t)/2.0, ymax_t + (ymax_t - ymin_t)/1.0

print ('x: {:.3f}, {:.3f}'.format(xmin, xmax))
print ('y: {:.3f}, {:.3f}'.format(ymin, ymax))
plt.xlim((xmin, xmax))
plt.ylim((ymin, ymax))

############################## train & plot ##############################
sess = tf.Session()
sess.run(tf.global_variables_initializer())
x_vis = sess.run(tf.squeeze(x_train_extended)*std_x + mean_x)
for epoch in range(1, epochs + 1):
    _, obj, var = sess.run([infer, loss, variance])

    if epoch % print_interval == 0:
        print ('[%d/%d]Loss: %5.4f, noise: %.4f'%(epoch, epochs, obj, var))

    if epoch % plot_interval == 0:
        mu, cov = sess.run([pred_mu, pred_cov])
        mu, cov = mu.squeeze() * std_y + mean_y, cov.squeeze() * (std_y ** 2)
        plt.clf()
        plt.xlim((xmin, xmax))
        plt.ylim((ymin, ymax))

        plt.plot(x_train, y_train, 'x')
        plt.plot(x_vis, mu, 'b')
        plt.fill_between(x_vis, mu - cov ** 0.5, mu + cov ** 0.5, alpha=0.2, color='g')
        plt.axvline(x=xmax_t , ymin=-1000, ymax=2000, linestyle='dashed', linewidth=2, color='r')
        plt.draw()
        plt.pause(1e-17)
        plt.savefig('results/time-series/%s_%s_epoch%d.pdf'%(args.name, args.kern, epoch))

print('Collapse Time = {}'.format(time.time() - start_time))

############################## compute final polynomial ##############################
parse_poly = False
def organize_symbol(sym):
    sym = sp.expand(sym)
    sym = sym.as_coefficients_dict()
    sym = [(k, v) for k, v in sym.items()]
    sym.sort(key=lambda x: x[1], reverse=True)
    return sym, [(k, v) for k, v in sym if v > 1e-4]

if parse_poly:
    ks = sp.symbols(['k' + str(i) for i in range(len(kernel))])
    weights, consts = [], []
    for i in range(len(wrapper.parameters) // 2):
        weights.append(wrapper.parameters[2 * i].value)
        consts.append(wrapper.parameters[2 * i + 1].value)
    weights = sess.run(weights)
    consts = sess.run(consts)

    symbols = [ks]
    for i, (ws, c) in enumerate(zip(weights, consts)):
        ## Linear layer
        fc = []
        for j in range(ws.shape[0]):
            tmp = c[j]
            for k in range(ws.shape[1]):
                tmp = tmp + symbols[k] * ws[j][k]
            fc.append(tmp)

        ## Product layer
        pd = []
        for j in range(len(fc) // 2):
            # for simplicity here, we assume Product step is 2.
            tmp = fc[2 * j] * fc[2 * j + 1]
            pd.append(tmp)
        symbols = pd

    print('function', organize_symbol(symbols[0])[1])
