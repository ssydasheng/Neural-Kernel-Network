from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import time

import numpy as np
import argparse
from scipy import stats
import gpflowSlim as gfs
from gpflowSlim.neural_kernel_network import NKNWrapper, NeuralKernelNetwork
import tensorflow as tf

from utils.create_logger import create_logger
from data import get_data
from utils.functions import median_distance_local, get_kemans_init
from kernels import KernelWrapper


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Neural-Kernel-Network')
parser.add_argument('--data', type=str, default="boston", help='choose data')
parser.add_argument('--split', type=str, default="uci_woval", help='way to split dataset')
parser.add_argument('--kern', type=str, default='nkn')
args = parser.parse_args()


############################# basic info #############################
N_RUNS=dict(
    uci_woval=10,
    uci_woval_pca=1,
    uci_woval_pca_train_val_split=1
)
FLOAT_TYPE = gfs.settings.float_type
SMALL_DATASETS=['boston', 'concrete', 'energy', 'wine', 'yacht']
epochs = 20000 #if args.data in SMALL_DATASETS else 10000


############################## NKN Info ##############################
def NKNInfo(input_dim, ls):
    kernel = dict(
        nkn=[
            {'name': 'Linear',  'params': {'input_dim': input_dim, 'ARD': True,                    'name': 'Linear1'}},
            {'name': 'Linear',  'params': {'input_dim': input_dim, 'ARD': True,                    'name': 'Linear2'}},
            {'name': 'RBF',     'params': {'input_dim': input_dim, 'lengthscales': ls / 6., 'ARD': True,     'name': 'RBF1'}},
            {'name': 'RBF',     'params': {'input_dim': input_dim, 'lengthscales': ls * 2 / 3., 'ARD': True, 'name': 'RBF2'}},
            {'name': 'RatQuad', 'params': {'input_dim': input_dim, 'alpha': 0.1, 'lengthscales': ls / 3.,    'name': 'RatQuad1'}},
            {'name': 'RatQuad', 'params': {'input_dim': input_dim, 'alpha': 1., 'lengthscales': ls / 3.,     'name': 'RatQuad2'}}],
        rbf=[
            {'name': 'RBF',     'params': {'input_dim': input_dim, 'lengthscales': ls, 'ARD': True, 'name': 'RBF1'}}],
        sm1=[{'name': 'SM', 'params': [
            {'w': 1.,
             'rbf': {'input_dim': input_dim, 'lengthscales': ls, 'ARD': True, 'name': 'SM-RBF0'},
             'cos': {'input_dim': input_dim, 'lengthscales': ls, 'ARD': True, 'name': 'SM-Cosine0'}},
        ]}],
        sm2 = [{'name': 'SM', 'params': [
            {'w': 1. / 2,
             'rbf': {'input_dim': input_dim, 'lengthscales': ls,      'ARD': True, 'name': 'SM-RBF0'},
             'cos': {'input_dim': input_dim, 'lengthscales': ls,      'ARD': True, 'name': 'SM-Cosine0'}},
            {'w': 1. / 2,
             'rbf': {'input_dim': input_dim, 'lengthscales': ls / 2,  'ARD': True, 'name': 'SM-RBF1'},
             'cos': {'input_dim': input_dim, 'lengthscales': ls / 2,  'ARD': True, 'name': 'SM-Cosine1'}},
        ]}],
        sm3=[{'name': 'SM', 'params': [
            {'w': 1. / 3,
             'rbf': {'input_dim': input_dim, 'lengthscales': ls * 2., 'ARD': True, 'name': 'SM-RBF0'},
             'cos': {'input_dim': input_dim, 'lengthscales': ls * 2., 'ARD': True, 'name': 'SM-Cosine0'}},
            {'w': 1. / 3,
             'rbf': {'input_dim': input_dim, 'lengthscales': ls / 2,  'ARD': True, 'name': 'SM-RBF1'},
             'cos': {'input_dim': input_dim, 'lengthscales': ls / 2,  'ARD': True, 'name': 'SM-Cosine1'}},
            {'w': 1. / 3,
             'rbf': {'input_dim': input_dim, 'lengthscales': ls,      'ARD': True, 'name': 'SM-RBF2'},
             'cos': {'input_dim': input_dim, 'lengthscales': ls,      'ARD': True, 'name': 'SM-Cosine2'}},
        ]}],
        sm4=[{'name': 'SM', 'params': [
            {'w': 1. / 4,
             'rbf': {'input_dim': input_dim, 'lengthscales': ls / 4., 'ARD': True, 'name': 'SM-RBF0'},
             'cos': {'input_dim': input_dim, 'lengthscales': ls / 4., 'ARD': True, 'name': 'SM-Cosine0'}},
            {'w': 1. / 4,
             'rbf': {'input_dim': input_dim, 'lengthscales': ls / 2,  'ARD': True, 'name': 'SM-RBF1'},
             'cos': {'input_dim': input_dim, 'lengthscales': ls / 2,  'ARD': True, 'name': 'SM-Cosine1'}},
            {'w': 1. / 4,
             'rbf': {'input_dim': input_dim, 'lengthscales': ls,      'ARD': True, 'name': 'SM-RBF2'},
             'cos': {'input_dim': input_dim, 'lengthscales': ls,      'ARD': True, 'name': 'SM-Cosine2'}},
            {'w': 1. / 4,
             'rbf': {'input_dim': input_dim, 'lengthscales': ls * 2,  'ARD': True, 'name': 'SM-RBF3'},
             'cos': {'input_dim': input_dim, 'lengthscales': ls * 2,  'ARD': True, 'name': 'SM-Cosine3'}}]}],
    )[args.kern]
    
    wrapper = dict(
        nkn=[
            {'name': 'Linear',  'params': {'input_dim': 6, 'output_dim': 8, 'name': 'layer1'}},
            {'name': 'Product', 'params': {'input_dim': 8, 'step': 2,       'name': 'layer2'}},
            {'name': 'Linear',  'params': {'input_dim': 4, 'output_dim': 4, 'name': 'layer3'}},
            {'name': 'Product', 'params': {'input_dim': 4, 'step': 2,       'name': 'layer4'}},
            {'name': 'Linear',  'params': {'input_dim': 2, 'output_dim': 1, 'name': 'layer5'}}],
        rbf=[],
        sm1=[],
        sm2=[],
        sm3=[],
        sm4=[]
    )[args.kern]
    return kernel, wrapper


def run(data, logger):
    tf.reset_default_graph()

    ############################## setup data ##############################
    x_train, y_train = data.x_train.astype(FLOAT_TYPE), data.y_train.astype(FLOAT_TYPE)
    x_test,  y_test  = data.x_test.astype(FLOAT_TYPE),  data.y_test.astype(FLOAT_TYPE)
    std_y_train = data.std_y_train

    N, nx = x_train.shape
    x = tf.placeholder(FLOAT_TYPE, shape=[None, nx])
    y = tf.placeholder(FLOAT_TYPE, shape=[None])

    ############################## build nkn ##############################
    ls = median_distance_local(data.x_train).astype('float32')
    ls[abs(ls) < 1e-6] = 1.
    kernel, wrapper = NKNInfo(input_dim=nx, ls=ls)
    wrapper = NKNWrapper(wrapper)
    nkn = NeuralKernelNetwork(nx, KernelWrapper(kernel), wrapper)

    ############################## build graph ##############################
    if args.data in SMALL_DATASETS:
        model = gfs.models.GPR(x, tf.expand_dims(y, 1), nkn, name='model')
    else:
        inducing_points = get_kemans_init(x_train, 1000)
        model = gfs.models.SGPR(x, tf.expand_dims(y, 1), nkn, Z=inducing_points)

    objective = model.objective
    optimizer = tf.train.AdamOptimizer(1e-3)
    infer = optimizer.minimize(objective)
    pred_mu, pred_cov = model.predict_y(x_test)

    ############################## session run ##############################
    rmse_results = []
    lld_results = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            indices = np.random.permutation(N)
            x_train, y_train = x_train[indices, :], y_train[indices]
            fd = {x: x_train, y: y_train}
            _, obj = sess.run([infer, objective], feed_dict=fd)

            if epoch % 100 == 0:
                logger.info('Epoch {}: Loss = {}'.format(epoch, obj))

                # evaluate
                mu, cov = sess.run(
                    [pred_mu, pred_cov], feed_dict={x: x_train, y: y_train}
                )
                mu, cov = mu.squeeze(), cov.squeeze()
                rmse = np.mean((mu - y_test) ** 2) ** .5 * std_y_train

                log_likelihood = np.mean(np.log(stats.norm.pdf(
                    y_test,
                    loc=mu,
                    scale=cov ** 0.5))) - np.log(std_y_train)
                logger.info('test rmse = {}'.format(rmse))
                logger.info('tset ll = {}'.format(log_likelihood))
                rmse_results.append(rmse)
                lld_results.append(log_likelihood)
    return rmse_results, lld_results


if __name__ == "__main__":
    start_time = time.time()
    tf.set_random_seed(123)
    np.random.seed(123)
    data_fetch = get_data(args.split)

    logger = create_logger('results/regression/' + args.split + '/' + args.data, 'reg', __file__)
    logger.info('| jitter level {}'.format(gfs.settings.jitter))
    logger.info('| float type {}'.format(FLOAT_TYPE))
    logger.info('| kernel {}'.format(args.kern))
    logger.info('| dataset {}'.format(args.data))
    logger.info('| split {}'.format(args.split))

    rmse_results = []
    ll_results = []
    for i in range(1, N_RUNS[args.split] + 1):
        logger.info("\n## RUN {}".format(i))
        data = data_fetch(args.data, i)
        rmse_result, ll_result = run(data, logger)
        rmse_results.append(rmse_result)
        ll_results.append(ll_result)
        logger.info('Collapse Time = {} s'.format(time.time() - start_time))

    ########################### logging results ###########################
    for i, (rmse_result, ll_result) in enumerate(zip(rmse_results,
                                                     ll_results)):
        logger.info("\n## Result-by-epoch for RUN {}".format(i))
        logger.info('# Test rmse = {}'.format(rmse_result))
        logger.info('# Test log likelihood = {}'.format(ll_result))

    for i in range(len(rmse_results[0])):
        logger.info("\n## AVERAGE for epoch {}".format(i))
        test_rmses = [a[i] for a in rmse_results]
        test_lls = [a[i] for a in ll_results]

        logger.info("Test rmse = {}/{}".format(
            np.mean(test_rmses), np.std(test_rmses) / N_RUNS[args.split] ** 0.5))
        logger.info("Test log likelihood = {}/{}".format(
            np.mean(test_lls), np.std(test_lls) / N_RUNS[args.split] ** 0.5))
        logger.info('NOTE: Test result above output mean and std. errors')

