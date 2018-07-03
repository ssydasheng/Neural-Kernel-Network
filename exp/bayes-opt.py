import os
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from gpflowSlim.neural_kernel_network import NKNWrapper, NeuralKernelNetwork
import tensorflow as tf
import argparse
import numpy as np

from bo_functions import Michalewicz, Stybtang, Stybtang_transform
from utils.create_logger import create_logger
from bayesianOpt import BayesianOptimization
from kernels import KernelWrapper

# Training settings
parser = argparse.ArgumentParser(description='Neural-Kernel-Network')
parser.add_argument('--name', type=str, default='sty')
parser.add_argument('--kern', type=str, default='rbf')
parser.add_argument('--run', type=int, default=-1, help='indx of run')
args = parser.parse_args()
logger = create_logger('results/bo/' + args.name, 'bo', __file__)
logger.info(args)

num_iters = 200
num_runs = 10
input_dim = 10
grid_size = 10000
iterations = 5000
all_dim_groups = []


def NKNInfo(dimGroups=None):
    ls = 0.3
    kernel = dict(
        oracle=[
            {'name': 'RBF', 'params': {'input_dim': len(group), 'lengthscales': ls, 'ARD': True, 'active_dims': group, 'name': 'RBF'+str(id)}}
            for id, group in enumerate(dimGroups)],
        nkn=[
            {'name': 'RBF', 'params': {'input_dim': 1, 'lengthscales': ls, 'ARD': True, 'active_dims': [id], 'name': 'RBF'+str(id)}}
            for id in range(input_dim)],
        rbf=[
            {'name': 'RBF', 'params': {'input_dim': input_dim, 'lengthscales': ls, 'ARD': True, 'name': 'RBF'}}]
    )[args.kern]

    wrapper = dict(
        oracle=[
            {'name': 'Linear',  'params': {'input_dim': len(dimGroups), 'output_dim': 1, 'name': 'layer1'}}],
        nkn=[
            {'name': 'Linear',  'params': {'input_dim': input_dim, 'output_dim': 8, 'name': 'layer1'}},
            {'name': 'Product', 'params': {'input_dim': 8, 'step': 2,               'name': 'layer2'}},
            {'name': 'Linear',  'params': {'input_dim': 4,         'output_dim': 4, 'name': 'layer3'}},
            {'name': 'Product', 'params': {'input_dim': 4, 'step': 2,               'name': 'layer4'}},
            {'name': 'Linear',  'params': {'input_dim': 2,         'output_dim': 1, 'name': 'layer5'}}
            ],
        rbf=[]
    )[args.kern]

    return kernel, wrapper


all_runs = range(num_runs) if args.run == -1 else [args.run]
np.random.seed(1234 if args.run == -1 else args.run)
for i in all_runs:
    tf.reset_default_graph()

    ################################### define functions ###################################
    if args.name == 'sty':
        Task = Stybtang(num_dims=input_dim)
        dim_groups = [[id] for id in range(input_dim)]
    if args.name == 'mich':
        Task = Michalewicz(num_dims=input_dim)
        dim_groups = [[id] for id in range(input_dim)]
    if args.name == 'sty_t':
        n = np.random.choice(range(3, input_dim-3))
        dims = np.random.permutation(input_dim)
        dim_groups, left_dims = [], input_dim
        for idx in range(n-1):
            max_ = min(left_dims - (n - idx - 1), int(1.5*left_dims // (n-idx)))
            nn = np.random.choice(range(1, int(1.5*left_dims // (n-idx))))
            dim_groups.append(dims[input_dim-left_dims: input_dim-left_dims+nn])
            left_dims = left_dims - nn
        dim_groups.append(dims[-left_dims:])

        logger.info(dim_groups)
        Task = Stybtang_transform(num_dims=input_dim, dim_groups=dim_groups)
    logger.info('Approximating function {}'.format(Task.name))

    ################################### define BO ###################################
    kernel, wrapper = NKNInfo(dim_groups)
    kernel = NeuralKernelNetwork(input_dim, KernelWrapper(kernel), NKNWrapper(wrapper))
    bo_model = BayesianOptimization(
        Task.func, kernel, logger, input_dim, grid_size,
        Task.min, Task.max, iterations=iterations)
    if not osp.exists('results/bo/'+args.name):
        os.makedirs('results/bo/'+args.name)
    bo_model.optimize(num_iters, 'bo/{}/{}_{}'.format(args.name, args.kern, args.run))
