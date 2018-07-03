import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import os.path as osp

path = 'results/bo'
TASKS = ['Stybtang', 'Stybtang_transform', 'Michalewicz']
tasks = ['sty', 'sty_t', 'mich']
ID=2
task = tasks[ID]
_NUM_RUNS = 10
_NUM_ITERS = 200 + 10

def minimum(arr, start=10):
    mins = [np.min(arr[:start])]
    for a in arr[start:]:
        if a < mins[-1]:
            mins.append(a)
        else:
            mins.append(mins[-1])
    return np.array(mins)

def save_final(task, kern):
    ys = []
    for i in range(_NUM_RUNS):
        with open("{}/{}/{}_{}.npz".format(path, task, kern, i), "rb") as outfile:
            content = np.load(outfile)
            y = content['all_y'].squeeze()
            ys.append(y)

    with open("{}/{}/{}.npz".format(path, task, kern), 'wb') as outfile:
        np.savez(outfile, y=ys)

def load_final(task, kern):
    ys = []
    with open("{}/{}/{}.npz".format(path, task, kern), "rb") as outfile:
        all = np.load(outfile)['y']
        for i in range(all.shape[0]):
            ys.append(minimum(all[i].squeeze()))
        mean, std = np.mean(ys, 0), np.std(ys, 0)
    return mean, std

save_final(task, 'nkn')
save_final(task, 'oracle')
save_final(task, 'rbf')

res_nkn, std_res_nkn       = load_final(task, 'nkn')
res_oracle, std_res_oracle = load_final(task, 'oracle')
res_rbf, std_res_rbf       = load_final(task, 'rbf')

plt.rcParams['legend.fontsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14

coeff, alpha = 0.2, 0.1

#########################################################
#                     SingleDim
#########################################################
plt.plot(range(len(res_oracle)), res_oracle, 'k-', linewidth=2)
plt.fill_between(range(len(res_oracle)), res_oracle-std_res_oracle * coeff, res_oracle+std_res_oracle * coeff, alpha=alpha, color='k')
plt.plot(range(len(res_nkn)), res_nkn, 'r-', linewidth=2)
plt.fill_between(range(len(res_nkn)), res_nkn-std_res_nkn * coeff, res_nkn+std_res_nkn * coeff, alpha=alpha, color='r')
plt.plot(range(len(res_rbf)), res_rbf, 'b-', linewidth=2)
plt.fill_between(range(len(res_rbf)), res_rbf-std_res_rbf * coeff, res_rbf+std_res_rbf * coeff, alpha=alpha, color='b')
plt.legend(['Oracle', 'NKN', 'RBF'])

#plt.ylim([-380, -200])
plt.xlim([0, 200])
plt.pause(0.001)
plt.tight_layout()
plt.savefig('{}/{}.png'.format(path, task), format='png', dvi=1200)
plt.savefig('{}/{}.pdf'.format(path, task), format='pdf', dvi=1200)
