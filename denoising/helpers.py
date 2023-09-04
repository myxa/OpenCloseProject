import matplotlib.pyplot as plt
import numpy as np


def reorder_matrix(matr, ids):
    sort_0 = matr[ids]
    return sort_0.T[ids]

def plot_corr_hist(ts1, ts2, title=''):
    l=[]
    for i in range(423):
        l.append(np.corrcoef(ts1[i], ts2[i])[0, 1])
    plt.hist(l, bins='auto')
    plt.title(f'{title}, median={round(np.median(l), 3)}');
