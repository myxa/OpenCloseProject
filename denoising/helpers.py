import matplotlib.pyplot as plt
import numpy as np
from nilearn.connectome import ConnectivityMeasure



def reorder_matrix(matr, ids):
    sort_0 = matr[ids]
    return sort_0.T[ids]


def plot_corr_hist(ts1, ts2, title=''):
    l=[]
    for i in range(423):
        l.append(np.corrcoef(ts1[i], ts2[i])[0, 1])
    plt.hist(l, bins='auto')
    plt.title(f'{title}, median={round(np.median(l), 3)}');


def functional_connectivity(ts, measure="correlation"):
    connectivity_measure = ConnectivityMeasure(kind=measure)
    fc = connectivity_measure.fit_transform(ts)
    for i in fc:
        np.fill_diagonal(i, 0)
    return fc

