import matplotlib.pyplot as plt
import numpy as np
from nilearn.connectome import ConnectivityMeasure



def reorder_matrix(matr, atlas_name):
    if atlas_name == 'HCPex':
        ids = np.loadtxt('../atlas/HCPex_id.txt', dtype=int) - 1
    elif atlas_name == 'Schaefer200':
        ids = np.loadtxt('../atlas/schaefer200_id.txt', dtype=int) - 1

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

    fc = []
    if isinstance(ts[0], list):
        for l in ts:
            calc = connectivity_measure.fit_transform(l)
            for i in calc:
                np.fill_diagonal(i, 0)
            fc.append(calc)

    elif isinstance(ts, np.ndarray) and len(ts.shape) == 2:
        fc = connectivity_measure.fit_transform([ts])
        for i in fc:
            np.fill_diagonal(i, 0)

        if fc.shape[0] == 1:
            fc = np.squeeze(fc)

    elif isinstance(ts, np.ndarray) and len(ts.shape) == 3:
        fc = connectivity_measure.fit_transform(ts)
        for i in fc:
            np.fill_diagonal(i, 0)

        if fc.shape[0] == 1:
            fc = np.squeeze(fc)

    return fc

