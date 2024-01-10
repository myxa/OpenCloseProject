import os
import matplotlib.pyplot as plt
import numpy as np
from nilearn.connectome import ConnectivityMeasure
import pandas as pd


def reorder_matrix(matr, atlas_name):
    """
    Reorder FC matrix in Left-Right order
    
    Parameters
    ----------
    matr: np.array
        Matrix to reorder
    atlas_name: str
        One of ['HCPex', 'Schaefer200']

    Returns
    -------
    np.array
        Reordered matrix (wow!)
    """
    if atlas_name == 'HCPex':
        ids = np.loadtxt('../atlas/HCPex_id.txt', dtype=int) - 1
    elif atlas_name == 'Schaefer200':
        ids = np.loadtxt('../atlas/schaefer200_id.txt', dtype=int) - 1

    if isinstance(matr, np.ndarray) and len(matr.shape) == 3:
        sort_0 = np.zeros_like(matr)
        for i in range(len(matr)):
            temp = matr[i][ids]
            sort_0[i] = temp.T[ids]
    else:
        sort_0 = matr[ids]
        sort_0 = sort_0.T[ids]

    return sort_0


def plot_corr_hist(ts1, ts2, title=''):
    l=[]
    for i in range(423):
        l.append(np.corrcoef(ts1[i], ts2[i])[0, 1])
    plt.hist(l, bins='auto')
    plt.title(f'{title}, median={round(np.median(l), 3)}');


def functional_connectivity(ts, measure="correlation"):
    """
    Functional connectivity calculation

    Parameters
    ----------
    ts: list, np.array
        List of np.arrays or np.array of shape (2, :, :) or (3, :, :)
    measure: str
        One of ["covariance", "correlation", "partial correlation", "tangent", "precision"]. By default "correlation"

    Returns
    -------
    list of np.arrays or np.array
    """

    connectivity_measure = ConnectivityMeasure(kind=measure)
    fc = []
    if isinstance(ts[0], list):
        for l in ts:
            calc = connectivity_measure.fit_transform(l)
            for i in calc:
                np.fill_diagonal(i, 0)
            fc.append(calc)

    elif (isinstance(ts, np.ndarray) and len(ts.shape) == 2):
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

    elif isinstance(ts, list):
        fc = connectivity_measure.fit_transform(ts)
        for i in fc:
            np.fill_diagonal(i, 0)


    return fc


def load_timeseries(path, sub=None, run=1, task='rest', strategy=4, atlas_name='AAL'):
    """
    Load time series from folder.
    Time-series should be stored in a folder for each subject. In subject folders there should be folder for atlas.
    In atlas folder there are csv files

    Parameters
    ----------
    path: str
        Path to folder with all subject folders
    sub: list of str, optional
        List of subjects to load data in form 'sub-n'. If None, all subjects are loaded (default None)
    run: int
        Run to load
    task: str
        Task to load
    strategy: int
        Strategy to load
    atlas_name: str
        Atlas name. Should be one of ['HCPex', 'Schaefer200', 'AAL']
    
    Returns
    -------
    List of numpy.arrays 
    """
    ts = []
    if sub is None:
        sub = os.listdir(path)
    failed = []
    for i in sub:
        try:
            name = f'{i}_task-{task}_run-{run}_time-series_{atlas_name}_strategy-{strategy}.csv'
            path_to_file = os.path.join(path, f'{i}', atlas_name, name)
            ts.append(pd.read_csv(path_to_file).values)
        except FileNotFoundError:
            failed.append(i)
            continue
    print('no files available:', failed)
    return ts
