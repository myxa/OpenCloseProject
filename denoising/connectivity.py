import numpy as np
from scipy import stats
from gglasso.problem import glasso_problem 
import numpy as np
from nilearn.connectome import ConnectivityMeasure


def graph_lasso(data, L1):
    '''
    Calculates the L1-regularized partial correlation matrix of a dataset. 
    Runs GGLasso's graphical lasso function (glasso_problem.solve()) and several other necessary steps.
    INPUT:
        data : a dataset with dimension [nNodes x nDatapoints]
        L1 : L1 (lambda1) hyperparameter value
    OUTPUT:
        glassoParCorr : regularized partial correlation coefficients (i.e., FC matrix)
        prec : precision matrix, where entries are not yet transformed into partial correlations (used to compute loglikelihood)
    '''

    nNodes = data.shape[0]

    # Z-score the data
    data_scaled = stats.zscore(data, axis=1)

    # Estimate the empirical covariance
    empCov = np.cov(data_scaled, rowvar=True)

    # Number of timepoints in data
    nTRs = data.shape[1]

    # Run glasso
    glasso = glasso_problem(empCov, nTRs, reg_params={'lambda1': L1}, latent=False, do_scaling=False)
    # glasso.model_selection(modelselect_params={'lambda1_range': L1s})
    glasso.solve(verbose=False)
    prec = np.squeeze(glasso.solution.precision_)

    # Transform precision matrix into regularized partial correlation matrix
    denom = np.atleast_2d(1. / np.sqrt(np.diag(prec)))
    glassoParCorr = -prec * denom * denom.T
    np.fill_diagonal(glassoParCorr, 0)

    return glassoParCorr, prec

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

    connectivity_measure = ConnectivityMeasure(kind=measure, standardize=False)
    fc = []
    if isinstance(ts[0], list):
        for l in ts:
            calc = connectivity_measure.fit_transform(l)
            for i in calc:
                np.fill_diagonal(i, 1)
            fc.append(calc)

    elif (isinstance(ts, np.ndarray) and len(ts.shape) == 2):
        fc = connectivity_measure.fit_transform([ts])
        for i in fc:
            np.fill_diagonal(i, 1)

        if fc.shape[0] == 1:
            fc = np.squeeze(fc)

    elif isinstance(ts, np.ndarray) and len(ts.shape) == 3:
        fc = connectivity_measure.fit_transform(ts)
        for i in fc:
            np.fill_diagonal(i, 1)

        if fc.shape[0] == 1:
            fc = np.squeeze(fc)

    elif isinstance(ts, list):
        fc = connectivity_measure.fit_transform(ts)
        for i in fc:
            np.fill_diagonal(i, 1)


    return fc