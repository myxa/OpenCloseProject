import os
import numpy as np
from scipy import stats
from scipy import linalg
from gglasso.problem import glasso_problem 
# (We began using the GGLasso package (https://gglasso.readthedocs.io/en/latest/) after sklearn's GraphicalLasso would not converge for all subjects for the tested hyperparameters)
from sklearn.covariance import log_likelihood, empirical_covariance

def graphicalLasso(data, L1):
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
