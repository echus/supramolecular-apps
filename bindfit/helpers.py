"""
" Miscellaneous helper functions, mostly for data munging
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.matlib as ml

def cov(y, residuals):
    cov = []
    
    # For each dataset
    # TODO do this the proper matrix way
    for d, r in zip(y, residuals):
        var = np.var(r, axis=1)/np.var(d, axis=1)
        cov.append(var)

    return cov

def rms(residuals):
    """
    Calculate RMS errors from residuals

    Arguments:
        residuals: array  3D array of residuals corresponding to each input
                          dataset

    Returns:
        array  2D array of RMS values for each fit
    """

    # For each input set of fits' residuals
    rms = []
    for r in residuals:
        a = np.array(r)
        rms.append(np.sqrt(np.sum(np.square(a), axis=1)))

    return rms

def normalise(y):
    """ 
    Normalise a 3D array of datasets.

    Arguments:
        y: array  3D array of input datasets

    Returns:
        array  3D array of normalised input datasets
    """

    #y_norm = y.copy()
    y_norm = np.zeros(y.shape)

    # Create matrix of initial values to subtract from original matrix
    # TODO do this the proper matrix way instead of looping
    # (loop over potentially more than one input dataset)
    # Transpose magic for easier repmat'n
    for i in range(y.shape[0]):
        initialmat = ml.repmat(y[i].T[0,:], len(y[i].T), 1)
        y_norm[i] = (y[i].T - initialmat).T

    return y_norm

def denormalise(y, fit_norm):
    """
    Denormalise a fit given original non-normalised input data and fitted data

    Arguments:
        y:     array  3D array of non-normalised input datasets
        fit_norm: array  3D array of normalised fitted data
    
    Returns:
        array  3D array of non-normalised fitted data
    """
    # PLACEHOLDER, this only calculates first dimension of potentially 
    # multi dimensional y fit array

    # Create matrix of initial data values to add to fit 
    # TODO do this the proper matrix way instead of looping
    # PLACEHOLDER uses only y[0] - the first of potential multiple inputs
    y0 = y[0]
    initialmat = ml.repmat(y0[:,0][np.newaxis].T, 1, y0.shape[1])
    #initialmat = ml.repmat(y[0].T[0,:], len(y[0].T), 1)

    # PLACEHOLDER 3rd axis added to calculated fit to mimic multiple
    # datasets and transpose to row matrix
    # De-normalize calculated y (add initial values back)
    fit_norm0 = fit_norm[0]
    fit = (fit_norm0 + initialmat)[np.newaxis]

    return fit

def dilute(x, y):
    """
    Apply dilution factor to a dataset

    Arguments:
        y:  array  3D array of non-normalised input dataset
        h0: array  1D array of input [H]0 concentrations

    Returns:
        array  3D array of input data with dilution factor applied
    """

    h0 = x[0]
    # PLACEHOLDER this only calculates dilution for the first dataset
    y0 = y[0]

    dilfac = h0/h0[0]
    dilmat = ml.repmat(dilfac, y0.shape[0], 1)
    
    # PLACEHOLDER add extra axis to simulate 3D dataset
    y_dil = (y0*dilmat)[np.newaxis]
    return y_dil
