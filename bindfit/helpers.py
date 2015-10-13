"""
" Miscellaneous helper functions, mostly for data munging
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.matlib as ml

def cov(data, residuals):
    cov = []
    
    # For each dataset
    # TODO do this the proper matrix way
    for d, r in zip(data, residuals):
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

def normalise(data):
    """ 
    Normalise a 3D array of datasets.

    Arguments:
        data: array  3D array of input datasets

    Returns:
        array  3D array of normalised input datasets
    """

    #data_norm = data.copy()
    data_norm = np.zeros(data.shape)

    # Create matrix of initial values to subtract from original matrix
    # TODO do this the proper matrix way instead of looping
    # (loop over potentially more than one input dataset)
    # Transpose magic for easier repmat'n
    for i in range(data.shape[0]):
        initialmat = ml.repmat(data[i].T[0,:], len(data[i].T), 1)
        data_norm[i] = (data[i].T - initialmat).T

    return data_norm

def denormalise(data, fit_norm):
    """
    Denormalise a fit given original non-normalised input data and fitted data

    Arguments:
        data:     array  3D array of non-normalised input datasets
        fit_norm: array  3D array of normalised fitted data
    
    Returns:
        array  3D array of non-normalised fitted data
    """
    # PLACEHOLDER, this only calculates first dimension of potentially 
    # multi dimensional y fit array

    # Create matrix of initial data values to add to fit 
    # TODO do this the proper matrix way instead of looping
    # PLACEHOLDER uses only data[0] - the first of potential multiple inputs
    data0 = data[0]
    initialmat = ml.repmat(data0[:,0][np.newaxis].T, 1, data0.shape[1])
    #initialmat = ml.repmat(data[0].T[0,:], len(data[0].T), 1)

    # PLACEHOLDER 3rd axis added to calculated fit to mimic multiple
    # datasets and transpose to row matrix
    # De-normalize calculated data (add initial values back)
    fit_norm0 = fit_norm[0]
    fit = (fit_norm0 + initialmat)[np.newaxis]

    return fit
