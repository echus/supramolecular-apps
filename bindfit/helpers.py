"""
" Miscellaneous helper functions, mostly for data munging
"""

from __future__ import division
from __future__ import print_function

from math import sqrt
import numpy as np
import numpy.matlib as ml

import logging
logger = logging.getLogger('supramolecular')

def cov(data, residuals, total=False):
    # TODO: TEMP
    # Add axis to single y arrays for generalised calcs
    if hasattr(residuals, "shape") and len(residuals.shape) == 1:
        residuals = residuals[np.newaxis]

    data_norm = normalise(data)

    if total:
        return np.var(residuals)/np.var(data_norm)
    else:
        return np.var(residuals, axis=1)/np.var(data_norm, axis=1)

def rms(residuals, total=False):
    """
    Calculate RMS errors from residuals

    Arguments:
        residuals: array  3D array of residuals corresponding to each input
                          y 

    Returns:
        array  1D array of RMS values for each fitted y
    """

    # TODO: TEMP
    # Add axis to single y arrays for generalised calcs
    if hasattr(residuals, "shape") and len(residuals.shape) == 1:
        residuals = residuals[np.newaxis]

    logger.debug("helpers.rms: called")

    r = np.array(residuals)
    sqr = np.square(r)
    # meansqr = np.mean(sqr, axis=1)
    # #rms = sqrt(sumsqr) - doesn't work when array elements are numpy.float64s!
    # rms = [ sqrt(s) for s in meansqr ]

    if total:
        return np.sqrt(np.mean(sqr)) 
    else:
        return np.sqrt(np.mean(sqr, axis=1))

def normalise(data):
    """ 
    Normalise a 2D array of observations.

    Arguments:
        data: ndarray  n x m array of n dependent variables, m observations

    Returns:
        ndarray  n x m array of normalised input data
    """

    logger.debug("helpers.normalise: called")
    logger.debug("helpers.normalise: input data")
    logger.debug(data)

    # Create matrix of initial values to subtract from original matrix
    initialmat = ml.repmat(data.T[0,:], len(data.T), 1).T
    data_norm = data - initialmat
    return data_norm

def denormalise(data, data_norm):
    """
    Denormalise a normalised dataset given original non-normalised input data

    Arguments:
        data:      ndarray  original n x m array
        data_norm: ndarray  normalised n x m array
    
    Returns:
        ndarray  n x m array of denormalised input data_norm
    """
    # Create matrix of initial data values to add to fit 
    initialmat = ml.repmat(data[:,0][np.newaxis].T, 1, data.shape[1])
    # De-normalize normalised data (add initial values back)
    data_denorm = data_norm + initialmat
    return data_denorm 

def dilute(h0, data):
    """
    Apply dilution factor to a dataset

    Arguments:
        xdata: ndarray  x x m array of m observations of independent variables
        ydata: ndarray  y x m array of non-normalised observations of dependent
                        variables

    Returns:
        ndarray  y x m array of input data with dilution factor applied
    """

    y = data

    dilfac = h0/h0[0]
    dilmat = ml.repmat(dilfac, y.shape[0], 1)
    y_dil = (y*dilmat)
    return y_dil
