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

def dilute(xdata, ydata):
    """
    Apply dilution factor to a dataset

    Arguments:
        xdata: ndarray  x x m array of m observations of independent variables
        ydata: ndarray  y x m array of non-normalised observations of dependent
                        variables

    Returns:
        ndarray  y x m array of input data with dilution factor applied
    """

    h0 = xdata[0]
    # PLACEHOLDER this only calculates dilution for the first dataset
    y = ydata

    dilfac = h0/h0[0]
    dilmat = ml.repmat(dilfac, y.shape[0], 1)
    y_dil = (y*dilmat)
    return y_dil

def calculate_coeffs(fitter, coeffs, ydata_init, h0_init=None):
    """
    Calculate "real" coefficients from their raw values and an input dataset

    Arguments:
        ydata_init: ndarray  1 x m array of non-normalised initial observations 
                             of dependent variables
        coeffs:     ndarray  
        h0_init:    float    Optional initial h0 value, if provided ydata_init 
                             is divided by this value before the calculation
    """
    

    # H coefficients
    h = np.copy(ydata_init)

    # Divide initial ydata values and coeffs by h0 in UV fitters
    if "uv" in fitter and h0_init is not None:
        h /= h0_init
        coeffs = np.copy(coeffs)/h0_init

    coeffs = np.array(coeffs)
    rows = coeffs.shape[0]
    if rows == 1:
        # 1:1 system
        hg = h + coeffs[0]
        return np.vstack((h, hg))
    elif rows == 2:
        # 1:2 or 2:1 system
        hg  = h - coeffs[0]
        hg2 = h - coeffs[1]
        return np.vstack((h, hg, hg2))
    else:
        pass # Throw error here
