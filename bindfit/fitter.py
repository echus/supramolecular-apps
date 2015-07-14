from __future__ import division
from __future__ import print_function

from math import sqrt
import numpy as np
import numpy.matlib as ml
import scipy
import scipy.optimize

from . import functions

def fit_nmr_1to1(h0, g0, data):
    """
    """

    # Subtract row 1 data point from all columns in data
    initial = ml.repmat(data[0,:], len(data), 1)
    data -= initial
    
    # Calculate residuals for static k value
    #ss = functions.nmr_1to1(1000, h0, g0, data)

    k_optim = scipy.optimize.minimize(functions.nmr_1to1,
                                      1000,
                                      args=(h0, g0, data),
                                      tol=10e-10,
                                      )

    k_optim_fmin = scipy.optimize.fmin(functions.nmr_1to1,
                                       1000,
                                       args=(h0, g0, data),
                                       xtol=10e-10,
                                       ftol=10e-10,
                                       )

    return k_optim_fmin
