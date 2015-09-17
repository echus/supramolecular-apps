from __future__ import division
from __future__ import print_function

from math import sqrt
import numpy as np
import numpy.matlib as ml
import scipy
import scipy.optimize

from . import functions

import logging
logger = logging.getLogger('supramolecular')

class Fitter():
    def __init__(self, function, algorithm='Nelder-Mead'):
        self.function = function
        self.algorithm = algorithm

    def fit(self, data, k_guess, tol=10e-18, niter=None):
        logger.debug("Fitter.fit: called")

        if niter is not None:
            opt = {
                  "maxiter": niter,
                  "maxfev": niter,  
                  }
        else:
            opt = {}

        result = scipy.optimize.minimize(self.function.lstsq,
                                         k_guess,
                                         args=(data, True),
                                         method=self.algorithm,
                                         tol=tol,
                                         options=opt,
                                        )

        self.result = result.x

        logger.debug("Fitter.fit: Result - "+str(result))

    def predict(self, data):
        y, residuals, coeffs, molefrac = self.function.lstsq(self.result, data)
        # PLACEHOLDER add 3rd access to 2D y result
        # In the future many different y results will need to be collated
        # here in a 3D array
        return y[np.newaxis], residuals, coeffs, molefrac
