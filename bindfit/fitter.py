from __future__ import division
from __future__ import print_function

from math import sqrt
from copy import deepcopy
import time
import numpy as np
import numpy.matlib as ml
import scipy
import scipy.optimize

from . import functions
from . import helpers 

import logging
logger = logging.getLogger('supramolecular')

class Fitter():
    def __init__(self, data, function, 
                 algorithm='Nelder-Mead',
                 normalise=True):
        # Original input data, no processing applied
        self.data = data
        self.function = function

        self.algorithm = algorithm

        self.normalise = normalise

        self.params = None
        self.time = None

    def _preprocess(self, data):
        # Preprocess data based on Fitter options
        # Returns modified processed copy of input data
        d = deepcopy(data)

        if self.normalise:
            d["y"] = helpers.normalise(d["y"])

        return d

    def _postprocess(self, fit):
        # Postprocess fitted data based on Fitter options 
        f = fit

        if self.normalise:
            f = helpers.denormalise(self.data["y"], fit)

        return f

    def run(self, k_guess, tol=10e-18, niter=None):
        logger.debug("Fitter.fit: called")

        if niter is not None:
            opt = {
                  "maxiter": niter,
                  "maxfev": niter,  
                  }
        else:
            opt = {}

        tic = time.clock()
        result = scipy.optimize.minimize(self.function.lstsq,
                                         k_guess,
                                         args=(self._preprocess(self.data), 
                                               True),
                                         method=self.algorithm,
                                         tol=tol,
                                         options=opt,
                                        )
        toc = time.clock()

        # Set optimised parameters
        self.params = result.x
        self.time = toc - tic 

        logger.debug("Fitter.fit: params - "+str(self.params))

    @property
    def fit(self):
        """
        Return fitted parameters and data
        """
        # Calculate fitted data with optimised parameters
        fit_norm, residuals, coeffs, molefrac = self.function.lstsq(self.params, 
                                                   self._preprocess(self.data))

        # Postprocess data (denormalise)
        fit = self._postprocess(fit_norm)

        return fit, residuals, coeffs, molefrac
