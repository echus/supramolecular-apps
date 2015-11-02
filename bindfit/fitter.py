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
        self.data = data # Original input data, no processing applied
        self.function = function

        # Fitter options
        self.algorithm = algorithm
        self.normalise = normalise

        # Populated on Fitter.run
        self.params = None
        self.time = None
        self.fit = None
        self.residuals = None
        self.coeffs = None
        self.molefrac = None

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

        # Generate options for optimizer
        if niter is not None:
            opt = {
                  "maxiter": niter,
                  "maxfev": niter,  
                  }
        else:
            opt = {}

        # Run optimizer 
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

        # Time taken to fit
        self.time = toc - tic 

        # Final optimised parameters
        self.params = result.x

        # Calculate fitted data with optimised parameters
        fit_norm, residuals, coeffs, molefrac = self.function.lstsq(self.params, 
                                                   self._preprocess(self.data))

        # Postprocess fitted data (denormalise)
        fit = self._postprocess(fit_norm)

        self.fit = fit
        self.residuals = residuals
        self.coeffs = coeffs

        # Calculate host molefraction from complexes and add as first row
        molefrac_host = np.ones(molefrac.shape[1])
        molefrac_host -= molefrac.sum(axis=0)
        self.molefrac = np.vstack((molefrac_host, molefrac))

    @property
    def statistics(self):
        """
        Return fit statistics

        Returns:
            Standard deviation of calculated y
            Standard deviation of calculated coefficients
            Asymptotic error for non-linear parameter estimate
        """
        fit, residuals, coeffs, molefrac = self.fit # Fit results
        # TODO deal with multi-y fit 3rd axis here? residuals is 3D
        residuals_sum = np.square(residuals) # Sum of squares of residuals

        # PLACEHOLDER add multi-y fit handling here
        # Calculate degrees of freedom 
        # = number of experimental datapoints - number of fitted parameters - 
        # number of calculated coefficients
        n = len(self.data["y"][0]) - len(self.params) - fit[2].size
        sd_y = np.sqrt(ss/n)

        pass
