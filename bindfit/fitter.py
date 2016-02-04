from __future__ import division
from __future__ import print_function

from math import sqrt
from copy import deepcopy
import time
from itertools import product

import numpy as np
import numpy.matlib as ml

import scipy
import scipy.optimize
from scipy import stats

from . import functions
from . import helpers 

import logging
logger = logging.getLogger('supramolecular')

class Fitter():
    def __init__(self, xdata, ydata, function, normalise=True):
        self.xdata = xdata # Original input data, no processing applied
        self.ydata = ydata # Original input data, no processing applied
        self.function = function

        # Fitter options
        self.normalise = normalise

        # Populated on Fitter.run
        self.params = None
        self.time = None
        self.fit = None
        self.residuals = None
        self.coeffs = None
        self.molefrac = None

    def _preprocess(self, ydata):
        # Preprocess data based on Fitter options
        # Returns modified processed copy of input data
        d = ydata

        if self.normalise:
            d = helpers.normalise(d)

        return d

    def _postprocess(self, ydata, yfit):
        # Postprocess fitted data based on Fitter options 
        f = yfit 

        if self.normalise:
            f = helpers.denormalise(ydata, yfit)

        return f 

    def run_scipy(self, params_init):
        """
        Arguments:
            params: dict  Initial parameter guesses for fitter    
        """
        logger.debug("Fitter.fit: called. Input params:")
        logger.debug(params_init)
        
        p = []
        for key, value in sorted(params_init.items()):
            p.append(value)
        
        # Run optimizer 
        x = self.xdata
        y = self._preprocess(self.ydata)

        tic = time.clock()
        result = scipy.optimize.minimize(self.function.objective,
                                         p,
                                         args=(x, y, True),
                                         method='Nelder-Mead',
                                         tol=1e-18,
                                        )
        toc = time.clock()

        logger.debug("Fitter.run: FIT FINISHED")
        logger.debug("Fitter.run: Fitter.function")
        logger.debug(self.function)
        logger.debug("Fitter.run: result.x")
        logger.debug(result.x)

        # Calculate fitted data with optimised parameters
        fit_norm, residuals, coeffs, molefrac = self.function.objective(
                                                    result.x, 
                                                    x, 
                                                    y, 
                                                    detailed=True,
                                                    force_molefrac=True)

        # Save time taken to fit
        self.time = toc - tic 

        # Save raw optimised params arra
        self._params_raw = result.x

        # Postprocess (denormalise) and save fitted data
        fit = self._postprocess(self.ydata, fit_norm)
        self.fit = fit

        self.residuals = residuals

        self.coeffs = coeffs

        # Calculate host molefraction from complexes and add as first row
        molefrac_host = np.ones(molefrac.shape[1])
        molefrac_host -= molefrac.sum(axis=0)
        self.molefrac = np.vstack((molefrac_host, molefrac))

        # Calculate fit uncertainty statistics
        ci = self.statistics()

        # Save final optimised parameters and errors as dictionary
        self.params = { name: {"value": param, "stderr": stderr, "init": params_init[name]} 
                        for (name, param, stderr) 
                        in zip(sorted(params_init), result.x, ci) }

        logger.debug("Fitter.run: PARAMS DICT")
        logger.debug(self.params)

    def statistics(self):
        """
        Return fit statistics after parameter optimisation

        Returns:
            Asymptotic error for non-linear parameter estimate
            # Standard deviation of calculated y
            # Standard deviation of calculated coefficients
        """
        # Calculate deLevie uncertainty
        d = 1e-6 # delta
        params = self._params_raw

        # 0. Calculate partial differentials for each parameter
        diffs = []
        for i, pi in enumerate(params):
            # Shift the ith parameter's value by delta
            pi_shift = pi*(1 + d)
            params_shift = np.copy(params)
            params_shift[i] = pi_shift

            # Calculate fit with modified parameter set
            x   = self.xdata
            y   = self._preprocess(self.ydata)
            fit_shift_norm, _, _, _ = self.function.objective(params_shift, 
                                                        x, 
                                                        y, 
                                                        detailed=True,
                                                        force_molefrac=True)
            fit_shift = self._postprocess(self.ydata, fit_shift_norm)
            
            # Calculate partial differential
            # Flatten numerator into 1D array (TODO: is this correct?)
            num   = (fit_shift - self.fit).flatten()
            denom = pi_shift - pi
            diffs.append(num/denom)

        diffs = np.array(diffs)

        # 1. Calculate PxP matrix M and invert
        P = len(params)
        M = np.zeros((P, P))
        for i, j in product(range(P), range(P)):
            M[i, j] = np.sum(diffs[i]*diffs[j])
        M_inv = np.linalg.inv(M)
        m_diag = np.diagonal(M_inv)

        # 2. Calculate standard deviations sigma of P parameters pi
        # Sum of squares of residuals
        ssr = np.sum(np.square(self.residuals))
        # Degrees of freedom:
        # N datapoints - N fitted params - N calculated coefficients
        d_free = len(self.ydata[0]) - len(params) - self.coeffs.size

        # TODO why d_free - 1?
        sigma = np.sqrt((m_diag*ssr)/d_free)

        # 3. Calculate confidence intervals
        # Calculate t-value at 95%
        # Studnt, n=d_free, p<0.05, 2-tail
        t = stats.t.ppf(1 - 0.025, d_free)

        ci = np.array([params - t*sigma, params + t*sigma])
        ci_percent = (100*t*sigma)/params

        return ci_percent
