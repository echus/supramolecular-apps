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
    def __init__(self, xdata, ydata, function, normalise=True, params=None):
        self.xdata = xdata # Original input data, no processing applied
        self.ydata = ydata # Original input data, no processing applied
        self.function = function

        # Fitter options
        self.normalise   = normalise

        # Populated on Fitter.run
        self._params_raw = None
        self.params      = params # Initialise with optimised param results
                                  # from previous run
                                  # Used with post-fit Monte Carlo calculation
        self.time        = None
        self.fit         = None
        self.residuals   = None
        self.coeffs      = None
        self.molefrac    = None

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

    def run_scipy(self, params_init, save=True, xdata=None, ydata=None, method='Nelder-Mead'):
        """
        Arguments:
            params_init: dict  Initial parameter guesses for fitter    
            save:        bool  If True, process and save optimisation results
                               If False, return raw optimised params
            xdata:       array Modified input array 
            ydata:       array Modified input array 
                               (used with save=False for Monte Carlo error 
                               calculation)
        """
        logger.debug("Fitter.fit: called. Input params:")
        logger.debug(params_init)

        # Set input data
        x = self.xdata if xdata is None else xdata
        y = self._preprocess(self.ydata if ydata is None else ydata)
        
        # Sort parameter dict into ordered array of parameters and bounds
        p = []
        b = []
        for key, value in sorted(params_init.items()):
            p.append(value["init"])
            b.append([value["bounds"]["min"],
                      value["bounds"]["max"]])

        logger.debug("Fitter.fit: params and bounds read:")
        logger.debug(p)
        logger.debug(b)

        # Run optimizer 
        tic = time.clock()
        result = scipy.optimize.minimize(self.function.objective,
                                         p,
                                         bounds=b,
                                         args=(x, y, True),
                                         method=method if method else "Nelder-Mead",
                                         tol=1e-18,
                                        )
        toc = time.clock()

        logger.debug("Fitter.run: FIT FINISHED")
        logger.debug("Fitter.run: Fitter.function")
        logger.debug(self.function)
        logger.debug("Fitter.run: result.x")
        logger.debug(result.x)

        # Calculate fitted data with optimised parameters
        # Force molefraction (not free concentration) calculation for proper 
        # fitting in UV models
        logger.debug("Fitter.run: Calculating optimised fit")
        logger.debug("Fitter.run: self.ydata vs. ydata")
        logger.debug(self.ydata)
        logger.debug(ydata)
        ydata_init = self.ydata[:,0] if ydata is None else ydata[:,0]
        logger.debug("Fitter.run: ydata_init")
        logger.debug(ydata_init)
        fit_norm, residuals, coeffs_raw, molefrac_raw, coeffs, molefrac = self.function.objective(result.x, x, y, scalar=False, ydata_init=ydata_init)

        # Postprocessing
        # Populate fit results dict
        results = {}

        # Save time taken to fit
        results["time"] = toc - tic 

        # Save raw optimised params arra
        results["_params_raw"] = result.x

        # Postprocess (denormalise) and save fitted data
        fit = self._postprocess(self.ydata if ydata is None else ydata, 
                                fit_norm)
        results["fit"] = fit

        results["residuals"] = residuals

        results["coeffs"] = coeffs
        results["coeffs_raw"] = coeffs_raw

        # Calculate final molefrac (including host, function-specific)
        results["molefrac"] = molefrac
        results["molefrac_raw"] = molefrac_raw

        # Calculate fit uncertainty statistics
        logger.debug("Fitter.run: Calculating uncertainty statistics")
        err = self.statistics(result.x, fit, coeffs_raw, residuals)
        logger.debug("Fitter.run: Done calculating uncertainty statistics")

        # Parse final optimised parameters and errors into parameters dict
        results["params"] = self.function.format_params(params_init, 
                                                        result.x, 
                                                        err)

        if save:
            # Save fit results dict to object instance
            for key, value in results.items():
                setattr(self, key, value)

            #logger.debug("Fitter.run: CALCULATING MONTE CARLO (TEST)")
            #self.calc_monte_carlo(5, [0.02, 0.01], 0.005)
        else:
            # Return results dict without saving
            return results

    def statistics(self, params, fit, coeffs, residuals):
        """
        Return fit statistics after parameter optimisation

        Returns:
            Asymptotic error for non-linear parameter estimate
            # Standard deviation of calculated y
            # Standard deviation of calculated coefficients
        """
        # Calculate deLevie uncertainty
        d = np.float64(1e-6) # delta
         
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
            ydata_init = self.ydata[:,0]
            fit_shift_norm, _, _, _, _, _ = self.function.objective(
                    params_shift, 
                    x, 
                    y, 
                    scalar=False,
                    ydata_init=ydata_init,
                    fit_coeffs=coeffs)
            fit_shift = self._postprocess(self.ydata, fit_shift_norm)
            
            # Calculate partial differential
            # Flatten numerator into 1D array (TODO: is this correct?)
            num   = (fit_shift - fit).flatten()
            denom = pi_shift - pi
            diffs.append(np.divide(num, denom))

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
        ssr = np.sum(np.square(residuals))
        # Degrees of freedom:
        # N datapoints - N fitted params - N calculated coefficients
        d_free = self.ydata.size - len(params) - coeffs.size

        sigma = np.sqrt((m_diag*ssr)/(d_free - 1))

        # 3. Calculate confidence intervals
        # Calculate t-value at 95%
        # Studnt, n=d_free, p<0.05, 2-tail
        t = stats.t.ppf(1 - 0.025, d_free)

        ci = np.array([params - t*sigma, params + t*sigma])
        ci_percent = (t*sigma)/params * 100

        return ci_percent

    def calc_monte_carlo(self, n_iter, xdata_error, ydata_error, method=None):
        """
        Calculate error on fit using a Monte Carlo method

        Arguments:
            n_iter:      Number of Monte Carlo iterations
            xdata_error: n array of n percentage errors corresponding to n 
                         rows of xdata
            ydata_error: Float corresponding to n percentage error on each
                         row of ydata

        Returns:
            something
        """

        xdata = self.xdata
        ydata = self.ydata
        
        # Convert params results to params_init array to use as input
        # to run_scipy
        # TODO: make params_init same format as params result
        # params_init = {}
        # for key, param in self.params.items():
        #     params_init[key] = param["value"]

        # Copy parameter results array and set inital values to optimised
        # parameter results to use as input to run_scipy
        params_init = {}
        for key, param in self.params.items():
            params_init[key] = param
            params_init[key]["init"] = param["value"]

        params_arr = np.zeros((n_iter, len(params_init)))

        for n in range(n_iter):
            # Calculate error multiplier arrays matching ydata, xdata shapes
            xdata_error_arr = np.random.standard_normal(xdata.shape)\
                              *ml.repmat(xdata_error, xdata.shape[1], 1).T\
                              + 1
            ydata_error_arr = np.random.standard_normal(ydata.shape)\
                              *ydata_error\
                              + 1

            # Calculated shifted input data
            xdata_shift = xdata*xdata_error_arr
            ydata_shift = ydata*ydata_error_arr

            logger.debug("Fitter.monte_carlo: params_init")
            logger.debug(params_init)

            results = self.run_scipy(params_init=params_init,
                                     save       =False, 
                                     xdata      =xdata_shift, 
                                     ydata      =ydata_shift,
                                     method     =method)

            logger.debug("Fitter.monte_carlo: results")
            logger.debug(results)

            # Log resulting params
            params_arr[n] = results["_params_raw"]

        percentile_params = np.percentile(params_arr, [2.5, 97.5], axis=0).T

        logger.debug("Fitter.monte_carlo: params_arr, percentile_params")
        logger.debug(params_arr)
        logger.debug(percentile_params)

        # Calculate errors and update input params dict with results
        for i, (key, param) in enumerate(sorted(self.params.items())):
            p = param["value"]            # Actual param result
            per = percentile_params[i] # Calc'd percentile for this param

            lower = (100*(per[0] - p))/p
            upper = (100*(per[1] - p))/p

            param["mc"] = [lower, upper]

        logger.debug("Fitter.monte_carlo: updated params dict")
        logger.debug(self.params)

        return self.params
