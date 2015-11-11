from __future__ import division
from __future__ import print_function

from math import sqrt
from copy import deepcopy
import time
import numpy as np
import numpy.matlib as ml
import lmfit

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

    def run(self, params_init):
        """
        Arguments:
            params: dict  Initial parameter guesses for fitter    
        """
        logger.debug("Fitter.fit: called. Input params:")
        logger.debug(params_init)
        
        p = lmfit.Parameters()
        for key, value in params_init.items():
            p.add(key, value=value)
        
        # Run optimizer 
        x = self.xdata
        y = self._preprocess(self.ydata)

        tic = time.clock()
        mini = lmfit.Minimizer(self.function.objective, p, fcn_args=(x, y))
        result = mini.minimize(method="leastsq")
        #ci = lmfit.conf_interval(mini, result)
        #logger.debug(str(lmfit.printfuncs.report_ci(ci)))
        #result = lmfit.minimize(self.function.objective, p, args=(x, y), method="nelder")
        toc = time.clock()

        logger.debug("Fitter.run: FIT FINISHED")
        logger.debug(lmfit.fit_report(result.params))
        if hasattr(result, "success"):
            logger.debug(result.success)
        if hasattr(result, "message"):
            logger.debug(result.message)
        logger.debug(result.nfev)
        logger.debug(result.init_vals)

        # Calculate fitted data with optimised parameters
        fit_norm, residuals, coeffs, molefrac = self.function.objective(
                                                    result.params, 
                                                    x, 
                                                    y, 
                                                    detailed=True)

        # Save time taken to fit
        self.time = toc - tic 

        # Save final optimised parameters and errors as dictionary
        #self.params = result.params.valuesdict()
        self.params = { key: {"value": param.value, "stderr": param.stderr, "init": params_init[key]} for (key, param) in result.params.items() }

        logger.debug("Fitter.run: PARAMS")
        logger.debug(self.params)

        # Postprocess (denormalise) and save fitted data
        fit = self._postprocess(self.ydata, fit_norm)
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
