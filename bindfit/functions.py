"""
" Module containing singletons representing fitter-specific minimisation
" functions.
"
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.matlib as ml

from . import helpers

import logging
logger = logging.getLogger('supramolecular')



#
# Base Function class template
#

class BaseFunction(object):
    # To use, choose an objective function and plotting mixin and create a class
    # like this:
    # class Function(PlotMixin, ObjectiveMixin, BaseFunction)
    #
    # Note the mixins are placed before BaseFunction! This is necessary
    # for the mixin functions to override BaseFunction template functions.
    # See here: https://www.ianlewis.org/en/mixins-and-python

    def __init__(self, f=None, normalise=True, flavour=""):
        self.f         = f
        self.normalise = normalise 
        self.flavour   = flavour

    def objective(self, params, xdata, ydata, scalar=False, *args, **kwargs):
        pass

    def format_x(self, xdata):
        pass

    def format_molefrac(self, molefrac):
        pass

    def format_coeffs(self, fitter, coeffs, ydata_init, h0_init=None):
        pass

    def format_params(self, params_init, params_raw, err):
        pass


#
# Objective function mixins
#

class BindingMixin():
    def objective(self, params, xdata, ydata, 
                  scalar=False, 
                  force_molefrac=False,
                  fit_coeffs=None,
                  *args, **kwargs):
        """
        Objective function:
        Performs least squares regression fitting via matrix division on provided
        NMR/UV dataset for a given binding constant K, and returns its sum of 
        least squares for optimisation OR full parameters, residuals and fitted
        results.

        Arguments:
            params: Parameter  lmfit Parameter object containing binding 
                               constant guesses
            datax:    ndarray    x x m array of x independent variables, m obs
            datay:    ndarray    y x m array of y dependent variables, m obs
            scalar:   bool       
            molefrac: bool       Force molefraction calculation (for UV 
                                 objective functions)

        Returns:
            float:  Sum of least squares
        """

        logger.debug("Function.objective: params, xdata shape, ydata shape")
        logger.debug(params)
        logger.debug(xdata.shape)
        logger.debug(ydata.shape)

        # Calculate predicted HG complex concentrations for this set of 
        # parameters and concentrations
        logger.debug("FLAVOUR RECEIVED BINDINGMIXIN:")
        logger.debug(self.flavour)
        molefrac = self.f(params, xdata, molefrac=force_molefrac, flavour=self.flavour)

        if self.normalise:
            # Don't solve first column (H)
            molefrac = molefrac[1:]

        # Solve by matrix division - linear regression by least squares
        # Equivalent to << coeffs = molefrac\ydata (EA = HG\DA) >> in Matlab

        if fit_coeffs is not None:
            coeffs = fit_coeffs
        else:
            coeffs, _, _, _ = np.linalg.lstsq(molefrac.T, ydata.T)

        # Calculate data from fitted parameters 
        # (will be normalised since input data was norm'd)
        # Result is column matrix - transform this into same shape as input
        # data array
        fit = molefrac.T.dot(coeffs).T

        logger.debug("Function.objective: fit shape")
        logger.debug(fit.shape)

        # Calculate residuals (fitted data - input data)
        residuals = fit - ydata

        # Transpose any column-matrices to rows
        if scalar:
            return np.square(residuals).sum()
        else:
            return fit, residuals, coeffs, molefrac

    def format_x(self, xdata):
        h0 = xdata[0]
        g0 = xdata[1]
        return g0/h0

    def format_molefrac(self, molefrac):
        # Calculate host molefraction from complexes and add as first row
        molefrac_host = np.ones(molefrac.shape[1])
        # TODO: temp delete for testing of initial value non-subtraction
        molefrac_host -= molefrac.sum(axis=0)
        return np.vstack((molefrac_host, molefrac))

    def format_coeffs(self, fitter, coeffs, ydata_init, h0_init=None):
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
            hg  = h + coeffs[0]
            hg2 = h + coeffs[1]
            return np.vstack((h, hg, hg2))
        else:
            pass # Throw error here

    def format_params(self, params_init, params_result, err):
        params = params_init

        for (name, param, stderr) in zip(sorted(params_init), 
                                         params_result, 
                                         err):
            params[name].update({
                "value":  param,
                "stderr": stderr,
                })

        return params

class AggMixin():
    def objective(self, params, xdata, ydata, 
                  scalar=False, 
                  force_molefrac=False,
                  fit_coeffs=None,
                  *args, **kwargs):
        """
        """

        logger.debug("Function.objective: params, xdata shape, ydata shape")
        logger.debug(params)
        logger.debug(xdata.shape)
        logger.debug(ydata.shape)

        # Calculate predicted complex concentrations for this set of 
        # parameters and concentrations
        molefrac = self.f(params, xdata, molefrac=force_molefrac, flavour=self.flavour)
        h  = molefrac[0]
        hs = molefrac[1]
        he = molefrac[2]
        hmat = np.array([h + he/2, hs + he/2])

        logger.debug("Function.objective: molefrac shape")
        logger.debug(molefrac.shape)

        # Solve by matrix division - linear regression by least squares
        # Equivalent to << coeffs = molefrac\ydata (EA = HG\DA) >> in Matlab

        if fit_coeffs is not None:
            coeffs = fit_coeffs
        else:
            coeffs, _, _, _ = np.linalg.lstsq(hmat.T, ydata.T)

        # Calculate data from fitted parameters 
        # (will be normalised since input data was norm'd)
        # Result is column matrix - transform this into same shape as input
        # data array
        fit = hmat.T.dot(coeffs).T

        logger.debug("Function.objective: fit")
        logger.debug(fit)

        # Calculate residuals (fitted data - input data)
        residuals = fit - ydata

        # Transpose any column-matrices to rows
        if scalar:
            return np.square(residuals).sum()
        else:
            return fit, residuals, coeffs, molefrac

    def format_x(self, xdata):
        return xdata[0]

    def format_molefrac(self, molefrac):
        return molefrac

    def format_coeffs(self, fitter, coeffs, ydata_init, h0_init=None):
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

    def format_params(self, params_init, params_result, err):
        params = params_init

        for (name, param, stderr) in zip(sorted(params_init), 
                                         params_result, 
                                         err):
            if name == "ke":
                params[name].update({
                    "value": [param, param/2],    # Calculate Kd if Ke
                    "stderr": [stderr, stderr/2], # parameter given
                    })
            else:
                params[name].update({
                    "value": param,   # Otherwise single param value
                    "stderr": stderr,
                    })

        return params



#
# Final class definitions
#

class FunctionBinding(BindingMixin, BaseFunction):
    pass

class FunctionAgg(AggMixin, BaseFunction):
    pass



#
# log(inhibitor) vs. normalised response test def
#

class FunctionInhibitorResponse(FunctionBinding):
    def objective(self, params, xdata, ydata, scalar=False, *args, **kwargs): 
        logger.debug("FunctionInhibitorResponse.objective: params, xdata, ydata")
        logger.debug(params)
        logger.debug(xdata)
        logger.debug(ydata)

        yfit = self.f(params, xdata)
        yfit = yfit[np.newaxis]

        # Calculate residuals (fitted data - input data)
        residuals = yfit - ydata

        logger.debug("FunctionInhibitorResponse.objective: yfit")
        logger.debug(yfit)

        if scalar:
            logger.debug("FIR.objective: returning residuals sum:")
            logger.debug(np.square(residuals).sum())
            return np.square(residuals).sum()
        else:
            logger.debug("FIR.objective: returning detailed fit:")
            # Transpose any column-matrices to rows
            return yfit, residuals, np.zeros(1, dtype="float64"), np.zeros((1,1), dtype="float64")

def inhibitor_response(params, xdata, *args, **kwargs):
    """
    Calculates predicted [HG] given data object parameters as input.
    """

    # Params sorted in alphabetical order
    hillslope = params[0]
    logIC50   = params[1]

    inhibitor = xdata[1] # xdata[0] is just 1s to fudge geq calc

    response = 100/(1+10**((logIC50 - inhibitor)*hillslope))

    return response

#
# End inhibitor vs. response test func
#


#
# Function definitions
#

def nmr_1to1(params, xdata, *args, **kwargs):
    """
    Calculates predicted [HG] given data object parameters as input.
    """

    k = params[0]
 
    h0 = xdata[0]
    g0 = xdata[1]

    # Calculate predicted [HG] concentration given input [H]0, [G]0 matrices 
    # and Ka guess
    hg = 0.5*(\
             (g0 + h0 + (1/k)) - \
             np.lib.scimath.sqrt(((g0+h0+(1/k))**2)-(4*((g0*h0))))\
             )
    h  = h0 - hg

    # Replace any non-real solutions with sqrt(h0*g0) 
    inds = np.imag(hg) > 0
    hg[inds] = np.sqrt(h0[inds] * g0[inds])

    # Convert [HG] concentration to molefraction for NMR
    hg /= h0
    h  /= h0

    # Make column vector
    #hg_mat = hg[np.newaxis]
    hg_mat = np.vstack((h, hg))

    return hg_mat

def uv_1to1(params, xdata, molefrac=False, *args, **kwargs):
    """
    Calculates predicted [HG] given data object parameters as input.
    """

    k = params[0]
 
    h0 = xdata[0]
    g0 = xdata[1]

    # Calculate predicted [HG] concentration given input [H]0, [G]0 matrices 
    # and Ka guess
    hg = 0.5*(\
             (g0 + h0 + (1/k)) - \
             np.lib.scimath.sqrt(((g0+h0+(1/k))**2)-(4*((g0*h0))))\
             )
    h  = h0 - hg

    # Replace any non-real solutions with sqrt(h0*g0) 
    inds = np.imag(hg) > 0
    hg[inds] = np.sqrt(h0[inds] * g0[inds])

    if molefrac:
        # Convert [HG] concentration to molefraction 
        hg /= h0
        h  /= h0

    # Make column vector
    # hg_mat = hg[np.newaxis]
    hg_mat = np.vstack((h, hg))

    return hg_mat

def uv_1to2(params, xdata, molefrac=False, flavour=""):
    """
    Calculates predicted [HG] and [HG2] given data object and binding constants
    as input.
    """

    k11 = params[0]
    if flavour == "noncoop" or flavour == "stat":
        k12 = k11/4
    else:
        k12 = params[1]
 
    h0 = xdata[0]
    g0 = xdata[1]

    # Calculate free guest concentration [G]: solve cubic
    a = np.ones(h0.shape[0])*k11*k12
    b = 2*k11*k12*h0 + k11 - g0*k11*k12
    c = 1 + k11*h0 - k11*g0
    d = -1. * g0

    # Rows: data points, cols: poly coefficients
    poly = np.column_stack((a, b, c, d))

    # Solve cubic in [G] for each observation
    g = np.zeros(h0.shape[0])
    for i, p in enumerate(poly):
        roots = np.roots(p)

        # Smallest real +ve root is [G]
        select = np.all([np.imag(roots) == 0, np.real(roots) >= 0], axis=0)
        if select.any():
            soln = roots[select].min()
            soln = float(np.real(soln))
        else:
            # No positive real roots, set solution to 0
            soln = 0.0
        
        g[i] = soln

    # Calculate [HG] and [HG2] complex concentrations 
    hg  = h0*((g*k11)/(1+(g*k11)+(g*g*k11*k12)))
    hg2 = h0*(((g*g*k11*k12))/(1+(g*k11)+(g*g*k11*k12)))
    h   = h0 - hg - hg2

    if molefrac:
        # Convert free concentrations to molefractions
        hg  /= h0
        hg2 /= h0
        h   /= h0

    if flavour == "add" or flavour == "stat":
        hg_mat = hg + 2*hg2
        hg_mat = hg_mat[np.newaxis]
    else:
        hg_mat = np.vstack((h, hg, hg2))

    return hg_mat

def nmr_1to2(params, xdata, flavour="", *args, **kwargs):
    """
    Calculates predicted [HG] and [HG2] given data object and binding constants
    as input.
    """

    logger.debug("FLAVOUR RECEIVED NMR1TO2:")
    logger.debug(flavour)

    k11 = params[0]
    if flavour == "noncoop" or flavour == "stat":
        k12 = k11/4
        logger.debug("FLAVOUR: noncoop or stat")
        logger.debug("k11, k12")
        logger.debug(k11)
        logger.debug(k12)
    else:
        k12 = params[1]
        logger.debug("FLAVOUR: none or add")
        logger.debug("k11, k12")
        logger.debug(k11)
        logger.debug(k12)

    h0  = xdata[0]
    g0  = xdata[1]

    # Calculate free guest concentration [G]: solve cubic
    a = np.ones(h0.shape[0])*k11*k12
    b = 2*k11*k12*h0 + k11 - g0*k11*k12
    c = 1 + k11*h0 - k11*g0
    d = -1. * g0

    # Rows: data points, cols: poly coefficients
    poly = np.column_stack((a, b, c, d))

    # Solve cubic in [G] for each observation
    g = np.zeros(h0.shape[0])
    for i, p in enumerate(poly):
        roots = np.roots(p)

        # Smallest real +ve root is [G]
        select = np.all([np.imag(roots) == 0, np.real(roots) >= 0], axis=0)
        if select.any():
            soln = roots[select].min()
            soln = float(np.real(soln))
        else:
            # No positive real roots, set solution to 0
            soln = 0.0
        
        g[i] = soln


    # Calculate [HG] and [HG2] complex concentrations 
    hg  = (g*k11)/(1+(g*k11)+(g*g*k11*k12))
    hg2 = ((g*g*k11*k12))/(1+(g*k11)+(g*g*k11*k12))
    h   = h0 - hg - hg2

    if flavour == "add" or flavour == "stat":
        logger.debug("FLAVOUR: add or stat")
        hg_mat = hg + 2*hg2
        hg_mat = hg_mat[np.newaxis]
    else:
        logger.debug("FLAVOUR: none or noncoop")
        hg_mat = np.vstack((h, hg, hg2))

    return hg_mat

def nmr_2to1(params, xdata, *args, **kwargs):
    """
    Calculates predicted [HG] and [H2G] given data object and binding constants
    as input.
    """

    k11 = params[0]
    k12 = params[1]

    h0  = xdata[0]
    g0  = xdata[1]

    # Calculate free host concentration [H]: solve cubic
    a = np.ones(h0.shape[0])*k11*k12
    b = 2*k11*k12*g0 + k11 - h0*k11*k12
    c = 1 + k11*g0 - k11*h0
    d = -1. * h0

    # Rows: data points, cols: poly coefficients
    poly = np.column_stack((a, b, c, d))

    # Solve cubic in [H] for each observation
    h = np.zeros(h0.shape[0])

    for i, p in enumerate(poly):
        roots = np.roots(p)
        # Smallest real +ve root is [H]
        select = np.all([np.imag(roots) == 0, np.real(roots) >= 0], axis=0)
        if select.any():
            soln = roots[select].min()
            soln = float(np.real(soln))
        else:
            # No positive real roots, set solution to 0
            soln = 0.0

        h[i] = soln

    # Calculate [HG] and [H2G] complex concentrations 
    hg  = (g0*h*k11)/(h0*(1+(h*k11)+(h*h*k11*k12)))
    h2g = (2*g0*h*h*k11*k12)/(h0*(1+(h*k11)+(h*h*k11*k12)))
    h   = h0 - hg - h2g

    hg_mat = np.vstack((h, hg, h2g))

    return hg_mat

def uv_2to1(params, xdata, molefrac=False, flavour=""):
    """
    Calculates predicted [HG] and [H2G] given data object and binding constants
    as input.
    """

    # Convenience
    k11 = params[0]
    k12 = params[1]

    h0  = xdata[0]
    g0  = xdata[1]

    # Calculate free host concentration [H]: solve cubic
    a = np.ones(h0.shape[0])*k11*k12
    b = 2*k11*k12*g0 + k11 - h0*k11*k12
    c = 1 + k11*g0 - k11*h0
    d = -1. * h0

    # Rows: data points, cols: poly coefficients
    poly = np.column_stack((a, b, c, d))

    # Solve cubic in [H] for each observation
    h = np.zeros(h0.shape[0])

    for i, p in enumerate(poly):
        roots = np.roots(p)
        # Smallest real +ve root is [H]
        select = np.all([np.imag(roots) == 0, np.real(roots) >= 0], axis=0)
        if select.any():
            soln = roots[select].min()
            soln = float(np.real(soln))
        else:
            # No positive real roots, set solution to 0
            soln = 0.0

        h[i] = soln

    # Calculate [HG] and [H2G] complex concentrations 
    hg  = g0*((h*k11)/(1+(h*k11)+(h*h*k11*k12)))
    h2g = g0*((2*h*h*k11*k12)/(1+(h*k11)+(h*h*k11*k12)))
    h   = h0 - hg - h2g

    if molefrac:
        # Convert free concentrations to molefractions
        hg  /= h0
        h2g /= h0
        h   /= h0

    hg_mat = np.vstack((h, hg, h2g))

    return hg_mat

def nmr_dimer(params, xdata, *args, **kwargs):
    """
    Calculates predicted [H] [Hs] and [He] given data object and binding
    constant as input.
    """

    ke = params[0]
    h0 = xdata[0]

    if ke == 0:
        # Avoid dividing by zero ...
        return np.array([h0*0, h0*0, h0*0])

    # Calculate free monomer concentration [H] or alpha: 
    # eq 143 from Thordarson book chapter
    h = ((2*ke*h0 + 1) - \
          np.lib.scimath.sqrt(((4*ke*h0 + 1)))\
          )/(2*ke*ke*h0*h0)

    # Calculate "in stack" concentration [Hs] or epislon: eq 149 
    # (rho = 1, n.b. one "h" missing) from Thordarson book chapter
    hs = (h*((h*ke*h0)**2))/((1 - h*ke*h0)**2)

    # Calculate "at end" concentration [He] or gamma: eq 150 (rho = 1) 
    # from Thordarson book chapter
    he = (2*h*h*ke*h0)/(1 - h*ke*h0)

    return np.vstack((h, hs, he)) 

def uv_dimer(params, xdata, *args, **kwargs):
    """
    Calculates predicted [H] [Hs] and [He] given data object and binding
    constant as input.
    """

    ke = params[0]
    h0 = xdata[0]

    if ke == 0:
        # Avoid dividing by zero ...
        return np.array([h0*0, h0*0, h0*0])

    # Calculate free monomer concentration [H] or alpha: 
    # eq 143 from Thordarson book chapter
    h = ((2*ke*h0 + 1) - \
          np.lib.scimath.sqrt(((4*ke*h0 + 1)))\
          )/(2*ke*ke*h0*h0)

    # Calculate "in stack" concentration [Hs] or epislon: eq 149 
    # (rho = 1, n.b. one "h" missing) from Thordarson book chapter
    hs = (h0*h*((h*ke*h0)**2))/((1 - h*ke*h0)**2)

    # Calculate "at end" concentration [He] or gamma: eq 150 (rho = 1) 
    # from Thordarson book chapter
    he = (h0*(2*h*h*ke*h0))/(1 - h*ke*h0)

    # Convert to free concentration
    hc = h0*h

    return np.vstack((hc, hs, he)) 

def nmr_coek(params, xdata, *args, **kwargs):
    """
    Calculates predicted [H] [Hs] and [He] given data object and binding constants
    as input.
    """

    ke = params[0]
    rho = params[1]

    h0  = xdata[0]

    # Calculate free monomer concentration [H] or alpha: 
    # eq 146 from Thordarson book chapter

    a = np.ones(h0.shape[0])*(((ke*h0)**2) - (rho*((ke*h0)**2)))
    b = 2*rho*ke*h0 - 2*ke*h0 - ((ke*h0)**2)
    c = 2*ke*h0 + 1
    d = -1. * np.ones(h0.shape[0])

    # Rows: data points, cols: poly coefficients
    poly = np.column_stack((a, b, c, d))

    # Solve cubic in [H] for each observation
    h = np.zeros(h0.shape[0])
    for i, p in enumerate(poly):
        roots = np.roots(p)

        # Smallest real +ve root is [H]
        select = np.all([np.imag(roots) == 0, np.real(roots) >= 0], axis=0)
        if select.any():
            soln = roots[select].min()
            soln = float(np.real(soln))
        else:
            # No positive real roots, set solution to 0
            soln = 0.0
        
        h[i] = soln

    # Calculate "in stack" concentration [Hs] or epislon: 
    # eq 149 from Thordarson book chapter
    hs = (rho*h*((h*ke*h0)**2))/((1-h*ke*h0)**2)

    # Calculate "at end" concentration [He] or gamma: 
    # eq 150 from Thordarson book chapter
    he = (2*rho*h*h*ke*h0)/(1-h*ke*h0)

    return np.vstack((h, hs, he))

def uv_coek(params, xdata, *args, **kwargs):
    """
    Calculates predicted [H] [Hs] and [He] given data object and binding constants
    as input.
    """

    ke = params[0]
    rho = params[1]

    h0  = xdata[0]

    # Calculate free monomer concentration [H] or alpha: 
    # eq 146 from Thordarson book chapter

    a = np.ones(h0.shape[0])*(((ke*h0)**2) - (rho*((ke*h0)**2)))
    b = 2*rho*ke*h0 - 2*ke*h0 - ((ke*h0)**2)
    c = 2*ke*h0 + 1
    d = -1. * np.ones(h0.shape[0])

    # Rows: data points, cols: poly coefficients
    poly = np.column_stack((a, b, c, d))

    # Solve cubic in [H] for each observation
    h = np.zeros(h0.shape[0])
    for i, p in enumerate(poly):
        roots = np.roots(p)

        # Smallest real +ve root is [H]
        select = np.all([np.imag(roots) == 0, np.real(roots) >= 0], axis=0)
        if select.any():
            soln = roots[select].min()
            soln = float(np.real(soln))
        else:
            # No positive real roots, set solution to 0
            soln = 0.0
        
        h[i] = soln
    
    # n.b. these fractions are multiplied by h0 

    # Calculate "in stack" concentration [Hs] or epislon: eq 149 from Thordarson book chapter
    hs = (h0*rho*h*((h*ke*h0)**2))/((1 - h*ke*h0)**2)

    # Calculate "at end" concentration [He] or gamma: eq 150 from Thordarson book chapter
    he = (h0*(2*rho*h*h*ke*h0))/(1 - h*ke*h0)
        
    # Convert to free concentration
    hc = h0*h

    return np.vstack((hc, hs, he))



def select(key, normalise=True, flavour=""):
    """
    Constructs and returns requested function object.

    Arguments:
        key:     string  Unique fitter function reference string, exposed by 
                         formatter.fitter_list
        flavour: string  Fitter flavour option, if selected
    """

    args_select = {
            "nmrdata":    ["FunctionBinding", ()],
            "nmr1to1":    ["FunctionBinding", (nmr_1to1,  normalise, flavour)],
            "nmr1to2":    ["FunctionBinding", (nmr_1to2,  normalise, flavour)],
            "nmr2to1":    ["FunctionBinding", (nmr_2to1,  normalise, flavour)],
            "uvdata":     ["FunctionBinding", ()],
            "uv1to1" :    ["FunctionBinding", (uv_1to1,   normalise, flavour)],
            "uv1to2" :    ["FunctionBinding", (uv_1to2,   normalise, flavour)],
            "uv2to1" :    ["FunctionBinding", (uv_2to1,   normalise, flavour)],
            "nmrdimer":   ["FunctionAgg",     (nmr_dimer, normalise, flavour)],
            "uvdimer":    ["FunctionAgg",     (uv_dimer,  normalise, flavour)],
            "nmrcoek":    ["FunctionAgg",     (nmr_coek,  normalise, flavour)],
            "uvcoek":     ["FunctionAgg",     (uv_coek,   normalise, flavour)],
            "inhibitor":  ["FunctionInhibitorResponse", (inhibitor_response)],
            }

    # Get appropriate class from global scope
    cls = globals()[args_select[key][0]]

    # Instantiation arguments
    args = args_select[key][1]

    # Construct and return
    return cls(*args)
