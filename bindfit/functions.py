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

    def __init__(self, f):
        self.f = f

    def objective(self, params, xdata, ydata, scalar=False, *args, **kwargs):
        pass

    def x_plot(self, xdata):
        pass

    def molefrac_plot(self, molefrac):
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

        logger.debug("Function.objective: params, xdata, ydata")
        logger.debug(params)
        logger.debug(xdata)
        logger.debug(ydata)

        # Calculate predicted HG complex concentrations for this set of 
        # parameters and concentrations
        molefrac = self.f(params, xdata, force_molefrac)

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

        logger.debug("Function.objective: fit")
        logger.debug(fit)

        # Calculate residuals (fitted data - input data)
        residuals = fit - ydata

        # Transpose any column-matrices to rows
        if scalar:
            return np.square(residuals).sum()
        else:
            return fit, residuals, coeffs, molefrac

    def x_plot(self, xdata):
        h0 = xdata[0]
        g0 = xdata[1]
        return g0/h0

    def molefrac_plot(self, molefrac):
        # Calculate host molefraction from complexes and add as first row
        molefrac_host = np.ones(molefrac.shape[1])
        molefrac_host -= molefrac.sum(axis=0)
        return np.vstack((molefrac_host, molefrac))

class DimerMixin():
    def objective(self, params, xdata, ydata, 
                  scalar=False, 
                  force_molefrac=False,
                  fit_coeffs=None,
                  *args, **kwargs):
        """
        """

        logger.debug("Function.objective: params, xdata, ydata")
        logger.debug(params)
        logger.debug(xdata)
        logger.debug(ydata)

        # Calculate predicted complex concentrations for this set of 
        # parameters and concentrations
        molefrac = self.f(params, xdata, force_molefrac)
        h  = molefrac[0]
        hs = molefrac[1]
        he = molefrac[2]
        hmat = np.array([h + he/2, hs + he/2])

        logger.debug("Function.objective: molefrac")
        logger.debug(molefrac)

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

    def x_plot(self, xdata):
        return xdata[0]

    def molefrac_plot(self, molefrac):
        return molefrac


#
# Final class definitions
#

class FunctionBinding(BindingMixin, BaseFunction):
    pass

class FunctionDimer(DimerMixin, BaseFunction):
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
    # if isinstance(params, list):
    #     k = params[0]
    # else:
    #     k = params["k"]
 
    h0 = xdata[0]
    g0 = xdata[1]

    # Calculate predicted [HG] concentration given input [H]0, [G]0 matrices 
    # and Ka guess
    hg = 0.5*(\
             (g0 + h0 + (1/k)) - \
             np.lib.scimath.sqrt(((g0+h0+(1/k))**2)-(4*((g0*h0))))\
             )

    # Replace any non-real solutions with sqrt(h0*g0) 
    inds = np.imag(hg) > 0
    hg[inds] = np.sqrt(h0[inds] * g0[inds])

    # Convert [HG] concentration to molefraction for NMR
    hg /= h0

    # Make column vector
    # hg = hg.reshape(len(hg), 1)
    hg = hg[np.newaxis]

    return hg

def uv_1to1(params, xdata, molefrac=False):
    """
    Calculates predicted [HG] given data object parameters as input.
    """

    k = params[0]
    # if isinstance(params, list):
    #     k = params[0]
    # else:
    #     k = params["k"]
 
    h0 = xdata[0]
    g0 = xdata[1]

    # Calculate predicted [HG] concentration given input [H]0, [G]0 matrices 
    # and Ka guess
    hg = 0.5*(\
             (g0 + h0 + (1/k)) - \
             np.lib.scimath.sqrt(((g0+h0+(1/k))**2)-(4*((g0*h0))))\
             )

    # Replace any non-real solutions with sqrt(h0*g0) 
    inds = np.imag(hg) > 0
    hg[inds] = np.sqrt(h0[inds] * g0[inds])

    if molefrac:
        # Convert [HG] concentration to molefraction 
        hg /= h0

    # Make column vector
    # hg = hg.reshape(len(hg), 1)
    hg = hg[np.newaxis]

    return hg

def uv_1to2(params, xdata, molefrac=False):
    """
    Calculates predicted [HG] and [HG2] given data object and binding constants
    as input.
    """

    k11 = params[0]
    k12 = params[1]
    # if isinstance(params, list):
    #     k11 = params[0]
    #     k12 = params[1]
    # else:
    #     k11 = params["k1"]
    #     k12 = params["k2"]
 
    h0 = xdata[0]
    g0 = xdata[1]

    #
    # Calculate free guest concentration [G]: solve cubic
    #
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

    #
    # Calculate [HG] and [HG2] complex concentrations 
    #
    hg = h0*((g*k11)/(1+(g*k11)+(g*g*k11*k12)))
    hg2 = h0*(((g*g*k11*k12))/(1+(g*k11)+(g*g*k11*k12)))

    if molefrac:
        # Convert free concentrations to molefractions
        hg  /= h0
        hg2 /= h0

    hg_mat = np.vstack((hg, hg2))

    # Transpose for matrix calculations
    # hg_mat = hg_mat.T

    return hg_mat

def nmr_1to2(params, xdata, *args, **kwargs):
    """
    Calculates predicted [HG] and [HG2] given data object and binding constants
    as input.
    """

    k11 = params[0]
    k12 = params[1]
    # if isinstance(params, list):
    #     k11 = params[0]
    #     k12 = params[1]
    # else:
    #     k11 = params["k1"]
    #     k12 = params["k2"]

    h0  = xdata[0]
    g0  = xdata[1]

    #
    # Calculate free guest concentration [G]: solve cubic
    #
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


    #
    # Calculate [HG] and [HG2] complex concentrations 
    #
    hg = (g*k11)/(1+(g*k11)+(g*g*k11*k12))
    hg2 = ((g*g*k11*k12))/(1+(g*k11)+(g*g*k11*k12))

    hg_mat = np.vstack((hg, hg2))


    # Transpose for matrix calculations
    # hg_mat = hg_mat.T

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

    #
    # Calculate free host concentration [H]: solve cubic
    #
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

    #
    # Calculate [HG] and [H2G] complex concentrations 
    #
    hg = (g0*h*k11)/(h0*(1+(h*k11)+(h*h*k11*k12)))
    h2g = (2*g0*h*h*k11*k12)/(h0*(1+(h*k11)+(h*h*k11*k12)))

    hg_mat = np.vstack((hg, h2g))

    # Transpose for matrix calculations
    # hg_mat = hg_mat.T

    return hg_mat

def uv_2to1(params, xdata, molefrac=False):
    """
    Calculates predicted [HG] and [H2G] given data object and binding constants
    as input.
    """

    # Convenience
    k11 = params[0]
    k12 = params[1]
    h0  = xdata[0]
    g0  = xdata[1]

    #
    # Calculate free host concentration [H]: solve cubic
    #
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

    #
    # Calculate [HG] and [H2G] complex concentrations 
    #
    hg = g0*((h*k11)/(1+(h*k11)+(h*h*k11*k12)))
    h2g = g0*((2*h*h*k11*k12)/(1+(h*k11)+(h*h*k11*k12)))

    if molefrac:
        # Convert free concentrations to molefractions
        hg  /= h0
        h2g /= h0

    hg_mat = np.vstack((hg, h2g))

    # Transpose for matrix calculations
    # hg_mat = hg_mat.T

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
    h = ((2*ke*h0+1) - \
          np.lib.scimath.sqrt(((4*ke*h0+1)))\
          )/(2*ke*ke*h0*h0)

    # Calculate "in stack" concentration [Hs] or epislon: eq 149 
    # (rho = 1, n.b. one "h" missing) from Thordarson book chapter
    hs=(h*((h*ke*h0)**2))/((1-h*ke*h0)**2)

    # Calculate "at end" concentration [He] or gamma: eq 150 (rho = 1) 
    # from Thordarson book chapter
    he=(2*h*h*ke*h0)/(1-h*ke*h0)

    return np.vstack((h, hs, he)) 

def uv_dimer(params, xdata, molefrac=False, *args, **kwargs):
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
    h = ((2*ke*h0+1) - \
          np.lib.scimath.sqrt(((4*ke*h0+1)))\
          )/(2*ke*ke*h0*h0)

    # Calculate "in stack" concentration [Hs] or epislon: eq 149 
    # (rho = 1, n.b. one "h" missing) from Thordarson book chapter
    hs=(h0*h*((h*ke*h0)**2))/((1-h*ke*h0)**2)

    # Calculate "at end" concentration [He] or gamma: eq 150 (rho = 1) 
    # from Thordarson book chapter
    he=(h0*(2*h*h*ke*h0))/(1-h*ke*h0)

    # Convert to free concentration
    hc = h0*h

    if molefrac:
        # Convert free concentrations to molefractions
        hc   = h
        hs  /= h0
        he  /= h0

    return np.vstack((hc, hs, he)) 




# Initialise singletons for each function
# Reference by dict key, ultimately exposed to use by formatter.fitter_list 
# dictionary
select = {
        "nmr1to1":    FunctionBinding(nmr_1to1),
        "nmr1to2":    FunctionBinding(nmr_1to2),
        "nmr2to1":    FunctionBinding(nmr_2to1),
        "uv1to1" :    FunctionBinding(uv_1to1),
        "uv1to2" :    FunctionBinding(uv_1to2),
        "uv2to1" :    FunctionBinding(uv_2to1),
        "nmrdimer":   FunctionDimer(nmr_dimer),
        "uvdimer":    FunctionDimer(uv_dimer),
        "inhibitor":  FunctionInhibitorResponse(inhibitor_response),
        }
