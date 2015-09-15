"""
" Module containing singletons representing fitter-specific minimisation
" functions.
"
"""

from __future__ import division
from __future__ import print_function

from math import sqrt
import numpy as np
import scipy as sp

import logging
logger = logging.getLogger('supramolecular')

class Function():
    def __init__(self, f):
        self.f = f

    def lstsq(self, k, data, sum_residuals=False):
        """
        Performs least squares regression fitting via matrix division on provided
        NMR data for a given binding constant K, and returns its sum of least
        squares for optimisation.

        Arguments:
            k   : float  Array of binding parameter guesses
            data: dict   Data dictionary of observed NMR resonances

        Returns:
            float:  Sum of least squares
        """

        # Call self.f to calculate predicted HG complex concentrations for this 
        # set of k
        molefrac = self.f(k, data)

        # Solve by matrix division - linear regression by least squares
        # Equivalent to << params = hg\obs >> in Matlab
        coeffs, residuals, rank, s = np.linalg.lstsq(molefrac, data["ynorm"].T)

        if sum_residuals:
            # For use during optimisation
            return residuals.sum()
        else:
            data_calculated = molefrac.dot(coeffs)

            logger.debug("Function.lstsq: linear regression molefrac and coeff")
            logger.debug(molefrac)
            logger.debug(coeffs)

            # Transpose any column-matrices to rows
            return data_calculated.T, residuals, coeffs, molefrac.T



#
# Function definitions
#

def nmr_1to1(k, data):
    """
    Calculates predicted [HG] given data object parameters as input.
    """

    h0  = data["h0"].T
    g0  = data["g0"].T

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
    hg = hg.reshape(len(hg), 1)

    return hg

def uv_1to1(k, data):
    """
    Calculates predicted [HG] given data object parameters as input.
    """

    h0  = data["h0"]
    g0  = data["g0"]

    # Calculate predicted [HG] concentration given input [H]0, [G]0 matrices 
    # and Ka guess
    hg = 0.5*(\
             (g0 + h0 + (1/k)) - \
             np.lib.scimath.sqrt(((g0+h0+(1/k))**2)-(4*((g0*h0))))\
             )

    # Replace any non-real solutions with sqrt(h0*g0) 
    inds = np.imag(hg) > 0
    hg[inds] = np.sqrt(h0[inds] * g0[inds])

    # Make column vector
    hg = hg.reshape(len(hg), 1)

    return hg

def uv_1to2(k, data):
    """
    Calculates predicted [HG] and [HG2] given data object and binding constants
    as input.
    """

    # Convenience
    k11 = k[0]
    k12 = k[1]
    h0 = data["h0"]
    g0 = data["g0"]


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

    hg_mat = np.vstack((hg, hg2))


    # Transpose for matrix calculations
    hg_mat = hg_mat.T

    return hg_mat

def nmr_1to2(k, data):
    """
    Calculates predicted [HG] and [HG2] given data object and binding constants
    as input.
    """

    # Convenience
    k11 = k[0]
    k12 = k[1]
    h0 = data["h0"]
    g0 = data["g0"]


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
    hg_mat = hg_mat.T

    return hg_mat



# Initialise singletons for each function
# Reference by dict key, ultimately exposed to use by formatter.fitter_list 
# dictionary
select = {
        "nmr1to1": Function(nmr_1to1),
        "nmr1to2": Function(nmr_1to2),
        "uv1to1" : Function(uv_1to1),
        "uv1to2" : Function(uv_1to2),
        }
