from __future__ import division
from __future__ import print_function

from math import sqrt
import numpy as np
import scipy as sp

def nmr_1to1(k, h0, g0, data):
    """
    Performs least squares regression fitting via matrix division on provided
    NMR data for a given binding constant K, and returns its sum of least
    squares for optimisation.

    Arguments:
        k   : float   Binding constant Ka guess
        h0  : n x 1 matrix  Total [H]0 concentration, n observations
        g0  : n x 1 matrix  Total [G]0 concentration, n observations
        data: n x m matrix  Observed NMR resonances, n observations, m spectra

    Returns:
        matrix  Sum of least squares
    """

    # Calculate [HG] concentration given input [H]0, [G]0 matrices and Ka guess
    hg = 0.5*(\
             (g0 + h0 + (1/k)) - \
             np.lib.scimath.sqrt(((g0+h0+(1/k))**2)-(4*((g0*h0))))\
             )

    # Replace any non-real solutions with sqrt(h0*g0) 
    inds = np.imag(hg) > 0
    hg[inds] = np.sqrt(h0[inds] * g0[inds])

    # Convert [HG] concentration to molefraction for NMR
    hg /= h0
    # Convert to column matrix for matrix calculations
    hg = hg.reshape(len(hg), 1)

    # Solve by matrix division - linear regression by least squares
    # Equivalent to << params = hg\data >> in Matlab
    params, residuals, rank, s = np.linalg.lstsq(hg, data)

    data_calculated = hg * params

    return residuals.sum()
