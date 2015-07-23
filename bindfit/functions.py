from __future__ import division
from __future__ import print_function

from math import sqrt
import numpy as np
import scipy as sp

class Function():
    def __init__(self, f):
        self.f = f

    def lstsq(self, k, data):
        """
        Performs least squares regression fitting via matrix division on provided
        NMR data for a given binding constant K, and returns its sum of least
        squares for optimisation.

        Arguments:
            k   : float   Binding constant Ka guess
            data: Data object of observed NMR resonances

        Returns:
            float:  Sum of least squares
        """

        # Call self.f to calculate predicted [HG] for this k
        hg = self.f(k, data)

        # Convert to column matrix for matrix calculations
        hg = hg.reshape(len(hg), 1)

        # Solve by matrix division - linear regression by least squares
        # Equivalent to << params = hg\obs >> in Matlab
        params, residuals, rank, s = np.linalg.lstsq(hg, data.observations)

        data_calculated = hg * params

        return residuals.sum()



#
# Function definitions
#

def nmr_1to1(k, data):
    """
    Calculates predicted [HG] given data object parameters as input.
    """

    h0  = data.params["h0"]
    g0  = data.params["g0"]

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

    return hg

NMR1to1 = Function(nmr_1to1)
