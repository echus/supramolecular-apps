from __future__ import division
from __future__ import print_function

from math import sqrt
import numpy as np
import numpy.matlib as ml
import scipy
import scipy.optimize

from . import functions

class Fitter():
    def __init__(self, function, algorithm=None):
        self.function = function
        #self.algorithm = algorithm

    def fit(self, data, k_guess=1000, tol=10e-10):
        # k_optim = scipy.optimize.minimize(functions.nmr_1to1,
        #                               1000,
        #                               args=(h0, g0, data),
        #                               tol=10e-10,
        #                               )
        self.result = scipy.optimize.fmin(self.function.lstsq,
                                          k_guess,
                                          args=(data, True),
                                          xtol=tol,
                                          ftol=tol
                                          )

    def predict(self, data):
        return self.function.lstsq(self.result, data) 
