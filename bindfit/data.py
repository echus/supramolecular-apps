from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.matlib as ml

class Data():
    def __init__(self, path):
        self._import_nmr_csv(path)
        self._normalise_observations()

    def _import_nmr_csv(self, path):
        with open(path) as f:
            raw = np.loadtxt(f, delimiter=",", skiprows=1)

        self.params = {
                      "h0": raw[:,0],
                      "g0": raw[:,1],
                      "geq": raw[:,1]/raw[:,0],
                      }
        
        self.observations = raw[:,2:]
        self.observations_initial = self.observations[0,:]

    def _normalise_observations(self):
        initialmat = ml.repmat(self.observations_initial, 
                               len(self.observations), 
                               1)
        self.observations -= initialmat
