from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from django.conf import settings

import os
import numpy as np

from . import functions
from .data import Data
from .fitter import Fitter

@api_view(['POST'])
def fit(request):
    """
    Request:
        input:
            type : string  Type of input file ["csv"]
            value: string  Input data ["/path/to/csv"]

        k_guess  : float   User guess of Ka
        algorithm: string  User selected fitting algorithm

    Response:
        data:       array  [ n x [array of [x, y] points] ]
                           Where n = number of experiments
                           x: Equivalent [G]/[H] concentration
                           y_n: Observed spectrum n
        fit:        array  As for data.
        residuals:
    """

    # Create Data object from input
    if request.data["input"]["type"] == "csv":
        data = Data(os.path.join(settings.MEDIA_ROOT, 
                                 request.data["input"]["value"]))
    else:
        pass
        # Error page

    # Create appropriate Fitter
    fitter = Fitter(functions.NMR1to1)

    # Run fitter on data
    fitter.fit(data, request.data["k_guess"])

    # Build response dict
    k = fitter.result

    observed = []
    predicted = []
    # Loop through each column of observed data and its respective predicted
    # best fit, create array of [x, y] point pairs for plottig
    for o, p in zip(data.observations.T, fitter.predict(data).T):
        geq = data.params["geq"]
        obs_plot  = [ [x, y] for x, y in zip(geq, o) ]
        pred_plot = [ [x, y] for x, y in zip(geq, p) ]
        observed.append(obs_plot)
        predicted.append(pred_plot)

    response = {
               "k": k,
               "data": observed,
               "fit": predicted,
               "residuals": [],
               }

    return Response(response)
