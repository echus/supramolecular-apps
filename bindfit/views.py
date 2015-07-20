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
        data:
            x : Equivalent Guest
            y1: Proton 1
            y2: Proton 2
            ...
        fit:
            x :
            y1:
            y2:
            ...
        residuals:
            x :
            y1:
            y2:
            ...
    """

    # Create Data object from input
    if request.data["input"]["type"] == "csv":
        data = Data(os.path.join(settings.MEDIA_ROOT, 
                                 request.data["input"]["value"]))
    else:
        pass
        # Error page

    # Create appropriate Fitter
    fitter = Fitter(functions.nmr_1to1)

    # Run fitter on data
    fitter.fit(data, request.data["k_guess"])

    # Build response dict
    response = {
               "k": fitter.result
               }

    return Response(response)
