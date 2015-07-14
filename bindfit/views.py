from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from django.conf import settings

import os
import numpy as np

from . import fitter

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

    # Process request

    # Import csv to numpy array
    #fn = request["input"]["value"]

    with open(os.path.join(settings.MEDIA_ROOT, "input.csv")) as f:
        raw_data = np.loadtxt(f, delimiter=",", skiprows=1)

    h0   = raw_data[:,0]
    g0   = raw_data[:,1]
    data = raw_data[:,2:]

    # Call fitting algorithm
    k = fitter.fit_nmr_1to1(h0, g0, data)

    # Build response dict
    response = {
            "k": k,
            }

    return Response(response)
