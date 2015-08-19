from rest_framework.views import APIView

from rest_framework.parsers import JSONParser, MultiPartParser 

from rest_framework.response import Response
from rest_framework import status

from django.conf import settings

import os
import numpy as np

from . import functions
from .data import Data
from .fitter import Fitter

import logging
logger = logging.getLogger('supramolecular')

class FitterView(APIView):
    parser_classes = (JSONParser,)

    # JSON fitter reference -> "functions" fitter function map 
    fitter_select = {
            "nmr1to1": "NMR1to1",
            "nmr1to2": "NMR1to2",
            "uv1to1":  "UV1to1",
            "uv1to2":  "UV1to2",
            }

    def post(self, request):
        """
        Request:
            input:
                type : string  Type of input file ["csv"]
                value: string  Input data ["/path/to/csv"]

            params   :
                0    : float   User guess of first parameter
                1    : float   User guess of second parameter
                ...  : ...

            algorithm: string  User selected fitting algorithm

            Note: params represented as dictionary instead of array to 
                  avoid issues with binding to arrays of primitives in JS 
                  frontend frameworks.

        Response:
            data     : array  [ n x [array of [x, y] points] ]
                               Where n = number of experiments
                               x: Equivalent [G]/[H] concentration
                               y_n: Observed spectrum n
            fit      : array  As for data.
            residuals:
            params   :
                0    : float   Fitted first parameter
                1    : float   Fitted second parameter
                ...  : ...
        """

        logger.debug("FitterView.post: called")

        # Parse request
        self.fitter = request.data["fitter"]

        params = request.data["params"]
        # Convert params dictionary to array for input to fitter
        self.params = [ params[str(i)] for i in range(len(params)) ]

        # Import data
        self.data = self.import_data(request.data["input"]["type"], 
                                     request.data["input"]["value"])

        # Call appropriate fitter
        self.fit = self.run_fitter()
        
        # Build response dict
        response = self.build_response()
        return Response(response)

    def import_data(self, fmt, value):
        # Import input file into Data object
        if fmt == "csv":
            input_path = os.path.join(settings.MEDIA_ROOT, value)
            data = Data(input_path)
        else:
            pass
            # Error response 

        return data

    def build_response(self):
        data = self.data
        fit  = self.fit

        # Loop through each column of observed data and its respective predicted
        # best fit, create array of [x, y] point pairs for plotting
        observed = []
        predicted = []
        for o, p in zip(data.observations.T, fit.predict(data).T):
            geq = data.params["geq"]
            obs_plot  = [ [x, y] for x, y in zip(geq, o) ]
            pred_plot = [ [x, y] for x, y in zip(geq, p) ]
            observed.append(obs_plot)
            predicted.append(pred_plot)

        # Convert fit result params to dictionary for response
        params = { i: param for (i, param) in enumerate(fit.result) }

        response = {
                "params": params,
                "data": observed,
                "fit": predicted,
                "residuals": [],
                }

        return response

    def run_fitter(self):
        # Initialise appropriate Fitter
        function = getattr(functions, self.fitter_select[self.fitter])
        fitter = Fitter(function)

        # Run fitter on data
        fitter.fit(self.data, self.params)

        logger.debug("FitterView.post: NMR1to1 fit")
        logger.debug("FitterView.post: fitter.result = "+str(fitter.result))
        logger.debug("FitterView.post: data.observations = "+str(self.data.observations))
        logger.debug("FitterView.post: fitter.predict(data) = "+str(fitter.predict(self.data)))

        return fitter 



class FitterOptionsView(APIView):
    parser_classes = (JSONParser,)

    # Default options for each fitter type
    default_options_select = {
            "nmr1to1": {
                "fitter": "nmr1to1",
                "input": {
                    "type": "csv",
                    "value": "input.csv",
                    },
                "params": {
                    0: 1000,
                    },
                },
            "nmr1to2": {
                "fitter": "nmr1to2",
                "input": {
                    "type": "csv",
                    "value": "input.csv",
                    },
                "params": {
                    0: 10000,
                    1: 1000,
                    },
                },
            "uv1to1": {
                "fitter": "uv1to1",
                "input": {
                    "type": "csv",
                    "value": "input.csv",
                    },
                "params": {
                    0: 1000,
                    },
                },
            "uv1to2": {
                "fitter": "uv1to2",
                "input": {
                    "type": "csv",
                    "value": "input.csv",
                    },
                "params": {
                    0: 10000,
                    1: 1000,
                    },
                },
            }


    def post(self, request):
        fitter = request.data["fitter"]
        response = self.default_options_select[fitter]
        return Response(response)



class FitterLabelsView(APIView):
    parser_classes = (JSONParser,)
    
    # Labels for each fitter type
    label_select = {
            "nmr1to1": {
                "x": {
                    "label": "Equivalent total [G]\u2080/[H]\u2080",
                    "units": "",
                    },
                "y": {
                    "label": "\u03B4",
                    "units": "ppm",
                    },
                "params": [
                    {"label": "K", "units": "M\u207B\u00B9"},
                    ]
                },
            "nmr1to2": {
                "x": {
                    "label": "Equivalent total [G]\u2080/[H]\u2080",
                    "units": "",
                    },
                "y": {
                    "label": "\u03B4",
                    "units": "ppm",
                    },
                "params": [
                    {"label": "K\u2081\u2081", "units": "M\u207B\u00B9"},
                    {"label": "K\u2081\u2082", "units": "M\u207B\u00B9"},
                    ]
                },
            "uv1to1": {
                "x": {
                    "label": "Equivalent total [G]\u2080/[H]\u2080",
                    "units": "",
                    },
                "y": {
                    "label": "\u03B4",
                    "units": "ppm",
                    },
                "params": [
                    {"label": "K", "units": "M\u207B\u00B9"},
                    ]
                },
            "uv1to2": {
                "x": {
                    "label": "Equivalent total [G]\u2080/[H]\u2080",
                    "units": "",
                    },
                "y": {
                    "label": "\u03B4",
                    "units": "ppm",
                    },
                "params": [
                    {"label": "K\u2081\u2081", "units": "M\u207B\u00B9"},
                    {"label": "K\u2081\u2082", "units": "M\u207B\u00B9"},
                    ]
                },
            }

    def post(self, request):
        logger.debug("FitterLabelsView.post: called")
        logger.debug(request.data)

        fitter = request.data["fitter"]
        response = self.label_select[fitter]
        return Response(response)



class FitterListView(APIView):
    parser_classes = (JSONParser,)
    
    def get(self, request):
        fitter_list = [
                {"name": "NMR 1:1", "key": "nmr1to1"},
                {"name": "NMR 1:2", "key": "nmr1to2"},
                {"name": "UV 1:1",  "key": "uv1to1"},
                {"name": "UV 1:2",  "key": "uv1to2"},
                ]
        return Response(fitter_list)



class UploadView(APIView):
    """
    Request:

    Response:
        string: Path to uploaded file on server
    """

    REQUEST_FILENAME = "input" 

    parser_classes = (MultiPartParser, )

    def put(self, request):
        f = request.FILES[self.REQUEST_FILENAME]

        filename = "input.csv"
        upload_path = os.path.join(settings.MEDIA_ROOT, filename) 

        logger.debug("UploadView.put: called")
        logger.debug("UploadView.put: f - "+str(f))

        with open(upload_path, 'wb+') as destination:
            destination.write(f.read())
            logger.debug("UploadView.put: f written to destination "+destination.name)

        response_dict = {
                "filename": filename,
                }

        return Response(response_dict, status=200)
