from rest_framework.views import APIView

from rest_framework.parsers import JSONParser, MultiPartParser 

from rest_framework.response import Response
from rest_framework import status

from django.contrib.sites.models import Site

from django.conf import settings

import os
import hashlib
import numpy as np

from . import functions
from .data import Data
from .fitter import Fitter

import logging
logger = logging.getLogger('supramolecular')

class FitView(APIView):
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

            params   : array   Array of objects
                [
                {value: float},   User guess of first parameter
                {value: float},   User guess of first parameter
                ...  : ...
                ]

            algorithm: string  User selected fitting algorithm

            Note: Each param represented as dictionary instead of simple
                  primitive to avoid issues with binding to arrays of 
                  primitives in some JS frontend frameworks.

        Response:
            data     : array  [ n x [array of [x, y] points] ]
                               Where n = number of experiments
                               x: Equivalent [G]/[H] concentration
                               y_n: Observed spectrum n
            fit      : array  As for data.
            residuals:
            params   :
                [
                {value: float},   User guess of first parameter
                {value: float},   User guess of first parameter
                ...  : ...
                ]
        """

        logger.debug("FitterView.post: called")

        # Parse request
        self.fitter = request.data["fitter"]

        params = request.data["params"]
        # Convert params dictionary to array for input to fitter
        self.params = [ float(p["value"]) for p in request.data["params"] ]

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

        # Convert fit result params to dictionaries for response
        params = [ {"value": param} for param in fit.result ]

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



class FitOptionsView(APIView):
    parser_classes = (JSONParser,)

    # Default options for each fitter type
    default_options_select = {
            "nmr1to1": {
                "fitter": "nmr1to1",
                "input": {
                    "type": "csv",
                    "value": "input.csv",
                    },
                "params": [
                    {"value": 1000},
                    ],
                },
            "nmr1to2": {
                "fitter": "nmr1to2",
                "input": {
                    "type": "csv",
                    "value": "input.csv",
                    },
                "params": [
                    {"value": 10000},
                    {"value": 1000},
                    ],
                },
            "uv1to1": {
                "fitter": "uv1to1",
                "input": {
                    "type": "csv",
                    "value": "input.csv",
                    },
                "params": [
                    {"value": 1000},
                    ],
                },
            "uv1to2": {
                "fitter": "uv1to2",
                "input": {
                    "type": "csv",
                    "value": "input.csv",
                    },
                "params": [
                    {"value": 10000},
                    {"value": 1000},
                    ],
                },
            }


    def post(self, request):
        fitter = request.data["fitter"]
        response = self.default_options_select[fitter]
        return Response(response)



class FitLabelsView(APIView):
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
                    "label": "[UV LABEL]",
                    "units": "[UV UNITS]",
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
                    "label": "[UV LABEL]",
                    "units": "[UV UNITS]",
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



class FitListView(APIView):
    parser_classes = (JSONParser,)
    
    def get(self, request):
        fitter_list = [
                {"name": "NMR 1:1", "key": "nmr1to1"},
                {"name": "NMR 1:2", "key": "nmr1to2"},
                {"name": "UV 1:1",  "key": "uv1to1"},
                {"name": "UV 1:2",  "key": "uv1to2"},
                ]
        return Response(fitter_list)



class FitExportView(APIView):
    parser_classes = (JSONParser,)
    
    def post(self, request):
        # Get data
        data   = np.array(request.data["data"]["data"])
        fit    = np.array(request.data["data"]["fit"])
        params = np.array([ p["value"] for p in request.data["data"]["params"] ])

        ncols = 1 + len(data)*2 # Number of cols in exported data =
                                # x axis + data y axes + fit y axes
        nrows = len(data[0])

        # Generate appropriate header for csv
        names = ["Equivalent total [G]0/[H]0",]
        names.extend([ "Data "+str(i) for i in range(len(data)) ])
        names.extend([  "Fit "+str(i) for i in range(len(fit))  ])
        header = ",".join(names)
        footer = ",".join([ str(p) for p in params ])
        
        # Init output array
        output = np.zeros((nrows, ncols), dtype='f8')

        # Populate x, y data and fits
        output[:,0] = np.array(data[0])[:,0] # x axis
        i = 1
        for d in data:
            output[:, i] = np.array(d)[:,1] # y axis
            i += 1

        for f in fit:
            output[:, i] = np.array(f)[:,1] # y axis
            i += 1

        export_path = os.path.join(settings.MEDIA_ROOT, "output.csv") 
        np.savetxt(export_path, output, header=header, footer=footer, fmt="%.18f", delimiter=",")

        export_url = settings.ROOT_URL+settings.MEDIA_URL+"output.csv"

        return Response({"url":export_url})



class UploadView(APIView):
    """
    Request:

    Response:
        string: Name of uploaded file on server
    """

    REQUEST_KEY = "input"
    MEDIA_INPUT_DIR = "input" 

    parser_classes = (MultiPartParser, )

    def put(self, request):
        f = request.FILES[self.REQUEST_KEY]

        # TODO chunk this pls
        buf = f.read()
        filename = self.generate_filename(buf)
        upload_path = os.path.join(settings.MEDIA_ROOT, 
                                   self.MEDIA_INPUT_DIR, 
                                   filename) 

        logger.debug("UploadView.put: called")
        logger.debug("UploadView.put: f - "+str(f))

        with open(upload_path, 'wb+') as destination:
            destination.write(buf)
            logger.debug("UploadView.put: f written to destination "+destination.name)

        filepath = os.path.join(self.MEDIA_INPUT_DIR, filename)
        response_dict = {
                "filename": filepath,
                }

        return Response(response_dict, status=200)

    def generate_filename(self, buf): 
        # Use SHA1 hash as filename
        hasher = hashlib.sha1()
        hasher.update(buf)
        filename = hasher.hexdigest()
        return filename
