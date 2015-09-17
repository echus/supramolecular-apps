from rest_framework.views import APIView

from rest_framework.parsers import JSONParser, MultiPartParser 

from rest_framework.response import Response
from rest_framework import status

from django.contrib.sites.models import Site

from django.conf import settings

import os
import string
import random
import pandas as pd
import numpy  as np

from . import models
from . import formatter
from . import functions
from .fitter import Fitter

import logging
logger = logging.getLogger('supramolecular')

class FitView(APIView):
    parser_classes = (JSONParser,)

    def post(self, request):
        """
        Request:
            data_id  : string  Reference to input data to use

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

        # Parse request options
        self.fitter = request.data["fitter"]

        params = request.data["params"]
        # Convert params dictionary to array for input to fitter
        self.params = [ float(p["value"]) for p in request.data["params"] ]

        # Get input data
        data = models.Data.objects.get(id=request.data["data_id"])
        self.data = data.to_dict()

        # Call appropriate fitter
        self.fit = self.run_fitter()
        
        # Build response dict
        response = self.build_response()
        return Response(response)

    def build_response(self):
        data = self.data
        fit  = self.fit.predict(self.data)
        params = self.fit.result

        response = {
                "data": self.data,
                "fit" : formatter.fit(y=fit[0],
                                      params=params,
                                      residuals=fit[1],
                                      coeffs=fit[2],
                                      molefrac=fit[3])
                }

        return response

    def run_fitter(self):
        # Initialise appropriate Fitter with specified minimisation function
        function = functions.select[self.fitter]
        fitter = Fitter(function)

        # Run fitter on data
        fitter.fit(self.data, self.params)

        return fitter 



class FitOptionsView(APIView):
    parser_classes = (JSONParser,)

    def post(self, request):
        return Response(formatter.options(request.data["fitter"]))



class FitLabelsView(APIView):
    parser_classes = (JSONParser,)
    
    def post(self, request):
        return Response(formatter.labels(request.data["fitter"]))



class FitListView(APIView):
    parser_classes = (JSONParser,)
    
    def get(self, request):
        return Response(formatter.fitter_list())



class FitSaveView(APIView):
    parser_classes = (JSONParser,)

    def post(self, request):
        name    = request.data["metadata"]["name"]
        notes   = request.data["metadata"]["notes"]

        fitter    = request.data["options"]["fitter"]
        data_id   = request.data["options"]["data_id"]
        params_in = [ p["value"] for p in request.data["options"]["params"] ]

        params_out = [ p["value"] for p in request.data["result"]["params"] ]
        y          = request.data["result"]["fit"]["y"]

        residuals        = request.data["result"]["residuals"]
        species_molefrac = request.data["result"]["species_molefrac"]
        species_coeff    = request.data["result"]["species_coeff"]

        data = models.Data.objects.get(id=data_id)

        fit = models.Fit(name=name, 
                         notes=notes,
                         data=data,
                         fitter=fitter,
                         params_guess=params_in,
                         params=params_out,
                         y=y,
                         residuals=residuals,
                         species_molefrac=species_molefrac,
                         species_coeff=species_coeff,
                         )
        fit.save()

        response = formatter.save(fit.id)
        return Response(response)



class FitRetrieveView(APIView):
    parser_classes = (JSONParser,)

    def get(self, request, id):
        fit = models.Fit.objects.get(id=id)
        response = fit.to_dict()
        return Response(response)
 


class FitExportView(APIView):
    parser_classes = (JSONParser,)
    
    def post(self, request):
        dt = 'f8'

        # Munge some data
        # Transpose 1D arrays -> 2D column arrays for hstack later
        # Input data
        data_h0  = np.array(request.data["data"]["x"][0],  dtype=dt)[np.newaxis].T
        data_g0  = np.array(request.data["data"]["x"][1],  dtype=dt)[np.newaxis].T
        data_geq = data_g0/data_h0
        # PLACEHOLDER deal with multi-D y inputs here later
        data_y   = np.array(request.data["data"]["y"][0],  dtype=dt).T

        # Input options
        options_fitter = request.data["options"]["fitter"]
        options_params = np.array([ p["value"] for p in request.data["options"]["params"] ], dtype=dt)

        # Fit results 
        # PLACEHOLDER deal with multi-D y inputs here later
        fit_y         = np.array(request.data["fit"]["y"][0],      dtype=dt).T
        fit_params    = np.array([ p["value"] for p in request.data["fit"]["params"] ], dtype=dt)
        fit_molefrac  = np.array(request.data["fit"]["molefrac"],  dtype=dt).T
        fit_coeffs    = np.array(request.data["fit"]["coeffs"],    dtype=dt)
        fit_residuals = np.array(request.data["fit"]["residuals"], dtype=dt)

        # Labels
        labels = formatter.labels(options_fitter)

        # Create output arrays
        data_array    = np.hstack((data_h0, data_g0, data_geq, data_y))
        options_array = np.concatenate(([options_fitter], options_params))
        fit_array_1   = np.hstack((data_h0, data_g0, data_geq, fit_y, fit_molefrac))
        fit_array_2   = fit_coeffs
        fit_array_3   = fit_residuals[np.newaxis] # To force horizontal array
                                                  # in DataFrame
        params_array  = fit_params 

        # Generate appropriate column titles
        data_names      = ["[H]0", "[G]0", "[G]0/[H]0 equivalent total"]
        data_names.extend([  "Data "+str(i+1) for i in range(fit_y.shape[1])  ])

        options_names      = ["Fitter"]
        options_names.extend([ p["label"] for p in labels["params"] ])

        fit_names_1      = ["[H]0", "[G]0", "[G]0/[H]0 equivalent total"]
        fit_names_1.extend([ "Fit "+str(i+1) for i in range(data_y.shape[1]) ])
        fit_names_1.extend([ "Fit molefrac "+str(i+1) for i in range(fit_molefrac.shape[1]) ])

        fit_names_2 = [ "Fit coeffs "+str(i+1) for i in range(fit_coeffs.shape[1]) ]
        fit_names_3 = [ "Fit residuals sum "+str(i+1) for i in range(fit_residuals.shape[0]) ]

        params_names = [ p["label"] for p in labels["params"] ]

        # Create data frames for export
        data_output    = pd.DataFrame(data_array,    columns=data_names)
        options_output = pd.DataFrame(options_array, index=options_names) 
        fit_output_1   = pd.DataFrame(fit_array_1,   columns=fit_names_1)
        fit_output_2   = pd.DataFrame(fit_array_2,   columns=fit_names_2)
        fit_output_3   = pd.DataFrame(fit_array_3,   columns=fit_names_3)
        fit_output     = pd.concat([fit_output_1, 
                                    fit_output_2,
                                    fit_output_3],
                                    axis=1, 
                                    join_axes=[fit_output_1.index])
        params_output  = pd.DataFrame(params_array,  index=params_names)

        # Create export file
        # Randomly generate export filename
        filename = id_generator()+".xlsx"
        export_path = os.path.join(settings.MEDIA_ROOT, "output", filename) 

        # Write all dataframes to excel file
        writer = pd.ExcelWriter(export_path)
        data_output.to_excel(writer, "Input Data", index=False)
        options_output.to_excel(writer, "Input Options", header=False)
        params_output.to_excel(writer, "Output Parameters", header=False)
        fit_output.to_excel(writer, "Output Fit", index=False)
        writer.save()

        export_url = settings.ROOT_URL+settings.MEDIA_URL+"output/"+filename

        return Response(formatter.export(export_url))



class UploadDataView(APIView):
    """
    Request:

    Response:
        string: Name of uploaded file on server
    """

    REQUEST_KEY = "input"

    parser_classes = (MultiPartParser, )

    def put(self, request):
        f = request.FILES[self.REQUEST_KEY]

        # Get file extension
        ext = os.path.splitext(str(f))[1][1:]

        if ext == "csv":
            d = models.Data.from_csv(f)
        elif ext == "xls" or ext == "xlsx":
            d = models.Data.from_xls(f)
        else:
            # Try reading from csv as default if no extension provided
            d = models.Data.from_csv(f)

        d.save()
        
        response = formatter.upload(d.id)
        
        return Response(response, status=200)



#
# Helper functions
#

def id_generator(size=5, chars=string.ascii_lowercase + string.ascii_uppercase + string.digits):
    # Generates random ID from a given list of characters
    # Used for random filenames on exporting
    return "".join(random.choice(chars) for _ in range(size))
