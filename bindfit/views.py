from rest_framework.views import APIView

from rest_framework.parsers import JSONParser, MultiPartParser 

from rest_framework.response import Response
from rest_framework import status

from django.contrib.sites.models import Site

from django.conf import settings

import os
import hashlib
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

        # Parse request
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

        response = formatter.fit(fitter=self.fitter,
                                 data=self.data,
                                 fit=fit[0],
                                 params=params,
                                 residuals=fit[1],
                                 species_coeff=fit[2],
                                 species_molefrac=fit[3])

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
 


class FitExportViewOld(APIView):
    parser_classes = (JSONParser,)
    
    def post(self, request):
        dt = 'f8'

        # Get data
        # Transpose geq 1D array -> 2D column array
        h0     = np.array(request.data["data"]["data"]["h0"], dtype=dt)[np.newaxis].T
        g0     = np.array(request.data["data"]["data"]["g0"], dtype=dt)[np.newaxis].T
        geq    = np.array(request.data["data"]["data"]["geq"], dtype=dt)[np.newaxis].T
        data   = np.array(request.data["data"]["data"]["y"],   dtype=dt).T
        fit    = np.array(request.data["data"]["fit"]["y"],    dtype=dt).T
        params = np.array([ p["value"] for p in request.data["data"]["params"] ], 
                          dtype=dt)

        # Generate appropriate header and footer info for csv
        names = ["[G]0", "[H]0", "[G]0/[H]0 equivalent total"]
        names.extend([ "Data "+str(i) for i in range(data.shape[1]) ])
        names.extend([  "Fit "+str(i) for i in range(fit.shape[1])  ])
        header = ",".join(names)
        footer = ",".join([ str(p) for p in params ])
        
        # Create output array
        output = np.hstack((h0, g0, geq, data, fit))

        # Create export file
        # Use SHA1 hash of array as filename to avoid duplication
        filename = hashlib.sha1(np.ascontiguousarray(output.data)).hexdigest()+".csv"

        export_path = os.path.join(settings.MEDIA_ROOT, "output", filename) 
        np.savetxt(export_path, output, header=header, footer=footer, fmt="%.18f", delimiter=",")

        export_url = settings.ROOT_URL+settings.MEDIA_URL+"output/"+filename

        return Response(formatter.export(export_url))





class FitExportView(APIView):
    parser_classes = (JSONParser,)
    
    def post(self, request):
        dt = 'f8'

        # Munge some data
        # Transpose 1D arrays -> 2D column arrays for hstack later
        data_h0  = np.array(request.data["result"]["data"]["h0"],  dtype=dt)[np.newaxis].T
        data_g0  = np.array(request.data["result"]["data"]["g0"],  dtype=dt)[np.newaxis].T
        data_geq = np.array(request.data["result"]["data"]["geq"], dtype=dt)[np.newaxis].T
        data_y   = np.array(request.data["result"]["data"]["y"],   dtype=dt).T

        options_params = np.array([ p["value"] for p in request.data["options"]["params"] ], dtype=dt)

        fit_y    = np.array(request.data["result"]["fit"]["y"],    dtype=dt).T
        fit_params = np.array([ p["value"] for p in request.data["result"]["params"] ], dtype=dt)

        # Create output array
        data_array = np.hstack((data_h0, data_g0, data_geq, data_y))
        fit_array  = np.hstack((data_h0, data_g0, data_geq, fit_y))

        # Generate appropriate column titles
        fit_names  = ["[G]0", "[H]0", "[G]0/[H]0 equivalent total"]
        fit_names.extend([ "Fit "+str(i) for i in range(data_y.shape[1]) ])
        data_names = ["[G]0", "[H]0", "[G]0/[H]0 equivalent total"]
        data_names.extend([  "Data "+str(i) for i in range(fit_y.shape[1])  ])

        # Create data frames for export
        data_output = pd.DataFrame(data_array, columns=data_names)
        fit_output  = pd.DataFrame(fit_array,  columns=data_names)

        # Create export file
        # Use SHA1 hash of fit array as filename to avoid duplication
        filename = hashlib.sha1(np.ascontiguousarray(fit_array.data)).hexdigest()+".xlsx"
        export_path = os.path.join(settings.MEDIA_ROOT, "output", filename) 

        # Write all dataframes to excel file
        writer = pd.ExcelWriter(export_path)
        data_output.to_excel(writer, "Input data")
        fit_output.to_excel(writer, "Fit")

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
