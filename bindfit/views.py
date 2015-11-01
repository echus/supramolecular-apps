from rest_framework.views import APIView

from rest_framework.parsers import JSONParser, MultiPartParser 

from rest_framework.response import Response
from rest_framework import status

from django.contrib.sites.models import Site

from django.conf import settings

import os
import string
import random
import datetime
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

        # Get input data entry
        data = models.Data.objects.get(id=request.data["data_id"])
        dilute = request.data["dilute"] # Dilution factor flag
        self.data = data.to_dict(dilute)

        logger.debug("views.FitView: data.to_dict() after retrieving")
        logger.debug(self.data)

        # Create and run appropriate fitter
        self.fit = self.run_fitter()
        
        # Build response dict
        response = self.build_response()
        return Response(response)

    def build_response(self):
        response = {
                "data": self.data,
                "fit" : formatter.fit(y=self.fit.fit,
                                      params=self.fit.params,
                                      residuals=self.fit.residuals,
                                      coeffs=self.fit.coeffs,
                                      molefrac=self.fit.molefrac,
                                      time=self.fit.time)
                }

        return response

    def run_fitter(self):
        # Initialise appropriate Fitter with specified minimisation function
        function = functions.select[self.fitter]
        fitter = Fitter(self.data, function)

        # Run fitter on data
        fitter.run(self.params)

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
        meta = request.data["meta"] # For readability
        meta_author    = meta.get("author", "")
        meta_name      = meta.get("name", "")
        meta_date      = meta.get("date", None)
        meta_ref       = meta.get("ref", "")
        meta_host      = meta.get("host", "")
        meta_guest     = meta.get("guest", "")
        meta_solvent   = meta.get("solvent", "")
        meta_temp      = meta.get("temp", None)
        meta_temp_unit = meta.get("temp_unit", None)
        meta_notes     = meta.get("notes", "")

        # Hack to deal with receiving "None" string
        if meta_temp == "None":
            meta_temp = None

        if meta_date == "None":
            meta_date = None

        options_fitter  = request.data["options"]["fitter"]
        options_data_id = request.data["options"]["data_id"]
        options_params  = [ p["value"] for p in request.data["options"]["params"] ]
        options_dilute  = request.data["options"]["dilute"]

        fit_params = [ p["value"] for p in request.data["fit"]["params"] ]
        fit_y      = request.data["fit"]["y"]

        fit_residuals = request.data["fit"]["residuals"]
        fit_molefrac  = request.data["fit"]["molefrac"]
        fit_coeffs    = request.data["fit"]["coeffs"]
        fit_time      = request.data["fit"]["time"]

        data = models.Data.objects.get(id=options_data_id)

        fit = models.Fit(meta_author=meta_author, 
                         meta_name=meta_name, 
                         meta_date=meta_date, 
                         meta_ref=meta_ref, 
                         meta_host=meta_host, 
                         meta_guest=meta_guest, 
                         meta_solvent=meta_solvent, 
                         meta_temp=meta_temp, 
                         meta_notes=meta_notes,
                         data=data,
                         options_fitter=options_fitter,
                         options_params=options_params,
                         options_dilute=options_dilute,
                         fit_params=fit_params,
                         fit_y=fit_y,
                         fit_residuals=fit_residuals,
                         fit_molefrac=fit_molefrac,
                         fit_coeffs=fit_coeffs,
                         fit_time=fit_time,
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
        data_x_labels = request.data["data"]["labels"]["x"]
        data_y_labels = request.data["data"]["labels"]["y"]
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
        # PLACEHOLDER deal with multi-D y inputs here later
        fit_residuals = np.array(request.data["fit"]["residuals"][0], dtype=dt).T
        fit_rms       = np.array(request.data["fit"]["rms"][0], dtype=dt).T
        fit_cov       = np.array(request.data["fit"]["cov"][0], dtype=dt).T
        fit_rms_total = request.data["fit"]["rms_total"][0]
        fit_cov_total = request.data["fit"]["cov_total"][0]

        # Labels
        labels = formatter.labels(options_fitter)

        # Create output arrays
        data_array     = np.hstack((data_h0, data_g0, data_geq, data_y))
        options_array  = np.concatenate(([options_fitter], options_params))
        fit_array      = np.hstack((data_h0, data_g0, data_geq, fit_y, fit_residuals, fit_molefrac))
        qof_array_1    = np.append(fit_rms, fit_rms_total)
        qof_array_2    = np.append(fit_cov, fit_cov_total)

        params_array_1 = fit_params[np.newaxis] # To force horizontal array in
                                                # DataFrame
        params_array_2 = fit_coeffs

        # Generate appropriate column titles
        data_names      = [ "x"+str(i+1)+": "+l for i, l in enumerate(data_x_labels) ]
        data_names.extend(["x3: G/H equivalent total"])
        data_names.extend([ "y"+str(i+1)+": "+l for i, l in enumerate(data_y_labels) ])
        logger.debug("DATA_NAMES")
        logger.debug(data_names)

        options_names      = ["Fitter"]
        options_names.extend([ p["label"] for p in labels["params"] ])

        fit_names      = [ "x"+str(i+1)+": "+l for i, l in enumerate(data_x_labels) ]
        fit_names.extend(["x3: G/H equivalent total"])
        fit_names.extend([ "y"+str(i+1)+": "+l for i, l in enumerate(data_y_labels) ])
        fit_names.extend([ "y"+str(i+1)+": Residuals" for i in range(fit_residuals.shape[1]) ])
        fit_names.extend([ "y"+str(i+1)+": Molefractions" for i in range(fit_molefrac.shape[1]) ])
        logger.debug("FIT_NAMES")
        logger.debug(fit_names)

        qof_names_1 = [ "RMS: "+l for l in data_y_labels ]
        qof_names_1.append("RMS: Total")
        qof_names_2 = [ "Covariance: "+l for l in data_y_labels ]
        qof_names_2.append("Covariance: Total")

        params_names_1 = [ p["label"] for p in labels["params"] ]
        params_names_2 = [ "Fit coeffs "+str(i+1) for i in range(fit_coeffs.shape[1]) ]

        # Create data frames for export
        data_output     = pd.DataFrame(data_array,     columns=data_names)
        options_output  = pd.DataFrame(options_array,  index=options_names) 
        fit_output      = pd.DataFrame(fit_array,      columns=fit_names)
        qof_output_1    = pd.DataFrame(qof_array_1,    index=qof_names_1)
        qof_output_2    = pd.DataFrame(qof_array_2,    index=qof_names_2)
        qof_output      = pd.concat([qof_output_1,
                                     qof_output_2],
                                     axis=0,
                                     join_axes=[qof_output_1.columns])
        params_output_1 = pd.DataFrame(params_array_1, columns=params_names_1)
        params_output_2 = pd.DataFrame(params_array_2, columns=params_names_2)
        params_output   = pd.concat([params_output_1,
                                     params_output_2],
                                     axis=1,
                                     join_axes=[params_output_1.index])

        # Create export file
        # Randomly generate export filename
        filename = id_generator()+".xlsx"
        export_path = os.path.join(settings.MEDIA_ROOT, "output", filename) 

        # Write all dataframes to excel file
        writer = pd.ExcelWriter(export_path)
        data_output.to_excel(writer, "Input Data", index=False)
        options_output.to_excel(writer, "Input Options", header=False)
        params_output.to_excel(writer, "Output Parameters", index=False)
        fit_output.to_excel(writer, "Output Fit", index=False)
        qof_output.to_excel(writer, "Output Fit Quality", header=False)
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
