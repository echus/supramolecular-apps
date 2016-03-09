import os
import string
import random
import datetime
import pandas as pd
import numpy  as np

from django.core.mail import send_mail

from rest_framework.views import APIView

from rest_framework.parsers import JSONParser, MultiPartParser 

from rest_framework.response import Response
from rest_framework import status

from haystack.query  import SearchQuerySet
from haystack.inputs import AutoQuery 

from django.contrib.sites.models import Site
from django.conf import settings

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
            data_id: string  Reference to input data to use

            params: {
                    k1: float   User guess of first parameter
                    k2: float   User guess of second parameter
                    ..: ...     ...
                    }

        Response:
            data_id:
            data:
                x:
                y:
            labels:
                data:
                    x:
                    y:

            fit:
                y:
                coeffs:
                molefrac:
                params: {
                        k1: dict    First parameter optimised results
                        k2: dict    Second parameter optimised results
                        ..: ...     ...
                        }
            qof:
                residuals:
                cov:
                cov_total:
                rms:
                rms_total:
            time:
            options:
                dilute:
        """

        logger.debug("FitterView.post: called")

        # Parse request options
        self.fitter_name = request.data["fitter"]

        # Get input data to fit from database
        dilute = request.data["options"]["dilute"] # Dilution factor flag
        data = models.Data.objects.get(id=request.data["data_id"]).to_dict(
                fitter=self.fitter_name,
                dilute=dilute)
        logger.debug("views.FitView: data.to_dict() after retrieving")
        logger.debug(data)
        datax = data["data"]["x"]
        datay = data["data"]["y"]

        params = { key: float(value) for key, value in request.data["params"].items() }

        # "Normalise" y data, i.e. subtract initial values from y data 
        # (silly name choice, sorry)
        normalise = request.data["options"].get("normalise", True)

        # Create and run appropriate fitter
        fitter = self.run_fitter(datax, datay, params, normalise)
        
        # Build response dict
        response = self.build_response(fitter, data, dilute)
        return Response(response)

    def build_response(self, fitter, data, dilute):
        # Combined fitter and data dictionaries
        response = formatter.fit(fitter    =self.fitter_name,
                                 data      =data,
                                 y         =fitter.fit,
                                 params    =fitter.params,
                                 residuals =fitter.residuals,
                                 coeffs    =fitter.coeffs,
                                 molefrac  =fitter.molefrac,
                                 time      =fitter.time,
                                 dilute    =dilute)
        return response

    def run_fitter(self, datax, datay, params, normalise):
        # Initialise Fitter with approriate objective function
        function = functions.select[self.fitter_name]
        fitter = Fitter(datax, datay, function, normalise=normalise)

        # Run fitter
        fitter.run_scipy(params)

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
        fit  = request.data
        meta = request.data["meta"]

        meta_options_searchable = meta["options"]["searchable"]

        meta_email     = meta.get("email",     "")
        meta_author    = meta.get("author",    "")
        meta_name      = meta.get("name",      "")
        meta_date      = meta.get("date",      None)
        meta_ref       = meta.get("ref",       "")
        meta_host      = meta.get("host",      "")
        meta_guest     = meta.get("guest",     "")
        meta_solvent   = meta.get("solvent",   "")
        meta_temp      = meta.get("temp",      None)
        meta_temp_unit = meta.get("temp_unit", None)
        meta_notes     = meta.get("notes",     "")

        # Hack to deal with receiving "None" string
        if meta_temp == "None" or meta_temp == "":
            meta_temp = None

        if meta_date == "None" or meta_date == "":
            meta_date = None

        options_fitter  = fit["fitter"]
        options_data_id = fit["data_id"]

        data = models.Data.objects.get(id=options_data_id)

        no_fit = fit["no_fit"]

        if not no_fit:
            options_dilute  = fit["options"]["dilute"]

            fit_params        = fit["fit"]["params"]
            fit_params_keys   = [ key for key in sorted(fit_params) ]
            fit_params_init   = [ fit_params[key]["init"]  
                                  for key in sorted(fit_params) ]
            fit_params_value  = [ fit_params[key]["value"] 
                                  for key in sorted(fit_params) ]
            fit_params_stderr = [ fit_params[key]["stderr"]
                                  for key in sorted(fit_params) ]

            fit_y      = fit["fit"]["y"]

            fit_molefrac  = fit["fit"]["molefrac"]
            fit_coeffs    = fit["fit"]["coeffs"]
            fit_time      = fit["time"]
            fit_residuals = fit["qof"]["residuals"]

            fit = models.Fit(no_fit=no_fit,
                             meta_options_searchable=meta_options_searchable, 
                             meta_email=meta_email, 
                             meta_author=meta_author, 
                             meta_name=meta_name, 
                             meta_date=meta_date, 
                             meta_ref=meta_ref, 
                             meta_host=meta_host, 
                             meta_guest=meta_guest, 
                             meta_solvent=meta_solvent, 
                             meta_temp=meta_temp, 
                             meta_notes=meta_notes,
                             data=data,
                             fitter_name=options_fitter,
                             options_dilute=options_dilute,
                             fit_params_keys=fit_params_keys,
                             fit_params_init=fit_params_init,
                             fit_params_value=fit_params_value,
                             fit_params_stderr=fit_params_stderr,
                             fit_y=fit_y,
                             fit_molefrac=fit_molefrac,
                             fit_coeffs=fit_coeffs,
                             qof_residuals=fit_residuals,
                             time=fit_time,
                             )
            fit.save()
        else:
            fit = models.Fit(no_fit=no_fit,
                             meta_options_searchable=meta_options_searchable, 
                             meta_email=meta_email, 
                             meta_author=meta_author, 
                             meta_name=meta_name, 
                             meta_date=meta_date, 
                             meta_ref=meta_ref, 
                             meta_host=meta_host, 
                             meta_guest=meta_guest, 
                             meta_solvent=meta_solvent, 
                             meta_temp=meta_temp, 
                             meta_notes=meta_notes,
                             data=data,
                             fitter_name=options_fitter,
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



class FitSearchEmailView(APIView):
    parser_classes = (JSONParser,)

    def post(self, request):
        view_url = request.data["view_url"] # Frontend view URL to use with fit IDs
        email    = request.data["email"]    # Email to retrieve fits for

        matches = models.Fit.objects.filter(meta_email=email)

        if matches:
            links = []
            for fit in matches:
                links.append(view_url+str(fit.id))

            body = "\n".join(links)

            send_mail("Your fit URLs", 
                      body, 
                      "BindFit Database <noreply@opendatafit.org>", 
                      [email], 
                      fail_silently=False)

            return Response({"detail": "Success! Check your email."}, 
                             status=status.HTTP_200_OK)
        else:
            # TODO return status here?
            return Response({"detail": "No matching fits found."}, 
                             status=status.HTTP_500_INTERNAL_SERVER_ERROR)



class FitSearchView(APIView):
    parser_classes = (JSONParser,)

    def post(self, request):
        r = request.data

        if type(r['query']) is str:
            # Simple search - searches for matches in all indexed fields
            query = r['query']

            # TODO:
            # Shouldn't have to filter for searchable=True here as they
            # shouldn't be indexed in the first place, but for some reason 
            # RealtimeSignalProcessor doesn't listen to the index_queryset
            # filtering in search_indexes.py.
            # Filtering added here as a temp fix.
            matches = SearchQuerySet().filter(content=AutoQuery(query)).filter(searchable=True)

        elif type(r['query']) is dict:
            # Advanced search (not implemented)
            # TODO: port to haystack, expand
            # Current code gets only exact matches directly from the database
            query = r['query']

            # Parse request names -> DB field names
            search = {}
            for (key, value) in query.items():
                if value:
                    search["meta_"+str(key)] = value

            # Get matching entries from DB
            matches = models.Fit.objects.filter(**search)

        else:
            # Bad query
            return Response({"detail": "Query must be a string or object."}, 
                             status=status.HTTP_400_BAD_REQUEST)

        summary_list = []

        for match in matches.all():
            summary = match.object.summary
            summary_list.append(summary)

        response = {"matches": summary_list}

        return Response(response)
 


class FitExportView(APIView):
    parser_classes = (JSONParser,)
    
    def post(self, request):
        dt = 'f8'

        fit  = request.data
        meta = request.data["meta"]

        labels = formatter.labels(fit["fitter"])
        user_labels = fit["labels"]

        # Munge some data
        # Transpose 1D arrays -> 2D column arrays for hstack later
        # Input data
        data_x_labels = user_labels["data"]["x"]["row_labels"]
        data_y_labels = user_labels["data"]["y"]["row_labels"]
        data_h0  = np.array(fit["data"]["x"][0],  dtype=dt)[np.newaxis].T
        data_g0  = np.array(fit["data"]["x"][1],  dtype=dt)[np.newaxis].T
        data_geq = data_g0/data_h0
        # PLACEHOLDER deal with multi-D y inputs here later
        data_y   = np.array(fit["data"]["y"],  dtype=dt).T

        # Input options
        options_fitter = fit["fitter"]
        options_params = np.array(
                [ fit["fit"]["params"][key]["init"] 
                  for key in sorted(fit["fit"]["params"]) ], 
                dtype=dt)

        # Fit results 
        # PLACEHOLDER deal with multi-D y inputs here later
        fit_y         = np.array(fit["fit"]["y"],      dtype=dt).T
        fit_params    = np.array(
                [ fit["fit"]["params"][key]["value"] 
                  for key in sorted(fit["fit"]["params"]) ], 
                dtype=dt)
        fit_molefrac  = np.array(fit["fit"]["molefrac"],  dtype=dt).T
        fit_coeffs    = np.array(fit["fit"]["coeffs"],    dtype=dt).T
        fit_coeffs_calc = np.array(fit["fit"]["coeffs_calc"],    dtype=dt).T
        # PLACEHOLDER deal with multi-D y inputs here later
        fit_residuals = np.array(fit["qof"]["residuals"], dtype=dt).T
        fit_rms       = np.array(fit["qof"]["rms"], dtype=dt).T
        fit_cov       = np.array(fit["qof"]["cov"], dtype=dt).T
        fit_rms_total = fit["qof"]["rms_total"]
        fit_cov_total = fit["qof"]["cov_total"]

        # Labels
        params_labels_dict = labels["fit"]["params"]
        params_labels      = [ params_labels_dict[key] for key in sorted(params_labels_dict) ]
        coeffs_calc_labels = labels["fit"]["coeffs_calc"]
        molefrac_labels    = labels["fit"]["molefrac"]

        # Create output arrays
        data_array     = np.hstack((data_h0, data_g0, data_geq, data_y))
        options_array  = np.concatenate(([options_fitter], options_params))
        fit_array      = np.hstack((data_h0, data_g0, data_geq, fit_y, fit_residuals, fit_molefrac))
        qof_array_1    = np.append(fit_rms, fit_rms_total)
        qof_array_2    = np.append(fit_cov, fit_cov_total)

        params_array_1 = fit_params[np.newaxis] # To force horizontal array in
                                                # DataFrame
        params_array_2 = fit_coeffs_calc
        params_array_3 = fit_coeffs

        # Generate appropriate column titles
        data_names      = [ "x"+str(i+1)+": "+l for i, l in enumerate(data_x_labels) ]
        data_names.extend(["x3: G/H equivalent total"])
        data_names.extend([ "y"+str(i+1)+": "+l for i, l in enumerate(data_y_labels) ])
        logger.debug("DATA_NAMES")
        logger.debug(data_names)

        options_names      = ["Fitter"]
        options_names.extend([ p["label"] for p in params_labels ])

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

        params_names_1 = [ p["label"] for p in params_labels ]
        params_names_2 = [ str(l)+" coeffs" for l in coeffs_calc_labels ]
        params_names_3 = [ "Raw coeffs"+str(i+1) for i in range(fit_coeffs.shape[1]) ]

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
        params_output_3 = pd.DataFrame(params_array_3, columns=params_names_3)
        params_output   = pd.concat([params_output_1,
                                     params_output_2,
                                     params_output_3],
                                     axis=1,
                                     join_axes=[params_output_2.index])

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
        # Read file
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
        
        # Get selected fitter key
        fitter = request.data["fitter"]

        # Return parsed data
        response = d.to_dict(fitter=fitter, dilute=False)
        return Response(response, status=200)



#
# Helper functions
#

def id_generator(size=5, chars=string.ascii_lowercase + string.ascii_uppercase + string.digits):
    # Generates random ID from a given list of characters
    # Used for random filenames on exporting
    return "".join(random.choice(chars) for _ in range(size))
