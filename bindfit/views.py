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


    def post(self, request):
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

        logger.debug("FitterView.post: called")

        # Import data
        data = self.import_data(request.data["input"]["type"], 
                                request.data["input"]["value"])

        k_guess = np.array(request.data["k_guess"], dtype=np.float64)

        # Call appropriate fitter

        # JSON fitter reference -> View fitter function map 
        # TODO move this definition elsewhere?
        fitter_select = {
                "nmr1to1": self.fit_nmr_1to1,
                "uv1to2":  self.fit_uv_1to2,
                }

        fitter = request.data["fitter"]
        fit = fitter_select[fitter](k_guess, data)
        
        # Build response dict
        response = self.build_response(data, fit)

        return Response(response)

    def import_data(self, fmt, value):
        # Import input file into Data object
        if fmt == "csv":
            input_path = os.path.join(settings.MEDIA_ROOT, value)
            data = Data(input_path)
        else:
            pass
            # Error page

        return data

    def build_response(self, data, fitter):
        # Build response dict

        # Loop through each column of observed data and its respective predicted
        # best fit, create array of [x, y] point pairs for plotting
        observed = []
        predicted = []
        for o, p in zip(data.observations.T, fitter.predict(data).T):
            geq = data.params["geq"]
            obs_plot  = [ [x, y] for x, y in zip(geq, o) ]
            pred_plot = [ [x, y] for x, y in zip(geq, p) ]
            observed.append(obs_plot)
            predicted.append(pred_plot)

        k = fitter.result

        response = {
                   "k": k,
                   "data": observed,
                   "fit": predicted,
                   "residuals": [],
                   }

        return response

    @staticmethod
    def fit_nmr_1to1(k_guess, data):
        # Initialise appropriate Fitter
        fitter = Fitter(functions.NMR1to1)

        # Run fitter on data
        fitter.fit(data, k_guess)

        logger.debug("FitterView.post: NMR1to1 fit")
        logger.debug("FitterView.post: fitter.result = "+str(fitter.result))
        logger.debug("FitterView.post: data.observations = "+str(data.observations))
        logger.debug("FitterView.post: fitter.predict(data) = "+str(fitter.predict(data)))

        return fitter 

    @staticmethod
    def fit_uv_1to2(k_guess, data):
        # Initialise appropriate Fitter
        fitter = Fitter(functions.UV1to2, algorithm="Nelder-Mead")

        # TESTING
        hg_mat = functions.UV1to2.f(k_guess, data)
        logger.debug("UV 1to2 HGMAT TEST")
        logger.debug(str(hg_mat))
        # END TESTING

        # Run fitter on data
        fitter.fit(data, k_guess)

        logger.debug("FitterView.post: UV1to2 fit")
        logger.debug("FitterView.post: fitter.result = "+str(fitter.result))
        logger.debug("FitterView.post: data.observations = "+str(data.observations))
        logger.debug("FitterView.post: fitter.predict(data) = "+str(fitter.predict(data)))

        return fitter



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
