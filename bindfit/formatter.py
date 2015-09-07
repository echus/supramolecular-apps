"""
Module defining API's standardised JSON responses to major endpoint requests.
"""

import numpy as np

def fitter_list():
    """ 
    Returns list of all available fitters 
    """

    fitter_list = [
            {"name": "NMR 1:1", "key": "nmr1to1"},
            {"name": "NMR 1:2", "key": "nmr1to2"},
            {"name": "UV 1:1",  "key": "uv1to1"},
            {"name": "UV 1:2",  "key": "uv1to2"},
            ]

    return fitter_list



def labels(fitter): 
    # Label definitions for each fitter type
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
                    "label": "Absorbance",
                    "units": "",
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
                    "label": "Absorbance",
                    "units": "",
                    },
                "params": [
                    {"label": "K\u2081\u2081", "units": "M\u207B\u00B9"},
                    {"label": "K\u2081\u2082", "units": "M\u207B\u00B9"},
                    ]
                },
            }

    return label_select[fitter]



def options(fitter):
    # Default options for each fitter type
    default_options_select = {
            "nmr1to1": {
                "fitter": "nmr1to1",
                "data_id": "",
                "params": [
                    {"value": 1000},
                    ],
                },
            "nmr1to2": {
                "fitter": "nmr1to2",
                "data_id": "",
                "params": [
                    {"value": 10000},
                    {"value": 1000},
                    ],
                },
            "uv1to1": {
                "fitter": "uv1to1",
                "data_id": "",
                "params": [
                    {"value": 1000},
                    ],
                },
            "uv1to2": {
                "fitter": "uv1to2",
                "data_id": "",
                "params": [
                    {"value": 10000},
                    {"value": 1000},
                    ],
                },
            }
    
    return default_options_select[fitter]



def fit (fitter, data, fit, params, residuals=None):
    """
    Return dictionary containing fit result information 
    (defines format used as JSON response in views)

    Arguments:
        fitter: string  Fitter type (eg: nmr1to1, uv1to2)
        data  : dict    Input data as per Data.to_dict() model
        fit   : array   Array of y arrays of values of fit results (vs. geq/h0/g0)
        params: array   Fitted parameters
    """
    response = {
            "data" : {
                "h0" : data["h0"],
                "g0" : data["g0"],
                "geq": data["geq"],
                "y"  : data["ynorm"],
                },

            "fit"  : {
                "y"  : fit,
                },

            "params"   : [ {"value": p} for p in params ],

            "residuals": residuals,
            }

    # TODO move this into a fitter-specific function hash
    # Fitter-type-specific post-processing
    if fitter == "uv1to2" or fitter == "uv1to1":
        response["data"]["y"] = np.abs(response["data"]["y"])
        response["fit"]["y"] = np.abs(response["fit"]["y"])

    return response



def save(fit_id):
    response = {
            "id": fit_id
            }
    return response

def export(url):
    response = {
            "url": url
            }
    return response

def upload(data_id):
    response = {
            "id": data_id
            }
    return response
