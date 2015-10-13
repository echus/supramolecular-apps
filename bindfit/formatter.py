"""
" Module defining API's standardised JSON responses to major endpoint requests
" Also defines standard computed properties for API responses 
" (eg: RMS, covariances)
"
" (Note: Computed properties placed here (not in DB model) for consistency and 
" minimum code duplication between saved fits and freshly fitted data in 
" views.py)
"""

import numpy as np

from . import helpers

import logging
logger = logging.getLogger('supramolecular')

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

def options(fitter, data_id=None, params=None):
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
    
    if data_id is not None and params is not None:
        return {
            "fitter": fitter,
            "data_id": data_id,
            "params": [ {"value": p} for p in params ]
            }
    else:
        return default_options_select[fitter]

def meta(author, 
         name, 
         date, 
         timestamp, 
         ref, 
         host, 
         guest, 
         solvent, 
         temp, 
         temp_unit,
         notes):
    response = {
            "author"   : author,
            "name"     : name,
            "date"     : str(date),
            "timestamp": str(timestamp),
            "ref"      : ref,
            "host"     : host,
            "guest"    : guest,
            "solvent"  : solvent,
            "temp"     : temp,
            "temp_unit": temp_unit,
            "notes"    : notes,
            }
    return response

def fit(y, params, residuals, molefrac, coeffs):
    """
    Return dictionary containing fit result information 
    (defines format used as JSON response in views)

    Arguments:
        y:         3D array   Array of 2D matrices of fit results 
        params:    1D array   Fitted parameters
        residuals: 3D array   Residuals for each fit
        molefrac:  2D array   Fitted species molefractions
        coeffs:    2D array   Fitted species coefficients

    Returns:
        {json}
        y:         3D array   Array of 2D matrices of fit results 
        params:    1D array   Fitted parameters
        residuals: 3D array   Residuals for each fit
        rms:       2D array   [computed] RMS errors for each fit set
        cov:       2D array   [computed] Covariances for each fit set
        molefrac:  2D array   Fitted species molefractions
        coeffs:    2D array   Fitted species coefficients
    """

    response = {
            "y"        : y,
            "params"   : [ {"value": p} for p in params ],
            "residuals": residuals,
            "rms"      : helpers.rms(residuals),
            "cov"      : helpers.cov(y, residuals),
            "molefrac" : molefrac,
            "coeffs"   : coeffs,
            }
    return response

def data(x, y):
    response = {
            "x" : x,
            "y" : y,
            }
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
