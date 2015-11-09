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
from copy import deepcopy

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

def options(fitter, data_id=None, params=None, dilute=False):
    # Default options for each fitter type
    default_options_select = {
            "nmr1to1": {
                "fitter": "nmr1to1",
                "data_id": "",
                "params": {
                    "k": 1000,
                    },
                "options": {
                    "dilute": False,
                    },
                "labels": {
                    "params": {
                        "k": {"label": "K", "units": "M\u207B\u00B9"},
                        },
                    },
                },
            "nmr1to2": {
                "fitter": "nmr1to2",
                "data_id": "",
                "params": [
                    {"value": 10000},
                    {"value": 1000},
                    ],
                "dilute": False,
                },
            "uv1to1": {
                "fitter": "uv1to1",
                "data_id": "",
                "params": [
                    {"value": 1000},
                    ],
                "dilute": True,
                },
            "uv1to2": {
                "fitter": "uv1to2",
                "data_id": "",
                "params": [
                    {"value": 10000},
                    {"value": 1000},
                    ],
                "dilute": True,
                },
            }
    
    if data_id is not None and params is not None:
        return {
            "fitter": fitter,
            "data_id": data_id,
            "params": [ {"value": p} for p in params ],
            "dilute": dilute
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

def fit(fitter, data, y, params, residuals, molefrac, coeffs, time):
    """
    Return dictionary containing fit result information 
    (defines format used as JSON response in views)

    Arguments:
        fitter:    string   Name (key) of fitter used
        data:      dict     Data/labels dict from data() to merge with fit info
        y:         ndarray  n x m array of fitted data
        params:    dict     Fitted parameters
        residuals: ndarray  Residuals for each fit
        molefrac:  ndarray  Fitted species molefractions
        coeffs:    ndarray  Fitted species coefficients
        time:      ndarray  Time taken to fit

    Returns:
        fit:
            y:
            coeffs:
            molefrac:
            params: 
                {
                    k1: float   Optimised first parameter value
                    k2: float   Optimised second parameter value
                    ..: ...     ...
                }
        qof:
            residuals:
            cov:
            cov_total:
            rms:
            rms_total:
        time:
        labels:
            fit: 
                y: {
                    row_labels:
                    axis_label:
                    axis_units:
                    }
                params: {
                    k1:
                    k2:
                    ...
                    }
    """

    fit = {
            "fit": {
                "y":        y,
                "coeffs":   coeffs,
                "molefrac": molefrac,
                "params":   params,
                },
            "qof": {
                "residuals": residuals,
                "rms"      : helpers.rms(residuals),
                "cov"      : helpers.cov(y, residuals),
                "rms_total": helpers.rms(residuals, total=True),
                "cov_total": helpers.cov(y, residuals, total=True),
                },
            "time": time,
            "labels": {
                "fit": {
                    "y": {
                        "row_labels": None,
                        "axis_label": None,
                        "axis_units": None,
                        },
                    # Get parameter labels from options definition
                    "params": options(fitter)["labels"]["params"],
                    },
                }
            }
    
    # Merge with data dictionary
    response = deepcopy(data)
    response.update(fit)

    # Manusally merge multi-level labels data ...
    labels = deepcopy(data["labels"])
    labels.update(fit["labels"])
    response["labels"] = labels

    return response

def data(data_id, x, y, x_labels, y_labels):
    response = {
            "data_id": data_id,
            "data": {
                "x": x,
                "y": y,
                },
            "labels": {
                "data": {
                    "x": {
                        "row_labels": x_labels,
                        "axis_label": None,
                        "axis_units": None,
                        },
                    "y": {
                        "row_labels": y_labels,
                        "axis_label": None,
                        "axis_units": None,
                        }
                    },
                },
            }
    return response

def save(fit_id):
    response = {
            "fit_id": fit_id
            }
    return response

def export(url):
    response = {
            "export_url": url
            }
    return response

def upload(data_id):
    response = {
            "data_id": data_id
            }
    return response
