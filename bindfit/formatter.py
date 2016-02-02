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
            {"name": "NMR",            "key": "nmrdata", "group": "Save data only (under construction)"},
            {"name": "UV",             "key": "uvdata",  "group": "Save data only (under construction)"},
            {"name": "NMR 1:1",        "key": "nmr1to1", "group": "NMR"},
            {"name": "NMR 1:2",        "key": "nmr1to2", "group": "NMR"},
            {"name": "NMR 2:1",        "key": "nmr2to1", "group": "NMR"},
            {"name": "UV 1:1",         "key": "uv1to1",  "group": "UV"},
            {"name": "UV 1:2",         "key": "uv1to2",  "group": "UV"},
            {"name": "UV 2:1",         "key": "uv2to1",  "group": "UV"},
            ]

    return fitter_list

def labels(fitter): 
    # Label definitions for each fitter type

    label_select = {
            "nmrdata": {
                "data": {
                    "x": {
                        "axis_label": "Equivalent total [G]\u2080/[H]\u2080",
                        "axis_units": "",
                        },
                    "y": {
                        "axis_label": "\u03B4",
                        "axis_units": "ppm",
                        },
                    },
                },

            "uvdata": {
                "data": {
                    "x": {
                        "axis_label": "Equivalent total [G]\u2080/[H]\u2080",
                        "axis_units": "",
                        },
                    "y": {
                        "axis_label": "Absorbance",
                        "axis_units": "",
                        },
                    },
                },
            
            "nmr1to1": {
                "data": {
                    "x": {
                        "axis_label": "Equivalent total [G]\u2080/[H]\u2080",
                        "axis_units": "",
                        },
                    "y": {
                        "axis_label": "\u03B4",
                        "axis_units": "ppm",
                        },
                    },
                "fit": {
                    "params": {
                        "k": {"label": "K", "units": "M\u207B\u00B9"},
                        },
                    "y": {
                        "axis_label": "\u03B4",
                        "axis_units": "ppm",
                        },
                    "coeffs_calc": ["H", "HG"],
                    "molefrac":    ["H", "HG"],
                    },
                },
            
            "nmr1to2": {
                "data": {
                    "x": {
                        "axis_label": "Equivalent total [G]\u2080/[H]\u2080",
                        "axis_units": "",
                        },
                    "y": {
                        "axis_label": "\u03B4",
                        "axis_units": "ppm",
                        },
                    },
                "fit": {
                    "params": {
                        "k1": {"label": "K\u2081\u2081", "units": "M\u207B\u00B9"},
                        "k2": {"label": "K\u2081\u2082", "units": "M\u207B\u00B9"},
                        },
                    "y": {
                        "axis_label": "\u03B4",
                        "axis_units": "ppm",
                        },
                    "coeffs_calc": ["H", "HG", "HG2"],
                    "molefrac":    ["H", "HG", "HG2"],
                    },
                },

            "nmr2to1": {
                "data": {
                    "x": {
                        "axis_label": "Equivalent total [G]\u2080/[H]\u2080",
                        "axis_units": "",
                        },
                    "y": {
                        "axis_label": "\u03B4",
                        "axis_units": "ppm",
                        },
                    },
                "fit": {
                    "params": {
                        "k1": {"label": "K\u2081\u2081", "units": "M\u207B\u00B9"},
                        "k2": {"label": "K\u2082\u2081", "units": "M\u207B\u00B9"},
                        },
                    "y": {
                        "axis_label": "\u03B4",
                        "axis_units": "ppm",
                        },
                    "coeffs_calc": ["H", "HG", "H2G"],
                    "molefrac":    ["H", "HG", "H2G"],
                    },
                },

            "uv1to1": {
                "data": {
                    "x": {
                        "axis_label": "Equivalent total [G]\u2080/[H]\u2080",
                        "axis_units": "",
                        },
                    "y": {
                        "axis_label": "Absorbance",
                        "axis_units": "",
                        },
                    },
                "fit": {
                    "params": {
                        "k": {"label": "K", "units": "M\u207B\u00B9"},
                        },
                    "y": {
                        "axis_label": "Absorbance",
                        "axis_units": "",
                        },
                    "coeffs_calc": ["H", "HG"],
                    "molefrac":    ["H", "HG"],
                    },
                },

            "uv1to2": {
                "data": {
                    "x": {
                        "axis_label": "Equivalent total [G]\u2080/[H]\u2080",
                        "axis_units": "",
                        },
                    "y": {
                        "axis_label": "Absorbance",
                        "axis_units": "",
                        },
                    },
                "fit": {
                    "params": {
                        "k1": {"label": "K\u2081\u2081", "units": "M\u207B\u00B9"},
                        "k2": {"label": "K\u2081\u2082", "units": "M\u207B\u00B9"},
                        },
                    "y": {
                        "axis_label": "Absorbance",
                        "axis_units": "",
                        },
                    "coeffs_calc": ["H", "HG", "HG2"],
                    "molefrac":    ["H", "HG", "HG2"],
                    },
                },

            "uv2to1": {
                "data": {
                    "x": {
                        "axis_label": "Equivalent total [G]\u2080/[H]\u2080",
                        "axis_units": "",
                        },
                    "y": {
                        "axis_label": "Absorbance",
                        "axis_units": "",
                        },
                    },
                "fit": {
                    "params": {
                        "k1": {"label": "K\u2081\u2081", "units": "M\u207B\u00B9"},
                        "k2": {"label": "K\u2082\u2081", "units": "M\u207B\u00B9"},
                        },
                    "y": {
                        "axis_label": "Absorbance",
                        "axis_units": "",
                        },
                    "coeffs_calc": ["H", "HG", "H2G"],
                    "molefrac":    ["H", "HG", "H2G"],
                    },
                },
            }

    return label_select[fitter]

def options(fitter, data_id=None, params=None, dilute=False):
    # Default options for each fitter type
    default_options_select = {
            "nmrdata": {
                "fitter": "nmrdata",
                "data_id": "",
                "params": {},
                "options": {
                    "dilute": False,
                    },
                },

            "uvdata": {
                "fitter": "uvdata",
                "data_id": "",
                "params": {},
                "options": {
                    "dilute": False,
                    },
                },

            "nmr1to1": {
                "fitter": "nmr1to1",
                "data_id": "",
                "params": {
                    "k": 1000,
                    },
                "options": {
                    "dilute": False,
                    },
                },

            "nmr1to2": {
                "fitter": "nmr1to2",
                "data_id": "",
                "params": {
                    "k1": 10000,
                    "k2": 1000,
                    },
                "options": {
                    "dilute": False,
                    },
                },

            "nmr2to1": {
                "fitter": "nmr2to1",
                "data_id": "",
                "params": {
                    "k1": 10000,
                    "k2": 1000,
                    },
                "options": {
                    "dilute": False,
                    },
                },

            "uv1to1": {
                "fitter": "uv1to1",
                "data_id": "",
                "params": {
                    "k": 1000,
                    },
                "options": {
                    "dilute": True,
                    },
                },

            "uv1to2": {
                "fitter": "uv1to2",
                "data_id": "",
                "params": {
                    "k1": 10000,
                    "k2": 1000,
                    },
                "options": {
                    "dilute": True,
                    },
                },

            "uv2to1": {
                "fitter": "uv2to1",
                "data_id": "",
                "params": {
                    "k1": 10000,
                    "k2": 1000,
                    },
                "options": {
                    "dilute": True,
                    },
                },
            }
    
    if data_id is not None and params is not None:
        return {
            "fitter": fitter,
            "data_id": data_id,
            "params": { key: float(value) for (key, value) in params.items() },
            "options": {
                "dilute": dilute,
                },
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

def fit(fitter, data, y=None, params=None, residuals=None, molefrac=None, 
        coeffs=None, time=None, dilute=None, no_fit=False):
    """
    Return dictionary containing fit result information 
    (defines format used as JSON response in views)

    Arguments:
        fitter:    string   Name (key) of fitter used
        data:      dict     Dictionary containing formatted input data
        y:         ndarray  n x m array of fitted data
        params:    dict     Fitted parameters
        residuals: ndarray  Residuals for each fit
        molefrac:  ndarray  Fitted species molefractions
        coeffs:    ndarray  Fitted species coefficients
        time:      ndarray  Time taken to fit
        dilute:    bool     (option) Dilution factor flag

    Returns:
        fit:
            y:
            coeffs:
            calc_coeffs:
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
    """

    if not no_fit:
        fit = {
                "no_fit": no_fit,
                "fitter": fitter,
                "fit": {
                    "y":           y,
                    "coeffs":      coeffs,
                    "coeffs_calc": helpers.calculate_coeffs(fitter,
                                                            coeffs,
                                                            data["data"]["y"][:,0],
                                                            data["data"]["x"][0][0]),
                    "molefrac":    molefrac,
                    "params":      params,
                    },
                "qof": {
                    "residuals": residuals,
                    "rms"      : helpers.rms(residuals),
                    "cov"      : helpers.cov(data["data"]["y"], residuals),
                    "rms_total": helpers.rms(residuals, total=True),
                    "cov_total": helpers.cov(data["data"]["y"], residuals, total=True),
                    },
                "time": time,
                "options": {
                    "dilute": dilute,
                    },
                }
    else:
        fit = {
                "no_fit": no_fit,
                "fitter": fitter,
                }   

    # Merge with data dictionary
    response = deepcopy(data)
    response.update(fit)

    return response

def data(data_id, x, y, labels_x, labels_y):
    response = {
            "data_id": data_id,
            "data": {
                "x": x,
                "y": y,
                },
            "labels": {
                "data": {
                    "x": {
                        "row_labels": labels_x,
                        },
                    "y": {
                        "row_labels": labels_y,
                        },
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
