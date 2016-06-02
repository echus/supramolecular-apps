from __future__ import division
from __future__ import print_function

from django.db import models
from django.contrib.postgres.fields import ArrayField, HStoreField

import numpy as np
import hashlib
import uuid

# For Excel read handling
from xlrd import open_workbook

from . import formatter 
from . import functions 
from . import helpers 

import logging
logger = logging.getLogger('supramolecular')

class Data(models.Model):
    # Primary key: SHA1 hash of imported numpy array
    id = models.CharField(max_length=40, primary_key=True)

    # 2D array of input x value fields (eg: [H]0 and [G]0 for NMR 1:1)
    x = ArrayField(
            ArrayField(models.FloatField())
            )

    # 3D array of variable length 2D input y value fields 
    # (eg: Proton 1, Proton 2, etc)
    y = ArrayField(
            ArrayField(
                ArrayField(models.FloatField())
                )
            )

    # Parsed header labels for each value
    labels_x = ArrayField(models.CharField(max_length=100, blank=True))
    labels_y = ArrayField(models.CharField(max_length=100, blank=True))

    @classmethod
    def from_csv(cls, fitter, f):
        # Read data into np array
        data = np.loadtxt(f, delimiter=",", skiprows=1)

        # Read header into list
        # TODO only read header line here
        # Read entire csv in again just to get column names to avoid
        # having to convert struct array to ndarray, because laziness
        struct = np.genfromtxt(f, dtype=float, delimiter=',', names=True) 
        header = list(struct.dtype.names)
        return cls.from_np(header, data, fitter=fitter)

    @classmethod
    def from_xls(cls, fitter, f):
        # Number of header rows to skip
        skiprows = 1
        dtype    = 'f8'
        
        # Read data from xls/x into python list
        # TODO use openpyxl here, uninstall xlrd
        header = []
        data = []
        
        with open_workbook(file_contents=f.read()) as wb:
            ws = wb.sheet_by_index(0)
            for r in range(ws.nrows):
                if r == 0:
                    header = ws.row_values(r)
                if r < skiprows:
                    continue
                data.append(ws.row_values(r))

        # Convert to float array
        data_parsed = np.array(data, dtype=dtype)

        # Parse header
        header_parsed = np.array(header)
        
        return cls.from_np(header_parsed, data_parsed, fitter=fitter)

    @classmethod
    def from_np(cls, header, array, fitter=None):
        # Use SHA1 hash of array as primary key to avoid duplication
        # TODO change this to hash both header and array??
        id = hashlib.sha1(np.ascontiguousarray(array.data)).hexdigest()

        # Data format definitions - number of columns expected in x
        # Defined as "many-to-one" dict with tuples and converted 
        # for convenience
        fmts_map = {
            ("default", "nmr1to1", "uv1to1"): {"x": 2},
            ("nmrdimer", "uvdimer", "nmrcoek", "uvcoek"): {"x": 1},
            }
        fmts = {}

        for k, v in fmts_map.items():
            for key in k:
                fmts[key] = v

        # Get appropriate data parsing format for this fitter
        fmt = fmts.get(fitter, fmts["default"])

        # Number of columns of xdata to parse
        nx = fmt["x"]

        x_labels = list(header[0:nx])
        x_raw = array[:,0:nx]
        x = [ list(x_raw[:,col]) for col in range(x_raw.shape[1]) ]

        y_labels = list(header[nx:])
        y_raw = array[:,nx:]
        # Add 3rd dimension to y for consistency w/ true 3D y inputs
        y = [[ list(y_raw[:,col]) for col in range(y_raw.shape[1]) ]]

        logger.debug("Data.from_np: x and y array shapes")
        logger.debug(np.array(x).shape)
        logger.debug(np.array(y).shape)

        return cls(id=id, x=x, y=y, labels_x=x_labels, labels_y=y_labels)

    def to_dict(self, fitter, dilute=False):
        x = np.array(self.x)
        y = np.array(self.y)[0]

        # Calculate x values for plotting
        x_plot = functions.construct(fitter).format_x(x)

        # Apply dilution factor if dilute option is set 
        if dilute:
            y = helpers.dilute(x[0], y)

        return formatter.data(self.id, x, x_plot, y, self.labels_x, self.labels_y)

class Fit(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Edit key
    edit_key = models.UUIDField(default=None, blank=True, null=True)

    # Options and flags
    # Fitter
    fitter_name = models.CharField(max_length=20)
    # Data store only flag (defined on fitter select)
    no_fit = models.BooleanField(default=False)
    # Options strings and flags (defined on fit)
    options_dilute    = models.BooleanField(default=False) # Dilution factor flag
    options_normalise = models.BooleanField(default=True)  # Normalisation (initial value subtraction) flag
    options_method    = models.CharField(default="Nelder-Mead",
                                         max_length=50, 
                                         blank=False) # Fit method
    options_flavour   = models.CharField(default="none",
                                         max_length=50,
                                         blank=False) # Fit flavour 

    # Time to fit
    time = models.FloatField(blank=True, null=True) # Time to fit

    # Metadata and save options
    meta_options_searchable = models.BooleanField(default=True) # Publish/make fit searchable

    meta_email     = models.CharField(max_length=200, blank=True)
    meta_author    = models.CharField(max_length=200, blank=True)
    meta_name      = models.CharField(max_length=200, blank=True)
    meta_date      = models.DateTimeField(blank=True, null=True)
    meta_timestamp = models.DateTimeField(auto_now_add=True)
    meta_ref       = models.CharField(max_length=200, blank=True)
    meta_host      = models.CharField(max_length=200, blank=True)
    meta_guest     = models.CharField(max_length=200, blank=True)
    meta_solvent   = models.CharField(max_length=200, blank=True)
    meta_temp      = models.DecimalField(max_digits=10, decimal_places=5, blank=True, null=True)
    meta_temp_unit = models.CharField(max_length=1, default="C")
    meta_notes     = models.CharField(max_length=10000, blank=True)

    # Link to raw data used for fit
    data = models.ForeignKey(Data)

    # Fit results
    # 1D array of fitted parameters
    fit_params_keys   = ArrayField(models.CharField(max_length=20), 
                                   blank=True, null=True)
    fit_params_init   = ArrayField(models.FloatField(), 
                                   blank=True, null=True)
    fit_params_bounds = ArrayField(
            ArrayField(models.FloatField()), 
            blank=True, null=True)
    # Array of parameter and sub-parameter result values
    # Returned as float if singular, array if multiple
    fit_params_value  = ArrayField(
            ArrayField(models.FloatField()), 
            blank=True, null=True) 
    # Array of parameter and sub-parameter result error values
    # Returned as float if singular, array if multiple
    fit_params_stderr = ArrayField(
            ArrayField(models.FloatField()), 
            blank=True, null=True)

    # 2D matrix of (calculated) fitted input y data
    fit_y = ArrayField(
            ArrayField(models.FloatField()),
            blank=True, null=True)

    fit_molefrac = ArrayField(
            ArrayField(models.FloatField()),
            blank=True, null=True)

    fit_coeffs   = ArrayField(
            ArrayField(models.FloatField()),
            blank=True, null=True)

    fit_molefrac_raw = ArrayField(
            ArrayField(models.FloatField()),
            blank=True, null=True)

    fit_coeffs_raw   = ArrayField(
            ArrayField(models.FloatField()),
            blank=True, null=True)

    qof_residuals = ArrayField(
            ArrayField(models.FloatField()),
            blank=True, null=True)



    def to_dict(self):
        meta_dict = {
            "author":    self.meta_author,
            "name":      self.meta_name,
            "date":      self.meta_date,
            "timestamp": self.meta_timestamp,
            "ref":       self.meta_ref,
            "host":      self.meta_host,
            "guest":     self.meta_guest,
            "solvent":   self.meta_solvent,
            "temp":      self.meta_temp,
            "temp_unit": self.meta_temp_unit,
            "notes":     self.meta_notes,
            "options_searchable": 
               self.meta_options_searchable,
            }

        if not self.no_fit:
            # Return full fit

            # Strip any padded values
            fit_params_value  = helpers.unpad_2d(self.fit_params_value)
            fit_params_stderr = helpers.unpad_2d(self.fit_params_stderr)

            # Convert parameter arrays to appropriate nested dict input to formatter
            # Backwards compatibility check for saved fits without bounds
            if self.fit_params_bounds:
                params = { key: {"init": init, 
                                 # Backwards compatibility for previously saved fits
                                 # Check for bare floats, null errors
                                 # If singular item in array, strip array before return
                                 "value":  value  if (not isinstance(value, list))  
                                                      or (len(value)  > 1) 
                                                  else value[0], 
                                 "stderr": stderr if (not isinstance(stderr, list)) 
                                                      or (len(stderr) > 1) 
                                                  else stderr[0],
                                 "bounds": {"min": bounds[0], "max": bounds[1]}}
                           for (key, init, value, stderr, bounds)
                           in zip(self.fit_params_keys,
                                  self.fit_params_init,
                                  fit_params_value,
                                  fit_params_stderr,
                                  self.fit_params_bounds) }
            else:
                params = { key: {"init": init, 
                                 # Backwards compatibility for previously saved fits
                                 # Check for bare floats, null errors
                                 # If singular item in array, strip array before return
                                 "value":  value  if (not isinstance(value, list))  
                                                      or (len(value)  > 1) 
                                                  else value[0], 
                                 "stderr": stderr if (not isinstance(stderr, list)) 
                                                      or (len(stderr) > 1) 
                                                  else stderr[0],
                                 "bounds": {"min": None, "max": None}}
                           for (key, init, value, stderr)
                           in zip(self.fit_params_keys,
                                  self.fit_params_init,
                                  fit_params_value,
                                  fit_params_stderr) }
                       
            response = formatter.fit(fitter      =self.fitter_name,
                                     data        =self.data.to_dict(self.fitter_name, self.options_dilute),
                                     y           =self.fit_y, 
                                     params      =params, 
                                     residuals   =self.qof_residuals,
                                     molefrac_raw=self.fit_molefrac_raw,
                                     coeffs_raw  =self.fit_coeffs_raw,
                                     molefrac    =self.fit_molefrac,
                                     coeffs      =self.fit_coeffs,
                                     time        =self.time,
                                     dilute      =self.options_dilute,
                                     normalise   =self.options_normalise,
                                     method      =self.options_method,
                                     flavour     =self.options_flavour,
                                     no_fit      =self.no_fit,
                                     meta_dict   =meta_dict,
                                     )
        else:
            # No fit, return only saved input data
            response = formatter.fit(self.fitter_name,
                                     self.data.to_dict(self.fitter_name, self.options_dilute),
                                     no_fit=self.no_fit,
                                     meta_dict=meta_dict,
                                     )

        return response

    @property
    def summary(self):
        # Return summary of fit (for search results and email retrieval)
        response = formatter.fit_summary(self.id, 
                                         self.fitter_name,
                                         self.meta_name, 
                                         self.meta_author, 
                                         self.meta_timestamp)
        return response
