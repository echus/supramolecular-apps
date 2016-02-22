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
    def from_csv(cls, f):
        # Read data into np array
        data = np.loadtxt(f, delimiter=",", skiprows=1)

        # Read header into list
        # TODO only read header line here
        # Read entire csv in again just to get column names to avoid
        # having to convert struct array to ndarray, because laziness
        struct = np.genfromtxt(f, dtype=float, delimiter=',', names=True) 
        header = list(struct.dtype.names)
        return cls.from_np(header, data)

    @classmethod
    def from_xls(cls, f):
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
        
        return cls.from_np(header_parsed, data_parsed)

    @classmethod
    def from_np(cls, header, array, fmt=None):
        # Use SHA1 hash of array as primary key to avoid duplication
        # TODO change this to hash both header and array??
        id = hashlib.sha1(np.ascontiguousarray(array.data)).hexdigest()

        if fmt == "2d":
            # Placeholder for handling decoding other array formats
            pass
        else:
            # Default format, 2 x cols and 1 2D y input
            x_labels = list(header[0:2])
            x_raw = array[:,0:2]
            x = [ list(x_raw[:,col]) for col in range(x_raw.shape[1]) ]

            y_labels = list(header[2:])
            y_raw = array[:,2:]
            # Add 3rd dimension to y for consistency w/ true 3D y inputs
            y = [[ list(y_raw[:,col]) for col in range(y_raw.shape[1]) ]]

        logger.debug("Data.from_np: x and y arrays")
        logger.debug(x)
        logger.debug(y)

        return cls(id=id, x=x, y=y, labels_x=x_labels, labels_y=y_labels)

    def to_dict(self, dilute=False):
        x = np.array(self.x)
        y = np.array(self.y)[0]

        # Apply dilution factor if dilute option is set 
        if dilute:
            y = helpers.dilute(x, y)

        return formatter.data(self.id, x, y, self.labels_x, self.labels_y)

class Fit(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Options and flags
    # Fitter
    fitter_name = models.CharField(max_length=20)
    # Data store only flag (defined on fitter select)
    no_fit = models.BooleanField(default=False)
    # Dilution flag (defined on fit)
    options_dilute = models.BooleanField(default=False) # Dilution factor flag

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
    fit_params_keys   = ArrayField(models.CharField(max_length=20), blank=True, null=True)
    fit_params_init   = ArrayField(models.FloatField(), blank=True, null=True)
    fit_params_value  = ArrayField(models.FloatField(), blank=True, null=True)
    fit_params_stderr = ArrayField(models.FloatField(), blank=True, null=True)

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

    qof_residuals = ArrayField(
            ArrayField(models.FloatField()),
            blank=True, null=True)


    def to_dict(self):
        if not self.no_fit:
            # Return full fit

            # Convert parameter arrays to appropriate nested dict input to formatter
            params = { key: {"init": init, "value": value, "stderr": stderr}
                       for (key, init, value, stderr)
                       in zip(self.fit_params_keys,
                              self.fit_params_init,
                              self.fit_params_value,
                              self.fit_params_stderr) }

            response = formatter.fit(self.fitter_name,
                                     self.data.to_dict(self.options_dilute),
                                     self.fit_y, 
                                     params, 
                                     self.qof_residuals,
                                     self.fit_molefrac,
                                     self.fit_coeffs,
                                     self.time,
                                     self.options_dilute,
                                     self.no_fit)
        else:
            # No fit, return only saved input data
            response = formatter.fit(self.fitter_name,
                                     self.data.to_dict(self.options_dilute),
                                     no_fit=self.no_fit)

        response["meta"] = formatter.meta(self.meta_author,
                                          self.meta_name,
                                          self.meta_date,
                                          self.meta_timestamp,
                                          self.meta_ref,
                                          self.meta_host,
                                          self.meta_guest,
                                          self.meta_solvent,
                                          self.meta_temp,
                                          self.meta_temp_unit,
                                          self.meta_notes)
        return response
