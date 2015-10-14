from __future__ import division
from __future__ import print_function

from django.db import models
from django.contrib.postgres.fields import ArrayField 

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

    @classmethod
    def from_csv(cls, f):
        raw = np.loadtxt(f, delimiter=",", skiprows=1)
        return cls.from_np(raw)

    @classmethod
    def from_xls(cls, f):
        # Number of header rows to skip
        skiprows = 1
        dtype    = 'f8'
        
        # Read data from xls/x into python list
        # TODO use openpyxl here, uninstall xlrd
        data = []
        
        with open_workbook(file_contents=f.read()) as wb:
            ws = wb.sheet_by_index(0)
            for r in range(ws.nrows):
                if r < skiprows:
                    continue
                data.append(ws.row_values(r))

        # Convert to float array
        raw = np.array(data, dtype=dtype)
        
        return cls.from_np(raw)

    @classmethod
    def from_np(cls, array, fmt=None):
        # Use SHA1 hash of array as primary key to avoid duplication
        id = hashlib.sha1(np.ascontiguousarray(array.data)).hexdigest()

        if fmt == "2d":
            # Placeholder for handling decoding other array formats
            pass
        else:
            # Default format, 2 x cols and 1 2D y input
            x_raw = array[:,0:2]
            x = [ list(x_raw[:,col]) for col in range(x_raw.shape[1]) ]

            y_raw = array[:,2:]
            # Add 3rd dimension to y for consistency w/ true 3D y inputs
            y = [[ list(y_raw[:,col]) for col in range(y_raw.shape[1]) ]]

        logger.debug("Data.from_np: x and y arrays")
        logger.debug(x)
        logger.debug(y)

        return cls(id=id, x=x, y=y)

    def to_dict(self, dilute=False):
        x = np.array(self.x)
        y = np.array(self.y)

        # Apply dilution factor if dilute option is set 
        if dilute:
            y = helpers.dilute(x, y)

        return formatter.data(x, y)

class Fit(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Metadata
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

    # Fit options 
    options_fitter = models.CharField(max_length=20)
    options_params = ArrayField(base_field=models.FloatField())
    options_dilute = models.BooleanField(default=False) # Dilution factor flag

    # Fit results
    # 1D array of fitted parameters
    fit_params = ArrayField(base_field=models.FloatField())

    # 3D array of 2D matrices of fitted data for each input y data
    fit_y = ArrayField(
            ArrayField(
                ArrayField(models.FloatField())
                )
            )
    
    fit_residuals = ArrayField(
            ArrayField(
                ArrayField(models.FloatField())
                )
            )

    fit_molefrac = ArrayField(
            ArrayField(models.FloatField())
            )

    fit_coeffs   = ArrayField(
            ArrayField(models.FloatField())
            )

    def to_dict(self):
        response = {
                "data": self.data.to_dict(self.options_dilute),
                "fit" : formatter.fit(self.fit_y, 
                                      self.fit_params, 
                                      self.fit_residuals,
                                      self.fit_molefrac,
                                      self.fit_coeffs),
                "meta": formatter.meta(self.meta_author,
                                       self.meta_name,
                                       self.meta_date,
                                       self.meta_timestamp,
                                       self.meta_ref,
                                       self.meta_host,
                                       self.meta_guest,
                                       self.meta_solvent,
                                       self.meta_temp,
                                       self.meta_temp_unit,
                                       self.meta_notes),
                "options": formatter.options(self.options_fitter,
                                             self.data.id,
                                             self.options_params,
                                             self.options_dilute),
                }
        return response
