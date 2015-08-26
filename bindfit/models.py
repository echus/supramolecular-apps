from django.db import models
from django.contrib.postgres.fields import ArrayField

class Fit(models.Model):
    name = models.CharField(max_length=200, blank=True)
    notes = models.CharField(max_length=1000, blank=True)

class Options(models.Model):
    fit = models.OneToOneField(Fit, primary_key=True)
    fitter = models.CharField(max_length=20)
    parameters = ArrayField(base_field=models.FloatField())

class Data(models.Model):
    fit = models.OneToOneField(Fit, primary_key=True)
    h0 = ArrayField(models.FloatField())
    g0 = ArrayField(models.FloatField())
    y = ArrayField(
            ArrayField(models.FloatField())
            )

    @property
    def geq(self):
        # Calculate equivalent [G]0/[H]0 concentration
        return None

class Result(models.Model):
    fit = models.OneToOneField(Fit, primary_key=True)
    y = ArrayField(
            ArrayField(models.FloatField())
            )
