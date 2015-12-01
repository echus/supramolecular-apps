# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import django.contrib.postgres.fields
import uuid


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Data',
            fields=[
                ('id', models.CharField(primary_key=True, serialize=False, max_length=40)),
                ('x', django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), size=None)),
                ('y', django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), size=None), size=None)),
                ('labels_x', django.contrib.postgres.fields.ArrayField(base_field=models.CharField(max_length=100, blank=True), size=None)),
                ('labels_y', django.contrib.postgres.fields.ArrayField(base_field=models.CharField(max_length=100, blank=True), size=None)),
            ],
        ),
        migrations.CreateModel(
            name='Fit',
            fields=[
                ('id', models.UUIDField(primary_key=True, default=uuid.uuid4, serialize=False, editable=False)),
                ('meta_author', models.CharField(max_length=200, blank=True)),
                ('meta_name', models.CharField(max_length=200, blank=True)),
                ('meta_date', models.DateTimeField(null=True, blank=True)),
                ('meta_timestamp', models.DateTimeField(auto_now_add=True)),
                ('meta_ref', models.CharField(max_length=200, blank=True)),
                ('meta_host', models.CharField(max_length=200, blank=True)),
                ('meta_guest', models.CharField(max_length=200, blank=True)),
                ('meta_solvent', models.CharField(max_length=200, blank=True)),
                ('meta_temp', models.DecimalField(null=True, decimal_places=5, max_digits=10, blank=True)),
                ('meta_temp_unit', models.CharField(default='C', max_length=1)),
                ('meta_notes', models.CharField(max_length=10000, blank=True)),
                ('fitter_name', models.CharField(max_length=20)),
                ('options_dilute', models.BooleanField(default=False)),
                ('fit_params_keys', django.contrib.postgres.fields.ArrayField(base_field=models.CharField(max_length=20), size=None)),
                ('fit_params_init', django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None)),
                ('fit_params_value', django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None)),
                ('fit_params_stderr', django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None)),
                ('fit_y', django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), size=None)),
                ('fit_molefrac', django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), size=None)),
                ('fit_coeffs', django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), size=None)),
                ('qof_residuals', django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), size=None)),
                ('time', models.FloatField()),
                ('data', models.ForeignKey(to='bindfit.Data')),
            ],
        ),
    ]
