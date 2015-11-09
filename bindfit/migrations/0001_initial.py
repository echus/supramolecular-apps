# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import uuid
import django.contrib.postgres.fields
import django.contrib.postgres.fields.hstore


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Data',
            fields=[
                ('id', models.CharField(serialize=False, max_length=40, primary_key=True)),
                ('x', django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), size=None)),
                ('y', django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), size=None), size=None)),
                ('labels_x', django.contrib.postgres.fields.ArrayField(base_field=models.CharField(max_length=100, blank=True), size=None)),
                ('labels_y', django.contrib.postgres.fields.ArrayField(base_field=models.CharField(max_length=100, blank=True), size=None)),
            ],
        ),
        migrations.CreateModel(
            name='Fit',
            fields=[
                ('id', models.UUIDField(serialize=False, primary_key=True, default=uuid.uuid4, editable=False)),
                ('meta_author', models.CharField(max_length=200, blank=True)),
                ('meta_name', models.CharField(max_length=200, blank=True)),
                ('meta_date', models.DateTimeField(null=True, blank=True)),
                ('meta_timestamp', models.DateTimeField(auto_now_add=True)),
                ('meta_ref', models.CharField(max_length=200, blank=True)),
                ('meta_host', models.CharField(max_length=200, blank=True)),
                ('meta_guest', models.CharField(max_length=200, blank=True)),
                ('meta_solvent', models.CharField(max_length=200, blank=True)),
                ('meta_temp', models.DecimalField(null=True, decimal_places=5, blank=True, max_digits=10)),
                ('meta_temp_unit', models.CharField(max_length=1, default='C')),
                ('meta_notes', models.CharField(max_length=10000, blank=True)),
                ('options_fitter', models.CharField(max_length=20)),
                ('options_params', django.contrib.postgres.fields.hstore.HStoreField()),
                ('options_dilute', models.BooleanField(default=False)),
                ('fit_params', django.contrib.postgres.fields.hstore.HStoreField()),
                ('fit_y', django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), size=None)),
                ('fit_residuals', django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), size=None)),
                ('fit_molefrac', django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), size=None)),
                ('fit_coeffs', django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), size=None)),
                ('fit_time', models.FloatField()),
                ('data', models.ForeignKey(to='bindfit.Data')),
            ],
        ),
    ]
