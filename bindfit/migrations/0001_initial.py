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
                ('id', models.CharField(serialize=False, max_length=40, primary_key=True)),
                ('h0', django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None)),
                ('g0', django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None)),
                ('y', django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), size=None)),
            ],
        ),
        migrations.CreateModel(
            name='Fit',
            fields=[
                ('id', models.UUIDField(serialize=False, editable=False, default=uuid.uuid4, primary_key=True)),
                ('name', models.CharField(max_length=200, blank=True)),
                ('notes', models.CharField(max_length=1000, blank=True)),
                ('fitter', models.CharField(max_length=20)),
                ('params_guess', django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None)),
                ('params', django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None)),
                ('y', django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), size=None)),
                ('data', models.ForeignKey(to='bindfit.Data')),
            ],
        ),
    ]
