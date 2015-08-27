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
                ('h0', django.contrib.postgres.fields.ArrayField(size=None, base_field=models.FloatField())),
                ('g0', django.contrib.postgres.fields.ArrayField(size=None, base_field=models.FloatField())),
                ('y', django.contrib.postgres.fields.ArrayField(size=None, base_field=django.contrib.postgres.fields.ArrayField(size=None, base_field=models.FloatField()))),
            ],
        ),
        migrations.CreateModel(
            name='Fit',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, serialize=False, editable=False, primary_key=True)),
                ('name', models.CharField(max_length=200, blank=True)),
                ('notes', models.CharField(max_length=1000, blank=True)),
            ],
        ),
        migrations.CreateModel(
            name='Options',
            fields=[
                ('fit', models.OneToOneField(primary_key=True, serialize=False, to='bindfit.Fit')),
                ('fitter', models.CharField(max_length=20)),
                ('parameters', django.contrib.postgres.fields.ArrayField(size=None, base_field=models.FloatField())),
            ],
        ),
        migrations.CreateModel(
            name='Result',
            fields=[
                ('fit', models.OneToOneField(primary_key=True, serialize=False, to='bindfit.Fit')),
                ('y', django.contrib.postgres.fields.ArrayField(size=None, base_field=django.contrib.postgres.fields.ArrayField(size=None, base_field=models.FloatField()))),
            ],
        ),
        migrations.AddField(
            model_name='data',
            name='fit',
            field=models.ForeignKey(default=None, null=True, to='bindfit.Fit', blank=True),
        ),
    ]
