# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import uuid
import django.contrib.postgres.fields


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Data',
            fields=[
                ('id', models.CharField(max_length=40, primary_key=True, serialize=False)),
                ('h0', django.contrib.postgres.fields.ArrayField(size=None, base_field=models.FloatField())),
                ('g0', django.contrib.postgres.fields.ArrayField(size=None, base_field=models.FloatField())),
                ('y', django.contrib.postgres.fields.ArrayField(size=None, base_field=django.contrib.postgres.fields.ArrayField(size=None, base_field=models.FloatField()))),
            ],
        ),
        migrations.CreateModel(
            name='Fit',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, serialize=False, primary_key=True, editable=False)),
                ('name', models.CharField(max_length=200, blank=True)),
                ('notes', models.CharField(max_length=1000, blank=True)),
                ('fitter', models.CharField(max_length=20)),
                ('params', django.contrib.postgres.fields.ArrayField(size=None, base_field=models.FloatField())),
            ],
        ),
        migrations.CreateModel(
            name='Result',
            fields=[
                ('fit', models.OneToOneField(primary_key=True, to='bindfit.Fit', serialize=False)),
                ('params', django.contrib.postgres.fields.ArrayField(size=None, base_field=models.FloatField())),
                ('y', django.contrib.postgres.fields.ArrayField(size=None, base_field=django.contrib.postgres.fields.ArrayField(size=None, base_field=models.FloatField()))),
            ],
        ),
        migrations.AddField(
            model_name='fit',
            name='data',
            field=models.ForeignKey(to='bindfit.Data'),
        ),
    ]
