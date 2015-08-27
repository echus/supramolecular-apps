# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import django.contrib.postgres.fields


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Data',
            fields=[
                ('id', models.AutoField(serialize=False, auto_created=True, verbose_name='ID', primary_key=True)),
                ('h0', django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None)),
                ('g0', django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None)),
                ('y', django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), size=None)),
            ],
        ),
        migrations.CreateModel(
            name='Fit',
            fields=[
                ('id', models.AutoField(serialize=False, auto_created=True, verbose_name='ID', primary_key=True)),
                ('name', models.CharField(max_length=200, blank=True)),
                ('notes', models.CharField(max_length=1000, blank=True)),
            ],
        ),
        migrations.CreateModel(
            name='Options',
            fields=[
                ('fit', models.OneToOneField(to='bindfit.Fit', serialize=False, primary_key=True)),
                ('fitter', models.CharField(max_length=20)),
                ('parameters', django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None)),
            ],
        ),
        migrations.CreateModel(
            name='Result',
            fields=[
                ('fit', models.OneToOneField(to='bindfit.Fit', serialize=False, primary_key=True)),
                ('y', django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), size=None)),
            ],
        ),
        migrations.AddField(
            model_name='data',
            name='fit',
            field=models.ForeignKey(to='bindfit.Fit'),
        ),
    ]
