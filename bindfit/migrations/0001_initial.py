# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import django.contrib.postgres.fields


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Fit',
            fields=[
                ('id', models.AutoField(primary_key=True, verbose_name='ID', auto_created=True, serialize=False)),
                ('name', models.CharField(blank=True, max_length=200)),
                ('notes', models.CharField(blank=True, max_length=1000)),
            ],
        ),
        migrations.CreateModel(
            name='Data',
            fields=[
                ('fit', models.OneToOneField(primary_key=True, to='bindfit.Fit', serialize=False)),
                ('h0', django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None)),
                ('g0', django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None)),
                ('y', django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), size=None)),
            ],
        ),
        migrations.CreateModel(
            name='Options',
            fields=[
                ('fit', models.OneToOneField(primary_key=True, to='bindfit.Fit', serialize=False)),
                ('fitter', models.CharField(max_length=20)),
                ('parameters', django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None)),
            ],
        ),
        migrations.CreateModel(
            name='Result',
            fields=[
                ('fit', models.OneToOneField(primary_key=True, to='bindfit.Fit', serialize=False)),
                ('y', django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), size=None)),
            ],
        ),
    ]
