# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import django.contrib.postgres.fields


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0005_auto_20160222_1540'),
    ]

    operations = [
        migrations.AddField(
            model_name='fit',
            name='fit_params_bounds',
            field=django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), size=None, blank=True, null=True),
        ),
        migrations.AddField(
            model_name='fit',
            name='options_flavour',
            field=models.CharField(max_length=50, default='', blank=True),
        ),
        migrations.AddField(
            model_name='fit',
            name='options_method',
            field=models.CharField(max_length=50, default='', blank=True),
        ),
        migrations.AddField(
            model_name='fit',
            name='options_normalise',
            field=models.BooleanField(default=True),
        ),
    ]
