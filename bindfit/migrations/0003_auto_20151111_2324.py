# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import django.contrib.postgres.fields


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0002_auto_20151111_2314'),
    ]

    operations = [
        migrations.AlterField(
            model_name='fit',
            name='fit_params_keys',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.CharField(max_length=20), size=None),
        ),
    ]
