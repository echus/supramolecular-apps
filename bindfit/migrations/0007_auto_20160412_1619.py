# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import django.contrib.postgres.fields


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0006_auto_20160411_1229'),
    ]

    operations = [
        migrations.AlterField(
            model_name='fit',
            name='fit_params_stderr',
            field=django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(size=None, base_field=models.FloatField()), size=None, null=True, blank=True),
        ),
        migrations.AlterField(
            model_name='fit',
            name='fit_params_value',
            field=django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(size=None, base_field=models.FloatField()), size=None, null=True, blank=True),
        ),
    ]
