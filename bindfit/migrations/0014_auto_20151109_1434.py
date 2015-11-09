# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import django.contrib.postgres.fields


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0013_fit_fit_time'),
    ]

    operations = [
        migrations.AlterField(
            model_name='fit',
            name='fit_residuals',
            field=django.contrib.postgres.fields.ArrayField(size=None, base_field=django.contrib.postgres.fields.ArrayField(size=None, base_field=models.FloatField())),
        ),
        migrations.AlterField(
            model_name='fit',
            name='fit_y',
            field=django.contrib.postgres.fields.ArrayField(size=None, base_field=django.contrib.postgres.fields.ArrayField(size=None, base_field=models.FloatField())),
        ),
    ]
