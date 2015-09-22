# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import django.contrib.postgres.fields


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0008_auto_20150921_1341'),
    ]

    operations = [
        migrations.AlterField(
            model_name='fit',
            name='fit_residuals',
            field=django.contrib.postgres.fields.ArrayField(size=None, base_field=django.contrib.postgres.fields.ArrayField(size=None, base_field=django.contrib.postgres.fields.ArrayField(size=None, base_field=models.FloatField()))),
        ),
    ]
