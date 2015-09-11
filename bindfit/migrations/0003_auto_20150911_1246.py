# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import django.contrib.postgres.fields


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0002_auto_20150902_2025'),
    ]

    operations = [
        migrations.AddField(
            model_name='fit',
            name='species_coeff',
            field=django.contrib.postgres.fields.ArrayField(default=[], size=None, base_field=django.contrib.postgres.fields.ArrayField(size=None, base_field=models.FloatField())),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='fit',
            name='species_molefrac',
            field=django.contrib.postgres.fields.ArrayField(default=[], size=None, base_field=django.contrib.postgres.fields.ArrayField(size=None, base_field=models.FloatField())),
            preserve_default=False,
        ),
    ]
