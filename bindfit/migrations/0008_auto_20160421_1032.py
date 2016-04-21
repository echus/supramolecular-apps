# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import django.contrib.postgres.fields


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0007_auto_20160412_1619'),
    ]

    operations = [
        migrations.AddField(
            model_name='fit',
            name='fit_coeffs_raw',
            field=django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), null=True, blank=True, size=None),
        ),
        migrations.AddField(
            model_name='fit',
            name='fit_molefrac_raw',
            field=django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), null=True, blank=True, size=None),
        ),
    ]
