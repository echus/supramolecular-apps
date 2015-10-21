# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import django.contrib.postgres.fields


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0011_fit_options_dilute'),
    ]

    operations = [
        migrations.AddField(
            model_name='data',
            name='x_labels',
            field=django.contrib.postgres.fields.ArrayField(default=['[H]0', '[G]0'], size=None, base_field=models.CharField(blank=True, max_length=100)),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='data',
            name='y_labels',
            field=django.contrib.postgres.fields.ArrayField(default=[], size=None, base_field=models.CharField(blank=True, max_length=100)),
            preserve_default=False,
        ),
    ]
