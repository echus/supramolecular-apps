# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import django.contrib.postgres.fields


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='fit',
            name='no_fit',
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name='fit',
            name='fit_coeffs',
            field=django.contrib.postgres.fields.ArrayField(blank=True, size=None, base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), null=True),
        ),
        migrations.AlterField(
            model_name='fit',
            name='fit_molefrac',
            field=django.contrib.postgres.fields.ArrayField(blank=True, size=None, base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), null=True),
        ),
        migrations.AlterField(
            model_name='fit',
            name='fit_params_init',
            field=django.contrib.postgres.fields.ArrayField(blank=True, size=None, base_field=models.FloatField(), null=True),
        ),
        migrations.AlterField(
            model_name='fit',
            name='fit_params_keys',
            field=django.contrib.postgres.fields.ArrayField(blank=True, size=None, base_field=models.CharField(max_length=20), null=True),
        ),
        migrations.AlterField(
            model_name='fit',
            name='fit_params_stderr',
            field=django.contrib.postgres.fields.ArrayField(blank=True, size=None, base_field=models.FloatField(), null=True),
        ),
        migrations.AlterField(
            model_name='fit',
            name='fit_params_value',
            field=django.contrib.postgres.fields.ArrayField(blank=True, size=None, base_field=models.FloatField(), null=True),
        ),
        migrations.AlterField(
            model_name='fit',
            name='fit_y',
            field=django.contrib.postgres.fields.ArrayField(blank=True, size=None, base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), null=True),
        ),
        migrations.AlterField(
            model_name='fit',
            name='qof_residuals',
            field=django.contrib.postgres.fields.ArrayField(blank=True, size=None, base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), null=True),
        ),
        migrations.AlterField(
            model_name='fit',
            name='time',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
