# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import django.contrib.postgres.fields


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='fit',
            old_name='options_fitter',
            new_name='fitter_name',
        ),
        migrations.RenameField(
            model_name='fit',
            old_name='fit_residuals',
            new_name='qof_residuals',
        ),
        migrations.RenameField(
            model_name='fit',
            old_name='fit_time',
            new_name='time',
        ),
        migrations.RemoveField(
            model_name='fit',
            name='fit_params',
        ),
        migrations.RemoveField(
            model_name='fit',
            name='options_params',
        ),
        migrations.AddField(
            model_name='fit',
            name='fit_params_init',
            field=django.contrib.postgres.fields.ArrayField(size=None, base_field=models.FloatField(), default=[]),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='fit',
            name='fit_params_keys',
            field=django.contrib.postgres.fields.ArrayField(size=None, base_field=models.FloatField(), default=[]),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='fit',
            name='fit_params_stderr',
            field=django.contrib.postgres.fields.ArrayField(size=None, base_field=models.FloatField(), default=[]),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='fit',
            name='fit_params_value',
            field=django.contrib.postgres.fields.ArrayField(size=None, base_field=models.FloatField(), default=[]),
            preserve_default=False,
        ),
    ]
