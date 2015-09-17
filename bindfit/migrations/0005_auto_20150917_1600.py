# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import django.contrib.postgres.fields


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0004_fit_residuals'),
    ]

    operations = [
        migrations.RenameField(
            model_name='fit',
            old_name='species_coeff',
            new_name='fit_coeffs',
        ),
        migrations.RenameField(
            model_name='fit',
            old_name='species_molefrac',
            new_name='fit_molefrac',
        ),
        migrations.RenameField(
            model_name='fit',
            old_name='params',
            new_name='fit_params',
        ),
        migrations.RenameField(
            model_name='fit',
            old_name='residuals',
            new_name='fit_residuals',
        ),
        migrations.RenameField(
            model_name='fit',
            old_name='name',
            new_name='meta_name',
        ),
        migrations.RenameField(
            model_name='fit',
            old_name='notes',
            new_name='meta_notes',
        ),
        migrations.RenameField(
            model_name='fit',
            old_name='fitter',
            new_name='options_fitter',
        ),
        migrations.RenameField(
            model_name='fit',
            old_name='params_guess',
            new_name='options_params',
        ),
        migrations.RemoveField(
            model_name='data',
            name='g0',
        ),
        migrations.RemoveField(
            model_name='data',
            name='h0',
        ),
        migrations.RemoveField(
            model_name='fit',
            name='y',
        ),
        migrations.AddField(
            model_name='data',
            name='x',
            field=django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), size=None, default=list),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='fit',
            name='fit_y',
            field=django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), size=None), size=None, default=list),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='data',
            name='y',
            field=django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), size=None), size=None),
        ),
    ]
