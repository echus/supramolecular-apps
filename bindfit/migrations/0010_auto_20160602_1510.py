# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0009_fit_edit_key'),
    ]

    operations = [
        migrations.AlterField(
            model_name='fit',
            name='options_flavour',
            field=models.CharField(max_length=50, default='none'),
        ),
        migrations.AlterField(
            model_name='fit',
            name='options_method',
            field=models.CharField(max_length=50, default='Nelder-Mead'),
        ),
    ]
