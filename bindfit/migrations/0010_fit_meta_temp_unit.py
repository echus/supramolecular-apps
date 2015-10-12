# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0009_auto_20150922_1818'),
    ]

    operations = [
        migrations.AddField(
            model_name='fit',
            name='meta_temp_unit',
            field=models.CharField(default='C', max_length=1),
        ),
    ]
