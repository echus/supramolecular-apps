# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0007_auto_20150921_1224'),
    ]

    operations = [
        migrations.AlterField(
            model_name='fit',
            name='meta_temp',
            field=models.DecimalField(max_digits=10, null=True, decimal_places=5, blank=True),
        ),
    ]
