# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0006_auto_20150921_1137'),
    ]

    operations = [
        migrations.AlterField(
            model_name='fit',
            name='meta_date',
            field=models.DateTimeField(null=True, blank=True),
        ),
        migrations.AlterField(
            model_name='fit',
            name='meta_temp',
            field=models.DecimalField(max_digits=10, decimal_places=10, null=True, blank=True),
        ),
    ]
