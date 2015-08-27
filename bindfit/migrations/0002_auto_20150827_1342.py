# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='data',
            name='fit',
            field=models.ForeignKey(blank=True, default=None, to='bindfit.Fit', null=True),
        ),
    ]
