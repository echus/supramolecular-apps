# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='fit',
            name='notes',
            field=models.CharField(max_length=10000, blank=True),
        ),
    ]
