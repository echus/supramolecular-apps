# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0002_auto_20150827_1342'),
    ]

    operations = [
        migrations.AlterField(
            model_name='data',
            name='id',
            field=models.CharField(max_length=40, primary_key=True, serialize=False),
        ),
    ]
