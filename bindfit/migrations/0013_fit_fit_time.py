# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0012_auto_20151021_1130'),
    ]

    operations = [
        migrations.AddField(
            model_name='fit',
            name='fit_time',
            field=models.FloatField(default=-1),
            preserve_default=False,
        ),
    ]
