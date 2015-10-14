# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0010_fit_meta_temp_unit'),
    ]

    operations = [
        migrations.AddField(
            model_name='fit',
            name='options_dilute',
            field=models.BooleanField(default=False),
        ),
    ]
