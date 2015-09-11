# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import django.contrib.postgres.fields


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0003_auto_20150911_1246'),
    ]

    operations = [
        migrations.AddField(
            model_name='fit',
            name='residuals',
            field=django.contrib.postgres.fields.ArrayField(size=None, base_field=models.FloatField(), default=[]),
            preserve_default=False,
        ),
    ]
