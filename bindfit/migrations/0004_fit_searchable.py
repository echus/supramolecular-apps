# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0003_fit_meta_email'),
    ]

    operations = [
        migrations.AddField(
            model_name='fit',
            name='searchable',
            field=models.BooleanField(default=True),
        ),
    ]
