# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0002_auto_20160202_1735'),
    ]

    operations = [
        migrations.AddField(
            model_name='fit',
            name='meta_email',
            field=models.CharField(blank=True, max_length=200),
        ),
    ]
