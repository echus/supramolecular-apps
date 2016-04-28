# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0008_auto_20160421_1032'),
    ]

    operations = [
        migrations.AddField(
            model_name='fit',
            name='edit_key',
            field=models.UUIDField(blank=True, default=None, null=True),
        ),
    ]
