# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0004_fit_searchable'),
    ]

    operations = [
        migrations.RenameField(
            model_name='fit',
            old_name='searchable',
            new_name='meta_options_searchable',
        ),
    ]
