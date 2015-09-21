# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import datetime
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('bindfit', '0005_auto_20150917_1600'),
    ]

    operations = [
        migrations.AddField(
            model_name='fit',
            name='meta_author',
            field=models.CharField(max_length=200, blank=True),
        ),
        migrations.AddField(
            model_name='fit',
            name='meta_date',
            field=models.DateTimeField(default=datetime.datetime(2015, 9, 21, 1, 37, 26, 881930, tzinfo=utc), blank=True),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='fit',
            name='meta_guest',
            field=models.CharField(max_length=200, blank=True),
        ),
        migrations.AddField(
            model_name='fit',
            name='meta_host',
            field=models.CharField(max_length=200, blank=True),
        ),
        migrations.AddField(
            model_name='fit',
            name='meta_ref',
            field=models.CharField(max_length=200, blank=True),
        ),
        migrations.AddField(
            model_name='fit',
            name='meta_solvent',
            field=models.CharField(max_length=200, blank=True),
        ),
        migrations.AddField(
            model_name='fit',
            name='meta_temp',
            field=models.DecimalField(max_digits=10, default=0, decimal_places=10, blank=True),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='fit',
            name='meta_timestamp',
            field=models.DateTimeField(auto_now_add=True, default=datetime.datetime(2015, 9, 21, 1, 37, 51, 987619, tzinfo=utc)),
            preserve_default=False,
        ),
    ]
