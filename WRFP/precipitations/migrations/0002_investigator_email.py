# Generated by Django 2.0.6 on 2019-01-22 04:36

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('precipitations', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='investigator',
            name='email',
            field=models.EmailField(default=django.utils.timezone.now, max_length=254),
            preserve_default=False,
        ),
    ]
