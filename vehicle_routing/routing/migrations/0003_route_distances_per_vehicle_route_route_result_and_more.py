# Generated by Django 5.0.6 on 2024-05-22 20:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('routing', '0002_remove_route_customers_remove_route_total_distance_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='route',
            name='distances_per_vehicle',
            field=models.JSONField(default=dict),
        ),
        migrations.AddField(
            model_name='route',
            name='route_result',
            field=models.JSONField(default=dict),
        ),
        migrations.AddField(
            model_name='route',
            name='total_distance',
            field=models.FloatField(default=0.0),
        ),
        migrations.AlterField(
            model_name='route',
            name='algorithm',
            field=models.CharField(default='ILP', max_length=50),
        ),
        migrations.AlterField(
            model_name='route',
            name='vehicle_capacity',
            field=models.IntegerField(verbose_name=100),
        ),
    ]
