from django.db import models
from django.contrib.postgres.fields import JSONField  # Use this for PostgreSQL, otherwise use TextField

class Customer(models.Model):
    latitude = models.FloatField()
    longitude = models.FloatField()
    demand_bags = models.IntegerField()
    demand_packs = models.IntegerField()

class Vehicle(models.Model):
    capacity = models.IntegerField()
    identifier = models.CharField(max_length=100)

from django.db import models

class Route(models.Model):
    dataset = models.FileField(upload_to='datasets/')
    num_vehicles = models.IntegerField()
    vehicle_capacity = models.IntegerField()
    algorithm = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)
    total_distance = models.FloatField(default=0)
    distances_per_vehicle = models.TextField(default='[]')
    routes = models.TextField(default='[]')
    depot_location = models.TextField(default='(0.0, 0.0)')


