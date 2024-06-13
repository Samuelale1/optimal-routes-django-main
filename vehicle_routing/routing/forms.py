from django import forms
from .models import Route

class RouteForm(forms.Form):
    dataset = forms.FileField(label='Upload Dataset')
    num_vehicles = forms.IntegerField(label='Number of Vehicles')
    vehicle_capacity = forms.IntegerField(label='Vehicle Capacity')
    ALGORITHM_CHOICES = [
        ('ILP', 'Integer Linear Programming'),
        ('LNS', 'Large Neighborhood Search'),
        ('Tabu', 'Tabu Search'),
    ]
    algorithm = forms.ChoiceField(choices=ALGORITHM_CHOICES, label='Choose Algorithm')