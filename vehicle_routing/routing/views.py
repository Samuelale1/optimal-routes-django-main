from django.shortcuts import render, redirect, get_object_or_404
from .forms import RouteForm
from .utils import process_data, generate_folium_map, run_optimization  
#from django.core.files.storage import FileSystemStorage
from .forms import RouteForm
from .models import Route
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .forms import RouteForm
from .models import Route
from django.shortcuts import render, redirect
from .utils import process_data, run_optimization, generate_folium_map
import json


def landing(request):
    return render(request, 'routing/landing.html')

def login_view(request):
    # Implement your login logic here
    return render(request, 'routing/login.html')

def register_view(request):
    # Implement your registration logic here
    return render(request, 'routing/register.html')

def dashboard(request):
    if request.method == 'POST':
        form = RouteForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.cleaned_data['dataset']
            num_vehicles = form.cleaned_data['num_vehicles']
            vehicle_capacity = form.cleaned_data['vehicle_capacity']
            algorithm = form.cleaned_data['algorithm']
            
            data = process_data(dataset)
            result = run_optimization(data, num_vehicles, vehicle_capacity, algorithm)
            
            route = Route.objects.create(
                dataset=dataset,
                num_vehicles=num_vehicles,
                vehicle_capacity=vehicle_capacity,
                algorithm=algorithm,
            )
            route.total_distance = result['total_distance']
            route.distances_per_vehicle = json.dumps(result['distances_per_vehicle'])
            route.routes = json.dumps(result['routes'])
            route.depot_location = json.dumps(data['depot_location'])
            route.save()
            
            return redirect('results', route_id=route.id)
    else:
        form = RouteForm()
    return render(request, 'routing/dashboard.html', {'form': form})

def results(request, route_id):
    route = get_object_or_404(Route, id=route_id)
    routes = json.loads(route.routes)
    folium_map = generate_folium_map(routes, json.loads(route.depot_location))
    map_html = folium_map._repr_html_()
    
    vehicle_distances = json.loads(route.distances_per_vehicle)
    vehicle_data = [(idx + 1, dist) for idx, dist in enumerate(vehicle_distances)]

    return render(request, 'routing/results.html', {
        'route': route,
        'map_html': map_html,
        'vehicle_data': vehicle_data,
    })