# utils.py

import pandas as pd
import folium
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from .algorithms import run_ilp, run_lns, tabu_search

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def process_data(file):
    df = pd.read_csv(file)

    depot = df[df['id'] == 0].iloc[0]
    customers = df[df['id'] != 0]

    depot_location = (depot['latitude'], depot['longitude'])
    customer_locations = customers[['latitude', 'longitude']].values
    customer_demands = customers['demand_bags'].values + customers['demand_packs'].values

    num_customers = len(customer_locations)
    distance_matrix = np.zeros((num_customers + 1, num_customers + 1))

    locations = np.vstack([depot_location, customer_locations])

    for i in range(num_customers + 1):
        for j in range(num_customers + 1):
            if i != j:
                distance_matrix[i, j] = haversine(locations[i][0], locations[i][1], locations[j][0], locations[j][1])

    return {
        'depot_location': depot_location,
        'customer_locations': customer_locations,
        'customer_demands': customer_demands,
        'distance_matrix': distance_matrix,
    }

def run_optimization(data, num_vehicles, vehicle_capacity, algorithm):
    if algorithm == 'ILP':
        return run_ilp(data, num_vehicles, vehicle_capacity)
    elif algorithm == 'LNS':
        return run_lns(data, num_vehicles, vehicle_capacity)
    elif algorithm == 'Tabu':
        return tabu_search(data, num_vehicles, vehicle_capacity)
    else:
        raise ValueError("Unsupported algorithm selected.")

def generate_folium_map(routes, depot_location):
    folium_map = folium.Map(location=depot_location, zoom_start=12)

    colors = ['red', 'blue', 'green', 'purple', 'orange']

    folium.Marker(
        location=depot_location,
        icon=folium.Icon(icon='industry', prefix='fa'),
        popup='Depot'
    ).add_to(folium_map)

    for i, route in enumerate(routes):
        color = colors[i % len(colors)]
        folium.PolyLine(route, color=color, weight=2.5, opacity=1).add_to(folium_map)

        for coord in route[1:]:
            folium.Marker(
                location=coord,
                icon=folium.Icon(color=color, icon='info-sign'),
                popup=f'Customer Location {coord}'
            ).add_to(folium_map)

    return folium_map