# algorithms.py

from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import random
from collections import deque
import numpy as np

# ILP Algorithm
def run_ilp(data, num_vehicles, vehicle_capacity):
    distance_matrix = data['distance_matrix']
    customer_demands = data['customer_demands']
    num_customers = len(customer_demands)

    prob = LpProblem("Vehicle_Routing_Problem", LpMinimize)

    x = LpVariable.dicts("x", ((i, j, k) for i in range(num_customers + 1)
                                         for j in range(num_customers + 1)
                                         for k in range(num_vehicles)), cat='Binary')
    u = LpVariable.dicts("u", (i for i in range(1, num_customers + 1)), lowBound=0, cat='Integer')

    prob += lpSum(distance_matrix[i][j] * x[i, j, k]
                  for i in range(num_customers + 1)
                  for j in range(num_customers + 1)
                  for k in range(num_vehicles))

    for j in range(1, num_customers + 1):
        prob += lpSum(x[i, j, k] for i in range(num_customers + 1)
                                 for k in range(num_vehicles) if i != j) == 1

    for k in range(num_vehicles):
        prob += lpSum(x[0, j, k] for j in range(1, num_customers + 1)) == 1
        prob += lpSum(x[i, 0, k] for i in range(1, num_customers + 1)) == 1

    for k in range(num_vehicles):
        for i in range(1, num_customers + 1):
            for j in range(1, num_customers + 1):
                if i != j:
                    prob += u[i] - u[j] + (num_customers * x[i, j, k]) <= num_customers - 1

    for k in range(num_vehicles):
        prob += lpSum(customer_demands[j - 1] * x[i, j, k]
                      for i in range(num_customers + 1)
                      for j in range(1, num_customers + 1) if i != j) <= vehicle_capacity

    prob.solve()

    routes = [[] for _ in range(num_vehicles)]
    for k in range(num_vehicles):
        current_location = 0
        while True:
            for j in range(num_customers + 1):
                if value(x[current_location, j, k]) == 1:
                    routes[k].append(j)
                    current_location = j
                    break
            if current_location == 0:
                break

    total_distance = value(prob.objective)
    distances_per_vehicle = [lpSum(distance_matrix[routes[k][i - 1]][routes[k][i]] for i in range(1, len(routes[k]))) for k in range(num_vehicles)]

    return {
        'total_distance': total_distance,
        'distances_per_vehicle': distances_per_vehicle,
        'routes': [[(data['customer_locations'][i - 1][0], data['customer_locations'][i - 1][1]) if i != 0 else data['depot_location'] for i in route] for route in routes]
    }

# Tabu Search Algorithm
def get_initial_solution(demands, capacity, num_vehicles):
    routes = [[] for _ in range(num_vehicles)]
    remaining_customers = set(range(len(demands)))
    vehicle_loads = [0] * num_vehicles

    for customer in remaining_customers:
        for v in range(num_vehicles):
            if vehicle_loads[v] + demands[customer] <= capacity:
                routes[v].append(customer)
                vehicle_loads[v] += demands[customer]
                break

    return routes

def swap_2opt(solution, i, j):
    new_solution = solution[:]
    new_solution[i:j+1] = reversed(new_solution[i:j+1])
    return new_solution

def relocate_customer(solution, route_idx1, route_idx2, customer_index):
    new_solution = [route[:] for route in solution]
    customer = new_solution[route_idx1].pop(customer_index)
    new_solution[route_idx2].append(customer)
    return new_solution

def generate_neighborhood(solution, distance_matrix, tabu_list, best_solution):
    neighborhood = []
    for route_idx, route in enumerate(solution):
        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route)):
                neighbor = swap_2opt(solution, i, j)
                if neighbor not in tabu_list and total_distance(neighbor, distance_matrix) < total_distance(best_solution, distance_matrix):
                    neighborhood.append(neighbor)

    for route_idx1, route1 in enumerate(solution):
        for route_idx2 in range(len(solution)):
            if route_idx1 != route_idx2:
                for i in range(1, len(route1)):
                    neighbor = relocate_customer(solution, route_idx1, route_idx2, i)
                    if neighbor not in tabu_list:
                        neighborhood.append(neighbor)
    return neighborhood

def total_distance(solution, distance_matrix):
    total = 0
    for route in solution:
        route = [0] + route + [0]
        for i in range(len(route) - 1):
            total += distance_matrix[route[i]][route[i+1]]
    return total

def tabu_search(data, num_vehicles, vehicle_capacity, max_iterations=100, tabu_tenure=10):
    customer_demands = data['customer_demands']
    capacity = vehicle_capacity

    initial_solution = get_initial_solution(customer_demands, capacity, num_vehicles)
    distance_matrix = data['distance_matrix']

    current_solution = initial_solution
    best_solution = initial_solution
    tabu_list = deque(maxlen=tabu_tenure)

    for _ in range(max_iterations):
        neighborhood = generate_neighborhood(current_solution, distance_matrix, tabu_list, best_solution)
        best_neighbor = None
        best_neighbor_cost = float('inf')

        for neighbor in neighborhood:
            neighbor_cost = total_distance(neighbor, distance_matrix)
            if neighbor not in tabu_list and neighbor_cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = neighbor_cost
            elif neighbor in tabu_list and neighbor_cost < total_distance(best_solution, distance_matrix):
                best_neighbor = neighbor
                best_neighbor_cost = neighbor_cost

        if best_neighbor is not None:
            current_solution = best_neighbor
            if total_distance(current_solution, distance_matrix) < total_distance(best_solution, distance_matrix):
                best_solution = current_solution
            tabu_list.append(best_neighbor)

    routes = [[0] + route + [0] for route in best_solution]
    total_dist = total_distance(best_solution, distance_matrix)
    distances_per_vehicle = [total_distance([route], distance_matrix) for route in best_solution]

    return {
        'total_distance': total_dist,
        'distances_per_vehicle': distances_per_vehicle,
        'routes': [[(data['customer_locations'][i - 1][0], data['customer_locations'][i - 1][1]) if i != 0 else data['depot_location'] for i in route] for route in routes]
    }

# Large Neighborhood Search (LNS) Algorithm
def destroy_solution(solution, num_customers):
    removed_customers = []
    for _ in range(num_customers):
        route = random.choice(solution)
        if route:
            customer = route.pop(random.randint(0, len(route) - 1))
            removed_customers.append(customer)
    return removed_customers

def insert_customers(solution, removed_customers, distance_matrix):
    for customer in removed_customers:
        best_increase = float('inf')
        best_position = None
        best_route = None
        for route in solution:
            for i in range(len(route) + 1):
                new_route = route[:i] + [customer] + route[i:]
                new_distance = 0
                for j in range(len(new_route) - 1):
                    new_distance += distance_matrix[new_route[j]][new_route[j + 1]]
                new_distance += distance_matrix[0][new_route[0]] + distance_matrix[new_route[-1]][0]
                increase = new_distance - total_distance([route], distance_matrix)
                if increase < best_increase:
                    best_increase = increase
                    best_position = i
                    best_route = route
        best_route.insert(best_position, customer)
    return solution

def run_lns (data, num_vehicles, vehicle_capacity, num_iterations=100, destroy_rate=0.2):
    customer_demands = data['customer_demands']
    capacity = vehicle_capacity

    def get_initial_solution(demands, capacity, num_vehicles):
        routes = [[] for _ in range(num_vehicles)]
        remaining_customers = set(range(1, len(demands) + 1))
        vehicle_loads = [0] * num_vehicles

        for customer in remaining_customers:
            for v in range(num_vehicles):
                if vehicle_loads[v] + demands[customer - 1] <= capacity:
                    routes[v].append(customer)
                    vehicle_loads[v] += demands[customer - 1]
                    break

        return routes

    initial_solution = get_initial_solution(customer_demands, capacity, num_vehicles)
    distance_matrix = data['distance_matrix']

    best_solution = initial_solution
    best_cost = total_distance(best_solution, distance_matrix)

    for _ in range(num_iterations):
        current_solution = [route[:] for route in best_solution]
        num_customers_to_remove = int(destroy_rate * sum(len(route) for route in current_solution))
        removed_customers = destroy_solution(current_solution, num_customers_to_remove)
        current_solution = insert_customers(current_solution, removed_customers, distance_matrix)

        current_cost = total_distance(current_solution, distance_matrix)
        if current_cost < best_cost:
            best_solution = current_solution
            best_cost = current_cost

    routes = [[0] + route + [0] for route in best_solution]
    total_dist = total_distance(best_solution, distance_matrix)
    distances_per_vehicle = [total_distance([route], distance_matrix) for route in best_solution]

    return {
        'total_distance': total_dist,
        'distances_per_vehicle': distances_per_vehicle,
        'routes': [[(data['customer_locations'][i - 1][0], data['customer_locations'][i - 1][1]) if i != 0 else data['depot_location'] for i in route] for route in routes]
    }
