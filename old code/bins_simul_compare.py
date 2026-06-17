import random
import time
import math
import numpy as np
import matplotlib.pyplot as plt

def initial_solution(num_items, num_bins):
    # Generate an initial random solution
    return [random.randint(0, num_bins - 1) for _ in range(num_items)]

def cost(solution, bin_capacities, bin_capacity):
    # Calculate the cost of the solution
    bin_weights = [0] * len(bin_capacities)
    for item, bin_index in enumerate(solution):
        bin_weights[bin_index] += bin_capacities[item]

    excess_weights = [max(0, weight - bin_capacity) for weight in bin_weights]
    total_cost = sum(excess_weights)
    return total_cost

def alm_cost(solution, bin_capacities, bin_capacity, penalty_factor, lagrange_multiplier):
    # Calculate the cost of the solution with the ALM-like penalty term
    bin_weights = [0] * len(bin_capacities)
    for item, bin_index in enumerate(solution):
        bin_weights[bin_index] += bin_capacities[item]

    excess_weights = [max(0, weight - bin_capacity) for weight in bin_weights]
    penalty_term = sum([max(0, weight) for weight in excess_weights])

    # Apply the penalty term and the Lagrange multiplier
    total_cost = sum(excess_weights) + penalty_factor * penalty_term + lagrange_multiplier * penalty_term

    return total_cost

def neighbor(solution, num_bins):
    # Generate a neighboring solution by moving an item to a different bin
    new_solution = solution.copy()
    item_to_move = random.randint(0, len(solution) - 1)
    new_bin = random.randint(0, num_bins - 1)

    while new_solution[item_to_move] == new_bin:
        new_bin = random.randint(0, num_bins - 1)

    new_solution[item_to_move] = new_bin
    return new_solution

def acceptance_probability(old_cost, new_cost, temperature):
    # Calculate the acceptance probability based on the cost difference and temperature
    if new_cost < old_cost:
        return 1.0
    return math.exp((old_cost - new_cost) / temperature)

def simulated_annealing(problem_info, alm=False, initial_temperature=1000, cooling_rate=0.85, num_iterations=100,
                        penalty_factor=0.5, initial_lagrange_multiplier=0.1, rho=1.1, track_metrics=False):
    label = ""
    if alm: label="Simulated Annealing with ALM" 
    else: label="Simulated Annealing"
    # Extract information from the problem_info dictionary
    bin_capacity = problem_info['bin_capacity']
    num_items = problem_info['num_items']
    num_bins = problem_info['num_bins']
    bin_capacities = problem_info['bin_capacities']

    # Initialize the current solution and its cost
    current_solution = initial_solution(num_items, num_bins)
    if alm:
        current_cost = alm_cost(current_solution, bin_capacities, bin_capacity, penalty_factor, initial_lagrange_multiplier)
    else:
        current_cost = cost(current_solution, bin_capacities, bin_capacity)

    # Initialize the best solution and its cost
    best_solution = current_solution
    best_cost = current_cost

    # Initialize the temperature
    temperature = initial_temperature

    # Initialize the lagrange multiplier
    lagrange_multiplier = initial_lagrange_multiplier

    #lists to track data
    cost_evolution = []
    temperature_evolution = []
    
    # Simulated Annealing main loop
    for iteration in range(num_iterations):
        # Generate a neighboring solution
        new_solution = neighbor(current_solution, num_bins)
        if alm:
            new_cost = alm_cost(new_solution, bin_capacities, bin_capacity, penalty_factor, lagrange_multiplier)
        else:
            new_cost = cost(new_solution, bin_capacities, bin_capacity)

        # Decide whether to accept the new solution
        if random.random() < acceptance_probability(current_cost, new_cost, temperature):
            current_solution = new_solution
            current_cost = new_cost

        # Update the best solution if needed
        if new_cost < best_cost:
            best_solution = new_solution
            best_cost = new_cost

        # Update the Lagrange multiplier using the penalty method
        if alm:
            lagrange_multiplier += rho * penalty_factor * sum([max(0, weight - bin_capacity) for weight in bin_capacities])

        # Cool down the temperature
        temperature *= cooling_rate

        #update data
        if track_metrics:
            cost_evolution.append(current_cost)
            temperature_evolution.append(temperature)
            
    # Count the number of unique bins in the best solution
    unique_bins = len(set(best_solution))

    #update convergence data
    #if track_metrics:
        #plot_metrics(cost_evolution, temperature_evolution, label)
        
    return best_solution, best_cost, unique_bins, cost_evolution, temperature_evolution

def plot_metrics(cost_evolution_sa, temperature_evolution_sa, cost_evolution_alm=None, temperature_evolution_alm=None):
    cost_derivative_sa = np.gradient(cost_evolution_sa); cost_derivative_alm = None
    
    # Plot cost evolution
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(cost_evolution_sa, label='Cost Evolution (Simulated Annealing)', color='goldenrod')
    
    if cost_evolution_alm is not None:
        plt.plot(cost_evolution_alm, label='Cost Evolution (Simulated Annealing with ALM)', linestyle='--', color = 'cornflowerblue')

    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Evolution Over Iterations')
    plt.legend()

    #Plot rate of descent of cost (derivative)
    plt.subplot(1,2,2)
    plt.plot(cost_derivative_sa, label='Cost Derivative', color = 'goldenrod')
    
    if cost_evolution_alm is not None:
        cost_derivative_alm = np.gradient(cost_evolution_alm)
        plt.plot(cost_derivative_alm, label='Cost Derivative (Simulated Annealing with ALM)', linestyle='--', color = 'cornflowerblue')
    
    plt.xlabel('Iterations')
    plt.ylabel('Derivative')
    plt.title('Cost Derivative Over Iterations')
    plt.legend()   
    # Plot temperature evolution
    #plt.subplot(1, 2, 2)
    #plt.plot(temperature_evolution_sa, label='Temperature Evolution (Simulated Annealing)', color='blue')
    
    #if temperature_evolution_alm is not None:
        #plt.plot(temperature_evolution_alm, label='Temperature Evolution (Simulated Annealing with ALM)', linestyle='--', color='orange')

    #plt.xlabel('Iterations')
    #plt.ylabel('Temperature')
    #plt.title('Temperature Evolution Over Iterations')
    #plt.legend()

    plt.tight_layout()
    plt.show()
       
# Example usage:
problem_info = {
    'id': "120",
    'bin_capacity': 150,
    'num_items': 120,
    'num_bins': 49,
    'bin_capacities': [97, 57, 81, 62, 75, 81, 23, 43, 50, 38, 60, 58, 70, 88, 36, 90, 37, 45, 45, 39, 44, 53, 70, 24, 82, 81, 47, 97, 35, 65, 74, 68, 49, 55, 52, 94, 95, 29, 99, 20, 22, 25, 49, 46, 98, 59, 98, 60, 23, 72, 33, 98, 80, 95, 78, 57, 67, 53, 47, 53, 36, 38, 92, 30, 80, 32, 97, 39, 80, 72, 55, 41, 60, 67, 53, 65, 95, 20, 66, 78, 98, 47, 100, 85, 53, 53, 67, 27, 22, 61, 43, 52, 76, 64, 61, 29, 30, 46, 79, 66, 27, 79, 98, 90, 22, 75, 57, 67, 36, 70, 99, 48, 43, 45, 71, 100, 88, 48, 27, 39]
}

# Run both simulated annealing approaches
start_time_baseline = time.time()
best_solution_baseline, best_cost_baseline, num_bins_used_baseline, cost_sa, temp_sa = simulated_annealing(problem_info, track_metrics=True)
end_time_baseline = time.time()

start_time_alm = time.time()
best_solution_alm, best_cost_alm, num_bins_used_alm, cost_alm, temp_alm = simulated_annealing(problem_info, alm=True, track_metrics=True)
end_time_alm = time.time()

# Print results and performance comparison
print("Baseline Simulated Annealing:")
print("Best Solution:", best_solution_baseline)
print("Best Cost:", best_cost_baseline)
print("Number of Bins Used:", num_bins_used_baseline)
print("Time Taken:", end_time_baseline - start_time_baseline)

print("\nSimulated Annealing with ALM:")
print("Best Solution:", best_solution_alm)
print("Best Cost:", best_cost_alm)
print("Number of Bins Used:", num_bins_used_alm)
print("Time Taken:", end_time_alm - start_time_alm)

plot_metrics(cost_sa, temp_sa, cost_alm, temp_alm)

#######COMPARE SCALABILITY WITH BEST FIT HEURISTIC##########
import numpy as np

# Function for Best Fit algorithm
def best_fit(problem_info):
    bin_capacity = problem_info['bin_capacity']
    num_items = problem_info['num_items']
    bin_capacities = problem_info['bin_capacities']

    # Initialize bins with capacities
    bins = [{'capacity': bin_capacity, 'items': []} for _ in range(num_items)]

    # Iterate through items and assign to the bin with the least remaining capacity
    for item in range(num_items):
        best_bin_index = np.argmin([bin['capacity'] for bin in bins])
        bins[best_bin_index]['items'].append(item)
        bins[best_bin_index]['capacity'] -= bin_capacities[item]

    # Count the number of unique bins used
    unique_bins = len([bin for bin in bins if bin['items']])

    return bins, unique_bins

# Function to evaluate scalability
def evaluate_scalability(algorithm, problem_sizes):
    execution_times = []

    for size in problem_sizes:
        # Generate a larger problem instance
        large_problem_info = {
            'id': str(size),
            'bin_capacity': size * 2,
            'num_items': size,
            'num_bins': size // 2,
            'bin_capacities': [random.randint(1, size) for _ in range(size)]
        }

        start_time = time.time()
        if algorithm == 'BestFit':
            best_fit(large_problem_info)
        elif algorithm == 'SimulatedAnnealing':
            simulated_annealing(large_problem_info)
        elif algorithm == 'SimulatedAnnealingALM':
            simulated_annealing(large_problem_info, alm=True)
        else:
            raise ValueError("Invalid algorithm")

        end_time = time.time()
        execution_times.append(end_time - start_time)

    return execution_times

# scalability eval.
problem_sizes = [50, 100, 150, 200, 250, 500, 1000]
#problem_sizes = [10, 20, 40, 50, 120]
best_fit_times = evaluate_scalability('BestFit', problem_sizes)
sa_times = evaluate_scalability('SimulatedAnnealing', problem_sizes)
#sa_alm_times = evaluate_scalability('SimulatedAnnealingALM', problem_sizes)

# Plot scalability results
plt.figure(figsize=(8, 6))
plt.plot(problem_sizes, best_fit_times, label='Best Fit', color='lightsteelblue')
plt.plot(problem_sizes, sa_times, label='Simulated Annealing',color='goldenrod')
#plt.plot(problem_sizes, sa_alm_times, label='Simulated Annealing with ALM', color='cornflowerblue')

plt.xlabel('Problem Size (items)')
plt.ylabel('Execution Time (seconds)')
plt.title('Scalability Evaluation for 100 Iterations')
plt.legend()
plt.show()