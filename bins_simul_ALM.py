import random
import time
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def initial_solution(num_items, num_bins):
    return [random.randint(0, num_bins - 1) for _ in range(num_items)]

def cost(solution, bin_capacities, bin_capacity, penalty_factor, lagrange_multiplier):
    # calculate the cost of the solution with the penalty term
    bin_weights = [0] * len(bin_capacities)
    for item, bin_index in enumerate(solution):
        bin_weights[bin_index] += bin_capacities[item]

    excess_weights = [max(0, weight - bin_capacity) for weight in bin_weights]
    penalty_term = sum([max(0, weight) for weight in excess_weights])

    # spply the penalty term and the Lagrange multiplier
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

def simulated_annealing(problem_info, initial_temperature=1000, cooling_rate=0.95, num_iterations=1000,
                        penalty_factor=0.5, initial_lagrange_multiplier=0, rho=1.1):
    bin_capacity = problem_info['bin_capacity']
    num_items = problem_info['num_items']
    num_bins = problem_info['num_bins']
    bin_capacities = problem_info['bin_capacities']

    # initialize variables
    current_solution = initial_solution(num_items, num_bins)
    current_cost = cost(current_solution, bin_capacities, bin_capacity, penalty_factor, initial_lagrange_multiplier)
    best_solution = current_solution
    best_cost = current_cost
    temperature = initial_temperature
    lagrange_multiplier = initial_lagrange_multiplier

    # main loop
    for iteration in range(num_iterations):
        if check_kkt_conditions(current_solution, bin_capacities, bin_capacity, lagrange_multiplier, tolerance=1e-6):
            break
        # generate the neighboring sol.
        new_solution = neighbor(current_solution, num_bins)
        new_cost = cost(new_solution, bin_capacities, bin_capacity, penalty_factor, lagrange_multiplier)

        # choose whether to accept this new sol.
        if random.random() < acceptance_probability(current_cost, new_cost, temperature):
            current_solution = new_solution
            current_cost = new_cost

        # ...and update if needed
        if new_cost < best_cost:
            best_solution = new_solution
            best_cost = new_cost

        # update Lagrange multiplier thru penalty
        lagrange_multiplier += rho * penalty_factor * sum([max(0, weight - bin_capacity) for weight in bin_capacities])

        # cool down
        temperature *= cooling_rate

    unique_bins = len(set(best_solution))

    return best_solution, best_cost, unique_bins

def check_kkt_conditions(solution, bin_capacities, bin_capacity, lagrange_multiplier, tolerance):
    # Calculate relevant quantities
    bin_weights = [0] * len(bin_capacities)
    for item, bin_index in enumerate(solution):
        bin_weights[bin_index] += bin_capacities[item]

    excess_weights = [max(0, weight - bin_capacity) for weight in bin_weights]
    
    # 1. Primal feasibility: Check if the solution is feasible
    primal_feasibility = all(weight <= bin_capacity for weight in bin_weights)
    if not primal_feasibility: return False
    
    # 2. Dual feasibility: Check if Lagrange multipliers are non-negative
    dual_feasibility = all(lagrange_multiplier >= 0 for _ in bin_weights)
    if not dual_feasibility: return False

    # 3. Complementary slackness: Check if complementary slackness conditions hold
    complementary_slackness = all(
        (weight == 0 and lagrange_multiplier > 0) or (weight > 0 and lagrange_multiplier == 0)
        for weight in excess_weights
    )
    if not complementary_slackness: return False
    
    # 4. Gradient of Lagrangian: Check if the gradient is close to zero
    lagrangian_gradient = sum([max(0, weight - bin_capacity) for weight in bin_capacities])
    gradient_close_to_zero = abs(lagrangian_gradient) < tolerance
    if not gradient_close_to_zero: return False
    
    # Check all conditions and return True if they are satisfied
    return True

def check_constraints(solution, bin_capacities):
    """
    Check if all constraints are satisfied for a given solution.

    Parameters:
    - solution (list): A list representing the bins assigned to each item.
    - bin_capacities (list): A list of the original capacities of each bin.

    Returns:
    - valid (bool): True if all constraints are satisfied, False otherwise.
    """
    num_bins = len(bin_capacities)
    
    # number of bins in the solution matches the original number of bins
    #if len(set(solution)) != num_bins:
       # return False
    
    # each item is assigned to a valid bin
    for item, bin_index in enumerate(solution):
        if bin_index < 0 or bin_index >= num_bins:
            return False
    
    # total weight in each bin is less than or equal to its capacity
    bin_weights = [0] * num_bins
    for item, bin_index in enumerate(solution):
        bin_weights[bin_index] += bin_capacities[item]
    
    for bin_index, weight in enumerate(bin_weights):
        if weight > bin_capacities[bin_index]:
            return False


# Example usage:
problem_info = {
    'id': "120",
    'bin_capacity': 150,
    'num_items': 120,
    'num_bins': 49,
    'bin_capacities': [97, 57, 81, 62, 75, 81, 23, 43, 50, 38, 60, 58, 70, 88, 36, 90, 37, 45, 45, 39, 44, 53, 70, 24, 82, 81, 47, 97, 35, 65, 74, 68, 49, 55, 52, 94, 95, 29, 99, 20, 22, 25, 49, 46, 98, 59, 98, 60, 23, 72, 33, 98, 80, 95, 78, 57, 67, 53, 47, 53, 36, 38, 92, 30, 80, 32, 97, 39, 80, 72, 55, 41, 60, 67, 53, 65, 95, 20, 66, 78, 98, 47, 100, 85, 53, 53, 67, 27, 22, 61, 43, 52, 76,64, 61, 29, 30, 46, 79, 66, 27, 79, 98, 90, 22, 75, 57, 67, 36, 70, 99, 48, 43, 45, 71, 100, 88, 48, 27, 39]
}

start = time.time()
best_solution, best_cost, num_bins_used = simulated_annealing(problem_info)
end = time.time()
#check = check_constraints(best_solution, problem_info['bin_capacities'])
#print('Check: ', check)
print("Best Solution:", best_solution)
print("Best Cost:", best_cost)
print("Number of Bins Used:", num_bins_used)
print(f"Time used: {end - start:.3g} s")

def read_bin_packing_file(file_path):
    with open(file_path, 'r') as file:
        num_problems = int(file.readline().strip())

        problems = []
        for _ in range(num_problems): 
            problem_id = file.readline().strip()
            #bin capacity, number of items, and number of bins in best sol.
            bin_capacity, num_items, num_bins = map(int, file.readline().split())

            # individual bin capacities (len = n)
            bin_capacities = [int(file.readline().strip()) for _ in range(num_items)]

            problem_info = {
                'id': problem_id,
                'bin_capacity': bin_capacity,
                'num_items': num_items,
                'num_bins': num_bins,
                'bin_capacities': bin_capacities
            }

            problems.append(problem_info)
    return problems

def main():
    total_times = {}
    for i in range(1,5):
        file_path = 'binpack'+str(i)+'.txt'
        problems = read_bin_packing_file(file_path)

        start_all = time.time()
        for ind, problem_info in enumerate(problems):
            size = problem_info['num_items']
            start_prob = time.time()
            _, _, bins_used = simulated_annealing(problem_info)
            end_prob = time.time()
            runtime = end_prob - start_prob
            if size not in total_times:
                total_times[size] = runtime
            else:
                prev = total_times[size]
                total_times[size] = (prev+runtime)/2

            print(f"{ind}: {bins_used == problem_info['num_bins']}, Time used: {end_prob - start_prob:.3g} s, number of items: {problem_info['num_items']}, bins: {bins_used}")

        total_t = time.time() - start_all
        print(f"Total time: {total_t:.3g}s")

    items = list(total_times.keys())
    runtimes = list(total_times.values())

    # Create a bar plot
    plt.bar(items, runtimes, color='blue', alpha=0.7, label='Runtime')

    # Calculate the regression line
    slope, intercept, _, _, _ = linregress(items, runtimes)
    regression_line = [slope * item + intercept for item in items]

    # Plot the regression line
    plt.plot(items, regression_line, color='red', label='Regression Line')

    plt.xlabel('Number of Items')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime vs Number of Items with Regression Line')
    plt.legend()
    plt.show()
 
#main()
