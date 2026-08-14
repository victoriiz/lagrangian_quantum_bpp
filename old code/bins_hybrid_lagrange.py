from dwave.system import LeapHybridCQMSampler
from dimod import ConstrainedQuadraticModel, Binary

import matplotlib.pyplot as plt
import numpy as np
import time 

from scipy.optimize import minimize

bin_capacity = None
weights = None
#constructs cqm based on lagrangian multipliers
def make_lagrangian_cqm(num_items, bin_capacity, weights, lambda_values, mu_values):
    start = time.time()
    cqm = ConstrainedQuadraticModel()

    # Objective function
    bin_used = [Binary(f'bin_used_{j}') for j in range(num_items)]
    cqm.set_objective(sum(bin_used) - sum(lambda_values[i] * bin_used[i] for i in range(num_items)) - sum(mu_values[j] * bin_used[j] * bin_capacity for j in range(num_items)))
    
    # BIN CONSTRAINTS
    item_in_bin = [[Binary(f'item_{i}_in_bin_{j}') for j in range(num_items)] for i in range(num_items)]

    for i in range(num_items):
        # EQUALITY CONSTRAINT: Each item in exactly one bin
        one_bin_per_item = cqm.add_constraint(sum(item_in_bin[i]) == 1, label=f'item_placing_{i}')

    for j in range(num_items):
        # INEQUALITY CONSTRAINT: Bin capacity constraint
        bin_up_to_capacity = cqm.add_constraint(
            sum(weights[i] * item_in_bin[i][j] for i in range(num_items)) - bin_used[j] * bin_capacity <= 0,
            label=f'capacity_bin_{j}')

    end = time.time()
    total_time = end - start
    print(f"Time taken for creation of CQM: {total_time:.3g}")
    return cqm, total_time

def lagrangian_relaxation(num_items, bin_capacity, weights):
    # initialize Lagrange multipliers
    lambda_values = [0.0] * num_items
    mu_values = [0.0] * num_items

    # CONVERGENCE CRITERIA
    convergence_threshold = 1e-4
    max_iterations = 100
    iteration = 0

    while iteration < max_iterations:
        # solves Lagrangian subproblem: result used to update 
        x, y = solve_lagrangian_subproblem(num_items, bin_capacity, weights, lambda_values, mu_values)

        # update Lagrange multipliers; check convergence
        lambda_values = update_lambda(lambda_values, x)
        mu_values = update_mu(mu_values, x, y)

        if is_converged(lambda_values, mu_values, convergence_threshold):
            break

        iteration += 1

    return lambda_values, mu_values

def solve_lagrangian_subproblem(num_items, bin_capacity, weights, lambda_values, mu_values):
    #decision variables
    x = [[0] * num_items for _ in range(num_items)]
    y = [0] * num_items

    # solve for x_ij based on Lagrange multipliers and constraints
    for i in range(num_items):
        for j in range(num_items):
            # Decision variable x_ij
            x[i][j] = 1 if (lambda_values[i] == 0 and j == i) or (lambda_values[i] > 0 and mu_values[j] == 0) else 0

    # solve for y_j based on Lagrange multipliers and constraints
    for j in range(num_items):
        total_weight = sum(weights[i] * x[i][j] for i in range(num_items))
        y[j] = 1 if total_weight > bin_capacity * mu_values[j] else 0

    return x, y

# updates Lagrange multipliers associated with equality constraints
def update_lambda(lambda_values, x):  
    alpha = 0.1  # step size
    updated_lambda_values = [lambda_values[i] + alpha * (1 - sum(x[i])) for i in range(len(lambda_values))]
    return updated_lambda_values

# updates Lagrange multipliers associated with inequality constraints
def update_mu(mu_values, x, y):
    alpha = 0.1  # step size
    updated_mu_values = [mu_values[j] + alpha * (sum(weights[i] * x[i][j] for i in range(len(x))) - bin_capacity * y[j]) for j in range(len(mu_values))]
    return updated_mu_values

# checks convergence based on the change in Lagrange multipliers
def is_converged(lambda_values, mu_values, convergence_threshold): 
    #Note: an alternative approach here is to check KTT conditions 
    return all(abs(lambda_values[i]) < convergence_threshold for i in range(len(lambda_values))) and all(abs(mu_values[j]) < convergence_threshold for j in range(len(mu_values)))

#EXAMPLE FOR TESTING MULTIPLIERS COMMENTED BELOW
#num_items = 5
#bin_capacity = 10
#weights = [4, 3, 2, 5, 1]

#lambda_values, mu_values = lagrangian_relaxation(num_items, bin_capacity, weights)
#print("Optimal Lagrange Multipliers (lambda):", lambda_values)
#print("Optimal Lagrange Multipliers (mu):", mu_values)
#x,y = solve_lagrangian_subproblem(num_items, bin_capacity, weights, lambda_values, mu_values)
#print(f"Number of bins: {sum(y)}")

# classical optimization subroutine to find the best solution/process quantum result
def classical_optimization(feasible_sampleset):
    if len(feasible_sampleset):
        #best = feasible_sampleset.first
        solutions = feasible_sampleset.samples()
        energies = feasible_sampleset.data_vectors['energy']
        sorted_indices = sorted(range(len(energies)), key=lambda k: energies[k])
        sorted_solutions = [solutions[i] for i in sorted_indices]
        #print("{} feasible solutions of {}.".format(len(feasible_sampleset), len(sampleset)))
        best = sorted_solutions[0]; lowest_energy = energies[sorted_indices[0]]
        selected_bins = [key for key, val in best.items() if 'bin_used' in key and val]
        print("{} bins are used.".format(len(selected_bins)))
        return len(selected_bins), lowest_energy, selected_bins
    else:
        return None, None, None  # in the case that no feasible solution found

def classical_plot(feasible_sampleset):
    energies = [] #all energies recorded with each iteration of search
    num_bins = [] #all solutions found with each iteration of search
    best_bins = None #best solution found
    all_energies = feasible_sampleset.data_vectors['energy']
    for idx, sample in enumerate(feasible_sampleset.samples()):  
        energy = all_energies[idx]
        selected_bins = [key for key, val in sample.items() if 'bin_used' in key and val]
        num_bins_used = len(selected_bins)
        energies.append(energy); num_bins.append(num_bins_used)
        if not best_bins or num_bins_used<best_bins: best_bins = num_bins_used
    return energies, num_bins, best_bins
  
#hybrid Lagrange model, but will return total energies and num_bin solutions
def hybrid_lagrange_plot(problem_info, iterations=5):
    bin_capacity = problem_info['bin_capacity']
    num_items = problem_info['num_items']
    bin_capacities = problem_info['bin_capacities']
    weights = bin_capacities
    
    # Quantum subroutine with Lagrangian relaxation
    for i in range(iterations):
        # Find optimal lagrange multipliers based on current sol.
        lambda_values, mu_values = lagrangian_relaxation(num_items, bin_capacity, bin_capacities)

        # Construct a CQM with updated Lagrange multipliers
        new_cqm, _ = make_lagrangian_cqm(num_items, bin_capacity, bin_capacities, lambda_values, mu_values)
        
        # use quantum annealing to sample feasible solutions
        sampler = LeapHybridCQMSampler(token='')
        sampler.parameters['num_reads'] = 100
        sampler.parameters['num_spin_reversal_transforms'] = 100
        sampleset = sampler.sample_cqm(new_cqm)

        # feasibility filtering
        feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)

    # classical subroutine
    energies, num_bins, num_bins_used = classical_plot(feasible_sampleset)

    return energies, num_bins, num_bins_used

# building the hybrid lagrange model (equivalent to above), aiming to ultimately test for scalability by returning times used and bin solutions  
def hybrid_lagrange(problem_info, iterations = 5):
    global bin_capacity; global weights
    bin_capacity = problem_info['bin_capacity']
    num_items = problem_info['num_items']
    bin_capacities = problem_info['bin_capacities']
    weights=bin_capacities
    
    all_times = []
    total_cqm_time = 0
    total_quantum_time = 0
    total_feasibility_time = 0
    #note: classical time is only computed once, so it can be automatically added to all_times
    
    # Quantum subroutine with Lagrangian relaxation
    for i in range(iterations):
        # Find optimal lagrange multipliers based on current sol.
        lambda_values, mu_values = lagrangian_relaxation(num_items, bin_capacity, weights)

        # construct a CQM with updated Lagrange multipliers
        new_cqm, cqmtime = make_lagrangian_cqm(num_items, bin_capacity, bin_capacities, lambda_values, mu_values)
        total_cqm_time += cqmtime
        
        # use quantum annealing to sample feasible solutions
        sampler = LeapHybridCQMSampler(token='DEV-09aa362cd4b0d85aff63ab0546a90bfebaa8eae7')
        sampler.parameters['num_reads'] = 100
        sampler.parameters['num_spin_reversal_transforms'] = 100
        start = time.time()
        sampleset = sampler.sample_cqm(new_cqm)
        end = time.time()
        quant_time = end-start
        total_quantum_time += quant_time

        # feasibility filtering
        start = time.time()
        feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)
        end = time.time()
        feas_time = end-start
        total_feasibility_time += feas_time

    #append all the total times to all_times, in the order they were computed
    all_times = [total_cqm_time, total_quantum_time, total_feasibility_time]
    
    # classical subroutine
    start = time.time()
    best_bins, lowest_energy, selected_bins = classical_optimization(feasible_sampleset)
    end = time.time()
    class_time = end - start
    all_times.append(class_time)

    return best_bins, lowest_energy, all_times

def plot_convergence(energies, num_bins):
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(energies, label='Energy', color = 'cornflowerblue')
    plt.title('Energy Progression over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(num_bins, label='Number of Bins Used', color='lightsteelblue')
    plt.title('Number of Bins Used Progression over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Bins Used')
    plt.legend()

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

def read_bin_packing_file(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        num_problems = int(file.readline().strip())

        problems = []
        for _ in range(num_problems): 
            problem_id = file.readline().strip()
            #bin capacity, number of items, and number of bins in best sol.
            bin_capacity, num_items, num_bins = map(int, file.readline().split())

            # individual bin capacities (len = n)
            bin_capacities = [int(num) for num in file.readline().split()]

            problem_info = {
                'id': problem_id,
                'bin_capacity': bin_capacity,
                'num_items': num_items,
                'num_bins': num_bins,
                'bin_capacities': bin_capacities
            }

            problems.append(problem_info)
    return problems

weights = None; bin_capacity = None
def test_plot():
    global weights; global bin_capacity
    start = time.time()
    num_items = problem_info['num_items']; bin_capacity = problem_info['bin_capacity']; weights=problem_info['bin_capacities']
    energies, num_bins, num_bins_used = hybrid_lagrange_plot(problem_info)
    end =time.time()
    total_time = end-start
    print(f"Number of bins used: {num_bins_used}. Total time: {total_time:.3g}")
    plot_convergence(energies, num_bins)
    
def test_main():
    start = time.time()
    num_items = problem_info['num_items']; bin_capacity = problem_info['bin_capacity']; weights=problem_info['bin_capacities']
    lambda_values, mu_values = lagrangian_relaxation(num_items, bin_capacity, weights)
    x,y = solve_lagrangian_subproblem(num_items, bin_capacity, weights, lambda_values, mu_values)
    print(f"Sum of y for subproblem: {sum(y)}")
    num_bins_used, lowest_energy, all_times = hybrid_lagrange(problem_info, iterations = 5)
    end = time.time()
    total_time = end-start
    #print(x)
    print(f"Number of Bins Used: {num_bins_used}. Energy: {lowest_energy}.")
    print(f"Time used: {total_time:.3g}")   

def main():
    sizes = [10, 20, 40, 50, 120]
    total_times = []
    cqm_time = []
    quantum_time = []
    filter_time = []
    classical_time = []
    
    ans_bins = []
    num_bins = []
    energies_found = []
    problems = read_bin_packing_file("sample.txt")
    for problem in problems:
        print("problem:",problem)
        ans_bins.append(problem['num_bins'])
        num_items = problem['num_items']; bin_capacity = problem['bin_capacity']; weights=problem['bin_capacities']
         
        start = time.time()
        num_bins_used, energy, all_times = hybrid_lagrange(problem, iterations=5)
        time_taken = time.time()-start
        total_times.append(round(time_taken, 3))
        
        num_bins.append(num_bins_used); energies_found.append(energy)
        
        cqm, quant, filtering, classical = [round(time, 3) for time in all_times]
        cqm_time.append(cqm); quantum_time.append(quant); filter_time.append(filtering); classical_time.append(classical)
        
        print("Done")
    
    print("problem:", problem_info)
    ans_bins.append(problem_info['num_bins'])
    num_items = problem_info['num_items']; bin_capacity = problem_info['bin_capacity']; weights=problem_info['bin_capacities']
        
    start = time.time()
    num_bins_used, energy, all_times = hybrid_lagrange(problem_info, iterations=5)
    time_taken = time.time()-start
    total_times.append(round(time_taken, 3))
        
    num_bins.append(num_bins_used); energies_found.append(energy)
        
    cqm, quant, filtering, classical = [round(time, 3) for time in all_times]
    cqm_time.append(cqm); quantum_time.append(quant); filter_time.append(filtering); classical_time.append(classical)
      
    ####PLOTS######
    # Plot 1: Line plot for times across sizes
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)

    plt.plot(sizes, total_times, label='Total Time', marker='o')
    plt.plot(sizes, cqm_time, label='CQM Time', marker='o')
    plt.plot(sizes, quantum_time, label='Quantum Time', marker='o')
    plt.plot(sizes, filter_time, label='Filter Time', marker='o')
    plt.plot(sizes, classical_time, label='Classical Time', marker='o')

    plt.xlabel('Sizes')
    plt.ylabel('Time (seconds)')
    plt.title('Time Comparison Across Problem Sizes')
    plt.legend()

    # Plot 2: Bar graph for ans_bins and num_bins, and a regression line for energies_found
    plt.subplot(1, 2, 2)

    # Bar graph for ans_bins and num_bins
    bar_width = 0.35
    index = np.arange(len(sizes))

    plt.bar(index, ans_bins, width=bar_width, label='Known Optimal Solution', color='cyan',align='center')
    plt.bar(index + bar_width, num_bins, width=bar_width, label='Found Solution', color='orange',align='center')

    plt.xlabel('Sizes')
    plt.ylabel('Bins')
    plt.title('Bin Solution Comparison Across Problem Sizes')
    plt.xticks(index + bar_width / 2, sizes)
    plt.legend()

    print(f"Bin solutions: {num_bins}")
    print(f"Energies found: {energies_found}")
    
    plt.tight_layout()
    plt.show()
    
test_plot()
