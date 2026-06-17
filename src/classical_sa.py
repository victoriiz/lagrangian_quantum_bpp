import random
import math
from src.utils import eval_violations, check_feasibility

def initial_sol(num_items, num_bins):
    return [random.randint(0, num_bins-1) for _ in range(num_items)]

def alm_cost(sol, bin_capacities, bin_cap, penalty, lagrange_mult):
    _, total_excess = eval_violations(sol, bin_capacities, bin_cap)
    return total_excess + (penalty*total_excess) * (lagrange_mult*total_excess)

def neighbor(sol, num_bins):
    new_sol = sol[:]
    item_to_move = random.randint(0, len(new_sol) - 1)
    new_bin = random.randint(0, num_bins-1)
    while new_sol[item_to_move] == new_bin:
        new_bin = random.randint(0, num_bins-1)
    new_sol[item_to_move] = new_bin
    return new_sol

def acceptance_probability(old_cost, new_cost, temperature):
    if new_cost < old_cost: return 1.0
    if temperature == 0.0: return 0.0
    return math.exp((old_cost - new_cost) / temperature)

def simul_annealing(problem_info, alm=False, initial_temp=100.0, cooling_rate=0.85,
                    num_iter=100, penalty=0.5, lagrange_mult=0.1, rho=1.1, track_metrics=False):
    bin_capacities = problem_info['bin_capacities']
    bin_cap = problem_info['bin_capacity']
    num_bins = problem_info['num_bins']
    num_items = problem_info['num_items']

    sol = initial_sol(num_items, num_bins)

    if alm: 
        curr_cost = alm_cost(sol, bin_capacities, bin_cap, penalty, lagrange_mult)
    else:
        _, total_excess = eval_violations(sol, bin_capacities, bin_cap)
        curr_cost = total_excess
    
    best_sol = sol
    best_cost = curr_cost
    temp = initial_temp
    lm = lagrange_mult

    costs = []
    temps = []
    stag_count = 0

    for i in range(num_iter):
        if alm and check_feasibility(sol, bin_capacities, bin_cap, stag_count):
            break
        new_sol = neighbor(sol, num_bins)
    
        if alm:
            new_cost = alm_cost(new_sol, bin_capacities, bin_cap, penalty, lm)
        else:
            _, total_excess = eval_violations(new_sol, bin_capacities, bin_cap)
            new_cost = total_excess
        
        if random.random() < acceptance_probability(curr_cost, new_cost, temp):
            sol = new_sol
            curr_cost = new_cost

        if new_cost < best_cost:
            best_sol = new_sol
            best_cost = new_cost
            stag_count = 0
        else:
            stag_count += 1
        
        if alm:
            _, active_excess = eval_violations(sol, bin_capacities, bin_cap)
            if active_excess > 0:
                lm += rho * penalty * active_excess

        temp *= cooling_rate

        if track_metrics:
            costs.append(curr_cost)
            temps.append(temp)
            
    unique_bins = len(set(best_sol))
    return best_sol, best_cost, unique_bins, costs, temps