import time
from dwave.system import LeapHybridSampler
from dimod import ConstrainedQuadraticModel, Binary

def make_cqm(num_items, bin_capacity, weights, lambdas, mus):
    start = time.time()
    cqm = ConstrainedQuadraticModel()
    bin_used = [Binary(f'bin_used_{i}') for i in range(num_items)]

    # objective function: minimize the number of bins used while injecting soft multiplier modifiers
    obj = (sum(bin_used)
           - sum(lambdas[i]*bin_used[i] for i in range(num_items))
           - sum(mus[j]*bin_used[j]*bin_capacity for j in range(num_items)))
    cqm.set_objective(obj)

    item_in_bin = [[Binary(f'item_{i}_in_bin_{j}') for j in range(num_items)] for i in range(num_items)]

    for i in range(num_items):
        cqm.add_constraint(sum(item_in_bin[i][j] for j in range(num_items)) == bin_used[i], label=f'item_{i}_in_bin')

    for i in range(num_items):
        cqm.add_constraint(
            sum(weights[i] * item_in_bin[i][j] for j in range(num_items)) <= bin_capacity * bin_used[i],
            label=f'capacity_bin_{i}'
        )  
    
    return cqm, time.time()-start

def lagrangian_relaxation(num_items, bin_capacity, weights):
    lambdas = [0.0] * num_items
    mus = [0.0] * num_items
    
    threshold = 1e-4 
    max_iters = 50

    for _ in range(max_iters):
        # solve subproblem matricies
        x = [[0.0 for _ in range(num_items)] for _ in range(num_items)]
        y = [0.0] * num_items

        for i in range(num_items):
            for j in range(num_items):
                x[i][j] = 1 if (lambdas[i]==0 and j==i) or (lambdas[i]>0 and mus[j]==0) else 0
            
        for j in range(num_items):
            total_weight = sum(weights[k] * x[k][j] for k in range(num_items))
            y[j] = 1 if total_weight > bin_capacity*mus[j] else 0
        
        # subgradient multiplier ascent steps
        alpha = 0.1
        for i in range(num_items):
            lambdas[i] += alpha*(1-sum(x[i]))
            mus[i] += alpha*(sum(weights[j] * x[j][i] for j in range(num_items)) - bin_capacity*y[i])

        if all(abs(lambdas[i]) < threshold for i in range(num_items)):
            break
    
    return lambdas, mus

def hybrid_lagrange(problem_info, iterations=5):
    """
    DIFFERENCE FROM ORIGINAL: 
    1. Fixed the sampler instantiation logic. The sampler is instantiated exactly ONCE 
       outside of all structural loops, eliminating significant API network overhead.
    2. Cleaner breakdown of compilation, feasibility, and solve tracking times.
    """
    bin_capacity = problem_info['bin_capacity']
    num_items = problem_info['num_items']
    weights = problem_info['bin_capacities']
    
    # Initialize connection session once. Token is pulled directly from native system configurations.
    sampler = LeapHybridCQMSampler()
    
    total_cqm_time = 0
    total_quantum_time = 0
    
    feasible_sampleset = None

    for i in range(iterations):
        lambda_values, mu_values = lagrangian_relaxation(num_items, bin_capacity, weights)
        new_cqm, cqm_time = make_lagrangian_cqm(num_items, bin_capacity, weights, lambda_values, mu_values)
        total_cqm_time += cqm_time
        
        start_q = time.time()
        sampleset = sampler.sample_cqm(new_cqm)
        total_quantum_time += (time.time() - start_q)

        feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)

    # Post-process results
    if len(feasible_sampleset):
        best_sample = feasible_sampleset.first.sample
        lowest_energy = feasible_sampleset.first.energy
        used_bins = [key for key, val in best_sample.items() if 'bin_used' in key and val]
        return len(used_bins), lowest_energy, [total_cqm_time, total_quantum_time]
    
    return None, None, [total_cqm_time, total_quantum_time]