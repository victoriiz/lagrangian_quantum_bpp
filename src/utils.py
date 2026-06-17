import numpy as np

def eval_violations(sol, item_weights, capacity):
    """
    Calculates total weight per bin and isolates exact capacity violations.
    """
    num_bins = max(sol) + 1 if sol else 0
    bin_weights = np.zeros(num_bins)
    for item_idx, bin_idx in enumerate(sol):
        bin_weights[bin_idx] += item_weights[item_idx]
    excess_wts = np.maximum(bin_weights-capacity, 0)
    total_excess = np.sum(excess_wts)
    return bin_weights, total_excess

def check_feasibility(sol, item_weights, capacity, stag_counter, max_patience=20):
    """
    Checks if a solution is feasible (i.e., no bin exceeds its capacity).
    """
    _, total_excess = eval_violations(sol, item_weights, capacity)
    if total_excess == 0 and stag_counter >= max_patience:
        return True
    return False