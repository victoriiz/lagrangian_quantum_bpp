import time
from src.baselines import best_fit
from src.classical_sa import simul_annealing
from src.quantum_cqm import hybrid_lagrange

problem_info = {
    'id': "Paper-Instance-120",
    'bin_capacity': 150,
    'num_items': 120,
    'num_bins': 49,
    'bin_capacities': [97, 57, 81, 62, 75, 81, 23, 43, 50, 38, 60, 58, 70, 88, 36, 90, 37, 45, 
                       45, 39, 44, 53, 70, 24, 82, 81, 47, 97, 35, 65, 74, 68, 49, 55, 52, 94, 
                       95, 29, 99, 20, 22, 25, 49, 46, 98, 59, 98, 60, 23, 72, 33, 98, 80, 95, 
                       78, 57, 67, 53, 47, 53, 36, 38, 92, 30, 80, 32, 97, 39, 80, 72, 55, 41, 
                       60, 67, 53, 65, 95, 20, 66, 78, 98, 47, 100, 85, 53, 53, 67, 27, 22, 61, 
                       43, 52, 76, 64, 61, 29, 30, 46, 79, 66, 27, 79, 98, 90, 22, 75, 57, 67, 
                       36, 70, 99, 48, 43, 45, 71, 100, 88, 48, 27, 39]
}

print("====================================================")
print("     STARTING METAHURISTIC EXPERIMENTAL RUNNERS     ")
print("====================================================\n")

# 1. Evaluate Best-Fit Baseline Heuristic
start = time.time()
bf_bins = best_fit(problem_info)
print(f"[BASELINE] Best Fit Bins Used: {bf_bins} | Time: {time.time() - start:.4f}s")

# 2. Evaluate Classical Simulated Annealing
start = time.time()
_, _, sa_bins, _, _ = simul_annealing(problem_info, alm=False, num_iter=500)
print(f"[CLASSICAL] Standard SA Bins Used: {sa_bins} | Time: {time.time() - start:.4f}s")

# 3. Evaluate Simulated Annealing with Augmented Lagrangian Methods (ALM)
start = time.time()
_, _, alm_bins, _, _ = simul_annealing(problem_info, alm=True, num_iter=500)
print(f"[CLASSICAL] ALM-Enhanced SA Bins Used: {alm_bins} | Time: {time.time() - start:.4f}s")

# 4. Evaluate Hybrid Lagrangian Quantum Pipeline
print("\n[QUANTUM] Connecting to D-Wave QPU endpoint...")
try:
    q_bins, energy, q_times = hybrid_lagrange(problem_info, iterations=3)
    print(f"[QUANTUM] Quantum Lagrangian Bins Used: {q_bins} (Objective Energy: {energy:.4f})")
    print(f"          CQM Formulation Time: {q_times[0]:.4f}s | QPU Hardware Sample Time: {q_times[1]:.4f}s")
except Exception as e:
    print(f"[QUANTUM ERROR] Connection or processing failure: {e}")
    print("                Verify your D-Wave environment configuration via 'dwave ping'.")

print("\n====================================================")
print("              EXPERIMENTATION COMPLETE              ")
print("====================================================")