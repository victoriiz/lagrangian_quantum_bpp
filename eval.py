import os
import time
import numpy as np
import matplotlib.pyplot as plt
from src.baselines import best_fit
from src.classical_sa import simul_annealing
from src.quantum_cqm import hybrid_lagrange

def run_benchmarks():
    sizes = [10, 20, 40, 50, 100]
    
    times_bf, times_sa, times_alm, times_quantum = [], [], [], []
    bins_bf, bins_sa, bins_alm, bins_quantum = [], [], [], []

    print("====================================================")
    print("     RUNNING SYSTEM SCALABILITY EVALUATION          ")
    print("====================================================\n")

    for size in sizes:
        print(f"Evaluating problem size: {size} items...")
        
        np.random.seed(42) 
        mock_capacities = np.random.randint(20, 100, size=size).tolist()
        
        bench_problem = {
            'id': f"Scale-Test-{size}",
            'bin_capacity': 150,
            'num_items': size,
            'num_bins': size,  
            'bin_capacities': mock_capacities
        }

        # 1. Benchmark Best-Fit Heuristic
        start = time.time()
        bf_res = best_fit(bench_problem)
        times_bf.append(time.time() - start)
        bins_bf.append(bf_res)

        # 2. Benchmark Standard Simulated Annealing
        start = time.time()
        _, _, sa_res, _, _ = simul_annealing(bench_problem, alm=False, num_iter=200)
        times_sa.append(time.time() - start)
        bins_sa.append(sa_res)

        # 3. Benchmark Augmented Lagrangian Simulated Annealing
        start = time.time()
        _, _, alm_res, _, _ = simul_annealing(bench_problem, alm=True, num_iter=200)
        times_alm.append(time.time() - start)
        bins_alm.append(alm_res)

        # 4. Benchmark Hybrid Lagrangian Quantum CQM Pipeline
        try:
            q_res, _, q_times = hybrid_lagrange(bench_problem, iterations=2)
            times_quantum.append(sum(q_times))
            bins_quantum.append(q_res if q_res is not None else size)
        except Exception:
            # Fallback if D-Wave API credentials are not active during the test run
            times_quantum.append(0)
            bins_quantum.append(0)

    os.makedirs('outputs', exist_ok=True)

    with open('outputs/benchmark_results.txt', 'w') as f:
        f.write("Problem_Size,BF_Bins,SA_Bins,ALM_Bins,Quantum_Bins,BF_Time,SA_Time,ALM_Time,Quantum_Time\n")
        for i, size in enumerate(sizes):
            f.write(f"{size},{bins_bf[i]},{bins_sa[i]},{bins_alm[i]},{bins_quantum[i]},"
                    f"{times_bf[i]:.4f},{times_sa[i]:.4f},{times_alm[i]:.4f},{times_quantum[i]:.4f}\n")
    print("\n[SUCCESS] Raw metrics saved to 'outputs/benchmark_results.txt'")

    generate_plots(sizes, times_bf, times_sa, times_alm, times_quantum, bins_bf, bins_sa, bins_alm, bins_quantum)

def generate_plots(sizes, t_bf, t_sa, t_alm, t_q, b_bf, b_sa, b_alm, b_q):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Algorithmic Runtime Execution Scalability (Time-to-Solution)
    ax1.plot(sizes, t_bf, label='Best-Fit Heuristic', marker='o', color='lightsteelblue', linestyle=':')
    ax1.plot(sizes, t_sa, label='Standard SA', marker='s', color='goldenrod')
    ax1.plot(sizes, t_alm, label='ALM-Enhanced SA', marker='^', color='darkorange')
    if any(t_q):  # Only plot quantum runtime if it successfully ran
        ax1.plot(sizes, t_q, label='Hybrid Lagrangian (Total)', marker='X', color='cornflowerblue')
    
    ax1.set_xlabel('Problem Size (Number of Items)')
    ax1.set_ylabel('Execution Runtime (Seconds)')
    ax1.set_title('Computational Runtime Complexity vs. Scale')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Plot 2: Solution Optimization Quality (Bin Packing Efficiency)
    bar_width = 0.2
    index = np.arange(len(sizes))

    ax2.bar(index - 1.5*bar_width, b_bf, width=bar_width, label='Best-Fit', color='lightstandalone', alpha=0.5)
    ax2.bar(index - 0.5*bar_width, b_sa, width=bar_width, label='Standard SA', color='goldenrod')
    ax2.bar(index + 0.5*bar_width, b_alm, width=bar_width, label='ALM SA', color='darkorange')
    if any(b_q):
        ax2.bar(index + 1.5*bar_width, b_q, width=bar_width, label='Hybrid Quantum', color='cornflowerblue')

    ax2.set_xlabel('Problem Size (Number of Items)')
    ax2.set_ylabel('Optimal Bin Count Solution')
    ax2.set_title('Optimization Accuracy & Solution Comparison')
    ax2.set_xticks(index)
    ax2.set_xticklabels(sizes)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax2.legend()

    plt.tight_layout()
    
    plt.savefig('outputs/experimental_performance_charts.pdf', format='pdf', dpi=300)
    print("[SUCCESS] Publication-ready chart saved to 'outputs/experimental_performance_charts.pdf'")
    plt.close()

if __name__ == "__main__":
    run_benchmarks()