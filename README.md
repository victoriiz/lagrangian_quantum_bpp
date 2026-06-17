Markdown
# Hybrid Quantum-Classical Metaheuristic Optimization for the Bin Packing Problem

This repository implements a **Hybrid Quantum-Classical Metaheuristic Framework** using **Lagrangian Relaxation** and **Augmented Lagrangian Methods (ALM)** to solve the NP-hard 1D Bin Packing Problem (BPP). 

By relaxing hard inequalities into dynamic, mathematically updated soft penalties within the objective function, this framework eliminates the need for binary slack variables—drastically reducing the qubit footprint required to solve heavily constrained combinatorial optimization problems on near-term quantum devices.

Algorithms tested:
- Best fit
- Simulated annealing
- Simulated annealing with the augmented Lagrangian method (ALM)
- Hybrid quantum/classical algorithm: quantum annealing with Lagrangian relaxation and classical subroutine

