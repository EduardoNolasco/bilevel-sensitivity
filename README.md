# Sensitivity-based bilevel optimisation (v1.0)

Python code for the numerical results of Nolasco, E., King, R. D. and Vassiliadis, V. S. (2026), *A Sensitivity-Based Method for Bilevel Optimization Problems: Theoretical Analysis and Computational Performance*, Computers & Chemical Engineering 213, 109709. https://doi.org/10.1016/j.compchemeng.2026.109709

It implements an augmented Lagrangian method that uses lower-level KKT sensitivities, and runs it on 15 BOLIB benchmark problems from several starting points.

## Usage 
pip install -r requirements.txt

python BLP_main.py # runs all problems and starting points

python results_aggregator.py # writes summary_results.csv


Start from an empty `results/` folder. Tested with Python 3.11, CasADi 3.6.3, NumPy 1.26.0, SciPy 1.11.3, Matplotlib 3.8.1 and pandas 2.2.3 on Windows.

## Notes

- Version 1.0 uses finite-difference gradients for the inner solves, not the adjoint formulation (3.9)-(3.10) of the paper.
- The status "Success (stalled)" means that the stall test ended the run, not the KKT tolerance; it does not certify optimality.

## Results

A full run (`reference_results/rerun_2026-10-09.csv`) reaches the objective value reported in Table 1 for 12 of the 13 problems. The exception is `AiyoshiShimizu1984Ex2` from (20, 20), which ends at F = 1.9e-4 instead of 5.

## Licence

MIT, see `LICENSE`.
