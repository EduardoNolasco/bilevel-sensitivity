# Sensitivity-based bilevel optimisation (BLP): legacy code

Code archived as used for the numerical results of

> Nolasco, E., King, R. D., Vassiliadis, V. S. (2026). *A Sensitivity-Based Method for Bilevel Optimization Problems: Theoretical Analysis and Computational Performance.* Computers & Chemical Engineering, 213, 109709. https://doi.org/10.1016/j.compchemeng.2026.109709

This is the version written in August 2025 (last solver commit `dafb3a3`, 27 August 2025). The results file was produced on a Windows 10 workstation (Intel i5-6500, 16 GB RAM). It is kept unchanged in behaviour so that it matches the reported numbers; only comments and dead code were cleaned.

## Files

| File | Purpose |
|---|---|
| `bolib_class.py` | `BilevelProblem` class and the 15 benchmark problems (BOLIB), with starting points and known solutions |
| `ALM_solver.py` | `SensitivityALMSolver`: augmented Lagrangian method with lower-level sensitivities |
| `BLP_main.py` | Runs every problem from every starting point; saves JSON results and plots |
| `file_utils.py`, `plot_utils.py` | Saving and loading results; convergence plots |
| `results_aggregator.py` | Collects all `results/*.json` into `summary_results.csv` |
| `analyze_results.py` | Regenerates the plots for one saved JSON file |

## Usage

```
pip install -r requirements.txt
python BLP_main.py                           # all problems, all starting points
python results_aggregator.py                 # writes summary_results.csv
python analyze_results.py results/<file>.json
```

Start from an empty `results/` folder: the aggregator reads every JSON file in it. Selecting the reported start for each problem, the (n, m, r, s) counts and the final table layout were done by hand and are not scripted.

## Notes on correspondence with the paper

The following describes what this code does, where it differs from the description in the paper.

- **Gradient of the inner problem.** The augmented Lagrangian subproblem is minimised with SciPy's L-BFGS-B using finite-difference gradients; each function evaluation re-solves the lower-level problem with IPOPT. The adjoint formulation (3.9)-(3.10) is not implemented in this version.
- **Sensitivities.** `_compute_sensitivity` is a forward-mode implementation of the system (3.4) (one right-hand side per upper-level variable). It is used only for the stationarity part of the KKT residual in the stopping test. It adds a diagonal shift of 1e-5 to the matrix, defines the active set by g_i > -1e-6, and does not include bounds on y.

## Reproducing the reported results

Tested with Python 3.11.13, CasADi 3.6.3, NumPy 1.26.0, SciPy 1.11.3, Matplotlib 3.8.1 and pandas 2.2.3 on Windows.

The outcome of these non-convex multi-start problems can depend on the platform and on library builds. A full re-run on a different Windows machine with the versions above (`reference_results/rerun_2026-10-09.csv`) reproduced the final solution in 12 of the 13 runs reported in Table 1. The exception is `AiyoshiShimizu1984Ex2` from x0 = (20, 20), which ended at a different point (F = 1.9e-4 instead of 5).

For `Outrata_Cervinka_2009` the starting points in this repository are (-10, 1), (-10, -1) and (10, 1); the run reported in the paper is the one from (-10, 1).

## Licence

MIT, see `LICENSE`.
