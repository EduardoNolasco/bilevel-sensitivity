<!-- DRAFT: review every statement, especially the section "Notes on correspondence with the paper", before publishing. -->

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
- **Stopping.** The KKT tolerance is 1e-5. The stall test requires ten consecutive outer iterations with ||x_k - x_{k-1}||_2 < 1e-7 and |F_k - F_{k-1}| < 1e-7. The L-BFGS-B tolerance is `gtol` = 1e-5. Most reported iteration counts (11 or 12) reflect this ten-iteration window.
- **Penalty update.** rho is multiplied by 2 when the feasibility violation exceeds 0.5 times the previous one, capped at 1e6 (rho_0 = 10, mu_0 = 0).
- **Regularisation.** The quadratic term 1e-6 * ||y||^2 is added to the lower-level objective for the problems flagged `is_ll_linear`: `Bard_1991_ex1`, `Bard_Book_1998`, `DempeEtal2012`, `DempeFranke2011Ex41`, `Dempe_Franke_2011_ex42`, `Dempe_Lohse_2011_ex31a`, `Dempe_Lohse_2011_ex31b`.
- **Problems not in Table 1.** `Dempe1992a` and `DempeFranke2011Ex41` did not reach the literature value from any starting point and are not reported.
- **Line search.** SciPy's built-in L-BFGS-B line search is used; its constants cannot be set through the interface used here.

## Reference results

[Add `summary_results.csv` / the labelled Excel file here, with the date of the run.]

## Licence

[Choose a licence, e.g. MIT, and add a `LICENSE` file.]
