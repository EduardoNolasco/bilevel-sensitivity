"""
Run the multi-start benchmark experiments (legacy script used for the numerical results of the paper).

For every problem returned by bolib_class.get_all_problems() and every starting point in its x0_list, the
sensitivity-based ALM solver is run. Each run is saved to results/*.json and its convergence plots to
results/plots/. Afterwards run results_aggregator.py to collect all JSON files into summary_results.csv.
"""
from bolib_class import get_all_problems
from ALM_solver import SensitivityALMSolver
from plot_utils import plot_convergence
from file_utils import save_results
import numpy as np
import os

def main():
    """
    Main script to run a batch of experiments for a list of problems.
    It iterates through each problem and each of its starting points,
    saving the results and plots for each individual run.
    """
    # Create results directories if they don't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('results/plots'):
        os.makedirs('results/plots')
        
    problem_factories = get_all_problems()

    for problem_func in problem_factories:
        problem = problem_func()
        print(f"\n\n{'='*20} Starting Experiment for Problem: {problem.name} {'='*20}")

        settings = {
            'll_print_level': 0,
            'tol': 1e-5,
            'tol_x': 1e-7, # Stall tolerance for x
            'tol_F': 1e-7, # Stall tolerance for F
            'rho_start': 10.0,
            'rho_max': 1e6,
            'max_outer_iter': 50,
            'inner_tol': 1e-5,
        }

        if not problem.x0_list:
            print(f"Warning: No starting points found for {problem.name}. Skipping.")
            continue

        for i, x0 in enumerate(problem.x0_list):
            print(f"\n--- Running Trial {i+1}/{len(problem.x0_list)} with x0 = {x0} ---")
            
            # Set the current starting point for this specific run
            problem.x0 = x0 
            
            solver = SensitivityALMSolver(problem, settings)
            solution = solver.solve()
            
            # --- Print Summary and Save Results ---
            print(f"\n--- Trial {i+1} Finished ---")
            print(f"Problem: '{problem.name}'")
            print(f"Initial x: {np.array2string(x0, formatter={'float_kind':lambda x: '%.4f' % x})}")
            print(f"Status: {solution['status']}")
            print(f"Final Objective F(x,y): {solution['F']:.6f}")
            print(f"Optimal x: {np.array2string(solution['x'], formatter={'float_kind':lambda x: '%.4f' % x})}")
            print(f"Optimal y: {np.array2string(solution['y'], formatter={'float_kind':lambda x: '%.4f' % x})}")

            save_results(solution, problem.name, x0)
            
            # --- Plotting and Comparison ---
            if solution['history']['F']:
                plot_convergence(solution, problem)
            
            print("\n--- Comparison with Known Solutions ---")
            formatter = {'float_kind':lambda x: "%.4f" % x}
            for sol_known in problem.solution:
                sol_type = sol_known.get('type', 'N/A').capitalize()
                print(f"--- {sol_type} Solution ---")
                print(f"  Known F: {sol_known['F']:.6f}")
                print(f"  Known x: {np.array2string(np.array(sol_known['x']), formatter=formatter)}")
                print(f"  Known y: {np.array2string(np.array(sol_known['y']), formatter=formatter)}")


if __name__ == "__main__":
    main()