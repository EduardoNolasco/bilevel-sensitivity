import sys
import os
from file_utils import load_results
from plot_utils import plot_convergence
from bolib_class import get_all_problems

def main():
    """
    A command-line tool to load a single result JSON file and generate
    the corresponding convergence and trajectory plots.
    """
    # 1. Check for command-line argument
    if len(sys.argv) != 2:
        print("Usage: python analyze_results.py <path_to_json_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        sys.exit(1)

    print(f"--- Analyzing results from: {os.path.basename(filepath)} ---")

    # 2. Load the results data
    results = load_results(filepath)

    # 3. Re-create the problem object to pass to the plot function
    # This is necessary for the adaptive plotting logic
    problem_name = results.get('problem_name')
    if not problem_name:
        print("Error: JSON file does not contain a 'problem_name' key.")
        sys.exit(1)

    problem_factories = get_all_problems()
    problem_lookup = {func().name: func for func in problem_factories}

    if problem_name not in problem_lookup:
        print(f"Error: Problem '{problem_name}' not found in the bolib_class library.")
        sys.exit(1)

    # Instantiate the correct problem
    problem = problem_lookup[problem_name]()
    # Set the specific x0 for this run to ensure correct plot titles
    problem.x0 = results.get('x0')
    
    # 4. Generate and save the plots
    if results['history']['F']:
        print("\n--- Generating and Saving Plots ---")
        plot_convergence(results, problem)
    else:
        print("No history data found in the file to plot.")

if __name__ == "__main__":
    main()