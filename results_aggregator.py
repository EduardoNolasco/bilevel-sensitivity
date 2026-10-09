import os
import json
import pandas as pd
import numpy as np
# EDITED: Import the problem definitions to access known solutions
from bolib_class import get_all_problems

def aggregate_results():
    """
    Scans the 'results' directory, loads all JSON result files,
    and aggregates them into a single summary CSV file, enriching the data
    with the known solutions from the problem library.
    """
    # --- 1. Create a lookup table for problem definitions ---
    # This maps the string name of a problem to its full object definition
    problem_factories = get_all_problems()
    problem_lookup = {func().name: func() for func in problem_factories}

    results_dir = 'results'
    if not os.path.exists(results_dir):
        print(f"Error: The '{results_dir}' directory does not exist. Run experiments first.")
        return

    all_results = []
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]

    if not json_files:
        print("No JSON result files found in the 'results' directory.")
        return

    print(f"Found {len(json_files)} result files. Aggregating...")

    for filename in json_files:
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)

        problem_name = data.get('problem_name', 'N/A')
        
        # --- 2. Fetch the known solution from the lookup table ---
        known_global = None
        if problem_name in problem_lookup:
            problem_def = problem_lookup[problem_name]
            if problem_def.solution:
                # Find the global known solution, default to first if not specified
                for sol in problem_def.solution:
                    if sol.get('type') == 'global':
                        known_global = sol
                        break
                if not known_global:
                    known_global = problem_def.solution[0]

        # --- 3. Build the row with both found and known data ---
        row = {
            'Problem_Name': problem_name,
            'Dimensions (n,m)': f"({len(data.get('x', []))}, {len(data.get('y', []))})",
            'x0': str(np.round(data.get('x0', []), 2)),
            'F_known': known_global.get('F') if known_global else 'N/A',
            'F_found': data.get('F'),
            'x_known': str(np.round(known_global.get('x', []), 2)) if known_global else 'N/A',
            'x_found': str(np.round(data.get('x', []), 2)),
            'y_known': str(np.round(known_global.get('y', []), 2)) if known_global else 'N/A',
            'y_found': str(np.round(data.get('y', []), 2)),
            'Status': data.get('status', 'N/A'),
            'Iterations': data.get('iterations'),
            'Time (s)': data.get('time')
        }
        all_results.append(row)

    # --- 4. Create and save the pandas DataFrame ---
    df = pd.DataFrame(all_results)
    
    column_order = [
        'Problem_Name', 'Dimensions (n,m)', 'x0', 'F_known', 'F_found',
        'x_known', 'x_found', 'y_known', 'y_found', 'Status', 'Iterations', 'Time (s)'
    ]
    df = df[column_order]

    df = df.sort_values(by=['Problem_Name', 'x0']).reset_index(drop=True)

    output_filename = 'summary_results.csv'
    df.to_csv(output_filename, index=False)
    
    print("\n--- Aggregation Complete ---")
    print(f"Summary table saved to '{output_filename}'")
    print("\n--- First 5 Rows of the Summary ---")
    print(df.head().to_string())

if __name__ == '__main__':
    # You may need to install pandas: pip install pandas
    aggregate_results()