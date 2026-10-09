import json
import numpy as np
import datetime
import os

class NpEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_results(solution, problem_name, x0):
    """
    Saves the solution dictionary to a uniquely named JSON file.
    
    The filename includes the problem name, a representation of the 
    starting point, and a timestamp to ensure no files are overwritten.
    """
    # Create a 'results' directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Create a clean string representation of the starting point for the filename
    x0_str = '_'.join(map(str, np.round(x0, 2)))
    filename = f"results/{problem_name}_x0_{x0_str}_{timestamp}.json"
    
    # Make a copy to add metadata without modifying the original dict
    solution_to_save = solution.copy()
    solution_to_save['problem_name'] = problem_name
    solution_to_save['x0'] = x0

    with open(filename, 'w') as f:
        json.dump(solution_to_save, f, indent=4, cls=NpEncoder)
    print(f"Results for x0={x0} saved to {filename}")

def load_results(filename):
    """Loads a solution dictionary from a JSON file."""
    with open(filename, 'r') as f:
        solution = json.load(f)
    print(f"Results loaded from {filename}")
    return solution