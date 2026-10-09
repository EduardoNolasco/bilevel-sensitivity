"""Convergence and variable-trajectory plots (PDF) for a single solver run."""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os


def plot_convergence(solution, problem):
    """
    Generates and saves a set of convergence plots for a given solution history.
    The plot layout is adaptive and focused on the most critical metrics.
    """
    history = solution['history']
    problem_name = problem.name
    x0 = problem.x0

    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'

    # --- Adaptive Plot Layout ---
    has_ul_cons = problem.num_upper_cons > 0
    
    if has_ul_cons:
        fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
        axes1 = axes1.flatten()
        # fig1.suptitle(f'Convergence for {problem_name} (x₀ = {np.round(x0, 2)})', fontsize=16)
    else:
        fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))
        # fig1.suptitle(f'Convergence for {problem_name} (x₀ = {np.round(x0, 2)})', fontsize=16)

    iterations = np.arange(len(history['F']))
    epsilon = 1e-16 # Small constant for log plotting

    # --- Plot 1: Upper-Level Objective (axes1[0]) ---
    ax = axes1[0]
    ax.plot(iterations, history['F'], 'o-', label=r'$F(x^k, \bar{y}(x^k))$')
    ax.set_title('Upper-Level Objective')
    ax.set_xlabel('Outer Iteration (k)')
    ax.set_ylabel(r'$F(x,y)$')
    ax.legend()
    ax.grid(True, linestyle=':')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- Plot 2: KKT Residual (axes1[1]) ---
    ax = axes1[1]
    # EDITED: Add epsilon to avoid log(0) errors
    kkt_data = np.maximum(history['r_kkt'], epsilon)
    ax.plot(iterations, kkt_data, 'o-', color='r', label=r'$r_{KKT}$')
    ax.set_title('KKT Residual')
    ax.set_xlabel('Outer Iteration (k)')
    ax.set_ylabel('Residual')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, linestyle=':')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


    # --- Plot 3 (Constrained): Upper-Level Multipliers (axes1[2]) ---
    ax = axes1[2]
    mu_history = np.array(history['mu'])
    for i in range(mu_history.shape[1]):
        ax.plot(iterations, mu_history[:, i], 'o-', label=fr'$\mu_{i+1}$')
    ax.set_title('Upper-Level Multipliers')
    ax.set_xlabel('Outer Iteration (k)')
    ax.set_ylabel(r'$\mu$')
    if mu_history.shape[1] <= 5: ax.legend()
    ax.grid(True, linestyle=':')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if has_ul_cons:
        # --- Plot 4 (Constrained Only): Constraint Violation (axes1[3]) ---
        ax = axes1[3]
        G_hist = np.array(history['G'])
        violations = np.linalg.norm(np.maximum(0, G_hist), ord=np.inf, axis=1)
        # EDITED: Add epsilon to avoid log(0) errors
        violations_data = np.maximum(violations, epsilon)
        ax.plot(iterations, violations_data, 'o-', color='orange', label=r'$\|\max(0, G(x^k))\|_{\infty}$')
        ax.set_title('Upper-Level Constraint Violation')
        ax.set_xlabel('Outer Iteration (k)')
        ax.set_ylabel('Violation')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, linestyle=':')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Second Figure for Variable Trajectories (Remains the same) ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    fig2.suptitle(f'Variable Trajectories for {problem_name} (x₀ = {np.round(x0, 2)})', fontsize=16)

    ax = axes2[0]
    x_hist = np.array(history['x'])
    for i in range(x_hist.shape[1]):
        ax.plot(iterations, x_hist[:, i], 'o-', label=f'$x_{i+1}$')
    ax.set_title('Upper-Level Variables'); ax.set_xlabel('Outer Iteration (k)'); ax.set_ylabel(r'$x$')
    if x_hist.shape[1] <= 5: ax.legend()
    ax.grid(True, linestyle=':'); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax = axes2[1]
    y_hist = np.array(history['y'])
    for i in range(y_hist.shape[1]):
        ax.plot(iterations, y_hist[:, i], 'o-', label=f'$y_{i+1}$')
    ax.set_title('Lower-Level Variables'); ax.set_xlabel('Outer Iteration (k)'); ax.set_ylabel(r'$y$')
    if y_hist.shape[1] <= 5: ax.legend()
    ax.grid(True, linestyle=':'); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # --- Save Figures ---
    plots_dir = os.path.join('results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    x0_str = '_'.join(map(str, np.round(x0, 2)))
    base_filename = f"{plots_dir}/{problem_name}_x0_{x0_str}"
    
    fig1.savefig(f"{base_filename}_convergence.pdf")
    fig2.savefig(f"{base_filename}_trajectories.pdf")
    
    print(f"Plots saved to {plots_dir}/{problem_name}_x0_{x0_str}_*.pdf")
    plt.close(fig1)
    plt.close(fig2)
