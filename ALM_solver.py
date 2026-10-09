"""
Sensitivity-based augmented Lagrangian (ALM) solver for bilevel programmes.
Version 1.0, as used to generate the numerical results of the paper.

Differences from the algorithm as described in the paper:
  * The inner ALM subproblem is minimised by SciPy's L-BFGS-B with gradients obtained by finite differences;
    every function evaluation re-solves the lower-level problem with IPOPT.
  * The forward-sensitivity routine (_compute_sensitivity) is used only to evaluate the stationarity residual
    r_stat of the stopping test. The adjoint formulation (3.9)-(3.10) is not implemented in this file.
  * Stall termination requires ten consecutive stalled outer iterations (tolerances set in BLP_main.py).
"""
import casadi as ca
import numpy as np
import time
from scipy.optimize import minimize

class SensitivityALMSolver:
    """
    A class to solve bilevel optimization problems using the ALM method with sensitivity analysis.
    """
    def __init__(self, problem, settings):
        """
        Initializes the solver by setting up the problem structure and
        compiling necessary CasADi functions.

        Args:
            problem (BilevelProblem): An object containing the symbolic
                                      definition of the bilevel problem.
            settings (dict): A dictionary of solver settings and hyperparameters.
        """

        self.problem = problem  # BilevelProblem instance
        self.settings = settings  # Solver settings

        # --- Unpack symbolic variables and expressions from the problem object ---
        x, y = problem.x, problem.y
        F, G = problem.F, problem.G
        f, g = problem.f, problem.g

        # Check if the problem is flagged as having a linear lower level.
        if getattr(problem, 'is_ll_linear', False):
            epsilon = self.settings.get('ll_regularization', 1e-6)
            print(f"Linear LL detected. Applying quadratic regularization with epsilon = {epsilon:.1e}")
            # Add the regularization term to the symbolic objective
            f = f + epsilon * ca.dot(y, y)

        # --- 1. Pre-compile all necessary functions using CasADi ---

        # Lower-level functions and derivatives
        self.ll_obj_func = ca.Function("ll_obj_func", [x, y], [f])
        self.ll_con_func = ca.Function("ll_constr_func", [x, y], [g])
        self.ll_grad_f_y = ca.Function("ll_obj_grad", [x, y], [ca.gradient(f, y)])        
        self.ll_jac_g_y = ca.Function("ll_constr_jac", [x, y], [ca.jacobian(g, y)])
        self.ll_jac_g_x = ca.Function('jac_g_x', [x, y], [ca.jacobian(g, x)])

        # Lower-level Lagrangian and its derivatives
        self.lambda_sym = ca.MX.sym("lambda", g.shape[0])
        self.ll_lagrangian = f + ca.dot(self.lambda_sym, g)
        self.ll_hess_L_yy = ca.Function("ll_hess_L_yy", [x, y, self.lambda_sym], [ca.hessian(self.ll_lagrangian, y)[0]])
        self.ll_grad_L_yx = ca.Function("ll_grad_L_yx", [x, y, self.lambda_sym], [ca.jacobian(ca.gradient(self.ll_lagrangian, y), x)])

        # Upper-level functions and derivatives
        self.ul_obj_func = ca.Function('F', [x, y], [F])
        self.ul_con_func = ca.Function('G', [x, y], [G])
        self.ul_grad_F_x = ca.Function('grad_F_x', [x, y], [ca.gradient(F, x)])
        self.ul_grad_F_y = ca.Function('grad_F_y', [x, y], [ca.gradient(F, y)])
        self.ul_jac_G_x = ca.Function('jac_G_x', [x, y], [ca.jacobian(G, x)])
        self.ul_jac_G_y = ca.Function('jac_G_y', [x, y], [ca.jacobian(G, y)])

        # --- 2. Set up the lower-level NLP solver ---
        p = x 
        nlp_ll = {'x': y, 'p': p, 'f': self.ll_obj_func(p, y), 'g': self.ll_con_func(p, y)}
        solver_options = {
            'ipopt.print_level': self.settings.get('ll_print_level', 0), 
            'print_time': False,
            'ipopt.sb': 'yes' # Suppresses IPOPT license message
        }
        self.ll_solver = ca.nlpsol('ll_solver', 'ipopt', nlp_ll, solver_options)

        print("Solver initialized successfully. All functions compiled.")

    def _solve_lower_level(self, x_k, y_guess=None):
        """
        Solves the lower-level problem for a given upper-level variable x_k.
        Includes robust error handling for solver failures.
        """
        if y_guess is None:
            y_guess = np.zeros(self.problem.num_lower_vars)
        
        lbg = -np.inf * np.ones(self.problem.num_lower_cons)
        ubg = np.zeros(self.problem.num_lower_cons)
        
        try:
            sol = self.ll_solver(x0=y_guess, p=x_k, lbx=self.problem.lb_y, ubx=self.problem.ub_y, 
                                 lbg=lbg, ubg=ubg)
            
            # EDITED: Check the solver's return status
            stats = self.ll_solver.stats()
            if not stats['success']:
                raise RuntimeError(
                    f"Lower-level solver failed at x = {x_k} with status: {stats['return_status']}"
                )
            
            return sol['x'].full().flatten(), sol['lam_g'].full().flatten()

        except Exception as e:
            # Re-raise the exception with more context
            print(f"Error during lower-level solve at x = {x_k}")
            raise e

    def _compute_sensitivity(self, x_k, y_k, lambda_k):
        """
        Computes the sensitivity matrix dy/dx by solving the system defined
        in Equation (3.4) of the paper (forward mode: one right-hand side per upper-level variable).
        """
        g_k = self.ll_con_func(x_k, y_k).full().flatten()
        active_set_tol = self.settings.get('active_set_tol', 1e-6)
        active_indices = np.where(g_k > -active_set_tol)[0]
        
        H_L_yy = self.ll_hess_L_yy(x_k, y_k, lambda_k).full()
        J_g_y = self.ll_jac_g_y(x_k, y_k).full()
        
        num_y, num_active = self.problem.num_lower_vars, len(active_indices)
        kkt_matrix = np.zeros((num_y + num_active, num_y + num_active))
        kkt_matrix[:num_y, :num_y] = H_L_yy
        
        if num_active > 0:
            J_g_y_active = J_g_y[active_indices, :]
            kkt_matrix[:num_y, num_y:] = J_g_y_active.T
            # CORRECTED: Added scaling by diagonal matrix of multipliers (Lambda_A)
            kkt_matrix[num_y:, :num_y] = np.diag(lambda_k[active_indices]) @ J_g_y_active

        grad_L_yx = self.ll_grad_L_yx(x_k, y_k, lambda_k).full()
        J_g_x = self.ll_jac_g_x(x_k, y_k).full()
        
        rhs = np.zeros((num_y + num_active, self.problem.num_upper_vars))
        rhs[:num_y, :] = -grad_L_yx
        
        if num_active > 0:
            J_g_x_active = J_g_x[active_indices, :]
            # CORRECTED: Added scaling by diagonal matrix of multipliers (Lambda_A)
            rhs[num_y:, :] = -np.diag(lambda_k[active_indices]) @ J_g_x_active

        # Legacy: small diagonal shift added to the sensitivity matrix; it is not described in the paper.
        reg_eps = self.settings.get('sensitivity_regularization', 1e-5)
        kkt_matrix += reg_eps * np.eye(kkt_matrix.shape[0])

        try:
            sensitivities = np.linalg.solve(kkt_matrix, rhs)
        except np.linalg.LinAlgError:
            print("Warning: KKT matrix is singular. Using pseudo-inverse.")
            sensitivities = np.linalg.pinv(kkt_matrix) @ rhs
            
        return sensitivities[:num_y, :]
    
    def _calculate_alm_objective(self, x_k, y_k, mu_k, rho_k):
        """
        Calculates the value of the PHR Augmented Lagrangian function.
        """
        F_k = self.ul_obj_func(x_k, y_k).full().flatten()
        G_k = self.ul_con_func(x_k, y_k).full().flatten()
        
        # PHR function value
        mu_plus = np.maximum(0, mu_k + rho_k * G_k)
        phr_term = (1 / (2 * rho_k)) * (np.sum(mu_plus**2) - np.sum(mu_k**2))
        
        return F_k + phr_term
    
    def _calculate_alm_gradient(self, x_k, y_k, lambda_k, mu_k, rho_k):
        dy_dx = self._compute_sensitivity(x_k, y_k, lambda_k)
        grad_F_x = self.ul_grad_F_x(x_k, y_k).full()
        grad_F_y = self.ul_grad_F_y(x_k, y_k).full()
        
        dF_dx = grad_F_x + dy_dx.T @ grad_F_y
        
        grad_alm = dF_dx
        G_k = self.ul_con_func(x_k, y_k).full().flatten()
        if G_k.size > 0:
            jac_G_x = self.ul_jac_G_x(x_k, y_k).full()
            jac_G_y = self.ul_jac_G_y(x_k, y_k).full()
            dG_dx = jac_G_x + jac_G_y @ dy_dx
            mu_plus = np.maximum(0, mu_k + rho_k * G_k)
            # CORRECTED: Reshape mu_plus to be a column vector for robust matrix multiplication
            grad_alm += dG_dx.T @ mu_plus.reshape(-1, 1)

        return grad_alm.flatten()

    def _calculate_kkt_residual(self, x_k, y_k, mu_k):
        _, lambda_k = self._solve_lower_level(x_k, y_guess=y_k)
        grad_L_F = self._calculate_alm_gradient(x_k, y_k, lambda_k, mu_k, 0) # ALM grad with rho=0 is Lagrangian grad
        r_stat = np.linalg.norm(grad_L_F, np.inf)
        
        G_k = self.ul_con_func(x_k, y_k).full().flatten()
        # CORRECTED: Handle case with no upper-level constraints
        r_feas = np.linalg.norm(np.maximum(0, G_k), np.inf) if G_k.size > 0 else 0
        r_comp = np.linalg.norm(mu_k * G_k, np.inf) if G_k.size > 0 else 0
        
        return max(r_stat, r_feas, r_comp), r_stat, r_feas, r_comp

    def solve(self):
        start_time = time.time()

        # --- 1. Initialization (Algorithm 1, line 1) ---
        # EDITED: Handle case where no starting point is provided
        if self.problem.x0 is not None:
            x_k = np.array(self.problem.x0)
            print(f"Starting point x0: {x_k}")
        else:
            print("Warning: No starting point (x0) provided. Defaulting to zeros.")
            x_k = np.zeros(self.problem.num_upper_vars)

        mu_k = np.zeros(self.problem.num_upper_cons)
        rho_k = self.settings.get('rho_start', 10.0)
        epsilon = self.settings.get('tol', 1e-4)
        epsilon_inner = self.settings.get('inner_tol_start', 0.1)  # unused in this version
        
        history = {
            'F': [], 'f': [], 'r_kkt': [], 'rho': [],
            'x': [], 'y': [], 'lambda': [], 'mu': [],
            'g': [], 'G': []
        }

        # --- 2. Initial Solve (Algorithm 1, line 2) ---
        y_k, lambda_k = self._solve_lower_level(x_k)

        F_k = self.ul_obj_func(x_k, y_k).full().item()
        f_k = self.ll_obj_func(x_k, y_k).full().item()
        g_k = self.ll_con_func(x_k, y_k).full().flatten()
        G_k = self.ul_con_func(x_k, y_k).full().flatten()
        r_kkt_val, _, _, _ = self._calculate_kkt_residual(x_k, y_k, mu_k)
        history['F'].append(F_k); history['f'].append(f_k)
        history['r_kkt'].append(r_kkt_val); history['rho'].append(rho_k)
        history['x'].append(x_k.copy()); history['y'].append(y_k.copy())
        history['lambda'].append(lambda_k.copy())
        history['mu'].append(mu_k.copy()); history['g'].append(g_k.copy())
        history['G'].append(G_k.copy())
        
        # CORRECTED: Handle case with no upper-level constraints
        r_feas_prev = np.linalg.norm(np.maximum(0, G_k), np.inf) if G_k.size > 0 else 0
        
        print("--- Starting Sensitivity-Based ALM ---")
        print(f"{'Iter':>4} {'F(x,y)':>12} {'KKT Residual':>14} {'rho':>10} {'Time (s)':>9}")
        print(f"{0:4d} {F_k:12.4e} {r_kkt_val:14.4e} {rho_k:10.2f} {'---':>9}")

        x_prev, F_prev = x_k.copy(), np.inf
        stall_counter = 0

        for k in range(self.settings.get('max_outer_iter', 100)):
            iter_start_time = time.time()
            
            # --- ARCHITECTURAL UPGRADE: Use SciPy to solve the inner subproblem ---
            def subproblem_obj(x_sub):
                try:
                    y_sub, _ = self._solve_lower_level(x_sub)
                    return self._calculate_alm_objective(x_sub, y_sub, mu_k, rho_k)
                except RuntimeError: 
                    return np.inf # Return a large value if the lower level fails to solve

            bounds = list(zip(self.problem.lb_x, self.problem.ub_x))
            # No gradient (jac) is passed: SciPy approximates the gradient of the ALM function by finite
            # differences, so each gradient costs extra lower-level solves.
            res = minimize(subproblem_obj, x_k, method='L-BFGS-B',
                           bounds=bounds, options={'gtol': self.settings.get('inner_tol', 1e-4)})

            x_k = res.x
            y_k, lambda_k = self._solve_lower_level(x_k)

            # --- Update duals and parameters ---
            G_k = self.ul_con_func(x_k, y_k).full().flatten()
            if G_k.size > 0:
                mu_k_next = np.maximum(0, mu_k + rho_k * G_k)
                r_feas_k = np.linalg.norm(np.maximum(0, G_k), np.inf)
                rho_max = self.settings.get('rho_max', 1e6)
                if r_feas_k > self.settings.get('feas_improv_factor', 0.5) * r_feas_prev and rho_k < rho_max:
                    rho_k *= self.settings.get('rho_increase_factor', 2.0)
                mu_k = mu_k_next; r_feas_prev = r_feas_k

            # --- 6. Logging and Convergence Check ---
            F_k = self.ul_obj_func(x_k, y_k).full().item()
            f_k = self.ll_obj_func(x_k, y_k).full().item()
            g_k = self.ll_con_func(x_k, y_k).full().flatten()

            kkt_results = self._calculate_kkt_residual(x_k, y_k, mu_k)
            r_kkt, r_stat, r_feas, r_comp = kkt_results
            
            # Store history
            history['F'].append(F_k)
            history['f'].append(f_k)
            history['r_kkt'].append(r_kkt)
            history['rho'].append(rho_k)
            history['x'].append(x_k.copy())
            history['y'].append(y_k.copy())
            history['lambda'].append(lambda_k.copy())
            history['mu'].append(mu_k.copy())
            history['g'].append(g_k.copy())
            history['G'].append(G_k.copy())
            
            print(f"{k+1:4d} {F_k:12.4e} {r_kkt:14.4e} {rho_k:10.2f} {time.time()-iter_start_time:9.2f}")

            # EDITED: Add secondary stopping criterion based on primal variable and objective change
            step_size = np.linalg.norm(x_k - x_prev)
            obj_change = abs(F_k - F_prev)
            tol_x = self.settings.get('tol_x', 1e-5)
            tol_F = self.settings.get('tol_F', 1e-5)

            if k > 0 and step_size < tol_x and obj_change < tol_F:
                stall_counter += 1
            else:
                stall_counter = 0

            if r_kkt < epsilon:
                print(f"\nConvergence achieved (KKT residual) after {k+1} iterations.")
                status = 'Success'; break
            if stall_counter >= 10: # Stop after 10 consecutive stalled iterations
                unmet_conditions = []
                if r_stat >= epsilon: unmet_conditions.append(f"stationarity ({r_stat:.2e})")
                if r_feas >= epsilon: unmet_conditions.append(f"feasibility ({r_feas:.2e})")
                if r_comp >= epsilon: unmet_conditions.append(f"complementarity ({r_comp:.2e})")
                if unmet_conditions:
                    reason = " and ".join(unmet_conditions)
                    print(f"  (Primary KKT criterion not met due to: {reason})")
                print(f"\nConvergence achieved (stalled for {stall_counter} iterations) after {k+1} iterations.")
                status = 'Success (stalled)'; break
            
            x_prev, F_prev = x_k.copy(), F_k
            
        else:
            print("\nMaximum outer iterations reached.")
            status = 'Max iterations reached'

        total_time = time.time() - start_time
        solution = {'x': x_k, 'y': y_k, 'F': F_k, 'status': status, 
                    'iterations': k+1, 'time': total_time, 'history': history}
        return solution