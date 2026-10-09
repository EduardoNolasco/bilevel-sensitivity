"""Library of benchmark bilevel problems (BOLIB), defined as CasADi expressions."""
import casadi as ca
import numpy as np
class BilevelProblem:
    """
    A class to represent a bilevel optimization problem.
    """
    def __init__(self, name, x, y, F, f, G, g, lb_x, ub_x, lb_y, ub_y, x0_list=None, solution=None, is_ll_linear=False):
        self.name = name 
        self.x = x  # CasADi symbolic MX variable for upper level
        self.y = y  # CasADi symbolic MX variable for lower level
        self.x0_list = x0_list  # Accepts a list of starting points
        self.solution = solution  # Dictionary with known optimal values, e.g., {'x': val, 'F': val}
        self.is_ll_linear = is_ll_linear

        # Upper-level components
        self.F = F   # Objective function
        self.G = G   # Inequality constraints
        self.lb_x = lb_x
        self.ub_x = ub_x
        self.num_upper_vars = x.shape[0]
        self.num_upper_cons = G.shape[0] 

        # Lower-level components
        self.f = f  # Objective function
        self.g = g  # Inequality constraints
        self.lb_y = lb_y
        self.ub_y = ub_y

        # Derived properties
        self.num_upper_vars = x.shape[0] if x is not None else 0
        self.num_upper_cons = G.shape[0] if G is not None else 0
        self.num_lower_vars = y.shape[0] if y is not None else 0
        self.num_lower_cons = g.shape[0] if g is not None else 0

# Add problems in the appropriate format.
# !!! Remember to add the problem to the get_all_problems function at the end of this file. !!!

def AiyoshiShimizu1984Ex2():
    n, m = 2, 2
    x = ca.MX.sym("x", n)
    y = ca.MX.sym("y", m)

    upper_obj = 2*x[0] + 2*x[1] - 3*y[0] - 3*y[1] - 60
    lower_obj = (y[0] - x[0] + 20)**2 + (y[1] - x[1] + 20)**2
    upper_cons = ca.vertcat(x[0] + x[1] + y[0] - 2*y[1] - 40)
    lower_cons = ca.vertcat(2*y[0] - x[0] + 10, 2*y[1] - x[1] + 10)

    lb_x = np.full(n, 0.0)
    ub_x = np.full(n, 50.0)
    lb_y = np.full(m, -10.0)
    ub_y = np.full(m, 20.0)

    # EDITED: Updated starting point and solution
    x0_list = [np.array([20.0, 20.0]),
               np.array([35.0, 35.0])]
    known_solutions = [
        {'type': 'global', 'x': [25.0, 30.0], 'y': [5.0, 10.0], 'F': 5.0},
        {'type': 'local',  'x': [0.0, 0.0],   'y': [-10.0, -10.0], 'F': 0.0}
    ]

    return BilevelProblem(name="AiyoshiShimizu1984Ex2", x=x, y=y, F=upper_obj, f=lower_obj, G=upper_cons, g=lower_cons,
                         lb_x=lb_x, ub_x=ub_x, lb_y=lb_y, ub_y=ub_y,
                         x0_list=x0_list, solution=known_solutions)

def AllendeStill2013():
    n, m = 2, 2
    x = ca.MX.sym("x", n); y = ca.MX.sym("y", m)
    F = x[0]**2 - 2*x[0] + x[1]**2 - 2*x[1] + y[0]**2 + y[1]**2
    f = y[0]**2 - 2*x[0]*y[0] + y[1]**2 - 2*x[1]*y[1]
    G = ca.vertcat(x[0] - 2)
    g = ca.vertcat((y[0] - 1)**2 - 0.25, (y[1] - 1)**2 - 0.25)
    lb_x = np.full(n, 0.0); ub_x = np.full(n, np.inf)
    lb_y = np.full(m, 0.0); ub_y = np.full(m, np.inf)
    x0_list = [np.array([1.6, 1.6]), np.array([2.0, 2.0]), np.array([0.1, 0.1])]
    known_solutions = [{'type': 'global', 'x': [0.5, 0.5], 'y': [0.5, 0.5], 'F': -1.0}]
    return BilevelProblem(name="AllendeStill2013", x=x, y=y, F=F, f=f, G=G, g=g, 
                         lb_x=lb_x, ub_x=ub_x, lb_y=lb_y, ub_y=ub_y, 
                         x0_list=x0_list, solution=known_solutions)

def Bard_1988_ex1():
    n, m = 1, 1
    x = ca.MX.sym("x", n); y = ca.MX.sym("y", m)
    F = (x[0] - 5)**2 + (2*y[0] + 1)**2
    f = (y[0] - 1)**2 - 1.5*x[0]*y[0]
    G = ca.vertcat()
    g = ca.vertcat(-3*x[0] + y[0] + 3, x[0] - 0.5*y[0] - 4, x[0] + y[0] - 7)
    lb_x = np.array([0.0]); ub_x = np.array([np.inf])
    lb_y = np.array([0.0]); ub_y = np.array([np.inf])
    x0_list = [np.array([5.0]), np.array([2.9]), np.array([2.0])] # Infeasible LL with x \leq 1, x \geq 5
    known_solutions = [
        {'type': 'global', 'x': [1.0], 'y': [0.0], 'F': 17.0},
        {'type': 'local', 'x': [5.0], 'y': [2.0], 'F': 25.0}
    ]
    return BilevelProblem(name="Bard_1988_ex1", x=x, y=y, F=F, f=f, G=G, g=g, 
                         lb_x=lb_x, ub_x=ub_x, lb_y=lb_y, ub_y=ub_y, 
                         x0_list=x0_list, solution=known_solutions)

def Bard_1991_ex1():
    n, m = 1, 2
    x = ca.MX.sym("x", n); y = ca.MX.sym("y", m)
    F = x[0] + y[1]
    f = 2*y[0] + x[0]*y[1]
    G = ca.vertcat()
    g = ca.vertcat(x[0] - y[0] - y[1] + 4)
    lb_x = np.array([2.0]); ub_x = np.array([4.0])
    lb_y = np.full(m, 0.0); ub_y = np.full(m, np.inf)
    x0_list = [np.array([1.1]), np.array([3.0]), np.array([4.0])]
    known_solutions = [{'type': 'global', 'x': [2.0], 'y': [6.0, 0.0], 'F': 2.0}]
    return BilevelProblem(name="Bard_1991_ex1", x=x, y=y, F=F, f=f, G=G, g=g, 
                         lb_x=lb_x, ub_x=ub_x, lb_y=lb_y, ub_y=ub_y, 
                         x0_list=x0_list, solution=known_solutions, is_ll_linear=True)

def Bard_Book_1998():
    n, m = 2, 2
    x = ca.MX.sym("x", n); y = ca.MX.sym("y", m)
    F = (y[0] - x[0] + 20)**2 + (y[1] - x[1] + 20)**2
    f = 2*x[0] + 2*x[1] - 3*y[0] - 3*y[1] - 60
    G = ca.vertcat()
    g = ca.vertcat(x[0] + x[1] + y[0] - 2*y[1] - 40, 2*y[0] - x[0] + 10, 2*y[1] - x[1] + 10)
    lb_x = np.full(n, 0.0); ub_x = np.full(n, 50.0)
    lb_y = np.full(m, -10.0); ub_y = np.full(m, 20.0)
    x0_list = [np.array([20.0, 30.0]), np.array([15.0, 25.0]), np.array([1.0, 1.0])]
    known_solutions = [{'type': 'global', 'x': [25.0, 30.0], 'y': [5.0, 10.0], 'F': 0.0}]
    return BilevelProblem(name="Bard_Book_1998", x=x, y=y, F=F, f=f, G=G, g=g, 
                         lb_x=lb_x, ub_x=ub_x, lb_y=lb_y, ub_y=ub_y, 
                         x0_list=x0_list, solution=known_solutions, is_ll_linear=True)

def ClarkWesterberg1990():
    """Defines the Clark & Westerberg (1990) problem."""
    # --- 1. Define Dimensions ---
    n = 1
    m = 1

    # --- 2. Define Symbolic Variables ---
    x = ca.MX.sym("x", n)
    y = ca.MX.sym("y", m)

    # --- 3. Define Objectives and Constraints ---
    upper_obj = (x - 3)**2 + (y - 2)**2
    lower_obj = (y - 5)**2

    upper_cons = ca.vertcat() 
    lower_cons = ca.vertcat(
        -2*x + y - 1,
        x - 2*y + 2,
        x + 2*y - 14
    )
    
    # --- 4. Define Bounds and Initial Point using NumPy ---
    lb_x = np.array([0.0])
    ub_x = np.array([8.0])
    lb_y = np.array([-100.0])
    ub_y = np.array([100.0])

    # Define starting point and known solution
    # x0 = np.array([5.0])
    x0_list = [
        np.array([0.5]),
        np.array([1.7]),
        np.array([4.0])
    ]

    known_solutions = [
        {'type':'global', 'x': [1.0], 'y': [3.0], 'F': 5},
        {'type':'local',  'x': [3.0], 'y': [5.0], 'F': 9.0}
        ]

    # Create and return a BilevelProblem object
    return BilevelProblem(
        name="ClarkWesterberg1990",
        x=x, y=y,
        F=upper_obj, f=lower_obj,
        G=upper_cons, g=lower_cons,
        lb_x=lb_x, ub_x=ub_x,
        lb_y=lb_y, ub_y=ub_y,
        x0_list=x0_list,
        solution=known_solutions
    )

def Dempe1992a():
    """Defines the Dempe1992a problem."""
    # --- 1. Define Dimensions ---
    n = 2
    m = 2

    # --- 2. Define Symbolic Variables ---
    x = ca.MX.sym("x", n)
    y = ca.MX.sym("y", m)

    # --- 3. Define Objectives and Constraints ---
    upper_obj = y[1]
    lower_obj = 0.5*(y[0] - 1)**2 + 0.5*y[1]**2

    upper_cons = ca.vertcat(
        x[0]**2 + (x[1] + 1)**2 - 1
    ) 
    lower_cons = ca.vertcat(
        y[0] + y[1]*x[0] + x[1]
    )

    # --- 4. Define Bounds and Initial Point using NumPy ---
    lb_x = np.full(n, -np.inf)
    ub_x = np.full(n, np.inf)
    lb_y = np.full(m, -np.inf)
    ub_y = np.array([0.0, np.inf])

    # Define starting point and known solution
    x0_list = [
        np.array([0.0,1.0]), # Smart guess near global optimum
        np.array([0.1,3.4]),   # Guess near local optimum
        np.array([1.1, 0.1])   # Another point
    ]

    known_solutions = [
        {'type':'possible', 'x': [0,0], 'y': [0, -0.5], 'F': -0.5}
        ]

    # Create and return a BilevelProblem object
    return BilevelProblem(
        name="Dempe1992a",
        x=x, y=y,
        F=upper_obj, f=lower_obj,
        G=upper_cons, g=lower_cons,
        lb_x=lb_x, ub_x=ub_x,
        lb_y=lb_y, ub_y=ub_y,
        x0_list=x0_list,
        solution=known_solutions
    )

def DempeEtal2012():
    n, m = 1, 1
    x = ca.MX.sym("x", n); y = ca.MX.sym("y", m)
    F = x[0]
    f = x[0]*y[0]
    G = ca.vertcat(); g = ca.vertcat()
    lb_x = np.array([-1.0]); ub_x = np.array([1.0])
    lb_y = np.array([0.0]); ub_y = np.array([1.0])
    x0_list = [np.array([-0.9]), np.array([0.0]), np.array([0.9])]
    known_solutions = [{'type': 'global', 'x': [-1.0], 'y': [1.0], 'F': -1.0}]
    return BilevelProblem(name="DempeEtal2012", x=x, y=y, F=F, f=f, G=G, g=g, 
                         lb_x=lb_x, ub_x=ub_x, lb_y=lb_y, ub_y=ub_y, 
                         x0_list=x0_list, solution=known_solutions, is_ll_linear=True)

def DempeFranke2011Ex41():
    n, m = 2, 2
    x = ca.MX.sym("x", n); y = ca.MX.sym("y", m)
    F = x[0] + y[0]**2 + y[1]**2
    f = x[0]*y[0] + x[1]*y[1]
    G = ca.vertcat()
    g = ca.vertcat(-2*y[0] + y[1], y[0] - 2)
    lb_x = np.full(n, -1.0); ub_x = np.full(n, 1.0)
    lb_y = np.array([-np.inf, 0.0]); ub_y = np.array([np.inf, 2.0])
    x0_list = [np.array([0.1, -0.9]), np.array([0.0, 0.0]), np.array([0.9, 0.9])]
    known_solutions = [{'type': 'global', 'x': [0.0, -1.0], 'y': [1.0, 2.0], 'F': 5.0}]
    return BilevelProblem(name="DempeFranke2011Ex41", x=x, y=y, F=F, f=f, G=G, g=g, 
                         lb_x=lb_x, ub_x=ub_x, lb_y=lb_y, ub_y=ub_y, 
                         x0_list=x0_list, solution=known_solutions, is_ll_linear=True)

def Dempe_Franke_2011_ex42():
    n, m = 2, 2
    x = ca.MX.sym("x", n); y = ca.MX.sym("y", m)
    F = x[0] + (y[0] - 1)**2 + y[1]**2
    f = x[0]*y[0] + x[1]*y[1]
    G = ca.vertcat()
    g = ca.vertcat(-y[0] + y[1] - 1, y[0] + y[1] - 3.5, y[1] - 2)
    lb_x = np.full(n, -1.0); ub_x = np.full(n, 1.0)
    lb_y = np.full(m, -np.inf); ub_y = np.full(m, np.inf)
    x0_list = [np.array([0.9, -0.9]), np.array([0.0, 0.0]), np.array([-0.9, 0.9])]
    known_solutions = [{'type': 'global', 'x': [1.0, -1.0], 'y': [0.0, 1.0], 'F': 3.0}]
    return BilevelProblem(name="Dempe_Franke_2011_ex42", x=x, y=y, F=F, f=f, G=G, g=g, 
                         lb_x=lb_x, ub_x=ub_x, lb_y=lb_y, ub_y=ub_y, 
                         x0_list=x0_list, solution=known_solutions, is_ll_linear=True)

def Dempe_Lohse_2011_ex31a():
    n, m = 2, 2
    x = ca.MX.sym("x", n); y = ca.MX.sym("y", m)
    F = (x[0] - 0.5)**2 + (x[1] - 0.5)**2 - 3*y[0] - 3*y[1]
    f = x[0]*y[0] + x[1]*y[1]
    G = ca.vertcat()
    g = ca.vertcat(y[0] + y[1] - 2, -y[0] + y[1])
    lb_x = np.full(n, -np.inf); ub_x = np.full(n, np.inf)
    lb_y = np.full(m, 0.0); ub_y = np.full(m, np.inf)
    x0_list = [np.array([-1.0, -1.0]), np.array([-0.4, -0.4]), np.array([-2.5, -7.5])]
    known_solutions = [
        {'type': 'global', 'x': [0.0, 0.0], 'y': [1.0, 1.0], 'F': -5.5},
        {'type': 'local', 'x': [0.5, 0.5], 'y': [0.0, 0.0], 'F': 0.0}
    ]
    return BilevelProblem(name="Dempe_Lohse_2011_ex31a", x=x, y=y, F=F, f=f, G=G, g=g, 
                         lb_x=lb_x, ub_x=ub_x, lb_y=lb_y, ub_y=ub_y, 
                         x0_list=x0_list, solution=known_solutions, is_ll_linear=True)

def Dempe_Lohse_2011_ex31b():
    n, m = 3, 3
    x = ca.MX.sym("x", n); y = ca.MX.sym("y", m)
    F = (x[0] - 0.5)**2 + (x[1] - 0.5)**2 + x[2]**2 - 3*y[0] - 3*y[1] - 6*y[2]
    f = x[0]*y[0] + x[1]*y[1] + x[2]*y[2]
    G = ca.vertcat()
    g = ca.vertcat(y[0] + y[1] + y[2] - 2, -y[0] + y[1])
    lb_x = np.full(n, -np.inf); ub_x = np.full(n, np.inf)
    lb_y = np.full(m, 0.0); ub_y = np.full(m, np.inf)
    x0_list = [np.array([1.0, 1.0, 1.0]), np.array([4.0, 4.0, 4.0]), np.array([-1.0, -1.0, -1.0])]
    known_solutions = [
        {'type': 'global', 'x': [0.5, 0.5, 0.0], 'y': [0.0, 0.0, 2.0], 'F': -12.0},
        {'type': 'local', 'x': [0.5, 0.5, 0.0], 'y': [1.0, 1.0, 0.0], 'F': -6.0}
    ]
    return BilevelProblem(name="Dempe_Lohse_2011_ex31b", x=x, y=y, F=F, f=f, G=G, g=g, 
                         lb_x=lb_x, ub_x=ub_x, lb_y=lb_y, ub_y=ub_y, 
                         x0_list=x0_list, solution=known_solutions, is_ll_linear=True)

def FloudasEtal2013():
    n, m = 2, 2
    x = ca.MX.sym("x", n); y = ca.MX.sym("y", m)
    F = 2*x[0] + 2*x[1] - 3*y[0] - 3*y[1] - 60
    f = (y[0] - x[0] + 20)**2 + (y[1] - x[1] + 20)**2
    G = ca.vertcat()
    g = ca.vertcat(x[0] + x[1] + y[0] - 2*y[1] - 40, 2*y[0] - x[0] + 10, 2*y[1] - x[1] + 10)
    lb_x = np.full(n, 0.0); ub_x = np.full(n, 50.0)
    lb_y = np.full(m, -10.0); ub_y = np.full(m, 20.0)
    x0_list = [np.array([1.0, 1.0]), np.array([25.0, 25.0]), np.array([10.0, 10.0])]
    known_solutions = [{'type': 'global', 'x': [0.0, 0.0], 'y': [-10.0, -10.0], 'F': 0.0}]
    return BilevelProblem(name="FloudasEtal2013", x=x, y=y, F=F, f=f, G=G, g=g, 
                         lb_x=lb_x, ub_x=ub_x, lb_y=lb_y, ub_y=ub_y, 
                         x0_list=x0_list, solution=known_solutions)

def Outrata_Cervinka_2009():
    n, m = 2, 2
    x = ca.MX.sym("x", n); y = ca.MX.sym("y", m)
    F = -2*x[0] - 0.5*x[1] - y[1]
    f = y[0] - y[1] + x[0]*y[0] + x[1]*y[1] + 0.5*(y[0]**2 + y[1]**2)
    G = ca.vertcat(x[0])
    g = ca.vertcat(y[1], y[1] - y[0], y[1] + y[0])
    lb_x = np.full(n, -np.inf); ub_x = np.full(n, np.inf)
    lb_y = np.full(m, -np.inf); ub_y = np.full(m, np.inf)
    x0_list = [np.array([-10.0, -1.0]), np.array([-10.0, 1.0]), np.array([10.0, 1.0])]
    known_solutions = [{'type': 'global', 'x': [0.0, 0.0], 'y': [0.0, 0.0], 'F': 0.0}]
    return BilevelProblem(name="Outrata_Cervinka_2009", x=x, y=y, F=F, f=f, G=G, g=g, 
                         lb_x=lb_x, ub_x=ub_x, lb_y=lb_y, ub_y=ub_y, 
                         x0_list=x0_list, solution=known_solutions)

def Shimizu_Aiyoshi_1981_ex2():
    n, m = 2, 2
    x = ca.MX.sym("x", n); y = ca.MX.sym("y", m)
    F = (x[0] - 30)**2 + (x[1] - 20)**2 - 20*y[0] + 20*y[1]
    f = (x[0] - y[0])**2 + (x[1] - y[1])**2
    G = ca.vertcat(-x[0] - 2*x[1] + 30, x[0] + x[1] - 25, x[1] - 15)
    g = ca.vertcat()
    lb_x = np.full(n, -np.inf); ub_x = np.full(n, np.inf)
    lb_y = np.full(m, 0.0); ub_y = np.full(m, 10.0)
    x0_list = [np.array([19.0, 6.0]), np.array([15.0, 10.0]), np.array([10.0, 1.0])]
    known_solutions = [{'type': 'global', 'x': [20.0, 5.0], 'y': [10.0, 5.0], 'F': 225.0}]
    return BilevelProblem(name="Shimizu_Aiyoshi_1981_ex2", x=x, y=y, F=F, f=f, G=G, g=g, 
                         lb_x=lb_x, ub_x=ub_x, lb_y=lb_y, ub_y=ub_y, 
                         x0_list=x0_list, solution=known_solutions)

def get_all_problems():
    """Returns a list of all problem-creating functions in the library."""
    return [AiyoshiShimizu1984Ex2,
            AllendeStill2013,
            Bard_1988_ex1,
            Bard_1991_ex1,
            Bard_Book_1998,
            ClarkWesterberg1990,
            Dempe1992a,
            DempeEtal2012,
            DempeFranke2011Ex41,
            Dempe_Franke_2011_ex42,
            Dempe_Lohse_2011_ex31a,
            Dempe_Lohse_2011_ex31b,
            FloudasEtal2013,
            Outrata_Cervinka_2009,
            Shimizu_Aiyoshi_1981_ex2]
