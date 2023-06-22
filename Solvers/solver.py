import numpy as np
from Solvers.dwave_solver import dwave_exact_solver, dwave_annealing_solver, dwave_quantum_solver
from Solvers.gurobi_solver import gurobi_solver

_ising_slover = {"gurobi": False, "dwave_exact": True, "dwave_annealing": True, "dwave_quantum": True}

def _ising_format(W, c):
  Q = W / 4
  np.fill_diagonal(Q, 0)
  c = c.reshape(-1, 1)
  bias = 0.5 * (c + np.sum(W, axis=1, keepdims=True))
  return Q, bias

def solve_qubo(Q, bias, solver, api_key=None):
  """
  
  Parameters
  ----------
  Q : numpy.ndarray
      QUBO matrix.
  bias : numpy.ndarray
      Bias vector.
  solver : str
      Name of the solver.
      
  Returns
  -------
  solution : numpy.ndarray
      Binary array of the solution.
  """
  if _ising_slover[solver]:
    Q, bias = _ising_format(Q, bias)

  if solver == "gurobi":
    return gurobi_solver(Q, bias)
  elif solver == "dwave_exact":
    return dwave_exact_solver(Q, bias)
  elif solver == "dwave_annealing":
    return dwave_annealing_solver(Q, bias)
  elif solver == "dwave_quantum":
    return dwave_quantum_solver(Q, bias, api_key)
  else:
    raise ValueError("Solver not found.")