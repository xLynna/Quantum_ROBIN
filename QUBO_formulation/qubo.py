import numpy as np
import heapq
from Solvers.solver import solve_qubo

# Without variable number limits
def definite_graph_to_qubo(G, penalty_reg=2):
  """Transform a graph G into a QUBO matrix Q and bias b for 
      the maximum clique problem.
  
  Parameters
  ----------
  G : np.ndarray
      Adjacency matrix of the graph.
      
  Returns
  -------
  Q : numpy.ndarray
      QUBO matrix.
  b : numpy.ndarray
      Bias vector.
  """
  n = G.shape[0]
  Q = np.zeros((n, n))
  b = -np.ones(n)

  Q[G == 0] = 1 # assign unit penalty to non-edges
  Q.fill_diagonal(0)
  
  return penalty_reg * Q, b

def weighted_graph_to_qubo(G, penalty_reg=1):
  """Transform a compatible weighted graph G into a QUBO matrix Q and bias b for 
      the maximum clique problem.

  Parameters
  ----------
  G : np.ndarray
      Weighted edge strength matrix of the graph.
      
  Returns
  -------
  Q : numpy.ndarray
      QUBO matrix.
  b : numpy.ndarray
      Bias vector.
  """
  n = G.shape[0]
  Q = G
  b = -np.ones(n)

  return penalty_reg * Q, b

def solve_graph(G, solver):
  """Solve the maximum clique problem for a graph G.
  """
  Q, b = definite_graph_to_qubo(G)
  
  #max_clique_mask
  return solve_qubo(Q, b, solver)

def solve_weighted_graph(G, solver):
  """Solve the maximum clique problem for a weighted graph G.
  """
  Q, b = weighted_graph_to_qubo(G)
  
  #max_clique_mask
  return solve_qubo(Q, b, solver)
