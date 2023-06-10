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
  np.fill_diagonal(Q, 0)
  
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

def n_invariant_ordinary_graph_to_qubo(H, penalty_reg=1, weighted=False):
  """Transform a graph G into a QUBO matrix Q and bias b for 
      the maximum clique problem.
  
  Parameters
  ----------
  H : np.ndarray
      Adjacency matrix of the graph, as the failed tests egde matrix.
      
  Returns
  -------
  Q : numpy.ndarray
      QUBO matrix.
  b : numpy.ndarray
      Bias vector.
  """
  n = H.shape[1]
  Q = np.zeros((n, n))
  b = -np.ones(n)

  for i in range(n):
    inds = np.argwhere(H[:, i] == 1).flatten()
    Q[i, :] = np.sum(H[inds, :])

  np.fill_diagonal(Q, 0) # TODO check if it's necessary
  if not weighted:
    Q = Q.astype(bool).astype(int)
  
  return penalty_reg * Q, b

def hypergraph_to_qubo(H, penalty_reg=1):
  """H is a inverted-hypergraph, each row represent a non-hyperedge.

  Parameters
  ----------
  H : np.ndarray (k, n nodes)
      Inverted asymmetric hypergraph regarding the failed tests.
  """

  n = H.shape[1]
  b = -np.ones(n)
  Q = H.T@H
  
  return penalty_reg * Q, b

# graph solver (connector to solvers)
def solve_graph(G, solver, penalty_reg=2):
  """Solve the maximum clique problem for a graph G.
  """
  Q, b = definite_graph_to_qubo(G, penalty_reg)
  
  #max_clique_mask
  return solve_qubo(Q, b, solver)

def solve_weighted_graph(G, solver):
  """Solve the maximum clique problem for a weighted graph G.
  """
  Q, b = weighted_graph_to_qubo(G)
  
  #max_clique_mask
  return solve_qubo(Q, b, solver)
