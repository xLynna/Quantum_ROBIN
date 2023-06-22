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
  Q = G.copy()
  b = -np.ones(n)
  Q = min_max_scale(Q, collapse=False)

  np.fill_diagonal(Q, 0)
  return penalty_reg * Q, b

def min_max_scale(Q, collapse=False, fill_diagonal=True):
  """Min-max scaling of the QUBO matrix Q.
  """
  if fill_diagonal:
    np.fill_diagonal(Q, 0)
  min_val = np.min(Q[Q > 0])
  max_val = np.max(Q[Q > 0])
  if collapse:
    Q[Q < (max_val + min_val) / 2] = 0
  else:
    Q = (Q - min_val) / (max_val - min_val)
  return Q


def n_invariant_ordinary_graph_to_qubo(H, penalty_reg=2, weighted=False):
  """Transform a graph G into a QUBO matrix Q and bias b for 
      the maximum clique problem.
  
  Parameters
  ----------
  H : np.ndarray
      Adjacency matrix of the graph.
      
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
    Q[i, :] = np.sum(H[inds, :], axis=0)

  if weighted:
    Q = (n-2) - Q # max possible connection - actual connection = penalty unit
    Q = min_max_scale(Q)
  else:
    Q = min_max_scale(Q, collapse=True)
    Q = (~(Q.astype(bool))).astype(int) # assign unit penalty to non-edges
  
  np.fill_diagonal(Q, 0)
  return penalty_reg * Q, b

def hypergraph_to_qubo(H, penalty_reg=2):
  """H is a inverted-hypergraph, each row represent a non-hyperedge.

  Parameters
  ----------
  H : np.ndarray (k, n nodes)
      Inverted asymmetric hypergraph regarding the failed tests.
  """

  m, n = H.shape
  b = np.zeros(n + 2*m)
  b[:n] = -1

  r_ind = np.arange(m).reshape(-1, 1)
  c_ind = 2*r_ind + np.array([0, 1])
  B = np.zeros((m, 2*m))
  B[r_ind, c_ind] = 1

  K = np.hstack((H, B))
  const = 4*np.ones((1, m))
  b -= (const@K).reshape(-1)
  Q = K.T@K

  np.fill_diagonal(Q[:n, :n], 0)
  return penalty_reg * Q, b

# graph solver (connector to solvers)
def solve_graph(G, solver, penalty_reg=2, api_key=None):
  """Solve the maximum clique problem for a graph G.
  """
  Q, b = definite_graph_to_qubo(G, penalty_reg)
  
  #max_clique_mask
  return solve_qubo(Q, b, solver, api_key)

def solve_weighted_graph(G, solver, penalty_reg=2, api_key=None):
  """Solve the maximum clique problem for a weighted graph G.
  """
  Q, b = weighted_graph_to_qubo(G, penalty_reg)
  
  #max_clique_mask
  return solve_qubo(Q, b, solver, api_key)
