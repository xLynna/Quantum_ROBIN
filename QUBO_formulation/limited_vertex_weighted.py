import numpy as np
import heapq
from QUBO_formulation.qubo import solve_weighted_graph

def _lightest_node(G):
  """Find the node whose edges have lowest weight in the graph G.
  
  Parameters
  ----------
  G : np.ndarray
      Adjacency weight matrix of the graph.
      
  Returns
  -------
  v : int
      Index of the node with the lightest edges.
  """
  degrees = np.sum(G, axis=0)
  v = np.argmin(degrees)
  return v

def _extract_subgraph(G, v):
  """Extract the subgraph of G induced by the neighbours of v.
  
  Parameters
  ----------
  G : np.ndarray
      Adjacency matrix of the graph.
  v : int
      Index of the node.
      
  Returns
  -------
  G : np.ndarray
      Adjacency matrix of the subgraph.
  """
  indices = np.arange(G.shape[0])
  v_neighbours_mask = G[v, :] == 1
  G = G[v_neighbours_mask, :][:, v_neighbours_mask].copy()
  return G, indices[v_neighbours_mask]

def _remove_vertex(G, v):
  """Remove the node v from the graph G.
  
  Parameters
  ----------
  G : np.ndarray
      Adjacency matrix of the graph.
  v : int
      Index of the node.
      
  Returns
  -------
  G : np.ndarray
      Adjacency matrix of the subgraph.
  """
  G = np.delete(G, v, axis=0)
  G = np.delete(G, v, axis=1)
  return G

#TODO how to dive and conquer the weighted graph for the max clique problem?

def split(G, vertex_limit, lower_bound, solver):
  """Split the graph G into subgraphs with a maximum number of nodes.
  
  Parameters
  ----------
  G : np.ndarray
      Adjacency matrix of the graph.
  vertex_limit : int
      Maximum number of nodes in the subgraphs.
  lower_bound : int
      Minimum number of nodes in the subgraphs.
      
  Returns
  -------
  max_clique_indices : set
      Indices of the nodes in the maximum clique.
  """
  max_clique_indices = None
  total_weight = np.sum(G)
  # negate graph size, graph, remaining nodes, splitted nodes
  subgraphs = [(-G.shape[0], G, np.arange[G.shape[0]] , set())]

  while len(subgraphs) > 0 and -subgraphs[0][0] > vertex_limit:
    _, sg, sg_indices, sg_pending_indices = heapq.heappop(subgraphs)
    v_ind = _lightest_node(sg)
    ssg, ssg_raw_indeices = _extract_subgraph(sg, v_ind)
    # indices in terms of the input graph
    ssg_indices = sg_indices[ssg_raw_indeices]

    sg = _remove_vertex(sg, v_ind)
    sg_indices = np.delete(sg_indices, v_ind)

    # Calculate the max clique from sg
    if sg_indices.shape[0] > 0:
      if sg_indices.shape[0] <= vertex_limit:
        # Obtain the max clique size and the related binary solution of the subgraph
        sg_solution_mask = solve_weighted_graph(sg, solver)
        # The size should include the vertices removed due to vertex limit
        combined_clique_size = len(sg_solution_mask) + len(sg_pending_indices)
        if combined_clique_size > lower_bound:
          lower_bound = combined_clique_size
          max_clique_indices = set(sg_indices[sg_solution_mask]) + sg_pending_indices
      else:
        heapq.heappush(subgraphs, (-sg.shape[0], sg, sg_indices, sg_pending_indices))
    
    # Calculate the max clique from ssg
    if ssg_indices.shape[0] > 0:
      if ssg_indices.shape[0] <= vertex_limit:
        ssg_solution_mask = solve_weighted_graph(ssg, solver)
        combined_clique_size = len(ssg_solution_mask) + len(sg_pending_indices) + 1 # v_ind
        if combined_clique_size > lower_bound:
          lower_bound = combined_clique_size
          max_clique_indices = set(ssg_indices[ssg_solution_mask]) + sg_pending_indices + {v_ind}
      else:
        heapq.heappush(subgraphs, (-ssg.shape[0], ssg, ssg_indices, sg_pending_indices + {v_ind}))

  return max_clique_indices, lower_bound
    