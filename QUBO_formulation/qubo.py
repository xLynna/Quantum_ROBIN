import numpy as np
import heapq

# Without variable number limits
def naive_definite_transformation(G):
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

    penalty_reg = 2
    
    return penalty_reg * Q, b

# With variable number limits
def extract_k_core(G, k):
    """Extract the k-core of a graph G inplace.
    
    Parameters
    ----------
    G : np.ndarray
        Adjacency matrix of the graph.
    k : int
        Number of nodes in the k-core.
        
    Returns
    -------
    G : np.ndarray (inplace)
        Sparse adjacency matrix of the k-core.
    ineices : np.ndarray
        Indices of the remain nodes in the k-core.
    """
    indices = np.arange(G.shape[0])

    while True:
      degrees = np.sum(G, axis=0)
      remain_mask = degrees >= k
      if np.all(degrees[remain_mask] >= k):
          break
      else:
          indices = indices[remain_mask]
          G = G[remain_mask, :][:, remain_mask]
    return indices

def reduce_graph(G, lower_bound):
    """Reduce the graph G to a subgraph with a minimum number of nodes.
    
    Parameters
    ----------
    G : np.ndarray
        Adjacency matrix of the graph.
    lower_bound : int
        Minimum number of nodes in the subgraph.
        
    Returns
    -------
    G : np.ndarray (inplace)
        Adjacency matrix of the subgraph.
    indices : np.ndarray
        Indices in terms of the input graph of the remain nodes in the subgraph.
    """
    meta_indices = extract_k_core(G, lower_bound)
    v = np.random.choice(len(meta_indices), 1)[0]
    v_neighbours_mask = G[v, :] == 1
    common_neighbours = np.logical_and(G, v_neighbours_mask).sum(axis=1, keepdims=True)
    # Remove the edges between the nodes if their common neighbours 
    # are less than the lower bound - 2
    G[v, common_neighbours < (lower_bound - 2)] = 0
    G[common_neighbours < (lower_bound - 2), v] = 0
    sub_indices = extract_k_core(G, lower_bound)
    return meta_indices[sub_indices]

def highest_degree_node(G):
    """Find the node with the highest degree in the graph G.
    
    Parameters
    ----------
    G : np.ndarray
        Adjacency matrix of the graph.
        
    Returns
    -------
    v : int
        Index of the node with the highest degree.
    """
    degrees = np.sum(G, axis=0)
    v = np.argmax(degrees)
    return v

def extract_subgraph(G, v):
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

def remove_vertex(G, v):
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

def solve(G):
   pass
   #return max_clique_mask

def split(G, vertex_limit, lower_bound):
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
    subGs : list
        List of adjacency matrices of the subgraphs.
    """
    max_clique_indices = None
    # negate graph size, graph, remaining nodes, splitted nodes
    subgraphs = [(-G.shape[0], G, np.arange[G.shape[0]] , set())]

    while len(subgraphs) > 0 and -subgraphs[0][0] > vertex_limit:
      _, sg, sg_indices, sg_pending_indices = heapq.heappop(subgraphs)
      v_ind = highest_degree_node(sg)
      ssg, ssg_raw_indeices = extract_subgraph(sg, v_ind)
      # indices in terms of the input graph
      ssg_indices = sg_indices[ssg_raw_indeices]

      sg = remove_vertex(sg, v_ind)
      sg_indices = np.delete(sg_indices, v_ind)

      # Calculate the max clique from sg
      sg_indices = sg_indices[reduce_graph(sg, lower_bound)]
      if sg_indices.shape[0] > 0:
        if sg_indices.shape[0] <= vertex_limit:
          # Obtain the max clique size and the related binary solution of the subgraph
          sg_solution_mask = solve(sg)
          # The size should include the vertices removed due to vertex limit
          combined_clique_size = len(sg_solution_mask) + len(sg_pending_indices)
          if combined_clique_size > lower_bound:
            lower_bound = combined_clique_size
            max_clique_indices = set(sg_indices[sg_solution_mask]) + sg_pending_indices
        else:
          heapq.heappush(subgraphs, (-sg.shape[0], sg, sg_indices, sg_pending_indices))
      
      # Calculate the max clique from ssg
      ssg_indices = ssg_indices[reduce_graph(ssg, lower_bound)]
      if ssg_indices.shape[0] > 0:
        if ssg_indices.shape[0] <= vertex_limit:
          ssg_solution_mask = solve(ssg)
          combined_clique_size = len(ssg_solution_mask) + len(sg_pending_indices) + 1 # v_ind
          if combined_clique_size > lower_bound:
            lower_bound = combined_clique_size
            max_clique_indices = set(ssg_indices[ssg_solution_mask]) + sg_pending_indices + {v_ind}
        else:
          heapq.heappush(subgraphs, (-ssg.shape[0], ssg, ssg_indices, sg_pending_indices + {v_ind}))

    return max_clique_indices, lower_bound
    