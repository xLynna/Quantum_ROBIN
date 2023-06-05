import numpy as np

# Without variable number limits
def naive_definite_transformation(G, n):
    """Transform a graph G into a QUBO matrix Q and bias b for 
       the maximum clique problem.
    
    Parameters
    ----------
    G : np.ndarray
        Adjacency matrix of the graph.
    n : int
        Number of nodes in the graph (number of measurements).
        
    Returns
    -------
    Q : numpy.ndarray
        QUBO matrix.
    b : numpy.ndarray
        Bias vector.
    """

    Q = np.zeros((n, n))
    b = -np.ones(n)

    Q[G == 0] = 1 # assign unit penalty to non-edges
    Q.fill_diagonal(0)

    penalty_reg = 2
    
    return penalty_reg * Q, b

# With variable number limits
def extract_k_core(G, k):
    """Extract the k-core of a graph G.
    
    Parameters
    ----------
    G : np.ndarray
        Adjacency matrix of the graph.
    k : int
        Number of nodes in the k-core.
        
    Returns
    -------
    G : np.ndarray
        Sparse adjacency matrix of the k-core.
    """
    G = G.copy()
    while True:
        degrees = np.sum(G, axis=0)
        reduce_mask = degrees < k
        if np.all(degrees[reduce_mask] == 0):
            break
        else:
            G[reduce_mask, :] = 0
            G[:, reduce_mask] = 0
    return G

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
    G : np.ndarray
        Adjacency matrix of the subgraph.
    """
    G = extract_k_core(G, lower_bound)
    vertex_indices = np.argwhere(np.sum(G, axis=0) > 0).flatten()
    v = np.random.choice(vertex_indices)
    v_neighbours_mask = G[v, :] == 1
    common_neighbours = np.logical_and(G, v_neighbours_mask).sum(axis=1, keepdims=True)
    # Remove the edges between the nodes if their common neighbours 
    # are less than the lower bound - 2
    G[common_neighbours < (lower_bound - 2)] = 0
    G = extract_k_core(G, lower_bound)
    return G
    