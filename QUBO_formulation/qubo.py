import numpy as np

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
    
    return penalty_reg*Q, b

    