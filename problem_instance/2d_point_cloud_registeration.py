import numpy as np

def model(R, t, p, epsilon):
  """
  :param R: rotation matrix (2, 2)
  :param t: translation vector (1, 2)
  :param p: point cloud, (n, 2)
  :param epsilon: noise, (n, 1)
  :return: transformed point cloud
  """
  return R.dot(p) + t + epsilon

def pairwise_compatibility_value(a, b):
  """
  b = model(R, t, a, epsilon)
  :param a: point cloud (n, 2)
  :param b: point cloud (n, 2)
  :return: pairwise error between point clouds
  """
  A = np.linalg.norm(a[np.newaxis, :, :] - a[:, np.newaxis, :], axis=2)
  B = np.linalg.norm(b[np.newaxis, :, :] - b[:, np.newaxis, :], axis=2)

  return np.abs(B - A)

def definite_graph(E, beta):
  """
  :param E: pairwise compatibility value (n, n)
  :param beta: threshold
  :return: Adjacency matrix of the graph
  """
  return (E < beta).astype(int)
    
    

