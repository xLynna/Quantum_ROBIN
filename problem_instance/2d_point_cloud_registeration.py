import numpy as np
import itertools

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

def area_invariant_compatibility_test(a, b):
  """
  :param a: point cloud (n, 2), ground truth
  :param b: point cloud (n, 2)
  :return: area invariant compatibility test
  """

  n = a.shape[0]
  failed_tests = []

  for comb in itertools.combinations(range(n), 3):
    A = a[comb]
    B = b[comb]
    min_area, max_area = extreme_area(A, 0.5)
    area = area_by_coord(B)
    if min_area < area < max_area:
      continue
    else:
      zeros = np.zeros(n)
      zeros[comb] = 1
      failed_tests.append(zeros)
  
  return np.array(failed_tests)

    
def _is_colinear(V):
  """
  :param V: three points (3, 2)
  :return: True if three points are colinear
  """
  x, y, z = V
  return (z[1] - y[1]) * (y[0] - x[0])  - (y[1] - x[1]) * (z[0] - y[0]) < 1e-5


def _calculate_height(x, y, z):
    base_vector = np.array(y) - np.array(x)
    norm_base_vector = base_vector / np.linalg.norm(base_vector)
    vertex_vector = np.array(z) - np.array(x)

    # Calculate the height vector by subtracting the projection of the vertex vector onto the base vector
    projection = np.dot(vertex_vector, norm_base_vector)
    height_vector = vertex_vector - projection * norm_base_vector

    # Calculate the height as the magnitude of the height vector
    height = np.linalg.norm(height_vector)

    return height

def extreme_area(V, beta):
  """
  :param V: three points (3, 2)
  :param beta: threshold
  :return: The minimal and the maximal area of the triangle
  """
  n = V.shape[0]

  if _is_colinear(V):
    diff = list(map(lambda x: x[0] - x[1], V[list(itertools.combinations(range(n), 2))]))
    max_base = np.max(np.norm(diff, axis=1))
    return 0, (max_base + 2 * beta) * beta
  
  x, y, z = V
  height = _calculate_height(x, y, z)
  base = np.linalg.norm(y - x)
  return 0.5 * (base - 2 * beta) * (height - 2 * beta), 0.5 * (base + 2 * beta) * (height + 2 * beta)

def area_by_coord(V):
  x, y, z = V
  return abs((x[0]*(y[1]-z[1]) + y[0]*(z[1]-x[1]) + z[0]*(x[1]-y[1])) / 2)
