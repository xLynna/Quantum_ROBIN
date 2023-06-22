import numpy as np
import itertools

def model(R, t, p, epsilon):
  """
  :param R: rotation matrix (2, d)
  :param t: translation vector (1, d)
  :param p: point cloud, (n, d)
  :param epsilon: noise, (n, d)
  :return: transformed point cloud
  """
  return p@R + t + epsilon

# pairwise compatibility test
def pairwise_compatibility_value(a, b):
  """
  b = model(R, t, a, epsilon)
  :param a: point cloud (n, d)
  :param b: point cloud (n, d)
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
  return (E < 2*beta).astype(int)

# 3-invariant compatibility test
def area_invariant_compatibility_test(a, b, beta, fail_edge=False):
  """
  :param a: point cloud (n, d), ground truth
  :param b: point cloud (n, d)
  :return: area invariant compatibility test
  """

  n = a.shape[0]
  tests = []

  for comb in itertools.combinations(range(n), 3):
    A = a[comb, :]
    B = b[comb, :]
    min_area, max_area = extreme_area(A, beta)
    area = area_by_coord(B)

    if fail_edge ^ (min_area < area < max_area):
      # record the successful/failed test
      zeros = np.zeros(n)
      zeros[comb, ] = 1
      tests.append(zeros.astype(int))
  
  return np.array(tests)
   
def _is_colinear(V):
  """
  :param V: three points (3, d)
  :return: True if three points are colinear
  """

  vectors = V[1:] - V[0]

  # Check if all vectors are parallel
  return np.allclose(np.cross(vectors[0], vectors[1]), 0)

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
  :param V: three points (3, d)
  :param beta: threshold
  :return: The minimal and the maximal area of the triangle
  """
  n = V.shape[0]

  if _is_colinear(V):
    diff = list(map(lambda x: x[0] - x[1], V[list(itertools.combinations(range(n), 2))]))
    max_base = np.max(np.linalg.norm(diff, axis=1))
    return 0, (max_base + 2 * beta) * beta
  
  x, y, z = V
  height = _calculate_height(x, y, z)
  base = np.linalg.norm(y - x)
  return np.maximum(0.0, 0.5 * (base - 2 * beta) * (height - 2 * beta)), 0.5 * (base + 2 * beta) * (height + 2 * beta)

def area_by_coord(V):
  vectors = V[1:] - V[0]
  return np.linalg.norm(np.cross(vectors[0], vectors[1])) / 2
