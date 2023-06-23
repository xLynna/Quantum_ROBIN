import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evaluation import evaluate
from problem_instance.point_cloud_registeration import model, pairwise_compatibility_value, definite_graph, area_invariant_compatibility_test
from QUBO_formulation.qubo import solve_graph, solve_weighted_graph, hypergraph_to_qubo, weighted_graph_to_qubo, n_invariant_ordinary_graph_to_qubo
from QUBO_formulation.limited_vertex import split
from Solvers.solver import solve_qubo
import time

def case1f(coordinates, new_coord_with_outliers, beta, solver, api_key=None):
  """Case 1f: 2-invariant compatibility test + ordinary graph"""
  weigthed_G = pairwise_compatibility_value(coordinates, new_coord_with_outliers)
  G = definite_graph(weigthed_G, beta)
  return solve_graph(G, solver, penalty_reg=2, api_key=api_key)

def case1s(coordinates, new_coord_with_outliers, beta, solver, api_key=None):
  """Case 1s: 2-invariant compatibility test + ordinary graph + graph splitting"""
  num_points = coordinates.shape[0]
  weigthed_G = pairwise_compatibility_value(coordinates, new_coord_with_outliers)
  G = definite_graph(weigthed_G, beta)
  sslg, _ = split(G, 45, 0, solver, api_key=api_key)
  sol = np.zeros(num_points, dtype=bool)
  sol[sslg] = True
  return sol

def case2(coordinates, new_coord_with_outliers, beta, solver, api_key=None):
  """Case 2: 2-invariant compatibility test + weighted graph"""
  weigthed_G = pairwise_compatibility_value(coordinates, new_coord_with_outliers)
  return solve_weighted_graph(weigthed_G, solver, 1, api_key=api_key)

def case3a(coordinates, new_coord_with_outliers, beta, solver, api_key=None):
  """Case 3af: 3-invariant compatibility test + ordinary graph"""
  num_points = coordinates.shape[0]
  H = area_invariant_compatibility_test(coordinates, new_coord_with_outliers, beta, fail_edge=False)
  Q, b = n_invariant_ordinary_graph_to_qubo(H, 1/num_points, weighted=True)
  return solve_qubo(Q, b, solver, api_key=api_key)

def case3b(coordinates, new_coord_with_outliers, beta, solver, api_key=None):
  """Case 3b: 3-invariant compatibility test + ordinary weighted graph"""
  num_points = coordinates.shape[0]
  H = area_invariant_compatibility_test(coordinates, new_coord_with_outliers, beta, fail_edge=False)
  Q, b = n_invariant_ordinary_graph_to_qubo(H, 1/num_points)
  return solve_qubo(Q, b, solver, api_key=api_key)

def case4(coordinates, new_coord_with_outliers, beta, solver, api_key=None):
  """Case 4: 3-invariant compatibility test + hyper graph"""
  num_points = coordinates.shape[0]
  H = area_invariant_compatibility_test(coordinates, new_coord_with_outliers, beta, fail_edge=True)
  Q, b = hypergraph_to_qubo(H, 2)
  return solve_qubo(Q, b, solver)[:num_points]