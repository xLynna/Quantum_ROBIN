import numpy as np
import neal
from dimod.reference.samplers import ExactSolver
import itertools

def _input_formmater(Q, bias):
  bias = bias.reshape(-1).tolist()
  num_vars = len(bias)
  indicies = np.arange(num_vars)
  indices_pairs = itertools.product(indicies, indicies)
  Q = dict(zip(indices_pairs, Q.flatten()))
  return bias, Q

def _binarise_variables(variables):
  return [1 if x == 1 else 0 for x in variables]

def dwave_exactsolver(Q, bias):
  sampler = ExactSolver()
  bias, Q = _input_formmater(Q, bias)
  response = sampler.sample_ising(bias, Q)
  solution = response.first.sample.values()
  return _binarise_variables(solution)

def dwave_annealing_solver(Q, bias):
  sampler = neal.SimulatedAnnealingSampler()
  bias, Q = _input_formmater(Q, bias)
  response = sampler.sample_ising(bias, Q)
  solution = response.first.sample.values()
  return _binarise_variables(solution)