import gurobipy as gp
import numpy as np

def _init_model(Q, bias):
  model = gp.Model("QUBO")
  bias = bias.reshape(-1)
  num_vars = len(bias)
  x = model.addVars(num_vars, vtype=gp.GRB.BINARY, name="x")
  quad_expr = gp.quicksum(Q[i, j] * x[i] * x[j] for i in range(num_vars) for j in range(num_vars))
  lin_expr = gp.quicksum(bias[i] * x[i] for i in range(num_vars))
  model.setObjective(quad_expr + lin_expr, gp.GRB.MINIMIZE)
  model.update()
  return model

def gurobi_solver(Q, bias):
  model = _init_model(Q, bias)
  model.optimize()
  if model.status == gp.GRB.OPTIMAL:
    return np.array([x.X for x in model.getVars()])
  else:
      raise ValueError("No solution found.")