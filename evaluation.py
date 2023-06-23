import numpy as np

# Class to record the shuffle permutation for the point cloud
class Evaluator:
    pass

def overall_accuracy(n, inlier_perm_mask, solution_mask):
   return np.sum(inlier_perm_mask == solution_mask) / n

def inlier_preserved_rate(n, inlier_perm_mask, solution_mask):
  num_inliers = np.sum(inlier_perm_mask)
  return np.sum(inlier_perm_mask & solution_mask) / num_inliers

def outlier_rejected_rate(n, inlier_perm_mask, solution_mask):
  num_outliers = n - np.sum(inlier_perm_mask)
  return np.sum(~inlier_perm_mask.astype(bool) & ~solution_mask.astype(bool)) / num_outliers


def evaluate(n, inlier_perm_mask, solution_mask, display=True):
  acc = overall_accuracy(n, inlier_perm_mask, solution_mask)
  ipr = inlier_preserved_rate(n, inlier_perm_mask, solution_mask)
  orr = outlier_rejected_rate(n, inlier_perm_mask, solution_mask)
  if display:
    print("Overall Accuracy: ", acc)
    print("Inlier Preserved Rate: ", ipr)
    print("Outlier Rejected Rate: ", orr)
  
  return acc, ipr, orr