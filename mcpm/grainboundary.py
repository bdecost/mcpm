""" Grain boundary properties -- default to cubic symmetry """

import numpy as np
from misori import misori


def uniform_mobility(qa, qb):
  return 1


def uniform_energy(qa, qb):
  return 1


def threshold_mobility(qa, qb, threshold=np.pi*30/180, ratio=0.001):
  angle = misori(qa, qb)
  return ratio if angle < threshold else 1


def threshold_energy(qa, qb, threshold=np.pi*30/180, ratio=0.5):
  angle = misori(qa, qb)
  return ratio if angle < threshold else 1
  
