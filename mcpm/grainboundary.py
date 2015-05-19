""" Grain boundary properties -- default to cubic symmetry """

import numpy as np
from misori import misori

# grain growth model parameters
# find the best way to set these at runtime
high_angle = np.pi*30/180
mobility_ratio = 0.001
energy_ratio = 0.5

def energy(*args):
  return uniform_energy()


def mobility(*args):
  return uniform_mobility()


def uniform_mobility(*args):
  return 1


def uniform_energy(*args):
  return 1


def threshold_mobility(qa, qb):
  angle = misori(qa, qb)
  return mobility_ratio if angle < high_angle else 1


def threshold_energy(qa, qb):
  angle = misori(qa, qb)
  return energy_ratio if angle < high_angle else 1
  
