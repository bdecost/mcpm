""" Grain boundary properties -- default to cubic symmetry """

import numpy as np
from misori import misori

from . import io

# grain growth model parameters
# find the best way to set these at runtime
high_angle = np.pi*30/180
mobility_ratio = None
energy_ratio = 0.5
quaternions = None
mobility_cache = {}


def energy(*args):
  return uniform_energy()


def mobility(*args):
  return uniform_mobility()


def uniform_mobility(*args):
  return 1


def uniform_energy(*args):
  return 1


def threshold_mobility(a, b):
  global mobility_cache
  key = tuple(sorted([a,b]))
  try:
    return mobility_cache[key]
  except KeyError:
    angle = misori(quaternions[a], quaternions[b])
    mobility = mobility_ratio if angle < high_angle else 1
    mobility_cache[key] = mobility
    return mobility


def threshold_energy(qa, qb):
  angle = misori(qa, qb)
  return energy_ratio if angle < high_angle else 1
  
def setup(options):
  global quaternions
  global mobility
  global mobility_ratio
  mobility_ratio = options.mobility
  if mobility_ratio != 1.0:
    quaternions = io.load_quaternions(options.infile)
    mobility = threshold_mobility

mobility = uniform_mobility
