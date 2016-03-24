""" Grain boundary properties -- default to cubic symmetry """

import numpy as np
from misori import misori

from . import io

# grain growth model parameters
# find the best way to set these at runtime
high_angle = np.pi*30/180
mobility_ratio = None
energy_ratio = 0.6
quaternions = None
colors = None
mobility_cache = {}
energy_cache = {}

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

def discrete_texture_mobility(a, b):
  global mobility_cache
  key = tuple(sorted([a,b]))
  try:
    return mobility_cache[key]
  except KeyError:
    mobility = 1.0
    if colors[a] == 0 or colors[b] == 0:
      pass
    elif colors[a] == colors[b]:
        mobility = mobility_ratio
    mobility_cache[key] = mobility
    return mobility

def setup_discrete_texture_energy(colors, energy_ratio):
    """ make a lookup table for energy values """
    energy_table = np.ones((len(colors), len(colors)), dtype=float)

    for color in np.unique(colors):
        if color > 0:
            idx = np.where(colors == color)
            y,x = np.meshgrid(idx,idx)
            energy_table[x,y] = energy_ratio

    np.fill_diagonal(energy_table, 0)
    
    return energy_table

def discrete_texture_energy(neighstates, refstate):
  global energy_cache
  return energy_cache[neighstates, refstate]

def setup(options):
  global quaternions
  global colors
  global mobility
  global mobility_ratio
  global energy
  global energy_ratio
  global high_angle
  mobility_ratio = options.mobility
  energy_ratio = options.energy
  
  if options.discrete == True:
    colors = io.load_colors(options.infile)
    # mobility = discrete_texture_mobility
    global energy
    global energy_cache
    energy_cache = setup_discrete_texture_energy(colors)
    energy = discrete_texture_energy
  else:
    if mobility_ratio != 1.0:
      quaternions = io.load_quaternions(options.infile)
      mobility = threshold_mobility
  high_angle = options.angle*np.pi/180
  
mobility = uniform_mobility
