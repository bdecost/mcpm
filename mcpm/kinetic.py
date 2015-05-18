""" Kinetic Monte Carlo solver """

import numpy as np
import pandas as pd
import argparse

from . import io
from . import spatial

radius = 1
dims = None

def site_propensity(site, nearest, kT, sites, weights):
  current_state = sites[site]
  neighs = spatial.neighbors(site, dims=dims, radius=radius)
  nearest_sites = neighs[nearest]
  nearest_states = sites[nearest_sites]
  states = pd.unique(nearest_states) # pd.unique faster than np.unique
  states = states[states != current_state]
  if states.size == 0:
    return 0

  delta = sites[neighs] != current_state
  current_energy = np.sum(np.multiply(delta, weights))

  prob = 0
  for proposed_state in states:
    sites[site] = proposed_state
    delta = sites[neighs] != proposed_state
    proposed_energy = np.sum(np.multiply(delta, weights))
    energy_change = proposed_energy - current_energy
  
    if energy_change <= 0:
      prob += 1
    elif kT > 0.0:
      prob += np.exp(-energy_change/kT)

  sites[site] = current_state
  return prob


def site_event(site, nearest, kT, weights, sites, propensity):
  threshold = np.random.uniform() * propensity[site]
  current_state = sites[site]
  neighs = spatial.neighbors(site, dims=dims, radius=radius)
  nearest_sites = neighs[nearest]
  states = pd.unique(sites[nearest_sites]) # pd.unique faster than np.unique
  states = states[states != current_state]

  delta = sites[neighs] != current_state
  current_energy = np.sum(np.multiply(delta, weights))

  prob = 0
  for proposed_state in states:
    sites[site] = proposed_state
    delta = sites[neighs] != proposed_state
    proposed_energy = np.sum(np.multiply(delta, weights))
    energy_change = proposed_energy - current_energy
  
    if energy_change <= 0:
      prob += 1
    elif kT > 0.0:
      prob += np.exp(-energy_change/kT)
    if prob >= threshold:
      break

  neighs = neighs[np.nonzero(weights)]
  for neigh in np.nditer(neighs):
    propensity[neigh] = site_propensity(neigh, nearest,
                                        kT, sites, weights)
  return


def select_site_iter(propensity):
  total_propensity = np.sum(propensity)
  index = np.argsort(propensity, axis=None)
  target = total_propensity * np.random.uniform()
  partial = np.array(0)
  # iterate in reverse:
  for site in np.nditer(index[::-1]):
    partial += propensity.ravel()[site]
    if partial >= target:
      break    
  time_step = -1.0/total_propensity * np.log(np.random.uniform())
  return site, time_step


def select_site(propensity):
  cumprop= np.cumsum(propensity)
  target = cumprop[-1] * np.random.uniform()
  site = np.searchsorted(cumprop, target)
  time_step = -1.0/cumprop[-1] * np.log(np.random.uniform())
  return site, time_step  


def all_propensity(sites, nearest, kT, weights):
  print('initializing kmc propensity')
  propensity = np.zeros(sites.shape, dtype=float)
  for site,__ in np.ndenumerate(sites.ravel()):
    propensity[site] = site_propensity(site, nearest, kT, sites, weights)
  return propensity


def iterate(sites, weights, options):
  global radius
  radius = options.radius
  global dims
  dims = sites.shape
  kT = options.kT
  length = options.length
  dump_frequency = options.freq

  time = 0
  neighbors = None
  if options.neighborlist:
    spatial.build_neighbor_list(sites, radius=radius)
  nearest = spatial.nearest_neighbor_mask(radius,sites.ndim)
  propensity = all_propensity(sites.ravel(),
                              nearest, kT, weights)
  while time < length:
    inner_time = 0
    print('time: {}'.format(time))
    io.dump_dream3d(sites, int(time))
    while inner_time < dump_frequency:
      site, time_step = select_site(propensity)
      site_event(site, nearest, kT, weights, sites.ravel(), propensity)
      inner_time += time_step
    time += inner_time
  io.dump_dream3d(sites, int(time))
  return time
