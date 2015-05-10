#!/usr/bin/env python

import numpy as np
import h5py
import pandas as pd

dist = 9
GRAIN_ID_PATH = 'DataContainers/SyntheticVolume/CellData/FeatureIds'

def load_dream3d(path):
  f = h5py.File(path)
  grain_ids = np.array(f[GRAIN_ID_PATH])
  f.close()
  return grain_ids.reshape((grain_ids.shape[0], grain_ids.shape[1]))


def dump_dream3d(sites, time):
  path = 'dump{0:06d}.dream3d'.format(time)
  f = h5py.File(path)
  f[GRAIN_ID_PATH] = sites
  f.close()
  return


def gaussian_mask(dist, sigma_squared=1, a=None, cutoff=0.01):
  if a is None:
    a = 1/(np.sqrt(2*sigma_squared*np.pi))
  pos = np.arange(-dist, dist+1)
  dist_x, dist_y = np.meshgrid(pos, pos)
  square_dist = dist_x**2 + dist_y**2
  weights = a * np.exp(-0.5 * square_dist / sigma_squared)
  weights[weights < cutoff] = 0
  return weights.flatten()


def uniform_mask(dist, sigma_squared=1, a=None):
  return np.ones((2*dist+1, 2*dist+1)).flatten()


def site_neighbors(site, dims=None, dist=1):
  idx, idy = np.unravel_index(site, dims=dims)
  x_range = np.arange(idx - dist, idx + dist + 1)
  y_range = np.arange(idy - dist, idy + dist + 1)
  neigh_idx, neigh_idy = np.meshgrid(x_range, y_range)
  neighs =  np.ravel_multi_index((neigh_idx, neigh_idy),
                                 dims=dims, mode='wrap')
  return neighs.flatten()


def neighbor_list(sites, dist=1):
  print('building neighbor list')
  check_neighs = site_neighbors(0, dims=sites.shape, dist=dist)
  num_neighs = check_neighs.size
  neighbors = np.zeros((sites.size, num_neighs),dtype=int)
  for site in np.arange(sites.size):
    neighbors[site] = site_neighbors(site, dims=sites.shape, dist=dist)

  return neighbors


def site_energy(site, kT, sites, weights):
  s = sites.ravel()
  current_state = s[site]
  neighs = site_neighbors(site, dims=sites.shape, dist=dist)
  delta = s[neighs] != current_state
  current_energy = np.sum(np.multiply(delta, weights))
  return current_energy


def energy_map(sites, kT, weights):
  energy = np.zeros_like(sites)
  e = energy.ravel()
  for site_id in np.ndindex(sites.ravel()):
    e[site_id] = site_energy(site_id, kT, sites, weights)
  return energy


def site_propensity(site, neighbors, nearest, kT, sites, weights):
  current_state = sites[site]
  nearest_sites = nearest[site]
  nearest_states = sites[nearest_sites]
  states = pd.unique(nearest_states) # pd.unique faster than np.unique
  states = states[states != current_state]
  if states.size == 0:
    return 0
  neighs = neighbors[site]

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


def kmc_event(site, neighbors, nearest, kT, weights, sites, propensity):
  threshold = np.random.uniform() * propensity[site]
  current_state = sites[site]

  states = pd.unique(sites[nearest[site]]) # pd.unique faster than np.unique
  states = states[states != current_state]

  neighs = neighbors[site]
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
    propensity[neigh] = site_propensity(neigh, neighbors, nearest,
                                        kT, sites, weights)
  return


def kmc_select_iter(propensity):
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


def kmc_select(propensity):
  cumprop= np.cumsum(propensity)
  target = cumprop[-1] * np.random.uniform()
  site = np.searchsorted(cumprop, target)
  time_step = -1.0/cumprop[-1] * np.log(np.random.uniform())
  return site, time_step  


def kmc_all_propensity(sites, neighbors, nearest, kT, weights):
  print('initializing kmc propensity')
  propensity = np.zeros(sites.shape, dtype=float)
  for site,__ in np.ndenumerate(sites.ravel()):
    propensity[site] = site_propensity(site, neighbors, nearest, kT, sites, weights)
  return propensity


def iterate_kmc(sites, kT, weights, length):
  time = 0
  dump_frequency = 1
  neighbors = neighbor_list(sites, dist=dist)
  nearest = neighbor_list(sites, dist=1)
  propensity = kmc_all_propensity(sites.ravel(),
                                  neighbors, nearest, kT, weights)
  while time < length:
    inner_time = 0
    print('time: {}'.format(time))
    dump_dream3d(sites, int(time))
    while inner_time < dump_frequency:
      site, time_step = kmc_select(propensity)
      kmc_event(site, neighbors, nearest, kT, weights, sites.ravel(), propensity)
      inner_time += time_step
    time += inner_time
  dump_dream3d(sites, int(time))
  return time


def rejection_event(site, kT, sites, weights):
  s = sites.ravel()
  current_state = s[site]
  nearest = site_neighbors(site, dims=sites.shape, dist=1)
  states = np.unique(s[nearest])
  states = states[states != current_state]
  if states.size == 0:
    return current_state

  neighs = site_neighbors(site, dims=sites.shape, dist=dist)
  delta = s[neighs] != current_state
  current_energy = np.sum(np.multiply(delta, weights))

  proposed_state = np.random.choice(states)
  s[site] = proposed_state

  delta = s[neighs] != proposed_state
  proposed_energy = np.sum(np.multiply(delta, weights))
  s[site] = current_state
  energy_change = proposed_energy - current_energy
  
  if energy_change > 0:
    if kT == 0:
      return current_state
    elif np.random.uniform() > np.exp(-energy_change/kT):
      return current_state

  return proposed_state


def rejection_timestep(sites, kT, weights):
  rejects = 0
  s = sites.ravel()
  for i in range(sites.size):
    site = np.random.randint(sites.size)
    current = s[site]
    s[site] = rejection_event(site, kT, sites, weights)
    rejects += (current == s[site])
  return rejects


def iterate_rejection(sites, kT, weights, length):
  dump_frequency = 10
  rejects = 0
  for time in np.arange(0, length+1, dump_frequency):
    print('time: {}'.format(time))
    dump_dream3d(sites, time)
    accepts = time*sites.size - rejects
    print('accepts: {}, rejects: {}'.format(accepts,rejects))
    for step in range(dump_frequency):
      rej = rejection_timestep(sites, kT, weights)
      rejects += rej
  dump_dream3d(sites, time)
  return


if __name__ == '__main__':
  # input_path = 'test.dream3d'
  input_path = 'input.dream3d'
  sites = load_dream3d(input_path)
  kT = 10**-2
  weights = gaussian_mask(dist,
                          sigma_squared=np.square(3),
                          a=1)
  length = 10
  # iterate_rejection(sites, kT, weights, length)
  iterate_kmc(sites, kT, weights, length)
