#!/usr/bin/env python

import numpy as np
import h5py

dist = 9

def load_dream3d(path):
  f = h5py.File(path)
  grain_ids = np.array(f['DataContainers/SyntheticVolume/CellData/FeatureIds'])
  f.close()
  return grain_ids.reshape((grain_ids.shape[0], grain_ids.shape[1]))


def dump_dream3d(sites, time):
  path = 'dump{0:06d}.dream3d'.format(time)
  f = h5py.File(path)
  f['DataContainers/SyntheticVolume/CellData/FeatureIds'] = sites
  f.close()
  return


def weight_mask(dist, sigma_squared=1, a=None, cutoff=0.01):
  if a is None:
    a = 1/(np.sqrt(2*sigma_squared*np.pi))
  pos = np.arange(-dist, dist+1)
  dist_x, dist_y = np.meshgrid(pos, pos)
  square_dist = dist_x**2 + dist_y**2
  weights = a * np.exp(-0.5 * square_dist / sigma_squared)
  weights[weights < cutoff] = 0
  return weights


def uniform_weight_mask(dist, sigma_squared=1, a=None):
  return np.ones((2*dist+1, 2*dist+1))


def site_neighbors(idx, idy, dims=None, dist=1):
  x_range = np.arange(idx - dist, idx + dist + 1)
  y_range = np.arange(idy - dist, idy + dist + 1)
  neigh_idx, neigh_idy = np.meshgrid(x_range, y_range)
  neighs = np.ravel_multi_index((neigh_idx, neigh_idy), dims=dims, mode='wrap')
  return np.unravel_index(neighs, dims=dims)


def site_energy(idx, idy, kT, sites, weights):
  current_state = sites[idx, idy]
  neighs = site_neighbors(idx, idy,
                          dims=sites.shape,
                          dist=dist)
  delta = sites[neighs] != current_state
  current_energy = np.sum(np.multiply(delta, weights))
  return current_energy


def energy_map(sites, kT, weights):
  e = np.zeros_like(sites)
  for site_id,s in np.ndenumerate(sites):
    e[site_id] = site_energy(site_id[0], site_id[1], kT, sites, weights)
  return e


def site_propensity(idx, idy, kT, sites, weights):
  current_state = sites[idx, idy]
  nearest = site_neighbors(idx, idy,
                           dims=sites.shape,
                           dist=1)
  states = np.unique(sites[nearest])
  states = states[states != current_state]
  if states.size == 0:
    return 0
  neighs = site_neighbors(idx, idy,
                          dims=sites.shape,
                          dist=dist)

  delta = sites[neighs] != current_state
  current_energy = np.sum(np.multiply(delta, weights))

  prob = 0
  for proposed_state in states:
    sites[idx, idy] = proposed_state
    delta = sites[neighs] != proposed_state
    proposed_energy = np.sum(np.multiply(delta, weights))
    energy_change = proposed_energy - current_energy
  
    if energy_change <= 0:
      prob += 1
    elif kT > 0.0:
      prob += np.exp(-energy_change/kT)

  sites[idx, idy] = current_state
  return prob


def kmc_event(site, kT, weights, sites, propensity):
  idx, idy = np.unravel_index(site, dims=sites.shape)
  threshold = np.random.uniform() * propensity[idx,idy]
  current_state = sites[idx, idy]
  nearest = site_neighbors(idx, idy,
                           dims=sites.shape,
                           dist=1)
  states = np.unique(sites[nearest])
  states = states[states != current_state]

  neighs = site_neighbors(idx, idy,
                          dims=sites.shape,
                          dist=dist)

  delta = sites[neighs] != current_state
  current_energy = np.sum(np.multiply(delta, weights))

  prob = 0
  for proposed_state in states:
    sites[idx, idy] = proposed_state
    delta = sites[neighs] != proposed_state
    proposed_energy = np.sum(np.multiply(delta, weights))
    energy_change = proposed_energy - current_energy
  
    if energy_change <= 0:
      prob += 1
    elif kT > 0.0:
      prob += np.exp(-energy_change/kT)
    if prob >= threshold:
      break
  neighs_x, neighs_y = np.array(neighs[0]), np.array(neighs[1])
  mask = np.nonzero(weights)
  neighs_x, neighs_y = neighs_x[mask], neighs_y[mask]
  for nx, ny in np.nditer([neighs_x, neighs_y]):
    propensity[nx,ny] = site_propensity(nx, ny, kT, sites, weights)
    
  return


def kmc_select(propensity):
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


def kmc_select_sort(propensity):
  cumprop= np.cumsum(propensity)
  target = cumprop[-1] * np.random.uniform()
  site = np.searchsorted(cumprop, target)
  time_step = -1.0/cumprop[-1] * np.log(np.random.uniform())
  return site, time_step  


def kmc_init(sites, kT, weights):
  print('initializing kmc data structures')
  propensity = np.zeros(sites.shape, dtype=float)
  for site,__ in np.ndenumerate(sites):
    idx, idy = site[0], site[1]
    propensity[site] = site_propensity(idx, idy, kT, sites, weights)
  return propensity


def iterate_kmc(sites, kT, weights, length):
  time = 0
  dump_frequency = 10
  propensity = kmc_init(sites, kT, weights)
  while time < length:
    inner_time = 0
    print('time: {}'.format(time))
    dump_dream3d(sites, int(time))
    while inner_time < dump_frequency:
      site, time_step = kmc_select_sort(propensity)
      kmc_event(site, kT, weights, sites, propensity)
      inner_time += time_step
    time += inner_time
  dump_dream3d(sites, int(time))
  return time


def rejection_event(idx, idy, kT, sites, weights):
  current_state = sites[idx, idy]
  nearest = site_neighbors(idx, idy,
                           dims=sites.shape,
                           dist=1)
  states = np.unique(sites[nearest])
  states = states[states != current_state]
  if states.size == 0:
    return current_state

  neighs = site_neighbors(idx, idy,
                          dims=sites.shape,
                          dist=dist)
  delta = sites[neighs] != current_state
  current_energy = np.sum(np.multiply(delta, weights))

  proposed_state = np.random.choice(states)
  sites[idx, idy] = proposed_state

  delta = sites[neighs] != proposed_state
  proposed_energy = np.sum(np.multiply(delta, weights))
  sites[idx, idy] = current_state
  energy_change = proposed_energy - current_energy
  
  if energy_change > 0:
    if kT == 0:
      return current_state
    elif np.random.uniform() > np.exp(-energy_change/kT):
      return current_state

  return proposed_state


def rejection_timestep(sites, kT, weights):
  rejects = 0
  for i in range(sites.size):
    idx = np.random.randint(sites.shape[0])
    idy = np.random.randint(sites.shape[1])
    current = sites[idx, idy]
    sites[idx, idy] = rejection_event(idx, idy, kT, sites, weights)
    rejects += (current == sites[idx, idy])
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
  weights = weight_mask(dist,
                        sigma_squared=np.square(3),
                        a=1)
  length = 2000
  # iterate_rejection(sites, kT, weights, length)
  iterate_kmc(sites, kT, weights, length)
