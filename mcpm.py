#!/usr/bin/env python

import numpy as np
import h5py
import pandas as pd
import argparse

GRAIN_ID_PATH = 'DataContainers/SyntheticVolume/CellData/FeatureIds'

def load_dream3d(path):
  f = h5py.File(path)
  grain_ids = np.array(f[GRAIN_ID_PATH])
  f.close()
  shape = tuple([s for s in grain_ids.shape if s > 1])
  return grain_ids.reshape(shape)


def dump_dream3d(sites, time):
  path = 'dump{0:06d}.dream3d'.format(time)
  f = h5py.File(path)
  f[GRAIN_ID_PATH] = sites
  f.close()
  return


def gaussian_mask(sites, dist, sigma_squared=1, a=None, cutoff=0.01):
  if a is None:
    a = 1/(np.sqrt(2*sigma_squared*np.pi))
  id_range = np.arange(-dist,dist+1)
  dist = np.meshgrid( *[id_range for d in range(sites.ndim)] )
  square_dist = np.sum(np.square(list(dist)), axis=0)
  weights = a * np.exp(-0.5 * square_dist / sigma_squared)
  weights[weights < cutoff] = 0
  return weights.flatten()


def uniform_mask(sites, dist=1):
  dims = tuple([2*dist+1 for __ in range(sites.ndim)])
  return np.ones(dims).flatten()
    
def nearest_neighbor_mask(dist, ndim):
  id_range = np.arange(-dist,dist+1)
  dist = np.meshgrid( *[id_range for d in range(ndim)] )
  square_dist = np.sum(np.square(list(dist)), axis=0)
  nearest = np.zeros_like(square_dist, dtype=bool)
  nearest[square_dist <= 2] = 1
  return nearest.flatten()


def site_neighbors(site, dims=None, dist=1):
  # square/cubic pixel neighborhood of 'radius' dist
  # for 2 and 3 dimensional periodic images
  index = np.unravel_index(site, dims=dims)
  id_range = [np.arange(idx-dist, idx+dist+1)
              for idx in index]
  neigh_ids = np.meshgrid(*id_range)
  neighs =  np.ravel_multi_index(neigh_ids,dims=dims, mode='wrap')
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
  neighs = neighbors[site]
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


def kmc_event(site, neighbors, nearest, kT, weights, sites, propensity):
  threshold = np.random.uniform() * propensity[site]
  current_state = sites[site]
  neighs = neighbors[site] # indices into sites
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


def iterate_kmc(sites, weights, options):
  dist = options.radius
  kT = options.kT
  length = options.length
  dump_frequency = options.freq
  
  time = 0
  neighbors = neighbor_list(sites, dist=dist)
  nearest = nearest_neighbor_mask(dist,sites.ndim)
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


def iterate_rejection(sites, kT, weights, length, dist=1):
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
  parser = argparse.ArgumentParser(prog='mcpm',
             description='''Kinetic Monte Carlo grain growth
                       simulations in 2 and 3 dimensions.''')

  parser.add_argument('--infile', nargs='?', default='input.dream3d',
                      help='DREAM3D file containing initial structure')
  parser.add_argument('--style', default='kmc',
                      choices=['kmc', 'reject'],
                      help='Monte Carlo style')
  parser.add_argument('--nbrhd', nargs='?', default='gaussian',
                      choices=['uniform', 'gaussian'],
                      help='pixel neighborhood weighting')
  parser.add_argument('--sigma', type=float, default=3,
                      help='smoothing parameter for gaussian neighborhood')
  parser.add_argument('--radius', type=int, default=9,
                      help='pixel neighborhood radius')
  parser.add_argument('--kT', type=float, default=.001,
                      help='Monte Carlo temperature kT')
  parser.add_argument('--length', type=float, default=100,
                      help='Simulation length in MCS')
  parser.add_argument('--cutoff', type=float, default=0.01,
                      help='interaction weight cutoff value')
  parser.add_argument('--norm', type=float, default=1,
                      help='gaussian kernel normalization constant a')
  parser.add_argument('--freq', type=float, default=10,
                      help='timesteps between system snapshots')
  parser.add_argument('--neighborlist', action='store_true',
                      help='''Compute explicit neighbor lists.
                              Problematic with large 3D systems.''')
  
  args = parser.parse_args()
  sites = load_dream3d(args.infile)

  if args.nbrhd == 'uniform':
    weights = uniform_mask(sites, dist=1)
  elif args.nbrhd == 'gaussian':
    weights = gaussian_mask(sites, args.radius, a=args.norm,
                sigma_squared=np.square(args.sigma))

  if args.style == 'reject':
    iterate_rejection(sites, args.kT, weights, args.length, dist=args.radius)
  elif args.style == 'kmc':
    iterate_kmc(sites, weights, args)
    
