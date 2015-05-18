""" Rejection Monte Carlo solver """

import numpy as np
import pandas as pd
import argparse

from . import io
from . import spatial

def site_event(site, kT, sites, weights):
  s = sites.ravel()
  current_state = s[site]
  nearest = spatial.neighbors(site, None, dims=sites.shape, radius=1)
  states = np.unique(s[nearest])
  states = states[states != current_state]
  if states.size == 0:
    return current_state

  neighs = spatial.neighbors(site, None, dims=sites.shape, radius=radius)
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


def timestep(sites, kT, weights):
  rejects = 0
  s = sites.ravel()
  for i in range(sites.size):
    site = np.random.randint(sites.size)
    current = s[site]
    s[site] = site_event(site, kT, sites, weights)
    rejects += (current == s[site])
  return rejects


def iterate(sites, weights, options):
  radius = options.radius
  kT = options.kT
  length = options.length
  dump_frequency = options.freq

  rejects = 0
  for time in np.arange(0, length+1, dump_frequency):
    print('time: {}'.format(time))
    io.dump_dream3d(sites, time)
    accepts = time*sites.size - rejects
    print('accepts: {}, rejects: {}'.format(accepts,rejects))
    for step in range(dump_frequency):
      rej = timestep(sites, kT, weights)
      rejects += rej
  io.dump_dream3d(sites, time)
  return
