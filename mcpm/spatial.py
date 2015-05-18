""" pixel neighborhood definitions """
import numpy as np

nbrlist = None
neighbors = None

def gaussian_mask(sites, radius, sigma_squared=1, a=None, cutoff=0.01):
  if a is None:
    a = 1/(np.sqrt(2*sigma_squared*np.pi))
  id_range = np.arange(-radius,radius+1)
  dist = np.meshgrid( *[id_range for d in range(sites.ndim)] )
  square_dist = np.sum(np.square(list(dist)), axis=0)
  weights = a * np.exp(-0.5 * square_dist / sigma_squared)
  weights[weights < cutoff] = 0
  return weights.ravel()


def uniform_mask(sites, radius=1):
  dims = tuple([2*radius+1 for __ in range(sites.ndim)])
  return np.ones(dims).ravel()


def nearest_neighbor_mask(radius, ndim):
  id_range = np.arange(-radius,radius+1)
  dist = np.meshgrid( *[id_range for d in range(ndim)] )
  square_dist = np.sum(np.square(list(dist)), axis=0)
  nearest = np.zeros_like(square_dist, dtype=bool)
  nearest[square_dist <= 2] = 1
  return nearest.ravel()


def lookup_neighbors(site, dims=None, radius=1):
  return nbrlist[site]


def neighbors(site, dims=None, radius=1):
  """ N-dimensional pixel neighborhood
      for periodic images on regular grids """
  index = np.unravel_index(site, dims=dims)
  id_range = [np.arange(idx-radius, idx+radius+1)
              for idx in index]
  neigh_ids = np.meshgrid(*id_range)
  neighs =  np.ravel_multi_index(neigh_ids,dims=dims, mode='wrap')
  return neighs.ravel()


def build_neighbor_list(sites, radius=1):
  global nbrlist
  global neighbors
  print('building neighbor list')
  check_neighs = neighbors(0, dims=sites.shape, radius=radius)
  num_neighs = check_neighs.size

  nbrlist = np.zeros((sites.size, num_neighs),dtype=int)
  for site, __ in np.ndenumerate(sites):
    nbrlist[site] = neighbors(site, dims=sites.shape, radius=radius)

  # reassign neighbors function to use lookup list
  neighbors = lookup_neighbors
  return
