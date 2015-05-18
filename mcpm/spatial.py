""" pixel neighborhood definitions """
import numpy as np

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
