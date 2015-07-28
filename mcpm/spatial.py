""" pixel neighborhood definitions """
import numpy as np

nbrlist = None
neighbors = None
nbr_range = np.arange(-1,2) 
lattice_basis = None

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

# def uniform_mask(sites, radius=1):
#   arr = np.array([[1,1,1],
#                   [1,1,1],
#                   [1,1,1]], dtype=float)
#   return arr.ravel()

# def uniform_mask(sites, radius=1):
#   arr = np.array([[0,0,0,0,0],
#                   [0,1,1,1,0],
#                   [0,1,1,1,0],
#                   [0,1,1,1,0],
#                   [0,0,0,0,0]], dtype=float)
#   return arr.ravel()

def strained_mask(sites, radius=1, strain=0.1):
  dims = tuple([2*radius+1 for __ in range(sites.ndim)])
  if dims != (3,3):
    import sys; sys.exit('strained_mask only implemented for 2D/8n square lattice')
  mask = np.array([[1,        1,  1],
                   [1+strain, 1, 1+strain],
                   [1,        1, 1]])
  return mask.ravel()

def nearest_neighbor_mask(radius, ndim):
  id_range = np.arange(-radius,radius+1)
  dist = np.meshgrid( *[id_range for d in range(ndim)] )
  square_dist = np.sum(np.square(list(dist)), axis=0)
  nearest = np.zeros_like(square_dist, dtype=bool)
  nearest[square_dist <= 2] = 1
  return nearest.ravel()


def lookup_neighbors(site, dims=None, radius=1):
  return nbrlist[site]


def meshgrid_neighbors(site, dims=None, radius=1):
  """ N-dimensional pixel neighborhood
      for periodic images on regular grids """
  index = np.unravel_index(site, dims=dims)
  id_range = [np.arange(idx-radius, idx+radius+1)
              for idx in index]
  neigh_ids = np.meshgrid(*id_range)
  neighs =  np.ravel_multi_index(neigh_ids,dims=dims, mode='wrap')
  return neighs.ravel()

def slice_neighbors(site, dims=None, radius=1):
  """ N-dimensional pixel neighborhood
      for periodic images on regular grids """
  index = np.unravel_index(site, dims=dims)
  islice = [(nbr_range + idx)%dims[i] for i,idx in enumerate(index)]
  neighs =  np.ravel_multi_index(islice,dims=dims, mode='wrap')
  return neighs.ravel()  

def ix_neighbors(site, dims=None, radius=1):
  """ use np.ix_ instead of np.meshgrid -- requires less memory """
  index = np.unravel_index(site, dims=dims)
  id_range = [np.arange(idx-radius, idx+radius+1)
              for idx in index]
  neighs = np.ravel_multi_index(np.ix_(*id_range),
                                dims=dims, mode='wrap')
  return neighs.ravel()

def view_neighbors(site, dims=None, radius=1):
  ishape = np.eye(len(dims))*(2*radius + 1)
  ishape[ishape == 0] += 1
  index = np.unravel_index(site, dims=dims)
  id_range = [np.arange(idx-radius, idx+radius+1).reshape(ishape[i])
              for i,idx in enumerate(index)]
  neighs = np.ravel_multi_index(tuple(id_range), dims=dims, mode='wrap')
  return neighs.ravel()

def view_neighbors_2d(site, dims=None, radius=1):
  ''' the listcomp in view_neighbors takes a lot of time. 
      here's a 2d-specific way to do it hopefully faster? '''
  index = np.unravel_index(site, dims=dims)
  id_range = ( (nbr_range+index[0]).reshape(lattice_basis[0]),
               (nbr_range+index[1]).reshape(lattice_basis[1]) )
  neighs = np.ravel_multi_index(id_range, dims=dims, mode='wrap')
  return neighs.ravel()

def view_neighbors_3d(site, dims=None, radius=1):
  ''' the listcomp in view_neighbors takes a lot of time. 
      here's a 3d-specific way to do it hopefully faster? '''
  index = np.unravel_index(site, dims=dims)
  id_range = ( (nbr_range+index[0]).reshape(lattice_basis[0]),
               (nbr_range+index[1]).reshape(lattice_basis[1]),
               (nbr_range+index[2]).reshape(lattice_basis[2]) )
  neighs = np.ravel_multi_index(id_range, dims=dims, mode='wrap')
  return neighs.ravel()

def roll_neighbors(sites, site, dims=None, radius=1):
  """ N-dimensional pixel neighborhood
      for periodic images on regular grids """
  index = np.unravel_index(site, dims=dims)
  neighs = sites.take(nbr_range+index, axis=0, mode='wrap')
  return neighs.flatten()  

def build_neighbor_list(sites, radius=1):
  global nbrlist
  if nbrlist is not None:
    print('nbrlist already exists')
    return
  global neighbors
  # neighbors = meshgrid_neighbors
  print('building neighbor list')
  check_neighs = neighbors(0, dims=sites.shape, radius=radius)
  num_neighs = check_neighs.size
  print('each site has {} neighbors'.format(num_neighs-1))
  
  nbrlist = np.zeros((sites.size, num_neighs),dtype=int)
  for index, __ in np.ndenumerate(sites):
    site = np.ravel_multi_index(index, sites.shape)
    nbrlist[site] = neighbors(site, dims=sites.shape, radius=radius)

  # reassign neighbors function to use lookup list
  neighbors = lookup_neighbors
  return

def load_neighbor_list(neighborpath):
  global nbrlist
  global neighbors
  nbrlist = np.load(neighborpath)
  neighbors = lookup_neighbors
  return

def setup(sites, options):
  global neighbors
  global nbr_range
  global lattice_basis
  if options.neighborfile:
    load_neighbor_list(options.neighborfile)
    return 
  # neighbors = meshgrid_neighbors
  neighbors = view_neighbors_2d if (sites.ndim == 2) else view_neighbors_3d
  nbr_range = np.arange(-options.radius, options.radius+1)
  lattice_basis = np.eye(len(sites.shape))*(2*options.radius + 1)
  lattice_basis[lattice_basis == 0] += 1

  if options.neighborlist:
    build_neighbor_list(sites, radius=options.radius)
  return
