""" simple HDF5 i/o routines (DREAM3D format) """

import numpy as np
import h5py

GRAIN_ID_PATH = 'DataContainers/SyntheticVolume/CellData/FeatureIds'
QUATERNION_PATH = 'DataContainers/SyntheticVolume/CellFeatureData/AvgQuats'
PRNG_STATE_PATH = 'DataContainers/SytheticVolume/prng_state'
ARGS_PATH = 'DataContainers/SytheticVolume/mcpm_args'

def load_dream3d(path):
  with h5py.File(path, 'r') as f:
    grain_ids = np.array(f[GRAIN_ID_PATH])
  shape = tuple([s for s in grain_ids.shape if s > 1])
  return grain_ids.reshape(shape)

def dimensions(dim_line):
    min_val, max_val = dim_line.split()
    return int(float(max_val) - float(min_val))

def load_spparks(path):
  """ load microstructure from spparks text dump """
  grain_ids = None
  with open(path, 'r') as f:
    for line in f:
      if "TIMESTEP" in line:
        time = next(f) # float()
      elif "NUMBER" in line:
        num_sites = next(f) # float()
      elif "BOX" in line:
        x = dimensions(next(f))
        y = dimensions(next(f))
        z = dimensions(next(f))
        grain_ids = np.zeros((x,y,z))
        # skip forward two lines
        next(f); line = next(f)
      else:
        # get id, spin, x, y, z
        idx, grain_id, x, y, z = map(int, line.split())
        grain_ids[(x,y,z)] = grain_id
    shape = tuple([s for s in grain_ids.shape if s > 1])
    return grain_ids.reshape(shape)

def load_prng_state(path):
  state = list(np.random.get_state())
  with h5py.File(path) as f:
    try:
      saved_state = f[PRNG_STATE_PATH]
      state[1] = saved_state
      np.random.set_state(tuple(state))
    except KeyError:
      print('unable to load saved state')
      save_prng_state(path)      
  return

def save_prng_state(path):
  # RandomState is a tuple
  # the important part is the 624 element array of integers.
  state = np.random.get_state()
  with h5py.File(path) as f:
    try:
      f[PRNG_STATE_PATH][...] = state[1]
    except KeyError:
      f[PRNG_STATE_PATH] = state[1]
  return

def dump_dream3d(sites, time, prefix='mcpm_dump'):
  path = '{0}{1:06d}.dream3d'.format(prefix, time)
  with h5py.File(path) as f:
    f[GRAIN_ID_PATH] = sites
  return

def load_quaternions(path):
  with h5py.File(path, 'r') as f:
    quaternions = np.array(f[QUATERNION_PATH], dtype=np.float32)
  return quaternions

def save_args(args):
  """ save command line arguments """
  with h5py.File(args.infile) as f:
    try:
      f.create_group(ARGS_PATH)
    except ValueError:
      del f[ARGS_PATH]
    h_args = f[ARGS_PATH]
    for key, value in args.__dict__.items():
      h_args[key] = value

  return

def load_prev_args(args):
  """ load args from a previous run 
      modifies values in the arguments namespace """
  with h5py.File(args.infile) as f:
    h_args = f[ARGS_PATH]
    for key in args.__dict__.keys():
      args.__dict__[key] = h_args[key].value
  return args
