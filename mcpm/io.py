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
      f.create_group(ARGSPATH)
    except ValueError:
      del f[ARGSPATH]
    h_args = f[ARGSPATH]
    for key, value in args.__dict__.items():
      h_args[key] = value

  return

def load_prev_args(args):
  """ load args from a previous run 
      modifies values in the arguments namespace """
  with h5py.File(args.infile) as f:
    h_args = f[ARGSPATH]
    for key in args.__dict__.keys():
      args.__dict__[key] = h_args[key].value
  return args
