""" simple HDF5 i/o routines (DREAM3D format) """

import numpy as np
import h5py

GRAIN_ID_PATH = 'DataContainers/SyntheticVolume/CellData/FeatureIds'
QUATERNION_PATH = 'DataContainers/SyntheticVolume/CellFeatureData/AvgQuats'
PRNG_STATE_PATH = 'DataContainers/SytheticVolume/prng_state'

def load_dream3d(path):
  f = h5py.File(path, 'r')
  grain_ids = np.array(f[GRAIN_ID_PATH])
  f.close()
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
      f[PRNG_STATE_PATH] = state[1]
  return

def dump_dream3d(sites, time):
  path = 'dump{0:06d}.dream3d'.format(time)
  f = h5py.File(path)
  f[GRAIN_ID_PATH] = sites
  f.close()
  return

def load_quaternions(path):
  f = h5py.File(path, 'r')
  quaternions = np.array(f[QUATERNION_PATH], dtype=np.float32)
  f.close()
  return quaternions
