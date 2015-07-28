""" simple HDF5 i/o routines (DREAM3D format) """

import numpy as np
import h5py

GRAIN_ID_PATH = 'DataContainers/SyntheticVolume/CellData/FeatureIds'
QUATERNION_PATH = 'DataContainers/SyntheticVolume/CellFeatureData/AvgQuats'

def load_dream3d(path):
  f = h5py.File(path, 'r')
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

def load_quaternions(path):
  f = h5py.File(path, 'r')
  quaternions = np.array(f[QUATERNION_PATH], dtype=np.float32)
  f.close()
  return quaternions
