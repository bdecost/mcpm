""" microstructure data output """
import h5py
import numpy as np

from . import io

stats_df = None  # pandas dataframe for stats output
unique_ids = None # list of Potts states
output_index = 0
stats_file = None

def initialize(sites, args):
  global unique_ids
  global stats_file
  stats_file = args.statsfile

  num_snapshots = 1 + (args.length / args.freq)
  unique_ids = np.sort(np.unique(sites))
  np.insert(unique_ids, 0, 0) # insert zero to match index with grain id
  zero_data = np.zeros((num_snapshots,unique_ids.size), dtype=float)
  with h5py.File(stats_file) as f:
    f['time'] = np.zeros(num_snapshots)
    f['grainsize'] = zero_data
  return
  
def compute(sites, time=None):
  global output_index
  global unique_ids
  global stats_file
  # get grain size in voxels
  grainsize = np.array([np.sum(sites == state) for state in unique_ids])
  with h5py.File(stats_file) as f:
    f['time'][output_index] = time
    f['grainsize'][output_index] = grainsize

  print('computed stats!')
  output_index += 1
  return

def finalize():
  pass
  return
