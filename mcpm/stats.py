""" microstructure data output """
import numpy as np
import pandas as pd

from . import io

stats_df = None  # pandas dataframe for stats output
state_ids = None # list of Potts states
output_index = 0
stats_file = None

def initialize(sites, args):
  global stats_df
  global state_ids
  global stats_file
  num_snapshots = 1 + (args.length / args.freq)
  state_ids = np.sort(np.unique(sites))
  zero_data = np.zeros((num_snapshots,state_ids.size+1), dtype=float)
  columns = ['time'] + list(map(str, state_ids))
  stats_df = pd.DataFrame(zero_data,
                          columns=columns,
                          index=np.arange(0,num_snapshots))
  stats_file = args.statsfile
  with pd.HDFStore(stats_file) as store:
    store['stats'] = stats_df
  return
  
def compute(sites, time=None):
  global stats_df
  global output_index
  # get grain size in voxels
  stats_dict = {str(state): np.sum(sites == state) for state in state_ids}
  stats_dict['time'] = time
  stats_df.loc[output_index] = pd.Series(stats_dict)
  print('computed stats!')
  checkpoint()
  output_index += 1
  return

def checkpoint():
  global stats_df
  with pd.HDFStore(stats_file) as store:
    del store['stats']
    store['stats'] = stats_df
  return

def finalize():
  checkpoint()
  return
