"""
Monte Carlo grain growth solver
kinetic or rejection MC schemes
traditional or gaussian pixel neighborhood
"""

from . import io
from . import stats
from . import kinetic
from . import rejection
from .spatial import uniform_mask, gaussian_mask, strained_mask

import argparse
import numpy as np

def main():
  parser = argparse.ArgumentParser(prog='mcpm',
             description="Kinetic Monte Carlo grain growth"
                         "simulations in 2 and 3 dimensions.",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('-i', '--infile', nargs='?', default='input.dream3d',
                      help='DREAM3D file containing initial structure')
  parser.add_argument('--style', default='kmc',
                      choices=['kmc', 'reject'],
                      help='Monte Carlo style')
  parser.add_argument('--nbrhd', nargs='?', default='gaussian',
                      choices=['uniform', 'gaussian', 'strained'],
                      help='pixel neighborhood weighting')
  parser.add_argument('--sigma', type=float, default=3,
                      help='smoothing parameter for gaussian neighborhood')
  parser.add_argument('--radius', type=int, default=9,
                      help='pixel neighborhood radius')
  parser.add_argument('--kT', type=float, default=.001,
                      help='Monte Carlo temperature kT')
  parser.add_argument('-l', '--length', type=float, default=100,
                      help='Simulation length in MCS')
  parser.add_argument('--cutoff', type=float, default=0.01,
                      help='interaction weight cutoff value')
  parser.add_argument('--norm', type=float, default=1,
                      help='gaussian kernel normalization constant a')
  parser.add_argument('--freq', type=float, default=10,
                      help='timesteps between system snapshots')
  parser.add_argument('--prefix', nargs='?', default='mcpm_dump',
                      help='prefix for dump file')
  parser.add_argument('--neighborlist', action='store_true',
                      help='''Compute explicit neighbor lists.
                              Problematic with large 3D systems.''')
  parser.add_argument('--mobility', type=float, default=1.0,
                      help='''use misorientation-threshold mobility. This is the mobility ratio.''')
  parser.add_argument('--angle', type=float, default=30.0,
                      help='high angle boundary cutoff in degrees')
  parser.add_argument('--statsfile', nargs='?', default='stats.h5',
                      help='HDF5 file for grain growth stats')
  parser.add_argument('--neighborfile', nargs='?', default='')
  parser.add_argument('--load_prng_state', action='store_true',
                      help='use the PRNG state stored in the input file')
  parser.add_argument('--nostats', action='store_true',
                      help='no statistics file.')
  parser.add_argument('--nodump', action='store_true',
                      help='no dream3d dump files.')
  
  args = parser.parse_args()
  sites = io.load_dream3d(args.infile)

  if args.load_prng_state:
    io.load_prng_state(args.infile)
  else:
    io.save_prng_state(args.infile)
  if not args.nostats:
    stats.initialize(sites, args)
  
  if args.nbrhd == 'uniform':
    weights = uniform_mask(sites, radius=args.radius)
  elif args.nbrhd == 'strained':
    weights = strained_mask(sites, radius=args.radius, strain=0.1)
  elif args.nbrhd == 'gaussian':
    weights = gaussian_mask(sites, args.radius, a=args.norm,
                            sigma_squared=np.square(args.sigma),
                            cutoff=args.cutoff)
  
  if args.style == 'reject':
    rejection.iterate(sites, weights, args)
  elif args.style == 'kmc':
    kinetic.iterate(sites, weights, args)

  io.save_args(args)
  if not args.nostats:
    stats.finalize()
