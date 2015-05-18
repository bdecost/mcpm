"""
Monte Carlo grain growth solver
kinetic or rejection MC schemes
traditional or gaussian pixel neighborhood
"""

from . import io
from . import kinetic
from . import rejection
from .spatial import uniform_mask, gaussian_mask

import argparse
import numpy as np

def main():
  parser = argparse.ArgumentParser(prog='mcpm',
             description='''Kinetic Monte Carlo grain growth
                       simulations in 2 and 3 dimensions.''')

  parser.add_argument('--infile', nargs='?', default='input.dream3d',
                      help='DREAM3D file containing initial structure')
  parser.add_argument('--style', default='kmc',
                      choices=['kmc', 'reject'],
                      help='Monte Carlo style')
  parser.add_argument('--nbrhd', nargs='?', default='gaussian',
                      choices=['uniform', 'gaussian'],
                      help='pixel neighborhood weighting')
  parser.add_argument('--sigma', type=float, default=3,
                      help='smoothing parameter for gaussian neighborhood')
  parser.add_argument('--radius', type=int, default=9,
                      help='pixel neighborhood radius')
  parser.add_argument('--kT', type=float, default=.001,
                      help='Monte Carlo temperature kT')
  parser.add_argument('--length', type=float, default=100,
                      help='Simulation length in MCS')
  parser.add_argument('--cutoff', type=float, default=0.01,
                      help='interaction weight cutoff value')
  parser.add_argument('--norm', type=float, default=1,
                      help='gaussian kernel normalization constant a')
  parser.add_argument('--freq', type=float, default=10,
                      help='timesteps between system snapshots')
  parser.add_argument('--neighborlist', action='store_true',
                      help='''Compute explicit neighbor lists.
                              Problematic with large 3D systems.''')
  
  args = parser.parse_args()
  sites = io.load_dream3d(args.infile)

  if args.nbrhd == 'uniform':
    weights = uniform_mask(sites, radius=1)
  elif args.nbrhd == 'gaussian':
    weights = gaussian_mask(sites, args.radius, a=args.norm,
                        sigma_squared=np.square(args.sigma))

  if args.style == 'reject':
    rejection.iterate(sites, weights, args)
  elif args.style == 'kmc':
    kinetic.iterate(sites, weights, args)
    
