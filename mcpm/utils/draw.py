""" draw 2D simulation snapshots """
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


import numpy as np
import h5py
import os
import glob
from skimage.segmentation import find_boundaries
import argparse
import subprocess

import misori

from .. import io

def rodrigues_colormap(quats):
  """ rodrigues vector mapped to unit cube as rgb values """
  quats = np.sort(np.abs(quats), axis=1) # sort ascending
  fz_half = np.tan(np.pi/8) # half cubic fundamental zone width
  rf = [misori.fz_rod(q)+fz_half for q in quats]
  return {index: a / (2*np.tan(np.pi/8)) for index, a in enumerate(rf)}

def binary_colormap(quats):
  cmap = {1: np.array([1.0, 1.0, 1.0]),
          2: np.array([.45, .57, 0.63]),
          3: np.array([0.6, 0.0, 0.0])}
  return {i: cmap[np.sum(quats[i,:] > 0)]
          for i in range(1,quats.shape[0])}

def map_colors(sites, color_map):
  ''' apply color mapping and fill in boundaries '''
  shape = sites.shape + (3,)
  colors = np.zeros(shape, dtype=float)
  for (x,y), s in np.ndenumerate(sites):
    colors[(x,y)] = color_map[s]
  colors[find_boundaries(sites), :] = np.array([.3,.3,.3])
  return colors

def mark_grain(sites, grain, idgrain):
  colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']

  # get grain centroids
  index = np.indices(sites.shape)
  idx , idy = index[0], index[1]
  size_x, size_y = sites.shape
  mask = sites == grain
  if np.any(mask):
    y, x = idy[mask], idx[mask]
    # fix periodic boundary splinching
    if np.max(y) - np.min(y) == size_y-1:
      y[y < size_y/2] += size_y
    if np.max(x) - np.min(x) == size_x-1:
      x[x < size_x/2] += size_x
    y, x = np.mean(y), np.mean(x)
    y = y if y < size_y else y - size_y
    x = x if x < size_x else x - size_x
    # use image coordinates -- forgot to change imshow
    plt.scatter(y, x, color=colors[idgrain], edgecolor='w')
  return

def draw(sites, outfile=None, colormap=None,
         vmin=None, vmax=None, mark_grains=[]):
  cmap = None
  if colormap is not None:
    colors = map_colors(sites, colormap)
  else:
    colors = sites
    colors[find_boundaries(sites)] = 0
    cmap=plt.get_cmap('Spectral')
  plt.imshow(colors, interpolation='none',
             cmap=cmap,
             vmin=vmin, vmax=vmax, origin='lower')
  for idg,grain in enumerate(mark_grains):
    mark_grain(sites, grain, idg) # calls plt.plot
  if outfile is not None:
    plt.savefig(outfile)
    plt.clf()
  else:
    return

def propensity(sites, outfile=None):
  return  
    
def sequence(tmpdir='images'):
  snapshots = glob.glob('dump*.dream3d')
  snapshots.sort()
  vmin, vmax = 0, 0
  for i,snapshot in enumerate(snapshots):
    print(snapshot)
    # name, ext = os.path.splitext(snapshot)
    outfile = 'snapshot{0:04d}.png'.format(i)
    s = load_dream3d(snapshot)
    if (vmin, vmax) == (0,0):
      vmax = s.max()
    draw(s, '{}/{}'.format(tmpdir,outfile), vmin=0, vmax=vmax)


def draw_snapshot():
  parser = argparse.ArgumentParser(prog='draw-snapshot',
             description='''Draw a 2D snapshot from a DREAM3D file''')
  parser.add_argument('-i', '--infile', nargs='?', default='input.dream3d',
                      help='path to dream3d file to draw')
  parser.add_argument('-o', '--outfile', nargs='?', default=None,
                      help='save image file to this path')
  parser.add_argument('-c', '--color', nargs='?', default='grain_id',
                      help='color scheme')
  parser.add_argument('--initial', nargs='?', default='input.dream3d',
                      help='initial snapshot with quaternions')

  args = parser.parse_args()
  
  sites = io.load_dream3d(args.infile)
  cmap = None
  if args.color == 'quaternion':
    quats = io.load_quaternions(args.initial)
    cmap = rodrigues_colormap(quats)
  draw(sites, colormap=cmap, outfile=args.outfile)


def animate_snapshots():
  parser = argparse.ArgumentParser(prog='mcpm-animate',
             description='''Animate 2D snapshots from file sequence''')
  parser.add_argument('-i', '--snapshots', nargs='+', required=True,
                      help='list of snapshots to animate')
  parser.add_argument('-o', '--outfile', nargs='?', default='grains.gif',
                      help='save image file to this path')
  parser.add_argument('--cleanup', action='store_true',
                      help='clean up temporary image files')
  parser.add_argument('-c', '--color', nargs='?', default='grain_id',
                      help='color scheme')
  parser.add_argument('--initial', nargs='?', default='input.dream3d',
                      help='initial snapshot with quaternions')
  parser.add_argument('--format', choices=['dream3d', 'spparks'], default='dream3d',
                      help='input file format')
  parser.add_argument('--ffmpeg', action='store_true',
                      help='use ffmpeg -> *.mov instead of imagemagick')

  args = parser.parse_args()

  plt.switch_backend('Agg')
  tmpdir = 'temp'
  try:
    os.mkdir(tmpdir)
  except FileExistsError:
    pass
  mark_grains = [] # list of grains to mark
  # mark_grains = [158, 2136, 2482, 300, 335, 39, 823]
  print('processing snapshots')
  args.snapshots.sort()
  vmin, vmax = 0, 0
  cmap = None
  for i,snapshot in enumerate(args.snapshots):
    print(snapshot)
    # name, ext = os.path.splitext(snapshot)
    name = 'snapshot{:04d}'.format(i)
    if args.format == 'dream3d':
      s = io.load_dream3d(snapshot)
    elif args.format == 'spparks':
      s = io.load_spparks(snapshot)
    if (vmin, vmax) == (0,0):
      vmax = s.max()
      if args.color == 'quaternion':
        quats = io.load_quaternions(args.initial)
        cmap = rodrigues_colormap(quats)
      elif args.color == 'binary':
        quats = io.load_quaternions(args.initial)
        cmap = binary_colormap(quats)

    draw(s, '{}/{}.png'.format(tmpdir,name),
         colormap=cmap, vmin=0, vmax=vmax,
         mark_grains=mark_grains)

  images = glob.glob('{}/*.png'.format(tmpdir))
  images.sort()
  if not args.ffmpeg: 
    print('calling imagemagick')
    subprocess.call(['convert'] + images + ['-delay', '25', args.outfile])
  else:
    print('calling ffmpeg')
    subprocess.call(['ffmpeg'] + '-i temp/snapshot%04d.png -codec png grains.mov'.split(' '))
    
  if args.cleanup:
    print('cleaning up')
    for i in images:
      os.remove(i)
    os.rmdir(tmpdir)
