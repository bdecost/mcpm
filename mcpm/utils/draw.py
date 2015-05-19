""" draw 2D simulation snapshots """

import numpy as np
import h5py
import os
import glob
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
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

def map_colors(sites, color_map):
  ''' apply color mapping and fill in boundaries '''
  shape = sites.shape + (3,)
  colors = np.zeros(shape, dtype=float)
  for (x,y), s in np.ndenumerate(sites):
    colors[(x,y)] = color_map[s]
  colors[find_boundaries(sites), :] = np.array([.3,.3,.3])
  return colors


def draw(sites, outfile=None, colormap=None, vmin=None, vmax=None):
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
  if outfile is not None:
    plt.savefig(outfile)
    plt.clf()
  else:
    plt.show()

def propensity(sites, outfile=None):
  return  
    
def sequence(tmpdir='images'):
  snapshots = glob.glob('dump*.dream3d')
  snapshots.sort()
  vmin, vmax = 0, 0
  for snapshot in snapshots:
    print(snapshot)
    name, ext = os.path.splitext(snapshot)
    s = load_dream3d(snapshot)
    if (vmin, vmax) == (0,0):
      vmax = s.max()
    draw(s, '{}/{}.png'.format(tmpdir,name), vmin=0, vmax=vmax)


def draw_snapshot():
  parser = argparse.ArgumentParser(prog='draw-snapshot',
             description='''Draw a 2D snapshot from a DREAM3D file''')
  parser.add_argument('-i', '--infile', nargs='?', default='input.dream3d',
                      help='path to dream3d file to draw')
  parser.add_argument('-o', '--outfile', nargs='?', default=None,
                      help='save image file to this path')
  parser.add_argument('-c', '--color', nargs='?', default='grain_id',
                      help='color scheme')

  args = parser.parse_args()
  
  sites = io.load_dream3d(args.infile)
  cmap = None
  if args.color == 'quaternion':
    quats = io.load_quaternions(args.infile)
    cmap = rodrigues_colormap(quats)
  draw(sites, colormap=cmap, outfile=args.outfile)


def animate_snapshots():
  parser = argparse.ArgumentParser(prog='draw-snapshot',
             description='''Draw a 2D snapshot from a DREAM3D file''')
  parser.add_argument('-i', '--snapshots', nargs='+', required=True,
                      help='list of dream3d snapshots to animate')
  parser.add_argument('-o', '--outfile', nargs='?', default='grains.gif',
                      help='save image file to this path')
  parser.add_argument('--cleanup', action='store_true',
                      help='clean up temporary image files')
  parser.add_argument('-c', '--color', nargs='?', default='grain_id',
                      help='color scheme')

  args = parser.parse_args()

  tmpdir = 'tmp'
  try:
    os.mkdir(tmpdir)
  except FileExistsError:
    pass

  print('processing snapshots')
  args.snapshots.sort()
  vmin, vmax = 0, 0
  cmap = None
  for snapshot in args.snapshots:
    print(snapshot)
    name, ext = os.path.splitext(snapshot)
    s = io.load_dream3d(snapshot)
    if (vmin, vmax) == (0,0):
      vmax = s.max()
      if args.color == 'quaternion':
        quats = io.load_quaternions(args.infile)
        cmap = rodrigues_colormap(quats)
    draw(s, '{}/{}.png'.format(tmpdir,name),
         colormap=cmap, vmin=0, vmax=vmax)

  images = glob.glob('{}/*.png'.format(tmpdir))
  images.sort()
  print('calling imagemagick')
  subprocess.call(['convert'] + images + ['-delay', '25', args.outfile])

  if args.cleanup:
    print('cleaning up')
    for i in images:
      os.remove(i)
    os.rmdir(tmpdir)
