""" draw 2D simulation snapshots """

import numpy as np
import h5py
import os
import glob
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
import argparse
import subprocess

from ..io import load_dream3d

def draw(sites, outfile=None, vmin=None, vmax=None):
  sites[find_boundaries(sites)] = 0
  plt.imshow(sites, interpolation='none',
             cmap=plt.get_cmap('Spectral'),
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
  args = parser.parse_args()
  
  sites = load_dream3d(args.infile)
  draw(sites, outfile=args.outfile)


def animate_snapshots():
  parser = argparse.ArgumentParser(prog='draw-snapshot',
             description='''Draw a 2D snapshot from a DREAM3D file''')
  parser.add_argument('-i', '--snapshots', nargs='+', required=True,
                      help='list of dream3d snapshots to animate')
  parser.add_argument('-o', '--outfile', nargs='?', default='grains.gif',
                      help='save image file to this path')
  parser.add_argument('-c', '--cleanup', action='store_true',
                      help='clean up temporary image files')
  args = parser.parse_args()

  tmpdir = 'tmp'
  try:
    os.mkdir(tmpdir)
  except FileExistsError:
    pass

  print('processing snapshots')
  args.snapshots.sort()
  vmin, vmax = 0, 0
  for snapshot in args.snapshots:
    print(snapshot)
    name, ext = os.path.splitext(snapshot)
    s = load_dream3d(snapshot)
    if (vmin, vmax) == (0,0):
      vmax = s.max()
    draw(s, '{}/{}.png'.format(tmpdir,name), vmin=0, vmax=vmax)

  images = glob.glob('{}/*.png'.format(tmpdir))
  images.sort()
  print('calling imagemagick')
  subprocess.call(['convert'] + images + ['-delay', '25', args.outfile])

  if args.cleanup:
    print('cleaning up')
    for i in images:
      os.remove(i)
    os.rmdir(tmpdir)
