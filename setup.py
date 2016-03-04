from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(name='mcpm',
      version='0.1',
      description='Kinetic Monte Carlo grain growth model',
      url='tbd',
      author='Brian DeCost',
      author_email='bdecost@andrew.cmu.edu',
      license='MIT',
      packages=['mcpm', 'mcpm.utils'],
      ext_modules=cythonize('mcpm/utils/unique.pyx'),
      include_dirs=[numpy.get_include()],
      entry_points={
        'console_scripts': [
          'mcpm = mcpm.mcpm:main',
          'mcpm-draw = mcpm.utils.draw:draw_snapshot',
          'mcpm-animate = mcpm.utils.draw:animate_snapshots',
          ],
        },
      zip_safe=False)
