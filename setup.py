from setuptools import setup

setup(name='mcpm',
      version='0.1',
      description='Kinetic Monte Carlo grain growth model',
      url='tbd',
      author='Brian DeCost',
      author_email='bdecost@andrew.cmu.edu',
      license='MIT',
      packages=['mcpm', 'mcpm.utils'],
      entry_points={
        'console_scripts': [
          'mcpm = mcpm.mcpm:main',
          'mcpm-draw = mcpm.utils.draw:draw_snapshot',
          'mcpm-animate = mcpm.utils.draw:animate_snapshots',
          ],
        },
      zip_safe=False)
