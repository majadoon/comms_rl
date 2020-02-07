from setuptools import setup

setup(name='comms_rl',
      version='0.1',
      packages=['comms_rl'],
      license='Copyright 2019. Nokia Bell-Labs.',
      author='Alvaro Valcarce',
      author_email='alvaro.valcarce_rial@nokia-bell-labs.com',
      description='Classes and modules for executing telecommunication problems as OpenAI Gym environments.',
      install_requires=['gym', 'numpy', 'scipy', 'sacred']  # Insert any other dependencies required
)