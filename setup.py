from pathlib import Path
from setuptools import setup

with open(Path(__file__).parent / 'README.md') as f:
    long_description = f.read()

setup(
    name='pie-torch',
    version='0.0',
    packages=['pietorch'],
    url='https://github.com/snavalm/mood22',
    license='MIT',
    author='Sergio Naval Marimont',
    author_email='sergio.naval-marimont@city.ac.uk',
    description='Implementation of the MOOD22 CitAI submission',
    long_description=long_description,
    install_requirements=[
        'monai>=1.1.0'
        'torch>=2.0.0'
        'pandas>=1.4.3'
        'nibabel>=4.0.1'
        'matplotlib>=3.5.2'
        'numpy'
    ]
)