from setuptools import setup, find_packages

setup(
      name='exmlc',
      version='0.1',
      packages=find_packages(),
      install_requires=[
            'scipy',
            'numpy',
            'networkx',
            'scikit-learn'
      ],
      license='LICENSE'
)
